"""Independent answer judge. Transport/protocol failures never become score zero."""
from __future__ import annotations
import json
import math
from pathlib import Path


class JudgeError(RuntimeError):
    pass


def parse_score(raw):
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    try:
        data = json.loads(text)
        value = data['Score']
        if isinstance(value, bool):
            raise ValueError('boolean score')
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('score outside [0,1]')
        if not isinstance(data.get('Justification'), str):
            raise ValueError('missing Justification')
        return score, data['Justification']
    except (TypeError, ValueError, KeyError) as exc:
        raise JudgeError('Judge response does not satisfy the scoring contract') from exc


class AnswerJudge:
    def __init__(self, config, key, *, client=None):
        self.config = config
        self.threshold = float(config.get('success_threshold', 0.8))
        if not 0 <= self.threshold <= 1:
            raise ValueError('success_threshold must be in [0,1]')
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url=config['base_url'],
                            timeout=float(config.get('timeout_seconds', 180)),
                            max_retries=int(config.get('max_retries', 1)))
        self.client = client
        path = config.get('prompt_file')
        self.prompt = Path(path).read_text(encoding='utf8') if path else (
            Path(__file__).parent/'prompt/judge.txt').read_text(encoding='utf8')

    def evaluate(self, question, reference, answer):
        kwargs = dict(self.config.get('request_options', {}))
        if {'model', 'messages', 'api_key', 'base_url'} & kwargs.keys():
            raise ValueError('Judge request_options cannot replace routing or messages')
        kwargs.setdefault('temperature', 0)
        try:
            response = self.client.chat.completions.create(
                model=self.config['model'], messages=[
                    {'role':'system', 'content':self.prompt},
                    {'role':'user', 'content':json.dumps({
                        'Question':question, 'Ground Truth Answer':reference,
                        'Predicted Answer':answer}, ensure_ascii=False)}], **kwargs)
            raw = response.choices[0].message.content or ''
            score, explanation = parse_score(raw)
        except Exception as exc:
            from remoteevolver.infra.errors import InfrastructureFailure
            raise InfrastructureFailure('judge', f'{type(exc).__name__}: judge failed; no score recorded') from exc
        usage = getattr(response, 'usage', None)
        return {'score':score, 'correct':score >= self.threshold,
                'success_threshold':self.threshold, 'justification':explanation,
                'model':self.config['model'], 'raw':raw,
                'usage':usage.model_dump() if usage is not None else {}}


class JudgeOracle:
    """Reference answers stay inside this evaluator, never in agent messages."""
    def __init__(self, judge, question, reference, *, scope='task', records=None, cache=None, errors=None):
        self.judge, self.question, self.reference = judge, question, reference
        self.errors = errors if errors is not None else []
        self.scope = scope
        self.records = records if records is not None else []
        self.cache = cache if cache is not None else {}

    def scoped(self, scope):
        return JudgeOracle(self.judge, self.question, self.reference, scope=scope,
                           records=self.records, cache=self.cache, errors=self.errors)

    def matches(self, answer):
        key = (self.scope, answer)
        if key not in self.cache:
            try:
                result = self.judge.evaluate(self.question, self.reference, answer)
            except Exception as exc:
                self.errors.append(exc)
                raise
            self.cache[key] = result
            self.records.append({'scope':self.scope, 'answer':answer, **result})
        return self.cache[key]['correct']

    def __str__(self):
        return '<independent evaluator; reference withheld>'
