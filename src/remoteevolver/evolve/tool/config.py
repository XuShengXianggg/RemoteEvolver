"""Configuration for GitHub coarse ranking and README assessment."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from pathlib import Path
import os
import math
import yaml

_DEFAULT_PROMPT = """\
你是 GitHub 仓库搜索助理。用户会描述一项他希望某个库帮他完成的能力。
请输出一条**最适合塞进 GitHub /search/repositories?q= 的搜索 query**。

(本 prompt 仅用于 raw_query 兜底 / CLI 直接搜场景;主流程已改走 planner.py 双通道。)

★ 核心规则 ★
1. **只输出 query 本身**,不要解释 / 引号 / Markdown / 前缀。
2. 用英文关键词(GitHub 索引以英文为主)。
3. **关键词控制在 2-3 个**,GitHub 是字面字符串匹配,词越多召回越低。
4. **不要加任何质量过滤限定符**(`stars:` / `archived:` / `pushed:` 等),
   这些过滤会埋掉刚发布但贴合任务的小库,质量评估交给后面的 5 维评分。
5. `language:` 不要写,GitHub 客户端层会自动加 `language:Python`。
6. **不要用 OR / 括号 / 复杂布尔**,GitHub 单 query 上限是 5 个 AND/OR/NOT 操作符。

例:
  用户:查股票实时价格         → stock price api
  用户:把网页转成 PDF          → html pdf converter
  用户:OCR 识别图片里的中文    → chinese ocr
  用户:用 bcrypt 算密码哈希    → bcrypt
  用户:RFC5545 ICS 日历生成    → icalendar

现在用户的需求是:
{capability}
"""


@dataclass
class ScoreConfig:
    relevance_weight: float = 30.0
    stars_log_weight: float = 3.0
    has_license_bonus: float = 2.0
    has_description_bonus: float = 1.0
    no_description_penalty: float = -1.0
    recent_year_threshold: int = 2025
    recent_bonus: float = 2.0
    stale_year_threshold: int = 2022
    stale_penalty: float = -3.0

@dataclass
class AssessmentWeights:
    semantic: float = 0.60
    structural: float = 0.15
    toolization: float = 0.15
    community: float = 0.05
    environment_penalty: float = 0.05

@dataclass
class AssessmentThresholds:
    semantic: float = 0.65
    structural: float = 0.40
    toolization: float = 0.40

@dataclass
class CommunityConfig:
    stars_log_divisor: float = 5.0
    stars_weight: float = 0.70
    license_bonus: float = 0.15
    description_bonus: float = 0.05
    recent_year_threshold: int = 2025
    recent_bonus: float = 0.10
    stale_year_threshold: int = 2021
    stale_penalty: float = -0.10

@dataclass
class AssessmentConfig:
    weights: AssessmentWeights = field(default_factory=AssessmentWeights)
    thresholds: AssessmentThresholds = field(default_factory=AssessmentThresholds)
    community: CommunityConfig = field(default_factory=CommunityConfig)

@dataclass
class RelaxConfig:
    drop_order: list[str] = field(default_factory=lambda: [
        'stars:>', 'stars:>=', 'pushed:>', 'pushed:>=', 'archived:false', 'license:',
    ])

@dataclass
class QueryBuilderConfig:
    prompt: str = _DEFAULT_PROMPT

@dataclass
class EvolveConfig:
    query_builder: QueryBuilderConfig = field(default_factory=QueryBuilderConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    assessment: AssessmentConfig = field(default_factory=AssessmentConfig)


def _section(cls, value, section):
    if not isinstance(value, dict):
        raise ValueError(f'{section} must be a mapping')
    defaults = cls()
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f'unknown {section} options: {sorted(unknown)}')
    result = {}
    for f in fields(cls):
        v = value.get(f.name, getattr(defaults, f.name))
        default = getattr(defaults, f.name)
        if isinstance(default, (int, float)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f'{section}.{f.name} must be a finite number')
            if isinstance(default, int) and int(v) != v:
                raise ValueError(f'{section}.{f.name} must be an integer')
            v = type(default)(v)
        result[f.name] = v
    return cls(**result)


def load_evolve_config(path: str | Path | None = None) -> EvolveConfig:
    """Read run-scoped YAML; omitted fields use the documented defaults.

    Resolve the environment at call time so switching runs does not retain a
    configuration path captured by an earlier module import.
    """
    selected = path if path is not None else os.environ.get('REMOTEEVOLVER_TOOL_CONFIG', '')
    raw = {}
    if selected:
        with Path(selected).open(encoding='utf-8') as stream:
            raw = yaml.safe_load(stream)
        if raw is None:
            raw = {}
    if not isinstance(raw, dict):
        raise ValueError('tool evolution configuration must be a mapping')
    unknown = set(raw) - {'score', 'assessment', 'query_builder', 'relax'}
    if unknown:
        raise ValueError(f'unknown tool evolution sections: {sorted(unknown)}')
    assessment = raw.get('assessment', {})
    if not isinstance(assessment, dict) or set(assessment) - {'weights', 'thresholds', 'community'}:
        raise ValueError('assessment accepts only weights, thresholds and community mappings')
    a = AssessmentConfig(
        weights=_section(AssessmentWeights, assessment.get('weights', {}), 'assessment.weights'),
        thresholds=_section(AssessmentThresholds, assessment.get('thresholds', {}), 'assessment.thresholds'),
        community=_section(CommunityConfig, assessment.get('community', {}), 'assessment.community'),
    )
    if any(v < 0 for v in vars(a.weights).values()):
        raise ValueError('assessment weights must be nonnegative')
    if any(not 0 <= v <= 1 for v in vars(a.thresholds).values()):
        raise ValueError('assessment thresholds must be between 0 and 1')
    if a.community.stars_log_divisor <= 0:
        raise ValueError('community.stars_log_divisor must be positive')
    score = _section(ScoreConfig, raw.get('score', {}), 'score')
    for label, c in [('score', score), ('assessment.community', a.community)]:
        if c.stale_year_threshold >= c.recent_year_threshold:
            raise ValueError(f'{label}: stale year must precede recent year')
    qb = _section(QueryBuilderConfig, raw.get('query_builder', {}), 'query_builder')
    if not isinstance(qb.prompt, str) or '{capability}' not in qb.prompt:
        raise ValueError('query_builder.prompt must contain {capability}')
    relax = _section(RelaxConfig, raw.get('relax', {}), 'relax')
    if not isinstance(relax.drop_order, list) or not all(isinstance(v, str) for v in relax.drop_order):
        raise ValueError('relax.drop_order must be a list of strings')
    return EvolveConfig(query_builder=qb, relax=relax, score=score, assessment=a)
