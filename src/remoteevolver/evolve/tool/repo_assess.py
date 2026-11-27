"""Assess README evidence and metadata with configured weights.

Model ratings and deterministic community scores feed a composite score.
Thresholds guide candidate selection; they are not hard rejection gates.
"""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ...llm.base import BaseLLM, Message
from .github import GitHubClient, Repo
from .config import AssessmentConfig, CommunityConfig, load_evolve_config



DEFAULT_MAX_CANDIDATES_TO_ASSESS = 5
DEFAULT_README_CHARS = 3000        # 评估前每个候选拉 README 多少字符
DEFAULT_README_MAX_WORKERS = 3


@dataclass
class AssessmentScore:
    """单个仓库的 5 维评估 + 综合分。"""
    semantic_relevance: float = 0.0
    structural_completeness: float = 0.0
    toolization_feasibility: float = 0.0
    community_reliability: float = 0.0
    environment_complexity: float = 0.0     # 这一项越大越糟
    composite: float = 0.0                  # 在 from_dimensions() 里算好
    passes_thresholds: bool = False
    reasoning: str = ""                     # LLM 给的一行总结

    @classmethod
    def from_dimensions(
        cls,
        semantic: float,
        structural: float,
        toolization: float,
        community: float,
        env_complexity: float,
        reasoning: str = "",
        config: AssessmentConfig | None = None,
    ) -> "AssessmentScore":
        config = config or load_evolve_config().assessment
        composite = (
            config.weights.semantic * semantic
            + config.weights.structural * structural
            + config.weights.toolization * toolization
            + config.weights.community * community
            - config.weights.environment_penalty * env_complexity
        )
        passes = (
            semantic >= config.thresholds.semantic
            and structural >= config.thresholds.structural
            and toolization >= config.thresholds.toolization
        )
        return cls(
            semantic_relevance=semantic,
            structural_completeness=structural,
            toolization_feasibility=toolization,
            community_reliability=community,
            environment_complexity=env_complexity,
            composite=composite,
            passes_thresholds=passes,
            reasoning=reasoning,
        )


@dataclass
class AssessedRepo:
    """评估完的候选 = 原 Repo + AssessmentScore。"""
    repo: Repo
    score: AssessmentScore




class RepositoryAssessmentInvalid(ValueError):
    """The LLM assessment was incomplete after one bounded retry."""


def _repository_assessment_tool(known_names: list[str]) -> dict:
    """Build a bounded structured-output contract for compatible clients."""
    return {
        "type": "function",
        "function": {
            "name": "submit_repository_assessments",
            "description": "Submit one evidence-grounded assessment per candidate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "assessments": {
                        "type": "array",
                        "minItems": len(known_names),
                        "maxItems": len(known_names),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "enum": known_names},
                                "semantic": {"type": "number", "enum": [1, 0.7, 0.4, 0]},
                                "structural": {"type": "number", "enum": [1, 0]},
                                "toolization": {"type": "number", "enum": [1, 0]},
                                "environment": {"type": "number", "enum": [0, 0.3, 0.7, 1]},
                                "reasoning": {"type": "string", "maxLength": 300},
                            },
                            "required": [
                                "id", "semantic", "structural", "toolization",
                                "environment", "reasoning",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["assessments"],
                "additionalProperties": False,
            },
        },
    }


def _parse_structured_assessment(arguments: object, known_names: list[str]) -> dict[str, dict]:
    if not isinstance(arguments, dict) or not isinstance(arguments.get("assessments"), list):
        raise RepositoryAssessmentInvalid("structured assessment is missing assessments[]")
    parsed: dict[str, dict] = {}
    for item in arguments["assessments"]:
        if not isinstance(item, dict):
            raise RepositoryAssessmentInvalid("structured assessment row is not an object")
        repo_id = str(item.get("id", ""))
        if repo_id not in known_names or repo_id in parsed:
            raise RepositoryAssessmentInvalid(
                f"structured assessment has unknown or duplicate id: {repo_id!r}"
            )
        parsed[repo_id] = {
            "semantic": item.get("semantic"),
            "structural": item.get("structural"),
            "toolization": item.get("toolization"),
            "env_complexity": item.get("environment"),
            "reasoning": str(item.get("reasoning", "")).strip(),
        }
    errors = _assessment_parse_errors(parsed, known_names)
    if errors:
        raise RepositoryAssessmentInvalid("; ".join(errors))
    return parsed


# ============================================================================
# 社区可靠性 —— 不走 LLM,用元数据算(便宜且稳定)
# ============================================================================

def _community_score(repo: Repo, config: CommunityConfig | None = None) -> float:
    """Compute a bounded metadata score without a model request."""
    c = config or load_evolve_config().assessment.community
    stars = min(math.log10(repo.stars + 1) / c.stars_log_divisor, 1.0)
    score = c.stars_weight * stars
    if repo.license:
        score += c.license_bonus
    if repo.description:
        score += c.description_bonus
    match = re.match(r"(\d{4})-", repo.pushed_at or "")
    if match:
        year = int(match.group(1))
        if year >= c.recent_year_threshold:
            score += c.recent_bonus
        elif year <= c.stale_year_threshold:
            score += c.stale_penalty
    return max(0.0, min(1.0, score))


# ============================================================================
# LLM 评估(语义 / 结构 / 可工具化 / 环境复杂度)
# ============================================================================

_ASSESS_PROMPT = """\
You are reviewing Python repositories for one user task. Evaluate every candidate.

**Scoring rules**: semantic uses four discrete match levels. Distinguish a
repository dedicated to the task from a general library that could support a
custom implementation. This distinction is the highest priority. structural
and toolization remain Yes/No fields.

  semantic     [1.0/0.7/0.4/0.0]  Match between the repository's primary purpose and the task.
                          1.0 = The repository is dedicated to this exact task;
                                its name or README identifies this as a core
                                function and exposes a directly usable API.
                          0.7 = A general library provides the required lower-level
                                capability, but substantial custom code is needed.
                          0.4 = Only partially or indirectly relevant.
                          0.0 = Unrelated.
  structural   [Yes/No]  Is the repository complete and usable?
                          Yes = README + dependency declaration
                                (pyproject/setup/requirements) + usage example.
                          No  = Toy, incomplete, or tutorial-only project.
  toolization  [Yes/No]  Can it be wrapped as a programmatic tool?
                          Yes = Clear Python function/class and an import or call example.
                          No  = GUI-only, notebook-only, an awesome list, or no callable API.

  env_complexity [0/0.3/0.7/1.0]  Dependency complexity; higher is worse.
                          0.0 = Python standard library only.
                          0.3 = Ordinary pip packages such as requests or numpy.
                          0.7 = Compiled extensions, CUDA, or large model weights.
                          1.0 = External binaries or system-level configuration.

[USER TASK]
{capability}

[CANDIDATES, EACH WITH A README EXCERPT]
{candidate_block}

[OUTPUT FORMAT - use exactly one block per candidate]
CANDIDATE 1: <full_name_1>
  semantic: <1.0 | 0.7 | 0.4 | 0.0> - <one-sentence reason grounded in the README>
  structural: <Yes|No> - <one-sentence reason>
  toolization: <Yes|No> - <one-sentence reason>
  env_complexity: <0 | 0.3 | 0.7 | 1.0> - <one-sentence reason>
  reasoning: <one concise summary sentence>

CANDIDATE 2: <full_name_2>
  ...

[HARD CONSTRAINTS]
  - semantic must be exactly 1.0, 0.7, 0.4, or 0.0; structural and
    toolization must be exactly Yes or No.
  - env_complexity must be exactly 0, 0.3, 0.7, or 1.0.
  - Do not lower semantic merely because a repository has few stars. Community
    reliability is computed separately. A dedicated zero-star repository can
    still receive semantic=1.0.
  - If description is empty but the README is strong, semantic and structural
    may still receive positive scores.
  - Write all reasons in English. Preserve repository names and API identifiers.
"""


def enrich_with_readme(
    candidates: list[Repo],
    github: GitHubClient | None = None,
    *,
    max_chars: int = DEFAULT_README_CHARS,
    max_workers: int = DEFAULT_README_MAX_WORKERS,
) -> list[Repo]:
    """B-2.5:对每个候选拉一份 README 摘要,挂到 Repo.readme 上,**原地修改并返回同一个列表**。

    每次失败回退到空 README,不抛 —— 个别仓库没 README / 触发限流不应阻断整批评估。
    """
    if not candidates:
        return candidates
    gh = github or GitHubClient()
    pending = [r for r in candidates if not r.readme]
    if not pending:
        return candidates

    # README 请求相互独立，可安全地做小规模并发。结果仍按 candidates 对象原位写回，
    # 不改变候选顺序；单仓库失败继续降级为空 README，与旧串行行为一致。
    worker_count = min(len(pending), max(1, int(max_workers)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(gh.fetch_readme, r.full_name, max_chars=max_chars)
            for r in pending
        ]
        for r, future in zip(pending, futures):
            try:
                r.readme = future.result()
            except Exception:  # noqa: BLE001
                r.readme = ""
    return candidates


def build_assessment_prompt(
    capability: str,
    candidates: list[Repo],
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES_TO_ASSESS,
) -> tuple[str, list[Repo]]:
    """Build the production text-format assessment prompt."""
    head = candidates[:max_candidates]
    block_lines = []
    for i, r in enumerate(head, 1):
        readme_block = r.readme.strip() if r.readme else "(README unavailable or empty)"
        block_lines.append(
            f"  CANDIDATE {i}: {r.full_name}\n"
            f"    description: {(r.description or '(none)')[:200]}\n"
            f"    language: {r.language or '(unknown)'}    license: {r.license or '(none)'}    "
            f"stars: {r.stars}    pushed: {r.pushed_at[:10] if r.pushed_at else '?'}\n"
            f"    README excerpt:\n"
            f"    ```\n{_indent(readme_block, '      ')}\n    ```"
        )
    return _ASSESS_PROMPT.format(
        capability=capability.strip(),
        candidate_block="\n\n".join(block_lines),
    ), head






def assess_candidates(
    capability: str,
    candidates: list[Repo],
    llm: BaseLLM,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES_TO_ASSESS,
) -> list[AssessedRepo]:
    """Assess enriched candidates; incomplete output raises after one retry."""
    if not candidates:
        return []

    prompt, head = build_assessment_prompt(
        capability, candidates, max_candidates=max_candidates,
    )
    known_names = [r.full_name for r in head]
    structured_method = getattr(llm, "generate_required_tool", None)
    structured_tool = _repository_assessment_tool(known_names)
    parsed: dict[str, dict] = {}
    errors: list[str] = []
    # This stage owns its one retry for either a request failure or incomplete
    # output. Disable a client's outer retry temporarily so the layers cannot
    # multiply into four physical requests.
    old_call_attempts = getattr(llm, "_call_attempts", None)
    try:
        if old_call_attempts is not None:
            llm._call_attempts = 1
        for attempt in range(2):
            try:
                if callable(structured_method):
                    resp = structured_method(
                        [Message(role="user", content=prompt)], structured_tool,
                    )
                else:
                    resp = llm.generate([Message(role="user", content=prompt)])
            except Exception:
                if attempt == 0:
                    continue
                raise
            if callable(structured_method):
                calls = [
                    call for call in (resp.tool_calls or [])
                    if call.name == "submit_repository_assessments"
                ]
                try:
                    if len(calls) != 1:
                        raise RepositoryAssessmentInvalid(
                            "structured assessment returned no unique tool call"
                        )
                    parsed = _parse_structured_assessment(
                        calls[0].arguments, known_names,
                    )
                    errors = []
                except RepositoryAssessmentInvalid as exc:
                    parsed = {}
                    errors = [str(exc)]
            else:
                raw = (resp.text or "").strip()
                parsed = _parse_assessment(raw, known_names) if raw else {}
                errors = _assessment_parse_errors(parsed, known_names)
            if not errors:
                break
        else:
            raise RepositoryAssessmentInvalid(
                "repository assessment remained incomplete after 2 attempts: "
                + "; ".join(errors[:12])
            )
    finally:
        if old_call_attempts is not None:
            llm._call_attempts = old_call_attempts

    # 即便 LLM 全空,也要回退到"全用元数据估一下"的最小可用结果,避免上层崩

    assessment_config = load_evolve_config().assessment
    out: list[AssessedRepo] = []
    for r in head:
        block = parsed.get(r.full_name, {})
        # semantic 用 4 级匹配度(专用库 1.0 vs 通用库 0.7);structural/toolization 仍二元
        sem = _to_semantic_score(block.get("semantic"))
        struct = _to_binary(block.get("structural"))
        tool = _to_binary(block.get("toolization"))
        # env_complexity 保留 4 级离散(0/0.3/0.7/1.0)
        env = _to_env_level(block.get("env_complexity"))
        comm = _community_score(r, assessment_config.community)  # 不走 LLM
        reason = block.get("reasoning", "") or "(The LLM did not provide reasoning.)"
        score = AssessmentScore.from_dimensions(
            semantic=sem, structural=struct, toolization=tool,
            community=comm, env_complexity=env, reasoning=reason, config=assessment_config,
        )
        out.append(AssessedRepo(repo=r, score=score))

    out.sort(key=lambda a: a.score.composite, reverse=True)
    return out








def pick_best(
    assessed: list[AssessedRepo],
) -> tuple[AssessedRepo | None, bool]:
    """从评估结果里挑 top-1,并返回 (它是否过阈值)。

    策略(soft):
      1. 优先返回 composite 最高 **且通过 thresholds** 的那一条 → (it, True)
      2. 如果没有任何一条过 thresholds,仍然返回 composite 最高那条 → (it, False)
         调用方决定是 warning 继续还是直接放弃。
    """
    if not assessed:
        return None, False
    # assess_candidates 已经按 composite 降序,但谨慎起见再排一次
    sorted_list = sorted(assessed, key=lambda a: a.score.composite, reverse=True)
    for a in sorted_list:
        if a.score.passes_thresholds:
            return a, True
    return sorted_list[0], False


# ============================================================================
# 解析助手
# ============================================================================

def _indent(text: str, prefix: str) -> str:
    """给多行文本每行加前缀。空行也保留前缀,便于代码块对齐。"""
    return "\n".join(prefix + line for line in text.splitlines())


def _clip01(v: float | None, *, default: float) -> float:
    """把 LLM 给的数字夹到 [0,1];None / 无效 → default。"""
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return max(0.0, min(1.0, f))


# R3:Yes/No → 1/0,保守默认 0(No)避免误报
_YES_TOKENS = {"yes", "y", "true", "1", "是", "对"}
_NO_TOKENS = {"no", "n", "false", "0", "否", "不"}


def _to_binary(v: object) -> float:
    """LLM 输出 Yes/No(也容忍 1/0 / true/false / 中文是否)→ 1.0 / 0.0。
    无法识别默认 0.0(保守:不轻易给通过)。
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        # 数值容忍:>= 0.5 当 Yes
        if math.isnan(float(v)) or math.isinf(float(v)):
            return 0.0
        return 1.0 if float(v) >= 0.5 else 0.0
    s = str(v).strip().lower()
    # 取第一个 token(LLM 偶尔会写 "Yes — 因为..."),逗号/空格切
    first = re.split(r"[\s,—–\-]", s, maxsplit=1)[0]
    if first in _YES_TOKENS:
        return 1.0
    if first in _NO_TOKENS:
        return 0.0
    # 字符串里的数字兜底:>= 0.5 当 Yes(防止 LLM 还是给 "0.7" 这种)
    try:
        f = float(first)
        if not (math.isnan(f) or math.isinf(f)):
            return 1.0 if f >= 0.5 else 0.0
    except ValueError:
        pass
    # 包含 yes/no 关键字也能识别
    if "yes" in s and "no" not in s.split("yes")[0]:
        return 1.0
    if "no" in s:
        return 0.0
    return 0.0


# semantic 匹配度:4 级离散(1.0 专用 / 0.7 通用能凑 / 0.4 沾边 / 0.0 跑题)
_SEM_LEVELS = (0.0, 0.4, 0.7, 1.0)


def _to_semantic_score(v: object) -> float:
    """把 LLM 给的 semantic 吸到 4 级 {0, 0.4, 0.7, 1.0}。
    兼容旧的 Yes/No(Yes→1.0,No→0.0)。无法识别默认 0.4(保守:沾边)。"""
    if v is None:
        return 0.4
    s = str(v).strip().lower()
    first = re.split(r"[\s,—–\-]", s, maxsplit=1)[0] if s else ""
    if first in _YES_TOKENS:
        return 1.0
    if first in _NO_TOKENS:
        return 0.0
    try:
        f = float(first)
    except ValueError:
        return 0.4
    if math.isnan(f) or math.isinf(f):
        return 0.4
    return min(_SEM_LEVELS, key=lambda x: abs(x - f))


# env_complexity 仍保留 4 级离散
_ENV_LEVELS = (0.0, 0.3, 0.7, 1.0)


def _to_env_level(v: object) -> float:
    """把 LLM 给的 env_complexity 吸到 4 级 {0, 0.3, 0.7, 1.0}。无法识别默认 0.3。"""
    if v is None:
        return 0.3
    try:
        f = float(v)
    except (TypeError, ValueError):
        # 字符串描述兜底
        s = str(v).strip().lower()
        if "std" in s or "纯" in s or "0.0" in s:
            return 0.0
        if "cuda" in s or "compil" in s or "0.7" in s:
            return 0.7
        if "binary" in s or "系统" in s or "1.0" in s:
            return 1.0
        return 0.3
    if math.isnan(f) or math.isinf(f):
        return 0.3
    # 吸到最近的 level
    return min(_ENV_LEVELS, key=lambda x: abs(x - f))


def _parse_assessment(text: str, known_names: list[str]) -> dict[str, dict]:
    """从 LLM 输出抽出 {full_name: {semantic, structural, toolization, env_complexity, reasoning}}。

    宽容策略:
      - 仓库段开头用 "CANDIDATE <n>: <full_name>" 识别,匹配 known_names
      - 后续维度行匹配 "<key>: <数字> — ..." 或 "<key>: <数字>"
      - 中文冒号 / 缩进 / 全角破折号 都允许
    """
    out: dict[str, dict] = {}
    current_name: str | None = None
    cur_block: dict = {}

    # 维度键的小写匹配前缀
    dim_keys = {
        "semantic": "semantic",
        "structural": "structural",
        "toolization": "toolization",
        "env_complexity": "env_complexity",
        "reasoning": "reasoning",
    }

    def _flush():
        nonlocal current_name, cur_block
        if current_name and cur_block:
            out[current_name] = cur_block
        current_name = None
        cur_block = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        # 候选块起始
        if upper.startswith("CANDIDATE"):
            _flush()
            # 抓 full_name:可能写在 ": " 之后
            after = line.split(":", 1)[-1].strip() if ":" in line else ""
            after = after.lstrip(":：").strip()
            # 用 known_names 兜底:LLM 偶尔把 number 当 full_name 写,这里强匹配
            matched = None
            for name in known_names:
                if name in line:
                    matched = name
                    break
            current_name = matched or after or None
            cur_block = {}
            continue

        # 维度行:在某个 candidate 段里才记录
        if not current_name:
            continue
        for key_lower, store_key in dim_keys.items():
            if line.lower().startswith(key_lower):
                # 形如 "semantic: Yes — 理由" / "env_complexity: 0.3 — 理由"
                after = line.split(":", 1)
                if len(after) != 2:
                    break
                rest = after[1].strip()
                if store_key == "reasoning":
                    cur_block[store_key] = rest
                else:
                    # R3:不在这里强转 float;留原始字符串,由 _to_binary / _to_env_level
                    # 在调用层做"Yes/No → 1/0"或"4 级吸附"。这样维度可以是 Yes/No 也可以是数字。
                    cur_block[store_key] = rest
                break

    _flush()
    return out


def _assessment_parse_errors(
    parsed: dict[str, dict], known_names: list[str],
) -> list[str]:
    """Reject incomplete scores instead of turning missing fields into defaults."""
    errors: list[str] = []
    expected = set(known_names)
    unexpected = sorted(set(parsed) - expected)
    if unexpected:
        errors.append(f"unknown candidates={unexpected}")
    required = ("semantic", "structural", "toolization", "env_complexity")
    for name in known_names:
        block = parsed.get(name)
        if block is None:
            errors.append(f"{name}: missing candidate")
            continue
        missing = [key for key in required if key not in block]
        if missing:
            errors.append(f"{name}: missing fields={missing}")
            continue
        if not _valid_semantic(block["semantic"]):
            errors.append(f"{name}: invalid semantic")
        if not _valid_binary(block["structural"]):
            errors.append(f"{name}: invalid structural")
        if not _valid_binary(block["toolization"]):
            errors.append(f"{name}: invalid toolization")
        if not _valid_env(block["env_complexity"]):
            errors.append(f"{name}: invalid env_complexity")
    return errors


def _score_token(value: object) -> str:
    text = str(value).strip().lower()
    return re.split(r"[\s,–—-]", text, maxsplit=1)[0] if text else ""


def _valid_semantic(value: object) -> bool:
    token = _score_token(value)
    if token in _YES_TOKENS | _NO_TOKENS:
        return True
    try:
        number = float(token)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number in _SEM_LEVELS


def _valid_binary(value: object) -> bool:
    token = _score_token(value)
    if token in _YES_TOKENS | _NO_TOKENS:
        return True
    try:
        number = float(token)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number in (0.0, 1.0)


def _valid_env(value: object) -> bool:
    token = _score_token(value)
    try:
        number = float(token)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number in _ENV_LEVELS
