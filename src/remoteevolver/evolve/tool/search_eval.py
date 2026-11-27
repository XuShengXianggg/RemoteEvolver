"""
remoteevolver.evolve.tool.search_eval —— 搜索结果相关性评估 + 查询重写

背景:
  现有 search.py 只在 GitHub 返回 0 条时才用 `_relax_query` 剥限定符重试。
  但更常见的失败情形是"返回 20 条但全是噪声"——教学项目 / UI 壳 / 跑分 demo /
  完全跑题的同名包等。这一层让 LLM 看候选列表,判"够不够相关"。
  不够就让 LLM 重写 query(去过具体场景描述、改用库术语),再发一轮。

两个函数:
  check_relevance(capability, candidates, llm) -> RelevanceVerdict
      LLM 看 (full_name / description / stars) 判每条相关性,统计 ≥thr 即 SUFFICIENT。

  reformulate_query(capability, prior_query, prior_verdict, llm) -> str
      LLM 基于"为什么不够相关"重写一条更宽松、更面向库术语的 query。

设计原则:
  - 不主动 clone / 不调 GitHub API:本模块纯输入(候选元数据)纯输出(verdict / 新 query)
  - 失败兜底:解析不出来时返回保守的 SUFFICIENT=True 让上层走原结果,避免死循环
  - 每轮可观测:调用方将 verdict 写入 audit log,后续可统计"几轮 ≥ K 才相关"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...llm.base import BaseLLM, Message
from .github import Repo
from .planner import SearchPlan, _parse_topics


# 默认参数(调用方可覆盖)
DEFAULT_MIN_RELEVANT = 2          # ≥ 多少条相关算 SUFFICIENT
DEFAULT_TOP_N_FOR_EVAL = 10       # 取前 N 条交给 LLM 评(再多 prompt 会胀)
DEFAULT_MAX_ROUNDS = 2            # 总轮数上限(首轮 + 最多 1 次重写)


@dataclass
class RelevanceVerdict:
    """LLM 对一轮候选的相关性裁定。"""
    sufficient: bool                          # ≥ min_relevant 条相关
    relevant_count: int                       # LLM 数出的相关候选数
    reasoning: str                            # 一行人类可读理由
    per_repo: dict[str, str] = field(default_factory=dict)  # full_name -> "相关|不相关 — 理由"
    raw: str = ""                             # LLM 原始输出,出锅时回查


_RELEVANCE_PROMPT = """\
You are reviewing GitHub repository relevance. Decide whether each candidate,
as a Python library, can help complete the user's task.

[USER TASK]
{capability}

[CANDIDATE REPOSITORIES, FIRST {n} IN COARSE-RANK ORDER]
{candidate_block}

[DECISION STANDARD]
"RELEVANT" means the repository exposes an API or capability that can actually
perform the task or a necessary part of it.
"IRRELEVANT" includes tutorials, toy implementations, UI shells, benchmark
demos, unrelated namesakes, datasets rather than libraries, and notebook-only
examples without a callable API.

Use only full_name and description. Do not infer unsupported functionality from
the repository name. If the description is empty or unrelated, default to
IRRELEVANT.

[OUTPUT FORMAT - follow exactly]
PER_REPO:
  <full_name_1>: <RELEVANT | IRRELEVANT> - <one-sentence reason>
  <full_name_2>: <RELEVANT | IRRELEVANT> - <one-sentence reason>
  ...

RELEVANT_COUNT: <integer number of RELEVANT candidates>
SUFFICIENT: <YES | NO>
  (Rule: YES if RELEVANT_COUNT >= {min_relevant}; otherwise NO.)
REASONING: <one concise summary sentence in English>
"""


def check_relevance(
    capability: str,
    candidates: list[Repo],
    llm: BaseLLM,
    *,
    top_n: int = DEFAULT_TOP_N_FOR_EVAL,
    min_relevant: int = DEFAULT_MIN_RELEVANT,
) -> RelevanceVerdict:
    """对一批候选做 LLM 相关性评估。

    候选 0 条时直接返回 SUFFICIENT=False(根本没东西可评)。
    解析失败时保守返回 SUFFICIENT=True(避免无限重写)。
    """
    if not candidates:
        return RelevanceVerdict(
            sufficient=False, relevant_count=0,
            reasoning="No candidates were returned.", raw="",
        )

    head = candidates[:top_n]
    block_lines = [
        f"  {i}. {r.full_name}  stars={r.stars}  -  {(r.description or '(no description)')[:140]}"
        for i, r in enumerate(head, 1)
    ]
    prompt = _RELEVANCE_PROMPT.format(
        capability=capability.strip(),
        n=len(head),
        candidate_block="\n".join(block_lines),
        min_relevant=min_relevant,
    )
    resp = llm.generate([Message(role="user", content=prompt)])
    raw = (resp.text or "").strip()
    if not raw:
        return RelevanceVerdict(
            sufficient=True, relevant_count=len(head),
            reasoning="The LLM returned an empty response; candidates were conservatively accepted.", raw="",
        )

    per_repo = _parse_per_repo(raw, [r.full_name for r in head])
    sufficient = _parse_sufficient(raw)
    count = _parse_relevant_count(raw, per_repo)
    reasoning = _parse_reason(raw)

    # 兜底:解析不到 SUFFICIENT 但 per_repo 完整 → 用 count 自己算
    if sufficient is None:
        sufficient = count >= min_relevant
    return RelevanceVerdict(
        sufficient=sufficient,
        relevant_count=count,
        reasoning=reasoning,
        per_repo=per_repo,
        raw=raw,
    )


# ============================================================================
# 查询重写
# ============================================================================

_REFORMULATE_PROMPT = """\
You rewrite GitHub search queries. The previous search did not return enough
relevant candidates, so produce a materially different query.

[USER TASK]
{capability}

[PREVIOUS QUERY]
{prior_query}

[WHY IT WAS INSUFFICIENT]
{prior_reason}

[REPRESENTATIVE IRRELEVANT RESULTS]
{prior_irrelevant_examples}

[REWRITE RULES]
1. Preserve the task's core intent.
2. Remove overly specific scenario terms such as company names, record IDs, or
   incidental data formats. Prefer general library or package terminology.
3. Use established English terms found in PyPI and GitHub descriptions.
4. Preserve language:python and archived:false. Other qualifiers may change:
   lower or remove an unhelpful stars threshold, and replace overly narrow
   keywords with one or two established alternatives.
5. Do not repeat the previous keyword combination.

Output only the new query string, without explanation, quotation marks, or Markdown:
"""


def reformulate_query(
    capability: str,
    prior_query: str,
    prior_verdict: RelevanceVerdict,
    llm: BaseLLM,
) -> str:
    """让 LLM 基于"为什么不够相关"重写 query。

    返回清洗后的新 query;若 LLM 给空或与原 query 完全相同,返回空串(调用方应停止重写)。
    """
    # 拿前两条"不相关"的样本作上下文,够 LLM 知道方向就行,不必全列
    irrelevant = [
        f"  - {name}: {verdict}"
        for name, verdict in prior_verdict.per_repo.items()
        if (
            verdict.upper().startswith("IRRELEVANT")
            or "不相关" in verdict
            or "不沾边" in verdict
        )
    ][:3]
    examples = "\n".join(irrelevant) if irrelevant else "  (none)"

    prompt = _REFORMULATE_PROMPT.format(
        capability=capability.strip(),
        prior_query=prior_query.strip(),
        prior_reason=prior_verdict.reasoning or "(none)",
        prior_irrelevant_examples=examples,
    )
    resp = llm.generate([Message(role="user", content=prompt)])
    new_q = (resp.text or "").strip()
    # 清洗:LLM 偶尔加引号 / "query:" 前缀(沿用 build_query_from_capability 的逻辑)
    new_q = new_q.strip("`'\"")
    new_q = re.sub(r"^(query|search|搜索|q)[:：]\s*", "", new_q, flags=re.IGNORECASE)
    if not new_q or new_q == prior_query.strip():
        return ""
    return new_q




# ============================================================================
# 解析助手
# ============================================================================

def _parse_per_repo(text: str, known_names: list[str]) -> dict[str, str]:
    """从 PER_REPO 段抽出 `<full_name>: <verdict> — <reason>` 行。

    用 known_names 兜底:LLM 偶尔把 full_name 写错或漏写,只匹配真实存在的仓库名。
    """
    out: dict[str, str] = {}
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        upper = s.upper()
        if upper.startswith("PER_REPO"):
            in_block = True
            continue
        if in_block:
            if any(upper.startswith(k) for k in
                   ("RELEVANT_COUNT", "SUFFICIENT", "REASONING")):
                break
            # 尝试匹配:行里若包含已知 full_name,就以那个 name 为键
            for name in known_names:
                if name in s:
                    # 抽冒号后的内容当 verdict
                    after = s.split(name, 1)[1].lstrip(":：").strip()
                    if after:
                        out[name] = after
                    break
    return out


def _parse_sufficient(text: str) -> bool | None:
    for line in text.splitlines():
        s = line.strip().upper()
        if s.startswith("SUFFICIENT"):
            if "YES" in s:
                return True
            if "NO" in s:
                return False
    return None


def _parse_relevant_count(text: str, per_repo: dict[str, str]) -> int:
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("RELEVANT_COUNT"):
            m = re.search(r"(\d+)", s)
            if m:
                return int(m.group(1))
    # 兜底:从 per_repo 数"相关"的条数
    return sum(
        1 for value in per_repo.values()
        if value.upper().startswith("RELEVANT") or value.startswith("相关")
    )


def _parse_reason(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("REASONING"):
            # 拿冒号后面的内容
            after = s.split(":", 1)
            if len(after) == 2:
                return after[1].strip()
    return ""


# ============================================================================
# R4:基于负样本的 SearchPlan 重写(温和措辞,不强制避开)
# ============================================================================

_REFORMULATE_PLAN_PROMPT = """\
You rewrite GitHub search plans. Earlier rounds did not find enough relevant
Python libraries. Produce a new plan based on the attempted directions.

[USER TASK]
{capability}

[ATTEMPTED DIRECTIONS]
{tried_block}

[GUIDANCE]
The attempted plan may have used scenario-specific keywords, uncommon terms,
or too much literal context. You may retain a useful prior topic, but text_hint
or topics must change materially by using an established term, broader concept,
or implementation-library name.

[OUTPUT FORMAT - no explanation or Markdown]
TEXT: <one lowercase English term without quotes; use NONE if unknown>
TOPICS: <comma-separated lowercase topics, hyphenating phrases; at most {max_topics}>

[EXAMPLE]
Task: "Hash a password with bcrypt"
Attempted: text=password-hashing, topics=[password, hash]
Output:
TEXT: bcrypt
TOPICS: bcrypt, password-hashing, cryptography
"""


def reformulate_plan(
    capability: str,
    prior_plans: list[SearchPlan],
    prior_verdicts: list[RelevanceVerdict],
    llm: BaseLLM,
    *,
    max_topics: int = 4,
) -> SearchPlan:
    """R4:基于负样本的 plan 重写。返回新的 SearchPlan;empty 表示放弃。

    措辞温和(不强制避开),只是提示哪些方向已经试过没找到东西。
    """
    if not prior_plans:
        # 没有历史 → 退化:LLM 重新出一份
        from .planner import plan_search
        return plan_search(capability, llm, max_topics=max_topics)

    # 渲染"已试过"上下文
    lines: list[str] = []
    for i, p in enumerate(prior_plans, 1):
        tried_t = p.text_hint or "(none)"
        tried_topics = ", ".join(p.topics) if p.topics else "(none)"
        v = prior_verdicts[i - 1] if i - 1 < len(prior_verdicts) else None
        rel = f" → relevant={v.relevant_count}" if v else ""
        lines.append(f"  round {i}: text={tried_t!r}, topics=[{tried_topics}]{rel}")
    tried_block = "\n".join(lines)

    prompt = _REFORMULATE_PLAN_PROMPT.format(
        capability=capability.strip(),
        tried_block=tried_block,
        max_topics=max_topics,
    )
    resp = llm.generate([Message(role="user", content=prompt)])
    raw = (resp.text or "").strip()
    if not raw:
        return SearchPlan()

    new_plan = _parse_reformulate_output(raw, max_topics=max_topics)

    # 兜底:如果 LLM 给的和最近一轮**字面完全相同** → 放弃(避免死循环)
    last = prior_plans[-1]
    if (new_plan.text_hint == last.text_hint
            and new_plan.topics == last.topics):
        return SearchPlan()

    new_plan.sources["text_hint"] = "llm_reform"
    new_plan.sources["topics"] = "llm_reform"
    return new_plan


def _parse_reformulate_output(raw: str, *, max_topics: int) -> SearchPlan:
    """从 LLM 重写输出抽 TEXT 和 TOPICS 两行。"""
    plan = SearchPlan()
    for line in raw.splitlines():
        s = line.strip()
        upper = s.upper()
        if upper.startswith("TEXT:"):
            val = s.split(":", 1)[1].strip().strip("'\"`").lower()
            first = val.split()[0] if val else ""
            if first and first != "none" and len(first) >= 3:
                if re.match(r"^[a-z][a-z0-9\-_]{1,29}$", first):
                    plan.text_hint = first
        elif upper.startswith("TOPICS:"):
            val = s.split(":", 1)[1].strip()
            plan.topics = _parse_topics(val, max_topics=max_topics)
    return plan
