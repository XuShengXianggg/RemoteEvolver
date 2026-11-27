"""
remoteevolver.evolve.workflow.verifier —— 分层可插拔验证器(难点1 + 难点6 的地基)

"工具跑通 ≠ 任务完成"。这里把"是否真成功"做成**分层、按可信度排序**的判定,
而不是一个 bool。每一级强度不同,技能记录会存"它当初过的是哪一级"(verification_level),
路由据此决定能不能直接复用。

级别(按可信度从弱到强):
  0 NONE        什么都没过
  1 TOOL_OK     工具链无报错(必要门,最弱)
  2 JUDGED      LLM 裁判判"完成了"(软信号,带置信度;建议用独立模型避免盲点)
  3 STRUCTURAL  确定性结构校验过(输出非空 / 含期望子串 / schema 合法)
  4 ORACLE      命中数据集给的预期输出(**后续训练微调时最强的信号**)
  5 DOMAIN      领域验证器过(遥感:CRS/分辨率/波段/数值范围/mask 等)

设计要点:
  - verify() 从弱到强逐级试,记录"通过的最高级别";任一**硬校验失败**
    (工具报错 / oracle 不匹配 / 领域规则不过 / 裁判判否)→ 整体 passed=False。
  - expected 与 domain_validators 现在留空插口;接数据集 / 遥感时填进来即可,
    框架不用改。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ...llm.base import BaseLLM, Message
from ._util import extract_json
from .dataflow import is_tool_error_result

# 级别常量
V_NONE = 0
V_TOOL_OK = 1
V_JUDGED = 2
V_STRUCTURAL = 3
V_ORACLE = 4
V_DOMAIN = 5

LEVEL_NAME = {
    V_NONE: "NONE", V_TOOL_OK: "TOOL_OK", V_JUDGED: "JUDGED",
    V_STRUCTURAL: "STRUCTURAL", V_ORACLE: "ORACLE", V_DOMAIN: "DOMAIN",
}


@dataclass
class VerifyContext:
    task: str
    results: list[str] = field(default_factory=list)   # 各步工具输出
    final: str = ""                                     # 最终答案(合成的或 ReAct 的)
    expect_substring: str = ""                          # 轻量结构期望(命中即 STRUCTURAL)
    expected: Any = None                                # 数据集预期输出(命中即 ORACLE)
    # Skill 源任务重放只允许“本次重放新合成的答案”命中 oracle，不能让任一
    # 中间结果或原任务旧答案代替它。其他验证场景保持原有宽松行为。
    oracle_final_only: bool = False


@dataclass
class VerifyResult:
    passed: bool
    level: int = V_NONE                  # 通过的最高级别
    confidence: float = 1.0              # JUDGED 级才 < 1
    reason: str = ""

    @property
    def level_name(self) -> str:
        return LEVEL_NAME.get(self.level, str(self.level))


# 领域验证器签名:吃 ctx,返回 (通过?, 说明)。遥感的 CRS/波段/数值范围检查以后注册进来。
DomainValidator = Callable[[VerifyContext], "tuple[bool, str]"]


def verify(
    ctx: VerifyContext,
    *,
    judge_llm: BaseLLM | None = None,
    domain_validators: tuple[DomainValidator, ...] = (),
) -> VerifyResult:
    """从弱到强逐级验证,返回通过的最高级别;任一硬校验失败则 passed=False。"""
    level = V_NONE

    # L0 工具链无报错(必要门)
    for r in ctx.results:
        if is_tool_error_result(r):
            return VerifyResult(False, V_NONE, reason=f"工具报错: {r[:120]}")
    level = V_TOOL_OK

    # L3 结构校验(确定性)。FlowEvo 对齐:功能正确性看"跑通 + 结果类型/结构一致",
    # **不强求重算出一模一样的精确值**。所以 expect_substring 命中 → 升 STRUCTURAL
    # (加分项);**不命中不再硬失败**(精确浮点/文件名做精确子串太脆,是我们之前
    # 的脆判据)。仍保留"输出为空"这种确定性硬失败。
    if ctx.final is not None and ctx.final.strip() == "" and not ctx.results:
        return VerifyResult(False, level, reason="输出为空")
    if ctx.expect_substring:
        hay = "\n".join(ctx.results) + "\n" + (ctx.final or "")
        if ctx.expect_substring in hay:
            level = max(level, V_STRUCTURAL)

    # L4 预期输出(数据集 oracle;现在 expected=None 时跳过)
    if ctx.expected is not None:
        if _match_expected(ctx, ctx.expected):
            level = max(level, V_ORACLE)
        else:
            return VerifyResult(False, level, reason="与预期输出不匹配(oracle)")

    # L2 LLM 裁判(软信号;只在还没拿到更强的确定性/ oracle 证据时才值得花这次调用)
    if judge_llm is not None and level < V_STRUCTURAL:
        ok, conf, why = _llm_judge(ctx, judge_llm)
        if not ok:
            return VerifyResult(False, level, confidence=conf,
                                reason=f"裁判判否: {why}")
        level = max(level, V_JUDGED)
        # 裁判通过但置信不满时,如实带回(供准入/路由按置信度收紧)
        if level == V_JUDGED:
            return _finish(ctx, level, conf, "裁判通过", domain_validators)

    return _finish(ctx, level, 1.0, "通过", domain_validators)


def _finish(ctx, level, conf, reason, domain_validators) -> VerifyResult:
    # L5 领域验证器(注册了才跑;遥感以后接)
    for dv in domain_validators:
        ok, why = dv(ctx)
        if not ok:
            return VerifyResult(False, level, confidence=conf,
                                reason=f"领域验证失败: {why}")
        level = max(level, V_DOMAIN)
    return VerifyResult(True, level, confidence=conf, reason=reason)


# ---- 内部 ---------------------------------------------------------------

def _match_expected(ctx: VerifyContext, expected: Any) -> bool:
    """预期输出匹配(oracle)。命中 final 或任一工具结果即算对。"""
    if oracle_match(ctx.final or "", expected):
        return True
    if ctx.oracle_final_only:
        return False
    return any(oracle_match(r, expected) for r in ctx.results)


# 答案归一化:思路借鉴 FlowEvo(Apache-2.0)code_math 的 _normalize_answer——
# 去空白/逗号/货币/百分号/末尾标点 + 小写 + 数字去尾零,让"3,000"=="3000"、"42."=="42"。
_STRIP_CHARS = " \t\n\r。.,，;；:：!！?？\"'`()（）[]【】"


def _normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(",", "").replace("$", "").replace("¥", "").replace("%", "")
    s = "".join(s.split())                       # 去掉所有空白
    s = s.strip(_STRIP_CHARS)
    # 纯数字去尾零:42.0 -> 42
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def oracle_match(pred: str, expected: Any) -> bool:
    """归一化后:相等 或 预期是预测的子串(应对"答案是 42"这类包裹)。"""
    matcher = getattr(expected, "matches", None)
    if callable(matcher):
        try:
            return bool(matcher(pred))
        except Exception:  # noqa: BLE001
            return False
    np = _normalize(pred)
    ne = _normalize(expected if isinstance(expected, str) else str(expected))
    if not ne:
        return False
    return np == ne or ne in np


_JUDGE_PROMPT = """\
判断下面的"答案"是否**真正完成了任务**(不是看起来合理,而是确实满足任务目标)。
Write the reason and every other generated free-text value in English.

任务:
{task}

工具执行得到的中间结果:
{results}

最终答案:
{final}

只输出一个 JSON(不要解释/代码围栏):
{{"ok": true 或 false, "confidence": 0到1之间的小数, "reason": "一句话理由"}}
"""


def _llm_judge(ctx: VerifyContext, judge_llm: BaseLLM) -> "tuple[bool, float, str]":
    joined = "\n".join(f"[{i}] {r[:400]}" for i, r in enumerate(ctx.results, 1)) or "(无)"
    prompt = _JUDGE_PROMPT.format(
        task=ctx.task.strip(), results=joined, final=(ctx.final or "").strip()[:600] or "(无)",
    )
    prompt += """

审核约束:
- 只检查任务要求的输出是否齐全,以及最终答案是否有工具结果支持。
- 比例和百分比须按等价形式理解,例如 0.25 等于 25%。
- 不得添加任务或工具证据中没有明确给出的统计规律、物理规律或领域假设。
- 判否必须指出一个明确矛盾,并引用任务、工具结果或最终答案中的原文证据。
- 在原有 ok/confidence/reason 字段外返回 failure_type、evidence_source、evidence_quote。
- failure_type 只能是 missing_required_output、tool_final_mismatch、unit_conversion_error、
  explicit_task_rule_violation 或 none；ok=true 时使用 none。
- evidence_source 只能是 task、results、final 或 none；evidence_quote 必须逐字来自对应来源。
"""
    resp = judge_llm.generate([Message(role="user", content=prompt)], tools=None)
    data = extract_json(resp.text)
    if data is None:
        # 裁判没给出可解析结果 → 保守判否(宁可不准入也别放进坏技能)
        return False, 0.0, "裁判未返回合法 JSON"
    ok = bool(data.get("ok", False))
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    reason = str(data.get("reason", ""))[:200]
    if not ok:
        failure_type = str(data.get("failure_type", ""))
        evidence_source = str(data.get("evidence_source", ""))
        evidence_quote = str(data.get("evidence_quote", "")).strip()
        allowed_types = {
            "missing_required_output", "tool_final_mismatch",
            "unit_conversion_error", "explicit_task_rule_violation",
        }
        sources = {
            "task": ctx.task,
            "results": "\n".join(ctx.results),
            "final": ctx.final or "",
        }
        source_text = sources.get(evidence_source, "")
        if (failure_type not in allowed_types or not evidence_quote
                or evidence_quote not in source_text):
            return False, 0.0, "裁判否定缺少可核验的原文证据"
    return ok, max(0.0, min(1.0, conf)), reason
