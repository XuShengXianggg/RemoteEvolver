"""
remoteevolver.evolve.workflow.record —— 工作流技能记录(SkillRecord)

对齐 FlowEvo 的技能记录 s=(f, σ, T, m, ℓ),但换成「工具调用」味:
  - f  可执行体   = call_template(参数化的工具调用序列)+ parameters(槽位)
  - σ  调用接口   = interface(槽位 -> 说明)
  - T  回放测试   = example_bindings(源任务的具体取值) + replay_expect(期望子串)
  - m  元数据      = task_pattern / source_task / guideline / 统计字段
  - ℓ  生命周期   = status(active|suppressed) + use/success/failure 计数 + contrastive

落盘:一技能一目录及 SKILL.md（兼容历史 JSON），由 SkillBank 管理。
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ContrastiveStats:
    """对比效用统计(curation 用):同一技能「用 vs 不用」的成败。

    本最小闭环只负责累计,真正的抑制逻辑下一版在 curation.py 里补。
    """
    guided_success: int = 0
    guided_total: int = 0
    unguided_success: int = 0
    unguided_total: int = 0

    def delta(self) -> float | None:
        """guided 成功率 - unguided 成功率;样本不足返回 None。"""
        if self.guided_total == 0 or self.unguided_total == 0:
            return None
        return (self.guided_success / self.guided_total
                - self.unguided_success / self.unguided_total)


@dataclass
class ReuseConstraints:
    """抽象复用边界；不包含源任务的具体路径、年份或答案。"""

    semantic_scope: str = ""
    hard_preconditions: list[str] = field(default_factory=list)
    soft_constraints: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    fallback_mode: str = "dynamic"          # conditioned | dynamic


@dataclass
class HybridTransition:
    """hybrid Skill 中必须由本轮 ReAct 完成的一段跨步骤衔接。"""

    after_step: int                          # -1 表示直接来自任务输入
    before_step: int
    operation: str
    input_contract: str
    output_contract: str
    instruction: str
    target_args: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)


@dataclass
class CapabilityPort:
    """A semantic whole-skill input or output, independent of tool arguments."""

    name: str
    description: str
    required: bool = True


@dataclass
class CapabilityInterface:
    """Whole-skill capability envelope used for routing compatibility."""

    inputs: list[CapabilityPort] = field(default_factory=list)
    outputs: list[CapabilityPort] = field(default_factory=list)


@dataclass
class HybridStep:
    """Per-tool split between deterministic wiring and ReAct-owned arguments."""

    step: int
    automatic_args: dict[str, Any] = field(default_factory=dict)
    react_args: list[str] = field(default_factory=list)


# 技能生命周期状态(仿 FlowEvo 的 SkillStatus,精简成 4 态):
#   draft       —— 已编译,还没过准入,不可被检索
#   active      —— 已准入,可被检索/直接执行
#   suppressed  —— curation 因稳定负迁移停用；artifact 本身仍保留
#   pruned      —— artifact 审计确认失效/不安全，不再参与
STATUS_DRAFT = "draft"
STATUS_ACTIVE = "active"
STATUS_SUPPRESSED = "suppressed"
STATUS_PRUNED = "pruned"

# 使用方式与生命周期正交：前者规定 Skill 如何参与求解，后者规定它当前是否可用。
USAGE_DIRECT = "direct"
USAGE_HYBRID = "hybrid"
USAGE_CONDITIONED_ONLY = "conditioned_only"
VALID_USAGE_MODES = frozenset({USAGE_DIRECT, USAGE_HYBRID, USAGE_CONDITIONED_ONLY})


@dataclass
class SkillRecord:
    skill_id: str
    task_pattern: str                       # 抽象描述,检索匹配用
    source_task: str                        # 编译来源的具体任务原文
    call_template: list[dict[str, Any]]     # [{"tool": "...", "args": {... 可含 "{slot}"}}]
    parameters: list[str]                   # 槽位名列表
    interface: dict[str, str]               # 槽位 -> 一句话说明(供 LLM 绑定)
    example_bindings: dict[str, Any]        # 源任务里这些槽位的具体取值(回放校验用)
    replay_expect: str                      # 用 example_bindings 重放时期望出现的子串
    source_replay_answer: str = ""           # 仅依据本次源任务重放产物重新合成的答案
    source_replay_reason: str = ""           # 准入验证结论，供审计和 Eval-driven 调优
    source_replay_steps: list[dict[str, Any]] = field(default_factory=list)
    cluster_key: str = ""                   # 同簇判定/去重键(task_pattern+工具序列+槽位数)
    guideline: str = ""                     # Layer-2 文本(skill-conditioned 时注入)
    abstract_stages: list[str] = field(default_factory=list)  # 纯语义高层阶段
    reuse_constraints: ReuseConstraints = field(default_factory=ReuseConstraints)
    capability_interface: CapabilityInterface = field(default_factory=CapabilityInterface)
    hybrid_steps: list[HybridStep] = field(default_factory=list)
    # Deprecated compatibility field. New persisted records do not include it.
    hybrid_transitions: list[HybridTransition] = field(default_factory=list)
    compiler_corrections: list[str] = field(default_factory=list)
    verification_level: int = 0             # 准入时通过的最高验证级别(见 verifier.V_*)
    direct_eligible: bool = False           # 是否够格直接执行(由验证级别决定)
    execution_enabled: bool = True          # Direct/Hybrid artifact 是否仍通过执行审计
    usage_mode: str = USAGE_CONDITIONED_ONLY  # direct|hybrid|conditioned_only
    status: str = STATUS_DRAFT              # draft|active|suppressed|pruned
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0           # 连续失败计数(curation 用)
    smoothed_utility: float = 1.0           # 指数平滑效用(0.5·prev+0.5·新);新技能乐观给 1.0
    audit_failure_count: int = 0
    last_audit_reason: str = ""
    last_audit_at: float = 0.0
    supersedes_skill_id: str = ""
    created_at: float = field(default_factory=time.time)
    contrastive: ContrastiveStats = field(default_factory=ContrastiveStats)

    # ---- 派生 -----------------------------------------------------------

    @property
    def utility(self) -> float:
        """平滑后的效用；用于检索重排与观测，不直接改变生命周期。"""
        return self.smoothed_utility

    @property
    def raw_utility(self) -> float:
        """success / use 原始命中率;没用过默认 1.0。"""
        return self.success_count / self.use_count if self.use_count else 1.0

    @property
    def active(self) -> bool:
        return self.status == STATUS_ACTIVE

    # ---- 工厂 -----------------------------------------------------------

    @staticmethod
    def new(
        *,
        task_pattern: str,
        source_task: str,
        call_template: list[dict[str, Any]],
        parameters: list[str],
        interface: dict[str, str],
        example_bindings: dict[str, Any],
        replay_expect: str,
        cluster_key: str = "",
        guideline: str = "",
        usage_mode: str = USAGE_DIRECT,
        reuse_constraints: ReuseConstraints | None = None,
        capability_interface: CapabilityInterface | None = None,
        hybrid_steps: list[HybridStep] | None = None,
        hybrid_transitions: list[HybridTransition] | None = None,
        compiler_corrections: list[str] | None = None,
        abstract_stages: list[str] | None = None,
    ) -> "SkillRecord":
        if usage_mode not in VALID_USAGE_MODES:
            raise ValueError(f"invalid usage_mode: {usage_mode!r}")
        semantic_only = usage_mode == USAGE_CONDITIONED_ONLY
        normalized_hybrid_steps = list(hybrid_steps or [])
        if not normalized_hybrid_steps and hybrid_transitions:
            normalized_hybrid_steps = _steps_from_legacy_transitions(
                hybrid_transitions, call_template,
            )
        return SkillRecord(
            skill_id=_make_id(task_pattern),
            task_pattern=task_pattern,
            source_task=source_task,
            call_template=[] if semantic_only else call_template,
            parameters=[] if semantic_only else parameters,
            interface={} if semantic_only else interface,
            example_bindings={} if semantic_only else example_bindings,
            replay_expect="" if semantic_only else replay_expect,
            cluster_key=cluster_key,
            guideline=guideline,
            abstract_stages=list(abstract_stages or []),
            reuse_constraints=reuse_constraints or ReuseConstraints(),
            capability_interface=capability_interface or CapabilityInterface(),
            hybrid_steps=normalized_hybrid_steps,
            hybrid_transitions=[],
            compiler_corrections=list(compiler_corrections or []),
            execution_enabled=not semantic_only,
            usage_mode=usage_mode,
        )

    # ---- 序列化 ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("hybrid_transitions", None)
        # The old prose slot interface is superseded by capability_interface.
        d.pop("interface", None)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SkillRecord":
        cs = d.get("contrastive", {}) or {}
        usage_mode = _load_usage_mode(d)
        semantic_only = usage_mode == USAGE_CONDITIONED_ONLY
        return SkillRecord(
            skill_id=d["skill_id"],
            task_pattern=d["task_pattern"],
            source_task=d.get("source_task", ""),
            call_template=[] if semantic_only else d.get("call_template", []),
            parameters=[] if semantic_only else d.get("parameters", []),
            interface={} if semantic_only else d.get("interface", {}),
            example_bindings={} if semantic_only else d.get("example_bindings", {}),
            replay_expect="" if semantic_only else d.get("replay_expect", ""),
            source_replay_answer=d.get("source_replay_answer", ""),
            source_replay_reason=d.get("source_replay_reason", ""),
            source_replay_steps=d.get("source_replay_steps", []),
            cluster_key=d.get("cluster_key", ""),
            guideline=d.get("guideline", ""),
            abstract_stages=[str(item) for item in (d.get("abstract_stages") or [])],
            reuse_constraints=_load_reuse_constraints(d.get("reuse_constraints")),
            capability_interface=_load_capability_interface(
                d.get("capability_interface")
            ),
            hybrid_steps=_load_hybrid_steps(
                d.get("hybrid_steps"), d.get("hybrid_transitions"),
                d.get("call_template", []),
            ),
            hybrid_transitions=[],
            compiler_corrections=[
                str(item) for item in (d.get("compiler_corrections") or [])
            ],
            verification_level=d.get("verification_level", 0),
            direct_eligible=d.get("direct_eligible", False),
            execution_enabled=d.get(
                "execution_enabled", usage_mode != USAGE_CONDITIONED_ONLY,
            ),
            usage_mode=usage_mode,
            status=d.get("status", STATUS_ACTIVE),
            use_count=d.get("use_count", 0),
            success_count=d.get("success_count", 0),
            failure_count=d.get("failure_count", 0),
            consecutive_failures=d.get("consecutive_failures", 0),
            smoothed_utility=d.get("smoothed_utility", 1.0),
            audit_failure_count=d.get("audit_failure_count", 0),
            last_audit_reason=d.get("last_audit_reason", ""),
            last_audit_at=d.get("last_audit_at", 0.0),
            supersedes_skill_id=d.get("supersedes_skill_id", ""),
            created_at=d.get("created_at", time.time()),
            contrastive=ContrastiveStats(**cs) if isinstance(cs, dict) else ContrastiveStats(),
        )


def _load_usage_mode(d: dict[str, Any]) -> str:
    """兼容旧记录：缺字段时根据既有 direct_eligible 保守迁移。"""
    mode = d.get("usage_mode")
    if mode in VALID_USAGE_MODES:
        return str(mode)
    return USAGE_DIRECT if d.get("direct_eligible", False) else USAGE_CONDITIONED_ONLY


def _load_reuse_constraints(value: Any) -> ReuseConstraints:
    if not isinstance(value, dict):
        return ReuseConstraints()
    names = ReuseConstraints.__dataclass_fields__
    return ReuseConstraints(**{k: v for k, v in value.items() if k in names})


def _load_hybrid_transitions(value: Any) -> list[HybridTransition]:
    if not isinstance(value, list):
        return []
    names = HybridTransition.__dataclass_fields__
    return [
        HybridTransition(**{k: v for k, v in item.items() if k in names})
        for item in value if isinstance(item, dict)
    ]


def _load_capability_interface(value: Any) -> CapabilityInterface:
    if not isinstance(value, dict):
        return CapabilityInterface()

    def ports(items: Any) -> list[CapabilityPort]:
        if not isinstance(items, list):
            return []
        out: list[CapabilityPort] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            if name and description:
                out.append(CapabilityPort(
                    name=name,
                    description=description,
                    required=bool(item.get("required", True)),
                ))
        return out

    return CapabilityInterface(
        inputs=ports(value.get("inputs")), outputs=ports(value.get("outputs")),
    )


def _load_hybrid_steps(
    value: Any, legacy: Any, call_template: list[dict[str, Any]],
) -> list[HybridStep]:
    if isinstance(value, list):
        out: list[HybridStep] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            step = item.get("step")
            automatic = item.get("automatic_args") or {}
            react = item.get("react_args") or []
            if (isinstance(step, int) and not isinstance(step, bool)
                    and isinstance(automatic, dict) and isinstance(react, list)):
                out.append(HybridStep(
                    step=step,
                    automatic_args=dict(automatic),
                    react_args=sorted({str(name) for name in react}),
                ))
        if out:
            return out
    return _steps_from_legacy_transitions(
        _load_hybrid_transitions(legacy), call_template,
    )


def _steps_from_legacy_transitions(
    transitions: list[HybridTransition], call_template: list[dict[str, Any]],
) -> list[HybridStep]:
    react_by_step: dict[int, set[str]] = {}
    for item in transitions:
        react_by_step.setdefault(item.before_step, set()).update(item.target_args)
    out: list[HybridStep] = []
    for index, step in enumerate(call_template or []):
        react = react_by_step.get(index, set())
        automatic = {
            str(name): value for name, value in (step.get("args") or {}).items()
            if name not in react
        }
        if react or automatic:
            out.append(HybridStep(
                step=index, automatic_args=automatic, react_args=sorted(react),
            ))
    return out


# ---- 内部 ---------------------------------------------------------------

_SLUG = re.compile(r"[^a-z0-9]+")


def _make_id(task_pattern: str) -> str:
    """skill_<slug>_<短uuid>;slug 取 task_pattern 里的 ascii 词。

    task_pattern 全是中文(ascii 词为空)时不再 fallback 成 'skill'(那会拼出
    难看的 skill_skill_...),直接 skill_<uuid>。
    """
    slug = _SLUG.sub("_", (task_pattern or "").lower()).strip("_")
    slug = "_".join(p for p in slug.split("_") if p)
    slug = "_".join(slug.split("_")[:5])
    prefix = f"skill_{slug}_" if slug else "skill_"
    return f"{prefix}{uuid.uuid4().hex[:8]}"
