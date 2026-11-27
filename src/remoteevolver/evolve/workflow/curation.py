"""
remoteevolver.evolve.workflow.curation —— 技能淘汰(skill curation)

借鉴 FlowEvo 两处机制,按本项目裁剪成一套:

  1. 对比抑制(策略层范式):同一技能「被用(guided)」vs「同簇任务没用它(unguided)」
     的成功率差 = contrastive delta。delta 持续为负(且样本够)→ 判负迁移 → 抑制。
  2. 绝对效果兜底:完整 Skill 调用成功率过低或连续失败 → 抑制。

普通任务效果永远不 prune。prune 只由 pipeline 的 artifact 审计入口处理。

入口 record_outcome():每个 episode 结束,外层把"这次用没用某技能、成没成"喂进来,
本模块更新统计 + 平滑效用 + 对比计数,然后跑一遍治理规则,返回触发的动作(供日志)。

只改 SkillRecord 的字段,不落盘;落盘由调用方(pipeline)负责。
"""

from __future__ import annotations

from .config import WorkflowConfig, DEFAULT
from .record import (
    SkillRecord,
    STATUS_ACTIVE,
    STATUS_SUPPRESSED,
)

# mode 取值:
#   direct / conditioned —— 实际用了技能 → guided 样本
#   unguided             —— 同簇任务这次没用该技能(基线) → unguided 样本
GUIDED_MODES = ("direct", "hybrid", "conditioned")


def record_outcome(
    skill: SkillRecord,
    mode: str,
    success: bool,
    cfg: WorkflowConfig = DEFAULT,
) -> str | None:
    """更新一个技能的统计并跑治理。返回 suppress 动作或 None。

    注意:unguided 样本不计 use_count(技能没被使用),只进对比基线。
    """
    if mode in GUIDED_MODES:
        skill.use_count += 1
        if success:
            skill.success_count += 1
            skill.consecutive_failures = 0
        else:
            skill.failure_count += 1
            skill.consecutive_failures += 1
        # 指数平滑效用
        a = cfg.smoothing_alpha
        skill.smoothed_utility = a * skill.smoothed_utility + (1 - a) * skill.raw_utility
        # 对比:guided
        skill.contrastive.guided_total += 1
        if success:
            skill.contrastive.guided_success += 1
    elif mode == "unguided":
        skill.contrastive.unguided_total += 1
        if success:
            skill.contrastive.unguided_success += 1

    return govern(skill, cfg)


def govern(skill: SkillRecord, cfg: WorkflowConfig = DEFAULT) -> str | None:
    """对一个 active 技能跑淘汰规则;命中则就地改 status,返回动作名。"""
    if skill.status != STATUS_ACTIVE:
        return None

    cs = skill.contrastive
    # ---- 抑制(suppress):多条触发任一即停 ----
    # (a) 对比 delta 路径
    delta = cs.delta()
    if (delta is not None
            and cs.guided_total >= cfg.contrastive_min_guided
            and cs.unguided_total >= cfg.contrastive_min_unguided
            and delta < cfg.contrastive_harm_threshold):
        skill.status = STATUS_SUPPRESSED
        return "suppress:contrastive"

    # (b) guided 成功率过低路径
    if cs.guided_total >= cfg.low_success_min_guided:
        gsr = cs.guided_success / cs.guided_total
        if gsr < cfg.low_success_rate_threshold:
            skill.status = STATUS_SUPPRESSED
            return "suppress:low_success"

    # (c) 连续失败。这里只接收上层已经排除缺工具、基础设施和未验证结果的
    # 完整 Skill 调用结果。
    if skill.consecutive_failures >= cfg.suppress_consecutive_failures:
        skill.status = STATUS_SUPPRESSED
        return "suppress:consecutive_failures"

    return None
