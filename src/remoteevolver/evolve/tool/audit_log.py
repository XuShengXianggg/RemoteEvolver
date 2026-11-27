"""
remoteevolver.evolve.tool.audit_log —— 工具自进化的审计日志

记录以下事件到 `data/evolve_log.jsonl`(每行一个 JSON 对象):

  - {"event": "search_round", ...}    M1 搜索的每一轮(query + LLM 相关性裁定)
  - {"event": "repo_assess",  ...}    M1 后的 5 维仓库评估(LLM 评分 + 综合分 + 阈值)
  - {"event": "evolve",       ...}    每次进化流水线跑完(成功 / 失败)

为什么用 JSONL 而不是单一 JSON 数组:
  - 追加便宜(open mode='a' 一行就完)
  - 不会因为某次写入崩坏整个文件
  - grep / jq / 自己 Python 解析都方便

不写到 data/long_term.json:
  - 那个文件是 LLM 反思机制的输入(execution_record / experience),由 reflector 维护
  - 审计日志是给人看的运行记录,职责分开
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from ...infra.paths import runtime_path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import EvolveResult
    from .repo_assess import AssessedRepo
    from .search import SearchRound


DEFAULT_LOG_PATH = runtime_path("logs/tool-evolution.jsonl")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())




def append_tool_validation_log(
    *,
    task_id: str,
    task: str,
    stage: str,
    candidate_tool: str,
    capability_contract: dict[str, Any],
    match_result: dict[str, Any],
    decision: str,
    details: dict[str, Any] | None = None,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    """Record selection, certification, and per-call contract evidence."""
    entry = {
        "event": "tool_validation",
        "ts": time.time(),
        "iso": _now_iso(),
        "task_id": task_id,
        "task": task,
        "stage": stage,
        "candidate_tool": candidate_tool,
        "capability_contract": capability_contract,
        "match_result": match_result,
        "hard_conflicts": match_result.get("hard_conflicts", []),
        "decision": decision,
        "details": details or {},
    }
    _write(log_path, entry)
    return entry


def append_search_round_log(
    capability: str,
    search_round: "SearchRound",
    *,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    """记录一次 M1 搜索循环的一轮(query + 命中数 + LLM 相关性裁定)。"""
    v = search_round.verdict
    entry: dict[str, Any] = {
        "event": "search_round",
        "ts": time.time(),
        "iso": _now_iso(),
        "capability": capability,
        "round_idx": search_round.round_idx,
        "query": search_round.query,
        "raw_repo_count": search_round.raw_repo_count,
        "verdict": (
            {
                "sufficient": v.sufficient,
                "relevant_count": v.relevant_count,
                "reasoning": v.reasoning,
                "per_repo": v.per_repo,
            }
            if v is not None else None
        ),
    }
    _write(log_path, entry)
    return entry


def append_repo_assess_log(
    capability: str,
    assessed: list["AssessedRepo"],
    *,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    """记录一次 M1 后的 5 维仓库评估(LLM 评分 + 综合分 + 阈值检查)。

    assessed 为空列表也写一条(记录"评估被跳过"),便于事后分析。
    """
    entry: dict[str, Any] = {
        "event": "repo_assess",
        "ts": time.time(),
        "iso": _now_iso(),
        "capability": capability,
        "count": len(assessed),
        "items": [
            {
                "full_name": a.repo.full_name,
                "stars": a.repo.stars,
                "semantic": a.score.semantic_relevance,
                "structural": a.score.structural_completeness,
                "toolization": a.score.toolization_feasibility,
                "community": a.score.community_reliability,
                "env_complexity": a.score.environment_complexity,
                "composite": a.score.composite,
                "passes_thresholds": a.score.passes_thresholds,
                "reasoning": a.score.reasoning,
            }
            for a in assessed
        ],
    }
    _write(log_path, entry)
    return entry


def append_evolve_log(
    result: "EvolveResult",
    *,
    started_at: float | None = None,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    """记录一次 evolve_from_* 调用的完整轨迹。"""
    end_ts = time.time()
    start_ts = started_at if started_at is not None else end_ts

    entry: dict[str, Any] = {
        "event": "evolve",
        "ts": end_ts,
        "iso": _now_iso(),
        "elapsed_sec": round(end_ts - start_ts, 2),
        "capability": result.capability,
        "repo": result.repo,
        "success": result.success,
        "failure_stage": result.failure_stage,
        "failure_kind": result.failure_kind,
        "error": (result.error or "")[:500],
        # M1 搜索阶段(capability 模式独有)
        "chosen_query": result.chosen_query,
        "candidates": [
            {
                "rank": i,
                "full_name": sr.repo.full_name,
                "stars": sr.repo.stars,
                "language": sr.repo.language,
                "license": sr.repo.license,
                "score": round(sr.score, 2),
                "score_reason": sr.score_reason,
            }
            for i, sr in enumerate(result.search_candidates, 1)
        ],
        # clone + install
        "cloned": result.cloned is not None,
        "pip_installed": (result.cloned.pip_installed if result.cloned else False),
        "package_name": (result.signals.package_name if result.signals else ""),
        # 测试参数(LLM 自动出或显式传)
        "test_args": result.test_args,
        "capability_contract": result.capability_contract,
        "engineering_verification": result.certification,
        "certification": result.certification,
        # 重试历史
        "attempts": [
            {
                "i": a.iteration,
                "test_stage": a.test_result.stage,
                "test_success": a.test_result.success,
                "function_name": a.test_result.function_name,
                "output_preview": (a.test_result.output or "")[:200],
                "traceback_tail": (a.test_result.traceback or "")[-400:],
            }
            for a in result.attempts
        ],
        "total_attempts": len(result.attempts),
        "repository_attempts": result.repository_attempts,
        # 注册结果
        "registered_as": (result.registration.tool_name if result.registration else ""),
        "registered_path": (str(result.registration.file_path) if result.registration else ""),
    }
    _write(log_path, entry)
    return entry


def _write(log_path: Path | str, entry: dict[str, Any]) -> None:
    """追加一行 JSON 到日志文件。父目录不存在会自动创建。"""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        _json_safe(entry),
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    )
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


# ---- 读取 / 查询(给 examples/show_evolve_log.py 用)----------------------

def load_log(log_path: Path | str = DEFAULT_LOG_PATH) -> list[dict[str, Any]]:
    """读全部条目。文件不存在返回空列表。"""
    p = Path(log_path)
    if not p.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            # 跳过损坏行,日志诚信原则:不为单点失败拒绝读其余
            continue
    return entries
