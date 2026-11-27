"""
remoteevolver.evolve.tool.pipeline —— 端到端自进化编排

两个入口:
  evolve_from_repo(repo, capability, test_args, llm, ...) -> EvolveResult
      指定仓库 + 显式 test_args 的精确控制版本。

  evolve_from_capability(capability, llm, ...) -> EvolveResult
      只给一句话需求。内部走 M1 搜索挑 top-1 仓库,LLM 自己产出 test_args,
      然后调用 evolve_from_repo。**这是 ReAct 自动触发将来要用的入口**。

流程:
  1. clone + pip install      cloner.clone_and_install
  2. 收集 README/包名信号       wrapper_gen.collect_repo_signals
  3. LLM 写第一版 wrapper      wrapper_gen.generate_wrapper
  4. (capability 模式)         test_args_gen.generate_test_args
  5. 冒烟测试                 test_runner.run_smoke_test
  6. 失败 → LLM 修 → 再测      最多 max_fix_attempts 次
  7. 通过 → 注册              registrar.register_wrapper

设计原则:
  - 任何一步失败都尽量在结果里如实回传 (failure_stage / error),不抛
    异常给上层,**便于 CLI / ReAct 决定下一步**
  - 中间产物(每次 LLM 给的 wrapper、每次测试结果)都收集起来,事后回放方便
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import time

from ...llm.base import BaseLLM
from ...infra.errors import is_transient_external_error
from ...console import safe_print as print
from .audit_log import (
    append_evolve_log,
    append_repo_assess_log,
    append_search_round_log,
)
from .capability_contract import (
    CapabilityContract,
    function_schema_from_wrapper,
)
from .checkpoint import (
    clear_repository_checkpoint,
    load_repository_checkpoint,
    save_repository_checkpoint,
)
from .cloner import ClonedRepo, CloneError, clone_and_install
from .code_audit import audit_wrapper_calls, format_issues_for_llm
from .fixer import fix_wrapper, fix_wrapper_for_consistency
from .invariants import check_output_invariants
from .registrar import RegistrationResult, register_wrapper
from .registration_probe import probe_registered_tool
from .repo_assess import (
    AssessedRepo,
    DEFAULT_MAX_CANDIDATES_TO_ASSESS,
    assess_candidates,
    enrich_with_readme,
    pick_best,
)
from .search import ScoredRepo, SearchRound, find_candidates
from .test_args_gen import generate_test_args
from .test_runner import TestResult, run_smoke_test
from .tool_certification import CertificationRecord, CertificationStore
from .wrapper_gen import (
    RepoSignals,
    RepositoryCapabilityUnsupported,
    collect_repo_signals,
    generate_wrapper,
)


@dataclass
class AttemptRecord:
    """每一轮 (生成 / 修复 → 测试) 的留痕。"""
    iteration: int                # 1 = 初次生成,2+ = 第几次修复
    wrapper_code: str
    test_result: TestResult


@dataclass
class EvolveResult:
    """端到端结果。"""
    success: bool
    capability: str
    repo: str
    failure_stage: str = ""       # "" 表示成功;"search"/"clone"/"signal"/"test_args"/"test_failed"/"max_attempts"/"register"
    error: str = ""               # 人类可读的错误概要
    failure_kind: str = ""        # infrastructure / repository / generation / validation / registration
    # 各阶段产物(供事后回放/诊断)
    search_candidates: list[ScoredRepo] = field(default_factory=list)  # 仅 capability 模式有
    chosen_query: str = ""        # 仅 capability 模式有:LLM 实际命中的 GitHub query
    cloned: ClonedRepo | None = None
    signals: RepoSignals | None = None
    test_args: dict[str, Any] = field(default_factory=dict)            # 实际用的测试参数
    attempts: list[AttemptRecord] = field(default_factory=list)
    registration: RegistrationResult | None = None
    repository_attempts: list[dict[str, Any]] = field(default_factory=list)
    capability_contract: dict[str, Any] = field(default_factory=dict)
    certification: dict[str, Any] = field(default_factory=dict)


def _should_try_next_repository(result: EvolveResult) -> bool:
    """Only repository-attributable failures justify paying for another clone/install.

    Infrastructure failures retain the exact repository checkpoint. A real-call MCP
    probe failure or an explicit repository rejection may use the one backup repo.
    Wrapper repair/validation exhaustion stops here instead of nesting another full
    repository evolution attempt.
    """
    return not result.success and (
        result.failure_kind == "repository"
        or result.failure_stage.startswith("register_probe")
    )


def _is_repairable_probe_failure(exc: Exception) -> bool:
    """Identify failures produced after discovery, during the real MCP call."""
    message = str(exc).lower()
    return isinstance(exc, TimeoutError) or any(marker in message for marker in (
        "real mcp invocation failed",
        "mcp registration probe",
        "tool call timed out",
    ))


# ---- 主入口 -------------------------------------------------------------

def evolve_from_repo(
    repo: str,
    capability: str,
    test_args: dict[str, Any] | None,
    llm: BaseLLM,
    *,
    max_fix_attempts: int = 2,
    force_clone: bool = False,
    overwrite_registered: bool = False,
    verbose: bool = True,
    sample_inputs: list[str] | None = None,
    prior_outputs: list[str] | None = None,
    capability_contract: CapabilityContract | dict[str, Any] | None = None,
    _skip_audit_log: bool = False,  # 内部用:evolve_from_capability 调用时设 True 避免重复日志
) -> EvolveResult:
    """从一个指定的 GitHub 仓库,自动产出并注册一个 remoteevolver 工具。

    Args:
        repo:             "owner/name" 或 https URL。
        capability:       一句话描述这个工具要做什么(进 prompt 用)。
        test_args:        冒烟测试参数 dict;键 = LLM 生成函数的形参名。
                          **传 None 时自动让 LLM 生成**(根据 wrapper 代码 + README)。
        llm:              LLM 客户端(用来生成 wrapper 和修复)。
        max_fix_attempts: 初版加修复后的总测试次数,默认 2(最多修复一次)。
        force_clone:      True 时把已存在的仓库目录删掉重 clone。
        overwrite_registered: True 时允许覆盖 remoteevolver/tools/ 已有同名文件。
        verbose:          True 时每个阶段打印一行进度。

    Returns:
        EvolveResult。success=False 时看 failure_stage 判断哪一步炸的。
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"[evolve] {msg}", flush=True)

    started_at = time.time()
    result = EvolveResult(success=False, capability=capability, repo=repo)

    def _finalize(r: EvolveResult) -> EvolveResult:
        """收尾:写一条 evolve 日志(_skip_audit_log=True 时跳过)然后返回 r。"""
        if not _skip_audit_log:
            try:
                append_evolve_log(r, started_at=started_at)
            except Exception as e:  # noqa: BLE001
                # 日志失败不能阻断主流程
                log(f"  ! audit log 写入失败(忽略): {type(e).__name__}: {e}")
        return r

    def _finish_wrapper_failure(
        exc: Exception,
        *,
        default_stage: str,
        context: str,
    ) -> EvolveResult:
        """Classify an explicit capability rejection as a repository failure."""
        if isinstance(exc, RepositoryCapabilityUnsupported):
            result.failure_stage = "repository_unsupported"
            result.failure_kind = "repository"
            result.error = (
                f"selected repository cannot supply the core capability: {exc.reason}"
            )
            log(f"  ✗ repository unsupported: {exc.reason}")
        else:
            result.failure_stage = default_stage
            result.failure_kind = (
                "infrastructure" if is_transient_external_error(exc) else "generation"
            )
            result.error = f"{context}: {type(exc).__name__}: {exc}"
        return _finalize(result)

    # ---- step 1: clone + pip install
    log(f"step 1/6 clone + install: {repo}")
    try:
        cloned = clone_and_install(repo, force=force_clone)
    except CloneError as e:
        result.failure_stage = "clone"
        result.failure_kind = (
            "infrastructure" if is_transient_external_error(e) else "repository"
        )
        result.error = str(e)
        log(f"  ✗ clone failed: {e}")
        return _finalize(result)
    result.cloned = cloned
    log(f"  ✓ clone -> {cloned.repo_dir}")
    if cloned.pip_installed:
        log(f"  ✓ pip install -e ok (package: {Path(cloned.repo_dir).name})")
    else:
        # 不立即失败 —— wrapper 可能能 raw import,或者 LLM 应对
        log(f"  ! pip install failed/skipped: {cloned.install_log_tail[:200]!r}")

    # ---- step 2: 收集信号(README + 包名 + C-1 AST 公开 API 扫描)
    log("step 2/6 collect repo signals (README / package name / AST public API)")
    try:
        signals = collect_repo_signals(
            cloned.repo_dir,
            cloned.full_name,
            scan_code=True,
            write_report_to=cloned.base_dir,  # → data/external_tools/<slug>/code_report.json
        )
    except Exception as e:  # noqa: BLE001
        result.failure_stage = "signals"
        result.failure_kind = "repository"
        result.error = f"collect signals failed: {type(e).__name__}: {e}"
        return _finalize(result)
    result.signals = signals
    api = signals.code_api
    api_brief = (
        f"  api found={api.found} funcs={api.total_functions()} classes={api.total_classes()} "
        f"exported={len(api.exported_names)}"
        if api else "  api=(skipped)"
    )
    log(f"  ✓ package={signals.package_name}  readme={len(signals.readme_excerpt)} chars")
    log(api_brief)

    # 能力描述用于指导生成、工具选择和后续接口适配，不再承担注册认证。
    # 无效或缺失的描述不得阻止一个通过工程检查的 wrapper 注册。
    contract: CapabilityContract | None = None
    if capability_contract is not None:
        try:
            contract = (
                capability_contract
                if isinstance(capability_contract, CapabilityContract)
                else CapabilityContract.from_dict(capability_contract)
            )
        except (TypeError, ValueError) as exc:
            log(f"  ! capability description invalid; continue without it: {exc}")
    result.capability_contract = contract.to_dict() if contract is not None else {}

    # ---- step 3: 初次生成 wrapper
    log("step 3/6 LLM generating initial wrapper...")
    try:
        code = generate_wrapper(
            capability, signals, llm,
        )
    except Exception as e:  # noqa: BLE001
        return _finish_wrapper_failure(
            e,
            default_stage="wrapper_generation",
            context="wrapper generation failed",
        )
    log(f"  ✓ got {len(code)} chars of code")

    # ---- step 4: 准备 test_args(显式传入或 LLM 自动生成)
    if test_args is None:
        log("step 4/6 LLM auto-generating test_args from wrapper + README...")
        if sample_inputs:
            log(f"  · 用任务真实输入样例 {len(sample_inputs)} 个约束文件型参数")
        if prior_outputs:
            log(f"  · 链上游产出 {len(prior_outputs)} 个 → 优先做输入(集成冒烟)")
        try:
            test_args = generate_test_args(
                capability, code, signals, llm,
                sample_inputs=sample_inputs, prior_outputs=prior_outputs,
            )
        except Exception as e:  # noqa: BLE001
            result.failure_stage = "test_args"
            result.failure_kind = (
                "infrastructure" if is_transient_external_error(e) else "generation"
            )
            result.error = str(e)
            log(f"  ✗ test_args generation failed: {e}")
            return _finalize(result)
        log(f"  ✓ test_args = {test_args}")
    else:
        log(f"step 4/6 using explicit test_args = {test_args}")
    result.test_args = dict(test_args)

    # ---- step 5: (C-2 一致性校验)+ 冒烟测试 + 修复 循环
    # 每一轮先做一致性校验,有幻觉 API 就让 LLM 直接定向修,不浪费冒烟测试
    # 一致性 OK → 跑冒烟测试,失败再用 traceback 修
    # 一致性 修 / 冒烟测试 修 都计入同一个 max_fix_attempts 预算
    for attempt in range(1, max_fix_attempts + 1):
        # 5a: 一致性校验(C-2)。signals.code_api 未知 / found=False 时返回空 → 跳过
        try:
            issues = audit_wrapper_calls(code, signals.code_api)
        except Exception as e:  # noqa: BLE001
            result.failure_stage = "code_audit"
            result.failure_kind = "validation"
            result.error = f"wrapper audit failed: {type(e).__name__}: {e}"
            return _finalize(result)
        unused_repository = next(
            (
                issue
                for issue in issues
                if issue.kind == "selected_repository_not_used"
            ),
            None,
        )
        if unused_repository is not None:
            return _finish_wrapper_failure(
                RepositoryCapabilityUnsupported(
                    "generated wrapper imports the selected repository but never uses it"
                ),
                default_stage="code_audit",
                context="selected repository use check failed",
            )
        if issues:
            log(f"step 5/6 attempt {attempt}/{max_fix_attempts} "
                f"× consistency CHECK: {len(issues)} hallucinated symbol(s)")
            for iss in issues[:5]:
                log(f"    - {iss.kind}: {iss.symbol}"
                    + (f"  suggest: {iss.suggestion}" if iss.suggestion else ""))
            # 用 TestResult.stage='consistency' 留痕,沿用现有 AttemptRecord 结构
            synthetic_tr = TestResult(
                success=False,
                stage="consistency",
                function_name="",
                output="",
                traceback=format_issues_for_llm(issues),
            )
            result.attempts.append(
                AttemptRecord(iteration=attempt, wrapper_code=code, test_result=synthetic_tr)
            )
            if attempt == max_fix_attempts:
                result.failure_stage = "consistency"
                result.error = (
                    f"{max_fix_attempts} 次重试后 wrapper 仍引用 {len(issues)} 个不存在的 API"
                )
                return _finalize(result)
            log(f"  → asking LLM to fix consistency (skip smoke test this round)")
            try:
                code = fix_wrapper_for_consistency(
                    code, issues, capability, signals, llm,
                )
            except Exception as e:  # noqa: BLE001
                return _finish_wrapper_failure(
                    e,
                    default_stage="wrapper_fix",
                    context="consistency fix failed",
                )
            continue  # 重新进 audit;不消耗冒烟测试

        # 5b: 一致性 OK,跑冒烟测试
        log(f"step 5/6 attempt {attempt}/{max_fix_attempts} smoke test")
        try:
            tr = run_smoke_test(code, test_args)
        except Exception as e:  # noqa: BLE001
            result.failure_stage = "smoke_runner"
            result.failure_kind = "validation"
            result.error = f"smoke runner failed: {type(e).__name__}: {e}"
            return _finalize(result)
        result.attempts.append(AttemptRecord(iteration=attempt, wrapper_code=code, test_result=tr))

        if tr.success:
            # 冒烟跑通后只做任务无关、可确定复现的产物健康检查。公式、阈值、
            # 领域范围和 NoData 策略不在注册前推断。
            known = list(sample_inputs or []) + list(prior_outputs or [])
            violations = check_output_invariants(test_args, tr.output, known_inputs=known)
            if not violations:
                log(f"  ✓ smoke test PASS  func={tr.function_name}  returned={tr.output[:100]}")
                break
            log(f"  ✗ smoke PASS 但违反通用不变量: {violations[:2]}  -- asking LLM to fix")
            tr = TestResult(
                success=False, function_name=tr.function_name, stage="invariant",
                output="",
                traceback=("冒烟跑通但输出违反确定性工程健康检查:\n  - "
                           + "\n  - ".join(violations)),
            )
            result.attempts[-1] = AttemptRecord(
                iteration=attempt, wrapper_code=code, test_result=tr
            )
            if attempt == max_fix_attempts:
                result.failure_stage = "invariant"
                result.error = "; ".join(violations)[:200]
                return _finalize(result)
            try:
                code = fix_wrapper(
                    code, tr, capability, signals, llm,
                )
            except Exception as e:  # noqa: BLE001
                return _finish_wrapper_failure(
                    e,
                    default_stage="wrapper_fix",
                    context="invariant fix failed",
                )
            continue
        log(f"  ✗ smoke test FAIL  stage={tr.stage}  -- asking LLM to fix")

        if attempt == max_fix_attempts:
            result.failure_stage = "max_attempts"
            result.error = f"{max_fix_attempts} 次重试后冒烟测试仍未通过;最后 stage={tr.stage}"
            return _finalize(result)

        # 让 LLM 看 traceback 修
        try:
            code = fix_wrapper(
                code, tr, capability, signals, llm,
            )
        except Exception as e:  # noqa: BLE001
            return _finish_wrapper_failure(
                e,
                default_stage="wrapper_fix",
                context="smoke fix failed",
            )
    else:
        # for-else 兜底(理论上不会到这里,因为 break 或 max_attempts return)
        result.failure_stage = "test_failed"
        result.error = "未知失败"
        return _finalize(result)

    # ---- step 6: 注册
    log(f"step 6/6 transactionally register and probe tool '{tr.function_name}'")
    try:
        schema = function_schema_from_wrapper(code, tr.function_name)
        pending = CertificationRecord.create(
            tool_name=tr.function_name,
            tool_code=code,
            wrapper_code=code,
            schema=schema,
            contract=contract,
            smoke_passed=True,
            invariants_passed=True,
            probe_passed=False,
            evidence={
                "smoke_output": tr.output[:1000],
                "probe_arguments": dict(test_args),
            },
        )
        verified_box: list[CertificationRecord] = []

        def _probe_and_finalize(provisional: RegistrationResult) -> None:
            probe = probe_registered_tool(tr.function_name, test_args)
            verified = CertificationRecord.create(
                tool_name=tr.function_name,
                tool_code=code,
                wrapper_code=code,
                schema=probe["schema"],
                contract=contract,
                smoke_passed=True,
                invariants_passed=True,
                probe_passed=True,
                evidence={
                    "smoke_output": tr.output[:1000],
                    "probe_arguments": dict(test_args),
                    "mcp_probe": {
                        "qualified_name": probe["qualified_name"],
                        "result": str(probe["result"])[:1000],
                    },
                },
            )
            if provisional.certification_path is None:
                raise RuntimeError("registration evidence path was not created")
            CertificationStore(provisional.certification_path.parent).save(verified)
            verified_box.append(verified)

        reg = register_wrapper(
            code, tr.function_name, overwrite=overwrite_registered,
            certification_record=pending.to_dict(),
            post_write_probe=_probe_and_finalize,
        )
        if not verified_box:
            raise RuntimeError("MCP registration probe did not finalize evidence")
        certification = verified_box[0]
    except Exception as e:  # noqa: BLE001
        if _is_repairable_probe_failure(e):
            log(f"  ! fresh MCP probe failed; repairing wrapper once: {e}")
            failure = TestResult(
                success=False,
                function_name=tr.function_name,
                stage="mcp_probe",
                traceback=(
                    "The standalone smoke call passed, but the same arguments "
                    "failed through a fresh MCP process. Remove blocking or "
                    "format-mismatched loading paths and make the registered "
                    f"call deterministic. Probe error: {type(e).__name__}: {e}"
                ),
            )
            try:
                code = fix_wrapper(
                    code, failure, capability, signals, llm,
                )
                repaired = run_smoke_test(code, test_args)
            except Exception as repair_exc:  # noqa: BLE001
                if isinstance(repair_exc, RepositoryCapabilityUnsupported):
                    return _finish_wrapper_failure(
                        repair_exc,
                        default_stage="register_probe_repair",
                        context="probe repair failed",
                    )
                result.failure_stage = "register_probe_repair"
                result.failure_kind = (
                    "infrastructure"
                    if is_transient_external_error(repair_exc)
                    else "generation"
                )
                result.error = (
                    f"probe repair failed: {type(repair_exc).__name__}: "
                    f"{repair_exc}"
                )
                return _finalize(result)
            result.attempts.append(AttemptRecord(
                iteration=max_fix_attempts + 1,
                wrapper_code=code,
                test_result=repaired,
            ))
            known = list(sample_inputs or []) + list(prior_outputs or [])
            violations = (
                check_output_invariants(
                    test_args, repaired.output, known_inputs=known,
                )
                if repaired.success else []
            )
            if not repaired.success or violations:
                result.failure_stage = "register_probe_repair"
                result.failure_kind = "validation"
                result.error = (
                    f"repaired wrapper smoke failed at {repaired.stage}"
                    if not repaired.success
                    else f"repaired wrapper violated invariant: {violations[0]}"
                )
                return _finalize(result)

            tr = repaired
            try:
                schema = function_schema_from_wrapper(code, tr.function_name)
                pending = CertificationRecord.create(
                    tool_name=tr.function_name,
                    tool_code=code,
                    wrapper_code=code,
                    schema=schema,
                    contract=contract,
                    smoke_passed=True,
                    invariants_passed=True,
                    probe_passed=False,
                    evidence={
                        "smoke_output": tr.output[:1000],
                        "probe_arguments": dict(test_args),
                    },
                )
                verified_box = []

                def _probe_repaired(provisional: RegistrationResult) -> None:
                    probe = probe_registered_tool(tr.function_name, test_args)
                    verified = CertificationRecord.create(
                        tool_name=tr.function_name,
                        tool_code=code,
                        wrapper_code=code,
                        schema=probe["schema"],
                        contract=contract,
                        smoke_passed=True,
                        invariants_passed=True,
                        probe_passed=True,
                        evidence={
                            "smoke_output": tr.output[:1000],
                            "probe_arguments": dict(test_args),
                            "mcp_probe": {
                                "qualified_name": probe["qualified_name"],
                                "result": str(probe["result"])[:1000],
                            },
                        },
                    )
                    if provisional.certification_path is None:
                        raise RuntimeError(
                            "registration evidence path was not created"
                        )
                    CertificationStore(
                        provisional.certification_path.parent
                    ).save(verified)
                    verified_box.append(verified)

                reg = register_wrapper(
                    code,
                    tr.function_name,
                    overwrite=overwrite_registered,
                    certification_record=pending.to_dict(),
                    post_write_probe=_probe_repaired,
                )
                if not verified_box:
                    raise RuntimeError(
                        "MCP registration probe did not finalize evidence"
                    )
                certification = verified_box[0]
            except Exception as retry_exc:  # noqa: BLE001
                result.failure_stage = "register_probe"
                result.failure_kind = "registration"
                result.error = str(retry_exc)
                return _finalize(result)
            result.registration = reg
            result.certification = certification.to_dict()
            result.success = True
            log("  probe repair passed smoke, invariants, and fresh MCP")
            return _finalize(result)
        result.failure_stage = "register"
        result.failure_kind = "registration"
        result.error = str(e)
        log(f"  ✗ register failed: {e}")
        return _finalize(result)

    result.registration = reg
    result.certification = certification.to_dict()
    log(f"  ✓ {reg.message}")
    log(f"DONE in {len(result.attempts)} attempt(s). Restart Agent to use the new tool: local__{tr.function_name}")
    result.success = True
    return _finalize(result)


# ---- capability-only 入口(ReAct 自动触发将来用这个) ---------------------

def evolve_from_capability(
    capability: str,
    llm: BaseLLM,
    *,
    top_k: int = 3,
    max_repo_attempts: int = 2,
    max_fix_attempts: int = 2,
    overwrite_registered: bool = False,
    verbose: bool = True,
    sample_inputs: list[str] | None = None,
    prior_outputs: list[str] | None = None,
    capability_contract: CapabilityContract | dict[str, Any] | None = None,
    resume_key: str | None = None,
) -> EvolveResult:
    """完全自动化:只给一句话能力描述,做完搜索 → 排序候选 → clone → 包装
    → LLM 自动出 test_args → 测试 → 修复 → 注册。

    sample_inputs:任务真实输入文件路径清单,透传给 test_args 生成,让 IO
    工具的冒烟测试用真数据而非编造路径。

    流程:
      1. M1 搜索:capability → top_k 个候选 GitHub 仓库
      2. 按评估顺序串行尝试,默认最多 2 个候选(首选 + 一个备选)
      3. 调 evolve_from_repo,test_args=None 让 LLM 自动生成

    失败回退:
      - 搜索 0 条     → failure_stage="search"
      - 仅 failure_kind="repository"(非瞬态 clone 失败、仓库信号缺失等)
        才切换下一个候选;生成、验证、注册和基础设施失败均停止,避免重复付出
        clone/install 成本却重现同一非仓库根因。
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"[evolve] {msg}", flush=True)

    started_at = time.time()
    result = EvolveResult(success=False, capability=capability, repo="")

    def _finalize(r: EvolveResult) -> EvolveResult:
        try:
            append_evolve_log(r, started_at=started_at)
        except Exception as e:  # noqa: BLE001
            log(f"  ! audit log 写入失败(忽略): {type(e).__name__}: {e}")
        return r

    def _merge_repository_result(sub: EvolveResult) -> EvolveResult:
        result.success = sub.success
        result.repo = sub.repo or result.repo
        result.failure_stage = sub.failure_stage
        result.error = sub.error
        result.failure_kind = sub.failure_kind
        result.cloned = sub.cloned
        result.signals = sub.signals
        result.test_args = sub.test_args
        result.attempts = sub.attempts
        result.registration = sub.registration
        result.capability_contract = sub.capability_contract
        result.certification = sub.certification
        return result

    # A task-level resume must continue the exact repository selected before
    # the infrastructure interruption.  Re-running search or ranking here can
    # silently change the unfinished capability and break workflow continuity.
    resumed_repository = (
        load_repository_checkpoint(resume_key, capability=capability)
        if resume_key else None
    )
    if resumed_repository:
        log(f"RESUME repository checkpoint: {resumed_repository}")
        result.repo = resumed_repository
        sub = evolve_from_repo(
            repo=resumed_repository,
            capability=capability,
            test_args=None,
            llm=llm,
            max_fix_attempts=max_fix_attempts,
            overwrite_registered=overwrite_registered,
            verbose=verbose,
            sample_inputs=sample_inputs,
            prior_outputs=prior_outputs,
            capability_contract=capability_contract,
            _skip_audit_log=True,
        )
        if sub.failure_kind != "infrastructure":
            clear_repository_checkpoint(resume_key)
        return _finalize(_merge_repository_result(sub))

    # ---- step 0: M1 搜索(多轮:LLM 相关性评估 + 查询重写)
    # B-2.5: top_k 拔到 50(参数 top_k 现在表示搜索池容量,不是最终输出)
    log(f"step 0/6 search GitHub for capability: {capability!r}")
    try:
        query, ranked, rounds = find_candidates(
            capability, llm=llm, top_k=max(top_k, 50), raw_query=False,
        )
    except Exception as e:  # noqa: BLE001
        result.failure_stage = "search"
        result.failure_kind = (
            "infrastructure" if is_transient_external_error(e) else "generation"
        )
        result.error = f"search threw: {type(e).__name__}: {e}"
        log(f"  ✗ search failed: {e}")
        return _finalize(result)

    result.chosen_query = query
    result.search_candidates = ranked

    for r in rounds:
        try:
            append_search_round_log(capability, r)
        except Exception as e:  # noqa: BLE001
            log(f"  ! search_round 日志失败(忽略): {type(e).__name__}: {e}")

    log(f"  ✓ rounds = {len(rounds)}  final_query = {query}")
    for r in rounds:
        v = r.verdict
        verdict_brief = (
            f"SUFFICIENT={'Y' if v.sufficient else 'N'} relevant={v.relevant_count}"
            if v else "no-eval"
        )
        log(f"    round {r.round_idx}: q={r.query[:80]!r}  hits={r.raw_repo_count}  {verdict_brief}")
    log(f"  ✓ pool size = {len(ranked)} candidates (after merge+rerank)")

    if not ranked:
        result.failure_stage = "search"
        result.error = "GitHub 搜索多轮后仍返回 0 条结果"
        return _finalize(result)

    # ---- step 0.3:只让代码粗排前 5 进入 README + 五维评估。
    raw_repos = [sr.repo for sr in ranked]
    filtered = raw_repos[:DEFAULT_MAX_CANDIDATES_TO_ASSESS]
    log(f"step 0.3/6 code-ranked top {len(filtered)} selected for README assessment:")
    for i, r in enumerate(filtered, 1):
        log(f"    {i}. {r.full_name}  ★{r.stars}  {(r.description or '')[:70]!r}")

    # Fetch README evidence and evaluate with configured weights.
    log(f"step 0.4/6 fetching README for {len(filtered)} candidates...")
    try:
        enriched = enrich_with_readme(filtered)
        readme_hits = sum(1 for r in enriched if r.readme)
        log(f"  ✓ README fetched: {readme_hits}/{len(enriched)} got non-empty content")
    except Exception as e:  # noqa: BLE001
        log(f"  ! enrich_with_readme 部分失败(忽略): {type(e).__name__}: {e}")
        enriched = filtered  # 即使 enrich 出错,readme 字段默认空,继续

    log(f"step 0.5/6 LLM 5-dim assessment (configured weights) on {len(enriched)} candidates...")
    try:
        assessed = assess_candidates(
            capability=capability,
            candidates=enriched,
            llm=llm,
        )
    except Exception as e:  # noqa: BLE001
        if is_transient_external_error(e):
            result.failure_stage = "repo_assess"
            result.failure_kind = "infrastructure"
            result.error = f"repo assessment threw: {type(e).__name__}: {e}"
            return _finalize(result)
        log(f"  ! assess_candidates 失败,退回粗筛 top-1: {type(e).__name__}: {e}")
        assessed = []

    try:
        append_repo_assess_log(capability, assessed)
    except Exception as e:  # noqa: BLE001
        log(f"  ! repo_assess 日志失败(忽略): {type(e).__name__}: {e}")

    ordered_repos = []
    if assessed:
        best, passes = pick_best(assessed)
        if best is None:
            chosen = enriched[0] if enriched else raw_repos[0]
            log(f"  ! pick_best 返回空,退回粗筛 top-1: {chosen.full_name}")
        else:
            chosen = best.repo
            tag = "✓ passes thresholds" if passes else "⚠ below thresholds, soft pick"
            log(f"  → picked: {chosen.full_name}  composite={best.score.composite:.2f}  {tag}")
            log(f"    sem={best.score.semantic_relevance:.2f} "
                f"struct={best.score.structural_completeness:.2f} "
                f"tool={best.score.toolization_feasibility:.2f} "
                f"comm={best.score.community_reliability:.2f} "
                f"env={best.score.environment_complexity:.2f}")
            log(f"    reason: {best.score.reasoning}")
        passing = [a.repo for a in assessed if a.score.passes_thresholds]
        below = [a.repo for a in assessed if not a.score.passes_thresholds]
        ordered_repos = passing + below
    else:
        chosen = enriched[0] if enriched else raw_repos[0]
        log(f"  → picked top-1 (no assessment): {chosen.full_name}")
        ordered_repos = list(enriched or raw_repos)

    # 保留评估顺序并去重。top-1 是首选，但仓库自身 clone/包装/测试失败时
    # 自动回退到 top-2/top-3；基础设施失败不切仓库，直接交给 harness。
    unique_repos = []
    seen_repos: set[str] = set()
    for candidate in ordered_repos:
        if candidate.full_name.lower() not in seen_repos:
            seen_repos.add(candidate.full_name.lower())
            unique_repos.append(candidate)
    candidates_to_try = unique_repos[:min(2, max(1, max_repo_attempts))]
    sub = EvolveResult(False, capability, "")
    for repo_idx, candidate in enumerate(candidates_to_try, 1):
        result.repo = candidate.full_name
        if resume_key:
            save_repository_checkpoint(
                resume_key,
                capability=capability,
                repository=candidate.full_name,
            )
        log(f"step 1-6 repository attempt {repo_idx}/{len(candidates_to_try)}: "
            f"{candidate.full_name}")
        sub = evolve_from_repo(
            repo=candidate.full_name,
            capability=capability,
            test_args=None,
            llm=llm,
            max_fix_attempts=max_fix_attempts,
            overwrite_registered=overwrite_registered,
            verbose=verbose,
            sample_inputs=sample_inputs,
            prior_outputs=prior_outputs,
            capability_contract=capability_contract,
            _skip_audit_log=True,
        )
        result.repository_attempts.append({
            "rank": repo_idx,
            "repo": candidate.full_name,
            "success": sub.success,
            "failure_stage": sub.failure_stage,
            "failure_kind": sub.failure_kind,
            "error": (sub.error or "")[:500],
            "total_attempts": len(sub.attempts),
        })
        if sub.success or not _should_try_next_repository(sub):
            break
        if repo_idx < len(candidates_to_try):
            log(f"  ↳ repository failed at {sub.failure_stage}; trying next candidate")

    # 把 sub 的字段合并回 result(留住 search 阶段的信息)
    if resume_key and sub.failure_kind != "infrastructure":
        clear_repository_checkpoint(resume_key)
    return _finalize(_merge_repository_result(sub))
