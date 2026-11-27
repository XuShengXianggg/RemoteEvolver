"""
remoteevolver.evolve.tool —— 工具自进化子系统(目标:能力描述 → 自动找/包装/测试/注册工具)

里程碑:
  M1  search             能力描述 → 候选 GitHub 仓库   (search.py + github.py)
  M2  clone + wrap       clone 仓库 + LLM 生成 wrapper (cloner.py + wrapper_gen.py)
  M3  test + fix         冒烟测试 + LLM 自动修复       (test_runner.py + fixer.py)
  M4  register           写到 remoteevolver/tools/ + 接 ReAct (registrar.py + pipeline.py)

所有可调参数在 configs/tool-evolution.yaml。
"""

from __future__ import annotations

from .audit_log import (
    append_evolve_log,
    append_repo_assess_log,
    append_search_round_log,
    load_log,
)
from .cloner import ClonedRepo, CloneError, clone_and_install
from .config import (
    EvolveConfig,
    QueryBuilderConfig,
    RelaxConfig,
    ScoreConfig,
    load_evolve_config,
)
from .code_audit import (
    CallIssue,
    ClassSig,
    FunctionSig,
    ModuleAPI,
    PackageAPI,
    audit_wrapper_calls,
    format_issues_for_llm,
    scan_package_api,
    write_code_report,
)
from .fixer import fix_wrapper, fix_wrapper_for_consistency
from .github import GitHubClient, GitHubError, RateLimitError, Repo
from .planner import (
    DEFAULT_MAX_TOPICS,
    DEFAULT_TEXT_HINT_MAX_TOKENS,
    DEFAULT_TOPICS_MAX_TOKENS,
    SearchPlan,
    extract_text_hint_regex,
    plan_search,
)
from .pipeline import AttemptRecord, EvolveResult, evolve_from_capability, evolve_from_repo
from .registrar import RegistrationResult, register_wrapper
from .repo_assess import (
    AssessedRepo,
    AssessmentScore,
    RepositoryAssessmentInvalid,
    assess_candidates,
    enrich_with_readme,
    pick_best,
)
from .search import ScoredRepo, SearchRound, build_query_from_capability, find_candidates
from .search_eval import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MIN_RELEVANT,
    DEFAULT_TOP_N_FOR_EVAL,
    RelevanceVerdict,
    check_relevance,
    reformulate_plan,
    reformulate_query,
)
from .test_args_gen import generate_test_args
from .test_runner import TestResult, run_smoke_test
from .wrapper_gen import (
    RepoSignals,
    RepositoryCapabilityUnsupported,
    collect_repo_signals,
    generate_wrapper,
    parse_wrapper_response,
)

__all__ = [
    # config
    "EvolveConfig", "QueryBuilderConfig", "RelaxConfig", "ScoreConfig",
    "load_evolve_config",
    # github
    "GitHubClient", "GitHubError", "RateLimitError", "Repo",
    # search (M1)
    "ScoredRepo", "SearchRound", "build_query_from_capability", "find_candidates",
    # planner (R1+R2 - 正则抽 text_hint + LLM 出 topics 双通道)
    "SearchPlan", "plan_search", "extract_text_hint_regex",
    "DEFAULT_MAX_TOPICS", "DEFAULT_TEXT_HINT_MAX_TOKENS", "DEFAULT_TOPICS_MAX_TOKENS",
    # search_eval (M1 - LLM 相关性评估 + 查询重写 + 语义粗筛 + R4 plan 重写)
    "RelevanceVerdict", "check_relevance", "reformulate_query",
    "reformulate_plan",
    "DEFAULT_MAX_ROUNDS", "DEFAULT_MIN_RELEVANT", "DEFAULT_TOP_N_FOR_EVAL",
    # repo_assess (M1 - 5 维仓库评估 + B-2.5 README 富化)
    "AssessedRepo", "AssessmentScore", "RepositoryAssessmentInvalid", "assess_candidates",
    "enrich_with_readme", "pick_best",
    # cloner (M2)
    "ClonedRepo", "CloneError", "clone_and_install",
    # wrapper_gen (M2)
    "RepoSignals", "RepositoryCapabilityUnsupported", "collect_repo_signals",
    "generate_wrapper", "parse_wrapper_response",
    # code_audit (C-1 / C-2) - AST 扫公开 API + 一致性校验
    "ClassSig", "FunctionSig", "ModuleAPI", "PackageAPI", "CallIssue",
    "scan_package_api", "write_code_report",
    "audit_wrapper_calls", "format_issues_for_llm",
    # test_runner (M3)
    "TestResult", "run_smoke_test",
    # fixer (M3 + C-2)
    "fix_wrapper", "fix_wrapper_for_consistency",
    # registrar (M4)
    "RegistrationResult", "register_wrapper",
    # test_args_gen (M3 auto)
    "generate_test_args",
    # audit log
    "append_evolve_log", "append_repo_assess_log",
    "append_search_round_log", "load_log",
    # pipeline (orchestration)
    "AttemptRecord", "EvolveResult", "evolve_from_repo", "evolve_from_capability",
]
