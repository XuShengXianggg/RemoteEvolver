"""
remoteevolver.core —— 用户面入口:Agent 门面类

把 LLM / MCP / memory / prompt / reflection 等内部依赖全装齐,对外只暴露:
  - run(task)        执行任务,返回最终答案
  - close()          关停 MCP 子进程(可重复;__exit__ 会自动触发)
  - __enter__ / __exit__  上下文管理
  - recall(query)    查长期经验(诊断/演示用)
  - config           只读访问当前配置(诊断用)

所有内部组件都是 _ 前缀,用户不应该也不需要直接访问。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import BaseAgent
from .infra.config import Config, load_config
from .llm import get_llm
from .llm.base import BaseLLM
from .mcpx import MCPClient
from .console import safe_print as print
from .infra.errors import InfrastructureFailure, is_transient_external_error
from .memory import get_memory_backend
from .memory.long_term import LongTermMemory, Entry
from .memory.short_term import ShortTermMemory
from .prompt.manager import PromptManager
from .react import ReActAgent
from .reflection.reflector import Reflector


# 任务文本里可能出现的路径 token:含分隔符的路径段(支持 / 与 \,以及盘符)。
_PATH_TOKEN = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|]+|[^\s\"'<>|]*[\\/][^\s\"'<>|]+")
# 一个目录里最多取几个样例文件喂冒烟测试(够 LLM 按用途配对即可,别太多撑爆 prompt)
_MAX_SAMPLE_FILES = 12


def _collect_sample_inputs(task: str, *, max_files: int = _MAX_SAMPLE_FILES) -> list[str]:
    """从任务文本里挖出**真实存在**的输入文件,作为冒烟测试样例(绝对路径)。

    通用做法,不绑定任何数据集:扫 task 里像路径的 token,凡是真实存在的——
      - 是目录 → 列出里面的文件(取前 max_files 个);
      - 是文件 → 直接收。
    一个都没扫到就返回 [](调用方退回旧行为:LLM 自己编 test_args)。

    返回绝对路径:冒烟测试在临时目录起子进程,相对路径会找不到文件。
    """
    found: list[str] = []
    seen: set[str] = set()

    def _add_file(p: Path) -> None:
        try:
            rp = str(p.resolve())
        except OSError:
            return
        if rp not in seen and p.is_file():
            seen.add(rp)
            found.append(rp)

    candidates = set()
    for m in _PATH_TOKEN.finditer(task or ""):
        tok = m.group(0).strip().strip(".,;:)]}。，").strip("'\"")
        if tok:
            candidates.add(tok)

    for tok in candidates:
        try:
            p = Path(tok)
            if not p.exists():
                continue
        except OSError:
            continue
        if p.is_dir():
            files = [c for c in sorted(p.iterdir()) if c.is_file()]
            budget = max_files - len(found)
            if len(files) > budget > 0:
                # 等距抽样而非取前 N:避免同名前缀(如全是 b04)挤满,漏掉 b06 这类
                # 另一半,保证样例覆盖目录里不同命名/类型,LLM 才好按用途配对。
                step = len(files) / budget
                files = [files[int(i * step)] for i in range(budget)]
            for child in files:
                if len(found) >= max_files:
                    break
                _add_file(child)
        else:
            _add_file(p)
        if len(found) >= max_files:
            break

    return found[:max_files]


def _collect_task_outputs() -> list[str]:
    """Return real files already produced inside the task-scoped output root."""
    import os

    raw = os.environ.get("EVOAGENT_OUTPUT_DIR", "").strip()
    if not raw:
        return []
    root = Path(raw)
    if not root.is_dir():
        return []
    return sorted(str(path.resolve()) for path in root.rglob("*") if path.is_file())


class Agent:
    """User-facing ReAct-first agent facade."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        config: Config | None = None,
        verbose: bool = False,
        evolve_enabled: bool = True,
        workflow_enabled: bool = True,
        judge_llm: BaseLLM | None = None,
        workflow_config=None,
    ) -> None:
        if config is None and config_path is None:
            raise ValueError("Pass an explicit configuration or configuration path")
        cfg = config if config is not None else load_config(config_path)
        self._verbose = verbose
        self._evolve_enabled = evolve_enabled
        self._workflow_enabled = workflow_enabled

        # ---- LLM 客户端(主)
        llm_cls = get_llm(cfg.llm.provider)
        self._llm: BaseLLM = llm_cls(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            temperature=cfg.llm.temperature,
        )

        # ---- MCP 客户端(构造即起子进程,start() 内会阻塞到握手成功)
        self._mcp = MCPClient(cfg.mcp_servers)
        self._mcp.start()

        # ---- 记忆层
        backend_cls = get_memory_backend(cfg.memory.long_term.backend)
        backend = backend_cls(path=cfg.memory.long_term.path)
        self._long_term = LongTermMemory(backend=backend)
        self._short_term = ShortTermMemory(
            max_messages=cfg.memory.short_term.max_messages,
        )

        # ---- prompt + reflection
        self._prompt_mgr = PromptManager(cfg.prompt)
        self._reflector = Reflector(
            llm=self._llm,
            prompt_manager=self._prompt_mgr,
            long_term_memory=self._long_term,
        )

        # ---- 工作流自进化(默认开启;显式传 False 才退化为纯 ReAct)
        self._workflow = None
        if workflow_enabled:
            from .evolve.workflow import WorkflowEvolver
            bank_dir = Path(cfg.memory.long_term.path).parent / "skills"
            self._workflow = WorkflowEvolver(
                llm=self._llm, mcp=self._mcp, bank_dir=bank_dir,
                judge_llm=judge_llm, verbose=verbose,
                **({"cfg": workflow_config} if workflow_config is not None else {}),
            )
            if verbose:
                print(f"[agent] 工作流自进化已开启,技能库={bank_dir} "
                      f"(现有 {self._workflow.bank.count()} 个技能)", flush=True)

        # ---- paradigm 路由(目前只有 react,预留 plan_execute 扩展点)
        if cfg.agent.paradigm == "react":
            from .evolve.tool.reliability import ToolReliabilityManager
            self._tool_reliability = ToolReliabilityManager(
                mode=cfg.tools.verification_mode,
                certification_dir=cfg.tools.certification_dir,
            )
            self._impl: BaseAgent = ReActAgent(
                llm=self._llm,
                mcp=self._mcp,
                short_term=self._short_term,
                long_term=self._long_term,
                prompt_manager=self._prompt_mgr,
                reflector=self._reflector,
                config=cfg,
                verbose=verbose,
                workflow=self._workflow,
                tool_reliability=self._tool_reliability,
                # 桥接:工作流归因"缺工具"时,直接驱动工具进化轴 + 带新工具重试。
                # 仅在工作流开启时接线(关闭时 ReAct 用不到)。
                # Fixed-tool experiments keep workflow evolution enabled while
                # disabling every route into tool evolution.
                evolve_tool_fn=self._evolve_tool if evolve_enabled else None,
                evolve_tool_for_task_fn=(
                    self._evolve_tool_for_react_task if evolve_enabled else None
                ),
            )
        else:
            raise ValueError(
                f"未知 agent.paradigm:{cfg.agent.paradigm!r};当前仅支持 'react'"
            )

        self._config = cfg
        if self._workflow is not None:
            from remoteevolver.admission import make_admission_runner
            self._workflow.set_admission_runner(make_admission_runner(self, self._workflow))

    # ---- 公开 API --------------------------------------------------------

    def run(self, task: str, expected: Any = None) -> str:
        """Run every task through the single ReAct-first execution path."""
        return self._impl.run(task, expected=expected)

    def _evolve_tool(
        self,
        capability: str,
        *,
        sample_inputs: list[str] | None = None,
        prior_outputs: list[str] | None = None,
        capability_contract: Any | None = None,
    ) -> bool:
        """Evolve one ReAct-requested capability and expose it through MCP."""
        from .evolve.tool import evolve_from_capability

        try:
            result = evolve_from_capability(
                capability=capability,
                llm=self._llm,
                overwrite_registered=True,
                top_k=int(os.environ.get('REMOTEEVOLVER_TOOL_TOP_K', '3')),
                max_repo_attempts=int(os.environ.get('REMOTEEVOLVER_TOOL_REPO_ATTEMPTS', '2')),
                max_fix_attempts=int(os.environ.get('REMOTEEVOLVER_TOOL_WRAPPER_ATTEMPTS', '2')),
                verbose=self._verbose,
                sample_inputs=sample_inputs,
                prior_outputs=prior_outputs,
                capability_contract=capability_contract,
            )
        except Exception as e:  # noqa: BLE001
            if is_transient_external_error(e):
                raise InfrastructureFailure("tool_evolution", e) from e
            print(f"[agent] 进化抛异常,降级 ReAct: {type(e).__name__}: {e}", flush=True)
            return False

        if not result.success:
            if result.failure_kind == "infrastructure":
                raise InfrastructureFailure(
                    f"tool_evolution.{result.failure_stage}", result.error
                )
            print(f"[agent] 进化失败 stage={result.failure_stage},降级 ReAct: {result.error[:200]}", flush=True)
            return False

        tool_name = result.registration.tool_name
        print(f"[agent] ✓ 新工具 local__{tool_name} 已注册,热重启 MCP...", flush=True)
        try:
            self._mcp.restart()
        except Exception as e:  # noqa: BLE001
            print(f"[agent] ✗ MCP 重启失败,新工具本轮可能不可用: {e}", flush=True)
            return False

        if self._verbose:
            print(f"[agent] MCP 重启完成,可用工具: {sorted(self._mcp.list_names())}", flush=True)
        return True

    def _evolve_tool_for_react_task(
        self, task: str, capability: dict[str, str] | str,
    ) -> bool:
        """Evolve one ReAct-requested capability with real task input evidence."""
        import json

        sample_inputs = _collect_sample_inputs(task)
        prior_outputs = _collect_task_outputs()
        if sample_inputs:
            print(
                f"[agent] ReAct 按需进化使用 {len(sample_inputs)} 个真实输入样例",
                flush=True,
            )
        if isinstance(capability, dict):
            description = json.dumps(capability, ensure_ascii=False, sort_keys=True)
            contract = {
                "operation": capability["operation"],
                "description": capability["expected_output"],
                "inputs": [{
                    "name": "task_inputs",
                    "type": "any",
                    "semantic_role": capability["input_description"],
                    "required": True,
                }],
                "outputs": [{
                    "name": "result",
                    "type": "any",
                    "semantic_role": capability["expected_output"],
                    "required": True,
                }],
                "formula_or_rule": capability["expected_output"],
                "preconditions": [capability["input_description"]],
                "postconditions": [capability["expected_output"]],
                "testable_assertions": [capability["single_operation_reason"]],
            }
        else:
            description = capability
            contract = None
        if contract is not None:
            equivalent = self._find_equivalent_registered_tool(contract, task=task)
            if equivalent is not None:
                tool_name, reason = equivalent
                print(
                    f"[agent] 已有工具 {tool_name} 的能力契约等价，跳过重复进化: {reason}",
                    flush=True,
                )
                return True
        return self._evolve_tool(
            description,
            sample_inputs=sample_inputs,
            prior_outputs=prior_outputs or None,
            capability_contract=contract,
        )

    def _find_equivalent_registered_tool(
        self, required_contract: dict[str, Any], *, task: str = "",
    ) -> tuple[str, str] | None:
        """Find a semantically equivalent visible tool before evolving a new one."""
        needed = ("_mcp", "_config", "_tool_reliability", "_llm")
        if not all(hasattr(self, name) for name in needed):
            return None
        from .evolve.tool.capability_contract import CapabilityContract
        from .evolve.tool.capability_matcher import judge_capability_equivalence

        required = CapabilityContract.from_task_requirement(required_contract)
        schemas = self._mcp.schemas(
            whitelist=self._config.tools.whitelist or None,
            blacklist=self._config.tools.blacklist or None,
        )
        eligible = self._tool_reliability.eligible_schemas_for_react(
            schemas, task=task,
        )
        offered_contracts = self._tool_reliability.offered_contracts(eligible)
        best: tuple[str, str, float] | None = None
        for tool_name, offered in offered_contracts.items():
            verdict = judge_capability_equivalence(required, offered, self._llm)
            if verdict.equivalent and (best is None or verdict.confidence > best[2]):
                best = (tool_name, verdict.reason, verdict.confidence)
        return (best[0], best[1]) if best is not None else None

    def close(self) -> None:
        """关停 MCP 子进程。可重复调用;__exit__ 会自动触发。"""
        self._mcp.stop()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def config(self) -> Config:
        """只读访问当前配置(诊断用)。"""
        return self._config

    def recall(self, query: str, top_k: int = 3) -> list[Entry]:
        """查长期经验里跟 query 相似的条目(诊断 / 演示自进化闭环用)。"""
        return self._long_term.search(query, top_k=top_k)
