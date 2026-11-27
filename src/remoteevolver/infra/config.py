"""
remoteevolver.infra.config —— 配置加载与校验

单一可信源：所有阈值、路径、模型名、MCP server、提示词模板路径都在 YAML 配置里；
私密信息（API key、私有 base_url）走 .env。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from .paths import runtime_path
from typing import Any

import yaml
from dotenv import load_dotenv


# ---- 配置 dataclass ----------------------------------------------------

@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    stream: bool = False


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolsConfig:
    max_on_demand_evolutions: int = 3
    whitelist: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    trusted_servers: list[str] = field(default_factory=list)
    always_exposed: list[str] = field(default_factory=lambda: ["get_filelist"])
    verification_mode: str = "strict"
    certification_dir: str = str(runtime_path("tools/.certifications"))


@dataclass
class AgentConfig:
    paradigm: str = "react"
    max_steps: int | None = None

    def __post_init__(self):
        if self.max_steps is not None and (isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps <= 0):
            raise ValueError("agent.max_steps must be null (unlimited) or a positive integer")


@dataclass
class ShortTermConfig:
    max_messages: int = 50


@dataclass
class LongTermConfig:
    backend: str = "file"
    path: str = str(runtime_path("memory/history.json"))
    top_k: int = 3


@dataclass
class MemoryConfig:
    short_term: ShortTermConfig = field(default_factory=ShortTermConfig)
    long_term: LongTermConfig = field(default_factory=LongTermConfig)


@dataclass
class ReflectionConfig:
    trigger: str = "on_finish"   # on_finish | on_failure | every_n_steps
    every_n_steps: int = 5


@dataclass
class PromptConfig:
    system: str = "system.txt"
    reflection: str = "reflection.txt"
    tool_use: str = "tool_use.txt"


@dataclass
class Config:
    llm: LLMConfig
    mcp_servers: list[MCPServerConfig]
    tools: ToolsConfig
    agent: AgentConfig
    memory: MemoryConfig
    reflection: ReflectionConfig
    prompt: PromptConfig


# ---- 加载入口 -----------------------------------------------------------

# 注意:llm.provider 和 memory.long_term.backend 的合法集合不在这里写死,
# 由 llm / memory 模块各自的 registry 提供 —— 见 _parse_llm / _parse_memory。
# 这样新增一个 provider / backend 只需要在对应模块加文件 + @register_xxx,
# 不需要回头改 infra/config.py。
_VALID_TRIGGERS = {"on_finish", "on_failure", "every_n_steps"}


def load_config(path: str | Path) -> Config:
    """加载 YAML 主配置，并从 .env 注入私密信息。

    override=True：让 .env 强制覆盖 shell/系统里残留的同名环境变量。
    否则一旦用户或系统级环境变量里有过期的 OPENAI_API_KEY / OPENAI_BASE_URL,
    框架会沿用旧值，而 .env 里的新值被静默忽略，表现为「test 脚本能通、框架报 401」。
    """
    # Environment loading belongs to the explicit application entry point.

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return Config(
        llm=_parse_llm(raw.get("llm", {})),
        mcp_servers=_parse_mcp_servers(raw.get("mcp_servers", []) or []),
        tools=_parse_tools(raw.get("tools", {}) or {}),
        agent=_parse_agent(raw.get("agent", {}) or {}),
        memory=_parse_memory(raw.get("memory", {}) or {}),
        reflection=_parse_reflection(raw.get("reflection", {}) or {}),
        prompt=_parse_prompt(raw.get("prompt", {}) or {}),
    )


# ---- 各段解析（含校验） -----------------------------------------------

def _parse_llm(raw: dict[str, Any]) -> LLMConfig:
    # 懒加载:把 infra → llm 的依赖延后到函数调用时,避免 import 期循环
    from ..llm import list_llms
    valid_providers = list_llms()

    provider = raw.get("provider", "openai")
    if provider not in valid_providers:
        raise ValueError(
            f"llm.provider={provider!r} 未注册，已注册：{valid_providers}"
        )

    api_key_env = raw.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ValueError(
            f"环境变量 {api_key_env} 未设置；请在 .env 或系统环境中提供"
        )

    base_url_env = raw.get("base_url_env")
    base_url = raw.get("base_url")
    if base_url_env and os.environ.get(base_url_env):
        base_url = os.environ[base_url_env]

    model_env = str(raw.get("model_env", "")).strip()
    model = os.environ.get(model_env, "").strip() if model_env else ""
    if not model:
        model = str(raw["model"]).strip()
    if not model:
        raise ValueError("llm.model must not be empty")

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(raw.get("temperature", 0.0)),
        stream=bool(raw.get("stream", False)),
    )


def _parse_mcp_servers(raw: list[dict[str, Any]]) -> list[MCPServerConfig]:
    servers = []
    for item in raw:
        if "name" not in item or "command" not in item:
            raise ValueError(f"mcp_servers 条目缺少 name 或 command：{item}")
        servers.append(MCPServerConfig(
            name=item["name"],
            command=item["command"],
            args=list(item.get("args", [])),
            env=dict(item.get("env", {})),
        ))
    return servers


def _parse_tools(raw: dict[str, Any]) -> ToolsConfig:
    max_on_demand_evolutions = int(raw.get("max_on_demand_evolutions", 3))
    if max_on_demand_evolutions <= 0:
        raise ValueError("tools.max_on_demand_evolutions 必须大于 0")
    mode = str(raw.get("verification_mode", "strict")).lower()
    if mode not in {"strict", "permissive"}:
        raise ValueError("tools.verification_mode 必须是 strict 或 permissive")
    return ToolsConfig(
        max_on_demand_evolutions=max_on_demand_evolutions,
        whitelist=list(raw.get("whitelist", []) or []),
        blacklist=list(raw.get("blacklist", []) or []),
        trusted_servers=list(raw.get("trusted_servers", []) or []),
        always_exposed=list(raw.get("always_exposed", ["get_filelist"]) or []),
        verification_mode=mode,
        certification_dir=str(raw.get("certification_dir", str(runtime_path("tools/.certifications")))),
    )


def _parse_agent(raw: dict[str, Any]) -> AgentConfig:
    return AgentConfig(
        paradigm=raw.get("paradigm", "react"),
        max_steps=raw.get("max_steps"),
    )


def _parse_memory(raw: dict[str, Any]) -> MemoryConfig:
    short = raw.get("short_term", {}) or {}
    long_ = raw.get("long_term", {}) or {}

    # 同 _parse_llm:合法 backend 集合由 memory 模块的 registry 提供,不在这里写死
    from ..memory import list_memory_backends
    valid_backends = list_memory_backends()

    backend = long_.get("backend", "file")
    if backend not in valid_backends:
        raise ValueError(
            f"memory.long_term.backend={backend!r} 未注册，已注册：{valid_backends}"
        )

    return MemoryConfig(
        short_term=ShortTermConfig(max_messages=int(short.get("max_messages", 50))),
        long_term=LongTermConfig(
            backend=backend,
            path=str(long_.get("path", str(runtime_path("memory/history.json")))),
            top_k=int(long_.get("top_k", 3)),
        ),
    )


def _parse_reflection(raw: dict[str, Any]) -> ReflectionConfig:
    trigger = raw.get("trigger", "on_finish")
    if trigger not in _VALID_TRIGGERS:
        raise ValueError(
            f"reflection.trigger={trigger!r} 不支持，可选：{sorted(_VALID_TRIGGERS)}"
        )
    return ReflectionConfig(
        trigger=trigger,
        every_n_steps=int(raw.get("every_n_steps", 5)),
    )


def _parse_prompt(raw: dict[str, Any]) -> PromptConfig:
    return PromptConfig(
        system=raw.get("system", "system.txt"),
        reflection=raw.get("reflection", "reflection.txt"),
        tool_use=raw.get("tool_use", "tool_use.txt"),
    )
