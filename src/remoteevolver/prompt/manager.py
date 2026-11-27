"""
remoteevolver.prompt.manager —— 模板加载与变量渲染

模板文件就放在本目录下（system.txt / reflection.txt / tool_use.txt），
通过 config.prompt.<name> 指定文件名。用 Python str.format_map 做最小渲染，
模板里写 {variable}，调用方传 kwargs 即可。
"""

from __future__ import annotations

from pathlib import Path

from ..infra.config import PromptConfig

_DEFAULT_DIR = Path(__file__).parent


class _SafeDict(dict):
    """str.format_map 用，找不到 key 时保留原样占位，避免渲染时炸。"""
    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "{" + key + "}"


class PromptManager:
    def __init__(self, cfg: PromptConfig, base_dir: Path | None = None) -> None:
        self._cfg = cfg
        self._base = base_dir or _DEFAULT_DIR
        self._cache: dict[str, str] = {}

    def render(self, name: str, **vars: object) -> str:
        """按名称加载模板（system / reflection / tool_use），用 vars 渲染。"""
        template = self._load(name)
        return template.format_map(_SafeDict(**{k: str(v) for k, v in vars.items()}))

    # ---- internals ------------------------------------------------------

    def _load(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]

        filename = getattr(self._cfg, name, None)
        if filename is None:
            raise KeyError(
                f"prompt 模板 {name!r} 未在配置中声明；已知：{list(self._cfg.__dict__)}"
            )
        path = self._base / filename
        if not path.exists():
            raise FileNotFoundError(f"模板文件不存在：{path}")
        text = path.read_text(encoding="utf-8")
        self._cache[name] = text
        return text
