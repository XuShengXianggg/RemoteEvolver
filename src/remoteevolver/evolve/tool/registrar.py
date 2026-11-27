"""Transactionally register generated wrappers in the run-scoped tool directory.
The owning Agent restarts MCP after success to refresh its schemas.
"""

from __future__ import annotations

import ast
import os
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from ...infra.paths import runtime_path
from typing import Callable

from .isolation import assert_registration_target_is_evolution_only


TOOLS_DIR = runtime_path("tools")
TOOLS_INIT = TOOLS_DIR / "__init__.py"


@dataclass
class RegistrationResult:
    tool_name: str
    file_path: Path
    init_updated: bool          # __init__.py 是否被修改(已存在则不动)
    message: str                # 给人看的状态摘要
    certification_path: Path | None = None


def register_wrapper(
    wrapper_code: str,
    tool_name: str,
    *,
    tools_dir: Path | str = TOOLS_DIR,
    overwrite: bool = False,
    certification_record: dict | None = None,
    post_write_probe: Callable[[RegistrationResult], None] | None = None,
) -> RegistrationResult:
    """落地 wrapper 文件 + 更新 tools/__init__.py 的 import 行。

    Args:
        wrapper_code: 已通过冒烟测试的完整 wrapper.py 文本。
        tool_name:    工具名,与 wrapper 里 @mcp.tool() 装饰的函数名一致。
        tools_dir:    remoteevolver/tools/ 路径(默认就这个;改它只在测试时)。
        overwrite:    True 时强制覆盖已有同名文件;False 时拒绝并抛 FileExistsError。
    """
    if not tool_name.isidentifier():
        raise ValueError(f"tool_name 必须是合法 Python 标识符:{tool_name!r}")

    tools_dir = Path(tools_dir)
    assert_registration_target_is_evolution_only(tools_dir)
    tools_dir.mkdir(parents=True, exist_ok=True)
    tool_path = tools_dir / f"{tool_name}.py"

    if tool_path.exists() and not overwrite:
        raise FileExistsError(
            f"{tool_path} 已存在;传 overwrite=True 强制覆盖,或换个 tool_name"
        )

    # 1) 写工具文件
    init_path = tools_dir / "__init__.py"
    ledger_path = tools_dir / ".evolved_tools.json"
    certification_path = tools_dir / ".certifications" / f"{tool_name}.json"
    old_tool = tool_path.read_bytes() if tool_path.exists() else None
    old_init = init_path.read_bytes() if init_path.exists() else None
    old_ledger = ledger_path.read_bytes() if ledger_path.exists() else None
    old_certification = certification_path.read_bytes() if certification_path.exists() else None
    try:
        registered = []
        if old_ledger is not None:
            value = json.loads(old_ledger.decode("utf-8"))
            registered = value if isinstance(value, list) else []
        if tool_name not in registered:
            registered.append(tool_name)
        # Write-ahead ledger: a hard kill after this point is recoverable by
        # strict cold-start reset, even if __init__.py was never updated.
        _atomic_write_text(
            ledger_path, json.dumps(sorted(registered), ensure_ascii=False) + "\n",
        )
        _atomic_write_text(tool_path, wrapper_code)
        if certification_record is not None:
            _atomic_write_text(
                certification_path,
                json.dumps(certification_record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            )
        init_updated = _ensure_import_line(init_path, tool_name)
        provisional = RegistrationResult(
            tool_name=tool_name, file_path=tool_path, init_updated=init_updated,
            message=f"wrote {tool_path}; {'updated' if init_updated else 'kept'} __init__.py",
            certification_path=(certification_path if certification_record is not None else None),
        )
        if post_write_probe is not None:
            post_write_probe(provisional)
    except BaseException:
        if old_tool is None:
            tool_path.unlink(missing_ok=True)
        else:
            _atomic_write_bytes(tool_path, old_tool)
        if old_init is None:
            init_path.unlink(missing_ok=True)
        else:
            _atomic_write_bytes(init_path, old_init)
        if old_ledger is None:
            ledger_path.unlink(missing_ok=True)
        else:
            _atomic_write_bytes(ledger_path, old_ledger)
        if old_certification is None:
            certification_path.unlink(missing_ok=True)
        else:
            _atomic_write_bytes(certification_path, old_certification)
        raise

    return provisional


# ---- 内部 --------------------------------------------------------------

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))

def _ensure_import_line(init_path: Path, tool_name: str) -> bool:
    """在 remoteevolver/tools/__init__.py 末尾追加 `from . import <tool_name>`,若未存在。

    返回是否实际修改了文件。
    """
    if not init_path.exists():
        _atomic_write_text(init_path, f"from . import {tool_name}  # noqa: F401\n")
        return True

    text = init_path.read_text(encoding="utf-8")
    # 简单匹配是否已经在 import 里
    tree = ast.parse(text, filename=str(init_path))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module is None
        for alias in node.names
    }
    if tool_name in imported:
        return False

    if not text.endswith("\n"):
        text += "\n"
    text += f"from . import {tool_name}  # noqa: F401 (added by evolve.tool.registrar)\n"
    _atomic_write_text(init_path, text)
    return True
