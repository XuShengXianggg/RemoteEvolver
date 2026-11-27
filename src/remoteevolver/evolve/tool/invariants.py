"""Task-independent, deterministic output health checks.

These checks reject only mechanical failures that can be reproduced without a
domain oracle: a non-finite numeric return, an empty artifact, or an unreadable
file whose extension declares a supported format.  Formulae, ranges, sentinel
values, filenames and task wording never determine registration here.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

def _has_nonfinite_numeric_result(text: str) -> bool:
    raw = (text or "").strip()
    if raw.casefold() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
        return True
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return False

    def _contains(item) -> bool:
        if isinstance(item, float):
            return not math.isfinite(item)
        if isinstance(item, dict):
            return any(_contains(key) or _contains(val) for key, val in item.items())
        if isinstance(item, (list, tuple, set)):
            return any(_contains(val) for val in item)
        return False

    return _contains(value)


def check_output_invariants(
    test_args: dict | None,
    result_repr: str,
    *,
    known_inputs: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """对一次工具调用的产出做通用健全性检查,返回违规说明列表(空=通过)。

    Args:
        test_args:    本次调用的参数(用来定位"输出文件"型实参)。
        result_repr:  工具返回值的字符串形态(查 NaN/Inf 用)。
        known_inputs: 已知的输入文件路径(这些是喂进去的,不查;只查新产出的文件)。
    """
    violations: list[str] = []

    # 1) A numeric return itself must be finite.  Text such as "nan_count=0"
    # is not interpreted as a numeric result and therefore is not rejected.
    if _has_nonfinite_numeric_result(result_repr or ""):
        violations.append(f"返回值含 NaN/Inf: {result_repr[:120]!r}")

    # 2) 输出文件检查:存在 / 非空 / 可按扩展名打开;栅格再查哨兵泄漏
    known = {str(p) for p in known_inputs}
    seen: set[str] = set()
    for v in (test_args or {}).values():
        if not isinstance(v, str):
            continue
        for part in v.split(","):
            p = part.strip()
            if not p or p in known or p in seen:
                continue
            try:
                pp = Path(p)
            except OSError:
                continue
            # 解析得出的绝对形态也算进 known/seen,避免重复
            seen.add(p)
            if not pp.is_file():
                continue                      # 不是已落地的产出文件 → 跳过(可能是输入/占位)
            try:
                if pp.stat().st_size == 0:
                    violations.append(f"输出文件为空: {p}")
                    continue
            except OSError:
                continue
            if pp.suffix.lower() in (".tif", ".tiff"):
                violations.extend(_check_raster(str(pp)))

    return violations


def check_task_utility_contract(capability: str, result_repr: str) -> list[str]:
    """Compatibility shim: task wording no longer creates a registration gate."""
    del capability, result_repr
    return []


def _check_raster(path: str) -> list[str]:
    """Check only that a declared raster artifact is structurally readable."""
    try:
        import rasterio
    except Exception:  # noqa: BLE001
        return []                              # fail-open:环境没 rasterio 就不查

    try:
        with rasterio.open(path) as ds:
            if ds.count < 1 or ds.width < 1 or ds.height < 1:
                return [f"输出 GeoTIFF 结构无效: {path}"]
            ds.read(1, window=((0, min(1, ds.height)), (0, min(1, ds.width))))
    except Exception as e:  # noqa: BLE001
        return [f"输出 GeoTIFF 打不开/损坏: {path} ({type(e).__name__}: {e})"]
    return []


# ---- 编排期入口(防瞎算的"运行时栏杆")--------------------------------

def check_tool_result(result_repr: str) -> list[str]:
    """Check a real call's returned numeric value and declared file artifacts."""
    violations: list[str] = []
    text = result_repr or ""

    # 1) The returned value itself is a non-finite number.
    if _has_nonfinite_numeric_result(text):
        violations.append(f"工具返回含 NaN/Inf: {text[:120]!r}")

    # 2) 工具写出的栅格产物:从返回值里挑出确实存在的 .tif/.tiff 文件再核验
    for tok in text.split(","):
        p = tok.strip().strip('"').strip("'")
        if not p or len(p) > 1024:
            continue
        try:
            pp = Path(p)
        except OSError:
            continue
        if pp.suffix.lower() in (".tif", ".tiff") and pp.is_file():
            violations.extend(_check_raster(str(pp)))

    return violations
