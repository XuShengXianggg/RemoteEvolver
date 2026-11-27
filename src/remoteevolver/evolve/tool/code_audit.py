"""
remoteevolver.evolve.tool.code_audit —— AST 扫描公开 API + 一致性校验(C-1 / C-2)

C-1 Code Report:
  扫 clone 下来的包根,**纯 AST 不执行**,产出"包真实导出的公开符号清单 + 签名"。
  把它塞进 wrapper 生成的 prompt,LLM 写代码不再凭记忆猜 API。

C-2 一致性校验(本文件后半):
  wrapper 生成后,AST 解析它的代码,把所有 `<package>.xxx(...)` 调用对照
  Code Report 中的真实导出符号清单,标出"幻觉 API"。

设计原则:
  - 纯 AST,不 import/不 exec:仓库随便用 sys.exit、动态 patch 都不会污染我们
  - "公开符号"语义:`__init__.py` 的 `__all__` 优先;否则 top-level 非下划线开头的
    def/class;`from .x import y` 这种再传递也算
  - 大库截断:超过 50 个公开函数,只保留入口文件(`__init__.py` / `__main__.py`)
    的导出 + 各模块 top-N 高频符号
"""

from __future__ import annotations

import ast
import difflib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


# 一些 hard-cap,防止大库把 prompt 撑爆
MAX_FUNCS_IN_REPORT = 50
MAX_CLASSES_IN_REPORT = 30
MAX_METHODS_PER_CLASS = 10
MAX_DOCSTRING_FIRST_LINE_CHARS = 150

# 扫描时跳过的目录
_SKIP_DIRS = {
    "__pycache__", "test", "tests", "testing",
    "docs", "doc", "examples", "example",
    "benchmarks", "benchmark",
    ".git", ".github", ".venv", "venv", "env",
    "build", "dist", ".tox",
}


@dataclass
class FunctionSig:
    """一个函数 / 类方法的签名摘要。"""
    name: str
    params: list[str] = field(default_factory=list)  # ["x: int", "y: str = 'a'"]
    return_annotation: str = ""                      # "str" 或 ""
    docstring_first_line: str = ""                   # 截到第一行,trim 过长


@dataclass
class ClassSig:
    """一个类的签名摘要 + 公开方法。"""
    name: str
    init_params: list[str] = field(default_factory=list)
    methods: list[FunctionSig] = field(default_factory=list)
    docstring_first_line: str = ""


@dataclass
class ModuleAPI:
    """一个 .py 模块里的公开符号。"""
    module_name: str           # "yfinance.ticker"
    file_relpath: str          # 相对 repo 根
    functions: list[FunctionSig] = field(default_factory=list)
    classes: list[ClassSig] = field(default_factory=list)


@dataclass
class PackageAPI:
    """一个包的总览。生成 wrapper 时把它塞进 prompt;写到磁盘当 Code Report。"""
    package_name: str
    repo_root_relpath: str = ""           # 包根相对 repo_dir,例 "src/yfinance" / "yfinance" / ""
    found: bool = False                   # 是否真的定位到了包根
    exported_names: list[str] = field(default_factory=list)  # __init__.py 的 __all__ 或导入再导出的名字
    modules: list[ModuleAPI] = field(default_factory=list)
    scan_warnings: list[str] = field(default_factory=list)   # 解析 warning(语法错、读不到等)

    def total_functions(self) -> int:
        return sum(len(m.functions) for m in self.modules)

    def total_classes(self) -> int:
        return sum(len(m.classes) for m in self.modules)

    def to_json(self) -> dict:
        return asdict(self)

    def to_prompt_summary(self) -> str:
        """生成给 LLM 读的紧凑摘要(进 wrapper prompt 用)。

        策略:
          1. 顶头列 __init__ 的 exported_names(LLM 看到这块就知道"我能直接调啥")
          2. 然后逐模块列 functions / classes,但有 hard-cap
          3. 每个函数:`def name(params) -> ret  # docstring 第一行`
        """
        if not self.found:
            return f"(AST 扫描未能定位到包 {self.package_name!r} 的根目录)"

        lines: list[str] = []
        if self.exported_names:
            lines.append(f"**__init__ 导出符号({len(self.exported_names)} 个)**:")
            lines.append("  " + ", ".join(self.exported_names[:80]))
            if len(self.exported_names) > 80:
                lines.append(f"  ... 还有 {len(self.exported_names)-80} 个未列")
            lines.append("")

        total_funcs = self.total_functions()
        total_classes = self.total_classes()
        lines.append(f"**模块详情(共 {len(self.modules)} 个模块,"
                     f"{total_funcs} 函数,{total_classes} 类)**:")

        funcs_shown = 0
        classes_shown = 0
        for mod in self.modules:
            if funcs_shown >= MAX_FUNCS_IN_REPORT and classes_shown >= MAX_CLASSES_IN_REPORT:
                lines.append(f"  ... 后续模块省略(达到展示上限)")
                break
            if not mod.functions and not mod.classes:
                continue
            lines.append(f"\n  模块 {mod.module_name}:")

            for fn in mod.functions:
                if funcs_shown >= MAX_FUNCS_IN_REPORT:
                    lines.append(f"    ... (函数省略)")
                    break
                params_str = ", ".join(fn.params)
                ret = f" -> {fn.return_annotation}" if fn.return_annotation else ""
                doc = f"  # {fn.docstring_first_line}" if fn.docstring_first_line else ""
                lines.append(f"    def {fn.name}({params_str}){ret}{doc}")
                funcs_shown += 1

            for cls in mod.classes:
                if classes_shown >= MAX_CLASSES_IN_REPORT:
                    lines.append(f"    ... (类省略)")
                    break
                init_params_str = ", ".join(cls.init_params)
                doc = f"  # {cls.docstring_first_line}" if cls.docstring_first_line else ""
                lines.append(f"    class {cls.name}({init_params_str}){doc}")
                for m in cls.methods[:MAX_METHODS_PER_CLASS]:
                    mp = ", ".join(m.params)
                    mret = f" -> {m.return_annotation}" if m.return_annotation else ""
                    lines.append(f"      .{m.name}({mp}){mret}")
                if len(cls.methods) > MAX_METHODS_PER_CLASS:
                    lines.append(f"      ... 还有 {len(cls.methods)-MAX_METHODS_PER_CLASS} 个方法未列")
                classes_shown += 1

        if self.scan_warnings:
            lines.append("\n**扫描 warning**:")
            for w in self.scan_warnings[:3]:
                lines.append(f"  - {w}")

        return "\n".join(lines)


# ============================================================================
# 扫描主入口
# ============================================================================

def scan_package_api(
    package_name: str,
    repo_dir: Path | str,
) -> PackageAPI:
    """对一个 clone 下来的仓库做 AST 扫描,产出 PackageAPI。

    package_name:    wrapper_gen.collect_repo_signals 已经猜出来的 import 名
    repo_dir:        仓库根目录(`data/external_tools/<slug>/repo/`)

    不会抛异常:任何步骤失败都会在 scan_warnings 里记一条,然后返回部分结果。
    """
    repo_path = Path(repo_dir)
    api = PackageAPI(package_name=package_name)

    package_root = _locate_package_root(package_name, repo_path)
    if package_root is None:
        api.scan_warnings.append(
            f"无法在 {repo_path} 下定位包 {package_name!r} 的根目录"
        )
        return api

    api.found = True
    api.repo_root_relpath = str(package_root.relative_to(repo_path))

    # 扫描所有 .py 文件
    py_files: list[Path] = []
    if package_root.is_file():
        # 单文件包(罕见但有)
        py_files = [package_root]
    else:
        for p in package_root.rglob("*.py"):
            # 跳过测试 / docs / build 目录(逐级匹配)
            if any(seg in _SKIP_DIRS for seg in p.relative_to(package_root).parts):
                continue
            py_files.append(p)

    for py_path in py_files:
        try:
            module_api = _scan_one_module(py_path, package_root, package_name, repo_path)
        except SyntaxError as e:
            api.scan_warnings.append(f"语法错: {py_path.name}: {e}")
            continue
        except Exception as e:  # noqa: BLE001
            api.scan_warnings.append(f"扫描出错: {py_path.name}: {type(e).__name__}: {e}")
            continue
        if module_api.functions or module_api.classes:
            api.modules.append(module_api)
        # __init__.py 单独处理 exported_names
        if py_path.name == "__init__.py" and py_path.parent == package_root:
            api.exported_names = _extract_exported_from_init(py_path)

    # __init__ 没找到 __all__ → 用 top-level 公开符号兜底
    if not api.exported_names and api.modules:
        # 把根级 __init__.py 模块(若存在)的 def/class 名当作 exported
        for mod in api.modules:
            if mod.module_name == package_name or mod.file_relpath.endswith("__init__.py"):
                api.exported_names = (
                    [f.name for f in mod.functions]
                    + [c.name for c in mod.classes]
                )
                break

    # 按"是否在 exported_names 里"排序模块,让重要的排前面
    if api.exported_names:
        api.modules.sort(
            key=lambda m: (
                0 if m.module_name == package_name else 1,  # 包根 __init__ 第一
                len([f for f in m.functions if f.name in api.exported_names]),
            ),
            reverse=False,
        )

    return api


def write_code_report(
    api: PackageAPI,
    repo_workspace: Path | str,
) -> Path:
    """把 PackageAPI 写到 `<workspace>/code_report.json`,返回写入的路径。

    repo_workspace 一般是 cloner 用的 `data/external_tools/<slug>/`。
    """
    p = Path(repo_workspace) / "code_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(api.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


# ============================================================================
# 内部:定位包根 + 扫单模块 + 解析 __init__ 导出
# ============================================================================

def _locate_package_root(package_name: str, repo_path: Path) -> Path | None:
    """在 repo_path 下找包根目录(含 __init__.py 的目录)或单文件 .py。

    常见布局优先级:
      1. repo_path/<package_name>/__init__.py
      2. repo_path/src/<package_name>/__init__.py
      3. repo_path/<package_name>.py     (单文件)
      4. repo_path/src/<package_name>.py
      5. 兜底:walk 一遍找第一个 __init__.py 所在目录名 == package_name 的
    """
    candidates = [
        repo_path / package_name / "__init__.py",
        repo_path / "src" / package_name / "__init__.py",
    ]
    for p in candidates:
        if p.exists():
            return p.parent

    # 单文件包
    file_candidates = [
        repo_path / f"{package_name}.py",
        repo_path / "src" / f"{package_name}.py",
    ]
    for p in file_candidates:
        if p.exists():
            return p

    # 兜底:深度搜索(可能错位)
    for p in repo_path.rglob("__init__.py"):
        # 跳过明显的非包目录
        if any(seg in _SKIP_DIRS for seg in p.relative_to(repo_path).parts):
            continue
        if p.parent.name == package_name:
            return p.parent

    return None


def _scan_one_module(
    py_path: Path,
    package_root: Path,
    package_name: str,
    repo_path: Path,
) -> ModuleAPI:
    """AST 解析单文件,抽 top-level 公开 def / class。"""
    source = py_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(py_path))

    # 模块名:基于 package_root 的相对路径,例 yfinance.ticker
    if package_root.is_file():
        module_name = package_name
        relpath = str(py_path.relative_to(repo_path))
    else:
        rel = py_path.relative_to(package_root)
        if rel.name == "__init__.py":
            module_name = package_name if rel.parent == Path(".") else (
                package_name + "." + ".".join(rel.parent.parts)
            )
        else:
            parts = list(rel.with_suffix("").parts)
            module_name = package_name + "." + ".".join(parts)
        relpath = str(py_path.relative_to(repo_path))

    module_api = ModuleAPI(module_name=module_name, file_relpath=relpath)

    for node in tree.body:
        # 跳过私有
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and not node.name.startswith("__"):
                continue
            module_api.functions.append(_extract_function_sig(node))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            module_api.classes.append(_extract_class_sig(node))

    return module_api


def _extract_function_sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionSig:
    """从 AST FunctionDef 抽签名 + docstring 第一行。"""
    params = _format_args(node.args)
    ret = ""
    if node.returns is not None:
        ret = _ast_unparse(node.returns)
    doc = ast.get_docstring(node) or ""
    first = doc.splitlines()[0].strip() if doc else ""
    if len(first) > MAX_DOCSTRING_FIRST_LINE_CHARS:
        first = first[:MAX_DOCSTRING_FIRST_LINE_CHARS] + "..."
    return FunctionSig(
        name=node.name,
        params=params,
        return_annotation=ret,
        docstring_first_line=first,
    )


def _extract_class_sig(node: ast.ClassDef) -> ClassSig:
    """从 AST ClassDef 抽 __init__ 签名 + 公开方法。"""
    init_params: list[str] = []
    methods: list[FunctionSig] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "__init__":
                init_params = _format_args(item.args, drop_self=True)
            elif not item.name.startswith("_"):
                methods.append(_extract_function_sig(item))
    doc = ast.get_docstring(node) or ""
    first = doc.splitlines()[0].strip() if doc else ""
    if len(first) > MAX_DOCSTRING_FIRST_LINE_CHARS:
        first = first[:MAX_DOCSTRING_FIRST_LINE_CHARS] + "..."
    return ClassSig(
        name=node.name,
        init_params=init_params,
        methods=methods,
        docstring_first_line=first,
    )


def _format_args(args: ast.arguments, *, drop_self: bool = False) -> list[str]:
    """把 ast.arguments 格式化成 ["x: int", "y: str = 'a'"] 列表。

    drop_self=True 时跳过第一个参数(用于类方法,排掉 self/cls)。
    """
    out: list[str] = []
    all_args = list(args.args)
    if drop_self and all_args:
        all_args = all_args[1:]

    # 默认值对齐:Python 默认值是从后往前贴的
    defaults = list(args.defaults)
    pad = len(all_args) - len(defaults)
    for i, a in enumerate(all_args):
        name = a.arg
        ann = f": {_ast_unparse(a.annotation)}" if a.annotation else ""
        if i >= pad:
            d = defaults[i - pad]
            out.append(f"{name}{ann} = {_ast_unparse(d)}")
        else:
            out.append(f"{name}{ann}")

    # *args
    if args.vararg:
        out.append(f"*{args.vararg.arg}")

    # keyword-only
    kw_defaults = list(args.kw_defaults)
    for i, a in enumerate(args.kwonlyargs):
        name = a.arg
        ann = f": {_ast_unparse(a.annotation)}" if a.annotation else ""
        d = kw_defaults[i] if i < len(kw_defaults) else None
        if d is not None:
            out.append(f"{name}{ann} = {_ast_unparse(d)}")
        else:
            out.append(name + ann)

    # **kwargs
    if args.kwarg:
        out.append(f"**{args.kwarg.arg}")

    return out


def _ast_unparse(node: ast.AST | None) -> str:
    """ast.unparse 的安全版:None / 异常 → 空串。"""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "?"


def _extract_exported_from_init(init_path: Path) -> list[str]:
    """从 __init__.py 抽公开导出:

    优先 __all__ = [...] 字面量;
    其次:`from .x import y` / `from .x import y as z` 把 y(或 z)收集进来;
    `from .x import *` 这种无法静态展开,先忽略(以模块的 top-level def/class 兜底)。
    """
    try:
        source = init_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(init_path))
    except Exception:  # noqa: BLE001
        return []

    # 1) __all__ 字面量
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return _const_list(node.value)

    # 2) from .x import y / y as z(再加上本文件 top-level 的公开符号)
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or node.level > 0):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                if not name.startswith("_"):
                    out.append(name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                out.append(node.name)

    # 去重保序
    seen: set[str] = set()
    unique: list[str] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def _const_list(node: ast.AST) -> list[str]:
    """把 [a, b, c] / (a, b, c) 字面量拆成 list[str]。非字符串元素跳过。"""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    out: list[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            out.append(elt.value)
    return out


# ============================================================================
# C-2: Wrapper 代码 vs PackageAPI 一致性校验
# ============================================================================

@dataclass
class CallIssue:
    """一条"幻觉 API"报告。"""
    kind: str                  # "undefined_attribute" | "unknown_direct_import"
    symbol: str                # 在 wrapper 里出现的引用,如 "yfinance.get_stock_price"
    line_no: int = 0           # wrapper 里的行号(直接 import 形态时为 0)
    suggestion: str = ""       # 最相近的真实符号(空 = 没靠谱的)


def audit_wrapper_calls(
    wrapper_code: str,
    package_api: PackageAPI | None,
    *,
    max_suggestions: int = 3,
) -> list[CallIssue]:
    """检查 wrapper 引用的 `<package>.xxx` / `from <package> import xxx` 是否真实存在。

    返回 CallIssue 列表;为空 = 没有幻觉。
    短路条件:
      - package_api 是 None → 返回 []
      - found=False / 0 真实符号时仍检查“只导入但未使用”;除此之外不校验 API 名称
      - wrapper 本身语法错 → 返回 [](冒烟测试会接住语法错)
    """
    if package_api is None:
        return []

    try:
        tree = ast.parse(wrapper_code)
    except SyntaxError:
        return []

    pkg_aliases, direct_imports = _collect_imports(tree, package_api.package_name)
    issues: list[CallIssue] = []

    # Importing the selected package is not evidence of using the repository.
    # Detect the deterministic loophole before validating individual API names.
    imported_names = pkg_aliases | set(direct_imports)
    used_imports = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in imported_names
    }
    if not imported_names or not used_imports:
        issues.append(CallIssue(
            kind="selected_repository_not_used",
            symbol=package_api.package_name,
            suggestion="reference a real API from the selected repository",
        ))

    if not package_api.found:
        return issues
    known = _collect_known_symbols(package_api)
    if not known:
        return issues

    # 1) `<alias>.<attr>` 形态:alias 指向 package 本身
    seen_attrs: set[tuple[str, str]] = set()  # 同一处重复出现的只报 1 次
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            alias = node.value.id
            if alias not in pkg_aliases:
                continue
            attr = node.attr
            if attr.startswith("_"):
                # private 默认认为是误调用,但不强报(可能是 __version__ 这种)
                continue
            key = (alias, attr)
            if key in seen_attrs:
                continue
            seen_attrs.add(key)
            if attr not in known:
                issues.append(CallIssue(
                    kind="undefined_attribute",
                    symbol=f"{alias}.{attr}",
                    line_no=getattr(node, "lineno", 0),
                    suggestion=_suggest(attr, known, n=max_suggestions),
                ))

    # 2) `from <package> import <real_name>` 形态
    seen_imports: set[str] = set()
    for real_name in direct_imports.values():
        if real_name in seen_imports:
            continue
        seen_imports.add(real_name)
        if real_name not in known:
            issues.append(CallIssue(
                kind="unknown_direct_import",
                symbol=f"from {package_api.package_name} import {real_name}",
                line_no=0,
                suggestion=_suggest(real_name, known, n=max_suggestions),
            ))

    return issues


def format_issues_for_llm(issues: list[CallIssue], *, max_lines: int = 12) -> str:
    """把 CallIssue 列表渲染成给 LLM 修复 prompt 用的紧凑文本。"""
    if not issues:
        return "(没有发现幻觉 API)"
    lines: list[str] = [
        f"共 {len(issues)} 处引用了不存在的符号 —— 必须改成"
        f"上面【公开 API】清单里真实存在的名字:"
    ]
    for i, iss in enumerate(issues[:max_lines], 1):
        line_tag = f" (行 {iss.line_no})" if iss.line_no else ""
        sug = f"  最接近的真实符号: {iss.suggestion}" if iss.suggestion else "  (公开 API 里没有相近的)"
        lines.append(f"  {i}. {iss.kind}:  {iss.symbol}{line_tag}\n{sug}")
    if len(issues) > max_lines:
        lines.append(f"  ... 还有 {len(issues)-max_lines} 条未列")
    return "\n".join(lines)


# ---- 内部:把 PackageAPI 摊平成符号集 + 解析 imports + 模糊建议 ----

def _collect_known_symbols(api: PackageAPI) -> set[str]:
    """把 PackageAPI 里所有真实存在的"可调用名字"摊平成一个集合。

    包括:exported_names + 所有模块的 functions/classes/methods 名。
    类方法名也加入,因为 LLM 可能调 `Class.method` 形式;
    我们这里不做严格的所属类匹配(那太复杂),粗略放行即可。
    """
    known: set[str] = set(api.exported_names)
    for m in api.modules:
        for f in m.functions:
            known.add(f.name)
        for c in m.classes:
            known.add(c.name)
            for method in c.methods:
                known.add(method.name)
    return known


def _collect_imports(
    tree: ast.Module,
    package_name: str,
) -> tuple[set[str], dict[str, str]]:
    """从 wrapper 的 AST 里抽:
      - pkg_aliases:    所有指向 package 本身的名字(`import pkg` 给 "pkg",
                       `import pkg as p` 给 "p",`import pkg.sub` 给 "pkg")
      - direct_imports: `from pkg import X [as Y]` 收的 {alias_in_code: real_name}
    """
    pkg_aliases: set[str] = set()
    direct_imports: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top == package_name:
                    pkg_aliases.add(alias.asname or top)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == package_name or mod.startswith(package_name + "."):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    direct_imports[alias.asname or alias.name] = alias.name
    return pkg_aliases, direct_imports


def _suggest(target: str, known: set[str], *, n: int = 3, cutoff: float = 0.55) -> str:
    """模糊找最接近 target 的真实符号,逗号拼回去。找不到返回空串。"""
    matches = difflib.get_close_matches(target, list(known), n=n, cutoff=cutoff)
    return ", ".join(matches)
