"""
remoteevolver.evolve.tool.cloner —— git clone + pip install

职责:
  - 给一个 GitHub clone_url 或 owner/repo,落到 ./data/external_tools/<name>/repo/
  - 同时尝试 pip install,让后续 wrapper.py 能直接 `import xxx`

第一版的简化(等跑通端到端再补):
  - 共用 remoteevolver 主 venv,不每工具一个独立 venv
  - 不指定 commit / tag,默认 clone 默认分支
  - shallow clone (--depth 1) 节省带宽

错误处理:
  - 所有外部命令失败统一抛 CloneError(带 stderr 摘要),上层一眼能看出是 git
    还是 pip 出问题
"""

from __future__ import annotations

import re
import os
import signal
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from ...infra.paths import runtime_path


DEFAULT_BASE_DIR = runtime_path("repositories")


class CloneError(Exception):
    """clone / install 失败统一异常,带 stderr 摘要给 LLM 看。"""


def _remove_tree(path: Path) -> None:
    """Remove a repository tree, including Windows read-only Git pack files."""
    if not path.exists():
        return

    def _remove_readonly(function, target, exc_info):  # noqa: ANN001, ARG001
        try:
            os.chmod(target, stat.S_IWRITE)
        except OSError:
            pass
        function(target)

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_remove_readonly)
        else:
            shutil.rmtree(path, onerror=_remove_readonly)
    except OSError as exc:
        raise CloneError(
            f"cannot clean stale repository before clone: {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if path.exists():
        raise CloneError(f"cannot clean stale repository before clone: {path}")


@dataclass
class ClonedRepo:
    """clone 完之后下游需要知道的信息。"""
    full_name: str              # "owner/repo"
    slug: str                   # "repo"(去掉 owner,落地目录名用)
    base_dir: Path              # data/external_tools/<slug>/
    repo_dir: Path              # data/external_tools/<slug>/repo/
    pip_installed: bool         # pip install 是否成功;失败时 wrapper 可能要 raw import
    install_log_tail: str       # pip install 输出末尾,失败时给 LLM 诊断用


# ---- 公开 API ----------------------------------------------------------

def clone_and_install(
    repo: str,
    *,
    base_dir: Path | str = DEFAULT_BASE_DIR,
    timeout: int = 180,
    force: bool = False,
    clone_attempts: int = 2,
) -> ClonedRepo:
    """clone GitHub 仓库到 ./data/external_tools/<slug>/repo/,并尝试 pip install。

    Args:
        repo:    "owner/name" 或完整 https clone_url。
        base_dir: 落地根目录,默认 data/external_tools/。
        timeout: 每个子命令超时(秒)。
        force:   True 时若目标目录已存在,先删后 clone;False 时直接复用。
        clone_attempts: 暂时性网络故障的最大 clone 尝试次数。默认同一仓库
            最多尝试两次；两次均失败后交由上层停止任务，等待人工排查网络。

    Returns:
        ClonedRepo,字段 pip_installed 告诉你 import 现在是否可用。

    Raises:
        CloneError: git clone 失败(网络/权限/不存在);pip install 失败**不会**抛
        —— 而是 pip_installed=False 让上层决定怎么处理。
    """
    timeout = float(os.getenv("REMOTEEVOLVER_CLONE_TIMEOUT", str(timeout)))
    full_name, clone_url = _normalize_repo(repo)
    slug = full_name.replace("/", "--")

    base_dir = Path(base_dir)
    work_dir = base_dir / slug
    repo_dir = work_dir / "repo"

    work_dir.mkdir(parents=True, exist_ok=True)

    # ---- git clone ----
    if repo_dir.exists():
        if force:
            _remove_tree(repo_dir)
        else:
            # 已经存在 → 跳过 clone,直接尝试 install
            pass

    if not repo_dir.exists():
        _clone_with_retries(
            clone_url, repo_dir, timeout=timeout, attempts=clone_attempts,
        )

    # ---- pip install ----
    pip_ok, pip_log = _try_pip_install(repo_dir, timeout=timeout)

    return ClonedRepo(
        full_name=full_name,
        slug=slug,
        base_dir=work_dir,
        repo_dir=repo_dir,
        pip_installed=pip_ok,
        install_log_tail=pip_log[-2000:] if pip_log else "",
    )


def _clone_with_retries(
    clone_url: str,
    repo_dir: Path,
    *,
    timeout: int,
    attempts: int,
    sleep=time.sleep,
) -> None:
    """对暂时性 git 网络错误做有界重试，并清掉失败留下的半成品目录。"""
    attempts = max(1, int(attempts))
    last_error: CloneError | None = None
    for attempt in range(1, attempts + 1):
        try:
            _run(
                [
                    "git",
                    "-c", "http.version=HTTP/1.1",
                    "-c", "http.lowSpeedLimit=1024",
                    "-c", "http.lowSpeedTime=30",
                    "clone", "--depth", "1", "--single-branch",
                    "--filter=blob:none", clone_url, str(repo_dir),
                ],
                timeout=timeout,
                error_prefix="git clone",
            )
            return
        except CloneError as exc:
            last_error = exc
            # git 失败经常留下目标目录；不清会让下一次误判为已 clone。
            if repo_dir.exists():
                _remove_tree(repo_dir)
            if attempt >= attempts or not _is_transient_clone_error(exc):
                break
            sleep(min(2 ** (attempt - 1), 4))
    assert last_error is not None
    if attempts > 1 and _is_transient_clone_error(last_error):
        raise CloneError(
            f"git clone transient failure after {attempts} attempts:\n{last_error}"
        ) from last_error
    raise last_error


def _is_transient_clone_error(exc: BaseException) -> bool:
    """区分可重试的链路抖动与仓库不存在、权限拒绝等确定性错误。"""
    text = str(exc).lower()
    markers = (
        "could not connect", "connection reset", "connection timed out",
        "recv failure", "failed to connect", "network is unreachable",
        "temporary failure", "remote end hung up", "unexpected eof",
        "tls", "ssl", "http 429", "http 500", "http 502", "http 503",
        "http 504", "timed out after",
    )
    return any(marker in text for marker in markers)


# ---- 内部 --------------------------------------------------------------

def _normalize_repo(repo: str) -> tuple[str, str]:
    """接受 'owner/name' 或 https URL,返回 (full_name, clone_url)。"""
    repo = repo.strip().rstrip("/")
    # https://github.com/owner/repo(.git)?
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", repo)
    if m:
        owner, name = m.group(1), m.group(2)
        return f"{owner}/{name}", f"https://github.com/{owner}/{name}.git"
    # owner/name
    m = re.match(r"^([^/\s]+)/([^/\s]+)$", repo)
    if m:
        owner, name = m.group(1), m.group(2)
        return f"{owner}/{name}", f"https://github.com/{owner}/{name}.git"
    raise CloneError(f"repo 格式不识别: {repo!r}(应为 'owner/name' 或 https url)")


def _try_pip_install(repo_dir: Path, *, timeout: int) -> tuple[bool, str]:
    """对 clone 下来的目录尝试 pip install,并立即验证 import 能跑。

    优先级:
      1. 仓库根有 pyproject.toml / setup.py → `pip install -e <repo_dir>`
      2. 没有可安装的入口 → 跳过,返回 (False, "no installable entry")

    返回 (是否成功, 完整日志)。失败不抛,让上层决定要不要继续。

    重要:本函数用 sys.executable 装 + 验证 ── 也就是说"跑 evolve_demo 的那个
    Python 装到哪里就验证哪里"。如果用户用 base conda 跑 demo 但用 .venv 跑
    Agent,这里 import 检查会**虚假成功**(装在 base 验证在 base),但 Agent
    那边会炸。CLI 层会显式告知 sys.executable 让用户对照,降低踩坑概率。
    """
    has_pyproject = (repo_dir / "pyproject.toml").exists()
    has_setup = (repo_dir / "setup.py").exists()
    if not (has_pyproject or has_setup):
        return False, "no pyproject.toml / setup.py — repo is not an installable package"

    cmd = [sys.executable, "-m", "pip", "install", "-e", str(repo_dir), "--quiet"]
    try:
        out = _run_process_tree_bounded(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"pip install timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return False, f"pip install threw: {type(e).__name__}: {e}"

    log = (out.stdout or "") + (out.stderr or "")
    if out.returncode != 0:
        return False, log

    # pip 报成功还不算完 —— 真 import 一次确认没装废。
    # 用 sys.executable -c 子进程,免得污染主进程 sys.modules。
    package_name = _guess_top_package_name(repo_dir)
    if package_name:
        check = _run_process_tree_bounded(
            [sys.executable, "-c", f"import {package_name}"], timeout=30,
        )
        if check.returncode != 0:
            return False, (
                log + f"\n[verify] pip OK 但 `import {package_name}` 失败:\n"
                f"{check.stderr.strip()}\n"
                "可能 setup 没声明这个 package,或装到了别的 Python env。"
            )
    return True, log


def _guess_top_package_name(repo_dir: Path) -> str:
    """非常粗的猜:从 pyproject.toml / setup.py 找 name,失败用目录名。"""
    pp = repo_dir / "pyproject.toml"
    if pp.exists():
        try:
            text = pp.read_text(encoding="utf-8", errors="replace")
            m = re.search(r'name\s*=\s*"([^"]+)"', text)
            if m:
                return m.group(1).strip().replace("-", "_")
        except Exception:  # noqa: BLE001
            pass
    sp = repo_dir / "setup.py"
    if sp.exists():
        try:
            text = sp.read_text(encoding="utf-8", errors="replace")
            m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', text)
            if m:
                return m.group(1).strip().replace("-", "_")
        except Exception:  # noqa: BLE001
            pass
    return repo_dir.name.replace("-", "_")


def _run(cmd: list[str], *, timeout: int, error_prefix: str) -> str:
    """跑外部命令,失败抛 CloneError(带 stderr 摘要)。"""
    try:
        out = _run_process_tree_bounded(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise CloneError(f"{error_prefix} timed out after {timeout}s: {' '.join(cmd)}") from e
    except FileNotFoundError as e:
        raise CloneError(f"{error_prefix} 命令未找到({cmd[0]});请确认已安装并在 PATH") from e

    if out.returncode != 0:
        stderr_tail = (out.stderr or "").strip().splitlines()[-5:]
        raise CloneError(
            f"{error_prefix} failed (exit {out.returncode})\n"
            f"  cmd : {' '.join(cmd)}\n"
            f"  tail: {chr(10).join(stderr_tail)}"
        )
    return out.stdout or ""


def _run_process_tree_bounded(
    cmd: list[str], *, timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run a command with a real wall-clock bound, including descendants.

    ``subprocess.run(timeout=...)`` kills only the direct child. On Windows,
    pip/build backends may keep inherited stdout/stderr handles open, causing
    ``communicate()`` to wait indefinitely after the nominal timeout. Start a
    new process group and terminate the entire tree before collecting output.
    """
    popen_kwargs: dict[str, object] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(proc)
        # Do not call communicate() again: a grandchild can retain the pipe
        # handles even after the direct child is gone and defeat the timeout.
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        raise subprocess.TimeoutExpired(
            cmd, timeout, output=exc.output, stderr=exc.stderr,
        ) from exc
    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr,
    )


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except Exception:  # noqa: BLE001
            proc.kill()
        if proc.poll() is None:
            proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
