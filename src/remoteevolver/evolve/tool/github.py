"""
remoteevolver.evolve.github —— GitHub REST API HTTP 客户端

只覆盖工具自进化用到的部分:
  - search_repositories(query, ...)

设计要点:
  - 走 stdlib urllib(零新依赖,跟 weather 工具同款)
  - 带身份认证(GITHUB_TOKEN)→ 5000 req/h、30 search/min
  - 首次握手到 api.github.com 在国内约 3 秒,所以 timeout 默认 30s + 3 次重试
  - 透传 GitHub 的 rate-limit 报错(403 + 'rate limit' / 'secondary rate limit')
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError


_API_BASE = os.getenv("REMOTEEVOLVER_GITHUB_API", "https://api.github.com").rstrip("/")


class GitHubError(Exception):
    """对外暴露的统一 GitHub 错误,避免上层处理一堆 urllib 异常类型。"""


class RateLimitError(GitHubError):
    """命中限流。e.reset_at_epoch 是预计恢复时间(unix epoch s),没有就是 None。"""

    def __init__(self, message: str, reset_at_epoch: int | None = None) -> None:
        super().__init__(message)
        self.reset_at_epoch = reset_at_epoch


@dataclass
class Repo:
    """搜索结果里一条仓库的元数据(只保留下游用到的字段)。"""
    full_name: str            # "owner/name"
    html_url: str
    clone_url: str            # https clone 地址,可直接 git clone
    description: str
    language: str             # 主语言,无则空串
    license: str              # SPDX id,无则空串
    stars: int
    forks: int
    pushed_at: str            # ISO8601;判断仓库活跃度用
    default_branch: str
    # B-2.5 新增:语义评估前会按需拉 README;在 search 阶段一般是空串。
    readme: str = ""


def _parse_repo(item: dict[str, Any]) -> Repo:
    lic = (item.get("license") or {}) or {}
    return Repo(
        full_name=item.get("full_name", ""),
        html_url=item.get("html_url", ""),
        clone_url=item.get("clone_url", ""),
        description=(item.get("description") or "").strip(),
        language=item.get("language") or "",
        license=lic.get("spdx_id") or "",
        stars=int(item.get("stargazers_count", 0)),
        forks=int(item.get("forks_count", 0)),
        pushed_at=item.get("pushed_at") or "",
        default_branch=item.get("default_branch") or "main",
    )


class GitHubClient:
    """对外只暴露 search_repositories;后续 clone/读 README 等扩展再加方法。"""

    def __init__(
        self,
        token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        user_agent: str = "remoteevolver-evolve/0.1",
    ) -> None:
        # token 缺省读 env;.env 由上层 load_dotenv() 负责加载
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN", "")
        self.timeout = float(os.getenv("REMOTEEVOLVER_GITHUB_TIMEOUT", str(timeout)))
        self.max_retries = max(1, max_retries)
        self.user_agent = user_agent

    # ---- 公开 API -------------------------------------------------------

    def search_repositories(
        self,
        query: str,
        *,
        sort: str = "",                # ""=GitHub best-match relevance(默认)| stars | forks | updated
        order: str = "desc",
        per_page: int = 50,            # B-2.5: 默认拉 50 条,留给 LLM 语义粗筛挑
    ) -> list[Repo]:
        """按 query 搜公开仓库。query 直接拼到 ?q=,可带 GitHub 限定符。

        sort:
          - ""      → 不传 sort 参数,GitHub 按 best-match(关键词相关性)返回。
                      **这是默认**,因为高 stars 老库会压制低 stars 但贴合任务的新库。
          - "stars" → 按 stars 降序(老行为,会埋没低星新库)
          - 其他    → forks / help-wanted-issues / updated
        """
        if not query.strip():
            raise GitHubError("query 不能为空")
        params: dict[str, str | int] = {
            "q": query,
            "order": order,
            "per_page": min(max(per_page, 1), 100),
        }
        if sort:
            params["sort"] = sort
        qs = urllib.parse.urlencode(params)
        data = self._get_json(f"/search/repositories?{qs}")
        return [_parse_repo(it) for it in data.get("items", [])]

    def fetch_readme(
        self,
        full_name: str,
        *,
        max_chars: int = 3000,
    ) -> str:
        """拉一个仓库的 README(走 GitHub Contents API 的 /readme 端点,自动识别文件名)。

        Args:
            full_name: "owner/repo"
            max_chars: 截断长度。README 太长会撑爆 LLM prompt,3000 字符已经够语义判断。

        Returns:
            README 文本(已做 base64 解码 + 截断)。任何失败都返回 "",不抛 ——
            search/assessment 阶段缺少 README 不是致命错误。
        """
        if "/" not in full_name:
            return ""
        try:
            data = self._get_json(f"/repos/{full_name}/readme")
        except (GitHubError, RateLimitError):
            return ""

        content_b64 = data.get("content") or ""
        if not content_b64:
            return ""
        try:
            # GitHub 的 base64 编码每 60 字符一个 \n,标准库能正确解码
            import base64
            raw = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
        return raw[:max_chars] if max_chars > 0 else raw

    # ---- 内部 ------------------------------------------------------------

    def _get_json(self, path: str) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{_API_BASE}{path}"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": self.user_agent,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_exc: BaseException | None = None
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
                if e.code in (403, 429) and "rate limit" in body.lower():
                    reset = e.headers.get("X-RateLimit-Reset")
                    raise RateLimitError(
                        f"GitHub rate limit hit: {body}",
                        reset_at_epoch=int(reset) if reset and reset.isdigit() else None,
                    ) from e
                # 4xx 一般是 query 问题,重试无意义
                if 400 <= e.code < 500:
                    raise GitHubError(f"HTTP {e.code} {e.reason}: {body}") from e
                last_exc = e
            except (URLError, TimeoutError) as e:
                # SSL 握手超时 / 暂时性网络错误 —— 重试有意义
                last_exc = e

            if attempt < self.max_retries:
                # 指数退避:1s, 2s, 4s ...
                time.sleep(2 ** (attempt - 1))

        raise GitHubError(
            f"GitHub 请求失败,重试 {self.max_retries} 次仍未成功:{last_exc!r}"
        ) from last_exc
