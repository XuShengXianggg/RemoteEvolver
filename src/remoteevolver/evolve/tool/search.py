"""
remoteevolver.evolve.tool.search —— 能力描述 → GitHub 搜索 → 评估 → top-N 候选

主入口:find_candidates(capability, llm, ...) -> (used_query, ranked, search_rounds)

流程(每一轮):
  1. (raw_query=False 时)LLM 把自然语言描述转成 GitHub 搜索 query。
  2. 调 GitHubClient.search_repositories;0 命中时按 _relax_query 剥限定符重试。
  3. 重排:对返回的仓库按公式打分(stars 对数 + license + 描述 + 活跃度)。
  4. **LLM 相关性评估**(search_eval.check_relevance):
       SUFFICIENT  → 结束,返回当前 ranked
       INSUFFICIENT → 让 LLM 重写 query(search_eval.reformulate_query),进下一轮
  5. 总轮数上限默认 3;最后一轮即便 INSUFFICIENT 也返回拿到的东西(由调用方兜底)。

所有可调参数(prompt / 降级顺序 / 打分权重)都在 configs/tool-evolution.yaml,
通过 remoteevolver.evolve.tool.config.EvolveConfig 注入。
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ...llm.base import BaseLLM, Message
from .config import EvolveConfig, ScoreConfig, load_evolve_config
from .github import GitHubClient, Repo
from .planner import SearchPlan, plan_search
from .search_eval import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MIN_RELEVANT,
    DEFAULT_TOP_N_FOR_EVAL,
    RelevanceVerdict,
    check_relevance,
    reformulate_plan,
    reformulate_query,
)


DEFAULT_SEARCH_MAX_WORKERS = 3


@dataclass
class ScoredRepo:
    """带分的仓库候选。分数越高越优,score_reason 给人看用的诊断。"""
    repo: Repo
    score: float
    score_reason: str


@dataclass
class SearchRound:
    """一次搜索循环的留痕。多轮串起来就是完整的搜索轨迹。

    R2 双通道后:
      - query 是"本轮所有通道 query 用 ' || ' 拼起来"的可读摘要(便于日志)
      - plan 字段记本轮真实的 SearchPlan(text_hint + topics + sources)
    """
    round_idx: int
    query: str                                 # 摘要:'pyjwt || topic:jwt || topic:auth'
    raw_repo_count: int                        # 本轮所有通道合并去重后的命中数
    verdict: RelevanceVerdict | None = None    # LLM 相关性裁定;raw_query 时 None
    plan: SearchPlan | None = None             # R2:留痕,便于事后看 LLM 出了啥


# ---- 能力描述 → 搜索 query ---------------------------------------------

def build_query_from_capability(
    capability: str, llm: BaseLLM, *, config: EvolveConfig | None = None,
) -> str:
    """让 LLM 把自然语言能力描述转 GitHub 搜索 query。

    Args:
        capability: 用户的自然语言描述。
        llm: LLM 客户端。
        config: 可注入的 EvolveConfig;None 时读 the run-scoped tool configuration。
    """
    cfg = config or load_evolve_config()
    prompt = cfg.query_builder.prompt
    if "{capability}" not in prompt:
        raise ValueError("query_builder.prompt 必须包含占位符 {capability}")
    msg = Message(role="user", content=prompt.format(capability=capability.strip()))
    resp = llm.generate([msg])
    q = (resp.text or "").strip()
    # 防御:LLM 偶尔会把 query 套引号 / 加 "搜索:" 前缀,简单清洗
    q = q.strip("`'\"")
    q = re.sub(r"^(query|search|搜索|q)[:：]\s*", "", q, flags=re.IGNORECASE)
    if not q:
        raise ValueError(f"LLM 返回空 query;原描述 = {capability!r}")
    return q


# ---- 重排打分 -----------------------------------------------------------

def _score(repo: Repo, w: ScoreConfig, need_kw: "frozenset[str]" = frozenset()) -> tuple[float, str]:
    """对单个 repo 打分。权重全部从 ScoreConfig 来,改 yaml 即可调整。

    ★ 相关性主导:先看仓库名/描述命中了多少个"需求关键词"(按比例 × relevance_weight),
      再把 star 作为辅助项加上 —— 让"符合需求"压过"名气",符合"小而贴合的库也该上"。
    """
    bits: list[str] = []
    s = 0.0
    # 相关性(主):full_name 含仓库名,description 是一句话简介
    if need_kw:
        hay = f"{repo.full_name} {repo.description or ''}".lower()
        hit = sum(1 for kw in need_kw if kw in hay)
        rel = (hit / len(need_kw)) * w.relevance_weight
        s += rel
        bits.append(f"rel={hit}/{len(need_kw)}*{w.relevance_weight:g}={rel:.1f}")
    # star(辅):log(stars+1),让 10000 星与 100 星差距合理而非线性
    star = math.log10(repo.stars + 1) * w.stars_log_weight
    s += star
    bits.append(f"log10(stars+1)*{w.stars_log_weight:g}={star:.1f}")

    if repo.license:
        s += w.has_license_bonus
        bits.append(f"license={repo.license}(+{w.has_license_bonus:g})")
    else:
        bits.append("no-license(0)")

    if repo.description:
        s += w.has_description_bonus
        bits.append(f"has-desc(+{w.has_description_bonus:g})")
    else:
        s += w.no_description_penalty
        bits.append(f"no-desc({w.no_description_penalty:+g})")

    # pushed_at 形如 "2024-08-12T03:11:22Z";粗略提取年份判断活跃度
    year_match = re.match(r"(\d{4})-", repo.pushed_at or "")
    if year_match:
        year = int(year_match.group(1))
        if year >= w.recent_year_threshold:
            s += w.recent_bonus
            bits.append(f"recent({year},+{w.recent_bonus:g})")
        elif year <= w.stale_year_threshold:
            s += w.stale_penalty
            bits.append(f"stale({year},{w.stale_penalty:+g})")

    return s, " ".join(bits)


def _rerank(
    repos: list[Repo], w: ScoreConfig, need_kw: "frozenset[str]" = frozenset(),
) -> list[ScoredRepo]:
    scored = [
        ScoredRepo(repo=r, **dict(zip(("score", "score_reason"), _score(r, w, need_kw))))
        for r in repos
    ]
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


def _need_keywords(plan: SearchPlan) -> set[str]:
    """从 plan 抽"需求关键词"(短语拆词 + text_hint),供粗排算相关性。长度<3 的丢掉。"""
    kw: set[str] = set()
    for ph in plan.phrases:
        for tok in ph.lower().split():
            if len(tok) >= 3:
                kw.add(tok)
    if plan.text_hint and len(plan.text_hint) >= 3:
        kw.add(plan.text_hint.lower())
    return kw


def _relax_query(query: str, drop_order: list[str]) -> list[str]:
    """生成 query 的渐进放宽版本:[原始, 去 stars, 去 pushed, ...]。

    drop_order 从 RelaxConfig 来。重复版本会被去重;
    不在 drop_order 里的限定符 = 永远不剥。
    """
    variants = [query]
    cur = query
    for qual in drop_order:
        # 匹配 "qual<value>",value 直到下一个空格或字符串末尾
        cur = re.sub(rf"\s*{re.escape(qual)}\S*", "", cur).strip()
        cur = re.sub(r"\s+", " ", cur)
        if cur and cur != variants[-1]:
            variants.append(cur)
    return variants


# ---- 对外主函数 ---------------------------------------------------------

def _search_one_query(
    query: str,
    gh: GitHubClient,
    cfg: EvolveConfig,
    *,
    per_page: int,
) -> tuple[str, list[Repo]]:
    """对一个 query 做"原查询 + 0 命中 relax 重试"。返回 (实际命中的 query, repos)。

    raw_query 模式下用(用户已经给了完整 GitHub query);R2 主流程不走这里。
    """
    for variant in _relax_query(query, cfg.relax.drop_order):
        repos = gh.search_repositories(variant, per_page=per_page)
        if repos:
            return variant, repos
    return query, []


def _dispatch_plan(
    plan: SearchPlan,
    gh: GitHubClient,
    *,
    per_page: int,
    language: str = "Python",   # R5: 由 client 层加而非 LLM query
    max_workers: int = DEFAULT_SEARCH_MAX_WORKERS,
) -> tuple[str, list[Repo], set[str]]:
    """R2 核心:把一个 SearchPlan 拆成多个 GitHub 调用并发出去,合并去重。

    Args:
        plan:        SearchPlan,可能有 text_hint 和/或 topics。两个都空就直接返回。
        gh:          GitHub 客户端。
        per_page:    单次调用拉多少条(GitHub 上限 100)。
        language:    R5:统一在 client 层加 language: 限定符,不让 LLM 写。
                     给 "" 表示完全不限语言。

    Returns:
        (summary_query, merged_repos):
            summary_query 用 ' || ' 拼接的可读字符串,只为日志展示
            merged_repos  按 full_name 去重后的 Repo 列表(顺序: text 命中优先)
    """
    queries = plan.queries()
    if not queries:
        return "", [], set()

    lang_suffix = f" language:{language}" if language else ""
    accumulated: dict[str, Repo] = {}
    precise: set[str] = set()       # 由"精确通道"(短语/库名,非 topic:)命中的 full_name(lower)
    actually_run: list[str] = []
    failures: list[BaseException] = []

    def run_query(q: str) -> list[Repo]:
        # GitHubClient 只读取不可变连接配置，每个请求自行创建 Request，适合有界线程并发。
        return gh.search_repositories(q + lang_suffix, per_page=per_page)

    # 请求并发执行，但严格按 SearchPlan 的原查询顺序合并结果。这样 phrases 优先、
    # setdefault 去重、summary 和“精确通道保护”的行为与旧串行实现完全一致。
    worker_count = min(len(queries), max(1, int(max_workers)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_query, q) for q in queries]
        for q, future in zip(queries, futures):
            full_q = q + lang_suffix
            try:
                repos = future.result()
            except Exception as exc:  # noqa: BLE001
                # 某一条挂了不阻塞其他通道；但所有通道都失败时必须向上抛，
                # 不能把网络/API 故障伪装成“搜索 0 条结果”。
                failures.append(exc)
                continue
            actually_run.append(full_q)
            is_precise = not q.lower().startswith("topic:")  # 短语/text_hint = 精确通道
            for r in repos:
                accumulated.setdefault(r.full_name, r)
                if is_precise:
                    precise.add(r.full_name.lower())

    if not actually_run and failures:
        raise failures[-1]
    summary = " || ".join(actually_run) if actually_run else (queries[0] + lang_suffix)
    return summary, list(accumulated.values()), precise


def find_candidates(
    capability: str,
    llm: BaseLLM | None = None,
    *,
    github: GitHubClient | None = None,
    top_k: int = 50,                 # B-2.5: 默认 50,留给下游语义粗筛挑
    per_page: int = 50,              # B-2.5: 默认 50,搭配 best-match 排序
    raw_query: bool = False,
    config: EvolveConfig | None = None,
    # 相关性评估 + 查询重写循环参数
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    eval_top_n: int = DEFAULT_TOP_N_FOR_EVAL,
    min_relevant: int = DEFAULT_MIN_RELEVANT,
    eval_disabled: bool = False,
) -> tuple[str, list[ScoredRepo], list[SearchRound]]:
    """主入口。返回 (最终 query, top-K 候选, 多轮搜索轨迹)。

    多轮搜索之间**会合并去重**(B-2.5):前几轮的结果不再被后几轮覆盖丢弃。
    这样即便相关性评估判 INSUFFICIENT,前轮里偶尔出现的好仓库也保留。

    Args:
        capability: 自然语言能力描述。
        llm: LLM 客户端;raw_query=True 或 eval_disabled=True 时可为 None。
        github: GitHub 客户端;默认现场 new 一个(读 GITHUB_TOKEN)。
        top_k: 最终返回多少条,默认 **50**(B-2.5,留给下游语义粗筛)。
        per_page: 单轮向 GitHub 拉多少条,默认 **50**。
        raw_query: True 表示 capability 已是 GitHub query,跳过首轮 LLM 转换。
        config: 可调参数;None 时读 configs/tool-evolution.yaml。
        max_rounds: 评估循环总轮数上限(首轮 + 最多 max_rounds-1 次重写)。
        eval_top_n: 每轮交给 LLM 评估的前 N 条(再多 prompt 会胀)。
        min_relevant: ≥ 多少条相关算 SUFFICIENT。
        eval_disabled: True 时跳过 LLM 相关性评估(回退到旧行为,仅做单轮搜索)。

    Returns:
        used_query   最后一次实际命中(非 0)的 query
        ranked       合并去重后,重排取 top-K(可能 INSUFFICIENT,由调用方处理)
        rounds       每一轮的 SearchRound 留痕,可写日志
    """
    cfg = config or load_evolve_config()
    gh = github or GitHubClient()
    rounds: list[SearchRound] = []
    accumulated: dict[str, Repo] = {}    # 跨轮去重池: full_name → Repo
    precise_names: set[str] = set()      # 精确通道(短语/库名)命中的 full_name(lower),保护不被 star 截断

    # ---- raw_query 模式:保留旧的"单 query + relax 兜底",绕过 planner
    if raw_query:
        q_str = capability.strip()
        if not q_str:
            raise ValueError("raw_query=True 时 capability 不能为空")
        used_query, repos = _search_one_query(q_str, gh, cfg, per_page=per_page)
        for r in repos:
            accumulated.setdefault(r.full_name, r)
        rounds.append(SearchRound(round_idx=1, query=used_query,
                                  raw_repo_count=len(repos), verdict=None))
        merged = list(accumulated.values())
        # raw query 下需求关键词 = query 里的普通词(去掉 language: 之类限定符)
        raw_kw = frozenset(t.lower() for t in re.split(r"\s+", q_str)
                           if len(t) >= 3 and ":" not in t)
        return used_query, _rerank(merged, cfg.score, raw_kw)[:top_k], rounds

    if llm is None:
        raise ValueError("raw_query=False 时必须传 llm")

    # ---- R2 主流程:双通道 planner + 多轮重写
    do_eval = not eval_disabled

    # 历史轨迹(给 R4 reformulate_plan 当负样本上下文)
    prior_plans: list[SearchPlan] = []
    prior_verdicts: list[RelevanceVerdict] = []

    plan = plan_search(capability, llm)
    used_query = ""
    need_kw: set[str] = set()            # 跨轮累积的需求关键词,供粗排算相关性

    for i in range(1, max_rounds + 1):
        need_kw |= _need_keywords(plan)
        round_summary, repos, precise = _dispatch_plan(plan, gh, per_page=per_page)
        if repos:
            used_query = round_summary
        for r in repos:
            accumulated.setdefault(r.full_name, r)
        precise_names |= precise

        if not do_eval:
            rounds.append(SearchRound(round_idx=i, query=round_summary,
                                      raw_repo_count=len(repos),
                                      verdict=None, plan=plan))
            break

        verdict = check_relevance(
            capability=capability,
            candidates=repos,
            llm=llm,
            top_n=eval_top_n,
            min_relevant=min_relevant,
        )
        rounds.append(SearchRound(round_idx=i, query=round_summary,
                                  raw_repo_count=len(repos),
                                  verdict=verdict, plan=plan))
        prior_plans.append(plan)
        prior_verdicts.append(verdict)

        if verdict.sufficient:
            break
        if i == max_rounds:
            break

        # R4: 用 reformulate_plan(带负样本反馈,温和不强制避开)替代旧 reformulate_query
        new_plan = reformulate_plan(
            capability=capability,
            prior_plans=prior_plans,
            prior_verdicts=prior_verdicts,
            llm=llm,
        )
        if new_plan.is_empty:
            break
        plan = new_plan

    merged = list(accumulated.values())
    ranked_all = _rerank(merged, cfg.score, frozenset(need_kw))
    # 精确通道命中的库**优先保留**(它们靠名字/描述字面命中 = 高相关),
    # 不让 star 重排把 0 星专用库挤出 top_k。其余按 star 分补足。
    precise_ranked = [r for r in ranked_all if r.repo.full_name.lower() in precise_names]
    rest = [r for r in ranked_all if r.repo.full_name.lower() not in precise_names]
    ranked = (precise_ranked + rest)[:top_k]
    return used_query, ranked, rounds
