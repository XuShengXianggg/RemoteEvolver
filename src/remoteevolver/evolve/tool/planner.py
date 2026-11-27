"""
remoteevolver.evolve.tool.planner —— 双通道搜索计划生成(R1 + R2)

参考 ToolRosella 的 planner.py 设计:不让 LLM 写"一条复杂 query 字符串",
而是把搜索意图拆成两个正交通道:

  text_hint:  ★ 单个英文词 ★  (库名 / 主流技术词,如 "pyjwt", "openpyxl")
              用来打到 GitHub 全文搜:  /search?q=pyjwt
  topics:     ★ 主题列表 ★    (短语,可带连字符,如 ["jwt", "json-web-token"])
              用来打 topic 限定符搜:    /search?q=topic:jwt

为什么这样设计:
  - **LLM 加更多词 ≠ 召回更高**(GitHub 是字面 AND 匹配)。强制每通道极简。
  - text_hint 只允许 1 个词,逼 LLM 不要堆砌(配合 LLM 调用时 max_tokens=24)
  - topics 是列表,各自跑一次搜索 → 多个角度并联,合并去重
  - **text_hint 优先正则抽**:用户已经在任务里写了"用 bcrypt 哈希"时,直接抽
    "bcrypt",根本不用调 LLM。失败再走 LLM。

输出结构:
  SearchPlan(text_hint, topics, sources)
    text_hint  : str        — 可能是空串(代表跳过全文搜),也可能是单个词
    topics     : list[str]  — 最多 ~4 个,不带 topic: 前缀
    sources    : dict       — 留痕:{'text_hint': 'regex' | 'llm' | 'none',
                                     'topics': 'llm' | 'none'}

不在这里做的事:
  - 不调 GitHub(那是 search.py 的事)
  - 不评分(那是 repo_assess.py 的事)
  - 不写 query qualifier(如 language:python 由 client 层加,见 R5)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...llm.base import BaseLLM, Message


# 单 LLM 调用的硬上限。topic 列表稍长,text_hint 一定要短。
DEFAULT_MAX_TOPICS = 4
# 召回:让 LLM 出"对象 × 操作"两组词,搜笛卡尔积(每个"对象 操作"2 词组合各搜一次)。
# 专用库常以“对象+操作”命名（如 raster_temporal_mean / spectral_index_batch），笛卡尔积能确定性
# 覆盖同义词组合(tif/geotiff/raster × average/mean),比让 LLM 自由编短语稳得多。
DEFAULT_MAX_SUBJECTS = 4
DEFAULT_MAX_OPERATIONS = 3
DEFAULT_MAX_PHRASES = 10           # 笛卡尔积后最多保留多少条 2 词查询(限 GitHub 调用数)
DEFAULT_TEXT_HINT_MAX_TOKENS = 24  # 略宽于 ToolRosella 的 16,容忍中文模型 verbose
DEFAULT_TOPICS_MAX_TOKENS = 128


@dataclass
class SearchPlan:
    """一轮搜索的多通道计划。

    三个正交通道,召回互补:
      phrases   : ★ 2-4 词短语 ★  (库名/描述字面词,如 "tif average geotiff")
                  打 GitHub 全文搜,专治"专用低星、无 topic"的库 —— 它们靠
                  名字/描述命中,而不是 topic 或单词。实测这是召回这类库的关键。
      text_hint : ★ 单个词 ★      (库名/主流技术词,如 "pyjwt")
      topics    : ★ 主题列表 ★    (如 ["jwt"]) —— 只对设了 topic 的库有效。
    """
    text_hint: str = ""               # 单词;空串表示这一通道跳过
    topics: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)  # 2-4 词字面短语,全文搜
    sources: dict[str, str] = field(default_factory=dict)  # 来源留痕

    @property
    def is_empty(self) -> bool:
        return not self.text_hint and not self.topics and not self.phrases

    def queries(self) -> list[str]:
        """这个 plan 要打几次 GitHub /search 调用,返回 query 字符串列表。

        顺序 = 短语(最贴合,优先)→ 单词 text_hint → topic 通道。
        合并去重时靠前的命中优先保留(见 search._dispatch_plan)。
        """
        out: list[str] = []
        seen: set[str] = set()

        def _add(q: str) -> None:
            q = q.strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)

        for p in self.phrases:
            _add(p)
        if self.text_hint:
            _add(self.text_hint)
        for t in self.topics:
            # 去掉空白 + 用连字符替换空格(GitHub topic 字面要求)
            slug = t.strip().lower().replace(" ", "-")
            if slug:
                _add(f"topic:{slug}")
        return out


# ============================================================================
# 正则前置:从用户任务文本直接抽 text_hint
# ============================================================================

# 形如 "用 bcrypt"、"调用 PyJWT"、"使用 openpyxl 库"、"with reportlab"、"via fpdf2"
# 严格限定:单一英文 token(允许小写/数字/下划线/-),长度 3-30
_LIB_HINT_PATTERNS = [
    # 中文用 X 库/模块/包
    r"用\s*([A-Za-z][A-Za-z0-9_\-]{2,29})\b\s*(库|模块|包|框架)?",
    r"使用\s*([A-Za-z][A-Za-z0-9_\-]{2,29})\b",
    r"调用\s*([A-Za-z][A-Za-z0-9_\-]{2,29})\b",
    r"通过\s*([A-Za-z][A-Za-z0-9_\-]{2,29})\b",
    # 英文 with X / via X / using X
    r"\b(?:with|via|using)\s+([A-Za-z][A-Za-z0-9_\-]{2,29})\b",
    # import X / from X
    r"\bimport\s+([A-Za-z][A-Za-z0-9_\-]{2,29})\b",
    r"\bfrom\s+([A-Za-z][A-Za-z0-9_\-]{2,29})\b",
    # 用引号包起的库名:"用 'fpdf2' 生成"
    r"['\"`]([A-Za-z][A-Za-z0-9_\-]{2,29})['\"`]",
]

# 黑名单:正则容易误抽的常见英文单词(语法/动词/介词等)
_HINT_BLACKLIST = {
    "the", "and", "with", "via", "use", "using", "for", "from", "import",
    "python", "library", "package", "module", "tool", "function",
    "create", "generate", "build", "make", "write", "read", "open",
    "true", "false", "none", "null",
    # 通用而非 PyPI 名
    "api", "http", "html", "json", "yaml", "xml", "url", "csv",
    # 任务关键词非库名
    "report", "table", "file", "stream",
}


def extract_text_hint_regex(capability: str) -> str:
    """从用户任务里用正则抽 text_hint。抽不到返回空串。

    返回 lowercase 后的第一个匹配,且不在黑名单里、长度 >= 3。
    """
    if not capability:
        return ""
    text = capability.strip()
    for pat in _LIB_HINT_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            cand = m.group(1).lower()
            if len(cand) < 3:
                continue
            if cand in _HINT_BLACKLIST:
                continue
            return cand
    return ""


# ============================================================================
# LLM:一次生成 text_hint + subjects/operations
# ============================================================================

_TERMS_PROMPT = """\
你在为 GitHub 全文搜索准备关键词。GitHub 是**字面 AND 匹配**,且专用库常以"对象+操作"
命名（如 raster_temporal_mean、spectral_index_batch）。所以请给两组词,我们会对**每个"对象 × 操作"
组合各搜一次**,用同义词覆盖来提高命中率。

同时给出一个适合单词全文搜索的 TEXT_HINT。三者必须在同一次回答中生成。

输出严格三行:
TEXT_HINT: 单个英文小写词，优先包名或具体技术词；没有合适词时填 NONE
SUBJECTS: 被处理的数据/对象名词,含常见同义词
OPERATIONS: 核心操作词,含同义词

规则:
- TEXT_HINT 必须恰好是一个英文小写词，不得包含空格、引号或解释。
- 每个词 = 1 个英文小写单词;SUBJECTS ≤ {max_subjects} 个,OPERATIONS ≤ {max_operations} 个。
- 只要**具体领域词**;不要 filler(batch/multiple/all/single/the/data/file/pixel-wise 等),
  也不要限定符(language: / topic: 等)。
- 任务若不是"对象+操作"型,就把核心概念词都放进 SUBJECTS,OPERATIONS 可留空。

示例(跨领域,学**模式**别抄词):
- "批量计算多张 GeoTIFF 的逐像素时间平均"
  TEXT_HINT: raster
  SUBJECTS: tif, geotiff, raster
  OPERATIONS: average, mean
- "计算 MODIS 两波段的 NDSI 雪指数栅格"
  TEXT_HINT: ndsi
  SUBJECTS: ndsi, geotiff, raster
  OPERATIONS: index
- "用 bcrypt 之类算法做密码哈希与校验"
  TEXT_HINT: bcrypt
  SUBJECTS: password, bcrypt
  OPERATIONS: hashing, verify

【能力】
{capability}

只输出三行,逗号分隔,无解释:
TEXT_HINT: ...
SUBJECTS: ...
OPERATIONS: ...
"""


_TOPICS_PROMPT = """\
Given the user query, identify up to {max_topics} GitHub topics that can find a repository
able to solve it.

Rules:
- Return ONLY a comma-separated list of noun-like topics.
- Use hyphens for multi-word topics (e.g. "json-web-token", not "json web token").
- Each topic must be lowercase ASCII.
- Prefer concrete technical topics over generic ones.
- Do NOT add explanations, prefixes, quotes, or markdown.

Examples (USER → OUTPUT):
- "用 bcrypt 算密码哈希"       → bcrypt, password-hashing
- "生成 2 页 PDF 报表"          → pdf, reportlab, pdf-generation
- "查股票实时价格"              → stock-market, yfinance, finance-api

User query:
{capability}
"""


def plan_search(
    capability: str,
    llm: BaseLLM | None = None,
    *,
    max_topics: int = DEFAULT_MAX_TOPICS,
) -> SearchPlan:
    """产出 SearchPlan:正则优先识别 hint,单次 LLM 调用生成完整搜索词计划。

    LLM=None 时只用正则;失败就 SearchPlan(空通道两个),让上层兜底用旧 query path。
    """
    plan = SearchPlan()

    # text_hint:正则结果优先;LLM 仍只调用一次,同时返回 hint 和短语词组。
    hint = extract_text_hint_regex(capability)
    if hint:
        plan.text_hint = hint
        plan.sources["text_hint"] = "regex"

    # phrases: 让 LLM 出"对象 SUBJECTS × 操作 OPERATIONS"两组词,搜笛卡尔积。
    # 这是召回主力——专用库以"对象+操作"命名,笛卡尔积确定性覆盖同义词组合。
    if llm is not None:
        prompt = _TERMS_PROMPT.format(
            capability=capability.strip(),
            max_subjects=DEFAULT_MAX_SUBJECTS,
            max_operations=DEFAULT_MAX_OPERATIONS,
        )
        resp = llm.generate([Message(role="user", content=prompt)])
        llm_hint, subjects, operations = _parse_search_plan(resp.text or "")
        if not hint:
            if llm_hint:
                plan.text_hint = llm_hint
                plan.sources["text_hint"] = "llm"
            else:
                plan.sources["text_hint"] = "none"
        plan.phrases = _cross_product(subjects, operations, max_phrases=DEFAULT_MAX_PHRASES)
        plan.sources["phrases"] = "llm" if plan.phrases else "none"
    else:
        if not hint:
            plan.sources["text_hint"] = "none"
        plan.sources["phrases"] = "none"

    # topics 通道已停用:实际仓库很少设 topic 标签,召回贡献小,还白费一次 LLM 调用。
    # 关键词召回改由 phrases(字面短语)+ text_hint(单词)承担。
    # (max_topics 形参保留,供 reformulate 等旧调用兼容,不再实际生成 topics。)
    del max_topics
    plan.topics = []
    plan.sources["topics"] = "disabled"
    return plan


def _parse_terms(text: str) -> "tuple[list[str], list[str]]":
    """从 LLM 输出抽 SUBJECTS / OPERATIONS 两组单词(小写、去 filler)。"""
    subjects: list[str] = []
    operations: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        low = line.lower()
        if low.startswith("subjects"):
            subjects = _clean_words(line.split(":", 1)[-1], DEFAULT_MAX_SUBJECTS)
        elif low.startswith("operations"):
            operations = _clean_words(line.split(":", 1)[-1], DEFAULT_MAX_OPERATIONS)
    return subjects, operations


def _parse_search_plan(text: str) -> "tuple[str, list[str], list[str]]":
    """解析单次规划响应，同时得到 text hint 和对象/操作词组。"""
    hint = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("text_hint"):
            continue
        raw = line.split(":", 1)[-1].strip().strip("'\"`").lower()
        candidate = raw.split()[0] if raw else ""
        if (
            candidate
            and candidate != "none"
            and len(candidate) >= 3
            and candidate not in _HINT_BLACKLIST
            and re.fullmatch(r"[a-z][a-z0-9_-]*", candidate)
        ):
            hint = candidate
        break
    subjects, operations = _parse_terms(text)
    return hint, subjects, operations


# 词级黑名单:这些是 filler / 限定符,进了笛卡尔积只会污染查询
_TERM_BLACKLIST = {
    "batch", "multiple", "all", "single", "the", "a", "an", "data", "file", "files",
    "pixelwise", "pixel", "wise", "language", "topic", "python", "tool", "library",
    "and", "or", "of", "for", "with", "to", "from", "list",
}


def _clean_words(segment: str, limit: int) -> list[str]:
    """把逗号/空白分隔的一段词规整成合法 GitHub 单词(小写、去黑名单、去重、限量)。"""
    out: list[str] = []
    seen: set[str] = set()
    for tok in re.split(r"[,\s]+", segment.strip()):
        w = tok.strip().strip("'\"`").lower()
        if not re.match(r"^[a-z0-9][a-z0-9\-]*$", w):
            continue
        if len(w) < 2 or w in _TERM_BLACKLIST or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out


def _cross_product(
    subjects: list[str], operations: list[str], *, max_phrases: int,
) -> list[str]:
    """对象 × 操作 笛卡尔积 → 2 词查询列表。操作为空则退化为 subjects 单词查询。

    没有 operations(单概念任务)时,直接用 subjects 各自当一条查询(如 "ndsi")。
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        if q and q not in seen:
            seen.add(q)
            out.append(q)

    # 先放"对象 操作"组合(命中专用库的主力),再放对象单词兜底
    for s in subjects:
        for o in operations:
            _add(s if s == o else f"{s} {o}")
    for s in subjects:
        _add(s)
    return out[:max_phrases]


def _parse_topics(text: str, *, max_topics: int) -> list[str]:
    """把 LLM 输出按逗号拆成 topic 列表;过滤过长/带空格/包含 markdown 的烂条目。"""
    # 先扔掉可能的 Markdown / 解释行
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 找出"看起来像逗号分隔列表"的那一行(优先选**没有冒号、没句号**的)
    candidate_line = ""
    for ln in lines:
        if "," in ln and ":" not in ln and "。" not in ln:
            candidate_line = ln
            break
    if not candidate_line:
        # fallback:把所有行拼起来,按逗号切
        candidate_line = " ".join(lines)

    out: list[str] = []
    seen: set[str] = set()
    for raw_tok in candidate_line.split(","):
        tok = raw_tok.strip().lower()
        # 清掉常见噪声:序号 / 引号 / Markdown
        tok = re.sub(r"^[\d\.\-\*\s]+", "", tok)
        tok = tok.strip("'\"` ")
        if not tok:
            continue
        if len(tok) > 40:        # 一看就是 LLM 在写句子
            continue
        if any(ch in tok for ch in " \t"):  # 词内带空格 → 不合法 GitHub topic
            tok = tok.replace(" ", "-")
        if not re.match(r"^[a-z][a-z0-9\-]{1,39}$", tok):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_topics:
            break
    return out
