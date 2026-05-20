from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import ssl
import urllib.request
import urllib.parse
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from benchmark.schemas import SourceDocument

logger = logging.getLogger(__name__)


class RetrieverAdapter(ABC):
    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[SourceDocument]: ...


class MockRetriever(RetrieverAdapter):
    async def search(self, query: str, limit: int = 5) -> list[SourceDocument]:
        now = datetime.utcnow()
        topic = query or "AI"
        docs = [
            SourceDocument(
                source_id="src_demo_001",
                title=f"{topic} 资讯：某公司发布新一代 AI 芯片",
                url="https://example.com/demo-ai-chip",
                source_type="news",
                publisher="Example News",
                published_at=now,
                content=(
                    "某公司于今日发布新一代 AI 芯片，宣称推理性能提升 30%。"
                    "该公司表示，新芯片面向数据中心推理场景，并计划在下季度开始向部分客户供货。"
                    "报道未披露独立第三方基准测试结果。"
                ),
                trust_level=3,
            ),
            SourceDocument(
                source_id="src_demo_002",
                title=f"{topic} 公司技术博客：新芯片架构说明",
                url="https://example.com/blog-ai-chip",
                source_type="company_blog",
                publisher="Example Corp",
                published_at=now - timedelta(hours=2),
                content=(
                    "Example Corp 在技术博客中称，新一代 AI 芯片采用改进的内存带宽设计。"
                    "公司给出的内部测试显示，部分推理任务性能提升约 30%。"
                    "博客强调，实际性能会随模型结构和部署环境变化。"
                ),
                trust_level=4,
            ),
            SourceDocument(
                source_id="src_demo_003",
                title=f"分析：{topic} 芯片性能声明仍需第三方验证",
                url="https://example.com/analysis-ai-chip",
                source_type="analysis",
                publisher="Example Research",
                published_at=now - timedelta(days=1),
                content=(
                    "行业分析师认为，厂商关于 AI 芯片性能提升 30% 的说法需要第三方测试验证。"
                    "目前公开材料主要来自公司新闻稿和技术博客，尚未看到独立实验室报告。"
                    "若性能声明成立，该产品可能影响云端推理成本结构。"
                ),
                trust_level=4,
            ),
        ]
        return docs[:limit]


# ---------------------------------------------------------------------------
# 联网检索器（百度百科 + 百度搜索，国内直连可用）
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")


class BaiduCultureRetriever(RetrieverAdapter):
    """通过百度百科 OpenAPI + 百度搜索联网检索社科文化语料。

    特性：
    - 国内直连，无需翻墙；
    - 仅依赖 Python 标准库（urllib），无额外 pip 依赖；
    - 使用 asyncio.to_thread 包装同步 HTTP，兼容 async 接口；
    - 支持关键词相关性评分 & trust_level 排序（遵循项目规范）；
    - 支持超时、最大重试、内容字数截断等配置。
    """

    # 百度百科开放 API（免费，无需 appkey）
    _BAIKE_API = "https://baike.baidu.com/api/openapi/BaikeLemmaCardApi"
    # 百度搜索建议 API（免费）
    _SUGGESTION_API = "https://suggestion.baidu.com/su"

    def __init__(
        self,
        trust_level: int = 4,
        max_content_chars: int = 2000,
        timeout: float = 10.0,
        max_retries: int = 2,
    ):
        self.trust_level = trust_level
        self.max_content_chars = max_content_chars
        self.timeout = timeout
        self.max_retries = max_retries
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    # ----------------------- 公开接口 -----------------------

    async def search(self, query: str, limit: int = 5) -> list[SourceDocument]:
        """搜索并返回 SourceDocument 列表，按相关性 + trust_level 排序。"""
        if not query or not query.strip():
            return []

        # 1. 构建候选关键词列表：原词 > 子词 > 建议词
        candidates = self._build_candidates(query)
        suggestions = await self._get_suggestions(query, limit=limit)
        candidates.extend(suggestions)
        # 去重保序
        seen: set[str] = set()
        unique_candidates: list[str] = []
        for c in candidates:
            key = c.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_candidates.append(c.strip())

        # 2. 逐个关键词查百度百科
        docs: list[SourceDocument] = []
        seen_titles: set[str] = set()
        for kw in unique_candidates:
            if len(docs) >= limit:
                break
            doc = await self._fetch_baike(kw)
            if doc is not None and doc.title not in seen_titles:
                seen_titles.add(doc.title)
                docs.append(doc)

        # 3. 评分排序
        tokens = self._tokenize(query)
        scored = [(self._score(doc, tokens), doc) for doc in docs]
        scored.sort(key=lambda pair: (pair[0], pair[1].trust_level), reverse=True)

        for score, doc in scored:
            doc.metadata["match_score"] = score

        return [doc for _, doc in scored[:limit]]

    @staticmethod
    def _build_candidates(query: str) -> list[str]:
        """从 query 生成候选查询词：原词 + 中文子词（2-4字窗口滑动）。"""
        candidates = [query]
        # 提取纯中文部分
        zh_chars = re.findall(r"[\u4e00-\u9fff]+", query)
        full_zh = "".join(zh_chars)
        if len(full_zh) > 2:
            # 按 2-4 字窗口切分，优先长词
            for size in (4, 3, 2):
                for i in range(0, len(full_zh) - size + 1):
                    word = full_zh[i : i + size]
                    if word != query:
                        candidates.append(word)
        return candidates

    # ----------------------- 百度建议 API -----------------------

    async def _get_suggestions(self, query: str, limit: int = 8) -> list[str]:
        """调用百度搜索建议接口获取相关词列表。"""
        params = {"wd": query, "action": "opensearch", "ie": "utf-8"}
        url = f"{self._SUGGESTION_API}?{urllib.parse.urlencode(params)}"
        data = await asyncio.to_thread(self._sync_get_text, url)
        if not data:
            return [query]
        try:
            parsed = json.loads(data)
            # opensearch 格式: [query, [suggestions...]]
            if isinstance(parsed, list) and len(parsed) >= 2:
                suggestions = parsed[1]
                if isinstance(suggestions, list):
                    return [str(s) for s in suggestions[:limit] if s]
        except (json.JSONDecodeError, IndexError):
            pass
        return [query]

    # ----------------------- 百度百科 API -----------------------

    async def _fetch_baike(self, keyword: str) -> SourceDocument | None:
        """调用百度百科 OpenAPI 获取词条信息。"""
        params = {
            "scope": "103",
            "format": "json",
            "appid": "379020",
            "bk_key": keyword,
            "bk_length": str(self.max_content_chars),
        }
        url = f"{self._BAIKE_API}?{urllib.parse.urlencode(params)}"
        raw = await asyncio.to_thread(self._sync_get_text, url)
        if not raw:
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        # API 返回错误时没有 title 字段
        title = data.get("title")
        if not title:
            return None

        # 拼接摘要 + 正文
        abstract = data.get("abstract", "")
        # 百科卡片可能有额外信息
        card = data.get("card", [])
        card_text = ""
        if isinstance(card, list):
            card_lines = [f"{item.get('key', '')}: {item.get('value', [' '])[0]}" for item in card if item.get("key")]
            card_text = "；".join(card_lines[:8])

        content = abstract
        if card_text:
            content = f"{abstract}\n\n基本信息：{card_text}"

        if not content.strip():
            return None

        if len(content) > self.max_content_chars:
            content = content[: self.max_content_chars] + "…"

        url_field = data.get("url") or f"https://baike.baidu.com/item/{urllib.parse.quote(title)}"
        source_id = f"baike_{hashlib.md5(title.encode()).hexdigest()[:12]}"

        return SourceDocument(
            source_id=source_id,
            title=title,
            url=url_field,
            source_type="wiki",
            publisher="百度百科",
            fetched_at=datetime.utcnow(),
            content=content,
            trust_level=self.trust_level,
            language="zh-CN",
            metadata={
                "retriever": "BaiduCultureRetriever",
                "keyword": keyword,
                "match_score": 0,
            },
        )

    # ----------------------- 通用 HTTP 工具 -----------------------

    def _sync_get_text(self, url: str) -> str | None:
        """同步 HTTP GET，带重试和超时，返回响应文本。"""
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept": "application/json, text/plain, */*",
                })
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_ctx) as resp:
                    return resp.read().decode("utf-8")
            except Exception as exc:
                logger.warning("Baidu API error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt == self.max_retries:
                    return None
        return None

    # ----------------------- 评分工具 -----------------------

    @staticmethod
    def _tokenize(query: str) -> list[str]:
        if not query:
            return []
        raw = re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", query.strip())
        return [token.lower() for token in raw if token]

    @staticmethod
    def _score(doc: SourceDocument, tokens: list[str]) -> int:
        """标题命中 +5，内容每命中一次 +1（上限5），同分按 trust_level 兜底。"""
        title = doc.title.lower()
        content = doc.content.lower()
        score = 0
        for tok in tokens:
            if tok in title:
                score += 5
            if tok in content:
                score += min(content.count(tok), 5)
        return score


# ---------------------------------------------------------------------------
# Wikipedia 联网检索器（备用，需翻墙）
# ---------------------------------------------------------------------------


class WikipediaCultureRetriever(RetrieverAdapter):
    """通过 Wikipedia MediaWiki API 联网检索社科文化语料。

    特性：
    - 仅依赖 Python 标准库（urllib），无额外 pip 依赖；
    - 使用 asyncio.to_thread 包装同步 HTTP，兼容 async 接口；
    - 支持关键词相关性评分 & trust_level 排序（遵循项目规范）；
    - 支持超时、最大重试、内容字数截断等配置。
    - 注意：国内需翻墙才可访问。
    """

    def __init__(
        self,
        language: str = "zh",
        trust_level: int = 4,
        max_content_chars: int = 2000,
        timeout: float = 15.0,
        max_retries: int = 2,
    ):
        self.language = language
        self.trust_level = trust_level
        self.max_content_chars = max_content_chars
        self.timeout = timeout
        self.max_retries = max_retries
        self._api_base = f"https://{language}.wikipedia.org/w/api.php"
        # 跳过 SSL 验证（部分代理环境需要；生产环境可移除）
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    # ----------------------- 公开接口 -----------------------

    async def search(self, query: str, limit: int = 5) -> list[SourceDocument]:
        """搜索并返回 SourceDocument 列表，按相关性 + trust_level 排序。"""
        if not query or not query.strip():
            return []

        search_results = await self._search_titles(query, limit=limit * 2)
        if not search_results:
            return []

        docs = await self._fetch_pages(search_results, limit=limit)

        # 基于 query 对结果打分排序
        tokens = self._tokenize(query)
        scored = [(self._score(doc, tokens), doc) for doc in docs]
        scored.sort(key=lambda pair: (pair[0], pair[1].trust_level), reverse=True)

        for score, doc in scored:
            doc.metadata["match_score"] = score

        return [doc for _, doc in scored[:limit]]

    # ----------------------- Wikipedia API 调用 -----------------------

    async def _search_titles(self, query: str, limit: int = 10) -> list[dict]:
        """调用 MediaWiki action=query&list=search 获取搜索结果列表。"""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": "0",
            "srlimit": str(limit),
            "format": "json",
            "utf8": "1",
        }
        data = await self._api_get(params)
        if data is None:
            return []
        return data.get("query", {}).get("search", [])

    async def _fetch_pages(self, search_results: list[dict], limit: int) -> list[SourceDocument]:
        """逐一获取每个搜索结果的页面摘要内容。"""
        docs: list[SourceDocument] = []
        for item in search_results:
            if len(docs) >= limit:
                break
            page_id = item.get("pageid")
            title = item.get("title", "")
            snippet = _TAG_RE.sub("", item.get("snippet", ""))

            doc = await self._fetch_page_content(page_id, title, snippet)
            if doc is not None:
                docs.append(doc)
        return docs

    async def _fetch_page_content(
        self,
        page_id: int,
        title: str,
        snippet: str,
    ) -> SourceDocument | None:
        """用 action=query&prop=extracts 拉取纯文本摘要。"""
        params = {
            "action": "query",
            "pageids": str(page_id),
            "prop": "extracts|info",
            "exintro": "true",
            "explaintext": "true",
            "exsectionformat": "plain",
            "inprop": "url",
            "format": "json",
            "utf8": "1",
        }
        data = await self._api_get(params)
        if data is None:
            return None

        pages = data.get("query", {}).get("pages", {})
        page_data = pages.get(str(page_id))
        if not page_data or page_data.get("missing") is not None:
            return None

        content = (page_data.get("extract") or snippet or "").strip()
        if not content:
            return None

        # 截断过长文本
        if len(content) > self.max_content_chars:
            content = content[: self.max_content_chars] + "…"

        full_url = page_data.get("fullurl") or f"https://{self.language}.wikipedia.org/wiki/{urllib.parse.quote(title)}"
        source_id = f"wiki_{page_id}_{hashlib.md5(title.encode()).hexdigest()[:8]}"
        lang_code = f"{self.language}-CN" if self.language == "zh" else self.language

        return SourceDocument(
            source_id=source_id,
            title=title,
            url=full_url,
            source_type="wiki",
            publisher="Wikipedia",
            fetched_at=datetime.utcnow(),
            content=content,
            trust_level=self.trust_level,
            language=lang_code,
            metadata={
                "retriever": "WikipediaCultureRetriever",
                "page_id": page_id,
                "snippet": snippet,
                "match_score": 0,
            },
        )

    # ----------------------- 通用 HTTP 工具 -----------------------

    async def _api_get(self, params: dict) -> dict | None:
        """带重试的 API GET（用 asyncio.to_thread 包装同步 urllib）。"""
        url = f"{self._api_base}?{urllib.parse.urlencode(params)}"
        return await asyncio.to_thread(self._sync_get, url)

    def _sync_get(self, url: str) -> dict | None:
        """同步 HTTP GET，带重试和超时。"""
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "EvoBench/1.0"})
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_ctx) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("Wikipedia API error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt == self.max_retries:
                    return None
        return None

    # ----------------------- 评分工具 -----------------------

    @staticmethod
    def _tokenize(query: str) -> list[str]:
        if not query:
            return []
        raw = re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", query.strip())
        return [token.lower() for token in raw if token]

    @staticmethod
    def _score(doc: SourceDocument, tokens: list[str]) -> int:
        """对返回结果打分：标题命中 +5，内容每命中一次 +1（上限5），同分按 trust_level 兜底。"""
        title = doc.title.lower()
        content = doc.content.lower()
        score = 0
        for tok in tokens:
            if tok in title:
                score += 5
            if tok in content:
                score += min(content.count(tok), 5)
        return score
