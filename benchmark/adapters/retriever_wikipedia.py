"""Wikipedia MediaWiki API 联网检索器（备用，需翻墙）。

作为 ``benchmark/adapters/retriever.py`` 的补充实现，独立成单文件以保持 main 原文件零修改。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import ssl
import urllib.request
import urllib.parse
from datetime import datetime

from benchmark.adapters.retriever import RetrieverAdapter
from benchmark.schemas import SourceDocument

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")


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
