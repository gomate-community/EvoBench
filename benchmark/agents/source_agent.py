from __future__ import annotations

from collections import defaultdict

from benchmark.adapters.retriever import MockRetriever
from benchmark.agents.base import AgentBase
from benchmark.schemas import SourceDocument


class SourceAgent(AgentBase):
    """Collect, normalize and rank source documents for sample generation.

    Production retrievers can return news, paper, financial-report, policy, patent,
    repository, or forum documents. This agent keeps the rest of the pipeline stable by
    de-duplicating, trust-ranking and annotating source metadata.
    """

    name = "source_agent"

    def __init__(self, retriever: MockRetriever, min_trust_level: int = 1):
        self.retriever = retriever
        self.min_trust_level = min_trust_level

    async def collect(self, topic: str, limit: int = 10) -> list[SourceDocument]:
        docs = await self.retriever.search(topic, limit=limit) # 调检索器
        return self.rank_sources(self.dedupe_sources(self.normalize_sources(docs)), query=topic)[:limit] # 排序、去重、归一化

    async def collect_multi_topic(self, topics: list[str], limit_per_topic: int = 5) -> list[SourceDocument]:
        docs: list[SourceDocument] = []
        for topic in topics:
            docs.extend(await self.collect(topic, limit=limit_per_topic))
        return self.rank_sources(self.dedupe_sources(docs))

    def normalize_sources(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        normalized: list[SourceDocument] = []
        for doc in docs:
            content = self.normalize_text(doc.content)
            title = self.normalize_text(doc.title)
            if not content or doc.trust_level < self.min_trust_level:
                continue
            doc.content = content
            doc.title = title
            doc.metadata.setdefault("content_chars", len(content))
            doc.metadata.setdefault("has_url", bool(doc.url))
            doc.metadata.setdefault("source_rank_reason", "ranked by trust, recency and content length")
            normalized.append(doc)
        return normalized

    def dedupe_sources(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        buckets: dict[str, SourceDocument] = {}
        for doc in docs:
            key = doc.url or f"{doc.publisher}:{doc.title}"
            if key not in buckets:
                buckets[key] = doc
                continue
            incumbent = buckets[key]
            if (doc.trust_level, len(doc.content)) > (incumbent.trust_level, len(incumbent.content)):
                buckets[key] = doc
        return list(buckets.values())

    def rank_sources(self, docs: list[SourceDocument], query: str | None = None) -> list[SourceDocument]:
        """排序：标题命中 query > trust_level > recency > content 长度。

        当提供 query 时，标题与 query 相互包含的 doc 优先级最高，避免
        百度子词扩展产生的泛词长文（如「人民」「保护」）把精确命中条目挤出 top-K。
        """
        q = (query or "").strip()

        def score(doc: SourceDocument) -> tuple[int, int, int, int]:
            title_hit = 0
            if q and doc.title:
                t = doc.title.strip()
                if q == t or q in t or t in q:
                    title_hit = 1
            recency = int(doc.published_at.timestamp()) if doc.published_at else 0
            return (title_hit, doc.trust_level, recency, min(len(doc.content), 5000))

        return sorted(docs, key=score, reverse=True)

    def group_by_publisher(self, docs: list[SourceDocument]) -> dict[str, list[SourceDocument]]:
        groups: dict[str, list[SourceDocument]] = defaultdict(list)
        for doc in docs:
            groups[doc.publisher or "unknown"].append(doc)
        return groups
