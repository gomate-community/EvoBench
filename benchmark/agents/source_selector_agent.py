from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from benchmark.agents.base import AgentBase
from benchmark.schemas import SourceDocument


@dataclass(frozen=True)
class SourceSelectionPolicy:
    min_trust_level: int = 1
    source_types: tuple[str, ...] = ()
    language: str | None = None
    max_age_days: int | None = None
    min_content_chars: int = 40
    require_url: bool = False
    dedupe_threshold: float = 0.92


class SourceSelectorAgent(AgentBase):
    """Shared source filtering and ranking layer for all generation skills.

    All task-specific skills receive already-normalized documents from this layer, so
    source governance is shared across d->x, d->(x,y), d->(x,T,y), and error-based
    augmentation tasks.
    """

    name = "source_selector_agent"

    def __init__(self, policy: SourceSelectionPolicy | None = None):
        self.policy = policy or SourceSelectionPolicy()

    def select(self, docs: Iterable[SourceDocument], limit: int = 20) -> list[SourceDocument]:
        filtered = [self._normalize(d) for d in docs if self._passes_policy(d)]
        return self._rank(self._dedupe(filtered))[:limit]

    def _passes_policy(self, doc: SourceDocument) -> bool:
        if doc.trust_level < self.policy.min_trust_level:
            return False
        if self.policy.source_types and doc.source_type not in self.policy.source_types:
            return False
        if self.policy.language and doc.language != self.policy.language:
            return False
        if self.policy.require_url and not doc.url:
            return False
        if len(doc.content or "") < self.policy.min_content_chars:
            return False
        if self.policy.max_age_days and doc.published_at:
            if doc.published_at < datetime.utcnow() - timedelta(days=self.policy.max_age_days):
                return False
        return True

    def _normalize(self, doc: SourceDocument) -> SourceDocument:
        doc.title = self.normalize_text(doc.title)
        doc.content = self.normalize_text(doc.content)
        doc.metadata.setdefault("selected_by", self.name)
        doc.metadata.setdefault("content_chars", len(doc.content))
        return doc

    def _dedupe(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        kept: list[SourceDocument] = []
        for doc in docs:
            if any(self.lexical_overlap(doc.content[:500], old.content[:500]) > self.policy.dedupe_threshold for old in kept):
                continue
            kept.append(doc)
        return kept

    def _rank(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        def score(doc: SourceDocument) -> tuple[int, int, int]:
            recency = int(doc.published_at.timestamp()) if doc.published_at else 0
            return doc.trust_level, recency, min(len(doc.content), 6000)

        return sorted(docs, key=score, reverse=True)
