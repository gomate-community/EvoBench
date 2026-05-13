from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from benchmark.adapters.llm import LLMAdapter, build_llm_adapter
from benchmark.core.config import settings


@dataclass(frozen=True)
class AgentRunConfig:
    language: str = "zh-CN"
    domain: str = "technology"
    min_claim_chars: int = 12
    max_question_chars: int = 500
    current_time: datetime | None = None
    enable_llm: bool = True
    llm_provider: str | None = None
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None

    @property
    def now(self) -> datetime:
        return self.current_time or datetime.now(timezone.utc).replace(tzinfo=None)


class AgentBase:
    name = "agent"

    def make_id(self, prefix: str, *parts: str, randomize: bool = False) -> str:
        if randomize:
            return f"{prefix}_{uuid4().hex[:10]}"
        raw = "||".join([self.name, *[p for p in parts if p]])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{digest}"

    def split_sentences(self, text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return []
        parts = re.split(r"(?<=[。！？!?；;\.])\s*", cleaned)
        return [p.strip(" \n\t。") for p in parts if len(p.strip()) >= 4]

    def normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def parse_json_object(self, raw: str) -> dict[str, Any]:
        """Best-effort parser for LLM JSON outputs."""
        raw = raw.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    def lexical_overlap(self, a: str, b: str) -> float:
        ta = set(re.findall(r"[\w\u4e00-\u9fff]+", (a or "").lower()))
        tb = set(re.findall(r"[\w\u4e00-\u9fff]+", (b or "").lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def llm_enabled(self) -> bool:
        config = getattr(self, "config", None)
        enabled = getattr(config, "enable_llm", True) if config is not None else True
        return bool(settings.llm_enabled and enabled)

    def get_llm(self) -> LLMAdapter:
        adapter = getattr(self, "_llm", None)
        if adapter is None:
            config = getattr(self, "config", None)
            self._llm = build_llm_adapter(
                provider=getattr(config, "llm_provider", None) if config is not None else None,
                temperature=getattr(config, "llm_temperature", None) if config is not None else None,
                max_tokens=getattr(config, "llm_max_tokens", None) if config is not None else None,
            )
        return self._llm

    async def llm_complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        fallback: str | Callable[[], str] | None = None,
    ) -> str:
        if not self.llm_enabled():
            return fallback() if callable(fallback) else (fallback or "")
        try:
            return await self.get_llm().complete(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            if callable(fallback):
                return fallback()
            if fallback is not None:
                return fallback
            raise

    async def llm_complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        fallback: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not self.llm_enabled():
            return fallback() if callable(fallback) else (fallback or {})
        try:
            return await self.get_llm().complete_json(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            if callable(fallback):
                return fallback()
            if fallback is not None:
                return fallback
            raise
