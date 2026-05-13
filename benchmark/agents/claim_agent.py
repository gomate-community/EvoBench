from __future__ import annotations

import re
from datetime import datetime

from benchmark.adapters.llm import LLMAdapter
from benchmark.agents.base import AgentBase, AgentRunConfig
from benchmark.schemas import Claim, EvidenceSpan, SourceDocument


class ClaimAgent(AgentBase):
    """Extract atomic, evidence-linked claims from source documents.

    This implementation is intentionally deterministic so the framework can run without
    external LLMs. In production, you can add an LLM extraction pass and keep this agent
    as a validator/fallback.
    """

    name = "claim_agent"

    def __init__(self, config: AgentRunConfig | None = None, llm: LLMAdapter | None = None):
        self.config = config or AgentRunConfig()
        self._llm = llm

    async def extract_claims(
        self,
        docs: list[SourceDocument],
        max_claims_per_doc: int = 5,
    ) -> list[Claim]:
        claims: list[Claim] = []
        for doc in docs:
            llm_claims = await self._extract_claims_with_llm(doc, max_claims_per_doc)
            if llm_claims:
                claims.extend(llm_claims)
                continue
            claims.extend(self._extract_claims_deterministic(doc, max_claims_per_doc))
        return self._dedupe_claims(claims)

    async def _extract_claims_with_llm(self, doc: SourceDocument, max_claims_per_doc: int) -> list[Claim]:
        payload = await self.llm_complete_json(
            self._claim_extraction_prompt(doc, max_claims_per_doc),
            system=(
                "你是一个面向 benchmark 构建的事实抽取助手。"
                "请仅根据给定文档抽取原子化 claim，输出严格 JSON："
                '{"claims":[{"text":"...","evidence_text":"...","subject":"...","predicate":"...","object":"..."}]}'
            ),
            temperature=0.0,
            max_tokens=1400,
            fallback={"claims": []},
        )
        raw_claims = payload.get("claims", [])
        if not isinstance(raw_claims, list):
            return []

        claims: list[Claim] = []
        for item in raw_claims[:max_claims_per_doc]:
            if not isinstance(item, dict):
                continue
            claim_text = self.normalize_text(str(item.get("text", "")))
            if len(claim_text) < self.config.min_claim_chars:
                continue
            evidence_text = self.normalize_text(str(item.get("evidence_text") or claim_text))
            evidence = EvidenceSpan(
                evidence_id=self.make_id("ev", doc.source_id, evidence_text),
                source_id=doc.source_id,
                text=evidence_text,
                support="supports",
                confidence=min(0.95, 0.5 + 0.1 * doc.trust_level),
            )
            claims.append(
                Claim(
                    claim_id=self.make_id("claim", doc.source_id, claim_text),
                    source_id=doc.source_id,
                    text=claim_text,
                    subject=self._nullable_text(item.get("subject")) or self._extract_subject(claim_text),
                    predicate=self._nullable_text(item.get("predicate")) or self._extract_predicate(claim_text),
                    object=self._nullable_text(item.get("object")) or self._extract_object(claim_text),
                    event_time=self._infer_event_time(claim_text, doc),
                    confidence=evidence.confidence,
                    evidence_spans=[evidence],
                    source_trust_level=doc.trust_level,
                    metadata={
                        "source_title": doc.title,
                        "publisher": doc.publisher,
                        "source_type": doc.source_type,
                        "claim_generator": "llm",
                    },
                )
            )
        return claims

    def _extract_claims_deterministic(self, doc: SourceDocument, max_claims_per_doc: int) -> list[Claim]:
        claims: list[Claim] = []
        for sentence in self._candidate_sentences(doc)[:max_claims_per_doc]:
            claim_text = self.normalize_text(sentence)
            evidence = EvidenceSpan(
                evidence_id=self.make_id("ev", doc.source_id, claim_text),
                source_id=doc.source_id,
                text=claim_text,
                support="supports",
                confidence=min(0.95, 0.45 + 0.1 * doc.trust_level),
            )
            claims.append(
                Claim(
                    claim_id=self.make_id("claim", doc.source_id, claim_text),
                    source_id=doc.source_id,
                    text=claim_text,
                    subject=self._extract_subject(claim_text),
                    predicate=self._extract_predicate(claim_text),
                    object=self._extract_object(claim_text),
                    event_time=self._infer_event_time(claim_text, doc),
                    confidence=evidence.confidence,
                    evidence_spans=[evidence],
                    source_trust_level=doc.trust_level,
                    metadata={
                        "source_title": doc.title,
                        "publisher": doc.publisher,
                        "source_type": doc.source_type,
                        "claim_generator": "rule",
                    },
                )
            )
        return claims

    def _claim_extraction_prompt(self, doc: SourceDocument, max_claims_per_doc: int) -> str:
        content = doc.content[:4000]
        return (
            f"文档标题：{doc.title}\n"
            f"来源类型：{doc.source_type}\n"
            f"发布时间：{doc.published_at.isoformat() if doc.published_at else 'unknown'}\n"
            f"最多抽取：{max_claims_per_doc} 条\n\n"
            "请抽取可验证、粒度尽量原子的事实性 claim。"
            "每条 claim 都要尽量配一段能直接支持它的 evidence_text。\n\n"
            f"文档内容：\n{content}"
        )

    def _nullable_text(self, value: object) -> str | None:
        text = self.normalize_text(str(value)) if value is not None else ""
        return text or None

    def _candidate_sentences(self, doc: SourceDocument) -> list[str]:
        sentences = self.split_sentences(doc.content)
        usable: list[str] = []
        for sentence in sentences:
            if len(sentence) < self.config.min_claim_chars:
                continue
            if self._looks_like_boilerplate(sentence):
                continue
            usable.append(sentence)
        return usable

    def _looks_like_boilerplate(self, sentence: str) -> bool:
        bad_patterns = ["点击", "订阅", "免责声明", "版权", "广告", "更多内容", "cookie"]
        return any(p in sentence.lower() for p in bad_patterns)

    def _extract_subject(self, text: str) -> str | None:
        # Heuristic: before common verbs or punctuation.
        match = re.match(r"(.{2,40}?)(?:发布|宣布|表示|计划|完成|推出|获得|增长|下降|称)", text)
        if match:
            return match.group(1).strip(" ，,：:")
        return text[:30] if text else None

    def _extract_predicate(self, text: str) -> str | None:
        for verb in ["发布", "宣布", "表示", "计划", "完成", "推出", "获得", "增长", "下降", "称", "提升"]:
            if verb in text:
                return verb
        return None

    def _extract_object(self, text: str) -> str | None:
        pred = self._extract_predicate(text)
        if not pred or pred not in text:
            return None
        return text.split(pred, 1)[-1].strip(" ，,。")[:80]

    def _infer_event_time(self, text: str, doc: SourceDocument) -> datetime | None:
        # For current-news benchmark construction, document publish time is a useful
        # conservative fallback when the sentence uses relative dates such as 今日/昨日.
        if any(token in text for token in ["今日", "今天", "昨日", "本周", "近日"]):
            return doc.published_at
        return doc.published_at

    def _dedupe_claims(self, claims: list[Claim]) -> list[Claim]:
        kept: list[Claim] = []
        for claim in claims:
            if any(self.lexical_overlap(claim.text, old.text) > 0.92 for old in kept):
                continue
            kept.append(claim)
        return kept
