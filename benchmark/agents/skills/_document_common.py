from __future__ import annotations

from benchmark.agents.skills.base import SkillBase
from benchmark.schemas import (
    AnnotationGuideline,
    ArtifactRole,
    EvidenceSpan,
    SampleArtifact,
    SampleInput,
    SourceDocument,
)


class DocumentSkillMixin(SkillBase):
    """Shared helpers for document-grounded skills."""

    def salient_sentences(self, doc: SourceDocument, limit: int = 4) -> list[str]:
        sentences = [sentence for sentence in self.split_sentences(doc.content) if len(sentence) >= 12]
        if not sentences and doc.content:
            sentences = [doc.content[:250]]
        return sentences[:limit]

    def build_evidence(self, doc: SourceDocument, text: str) -> EvidenceSpan:
        start = doc.content.find(text) if text else -1
        return EvidenceSpan(
            evidence_id=self.make_id("ev", doc.source_id, text),
            source_id=doc.source_id,
            text=text,
            start_char=start if start >= 0 else None,
            end_char=start + len(text) if start >= 0 else None,
            support="supports",
            confidence=min(0.95, 0.45 + doc.trust_level * 0.1),
        )

    def base_input(self, doc: SourceDocument, evidence: EvidenceSpan) -> SampleInput:
        return SampleInput(
            documents=[doc],
            artifacts=[
                SampleArtifact(
                    role=ArtifactRole.document,
                    key="d",
                    value=doc.content,
                    metadata={"source_id": doc.source_id},
                ),
                SampleArtifact(
                    role=ArtifactRole.evidence,
                    key="evidence",
                    value=evidence.text,
                    evidence_ids=[evidence.evidence_id],
                ),
            ],
        )

    def guideline(self, *, need_reasoning: bool = False) -> AnnotationGuideline:
        return AnnotationGuideline(
            label_schema={
                "x": "question/instruction",
                "y": "answer",
                "T": "solution steps" if need_reasoning else "optional",
            },
            positive_criteria=[
                "样本必须仅依赖给定文档。",
                "答案应被最小证据片段直接支持。",
                "问题里不应泄露答案。",
            ],
            negative_criteria=[
                "不得引入外部事实。",
                "不得生成无法从文档验证的答案。",
                "不得把背景介绍误写成结论。",
            ],
            edge_cases=[
                "如果证据不足，应生成拒答型或证据不足型样本。",
            ],
            human_review_required=need_reasoning,
            review_priority="medium" if need_reasoning else "low",
        )
