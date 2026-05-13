from __future__ import annotations

from benchmark.agents.skills._document_common import DocumentSkillMixin
from benchmark.agents.skills.doc_to_answer.prompts import INSTRUCTION, SOURCE_REF_REASON
from benchmark.schemas import (
    ArtifactRole,
    SampleArtifact,
    SampleOutput,
    SourceDocument,
    SourceReference,
    TaskType,
    UnifiedSample,
    VerificationMethod,
)


class DocumentToAnswerSkill(DocumentSkillMixin):
    """Generate answer-like targets from source documents."""

    async def generate(self, *, documents: list[SourceDocument] | None = None, error_samples=None, limit: int = 10) -> list[UnifiedSample]:
        docs = documents or []
        cfg = self._merged_config()
        samples: list[UnifiedSample] = []
        for doc in docs:
            for sentence in self.salient_sentences(doc, limit=cfg.get("sentences_per_doc", 1)):
                evidence = self.build_evidence(doc, sentence)
                answer = sentence.strip()
                samples.append(
                    UnifiedSample(
                        sample_id=self.make_id("sample", self.definition.skill_id, doc.source_id, answer),
                        task_type=TaskType.document_to_x,
                        skill_id=self.definition.skill_id,
                        domain=self.context.config.domain,
                        language=self.context.config.language,
                        input=self.base_input(doc, evidence),
                        output=SampleOutput(
                            artifacts=[
                                SampleArtifact(
                                    role=ArtifactRole.answer,
                                    key="x",
                                    value=answer,
                                    evidence_ids=[evidence.evidence_id],
                                )
                            ],
                            target_schema=self.definition.output_schema,
                        ),
                        source_refs=[SourceReference.from_doc(doc, SOURCE_REF_REASON)],
                        evidence=[evidence],
                        instruction=INSTRUCTION,
                        verification_method=VerificationMethod.evidence_overlap,
                        annotation_guideline=self.guideline(),
                        difficulty_estimate=0.35,
                        tags=["d_to_x", "answer", doc.source_type],
                        metadata={"topic": self.context.topic, "output_kind": "answer"},
                    )
                )
                if len(samples) >= limit:
                    return samples
        return samples
