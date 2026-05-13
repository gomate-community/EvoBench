from __future__ import annotations

from benchmark.agents.skills._document_common import DocumentSkillMixin
from benchmark.agents.skills.doc_to_question.prompts import INSTRUCTION, SOURCE_REF_REASON, make_question
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


class DocumentToQuestionSkill(DocumentSkillMixin):
    """Generate document-grounded questions from source documents."""

    async def generate(self, *, documents: list[SourceDocument] | None = None, error_samples=None, limit: int = 10) -> list[UnifiedSample]:
        docs = documents or []
        cfg = self._merged_config()
        samples: list[UnifiedSample] = []
        for doc in docs:
            for sentence in self.salient_sentences(doc, limit=cfg.get("sentences_per_doc", 1)):
                evidence = self.build_evidence(doc, sentence)
                question = self._make_question(sentence)
                samples.append(
                    UnifiedSample(
                        sample_id=self.make_id("sample", self.definition.skill_id, doc.source_id, question),
                        task_type=TaskType.document_to_x,
                        skill_id=self.definition.skill_id,
                        domain=self.context.config.domain,
                        language=self.context.config.language,
                        input=self.base_input(doc, evidence),
                        output=SampleOutput(
                            artifacts=[
                                SampleArtifact(
                                    role=ArtifactRole.question,
                                    key="x",
                                    value=question,
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
                        tags=["d_to_x", "question", doc.source_type],
                        metadata={"topic": self.context.topic, "output_kind": "question"},
                    )
                )
                if len(samples) >= limit:
                    return samples
        return samples

    def _make_question(self, sentence: str) -> str:
        return make_question(sentence)
