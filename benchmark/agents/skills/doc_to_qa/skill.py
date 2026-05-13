from __future__ import annotations

from benchmark.agents.skills._document_common import DocumentSkillMixin
from benchmark.agents.skills.doc_to_qa.prompts import (
    INSTRUCTION,
    SOURCE_REF_REASON,
    answer_from_sentence,
    question_from_sentence,
)
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


class DocumentToQASkill(DocumentSkillMixin):
    """Generate evidence-grounded QA pairs from a document."""

    async def generate(self, *, documents: list[SourceDocument] | None = None, error_samples=None, limit: int = 10) -> list[UnifiedSample]:
        docs = documents or []
        cfg = self._merged_config()
        samples: list[UnifiedSample] = []
        for doc in docs:
            for sentence in self.salient_sentences(doc, limit=cfg.get("pairs_per_doc", 2)):
                evidence = self.build_evidence(doc, sentence)
                question = self._question_from_sentence(sentence, doc)
                answer = self._answer_from_sentence(sentence)
                samples.append(
                    UnifiedSample(
                        sample_id=self.make_id("sample", self.definition.skill_id, doc.source_id, question, answer),
                        task_type=TaskType.document_to_xy,
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
                                ),
                                SampleArtifact(
                                    role=ArtifactRole.answer,
                                    key="y",
                                    value=answer,
                                    evidence_ids=[evidence.evidence_id],
                                ),
                            ],
                            target_schema={"x": "question", "y": "answer"},
                        ),
                        source_refs=[SourceReference.from_doc(doc, SOURCE_REF_REASON)],
                        evidence=[evidence],
                        instruction=INSTRUCTION,
                        verification_method=VerificationMethod.evidence_overlap,
                        annotation_guideline=self.guideline(),
                        difficulty_estimate=0.45,
                        tags=["d_to_xy", "qa", doc.source_type],
                        metadata={"topic": self.context.topic},
                    )
                )
                if len(samples) >= limit:
                    return samples
        return samples

    def _question_from_sentence(self, sentence: str, doc: SourceDocument) -> str:
        return question_from_sentence(sentence, doc)

    def _answer_from_sentence(self, sentence: str) -> str:
        return answer_from_sentence(sentence)
