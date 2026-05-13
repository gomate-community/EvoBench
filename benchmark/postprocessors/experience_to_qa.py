from __future__ import annotations

from typing import Iterable

from benchmark.agents.base import AgentBase
from benchmark.schemas import (
    AnnotationGuideline,
    ArtifactRole,
    DOC_TO_QA_OUTPUT_SCHEMA,
    SampleArtifact,
    SampleOutput,
    TaskType,
    UnifiedSample,
    VerificationMethod,
)


class ExperienceToQAPostprocessor(AgentBase):
    """Convert structured experience cards into downstream QA samples."""

    name = "experience_to_qa"

    def transform(
        self,
        samples: Iterable[UnifiedSample],
        *,
        limit: int | None = None,
        source_skill_id: str = "paper_to_experience",
    ) -> list[UnifiedSample]:
        qa_samples: list[UnifiedSample] = []
        for sample in samples:
            if sample.skill_id != source_skill_id:
                continue
            if not isinstance(sample.y, dict):
                continue
            qa_sample = self._convert_sample(sample)
            qa_samples.append(qa_sample)
            if limit is not None and len(qa_samples) >= limit:
                return qa_samples
        return qa_samples

    def _convert_sample(self, sample: UnifiedSample) -> UnifiedSample:
        experience = sample.y if isinstance(sample.y, dict) else {}
        experience_type = str(experience.get("experience_type", "experience"))
        question = self._compose_question(str(sample.x), experience_type, experience)
        answer = self._compose_answer(experience)
        evidence_ids = [ev.evidence_id for ev in sample.evidence]
        parent_ids = [sample.sample_id, *sample.parent_sample_ids]

        return UnifiedSample(
            sample_id=self.make_id("sample", self.name, sample.sample_id, question),
            task_type=TaskType.document_to_xy,
            skill_id=self.name,
            domain=sample.domain,
            language=sample.language,
            input=sample.input,
            output=SampleOutput(
                artifacts=[
                    SampleArtifact(
                        role=ArtifactRole.question,
                        key="x",
                        value=question,
                        evidence_ids=evidence_ids,
                    ),
                    SampleArtifact(
                        role=ArtifactRole.answer,
                        key="y",
                        value=answer,
                        evidence_ids=evidence_ids,
                    ),
                    SampleArtifact(
                        role=ArtifactRole.label,
                        key="experience_type",
                        value=experience_type,
                        evidence_ids=evidence_ids,
                    ),
                ],
                target_schema=DOC_TO_QA_OUTPUT_SCHEMA,
            ),
            source_refs=sample.source_refs,
            evidence=sample.evidence,
            instruction="Convert an experience card into a reusable future-facing QA sample.",
            verification_method=VerificationMethod.evidence_overlap,
            annotation_guideline=self._guideline(),
            difficulty_estimate=min(0.8, sample.difficulty_estimate + 0.05),
            ambiguity_risk=sample.ambiguity_risk,
            parent_sample_ids=parent_ids,
            tags=[*sample.tags, "experience_to_qa", experience_type],
            metadata={
                **sample.metadata,
                "source_skill_id": sample.skill_id,
                "postprocessor": self.name,
            },
        )

    def _compose_question(self, future_problem: str, experience_type: str, experience: dict) -> str:
        title = str(experience.get("experience_title", "")).strip()
        if experience_type == "fact":
            return f"For the future problem '{future_problem}', what validated factual experience from {title or 'the paper'} should guide the answer?"
        if experience_type == "strategy":
            return f"For the future problem '{future_problem}', what strategy from {title or 'the paper'} should be applied first?"
        if experience_type == "mechanism":
            return f"For the future problem '{future_problem}', what mechanism from {title or 'the paper'} best explains the expected behavior?"
        if experience_type == "boundary":
            return f"For the future problem '{future_problem}', what scope limit from {title or 'the paper'} should constrain the answer?"
        if experience_type == "failure":
            return f"For the future problem '{future_problem}', what failed attempt or negative lesson from {title or 'the paper'} should be avoided?"
        return f"For the future problem '{future_problem}', what transferable experience from {title or 'the paper'} should be kept in mind?"

    def _compose_answer(self, experience: dict) -> str:
        statement = str(experience.get("experience_statement", "")).strip()
        advice = str(experience.get("actionable_advice", "")).strip()
        applicability = str(experience.get("applicability", "")).strip()
        caveats = str(experience.get("caveats", "")).strip()
        parts = [part for part in [statement, advice, applicability, caveats] if part]
        return " ".join(parts)

    def _guideline(self) -> AnnotationGuideline:
        return AnnotationGuideline(
            label_schema={"x": "question", "y": "answer"},
            positive_criteria=[
                "Question should target future use of the extracted experience.",
                "Answer should preserve the source experience and actionable advice.",
                "The transformed QA must remain evidence-grounded.",
            ],
            negative_criteria=[
                "Do not introduce new claims beyond the experience card.",
                "Do not drop important caveats or applicability constraints.",
            ],
            edge_cases=[
                "If the original experience is highly conditional, keep the condition explicit in the answer.",
            ],
            human_review_required=False,
            review_priority="low",
        )
