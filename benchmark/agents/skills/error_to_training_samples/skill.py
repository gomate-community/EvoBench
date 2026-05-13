from __future__ import annotations

from benchmark.agents.skills.base import SkillBase
from benchmark.agents.skills.error_to_training_samples.prompts import (
    BOUNDARY_INSTRUCTION,
    CONTRASTIVE_INSTRUCTION,
    CORRECTED_INSTRUCTION,
)
from benchmark.schemas import (
    AnnotationGuideline,
    ArtifactRole,
    ErrorBoundarySampleSchema,
    ErrorContrastiveSampleSchema,
    ErrorCorrectionDeltaSchema,
    ErrorCorrectedSampleSchema,
    ErrorSample,
    SampleArtifact,
    SampleInput,
    SampleOutput,
    TaskType,
    UnifiedSample,
    VerificationMethod,
    ERROR_TO_TRAINING_BOUNDARY_OUTPUT_SCHEMA,
    ERROR_TO_TRAINING_CONTRASTIVE_OUTPUT_SCHEMA,
    ERROR_TO_TRAINING_CORRECTED_OUTPUT_SCHEMA,
)


class ErrorToTrainingSamplesSkill(SkillBase):
    """Generate corrected, contrastive, and boundary samples from errors."""

    async def generate(self, *, documents=None, error_samples: list[ErrorSample] | None = None, limit: int = 10) -> list[UnifiedSample]:
        errors = error_samples or []
        cfg = self._merged_config()
        include_contrastive = cfg.get("include_contrastive", True)
        include_boundary = cfg.get("include_boundary", True)
        samples: list[UnifiedSample] = []
        for err in errors:
            samples.append(self._corrected_sample(err))
            if include_contrastive:
                samples.append(self._contrastive_sample(err))
            if include_boundary:
                samples.append(self._boundary_sample(err))
            if len(samples) >= limit:
                return samples[:limit]
        return samples[:limit]

    def _base_input(self, err: ErrorSample) -> SampleInput:
        return SampleInput(
            artifacts=[
                SampleArtifact(role=ArtifactRole.question, key="original_input", value=err.input_text),
                SampleArtifact(role=ArtifactRole.answer, key="wrong_output", value=err.wrong_output),
                SampleArtifact(role=ArtifactRole.label, key="error_type", value=err.error_type),
            ],
            raw={"error_sample": err.model_dump()},
        )

    def _guideline(self) -> AnnotationGuideline:
        return AnnotationGuideline(
            label_schema={
                "x": "new input",
                "y": "correct output",
                "error_type": "diagnosis",
                "negative_y": "optional wrong answer",
            },
            positive_criteria=[
                "The new sample should target the same failure mode as the original error.",
                "The corrected answer should be more constrained and more faithful than the wrong output.",
                "Negative or contrastive variants should expose the same class of mistake clearly.",
            ],
            negative_criteria=[
                "Do not duplicate the source sample without introducing a meaningful learning signal.",
                "Do not emit a training sample with no usable target behavior.",
            ],
            edge_cases=[
                "If expected_output is missing, prefer human review or an abstaining target.",
            ],
            human_review_required=True,
            review_priority="high",
        )

    def _corrected_sample(self, err: ErrorSample) -> UnifiedSample:
        expected = err.expected_output if err.expected_output is not None else "A human reviewer should provide the corrected target."
        x = (
            "Fix the incorrect output for the task below and produce the corrected answer.\n"
            f"Task: {err.input_text}\n"
            f"Wrong output: {err.wrong_output}"
        )
        correction = ErrorCorrectionDeltaSchema.model_validate({"from": err.wrong_output, "to": expected})
        payload = ErrorCorrectedSampleSchema(x=x, y=expected, correction=correction)
        return UnifiedSample(
            sample_id=self.make_id("sample", self.definition.skill_id, err.error_id, "corrected"),
            task_type=TaskType.error_to_training_set,
            skill_id=self.definition.skill_id,
            domain=self.context.config.domain,
            language=self.context.config.language,
            input=self._base_input(err),
            output=SampleOutput(
                artifacts=[
                    SampleArtifact(role=ArtifactRole.question, key="x", value=payload.x),
                    SampleArtifact(role=ArtifactRole.answer, key="y", value=payload.y),
                    SampleArtifact(role=ArtifactRole.correction, key="correction", value=payload.correction.model_dump(by_alias=True)),
                ],
                target_schema=ERROR_TO_TRAINING_CORRECTED_OUTPUT_SCHEMA,
            ),
            instruction=CORRECTED_INSTRUCTION,
            verification_method=VerificationMethod.human if err.expected_output is None else VerificationMethod.exact_match,
            annotation_guideline=self._guideline(),
            status="candidate",
            split="train",
            difficulty_estimate=0.6,
            ambiguity_risk=0.25 if err.expected_output is None else 0.1,
            parent_sample_ids=[err.source_sample_id] if err.source_sample_id else [],
            tags=["error_aug", "corrected", err.error_type],
            metadata={"error_id": err.error_id, "variant": "corrected"},
        )

    def _contrastive_sample(self, err: ErrorSample) -> UnifiedSample:
        preferred = err.expected_output if err.expected_output is not None else "A human reviewer should select the preferred output."
        x = (
            "Compare the candidate outputs below and choose the one that best satisfies the task.\n"
            f"Task: {err.input_text}"
        )
        critique = err.diagnosis or f"Error type: {err.error_type}"
        payload = ErrorContrastiveSampleSchema(
            x=x,
            y={"preferred": preferred, "rejected": err.wrong_output, "error_type": err.error_type},
            critique=critique,
        )
        return UnifiedSample(
            sample_id=self.make_id("sample", self.definition.skill_id, err.error_id, "contrastive"),
            task_type=TaskType.error_to_training_set,
            skill_id=self.definition.skill_id,
            domain=self.context.config.domain,
            language=self.context.config.language,
            input=self._base_input(err),
            output=SampleOutput(
                artifacts=[
                    SampleArtifact(role=ArtifactRole.question, key="x", value=payload.x),
                    SampleArtifact(role=ArtifactRole.label, key="y", value=payload.y.model_dump()),
                    SampleArtifact(role=ArtifactRole.critique, key="critique", value=payload.critique),
                ],
                target_schema=ERROR_TO_TRAINING_CONTRASTIVE_OUTPUT_SCHEMA,
            ),
            instruction=CONTRASTIVE_INSTRUCTION,
            verification_method=VerificationMethod.human,
            annotation_guideline=self._guideline(),
            split="train",
            difficulty_estimate=0.65,
            ambiguity_risk=0.2,
            parent_sample_ids=[err.source_sample_id] if err.source_sample_id else [],
            tags=["error_aug", "contrastive", err.error_type],
            metadata={"error_id": err.error_id, "variant": "contrastive"},
        )

    def _boundary_sample(self, err: ErrorSample) -> UnifiedSample:
        safe_answer = err.expected_output if err.expected_output is not None else "Abstain or ask for the missing information instead of guessing."
        x = (
            "Handle a boundary case related to the original failure without repeating the same mistake.\n"
            f"Original task: {err.input_text}\n"
            "Boundary constraint: use only the information in the prompt and do not invent missing facts."
        )
        payload = ErrorBoundarySampleSchema(
            x=x,
            T=[
                "Identify the original failure mode.",
                "Check the boundary constraint carefully.",
                "Produce a safe answer that does not cross the stated limits.",
            ],
            y=safe_answer,
        )
        return UnifiedSample(
            sample_id=self.make_id("sample", self.definition.skill_id, err.error_id, "boundary"),
            task_type=TaskType.error_to_training_set,
            skill_id=self.definition.skill_id,
            domain=self.context.config.domain,
            language=self.context.config.language,
            input=self._base_input(err),
            output=SampleOutput(
                artifacts=[
                    SampleArtifact(role=ArtifactRole.question, key="x", value=payload.x),
                    SampleArtifact(role=ArtifactRole.answer, key="y", value=payload.y),
                    SampleArtifact(role=ArtifactRole.reasoning_trace, key="T", value=payload.T),
                ],
                target_schema=ERROR_TO_TRAINING_BOUNDARY_OUTPUT_SCHEMA,
            ),
            instruction=BOUNDARY_INSTRUCTION,
            verification_method=VerificationMethod.human,
            annotation_guideline=self._guideline(),
            split="train",
            difficulty_estimate=0.7,
            ambiguity_risk=0.25,
            parent_sample_ids=[err.source_sample_id] if err.source_sample_id else [],
            tags=["error_aug", "boundary", err.error_type],
            metadata={"error_id": err.error_id, "variant": "boundary"},
        )
