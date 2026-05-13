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
    ErrorSample,
    SampleArtifact,
    SampleInput,
    SampleOutput,
    TaskType,
    UnifiedSample,
    VerificationMethod,
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
                "新样本应针对原错误类型。",
                "正确答案应比原错误输出更受约束。",
                "负样本必须能暴露同类错误。",
            ],
            negative_criteria=[
                "不得简单复制原样本而不变化。",
                "不得生成没有正确答案的训练样本。",
            ],
            edge_cases=[
                "如果 expected_output 缺失，应先生成纠错说明或进入人工复核。",
            ],
            human_review_required=True,
            review_priority="high",
        )

    def _corrected_sample(self, err: ErrorSample) -> UnifiedSample:
        expected = err.expected_output if err.expected_output is not None else "需要人工补充标准答案"
        x = f"请修正以下任务中的错误输出，并给出符合要求的答案。\n任务：{err.input_text}\n错误输出：{err.wrong_output}"
        y = expected
        return UnifiedSample(
            sample_id=self.make_id("sample", self.definition.skill_id, err.error_id, "corrected"),
            task_type=TaskType.error_to_training_set,
            skill_id=self.definition.skill_id,
            domain=self.context.config.domain,
            language=self.context.config.language,
            input=self._base_input(err),
            output=SampleOutput(
                artifacts=[
                    SampleArtifact(role=ArtifactRole.question, key="x", value=x),
                    SampleArtifact(role=ArtifactRole.answer, key="y", value=y),
                    SampleArtifact(role=ArtifactRole.correction, key="correction", value={"from": err.wrong_output, "to": y}),
                ],
                target_schema={"x": "instruction", "y": "corrected_answer"},
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
        expected = err.expected_output if err.expected_output is not None else "待人工确认的优选答案"
        x = f"比较两个候选答案，选择更符合任务要求的一项。\n任务：{err.input_text}"
        y = {"preferred": expected, "rejected": err.wrong_output, "error_type": err.error_type}
        return UnifiedSample(
            sample_id=self.make_id("sample", self.definition.skill_id, err.error_id, "contrastive"),
            task_type=TaskType.error_to_training_set,
            skill_id=self.definition.skill_id,
            domain=self.context.config.domain,
            language=self.context.config.language,
            input=self._base_input(err),
            output=SampleOutput(
                artifacts=[
                    SampleArtifact(role=ArtifactRole.question, key="x", value=x),
                    SampleArtifact(role=ArtifactRole.label, key="y", value=y),
                    SampleArtifact(
                        role=ArtifactRole.critique,
                        key="critique",
                        value=err.diagnosis or f"错误类型：{err.error_type}",
                    ),
                ],
                target_schema={"x": "comparison_prompt", "y": "preference_pair"},
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
        expected = err.expected_output if err.expected_output is not None else "证据不足时应拒答或请求补充信息"
        x = (
            "请处理一个与原错误相邻的边界场景，避免重复同类错误。\n"
            f"原任务：{err.input_text}\n边界约束：只使用题面信息，不得补充未给出的事实。"
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
                    SampleArtifact(role=ArtifactRole.question, key="x", value=x),
                    SampleArtifact(role=ArtifactRole.answer, key="y", value=expected),
                    SampleArtifact(
                        role=ArtifactRole.reasoning_trace,
                        key="T",
                        value=["识别原错误类型。", "检查边界约束。", "生成不过界的答案。"],
                    ),
                ],
                target_schema={"x": "boundary_instruction", "T": "solution_steps", "y": "safe_answer"},
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
