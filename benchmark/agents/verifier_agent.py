from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from benchmark.agents.base import AgentBase
from benchmark.schemas import AnswerType, BenchmarkItem, QualitySignals, UnifiedSample, VerificationMethod


@dataclass(frozen=True)
class VerificationPolicy:
    min_question_chars: int = 12
    min_evidence_count: int = 1
    max_ambiguity_risk: float = 0.55
    max_leakage_risk: float = 0.75
    min_evidence_coverage: float = 0.35
    require_answer: bool = True
    reject_near_duplicates: bool = True


class VerifierAgent(AgentBase):
    """Independent quality gate for generated annotation samples.

    It validates answerability, evidence grounding, schema consistency, leakage risk,
    ambiguity risk and near-duplicate questions. It should run after QuestionAgent and
    before items enter the active benchmark pool.
    """

    name = "verifier_agent"

    def __init__(self, policy: VerificationPolicy | None = None):
        self.policy = policy or VerificationPolicy()
        self._seen_questions: list[str] = []

    async def verify(self, item: BenchmarkItem) -> BenchmarkItem:
        return self.verify_batch([item])[0]

    def verify_batch(self, items: Iterable[BenchmarkItem]) -> list[BenchmarkItem]:
        verified: list[BenchmarkItem] = []
        for item in items:
            reasons = self._rejection_reasons(item)
            if self.policy.reject_near_duplicates and self._is_duplicate(item.question):
                reasons.append("near_duplicate_question")
            coverage = self._evidence_coverage(item)
            item.quality_signals = item.quality_signals or QualitySignals()
            item.quality_signals.evidence_coverage = coverage
            item.quality_signals.answerability = self._answerability(item)
            item.quality_signals.clarity = max(0.0, 1.0 - item.ambiguity_risk)
            item.quality_signals.novelty = max(0.0, 1.0 - item.leakage_risk)
            item.quality_signals.rejection_reasons = reasons
            item.quality_signals.quality_gate_passed = not reasons
            item.status = "verified" if not reasons else "rejected"
            if item.status == "verified":
                self._seen_questions.append(item.question)
            verified.append(item)
        return verified



    def verify_samples(self, samples: Iterable[UnifiedSample]) -> list[UnifiedSample]:
        """Quality gate for the new unified schema."""
        verified: list[UnifiedSample] = []
        seen: list[str] = []
        for sample in samples:
            reasons = self._sample_rejection_reasons(sample)
            x_text = str(sample.artifact("x", ""))
            if self.policy.reject_near_duplicates and any(self.lexical_overlap(x_text, old) > 0.95 for old in seen):
                reasons.append("near_duplicate_x")
            coverage = self._sample_evidence_coverage(sample)
            sample.quality_signals.evidence_coverage = coverage
            sample.quality_signals.answerability = 0.75 if sample.artifact("y") is None and sample.task_type.value == "error_to_training_set" else 1.0
            sample.quality_signals.clarity = max(0.0, 1.0 - sample.ambiguity_risk)
            sample.quality_signals.novelty = max(0.0, 1.0 - sample.leakage_risk)
            sample.quality_signals.rejection_reasons = sorted(set(reasons))
            sample.quality_signals.quality_gate_passed = not reasons
            sample.status = "verified" if not reasons else "rejected"
            if sample.status == "verified":
                seen.append(x_text)
            verified.append(sample)
        return verified

    def _sample_rejection_reasons(self, sample: UnifiedSample) -> list[str]:
        reasons: list[str] = []
        if not sample.output.artifacts:
            reasons.append("missing_output_artifacts")
        if not sample.artifact("x") and sample.task_type.value.startswith("document_to"):
            reasons.append("missing_x")
        if sample.task_type.value in {"document_to_xy", "document_to_xty"} and not sample.artifact("y"):
            reasons.append("missing_y")
        if sample.task_type.value == "document_to_xty" and not sample.artifact("T"):
            reasons.append("missing_T")
        if len(sample.evidence) < self.policy.min_evidence_count and sample.task_type.value != "error_to_training_set":
            reasons.append("missing_evidence")
        if sample.ambiguity_risk > self.policy.max_ambiguity_risk:
            reasons.append("ambiguity_risk_too_high")
        if sample.leakage_risk > self.policy.max_leakage_risk:
            reasons.append("leakage_risk_too_high")
        if self._sample_evidence_coverage(sample) < self.policy.min_evidence_coverage and sample.task_type.value != "error_to_training_set":
            reasons.append("low_evidence_coverage")
        return sorted(set(reasons))

    def _sample_evidence_coverage(self, sample: UnifiedSample) -> float:
        if not sample.evidence:
            return 0.0
        output_text = " ".join(str(a.value) for a in sample.output.artifacts)
        input_text = " ".join(str(a.value) for a in sample.input.artifacts)
        haystack = output_text + " " + input_text
        overlaps = []
        for ev in sample.evidence:
            overlap = self.lexical_overlap(ev.text, haystack)
            if ev.text and ev.text in haystack:
                overlap = max(overlap, 1.0)
            overlaps.append(overlap)
        return max(overlaps) if overlaps else 0.0

    def _rejection_reasons(self, item: BenchmarkItem) -> list[str]:
        reasons: list[str] = []
        if len(item.question.strip()) < self.policy.min_question_chars:
            reasons.append("question_too_short")
        if self.policy.require_answer and not self._has_answer(item):
            reasons.append("missing_answer")
        if len(item.evidence) < self.policy.min_evidence_count:
            reasons.append("missing_evidence")
        if item.ambiguity_risk > self.policy.max_ambiguity_risk:
            reasons.append("ambiguity_risk_too_high")
        if item.leakage_risk > self.policy.max_leakage_risk:
            reasons.append("leakage_risk_too_high")
        if self._evidence_coverage(item) < self.policy.min_evidence_coverage:
            reasons.append("low_evidence_coverage")
        reasons.extend(self._schema_reasons(item))
        return sorted(set(reasons))

    def _has_answer(self, item: BenchmarkItem) -> bool:
        if item.answer is None:
            return False
        if isinstance(item.answer, str):
            return bool(item.answer.strip())
        if isinstance(item.answer, (list, dict)):
            return bool(item.answer)
        return True

    def _schema_reasons(self, item: BenchmarkItem) -> list[str]:
        reasons: list[str] = []
        if item.answer_type == AnswerType.single_choice and item.options and item.answer not in item.options:
            reasons.append("answer_not_in_options")
        if item.answer_type == AnswerType.source_selection and item.answer not in item.source_ids and not any(
            str(item.answer) in opt for opt in item.options
        ):
            reasons.append("source_answer_not_in_options")
        if item.verification_method == VerificationMethod.rubric and not item.rubric:
            reasons.append("rubric_required")
        if item.answer_type == AnswerType.structured_json and not isinstance(item.answer, dict):
            reasons.append("structured_answer_must_be_dict")
        return reasons

    def _evidence_coverage(self, item: BenchmarkItem) -> float:
        if not item.evidence:
            return 0.0
        answer_text = str(item.answer)
        question_text = item.question
        overlaps = []
        for ev in item.evidence:
            overlap = max(self.lexical_overlap(ev, question_text), self.lexical_overlap(ev, answer_text))
            # Exact containment is common in generated evidence-selection tasks.
            if ev and (ev in question_text or ev in answer_text):
                overlap = max(overlap, 1.0)
            overlaps.append(overlap)
        return max(overlaps) if overlaps else 0.0

    def _answerability(self, item: BenchmarkItem) -> float:
        if item.answer_type == AnswerType.abstain:
            return 0.75
        if not item.evidence:
            return 0.0
        if any(ev in item.question or ev in str(item.answer) for ev in item.evidence):
            return 1.0
        return min(1.0, 0.5 + self._evidence_coverage(item))

    def _is_duplicate(self, question: str) -> bool:
        normalized = re.sub(r"\s+", " ", question).strip().lower()
        return any(self.lexical_overlap(normalized, old) > 0.95 for old in self._seen_questions)
