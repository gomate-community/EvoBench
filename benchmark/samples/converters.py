from __future__ import annotations

from benchmark.schemas import AnswerType, BenchmarkItem, UnifiedSample, VerificationMethod


def unified_to_benchmark_item(sample: UnifiedSample) -> BenchmarkItem:
    """Convert a UnifiedSample to the old BenchmarkItem shape for objective scoring/Arena.

    The conversion is intentionally lossy but preserves the full unified payload under
    input_payload/output_payload/metadata.
    """
    x = sample.artifact("x", "")
    y = sample.artifact("y", sample.artifact("answer", ""))
    trace = sample.artifact("T", [])
    source_ids = [ref.source_id for ref in sample.source_refs]
    return BenchmarkItem(
        question_id=sample.sample_id,
        skill_id=sample.skill_id,
        domain=sample.domain,
        question=str(x),
        answer=y,
        answer_type=AnswerType.structured_json if isinstance(y, dict) else AnswerType.short_text,
        evidence=[ev.text for ev in sample.evidence],
        verification_method=sample.verification_method or VerificationMethod.rubric,
        source_ids=source_ids,
        difficulty_estimate=sample.difficulty_estimate,
        leakage_risk=sample.leakage_risk,
        ambiguity_risk=sample.ambiguity_risk,
        status=sample.status if sample.status in {"candidate", "verified", "rejected", "active", "retired"} else "candidate",
        split=sample.split if sample.split in {"anchor", "fresh", "adversarial", "canary", "public"} else "fresh",
        task_type=sample.task_type,
        input_payload=sample.input.model_dump(),
        output_payload=sample.output.model_dump(),
        source_refs=sample.source_refs,
        reasoning_trace=trace if isinstance(trace, list) else [str(trace)] if trace else [],
        parent_sample_ids=sample.parent_sample_ids,
        instruction=sample.instruction,
        expected_evidence_ids=[ev.evidence_id for ev in sample.evidence],
        annotation_guideline=sample.annotation_guideline,
        quality_signals=sample.quality_signals,
        tags=sample.tags,
        metadata={**sample.metadata, "schema_version": sample.schema_version},
    )
