from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnswerType(str, Enum):
    single_choice = "single_choice"
    multi_choice = "multi_choice"
    ranking = "ranking"
    short_text = "short_text"
    evidence_based = "evidence_based"
    abstain = "abstain"
    boolean = "boolean"
    structured_json = "structured_json"
    source_selection = "source_selection"
    rubric = "rubric"


class VerificationMethod(str, Enum):
    exact_match = "exact_match"
    regex = "regex"
    programmatic = "programmatic"
    multi_source = "multi_source"
    official_source = "official_source"
    human = "human"
    judge = "judge"
    evidence_overlap = "evidence_overlap"
    contradiction_check = "contradiction_check"
    rubric = "rubric"


class SampleType(str, Enum):
    fact_verification = "fact_verification"
    temporal_awareness = "temporal_awareness"
    source_attribution = "source_attribution"
    evidence_selection = "evidence_selection"
    conflict_resolution = "conflict_resolution"
    data_value_judgement = "data_value_judgement"
    abstention = "abstention"
    causal_attribution = "causal_attribution"


class TaskType(str, Enum):
    """Top-level generation task family.

    The benchmark framework treats each task as a configurable Skill.  The symbols
    follow the user's notation:
    - document_to_x: d -> x
    - document_to_xy: d -> (x, y)
    - document_to_xty: d -> (x, T, y), where T is a concise solution trace / steps
    - error_to_training_set: e -> new training sample collection
    """

    document_to_x = "document_to_x"
    document_to_xy = "document_to_xy"
    document_to_xty = "document_to_xty"
    error_to_training_set = "error_to_training_set"


class ArtifactRole(str, Enum):
    document = "document"
    question = "question"  # x
    answer = "answer"  # y
    reasoning_trace = "reasoning_trace"  # T, externalized solution steps, not hidden CoT
    evidence = "evidence"
    label = "label"
    critique = "critique"
    correction = "correction"
    metadata = "metadata"


class SkillCategory(str, Enum):
    generation = "generation"
    transformation = "transformation"
    augmentation = "augmentation"
    verification = "verification"
    judging = "judging"


class ReviewPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class SourceDocument(BaseModel):
    source_id: str
    title: str
    url: str | None = None
    source_type: str = "news"
    publisher: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    content: str
    trust_level: int = Field(default=3, ge=1, le=5)
    language: str = "zh-CN"
    authors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceReference(BaseModel):
    source_id: str
    title: str | None = None
    url: str | None = None
    publisher: str | None = None
    source_type: str | None = None
    published_at: datetime | None = None
    trust_level: int | None = Field(default=None, ge=1, le=5)
    selected_reason: str | None = None

    @classmethod
    def from_doc(cls, doc: SourceDocument, selected_reason: str | None = None) -> "SourceReference":
        return cls(
            source_id=doc.source_id,
            title=doc.title,
            url=doc.url,
            publisher=doc.publisher,
            source_type=doc.source_type,
            published_at=doc.published_at,
            trust_level=doc.trust_level,
            selected_reason=selected_reason,
        )


class EvidenceSpan(BaseModel):
    evidence_id: str
    source_id: str
    text: str
    start_char: int | None = None
    end_char: int | None = None
    quote_type: Literal["direct", "paraphrase", "derived"] = "direct"
    support: Literal["supports", "refutes", "neutral"] = "supports"
    confidence: float = Field(default=0.8, ge=0, le=1)


class Claim(BaseModel):
    claim_id: str
    source_id: str
    text: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    event_time: datetime | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    source_trust_level: int = Field(default=3, ge=1, le=5)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampleArtifact(BaseModel):
    """Generic input/output unit used by all task skills."""

    role: ArtifactRole
    key: str
    value: Any
    content_type: str = "text/plain"
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampleInput(BaseModel):
    documents: list[SourceDocument] = Field(default_factory=list)
    artifacts: list[SampleArtifact] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class SampleOutput(BaseModel):
    artifacts: list[SampleArtifact] = Field(default_factory=list)
    target_schema: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        for artifact in self.artifacts:
            if artifact.key == key:
                return artifact.value
        return default


class ErrorSample(BaseModel):
    error_id: str
    input_text: str
    wrong_output: Any
    expected_output: Any | None = None
    error_type: str = "unknown"
    source_sample_id: str | None = None
    diagnosis: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnnotationGuideline(BaseModel):
    label_schema: dict[str, Any] = Field(default_factory=dict)
    positive_criteria: list[str] = Field(default_factory=list)
    negative_criteria: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    review_priority: ReviewPriority = ReviewPriority.low


class QualitySignals(BaseModel):
    source_support_count: int = 0
    source_refute_count: int = 0
    evidence_coverage: float = Field(default=0.0, ge=0, le=1)
    answerability: float = Field(default=1.0, ge=0, le=1)
    clarity: float = Field(default=0.5, ge=0, le=1)
    novelty: float = Field(default=0.5, ge=0, le=1)
    estimated_human_minutes: float = Field(default=2.0, ge=0)
    quality_gate_passed: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)


class SkillDefinition(BaseModel):
    skill_id: str
    name: str
    task_type: TaskType
    category: SkillCategory = SkillCategory.generation
    description: str = ""
    enabled: bool = True
    input_requirements: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    quality_rules: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class UnifiedSample(BaseModel):
    """Unified schema for benchmark, SFT, evaluation and error-augmentation samples.

    All tasks are represented as:
        input: documents / claims / error artifacts / raw payload
        output: typed artifacts such as x, y, T, labels and critiques

    This lets d->x, d->(x,y), d->(x,T,y) and error-based augmentation share the
    same storage, verification, filtering and lineage logic.
    """

    sample_id: str
    schema_version: str = "sample_schema_v1"
    task_type: TaskType
    skill_id: str
    domain: str = "technology"
    language: str = "zh-CN"
    input: SampleInput = Field(default_factory=SampleInput)
    output: SampleOutput = Field(default_factory=SampleOutput)
    source_refs: list[SourceReference] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    instruction: str | None = None
    verification_method: VerificationMethod = VerificationMethod.rubric
    annotation_guideline: AnnotationGuideline = Field(default_factory=AnnotationGuideline)
    quality_signals: QualitySignals = Field(default_factory=QualitySignals)
    status: Literal["candidate", "verified", "rejected", "active", "retired"] = "candidate"
    split: Literal["anchor", "fresh", "adversarial", "canary", "public", "train", "dev"] = "fresh"
    difficulty_estimate: float = Field(default=0.5, ge=0, le=1)
    leakage_risk: float = Field(default=0.0, ge=0, le=1)
    ambiguity_risk: float = Field(default=0.0, ge=0, le=1)
    parent_sample_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def artifact(self, key: str, default: Any = None) -> Any:
        return self.output.get(key, default)

    @property
    def x(self) -> Any:
        return self.artifact("x")

    @property
    def y(self) -> Any:
        return self.artifact("y")

    @property
    def T(self) -> Any:
        return self.artifact("T")


class BenchmarkItem(BaseModel):
    """Backward-compatible benchmark item.

    v3 keeps this model so the v1/v2 evaluation, arena and repository code continues
    to work. New task-skill pipelines should prefer UnifiedSample and can convert to a
    BenchmarkItem when they need objective scoring or Arena battles.
    """

    question_id: str
    skill_id: str
    domain: str
    question: str
    answer: Any
    answer_type: AnswerType
    evidence: list[str] = Field(default_factory=list)
    verification_method: VerificationMethod
    source_ids: list[str] = Field(default_factory=list)
    freshness_window_days: int | None = None
    difficulty_estimate: float = Field(default=0.5, ge=0, le=1)
    leakage_risk: float = Field(default=0.0, ge=0, le=1)
    ambiguity_risk: float = Field(default=0.0, ge=0, le=1)
    status: Literal["candidate", "verified", "rejected", "active", "retired"] = "candidate"
    split: Literal["anchor", "fresh", "adversarial", "canary", "public"] = "fresh"
    version: str = "v0"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    sample_type: SampleType | None = None
    task_type: TaskType | None = None
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[SourceReference] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    parent_sample_ids: list[str] = Field(default_factory=list)
    instruction: str | None = None
    options: list[str] = Field(default_factory=list)
    expected_evidence_ids: list[str] = Field(default_factory=list)
    rubric: dict[str, Any] = Field(default_factory=dict)
    annotation_guideline: AnnotationGuideline = Field(default_factory=AnnotationGuideline)
    quality_signals: QualitySignals = Field(default_factory=QualitySignals)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    model_id: str
    question_id: str
    answer: Any
    reasoning_summary: str | None = None
    evidence_used: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency_ms: int | None = None
    cost_usd: float | None = None


class ScoreResult(BaseModel):
    question_id: str
    model_id: str
    score: float = Field(ge=0, le=1)
    dimensions: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class BattleResult(BaseModel):
    battle_id: str
    question_id: str
    model_a: str
    model_b: str
    winner: Literal["a", "b", "tie"]
    judge_id: str
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class GenerationRequest(BaseModel):
    topic: str
    domain: str = "technology"
    sample_types: list[SampleType] = Field(default_factory=lambda: [SampleType.fact_verification])
    limit: int = Field(default=10, ge=1, le=100)
    language: str = "zh-CN"
    include_adversarial: bool = True
    require_multi_source: bool = False


class SkillGenerationRequest(BaseModel):
    topic: str | None = None
    corpus_jsonl_path: str | None = None
    samples_jsonl_path: str | None = None
    task_type: TaskType | None = None
    skill_ids: list[str] = Field(default_factory=list)
    domain: str = "technology"
    language: str = "zh-CN"
    limit: int = Field(default=10, ge=1, le=100)
    source_filter: dict[str, Any] = Field(default_factory=dict)
    skill_config: dict[str, Any] = Field(default_factory=dict)
    documents: list[SourceDocument] = Field(default_factory=list)
    error_samples: list[ErrorSample] = Field(default_factory=list)


class SkillGenerationResult(BaseModel):
    request: SkillGenerationRequest
    samples: list[UnifiedSample] = Field(default_factory=list)
    rejected: list[UnifiedSample] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
