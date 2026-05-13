import pytest

from benchmark.agents.base import AgentRunConfig
from benchmark.agents.skills import SkillContext, SkillRegistry
from benchmark.core.config import settings
from benchmark.pipelines.sample_generation import SampleGenerationPipeline
from benchmark.postprocessors.experience_to_qa import ExperienceToQAPostprocessor
from benchmark.schemas import ErrorSample, SkillGenerationRequest, SourceDocument, TaskType
from benchmark.storage.db import init_storage


@pytest.mark.asyncio
async def test_document_to_qa_skill_outputs_unified_schema():
    doc = SourceDocument(
        source_id="doc1",
        title="AI Test",
        content="Alpha 公司发布了新的 AI 芯片，称推理性能提升 30%。该芯片面向数据中心场景。",
        trust_level=4,
    )
    skill = SkillRegistry().create("doc_to_qa", context=SkillContext(topic="AI"))
    samples = await skill.generate(documents=[doc], limit=2)
    assert samples
    sample = samples[0]
    assert sample.task_type == TaskType.document_to_xy
    assert sample.x
    assert sample.y
    assert sample.evidence
    assert sample.output.target_schema == {"x": "question", "y": "answer"}


@pytest.mark.asyncio
async def test_error_to_training_samples_skill_generates_collection():
    err = ErrorSample(
        error_id="err1",
        input_text="根据证据回答 Alpha 芯片性能提升多少？",
        wrong_output="提升 50%",
        expected_output="提升 30%",
        error_type="numeric_mismatch",
    )
    skill = SkillRegistry().create("error_to_training_samples")
    samples = await skill.generate(error_samples=[err], limit=3)
    assert len(samples) == 3
    assert all(s.task_type == TaskType.error_to_training_set for s in samples)
    assert {s.metadata["variant"] for s in samples} == {"corrected", "contrastive", "boundary"}


@pytest.mark.asyncio
async def test_paper_to_experience_skill_outputs_experience_cards():
    doc = SourceDocument(
        source_id="paper1",
        title="Efficient Retrieval for Long-Context Reasoning",
        source_type="paper",
        content=(
            "We study long-context reasoning over technical documents. "
            "Our experiments show that retrieval-first planning reduces token usage by 28 percent. "
            "The method works best when evidence chunks are ranked before synthesis. "
            "We also note that end-to-end prompting often hides missing evidence and leads to overconfident answers."
        ),
        trust_level=5,
    )
    skill = SkillRegistry().create("paper_to_experience", context=SkillContext(topic="reasoning"))
    samples = await skill.generate(documents=[doc], limit=3)
    assert samples
    sample = samples[0]
    assert sample.task_type == TaskType.document_to_xy
    assert sample.x
    assert isinstance(sample.y, dict)
    assert sample.y["experience_type"] in {"fact", "strategy", "cognitive"}
    assert "actionable_advice" in sample.y
    assert sample.output.target_schema == {"x": "future_problem", "y": "experience_card"}


@pytest.mark.asyncio
async def test_paper_to_experience_supports_more_experiences_than_default_types():
    doc = SourceDocument(
        source_id="paper_extra",
        title="Evidence Planning Patterns",
        source_type="paper",
        content=(
            "Evidence ranking improves retrieval precision. "
            "Planning before synthesis reduces token cost. "
            "Missing citations often lead to overconfident answers. "
            "Human review catches edge cases missed by automated scoring."
        ),
        trust_level=5,
    )
    skill = SkillRegistry().create(
        "paper_to_experience",
        context=SkillContext(
            topic="reasoning",
            config=AgentRunConfig(enable_llm=False),
            skill_config={"experiences_per_doc": 4},
        ),
    )
    samples = await skill.generate(documents=[doc], limit=4)
    assert len(samples) == 4
    assert len({sample.sample_id for sample in samples}) == 4
    assert all(sample.y["experience_type"] in {"fact", "strategy", "cognitive"} for sample in samples)


def test_experience_to_qa_postprocessor_converts_experience_cards():
    source_doc = SourceDocument(
        source_id="paper2",
        title="Planning with Retrieved Evidence",
        source_type="paper",
        content="Retrieved evidence planning improves reasoning quality and reduces hallucination.",
        trust_level=5,
    )
    processor = ExperienceToQAPostprocessor()

    from benchmark.schemas import ArtifactRole, EvidenceSpan, SampleArtifact, SampleInput, SampleOutput, UnifiedSample, VerificationMethod

    evidence = EvidenceSpan(
        evidence_id="ev1",
        source_id="paper2",
        text="Retrieved evidence planning improves reasoning quality and reduces hallucination.",
    )
    source = UnifiedSample(
        sample_id="sample_exp_1",
        task_type=TaskType.document_to_xy,
        skill_id="paper_to_experience",
        input=SampleInput(documents=[source_doc]),
        output=SampleOutput(
            artifacts=[
                SampleArtifact(role=ArtifactRole.question, key="x", value="How should future reasoning tasks use retrieved evidence?"),
                SampleArtifact(
                    role=ArtifactRole.answer,
                    key="y",
                    value={
                        "experience_type": "strategy",
                        "experience_title": "Retrieval-first planning",
                        "experience_statement": "Plan with retrieved evidence before synthesis.",
                        "applicability": "Useful for evidence-heavy reasoning tasks.",
                        "actionable_advice": "Rank evidence chunks before writing the final answer.",
                        "caveats": "Validate evidence quality before trusting the plan.",
                    },
                ),
            ],
            target_schema={"x": "future_problem", "y": "experience_card"},
        ),
        evidence=[evidence],
        verification_method=VerificationMethod.evidence_overlap,
        source_refs=[],
    )
    qa_samples = processor.transform([source], limit=3)
    assert len(qa_samples) == 1
    qa = qa_samples[0]
    assert qa.skill_id == "experience_to_qa"
    assert qa.task_type == TaskType.document_to_xy
    assert isinstance(qa.x, str) and qa.x
    assert isinstance(qa.y, str) and "retrieved evidence" in qa.y.lower()


@pytest.mark.asyncio
async def test_sample_generation_pipeline_respects_custom_samples_output(tmp_path, monkeypatch):
    output_path = tmp_path / "custom_samples.jsonl"
    default_samples_path = tmp_path / "default_samples.jsonl"
    default_corpus_path = tmp_path / "corpus.jsonl"
    legacy_items_path = tmp_path / "items.jsonl"
    monkeypatch.setattr(settings, "llm_enabled", False)
    monkeypatch.setattr(settings, "samples_jsonl_path", str(default_samples_path))
    monkeypatch.setattr(settings, "corpus_jsonl_path", str(default_corpus_path))
    monkeypatch.setattr(settings, "items_jsonl_path", str(legacy_items_path))

    doc = SourceDocument(
        source_id="paper_output",
        title="Configurable Output Paths",
        source_type="paper",
        content="Structured samples should be written to the requested JSONL file.",
        trust_level=5,
    )
    request = SkillGenerationRequest(
        documents=[doc],
        skill_ids=["paper_to_experience"],
        limit=1,
        samples_jsonl_path=str(output_path),
    )

    result = await SampleGenerationPipeline().run_request(request)

    assert result.samples
    assert output_path.exists()
    assert result.samples[0].sample_id in output_path.read_text(encoding="utf-8")
    assert default_samples_path.read_text(encoding="utf-8") == ""


def test_init_storage_does_not_create_legacy_items_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "corpus_jsonl_path", str(tmp_path / "corpus.jsonl"))
    monkeypatch.setattr(settings, "samples_jsonl_path", str(tmp_path / "samples.jsonl"))
    monkeypatch.setattr(settings, "items_jsonl_path", str(tmp_path / "items.jsonl"))

    init_storage()

    assert (tmp_path / "corpus.jsonl").exists()
    assert (tmp_path / "samples.jsonl").exists()
    assert not (tmp_path / "items.jsonl").exists()
