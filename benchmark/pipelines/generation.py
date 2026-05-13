from __future__ import annotations

from benchmark.adapters.llm import LLMAdapter, build_llm_adapter
from benchmark.adapters.retriever import MockRetriever, RetrieverAdapter
from benchmark.agents.base import AgentRunConfig
from benchmark.agents.claim_agent import ClaimAgent
from benchmark.agents.question_agent import QuestionAgent
from benchmark.agents.source_agent import SourceAgent
from benchmark.agents.verifier_agent import VerifierAgent
from benchmark.core.config import settings
from benchmark.schemas import GenerationRequest, SampleType, SkillGenerationRequest, SkillGenerationResult
from benchmark.storage.repository import BenchmarkRepository
from benchmark.pipelines.sample_generation import SampleGenerationPipeline


class GenerationPipeline:
    def __init__(self, retriever: RetrieverAdapter | None = None, llm: LLMAdapter | None = None):
        retriever = retriever or MockRetriever()
        self.llm = llm or (build_llm_adapter() if settings.llm_enabled else None)
        self.source = SourceAgent(retriever)
        self.repo = BenchmarkRepository()
        self.verifier = VerifierAgent()

    async def run(
        self,
        topic: str,
        limit: int = 10,
        sample_types: list[SampleType] | None = None,
        domain: str = "technology",
        language: str = "zh-CN",
    ) -> int:
        request = GenerationRequest(
            topic=topic,
            limit=limit,
            domain=domain,
            language=language,
            sample_types=sample_types
            or [
                SampleType.fact_verification,
                SampleType.source_attribution,
                SampleType.evidence_selection,
                SampleType.temporal_awareness,
                SampleType.abstention,
                SampleType.data_value_judgement,
            ],
        )
        return await self.run_request(request)


    async def run_skill_request(self, request: SkillGenerationRequest) -> SkillGenerationResult:
        """v3 unified task/skill generation entrypoint."""
        return await SampleGenerationPipeline(retriever=self.source.retriever, llm=self.llm).run_request(request)

    async def run_request(self, request: GenerationRequest) -> int:
        docs = await self.source.collect(request.topic, limit=request.limit)
        self.repo.upsert_documents(docs)
        config = AgentRunConfig(language=request.language, domain=request.domain)
        claim_agent = ClaimAgent(config=config, llm=self.llm)
        question_agent = QuestionAgent(config=config, llm=self.llm)
        claims = await claim_agent.extract_claims(docs)
        selected_sample_types = list(request.sample_types)
        if request.include_adversarial and SampleType.conflict_resolution not in selected_sample_types:
            selected_sample_types.append(SampleType.conflict_resolution)
        items = await question_agent.generate(claims, docs=docs, sample_types=selected_sample_types)
        verified = self.verifier.verify_batch(items)
        saved = 0
        for item in verified:
            self.repo.upsert_item(item)
            saved += 1
        return saved
