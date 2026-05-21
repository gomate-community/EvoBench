from __future__ import annotations

from benchmark.adapters.llm import LLMAdapter, build_llm_adapter
from benchmark.adapters.retriever import MockRetriever, RetrieverAdapter
from benchmark.agents.sample_factory_agent import SampleFactoryAgent
from benchmark.agents.source_agent import SourceAgent
from benchmark.agents.source_selector_agent import SourceSelectionPolicy, SourceSelectorAgent
from benchmark.agents.skills import SkillRegistry
from benchmark.agents.verifier_agent import VerifierAgent
from benchmark.core.config import settings
from benchmark.schemas import SkillGenerationRequest, SkillGenerationResult, SourceDocument, TaskType
from benchmark.storage.repository import BenchmarkRepository


class SampleGenerationPipeline:
    """Unified pipeline: shared source selection + configurable skill execution."""

    def __init__(
        self,
        retriever: RetrieverAdapter | None = None,
        registry: SkillRegistry | None = None,
        source_policy: SourceSelectionPolicy | None = None,
        llm: LLMAdapter | None = None,
    ):
        retriever = retriever or MockRetriever()
        self.llm = llm or (build_llm_adapter() if settings.llm_enabled else None)
        self.source_policy = source_policy or SourceSelectionPolicy()
        self.source_agent = SourceAgent(retriever)
        self.selector = SourceSelectorAgent(self.source_policy)
        self.factory = SampleFactoryAgent(registry or SkillRegistry(), llm=self.llm)
        self.verifier = VerifierAgent()
        self.repo = BenchmarkRepository()

    async def run_request(self, request: SkillGenerationRequest, save: bool = True) -> SkillGenerationResult:
        repo = self._repository_for_request(request)
        docs = request.documents or await self._collect_docs(request)
        if docs:
            self.repo.upsert_documents(docs)
        selected_docs = self._select_docs(docs, request)
        samples = await self.factory.generate(request, documents=selected_docs, error_samples=request.error_samples)
        verified = self.verifier.verify_samples(samples)
        accepted = [s for s in verified if s.status == "verified"]
        rejected = [s for s in verified if s.status == "rejected"]
        if save:
            for sample in verified:
                repo.upsert_sample(sample)
        return SkillGenerationResult(
            request=request,
            samples=accepted,
            rejected=rejected,
            metrics={
                "input_docs": len(docs),
                "selected_docs": len(selected_docs),
                "generated": len(samples),
                "verified": len(accepted),
                "rejected": len(rejected),
            },
        )

    def _repository_for_request(self, request: SkillGenerationRequest) -> BenchmarkRepository:
        if request.samples_jsonl_path:
            return BenchmarkRepository(samples_path=request.samples_jsonl_path)
        return self.repo

    def _select_docs(self, docs: list[SourceDocument], request: SkillGenerationRequest) -> list[SourceDocument]:
        policy_kwargs = {
            "min_trust_level": self.source_policy.min_trust_level,
            "source_types": self.source_policy.source_types,
            "language": self.source_policy.language,
            "max_age_days": self.source_policy.max_age_days,
            "min_content_chars": self.source_policy.min_content_chars,
            "require_url": self.source_policy.require_url,
            "dedupe_threshold": self.source_policy.dedupe_threshold,
        }

        if request.corpus_jsonl_path:
            policy_kwargs["min_content_chars"] = min(policy_kwargs["min_content_chars"], 1)

        if "min_trust_level" in request.source_filter:
            policy_kwargs["min_trust_level"] = int(request.source_filter["min_trust_level"])
        if "language" in request.source_filter:
            policy_kwargs["language"] = request.source_filter["language"]
        if "max_age_days" in request.source_filter:
            policy_kwargs["max_age_days"] = int(request.source_filter["max_age_days"])
        if "min_content_chars" in request.source_filter:
            policy_kwargs["min_content_chars"] = int(request.source_filter["min_content_chars"])
        if "require_url" in request.source_filter:
            policy_kwargs["require_url"] = bool(request.source_filter["require_url"])
        if "dedupe_threshold" in request.source_filter:
            policy_kwargs["dedupe_threshold"] = float(request.source_filter["dedupe_threshold"])
        if "source_types" in request.source_filter:
            policy_kwargs["source_types"] = tuple(request.source_filter["source_types"])

        selector = SourceSelectorAgent(SourceSelectionPolicy(**policy_kwargs))
        return selector.select(docs, limit=max(request.limit, len(docs)))

    async def _collect_docs(self, request: SkillGenerationRequest) -> list[SourceDocument]:
        if request.task_type == TaskType.error_to_training_set or request.skill_ids == ["error_to_training_samples"]:
            return []
        if request.corpus_jsonl_path:
            docs = self.repo.load_corpus_jsonl(request.corpus_jsonl_path)
            if request.topic:
                topic = request.topic.lower()
                docs = [doc for doc in docs if topic in doc.title.lower() or topic in doc.content.lower()]
            return docs[: request.limit]
        if not request.topic:
            return self.repo.list_documents(limit=request.limit)
        return await self.source_agent.collect(request.topic, limit=request.limit)
