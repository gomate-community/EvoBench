from __future__ import annotations

from benchmark.adapters.llm import LLMAdapter
from benchmark.agents.base import AgentRunConfig
from benchmark.agents.skills import SkillContext, SkillRegistry
from benchmark.schemas import ErrorSample, SkillGenerationRequest, SourceDocument, TaskType, UnifiedSample


class SampleFactoryAgent:
    """Orchestrates configurable skills for heterogeneous sample generation tasks."""

    name = "sample_factory_agent"

    def __init__(self, registry: SkillRegistry | None = None, llm: LLMAdapter | None = None):
        self.registry = registry or SkillRegistry()
        self.llm = llm

    async def generate(
        self,
        request: SkillGenerationRequest,
        *,
        documents: list[SourceDocument] | None = None,
        error_samples: list[ErrorSample] | None = None,
    ) -> list[UnifiedSample]:
        docs = documents if documents is not None else request.documents
        errors = error_samples if error_samples is not None else request.error_samples
        context = SkillContext(
            config=AgentRunConfig(language=request.language, domain=request.domain),
            skill_config=request.skill_config,
            topic=request.topic,
            llm=self.llm,
        )
        skills = self._resolve_skills(request, context)
        samples: list[UnifiedSample] = []
        remaining = request.limit
        for skill in skills:
            if remaining <= 0:
                break
            produced = await skill.generate(documents=docs, error_samples=errors, limit=remaining)
            samples.extend(produced)
            remaining = request.limit - len(samples)
        return samples[: request.limit]

    def _resolve_skills(self, request: SkillGenerationRequest, context: SkillContext):
        if request.skill_ids:
            return [self.registry.create(skill_id, context=context) for skill_id in request.skill_ids]
        if request.task_type:
            return self.registry.create_for_task(request.task_type, context=context)
        # Default to the most common document QA skill when task is omitted.
        return [self.registry.create("doc_to_qa", context=context)]
