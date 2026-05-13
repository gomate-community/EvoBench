from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from benchmark.adapters.llm import LLMAdapter
from benchmark.agents.base import AgentBase, AgentRunConfig
from benchmark.schemas import ErrorSample, SkillDefinition, SourceDocument, UnifiedSample


@dataclass
class SkillContext:
    config: AgentRunConfig = field(default_factory=AgentRunConfig)
    skill_config: dict[str, Any] = field(default_factory=dict)
    topic: str | None = None
    llm: LLMAdapter | None = None


class SkillBase(AgentBase):
    """Base class for standalone sample generation skills."""

    definition: SkillDefinition

    def __init__(self, definition: SkillDefinition, context: SkillContext | None = None):
        self.definition = definition
        self.context = context or SkillContext()
        self.name = definition.skill_id
        if self.context.llm is not None:
            self._llm = self.context.llm

    async def generate(
        self,
        *,
        documents: list[SourceDocument] | None = None,
        error_samples: list[ErrorSample] | None = None,
        limit: int = 10,
    ) -> list[UnifiedSample]:
        raise NotImplementedError

    def _merged_config(self) -> dict[str, Any]:
        merged = dict(self.definition.config)
        merged.update(self.context.skill_config)
        return merged


class SkillFactory(Protocol):
    def __call__(self, definition: SkillDefinition, context: SkillContext | None = None) -> SkillBase: ...
