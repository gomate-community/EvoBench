from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.agents.skills.base import SkillBase, SkillContext
from benchmark.agents.skills.doc_to_answer import DocumentToAnswerSkill
from benchmark.agents.skills.doc_to_qa import DocumentToQASkill
from benchmark.agents.skills.doc_to_qa_steps import DocumentToQAStepsSkill
from benchmark.agents.skills.doc_to_question import DocumentToQuestionSkill
from benchmark.agents.skills.error_to_training_samples import ErrorToTrainingSamplesSkill
from benchmark.agents.skills.paper_to_experience import PaperToExperienceSkill
from benchmark.schemas import (
    DOC_TO_ANSWER_OUTPUT_SCHEMA,
    DOC_TO_QA_OUTPUT_SCHEMA,
    DOC_TO_QA_STEPS_OUTPUT_SCHEMA,
    DOC_TO_QUESTION_OUTPUT_SCHEMA,
    ERROR_TO_TRAINING_OUTPUT_SCHEMAS,
    PAPER_TO_EXPERIENCE_OUTPUT_SCHEMA,
    SkillDefinition,
    TaskType,
)


class SkillRegistry:
    """Registry mapping skill definitions to executable skill classes."""

    _builtin_output_schemas: dict[str, dict[str, Any]] = {
        "doc_to_question": DOC_TO_QUESTION_OUTPUT_SCHEMA,
        "doc_to_answer": DOC_TO_ANSWER_OUTPUT_SCHEMA,
        "doc_to_qa": DOC_TO_QA_OUTPUT_SCHEMA,
        "doc_to_qa_steps": DOC_TO_QA_STEPS_OUTPUT_SCHEMA,
        "paper_to_experience": PAPER_TO_EXPERIENCE_OUTPUT_SCHEMA,
        "error_to_training_samples": ERROR_TO_TRAINING_OUTPUT_SCHEMAS,
    }

    _factories: dict[str, type[SkillBase]] = {
        "doc_to_question": DocumentToQuestionSkill,
        "doc_to_answer": DocumentToAnswerSkill,
        "doc_to_qa": DocumentToQASkill,
        "doc_to_qa_steps": DocumentToQAStepsSkill,
        "paper_to_experience": PaperToExperienceSkill,
        "error_to_training_samples": ErrorToTrainingSamplesSkill,
    }

    def __init__(self, definitions: list[SkillDefinition] | None = None):
        normalized = [self._normalize_definition(definition) for definition in (definitions or self.default_definitions()) if definition.enabled]
        self.definitions = {definition.skill_id: definition for definition in normalized}

    @classmethod
    def _normalize_definition(cls, definition: SkillDefinition) -> SkillDefinition:
        output_schema = cls._builtin_output_schemas.get(definition.skill_id)
        if output_schema is None:
            return definition
        return definition.model_copy(deep=True, update={"output_schema": output_schema})

    @classmethod
    def default_definitions(cls) -> list[SkillDefinition]:
        return [
            SkillDefinition(
                skill_id="doc_to_question",
                name="Document to Question",
                task_type=TaskType.document_to_x,
                description="Generate document-grounded questions.",
                output_schema=DOC_TO_QUESTION_OUTPUT_SCHEMA,
                quality_rules={"min_evidence_coverage": 0.35},
                config={"sentences_per_doc": 1},
                tags=["d_to_x", "question_generation"],
            ),
            SkillDefinition(
                skill_id="doc_to_answer",
                name="Document to Answer",
                task_type=TaskType.document_to_x,
                description="Generate document-grounded answers or summary spans.",
                output_schema=DOC_TO_ANSWER_OUTPUT_SCHEMA,
                quality_rules={"min_evidence_coverage": 0.35},
                config={"sentences_per_doc": 1},
                tags=["d_to_x", "answer_generation"],
            ),
            SkillDefinition(
                skill_id="doc_to_qa",
                name="Document to QA Pair",
                task_type=TaskType.document_to_xy,
                description="Generate document-grounded question and answer pairs.",
                output_schema=DOC_TO_QA_OUTPUT_SCHEMA,
                quality_rules={"min_evidence_coverage": 0.4},
                config={"pairs_per_doc": 2},
                tags=["d_to_xy", "qa"],
            ),
            SkillDefinition(
                skill_id="doc_to_qa_steps",
                name="Document to QA with Solution Steps",
                task_type=TaskType.document_to_xty,
                description="Generate document-grounded QA samples with concise externalized solution steps.",
                output_schema=DOC_TO_QA_STEPS_OUTPUT_SCHEMA,
                quality_rules={"min_evidence_coverage": 0.4, "human_review_required": True},
                config={"items_per_doc": 2},
                tags=["d_to_xty", "solution_steps"],
            ),
            SkillDefinition(
                skill_id="paper_to_experience",
                name="Paper to Experience",
                task_type=TaskType.document_to_xy,
                description="Extract fact, strategy, mechanism, boundary, and failure experience from academic papers.",
                output_schema=PAPER_TO_EXPERIENCE_OUTPUT_SCHEMA,
                quality_rules={"min_evidence_coverage": 0.45, "human_review_required": True},
                config={
                    "experiences_per_doc": 3,
                    "experience_types": ["fact", "strategy", "mechanism", "boundary", "failure"],
                    "strict_paper_only": False,
                },
                tags=["d_to_xy", "paper_experience", "knowledge_transfer"],
            ),
            SkillDefinition(
                skill_id="error_to_training_samples",
                name="Error Sample Augmentation",
                task_type=TaskType.error_to_training_set,
                description="Generate corrected, contrastive, and boundary training samples from error cases.",
                output_schema=ERROR_TO_TRAINING_OUTPUT_SCHEMAS,
                quality_rules={"human_review_required": True},
                config={"include_contrastive": True, "include_boundary": True},
                tags=["error_aug", "training_data"],
            ),
        ]

    @classmethod
    def from_file(cls, path: str | Path) -> "SkillRegistry":
        p = Path(path)
        if not p.exists():
            return cls()
        data = cls._load_config(p)
        raw_defs = data.get("skills", data if isinstance(data, list) else [])
        definitions = [SkillDefinition.model_validate(item) for item in raw_defs]
        return cls(definitions)

    @staticmethod
    def _load_config(path: Path) -> Any:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text) or {}
        except Exception:
            return {}

    def list_definitions(self, task_type: TaskType | None = None) -> list[SkillDefinition]:
        definitions = list(self.definitions.values())
        if task_type:
            definitions = [definition for definition in definitions if definition.task_type == task_type]
        return definitions

    def get_definition(self, skill_id: str) -> SkillDefinition:
        if skill_id not in self.definitions:
            available = ", ".join(sorted(self.definitions))
            raise KeyError(f"Unknown skill_id={skill_id}. Available skills: {available}")
        return self.definitions[skill_id]

    def create(self, skill_id: str, context: SkillContext | None = None) -> SkillBase:
        definition = self.get_definition(skill_id)
        if skill_id not in self._factories:
            available = ", ".join(sorted(self._factories))
            raise KeyError(f"Unregistered skill_id={skill_id}. Registered skills: {available}")
        return self._factories[skill_id](definition, context=context)

    def create_for_task(self, task_type: TaskType, context: SkillContext | None = None) -> list[SkillBase]:
        return [self.create(definition.skill_id, context=context) for definition in self.list_definitions(task_type)]
