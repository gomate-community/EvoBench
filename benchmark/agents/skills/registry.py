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
from benchmark.schemas import SkillDefinition, TaskType


class SkillRegistry:
    """Registry mapping skill definitions to executable skill classes."""

    _factories: dict[str, type[SkillBase]] = {
        "doc_to_question": DocumentToQuestionSkill,
        "doc_to_answer": DocumentToAnswerSkill,
        "doc_to_qa": DocumentToQASkill,
        "doc_to_qa_steps": DocumentToQAStepsSkill,
        "paper_to_experience": PaperToExperienceSkill,
        "error_to_training_samples": ErrorToTrainingSamplesSkill,
    }

    def __init__(self, definitions: list[SkillDefinition] | None = None):
        self.definitions = {definition.skill_id: definition for definition in (definitions or self.default_definitions()) if definition.enabled}

    @classmethod
    def default_definitions(cls) -> list[SkillDefinition]:
        return [
            SkillDefinition(
                skill_id="doc_to_question",
                name="Document to Question",
                task_type=TaskType.document_to_x,
                description="基于文档生成可回答、可验证的问题 x。",
                output_schema={"x": "question"},
                quality_rules={"min_evidence_coverage": 0.35},
                config={"sentences_per_doc": 1},
                tags=["d_to_x", "question_generation"],
            ),
            SkillDefinition(
                skill_id="doc_to_answer",
                name="Document to Answer",
                task_type=TaskType.document_to_x,
                description="基于文档生成答案或摘要片段 x。",
                output_schema={"x": "answer"},
                quality_rules={"min_evidence_coverage": 0.35},
                config={"sentences_per_doc": 1},
                tags=["d_to_x", "answer_generation"],
            ),
            SkillDefinition(
                skill_id="doc_to_qa",
                name="Document to QA Pair",
                task_type=TaskType.document_to_xy,
                description="基于文档生成问题 x 和答案 y。",
                output_schema={"x": "question", "y": "answer"},
                quality_rules={"min_evidence_coverage": 0.4},
                config={"pairs_per_doc": 2},
                tags=["d_to_xy", "qa"],
            ),
            SkillDefinition(
                skill_id="doc_to_qa_steps",
                name="Document to QA with Solution Steps",
                task_type=TaskType.document_to_xty,
                description="基于文档生成问题 x、外显步骤 T 和答案 y。",
                output_schema={"x": "question", "T": "solution_steps", "y": "answer"},
                quality_rules={"min_evidence_coverage": 0.4, "human_review_required": True},
                config={"items_per_doc": 2},
                tags=["d_to_xty", "solution_steps"],
            ),
            SkillDefinition(
                skill_id="paper_to_experience",
                name="Paper to Experience",
                task_type=TaskType.document_to_xy,
                description="基于学术论文抽取可迁移的事实经验、策略经验和认知经验。",
                output_schema={"x": "future_problem", "y": "experience_card"},
                quality_rules={"min_evidence_coverage": 0.45, "human_review_required": True},
                config={
                    "experiences_per_doc": 3,
                    "experience_types": ["fact", "strategy", "cognitive"],
                    "strict_paper_only": False,
                },
                tags=["d_to_xy", "paper_experience", "knowledge_transfer"],
            ),
            SkillDefinition(
                skill_id="error_to_training_samples",
                name="Error Sample Augmentation",
                task_type=TaskType.error_to_training_set,
                description="基于错误样本生成纠错、对比和边界训练样本集合。",
                output_schema={"x": "new_input", "y": "target_or_preference", "T": "optional_steps"},
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
