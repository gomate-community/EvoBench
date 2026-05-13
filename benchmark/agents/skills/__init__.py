from benchmark.agents.skills.base import SkillBase, SkillContext
from benchmark.agents.skills.doc_to_answer import DocumentToAnswerSkill
from benchmark.agents.skills.doc_to_qa import DocumentToQASkill
from benchmark.agents.skills.doc_to_qa_steps import DocumentToQAStepsSkill
from benchmark.agents.skills.doc_to_question import DocumentToQuestionSkill
from benchmark.agents.skills.error_to_training_samples import ErrorToTrainingSamplesSkill
from benchmark.agents.skills.paper_to_experience import PaperToExperienceSkill
from benchmark.agents.skills.registry import SkillRegistry

ErrorAugmentationSkill = ErrorToTrainingSamplesSkill

__all__ = [
    "SkillBase",
    "SkillContext",
    "DocumentToQuestionSkill",
    "DocumentToAnswerSkill",
    "DocumentToQASkill",
    "DocumentToQAStepsSkill",
    "PaperToExperienceSkill",
    "ErrorToTrainingSamplesSkill",
    "ErrorAugmentationSkill",
    "SkillRegistry",
]
