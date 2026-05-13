from __future__ import annotations

from benchmark.schemas import SourceDocument

SOURCE_REF_REASON = "selected for paper_to_experience generation"
INSTRUCTION = (
    "Extract transferable experience from the paper. "
    "x is a future problem scenario, and y is a structured experience card."
)

SYSTEM_PROMPT = (
    "You extract reusable experience from academic papers. "
    "Only use the paper content. "
    "If the requested count is larger than the number of experience types, you may return multiple experiences of the same type. "
    'Return JSON in the form: {"experiences":[{"experience_type":"fact|strategy|cognitive","experience_title":"...","experience_statement":"...","future_problem":"...","applicability":"...","actionable_advice":"...","caveats":"...","evidence_text":"..."}]}'
)


def build_experience_prompt(doc: SourceDocument, experience_types: list[str], max_per_doc: int) -> str:
    return (
        f"Paper title: {doc.title}\n"
        f"Source type: {doc.source_type}\n"
        f"Allowed experience types: {', '.join(experience_types)}\n"
        f"Maximum experiences: {max_per_doc}\n\n"
        "Extract reusable experience that can help solve future related problems.\n"
        "Type definitions:\n"
        "1. fact: verified facts, boundary conditions, or quantitative findings.\n"
        "2. strategy: methods, workflows, procedures, or decision strategies.\n"
        "3. cognitive: reasoning frameworks, pitfalls, or cautionary patterns.\n\n"
        "The same experience type may appear more than once if the paper supports multiple distinct experiences.\n"
        "For each experience include future_problem, applicability, actionable_advice, caveats, and evidence_text.\n\n"
        f"Paper content:\n{doc.content[:5000]}"
    )


def future_problem_prompt(experience_type: str, doc: SourceDocument) -> str:
    if experience_type == "fact":
        return f"When facing a future problem similar to '{doc.title}', which validated facts or boundary conditions should be checked first?"
    if experience_type == "strategy":
        return f"When facing a future task similar to '{doc.title}', which strategy or workflow should be tried first?"
    return f"When facing a future problem similar to '{doc.title}', which reasoning mistakes or blind spots should be watched for?"
