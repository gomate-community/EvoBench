from benchmark.evaluation.scorers import ObjectiveScorer
from benchmark.schemas import AnswerType, BenchmarkItem, ModelResponse, VerificationMethod


def test_objective_scorer_exact():
    item = BenchmarkItem(
        question_id="q1",
        skill_id="fact_verification",
        domain="tech",
        question="Q",
        answer="A",
        answer_type=AnswerType.short_text,
        evidence=["E"],
        verification_method=VerificationMethod.exact_match,
    )
    resp = ModelResponse(model_id="m", question_id="q1", answer="A", evidence_used=["E"])
    score = ObjectiveScorer().score(item, resp)
    assert score.score == 1.0
