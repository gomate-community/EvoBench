from __future__ import annotations

from benchmark.evaluation.aggregation import aggregate_scores, final_score
from benchmark.evaluation.scorers import ObjectiveScorer
from benchmark.schemas import ModelResponse
from benchmark.storage.repository import BenchmarkRepository


class EvaluationPipeline:
    def __init__(self):
        self.repo = BenchmarkRepository()
        self.scorer = ObjectiveScorer()

    async def evaluate_mock_model(self, model_id: str) -> dict:
        items = self.repo.list_items(status="verified", limit=100)
        scores = []
        for item in items:
            response = ModelResponse(
                model_id=model_id,
                question_id=item.question_id,
                answer=item.answer,
                evidence_used=item.evidence[:1],
                confidence=0.8,
            )
            scores.append(self.scorer.score(item, response))
        agg = aggregate_scores(scores)
        agg["arena_score"] = 50.0
        agg["final_score"] = final_score(agg["verifiable_score"], agg["arena_score"], agg["reliability_score"])
        agg["evaluated_items"] = len(items)
        return agg
