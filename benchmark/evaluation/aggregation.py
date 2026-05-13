from benchmark.schemas import ScoreResult


def aggregate_scores(scores: list[ScoreResult]) -> dict[str, float]:
    if not scores:
        return {"verifiable_score": 0.0, "reliability_score": 0.0}
    avg = sum(s.score for s in scores) / len(scores)
    evidence = sum(s.dimensions.get("evidence_faithfulness", 0.0) for s in scores) / len(scores)
    return {"verifiable_score": avg * 100, "reliability_score": evidence * 100}


def final_score(verifiable_score: float, arena_score: float, reliability_score: float, leakage_penalty: float = 0.0) -> float:
    return max(0.0, 0.50 * verifiable_score + 0.25 * arena_score + 0.20 * reliability_score - leakage_penalty)
