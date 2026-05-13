from benchmark.agents.judge_agent import JudgeAgent
from benchmark.arena.elo import update_elo
from benchmark.schemas import BenchmarkItem, ModelResponse


class ArenaService:
    def __init__(self, judge: JudgeAgent):
        self.judge = judge

    async def run_battle(self, item: BenchmarkItem, a: ModelResponse, b: ModelResponse, ratings: dict[str, float]) -> dict:
        result = await self.judge.judge_pair(item, a, b)
        result_a = 1.0 if result.winner == "a" else 0.0 if result.winner == "b" else 0.5
        ra, rb = ratings.get(a.model_id, 1000.0), ratings.get(b.model_id, 1000.0)
        new_a, new_b = update_elo(ra, rb, result_a)
        ratings[a.model_id] = new_a
        ratings[b.model_id] = new_b
        return {"battle": result, "ratings": ratings}
