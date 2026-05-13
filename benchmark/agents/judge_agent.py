from __future__ import annotations

from uuid import uuid4

from benchmark.agents.base import AgentBase
from benchmark.schemas import AnswerType, BattleResult, BenchmarkItem, ModelResponse


class JudgeAgent(AgentBase):
    """Rubric-based pairwise judge with deterministic fallback scoring.

    This judge is suitable for local development and as a guardrail around LLM judges.
    Production deployments can subclass it and override `_score_single` with a model
    call, while preserving the same rubric dimensions and bias controls.
    """

    name = "judge_agent"

    def __init__(self, judge_id: str = "rubric_judge_v2"):
        self.judge_id = judge_id

    async def judge_pair(self, item: BenchmarkItem, a: ModelResponse, b: ModelResponse) -> BattleResult:
        score_a, dim_a = self._score_single(item, a)
        score_b, dim_b = self._score_single(item, b)
        delta = score_a - score_b
        if abs(delta) < 0.05:
            winner = "tie"
        elif delta > 0:
            winner = "a"
        else:
            winner = "b"
        return BattleResult(
            battle_id=f"battle_{uuid4().hex[:10]}",
            question_id=item.question_id,
            model_a=a.model_id,
            model_b=b.model_id,
            winner=winner,
            judge_id=self.judge_id,
            rubric_scores={
                "a_total": round(score_a, 4),
                "b_total": round(score_b, 4),
                "a_answer_accuracy": dim_a["answer_accuracy"],
                "b_answer_accuracy": dim_b["answer_accuracy"],
                "a_evidence_faithfulness": dim_a["evidence_faithfulness"],
                "b_evidence_faithfulness": dim_b["evidence_faithfulness"],
                "a_uncertainty_handling": dim_a["uncertainty_handling"],
                "b_uncertainty_handling": dim_b["uncertainty_handling"],
            },
            explanation=(
                "Rubric Judge v2: 综合答案正确性、证据忠实度、结构合规、拒答/不确定性处理进行成对比较。"
            ),
        )

    def _score_single(self, item: BenchmarkItem, response: ModelResponse) -> tuple[float, dict[str, float]]:
        answer_accuracy = self._answer_accuracy(item, response)
        evidence_faithfulness = self._evidence_faithfulness(item, response)
        structure = self._structure_compliance(item, response)
        uncertainty = self._uncertainty_handling(item, response)
        weights = item.rubric or {
            "answer_accuracy": 0.45,
            "evidence_faithfulness": 0.30,
            "structure_compliance": 0.15,
            "uncertainty_handling": 0.10,
        }
        # Normalize arbitrary rubric labels to default dimensions when necessary.
        if not any(k in weights for k in ["answer_accuracy", "evidence_faithfulness", "structure_compliance", "uncertainty_handling"]):
            weights = {
                "answer_accuracy": 0.45,
                "evidence_faithfulness": 0.30,
                "structure_compliance": 0.15,
                "uncertainty_handling": 0.10,
            }
        score = (
            weights.get("answer_accuracy", weights.get("label_correctness", 0.45)) * answer_accuracy
            + weights.get("evidence_faithfulness", weights.get("evidence_support", 0.30)) * evidence_faithfulness
            + weights.get("structure_compliance", 0.15) * structure
            + weights.get("uncertainty_handling", weights.get("uncertainty_expression", 0.10)) * uncertainty
        )
        total_weight = sum(
            [
                weights.get("answer_accuracy", weights.get("label_correctness", 0.45)),
                weights.get("evidence_faithfulness", weights.get("evidence_support", 0.30)),
                weights.get("structure_compliance", 0.15),
                weights.get("uncertainty_handling", weights.get("uncertainty_expression", 0.10)),
            ]
        )
        score = score / total_weight if total_weight else 0.0
        return max(0.0, min(1.0, score)), {
            "answer_accuracy": round(answer_accuracy, 4),
            "evidence_faithfulness": round(evidence_faithfulness, 4),
            "structure_compliance": round(structure, 4),
            "uncertainty_handling": round(uncertainty, 4),
        }

    def _answer_accuracy(self, item: BenchmarkItem, response: ModelResponse) -> float:
        gold = item.answer
        pred = response.answer
        if isinstance(gold, dict) and isinstance(pred, dict):
            if not gold:
                return 0.0
            matched = sum(1 for k, v in gold.items() if k in pred and self._soft_equal(v, pred[k]))
            return matched / len(gold)
        if isinstance(gold, list) and isinstance(pred, list):
            if not gold:
                return 0.0
            return len(set(map(str, gold)) & set(map(str, pred))) / len(set(map(str, gold)))
        return 1.0 if self._soft_equal(gold, pred) else max(0.0, self.lexical_overlap(str(gold), str(pred)))

    def _evidence_faithfulness(self, item: BenchmarkItem, response: ModelResponse) -> float:
        if not item.evidence:
            return 0.0
        used = response.evidence_used or []
        pred_text = str(response.answer)
        scores = []
        for ev in item.evidence:
            direct = 1.0 if ev in used or ev in pred_text else 0.0
            overlap = max([self.lexical_overlap(ev, str(u)) for u in used] or [0.0])
            scores.append(max(direct, overlap))
        return max(scores) if scores else 0.0

    def _structure_compliance(self, item: BenchmarkItem, response: ModelResponse) -> float:
        if item.answer_type in {AnswerType.structured_json, AnswerType.rubric}:
            return 1.0 if isinstance(response.answer, dict) else 0.2
        if item.answer_type == AnswerType.source_selection:
            return 1.0 if isinstance(response.answer, str) else 0.3
        return 1.0

    def _uncertainty_handling(self, item: BenchmarkItem, response: ModelResponse) -> float:
        text = str(response.answer).lower() + " " + (response.reasoning_summary or "").lower()
        uncertainty_terms = ["insufficient", "uncertain", "unknown", "证据不足", "无法判断", "不确定"]
        if item.answer_type == AnswerType.abstain:
            return 1.0 if any(t in text for t in uncertainty_terms) else 0.0
        if item.ambiguity_risk >= 0.3:
            return 1.0 if any(t in text for t in uncertainty_terms) else 0.4
        return 1.0

    def _soft_equal(self, a, b) -> bool:
        return str(a).strip().lower() == str(b).strip().lower() or str(a).strip() in str(b).strip()
