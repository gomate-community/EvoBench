from __future__ import annotations

import re
from typing import Any

from benchmark.schemas import AnswerType, BenchmarkItem, ModelResponse, ScoreResult, VerificationMethod


class ObjectiveScorer:
    def score(self, item: BenchmarkItem, response: ModelResponse) -> ScoreResult:
        answer_score = self._answer_score(item, response.answer)
        evidence_score = self._evidence_score(item, response)
        structure_score = self._structure_score(item, response.answer)
        uncertainty_score = self._uncertainty_score(item, response)
        final = 0.55 * answer_score + 0.25 * evidence_score + 0.10 * structure_score + 0.10 * uncertainty_score
        return ScoreResult(
            question_id=item.question_id,
            model_id=response.model_id,
            score=max(0.0, min(1.0, final)),
            dimensions={
                "answer_accuracy": answer_score,
                "evidence_faithfulness": evidence_score,
                "structure_compliance": structure_score,
                "uncertainty_handling": uncertainty_score,
            },
            explanation="自动评分：答案 55%，证据 25%，结构 10%，不确定性/拒答 10%。",
        )

    def _answer_score(self, item: BenchmarkItem, pred: Any) -> float:
        gold = item.answer
        if item.verification_method == VerificationMethod.regex:
            return float(bool(re.search(str(gold), str(pred))))
        if isinstance(gold, dict) and isinstance(pred, dict):
            if not gold:
                return 0.0
            matched = sum(1 for k, v in gold.items() if k in pred and self._soft_equal(v, pred[k]))
            return matched / len(gold)
        if isinstance(gold, list) and isinstance(pred, list):
            gold_set = set(map(str, gold))
            pred_set = set(map(str, pred))
            return len(gold_set & pred_set) / max(1, len(gold_set))
        return float(self._soft_equal(gold, pred) or str(gold).strip() in str(pred).strip())

    def _evidence_score(self, item: BenchmarkItem, response: ModelResponse) -> float:
        if not item.evidence:
            return 0.0
        haystack = " ".join(map(str, response.evidence_used)) + " " + str(response.answer)
        return float(any(ev in haystack or self._token_overlap(ev, haystack) > 0.6 for ev in item.evidence))

    def _structure_score(self, item: BenchmarkItem, pred: Any) -> float:
        if item.answer_type in {AnswerType.structured_json, AnswerType.rubric}:
            return 1.0 if isinstance(pred, dict) else 0.2
        if item.answer_type == AnswerType.source_selection:
            return 1.0 if isinstance(pred, str) else 0.2
        return 1.0

    def _uncertainty_score(self, item: BenchmarkItem, response: ModelResponse) -> float:
        text = (str(response.answer) + " " + (response.reasoning_summary or "")).lower()
        terms = ["insufficient", "unknown", "uncertain", "证据不足", "无法判断", "不确定"]
        if item.answer_type == AnswerType.abstain:
            return 1.0 if any(t in text for t in terms) else 0.0
        if item.ambiguity_risk >= 0.3:
            return 1.0 if any(t in text for t in terms) else 0.5
        return 1.0

    def _soft_equal(self, a: Any, b: Any) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            return all(k in b and self._soft_equal(v, b[k]) for k, v in a.items())
        return str(a).strip().lower() == str(b).strip().lower()

    def _token_overlap(self, a: str, b: str) -> float:
        ta = set(re.findall(r"[\w\u4e00-\u9fff]+", (a or "").lower()))
        tb = set(re.findall(r"[\w\u4e00-\u9fff]+", (b or "").lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta)
