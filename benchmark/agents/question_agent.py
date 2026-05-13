from __future__ import annotations

from itertools import islice

from benchmark.adapters.llm import LLMAdapter
from benchmark.agents.base import AgentBase, AgentRunConfig
from benchmark.schemas import (
    AnnotationGuideline,
    AnswerType,
    BenchmarkItem,
    Claim,
    QualitySignals,
    SampleType,
    SourceDocument,
    VerificationMethod,
)


class QuestionAgent(AgentBase):
    """Generate multiple high-quality annotation sample types from evidence-linked claims."""

    name = "question_agent"

    def __init__(self, config: AgentRunConfig | None = None, llm: LLMAdapter | None = None):
        self.config = config or AgentRunConfig()
        self._llm = llm

    async def generate(
        self,
        claims: list[Claim],
        skill_id: str = "fact_verification",
        docs: list[SourceDocument] | None = None,
        sample_types: list[SampleType] | None = None,
    ) -> list[BenchmarkItem]:
        docs = docs or []
        if sample_types is None:
            try:
                sample_types = [SampleType(skill_id)]
            except ValueError:
                sample_types = [SampleType.fact_verification]

        doc_map = {d.source_id: d for d in docs}
        items: list[BenchmarkItem] = []
        for claim in claims:
            llm_overrides = await self._question_overrides_with_llm(claim, sample_types, doc_map, docs)
            for sample_type in sample_types:
                built = self._build_item(sample_type, claim, doc_map, docs)
                if built is not None:
                    built = self._apply_llm_override(built, llm_overrides.get(sample_type.value))
                    items.append(built)
        return self._dedupe_items(items)

    async def _question_overrides_with_llm(
        self,
        claim: Claim,
        sample_types: list[SampleType],
        doc_map: dict[str, SourceDocument],
        docs: list[SourceDocument],
    ) -> dict[str, dict]:
        if not sample_types:
            return {}
        source_doc = doc_map.get(claim.source_id)
        related_docs = [doc for doc in docs if doc.source_id != claim.source_id][:2]
        payload = await self.llm_complete_json(
            self._question_override_prompt(claim, sample_types, source_doc, related_docs),
            system=(
                "你是 benchmark 样本生成助手。"
                "请为每种 sample_type 输出更自然、更可验证的问题表达。"
                '返回 JSON：{"items":{"sample_type":{"question":"...","instruction":"..."}}}'
            ),
            temperature=0.2,
            max_tokens=1600,
            fallback={"items": {}},
        )
        items = payload.get("items", {})
        return items if isinstance(items, dict) else {}

    def _question_override_prompt(
        self,
        claim: Claim,
        sample_types: list[SampleType],
        source_doc: SourceDocument | None,
        related_docs: list[SourceDocument],
    ) -> str:
        source_summary = (
            f"主来源：{source_doc.title}\n"
            f"主来源内容摘要：{source_doc.content[:800]}\n"
            if source_doc
            else "主来源：unknown\n"
        )
        related_summary = "\n".join(
            f"- {doc.source_id}: {doc.title} | {doc.content[:240]}" for doc in related_docs
        )
        return (
            f"claim: {claim.text}\n"
            f"evidence: {' | '.join(span.text for span in claim.evidence_spans[:2])}\n"
            f"sample_types: {[sample_type.value for sample_type in sample_types]}\n\n"
            f"{source_summary}"
            f"相关来源：\n{related_summary or '无'}\n\n"
            "请为每个 sample_type 提供：\n"
            "1. question: 更自然、更清晰、适合评测的问题\n"
            "2. instruction: 一句简短标注或回答要求\n"
            "不要编造新事实，不要泄露答案。"
        )

    def _apply_llm_override(self, item: BenchmarkItem, override: dict | None) -> BenchmarkItem:
        if not isinstance(override, dict):
            return item
        question = self.normalize_text(str(override.get("question", "")))
        instruction = self.normalize_text(str(override.get("instruction", "")))
        if question:
            item.question = question[: self.config.max_question_chars]
        if instruction:
            item.instruction = instruction
        if question or instruction:
            item.metadata = {**item.metadata, "question_generator": "llm_assisted"}
        return item

    def _build_item(
        self,
        sample_type: SampleType,
        claim: Claim,
        doc_map: dict[str, SourceDocument],
        docs: list[SourceDocument],
    ) -> BenchmarkItem | None:
        builders = {
            SampleType.fact_verification: self._fact_verification,
            SampleType.temporal_awareness: self._temporal_awareness,
            SampleType.source_attribution: self._source_attribution,
            SampleType.evidence_selection: self._evidence_selection,
            SampleType.conflict_resolution: self._conflict_resolution,
            SampleType.data_value_judgement: self._data_value_judgement,
            SampleType.abstention: self._abstention,
            SampleType.causal_attribution: self._causal_attribution,
        }
        return builders[sample_type](claim, doc_map, docs)

    def _base_item(
        self,
        sample_type: SampleType,
        claim: Claim,
        question: str,
        answer,
        answer_type: AnswerType,
        verification_method: VerificationMethod,
        *,
        options: list[str] | None = None,
        rubric: dict | None = None,
        split: str = "fresh",
        difficulty: float = 0.5,
        ambiguity: float = 0.1,
        leakage: float = 0.1,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        guideline: AnnotationGuideline | None = None,
    ) -> BenchmarkItem:
        evidence_texts = [span.text for span in claim.evidence_spans] or [claim.text]
        evidence_ids = [span.evidence_id for span in claim.evidence_spans]
        return BenchmarkItem(
            question_id=self.make_id("q", sample_type.value, claim.claim_id, question),
            skill_id=sample_type.value,
            sample_type=sample_type,
            domain=self.config.domain,
            instruction=self._instruction_for(sample_type),
            question=question[: self.config.max_question_chars],
            answer=answer,
            answer_type=answer_type,
            options=options or [],
            evidence=evidence_texts,
            expected_evidence_ids=evidence_ids,
            verification_method=verification_method,
            source_ids=[claim.source_id],
            freshness_window_days=30 if sample_type in {SampleType.temporal_awareness, SampleType.fact_verification} else None,
            difficulty_estimate=difficulty,
            leakage_risk=leakage,
            ambiguity_risk=ambiguity,
            split=split,  # type: ignore[arg-type]
            rubric=rubric or {},
            annotation_guideline=guideline or self._guideline_for(sample_type),
            quality_signals=QualitySignals(
                source_support_count=1,
                evidence_coverage=1.0 if evidence_texts else 0.0,
                answerability=1.0,
                clarity=max(0.2, 1.0 - ambiguity),
                novelty=max(0.2, 1.0 - leakage),
            ),
            tags=tags or [sample_type.value, claim.metadata.get("source_type", "source")],
            metadata={"claim_id": claim.claim_id, **(metadata or {})},
        )

    def _fact_verification(self, claim: Claim, doc_map: dict[str, SourceDocument], docs: list[SourceDocument]) -> BenchmarkItem:
        question = f"判断以下陈述是否被给定证据支持，并给出最小证据句。\n陈述：{claim.text}"
        return self._base_item(
            SampleType.fact_verification,
            claim,
            question,
            {"label": "supported", "claim": claim.text},
            AnswerType.structured_json,
            VerificationMethod.evidence_overlap,
            rubric={
                "label_correctness": 0.5,
                "evidence_support": 0.3,
                "no_overclaiming": 0.2,
            },
            difficulty=0.35,
        )

    def _temporal_awareness(self, claim: Claim, doc_map: dict[str, SourceDocument], docs: list[SourceDocument]) -> BenchmarkItem:
        doc = doc_map.get(claim.source_id)
        published = doc.published_at.isoformat() if doc and doc.published_at else "unknown"
        question = (
            "请根据证据判断该陈述的时间属性。输出 current / historical / unknown 之一，"
            f"并说明依据。\n陈述：{claim.text}\n证据发布时间：{published}"
        )
        label = "current" if claim.event_time else "unknown"
        return self._base_item(
            SampleType.temporal_awareness,
            claim,
            question,
            {"temporal_label": label, "event_time": claim.event_time.isoformat() if claim.event_time else None},
            AnswerType.structured_json,
            VerificationMethod.multi_source,
            rubric={"time_label": 0.5, "event_time_grounding": 0.3, "uncertainty": 0.2},
            difficulty=0.5,
            tags=["temporal", "freshness"],
        )

    def _source_attribution(
        self, claim: Claim, doc_map: dict[str, SourceDocument], docs: list[SourceDocument]
    ) -> BenchmarkItem | None:
        positive = doc_map.get(claim.source_id)
        if not positive:
            return None
        distractors = [d for d in docs if d.source_id != claim.source_id]
        selected = [positive, *list(islice(distractors, 3))]
        if len(selected) < 2:
            # Still useful as direct source attribution, but less discriminative.
            selected = [positive]
        options = [f"{d.source_id} | {d.publisher or 'unknown'} | {d.title}" for d in selected]
        question = f"以下陈述最直接由哪个来源支持？\n陈述：{claim.text}\n候选来源：\n" + "\n".join(options)
        return self._base_item(
            SampleType.source_attribution,
            claim,
            question,
            claim.source_id,
            AnswerType.source_selection,
            VerificationMethod.exact_match,
            options=options,
            difficulty=0.45 + min(0.25, 0.05 * len(options)),
            tags=["attribution", "source_selection"],
        )

    def _evidence_selection(self, claim: Claim, doc_map: dict[str, SourceDocument], docs: list[SourceDocument]) -> BenchmarkItem:
        doc = doc_map.get(claim.source_id)
        candidate_sentences = self.split_sentences(doc.content)[:6] if doc else [claim.text]
        if claim.text not in candidate_sentences:
            candidate_sentences = [claim.text, *candidate_sentences]
        question = (
            "从候选证据句中选择最能支持陈述的一句。\n"
            f"陈述：{claim.text}\n候选证据：\n"
            + "\n".join(f"[{i}] {s}" for i, s in enumerate(candidate_sentences))
        )
        return self._base_item(
            SampleType.evidence_selection,
            claim,
            question,
            claim.text,
            AnswerType.evidence_based,
            VerificationMethod.evidence_overlap,
            options=candidate_sentences,
            difficulty=0.4 + min(0.3, 0.03 * len(candidate_sentences)),
            tags=["evidence", "span_selection"],
        )

    def _conflict_resolution(
        self, claim: Claim, doc_map: dict[str, SourceDocument], docs: list[SourceDocument]
    ) -> BenchmarkItem:
        # Without a true contradictory source, generate a safe conflict-check sample that
        # expects the model to avoid fabricating contradictions.
        related = [d for d in docs if d.source_id != claim.source_id and self.lexical_overlap(d.content, claim.text) > 0.05]
        evidence_bundle = [claim.text] + [self.split_sentences(d.content)[0] for d in related[:2] if self.split_sentences(d.content)]
        question = (
            "判断证据之间是否存在实质性冲突。若无冲突，输出 no_conflict；若有冲突，"
            "指出冲突点和更可信来源。\n证据：\n"
            + "\n".join(f"- {e}" for e in evidence_bundle)
        )
        answer = {"conflict_label": "no_conflict", "supported_claim": claim.text, "uncertainty": "low"}
        return self._base_item(
            SampleType.conflict_resolution,
            claim,
            question,
            answer,
            AnswerType.structured_json,
            VerificationMethod.contradiction_check,
            rubric={"conflict_detection": 0.4, "source_reliability": 0.3, "uncertainty_expression": 0.3},
            split="adversarial",
            difficulty=0.65,
            ambiguity=0.25 if len(evidence_bundle) == 1 else 0.15,
            tags=["conflict", "adversarial"],
        )

    def _data_value_judgement(self, claim: Claim, doc_map: dict[str, SourceDocument], docs: list[SourceDocument]) -> BenchmarkItem:
        question = (
            "请评估以下信息是否属于高价值数据样本，并从时效性、稀缺性、可验证性、"
            f"决策影响四个维度给出 1-5 分。\n信息：{claim.text}"
        )
        answer = {
            "value_label": "medium_high" if claim.source_trust_level >= 3 else "medium",
            "scores": {
                "freshness": 4 if claim.event_time else 3,
                "rarity": 3,
                "verifiability": min(5, claim.source_trust_level + 1),
                "decision_impact": 3,
            },
        }
        return self._base_item(
            SampleType.data_value_judgement,
            claim,
            question,
            answer,
            AnswerType.rubric,
            VerificationMethod.rubric,
            rubric={"freshness": 0.25, "rarity": 0.25, "verifiability": 0.25, "decision_impact": 0.25},
            difficulty=0.7,
            ambiguity=0.3,
            tags=["data_value", "rubric"],
        )

    def _abstention(self, claim: Claim, doc_map: dict[str, SourceDocument], docs: list[SourceDocument]) -> BenchmarkItem:
        unsupported_claim = f"{claim.text}，且该变化已经被所有主要来源确认。"
        question = (
            "仅根据给定证据回答陈述是否成立。若证据不足，必须输出 insufficient_evidence。\n"
            f"陈述：{unsupported_claim}\n给定证据：{claim.text}"
        )
        return self._base_item(
            SampleType.abstention,
            claim,
            question,
            {"label": "insufficient_evidence", "reason": "原证据未支持‘所有主要来源确认’这一外推结论"},
            AnswerType.abstain,
            VerificationMethod.evidence_overlap,
            split="adversarial",
            difficulty=0.6,
            ambiguity=0.15,
            tags=["abstention", "overclaiming", "adversarial"],
            metadata={"unsupported_claim": unsupported_claim},
        )

    def _causal_attribution(self, claim: Claim, doc_map: dict[str, SourceDocument], docs: list[SourceDocument]) -> BenchmarkItem:
        question = (
            "判断以下陈述是否包含明确因果归因。若只存在相关性或宣传性说法，"
            f"请标注为 correlation_or_claim。\n陈述：{claim.text}"
        )
        causal_markers = ["导致", "因为", "由于", "使得", "推动", "归因于"]
        label = "causal_claim" if any(m in claim.text for m in causal_markers) else "correlation_or_claim"
        return self._base_item(
            SampleType.causal_attribution,
            claim,
            question,
            {"causal_label": label},
            AnswerType.structured_json,
            VerificationMethod.rubric,
            difficulty=0.55,
            tags=["causal", "attribution"],
        )

    def _instruction_for(self, sample_type: SampleType) -> str:
        instructions = {
            SampleType.fact_verification: "基于证据判断陈述是否被支持，不允许使用外部知识补全。",
            SampleType.temporal_awareness: "识别事件时间、发布时间和答案时效性，避免把过时信息当作当前事实。",
            SampleType.source_attribution: "选择最直接支持陈述的来源，并保留来源 ID。",
            SampleType.evidence_selection: "选择最小充分证据，不要选择泛泛背景句。",
            SampleType.conflict_resolution: "比较多条证据是否冲突，必要时表达不确定性。",
            SampleType.data_value_judgement: "按 rubric 判断数据价值，不以语言流畅度替代事实价值。",
            SampleType.abstention: "证据不足时应拒答或标注 insufficient_evidence。",
            SampleType.causal_attribution: "区分因果归因、相关性和宣传性声明。",
        }
        return instructions[sample_type]

    def _guideline_for(self, sample_type: SampleType) -> AnnotationGuideline:
        if sample_type == SampleType.data_value_judgement:
            return AnnotationGuideline(
                label_schema={"value_label": ["low", "medium", "medium_high", "high"], "scores": "1-5 by dimension"},
                positive_criteria=["评分必须引用证据中的事实属性", "高价值判断需说明决策影响或稀缺性"],
                negative_criteria=["不得仅因题材热门就判为高价值", "不得忽略来源可信度"],
                edge_cases=["宣传稿但缺少独立验证时，可验证性应降分"],
                human_review_required=True,
                review_priority="high",
            )
        if sample_type in {SampleType.conflict_resolution, SampleType.abstention}:
            return AnnotationGuideline(
                label_schema={"label": "task-specific structured JSON"},
                positive_criteria=["答案必须被给定证据支持", "证据不足时不得猜测"],
                negative_criteria=["不得引入题外来源", "不得把单一来源外推为多源共识"],
                edge_cases=["信息源之间只是表述不同，不一定构成事实冲突"],
                human_review_required=True,
                review_priority="medium",
            )
        return AnnotationGuideline(
            label_schema={"answer": "task-specific"},
            positive_criteria=["答案与证据一致", "证据跨度最小充分"],
            negative_criteria=["答案不能超出证据", "不能混淆发布时间与事件时间"],
            edge_cases=[],
            human_review_required=False,
            review_priority="low",
        )

    def _dedupe_items(self, items: list[BenchmarkItem]) -> list[BenchmarkItem]:
        kept: list[BenchmarkItem] = []
        for item in items:
            if any(self.lexical_overlap(item.question, old.question) > 0.95 for old in kept):
                continue
            kept.append(item)
        return kept
