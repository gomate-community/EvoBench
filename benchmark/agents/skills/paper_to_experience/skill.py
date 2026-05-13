from __future__ import annotations

from typing import Any

from benchmark.agents.skills._document_common import DocumentSkillMixin
from benchmark.agents.skills.paper_to_experience.prompts import (
    INSTRUCTION,
    SOURCE_REF_REASON,
    SYSTEM_PROMPT,
    build_experience_prompt,
    future_problem_prompt,
)
from benchmark.schemas import (
    AnnotationGuideline,
    ArtifactRole,
    ExperienceCard,
    ExperienceType,
    PAPER_TO_EXPERIENCE_OUTPUT_SCHEMA,
    SampleArtifact,
    SampleOutput,
    SourceDocument,
    SourceReference,
    TaskType,
    UnifiedSample,
    VerificationMethod,
)


class PaperToExperienceSkill(DocumentSkillMixin):
    """Extract transferable experience cards from academic papers."""

    DEFAULT_TYPES = tuple(experience_type.value for experience_type in ExperienceType)

    async def generate(self, *, documents: list[SourceDocument] | None = None, error_samples=None, limit: int = 10) -> list[UnifiedSample]:
        docs = documents or []
        cfg = self._merged_config()
        samples: list[UnifiedSample] = []
        max_per_doc = max(1, int(cfg.get("experiences_per_doc", 3)))
        experience_types = self._normalize_types(cfg.get("experience_types"))
        strict_paper_only = bool(cfg.get("strict_paper_only", False))

        for doc in docs:
            if strict_paper_only and not self._looks_like_academic_paper(doc):
                continue

            cards = await self._extract_experience_cards(doc, experience_types, max_per_doc)
            for card_index, card_data in enumerate(cards[:max_per_doc], start=1):
                card = self._build_experience_card(doc, card_data)
                evidence_text = card_data.get("evidence_text") or card.experience_statement or doc.content[:250]
                evidence = self.build_evidence(doc, str(evidence_text))
                experience_type = card.experience_type.value
                future_question = str(card_data.get("future_problem") or self._future_problem_prompt(experience_type, doc))
                answer = card.model_dump(mode="json")
                samples.append(
                    UnifiedSample(
                        sample_id=self.make_id(
                            "sample",
                            self.definition.skill_id,
                            doc.source_id,
                            str(card_index),
                            experience_type,
                            future_question,
                            card.experience_statement,
                        ),
                        task_type=TaskType.document_to_xy,
                        skill_id=self.definition.skill_id,
                        domain=self.context.config.domain,
                        language=self.context.config.language,
                        input=self.base_input(doc, evidence),
                        output=SampleOutput(
                            artifacts=[
                                SampleArtifact(
                                    role=ArtifactRole.question,
                                    key="x",
                                    value=future_question,
                                    evidence_ids=[evidence.evidence_id],
                                ),
                                SampleArtifact(
                                    role=ArtifactRole.answer,
                                    key="y",
                                    value=answer,
                                    evidence_ids=[evidence.evidence_id],
                                ),
                                SampleArtifact(
                                    role=ArtifactRole.label,
                                    key="experience_type",
                                    value=experience_type,
                                    evidence_ids=[evidence.evidence_id],
                                ),
                            ],
                            target_schema=self.definition.output_schema or PAPER_TO_EXPERIENCE_OUTPUT_SCHEMA,
                        ),
                        source_refs=[SourceReference.from_doc(doc, SOURCE_REF_REASON)],
                        evidence=[evidence],
                        instruction=INSTRUCTION,
                        verification_method=VerificationMethod.evidence_overlap,
                        annotation_guideline=self._experience_guideline(),
                        difficulty_estimate=0.62,
                        ambiguity_risk=0.2,
                        tags=["paper_experience", experience_type, doc.source_type],
                        metadata={
                            "topic": self.context.topic,
                            "paper_title": doc.title,
                            "experience_type": experience_type,
                            "source_kind": "academic_paper" if self._looks_like_academic_paper(doc) else doc.source_type,
                        },
                    )
                )
                if len(samples) >= limit:
                    return samples
        return samples

    async def _extract_experience_cards(
        self,
        doc: SourceDocument,
        experience_types: list[str],
        max_per_doc: int,
    ) -> list[dict[str, Any]]:
        fallback = self._fallback_cards(doc, experience_types, max_per_doc)
        payload = await self.llm_complete_json(
            self._experience_prompt(doc, experience_types, max_per_doc),
            system=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=2200,
            fallback={"experiences": fallback},
        )
        experiences = payload.get("experiences", [])
        if not isinstance(experiences, list) or not experiences:
            return fallback

        cards: list[dict[str, Any]] = []
        for item in experiences:
            if not isinstance(item, dict):
                continue
            experience_type = str(item.get("experience_type", "")).strip().lower()
            if experience_type not in experience_types:
                continue
            statement = self.normalize_text(str(item.get("experience_statement", "")))
            evidence_text = self.normalize_text(str(item.get("evidence_text", statement)))
            if not statement:
                continue
            cards.append(
                {
                    "experience_type": experience_type,
                    "experience_title": self.normalize_text(str(item.get("experience_title", ""))) or self._default_title(experience_type, doc),
                    "statement_nature": self._normalize_statement_nature(item.get("statement_nature"), experience_type),
                    "experience_statement": statement,
                    "future_problem": self.normalize_text(str(item.get("future_problem", ""))) or self._future_problem_prompt(experience_type, doc),
                    "applicability": self.normalize_text(str(item.get("applicability", ""))) or self._default_applicability(doc, experience_type),
                    "supporting_evidence": self.normalize_text(str(item.get("supporting_evidence", ""))) or evidence_text or statement,
                    "paper_location": self.normalize_text(str(item.get("paper_location", ""))) or self._default_paper_location(doc),
                    "is_verifiable": self._coerce_bool(item.get("is_verifiable"), default=True),
                    "verification_method": self.normalize_text(str(item.get("verification_method", ""))) or self._default_verification_method(experience_type),
                    "possible_counterexample": self.normalize_text(str(item.get("possible_counterexample", ""))) or self._default_possible_counterexample(experience_type),
                    "confidence": self._coerce_confidence(item.get("confidence"), default=self._default_confidence(experience_type)),
                    "benchmark_transformable": self._coerce_bool(item.get("benchmark_transformable"), default=self._default_benchmark_transformable(experience_type)),
                    "actionable_advice": self.normalize_text(str(item.get("actionable_advice", ""))) or self._default_advice(experience_type, statement),
                    "caveats": self.normalize_text(str(item.get("caveats", ""))) or self._default_caveat(experience_type),
                    "evidence_text": evidence_text or statement,
                }
            )
        return self._fill_missing_cards(cards=cards, fallback=fallback, max_per_doc=max_per_doc)

    def _build_experience_card(self, doc: SourceDocument, card_data: dict[str, Any]) -> ExperienceCard:
        experience_type = str(card_data.get("experience_type", ExperienceType.fact.value)).strip().lower()
        return ExperienceCard(
            experience_type=ExperienceType(experience_type),
            experience_title=str(card_data.get("experience_title") or self._default_title(experience_type, doc)),
            statement_nature=self._normalize_statement_nature(card_data.get("statement_nature"), experience_type),
            experience_statement=str(card_data.get("experience_statement") or doc.content[:250]),
            applicability=str(card_data.get("applicability") or self._default_applicability(doc, experience_type)),
            supporting_evidence=str(card_data.get("supporting_evidence") or card_data.get("evidence_text") or doc.content[:250]),
            paper_location=str(card_data.get("paper_location") or self._default_paper_location(doc)),
            is_verifiable=self._coerce_bool(card_data.get("is_verifiable"), default=True),
            verification_method=str(card_data.get("verification_method") or self._default_verification_method(experience_type)),
            possible_counterexample=str(card_data.get("possible_counterexample") or self._default_possible_counterexample(experience_type)),
            confidence=self._coerce_confidence(card_data.get("confidence"), default=self._default_confidence(experience_type)),
            benchmark_transformable=self._coerce_bool(
                card_data.get("benchmark_transformable"),
                default=self._default_benchmark_transformable(experience_type),
            ),
            actionable_advice=str(card_data.get("actionable_advice") or self._default_advice(experience_type, doc.content[:250])),
            caveats=str(card_data.get("caveats") or self._default_caveat(experience_type)),
        )

    def _fallback_cards(self, doc: SourceDocument, experience_types: list[str], max_per_doc: int) -> list[dict[str, Any]]:
        sentences = self.salient_sentences(doc, limit=max(max_per_doc, len(experience_types), 3))
        if not sentences and doc.content:
            sentences = [doc.content[:250]]
        if not sentences:
            sentences = [doc.title[:250] or "No content available."]
        cards: list[dict[str, Any]] = []
        type_sequence = self._expand_experience_types(experience_types, max_per_doc)
        for index, experience_type in enumerate(type_sequence):
            sentence = sentences[min(index, len(sentences) - 1)]
            cards.append(
                {
                    "experience_type": experience_type,
                    "experience_title": self._default_title(experience_type, doc),
                    "statement_nature": self._default_statement_nature(experience_type),
                    "experience_statement": sentence,
                    "future_problem": self._future_problem_prompt(experience_type, doc),
                    "applicability": self._default_applicability(doc, experience_type),
                    "supporting_evidence": sentence,
                    "paper_location": self._default_paper_location(doc),
                    "is_verifiable": True,
                    "verification_method": self._default_verification_method(experience_type),
                    "possible_counterexample": self._default_possible_counterexample(experience_type),
                    "confidence": self._default_confidence(experience_type),
                    "benchmark_transformable": self._default_benchmark_transformable(experience_type),
                    "actionable_advice": self._default_advice(experience_type, sentence),
                    "caveats": self._default_caveat(experience_type),
                    "evidence_text": sentence,
                }
            )
        return cards

    def _fill_missing_cards(
        self,
        *,
        cards: list[dict[str, Any]],
        fallback: list[dict[str, Any]],
        max_per_doc: int,
    ) -> list[dict[str, Any]]:
        if len(cards) >= max_per_doc:
            return cards[:max_per_doc]
        existing_keys = {
            (
                str(card.get("experience_type", "")),
                str(card.get("experience_statement", "")),
                str(card.get("future_problem", "")),
            )
            for card in cards
        }
        for item in fallback:
            key = (
                str(item.get("experience_type", "")),
                str(item.get("experience_statement", "")),
                str(item.get("future_problem", "")),
            )
            if key in existing_keys:
                continue
            cards.append(item)
            existing_keys.add(key)
            if len(cards) >= max_per_doc:
                break
        return cards[:max_per_doc] or fallback[:max_per_doc]

    def _expand_experience_types(self, experience_types: list[str], max_per_doc: int) -> list[str]:
        if not experience_types:
            experience_types = list(self.DEFAULT_TYPES)
        expanded: list[str] = []
        while len(expanded) < max_per_doc:
            expanded.extend(experience_types)
        return expanded[:max_per_doc]

    def _experience_prompt(self, doc: SourceDocument, experience_types: list[str], max_per_doc: int) -> str:
        return build_experience_prompt(doc, experience_types, max_per_doc)

    def _normalize_types(self, raw: Any) -> list[str]:
        if not raw:
            return list(self.DEFAULT_TYPES)
        if isinstance(raw, str):
            candidates = [part.strip().lower() for part in raw.split(",") if part.strip()]
        elif isinstance(raw, list):
            candidates = [str(part).strip().lower() for part in raw if str(part).strip()]
        else:
            return list(self.DEFAULT_TYPES)
        allowed = [candidate for candidate in candidates if candidate in self.DEFAULT_TYPES]
        return allowed or list(self.DEFAULT_TYPES)

    def _looks_like_academic_paper(self, doc: SourceDocument) -> bool:
        source_type = (doc.source_type or "").lower()
        if source_type in {"paper", "academic_paper", "conference_paper", "journal_article", "preprint"}:
            return True
        content = (doc.title + " " + doc.content[:1200]).lower()
        cues = ["abstract", "introduction", "method", "experiment", "conclusion", "paper", "study", "dataset"]
        return any(cue in content for cue in cues)

    def _future_problem_prompt(self, experience_type: str, doc: SourceDocument) -> str:
        return future_problem_prompt(experience_type, doc)

    def _default_title(self, experience_type: str, doc: SourceDocument) -> str:
        labels = {
            ExperienceType.fact.value: "Fact Experience",
            ExperienceType.strategy.value: "Strategy Experience",
            ExperienceType.mechanism.value: "Mechanism Experience",
            ExperienceType.boundary.value: "Boundary Experience",
            ExperienceType.failure.value: "Failure Experience",
        }
        return f"{labels.get(experience_type, 'Experience')} | {doc.title[:50]}"

    def _default_applicability(self, doc: SourceDocument, experience_type: str) -> str:
        if experience_type == ExperienceType.fact.value:
            return f"Use when the new task shares the key conditions, setup, or evaluation lens of '{doc.title}'."
        if experience_type == ExperienceType.strategy.value:
            return f"Use when future work has similar constraints, resources, or optimization goals to '{doc.title}'."
        if experience_type == ExperienceType.mechanism.value:
            return f"Use when you need to explain why a result in work related to '{doc.title}' appears or disappears."
        if experience_type == ExperienceType.boundary.value:
            return f"Use when you must judge whether conclusions from '{doc.title}' still hold under changed conditions."
        return f"Use when future exploration resembles '{doc.title}' and it is important to avoid repeating a known ineffective attempt."

    def _default_statement_nature(self, experience_type: str) -> str:
        if experience_type == ExperienceType.fact.value:
            return "evidence_supported_conclusion"
        if experience_type == ExperienceType.mechanism.value:
            return "mechanism_explanation"
        if experience_type == ExperienceType.failure.value:
            return "evidence_supported_conclusion"
        if experience_type == ExperienceType.boundary.value:
            return "evidence_supported_conclusion"
        return "synthesized_summary"

    def _default_paper_location(self, doc: SourceDocument) -> str:
        source_type = doc.source_type or "paper"
        return f"Not explicitly localized; inspect the most relevant passage in the {source_type} text around the supporting evidence."

    def _default_verification_method(self, experience_type: str) -> str:
        if experience_type == ExperienceType.fact.value:
            return "Check the reported experiment, observation, or theorem under the stated conditions; reproduce if feasible."
        if experience_type == ExperienceType.strategy.value:
            return "Verify by re-running the strategy under similar constraints and comparing with the paper's reported baseline."
        if experience_type == ExperienceType.mechanism.value:
            return "Verify with ablation studies, controls, or theoretical analysis that tests the claimed causal chain."
        if experience_type == ExperienceType.boundary.value:
            return "Vary the stated condition or task regime and confirm where the reported advantage weakens or disappears."
        return "Verify that the negative result or side effect reappears under the reported setup and does not disappear after controlling for confounders."

    def _default_possible_counterexample(self, experience_type: str) -> str:
        return self._default_caveat(experience_type)

    def _default_confidence(self, experience_type: str) -> float:
        if experience_type == ExperienceType.mechanism.value:
            return 0.6
        if experience_type == ExperienceType.strategy.value:
            return 0.65
        return 0.7

    def _default_benchmark_transformable(self, experience_type: str) -> bool:
        return experience_type in {
            ExperienceType.fact.value,
            ExperienceType.strategy.value,
            ExperienceType.boundary.value,
            ExperienceType.failure.value,
        }

    def _normalize_statement_nature(self, raw: Any, experience_type: str) -> str:
        allowed = {
            "author_claim",
            "evidence_supported_conclusion",
            "mechanism_explanation",
            "speculative_hypothesis",
            "synthesized_summary",
        }
        value = self.normalize_text(str(raw or "")).lower()
        if value in allowed:
            return value
        return self._default_statement_nature(experience_type)

    def _coerce_bool(self, raw: Any, *, default: bool) -> bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        text = self.normalize_text(str(raw or "")).lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
        return default

    def _coerce_confidence(self, raw: Any, *, default: float) -> float:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return min(1.0, max(0.0, value))

    def _default_advice(self, experience_type: str, evidence_text: str) -> str:
        if experience_type == ExperienceType.fact.value:
            return f"Treat this as a validated fact only after checking that the new setting matches the supported conditions. Evidence cue: {evidence_text[:120]}"
        if experience_type == ExperienceType.strategy.value:
            return f"Try this workflow first, but preserve the paper's stated preconditions and evaluate gains incrementally. Evidence cue: {evidence_text[:120]}"
        if experience_type == ExperienceType.mechanism.value:
            return f"Use this explanation as a causal hypothesis and verify it with ablations or controls before relying on it. Evidence cue: {evidence_text[:120]}"
        if experience_type == ExperienceType.boundary.value:
            return f"Check these scope limits before transferring the method or claim into a new benchmark or task. Evidence cue: {evidence_text[:120]}"
        return f"Preserve this negative result as a warning signal so future agents can skip the same low-value path. Evidence cue: {evidence_text[:120]}"

    def _default_caveat(self, experience_type: str) -> str:
        if experience_type == ExperienceType.fact.value:
            return "Do not over-generalize beyond the paper's supported objects, conditions, metrics, or observed range."
        if experience_type == ExperienceType.strategy.value:
            return "A useful strategy can still fail when compute budget, data quality, or assumptions differ materially."
        if experience_type == ExperienceType.mechanism.value:
            return "Separate correlation from causation and keep any uncertainty explicit if the mechanism is only partially supported."
        if experience_type == ExperienceType.boundary.value:
            return "Boundary experience is only useful if the limiting condition is explicit enough to prevent mistaken transfer."
        return "Keep failure experience only when the paper gives enough negative evidence to guide future decisions credibly."

    def _experience_guideline(self) -> AnnotationGuideline:
        return AnnotationGuideline(
            label_schema={
                "x": "future problem",
                "y": {
                    "experience_type": [experience_type.value for experience_type in ExperienceType],
                    "experience_title": "short title",
                    "statement_nature": [
                        "author_claim",
                        "evidence_supported_conclusion",
                        "mechanism_explanation",
                        "speculative_hypothesis",
                        "synthesized_summary",
                    ],
                    "experience_statement": "core transferable experience",
                    "applicability": "where this experience applies",
                    "supporting_evidence": "direct evidence from the paper",
                    "paper_location": "section, table, figure, paragraph, or approximate location",
                    "is_verifiable": "whether the experience can be checked",
                    "verification_method": "how to verify or falsify it",
                    "possible_counterexample": "counterexample or failure condition",
                    "confidence": "0-1 confidence score",
                    "benchmark_transformable": "whether it can become a benchmark task",
                    "actionable_advice": "what to do next",
                    "caveats": "limits, risks, or uncertainty",
                },
            },
            positive_criteria=[
                "Fact experience must be supported by experiments, data, theorem-like claims, or direct observations from the paper.",
                "Strategy experience must contain an executable recommendation plus conditions, gains, or risks.",
                "Mechanism experience must include an explanation chain and avoid overstating causality.",
                "Boundary experience must make the limiting condition or scope restriction explicit.",
                "Failure experience must preserve a meaningful negative result, failed attempt, or side effect.",
            ],
            negative_criteria=[
                "Do not introduce conclusions that are not grounded in the paper.",
                "Do not collapse different experience types into the same card without a clear dominant label.",
                "Do not omit caveats when the original result is conditional or uncertain.",
            ],
            edge_cases=[
                "If the paper only weakly supports a claim, keep the uncertainty in the mechanism or caveat fields.",
                "If a result only holds for a subset of tasks or data regimes, prefer boundary experience or make the scope explicit.",
            ],
            human_review_required=True,
            review_priority="medium",
        )
