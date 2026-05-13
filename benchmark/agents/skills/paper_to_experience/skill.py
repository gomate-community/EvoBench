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
    SampleArtifact,
    SampleOutput,
    SourceDocument,
    SourceReference,
    TaskType,
    UnifiedSample,
    VerificationMethod,
)


class PaperToExperienceSkill(DocumentSkillMixin):
    """Extract reusable experience cards from academic papers."""

    DEFAULT_TYPES = ("fact", "strategy", "cognitive")

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
            for card_index, card in enumerate(cards[:max_per_doc], start=1):
                evidence_text = card.get("evidence_text") or card.get("experience_statement") or doc.content[:250]
                evidence = self.build_evidence(doc, str(evidence_text))
                experience_type = str(card.get("experience_type", "fact"))
                future_question = str(card.get("future_problem", self._future_problem_prompt(experience_type, doc)))
                answer = {
                    "experience_type": experience_type,
                    "experience_title": str(card.get("experience_title", self._default_title(experience_type, doc))),
                    "experience_statement": str(card.get("experience_statement", evidence.text)),
                    "applicability": str(card.get("applicability", self._default_applicability(doc))),
                    "actionable_advice": str(card.get("actionable_advice", self._default_advice(experience_type, evidence.text))),
                    "caveats": str(card.get("caveats", self._default_caveat(experience_type))),
                }
                samples.append(
                    UnifiedSample(
                        sample_id=self.make_id(
                            "sample",
                            self.definition.skill_id,
                            doc.source_id,
                            str(card_index),
                            experience_type,
                            future_question,
                            str(answer["experience_statement"]),
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
                            target_schema={"x": "future_problem", "y": "experience_card"},
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
            max_tokens=1800,
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
                    "experience_statement": statement,
                    "future_problem": self.normalize_text(str(item.get("future_problem", ""))) or self._future_problem_prompt(experience_type, doc),
                    "applicability": self.normalize_text(str(item.get("applicability", ""))) or self._default_applicability(doc),
                    "actionable_advice": self.normalize_text(str(item.get("actionable_advice", ""))) or self._default_advice(experience_type, statement),
                    "caveats": self.normalize_text(str(item.get("caveats", ""))) or self._default_caveat(experience_type),
                    "evidence_text": evidence_text or statement,
                }
            )
        return self._fill_missing_cards(cards=cards, fallback=fallback, max_per_doc=max_per_doc)

    def _fallback_cards(self, doc: SourceDocument, experience_types: list[str], max_per_doc: int) -> list[dict[str, Any]]:
        sentences = self.salient_sentences(doc, limit=max(max_per_doc, len(experience_types), 3))
        if not sentences and doc.content:
            sentences = [doc.content[:250]]
        cards: list[dict[str, Any]] = []
        type_sequence = self._expand_experience_types(experience_types, max_per_doc)
        for index, experience_type in enumerate(type_sequence):
            sentence = sentences[min(index, len(sentences) - 1)]
            cards.append(
                {
                    "experience_type": experience_type,
                    "experience_title": self._default_title(experience_type, doc),
                    "experience_statement": sentence,
                    "future_problem": self._future_problem_prompt(experience_type, doc),
                    "applicability": self._default_applicability(doc),
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
        cues = ["abstract", "introduction", "method", "experiment", "conclusion", "论文", "实验", "方法", "研究"]
        return any(cue in content for cue in cues)

    def _future_problem_prompt(self, experience_type: str, doc: SourceDocument) -> str:
        return future_problem_prompt(experience_type, doc)

    def _default_title(self, experience_type: str, doc: SourceDocument) -> str:
        labels = {
            "fact": "事实经验",
            "strategy": "策略经验",
            "cognitive": "认知经验",
        }
        return f"{labels.get(experience_type, '经验')} | {doc.title[:50]}"

    def _default_applicability(self, doc: SourceDocument) -> str:
        return f"适用于与《{doc.title}》研究对象、约束条件或任务目标相近的场景。"

    def _default_advice(self, experience_type: str, evidence_text: str) -> str:
        if experience_type == "fact":
            return f"先把该论文支持的核心事实纳入问题建模，再用新场景的数据验证是否仍然成立。证据线索：{evidence_text[:80]}"
        if experience_type == "strategy":
            return f"优先复用论文中被证明有效的方法步骤，并在新场景中逐步验证关键假设。证据线索：{evidence_text[:80]}"
        return f"在决策前先检查自己是否忽略了论文提示的判断框架或局限，再决定是否迁移结论。证据线索：{evidence_text[:80]}"

    def _default_caveat(self, experience_type: str) -> str:
        if experience_type == "fact":
            return "注意该事实通常依赖论文中的实验条件、数据分布或评价口径。"
        if experience_type == "strategy":
            return "注意该策略可能受资源预算、任务规模或先验假设影响。"
        return "注意该认知经验更像判断框架，不能替代证据本身。"

    def _experience_guideline(self) -> AnnotationGuideline:
        return AnnotationGuideline(
            label_schema={
                "x": "future problem",
                "y": {
                    "experience_type": ["fact", "strategy", "cognitive"],
                    "experience_title": "short title",
                    "experience_statement": "core reusable experience",
                    "applicability": "when to use",
                    "actionable_advice": "what to do",
                    "caveats": "boundary or risk",
                },
            },
            positive_criteria=[
                "经验必须由论文证据直接支持。",
                "经验必须能迁移到未来相似问题，而不是只复述论文标题。",
                "future_problem 应与经验的适用场景保持一致。",
            ],
            negative_criteria=[
                "不得引入论文之外的结论。",
                "不得把纯背景描述误写成可执行经验。",
                "不得混淆事实经验、策略经验和认知经验。",
            ],
            edge_cases=[
                "如果论文只支持局部结论，caveats 中必须明确边界条件。",
            ],
            human_review_required=True,
            review_priority="medium",
        )
