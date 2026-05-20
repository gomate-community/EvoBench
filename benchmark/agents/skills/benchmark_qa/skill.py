from __future__ import annotations

import json
import logging
import uuid

from benchmark.agents.skills._document_common import DocumentSkillMixin
from benchmark.agents.skills.benchmark_qa.prompts import (
    GENERATION_PROMPT,
    INSTRUCTION,
    SOURCE_REF_REASON,
    SYSTEM_PROMPT,
)
from benchmark.schemas import (
    ArtifactRole,
    SampleArtifact,
    SampleOutput,
    SourceDocument,
    SourceReference,
    TaskType,
    UnifiedSample,
    VerificationMethod,
)

logger = logging.getLogger(__name__)

RISK_DIFFICULTY_MAP = {"high": 0.85, "medium": 0.65, "low": 0.45}


class BenchmarkQASkill(DocumentSkillMixin):
    """Generate normal, counterfactual, and risk-annotated QA triplets from documents."""

    async def generate(
        self,
        *,
        documents: list[SourceDocument] | None = None,
        error_samples=None,
        limit: int = 10,
    ) -> list[UnifiedSample]:
        docs = documents or []
        cfg = self._merged_config()
        groups_per_doc = cfg.get("groups_per_doc", 2)

        llm = getattr(self, "_llm", None)
        if llm is None:
            logger.warning("BenchmarkQASkill 需要 LLM，当前 LLM 不可用")
            return []

        samples: list[UnifiedSample] = []
        for doc in docs:
            if len(samples) >= limit:
                break
            # 每组产出 3 条样本，计算需要多少组
            remaining = limit - len(samples)
            n_groups = min(groups_per_doc, remaining // 3 or 1)
            triplets = await self._generate_triplets(llm, doc, n_groups)
            for triplet in triplets:
                if len(samples) >= limit:
                    break
                group_samples = self._build_triplet_samples(doc, triplet)
                for s in group_samples:
                    if len(samples) >= limit:
                        break
                    samples.append(s)
        return samples

    async def _generate_triplets(self, llm, doc: SourceDocument, n: int) -> list[dict]:
        """调用 LLM 生成 n 组三元组"""
        prompt = GENERATION_PROMPT.format(
            n=n,
            title=doc.title,
            content=doc.content[:3000],
        )
        try:
            raw = await llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.4, max_tokens=2048)
            return self._parse_response(raw)
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
            return []

    def _build_triplet_samples(self, doc: SourceDocument, triplet: dict) -> list[UnifiedSample]:
        """从一组三元组构建 3 条 UnifiedSample"""
        question = triplet.get("question", "")
        fact_answer = triplet.get("fact_answer", "")
        counterfactual_answer = triplet.get("counterfactual_answer", "")
        entity_replaced = triplet.get("entity_replaced", "")
        risk_statement = triplet.get("risk_statement", "")
        risk_level = triplet.get("risk_level", "medium")
        risk_reason = triplet.get("risk_reason", "")
        evidence_text = triplet.get("evidence", "")

        if not question or not fact_answer:
            return []

        group_id = uuid.uuid4().hex[:12]
        evidence = self.build_evidence(doc, evidence_text or fact_answer)
        source_ref = SourceReference.from_doc(doc, SOURCE_REF_REASON)
        base_input = self.base_input(doc, evidence)

        results: list[UnifiedSample] = []

        # ─── 样本 A: 正常样本 ─────────────────────────────────────────────
        results.append(
            UnifiedSample(
                sample_id=self.make_id("sample", "benchmark_qa", "normal", doc.source_id, question),
                task_type=TaskType.document_to_xy,
                skill_id=self.definition.skill_id,
                domain=self.context.config.domain,
                language=self.context.config.language,
                input=base_input,
                output=SampleOutput(
                    artifacts=[
                        SampleArtifact(role=ArtifactRole.question, key="x", value=question, evidence_ids=[evidence.evidence_id]),
                        SampleArtifact(role=ArtifactRole.answer, key="y", value=fact_answer, evidence_ids=[evidence.evidence_id]),
                    ],
                    target_schema=self.definition.output_schema,
                ),
                source_refs=[source_ref],
                evidence=[evidence],
                instruction=INSTRUCTION,
                verification_method=VerificationMethod.evidence_overlap,
                annotation_guideline=self.guideline(),
                difficulty_estimate=0.4,
                tags=["benchmark_qa", "normal", doc.source_type],
                metadata={
                    "topic": self.context.topic,
                    "sample_type": "normal",
                    "group_id": group_id,
                    "generation_mode": "llm",
                },
            )
        )

        # ─── 样本 B: 反事实样本 ───────────────────────────────────────────
        if counterfactual_answer:
            results.append(
                UnifiedSample(
                    sample_id=self.make_id("sample", "benchmark_qa", "counterfactual", doc.source_id, question),
                    task_type=TaskType.document_to_xy,
                    skill_id=self.definition.skill_id,
                    domain=self.context.config.domain,
                    language=self.context.config.language,
                    input=base_input,
                    output=SampleOutput(
                        artifacts=[
                            SampleArtifact(role=ArtifactRole.question, key="x", value=question, evidence_ids=[evidence.evidence_id]),
                            SampleArtifact(role=ArtifactRole.answer, key="y", value=counterfactual_answer, evidence_ids=[evidence.evidence_id]),
                        ],
                        target_schema=self.definition.output_schema,
                    ),
                    source_refs=[source_ref],
                    evidence=[evidence],
                    instruction=INSTRUCTION,
                    verification_method=VerificationMethod.contradiction_check,
                    annotation_guideline=self.guideline(),
                    difficulty_estimate=0.6,
                    tags=["benchmark_qa", "counterfactual", doc.source_type],
                    metadata={
                        "topic": self.context.topic,
                        "sample_type": "counterfactual",
                        "group_id": group_id,
                        "entity_replaced": entity_replaced,
                        "generation_mode": "llm",
                    },
                )
            )

        # ─── 样本 C: 风险样本 ─────────────────────────────────────────────
        if risk_statement:
            risk_difficulty = RISK_DIFFICULTY_MAP.get(risk_level, 0.65)
            results.append(
                UnifiedSample(
                    sample_id=self.make_id("sample", "benchmark_qa", "risk", doc.source_id, risk_statement),
                    task_type=TaskType.document_to_xy,
                    skill_id=self.definition.skill_id,
                    domain=self.context.config.domain,
                    language=self.context.config.language,
                    input=base_input,
                    output=SampleOutput(
                        artifacts=[
                            SampleArtifact(role=ArtifactRole.question, key="x", value=risk_statement, evidence_ids=[]),
                            SampleArtifact(role=ArtifactRole.answer, key="y", value=f"该陈述存在{risk_level}风险：{risk_reason}", evidence_ids=[]),
                        ],
                        target_schema=self.definition.output_schema,
                    ),
                    source_refs=[source_ref],
                    evidence=[evidence],
                    instruction=INSTRUCTION,
                    verification_method=VerificationMethod.human,
                    annotation_guideline=self.guideline(),
                    difficulty_estimate=risk_difficulty,
                    tags=["benchmark_qa", "risk", f"risk_{risk_level}", doc.source_type],
                    metadata={
                        "topic": self.context.topic,
                        "sample_type": "risk",
                        "group_id": group_id,
                        "risk_level": risk_level,
                        "risk_reason": risk_reason,
                        "generation_mode": "llm",
                    },
                )
            )

        return results

    @staticmethod
    def _parse_response(raw: str) -> list[dict]:
        """解析 LLM 返回的 JSON 数组"""
        import re

        text = raw.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            return []
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    return [item for item in data if isinstance(item, dict)]
                except json.JSONDecodeError:
                    pass
            return []
