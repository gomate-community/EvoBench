from __future__ import annotations

import json
import logging

from benchmark.agents.skills._document_common import DocumentSkillMixin
from benchmark.agents.skills.doc_to_qa.prompts import (
    INSTRUCTION,
    QA_GENERATION_PROMPT,
    SOURCE_REF_REASON,
    SYSTEM_PROMPT,
    answer_from_sentence,
    question_from_sentence,
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


class DocumentToQASkill(DocumentSkillMixin):
    """Generate evidence-grounded QA pairs from a document using LLM."""

    async def generate(self, *, documents: list[SourceDocument] | None = None, error_samples=None, limit: int = 10) -> list[UnifiedSample]:
        docs = documents or []
        cfg = self._merged_config()
        pairs_per_doc = cfg.get("pairs_per_doc", 3)

        # 判断是否有可用 LLM
        llm = getattr(self, "_llm", None)
        if llm is None:
            logger.info("LLM 不可用，回退到模板模式")
            return self._generate_template(docs, limit, pairs_per_doc)

        # ─── LLM 模式 ─────────────────────────────────────────────────────
        samples: list[UnifiedSample] = []
        for doc in docs:
            if len(samples) >= limit:
                break
            n = min(pairs_per_doc, limit - len(samples))
            qa_pairs = await self._generate_with_llm(llm, doc, n)
            for qa in qa_pairs:
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                evidence_text = qa.get("evidence", "")
                if not question or not answer:
                    continue
                evidence = self.build_evidence(doc, evidence_text or answer)
                samples.append(
                    UnifiedSample(
                        sample_id=self.make_id("sample", self.definition.skill_id, doc.source_id, question, answer),
                        task_type=TaskType.document_to_xy,
                        skill_id=self.definition.skill_id,
                        domain=self.context.config.domain,
                        language=self.context.config.language,
                        input=self.base_input(doc, evidence),
                        output=SampleOutput(
                            artifacts=[
                                SampleArtifact(role=ArtifactRole.question, key="x", value=question, evidence_ids=[evidence.evidence_id]),
                                SampleArtifact(role=ArtifactRole.answer, key="y", value=answer, evidence_ids=[evidence.evidence_id]),
                            ],
                            target_schema=self.definition.output_schema,
                        ),
                        source_refs=[SourceReference.from_doc(doc, SOURCE_REF_REASON)],
                        evidence=[evidence],
                        instruction=INSTRUCTION,
                        verification_method=VerificationMethod.evidence_overlap,
                        annotation_guideline=self.guideline(),
                        difficulty_estimate=0.6,
                        tags=["d_to_xy", "qa", "llm_generated", doc.source_type],
                        metadata={"topic": self.context.topic, "generation_mode": "llm"},
                    )
                )
                if len(samples) >= limit:
                    break
        return samples

    async def _generate_with_llm(self, llm, doc: SourceDocument, n: int) -> list[dict]:
        """调用 LLM 生成 n 个 QA 对"""
        prompt = QA_GENERATION_PROMPT.format(
            n=n,
            title=doc.title,
            content=doc.content[:3000],  # 截断过长内容
        )
        try:
            raw = await llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.3, max_tokens=2048)
            return self._parse_qa_response(raw)
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}，回退模板模式处理该文档")
            return []

    @staticmethod
    def _parse_qa_response(raw: str) -> list[dict]:
        """解析 LLM 返回的 JSON 数组"""
        text = raw.strip()
        # 尝试提取 ```json ... ``` 代码块
        if "```" in text:
            import re
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            return []
        except json.JSONDecodeError:
            # 尝试找到 [ ... ] 部分
            import re
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    return [item for item in data if isinstance(item, dict)]
                except json.JSONDecodeError:
                    pass
            return []

    # ─── 模板兜底模式（无 LLM 时）──────────────────────────────────────────

    def _generate_template(self, docs: list[SourceDocument], limit: int, pairs_per_doc: int) -> list[UnifiedSample]:
        """原始模板模式，不调用 LLM"""
        samples: list[UnifiedSample] = []
        for doc in docs:
            for sentence in self.salient_sentences(doc, limit=pairs_per_doc):
                evidence = self.build_evidence(doc, sentence)
                question = question_from_sentence(sentence, doc)
                answer = answer_from_sentence(sentence)
                samples.append(
                    UnifiedSample(
                        sample_id=self.make_id("sample", self.definition.skill_id, doc.source_id, question, answer),
                        task_type=TaskType.document_to_xy,
                        skill_id=self.definition.skill_id,
                        domain=self.context.config.domain,
                        language=self.context.config.language,
                        input=self.base_input(doc, evidence),
                        output=SampleOutput(
                            artifacts=[
                                SampleArtifact(role=ArtifactRole.question, key="x", value=question, evidence_ids=[evidence.evidence_id]),
                                SampleArtifact(role=ArtifactRole.answer, key="y", value=answer, evidence_ids=[evidence.evidence_id]),
                            ],
                            target_schema=self.definition.output_schema,
                        ),
                        source_refs=[SourceReference.from_doc(doc, SOURCE_REF_REASON)],
                        evidence=[evidence],
                        instruction=INSTRUCTION,
                        verification_method=VerificationMethod.evidence_overlap,
                        annotation_guideline=self.guideline(),
                        difficulty_estimate=0.45,
                        tags=["d_to_xy", "qa", doc.source_type],
                        metadata={"topic": self.context.topic, "generation_mode": "template"},
                    )
                )
                if len(samples) >= limit:
                    return samples
        return samples
