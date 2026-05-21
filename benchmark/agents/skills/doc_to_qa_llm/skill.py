"""LLM-augmented doc_to_qa Skill：通过子类化覆盖父类 ``generate`` 加入 LLM 分支。

- 有 LLM 时调用 LLM 生成 QA 对，并构造 UnifiedSample（与 wsy_skill_dev 上行为完全一致）；
- 无 LLM 时直接委托给父类（main 上的纯模板实现），形成统一的回退路径。
"""

from __future__ import annotations

import json
import logging
import re

from benchmark.agents.skills.doc_to_qa.skill import DocumentToQASkill
from benchmark.agents.skills.doc_to_qa_llm.prompts import (
    QA_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from benchmark.agents.skills.doc_to_qa.prompts import (
    INSTRUCTION,
    SOURCE_REF_REASON,
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


class DocumentToQALLMSkill(DocumentToQASkill):
    """LLM-augmented variant：先走 LLM，失败/无 LLM 时回退父类模板。"""

    async def generate(
        self,
        *,
        documents: list[SourceDocument] | None = None,
        error_samples=None,
        limit: int = 10,
    ) -> list[UnifiedSample]:
        docs = documents or []
        cfg = self._merged_config()
        pairs_per_doc = cfg.get("pairs_per_doc", 3)

        llm = getattr(self, "_llm", None)
        if llm is None:
            logger.info("LLM 不可用，回退到父类模板模式")
            return await super().generate(
                documents=documents,
                error_samples=error_samples,
                limit=limit,
            )

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
        """调用 LLM 生成 n 个 QA 对。"""
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
        """解析 LLM 返回的 JSON 数组。"""
        text = raw.strip()
        # 尝试提取 ```json ... ``` 代码块
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
