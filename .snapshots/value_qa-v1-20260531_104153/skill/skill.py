"""value_qa Skill 主体（M3 完整版：三阶段 LLM 调用）。

阶段 1：_extract_and_route   - 单次 LLM 调用，从 doc 抽取最多 facts_per_doc 条事实并路由到维度
阶段 2/3：_generate_triplet  - 每条事实 1 次 LLM 调用，生成 high/medium/low 三档样本
后处理：_align_annotations   - 用 substring 兜底定位 value_annotation 的 char offset

每个 doc 总 LLM 调用数 ≈ 1 + facts_per_doc；每条 fact 产 3 条样本。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from benchmark.agents.skills._document_common import DocumentSkillMixin
from benchmark.agents.skills.value_qa.dimensions import (
    MEDIUM_PATTERN_IDS,
    SECONDARY_DIMENSION_IDS,
    list_secondary_brief,
    load_dimensions,
    medium_pattern_for,
    primary_id_en,
    primary_of,
    reference_layers,
)
from benchmark.agents.skills.value_qa.prompts import (
    INSTRUCTION,
    ROUTE_PROMPT,
    SOURCE_REF_REASON,
    SYSTEM_PROMPT,
    TRIPLET_PROMPT,
)
from benchmark.agents.skills.value_qa.taxonomy import (
    VALID_LEVELS,
    list_layer_keys,
    load_taxonomy,
)
from benchmark.schemas import (
    ArtifactRole,
    SampleArtifact,
    SampleOutput,
    SourceDocument,
    SourceReference,
    TaskType,
    UnifiedSample,
    ValueAnnotation,
    VerificationMethod,
)

logger = logging.getLogger(__name__)


LEVEL_DIFFICULTY = {"high": 0.40, "medium": 0.55, "low": 0.75}


class ValueQASkill(DocumentSkillMixin):
    """Value-aligned QA samples grounded in the 6 secondary evaluation dimensions."""

    async def generate(
        self,
        *,
        documents: list[SourceDocument] | None = None,
        error_samples=None,
        limit: int = 10,
    ) -> list[UnifiedSample]:
        docs = documents or []
        cfg = self._merged_config()
        facts_per_doc = int(cfg.get("facts_per_doc", 2))

        try:
            dimensions_data = load_dimensions()
        except Exception as exc:  # noqa: BLE001
            logger.error("[value_qa] failed to load evaluation dimensions: %s", exc)
            return []
        try:
            taxonomy = load_taxonomy()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[value_qa] taxonomy unavailable, anchors will be empty: %s", exc)
            taxonomy = {}

        if not self.llm_enabled():
            logger.warning("[value_qa] LLM disabled, returning empty list")
            return []
        # 优先复用 SampleFactoryAgent 注入的 _llm；独立调用时由 get_llm 懒构造
        llm = getattr(self, "_llm", None) or self.get_llm()

        samples: list[UnifiedSample] = []
        for doc in docs:
            if len(samples) >= limit:
                break
            facts = await self._extract_and_route(llm, doc, facts_per_doc, dimensions_data)
            for fact_idx, fact_item in enumerate(facts):
                if len(samples) >= limit:
                    break
                triplet = await self._generate_triplet(
                    llm, doc, fact_item, dimensions_data, taxonomy, fact_idx
                )
                for sample in triplet:
                    if len(samples) >= limit:
                        break
                    samples.append(sample)
        return samples

    # ──────────────────────────────────────────────────────────────────────
    # 阶段 1：事实抽取 + 二级评估维度路由
    # ──────────────────────────────────────────────────────────────────────
    async def _extract_and_route(
        self,
        llm,
        doc: SourceDocument,
        facts_per_doc: int,
        dimensions_data: dict[str, Any],
    ) -> list[dict[str, str]]:
        dimensions_block = self._format_dimensions_brief(dimensions_data)
        prompt = ROUTE_PROMPT.format(
            max_facts=facts_per_doc + 2,  # 多抽几个，便于 skip 过滤后还有量
            dimensions_block=dimensions_block,
            title=doc.title,
            content=doc.content[:3000],
        )
        try:
            raw = await llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.3, max_tokens=1024)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[value_qa] route LLM call failed: %s", exc)
            return []

        items = self._parse_json_list(raw)
        kept: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            fact = (item.get("fact") or "").strip()
            secondary = (item.get("evaluation_dimension") or "").strip()
            if not fact:
                continue
            if secondary not in SECONDARY_DIMENSION_IDS:
                # skip / 非法维度 一律丢弃
                continue
            kept.append(
                {
                    "fact": fact,
                    "evaluation_dimension": secondary,
                    "reason": item.get("reason", ""),
                }
            )
            if len(kept) >= facts_per_doc:
                break
        if not kept:
            logger.info("[value_qa] doc %s yielded 0 routable facts", doc.source_id)
        return kept

    # ──────────────────────────────────────────────────────────────────────
    # 阶段 2/3：三档对照样本生成
    # ──────────────────────────────────────────────────────────────────────
    async def _generate_triplet(
        self,
        llm,
        doc: SourceDocument,
        fact_item: dict[str, str],
        dimensions_data: dict[str, Any],
        taxonomy: dict[str, Any],
        fact_idx: int,
    ) -> list[UnifiedSample]:
        secondary = fact_item["evaluation_dimension"]
        fact = fact_item["fact"]
        primary = primary_of(dimensions_data, secondary) or ""
        secondary_def = self._secondary_def(dimensions_data, secondary)

        anchors = self._anchors_from_layers(
            taxonomy, reference_layers(dimensions_data, secondary)
        )
        anchors_block = self._format_anchors(anchors) or "(无可参考锚点；可不引用)"

        medium_patterns = medium_pattern_for(dimensions_data, secondary)
        medium_block = self._format_medium_patterns(medium_patterns)

        prompt = TRIPLET_PROMPT.format(
            primary_id=primary,
            secondary_id=secondary,
            dimension_definition=(secondary_def.get("definition") or "").strip(),
            high_criteria=(secondary_def.get("high_criteria") or "").strip(),
            low_criteria=(secondary_def.get("low_criteria") or "").strip(),
            medium_patterns_block=medium_block,
            anchors_block=anchors_block,
            fact=fact,
            title=doc.title,
            content=doc.content[:2500],
        )
        try:
            raw = await llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.5, max_tokens=2048)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[value_qa] triplet LLM call failed: %s", exc)
            return []

        triplet_obj = self._parse_json_object(raw)
        if not triplet_obj:
            return []

        # 锚点合法 key（仅用于 annotation.value_keys 范围限制）；现在已是软引用
        legal_keys: set[str] = set()
        for layer in reference_layers(dimensions_data, secondary):
            legal_keys.update(list_layer_keys(taxonomy, layer))

        out: list[UnifiedSample] = []
        for level in VALID_LEVELS:
            payload = triplet_obj.get(level)
            if not isinstance(payload, dict):
                continue
            sample = self._build_sample(
                doc=doc,
                primary=primary,
                secondary=secondary,
                level=level,
                fact=fact,
                fact_idx=fact_idx,
                payload=payload,
                legal_keys=legal_keys,
            )
            if sample is not None:
                out.append(sample)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # 单条样本构造 + value_annotation 校准
    # ──────────────────────────────────────────────────────────────────────
    def _build_sample(
        self,
        *,
        doc: SourceDocument,
        primary: str,
        secondary: str,
        level: str,
        fact: str,
        fact_idx: int,
        payload: dict[str, Any],
        legal_keys: set[str],
    ) -> UnifiedSample | None:
        question = (payload.get("question") or "").strip()
        answer = (payload.get("answer") or "").strip()
        evidence_quote = (payload.get("evidence_quote") or "").strip()
        if not question or not answer:
            return None

        evidence = self.build_evidence(doc, evidence_quote or fact)
        source_ref = SourceReference.from_doc(doc, SOURCE_REF_REASON)
        base_input = self.base_input(doc, evidence)

        annotations = self._align_annotations(
            payload.get("value_annotations") or [],
            question=question,
            answer=answer,
            evidence_text=evidence.text,
            legal_keys=legal_keys,
        )
        # medium 样本捕获 medium_pattern；其余档位强制为 None
        medium_pattern: str | None = None
        if level == "medium":
            raw_pattern = (payload.get("medium_pattern") or "").strip()
            medium_pattern = raw_pattern if raw_pattern in MEDIUM_PATTERN_IDS else "unknown"

        # 标注命中的锚点 key 汇总（仅供分析；不再作硬约束）
        core_values_aligned = sorted(
            {key for ann in annotations for key in ann.value_keys if key in legal_keys}
        )

        primary_en = primary_id_en(primary)

        metadata: dict[str, Any] = {
            "topic": self.context.topic,
            # 一级 / 二级评估维度（新体系核心字段）
            "primary_metric": primary,
            "evaluation_dimension": secondary,
            # 兼容旧字段：value_dimension 仍保留为一级英文 id（cultural_fidelity / ethics_compliance）
            "value_dimension": primary_en,
            "value_level": level,
            "fact": fact,
            "fact_idx": fact_idx,
            "core_values_aligned": [{"key": k} for k in core_values_aligned],
            "generation_mode": "llm",
        }
        if medium_pattern is not None:
            metadata["medium_pattern"] = medium_pattern

        tags = ["value_qa", primary_en or "unknown_primary", secondary, f"level_{level}", doc.source_type]
        if medium_pattern:
            tags.append(f"medium_{medium_pattern}")

        return UnifiedSample(
            sample_id=self.make_id(
                "sample",
                "value_qa",
                primary_en or "unknown",
                secondary,
                level,
                doc.source_id,
                f"f{fact_idx}",
                question,
            ),
            task_type=TaskType.document_to_xy,
            skill_id=self.definition.skill_id,
            domain=self.context.config.domain,
            language=self.context.config.language,
            input=base_input,
            output=SampleOutput(
                artifacts=[
                    SampleArtifact(
                        role=ArtifactRole.question,
                        key="x",
                        value=question,
                        evidence_ids=[evidence.evidence_id],
                    ),
                    SampleArtifact(
                        role=ArtifactRole.answer,
                        key="y",
                        value=answer,
                        evidence_ids=[evidence.evidence_id],
                    ),
                ],
                target_schema=self.definition.output_schema,
            ),
            source_refs=[source_ref],
            evidence=[evidence],
            instruction=INSTRUCTION,
            verification_method=VerificationMethod.value_alignment,
            annotation_guideline=self.guideline(),
            difficulty_estimate=LEVEL_DIFFICULTY.get(level, 0.5),
            value_annotations=annotations,
            tags=tags,
            metadata=metadata,
        )

    def _align_annotations(
        self,
        raw_annotations: list[Any],
        *,
        question: str,
        answer: str,
        evidence_text: str,
        legal_keys: set[str],
    ) -> list[ValueAnnotation]:
        field_text_map = {"question": question, "answer": answer, "evidence": evidence_text}
        out: list[ValueAnnotation] = []
        for raw in raw_annotations or []:
            if not isinstance(raw, dict):
                continue
            field = raw.get("field")
            text = (raw.get("text") or "").strip()
            label = raw.get("label")
            polarity = raw.get("polarity") or "neutral"
            value_layer = raw.get("value_layer") or ""
            value_keys = [k for k in (raw.get("value_keys") or []) if k in legal_keys]
            rationale = raw.get("rationale") or ""

            if field not in field_text_map or not text or not label:
                continue
            haystack = field_text_map[field]
            start = haystack.find(text) if haystack else -1
            end = start + len(text) if start >= 0 else -1

            try:
                ann = ValueAnnotation(
                    field=field,
                    text=text,
                    start=start,
                    end=end,
                    label=label,
                    polarity=polarity,
                    value_layer=value_layer if value_layer in {"A", "B", "C", "D"} else "",
                    value_keys=value_keys,
                    rationale=rationale,
                )
            except Exception as exc:  # noqa: BLE001  -- 非法枚举值等
                logger.debug("[value_qa] drop invalid annotation: %s (%s)", raw, exc)
                continue
            out.append(ann)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _format_dimensions_brief(data: dict[str, Any]) -> str:
        """阶段 1 提示词中的 6 个评估维度清单（id ｜ 一级 ｜ 定义）。"""
        lines: list[str] = []
        for d in list_secondary_brief(data):
            definition = d["definition"].replace("\n", " ").strip()
            lines.append(f'- {d["id"]} ｜ {d["primary"]} ｜ {definition}')
        return "\n".join(lines)

    @staticmethod
    def _secondary_def(data: dict[str, Any], secondary_id: str) -> dict[str, Any]:
        for d in data.get("secondary_dimensions") or []:
            if d.get("id") == secondary_id:
                return d
        return {}

    @staticmethod
    def _anchors_from_layers(taxonomy: dict[str, Any], layers: list[str]) -> list[dict[str, Any]]:
        """按 4 层 taxonomy 的层 ID 列表（如 ['B','A']）抽出 anchor 列表。"""
        out: list[dict[str, Any]] = []
        if not taxonomy:
            return out
        for layer in layers:
            layer_def = taxonomy.get("layers", {}).get(layer, {})
            for item in layer_def.get("items", []):
                out.append({**item, "layer": layer})
            for domain_name, domain_def in layer_def.get("domains", {}).items():
                for item in domain_def.get("items", []):
                    out.append({**item, "layer": layer, "domain": domain_name})
        return out

    @staticmethod
    def _format_medium_patterns(patterns: list[dict[str, Any]]) -> str:
        if not patterns:
            return "(该维度未配置 medium_patterns；请按通用 letter_vs_spirit/partial_correct/hedged 灰区思路生成)"
        lines: list[str] = []
        for p in patterns:
            pid = p.get("pattern_id", "")
            label = p.get("label", "")
            hint = p.get("hint", "")
            lines.append(f"- {pid}（{label}）：{hint}")
        return "\n".join(lines)

    @staticmethod
    def _format_anchors(anchors: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for a in anchors:
            layer = a.get("layer", "")
            domain = a.get("domain", "")
            tag = f"[{layer}{('-' + domain) if domain else ''}]"
            lines.append(f"{tag} {a['key']} | {a.get('label','')} | {a.get('gloss','')}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json_list(raw: str) -> list[Any]:
        text = (raw or "").strip()
        if not text:
            return []
        if "```" in text:
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                return []
            try:
                data = json.loads(m.group(0))
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                return []

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            return {}
        if "```" in text:
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return {}
            try:
                data = json.loads(m.group(0))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
