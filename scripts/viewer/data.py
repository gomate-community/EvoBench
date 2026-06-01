"""data.py - 纯数据加载层，不依赖 benchmark 包。

仅读 jsonl 文件 + 抽取 viewer 需要的字段，与业务代码零耦合。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SampleRow:
    """Viewer 展示用的扁平化行，一一对应 jsonl 一行。"""

    idx: int
    raw: dict[str, Any]
    # 顶层
    sample_id: str = ""
    skill_id: str = ""
    task_type: str = ""
    status: str = ""
    difficulty: float = 0.0
    leakage_risk: float = 0.0
    ambiguity_risk: float = 0.0
    created_at: str = ""
    tags: list[str] = field(default_factory=list)
    # metadata
    topic: str = ""
    sample_type: str = ""
    group_id: str = ""
    risk_level: str = ""
    risk_reason: str = ""
    entity_replaced: str = ""
    risk_phrases: list[dict] = field(default_factory=list)
    # input
    doc_title: str = ""
    doc_content: str = ""
    doc_source_type: str = ""
    doc_url: str = ""
    evidence_text: str = ""
    evidence_start: int = -1
    evidence_end: int = -1
    evidence_confidence: float = 0.0
    # output
    question: str = ""
    answer: str = ""
    # quality
    evidence_coverage: float = 0.0
    answerability: float = 0.0
    clarity: float = 0.0
    novelty: float = 0.0
    quality_gate: bool = True
    rejection_reasons: list[str] = field(default_factory=list)
    # value_qa 专属（其他 skill 默认空）
    value_dimension: str = ""        # 一级英文 id（cultural_fidelity / ethics_compliance），保持兼容
    primary_metric: str = ""         # 一级中文 id（文化忠实度 / 伦理合规度）
    evaluation_dimension: str = ""   # 二级评估维度（6 选 1）
    medium_pattern: str = ""         # medium 档的灰区模式 id
    value_level: str = ""
    fact: str = ""
    fact_idx: int = -1
    core_values_aligned: list[dict] = field(default_factory=list)
    value_annotations: list[dict] = field(default_factory=list)
    verification_method: str = ""

    def __post_init__(self) -> None:
        """归一化 None：jsonl 里部分字段值为 null（如 evidence start_char/end_char）
        会绕过 dict.get 默认值，必须在这里兜底，避免与 int/float 比较时 TypeError。"""
        # int 字段
        for f_name, default in (
            ("evidence_start", -1),
            ("evidence_end", -1),
            ("fact_idx", -1),
        ):
            if getattr(self, f_name, None) is None:
                setattr(self, f_name, default)
        # float 字段
        for f_name in (
            "difficulty",
            "leakage_risk",
            "ambiguity_risk",
            "evidence_confidence",
            "evidence_coverage",
            "answerability",
            "clarity",
            "novelty",
        ):
            if getattr(self, f_name, None) is None:
                setattr(self, f_name, 0.0)
        # str 字段
        for f_name in (
            "evidence_text",
            "topic",
            "doc_title",
            "doc_content",
            "doc_source_type",
            "doc_url",
            "question",
            "answer",
            "value_dimension",
            "primary_metric",
            "evaluation_dimension",
            "medium_pattern",
            "value_level",
            "fact",
            "verification_method",
        ):
            if getattr(self, f_name, None) is None:
                setattr(self, f_name, "")
        # bool 字段
        if getattr(self, "quality_gate", None) is None:
            self.quality_gate = True


def _extract_row(idx: int, obj: dict) -> SampleRow:
    """从原始 JSON 对象提取 Viewer 关心的字段。"""
    meta = obj.get("metadata") or {}
    inp = obj.get("input") or {}
    out = obj.get("output") or {}
    qs = obj.get("quality_signals") or {}
    evidences = obj.get("evidence") or []

    # 文档
    docs = inp.get("documents") or []
    doc = docs[0] if docs else {}

    # evidence (取第一条)
    ev = evidences[0] if evidences else {}

    # output artifacts -> question (x) / answer (y)
    question = ""
    answer = ""
    for art in out.get("artifacts") or []:
        if art.get("key") == "x" or art.get("role") == "question":
            question = art.get("value", "")
        elif art.get("key") == "y" or art.get("role") == "answer":
            answer = art.get("value", "")

    return SampleRow(
        idx=idx,
        raw=obj,
        sample_id=obj.get("sample_id", ""),
        skill_id=obj.get("skill_id", ""),
        task_type=obj.get("task_type", ""),
        status=obj.get("status", ""),
        difficulty=obj.get("difficulty_estimate", 0.0),
        leakage_risk=obj.get("leakage_risk", 0.0),
        ambiguity_risk=obj.get("ambiguity_risk", 0.0),
        created_at=obj.get("created_at", ""),
        tags=obj.get("tags") or [],
        # metadata
        topic=meta.get("topic", ""),
        sample_type=meta.get("sample_type", ""),
        group_id=meta.get("group_id", ""),
        risk_level=meta.get("risk_level", ""),
        risk_reason=meta.get("risk_reason", ""),
        entity_replaced=meta.get("entity_replaced", ""),
        risk_phrases=meta.get("risk_phrases") or [],
        # doc
        doc_title=doc.get("title", ""),
        doc_content=doc.get("content", ""),
        doc_source_type=doc.get("source_type", ""),
        doc_url=doc.get("url", ""),
        # evidence
        evidence_text=ev.get("text") or "",
        evidence_start=ev.get("start_char") if ev.get("start_char") is not None else -1,
        evidence_end=ev.get("end_char") if ev.get("end_char") is not None else -1,
        evidence_confidence=ev.get("confidence") if ev.get("confidence") is not None else 0.0,
        # output
        question=question,
        answer=answer,
        # quality
        evidence_coverage=qs.get("evidence_coverage", 0.0),
        answerability=qs.get("answerability", 0.0),
        clarity=qs.get("clarity", 0.0),
        novelty=qs.get("novelty", 0.0),
        quality_gate=qs.get("quality_gate_passed", True),
        rejection_reasons=qs.get("rejection_reasons") or [],
        # value_qa
        value_dimension=meta.get("value_dimension", ""),
        primary_metric=meta.get("primary_metric", ""),
        evaluation_dimension=meta.get("evaluation_dimension", ""),
        medium_pattern=meta.get("medium_pattern", ""),
        value_level=meta.get("value_level", ""),
        fact=meta.get("fact", ""),
        fact_idx=meta.get("fact_idx", -1),
        core_values_aligned=meta.get("core_values_aligned") or [],
        value_annotations=obj.get("value_annotations") or [],
        verification_method=obj.get("verification_method", ""),
    )


def load_samples(path: str | Path) -> list[SampleRow]:
    """加载 jsonl，返回扁平化行列表。"""
    rows: list[SampleRow] = []
    p = Path(path)
    if not p.exists():
        return rows
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append(_extract_row(i, obj))
    return rows


# ---------- 聚合 ----------


@dataclass
class KPIStats:
    total: int = 0
    triplet_complete_rate: float = 0.0
    type_dist: dict[str, int] = field(default_factory=dict)
    risk_dist: dict[str, int] = field(default_factory=dict)
    avg_evidence_coverage: float = 0.0
    avg_difficulty: float = 0.0
    quality_gate_pass_rate: float = 0.0
    topic_count: int = 0
    avg_question_len: float = 0.0
    answer_unique_rate: float = 0.0


def compute_kpi(rows: list[SampleRow]) -> KPIStats:
    if not rows:
        return KPIStats()

    total = len(rows)
    type_dist = Counter(r.sample_type for r in rows)
    risk_dist = Counter(r.risk_level for r in rows if r.risk_level)

    # triplet 完整性
    groups: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.group_id:
            groups[r.group_id].add(r.sample_type)
    complete = sum(1 for types in groups.values() if len(types) >= 3)
    triplet_rate = complete / len(groups) if groups else 0.0

    avg_cov = sum(r.evidence_coverage for r in rows) / total
    avg_diff = sum(r.difficulty for r in rows) / total
    gate_pass = sum(1 for r in rows if r.quality_gate) / total

    topics = set(r.topic for r in rows if r.topic)
    avg_q_len = sum(len(r.question) for r in rows) / total
    answers = [r.answer for r in rows]
    unique_answers = len(set(answers))
    answer_uniq = unique_answers / total if total else 0.0

    return KPIStats(
        total=total,
        triplet_complete_rate=triplet_rate,
        type_dist=dict(type_dist),
        risk_dist=dict(risk_dist),
        avg_evidence_coverage=avg_cov,
        avg_difficulty=avg_diff,
        quality_gate_pass_rate=gate_pass,
        topic_count=len(topics),
        avg_question_len=avg_q_len,
        answer_unique_rate=answer_uniq,
    )


def group_triplets(rows: list[SampleRow]) -> dict[str, list[SampleRow]]:
    """按 group_id 分组。"""
    groups: dict[str, list[SampleRow]] = defaultdict(list)
    for r in rows:
        if r.group_id:
            groups[r.group_id].append(r)
    return dict(groups)


# ---------- value_qa 专属聚合 ----------


@dataclass
class ValueKPIStats:
    total: int = 0
    triplet_complete_rate: float = 0.0
    dim_dist: dict[str, int] = field(default_factory=dict)             # 一级（英文 id）分布，向后兼容
    primary_dist: dict[str, int] = field(default_factory=dict)         # 一级（中文）分布
    evaluation_dim_dist: dict[str, int] = field(default_factory=dict)  # 二级评估维度分布
    medium_pattern_dist: dict[str, int] = field(default_factory=dict)  # medium 档的灰区模式分布
    level_dist: dict[str, int] = field(default_factory=dict)
    avg_evidence_coverage: float = 0.0
    avg_difficulty: float = 0.0
    quality_gate_pass_rate: float = 0.0
    topic_count: int = 0
    high_with_positive_rate: float = 0.0
    low_with_negative_rate: float = 0.0
    medium_with_pattern_rate: float = 0.0  # medium 样本中 metadata.medium_pattern ∈ 合法集合的比例


def _value_group_key(row: SampleRow) -> str:
    """value_qa 三联组分组键：同一 doc 同一 fact 文本的 high/medium/low 视为一组。

    用 fact 文本本身 hash 而非 fact_idx，因为 fact_idx 在每次 cli 运行内重置，
    多次跑同一 doc 会让不同 fact 共用相同 idx，导致跨 fact 误合并。
    """
    doc = row.doc_title or row.topic or "?"
    fact_key = hashlib.md5((row.fact or "").encode("utf-8")).hexdigest()[:8] if row.fact else f"idx{row.fact_idx}"
    return f"{doc}#{fact_key}"


def group_value_triplets(rows: list[SampleRow]) -> dict[str, list[SampleRow]]:
    """按 (doc_title, fact 文本 hash) 分组 value_qa 样本。"""
    groups: dict[str, list[SampleRow]] = defaultdict(list)
    for r in rows:
        if r.value_level:
            groups[_value_group_key(r)].append(r)
    return dict(groups)


def compute_value_kpi(rows: list[SampleRow]) -> ValueKPIStats:
    if not rows:
        return ValueKPIStats()

    total = len(rows)
    dim_dist = Counter(r.value_dimension for r in rows if r.value_dimension)
    primary_dist = Counter(r.primary_metric for r in rows if r.primary_metric)
    evaluation_dim_dist = Counter(r.evaluation_dimension for r in rows if r.evaluation_dimension)
    medium_pattern_dist = Counter(
        r.medium_pattern for r in rows if r.value_level == "medium" and r.medium_pattern
    )
    level_dist = Counter(r.value_level for r in rows if r.value_level)

    # 三联组完整性：同 (doc, fact_idx) 含 high/medium/low 三档
    groups = group_value_triplets(rows)
    complete = 0
    for items in groups.values():
        levels = {r.value_level for r in items}
        if {"high", "medium", "low"}.issubset(levels):
            complete += 1
    triplet_rate = complete / len(groups) if groups else 0.0

    avg_cov = sum(r.evidence_coverage for r in rows) / total
    avg_diff = sum(r.difficulty for r in rows) / total
    gate_pass = sum(1 for r in rows if r.quality_gate) / total

    topics = set(r.topic for r in rows if r.topic)

    # high 含 positive、low 含 negative 的比例
    highs = [r for r in rows if r.value_level == "high"]
    lows = [r for r in rows if r.value_level == "low"]
    mediums = [r for r in rows if r.value_level == "medium"]

    def _has_polarity(row: SampleRow, target: str) -> bool:
        return any(a.get("polarity") == target for a in row.value_annotations)

    high_pos_rate = (sum(1 for r in highs if _has_polarity(r, "positive")) / len(highs)) if highs else 0.0
    low_neg_rate = (sum(1 for r in lows if _has_polarity(r, "negative")) / len(lows)) if lows else 0.0
    valid_patterns = {"letter_vs_spirit", "partial_correct", "hedged", "unknown"}
    medium_pattern_rate = (
        sum(1 for r in mediums if r.medium_pattern in valid_patterns) / len(mediums)
    ) if mediums else 0.0

    return ValueKPIStats(
        total=total,
        triplet_complete_rate=triplet_rate,
        dim_dist=dict(dim_dist),
        primary_dist=dict(primary_dist),
        evaluation_dim_dist=dict(evaluation_dim_dist),
        medium_pattern_dist=dict(medium_pattern_dist),
        level_dist=dict(level_dist),
        avg_evidence_coverage=avg_cov,
        avg_difficulty=avg_diff,
        quality_gate_pass_rate=gate_pass,
        topic_count=len(topics),
        high_with_positive_rate=high_pos_rate,
        low_with_negative_rate=low_neg_rate,
        medium_with_pattern_rate=medium_pattern_rate,
    )
