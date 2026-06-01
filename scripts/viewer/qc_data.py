"""qc_data.py - QC 评分数据层（不依赖 benchmark 包）。

负责：
    - 加载 round1_samples.jsonl + round1_index.json
    - 加载 / 保存 round1_scores.jsonl（每条 = 一个三联组的评分记录）
    - 计算评分进度统计
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ============================================================
# Schema
# ============================================================


@dataclass
class TripletScore:
    """一个三联组的评分。"""

    triplet_id: str
    # D1/D2 共享一个分（评的是整个三联组）
    d1_fact_judgable: int = 0       # 1-3，0=未评
    d2_triplet_discrim: int = 0     # 1-3，0=未评
    # D3/D4 三档分开
    d3_answer_fullness: dict[str, int] = field(default_factory=lambda: {"high": 0, "medium": 0, "low": 0})
    d4_annotation_correct: dict[str, int] = field(default_factory=lambda: {"high": 0, "medium": 0, "low": 0})
    # 自由文本
    issue_tags: list[str] = field(default_factory=list)
    note: str = ""
    reviewer: str = ""
    ts: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_complete(self) -> bool:
        """所有必填评分项都打过分（>=1）才算 complete。"""
        if self.d1_fact_judgable < 1 or self.d2_triplet_discrim < 1:
            return False
        for lv in ("high", "medium", "low"):
            if self.d3_answer_fullness.get(lv, 0) < 1:
                return False
            # D4 的 0 分可能合法（如 "annotation 完全缺失"），但 medium 档 D4 经常本就空
            # 这里仍要求 ≥0，即必须给出明确选择
        return True


# ============================================================
# IO
# ============================================================

# Issue 标签预设（多选）
ISSUE_TAGS = [
    "fact-纯客观",
    "fact-含价值含义",
    "三档雷同",
    "high/low 立场不鲜明",
    "medium 不是灰区",
    "answer-套话",
    "answer-空泛",
    "answer-偏题",
    "annotation-text 不是子串",
    "annotation-polarity 错",
    "annotation-缺失",
    "evidence-不来自原文",
    "其它",
]


def load_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def samples_by_triplet(index: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """index + samples → {triplet_id: {"high": {...}, "medium": {...}, "low": {...}}}"""
    by_id = {s.get("sample_id"): s for s in samples}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in index:
        tid = entry["triplet_id"]
        out[tid] = {
            "high": by_id.get(entry["high_id"], {}),
            "medium": by_id.get(entry["medium_id"], {}),
            "low": by_id.get(entry["low_id"], {}),
        }
    return out


def load_scores(path: Path) -> dict[str, TripletScore]:
    if not path.exists():
        return {}
    out: dict[str, TripletScore] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["triplet_id"]] = TripletScore(
            triplet_id=d["triplet_id"],
            d1_fact_judgable=d.get("d1_fact_judgable", 0),
            d2_triplet_discrim=d.get("d2_triplet_discrim", 0),
            d3_answer_fullness=d.get("d3_answer_fullness") or {"high": 0, "medium": 0, "low": 0},
            d4_annotation_correct=d.get("d4_annotation_correct") or {"high": 0, "medium": 0, "low": 0},
            issue_tags=d.get("issue_tags") or [],
            note=d.get("note", ""),
            reviewer=d.get("reviewer", ""),
            ts=d.get("ts", ""),
        )
    return out


def save_scores(path: Path, scores: dict[str, TripletScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sc in scores.values():
            f.write(json.dumps(sc.to_jsonable(), ensure_ascii=False) + "\n")


def upsert_score(scores: dict[str, TripletScore], score: TripletScore) -> None:
    score.ts = _dt.datetime.now().isoformat(timespec="seconds")
    scores[score.triplet_id] = score


# ============================================================
# 进度
# ============================================================


@dataclass
class ProgressStats:
    total_triplets: int = 0
    completed: int = 0
    avg_d1: float = 0.0
    avg_d2: float = 0.0
    avg_d3: float = 0.0
    avg_d4: float = 0.0


def compute_progress(index: list[dict[str, Any]], scores: dict[str, TripletScore]) -> ProgressStats:
    total = len(index)
    completed_scores = [sc for sc in scores.values() if sc.is_complete]
    n = len(completed_scores)
    if n == 0:
        return ProgressStats(total_triplets=total)
    avg_d1 = sum(sc.d1_fact_judgable for sc in completed_scores) / n
    avg_d2 = sum(sc.d2_triplet_discrim for sc in completed_scores) / n
    d3_vals = [v for sc in completed_scores for v in sc.d3_answer_fullness.values() if v > 0]
    d4_vals = [v for sc in completed_scores for v in sc.d4_annotation_correct.values() if v > 0]
    return ProgressStats(
        total_triplets=total,
        completed=n,
        avg_d1=avg_d1,
        avg_d2=avg_d2,
        avg_d3=sum(d3_vals) / len(d3_vals) if d3_vals else 0.0,
        avg_d4=sum(d4_vals) / len(d4_vals) if d4_vals else 0.0,
    )
