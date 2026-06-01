"""scan_qa_self_contained.py - 扫描 verified.jsonl，识别违反「QA 语义自足性」的样本。

判定规则（任一命中即标记为违规）：
1. 黑名单短语：question 或 answer 中包含「根据文档/根据文中/文中提到/原文写到/...」等指代源文措辞
2. 代词主语：question 以「这部剧/这个系统/这件事/这一现象/这一说法/该剧/该系统/该现象/...」开头
3. 主语缺失：question 句首找不到具名主语（启发式：以"如何/怎样/为什么/什么/是否"开头但前面无名词）

用法：
    python scripts/scan_qa_self_contained.py
    python scripts/scan_qa_self_contained.py --show 30   # 列出前 30 条违规明细
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "data" / "samples" / "value_qa" / "verified.jsonl"

# 黑名单短语（出现即违规）
BLACKLIST = [
    "根据文档", "根据文中", "根据原文", "根据上述",
    "文档明确", "文档提到", "文档说", "文档中",
    "文中提到", "文中说", "文中明确", "文中表述",
    "原文写到", "原文提到", "原文中", "原文表述",
    "该文提及", "该文说",
    "上述文档", "上述内容", "上面的文档",
    "这一说法", "这一论断", "这一描述",
    "这部剧", "这个系统", "这件事", "这一现象",
    "该剧", "该系统",
]

# 代词开头主语（question 以此开头视为主语不明确）
PRONOUN_LEADS = [
    "这部", "这个", "这件", "这一", "这种", "这类", "这些",
    "该剧", "该系统", "该现象", "该论断", "该说法",
    "它", "他",
]


def get_qa(sample: dict) -> tuple[str, str]:
    arts = sample.get("output", {}).get("artifacts", []) or []
    q = next((a.get("value", "") for a in arts if a.get("role") == "question"), "")
    a = next((a.get("value", "") for a in arts if a.get("role") == "answer"), "")
    return q, a


def find_blacklist_hits(text: str) -> list[str]:
    return [bw for bw in BLACKLIST if bw in text]


def is_pronoun_lead(question: str) -> str | None:
    q = question.strip().lstrip("「『\"\"")
    for lead in PRONOUN_LEADS:
        if q.startswith(lead):
            return lead
    return None


def scan(rows: list[dict]) -> tuple[list[dict], dict]:
    violations: list[dict] = []
    stats = {
        "total": len(rows),
        "blacklist_q": 0,
        "blacklist_a": 0,
        "pronoun_lead_q": 0,
        "any_violation": 0,
        "phrase_freq": Counter(),
        "by_level": Counter(),
        "by_dimension": Counter(),
    }

    for i, r in enumerate(rows):
        q, a = get_qa(r)
        meta = r.get("metadata", {}) or {}
        lv = meta.get("value_level", "?")
        dim = meta.get("evaluation_dimension", "?")

        q_hits = find_blacklist_hits(q)
        a_hits = find_blacklist_hits(a)
        pron = is_pronoun_lead(q)

        if not (q_hits or a_hits or pron):
            continue

        stats["any_violation"] += 1
        if q_hits:
            stats["blacklist_q"] += 1
        if a_hits:
            stats["blacklist_a"] += 1
        if pron:
            stats["pronoun_lead_q"] += 1
        for hit in q_hits + a_hits:
            stats["phrase_freq"][hit] += 1
        if pron:
            stats["phrase_freq"][f"<代词主语:{pron}>"] += 1
        stats["by_level"][lv] += 1
        stats["by_dimension"][dim] += 1

        violations.append({
            "idx": i,
            "sample_id": r.get("sample_id", "?"),
            "level": lv,
            "dimension": dim,
            "question": q,
            "answer_head": a[:120],
            "q_hits": q_hits,
            "a_hits": a_hits,
            "pronoun_lead": pron,
        })

    return violations, stats


def report(stats: dict, violations: list[dict], show: int = 0) -> None:
    print(f"\n{'='*72}")
    print(f"  QA 语义自足性扫描报告 - 共 {stats['total']} 条样本")
    print(f"{'='*72}")
    n = stats["total"]
    v = stats["any_violation"]
    print(f"\n📊 违规总览")
    print(f"  总违规: {v}/{n} ({v/n*100:.1f}%)")
    print(f"  ├─ question 含黑名单短语: {stats['blacklist_q']}")
    print(f"  ├─ answer   含黑名单短语: {stats['blacklist_a']}")
    print(f"  └─ question 代词起头:    {stats['pronoun_lead_q']}")

    print(f"\n📊 高频违规短语 Top 15")
    for k, c in stats["phrase_freq"].most_common(15):
        print(f"  {c:4d}  {k}")

    print(f"\n📊 按 value_level 分布")
    for k, c in stats["by_level"].most_common():
        print(f"  {k:8s}: {c}")

    print(f"\n📊 按维度分布")
    for k, c in stats["by_dimension"].most_common():
        print(f"  {k}: {c}")

    if show > 0:
        print(f"\n📋 违规明细（前 {show} 条）")
        for v in violations[:show]:
            tags = []
            if v["q_hits"]:
                tags.append(f"Q:{','.join(v['q_hits'])}")
            if v["a_hits"]:
                tags.append(f"A:{','.join(v['a_hits'])}")
            if v["pronoun_lead"]:
                tags.append(f"主语:{v['pronoun_lead']}")
            print(f"\n[#{v['idx']:3d}] [{v['level']:6s}] {' | '.join(tags)}")
            print(f"  Q: {v['question']}")
            print(f"  A: {v['answer_head']}...")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--show", type=int, default=0, help="展示前 N 条违规明细")
    ap.add_argument("--out", default=None, help="可选：把违规清单写到 json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.src).read_text(encoding="utf-8").splitlines() if line.strip()]
    violations, stats = scan(rows)
    report(stats, violations, show=args.show)

    if args.out:
        Path(args.out).write_text(json.dumps({"stats": {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}, "violations": violations}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n违规清单已写入: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
