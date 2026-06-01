#!/usr/bin/env python3
"""sample_qc.py - 从 verified.jsonl 分层抽样 30 条做人工质检。

抽样规则:
    1. 总量 30 条 = 10 个完整三联组（high/medium/low 一起抽，保证三档区分度可评）
    2. 按二级评估维度分层（保证 6 维都有覆盖，比例向数据量倾斜）
    3. 同维度内 medium_pattern 多样性优先（letter / partial / hedged 尽量都覆盖）
    4. 不重复 fact

输出:
    data/qc/round1_samples.jsonl     -- 30 条原始 sample（保留全部字段）
    data/qc/round1_index.json        -- 10 组 triplet 索引（id/维度/medium_pattern/三档 sample_id）
    data/qc/sampling_summary.md      -- 人类可读的抽样报告

用法:
    python scripts/sample_qc.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# 抽样配额：维度 -> 三联组数（共 10 组）
DIMENSION_QUOTA = {
    "价值观忠实度": 4,
    "文化元素符合度": 2,
    "行为社会规范符合度": 1,
    "伤害风险合规度": 1,
    "公平合规度": 1,
    "隐私合规度": 1,
}
TOTAL_GROUPS = sum(DIMENSION_QUOTA.values())  # 10
TOTAL_SAMPLES = TOTAL_GROUPS * 3              # 30
SEED = 42

# 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 默认（round1）路径——保持向后兼容
SRC = PROJECT_ROOT / "data" / "samples" / "value_qa" / "verified.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "qc"
SAMPLES_OUT = OUT_DIR / "round1_samples.jsonl"
INDEX_OUT = OUT_DIR / "round1_index.json"
SUMMARY_OUT = OUT_DIR / "sampling_summary.md"


def configure_paths(round_id: int) -> None:
    """根据 round_id 切换输入/输出路径。round1 保持原文件名兼容。"""
    global SRC, SAMPLES_OUT, INDEX_OUT, SUMMARY_OUT
    if round_id == 1:
        return  # 默认配置即 round1
    SRC = PROJECT_ROOT / "data" / "samples" / "value_qa" / f"verified.round{round_id}_raw.jsonl"
    SAMPLES_OUT = OUT_DIR / f"round{round_id}_samples.jsonl"
    INDEX_OUT = OUT_DIR / f"round{round_id}_index.json"
    SUMMARY_OUT = OUT_DIR / f"round{round_id}_summary.md"


def fact_key(fact: str) -> str:
    return hashlib.md5((fact or "").encode("utf-8")).hexdigest()[:8]


def triplet_key(meta: dict[str, Any]) -> str:
    """三联组键：用 doc 不可靠（缺失），用 (dim, fact_md5) 替代。"""
    dim = meta.get("evaluation_dimension", "?")
    return f"{dim}#{fact_key(meta.get('fact', ''))}"


def load_rows() -> list[dict]:
    rows = []
    for i, line in enumerate(SRC.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj["_idx"] = i
        rows.append(obj)
    return rows


def group_triplets(rows: list[dict]) -> dict[str, list[dict]]:
    """按 triplet_key 分组，仅保留三档齐全的组。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        meta = r.get("metadata") or {}
        if not meta.get("value_level"):
            continue
        groups[triplet_key(meta)].append(r)
    full = {}
    for k, items in groups.items():
        levels = {(it.get("metadata") or {}).get("value_level") for it in items}
        if {"high", "medium", "low"}.issubset(levels):
            # 仅保留每档第一条（去重保险，理论上 dedupe 后已唯一）
            uniq: dict[str, dict] = {}
            for it in items:
                lv = (it.get("metadata") or {}).get("value_level")
                if lv and lv not in uniq:
                    uniq[lv] = it
            full[k] = [uniq["high"], uniq["medium"], uniq["low"]]
    return full


def stratified_pick(groups: dict[str, list[dict]]) -> list[tuple[str, list[dict]]]:
    """分层抽样：维度 -> 配额，pattern 多样性优先。"""
    rng = random.Random(SEED)
    by_dim: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for k, triplet in groups.items():
        meta = triplet[0].get("metadata") or {}
        dim = meta.get("evaluation_dimension", "?")
        by_dim[dim].append((k, triplet))

    picked: list[tuple[str, list[dict]]] = []
    for dim, quota in DIMENSION_QUOTA.items():
        candidates = by_dim.get(dim, [])
        if not candidates:
            print(f"[WARN] 维度 '{dim}' 无可用三联组，跳过", file=sys.stderr)
            continue
        rng.shuffle(candidates)
        # pattern 多样性优先：先按 medium 档的 medium_pattern 分桶，轮转抽
        by_pattern: dict[str, list] = defaultdict(list)
        for k, triplet in candidates:
            medium = next((it for it in triplet if (it.get("metadata") or {}).get("value_level") == "medium"), None)
            patt = ((medium or {}).get("metadata") or {}).get("medium_pattern", "?")
            by_pattern[patt].append((k, triplet))
        # 轮转：letter → partial → hedged → 其它
        pattern_order = ["letter_vs_spirit", "partial_correct", "hedged"]
        ordered_patts = [p for p in pattern_order if p in by_pattern] + [p for p in by_pattern if p not in pattern_order]
        chosen: list[tuple[str, list[dict]]] = []
        round_no = 0
        while len(chosen) < quota:
            progress = False
            for p in ordered_patts:
                if len(chosen) >= quota:
                    break
                bucket = by_pattern[p]
                if round_no < len(bucket):
                    chosen.append(bucket[round_no])
                    progress = True
            if not progress:
                break
            round_no += 1
        picked.extend(chosen[:quota])
    return picked


def build_index(picked: list[tuple[str, list[dict]]]) -> list[dict]:
    """为 viewer 评分页生成索引。"""
    out = []
    for k, triplet in picked:
        high = next((it for it in triplet if (it.get("metadata") or {}).get("value_level") == "high"), None)
        medium = next((it for it in triplet if (it.get("metadata") or {}).get("value_level") == "medium"), None)
        low = next((it for it in triplet if (it.get("metadata") or {}).get("value_level") == "low"), None)
        meta = (high or medium or low).get("metadata") or {}
        out.append({
            "triplet_id": k,
            "evaluation_dimension": meta.get("evaluation_dimension", ""),
            "primary_metric": meta.get("primary_metric", ""),
            "topic": meta.get("topic", ""),
            "fact": meta.get("fact", ""),
            "fact_md5": fact_key(meta.get("fact", "")),
            "medium_pattern": ((medium or {}).get("metadata") or {}).get("medium_pattern", ""),
            "high_id": (high or {}).get("sample_id", ""),
            "medium_id": (medium or {}).get("sample_id", ""),
            "low_id": (low or {}).get("sample_id", ""),
        })
    return out


def write_summary(picked: list[tuple[str, list[dict]]], total_groups: int) -> str:
    lines = ["# QC Round 1 抽样报告", ""]
    lines.append(f"- 源文件: `{SRC.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- 总三联组: {total_groups}")
    lines.append(f"- 抽样三联组: {len(picked)} / 目标 {TOTAL_GROUPS}")
    lines.append(f"- 抽样样本: {len(picked) * 3} / 目标 {TOTAL_SAMPLES}")
    lines.append(f"- 随机种子: {SEED}")
    lines.append("")

    lines.append("## 维度分布")
    lines.append("| 二级维度 | 配额 | 实际 |")
    lines.append("| --- | ---: | ---: |")
    actual_dim = Counter()
    for k, triplet in picked:
        dim = (triplet[0].get("metadata") or {}).get("evaluation_dimension", "?")
        actual_dim[dim] += 1
    for dim, quota in DIMENSION_QUOTA.items():
        lines.append(f"| {dim} | {quota} | {actual_dim.get(dim, 0)} |")
    lines.append("")

    lines.append("## medium_pattern 分布")
    actual_patt = Counter()
    for _, triplet in picked:
        medium = next((it for it in triplet if (it.get("metadata") or {}).get("value_level") == "medium"), None)
        patt = ((medium or {}).get("metadata") or {}).get("medium_pattern", "?")
        actual_patt[patt] += 1
    for p, n in sorted(actual_patt.items(), key=lambda x: -x[1]):
        lines.append(f"- {p}: {n}")
    lines.append("")

    lines.append("## 抽中的三联组")
    lines.append("| # | 维度 | medium_pattern | topic | fact (前 60 字) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for i, (k, triplet) in enumerate(picked, 1):
        meta = triplet[0].get("metadata") or {}
        medium = next((it for it in triplet if (it.get("metadata") or {}).get("value_level") == "medium"), None)
        patt = ((medium or {}).get("metadata") or {}).get("medium_pattern", "?")
        fact = (meta.get("fact") or "").replace("|", "/").replace("\n", " ")[:60]
        lines.append(f"| {i} | {meta.get('evaluation_dimension', '?')} | {patt} | {meta.get('topic', '?')} | {fact} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1, help="round id（1=默认基线，2=用 round2 增量数据）")
    args = ap.parse_args()
    configure_paths(args.round)

    if not SRC.exists():
        print(f"[ERROR] 找不到源文件: {SRC}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    print(f"加载 {len(rows)} 条样本")
    groups = group_triplets(rows)
    print(f"完整三联组: {len(groups)} 组")
    picked = stratified_pick(groups)
    print(f"抽中三联组: {len(picked)} 组 = {len(picked) * 3} 条")

    # 落盘 samples
    with SAMPLES_OUT.open("w", encoding="utf-8") as f:
        for _, triplet in picked:
            for it in triplet:
                # 去掉临时字段
                obj = {k: v for k, v in it.items() if k != "_idx"}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 落盘 index
    index = build_index(picked)
    INDEX_OUT.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # 落盘 summary
    SUMMARY_OUT.write_text(write_summary(picked, len(groups)), encoding="utf-8")

    print(f"\n输出:\n  {SAMPLES_OUT}\n  {INDEX_OUT}\n  {SUMMARY_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
