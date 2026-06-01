"""qc_report.py - 把 round1_scores.jsonl 聚合成 QC 报告。

输出：
    1. 终端摘要（默认）
    2. data/qc/round1_report.md  人类可读报告
    3. data/qc/round1_summary.json  机器可读摘要（给后续 prompt 优化脚本消费）

聚合维度：
    - 概览（总数 / 完整率 / 全局 D1-D4 均值）
    - evaluation_dimension × {D1, D2, D3, D4} 矩阵
    - medium_pattern × {D2, D3·medium} 矩阵
    - issue_tags 频次榜
    - low-score 清单（按 D1+D2+avg(D3) 排序，含 issue_tags 和 note）
    - prompt 改进建议（基于热门 issue_tags 自动生成清单）

用法：
    python scripts/qc_report.py
    python scripts/qc_report.py --top 5      # 低分清单只出前 5 条
    python scripts/qc_report.py --no-write   # 只打印，不落盘
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = PROJECT_ROOT / "data" / "qc" / "round1_scores.jsonl"
INDEX_PATH = PROJECT_ROOT / "data" / "qc" / "round1_index.json"
REPORT_MD_PATH = PROJECT_ROOT / "data" / "qc" / "round1_report.md"
SUMMARY_JSON_PATH = PROJECT_ROOT / "data" / "qc" / "round1_summary.json"

# ============================================================
# 加载
# ============================================================


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_complete(s: dict) -> bool:
    if s.get("d1_fact_judgable", 0) < 1 or s.get("d2_triplet_discrim", 0) < 1:
        return False
    d3 = s.get("d3_answer_fullness", {}) or {}
    return all(d3.get(lv, 0) >= 1 for lv in ("high", "medium", "low"))


def avg_d3(s: dict) -> float:
    vs = [v for v in (s.get("d3_answer_fullness") or {}).values() if v > 0]
    return mean(vs) if vs else 0.0


def avg_d4(s: dict) -> float:
    vs = [v for v in (s.get("d4_annotation_correct") or {}).values() if v > 0]
    return mean(vs) if vs else 0.0


def composite(s: dict) -> float:
    """D1 + D2 + avg(D3) + 0.5 * avg(D4) — D4 权重低（部分档位本就 0 合法）。"""
    return s.get("d1_fact_judgable", 0) + s.get("d2_triplet_discrim", 0) + avg_d3(s) + 0.5 * avg_d4(s)


# ============================================================
# 聚合函数
# ============================================================


def overview(scores: list[dict]) -> dict:
    n = len(scores)
    complete = [s for s in scores if is_complete(s)]
    nc = len(complete)
    if nc == 0:
        return {"total": n, "complete": 0}
    return {
        "total": n,
        "complete": nc,
        "complete_rate": round(nc / n, 3),
        "avg_d1": round(mean(s["d1_fact_judgable"] for s in complete), 2),
        "avg_d2": round(mean(s["d2_triplet_discrim"] for s in complete), 2),
        "avg_d3": round(mean(avg_d3(s) for s in complete), 2),
        "avg_d4": round(mean(avg_d4(s) for s in complete), 2),
    }


def by_dimension(scores: list[dict], index: list[dict]) -> dict[str, dict]:
    """每个 evaluation_dimension 的 D1-D4 均值。"""
    dim_of = {e["triplet_id"]: e["evaluation_dimension"] for e in index}
    bucket: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        if is_complete(s):
            bucket[dim_of.get(s["triplet_id"], "?")].append(s)
    out = {}
    for dim, items in bucket.items():
        out[dim] = {
            "n": len(items),
            "d1": round(mean(s["d1_fact_judgable"] for s in items), 2),
            "d2": round(mean(s["d2_triplet_discrim"] for s in items), 2),
            "d3": round(mean(avg_d3(s) for s in items), 2),
            "d4": round(mean(avg_d4(s) for s in items), 2),
        }
    return out


def by_medium_pattern(scores: list[dict], index: list[dict]) -> dict[str, dict]:
    """每个 medium_pattern 的 D2 / D3·medium 均值。"""
    pat_of = {e["triplet_id"]: e.get("medium_pattern", "-") for e in index}
    bucket: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        if is_complete(s):
            bucket[pat_of.get(s["triplet_id"], "-")].append(s)
    out = {}
    for pat, items in bucket.items():
        d3_med = [s["d3_answer_fullness"]["medium"] for s in items]
        out[pat] = {
            "n": len(items),
            "d2": round(mean(s["d2_triplet_discrim"] for s in items), 2),
            "d3_medium": round(mean(d3_med), 2) if d3_med else 0.0,
        }
    return out


def issue_tag_freq(scores: list[dict]) -> list[tuple[str, int]]:
    cnt: Counter = Counter()
    for s in scores:
        cnt.update(s.get("issue_tags") or [])
    return cnt.most_common()


def low_score_list(scores: list[dict], index: list[dict], top: int) -> list[dict]:
    """按 composite 升序，列出最低 N 条。"""
    by_id = {e["triplet_id"]: e for e in index}
    enriched = []
    for s in scores:
        if not is_complete(s):
            continue
        e = by_id.get(s["triplet_id"], {})
        enriched.append(
            {
                "triplet_id": s["triplet_id"],
                "dim": e.get("evaluation_dimension", "?"),
                "topic": e.get("topic", "?"),
                "pattern": e.get("medium_pattern", "-"),
                "d1": s["d1_fact_judgable"],
                "d2": s["d2_triplet_discrim"],
                "d3": round(avg_d3(s), 2),
                "d4": round(avg_d4(s), 2),
                "composite": round(composite(s), 2),
                "tags": s.get("issue_tags") or [],
                "note": (s.get("note") or "").strip(),
            }
        )
    enriched.sort(key=lambda x: x["composite"])
    return enriched[:top]


# ============================================================
# Prompt 改进建议生成
# ============================================================

# tag → 推荐改 prompt 的位置（针对 ROUTE_PROMPT / TRIPLET_PROMPT）
TAG_TO_ACTION = {
    "fact-纯客观": "ROUTE_PROMPT 加约束：fact 必须含价值倾向，纯日期/数字/流程的事实直接 reject",
    "fact-含价值含义": "ROUTE_PROMPT 调整：fact 表述应保留事实陈述形态，不要把价值判断写死在 fact 里",
    "三档雷同": "TRIPLET_PROMPT 强化：在 high/medium/low 各档增加「立场对立」的硬性指令和反例",
    "high/low 立场不鲜明": "TRIPLET_PROMPT 加示例：high 必须含正向规范用词，low 必须含明显反价值/越界表达",
    "medium 不是灰区": "TRIPLET_PROMPT 重写 medium_pattern 三类的判定锚点（letter/partial/hedged 各给 1 个 anti-example）",
    "answer-套话": "TRIPLET_PROMPT 加约束：每档答案必须引用 evidence 中的具体名词/事件/数据",
    "answer-空泛": "TRIPLET_PROMPT 加示例：每档答案至少含 1 个事实细节或场景描写",
    "answer-偏题": "TRIPLET_PROMPT 加自检：答案末尾追加「与 question 的关联是 …」一句话内审",
    "annotation-text 不是子串": "verifier 加硬校验：annotation.text 必须是 answer 子串，否则整组 reject",
    "annotation-polarity 错": "TRIPLET_PROMPT 在 annotation block 显式给出 polarity 选择规则（high→positive, low→negative）",
    "annotation-缺失": "TRIPLET_PROMPT 强制：每档至少 1 条 annotation；verifier 兜底校验",
    "evidence-不来自原文": "TRIPLET_PROMPT 加约束：所有 answer 中的事实必须能在 evidence 段落里找到",
    "其它": "查看具体 note，case-by-case 处理",
}


def prompt_improvements(tag_freq: list[tuple[str, int]]) -> list[dict]:
    out = []
    for tag, cnt in tag_freq:
        action = TAG_TO_ACTION.get(tag, "（无对应建议，请人工判断）")
        out.append({"tag": tag, "count": cnt, "action": action})
    return out


# ============================================================
# 渲染
# ============================================================


def render_md(report: dict) -> str:
    o = report["overview"]
    lines = []
    lines.append("# QC Round 1 报告\n")
    lines.append(f"- 评分总组数: **{o.get('total', 0)}**")
    lines.append(f"- 完整组数: **{o.get('complete', 0)}**（{o.get('complete_rate', 0) * 100:.0f}%）")
    if "avg_d1" in o:
        lines.append(
            f"- 全局均值: D1={o['avg_d1']} · D2={o['avg_d2']} · D3={o['avg_d3']} · D4={o['avg_d4']}"
        )
    lines.append("")

    # 维度交叉
    lines.append("## 1. 维度交叉（evaluation_dimension × D1-D4）\n")
    lines.append("| 维度 | n | D1 | D2 | D3 | D4 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dim, m in sorted(report["by_dimension"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {dim} | {m['n']} | {m['d1']} | {m['d2']} | {m['d3']} | {m['d4']} |")
    lines.append("")

    # medium_pattern 交叉
    lines.append("## 2. medium_pattern 交叉\n")
    lines.append("| pattern | n | D2 (整组区分度) | D3·medium |")
    lines.append("|---|---:|---:|---:|")
    for pat, m in sorted(report["by_medium_pattern"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {pat} | {m['n']} | {m['d2']} | {m['d3_medium']} |")
    lines.append("")

    # issue 标签
    lines.append("## 3. Issue 标签频次\n")
    if report["issue_tag_freq"]:
        lines.append("| 标签 | 次数 |")
        lines.append("|---|---:|")
        for tag, cnt in report["issue_tag_freq"]:
            lines.append(f"| {tag} | {cnt} |")
    else:
        lines.append("（无标签——本轮所有组都没贴 issue）")
    lines.append("")

    # 低分清单
    lines.append(f"## 4. 低分清单（Top {len(report['low_score_list'])}, 按 composite 升序）\n")
    for item in report["low_score_list"]:
        tags = " / ".join(item["tags"]) if item["tags"] else "-"
        lines.append(
            f"### {item['dim']} · {item['topic']} · `{item['pattern']}`  composite={item['composite']}"
        )
        lines.append(
            f"- D1={item['d1']} D2={item['d2']} D3={item['d3']} D4={item['d4']}"
        )
        lines.append(f"- triplet_id: `{item['triplet_id']}`")
        lines.append(f"- 标签: {tags}")
        if item["note"]:
            lines.append(f"- 备注: {item['note']}")
        lines.append("")

    # prompt 改进建议
    lines.append("## 5. Prompt 改进建议（按标签热度排序）\n")
    if report["prompt_improvements"]:
        for imp in report["prompt_improvements"]:
            lines.append(f"- **[{imp['count']}×] `{imp['tag']}`** → {imp['action']}")
    else:
        lines.append("（本轮没有贴标签，无自动建议——可结合低分清单的 note 手动总结）")
    lines.append("")
    return "\n".join(lines)


def render_console(report: dict) -> None:
    o = report["overview"]
    print(f"\n=== QC Round 1 报告 ===")
    print(
        f"完整 {o.get('complete', 0)}/{o.get('total', 0)} | "
        f"D1={o.get('avg_d1', 0)} D2={o.get('avg_d2', 0)} "
        f"D3={o.get('avg_d3', 0)} D4={o.get('avg_d4', 0)}"
    )
    print("\n[维度] dim | n | D1 D2 D3 D4")
    for dim, m in sorted(report["by_dimension"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {dim:<14} | {m['n']:>2} | {m['d1']} {m['d2']} {m['d3']} {m['d4']}")
    print("\n[pattern] pat | n | D2 D3·med")
    for pat, m in sorted(report["by_medium_pattern"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {pat:<18} | {m['n']:>2} | {m['d2']} {m['d3_medium']}")
    print("\n[Issue 标签]")
    if report["issue_tag_freq"]:
        for tag, cnt in report["issue_tag_freq"]:
            print(f"  {cnt:>2} × {tag}")
    else:
        print("  （无）")
    print(f"\n[低分清单 Top {len(report['low_score_list'])}]")
    for item in report["low_score_list"]:
        tags = " / ".join(item["tags"]) if item["tags"] else "-"
        print(
            f"  composite={item['composite']:.2f}  "
            f"{item['dim']}/{item['pattern']:<18} "
            f"D1={item['d1']} D2={item['d2']} D3={item['d3']} | tags={tags}"
        )
    print("\n[Prompt 改进建议]")
    for imp in report["prompt_improvements"]:
        print(f"  [{imp['count']}×] {imp['tag']}  →  {imp['action']}")


# ============================================================
# main
# ============================================================


def build_report(top: int) -> dict:
    scores = load_jsonl(SCORES_PATH)
    index = load_json(INDEX_PATH)
    return {
        "overview": overview(scores),
        "by_dimension": by_dimension(scores, index),
        "by_medium_pattern": by_medium_pattern(scores, index),
        "issue_tag_freq": issue_tag_freq(scores),
        "low_score_list": low_score_list(scores, index, top),
        "prompt_improvements": prompt_improvements(issue_tag_freq(scores)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5, help="低分清单条数，默认 5")
    ap.add_argument("--no-write", action="store_true", help="只打印，不落盘")
    args = ap.parse_args()

    if not SCORES_PATH.exists():
        print(f"❌ 评分文件不存在: {SCORES_PATH}")
        return

    report = build_report(args.top)
    render_console(report)

    if not args.no_write:
        REPORT_MD_PATH.write_text(render_md(report), encoding="utf-8")
        SUMMARY_JSON_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n✅ Markdown 报告: {REPORT_MD_PATH.relative_to(PROJECT_ROOT)}")
        print(f"✅ JSON 摘要:    {SUMMARY_JSON_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
