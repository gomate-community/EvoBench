"""rewrite_qa.py - 把 verified.jsonl 中违反 QA 自足性的样本改写成合规版本。

两阶段：
- 阶段 1：机械替换（前缀「根据文档/根据文中/...」类去掉，answer 中「文中提到 X」→「X」）
- 阶段 2：LLM 兜底改写代词主语类（「这部剧/该剧/这个系统/这一现象/...」需要结合 fact 判断指代）

用法：
    python scripts/rewrite_qa.py --src data/samples/value_qa/verified.jsonl \
        --out data/samples/value_qa/verified.rewriting.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.adapters.llm import build_llm_adapter  # noqa: E402

# === 黑名单短语（与 scan_qa_self_contained.py 保持一致） ===
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
PRONOUN_LEADS = [
    "这部", "这个", "这件", "这一", "这种", "这类", "这些",
    "该剧", "该系统", "该现象", "该论断", "该说法",
    "它", "他",
]

# === 阶段 1：机械替换规则（按顺序匹配，第一条命中后替换并退出） ===
# 形如 "根据文档，X" → "X"，"文中提到 Y" → "Y"
PREFIX_PATTERNS = [
    # question 前缀类（去掉前缀，保留主体）
    (re.compile(r"^根据文档[，,：:]?\s*"), ""),
    (re.compile(r"^根据文中[，,：:]?\s*"), ""),
    (re.compile(r"^根据原文[，,：:]?\s*"), ""),
    (re.compile(r"^根据上述[文档内容]*[，,：:]?\s*"), ""),
    (re.compile(r"^根据文段[，,：:]?\s*"), ""),
    (re.compile(r"^结合文档[，,：:]?\s*"), ""),
    (re.compile(r"^结合文中[内容]*[，,：:]?\s*"), ""),
    # answer 句首类
    (re.compile(r"^文中明确指出[，,：:]?\s*"), ""),
    (re.compile(r"^文中明确表述[，,：:]?\s*"), ""),
    (re.compile(r"^文中明确[，,：:]?\s*"), ""),
    (re.compile(r"^文中提到[，,：:]?\s*"), ""),
    (re.compile(r"^文中说[，,：:]?\s*"), ""),
    (re.compile(r"^文档明确指出[，,：:]?\s*"), ""),
    (re.compile(r"^文档明确[，,：:]?\s*"), ""),
    (re.compile(r"^文档提到[，,：:]?\s*"), ""),
    (re.compile(r"^文档说[，,：:]?\s*"), ""),
    (re.compile(r"^原文写到[，,：:]?\s*"), ""),
    (re.compile(r"^原文提到[，,：:]?\s*"), ""),
    (re.compile(r"^该文提及[，,：:]?\s*"), ""),
    (re.compile(r"^上述文档[内容]*[显示表明指出说][，,：:]?\s*"), ""),
    # 句中夹杂的去除（保守只去明确无歧义的）
    (re.compile(r"，文中明确指出[，,：:]?"), "，"),
    (re.compile(r"，文中提到[，,：:]?"), "，"),
    (re.compile(r"，文档明确[，,：:]?"), "，"),
    (re.compile(r"，根据文档[，,：:]?"), "，"),
]


def get_qa_indices(sample: dict) -> tuple[int | None, int | None]:
    arts = sample.get("output", {}).get("artifacts", []) or []
    qi = ai = None
    for i, a in enumerate(arts):
        if a.get("role") == "question" and qi is None:
            qi = i
        if a.get("role") == "answer" and ai is None:
            ai = i
    return qi, ai


def get_qa_text(sample: dict) -> tuple[str, str]:
    arts = sample.get("output", {}).get("artifacts", []) or []
    q = next((a.get("value", "") for a in arts if a.get("role") == "question"), "")
    a = next((a.get("value", "") for a in arts if a.get("role") == "answer"), "")
    return q, a


def has_blacklist(text: str) -> bool:
    return any(bw in text for bw in BLACKLIST)


def has_pronoun_lead(question: str) -> bool:
    q = question.strip().lstrip("「『\"\"")
    return any(q.startswith(lead) for lead in PRONOUN_LEADS)


def mechanical_clean(text: str) -> str:
    """对单段文本应用前缀类正则替换，可能多次循环（去掉之后开头还是黑名单的话）。"""
    out = text
    for _ in range(3):  # 最多 3 轮，避免死循环
        before = out
        for pat, repl in PREFIX_PATTERNS:
            out = pat.sub(repl, out, count=1)
        out = out.lstrip(" ，,：:")
        if out == before:
            break
    return out


# === 阶段 2：LLM 改写 prompt ===
LLM_REWRITE_PROMPT = """你的任务：把一组 question/answer 改写成「语义自足」版本，确保读者不需要看原始文档就能理解 QA 在问什么、答什么。

# 改写规则（强约束）
1. **保留立场不变**：question 表达的探究点、answer 的判断方向（赞同/中立/反驳）和事实细节必须完全保留。
2. **禁用所有指代源文的措辞**：「根据文档/根据文中/文中提到/文档明确/原文写到/这一说法/这一论断/...」全部禁用。
3. **代词主语必须替换为完整名词**：例如「这部剧」→「电视剧《XXX》」，「该系统」→「学术不端检测系统 TMLC」，「这一现象」→「XX 现象」。具体应替换为什么由 fact 内容决定。
4. **句子要自然流畅**，不能僵硬。
5. **绝对不要新增 fact 没有的信息**——只改写措辞，不编造细节。

# 输入
- topic: {topic}
- evaluation_dimension: {dimension}
- value_level: {level}
- fact（QA 依据的事实）: {fact}
- 原 question: {question}
- 原 answer: {answer}

# 输出（只输出 JSON，不要 markdown 围栏）
{{"question": "...", "answer": "..."}}
"""


async def llm_rewrite(adapter, sample: dict, q: str, a: str) -> tuple[str, str]:
    meta = sample.get("metadata", {}) or {}
    fact = meta.get("fact", "") or ""
    if not fact:
        # 兼容字段
        fact = (sample.get("input", {}) or {}).get("fact", "")

    prompt = LLM_REWRITE_PROMPT.format(
        topic=meta.get("topic", "?"),
        dimension=meta.get("evaluation_dimension", "?"),
        level=meta.get("value_level", "?"),
        fact=fact[:500],
        question=q,
        answer=a,
    )
    data = await adapter.complete_json(prompt, temperature=0.2, max_tokens=600)
    new_q = (data.get("question") or "").strip()
    new_a = (data.get("answer") or "").strip()
    if not new_q or not new_a:
        return q, a  # LLM 失败则返回原文
    return new_q, new_a


# === 主流程 ===
def stage1_mechanical(rows: list[dict]) -> tuple[int, list[int]]:
    """对所有样本应用机械替换，原地修改 rows。返回 (修复条数, 仍有违规的行 idx)。"""
    fixed = 0
    still_bad: list[int] = []
    for i, r in enumerate(rows):
        arts = r.get("output", {}).get("artifacts", []) or []
        qi, ai = get_qa_indices(r)
        if qi is None or ai is None:
            continue
        q_orig = arts[qi].get("value", "")
        a_orig = arts[ai].get("value", "")
        q_new = mechanical_clean(q_orig)
        a_new = mechanical_clean(a_orig)
        if q_new != q_orig or a_new != a_orig:
            arts[qi]["value"] = q_new
            arts[ai]["value"] = a_new
            fixed += 1
        # 检查是否仍有违规
        if has_blacklist(q_new) or has_blacklist(a_new) or has_pronoun_lead(q_new):
            still_bad.append(i)
    return fixed, still_bad


async def stage2_llm(rows: list[dict], bad_indices: list[int], concurrency: int = 5) -> tuple[int, list[int]]:
    """对仍有违规的样本走 LLM 改写。返回 (改写条数, 仍违规行 idx)。"""
    if not bad_indices:
        return 0, []
    adapter = build_llm_adapter()
    sem = asyncio.Semaphore(concurrency)
    fixed_count = 0
    still_bad: list[int] = []

    async def run(i: int):
        nonlocal fixed_count
        async with sem:
            r = rows[i]
            arts = r.get("output", {}).get("artifacts", []) or []
            qi, ai = get_qa_indices(r)
            if qi is None or ai is None:
                return
            q_orig = arts[qi].get("value", "")
            a_orig = arts[ai].get("value", "")
            try:
                new_q, new_a = await llm_rewrite(adapter, r, q_orig, a_orig)
            except Exception as exc:
                print(f"  ⚠️ #{i} LLM 失败: {exc}")
                still_bad.append(i)
                return
            # 验证：改写后不能再有违规
            if has_blacklist(new_q) or has_blacklist(new_a) or has_pronoun_lead(new_q):
                print(f"  ⚠️ #{i} LLM 改写后仍违规，保留原文")
                still_bad.append(i)
                return
            arts[qi]["value"] = new_q
            arts[ai]["value"] = new_a
            fixed_count += 1
            print(f"  ✓ #{i} LLM 改写完成")

    await asyncio.gather(*[run(i) for i in bad_indices])
    return fixed_count, still_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(PROJECT_ROOT / "data" / "samples" / "value_qa" / "verified.jsonl"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "samples" / "value_qa" / "verified.rewriting.jsonl"))
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--skip-llm", action="store_true", help="只做阶段1，跳过 LLM 改写")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"📥 读入 {len(rows)} 条样本")

    print(f"\n=== 阶段 1：机械替换 ===")
    fixed1, still_bad = stage1_mechanical(rows)
    print(f"  机械修复 {fixed1} 条，仍违规 {len(still_bad)} 条")

    if not args.skip_llm and still_bad:
        print(f"\n=== 阶段 2：LLM 兜底（concurrency={args.concurrency}）===")
        fixed2, still_bad2 = asyncio.run(stage2_llm(rows, still_bad, concurrency=args.concurrency))
        print(f"  LLM 修复 {fixed2} 条，最终仍违规 {len(still_bad2)} 条")

    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"\n💾 已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
