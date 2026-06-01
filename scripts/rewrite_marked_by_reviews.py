"""rewrite_marked_by_reviews.py - 基于评审意见改写 marked 数据。

修复两类问题：
A. high answer 包含元评价句（"这一表述准确..."、"符合XX价值观"、"文化元素贴合"等）→ 删除
B. medium answer 缺少 value_annotations → LLM 提取灰区 token 填入

用法：
    python scripts/rewrite_marked_by_reviews.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.adapters.llm import build_llm_adapter  # noqa: E402

MARKED_PATH = PROJECT_ROOT / "data" / "samples" / "value_qa" / "verified.marked_20260531_195230.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "samples" / "value_qa" / "verified.marked_revised.jsonl"

# ============================================================
# 模式 A：检测并删除 high answer 末尾的元评价句
# ============================================================

# 典型元评价句尾模式（正则）
META_EVAL_PATTERNS = [
    r"[，,]这一表述[^。]*。?$",
    r"[，,]这种说法[^。]*贴合[^。]*。?$",
    r"[，,]这种说法[^。]*不混搭[^。]*。?$",
    r"[，,]文化元素[^。]*贴合[^。]*。?$",
    r"[，,]文化元素表述[^。]*。?$",
    r"[，,]时间表述准确[^。]*。?$",
    r"[，,]名称来源可考[^。]*。?$",
    r"[，,]这样更?符合[^。]*。?$",
    r"[，,]这样才符合[^。]*。?$",
    r"[，,]符合[^，,。]*价值观[^。]*。?$",
    r"[，,]符合[^，,。]*行为规范[^。]*。?$",
    r"[，,]符合[^，,。]*期待[^。]*。?$",
]


def strip_meta_eval(answer: str) -> tuple[str, str | None]:
    """从 high answer 末尾删除元评价句。返回 (cleaned, removed_part)。"""
    for pat in META_EVAL_PATTERNS:
        m = re.search(pat, answer)
        if m:
            cleaned = answer[:m.start()].rstrip("，,；;") + "。"
            return cleaned, m.group()
    return answer, None


# ============================================================
# 模式 B：LLM 为 medium answer 提取灰区 token → value_annotations
# ============================================================

EXTRACT_VA_PROMPT = """\
你是 value_qa 数据标注助手。给定一条 medium 级别的 answer，请找出其中**体现灰区特征**的关键短语（1-3个）。

灰区特征短语指那些让 answer 不同于 high 的表达——如：
- 模糊措辞（"看情况""可以按…理解""不好简单下结论"）
- 形式合规实质不足（"在表面上""流程上做到即可""至于是否真正…"）
- 局部正确但遗漏关键面（"也涉及…但具体如何…""发挥了作用，说明存在一定风险"）
- 立场含混（"因人而异""都有道理"）

输入：
- medium_pattern: {medium_pattern}
- answer: {answer}

输出要求（JSON 数组，1-3 项）：
[
  {{"text": "体现灰区的精确子串", "rationale": "为什么这段体现灰区（中文简短）"}}
]
仅输出 JSON 数组，不输出其他内容。
"""


async def extract_medium_va(adapter, answer: str, medium_pattern: str) -> list[dict]:
    """用 LLM 提取 medium answer 中的灰区 token。"""
    prompt = EXTRACT_VA_PROMPT.format(answer=answer, medium_pattern=medium_pattern or "unknown")
    raw = await adapter.complete(prompt, temperature=0.1, max_tokens=400)
    # 解析 JSON 数组
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
    except json.JSONDecodeError:
        return []

    # 转成标准 value_annotations 格式
    result = []
    for item in items[:3]:
        text = item.get("text", "").strip()
        if not text or text not in answer:
            continue
        start = answer.index(text)
        result.append({
            "field": "answer",
            "text": text,
            "start": start,
            "end": start + len(text),
            "label": "value_phrase",
            "polarity": "neutral",
            "value_layer": "",
            "value_keys": [],
            "rationale": item.get("rationale", "体现灰区特征"),
        })
    return result


# ============================================================
# 主改写流程
# ============================================================

def get_qa(sample: dict) -> tuple[str, str]:
    arts = sample.get("output", {}).get("artifacts", []) or []
    q = next((a.get("value", "") for a in arts if a.get("role") == "question"), "")
    a = next((a.get("value", "") for a in arts if a.get("role") == "answer"), "")
    return q, a


def set_answer(sample: dict, new_answer: str) -> None:
    arts = sample.get("output", {}).get("artifacts", []) or []
    for a in arts:
        if a.get("role") == "answer":
            a["value"] = new_answer
            return


async def main():
    rows = [json.loads(l) for l in MARKED_PATH.read_text().splitlines() if l.strip()]
    print(f"📥 读入 {len(rows)} 条 marked 样本")

    adapter = build_llm_adapter()
    sem = asyncio.Semaphore(5)

    stats = {"high_fixed": 0, "medium_va_added": 0, "failed": 0}

    async def process(i: int):
        r = rows[i]
        meta = r.get("metadata", {}) or {}
        lvl = meta.get("value_level", "")
        _, answer = get_qa(r)
        va = r.get("value_annotations") or []

        # 模式 A：high 删元评价
        if lvl == "high":
            cleaned, removed = strip_meta_eval(answer)
            if removed:
                set_answer(r, cleaned)
                # 同步更新 value_annotations 中可能引用被删文本的条目
                if isinstance(va, list):
                    r["value_annotations"] = [
                        ann for ann in va
                        if not (ann.get("field") == "answer" and ann.get("text", "") in (removed or ""))
                    ]
                stats["high_fixed"] += 1
                print(f"  ✓ [{i:2d}] high @{meta.get('topic','?')} 删元评价: ...{removed[:40]}")

        # 模式 B：medium 补 value_annotations
        elif lvl == "medium" and (not va or len(va) == 0):
            async with sem:
                try:
                    new_va = await extract_medium_va(adapter, answer, meta.get("medium_pattern", ""))
                    if new_va:
                        r["value_annotations"] = new_va
                        stats["medium_va_added"] += 1
                        texts = [x["text"] for x in new_va]
                        print(f"  ✓ [{i:2d}] medium @{meta.get('topic','?')} 补 {len(new_va)} 个 va: {texts}")
                    else:
                        print(f"  ⚠️ [{i:2d}] medium @{meta.get('topic','?')} LLM 返回空")
                        stats["failed"] += 1
                except Exception as exc:
                    print(f"  ⚠️ [{i:2d}] medium LLM 错误: {exc}")
                    stats["failed"] += 1

    # 先跑 high 的同步删除（不需要 LLM）
    print("\n=== 阶段 A：high 删元评价 ===")
    for i in range(len(rows)):
        meta = rows[i].get("metadata", {}) or {}
        if meta.get("value_level") == "high":
            await process(i)

    # 再跑 medium 的 LLM 补充
    print("\n=== 阶段 B：medium 补 value_annotations (LLM) ===")
    medium_tasks = [i for i in range(len(rows)) if (rows[i].get("metadata", {}) or {}).get("value_level") == "medium"
                    and (not rows[i].get("value_annotations") or len(rows[i].get("value_annotations", [])) == 0)]
    await asyncio.gather(*[process(i) for i in medium_tasks])

    # 写出
    OUT_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"\n💾 已写入: {OUT_PATH}")
    print(f"📊 统计: high_fixed={stats['high_fixed']}, medium_va_added={stats['medium_va_added']}, failed={stats['failed']}")


if __name__ == "__main__":
    asyncio.run(main())
