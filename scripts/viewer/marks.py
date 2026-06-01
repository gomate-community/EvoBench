"""marks.py - viewer 内部的样本标记 + 评审意见持久化层。

设计：
- 标记不污染源数据 verified.jsonl，独立存储到 _marks.jsonl（同目录）。
- 评审意见同样独立存储到 _comments.jsonl（同目录）。
- 文件格式：每行一条 JSON
- API：load_marks / toggle_mark / clear_marks / export_marked
       load_comments / save_comment / delete_comment / clear_comments / export_comments

约束：
- sample_id 是稳定唯一 key
- 每次操作即落盘，避免 streamlit 进程异常丢数据
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _marks_path(samples_jsonl: Path) -> Path:
    """根据样本文件路径推导对应的 _marks_{stem}.jsonl 路径（按数据文件隔离）。"""
    stem = samples_jsonl.stem  # e.g. "verified" or "verified.round3_6dim_20260601_023005"
    p = samples_jsonl.parent / f"_marks_{stem}.jsonl"
    # 迁移兼容：旧版只有单一 _marks.jsonl，首次使用新路径时自动迁移
    if not p.exists():
        legacy = samples_jsonl.parent / "_marks.jsonl"
        if legacy.exists() and stem == "verified":
            legacy.rename(p)
    return p


def load_marks(samples_jsonl: Path) -> set[str]:
    """读取已标记的 sample_id 集合。文件不存在则返回空 set。"""
    p = _marks_path(samples_jsonl)
    if not p.exists():
        return set()
    out: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            sid = obj.get("sample_id")
            if sid:
                out.add(sid)
        except json.JSONDecodeError:
            continue
    return out


def save_marks(samples_jsonl: Path, marks: set[str]) -> None:
    """全量重写 _marks.jsonl。"""
    p = _marks_path(samples_jsonl)
    now = datetime.now().isoformat(timespec="seconds")
    lines = [json.dumps({"sample_id": s, "marked_at": now}, ensure_ascii=False) for s in sorted(marks)]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def toggle_mark(samples_jsonl: Path, sample_id: str) -> bool:
    """切换标记状态。返回切换后是否已标记。"""
    marks = load_marks(samples_jsonl)
    if sample_id in marks:
        marks.discard(sample_id)
        new_state = False
    else:
        marks.add(sample_id)
        new_state = True
    save_marks(samples_jsonl, marks)
    return new_state


def add_marks(samples_jsonl: Path, sample_ids: list[str]) -> int:
    """批量添加标记（用于"标记整三联组"）。返回新增数量。"""
    marks = load_marks(samples_jsonl)
    before = len(marks)
    marks.update(sample_ids)
    save_marks(samples_jsonl, marks)
    return len(marks) - before


def remove_marks(samples_jsonl: Path, sample_ids: list[str]) -> int:
    """批量取消标记。返回实际移除数量。"""
    marks = load_marks(samples_jsonl)
    before = len(marks)
    for sid in sample_ids:
        marks.discard(sid)
    save_marks(samples_jsonl, marks)
    return before - len(marks)


def clear_marks(samples_jsonl: Path) -> int:
    """清空全部标记，返回原标记数。"""
    n = len(load_marks(samples_jsonl))
    save_marks(samples_jsonl, set())
    return n


def export_marked(samples_jsonl: Path, out_filename: str | None = None) -> tuple[Path, int]:
    """从 samples_jsonl 中筛选已标记的样本，输出到同目录的 out_filename。

    out_filename 默认按时间戳生成：verified.marked_YYYYMMDD_HHMMSS.jsonl
    返回 (输出路径, 导出条数)。
    """
    marks = load_marks(samples_jsonl)
    if out_filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"verified.marked_{ts}.jsonl"
    out_path = samples_jsonl.parent / out_filename
    if not marks:
        out_path.write_text("", encoding="utf-8")
        return out_path, 0

    selected: list[dict] = []
    for line in samples_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("sample_id") in marks:
            selected.append(obj)

    out_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in selected) + "\n",
        encoding="utf-8",
    )
    return out_path, len(selected)


# ============================================================
# 评审意见（comments）相关函数
# ============================================================

def _comments_path(samples_jsonl: Path) -> Path:
    """根据样本文件路径推导对应的 _comments_{stem}.jsonl 路径（按数据文件隔离）。"""
    stem = samples_jsonl.stem
    p = samples_jsonl.parent / f"_comments_{stem}.jsonl"
    # 迁移兼容：旧版只有单一 _comments.jsonl，首次使用新路径时自动迁移
    if not p.exists():
        legacy = samples_jsonl.parent / "_comments.jsonl"
        if legacy.exists() and stem == "verified":
            legacy.rename(p)
    return p


def load_comments(samples_jsonl: Path) -> dict[str, dict]:
    """读取 sample_id → {comment, updated_at} 映射。文件不存在则返回空 dict。
    同一 sample_id 后写入覆盖前者（按文件出现顺序，取最后一条）。
    """
    p = _comments_path(samples_jsonl)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            sid = obj.get("sample_id")
            if sid:
                out[sid] = {
                    "comment": obj.get("comment", ""),
                    "updated_at": obj.get("updated_at", ""),
                }
        except json.JSONDecodeError:
            continue
    return out


def _save_comments(samples_jsonl: Path, comments: dict[str, dict]) -> None:
    """全量重写 _comments.jsonl。"""
    p = _comments_path(samples_jsonl)
    lines = []
    for sid in sorted(comments.keys()):
        c = comments[sid]
        lines.append(json.dumps({
            "sample_id": sid,
            "comment": c.get("comment", ""),
            "updated_at": c.get("updated_at", ""),
        }, ensure_ascii=False))
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_comment(samples_jsonl: Path, sample_id: str, comment: str) -> None:
    """写入/更新一条评审意见。空字符串视为删除。"""
    comments = load_comments(samples_jsonl)
    text = (comment or "").strip()
    if not text:
        comments.pop(sample_id, None)
    else:
        comments[sample_id] = {
            "comment": text,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    _save_comments(samples_jsonl, comments)


def delete_comment(samples_jsonl: Path, sample_id: str) -> bool:
    """删除一条评审意见。返回是否真的删了。"""
    comments = load_comments(samples_jsonl)
    if sample_id in comments:
        del comments[sample_id]
        _save_comments(samples_jsonl, comments)
        return True
    return False


def clear_comments(samples_jsonl: Path) -> int:
    """清空全部评审意见，返回原数量。"""
    n = len(load_comments(samples_jsonl))
    _save_comments(samples_jsonl, {})
    return n


def export_comments(samples_jsonl: Path, out_filename: str | None = None) -> tuple[Path, int]:
    """把所有评审意见连同样本上下文导出，便于后续 LLM 改写或人工修订。

    out_filename 默认时间戳: reviews_YYYYMMDD_HHMMSS.jsonl
    导出字段：sample_id / global_no / comment / updated_at / question / answer /
              value_level / topic / evaluation_dimension / primary_metric
    """
    comments = load_comments(samples_jsonl)
    if out_filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"reviews_{ts}.jsonl"
    out_path = samples_jsonl.parent / out_filename

    if not comments:
        out_path.write_text("", encoding="utf-8")
        return out_path, 0

    # 读源样本，按 sample_id 索引
    src_rows = {}
    src_order = {}
    for i, line in enumerate(samples_jsonl.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = obj.get("sample_id")
        if sid:
            src_rows[sid] = obj
            src_order[sid] = i + 1

    # 拼输出
    selected = []
    for sid, c in comments.items():
        if sid not in src_rows:
            continue  # 样本可能已被删除
        s = src_rows[sid]
        # 提取 question/answer
        arts = (s.get("output", {}) or {}).get("artifacts", []) or []
        question = next((a.get("value", "") for a in arts if a.get("role") == "question"), "")
        answer = next((a.get("value", "") for a in arts if a.get("role") == "answer"), "")
        meta = s.get("metadata", {}) or {}
        selected.append({
            "sample_id": sid,
            "global_no": src_order.get(sid),
            "comment": c.get("comment", ""),
            "updated_at": c.get("updated_at", ""),
            "question": question,
            "answer": answer,
            "value_level": meta.get("value_level"),
            "topic": meta.get("topic"),
            "evaluation_dimension": meta.get("evaluation_dimension"),
            "primary_metric": meta.get("primary_metric"),
        })

    # 按全局序号排序
    selected.sort(key=lambda r: r.get("global_no") or 0)

    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + "\n",
        encoding="utf-8",
    )
    return out_path, len(selected)
