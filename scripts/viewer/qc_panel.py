"""qc_panel.py - QC 评分页面渲染（Streamlit）。

入口: render_qc_page(samples_path, index_path, scores_path)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from viewer.qc_data import (
    ISSUE_TAGS,
    TripletScore,
    compute_progress,
    load_index,
    load_samples,
    load_scores,
    samples_by_triplet,
    save_scores,
    upsert_score,
)

# ---------- 工具 ----------


def _level_emoji(lv: str) -> str:
    return {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(lv, "⚪")


def _get_qa(sample: dict[str, Any]) -> tuple[str, str]:
    out = sample.get("output") or {}
    q, a = "", ""
    for art in out.get("artifacts") or []:
        if art.get("key") == "x" or art.get("role") == "question":
            q = art.get("value", "")
        elif art.get("key") == "y" or art.get("role") == "answer":
            a = art.get("value", "")
    return q, a


def _get_evidence(sample: dict[str, Any]) -> str:
    ev = (sample.get("evidence") or [{}])[0]
    return ev.get("text") or ""


def _annotation_brief(annotations: list[dict]) -> list[str]:
    out = []
    for a in annotations or []:
        polarity = a.get("polarity", "?")
        emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}.get(polarity, "❓")
        text = (a.get("text") or "")[:25]
        label = a.get("label", "?")
        out.append(f"{emoji} `{label}` · {text}{'…' if len(a.get('text', '')) > 25 else ''}")
    return out


# ---------- 单档卡片 ----------


def _render_level_card(level: str, sample: dict[str, Any]) -> None:
    emoji = _level_emoji(level)
    q, a = _get_qa(sample)
    ev = _get_evidence(sample)
    meta = sample.get("metadata") or {}
    annotations = sample.get("value_annotations") or []
    medium_pattern = meta.get("medium_pattern", "") if level == "medium" else ""

    title = f"{emoji} **{level.upper()}**"
    if medium_pattern:
        title += f" · `pattern={medium_pattern}`"
    st.markdown(title)
    st.markdown(f"**Q**: {q}")
    st.markdown(f"**A**: {a}")
    with st.expander("Evidence + Annotations", expanded=False):
        st.caption("**Evidence**")
        st.text(ev or "(空)")
        st.caption(f"**Value Annotations** ({len(annotations)})")
        if annotations:
            for line in _annotation_brief(annotations):
                st.markdown(f"- {line}")
        else:
            st.markdown("- (无)")


# ---------- 评分控件 ----------

_D1_LABELS = {0: "未评", 1: "1 几乎纯客观", 2: "2 部分有价值", 3: "3 明确价值含义"}
_D2_LABELS = {0: "未评", 1: "1 三档雷同", 2: "2 区分但不强", 3: "3 三档语义清晰对立"}
_D3_LABELS = {0: "未评", 1: "1 套话/空泛", 2: "2 立场对但偏抽象", 3: "3 具体引用 + 立场鲜明"}
_D4_LABELS = {0: "未评 / 无 annotation", 1: "1 严重错误", 2: "2 部分偏移", 3: "3 全部正确"}


def _radio_score(label: str, key: str, current: int, opts: dict[int, str]) -> int:
    options = list(opts.keys())
    idx = options.index(current) if current in options else 0
    chosen = st.radio(label, options, index=idx, format_func=lambda v: opts[v], key=key, horizontal=True)
    return int(chosen)


def _render_score_form(score: TripletScore, triplet_idx: int) -> TripletScore:
    """渲染评分控件，返回更新后的 TripletScore。"""
    # 完整度提示
    missing: list[str] = []
    if score.d1_fact_judgable < 1:
        missing.append("D1")
    if score.d2_triplet_discrim < 1:
        missing.append("D2")
    for lv in ("high", "medium", "low"):
        if score.d3_answer_fullness.get(lv, 0) < 1:
            missing.append(f"D3·{lv}")
    if missing:
        st.warning(f"⚠️ 本组还缺：{', '.join(missing)}（D4 允许 0）")
    else:
        st.success("✅ 本组所有必填项已就绪")

    st.markdown("#### 📝 评分")
    col1, col2 = st.columns(2)
    with col1:
        d1 = _radio_score("D1 fact 价值可判定性 (整组)", f"d1_{triplet_idx}", score.d1_fact_judgable, _D1_LABELS)
    with col2:
        d2 = _radio_score("D2 三档语义区分度 (整组)", f"d2_{triplet_idx}", score.d2_triplet_discrim, _D2_LABELS)

    st.markdown("**D3 answer 充实度（每档独立）**")
    c_h, c_m, c_l = st.columns(3)
    with c_h:
        d3_h = _radio_score("🟢 high", f"d3h_{triplet_idx}", score.d3_answer_fullness.get("high", 0), _D3_LABELS)
    with c_m:
        d3_m = _radio_score("🟡 medium", f"d3m_{triplet_idx}", score.d3_answer_fullness.get("medium", 0), _D3_LABELS)
    with c_l:
        d3_l = _radio_score("🔴 low", f"d3l_{triplet_idx}", score.d3_answer_fullness.get("low", 0), _D3_LABELS)

    st.markdown("**D4 annotation 准确性（每档独立）**")
    c_h2, c_m2, c_l2 = st.columns(3)
    with c_h2:
        d4_h = _radio_score("🟢 high", f"d4h_{triplet_idx}", score.d4_annotation_correct.get("high", 0), _D4_LABELS)
    with c_m2:
        d4_m = _radio_score("🟡 medium", f"d4m_{triplet_idx}", score.d4_annotation_correct.get("medium", 0), _D4_LABELS)
    with c_l2:
        d4_l = _radio_score("🔴 low", f"d4l_{triplet_idx}", score.d4_annotation_correct.get("low", 0), _D4_LABELS)

    issue_tags = st.multiselect(
        "🏷️ Issue 标签 (可多选)", ISSUE_TAGS, default=score.issue_tags, key=f"tags_{triplet_idx}"
    )
    note = st.text_area("✍️ 备注 (可选)", value=score.note, key=f"note_{triplet_idx}", height=80)

    return TripletScore(
        triplet_id=score.triplet_id,
        d1_fact_judgable=d1,
        d2_triplet_discrim=d2,
        d3_answer_fullness={"high": d3_h, "medium": d3_m, "low": d3_l},
        d4_annotation_correct={"high": d4_h, "medium": d4_m, "low": d4_l},
        issue_tags=issue_tags,
        note=note,
        reviewer=score.reviewer,
        ts=score.ts,
    )


# ---------- 顶层入口 ----------


def render_qc_page(samples_path: Path, index_path: Path, scores_path: Path) -> None:
    st.title("✏️ QC 评分 · Round 1")
    st.caption("基于 30 条分层抽样样本（10 三联组）做人工质检，评分自动落到 round1_scores.jsonl")

    # 数据准备
    if not samples_path.exists() or not index_path.exists():
        st.error(
            f"❌ 缺少数据文件：\n- {samples_path}\n- {index_path}\n"
            "请先运行 `python scripts/sample_qc.py` 生成抽样集。"
        )
        return

    index = load_index(index_path)
    samples = load_samples(samples_path)
    triplets = samples_by_triplet(index, samples)

    # session 缓存 scores（避免每次 rerender 重读）
    cache_key = f"qc_scores_{scores_path}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = load_scores(scores_path)
    scores: dict[str, TripletScore] = st.session_state[cache_key]

    # 顶部 reviewer 输入
    reviewer = st.sidebar.text_input("👤 评分员", value=st.session_state.get("qc_reviewer", "shiyewang"), key="qc_reviewer")

    # 进度面板
    progress = compute_progress(index, scores)
    pct = progress.completed / progress.total_triplets if progress.total_triplets else 0.0
    st.progress(pct, text=f"已评 {progress.completed} / {progress.total_triplets} 三联组（{pct * 100:.0f}%）")
    if progress.completed > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("D1 avg", f"{progress.avg_d1:.2f}")
        c2.metric("D2 avg", f"{progress.avg_d2:.2f}")
        c3.metric("D3 avg", f"{progress.avg_d3:.2f}")
        c4.metric("D4 avg", f"{progress.avg_d4:.2f}")

    # 未完成清单（明确告诉用户每个未完成组缺哪些项）
    incomplete_lines: list[str] = []
    for i, entry in enumerate(index, 1):
        tid = entry["triplet_id"]
        sc = scores.get(tid)
        miss: list[str] = []
        if sc is None:
            miss.append("整组未评")
        else:
            if sc.d1_fact_judgable < 1:
                miss.append("D1")
            if sc.d2_triplet_discrim < 1:
                miss.append("D2")
            for lv in ("high", "medium", "low"):
                if sc.d3_answer_fullness.get(lv, 0) < 1:
                    miss.append(f"D3·{lv}")
        if miss:
            incomplete_lines.append(
                f"- **#{i}** `{entry['evaluation_dimension']}` · {entry.get('topic', '?')} → 缺：**{', '.join(miss)}**"
            )
    if incomplete_lines:
        st.error(
            "❗ 未完成清单（共 {n} 组）：\n\n{body}\n\n"
            "👉 用上方下拉跳到对应组，把这些项打分后再保存。".format(
                n=len(incomplete_lines), body="\n".join(incomplete_lines)
            )
        )
    else:
        st.success("🎉 全部 10 组评分已完整！")
    st.divider()

    # 三联组选择器
    def _label(i: int, entry: dict) -> str:
        tid = entry["triplet_id"]
        done = "✅" if (tid in scores and scores[tid].is_complete) else "⬜"
        return f"#{i + 1} [{done}] {entry['evaluation_dimension']} · {entry.get('topic', '?')} · {entry.get('medium_pattern', '?')}"

    # 在 widget 实例化前消化挂起的跳转意图（避免 StreamlitAPIException）
    if "qc_pending_idx" in st.session_state:
        st.session_state["qc_sel_idx"] = st.session_state.pop("qc_pending_idx")

    sel_idx = st.selectbox(
        "选择三联组",
        list(range(len(index))),
        format_func=lambda i: _label(i, index[i]),
        key="qc_sel_idx",
    )
    entry = index[sel_idx]
    tid = entry["triplet_id"]
    triplet = triplets.get(tid, {})

    # 元信息
    st.markdown(
        f"**维度**: `{entry['evaluation_dimension']}` （{entry.get('primary_metric', '?')}） · "
        f"**topic**: `{entry.get('topic', '?')}` · "
        f"**medium_pattern**: `{entry.get('medium_pattern', '-')}`"
    )
    with st.expander("📌 fact 原文", expanded=False):
        st.write(entry.get("fact", ""))

    # 三档样本
    col_h, col_m, col_l = st.columns(3)
    with col_h:
        _render_level_card("high", triplet.get("high", {}))
    with col_m:
        _render_level_card("medium", triplet.get("medium", {}))
    with col_l:
        _render_level_card("low", triplet.get("low", {}))

    st.divider()

    # 评分表单
    cur_score = scores.get(tid) or TripletScore(triplet_id=tid, reviewer=reviewer)
    cur_score.reviewer = reviewer
    new_score = _render_score_form(cur_score, sel_idx)

    # 保存按钮
    cs1, cs2, cs3 = st.columns([1, 1, 4])
    with cs1:
        if st.button("💾 保存本组评分", type="primary", key=f"save_{sel_idx}"):
            upsert_score(scores, new_score)
            save_scores(scores_path, scores)
            st.success(f"✅ 已保存 → {scores_path}")
            st.rerun()
    with cs2:
        # 找下一个未完成的组（从 sel_idx+1 开始环绕一圈，跳过 is_complete 的）
        def _next_incomplete(start: int) -> int | None:
            n = len(index)
            for offset in range(1, n + 1):
                cand = (start + offset) % n
                tid_c = index[cand]["triplet_id"]
                sc = scores.get(tid_c)
                if sc is None or not sc.is_complete:
                    return cand
            return None

        if st.button("⏭ 保存并跳到下一未完成组", key=f"save_next_{sel_idx}"):
            upsert_score(scores, new_score)
            save_scores(scores_path, scores)
            # 用最新 scores 字典找未完成组（含刚保存的本组：若仍不完整会被再次定位到）
            nxt = _next_incomplete(sel_idx)
            if nxt is None:
                st.success("🎉 全部 10 组评分已完整！可以跑 qc_report 了")
            else:
                st.session_state["qc_pending_idx"] = nxt
                st.rerun()
    with cs3:
        if cur_score.ts:
            st.caption(f"上次保存: {cur_score.ts}  reviewer={cur_score.reviewer}")
