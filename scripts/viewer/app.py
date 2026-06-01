"""EvoBench Sample Viewer - Streamlit 主入口

启动: cd EvoBench && streamlit run scripts/viewer/app.py
依赖: pip install streamlit pandas  (不依赖 benchmark 包)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 保证包内相对 import 可用
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from viewer.data import (
    KPIStats,
    SampleRow,
    ValueKPIStats,
    _value_group_key,
    compute_kpi,
    compute_value_kpi,
    group_triplets,
    group_value_triplets,
    load_samples,
)
from viewer.components import (
    render_evidence,
    render_kpi,
    render_quality,
    render_risk_annotation,
    render_triplet,
    render_value_annotations,
    render_value_kpi,
    render_value_triplet,
)
from viewer.marks import (
    add_marks,
    clear_comments,
    clear_marks,
    delete_comment,
    export_comments,
    export_marked,
    load_comments,
    load_marks,
    remove_marks,
    save_comment,
    toggle_mark,
)

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(page_title="EvoBench Sample Viewer", layout="wide", page_icon="🔬")
st.title("🔬 EvoBench · Sample Viewer")

# ============================================================
# Sidebar - 数据源 + 过滤
# ============================================================

PROJECT_ROOT = _SCRIPT_DIR.parents[1]
_SKILL_PRESETS = {
    "benchmark_qa": PROJECT_ROOT / "data" / "samples" / "benchmark_qa" / "verified.jsonl",
    "value_qa": PROJECT_ROOT / "data" / "samples" / "value_qa" / "verified.jsonl",
}
DEFAULT_PATH = _SKILL_PRESETS["benchmark_qa"]

# QC 评分模式数据路径（按 round_id 拼）
def _qc_paths(round_id: int) -> tuple:
    base = PROJECT_ROOT / "data" / "qc"
    return (
        base / f"round{round_id}_samples.jsonl",
        base / f"round{round_id}_index.json",
        base / f"round{round_id}_scores.jsonl",
    )


with st.sidebar:
    st.header("📂 数据源")
    skill_mode = st.radio(
        "Skill 模式",
        list(_SKILL_PRESETS.keys()) + ["QC 评分"],
        horizontal=True,
        key="skill_mode",
    )

# QC 模式独立分支：不走主表格逻辑
if skill_mode == "QC 评分":
    from viewer.qc_panel import render_qc_page

    # round 切换：自动列出 data/qc/ 下已存在的 roundN_samples.jsonl
    qc_dir = PROJECT_ROOT / "data" / "qc"
    available_rounds = sorted(
        int(p.stem.replace("round", "").replace("_samples", ""))
        for p in qc_dir.glob("round*_samples.jsonl")
    )
    if not available_rounds:
        available_rounds = [1]
    with st.sidebar:
        round_id = st.radio(
            "Round",
            available_rounds,
            format_func=lambda r: f"Round {r}",
            horizontal=True,
            key="qc_round_id",
        )
    samples_p, index_p, scores_p = _qc_paths(round_id)
    render_qc_page(samples_p, index_p, scores_p)
    st.stop()

with st.sidebar:
    # 扫描目录下可选的 jsonl 文件（排除 _ 开头的辅助文件），按修改时间倒序（最新在上）
    _data_dir = _SKILL_PRESETS[skill_mode].parent
    _jsonl_files = [
        f.name for f in sorted(
            (f for f in _data_dir.glob("*.jsonl") if not f.name.startswith("_")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
    ]
    _default_idx = _jsonl_files.index("verified.jsonl") if "verified.jsonl" in _jsonl_files else 0
    selected_file = st.selectbox(
        "📂 数据文件",
        _jsonl_files,
        index=_default_idx,
        key=f"data_file_{skill_mode}",
    )
    p = _data_dir / selected_file
    if not p.exists():
        st.error(f"文件不存在: {p}")
        st.stop()

    # 加载（带缓存）
    @st.cache_data
    def _load(path_str: str, mtime: float) -> list[dict]:
        rows = load_samples(path_str)
        return [r.__dict__ for r in rows]  # cache 要求 serializable

    raw_rows = _load(str(p), p.stat().st_mtime)
    all_rows = [SampleRow(**d) for d in raw_rows]

    st.metric("总样本数", len(all_rows))

    # ----- 标记功能（仅 value_qa） -----
    is_value_qa = (skill_mode == "value_qa")
    marks_set: set[str] = load_marks(p) if is_value_qa else set()
    comments_map: dict[str, dict] = load_comments(p) if is_value_qa else {}
    if is_value_qa:
        st.divider()
        st.subheader("⭐ 标记")
        st.caption(f"📎 关联文件: `{selected_file}`")
        c1, c2 = st.columns(2)
        c1.metric("已标记", len(marks_set))
        c2.metric("未标记", len(all_rows) - len(marks_set))

        sel_mark_filter = st.radio(
            "筛选标记状态",
            ["全部", "⭐ 仅已标记", "⚪ 仅未标记"],
            horizontal=True,
            key="mark_filter",
        )

        cexp, cclr = st.columns(2)
        if cexp.button("💾 导出已标记", use_container_width=True, disabled=(len(marks_set) == 0)):
            out_path, n = export_marked(p)
            st.success(f"已导出 {n} 条 → `{out_path.name}`")
        if cclr.button("🗑 清空标记", use_container_width=True, disabled=(len(marks_set) == 0)):
            n = clear_marks(p)
            st.success(f"已清空 {n} 条标记")
            st.rerun()

        st.divider()
        st.subheader("💬 评审意见")
        st.metric("已写意见", len(comments_map))

        sel_comment_filter = st.radio(
            "筛选意见状态",
            ["全部", "💬 仅有意见", "⚫ 仅无意见"],
            horizontal=True,
            key="comment_filter",
        )

        cexp2, cclr2 = st.columns(2)
        if cexp2.button("📝 导出评审意见", use_container_width=True, disabled=(len(comments_map) == 0)):
            out_path, n = export_comments(p)
            st.success(f"已导出 {n} 条 → `{out_path.name}`")
        if cclr2.button("🗑 清空意见", use_container_width=True, disabled=(len(comments_map) == 0)):
            n = clear_comments(p)
            st.success(f"已清空 {n} 条意见")
            st.rerun()
    else:
        sel_mark_filter = "全部"
        sel_comment_filter = "全部"

    st.divider()

    # 多维过滤
    st.header("🔍 过滤")

    if skill_mode == "value_qa":
        all_primaries = sorted(set(r.primary_metric for r in all_rows if r.primary_metric))
        sel_primaries = st.multiselect("一级指标 (primary_metric)", all_primaries)

        # 二级维度候选项联动：选了一级指标时只显示对应的二级维度
        if sel_primaries:
            all_secondaries = sorted(set(
                r.evaluation_dimension for r in all_rows
                if r.evaluation_dimension and r.primary_metric in sel_primaries
            ))
        else:
            all_secondaries = sorted(set(r.evaluation_dimension for r in all_rows if r.evaluation_dimension))
        sel_secondaries = st.multiselect("二级评估维度 (evaluation_dimension)", all_secondaries)

        all_levels = sorted(set(r.value_level for r in all_rows if r.value_level))
        all_patterns = sorted(set(
            r.medium_pattern for r in all_rows if r.value_level == "medium" and r.medium_pattern
        ))
        sel_levels = st.multiselect("value_level", all_levels)
        sel_patterns = st.multiselect("medium_pattern (仅对 medium)", all_patterns)
        sel_types: list[str] = []
        sel_risks: list[str] = []
    else:
        all_types = sorted(set(r.sample_type for r in all_rows if r.sample_type))
        all_risks = sorted(set(r.risk_level for r in all_rows if r.risk_level))
        sel_types = st.multiselect("sample_type", all_types)
        sel_risks = st.multiselect("risk_level", all_risks)
        sel_primaries: list[str] = []
        sel_secondaries: list[str] = []
        sel_levels: list[str] = []
        sel_patterns: list[str] = []

    # topic 候选项联动：根据已选的一级/二级指标动态过滤
    if sel_primaries or sel_secondaries:
        _topic_pool = all_rows
        if sel_primaries:
            _topic_pool = [r for r in _topic_pool if r.primary_metric in sel_primaries]
        if sel_secondaries:
            _topic_pool = [r for r in _topic_pool if r.evaluation_dimension in sel_secondaries]
        all_topics = sorted(set(r.topic for r in _topic_pool if r.topic))
    else:
        all_topics = sorted(set(r.topic for r in all_rows if r.topic))

    sel_topics = st.multiselect("topic", all_topics)
    sel_gate = st.radio("quality_gate", ["全部", "✅ 通过", "❌ 未通过"], horizontal=True)

    search_text = st.text_input("🔎 搜索 Q/A 内容")

    diff_range = st.slider("difficulty 区间", 0.0, 1.0, (0.0, 1.0), step=0.05)
    cov_min = st.slider("最低 evidence_coverage", 0.0, 1.0, 0.0, step=0.05)

# ============================================================
# 过滤
# ============================================================

filtered = all_rows
if sel_types:
    filtered = [r for r in filtered if r.sample_type in sel_types]
if sel_risks:
    filtered = [r for r in filtered if r.risk_level in sel_risks]
if sel_primaries:
    filtered = [r for r in filtered if r.primary_metric in sel_primaries]
if sel_secondaries:
    filtered = [r for r in filtered if r.evaluation_dimension in sel_secondaries]
if sel_levels:
    filtered = [r for r in filtered if r.value_level in sel_levels]
if sel_patterns:
    filtered = [r for r in filtered if r.value_level == "medium" and r.medium_pattern in sel_patterns]
if sel_topics:
    filtered = [r for r in filtered if r.topic in sel_topics]
if sel_gate == "✅ 通过":
    filtered = [r for r in filtered if r.quality_gate]
elif sel_gate == "❌ 未通过":
    filtered = [r for r in filtered if not r.quality_gate]
if search_text:
    q = search_text.lower()
    filtered = [r for r in filtered if q in r.question.lower() or q in r.answer.lower()]
filtered = [r for r in filtered if diff_range[0] <= r.difficulty <= diff_range[1]]
filtered = [r for r in filtered if r.evidence_coverage >= cov_min]

# 标记筛选（仅 value_qa）
if sel_mark_filter == "⭐ 仅已标记":
    filtered = [r for r in filtered if r.sample_id in marks_set]
elif sel_mark_filter == "⚪ 仅未标记":
    filtered = [r for r in filtered if r.sample_id not in marks_set]

# 意见筛选（仅 value_qa）
if sel_comment_filter == "💬 仅有意见":
    filtered = [r for r in filtered if r.sample_id in comments_map]
elif sel_comment_filter == "⚫ 仅无意见":
    filtered = [r for r in filtered if r.sample_id not in comments_map]

# ============================================================
# KPI 区
# ============================================================

if skill_mode == "value_qa":
    render_value_kpi(compute_value_kpi(filtered))
else:
    render_kpi(compute_kpi(filtered))
st.divider()

# ============================================================
# 表格区
# ============================================================

_TYPE_EMOJI_MAP = {"normal": "🟢", "counterfactual": "🟡", "risk": "🔴"}
_LEVEL_EMOJI_MAP = {"high": "🟢", "medium": "🟡", "low": "🔴"}

# 全局序号映射：sample_id → 1-based 行号（基于 all_rows 顺序，过滤后保持稳定）
_global_no = {r.sample_id: i + 1 for i, r in enumerate(all_rows) if r.sample_id}

if skill_mode == "value_qa":
    df = pd.DataFrame([{
        "#": _global_no.get(r.sample_id, "-"),
        "★": "⭐" if r.sample_id in marks_set else "",
        "💬": "💬" if r.sample_id in comments_map else "",
        "doc": r.doc_title or r.topic,
        "primary": r.primary_metric or (r.value_dimension or "-"),
        "secondary": r.evaluation_dimension or "-",
        "level": f"{_LEVEL_EMOJI_MAP.get(r.value_level, '⚪')} {r.value_level}",
        "pattern": r.medium_pattern if r.value_level == "medium" else "-",
        "fact_idx": r.fact_idx,
        "question": r.question[:40],
        "answer": r.answer[:40],
        "cov": f"{r.evidence_coverage:.2f}",
        "diff": f"{r.difficulty:.2f}",
        "gate": "✅" if r.quality_gate else "❌",
        "_idx": i,
    } for i, r in enumerate(filtered)])
else:
    df = pd.DataFrame([{
        "#": _global_no.get(r.sample_id, "-"),
        "topic": r.topic,
        "type": f"{_TYPE_EMOJI_MAP.get(r.sample_type, '⚪')} {r.sample_type}",
        "risk": r.risk_level or "-",
        "question": r.question[:40],
        "answer": r.answer[:40],
        "cov": f"{r.evidence_coverage:.2f}",
        "diff": f"{r.difficulty:.2f}",
        "gate": "✅" if r.quality_gate else "❌",
        "group_id": r.group_id,
        "_idx": i,
    } for i, r in enumerate(filtered)])

if df.empty:
    st.warning("无匹配样本")
    st.stop()

st.subheader(f"📋 样本列表 ({len(filtered)} 条)")
st.caption("👇 点击表格中的任意一行，下方将展示该样本的详情")

# 表格作为唯一选择器（streamlit ≥ 1.28 原生 row selection）
event = st.dataframe(
    df.drop(columns=["_idx"]),
    use_container_width=True,
    height=min(420, 35 * len(df) + 38),
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="sample_table",
)

selected_rows = event.selection.rows if event and event.selection else []
selected_idx = selected_rows[0] if selected_rows else None

st.divider()

# ============================================================
# 详情面板
# ============================================================

if selected_idx is None:
    st.info("👆 请在上方表格中点击一行查看详情")
    st.stop()

if selected_idx >= len(filtered):
    st.info("⚠️ 筛选条件变化，请重新点击表格中的一行")
    st.stop()

row = filtered[selected_idx]

if skill_mode == "value_qa":
    _sel_emoji = _LEVEL_EMOJI_MAP.get(row.value_level, "⚪")
    _crumb = " · ".join(
        x for x in [
            row.primary_metric or row.value_dimension,
            row.evaluation_dimension,
        ] if x
    ) or "?"
    _pattern_tag = f" · pattern={row.medium_pattern}" if (row.value_level == "medium" and row.medium_pattern) else ""
    st.markdown(
        f"### 🔍 当前选中：{_sel_emoji} **{row.value_level}** · "
        f"`{_crumb}`{_pattern_tag} · {row.doc_title or row.topic} "
        f"· {row.question[:50]}{'...' if len(row.question) > 50 else ''}"
    )

    # ----- 标记按钮区 -----
    _is_marked = row.sample_id in marks_set
    _global_idx = _global_no.get(row.sample_id, "-")
    mc1, mc2, mc3 = st.columns([1, 1, 2])
    _btn_label = "✗ 取消标记当前条" if _is_marked else "✓ 标记当前条为合格"
    if mc1.button(f"{_btn_label}（# {_global_idx}）", use_container_width=True, key="btn_toggle_one"):
        toggle_mark(p, row.sample_id)
        st.rerun()

    # 找出同三联组所有 sample_id
    _v_groups_all = group_value_triplets(all_rows)
    _gkey = _value_group_key(row)
    _trip_ids = [r.sample_id for r in _v_groups_all.get(_gkey, []) if r.sample_id]
    _trip_marked = sum(1 for sid in _trip_ids if sid in marks_set)
    _trip_total = len(_trip_ids)
    if _trip_total > 1:
        _all_in = (_trip_marked == _trip_total)
        _trip_label = (
            f"✗ 取消同三联组（{_trip_total} 条全部）"
            if _all_in
            else f"✓ 标记同三联组（已标 {_trip_marked}/{_trip_total}）"
        )
        if mc2.button(_trip_label, use_container_width=True, key="btn_toggle_triplet"):
            if _all_in:
                remove_marks(p, _trip_ids)
            else:
                add_marks(p, _trip_ids)
            st.rerun()
    mc3.markdown(f"**当前状态**：{'⭐ 已标记' if _is_marked else '⚪ 未标记'}  ·  `sample_id={row.sample_id}`")

    # ----- 评审意见区 -----
    _existing_comment = comments_map.get(row.sample_id, {}).get("comment", "")
    _existing_ts = comments_map.get(row.sample_id, {}).get("updated_at", "")
    with st.expander(
        f"💬 评审意见（可选） {'· 已填写' if _existing_comment else '· 暂无'}",
        expanded=bool(_existing_comment),
    ):
        if _existing_ts:
            st.caption(f"上次更新：{_existing_ts}")
        _input_key = f"comment_input_{row.sample_id}"
        comment_text = st.text_area(
            "对该样本的改进建议（保留为空表示不留意见）",
            value=_existing_comment,
            height=110,
            placeholder="例如：answer 末尾出现『这种说法』指代不清，建议替换为完整名词主语；或：medium 段对 '敬意' 缺失体现不足。",
            key=_input_key,
        )
        bc1, bc2, bc3 = st.columns([1, 1, 2])
        if bc1.button("💾 保存意见", use_container_width=True, key=f"btn_save_cmt_{row.sample_id}"):
            save_comment(p, row.sample_id, comment_text)
            st.success("已保存")
            st.rerun()
        if _existing_comment and bc2.button("🗑 删除意见", use_container_width=True, key=f"btn_del_cmt_{row.sample_id}"):
            delete_comment(p, row.sample_id)
            st.success("已删除")
            st.rerun()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔀 三联组对照",
        "📑 Evidence 高亮",
        "📊 Quality 信号",
        "🏷️ Value 标注",
        "{ } Raw JSON",
    ])

    with tab1:
        v_groups = group_value_triplets(filtered)
        gkey = _value_group_key(row)
        if gkey in v_groups:
            render_value_triplet(v_groups[gkey])
        else:
            all_v_groups = group_value_triplets(all_rows)
            if gkey in all_v_groups:
                st.caption("⚠️ 部分同组样本不在当前过滤结果中，已从全量加载")
                render_value_triplet(all_v_groups[gkey])
            else:
                st.info("无三联组")

    with tab2:
        render_evidence(row)

    with tab3:
        render_quality(row)

    with tab4:
        render_value_annotations(row)

    with tab5:
        st.json(row.raw)
else:
    # 当前选中样本简要标识
    _sel_emoji = _TYPE_EMOJI_MAP.get(row.sample_type, "⚪")
    st.markdown(
        f"### 🔍 当前选中：{_sel_emoji} **{row.sample_type}** "
        f"· `{row.topic}` · {row.question[:50]}{'...' if len(row.question) > 50 else ''}"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔀 Triplet 对照",
        "📑 Evidence 高亮",
        "📊 Quality 信号",
        "⚠️ Risk 标注",
        "{ } Raw JSON",
    ])

    with tab1:
        groups = group_triplets(filtered)
        if row.group_id and row.group_id in groups:
            render_triplet(groups[row.group_id])
        else:
            # 如果当前过滤条件隐藏了同组样本，从全量拉
            all_groups = group_triplets(all_rows)
            if row.group_id and row.group_id in all_groups:
                st.caption("⚠️ 部分同组样本不在当前过滤结果中，已从全量加载")
                render_triplet(all_groups[row.group_id])
            else:
                st.info("无 group_id 或独立样本")

    with tab2:
        render_evidence(row)

    with tab3:
        render_quality(row)

    with tab4:
        render_risk_annotation(row)

    with tab5:
        st.json(row.raw)
