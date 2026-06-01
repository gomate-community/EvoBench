"""components.py - Streamlit 渲染组件，与业务代码零耦合。

每个函数对应 viewer 的一个 UI 区块。
"""
from __future__ import annotations

import difflib
import html as html_mod
import re
from typing import Any

import streamlit as st

from .data import KPIStats, SampleRow, ValueKPIStats

# ============================================================
# 风险词典（兜底高亮用）
# ============================================================
_RISK_LEXICON: set[str] = {
    "低等", "愚昧", "野蛮", "落后", "懒惰", "肮脏", "劣等",
    "难管理", "管不了", "管理麻烦", "难融合", "难教化",
    "低素质", "没文化", "脏乱差", "不开化", "蛮荒",
    "笨", "蠢", "废物", "垃圾", "劣根性",
    "暴力", "血腥", "杀", "打死", "灭族",
    "偷", "骗", "不守法", "违法", "犯罪",
    "下贱", "卑鄙", "恶心", "变态",
}


# ============================================================
# KPI 卡片
# ============================================================

_TYPE_CHIP_COLOR = {
    "normal": ("#22c55e", "🟢"),
    "counterfactual": ("#eab308", "🟡"),
    "risk": ("#ef4444", "🔴"),
}
_RISK_CHIP_COLOR = {
    "high": ("#dc2626", "🔴"),
    "medium": ("#f97316", "🟠"),
    "low": ("#facc15", "🟡"),
}


def _render_chips(label: str, dist: dict, color_map: dict) -> str:
    """把分布字典渲染成横排彩色 chip。"""
    if not dist:
        body = '<span style="color:#94a3b8">—</span>'
    else:
        chips = []
        # 按 color_map 顺序，没在 color_map 的排后面
        order = list(color_map.keys())
        ordered = sorted(dist.items(), key=lambda kv: (order.index(kv[0]) if kv[0] in order else 99, kv[0]))
        for k, v in ordered:
            color, emoji = color_map.get(k, ("#64748b", "⚪"))
            chips.append(
                f'<span style="display:inline-block;background:{color}20;color:{color};'
                f'border:1px solid {color}55;border-radius:12px;padding:2px 10px;'
                f'margin-right:6px;font-size:0.85em;font-weight:500">'
                f"{emoji} {k} <b>{v}</b></span>"
            )
        body = "".join(chips)
    return (
        f'<div style="margin-bottom:8px">'
        f'<div style="color:#64748b;font-size:0.78em;margin-bottom:4px">{label}</div>'
        f"<div>{body}</div></div>"
    )


def _kpi_card(label: str, value: str, hint: str = "", accent: str = "#0f172a") -> str:
    """统一尺寸的紧凑 KPI 卡片。"""
    hint_html = (
        f'<div style="color:#94a3b8;font-size:0.7em;margin-top:2px">{hint}</div>'
        if hint
        else ""
    )
    return (
        f'<div style="padding:10px 14px;border:1px solid #e2e8f0;border-radius:10px;'
        f'background:#fafafa;min-height:74px;display:flex;flex-direction:column;'
        f'justify-content:center">'
        f'<div style="color:#64748b;font-size:0.78em;letter-spacing:0.02em">{label}</div>'
        f'<div style="color:{accent};font-size:1.35em;font-weight:600;line-height:1.2;'
        f'margin-top:2px">{value}</div>'
        f"{hint_html}</div>"
    )


def render_kpi(kpi: KPIStats) -> None:
    """顶部 KPI 区：2 行 × 4 列紧凑卡片 + 底部双列分布。"""

    # ─── 阈值上色辅助 ───
    def _ok_color(val: float, good: float = 0.9, warn: float = 0.7) -> str:
        if val >= good:
            return "#16a34a"  # 绿
        if val >= warn:
            return "#d97706"  # 橙
        return "#dc2626"  # 红

    cards = [
        _kpi_card("样本总数", f"{kpi.total}"),
        _kpi_card(
            "Triplet 完整率",
            f"{kpi.triplet_complete_rate:.0%}",
            accent=_ok_color(kpi.triplet_complete_rate),
        ),
        _kpi_card(
            "质量门通过率",
            f"{kpi.quality_gate_pass_rate:.0%}",
            accent=_ok_color(kpi.quality_gate_pass_rate, 0.95, 0.8),
        ),
        _kpi_card("Topic 覆盖", f"{kpi.topic_count}"),
        _kpi_card(
            "Answer 唯一率",
            f"{kpi.answer_unique_rate:.0%}",
            accent=_ok_color(kpi.answer_unique_rate, 0.9, 0.7),
        ),
        _kpi_card(
            "avg Evidence 覆盖",
            f"{kpi.avg_evidence_coverage:.2f}",
            accent=_ok_color(kpi.avg_evidence_coverage, 0.8, 0.5),
        ),
        _kpi_card("avg Difficulty", f"{kpi.avg_difficulty:.2f}"),
        _kpi_card("avg Q 字数", f"{kpi.avg_question_len:.0f}", hint="字符"),
    ]

    # 2 行 × 4 列
    for row_start in (0, 4):
        cols = st.columns(4, gap="small")
        for i, col in enumerate(cols):
            with col:
                st.markdown(cards[row_start + i], unsafe_allow_html=True)

    st.write("")  # 间距

    # 分布行：左 类型分布 | 右 Risk 分布
    dcols = st.columns(2, gap="small")
    with dcols[0]:
        st.markdown(
            _render_chips("类型分布", kpi.type_dist, _TYPE_CHIP_COLOR),
            unsafe_allow_html=True,
        )
    with dcols[1]:
        st.markdown(
            _render_chips("Risk 分布", kpi.risk_dist, _RISK_CHIP_COLOR),
            unsafe_allow_html=True,
        )


# ============================================================
# Triplet 三联组对照
# ============================================================

_TYPE_ORDER = {"normal": 0, "counterfactual": 1, "risk": 2}
_TYPE_EMOJI = {"normal": "🟢", "counterfactual": "🟡", "risk": "🔴"}
_TYPE_COLOR = {"normal": "#dcfce7", "counterfactual": "#fef9c3", "risk": "#fee2e2"}


def render_triplet(group: list[SampleRow]) -> None:
    """三联组并排渲染。"""
    sorted_group = sorted(group, key=lambda r: _TYPE_ORDER.get(r.sample_type, 9))
    cols = st.columns(len(sorted_group))

    # 找 normal answer 用于 diff 兜底
    normal_answer = ""
    for r in sorted_group:
        if r.sample_type == "normal":
            normal_answer = r.answer
            break

    for col, row in zip(cols, sorted_group):
        with col:
            emoji = _TYPE_EMOJI.get(row.sample_type, "⚪")
            st.markdown(f"### {emoji} {row.sample_type}")
            st.markdown(f"**Q:** {row.question}")

            # answer 渲染（risk 高亮）
            if row.sample_type == "risk":
                highlighted = _highlight_risk_answer(row, normal_answer)
                st.markdown(f"**A:** {highlighted}", unsafe_allow_html=True)
            else:
                st.markdown(f"**A:** {row.answer}")

            # 辅助信息
            if row.entity_replaced:
                st.caption(f"🔄 entity_replaced: `{row.entity_replaced}`")
            if row.risk_level:
                st.caption(f"⚠️ risk_level: **{row.risk_level}**")
            if row.risk_reason:
                st.caption(f"💬 {row.risk_reason}")

            st.caption(f"difficulty={row.difficulty:.2f}  cov={row.evidence_coverage:.2f}")


# ============================================================
# Risk 高亮（三层策略）
# ============================================================

def _highlight_risk_answer(row: SampleRow, normal_answer: str) -> str:
    """对 risk answer 做风险词高亮，三层兜底。"""
    answer = row.answer
    if not answer:
        return ""

    # 优先层：metadata.risk_phrases（结构化 span）
    if row.risk_phrases:
        return _highlight_by_phrases(answer, row.risk_phrases)

    # 第二层：与 normal answer diff
    if normal_answer:
        result = _highlight_by_diff(answer, normal_answer)
        if result != html_mod.escape(answer):
            return result

    # 第三层：词典匹配
    return _highlight_by_lexicon(answer)


def _highlight_by_phrases(answer: str, phrases: list[dict]) -> str:
    """按 risk_phrases 的 start/end 高亮。"""
    # 收集所有区间
    spans: list[tuple[int, int, str, str]] = []
    for p in phrases:
        if p.get("field", "answer") != "answer":
            continue
        s, e = p.get("start", -1), p.get("end", -1)
        text = p.get("text", "")
        reason = p.get("reason", "")
        cat = p.get("category", "")
        if s >= 0 and e > s and e <= len(answer):
            spans.append((s, e, cat, reason))
        elif text and text in answer:
            # span 不对，回退到 find
            idx = answer.find(text)
            if idx >= 0:
                spans.append((idx, idx + len(text), cat, reason))

    if not spans:
        return html_mod.escape(answer)

    # 按起点排序，渲染
    spans.sort(key=lambda x: x[0])
    parts: list[str] = []
    cursor = 0
    for s, e, cat, reason in spans:
        if s > cursor:
            parts.append(html_mod.escape(answer[cursor:s]))
        tooltip = f"{cat}: {reason}" if reason else cat
        parts.append(
            f'<mark style="background:#fca5a5" title="{html_mod.escape(tooltip)}">'
            f"{html_mod.escape(answer[s:e])}</mark>"
        )
        cursor = e
    if cursor < len(answer):
        parts.append(html_mod.escape(answer[cursor:]))
    return "".join(parts)


def _highlight_by_diff(risk_answer: str, normal_answer: str) -> str:
    """diff 兜底：risk answer 相比 normal 多出的字段高亮。"""
    matcher = difflib.SequenceMatcher(None, normal_answer, risk_answer)
    parts: list[str] = []
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        segment = risk_answer[j1:j2]
        if tag == "equal":
            parts.append(html_mod.escape(segment))
        elif tag in ("insert", "replace"):
            parts.append(
                f'<mark style="background:#fdba74" title="diff: risk 新增/替换">'
                f"{html_mod.escape(segment)}</mark>"
            )
    return "".join(parts)


def _highlight_by_lexicon(answer: str) -> str:
    """词典兜底：扫描敏感词。"""
    escaped = html_mod.escape(answer)
    for word in sorted(_RISK_LEXICON, key=len, reverse=True):
        ew = html_mod.escape(word)
        if ew in escaped:
            escaped = escaped.replace(
                ew,
                f'<mark style="background:#fecaca" title="词典命中">{ew}</mark>',
            )
    return escaped


# ============================================================
# Evidence 文档高亮
# ============================================================

def render_evidence(row: SampleRow) -> None:
    """文档全文 + evidence 区间高亮。"""
    content = row.doc_content
    if not content:
        st.info("无文档内容")
        return

    st.caption(f"📄 {row.doc_title} · {row.doc_source_type} · {row.doc_url}")

    if row.evidence_start >= 0 and row.evidence_end > row.evidence_start:
        s, e = row.evidence_start, row.evidence_end
        before = html_mod.escape(content[:s])
        mid = html_mod.escape(content[s:e])
        after = html_mod.escape(content[e:])
        html_str = (
            f'<div style="white-space:pre-wrap;font-size:0.85em;line-height:1.6">'
            f"{before}"
            f'<mark style="background:#bfdbfe;padding:2px 0" '
            f'title="evidence (confidence={row.evidence_confidence:.2f})">{mid}</mark>'
            f"{after}</div>"
        )
        st.markdown(html_str, unsafe_allow_html=True)
    else:
        st.text(content[:2000])

    if row.evidence_text:
        st.caption(f"🔍 evidence: \"{row.evidence_text}\" (conf={row.evidence_confidence:.2f})")


# ============================================================
# Quality 信号
# ============================================================

def render_quality(row: SampleRow) -> None:
    """质量信号条形图。"""
    metrics = [
        ("Evidence Coverage", row.evidence_coverage),
        ("Answerability", row.answerability),
        ("Clarity", row.clarity),
        ("Novelty", row.novelty),
        ("Difficulty", row.difficulty),
        ("Leakage Risk", row.leakage_risk),
        ("Ambiguity Risk", row.ambiguity_risk),
    ]
    for label, val in metrics:
        col1, col2 = st.columns([3, 1])
        col1.progress(min(val, 1.0), text=label)
        col2.write(f"{val:.2f}")

    gate_icon = "✅" if row.quality_gate else "❌"
    st.markdown(f"**Quality Gate:** {gate_icon}")
    if row.rejection_reasons:
        st.warning(f"Rejection: {', '.join(row.rejection_reasons)}")


# ============================================================
# Risk Annotation 面板
# ============================================================

def render_risk_annotation(row: SampleRow) -> None:
    """独立的 risk 分析面板。"""
    if row.sample_type != "risk":
        st.info("仅 risk 类型样本显示此面板")
        return

    st.markdown(f"**Risk Level:** `{row.risk_level}`")
    st.markdown(f"**Risk Reason:** {row.risk_reason}")

    if row.risk_phrases:
        st.markdown("**Risk Phrases (结构化):**")
        for p in row.risk_phrases:
            cat = p.get("category", "?")
            text = p.get("text", "")
            reason = p.get("reason", "")
            st.markdown(f"- 🏷️ `{cat}` — **\"{text}\"** — {reason}")
    else:
        st.caption("⚠️ 无结构化 risk_phrases，使用 diff/词典兜底高亮")


# ============================================================
# value_qa 专属渲染
# ============================================================

_VALUE_DIM_CHIP = {
    "cultural_fidelity": ("#0ea5e9", "🏛️"),
    "ethics_compliance": ("#a855f7", "⚖️"),
    # 一级中文 id
    "文化忠实度": ("#0ea5e9", "🏛️"),
    "伦理合规度": ("#a855f7", "⚖️"),
}
_VALUE_SECONDARY_CHIP = {
    "文化元素符合度": ("#0284c7", "🪔"),
    "行为社会规范符合度": ("#0369a1", "🤝"),
    "价值观忠实度": ("#0c4a6e", "🌱"),
    "公平合规度": ("#9333ea", "⚖️"),
    "伤害风险合规度": ("#7e22ce", "🛡️"),
    "隐私合规度": ("#6b21a8", "🔒"),
}
_MEDIUM_PATTERN_CHIP = {
    "letter_vs_spirit": ("#f97316", "📜"),
    "partial_correct": ("#eab308", "🧩"),
    "hedged": ("#6b7280", "🌫️"),
    "unknown": ("#94a3b8", "❔"),
}
_VALUE_LEVEL_CHIP = {
    "high": ("#16a34a", "🟢"),
    "medium": ("#eab308", "🟡"),
    "low": ("#dc2626", "🔴"),
}
_VALUE_LEVEL_ORDER = {"high": 0, "medium": 1, "low": 2}
_VALUE_LEVEL_BG = {"high": "#dcfce7", "medium": "#fef9c3", "low": "#fee2e2"}
_POLARITY_BG = {"positive": "#bbf7d0", "negative": "#fecaca", "neutral": "#e2e8f0"}
_POLARITY_EMOJI = {"positive": "✅", "negative": "❌", "neutral": "➖"}


def render_value_kpi(kpi: ValueKPIStats) -> None:
    """value_qa KPI 区：维度/档位分布 + 三联组完整率 + 极性合规率。"""

    def _ok_color(val: float, good: float = 0.9, warn: float = 0.7) -> str:
        if val >= good:
            return "#16a34a"
        if val >= warn:
            return "#d97706"
        return "#dc2626"

    cards = [
        _kpi_card("样本总数", f"{kpi.total}"),
        _kpi_card(
            "三联组完整率",
            f"{kpi.triplet_complete_rate:.0%}",
            hint="同 (doc, fact_idx) 含 high/med/low",
            accent=_ok_color(kpi.triplet_complete_rate),
        ),
        _kpi_card(
            "质量门通过率",
            f"{kpi.quality_gate_pass_rate:.0%}",
            accent=_ok_color(kpi.quality_gate_pass_rate, 0.95, 0.8),
        ),
        _kpi_card("Topic 覆盖", f"{kpi.topic_count}"),
        _kpi_card(
            "high 含 positive 标注",
            f"{kpi.high_with_positive_rate:.0%}",
            hint="high 档应含 positive 标注",
            accent=_ok_color(kpi.high_with_positive_rate),
        ),
        _kpi_card(
            "low 含 negative 标注",
            f"{kpi.low_with_negative_rate:.0%}",
            hint="low 档应含 negative 标注",
            accent=_ok_color(kpi.low_with_negative_rate),
        ),
        _kpi_card(
            "medium 含 pattern",
            f"{kpi.medium_with_pattern_rate:.0%}",
            hint="medium 档应给出 medium_pattern",
            accent=_ok_color(kpi.medium_with_pattern_rate),
        ),
        _kpi_card(
            "avg Evidence 覆盖",
            f"{kpi.avg_evidence_coverage:.2f}",
            accent=_ok_color(kpi.avg_evidence_coverage, 0.8, 0.5),
        ),
    ]

    for row_start in (0, 4):
        cols = st.columns(4, gap="small")
        for i, col in enumerate(cols):
            with col:
                st.markdown(cards[row_start + i], unsafe_allow_html=True)

    st.write("")

    # 三组分布：一级 / 二级 / 档位 / medium_pattern
    primary_dist = kpi.primary_dist or kpi.dim_dist
    dcols = st.columns(2, gap="small")
    with dcols[0]:
        st.markdown(
            _render_chips("一级指标分布", primary_dist, _VALUE_DIM_CHIP),
            unsafe_allow_html=True,
        )
    with dcols[1]:
        st.markdown(
            _render_chips("档位分布", kpi.level_dist, _VALUE_LEVEL_CHIP),
            unsafe_allow_html=True,
        )
    dcols2 = st.columns(2, gap="small")
    with dcols2[0]:
        st.markdown(
            _render_chips("二级评估维度分布", kpi.evaluation_dim_dist, _VALUE_SECONDARY_CHIP),
            unsafe_allow_html=True,
        )
    with dcols2[1]:
        st.markdown(
            _render_chips("medium_pattern 分布", kpi.medium_pattern_dist, _MEDIUM_PATTERN_CHIP),
            unsafe_allow_html=True,
        )


def render_value_triplet(group: list[SampleRow]) -> None:
    """value_qa 三联组并排渲染（high / medium / low），各档底色区分。"""
    sorted_group = sorted(group, key=lambda r: _VALUE_LEVEL_ORDER.get(r.value_level, 9))
    if not sorted_group:
        st.info("无三联组样本")
        return

    cols = st.columns(len(sorted_group))
    for col, row in zip(cols, sorted_group):
        with col:
            color, emoji = _VALUE_LEVEL_CHIP.get(row.value_level, ("#64748b", "⚪"))
            bg = _VALUE_LEVEL_BG.get(row.value_level, "#f1f5f9")
            crumb = " · ".join(
                x for x in [
                    row.primary_metric or row.value_dimension,
                    row.evaluation_dimension,
                ] if x
            )
            pattern_tag = (
                f' <span style="color:#f97316;font-size:0.8em">[{row.medium_pattern}]</span>'
                if (row.value_level == "medium" and row.medium_pattern)
                else ""
            )
            st.markdown(
                f'<div style="padding:6px 10px;border-radius:8px;background:{bg};'
                f'border:1px solid {color}55;margin-bottom:8px">'
                f'<b style="color:{color}">{emoji} {row.value_level}</b>'
                f' · <span style="color:#64748b;font-size:0.85em">{crumb}</span>'
                f"{pattern_tag}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Q:** {row.question}")
            highlighted_a = _highlight_value_text(row.answer, row.value_annotations, "answer")
            st.markdown(f"**A:** {highlighted_a}", unsafe_allow_html=True)
            if row.fact:
                st.caption(f"📌 fact: {row.fact}")
            if row.core_values_aligned:
                keys = ", ".join(c.get("key", "?") for c in row.core_values_aligned)
                st.caption(f"🎯 core_values: `{keys}`")
            st.caption(
                f"difficulty={row.difficulty:.2f}  cov={row.evidence_coverage:.2f}"
            )


def _highlight_value_text(text: str, annotations: list[dict], field: str) -> str:
    """根据 annotations 的 (start,end,polarity) 在指定 field 文本上高亮。"""
    if not text:
        return ""
    spans: list[tuple[int, int, str, str, str]] = []
    for a in annotations or []:
        if a.get("field") != field:
            continue
        s = a.get("start", -1)
        e = a.get("end", -1)
        polarity = a.get("polarity", "neutral")
        label = a.get("label", "")
        keys = ",".join(a.get("value_keys") or [])
        if 0 <= s < e <= len(text):
            spans.append((s, e, polarity, label, keys))
        else:
            t = a.get("text") or ""
            if t and t in text:
                idx = text.find(t)
                spans.append((idx, idx + len(t), polarity, label, keys))

    if not spans:
        return html_mod.escape(text)

    spans.sort(key=lambda x: x[0])
    parts: list[str] = []
    cursor = 0
    for s, e, polarity, label, keys in spans:
        if s < cursor:
            continue  # 跳过重叠
        if s > cursor:
            parts.append(html_mod.escape(text[cursor:s]))
        bg = _POLARITY_BG.get(polarity, "#e2e8f0")
        emoji = _POLARITY_EMOJI.get(polarity, "")
        tooltip = f"{label} | {polarity} | {keys}".strip(" |")
        parts.append(
            f'<mark style="background:{bg};padding:1px 2px;border-radius:3px" '
            f'title="{html_mod.escape(tooltip)}">'
            f"{emoji}{html_mod.escape(text[s:e])}</mark>"
        )
        cursor = e
    if cursor < len(text):
        parts.append(html_mod.escape(text[cursor:]))
    return "".join(parts)


def render_value_annotations(row: SampleRow) -> None:
    """value annotations 列表 + 在 answer/evidence 上高亮 polarity。"""
    if not row.value_annotations and row.verification_method != "value_alignment":
        st.info("仅 value_qa 样本显示此面板")
        return

    # 顶部元信息
    color, emoji = _VALUE_LEVEL_CHIP.get(row.value_level, ("#64748b", "⚪"))
    primary_label = row.primary_metric or row.value_dimension
    dim_color, dim_emoji = _VALUE_DIM_CHIP.get(primary_label, ("#64748b", "🏷️"))
    sec_color, sec_emoji = _VALUE_SECONDARY_CHIP.get(row.evaluation_dimension, ("#64748b", "🏷️"))
    parts = [
        f'<span style="background:{color}20;color:{color};border:1px solid {color}55;'
        f'border-radius:12px;padding:2px 10px;margin-right:8px;font-size:0.85em">'
        f"{emoji} {row.value_level}</span>",
    ]
    if primary_label:
        parts.append(
            f'<span style="background:{dim_color}20;color:{dim_color};border:1px solid {dim_color}55;'
            f'border-radius:12px;padding:2px 10px;margin-right:8px;font-size:0.85em">'
            f"{dim_emoji} {primary_label}</span>"
        )
    if row.evaluation_dimension:
        parts.append(
            f'<span style="background:{sec_color}20;color:{sec_color};border:1px solid {sec_color}55;'
            f'border-radius:12px;padding:2px 10px;margin-right:8px;font-size:0.85em">'
            f"{sec_emoji} {row.evaluation_dimension}</span>"
        )
    if row.value_level == "medium" and row.medium_pattern:
        mp_color, mp_emoji = _MEDIUM_PATTERN_CHIP.get(row.medium_pattern, ("#64748b", "❔"))
        parts.append(
            f'<span style="background:{mp_color}20;color:{mp_color};border:1px solid {mp_color}55;'
            f'border-radius:12px;padding:2px 10px;font-size:0.85em">'
            f"{mp_emoji} {row.medium_pattern}</span>"
        )
    st.markdown(
        f'<div style="margin-bottom:10px">' + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )
    if row.fact:
        st.markdown(f"**📌 fact:** {row.fact}")
    if row.core_values_aligned:
        keys = ", ".join(c.get("key", "?") for c in row.core_values_aligned)
        st.markdown(f"**🎯 core_values_aligned:** `{keys}`")

    # Answer 高亮预览
    st.markdown("---")
    st.markdown("**Answer 高亮：**")
    st.markdown(
        _highlight_value_text(row.answer, row.value_annotations, "answer"),
        unsafe_allow_html=True,
    )

    # Evidence 高亮预览
    if row.evidence_text:
        st.markdown("**Evidence 高亮：**")
        st.markdown(
            _highlight_value_text(row.evidence_text, row.value_annotations, "evidence"),
            unsafe_allow_html=True,
        )

    # Annotation 列表
    if row.value_annotations:
        st.markdown("---")
        st.markdown("**Annotations 明细：**")
        for a in row.value_annotations:
            polarity = a.get("polarity", "neutral")
            bg = _POLARITY_BG.get(polarity, "#e2e8f0")
            emoji = _POLARITY_EMOJI.get(polarity, "")
            field = a.get("field", "?")
            label = a.get("label", "?")
            text = a.get("text", "")
            layer = a.get("value_layer", "?")
            keys = ", ".join(a.get("value_keys") or [])
            rationale = a.get("rationale", "")
            st.markdown(
                f'<div style="border-left:3px solid; padding:4px 8px;margin-bottom:6px;'
                f'background:{bg}40">'
                f'<div><b>{emoji} {polarity}</b> · '
                f'<code>{field}</code> · <code>{label}</code> · '
                f'layer=<code>{layer}</code> · keys=<code>{keys}</code></div>'
                f'<div style="margin-top:2px">📝 "{html_mod.escape(text)}"</div>'
                f'<div style="color:#64748b;font-size:0.85em;margin-top:2px">'
                f'💬 {html_mod.escape(rationale)}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("⚠️ 无 value_annotations")
