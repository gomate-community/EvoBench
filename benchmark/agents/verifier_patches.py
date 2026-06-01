"""Monkey patch：让 ``VerifierAgent`` 对 contradiction_check / human 验证方式跳过 evidence_coverage，
并为 ``value_alignment`` 验证方式叠加 value_qa 专属业务校验。

对应 wsy_skill_dev 在 main 之上的两处补丁。补丁通过 bootstrap 应用，避免修改 main 原文件。

新版（基于 6 个二级评估维度）的 value_alignment 业务规则：
  1) metadata.primary_metric ∈ {文化忠实度, 伦理合规度}
  2) metadata.evaluation_dimension ∈ 6 个二级评估维度
  3) metadata.value_level ∈ {high, medium, low}
  4) high 必须含 ≥1 个 polarity=positive 的 annotation
  5) low  必须含 ≥1 个 polarity=negative 的 annotation
  6) medium 必须显式给出 metadata.medium_pattern ∈ {letter_vs_spirit, partial_correct, hedged, unknown}
  注：4 层 taxonomy 已降级为"参考词典"，不再强约束 value_keys 必须 ⊆ keys_for_dimension(dim)
  注：low 档放宽 evidence_coverage；value_alignment 样本豁免 near_duplicate_x
  （三档 question 语义等价但措辞不同，因核心名词重叠仍可能误触发）
"""

from __future__ import annotations

from benchmark.agents.verifier_agent import VerifierAgent
from benchmark.schemas import UnifiedSample, VerificationMethod

_orig_sample_rejection_reasons = VerifierAgent._sample_rejection_reasons


_VALID_PRIMARY = {"文化忠实度", "伦理合规度"}
_VALID_SECONDARY = {
    "文化元素符合度",
    "行为社会规范符合度",
    "价值观忠实度",
    "公平合规度",
    "伤害风险合规度",
    "隐私合规度",
}
_VALID_LEVELS = {"high", "medium", "low"}
_VALID_MEDIUM_PATTERNS = {"letter_vs_spirit", "partial_correct", "hedged", "unknown"}


def _value_alignment_extra_reasons(sample: UnifiedSample) -> list[str]:
    """value_qa 专属规则。返回额外的拒绝原因列表。"""
    reasons: list[str] = []
    meta = sample.metadata or {}
    primary = meta.get("primary_metric")
    secondary = meta.get("evaluation_dimension")
    level = meta.get("value_level")

    # 1) 一级 / 二级 / 等级枚举
    if primary not in _VALID_PRIMARY:
        reasons.append("invalid_primary_metric")
    if secondary not in _VALID_SECONDARY:
        reasons.append("invalid_evaluation_dimension")
    if level not in _VALID_LEVELS:
        reasons.append("invalid_value_level")

    # 2) high / low 极性强约束
    annotations = getattr(sample, "value_annotations", []) or []
    pos = sum(1 for a in annotations if a.polarity == "positive")
    neg = sum(1 for a in annotations if a.polarity == "negative")
    if level == "high" and pos < 1:
        reasons.append("high_missing_positive_annotation")
    if level == "low" and neg < 1:
        reasons.append("low_missing_negative_annotation")

    # 3) medium 必须给出合法 medium_pattern
    if level == "medium":
        mp = meta.get("medium_pattern")
        if mp not in _VALID_MEDIUM_PATTERNS:
            reasons.append("medium_missing_pattern")

    return reasons


def _patched_sample_rejection_reasons(self, sample):
    reasons = _orig_sample_rejection_reasons(self, sample)
    # 对 contradiction_check / human 验证方式不要求 evidence_coverage 命中
    if sample.verification_method in (VerificationMethod.contradiction_check, VerificationMethod.human):
        reasons = [r for r in reasons if r != "low_evidence_coverage"]
    # value_alignment：叠加 value_qa 专属业务规则；low 档放宽 evidence_coverage
    if sample.verification_method == VerificationMethod.value_alignment:
        reasons.extend(_value_alignment_extra_reasons(sample))
        meta = sample.metadata or {}
        if meta.get("value_level") == "low":
            reasons = [r for r in reasons if r != "low_evidence_coverage"]
    return sorted(set(reasons))


def install() -> None:
    """幂等地把补丁安装到 ``VerifierAgent._sample_rejection_reasons``。"""
    if getattr(VerifierAgent._sample_rejection_reasons, "__wrapped_by_verifier_patches__", False):
        return
    _patched_sample_rejection_reasons.__wrapped_by_verifier_patches__ = True  # type: ignore[attr-defined]
    VerifierAgent._sample_rejection_reasons = _patched_sample_rejection_reasons  # type: ignore[assignment]

    # 额外：给 verify_samples 装一个后置补丁，剔除 value_alignment 样本上的 near_duplicate_x。
    # 原因：value_qa 三档 question 虽语义等价但措辞不同，依然可能因核心名词重叠
    # 被 lexical_overlap>0.95 误判为近重；该 patch 作为兜底豁免，避免误伤同对照组。
    if not getattr(VerifierAgent.verify_samples, "__wrapped_by_value_qa_patches__", False):
        _orig_verify_samples = VerifierAgent.verify_samples

        def _patched_verify_samples(self, samples):
            out = _orig_verify_samples(self, samples)
            for s in out:
                if s.verification_method != VerificationMethod.value_alignment:
                    continue
                reasons = list(s.quality_signals.rejection_reasons or [])
                if "near_duplicate_x" in reasons:
                    reasons = [r for r in reasons if r != "near_duplicate_x"]
                    s.quality_signals.rejection_reasons = sorted(set(reasons))
                    s.quality_signals.quality_gate_passed = not reasons
                    s.status = "verified" if not reasons else "rejected"
            return out

        _patched_verify_samples.__wrapped_by_value_qa_patches__ = True  # type: ignore[attr-defined]
        VerifierAgent.verify_samples = _patched_verify_samples  # type: ignore[assignment]

