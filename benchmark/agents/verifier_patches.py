"""Monkey patch：让 ``VerifierAgent`` 对 contradiction_check / human 验证方式跳过 evidence_coverage。

对应 wsy_skill_dev 在 main 之上的两行补丁。补丁通过 bootstrap 应用，避免修改 main 原文件。
"""

from __future__ import annotations

from benchmark.agents.verifier_agent import VerifierAgent
from benchmark.schemas import VerificationMethod

_orig_sample_rejection_reasons = VerifierAgent._sample_rejection_reasons


def _patched_sample_rejection_reasons(self, sample):
    reasons = _orig_sample_rejection_reasons(self, sample)
    # 对 contradiction_check / human 验证方式不要求 evidence_coverage 命中
    if sample.verification_method in (VerificationMethod.contradiction_check, VerificationMethod.human):
        reasons = [r for r in reasons if r != "low_evidence_coverage"]
    return sorted(set(reasons))


def install() -> None:
    """幂等地把补丁安装到 ``VerifierAgent._sample_rejection_reasons``。"""
    if getattr(VerifierAgent._sample_rejection_reasons, "__wrapped_by_verifier_patches__", False):
        return
    _patched_sample_rejection_reasons.__wrapped_by_verifier_patches__ = True  # type: ignore[attr-defined]
    VerifierAgent._sample_rejection_reasons = _patched_sample_rejection_reasons  # type: ignore[assignment]
