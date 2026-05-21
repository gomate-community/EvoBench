"""Monkey patch：为 ``benchmark.adapters.llm.build_llm_adapter`` 增加 ``rightcode`` provider 别名。

``rightcode`` 视为 ``openai_compatible`` 的等价别名。该补丁在 ``benchmark.bootstrap`` 启动时
应用，避免修改 main 原 ``llm.py``。
"""

from __future__ import annotations

from benchmark.adapters import llm as _llm
from benchmark.core.config import settings

_orig_build_llm_adapter = _llm.build_llm_adapter


def _patched_build_llm_adapter(provider: str | None = None, **kwargs):
    """把 ``rightcode`` 别名归一化为 ``openai_compatible`` 再委托给原函数。

    与原函数一致：未传 ``provider`` 时回退到 ``settings.llm_provider``；只要解析后是 ``rightcode``
    都改写成 ``openai_compatible``。
    """
    resolved = (provider or settings.llm_provider or "").lower()
    if resolved == "rightcode":
        provider = "openai_compatible"
    return _orig_build_llm_adapter(provider=provider, **kwargs)


def install() -> None:
    """幂等地把补丁安装到 ``benchmark.adapters.llm.build_llm_adapter``。"""
    if getattr(_llm.build_llm_adapter, "__wrapped_by_rightcode__", False):
        return
    _patched_build_llm_adapter.__wrapped_by_rightcode__ = True  # type: ignore[attr-defined]
    _llm.build_llm_adapter = _patched_build_llm_adapter  # type: ignore[assignment]
