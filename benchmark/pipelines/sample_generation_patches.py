"""Monkey patch：把 ``SampleGenerationPipeline`` 的默认 retriever 从 MockRetriever 切到 BaiduCultureRetriever。

对应 wsy_skill_dev 在 main 之上把 ``retriever or MockRetriever()`` 改成 ``retriever or BaiduCultureRetriever()`` 的行为，
通过 bootstrap 应用，避免修改 main 原文件。

注意：所有对 ``SampleGenerationPipeline`` / ``BaiduCultureRetriever`` 的 import 都被延迟到
``install()`` 函数体内，确保 ``llm_rightcode.install()`` 先于 ``sample_generation`` 模块被 import，
从而让 ``sample_generation.py`` 顶部 ``from benchmark.adapters.llm import build_llm_adapter`` 拿到
的是 patched 后的函数引用。
"""

from __future__ import annotations


def install() -> None:
    """幂等地把补丁安装到 ``SampleGenerationPipeline.__init__``。"""
    from benchmark.adapters.retriever_baidu import BaiduCultureRetriever
    from benchmark.pipelines.sample_generation import SampleGenerationPipeline

    if getattr(SampleGenerationPipeline.__init__, "__wrapped_by_sgp_patches__", False):
        return

    _orig_init = SampleGenerationPipeline.__init__

    def _patched_init(self, retriever=None, *args, **kwargs):
        return _orig_init(self, retriever=retriever or BaiduCultureRetriever(), *args, **kwargs)

    _patched_init.__wrapped_by_sgp_patches__ = True  # type: ignore[attr-defined]
    SampleGenerationPipeline.__init__ = _patched_init  # type: ignore[assignment]
