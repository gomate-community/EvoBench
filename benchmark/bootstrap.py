"""项目级 bootstrap：集中安装所有 wsy_skill_dev 引入的扩展能力，避免修改 main 任何原文件。

工作内容（幂等）：
1. ``llm_rightcode`` provider 别名 monkey patch；
2. ``VerifierAgent`` 跳过 contradiction_check / human 的 evidence_coverage 检查；
3. ``SampleGenerationPipeline`` 默认 retriever 切到 ``BaiduCultureRetriever``；
4. 把 ``benchmark_qa`` Skill 注册进 ``SkillRegistry`` 的 ``_factories`` / ``_builtin_output_schemas`` /
   ``default_definitions``；
5. 把 ``doc_to_qa`` 的工厂替换为 ``DocumentToQALLMSkill``，让默认行为变为 LLM 模式（无 LLM 时自动回退父类模板）；
6. ``BenchmarkRepository`` 与 ``init_storage`` 切到 ``data/corpus/{source}/`` 与 ``data/samples/{skill_id}/{status}/`` 分类目录布局
   （显式给 ``--corpus-jsonl`` / ``--samples-jsonl`` 时退回单文件模式）。

激活方式：
- 项目根 ``sitecustomize.py`` 在 Python 启动时自动 ``import benchmark.bootstrap``；
- 测试中由 ``tests/conftest.py`` 显式 import 保证激活；
- 任何代码也可手动 ``import benchmark.bootstrap`` 触发。
"""

from __future__ import annotations

from benchmark.adapters import llm_rightcode
from benchmark.agents import verifier_patches
from benchmark.pipelines import sample_generation_patches
from benchmark.storage import db_patches, repository_patches
from benchmark.agents.skills.benchmark_qa.schema import BENCHMARK_QA_OUTPUT_SCHEMA
from benchmark.agents.skills.benchmark_qa.skill import BenchmarkQASkill
from benchmark.agents.skills.doc_to_qa_llm.skill import DocumentToQALLMSkill
from benchmark.agents.skills.registry import SkillRegistry
from benchmark.schemas import SkillDefinition, TaskType

_installed = False


def _register_skills() -> None:
    """把 benchmark_qa 注册进 SkillRegistry，并把 doc_to_qa 工厂替换为 LLM 子类版本。"""
    # 1) 工厂表：新增 benchmark_qa；把 doc_to_qa 默认替换为 LLM 子类。
    SkillRegistry._factories["benchmark_qa"] = BenchmarkQASkill
    SkillRegistry._factories["doc_to_qa"] = DocumentToQALLMSkill

    # 2) output_schema 表：新增 benchmark_qa。
    SkillRegistry._builtin_output_schemas["benchmark_qa"] = BENCHMARK_QA_OUTPUT_SCHEMA

    # 3) default_definitions：包裹原 classmethod，追加 benchmark_qa 定义。
    if getattr(SkillRegistry.default_definitions, "__wrapped_by_bootstrap__", False):
        return

    _orig_default_definitions = SkillRegistry.default_definitions

    def _patched_default_definitions(cls) -> list[SkillDefinition]:
        defs = list(_orig_default_definitions.__func__(cls))
        # 避免重复追加
        if any(d.skill_id == "benchmark_qa" for d in defs):
            return defs
        defs.append(
            SkillDefinition(
                skill_id="benchmark_qa",
                name="Benchmark QA (Normal + Counterfactual + Risk)",
                task_type=TaskType.document_to_xy,
                description="Generate triplet samples: normal fact QA, counterfactual QA, and risk-annotated statement.",
                output_schema=BENCHMARK_QA_OUTPUT_SCHEMA,
                quality_rules={"min_evidence_coverage": 0.3, "human_review_required": True},
                config={"groups_per_doc": 2},
                tags=["benchmark_qa", "counterfactual", "risk"],
            )
        )
        return defs

    _patched_default_definitions.__wrapped_by_bootstrap__ = True  # type: ignore[attr-defined]
    SkillRegistry.default_definitions = classmethod(_patched_default_definitions)  # type: ignore[assignment]


def apply() -> None:
    """幂等地装载所有补丁。"""
    global _installed
    if _installed:
        return
    llm_rightcode.install()
    verifier_patches.install()
    sample_generation_patches.install()
    db_patches.install()
    repository_patches.install()
    _register_skills()
    _installed = True


# import 时立即激活，让 ``import benchmark.bootstrap`` 即生效。
apply()
