"""benchmark_qa Skill 专属的 output schema 定义。

从 main 原 ``benchmark/schemas.py`` 抽离，放在 skill 子包内符合 ``Skill 模块标准化目录结构与职责``
（每个 skill 自包含 schema/prompts/skill）。bootstrap 会把该常量注入到 ``SkillRegistry`` 的
``_builtin_output_schemas`` 和 ``default_definitions``。
"""

from __future__ import annotations

from benchmark.schemas import DocumentQASampleSchema, build_output_schema

BENCHMARK_QA_OUTPUT_SCHEMA = build_output_schema(
    schema_name="benchmark_qa_sample",
    artifact_map={"x": "question_or_statement", "y": "answer_or_assessment"},
    model=DocumentQASampleSchema,
    description="Output schema for benchmark QA samples (normal / counterfactual / risk variants).",
)
