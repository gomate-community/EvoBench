"""value_qa Skill 专属的 output schema 定义。

复用通用 DocumentQASampleSchema (x=question, y=answer)；维度/等级/标注等扩展
信息通过 ``UnifiedSample.metadata`` 与 ``UnifiedSample.value_annotations`` 承载，
不污染通用 schema。
"""

from __future__ import annotations

from benchmark.schemas import DocumentQASampleSchema, build_output_schema

VALUE_QA_OUTPUT_SCHEMA = build_output_schema(
    schema_name="value_qa_sample",
    artifact_map={"x": "question", "y": "answer"},
    model=DocumentQASampleSchema,
    description=(
        "Output schema for value_qa samples. Each sample is one of "
        "high / medium / low value tier under either cultural_fidelity or "
        "ethics_compliance dimension; value-aligned spans are recorded in "
        "UnifiedSample.value_annotations."
    ),
)
