# 统一样本 Schema 设计

## 目标

统一支持以下任务：

1. `d -> x`：基于文档生成问题、指令、答案片段或其他单目标样本。
2. `d -> (x, y)`：基于文档生成问题和答案。
3. `d -> (x, T, y)`：基于文档生成问题、显式解题/标注步骤和答案。
4. `error -> training samples`：基于已有错误样本生成新的训练样本集合。

## 设计原则

- 样本不是固定 QA，而是统一表示为 `input -> output artifacts`。
- `x`、`y`、`T` 只是 output artifact 的 key，不绑定具体任务。
- 文档、错误样本、证据、claim、人工标注说明统一进入 input/provenance。
- Skill 独立：不同任务生成过程均实现为可注册的 Skill。
- 源数据共享：所有文档先经过统一 SourceSelector，再进入 Skill。

## 核心对象

### UnifiedSample

| 字段 | 说明 |
|---|---|
| sample_id | 样本唯一 ID |
| task_type | 任务类型，例如 document_to_xy |
| skill_id | 生成该样本的 Skill |
| input | 输入文档、错误样本、claim、额外 artifact |
| output | 输出 artifact，如 x/y/T |
| source_refs | 源数据引用 |
| evidence | 证据片段 |
| annotation_guideline | 人工审核与标注规则 |
| quality_signals | 自动质量门控信号 |
| parent_sample_ids | 由已有样本扩增时的父样本 |

### SampleArtifact

| 字段 | 说明 |
|---|---|
| role | document/question/answer/reasoning_trace/evidence/label/critique/correction |
| key | 业务键，例如 d/x/y/T |
| value | 任意 JSON 可序列化内容 |
| evidence_ids | 支持该 artifact 的证据 ID |
| metadata | 扩展字段 |

### SkillDefinition

| 字段 | 说明 |
|---|---|
| skill_id | Skill ID |
| task_type | 对应任务类型 |
| input_requirements | 输入要求 |
| output_schema | 输出结构，例如 `{x: question, y: answer}` |
| quality_rules | 质量门控要求 |
| config | Skill 私有配置 |

## 新增任务的推荐流程

1. 定义任务类型或复用已有 TaskType。
2. 新增 SkillDefinition。
3. 实现 `SkillBase.generate()`。
4. 输出 `UnifiedSample`。
5. 通过 `VerifierAgent.verify_samples()`。
6. 存入 `benchmark_samples`。
7. 必要时用 `unified_to_benchmark_item()` 转成旧 QA 结构做客观评测。
