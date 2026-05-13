# EvoBench v3

A skill-driven framework for generating, validating, and evolving high-quality benchmark samples across tasks.


默认数据文件：

- `data/corpus.jsonl`: 原始文档语料
- `data/samples.jsonl`: 统一样本 `UnifiedSample`

## Skills Layout

```text
benchmark/
  agents/
    skills/
      base.py
      registry.py
      _document_common.py
      doc_to_question/
      doc_to_answer/
      doc_to_qa/
      doc_to_qa_steps/
      paper_to_experience/
      error_to_training_samples/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m benchmark.cli init-storage
python -m benchmark.cli list-skills
```

## Local LLM

项目默认使用你本地部署的 OpenAI 兼容服务：

```env
LLM_ENABLED=true
LLM_PROVIDER=openai_compatible
LLM_MODEL=qwen3-32b
LLM_API_KEY=EMPTY
LLM_API_BASE=http://10.208.62.156:8002/v1
```

完整示例见 `.env.example`。

## JSONL Storage

`corpus.jsonl` 里每一行都是一个 `SourceDocument`。最小示例：

```json
{"source_id":"doc_001","title":"Alpha 芯片发布","source_type":"news","publisher":"Example News","content":"Alpha 公司发布新一代 AI 芯片，推理性能提升 30%。","trust_level":4,"language":"zh-CN"}
{"source_id":"doc_002","title":"Alpha 芯片分析","source_type":"analysis","publisher":"Example Research","content":"行业分析指出，该性能提升声明仍需第三方验证。","trust_level":4,"language":"zh-CN"}
```

`samples.jsonl` 里每一行都是一个 `UnifiedSample`。

## Generate Samples

`generate-samples` 的入口参数都收敛到 `SkillGenerationRequest`，但它们分属两层：

- 文档来源层：`--corpus-jsonl`、`--topic`
- 技能选择层：`--skill-ids`、`--task-type`、`--skill-config`

推荐先记住两条规则：

- 只要 `skill_ids` 非空，真正执行哪些 skill 就只看 `skill_ids`，`task_type` 不参与选 skill。
- 只有 `skill_ids` 为空时，`task_type` 才会按任务族自动展开成一组 skill。

### 参数含义

| 参数 | 类型 | 作用位置 | 实际行为 |
| --- | --- | --- | --- |
| `--corpus-jsonl` | `Path` | 文档收集层 | 从指定 `jsonl` 文件加载 `SourceDocument`，而不是走 retriever。 |
| `--topic` | `str` | 文档收集层 | 如果配合 `--corpus-jsonl`，会在标题和正文中做包含式过滤；如果不传 `--corpus-jsonl`，则作为 retriever 查询词。 |
| `--skill-ids` | 逗号分隔字符串 | skill 解析层 | 直接指定要跑哪些 skill，顺序就是执行顺序。CLI 默认值是 `doc_to_qa`。 |
| `--task-type` | `TaskType` 枚举 | skill 解析层 | 只有在 `skill_ids` 为空时才用于“按任务族选 skill”。 |
| `--skill-config` | 可重复 `key=value` | skill 执行层 | 会透传给所有被选中的 skill，并覆盖各 skill 自己的默认 config。 |
| `--limit` | `int` | pipeline 层 | 是本次请求的总样本上限，不是“每个 skill 各生成多少条”。 |

### `task_type` 取值

`TaskType` 目前只有 4 个值，它描述的是样本输出形态，而不是具体 skill 名称：

| `task_type` | 语义 | 典型输出 |
| --- | --- | --- |
| `document_to_x` | `d -> x` | 只产出一个 `x` |
| `document_to_xy` | `d -> (x, y)` | 同时产出 `x` 和 `y` |
| `document_to_xty` | `d -> (x, T, y)` | 产出 `x`、步骤/轨迹 `T`、答案 `y` |
| `error_to_training_set` | `e -> training samples` | 基于错误样本生成训练样本 |

这里的 `x / y / T` 是统一 artifact key，但不同 skill 对它们的业务含义可以不同，例如：

- `doc_to_qa` 的 `x` 是 question，`y` 是 answer。
- `paper_to_experience` 的 `x` 是 future problem，`y` 是 experience card。
- `doc_to_qa_steps` 的 `T` 是外显步骤，不是隐藏 CoT。

### `skill_id` 与 `task_type` 的对应关系

当前 registry 里的映射关系如下：

| `skill_id` | 对应 `task_type` | `x / y / T` 含义 | 默认配置键 |
| --- | --- | --- | --- |
| `doc_to_question` | `document_to_x` | `x=question` | `sentences_per_doc` |
| `doc_to_answer` | `document_to_x` | `x=answer` | `sentences_per_doc` |
| `doc_to_qa` | `document_to_xy` | `x=question`, `y=answer` | `pairs_per_doc` |
| `paper_to_experience` | `document_to_xy` | `x=future_problem`, `y=experience_card` | `experiences_per_doc`, `experience_types`, `strict_paper_only` |
| `doc_to_qa_steps` | `document_to_xty` | `x=question`, `T=solution_steps`, `y=answer` | `items_per_doc` |
| `error_to_training_samples` | `error_to_training_set` | 变体不同，可能产出 `x/y`，也可能产出 `x/T/y` | `include_contrastive`, `include_boundary` |

如果你想看代码中的实时定义，可以运行：

```bash
python -m benchmark.cli list-skills
python -m benchmark.cli list-skills --task-type document_to_xy
```

### 参数之间的优先级和坑点

1. `skill_ids` 优先级高于 `task_type`

下面这个命令实际执行的仍然是 `doc_to_qa`，不是 `document_to_x`：

```bash
python -m benchmark.cli generate-samples \
  --corpus-jsonl data/corpus.jsonl \
  --skill-ids doc_to_qa \
  --task-type document_to_x \
  --limit 10
```

原因是 `SampleFactoryAgent._resolve_skills()` 会先判断 `request.skill_ids`，非空就直接按 `skill_ids` 创建 skill。

2. CLI 里想“只用 `task_type` 选 skill”，必须显式清空 `skill_ids`

因为 CLI 参数 `--skill-ids` 的默认值就是 `doc_to_qa`，所以命令行里省略它，不代表空列表，而代表“固定跑 `doc_to_qa`”。

如果你真想按任务族自动展开，需要这样写：

```bash
python -m benchmark.cli generate-samples \
  --corpus-jsonl data/corpus.jsonl \
  --skill-ids "" \
  --task-type document_to_xy \
  --limit 10
```

这时会按 registry 顺序依次尝试当前 `document_to_xy` 下的所有 skill，也就是：

- `doc_to_qa`
- `paper_to_experience`

注意 `limit` 是总上限，所以前面的 skill 先产出够了，后面的 skill 可能完全跑不到。

3. `task_type` 和 `skill_ids` 当前不会自动校验一致

代码不会因为你传了“不匹配组合”而报错，最终样本的 `sample.task_type` 以具体 skill 自己写入的值为准。

这意味着：

- `--skill-ids paper_to_experience --task-type document_to_x` 仍会生成 `document_to_xy` 样本。
- 最安全的做法是：如果已经明确写了 `skill_ids`，就把 `task_type` 写成对应值，或者干脆不写。

4. `error_to_training_set` 是特例

`error_to_training_samples` 不是从 corpus 文档生样本，而是从 `error_samples` 生样本。当前 pipeline 在以下任一条件满足时会跳过文档收集：

- `task_type == error_to_training_set`
- `skill_ids == ["error_to_training_samples"]`

因此它更适合走 `generate-from-errors`，而不是 `--corpus-jsonl` 这条链路。

5. API 和 CLI 在 `skill_ids` 默认值上不一样

`SkillGenerationRequest` 模型里的 `skill_ids` 默认是空列表，所以 API 请求里如果省略 `skill_ids`，`task_type` 会正常生效；CLI 则因为 `--skill-ids` 默认是 `doc_to_qa`，行为不同。

### 推荐用法

最稳定、最不容易混淆的是“按 skill 精确指定”：

```bash
python -m benchmark.cli generate-samples \
  --corpus-jsonl data/corpus.jsonl \
  --skill-ids doc_to_qa \
  --task-type document_to_xy \
  --limit 10
```

给 skill 传私有参数：

```bash
python -m benchmark.cli generate-samples \
  --corpus-jsonl data/corpus.jsonl \
  --skill-ids doc_to_qa \
  --task-type document_to_xy \
  --skill-config pairs_per_doc=3 \
  --limit 10
```

例如 `doc_to_qa` 里，`pairs_per_doc` 可以控制每篇文档最多生成多少条 q-a。

如果要传多个参数，可以重复写：

```bash
python -m benchmark.cli generate-samples \
  --corpus-jsonl data/corpus.jsonl \
  --skill-ids doc_to_qa \
  --task-type document_to_xy \
  --skill-config pairs_per_doc=3 \
  --skill-config custom_flag=true \
  --limit 10
```

如果一次选择多个 skill，这些 `skill_config` 会共享给所有 skill；各 skill 只会消费自己认识的键，其他键会被忽略。

从学术论文抽取可迁移经验：

```bash
python -m benchmark.cli generate-samples \
  --corpus-jsonl data/corpus.jsonl \
  --skill-ids paper_to_experience \
  --task-type document_to_xy \
  --skill-config experiences_per_doc=4 \
  --samples-jsonl data/samples.jsonl \
  --limit 10
```

如果只想通过 `task_type` 自动选 skill，可以这样写：

```bash
python -m benchmark.cli generate-samples \
  --corpus-jsonl data/corpus.jsonl \
  --skill-ids "" \
  --task-type document_to_x \
  --limit 10
```

这个命令会按顺序尝试 `doc_to_question` 和 `doc_to_answer`。

初始化论文语料模板：

```bash
python -m benchmark.cli init-paper-corpus-template
```

这会生成一个可批量填写的模板文件 `data/paper_corpus_template.jsonl`。模板源文件也放在：

- `benchmark/agents/skills/templates/paper_corpus_template.jsonl`

`paper_to_experience` 会输出：

- `x`: 未来可能遇到的相关问题场景
- `y`: 结构化经验卡，包含 `experience_type`、`experience_statement`、`applicability`、`actionable_advice`、`caveats`

它支持的经验类型有三种：

- `fact`: 事实经验
- `strategy`: 策略经验
- `mechanism`: 机制经验
- `boundary`: 边界经验
- `failure`: 失败经验

如果只想抽取部分类型，可以这样传：

```bash
python -m benchmark.cli generate-samples \
  --corpus-jsonl data/corpus.jsonl \
  --skill-ids paper_to_experience \
  --task-type document_to_xy \
  --skill-config experience_types='["fact","strategy","boundary"]' \
  --skill-config strict_paper_only=true \
  --limit 10
```

把经验卡后处理成 QA：

```bash
python -m benchmark.cli postprocess-experience-to-qa \
  --source-skill-id paper_to_experience \
  --samples-jsonl data/samples.jsonl \
  --limit 100
```

这一步会从 `samples.jsonl` 里读取 `paper_to_experience` 样本，再生成新的 `experience_to_qa` 样本并回写到 `samples.jsonl`。

如果只想针对某个主题过滤语料：

```bash
python -m benchmark.cli generate-samples \
  --corpus-jsonl data/corpus.jsonl \
  --topic Alpha \
  --skill-ids doc_to_qa \
  --task-type document_to_xy \
  --limit 10
```

运行后：

- 原始文档会保存在 `corpus.jsonl`
- 生成结果会保存在 `samples.jsonl`

查看语料和样本：

```bash
python -m benchmark.cli list-corpus --limit 20
python -m benchmark.cli list-samples --skill-id doc_to_qa --limit 20
```

## Other Commands

从 retriever 抓取文档并生成旧版 benchmark items：

```bash
python -m benchmark.cli ingest-sample --topic AI --limit 5
```

基于错误样本生成训练数据：

```bash
python -m benchmark.cli generate-from-errors --errors-json errors.json --limit 20
```

## API

```bash
uvicorn benchmark.api.main:app --reload
```

从 `corpus.jsonl` 生成样本：

```bash
curl -X POST http://127.0.0.1:8000/samples/generate \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_jsonl_path": "data/corpus.jsonl",
    "samples_jsonl_path": "data/samples.jsonl",
    "task_type": "document_to_xy",
    "skill_ids": ["doc_to_qa"],
    "limit": 5,
    "language": "zh-CN",
    "domain": "technology"
  }'
```

如果你希望 API 按 `task_type` 自动选 skill，可以直接省略 `skill_ids` 字段；这一点和 CLI 不同。

查看当前语料：

```bash
curl "http://127.0.0.1:8000/corpus?limit=20"
```

## Test

```bash
PYTHONPATH=. pytest -q
```
