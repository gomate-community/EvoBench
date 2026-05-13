# doc_to_qa

将单篇文档转换为 `d -> (x, y)` 形式的问答对样本。

## Input

- `documents`

## Output

- `x`: 问题
- `y`: 与问题配套、可由证据支持的答案

## Notes

- 问答都必须能回溯到同一条证据。
- 适合作为统一样本 schema 的基础 QA skill。
