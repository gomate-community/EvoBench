# doc_to_question

将单篇文档转换为 `d -> x` 形式的问题样本。

## Input

- `documents`

## Output

- `x`: 可被文档证据直接支持的问题

## Notes

- 问题必须可回答、可验证。
- 问题中不能直接泄露答案。
- 证据片段会回填到 `SampleInput` 和 `evidence` 字段中。
