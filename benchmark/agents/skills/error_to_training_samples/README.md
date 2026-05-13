# error_to_training_samples

将错误样本转换为训练数据集合。

## Input

- `error_samples`

## Output

- `corrected`: 纠错型 SFT 样本
- `contrastive`: 偏好/对比样本
- `boundary`: 边界场景样本

## Notes

- 适合从线上错例或评测错例反推训练数据。
- `expected_output` 缺失时会退化为需要人工复核的样本。
