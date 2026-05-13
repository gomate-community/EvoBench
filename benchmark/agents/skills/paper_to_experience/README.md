# paper_to_experience

从一篇学术论文中抽取可迁移经验，用于支持未来相似问题的解决。

## Input

- `documents`

## Output

- `x`: 一个未来可能遇到的相关问题场景
- `y`: 一张结构化经验卡

## Experience Types

- `fact`: 事实经验，强调被论文支持的规律、边界、定量发现
- `strategy`: 策略经验，强调方法、流程、步骤与决策策略
- `cognitive`: 认知经验，强调思维框架、常见误判、注意事项

## Notes

- 每条经验都必须回链到论文中的最小证据。
- `y` 会包含 `experience_type`、`experience_statement`、`applicability`、`actionable_advice`、`caveats`。
- 适合做“从论文沉淀经验，再迁移到未来问题”的样本构建。
