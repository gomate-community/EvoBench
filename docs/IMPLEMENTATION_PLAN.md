# 实施开发计划

## M1：工程骨架与题库元数据（第 1-2 周）
- 完成核心 Schema、数据库、Repository、CLI 与 API。
- 建立 Skill 配置、评分权重配置和样例数据生成链路。
- 验收：本地可完成 init-db、ingest-sample、run-eval。

## M2：RAG 与多源接入（第 3-4 周）
- 接入新闻、论文、政策、财报等 Source Adapter。
- 增加来源可信度、发布时间、事件时间、引用链记录。
- 验收：每道题可追溯到原始来源与证据片段。

## M3：题目验证与质量控制（第 5-6 周）
- 实现独立 Verifier、语义去重、难度估计、歧义检测。
- 建立风险分层抽检队列和人工审核接口。
- 验收：候选题进入 verified/ rejected/ review 三类状态。

## M4：自动评分与 Arena（第 7-8 周）
- 完善 exact match、regex、程序化验证和 Judge 评分。
- 实现双盲 Battle、顺序随机、Elo/Anchor 归一化。
- 验收：输出按 Skill 分组的模型分数和总分。

## M5：治理、监控与生产化（第 9-10 周）
- 增加 Anchor/Fresh/Adversarial/Canary 分层题库管理。
- 增加泄漏风险、题目淘汰、榜单版本化、审计日志。
- 验收：支持定期滚动更新和跨版本可比分析。
