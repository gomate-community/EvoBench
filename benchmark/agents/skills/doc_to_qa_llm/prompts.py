"""doc_to_qa_llm Skill 专用 LLM Prompt（独立于 main 上 doc_to_qa/prompts.py）。"""

from __future__ import annotations

SYSTEM_PROMPT = """\
你是一个高质量 benchmark 样本生成专家。你的任务是根据给定文档生成事实性问答对。

要求：
1. 问题必须完全基于文档内容，不得引入外部知识
2. 答案必须能从文档中直接找到证据支持
3. 问题应有深度，不是简单的"是什么"，要涵盖因果、比较、过程等类型
4. 答案应准确、简洁、有信息量
5. 每对 QA 之间应尽量覆盖文档不同方面
"""

QA_GENERATION_PROMPT = """\
请根据以下文档生成 {n} 个高质量问答对。

## 文档标题
{title}

## 文档内容
{content}

## 输出格式
请严格按以下 JSON 数组格式输出，不要有其他文字：
```json
[
  {{"question": "...", "answer": "...", "evidence": "..."}},
  ...
]
```

其中 evidence 是答案在文档中的原文依据（直接摘抄）。
"""
