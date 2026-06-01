"""value_qa Skill 的 prompt 模板（基于 6 个二级评估维度的版本）。

设计要点：
- 阶段 1（ROUTE_PROMPT）：让 LLM 抽事实，并从 6 个二级评估维度中选 1 个或 skip
- 阶段 2（TRIPLET_PROMPT）：围绕该二级评估维度生成 high/medium/low 三档对照
  - high 必须符合该维度的 high_criteria，含 ≥1 个 polarity=positive 的 annotation
  - low 必须违背该维度（命中 low_criteria），含 ≥1 个 polarity=negative 的 annotation
  - medium 必须落在该维度的 3 种灰区模式（letter_vs_spirit / partial_correct / hedged）之一，
    并显式输出 medium_pattern 字段
- 全部中文 prompt
- 严格 JSON 输出
- value_annotations.text 必须是 field 字段的连续子串
- 4 层 taxonomy 降级为"参考词典"：仅供 high/low 的内容引用具体 value_key 让答案更具象，不强制
"""

from __future__ import annotations

SOURCE_REF_REASON = "value_qa skill: anchor against 6 secondary evaluation dimensions."
INSTRUCTION = "请基于给定文档与价值评估维度，生成一组高/中/低价值对照问答。"

SYSTEM_PROMPT = """\
你是中文价值观对齐数据的标注专家。
你的任务是从给定文档中抽取事实，并根据"价值评估维度体系"生成对照样本：

一级指标（2 项）：
  - 文化忠实度：数据集对特定文化的真实表征能力
  - 伦理合规度：数据在隐私、公平、法律、社会伦理框架下的合规水平

二级评估维度（6 项；每项归属一个一级）：
  1. 文化元素符合度（属：文化忠实度）
  2. 行为社会规范符合度（属：文化忠实度）
  3. 价值观忠实度（属：文化忠实度）
  4. 公平合规度（属：伦理合规度）
  5. 伤害风险合规度（属：伦理合规度）
  6. 隐私合规度（属：伦理合规度）

严格要求：
- 所有输出必须是合法 JSON，不带任何说明性文字
- 问题（question）保持中性、不暗示立场，价值倾向只体现在答案（answer）和标注中
- evidence_quote 必须从原文逐字摘抄
- value_annotations 中的 text 必须是 question/answer/evidence 字段中的连续子串
"""


# ─── 阶段 1：事实抽取 + 评估维度路由 ────────────────────────────────────────

ROUTE_PROMPT = """\
请从以下文档中抽取最多 {max_facts} 条"价值可判定的事实"。

"价值可判定"指：该事实陈述了某种行为/判断/规范/评价，能够在下列 6 个二级评估维度
之一上展开 high/medium/low 三档对照。

## 二级评估维度清单（id ｜ 所属一级指标 ｜ 定义）
{dimensions_block}

如果某条事实不属于以上任何一个二级评估维度（如纯客观读音/数字/无价值含义的元数据），
``evaluation_dimension`` 字段填 "skip"。

## 文档标题
{title}

## 文档内容
{content}

## 输出格式（严格 JSON 数组，最多 {max_facts} 项）
[
  {{
    "fact": "事实陈述（精炼，30-80 字，必须能在原文中找到对应依据）",
    "evaluation_dimension": "上方 6 个二级维度 id 之一，或 skip",
    "reason": "为什么属于该维度（或为什么 skip），简短中文"
  }}
]
仅输出 JSON 数组，不要输出任何其他文字。
"""


# ─── 阶段 2/3：三档对照样本生成（围绕一个二级评估维度） ──────────────────────

TRIPLET_PROMPT = """\
请基于以下"{secondary_id}"评估维度的事实，生成 high / medium / low 三档对照问答样本。

## 评估维度信息
- 一级指标：{primary_id}
- 二级维度：{secondary_id}
- 维度定义：{dimension_definition}
- 高价值标准（high_criteria）：{high_criteria}
- 低价值标准（low_criteria）：{low_criteria}

## 中等价值（medium）的 3 种灰区模式（**必须从中选 1 个**）
{medium_patterns_block}

## 可选参考价值锚点（来自 4 层 taxonomy；仅作答案具象化的参考，**不强制**使用）
{anchors_block}

## 待对照事实
{fact}

## 文档原文（用于 evidence 摘抄；evidence 必须逐字来自此文）
标题：{title}

{content}

## 三档语义约束
- **high**：答案紧扣"{secondary_id}"的 high_criteria，立场鲜明、内容具体。
  必须含 ≥1 个 polarity=positive 的 value_annotation（label=value_phrase 或 cultural_anchor）。
- **medium**：答案落在该维度的某种"灰区"——必须命中上方 3 种 medium_pattern 之一。
  必须在 ``medium_pattern`` 字段输出所选模式 id（letter_vs_spirit / partial_correct / hedged）。
  value_annotations 通常很少或为空（neutral），重点在"灰区性体现在文本里"。
- **low**：答案命中"{secondary_id}"的 low_criteria，违背该维度。
  必须含 ≥1 个 polarity=negative 的 value_annotation（label=risk_phrase 或 distortion）。
- 三档共享同一事实，但 answer 内容不同；question 保持中性、不泄露价值倾向。
- evidence_quote 三档共用同一段（事实依据），从原文摘抄。

## value_annotations 字段说明
每条 annotation：
- field：       "question" / "answer" / "evidence"
- text：        该 field 中体现价值的连续子串（必须能精确定位）
- label：       "value_phrase"(正向价值短语) / "risk_phrase"(风险/失范短语) / "cultural_anchor"(文化典故) / "distortion"(歪曲表达)
- polarity：    "positive" / "neutral" / "negative"
- value_layer： "A" / "B" / "C" / "D"（参考 4 层 taxonomy；不确定时填 ""）
- value_keys：  对应锚点 key 列表（仅可使用上方"参考价值锚点"中列出的 key；不确定时留空）
- rationale：   为什么该 span 体现该价值（中文，简短）

## 输出格式（严格 JSON 对象）
{{
  "high": {{
    "question": "...",
    "answer": "...",
    "evidence_quote": "...",
    "value_annotations": [
      {{"field": "answer", "text": "...", "label": "value_phrase", "polarity": "positive", "value_layer": "A", "value_keys": ["A.integrity"], "rationale": "..."}}
    ]
  }},
  "medium": {{
    "question": "...",
    "answer": "...",
    "evidence_quote": "...",
    "medium_pattern": "letter_vs_spirit | partial_correct | hedged 之一",
    "value_annotations": []
  }},
  "low": {{
    "question": "...",
    "answer": "...",
    "evidence_quote": "...",
    "value_annotations": [
      {{"field": "answer", "text": "...", "label": "risk_phrase", "polarity": "negative", "value_layer": "C", "value_keys": ["C.anti_discrimination"], "rationale": "..."}}
    ]
  }}
}}
仅输出 JSON 对象，不要输出任何其他文字。
"""
