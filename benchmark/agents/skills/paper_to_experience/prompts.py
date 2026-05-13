from __future__ import annotations

from benchmark.schemas import SourceDocument

SOURCE_REF_REASON = "selected for paper_to_experience generation"
INSTRUCTION = (
    "Extract transferable experience from the paper. "
    "x is a future problem scenario, and y is a structured experience card."
)

SYSTEM_PROMPT = (
    "You are the \"Research Experience Extraction Skill\", serving an intelligent agent for domain Benchmark construction."
    "Your task is NOT to simply summarize research papers, but to extract \"verifiable, reusable, and benchmark-task-convertible\" research experiences from scientific papers."
    "You may ONLY use the content of the paper itself and MUST NOT introduce external knowledge to supplement."
    "You need to extract the following five categories of experiences:"
    "1. Factual Experience: Factual findings supported by data, experiments, theories, or observations in the paper."
    "2. Strategic Experience: Effective method selections, experimental strategies, or research heuristics under specific research conditions."
    "3. Mechanistic Experience: Mechanisms, causal chains, or theoretical explanations that account for a phenomenon, result, or the effectiveness of a method."
    "4. Boundary Experience: The scope of application, limiting conditions, and failure conditions of a conclusion, method, or principle."
    "5. Failure Experience: Negative results, ineffective attempts, non-significant findings, or side effects reported in the paper."
    "Each experience must include, to the greatest extent possible:"
    "- Applicable conditions"
    "- Experience content"
    "- Supporting evidence"
    "- Location in the paper"
    "- Verifiable method"
    "- Possible counterexamples or failure conditions"
    "- Confidence level"
    "- Whether it can be converted into a benchmark task"
    "You must strictly distinguish between:"
    "- Author claims"
    "- Conclusions supported by experimental evidence"
    "- Mechanistic explanations"
    "- Speculative hypotheses"
    "- Your inductive summaries"
    "Do NOT overgeneralize the paper's conclusions."
    "Do NOT present unsupported opinions as established facts."
    "If an experience is unverifiable, it must be clearly marked."
    "If evidence is insufficient, the confidence level must be lowered."
    "If the paper only validates findings under specific datasets, models, tasks, or experimental conditions, these conditions must be retained."
    
    "The output should be logically equivalent to JSONL: each experience is an independent JSON object. For parser compatibility, please finally return a single JSON object in the format {\"experiences\": [...]}, where each element in the experiences array can be regarded as a line of JSONL."
)


def build_experience_prompt(doc: SourceDocument, experience_types: list[str], max_per_doc: int) -> str:
    return (
        f"论文标题: {doc.title}\n"
        f"来源类型: {doc.source_type}\n"
        f"允许的经验类型: {', '.join(experience_types)}\n"
        f"最多输出经验条数: {max_per_doc}\n\n"
        "请从论文中抽取可迁移的科研经验，用于后续 benchmark 任务构建。\n\n"
        "经验类型定义：\n"
        "1. fact / 事实经验：被数据、实验、理论或观察支持的事实性发现，适合回答“在什么条件下，什么现象成立？”。\n"
        "2. strategy / 策略经验：在特定科研条件下有效的方法选择、实验策略或研究启发式，适合回答“应该优先尝试什么方法？为什么？”。\n"
        "3. mechanism / 机制经验：解释某种现象、结果或方法有效性的机制、因果链或理论解释，适合回答“为什么会发生？”。\n"
        "4. boundary / 边界经验：某个结论、方法或规律的适用范围、限制条件和失效条件，适合回答“什么时候不成立？”。\n"
        "5. failure / 失败经验：论文中报告的负结果、无效尝试、不显著结果或副作用，适合回答“哪些看起来合理但实际无效或有风险？”。\n\n"
        "每条经验尽量包含以下字段：\n"
        "- experience_type\n"
        "- experience_title\n"
        "- statement_nature: author_claim | evidence_supported_conclusion | mechanism_explanation | speculative_hypothesis | synthesized_summary\n"
        "- experience_statement\n"
        "- applicability\n"
        "- supporting_evidence\n"
        "- evidence_text\n"
        "- paper_location\n"
        "- is_verifiable\n"
        "- verification_method\n"
        "- possible_counterexample\n"
        "- confidence: 0 到 1 之间\n"
        "- benchmark_transformable: true/false\n"
        "- future_problem\n"
        "- actionable_advice\n"
        "- caveats\n\n"
        "抽取要求：\n"
        "- 不要简单摘要全文，要抽取可复用的科研经验。\n"
        "- 每条经验都要尽量保留成立条件、实验环境、数据集、模型、任务范围或理论前提。\n"
        "- 如果论文证据只能支持作者主张而不能支持强结论，statement_nature 不要写成 evidence_supported_conclusion。\n"
        "- 如果内容更像机制解释或推测，请明确标记 mechanism_explanation 或 speculative_hypothesis。\n"
        "- 如果经验不可验证，is_verifiable 设为 false，并说明 verification_method 为什么不足或不可执行。\n"
        "- 如果证据较弱、结论不显著、或只有局部支持，请降低 confidence。\n"
        "- 如果经验可转化为 benchmark 任务，benchmark_transformable 设为 true；否则设为 false。\n"
        "- 同一类型可以出现多条，但必须彼此有区分度。\n\n"
        "输出格式要求：\n"
        '- 返回单个 JSON 对象：{"experiences":[...]}。\n'
        "- experiences 中每个元素代表一条独立经验，逻辑上等价于一行 JSONL。\n"
        "- 不要输出任何 JSON 之外的解释文字。\n\n"
        f"论文内容:\n{doc.content[:5000]}"
    )


def future_problem_prompt(experience_type: str, doc: SourceDocument) -> str:
    if experience_type == "fact":
        return f"When facing a future problem similar to '{doc.title}', which validated fact should be checked first, and under what conditions does it hold?"
    if experience_type == "strategy":
        return f"When facing a future task similar to '{doc.title}', which strategy or workflow should be attempted first, and why?"
    if experience_type == "mechanism":
        return f"When explaining a future result related to '{doc.title}', which mechanism or causal chain should be examined first?"
    if experience_type == "boundary":
        return f"When transferring insights from '{doc.title}' to a new setting, which scope limits or failure conditions should be checked first?"
    return f"When planning future work related to '{doc.title}', which failed attempt or negative result should be avoided first?"
