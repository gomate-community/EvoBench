SOURCE_REF_REASON = "selected for doc_to_qa_steps generation"
INSTRUCTION = "Generate question x, concise external solution steps T, and answer y from the document."


def question_from_sentence(sentence: str) -> str:
    head = sentence[:70].rstrip(",.;:!? ")
    return f"Based on the document, how should this information be summarized: {head}?"


def default_trace() -> list[str]:
    return [
        "Locate the evidence sentence most relevant to the question.",
        "Check that the sentence directly supports the answer without outside facts.",
        "Compress the subject, action, and result into a concise answer.",
    ]
