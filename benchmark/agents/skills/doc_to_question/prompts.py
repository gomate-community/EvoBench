from __future__ import annotations


SOURCE_REF_REASON = "selected for doc_to_question generation"
INSTRUCTION = "Generate a question x that is answerable and verifiable from the document."


def make_question(sentence: str) -> str:
    head = sentence[:90].rstrip(",.;:!? ")
    return f"What core fact is supported by this evidence? Evidence hint: {head}"
