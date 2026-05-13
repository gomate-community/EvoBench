from __future__ import annotations

from benchmark.schemas import SourceDocument

SOURCE_REF_REASON = "selected for doc_to_qa generation"
INSTRUCTION = "Generate a question x and answer y grounded directly in the document evidence."


def question_from_sentence(sentence: str, doc: SourceDocument) -> str:
    subject = sentence.split("\uff0c", 1)[0].split(",", 1)[0][:50]
    if len(subject) < 6:
        subject = doc.title[:50]
    return f"According to the document, what is the key information about {subject}?"


def answer_from_sentence(sentence: str) -> str:
    return sentence.strip(" .")
