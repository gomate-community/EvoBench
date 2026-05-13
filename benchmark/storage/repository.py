from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from benchmark.schemas import BenchmarkItem, SourceDocument, TaskType, UnifiedSample
from benchmark.storage.db import init_storage
from benchmark.core.config import settings

T = TypeVar("T")


class BenchmarkRepository:
    def __init__(
        self,
        *,
        corpus_path: str | Path | None = None,
        samples_path: str | Path | None = None,
        items_path: str | Path | None = None,
    ):
        self.corpus_path = Path(corpus_path or settings.corpus_jsonl_path)
        self.samples_path = Path(samples_path or settings.samples_jsonl_path)
        self.items_path = Path(items_path or settings.items_jsonl_path)
        init_storage()

    def upsert_document(self, doc: SourceDocument) -> None:
        self._upsert_model(self.corpus_path, "source_id", doc)

    def upsert_documents(self, docs: Iterable[SourceDocument]) -> None:
        for doc in docs:
            self.upsert_document(doc)

    def list_documents(
        self,
        *,
        limit: int = 100,
        source_type: str | None = None,
        language: str | None = None,
        min_trust_level: int | None = None,
        topic: str | None = None,
    ) -> list[SourceDocument]:
        docs = self._load_models(self.corpus_path, SourceDocument.model_validate)
        if source_type:
            docs = [doc for doc in docs if doc.source_type == source_type]
        if language:
            docs = [doc for doc in docs if doc.language == language]
        if min_trust_level is not None:
            docs = [doc for doc in docs if doc.trust_level >= min_trust_level]
        if topic:
            topic_lower = topic.lower()
            docs = [
                doc
                for doc in docs
                if topic_lower in doc.title.lower() or topic_lower in doc.content.lower()
            ]
        return docs[:limit]

    def load_corpus_jsonl(self, path: str | Path) -> list[SourceDocument]:
        return self._load_models(Path(path), SourceDocument.model_validate)

    def upsert_item(self, item: BenchmarkItem) -> None:
        self._upsert_model(self.items_path, "question_id", item)

    def list_items(self, status: str | None = None, limit: int = 100) -> list[BenchmarkItem]:
        items = self._load_models(self.items_path, BenchmarkItem.model_validate)
        if status:
            items = [item for item in items if item.status == status]
        return items[:limit]

    def upsert_sample(self, sample: UnifiedSample) -> None:
        self._upsert_model(self.samples_path, "sample_id", sample)

    def list_samples(
        self,
        *,
        status: str | None = None,
        task_type: TaskType | str | None = None,
        skill_id: str | None = None,
        limit: int = 100,
    ) -> list[UnifiedSample]:
        samples = self._load_models(self.samples_path, UnifiedSample.model_validate)
        if status:
            samples = [sample for sample in samples if sample.status == status]
        if task_type:
            value = task_type.value if isinstance(task_type, TaskType) else str(task_type)
            samples = [sample for sample in samples if sample.task_type.value == value]
        if skill_id:
            samples = [sample for sample in samples if sample.skill_id == skill_id]
        return samples[:limit]

    def _upsert_model(self, path: Path, key_field: str, model: Any) -> None:
        records = self._read_jsonl(path)
        payload = model.model_dump(mode="json")
        key_value = payload[key_field]
        updated = False
        for index, record in enumerate(records):
            if record.get(key_field) == key_value:
                records[index] = payload
                updated = True
                break
        if not updated:
            records.append(payload)
        self._write_jsonl(path, records)

    def _load_models(self, path: Path, validator: Callable[[dict[str, Any]], T]) -> list[T]:
        return [validator(record) for record in self._read_jsonl(path)]

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    items.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc.msg}") from exc
        return items

    def _write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
