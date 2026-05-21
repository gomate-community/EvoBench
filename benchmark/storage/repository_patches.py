"""对 ``BenchmarkRepository`` 的 monkey patch：把读写从单一扁平 jsonl 切换到分类目录。

激活后行为变化：
- ``upsert_document``：按 ``resolve_corpus_path(doc)`` 写入分类子文件；
- ``upsert_sample``：按 ``resolve_sample_path(sample)`` 写入；
- ``list_documents`` / ``list_samples``：遍历分类目录所有 jsonl（含 ``_legacy/``），合并去重后再走原过滤逻辑；
- 用户显式给 ``BenchmarkRepository(corpus_path=..., samples_path=...)`` 时打 ``_explicit_paths`` 标志，所有方法退回原逻辑（即仍写 / 读那一个文件），保证 dev 调试与 tmp_path 测试不受影响；
- ``load_corpus_jsonl(path)`` 不动。

补丁幂等：通过 ``__wrapped_by_data_layout__`` 标记。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.storage import data_layout
from benchmark.storage.repository import BenchmarkRepository


def _is_explicit(self: BenchmarkRepository) -> bool:
    return bool(getattr(self, "_explicit_paths", False))


def install() -> None:
    if getattr(BenchmarkRepository.__init__, "__wrapped_by_data_layout__", False):
        return

    _orig_init = BenchmarkRepository.__init__
    _orig_upsert_document = BenchmarkRepository.upsert_document
    _orig_upsert_sample = BenchmarkRepository.upsert_sample
    _orig_list_documents = BenchmarkRepository.list_documents
    _orig_list_samples = BenchmarkRepository.list_samples

    def _patched_init(
        self: BenchmarkRepository,
        *,
        corpus_path: str | Path | None = None,
        samples_path: str | Path | None = None,
        items_path: str | Path | None = None,
    ) -> None:
        _orig_init(
            self,
            corpus_path=corpus_path,
            samples_path=samples_path,
            items_path=items_path,
        )
        # 显式给单文件路径时退回原行为
        self._explicit_paths = bool(corpus_path or samples_path)
        if not self._explicit_paths:
            data_layout.ensure_layout()

    def _patched_upsert_document(self: BenchmarkRepository, doc: Any) -> None:
        if _is_explicit(self):
            return _orig_upsert_document(self, doc)
        target = data_layout.resolve_corpus_path(doc)
        self._upsert_model(target, "source_id", doc)

    def _patched_upsert_sample(self: BenchmarkRepository, sample: Any) -> None:
        if _is_explicit(self):
            return _orig_upsert_sample(self, sample)
        target = data_layout.resolve_sample_path(sample)
        self._upsert_model(target, "sample_id", sample)

    def _load_all(self: BenchmarkRepository, paths: list[Path], validator) -> list:
        seen: set[str] = set()
        merged_records: list[dict[str, Any]] = []
        for p in paths:
            for record in self._read_jsonl(p):
                key = record.get("source_id") or record.get("sample_id")
                if key is None:
                    merged_records.append(record)
                    continue
                if key in seen:
                    continue
                seen.add(key)
                merged_records.append(record)
        return [validator(r) for r in merged_records]

    def _patched_list_documents(
        self: BenchmarkRepository,
        *,
        limit: int = 100,
        source_type: str | None = None,
        language: str | None = None,
        min_trust_level: int | None = None,
        topic: str | None = None,
    ):
        if _is_explicit(self):
            return _orig_list_documents(
                self,
                limit=limit,
                source_type=source_type,
                language=language,
                min_trust_level=min_trust_level,
                topic=topic,
            )
        from benchmark.schemas import SourceDocument

        docs = _load_all(self, data_layout.iter_corpus_paths(), SourceDocument.model_validate)
        if source_type:
            docs = [d for d in docs if d.source_type == source_type]
        if language:
            docs = [d for d in docs if d.language == language]
        if min_trust_level is not None:
            docs = [d for d in docs if d.trust_level >= min_trust_level]
        if topic:
            topic_lower = topic.lower()
            docs = [
                d for d in docs
                if topic_lower in d.title.lower() or topic_lower in d.content.lower()
            ]
        return docs[:limit]

    def _patched_list_samples(
        self: BenchmarkRepository,
        *,
        status: str | None = None,
        task_type: Any = None,
        skill_id: str | None = None,
        limit: int = 100,
    ):
        if _is_explicit(self):
            return _orig_list_samples(
                self,
                status=status,
                task_type=task_type,
                skill_id=skill_id,
                limit=limit,
            )
        from benchmark.schemas import TaskType, UnifiedSample

        samples = _load_all(self, data_layout.iter_sample_paths(), UnifiedSample.model_validate)
        if status:
            samples = [s for s in samples if s.status == status]
        if task_type:
            value = task_type.value if isinstance(task_type, TaskType) else str(task_type)
            samples = [s for s in samples if s.task_type.value == value]
        if skill_id:
            samples = [s for s in samples if s.skill_id == skill_id]
        return samples[:limit]

    _patched_init.__wrapped_by_data_layout__ = True  # type: ignore[attr-defined]
    BenchmarkRepository.__init__ = _patched_init  # type: ignore[assignment]
    BenchmarkRepository.upsert_document = _patched_upsert_document  # type: ignore[assignment]
    BenchmarkRepository.upsert_sample = _patched_upsert_sample  # type: ignore[assignment]
    BenchmarkRepository.list_documents = _patched_list_documents  # type: ignore[assignment]
    BenchmarkRepository.list_samples = _patched_list_samples  # type: ignore[assignment]
