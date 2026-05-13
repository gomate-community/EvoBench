from __future__ import annotations

from pathlib import Path

from benchmark.core.config import settings


def _ensure_jsonl_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def init_storage() -> None:
    _ensure_jsonl_file(Path(settings.corpus_jsonl_path))
    _ensure_jsonl_file(Path(settings.samples_jsonl_path))


def init_db() -> None:
    """Backward-compatible alias kept for CLI/API imports."""
    init_storage()
