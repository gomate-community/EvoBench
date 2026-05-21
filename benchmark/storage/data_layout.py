"""data/ 目录分类布局：把扁平的 corpus.jsonl / samples.jsonl 拆成按来源 / 技能 / 状态分类的两级目录。

设计原则：
- 纯函数模块，不引入业务对象之外的依赖；
- main 分支文件零修改，仅通过 ``repository_patches`` / ``db_patches`` 注入到 ``BenchmarkRepository`` 与 ``init_storage``；
- 用户显式给 ``--corpus-jsonl`` / ``--samples-jsonl`` 仍走原单文件模式，本模块不介入。

目录约定：
    data/
      corpus/{source_type}/{topic}.jsonl
      corpus/_legacy/*.jsonl           # 旧扁平文件迁移落脚
      samples/{skill_id}/{status}.jsonl
      samples/_legacy/*.jsonl
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from benchmark.core.config import settings

DEFAULT_CORPUS_DIR = "./data/corpus"
DEFAULT_SAMPLES_DIR = "./data/samples"
LEGACY_SUBDIR = "_legacy"
DEFAULT_SOURCE = "user"
DEFAULT_TOPIC = "general"
DEFAULT_STATUS = "pending"

_SLUG_RE = re.compile(r"[^\w\-\u4e00-\u9fff]+", re.UNICODE)


def corpus_root() -> Path:
    return Path(getattr(settings, "corpus_dir", DEFAULT_CORPUS_DIR))


def samples_root() -> Path:
    return Path(getattr(settings, "samples_dir", DEFAULT_SAMPLES_DIR))


def slugify(text: str | None, *, max_len: int = 40, fallback: str = "untitled") -> str:
    """把任意字符串规范化为安全的文件名分量：保留中英文与数字，其他替换为 ``_``。"""
    if not text:
        return fallback
    cleaned = _SLUG_RE.sub("_", str(text).strip())
    cleaned = cleaned.strip("_") or fallback
    return cleaned[:max_len]


def _doc_topic(doc: Any) -> str:
    """从 SourceDocument 推断 topic：优先 metadata.topic，回退到 title 前 20 字。"""
    metadata = getattr(doc, "metadata", None) or {}
    topic = metadata.get("topic") if isinstance(metadata, dict) else None
    if topic:
        return slugify(topic)
    title = getattr(doc, "title", None)
    if title:
        return slugify(title, max_len=20)
    return DEFAULT_TOPIC


def _doc_source(doc: Any) -> str:
    source_type = getattr(doc, "source_type", None)
    return slugify(source_type, max_len=30) if source_type else DEFAULT_SOURCE


def resolve_corpus_path(doc: Any) -> Path:
    """根据 SourceDocument 决定它该落到 ``corpus/{source}/{topic}.jsonl``。"""
    return corpus_root() / _doc_source(doc) / f"{_doc_topic(doc)}.jsonl"


def resolve_sample_path(sample: Any) -> Path:
    """根据 UnifiedSample 决定它该落到 ``samples/{skill_id}/{status}.jsonl``。"""
    skill_id = getattr(sample, "skill_id", None) or "unknown"
    status = getattr(sample, "status", None) or DEFAULT_STATUS
    return samples_root() / slugify(skill_id, max_len=64) / f"{slugify(status, max_len=32)}.jsonl"


def _iter_jsonl_under(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*.jsonl")):
        if path.is_file():
            yield path


def iter_corpus_paths() -> list[Path]:
    """枚举 ``data/corpus/`` 下所有 jsonl（含 ``_legacy/``）。"""
    return list(_iter_jsonl_under(corpus_root()))


def iter_sample_paths() -> list[Path]:
    """枚举 ``data/samples/`` 下所有 jsonl（含 ``_legacy/``）。"""
    return list(_iter_jsonl_under(samples_root()))


def ensure_layout() -> None:
    """建好分类目录骨架与 ``_legacy/`` 子目录，幂等。"""
    (corpus_root() / LEGACY_SUBDIR).mkdir(parents=True, exist_ok=True)
    (samples_root() / LEGACY_SUBDIR).mkdir(parents=True, exist_ok=True)
