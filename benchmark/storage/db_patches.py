"""对 ``benchmark.storage.db.init_storage`` 的 monkey patch：在保留原行为的基础上额外建好分类目录骨架。

不删旧文件，仅追加目录创建。补丁幂等。
"""

from __future__ import annotations

from benchmark.storage import data_layout, db


def install() -> None:
    if getattr(db.init_storage, "__wrapped_by_data_layout__", False):
        return

    _orig_init_storage = db.init_storage

    def _patched_init_storage() -> None:
        _orig_init_storage()
        data_layout.ensure_layout()

    _patched_init_storage.__wrapped_by_data_layout__ = True  # type: ignore[attr-defined]
    db.init_storage = _patched_init_storage  # type: ignore[assignment]
