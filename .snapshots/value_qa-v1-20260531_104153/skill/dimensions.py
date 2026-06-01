"""value_qa skill - 加载 ``configs/value_evaluation_dimensions.yaml``。

体系：
  - 一级指标 primary_metrics（2 项）
  - 二级评估维度 secondary_dimensions（6 项；每项归属一个一级）
  - medium_pattern_ids（letter_vs_spirit / partial_correct / hedged / unknown）

与 4 层 taxonomy 的关系：
  - 4 层 taxonomy 不再作为档位判定锚点
  - 通过 ``reference_taxonomy_layers`` 软关联推荐参考层（A/B/C/D）
  - 生成 high/low 时可让 LLM 引用具体 value_key 让答案具象，但不强制
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_DIMENSIONS_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "value_evaluation_dimensions.yaml"
)

# 一级指标（id -> id_en）。供 metadata.value_dimension（保持 cultural_fidelity / ethics_compliance）
# 与 metadata.primary_metric（中文）相互映射。
PRIMARY_ID_TO_EN = {
    "文化忠实度": "cultural_fidelity",
    "伦理合规度": "ethics_compliance",
}
PRIMARY_EN_TO_ID = {v: k for k, v in PRIMARY_ID_TO_EN.items()}

# 二级评估维度 6 项（中文 id 即对外 id）
SECONDARY_DIMENSION_IDS = (
    "文化元素符合度",
    "行为社会规范符合度",
    "价值观忠实度",
    "公平合规度",
    "伤害风险合规度",
    "隐私合规度",
)

MEDIUM_PATTERN_IDS = ("letter_vs_spirit", "partial_correct", "hedged", "unknown")


@lru_cache(maxsize=4)
def load_dimensions(path: str | None = None) -> dict[str, Any]:
    """加载并缓存评估维度 yaml。"""

    target = Path(path) if path else _DEFAULT_DIMENSIONS_PATH
    if not target.exists():
        raise FileNotFoundError(f"value evaluation dimensions not found: {target}")
    with target.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def list_secondary(data: dict[str, Any]) -> list[dict[str, Any]]:
    """返回 6 个二级评估维度的完整定义列表。"""

    return list(data.get("secondary_dimensions") or [])


def get_secondary(data: dict[str, Any], dimension_id: str) -> dict[str, Any] | None:
    """按二级维度 id 查找定义。"""

    for d in data.get("secondary_dimensions") or []:
        if d.get("id") == dimension_id:
            return d
    return None


def primary_of(data: dict[str, Any], dimension_id: str) -> str | None:
    """二级维度 -> 所属一级指标 id（中文）。"""

    d = get_secondary(data, dimension_id)
    return d.get("primary") if d else None


def primary_id_en(primary_id: str) -> str:
    """一级指标中文 id -> 英文（用于 value_dimension 字段兼容）。"""

    return PRIMARY_ID_TO_EN.get(primary_id, "")


def medium_pattern_ids(data: dict[str, Any]) -> list[str]:
    return list(data.get("medium_pattern_ids") or list(MEDIUM_PATTERN_IDS))


def medium_pattern_for(data: dict[str, Any], dimension_id: str) -> list[dict[str, Any]]:
    """某二级维度下的 medium_pattern 列表（含 hint）。"""

    d = get_secondary(data, dimension_id)
    return list(d.get("medium_patterns") or []) if d else []


def reference_layers(data: dict[str, Any], dimension_id: str) -> list[str]:
    """某二级维度推荐的 4 层 taxonomy 参考层。"""

    d = get_secondary(data, dimension_id)
    return list(d.get("reference_taxonomy_layers") or []) if d else []


def uncertain_strategy(data: dict[str, Any]) -> str:
    return data.get("defaults", {}).get("uncertain_strategy", "skip")


def list_secondary_brief(data: dict[str, Any]) -> list[dict[str, str]]:
    """供 ROUTE_PROMPT 列出 6 个维度的简要清单（id / primary / definition）。"""

    out: list[dict[str, str]] = []
    for d in list_secondary(data):
        out.append(
            {
                "id": d.get("id", ""),
                "primary": d.get("primary", ""),
                "definition": (d.get("definition") or "").strip(),
            }
        )
    return out
