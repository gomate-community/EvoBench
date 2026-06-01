"""value_qa skill - 4-layer mainstream-value taxonomy loader and helpers.

Loads ``configs/value_taxonomy.yaml`` (single source of truth for value keys)
and exposes utilities for prompt construction and verifier validation.

The taxonomy layout is documented in the yaml header; this module only
provides typed accessors and is import-safe (yaml is loaded lazily on first
access so that ``import benchmark.bootstrap`` does not pay the cost).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Default location, resolvable both from repo root and installed layout.
_DEFAULT_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "value_taxonomy.yaml"
)


@lru_cache(maxsize=4)
def load_taxonomy(path: str | None = None) -> dict[str, Any]:
    """Load and cache the taxonomy yaml.

    Pass ``path=None`` to use the default ``configs/value_taxonomy.yaml``.
    """

    target = Path(path) if path else _DEFAULT_TAXONOMY_PATH
    if not target.exists():
        raise FileNotFoundError(f"value taxonomy not found: {target}")
    with target.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def list_layer_keys(taxonomy: dict[str, Any], layer: str) -> list[str]:
    """Flat list of all keys under a layer (D layer flattens across domains)."""

    layer_def = taxonomy.get("layers", {}).get(layer, {})
    if not layer_def:
        return []
    keys: list[str] = []
    if "items" in layer_def:
        keys.extend(item["key"] for item in layer_def["items"] if "key" in item)
    if "domains" in layer_def:
        for domain_def in layer_def["domains"].values():
            keys.extend(item["key"] for item in domain_def.get("items", []) if "key" in item)
    return keys


def keys_for_dimension(taxonomy: dict[str, Any], dimension: str) -> set[str]:
    """All legal value_keys for a given dimension (cultural_fidelity / ethics_compliance)."""

    layers = taxonomy.get("dimension_layer_map", {}).get(dimension, [])
    out: set[str] = set()
    for layer in layers:
        out.update(list_layer_keys(taxonomy, layer))
    return out


def anchors_for_dimension(taxonomy: dict[str, Any], dimension: str) -> list[dict[str, Any]]:
    """Flat anchor list (key/label/gloss/layer) used to inject into prompts."""

    layers = taxonomy.get("dimension_layer_map", {}).get(dimension, [])
    out: list[dict[str, Any]] = []
    for layer in layers:
        layer_def = taxonomy.get("layers", {}).get(layer, {})
        for item in layer_def.get("items", []):
            out.append({**item, "layer": layer})
        for domain_name, domain_def in layer_def.get("domains", {}).items():
            for item in domain_def.get("items", []):
                out.append({**item, "layer": layer, "domain": domain_name})
    return out


def hint_for_dimension(taxonomy: dict[str, Any], dimension: str) -> list[str]:
    return taxonomy.get("defaults", {}).get("hints", {}).get(dimension, [])


def uncertain_strategy(taxonomy: dict[str, Any]) -> str:
    return taxonomy.get("defaults", {}).get("uncertain_strategy", "skip")


VALID_DIMENSIONS = ("cultural_fidelity", "ethics_compliance")
VALID_LEVELS = ("high", "medium", "low")
