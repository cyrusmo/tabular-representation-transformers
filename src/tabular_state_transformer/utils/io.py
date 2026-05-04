from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_merged_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = read_yaml(config_path)
    base_dir = config_path.parent.parent.parent if config_path.parts else Path.cwd()
    for key in ("data_config", "model_config"):
        if key in config:
            child_path = Path(config[key])
            if not child_path.is_absolute():
                child_path = base_dir / child_path
            config[key.removesuffix("_config")] = read_yaml(child_path)
    return config
