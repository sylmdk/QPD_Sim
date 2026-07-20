from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(
    path: str | Path,
    required_sections: tuple[str, ...] = (),
) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    missing = [key for key in required_sections if key not in config]
    if missing:
        raise ValueError(f"Missing YAML sections: {', '.join(missing)}")
    return config


def copy_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a mapping")
    return deepcopy(value)
