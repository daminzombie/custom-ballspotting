import dataclasses
import json
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


def load_json_config(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def resolve_config_path(path: str, config_path: str | None) -> str:
    if Path(path).is_absolute() or config_path is None:
        return path
    return str((Path(config_path).resolve().parent / path).resolve())


def dataclass_from_dict(cls: type[T], values: dict[str, Any]) -> T:
    fields = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in fields})


def merge_values(
    config_values: dict[str, Any],
    cli_values: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(config_values)
    for key, value in cli_values.items():
        if value is not None:
            merged[key] = value
    return merged
