"""Access to parameter dict entries by dotted path, e.g. "rw_motor.max_rpm"."""

from typing import Any


def get_path(params: dict, path: str) -> Any:
    node: Any = params
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(path)
        node = node[key]
    return node


def set_path(params: dict, path: str, value: Any) -> None:
    """Set an existing entry, missing keys raise KeyError."""
    keys = path.split(".")
    node: Any = params
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(path)
        node = node[key]
    if not isinstance(node, dict) or keys[-1] not in node:
        raise KeyError(path)
    node[keys[-1]] = value


def is_numeric_leaf(value: Any) -> bool:
    # Switches like lt_motor.activate are bools and can't be dispersed
    return isinstance(value, (int, float)) and not isinstance(value, bool)
