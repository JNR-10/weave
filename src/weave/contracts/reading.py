"""Strict readers for the team's JSON inputs (cluster state, candidate sets, experiment plans).

Each reader raises ValueError whose message starts with the field's path ("devices[1].device_id:
...") so whoever fixes the input knows exactly where to look. They were first written twice, in
weave.scheduler.candidates and weave.scheduler.policies, and live here so every reader of a
committed contract refuses bad input the same way.
"""

from typing import Any


def field_path(parent: str, key: str | int) -> str:
    """The dotted/indexed path of `key` under `parent`, as error messages print it."""
    if isinstance(key, int):
        return f"{parent}[{key}]"
    return f"{parent}.{key}" if parent else key


def required(obj: dict[str, Any], key: str, parent: str) -> Any:
    """The value at `key`; a missing key is an error, never a default."""
    if key not in obj:
        raise ValueError(f"{field_path(parent, key)}: required")
    return obj[key]


def as_object(value: Any, path: str) -> dict[str, Any]:
    """`value` itself if it is a JSON object."""
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(value).__name__}")
    return value


def text_field(obj: dict[str, Any], key: str, parent: str) -> str:
    """A required non-empty string."""
    value = required(obj, key, parent)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_path(parent, key)}: expected a non-empty string, got {value!r}")
    return value


def bool_field(obj: dict[str, Any], key: str, parent: str) -> bool:
    """A required JSON true or false; 0 and 1 are refused."""
    value = required(obj, key, parent)
    if not isinstance(value, bool):
        raise ValueError(f"{field_path(parent, key)}: expected true or false, got {value!r}")
    return value


def positive_int(value: Any, path: str) -> int:
    """`value` itself if it is an integer of at least 1."""
    # bool is a subclass of int in Python, so `type(...) is int` keeps true/false out.
    if type(value) is not int or value < 1:
        raise ValueError(f"{path}: expected a positive integer, got {value!r}")
    return value


def positive_field(obj: dict[str, Any], key: str, parent: str) -> int:
    """A required integer field of at least 1."""
    return positive_int(required(obj, key, parent), field_path(parent, key))
