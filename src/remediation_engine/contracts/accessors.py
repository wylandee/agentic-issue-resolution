"""Typed accessors for Pydantic-or-dict state projections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def model_or_dict_value(item: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a Pydantic object or a mapping.

    Args:
        item: Object or mapping containing the requested field.
        key: Field name to read.
        default: Value returned when the field is absent.

    Returns:
        The field value, or ``default`` when it is absent.
    """
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)
