"""Public contract for initial task-entity annotations.

The public identifiers deliberately carry no task role, class name, or private
LIBERO instance name.  They are only stable across cameras within one episode.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TASK_ENTITY_ANNOTATION_SCHEMA_VERSION = "libero.task_entities.v1"
TASK_ENTITY_PREFIX = "entity_"


def task_entity_id(index: int) -> str:
    """Return the canonical anonymous identifier for one task entity."""

    if index < 0:
        raise ValueError("task-entity index must be non-negative")
    return f"{TASK_ENTITY_PREFIX}{index:03d}"


def validate_task_entity_mapping(value: Any) -> tuple[str, ...]:
    """Validate a non-empty, contiguous anonymous entity mapping."""

    if not isinstance(value, Mapping) or not value:
        raise ValueError("task_entities must be a non-empty mapping")
    identifiers = tuple(value)
    expected = tuple(task_entity_id(index) for index in range(len(identifiers)))
    if identifiers != expected:
        raise ValueError(
            "task-entity identifiers must be contiguous entity_000... entries"
        )
    return identifiers
