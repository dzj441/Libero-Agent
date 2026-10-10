"""Safe loading of task-level visual references.

Task references are task specifications, not demonstrations.  They contain no
actions, state trajectories, or object metadata.  The resolver is intentionally
allow-listed so an ordinary LIBERO task can never acquire a reference by
changing a path supplied by an agent.  The live environment may materialize
the reference alongside the current public observation; this module only
resolves and validates the repository-owned source asset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image


_REFERENCE_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "task_references"
)
_TASK_REFERENCES: Final[dict[tuple[str, int], Path]] = {
    ("long_horizon_v2", 5): (
        Path("long_horizon_v2") / "prepare_cooking_goal_rgb.png"
    ),
}


def resolve_task_reference_path(suite: str, task_id: int) -> Path | None:
    """Resolve the allow-listed reference path for ``suite`` and ``task_id``.

    Only explicitly registered visual-goal variants have task-level RGB
    references. ``None`` is returned for textual variants and every other
    suite/task rather than attempting to interpret a caller-provided path.
    All path components are checked for symlinks and containment under the
    repository asset root.
    """

    relative_path = _TASK_REFERENCES.get((str(suite), int(task_id)))
    if relative_path is None:
        return None

    root = _REFERENCE_ROOT
    candidate = root / relative_path
    _reject_symlink_components(candidate, root)

    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve(strict=False)
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("task reference path escapes the repository asset root") from exc
    if not candidate_resolved.is_file():
        raise FileNotFoundError(f"task reference does not exist: {candidate_resolved}")
    return candidate_resolved


def load_task_reference_rgb(suite: str, task_id: int) -> np.ndarray | None:
    """Load an allow-listed task reference as a copied RGB uint8 HWC array."""

    path = resolve_task_reference_path(suite, task_id)
    if path is None:
        return None

    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError(
                    f"task reference must be an RGB PNG, got mode {image.mode!r}"
                )
            array = np.asarray(image)
    except (OSError, ValueError):
        raise

    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            "task reference must load as uint8 HWC3, "
            f"got dtype={array.dtype}, shape={array.shape}"
        )
    return np.ascontiguousarray(array.copy())


def _reject_symlink_components(path: Path, root: Path) -> None:
    """Reject a symlink in any component between ``root`` and ``path``."""

    root_absolute = root.absolute()
    path_absolute = path.absolute()
    try:
        relative_parts = path_absolute.relative_to(root_absolute).parts
    except ValueError as exc:
        raise ValueError("task reference path is outside the repository asset root") from exc

    current = root_absolute
    if current.is_symlink():
        raise ValueError("task reference asset root must not be a symlink")
    for component in relative_parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"task reference path must not contain symlinks: {current}")


__all__ = ["load_task_reference_rgb", "resolve_task_reference_path"]
