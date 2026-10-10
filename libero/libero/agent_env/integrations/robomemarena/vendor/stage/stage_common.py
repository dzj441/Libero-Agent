"""Minimal simulator helpers used by the frozen RoboMemArena stage checker."""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_env_class() -> type:
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv


def _resolve_body_id(env: Any, name: str) -> int | None:
    candidates = [name]
    if not name.endswith("_main"):
        candidates.append(f"{name}_main")
    else:
        candidates.append(name[:-5])
    for candidate in candidates:
        try:
            return int(env.sim.model.body_name2id(candidate))
        except Exception:
            continue
    return None


def _body_pos(env: Any, name: str) -> np.ndarray | None:
    body_id = _resolve_body_id(env, name)
    if body_id is None:
        return None
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float32)

