"""Private evaluators for the frozen ten-task V2 perception suite."""

from __future__ import annotations

from typing import Any

from .long_horizon import PhysicalWeighingEvaluator
from .vlabench.button import VLABenchPhysicalPropertyButtonEvaluator


PERCEPTION_V2_SUITE = "perception_v2"
PERCEPTION_V2_TASK_COUNT = 10

# These tasks reuse the real three-button VLABench fixture. The target is a
# property of the rendered candidates; the answer index remains host-private.
_BUTTON_TARGET_INDEX_BY_TASK_ID = {
    5: 0,  # ball
    8: 1,  # polished / highest-reflectance panel (centre after layout swap)
    9: 2,  # largest cylinder
}


class PerceptionV2ButtonEvaluator(VLABenchPhysicalPropertyButtonEvaluator):
    """Contact-gated checker for a fixed visual-selection answer."""

    schema_version = "libero.perception_v2.button.v1"

    def __init__(self, env: Any, *, target_index: int, task_id: int) -> None:
        if not 0 <= int(target_index) < len(self.button_names):
            raise ValueError(f"invalid perception button target index: {target_index}")
        self.env = env
        self.family_id = "perception_v2"
        self.seed = -1
        self.task_id = int(task_id)
        self._question = None
        self._target_index = int(target_index)
        self._target_button = self.button_names[self._target_index]
        self.reset()


def perception_v2_private_evaluator(
    env: Any,
    *,
    suite: str,
    task_id: int,
) -> PhysicalWeighingEvaluator | PerceptionV2ButtonEvaluator | None:
    """Select the authoritative private checker for non-terminal V2 tasks."""

    if suite != PERCEPTION_V2_SUITE:
        return None
    normalized_task_id = int(task_id)
    if not 0 <= normalized_task_id < PERCEPTION_V2_TASK_COUNT:
        raise ValueError(
            f"unknown {PERCEPTION_V2_SUITE} task_id={normalized_task_id}"
        )
    if normalized_task_id == 7:
        return PhysicalWeighingEvaluator(env, task_id=normalized_task_id)
    target_index = _BUTTON_TARGET_INDEX_BY_TASK_ID.get(normalized_task_id)
    if target_index is not None:
        return PerceptionV2ButtonEvaluator(
            env,
            target_index=target_index,
            task_id=normalized_task_id,
        )
    return None


__all__ = [
    "PERCEPTION_V2_SUITE",
    "PERCEPTION_V2_TASK_COUNT",
    "PerceptionV2ButtonEvaluator",
    "perception_v2_private_evaluator",
]
