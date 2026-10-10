"""Private temporal evaluators for the diverse V2 long-horizon candidates.

The public BDDL goals deliberately remain flat terminal predicates so ordinary
LIBERO tooling can inspect them.  This module is evaluator-private and restores
the event history required by the source tasks without exposing object ids,
mass classes, or progress to the Agent.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .manipulation import (
    _OrderedEvaluator,
    _bodies_in_contact,
    _body_position,
    _joint_position,
    _sim_time,
)


LONG_HORIZON_V2_SUITE = "long_horizon_v2"
LONG_HORIZON_V2_TASK_NAMES = (
    "rotate_three_bowls_using_empty_plate",
    "weigh_and_place_heaviest_mug",
    "place_mug_and_brew_coffee",
    "prepare_cooking_with_two_stoves_and_microwave",
    "prepare_cooking_text_goal",
    "prepare_cooking_goal_image",
)


def _domain(env: Any) -> Any:
    return getattr(env, "env", env)


def _evaluate_predicate(env: Any, predicate: tuple[str, ...]) -> bool:
    evaluator = getattr(_domain(env), "_eval_predicate", None)
    if not callable(evaluator):
        return False
    try:
        return bool(evaluator(list(predicate)))
    except Exception:
        return False


def _object_grasped(env: Any, logical_name: str) -> bool:
    domain = _domain(env)
    try:
        object_model = domain.objects_dict[logical_name]
        return bool(
            domain._check_grasp(
                gripper=domain.robots[0].gripper,
                object_geoms=object_model.contact_geoms,
            )
        )
    except Exception:
        return False


def _eef_position(env: Any) -> np.ndarray | None:
    domain = _domain(env)
    try:
        site_id = int(domain.robots[0].eef_site_id)
        return np.asarray(
            domain.sim.data.site_xpos[site_id],
            dtype=np.float64,
        ).copy()
    except Exception:
        return None


def _released_and_clear(
    env: Any,
    logical_name: str,
    *,
    clearance_m: float = 0.055,
) -> bool:
    if _object_grasped(env, logical_name):
        return False
    object_position = _body_position(env, logical_name)
    eef_position = _eef_position(env)
    if object_position is None or eef_position is None:
        return False
    return bool(np.linalg.norm(object_position - eef_position) >= clearance_m)


def _set_geom_alpha(env: Any, suffix: str, alpha: float) -> bool:
    model = env.sim.model
    changed = False
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id)
        if name is not None and str(name).endswith(suffix):
            model.geom_rgba[geom_id, 3] = float(alpha)
            changed = True
    return changed


class _StrictPredicateSequenceEvaluator(_OrderedEvaluator):
    """Require predicate transitions in order and latch early transitions."""

    schema_version = "libero.long_horizon_v2_private.v1"
    stage_predicates: tuple[tuple[str, ...], ...] = ()
    release_objects: tuple[str | None, ...] = ()

    def _reset_task(self) -> None:
        if len(self.stage_names) != len(self.stage_predicates):
            raise RuntimeError("stage names and predicates must have equal length")
        if self.release_objects and len(self.release_objects) != len(
            self.stage_names
        ):
            raise RuntimeError("release requirements must match the stage count")
        self._ordering_violation: str | None = None

    def _stage_is_complete(self, index: int) -> bool:
        if not _evaluate_predicate(self.env, self.stage_predicates[index]):
            return False
        if not self.release_objects:
            return True
        object_name = self.release_objects[index]
        return object_name is None or _released_and_clear(self.env, object_name)

    def _sample(self) -> None:
        if self._ordering_violation is not None or self._stage_index >= len(
            self.stage_names
        ):
            return

        # Completing any later semantic step before the current one makes the
        # trace invalid, even if the Agent subsequently undoes and redoes it.
        for later_index in range(self._stage_index + 1, len(self.stage_names)):
            if self._stage_is_complete(later_index):
                self._ordering_violation = (
                    f"{self.stage_names[later_index]}_before_"
                    f"{self.stage_names[self._stage_index]}"
                )
                return

        if self._stage_is_complete(self._stage_index):
            self._mark_next(self.stage_names[self._stage_index])

    def _terminal_predicates(self) -> tuple[tuple[str, ...], ...]:
        return self.stage_predicates

    def result(self) -> dict[str, Any]:
        terminal = {
            " ".join(predicate): _evaluate_predicate(self.env, predicate)
            for predicate in self._terminal_predicates()
        }
        success = bool(
            self._ordering_violation is None
            and self._stage_index == len(self.stage_names)
            and all(terminal.values())
        )
        return {
            **self._base_result(success=success),
            "terminal_state_checks": terminal,
            "ordering_violation": self._ordering_violation,
            "failure_reason": (
                None
                if success
                else self._ordering_violation or "ordered_sequence_incomplete"
            ),
        }


class CyclicBowlRotationEvaluator(_StrictPredicateSequenceEvaluator):
    """Restore LIBERO-Mem Task 8's four ordered buffer transitions."""

    bowl_names = (
        "akita_black_bowl_1",
        "akita_black_bowl_2",
        "akita_black_bowl_3",
    )
    tabletop_body_name = "table"

    stage_names = (
        "01_Move_Left_Bowl_To_Empty_Plate",
        "02_Move_Middle_Bowl_To_Left_Plate",
        "03_Move_Right_Bowl_To_Middle_Plate",
        "04_Move_Buffered_Bowl_To_Right_Plate",
    )
    stage_predicates = (
        ("on", "akita_black_bowl_1", "plate_4"),
        ("on", "akita_black_bowl_2", "plate_1"),
        ("on", "akita_black_bowl_3", "plate_2"),
        ("on", "akita_black_bowl_1", "plate_3"),
    )
    release_objects = (
        "akita_black_bowl_1",
        "akita_black_bowl_2",
        "akita_black_bowl_3",
        "akita_black_bowl_1",
    )

    def _reset_task(self) -> None:
        super()._reset_task()
        self._tabletop_contact_violation: str | None = None

    def _sample(self) -> None:
        if self._tabletop_contact_violation is not None:
            return
        for bowl_name in self.bowl_names:
            if _bodies_in_contact(self.env, bowl_name, self.tabletop_body_name):
                self._tabletop_contact_violation = f"{bowl_name}_touched_tabletop"
                return
        super()._sample()

    def _terminal_predicates(self) -> tuple[tuple[str, ...], ...]:
        return self.stage_predicates[1:]

    def result(self) -> dict[str, Any]:
        result = super().result()
        result["tabletop_contact_violation"] = self._tabletop_contact_violation
        if self._tabletop_contact_violation is not None:
            result["success"] = False
            result["failure_reason"] = self._tabletop_contact_violation
        sequence_success = bool(result["success"])
        buffer_cleared = not _evaluate_predicate(
            self.env,
            ("on", "akita_black_bowl_1", "plate_4"),
        )
        result["terminal_state_checks"]["temporary buffer plate cleared"] = (
            buffer_cleared
        )
        result["success"] = bool(sequence_success and buffer_cleared)
        if sequence_success and not buffer_cleared:
            result["failure_reason"] = "temporary_buffer_not_cleared"
        return result


class PhysicalWeighingEvaluator:
    """Require all three visually identical mugs to be physically sampled."""

    schema_version = "libero.long_horizon_v2_private.v1"
    mug_names = ("light_mug_1", "medium_mug_1", "heavy_mug_1")
    lift_threshold_m = 0.04

    def __init__(self, env: Any, *, task_id: int) -> None:
        self.env = env
        self.task_id = int(task_id)
        self._active = False
        self.reset()

    def reset(self) -> None:
        self._control_steps = 0
        self._physics_substeps = 0
        self._active = True
        self._sampled = {name: False for name in self.mug_names}
        self._initial_heights = {
            name: (
                float(position[2])
                if (position := _body_position(self.env, name)) is not None
                else float("nan")
            )
            for name in self.mug_names
        }
        self._stage_events: list[dict[str, Any]] = []
        self._placed_event_recorded = False

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        if not self._active:
            return
        self._control_steps += 1
        self._sample()

    def observe_physics_substep(self) -> None:
        if not self._active:
            return
        self._physics_substeps += 1
        self._sample()

    def _record(self, name: str) -> None:
        event: dict[str, Any] = {
            "stage_index": len(self._stage_events),
            "stage_name": name,
            "control_step": self._control_steps,
            "physics_substep": self._physics_substeps,
        }
        sim_time = _sim_time(self.env)
        if sim_time is not None:
            event["sim_time_s"] = sim_time
        self._stage_events.append(event)

    def _heavy_placed(self) -> bool:
        return _evaluate_predicate(
            self.env,
            ("in", "heavy_mug_1", "target_bin_1_contain_region"),
        )

    def _sample(self) -> None:
        for name in self.mug_names:
            if self._sampled[name] or not _object_grasped(self.env, name):
                continue
            position = _body_position(self.env, name)
            initial_height = self._initial_heights[name]
            if (
                position is not None
                and np.isfinite(initial_height)
                and position[2] >= initial_height + self.lift_threshold_m
            ):
                self._sampled[name] = True
                self._record(f"Weigh_{name}")
        if (
            all(self._sampled.values())
            and self._heavy_placed()
            and _released_and_clear(self.env, "heavy_mug_1")
            and not self._placed_event_recorded
        ):
            self._placed_event_recorded = True
            self._record("Place_Heaviest_Mug_In_Green_Bin")

    def result(self) -> dict[str, Any]:
        placed = self._heavy_placed()
        released = _released_and_clear(self.env, "heavy_mug_1")
        all_weighed = all(self._sampled.values())
        success = bool(all_weighed and placed and released)
        completed = sum(self._sampled.values()) + int(self._placed_event_recorded)
        required_names = [
            *(f"Weigh_{name}" for name in self.mug_names),
            "Place_Heaviest_Mug_In_Green_Bin",
        ]
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "success": success,
            "required_stage_count": len(required_names),
            "completed_required_stage_count": completed,
            "stage_score_percent": 100.0 * completed / len(required_names),
            "required_stage_names": required_names,
            "completed_stage_names": [
                *(
                    f"Weigh_{name}"
                    for name in self.mug_names
                    if self._sampled[name]
                ),
                *(
                    ["Place_Heaviest_Mug_In_Green_Bin"]
                    if self._placed_event_recorded
                    else []
                ),
            ],
            "terminal_state_checks": {
                "all_three_mugs_physically_lifted": all_weighed,
                "heaviest_mug_in_green_bin": placed,
                "heaviest_mug_released_and_gripper_clear": released,
            },
            "weighing_interactions": dict(self._sampled),
            "control_steps_observed": self._control_steps,
            "physics_substeps_observed": self._physics_substeps,
            "failure_reason": (
                None if success else "weighing_or_placement_incomplete"
            ),
            "stage_events": list(self._stage_events),
        }


class CoffeeBrewingEvaluator(_OrderedEvaluator):
    """Require placement, a new post-placement press, and final release."""

    schema_version = "libero.long_horizon_v2_private.v1"
    stage_names = (
        "01_Place_Mug_Under_Dispenser",
        "02_Press_Button_While_Mug_Is_Placed",
        "03_Release_Mug_Under_Dispenser",
    )
    press_threshold_m = 0.015
    release_threshold_m = 0.003

    def _reset_task(self) -> None:
        initial = _joint_position(self.env, "coffee_machine_1_button_slide")
        self._initial_button_qpos = 0.0 if initial is None else float(initial)
        self._max_button_displacement = 0.0
        self._button_armed_after_placement = False
        self._brewed = False
        _set_geom_alpha(self.env, "coffee_liquid_visual", 0.0)

    def _mug_under_dispenser(self) -> bool:
        return _evaluate_predicate(
            self.env,
            ("in", "mug_1", "coffee_machine_1_mug_region"),
        )

    def _sample(self) -> None:
        mug_placed = self._mug_under_dispenser()
        if self._stage_index == 0 and mug_placed:
            self._mark_next(self.stage_names[0])

        qpos = _joint_position(self.env, "coffee_machine_1_button_slide")
        displacement = (
            0.0 if qpos is None else float(qpos - self._initial_button_qpos)
        )
        self._max_button_displacement = max(
            self._max_button_displacement,
            displacement,
        )
        if (
            self._stage_index == 1
            and mug_placed
            and displacement <= self.release_threshold_m
        ):
            self._button_armed_after_placement = True
        if (
            self._button_armed_after_placement
            and mug_placed
            and displacement >= self.press_threshold_m
        ):
            self._brewed = True
            _set_geom_alpha(self.env, "coffee_liquid_visual", 1.0)
        if self._stage_index == 1 and self._brewed:
            self._mark_next(self.stage_names[1])

        released = _released_and_clear(self.env, "mug_1")
        if self._stage_index == 2 and self._brewed and mug_placed and released:
            self._mark_next(self.stage_names[2])

    def result(self) -> dict[str, Any]:
        mug_placed = self._mug_under_dispenser()
        released = _released_and_clear(self.env, "mug_1")
        success = bool(
            self._stage_index == len(self.stage_names)
            and self._brewed
            and mug_placed
            and released
        )
        return {
            **self._base_result(success=success),
            "terminal_state_checks": {
                "mug_under_dispenser": mug_placed,
                "button_pressed_after_mug_placement": self._brewed,
                "button_armed_after_mug_placement": (
                    self._button_armed_after_placement
                ),
                "mug_released_and_gripper_clear": released,
            },
            "max_button_displacement_m": self._max_button_displacement,
            "required_button_displacement_m": self.press_threshold_m,
            "failure_reason": None if success else "coffee_sequence_incomplete",
        }


class CookingPreparationEvaluator(_StrictPredicateSequenceEvaluator):
    """Repair LiLo-VLA's Ultra-Long cooking language into five real stages."""

    stage_names = (
        "01_Place_Moka_Pot_On_Second_Stove",
        "02_Turn_On_Second_Stove",
        "03_Place_Frying_Pan_On_First_Stove",
        "04_Turn_On_First_Stove",
        "05_Open_Microwave",
    )
    stage_predicates = (
        ("on", "moka_pot_1", "flat_stove_2_cook_region"),
        ("turnon", "flat_stove_2"),
        ("on", "chefmate_8_frypan_1", "flat_stove_1_cook_region"),
        ("turnon", "flat_stove_1"),
        ("open", "microwave_1"),
    )
    release_objects = (
        "moka_pot_1",
        None,
        "chefmate_8_frypan_1",
        None,
        None,
    )


class CookingPreparationGoalEvaluator:
    """Track the five LiLo final goals without imposing a hidden order.

    The text and image variants are a matched goal-decomposition pair. A
    single goal image cannot communicate action order, so both variants use
    the same unordered terminal-goal semantics while retaining per-goal
    progress for evaluator-side scoring.
    """

    schema_version = "libero.long_horizon_v2_goal.v1"
    goal_names = (
        "Moka_Pot_On_Second_Stove",
        "Second_Stove_On",
        "Frying_Pan_On_First_Stove",
        "First_Stove_On",
        "Microwave_Open",
    )
    goal_predicates = CookingPreparationEvaluator.stage_predicates

    def __init__(self, env: Any, *, task_id: int) -> None:
        self.env = env
        self.task_id = int(task_id)
        self.reset()

    def reset(self) -> None:
        self._control_steps = 0
        completed = self._completed_goal_names()
        self._initial_completed_goal_count = len(completed)
        self._max_completed_goal_count = len(completed)
        self._last_completed_goal_names = completed
        self._progress_events: list[dict[str, Any]] = []

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        self._control_steps += 1
        completed = self._completed_goal_names()
        if completed != self._last_completed_goal_names:
            self._max_completed_goal_count = max(
                self._max_completed_goal_count,
                len(completed),
            )
            self._progress_events.append(
                {
                    "control_step": self._control_steps,
                    "completed_goal_count": len(completed),
                    "completed_goal_names": list(completed),
                }
            )
            self._last_completed_goal_names = completed

    def _goal_checks(self) -> dict[str, bool]:
        return {
            name: _evaluate_predicate(self.env, predicate)
            for name, predicate in zip(self.goal_names, self.goal_predicates)
        }

    def _completed_goal_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, complete in self._goal_checks().items() if complete
        )

    def result(self) -> dict[str, Any]:
        checks = self._goal_checks()
        completed = tuple(name for name, value in checks.items() if value)
        completed_count = len(completed)
        self._max_completed_goal_count = max(
            self._max_completed_goal_count,
            completed_count,
        )
        required_count = len(self.goal_names)
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "success": bool(all(checks.values())),
            "progress_type": "unordered_goal_predicates",
            "required_stage_count": required_count,
            "completed_required_stage_count": completed_count,
            "stage_score_percent": 100.0 * completed_count / required_count,
            "initial_completed_goal_count": self._initial_completed_goal_count,
            "max_completed_goal_count": self._max_completed_goal_count,
            "regressed_from_max": (
                completed_count < self._max_completed_goal_count
            ),
            "required_stage_names": list(self.goal_names),
            "completed_stage_names": list(completed),
            "terminal_state_checks": checks,
            "control_steps_observed": self._control_steps,
            "progress_events": list(self._progress_events),
            "failure_reason": (
                None if all(checks.values()) else "cooking_goals_incomplete"
            ),
        }


def long_horizon_v2_private_evaluator(
    env: Any,
    *,
    suite: str,
    task_id: int,
) -> (
    CyclicBowlRotationEvaluator
    | PhysicalWeighingEvaluator
    | CoffeeBrewingEvaluator
    | CookingPreparationEvaluator
    | CookingPreparationGoalEvaluator
    | None
):
    """Select the authoritative checker for candidates and goal variants."""

    if suite != LONG_HORIZON_V2_SUITE:
        return None
    evaluators = (
        CyclicBowlRotationEvaluator,
        PhysicalWeighingEvaluator,
        CoffeeBrewingEvaluator,
        CookingPreparationEvaluator,
        CookingPreparationGoalEvaluator,
        CookingPreparationGoalEvaluator,
    )
    normalized_task_id = int(task_id)
    if not 0 <= normalized_task_id < len(evaluators):
        raise ValueError(f"unsupported long_horizon_v2 task_id {task_id}")
    return evaluators[normalized_task_id](env, task_id=normalized_task_id)


__all__ = [
    "CoffeeBrewingEvaluator",
    "CookingPreparationGoalEvaluator",
    "CookingPreparationEvaluator",
    "CyclicBowlRotationEvaluator",
    "LONG_HORIZON_V2_SUITE",
    "LONG_HORIZON_V2_TASK_NAMES",
    "PhysicalWeighingEvaluator",
    "long_horizon_v2_private_evaluator",
]
