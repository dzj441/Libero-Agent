"""Private evaluator wiring for the first V2 manipulation adaptations.

The hammer and wipe evaluators are intentionally carried over from the audited
``335e07f`` prototype. The initial door port uses the stock robosuite success
semantics through its BDDL ``Open`` predicate; checker hardening is a later
milestone. The wine-pouring task extracts the first physical pour event from
RoboMemArena Task 10 and reuses its audited live-state counter.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..integrations.robomemarena.vendor.stage.shared_pour_counter import (
    PourCounterConfig,
    SharedPourCounter,
)


MUJOCO_ADAPTED_V2_SUITE = "mujoco_adapted_v2"
MUJOCO_ADAPTED_V2_TASK_NAMES = (
    "unlatch_and_open_door",
    "drive_nail_with_hammer",
    "pour_wine_into_mug",
    "wipe_spill_from_table",
)


def _model(env: Any) -> Any:
    return env.sim.model


def _data(env: Any) -> Any:
    return env.sim.data


def _name_candidates(logical_name: str) -> tuple[str, ...]:
    return (
        logical_name,
        f"{logical_name}_main",
        f"{logical_name}_object",
    )


def _body_id(env: Any, logical_name: str) -> int | None:
    for candidate in _name_candidates(logical_name):
        try:
            return int(_model(env).body_name2id(candidate))
        except Exception:
            continue
    return None


def _body_position(env: Any, logical_name: str) -> np.ndarray | None:
    body_id = _body_id(env, logical_name)
    if body_id is None:
        return None
    return np.asarray(_data(env).body_xpos[body_id], dtype=np.float64).copy()


def _joint_position(env: Any, logical_name: str) -> float | None:
    for candidate in _name_candidates(logical_name):
        try:
            joint_id = int(_model(env).joint_name2id(candidate))
            address = int(_model(env).jnt_qposadr[joint_id])
            return float(_data(env).qpos[address])
        except Exception:
            continue
    return None


def _body_descendants(model: Any, root_body_id: int) -> set[int]:
    parents = np.asarray(model.body_parentid, dtype=int).reshape(-1)
    descendants = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parents):
            if int(parent_id) in descendants and body_id not in descendants:
                descendants.add(int(body_id))
                changed = True
    return descendants


def _bodies_in_contact(env: Any, first_name: str, second_name: str) -> bool:
    first_id = _body_id(env, first_name)
    second_id = _body_id(env, second_name)
    if first_id is None or second_id is None:
        return False
    model = _model(env)
    data = _data(env)
    first = _body_descendants(model, first_id)
    second = _body_descendants(model, second_id)
    geom_bodies = np.asarray(model.geom_bodyid, dtype=int).reshape(-1)
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        try:
            body_a = int(geom_bodies[int(contact.geom1)])
            body_b = int(geom_bodies[int(contact.geom2)])
        except (IndexError, TypeError, ValueError):
            continue
        if not (
            (body_a in first and body_b in second)
            or (body_a in second and body_b in first)
        ):
            continue
        if float(getattr(contact, "dist", 0.0)) <= 1e-3:
            return True
    return False


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
        return np.asarray(_data(env).site_xpos[site_id], dtype=np.float64).copy()
    except Exception:
        return None


def _released_and_clear(
    env: Any, logical_name: str, *, clearance_m: float = 0.055
) -> bool:
    if _object_grasped(env, logical_name):
        return False
    object_position = _body_position(env, logical_name)
    eef_position = _eef_position(env)
    if object_position is None or eef_position is None:
        return False
    return bool(np.linalg.norm(object_position - eef_position) >= clearance_m)


def _sim_time(env: Any) -> float | None:
    try:
        return float(_data(env).time)
    except Exception:
        return None


def _set_geom_alpha(env: Any, suffix: str, alpha: float) -> bool:
    model = _model(env)
    changed = False
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id)
        if name is not None and str(name).endswith(suffix):
            model.geom_rgba[geom_id, 3] = float(alpha)
            changed = True
    return changed


class _OrderedEvaluator:
    schema_version = "libero.mujoco_adapted_private.v1"
    stage_names: tuple[str, ...] = ()

    def __init__(self, env: Any, *, task_id: int) -> None:
        self.env = env
        self.task_id = int(task_id)
        self._active = False
        self.reset()

    def reset(self) -> None:
        self._control_steps = 0
        self._physics_substeps = 0
        self._stage_index = 0
        self._stage_events: list[dict[str, Any]] = []
        self._active = True
        self._reset_task()

    def _reset_task(self) -> None:
        pass

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

    def _sample(self) -> None:
        raise NotImplementedError

    def _mark_next(self, stage_name: str) -> bool:
        if self._stage_index >= len(self.stage_names):
            return False
        if self.stage_names[self._stage_index] != stage_name:
            return False
        event: dict[str, Any] = {
            "stage_index": self._stage_index,
            "stage_name": stage_name,
            "control_step": self._control_steps,
            "physics_substep": self._physics_substeps,
        }
        sim_time = _sim_time(self.env)
        if sim_time is not None:
            event["sim_time_s"] = sim_time
        self._stage_events.append(event)
        self._stage_index += 1
        return True

    def _base_result(self, *, success: bool) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "success": bool(success),
            "required_stage_count": len(self.stage_names),
            "completed_required_stage_count": self._stage_index,
            "stage_score_percent": (
                100.0 * self._stage_index / len(self.stage_names)
                if self.stage_names
                else 0.0
            ),
            "ordered_stage_names": list(self.stage_names),
            "completed_stage_names": list(self.stage_names[: self._stage_index]),
            "control_steps_observed": self._control_steps,
            "physics_substeps_observed": self._physics_substeps,
            "stage_events": list(self._stage_events),
        }


class HammerNailEvaluator(_OrderedEvaluator):
    stage_names = (
        "01_Lift_Hammer",
        "02_Hammer_Contacts_Nail",
        "03_Drive_Nail",
    )
    driven_threshold_m = 0.075

    def _reset_task(self) -> None:
        hammer = _body_position(self.env, "hammer_1")
        self._initial_hammer_height = (
            float(hammer[2]) if hammer is not None else float("nan")
        )
        initial = _joint_position(self.env, "hammer_block_1_nail_slide")
        self._initial_nail_qpos = 0.0 if initial is None else float(initial)
        self._max_nail_displacement = 0.0
        self._hammer_nail_contact_seen = False

    def _sample(self) -> None:
        hammer = _body_position(self.env, "hammer_1")
        qpos = _joint_position(self.env, "hammer_block_1_nail_slide")
        displacement = 0.0 if qpos is None else float(qpos - self._initial_nail_qpos)
        self._max_nail_displacement = max(self._max_nail_displacement, displacement)
        lifted = bool(
            hammer is not None
            and np.isfinite(self._initial_hammer_height)
            and hammer[2] >= self._initial_hammer_height + 0.045
        )
        contact = _bodies_in_contact(self.env, "hammer_1", "hammer_block_1_nail_link")
        if contact:
            self._hammer_nail_contact_seen = True
        if self._stage_index == 0 and lifted:
            self._mark_next(self.stage_names[0])
        if self._stage_index == 1 and contact:
            self._mark_next(self.stage_names[1])
        if (
            self._stage_index == 2
            and self._hammer_nail_contact_seen
            and displacement >= self.driven_threshold_m
        ):
            self._mark_next(self.stage_names[2])

    def result(self) -> dict[str, Any]:
        qpos = _joint_position(self.env, "hammer_block_1_nail_slide")
        displacement = 0.0 if qpos is None else float(qpos - self._initial_nail_qpos)
        driven = displacement >= self.driven_threshold_m
        success = bool(
            self._stage_index == len(self.stage_names)
            and self._hammer_nail_contact_seen
            and driven
        )
        return {
            **self._base_result(success=success),
            "terminal_state_checks": {
                "hammer_nail_contact_observed": self._hammer_nail_contact_seen,
                "nail_driven_to_threshold": driven,
            },
            "nail_displacement_m": displacement,
            "max_nail_displacement_m": self._max_nail_displacement,
            "required_nail_displacement_m": self.driven_threshold_m,
            "failure_reason": None if success else "nail_not_physically_driven",
        }


class PourWineEvaluator(_OrderedEvaluator):
    """One complete physical pour extracted from RoboMemArena Task 10."""

    schema_version = "libero.robomemarena_atomic_pour.v1"
    stage_names = (
        "01_Lift_Wine_Bottle",
        "02_Pour_Wine_Into_Mug",
    )
    source_name = "wine_bottle_1"
    target_kind = "site"
    target_name = "white_yellow_mug_1_default_site"
    counter_config = PourCounterConfig()

    def _reset_task(self) -> None:
        initial_source_position = _body_position(self.env, self.source_name)
        if initial_source_position is None:
            raise RuntimeError(
                f"pour checker could not resolve source body {self.source_name!r}"
            )
        self._initial_source_position = initial_source_position
        self._lifted = False
        self._counter = SharedPourCounter(
            source_name=self.source_name,
            target_kind=self.target_kind,
            target_name=self.target_name,
            initial_source_pos=initial_source_position.copy(),
            config=self.counter_config,
        )

    def observe_physics_substep(self) -> None:
        # RoboMemArena's source counter is calibrated in 20 Hz control steps.
        # Keep substep telemetry without letting MuJoCo's internal rate satisfy
        # its dwell thresholds early.
        if self._active:
            self._physics_substeps += 1

    def _sample(self) -> None:
        source_position = _body_position(self.env, self.source_name)
        if source_position is None:
            return
        lifted = bool(
            source_position[2]
            > self._initial_source_position[2] + self.counter_config.lift_delta
        )
        self._lifted = self._lifted or lifted
        if self._stage_index == 0 and lifted:
            self._mark_next(self.stage_names[0])

        event_count = self._counter.update(self.env, self._control_steps)
        if self._stage_index == 1 and event_count >= 1:
            self._mark_next(self.stage_names[1])

    def result(self) -> dict[str, Any]:
        event_count = int(self._counter.event_count)
        success = bool(
            self._stage_index == len(self.stage_names)
            and self._lifted
            and event_count >= 1
        )
        return {
            **self._base_result(success=success),
            "terminal_state_checks": {
                "wine_bottle_lift_observed": self._lifted,
                "pour_event_observed": event_count >= 1,
            },
            "pour_event_count": event_count,
            "pour_events": list(self._counter.events),
            "target_radius_m": self.counter_config.target_radius,
            "tilt_enter_rad": self.counter_config.tilt_enter_rad,
            "tilt_dwell_control_steps": self.counter_config.enter_dwell,
            "failure_reason": None if success else "physical_pour_not_observed",
        }


class SpillWipeEvaluator(_OrderedEvaluator):
    """Require grasped, live table contact across the whole visible spill."""

    stage_names = (
        "01_Grasp_Sponge",
        "02_Wipe_All_Spill_Markers",
        "03_Release_Sponge_On_Blue_Rest_Area",
    )
    marker_suffixes = tuple(f"spill_marker_{index}" for index in range(7))
    coverage_radius_m = 0.052

    def _reset_task(self) -> None:
        self._covered = {suffix: False for suffix in self.marker_suffixes}
        self._grasp_seen = False
        for suffix in self.marker_suffixes:
            _set_geom_alpha(self.env, suffix, 0.92)

    def _marker_position(self, suffix: str) -> np.ndarray | None:
        model = _model(self.env)
        data = _data(self.env)
        for geom_id in range(int(model.ngeom)):
            name = model.geom_id2name(geom_id)
            if name is not None and str(name).endswith(suffix):
                return np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy()
        return None

    def _sample(self) -> None:
        grasped = _object_grasped(self.env, "sponge_1")
        if grasped:
            self._grasp_seen = True
        if self._stage_index == 0 and grasped:
            self._mark_next(self.stage_names[0])

        sponge = _body_position(self.env, "sponge_1")
        touching_table = _bodies_in_contact(self.env, "sponge_1", "table")
        if self._grasp_seen and grasped and touching_table and sponge is not None:
            for suffix in self.marker_suffixes:
                marker = self._marker_position(suffix)
                if marker is None:
                    continue
                if np.linalg.norm(sponge[:2] - marker[:2]) <= self.coverage_radius_m:
                    self._covered[suffix] = True
                    _set_geom_alpha(self.env, suffix, 0.0)

        all_covered = all(self._covered.values())
        if self._stage_index == 1 and all_covered:
            self._mark_next(self.stage_names[1])
        rested = _evaluate_predicate(
            self.env,
            ("on", "sponge_1", "spill_patch_1_sponge_rest_region"),
        )
        released = _released_and_clear(self.env, "sponge_1")
        if self._stage_index == 2 and rested and released:
            self._mark_next(self.stage_names[2])

    def result(self) -> dict[str, Any]:
        rested = _evaluate_predicate(
            self.env,
            ("on", "sponge_1", "spill_patch_1_sponge_rest_region"),
        )
        released = _released_and_clear(self.env, "sponge_1")
        all_covered = all(self._covered.values())
        success = bool(
            self._stage_index == len(self.stage_names)
            and all_covered
            and rested
            and released
        )
        return {
            **self._base_result(success=success),
            "terminal_state_checks": {
                "all_spill_markers_wiped": all_covered,
                "sponge_on_blue_rest_area": rested,
                "sponge_released_and_gripper_clear": released,
            },
            "covered_marker_count": sum(self._covered.values()),
            "required_marker_count": len(self._covered),
            "marker_coverage": dict(self._covered),
            "failure_reason": None if success else "wipe_or_release_incomplete",
        }


def mujoco_adapted_v2_private_evaluator(
    env: Any,
    *,
    suite: str,
    task_id: int,
) -> HammerNailEvaluator | PourWineEvaluator | SpillWipeEvaluator | None:
    """Return the private checker for an event-based adapted task."""

    if suite != MUJOCO_ADAPTED_V2_SUITE:
        return None
    normalized_task_id = int(task_id)
    if normalized_task_id == 1:
        return HammerNailEvaluator(env, task_id=normalized_task_id)
    if normalized_task_id == 2:
        return PourWineEvaluator(env, task_id=normalized_task_id)
    if normalized_task_id == 3:
        return SpillWipeEvaluator(env, task_id=normalized_task_id)
    return None


__all__ = [
    "HammerNailEvaluator",
    "MUJOCO_ADAPTED_V2_SUITE",
    "MUJOCO_ADAPTED_V2_TASK_NAMES",
    "PourWineEvaluator",
    "SpillWipeEvaluator",
    "mujoco_adapted_v2_private_evaluator",
]
