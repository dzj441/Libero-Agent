"""Private geometry checker for the VLABench chemistry-tube task.

The public task observation only contains anonymous entities.  This module is
host-side evaluation code: it is intentionally selected by the agent factory
and is never included in the projected observation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


VLABENCH_TUBE_PRECISION_SUITE = "vlabench_tube_precision"
VLABENCH_TUBE_PRECISION_TASK_ID = 0
VLABENCH_TUBE_REARRANGEMENT_TASK_ID = 1


def _body_descendants(model: Any, root_body_id: int) -> set[int]:
    """Return a MuJoCo body and every nested source-asset body below it."""

    parent_ids = np.asarray(getattr(model, "body_parentid"), dtype=int).reshape(-1)
    descendants = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parent_ids):
            if int(parent_id) in descendants and body_id not in descendants:
                descendants.add(int(body_id))
                changed = True
    return descendants


def _has_live_body_contact(env: Any, first_body_id: int, second_body_id: int) -> bool:
    """Check actual MuJoCo contact, including nested child bodies."""

    data = env.sim.data
    model = env.sim.model
    first = _body_descendants(model, first_body_id)
    second = _body_descendants(model, second_body_id)
    geom_body_ids = np.asarray(getattr(model, "geom_bodyid"), dtype=int).reshape(-1)
    for contact_index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[contact_index]
        try:
            body1 = int(geom_body_ids[int(contact.geom1)])
            body2 = int(geom_body_ids[int(contact.geom2)])
        except (IndexError, TypeError, ValueError):
            continue
        if not ((body1 in first and body2 in second) or (body1 in second and body2 in first)):
            continue
        # Contacts inside MuJoCo's small margin have a positive distance;
        # keep only near-zero contacts so a deliberate hover cannot pass.
        distance = float(getattr(contact, "dist", 0.0))
        if np.isfinite(distance) and distance <= 1e-3:
            return True
    return False


class ChemistryTubePrecisionEvaluator:
    """Check insertion into the named rack slot with geometric tolerances."""

    target_name = "blue_tube_1"
    rack_name = "tube_stand_1"
    slot_name = "tube_stand_1_rear_center_slot"

    lateral_tolerance_m = 0.018
    insertion_depth_range_m = (0.032, 0.085)
    # The released rack has compliant mesh contacts; a released tube can
    # settle with a few degrees of lean while remaining captured by the hole.
    # 0.97 corresponds to roughly 14 degrees and still rejects a visibly
    # tilted insertion.
    axis_alignment_tolerance = 0.97
    linear_speed_tolerance_mps = 0.08
    angular_speed_tolerance_rps = 0.60

    def __init__(self, env: Any) -> None:
        self.env = env
        self._control_steps = 0
        self._success_events = 0

    def reset(self) -> None:
        self._control_steps = 0
        self._success_events = 0

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        self._control_steps += 1
        if self._state()["success"]:
            self._success_events += 1

    def result(self) -> dict[str, Any]:
        state = self._state()
        return {
            "schema_version": "libero.vlabench_tube_precision_private.v1",
            "success": bool(state["success"]),
            "target_identity": self.target_name,
            "slot_identity": self.slot_name,
            "lateral_error_m": state["lateral_error_m"],
            "insertion_depth_m": state["insertion_depth_m"],
            "axis_alignment": state["axis_alignment"],
            "linear_speed_mps": state["linear_speed_mps"],
            "angular_speed_rps": state["angular_speed_rps"],
            "checks": dict(state["checks"]),
            "control_steps_observed": self._control_steps,
            "success_observations": self._success_events,
        }

    def _state(self) -> dict[str, Any]:
        try:
            target_body = self._body_id(self.target_name)
            axis_body = self._axis_body_id(self.target_name)
            rack_body = self._body_id(self.rack_name)
            slot = self._site_position(self.slot_name)
            target_position = np.asarray(self.env.sim.data.body_xpos[target_body], dtype=float)
            rack_position = np.asarray(self.env.sim.data.body_xpos[rack_body], dtype=float)
            xmat = np.asarray(self.env.sim.data.body_xmat[axis_body], dtype=float).reshape(3, 3)
            axis = xmat[:, 1]
            axis_alignment = float(abs(np.dot(axis, np.asarray((0.0, 0.0, 1.0)))))
            lateral_error = float(np.linalg.norm(target_position[:2] - slot[:2]))
            insertion_depth = float(target_position[2] - rack_position[2])
            cvel = np.asarray(self.env.sim.data.cvel[target_body], dtype=float)
            angular_speed = float(np.linalg.norm(cvel[:3]))
            linear_speed = float(np.linalg.norm(cvel[3:]))
            rack_contact = _has_live_body_contact(self.env, target_body, rack_body)
            checks = {
                "slot_lateral_error": lateral_error <= self.lateral_tolerance_m,
                "insertion_depth": (
                    self.insertion_depth_range_m[0]
                    <= insertion_depth
                    <= self.insertion_depth_range_m[1]
                ),
                "tube_axis_vertical": axis_alignment >= self.axis_alignment_tolerance,
                "tube_stable": (
                    linear_speed <= self.linear_speed_tolerance_mps
                    and angular_speed <= self.angular_speed_tolerance_rps
                ),
                "rack_live_contact": rack_contact,
            }
            return {
                "success": all(checks.values()),
                "lateral_error_m": lateral_error,
                "insertion_depth_m": insertion_depth,
                "axis_alignment": axis_alignment,
                "linear_speed_mps": linear_speed,
                "angular_speed_rps": angular_speed,
                "checks": checks,
            }
        except Exception as exc:
            # A malformed/partially compiled scene is never a successful
            # insertion.  Keep the diagnostic in the private result without
            # exposing model names or exceptions to the agent.
            return {
                "success": False,
                "lateral_error_m": float("inf"),
                "insertion_depth_m": float("nan"),
                "axis_alignment": 0.0,
                "linear_speed_mps": float("inf"),
                "angular_speed_rps": float("inf"),
                "checks": {
                    "slot_lateral_error": False,
                    "insertion_depth": False,
                    "tube_axis_vertical": False,
                    "tube_stable": False,
                    "rack_live_contact": False,
                    "checker_error": type(exc).__name__,
                },
            }

    def _body_id(self, logical_name: str) -> int:
        model = self.env.sim.model
        for name in (f"{logical_name}_main", logical_name):
            try:
                return int(model.body_name2id(name))
            except (KeyError, ValueError):
                continue
        raise KeyError(f"No MuJoCo body for {logical_name!r}")

    def _site_position(self, name: str) -> np.ndarray:
        data = self.env.sim.data
        try:
            return np.asarray(data.get_site_xpos(name), dtype=float)
        except (AttributeError, KeyError, ValueError):
            site_id = int(self.env.sim.model.site_name2id(name))
            return np.asarray(data.site_xpos[site_id], dtype=float)

    def _axis_body_id(self, logical_name: str) -> int:
        """Return the fixed-orientation child carrying tube geometry."""

        model = self.env.sim.model
        for name in (f"{logical_name}_tube_axis", f"{logical_name}_main"):
            try:
                return int(model.body_name2id(name))
            except (KeyError, ValueError):
                continue
        raise KeyError(f"No orientation body for {logical_name!r}")


class ChemistryTubeRearrangementEvaluator(ChemistryTubePrecisionEvaluator):
    """Strict private checker for two tubes mapped to two labeled slots."""

    tube_slots = {
        "blue_tube_1": "tube_stand_1_left_slot",
        "green_tube_1": "tube_stand_1_right_slot",
    }

    def result(self) -> dict[str, Any]:
        state = self._state()
        return {
            "schema_version": "libero.vlabench_tube_rearrangement_private.v1",
            "success": bool(state["success"]),
            "checks": dict(state["checks"]),
            "tubes": state["tubes"],
            "control_steps_observed": self._control_steps,
            "success_observations": self._success_events,
        }

    def _state(self) -> dict[str, Any]:
        tube_states: dict[str, dict[str, Any]] = {}
        checks: dict[str, bool] = {}
        try:
            rack_body = self._body_id("tube_stand_1")
            rack_position = np.asarray(
                self.env.sim.data.body_xpos[rack_body], dtype=float
            )
            for tube_name, slot_name in self.tube_slots.items():
                tube_body = self._body_id(tube_name)
                axis_body = self._axis_body_id(tube_name)
                slot = self._site_position(slot_name)
                tube_position = np.asarray(
                    self.env.sim.data.body_xpos[tube_body], dtype=float
                )
                xmat = np.asarray(
                    self.env.sim.data.body_xmat[axis_body], dtype=float
                ).reshape(3, 3)
                axis_alignment = float(
                    abs(np.dot(xmat[:, 1], np.asarray((0.0, 0.0, 1.0))))
                )
                lateral_error = float(
                    np.linalg.norm(tube_position[:2] - slot[:2])
                )
                insertion_depth = float(tube_position[2] - rack_position[2])
                cvel = np.asarray(self.env.sim.data.cvel[tube_body], dtype=float)
                angular_speed = float(np.linalg.norm(cvel[:3]))
                linear_speed = float(np.linalg.norm(cvel[3:]))
                rack_contact = _has_live_body_contact(
                    self.env, tube_body, rack_body
                )
                tube_checks = {
                    "slot_lateral_error": lateral_error <= self.lateral_tolerance_m,
                    "insertion_depth": (
                        self.insertion_depth_range_m[0]
                        <= insertion_depth
                        <= self.insertion_depth_range_m[1]
                    ),
                    "tube_axis_vertical": axis_alignment
                    >= self.axis_alignment_tolerance,
                    "tube_stable": (
                        linear_speed <= self.linear_speed_tolerance_mps
                        and angular_speed <= self.angular_speed_tolerance_rps
                    ),
                    "rack_live_contact": rack_contact,
                }
                checks.update(
                    {f"{tube_name}.{key}": value for key, value in tube_checks.items()}
                )
                tube_states[tube_name] = {
                    "slot_identity": slot_name,
                    "lateral_error_m": lateral_error,
                    "insertion_depth_m": insertion_depth,
                    "axis_alignment": axis_alignment,
                    "linear_speed_mps": linear_speed,
                    "angular_speed_rps": angular_speed,
                    "checks": tube_checks,
                }
            return {
                "success": bool(tube_states) and all(checks.values()),
                "checks": checks,
                "tubes": tube_states,
            }
        except Exception as exc:
            return {
                "success": False,
                "checks": {"checker_error": type(exc).__name__},
                "tubes": tube_states,
            }


def vlabench_tube_precision_private_evaluator(
    env: Any,
    *,
    suite: str,
    task_id: int,
) -> ChemistryTubePrecisionEvaluator | None:
    """Return the private checker for this suite/task pair."""

    if suite == VLABENCH_TUBE_PRECISION_SUITE and task_id == VLABENCH_TUBE_PRECISION_TASK_ID:
        return ChemistryTubePrecisionEvaluator(env)
    if suite == VLABENCH_TUBE_PRECISION_SUITE and task_id == VLABENCH_TUBE_REARRANGEMENT_TASK_ID:
        return ChemistryTubeRearrangementEvaluator(env)
    return None


__all__ = [
    "ChemistryTubePrecisionEvaluator",
    "ChemistryTubeRearrangementEvaluator",
    "VLABENCH_TUBE_PRECISION_SUITE",
    "VLABENCH_TUBE_REARRANGEMENT_TASK_ID",
    "vlabench_tube_precision_private_evaluator",
]
