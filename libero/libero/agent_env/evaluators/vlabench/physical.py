"""Private physical checker and factory for the VLABench seesaw task.

The upstream ``simple_seesaw_use`` task only reports whether the requested
object is grasped.  This adapter keeps the mechanism and replaces that weak
terminal condition with an ordered, evaluator-private contract:

1. place both counterweights on the far board end (in either order);
2. observe a real hinge/target lift caused by the resulting dynamics; and
3. place the target in the basket.

All predicates are computed from the live MuJoCo state.  No action token,
button, or language answer can advance a stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ...runtime.control import MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS, OSCControlConfig
from ...runtime.contracts.release import (
    RELEASE_RESOURCE_DIRECTORY,
    validate_release_task,
)
from ...runtime.environment import LiberoAgentEnv
from ...runtime.observation import TaskEntitySelection
from ...runtime.profiles import ObservationProfile
from ...runtime.references import load_task_reference_rgb


VLABENCH_PHYSICAL_SUITE = "vlabench_physical"
VLABENCH_PHYSICAL_TASK_NAME = "basic_seesaw_usage"
VLABENCH_PHYSICAL_BDDL_FILENAME = "basic_seesaw_usage.bddl"
VLABENCH_PHYSICAL_INIT_FILENAME = "basic_seesaw_usage.pruned_init"
VLABENCH_PHYSICAL_INSTRUCTION = (
    "Lift the hidden object and place it in the basket."
)
VLABENCH_ADAPTIVE_SEESAW_TASK_NAME = "adaptive_seesaw_usage"
VLABENCH_ADAPTIVE_SEESAW_BDDL_FILENAME = "adaptive_seesaw_usage.bddl"
VLABENCH_ADAPTIVE_SEESAW_INIT_FILENAME = "adaptive_seesaw_usage.pruned_init"
VLABENCH_ADAPTIVE_SEESAW_INSTRUCTION = (
    "Probe the seesaw with a subset of neutral weights until the board tilts, "
    "then place the payload in the basket."
)
VLABENCH_ACTIVE_COMPARISON_TASK_NAME = "active_weight_comparison"
VLABENCH_ACTIVE_COMPARISON_BDDL_FILENAME = "active_weight_comparison.bddl"
VLABENCH_ACTIVE_COMPARISON_INIT_FILENAME = "active_weight_comparison.pruned_init"
VLABENCH_ACTIVE_COMPARISON_INSTRUCTION = (
    "Compare two neutral weights by placing each on the marked far end of the "
    "seesaw in turn, then place the dynamically heavier one in the basket."
)


@dataclass(frozen=True)
class VLABenchPhysicalTaskSpec:
    """Stable task metadata used by the benchmark and private evaluator."""

    task_id: int
    name: str
    instruction: str
    bddl_file: str
    init_states_file: str
    required_stage_names: tuple[str, ...]


VLABENCH_PHYSICAL_TASK = VLABenchPhysicalTaskSpec(
    task_id=0,
    name=VLABENCH_PHYSICAL_TASK_NAME,
    instruction=VLABENCH_PHYSICAL_INSTRUCTION,
    bddl_file=VLABENCH_PHYSICAL_BDDL_FILENAME,
    init_states_file=VLABENCH_PHYSICAL_INIT_FILENAME,
    required_stage_names=(
        "01_Place_First_Counterweight",
        "02_Place_Second_Counterweight",
        "03_Physical_Lift_Of_Target",
        "04_Place_Target_In_Basket",
    ),
)

VLABENCH_ADAPTIVE_SEESAW_TASK = VLABenchPhysicalTaskSpec(
    task_id=1,
    name=VLABENCH_ADAPTIVE_SEESAW_TASK_NAME,
    instruction=VLABENCH_ADAPTIVE_SEESAW_INSTRUCTION,
    bddl_file=VLABENCH_ADAPTIVE_SEESAW_BDDL_FILENAME,
    init_states_file=VLABENCH_ADAPTIVE_SEESAW_INIT_FILENAME,
    required_stage_names=(
        "01_Place_First_Mass_Probe",
        "02_Select_Two_Counterweights",
        "03_Observed_Mass_Threshold_Tilt_And_Payload_Lift",
        "04_Place_Probed_Payload_In_Basket",
    ),
)

VLABENCH_ACTIVE_COMPARISON_TASK = VLABenchPhysicalTaskSpec(
    task_id=2,
    name=VLABENCH_ACTIVE_COMPARISON_TASK_NAME,
    instruction=VLABENCH_ACTIVE_COMPARISON_INSTRUCTION,
    bddl_file=VLABENCH_ACTIVE_COMPARISON_BDDL_FILENAME,
    init_states_file=VLABENCH_ACTIVE_COMPARISON_INIT_FILENAME,
    required_stage_names=(
        "01_Probe_First_Candidate",
        "02_Probe_Second_Candidate_After_Reset",
        "03_Mass_Response_Agrees_With_MuJoCo",
        "04_Place_Dynamically_Heavier_Candidate",
    ),
)

VLABENCH_PHYSICAL_TASKS = (
    VLABENCH_PHYSICAL_TASK,
    VLABENCH_ADAPTIVE_SEESAW_TASK,
    VLABENCH_ACTIVE_COMPARISON_TASK,
)


def _name_variants(name: str) -> tuple[str, ...]:
    if name.endswith("_main"):
        return (name, name[:-5])
    return (name, f"{name}_main")


def _body_position(env: Any, name: str) -> np.ndarray | None:
    for candidate in _name_variants(name):
        try:
            body_id = env.sim.model.body_name2id(candidate)
        except Exception:
            continue
        return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()
    return None


def _site_position(env: Any, name: str) -> np.ndarray | None:
    for candidate in _name_variants(name):
        try:
            site_id = env.sim.model.site_name2id(candidate)
        except Exception:
            continue
        return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()
    return None


def _joint_position(env: Any, name: str) -> float | None:
    for candidate in _name_variants(name):
        try:
            joint_id = env.sim.model.joint_name2id(candidate)
        except Exception:
            continue
        address = int(env.sim.model.jnt_qposadr[joint_id])
        return float(env.sim.data.qpos[address])
    return None


def _joint_velocity(env: Any, name: str) -> float | None:
    """Read a scalar hinge velocity across MuJoCo/robosuite wrappers."""

    for candidate in _name_variants(name):
        try:
            joint_id = env.sim.model.joint_name2id(candidate)
            address = int(env.sim.model.jnt_dofadr[joint_id])
            return float(env.sim.data.qvel[address])
        except Exception:
            continue
    return None


def _body_id(env: Any, name: str) -> int | None:
    """Resolve a body without exposing model names to the public agent."""

    for candidate in _name_variants(name):
        try:
            return int(env.sim.model.body_name2id(candidate))
        except Exception:
            continue
    return None


def _body_mass(env: Any, name: str) -> float | None:
    body_id = _body_id(env, name)
    if body_id is None:
        return None
    try:
        value = float(env.sim.model.body_mass[body_id])
    except Exception:
        return None
    return value if np.isfinite(value) and value > 0.0 else None


def _all_body_ids(env: Any, name: str) -> set[int]:
    """Return a body's descendants for contact matching across XML wrappers."""

    root = _body_id(env, name)
    if root is None:
        return set()
    ids = {root}
    try:
        parent = np.asarray(env.sim.model.body_parentid, dtype=np.int64)
        changed = True
        while changed:
            changed = False
            for body_id, parent_id in enumerate(parent):
                if int(parent_id) in ids and body_id not in ids:
                    ids.add(int(body_id))
                    changed = True
    except Exception:
        pass
    return ids


def _contact_with_seesaw(env: Any, object_name: str) -> bool:
    """Return whether MuJoCo reported a live object/articulated-board contact.

    Proximity is intentionally not enough for the expansion tasks.  The
    contact pair is read from ``mjData.contact`` and matched against the body
    carrying ``seesaw_hinge`` (plus any descendants).  Resolving the board
    from the joint is important for the external asset: its fixed ``rest``
    block sits underneath the moving board and must not count as a placed
    counterweight.
    """

    object_ids = _all_body_ids(env, object_name)
    model = env.sim.model
    board_ids: set[int] = set()
    for joint_name in ("seesaw_1_seesaw_hinge", "seesaw_hinge"):
        try:
            joint_id = int(model.joint_name2id(joint_name))
            board_ids.add(int(model.jnt_bodyid[joint_id]))
            break
        except Exception:
            continue
    if board_ids:
        try:
            parent = np.asarray(model.body_parentid, dtype=np.int64)
            changed = True
            while changed:
                changed = False
                for body_id, parent_id in enumerate(parent):
                    if int(parent_id) in board_ids and body_id not in board_ids:
                        board_ids.add(int(body_id))
                        changed = True
        except Exception:
            pass
    if not object_ids or not board_ids:
        return False
    try:
        contacts = env.sim.data.contact
        count = int(env.sim.data.ncon)
    except Exception:
        return False
    try:
        geom_bodyid = np.asarray(env.sim.model.geom_bodyid, dtype=np.int64)
    except Exception:
        return False
    for index in range(max(0, count)):
        try:
            contact = contacts[index]
            first = int(geom_bodyid[int(contact.geom1)])
            second = int(geom_bodyid[int(contact.geom2)])
        except Exception:
            continue
        if (first in object_ids and second in board_ids) or (
            second in object_ids and first in board_ids
        ):
            return True
    return False


def _object_grasped(env: Any, object_name: str) -> bool:
    """Use robosuite's live finger-contact predicate for one movable object."""

    base_env = getattr(env, "env", env)
    logical_name = object_name.removesuffix("_main")
    try:
        obj = base_env.objects_dict[logical_name]
        return bool(
            base_env._check_grasp(
                base_env.robots[0].gripper,
                obj.contact_geoms,
            )
        )
    except Exception:
        # The benchmark runtime always exposes this robosuite contract. A
        # missing predicate must not turn a still-held weight into a release.
        return True


def _available_weight_names(env: Any, candidates: Sequence[str]) -> tuple[str, ...]:
    """Keep only instances present in the live model, preserving BDDL order."""

    return tuple(name for name in candidates if _body_position(env, name) is not None)


def _far_end_position(env: Any) -> np.ndarray | None:
    """Resolve the task anchor across primitive and external seesaw variants."""

    for name in (
        "seesaw_1_seesaw_right_region",
        "seesaw_1_horizontal_radius_site",
        "seesaw_1_top_site",
    ):
        position = _site_position(env, name)
        if position is not None:
            return position
    # A source variant may omit all placement sites.  Its articulated body is
    # still named by the adapter, so the body's world position is a safe
    # coarse fallback for horizontal proximity; collision remains the final
    # physical source of truth during simulation.
    for name in ("seesaw_1_board", "seesaw_1_seesaw"):
        position = _body_position(env, name)
        if position is not None:
            return position
    return None


def _target_in_basket(env: Any) -> bool:
    """Use the parsed BDDL ``In`` predicate as the terminal geometry check."""

    try:
        return bool(env._check_success())
    except Exception:
        try:
            return bool(env.check_success())
        except Exception:
            return False


def _target_lifted(state: Mapping[str, Any], env: Any) -> bool:
    initial = state.get("initial_target_position")
    current = _body_position(env, "target_1")
    if initial is None or current is None:
        return False
    # The guard and mesh settle differently across MuJoCo minor versions;
    # four centimetres is still a substantial payload lift, while the hinge
    # threshold below prevents a direct tabletop / basket shortcut.
    return float(current[2] - initial[2]) >= 0.04


def _counterweight_on_far_end(env: Any, name: str) -> bool:
    current = _body_position(env, name)
    far_site = _far_end_position(env)
    if current is None or far_site is None:
        return False

    # The site is a board-local placement anchor; its world position is the
    # strongest available geometry signal and remains valid for the external
    # asset when it exposes the same documented right-end site.  A generous
    # tolerance covers gripper release and the small collision settling drift.
    horizontal_distance = float(np.linalg.norm(current[:2] - far_site[:2]))
    vertical_offset = float(current[2] - far_site[2])
    return horizontal_distance <= 0.12 and -0.15 <= vertical_offset <= 0.20


def _counterweights_on_far_end(env: Any) -> tuple[str, ...]:
    """Return the currently supported weights without assigning identities."""

    return tuple(
        name
        for name in ("weight_1", "weight_2")
        if _counterweight_on_far_end(env, name)
    )


def _supported_counterweights_on_far_end(env: Any) -> tuple[str, ...]:
    """Return weights whose collision geom is physically on the far board."""

    return tuple(
        name
        for name in ("weight_1", "weight_2")
        if _counterweight_on_far_end(env, name)
        and _contact_with_seesaw(env, name)
    )


def _released_counterweights_on_far_end(env: Any) -> tuple[str, ...]:
    """Return physically supported far-end weights no longer in the gripper."""

    return tuple(
        name
        for name in _supported_counterweights_on_far_end(env)
        if not _object_grasped(env, name)
    )


def _hinge_lifted(state: Mapping[str, Any], env: Any) -> bool:
    initial = state.get("initial_hinge_qpos")
    current = _joint_position(env, "seesaw_1_seesaw_hinge")
    if initial is None or current is None:
        return False
    return float(current - initial) >= 0.12


@dataclass(frozen=True)
class _Stage:
    name: str
    check: Callable[[Any, Mapping[str, Any]], bool]


class VLABenchSeesawEvaluator:
    """Host-private ordered checker for ``basic_seesaw_usage``.

    The evaluator records a lift event before accepting the terminal basket
    placement.  Thus moving the target directly to the basket at reset or
    before using the seesaw cannot produce success.
    """

    schema_version = "libero.vlabench_physical_private_evaluation.v1"

    def __init__(self, env: Any) -> None:
        self.env = env
        self._stages: tuple[_Stage, ...] = ()
        self._stage_done: dict[str, bool] = {}
        self._stage_index = 0
        self._state: dict[str, Any] = {}
        self._events: list[dict[str, Any]] = []
        self._control_steps = 0
        self._active = False
        self._lift_event_seen = False
        self._max_hinge_delta = 0.0
        self._max_target_lift = 0.0
        self._max_pre_two_hinge_delta = 0.0
        self._max_pre_two_target_lift = 0.0
        self._max_post_two_hinge_delta = 0.0
        self._max_post_two_target_lift = 0.0
        self._premature_lift_detected = False

    def reset(self) -> None:
        target = _body_position(self.env, "target_1")
        hinge = _joint_position(self.env, "seesaw_1_seesaw_hinge")
        if target is None or hinge is None:
            raise RuntimeError(
                "VLABench seesaw checker could not resolve target body or hinge"
            )
        self._state = {
            "initial_target_position": target,
            "initial_hinge_qpos": hinge,
        }
        self._stages = (
            _Stage(
                "01_Place_First_Counterweight",
                lambda env, state: bool(
                    _released_counterweights_on_far_end(env)
                ),
            ),
            _Stage(
                "02_Place_Second_Counterweight",
                lambda env, state: len(
                    _released_counterweights_on_far_end(env)
                )
                == 2,
            ),
            _Stage(
                "03_Physical_Lift_Of_Target",
                lambda env, state: (
                    not self._premature_lift_detected
                    and len(_released_counterweights_on_far_end(env)) == 2
                    and self._max_post_two_hinge_delta >= 0.12
                    and self._max_post_two_target_lift >= 0.04
                ),
            ),
            _Stage(
                "04_Place_Target_In_Basket",
                lambda env, state: _target_in_basket(env),
            ),
        )
        self._stage_done = {stage.name: False for stage in self._stages}
        self._stage_index = 0
        self._events = []
        self._control_steps = 0
        self._active = True
        self._lift_event_seen = False
        self._max_hinge_delta = 0.0
        self._max_target_lift = 0.0
        self._max_pre_two_hinge_delta = 0.0
        self._max_pre_two_target_lift = 0.0
        self._max_post_two_hinge_delta = 0.0
        self._max_post_two_target_lift = 0.0
        self._premature_lift_detected = False

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        if not self._active:
            return
        self._control_steps += 1
        hinge = _joint_position(self.env, "seesaw_1_seesaw_hinge")
        target = _body_position(self.env, "target_1")
        hinge_delta = 0.0
        target_lift = 0.0
        if hinge is not None and self._state.get("initial_hinge_qpos") is not None:
            hinge_delta = float(hinge - self._state["initial_hinge_qpos"])
            self._max_hinge_delta = max(
                self._max_hinge_delta,
                hinge_delta,
            )
        if target is not None and self._state.get("initial_target_position") is not None:
            target_lift = float(
                target[2] - self._state["initial_target_position"][2]
            )
            self._max_target_lift = max(
                self._max_target_lift,
                target_lift,
            )

        both_stage_done = self._stage_done.get(
            "02_Place_Second_Counterweight", False
        )
        if both_stage_done:
            if not self._lift_event_seen:
                self._max_post_two_hinge_delta = max(
                    self._max_post_two_hinge_delta, hinge_delta
                )
                self._max_post_two_target_lift = max(
                    self._max_post_two_target_lift, target_lift
                )
        else:
            self._max_pre_two_hinge_delta = max(
                self._max_pre_two_hinge_delta, hinge_delta
            )
            self._max_pre_two_target_lift = max(
                self._max_pre_two_target_lift, target_lift
            )
            if hinge_delta >= 0.12 and target_lift >= 0.04:
                self._premature_lift_detected = True

        if self._stage_index >= len(self._stages):
            return
        stage = self._stages[self._stage_index]
        if stage.check(self.env, self._state):
            self._stage_done[stage.name] = True
            supported_weights = _supported_counterweights_on_far_end(self.env)
            self._events.append(
                {
                    "stage_index": self._stage_index,
                    "stage_name": stage.name,
                    "control_step": self._control_steps,
                    "sim_time_s": float(self.env.sim.data.time),
                    "counterweights_on_far_end": list(supported_weights),
                    "counterweights_grasped": {
                        name: _object_grasped(self.env, name)
                        for name in ("weight_1", "weight_2")
                    },
                }
            )
            if stage.name == "02_Place_Second_Counterweight":
                # Start a new causal response window. Maxima accumulated while
                # fewer than two released weights were supported cannot
                # satisfy lift.
                self._max_post_two_hinge_delta = max(0.0, hinge_delta)
                self._max_post_two_target_lift = max(0.0, target_lift)
            if stage.name == "03_Physical_Lift_Of_Target":
                self._lift_event_seen = True
            self._stage_index += 1

    def result(self) -> dict[str, Any]:
        if not self._state:
            raise RuntimeError("VLABench seesaw checker result requested before reset")
        stage_success = all(self._stage_done.values())
        terminal_checks = {
            "target_in_basket": _target_in_basket(self.env),
            "physical_lift_observed": self._lift_event_seen,
            "both_counterweights_placed": len(
                _released_counterweights_on_far_end(self.env)
            )
            == 2,
            "no_lift_before_two_counterweights": not self._premature_lift_detected,
        }
        terminal_success = all(terminal_checks.values())
        success = bool(stage_success and terminal_success)
        if self._premature_lift_detected:
            failure_reason = "physical_lift_before_two_counterweights"
        elif not stage_success:
            failure_reason = "incomplete_ordered_stage"
        elif not terminal_success:
            failure_reason = "terminal_physical_state_invalid"
        else:
            failure_reason = None
        completed = [name for name, done in self._stage_done.items() if done]
        return {
            "schema_version": self.schema_version,
            "task_id": 0,
            "success": success,
            "required_stage_count": len(self._stages),
            "completed_required_stage_count": len(completed),
            "stage_score_percent": 100.0 * len(completed) / len(self._stages),
            "ordered_stage_names": [stage.name for stage in self._stages],
            "completed_stage_names": completed,
            "terminal_state_success": terminal_success,
            "terminal_state_checks": terminal_checks,
            "control_steps_observed": self._control_steps,
            "max_hinge_delta_rad": self._max_hinge_delta,
            "max_target_lift_m": self._max_target_lift,
            "max_pre_two_hinge_delta_rad": self._max_pre_two_hinge_delta,
            "max_pre_two_target_lift_m": self._max_pre_two_target_lift,
            "max_post_two_hinge_delta_rad": self._max_post_two_hinge_delta,
            "max_post_two_target_lift_m": self._max_post_two_target_lift,
            "premature_lift_detected": self._premature_lift_detected,
            "failure_reason": failure_reason,
            "stage_events": list(self._events),
        }


class VLABenchAdaptiveSeesawEvaluator:
    """Checker for the mass-selection seesaw variant.

    Three visually identical weights expose different inertial masses.  The
    successful policy is deliberately not encoded as a name: exactly two
    weights must make live contact with the far board, their measured mass
    must clear a reset-specific torque threshold, and MuJoCo must then report
    both a real hinge tilt and a physical lift of the payload. A direct basket
    placement, all-three
    placement, or a qpos teleport without contact/response remains incomplete.
    """

    schema_version = "libero.vlabench_physical_private_evaluation.v2"
    weight_names = ("weight_a", "weight_b", "weight_c")

    def __init__(self, env: Any) -> None:
        self.env = env
        self._state: dict[str, Any] = {}
        self._stages: tuple[_Stage, ...] = ()
        self._stage_done: dict[str, bool] = {}
        self._stage_index = 0
        self._events: list[dict[str, Any]] = []
        self._control_steps = 0
        self._contact_seen: set[str] = set()
        self._contact_order: list[str] = []
        self._max_hinge_delta = 0.0
        self._max_target_lift = 0.0
        self._selected_weight_names: tuple[str, ...] = ()
        self._required_mass_kg = float("nan")

    def reset(self) -> None:
        target = _body_position(self.env, "payload_1")
        if target is None:
            # Keep the checker useful with a generic fake that reuses the
            # basic-task target name.
            target = _body_position(self.env, "target_1")
        hinge = _joint_position(self.env, "seesaw_1_seesaw_hinge")
        names = _available_weight_names(self.env, self.weight_names)
        masses = {name: _body_mass(self.env, name) for name in names}
        finite_masses = [mass for mass in masses.values() if mass is not None]
        if target is None or hinge is None or len(names) != 3 or len(finite_masses) != 3:
            raise RuntimeError(
                "adaptive seesaw checker could not resolve payload, hinge, or masses"
            )
        # A fraction of total available mass keeps the threshold tied to the
        # actual fixture configuration.  Requiring exactly two means the
        # agent must make a closed-loop choice instead of dumping everything.
        total_mass = float(sum(finite_masses))
        self._required_mass_kg = max(0.80, 0.58 * total_mass)
        self._state = {
            "initial_target_position": target,
            "initial_hinge_qpos": hinge,
            "weight_masses_kg": masses,
            "available_weight_names": names,
        }
        self._stages = (
            _Stage(
                "01_Place_First_Mass_Probe",
                lambda env, state: len(self._weights_on_board()) == 1
                and bool(self._contact_seen.intersection(self._weights_on_board())),
            ),
            _Stage(
                "02_Select_Two_Counterweights",
                lambda env, state: len(self._contact_order) == 2,
            ),
            _Stage(
                "03_Observed_Mass_Threshold_Tilt_And_Payload_Lift",
                lambda env, state: (
                    self._selected_mass() >= self._required_mass_kg
                    and self._max_hinge_delta >= 0.12
                    and self._max_target_lift >= 0.04
                    and len(self._contact_order) == 2
                    and all(name in self._contact_seen for name in self._selected_weight_names)
                ),
            ),
            _Stage(
                "04_Place_Probed_Payload_In_Basket",
                lambda env, state: _target_in_basket(env),
            ),
        )
        self._stage_done = {stage.name: False for stage in self._stages}
        self._stage_index = 0
        self._events = []
        self._control_steps = 0
        self._contact_seen = set()
        self._contact_order = []
        self._max_hinge_delta = 0.0
        self._max_target_lift = 0.0
        self._selected_weight_names = ()

    def _weights_on_board(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self._state.get("available_weight_names", self.weight_names)
            if _counterweight_on_far_end(self.env, name)
        )

    def _selected_mass(self) -> float:
        masses = self._state.get("weight_masses_kg", {})
        return float(sum(float(masses.get(name, 0.0) or 0.0) for name in self._selected_weight_names))

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        if not self._state:
            return
        self._control_steps += 1
        for name in self._state["available_weight_names"]:
            if _contact_with_seesaw(self.env, name):
                if name not in self._contact_seen:
                    self._contact_order.append(name)
                self._contact_seen.add(name)
        hinge = _joint_position(self.env, "seesaw_1_seesaw_hinge")
        target = _body_position(self.env, "payload_1")
        if target is None:
            target = _body_position(self.env, "target_1")
        if hinge is not None:
            self._max_hinge_delta = max(
                self._max_hinge_delta,
                float(hinge - self._state["initial_hinge_qpos"]),
            )
        if target is not None:
            self._max_target_lift = max(
                self._max_target_lift,
                float(target[2] - self._state["initial_target_position"][2]),
            )
        if self._stage_index >= len(self._stages):
            return
        stage = self._stages[self._stage_index]
        if stage.check(self.env, self._state):
            if stage.name == "02_Select_Two_Counterweights":
                self._selected_weight_names = tuple(self._contact_order[:2])
            supported = self._weights_on_board()
            self._stage_done[stage.name] = True
            self._events.append(
                {
                    "stage_index": self._stage_index,
                    "stage_name": stage.name,
                    "control_step": self._control_steps,
                    "sim_time_s": float(self.env.sim.data.time),
                    "weights_on_far_end": list(supported),
                    "weight_masses_kg": {
                        name: self._state["weight_masses_kg"].get(name)
                        for name in supported
                    },
                    "contact_observed": sorted(self._contact_seen),
                    "contact_order_private": list(self._contact_order),
                }
            )
            self._stage_index += 1

    def result(self) -> dict[str, Any]:
        if not self._state:
            raise RuntimeError("adaptive seesaw checker result requested before reset")
        completed = [name for name, done in self._stage_done.items() if done]
        threshold_met = self._selected_mass() >= self._required_mass_kg
        terminal_checks = {
            "payload_in_basket": _target_in_basket(self.env),
            "physical_tilt_observed": self._stage_done.get(
                "03_Observed_Mass_Threshold_Tilt_And_Payload_Lift", False
            ),
            "payload_lift_observed": self._max_target_lift >= 0.04,
            "exactly_two_counterweights_selected": len(self._selected_weight_names) == 2,
            "selected_mass_threshold_met": bool(threshold_met),
            "selected_weights_contacted_seesaw": bool(self._selected_weight_names)
            and all(name in self._contact_seen for name in self._selected_weight_names),
        }
        stage_success = len(completed) == len(self._stages)
        terminal_success = all(terminal_checks.values())
        return {
            "schema_version": self.schema_version,
            "task_id": 1,
            "success": bool(stage_success and terminal_success),
            "required_stage_count": len(self._stages),
            "completed_required_stage_count": len(completed),
            "stage_score_percent": 100.0 * len(completed) / len(self._stages),
            "ordered_stage_names": [stage.name for stage in self._stages],
            "completed_stage_names": completed,
            "terminal_state_success": terminal_success,
            "terminal_state_checks": terminal_checks,
            "control_steps_observed": self._control_steps,
            "available_weight_masses_kg": dict(self._state["weight_masses_kg"]),
            "required_mass_threshold_kg": self._required_mass_kg,
            "selected_weight_names_private": list(self._selected_weight_names),
            "selected_mass_kg": self._selected_mass(),
            "max_hinge_delta_rad": self._max_hinge_delta,
            "max_target_lift_m": self._max_target_lift,
            "contact_observed_private": sorted(self._contact_seen),
            "contact_order_private": list(self._contact_order),
            "failure_reason": None if stage_success and terminal_success else "incomplete_physical_contract",
            "stage_events": list(self._events),
        }


def _object_in_basket(env: Any, object_name: str) -> bool:
    """Require the parsed ``In`` region *and* live basket contact.

    A broad body/site distance check is not sufficient here: it accepts a
    candidate hovering well above the basket, which is exactly the terminal
    shortcut this task is meant to rule out.  The LIBERO BDDL environment
    already owns the authoritative ``SiteObjectState`` for the contain region
    and ``ObjectState`` contact implementation, so use both of those live
    predicates.  Returning ``False`` when the parsed state API is absent keeps
    this private checker fail-closed for incomplete test doubles as well.
    """

    bddl_env = getattr(env, "env", env)
    states = getattr(bddl_env, "object_states_dict", None)
    if not isinstance(states, Mapping):
        return False
    region_state = states.get("basket_1_contain_region")
    object_state = states.get(object_name)
    basket_state = states.get("basket_1")
    if region_state is None or object_state is None or basket_state is None:
        return False
    try:
        # SiteObjectState.check_contain is the same region-aware predicate
        # used by the BDDL ``In`` goal.  ObjectState.check_contact reads the
        # current MuJoCo contact graph, so a suspended object cannot pass.
        contained = bool(region_state.check_contain(object_state))
        touching_basket = bool(basket_state.check_contact(object_state))
    except Exception:
        return False
    return contained and touching_basket


class VLABenchActiveWeightComparisonEvaluator:
    """Compare two same-looking weights through sequential live dynamics.

    Each candidate is placed on the same far board end, allowed to perturb the
    hinge, and then removed before the second trial.  The checker compares the
    measured joint responses against MuJoCo body masses, records contact pairs,
    and only accepts the dynamically heavier candidate in the basket.  No
    candidate name is treated as the answer.
    """

    schema_version = "libero.vlabench_physical_private_evaluation.v2"
    candidate_names = ("candidate_1", "candidate_2")

    def __init__(self, env: Any) -> None:
        self.env = env
        self._state: dict[str, Any] = {}
        self._stages: tuple[_Stage, ...] = ()
        self._stage_done: dict[str, bool] = {}
        self._stage_index = 0
        self._events: list[dict[str, Any]] = []
        self._control_steps = 0
        self._contact_seen: set[str] = set()
        self._trial_baseline: dict[str, float] = {}
        self._trial_response: dict[str, float] = {}
        self._first_name: str | None = None
        self._second_name: str | None = None
        self._winner_name: str | None = None
        self._neutral_stable_steps = 0
        self._required_neutral_stable_steps = 3
        self._reset_gate_open = False
        self._second_contact_after_reset = False

    def reset(self) -> None:
        hinge = _joint_position(self.env, "seesaw_1_seesaw_hinge")
        names = _available_weight_names(self.env, self.candidate_names)
        masses = {name: _body_mass(self.env, name) for name in names}
        if hinge is None or len(names) != 2 or any(mass is None for mass in masses.values()):
            raise RuntimeError("active comparison checker could not resolve hinge or masses")
        self._state = {
            "initial_hinge_qpos": hinge,
            "candidate_masses_kg": masses,
            "candidate_names": names,
        }
        self._stages = (
            _Stage(
                "01_Probe_First_Candidate",
                lambda env, state: self._first_name is not None
                and _counterweight_on_far_end(env, self._first_name)
                and self._first_name in self._contact_seen
                and self._trial_response.get(self._first_name, 0.0) >= 0.005,
            ),
            _Stage(
                "02_Probe_Second_Candidate_After_Reset",
                lambda env, state: self._second_name is not None
                and not _counterweight_on_far_end(env, self._first_name or "")
                and _counterweight_on_far_end(env, self._second_name)
                and self._second_contact_after_reset
                and self._reset_gate_open
                and self._neutral_stable_steps >= self._required_neutral_stable_steps
                and self._trial_response.get(self._second_name, 0.0) >= 0.005,
            ),
            _Stage(
                "03_Mass_Response_Agrees_With_MuJoCo",
                lambda env, state: self._responses_agree_with_mass(),
            ),
            _Stage(
                "04_Place_Dynamically_Heavier_Candidate",
                lambda env, state: self._winner_name is not None
                and _object_in_basket(env, self._winner_name),
            ),
        )
        self._stage_done = {stage.name: False for stage in self._stages}
        self._stage_index = 0
        self._events = []
        self._control_steps = 0
        self._contact_seen = set()
        self._trial_baseline = {}
        self._trial_response = {}
        self._first_name = None
        self._second_name = None
        self._winner_name = None
        self._neutral_stable_steps = 0
        self._reset_gate_open = False
        self._second_contact_after_reset = False

    def _on_board_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self._state["candidate_names"]
            if _counterweight_on_far_end(self.env, name)
        )

    def _responses_agree_with_mass(self) -> bool:
        masses = self._state.get("candidate_masses_kg", {})
        if len(self._trial_response) != 2:
            return False
        ordered_by_mass = sorted(
            self._state["candidate_names"], key=lambda name: float(masses[name])
        )
        lighter, heavier = ordered_by_mass
        lighter_response = float(self._trial_response.get(lighter, -1.0))
        heavier_response = float(self._trial_response.get(heavier, -1.0))
        self._winner_name = heavier
        return (
            lighter_response >= 0.005
            and heavier_response >= 0.005
            and heavier_response > lighter_response + 0.003
        )

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        if not self._state:
            return
        self._control_steps += 1
        on_board = self._on_board_names()
        for name in self._state["candidate_names"]:
            if _contact_with_seesaw(self.env, name):
                self._contact_seen.add(name)

        # A one-sided board can remain at its travel stop after the first
        # probe is removed.  Require several consecutive control observations
        # with the first candidate off the board, the hinge back near its
        # reset qpos, and low velocity before admitting the second probe.
        # This prevents residual first-trial motion from becoming false mass
        # evidence for the second candidate.
        if self._first_name is not None and self._second_name is None:
            hinge_for_reset = _joint_position(self.env, "seesaw_1_seesaw_hinge")
            velocity_for_reset = _joint_velocity(self.env, "seesaw_1_seesaw_hinge")
            initial_hinge = float(self._state["initial_hinge_qpos"])
            any_candidate_contact = any(
                _contact_with_seesaw(self.env, name)
                for name in self._state["candidate_names"]
            )
            near_reset = hinge_for_reset is not None and abs(
                float(hinge_for_reset) - initial_hinge
            ) <= 0.05
            settled = velocity_for_reset is None or abs(float(velocity_for_reset)) <= 0.20
            # Both candidates must be off the board during the neutral
            # window.  Once the window completes, latch the gate so the
            # second probe's own hinge response can move away from qpos=0
            # without erasing the fact that reset happened first.
            if not on_board and not any_candidate_contact and near_reset and settled:
                self._neutral_stable_steps += 1
                if self._neutral_stable_steps >= self._required_neutral_stable_steps:
                    self._reset_gate_open = True
            elif not self._reset_gate_open:
                self._neutral_stable_steps = 0

        hinge = _joint_position(self.env, "seesaw_1_seesaw_hinge")
        if hinge is not None:
            # Do not record an unqualified second object that was placed
            # while the first trial was still settling.  Its response becomes
            # eligible only after the reset gate below assigns _second_name.
            tracked_names = []
            if self._first_name is not None and self._first_name in on_board:
                tracked_names.append(self._first_name)
            if self._second_name is not None and self._second_name in on_board:
                tracked_names.append(self._second_name)
            for name in tracked_names:
                # Both probes are compared against the reset hinge pose.  A
                # first trial can leave a one-sided board near its stop; a
                # per-trial baseline would erase exactly the mass-dependent
                # signal the second probe is meant to compare.
                self._trial_baseline.setdefault(
                    name, float(self._state["initial_hinge_qpos"])
                )
                response = abs(float(hinge) - self._trial_baseline[name])
                self._trial_response[name] = max(
                    self._trial_response.get(name, 0.0), response
                )

        if self._first_name is None:
            if len(on_board) == 1:
                self._first_name = on_board[0]
        elif self._second_name is None:
            if (
                self._reset_gate_open
                and self._neutral_stable_steps >= self._required_neutral_stable_steps
                and len(on_board) == 1
                and on_board[0] != self._first_name
                and _contact_with_seesaw(self.env, on_board[0])
            ):
                self._second_name = on_board[0]
                self._second_contact_after_reset = True

        if self._stage_index >= len(self._stages):
            return
        stage = self._stages[self._stage_index]
        if stage.check(self.env, self._state):
            self._stage_done[stage.name] = True
            self._events.append(
                {
                    "stage_index": self._stage_index,
                    "stage_name": stage.name,
                    "control_step": self._control_steps,
                    "sim_time_s": float(self.env.sim.data.time),
                    "first_candidate_private": self._first_name,
                    "second_candidate_private": self._second_name,
                    "candidate_masses_kg": dict(self._state["candidate_masses_kg"]),
                    "trial_response_rad": dict(self._trial_response),
                    "contact_observed": sorted(self._contact_seen),
                    "neutral_hinge_stable_steps": self._neutral_stable_steps,
                }
            )
            self._stage_index += 1

    def result(self) -> dict[str, Any]:
        if not self._state:
            raise RuntimeError("active comparison checker result requested before reset")
        completed = [name for name, done in self._stage_done.items() if done]
        winner_in_basket = bool(self._winner_name and _object_in_basket(self.env, self._winner_name))
        terminal_checks = {
            "mass_response_order_observed": self._stage_done.get(
                "03_Mass_Response_Agrees_With_MuJoCo", False
            ),
            "both_candidates_contacted_seesaw": all(
                name in self._contact_seen for name in self._state["candidate_names"]
            ),
            "dynamically_heavier_candidate_in_basket": winner_in_basket,
        }
        stage_success = len(completed) == len(self._stages)
        terminal_success = all(terminal_checks.values())
        return {
            "schema_version": self.schema_version,
            "task_id": 2,
            "success": bool(stage_success and terminal_success),
            "required_stage_count": len(self._stages),
            "completed_required_stage_count": len(completed),
            "stage_score_percent": 100.0 * len(completed) / len(self._stages),
            "ordered_stage_names": [stage.name for stage in self._stages],
            "completed_stage_names": completed,
            "terminal_state_success": terminal_success,
            "terminal_state_checks": terminal_checks,
            "control_steps_observed": self._control_steps,
            "candidate_masses_kg": dict(self._state["candidate_masses_kg"]),
            "trial_response_rad": dict(self._trial_response),
            "winner_private": self._winner_name,
            "contact_observed_private": sorted(self._contact_seen),
            "neutral_hinge_stable_steps": self._neutral_stable_steps,
            "required_neutral_hinge_stable_steps": self._required_neutral_stable_steps,
            "reset_gate_open": self._reset_gate_open,
            "second_contact_after_reset": self._second_contact_after_reset,
            "failure_reason": None if stage_success and terminal_success else "incomplete_physical_comparison",
            "stage_events": list(self._events),
        }


def vlabench_physical_private_evaluator(
    env: Any, *, suite: str, task_id: int
) -> (
    VLABenchSeesawEvaluator
    | VLABenchAdaptiveSeesawEvaluator
    | VLABenchActiveWeightComparisonEvaluator
    | None
):
    """Return the host-private checker for a physical suite task."""

    if suite != VLABENCH_PHYSICAL_SUITE:
        return None
    checkers = (
        VLABenchSeesawEvaluator,
        VLABenchAdaptiveSeesawEvaluator,
        VLABenchActiveWeightComparisonEvaluator,
    )
    if not 0 <= int(task_id) < len(checkers):
        raise ValueError(f"unknown vlabench_physical task_id={task_id}")
    return checkers[int(task_id)](env)


def make_vlabench_physical_agent_env(
    *,
    task_id: int = 0,
    init_state_id: int = 0,
    profile: ObservationProfile | int | str = ObservationProfile.LEVEL3,
    seed: int = 0,
    camera_height: int = 256,
    camera_width: int = 256,
    task_entities: TaskEntitySelection | None = None,
    control_config: OSCControlConfig | None = None,
    initial_settle_control_steps: int = 0,
    max_agent_steps: int | None = None,
    native_sequence_submission_limit: int | None = (
        MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS
    ),
    private_control_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
    render_gpu_device_id: int = -1,
    bddl_root: str | os.PathLike[str] | None = None,
    init_states_root: str | os.PathLike[str] | None = None,
    **env_kwargs: Any,
) -> LiberoAgentEnv:
    """Create the seesaw task through the ordinary agent-safe environment."""

    if not 0 <= int(task_id) < len(VLABENCH_PHYSICAL_TASKS):
        raise ValueError(
            f"vlabench_physical task_id must be in [0, {len(VLABENCH_PHYSICAL_TASKS)})"
        )
    from pathlib import Path

    import torch
    import libero.libero as libero_package
    from libero.libero.envs import OffScreenRenderEnv

    package_root = Path(libero_package.__file__).resolve().parent
    validate_release_task(VLABENCH_PHYSICAL_SUITE, task_id)
    bddl_root_path = (
        Path(bddl_root) if bddl_root is not None else package_root / "bddl_files"
    )
    init_root_path = (
        Path(init_states_root)
        if init_states_root is not None
        else package_root / "init_files"
    )
    task_spec = VLABENCH_PHYSICAL_TASKS[int(task_id)]
    bddl_path = (
        bddl_root_path / RELEASE_RESOURCE_DIRECTORY / task_spec.bddl_file
    )
    init_path = (
        init_root_path / RELEASE_RESOURCE_DIRECTORY / task_spec.init_states_file
    )
    try:
        init_states = torch.load(os.fspath(init_path), weights_only=False)
    except TypeError:
        init_states = torch.load(os.fspath(init_path))
    if not 0 <= int(init_state_id) < len(init_states):
        raise ValueError(
            f"init_state_id must be in [0, {len(init_states)}), got {init_state_id}"
        )
    reserved = {
        "bddl_file_name",
        "camera_names",
        "camera_heights",
        "camera_widths",
        "camera_depths",
        "camera_segmentations",
        "use_object_obs",
        "ignore_done",
        "initialization_noise",
        "render_gpu_device_id",
        "horizon",
    }
    conflicts = reserved.intersection(env_kwargs)
    if conflicts:
        raise ValueError(f"factory-managed env kwargs cannot be overridden: {sorted(conflicts)}")
    np.random.seed(seed)
    env = OffScreenRenderEnv(
        bddl_file_name=os.fspath(bddl_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_depths=True,
        camera_segmentations="instance",
        use_object_obs=False,
        ignore_done=True,
        initialization_noise=None,
        render_gpu_device_id=render_gpu_device_id,
        horizon=10000,
        **env_kwargs,
    )
    env.seed(seed)
    evaluator = vlabench_physical_private_evaluator(
        env, suite=VLABENCH_PHYSICAL_SUITE, task_id=int(task_id)
    )
    return LiberoAgentEnv(
        env,
        profile=profile,
        camera_height=camera_height,
        camera_width=camera_width,
        task_instruction=task_spec.instruction,
        initial_state=np.asarray(init_states[int(init_state_id)]),
        task_entities=task_entities,
        control_config=control_config,
        initial_settle_control_steps=initial_settle_control_steps,
        settle_before_initial_state=True,
        max_agent_steps=max_agent_steps,
        native_sequence_submission_limit=native_sequence_submission_limit,
        private_control_step_callback=private_control_step_callback,
        private_episode_evaluator=evaluator,
        task_reference_rgb=load_task_reference_rgb(VLABENCH_PHYSICAL_SUITE, task_id),
    )


__all__ = [
    "VLABENCH_ACTIVE_COMPARISON_BDDL_FILENAME",
    "VLABENCH_ACTIVE_COMPARISON_INIT_FILENAME",
    "VLABENCH_ACTIVE_COMPARISON_INSTRUCTION",
    "VLABENCH_ACTIVE_COMPARISON_TASK",
    "VLABENCH_ACTIVE_COMPARISON_TASK_NAME",
    "VLABENCH_ADAPTIVE_SEESAW_BDDL_FILENAME",
    "VLABENCH_ADAPTIVE_SEESAW_INIT_FILENAME",
    "VLABENCH_ADAPTIVE_SEESAW_INSTRUCTION",
    "VLABENCH_ADAPTIVE_SEESAW_TASK",
    "VLABENCH_ADAPTIVE_SEESAW_TASK_NAME",
    "VLABENCH_PHYSICAL_BDDL_FILENAME",
    "VLABENCH_PHYSICAL_INIT_FILENAME",
    "VLABENCH_PHYSICAL_INSTRUCTION",
    "VLABENCH_PHYSICAL_SUITE",
    "VLABENCH_PHYSICAL_TASK",
    "VLABENCH_PHYSICAL_TASKS",
    "VLABENCH_PHYSICAL_TASK_NAME",
    "VLABenchPhysicalTaskSpec",
    "VLABenchSeesawEvaluator",
    "VLABenchAdaptiveSeesawEvaluator",
    "VLABenchActiveWeightComparisonEvaluator",
    "make_vlabench_physical_agent_env",
    "vlabench_physical_private_evaluator",
]
