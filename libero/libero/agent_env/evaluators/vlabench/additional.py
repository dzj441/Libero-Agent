"""Private evaluators for the retained VLABench additional task modes.

The task language and projected observations never contain these predicates.
Each evaluator reads only live MuJoCo state and records progress observed on a
real trajectory.  BDDL remains useful for reset / ordinary diagnostics, while
the private contracts below provide the stronger semantics that the upstream
``find_unseen_object`` and ``cluster_series`` tasks require. The dining-table
prototype remains implemented below for historical tests, but is no longer a
registered benchmark task because its adapted utensil assets are visually
ambiguous.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


VLABENCH_ADDITIONAL_SUITE = "vlabench_additional"
VLABENCH_ADDITIONAL_TASK_NAMES = (
    "find_unseen_object",
    "cluster_series",
)


# The checker speaks in task-level drawer labels, while the installed source
# cabinet variants use different joint tokens.  Keep the logical names first
# so external VLABench assets and lightweight test doubles remain compatible,
# then accept the names from LIBERO's bundled short_cabinet.xml fallback.
_DRAWER_JOINT_ALIASES: dict[str, tuple[str, ...]] = {
    "short_cabinet_1_top_drawer": (
        "short_cabinet_1_top_region",
        "short_cabinet_1_top_level",
        "short_cabinet_1_drawer_top",
        "short_cabinet_1_drawer_high",
    ),
    "short_cabinet_1_middle_drawer": (
        "short_cabinet_1_middle_level",
        "short_cabinet_1_middle_region",
        "short_cabinet_1_drawer_middle",
        "short_cabinet_1_drawer_mid",
    ),
    "short_cabinet_1_bottom_drawer": (
        "short_cabinet_1_bottom_region",
        "short_cabinet_1_bottom_level",
        "short_cabinet_1_drawer_bottom",
        "short_cabinet_1_drawer_low",
    ),
}


def _joint_name_variants(logical_name: str) -> tuple[str, ...]:
    """Return logical and source-asset aliases for one drawer joint."""

    base_name = str(logical_name).removesuffix("_main")
    aliases = _DRAWER_JOINT_ALIASES.get(base_name, ())
    candidates: list[str] = []
    for candidate in (base_name, *aliases):
        candidates.extend((candidate, f"{candidate}_main"))
    return tuple(dict.fromkeys(candidates))


def _body_id(env: Any, logical_name: str) -> int | None:
    """Resolve a BDDL logical body through robosuite's naming prefix."""

    model = getattr(getattr(env, "sim", None), "model", None)
    if model is None:
        return None
    for candidate in (f"{logical_name}_main", logical_name):
        try:
            return int(model.body_name2id(candidate))
        except Exception:
            continue
    return None


def _body_position(env: Any, logical_name: str) -> np.ndarray | None:
    body_id = _body_id(env, logical_name)
    if body_id is None:
        return None
    try:
        return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()
    except Exception:
        return None


def _object_world_extent(env: Any, logical_name: str) -> np.ndarray | None:
    """Return a conservative XYZ extent for all geoms under one object.

    MuJoCo exposes mesh bounding half-sizes through ``geom_size``.  Rotating
    those half-sizes by ``geom_xmat`` is enough to reject the former dining
    shortcut where a plate or utensil had the right centre position while
    standing on edge.
    """

    body_id = _body_id(env, logical_name)
    model = getattr(getattr(env, "sim", None), "model", None)
    data = getattr(getattr(env, "sim", None), "data", None)
    if body_id is None or model is None or data is None:
        return None
    try:
        descendants = {body_id}
        changed = True
        while changed:
            changed = False
            for candidate, parent in enumerate(np.asarray(model.body_parentid)):
                if int(parent) in descendants and candidate not in descendants:
                    descendants.add(candidate)
                    changed = True
        lower: list[np.ndarray] = []
        upper: list[np.ndarray] = []
        for geom_id, geom_body in enumerate(np.asarray(model.geom_bodyid)):
            if int(geom_body) not in descendants:
                continue
            # Measure the physical support envelope, not presentation-only
            # group-1 visuals.  The VLABench tableware adapter adds a
            # collision-free wrist silhouette so thin utensils remain
            # legible; including that silhouette here would make a flat fork
            # appear vertically oversized and falsely fail the orientation
            # predicate.
            if int(model.geom_group[geom_id]) != 0:
                continue
            geom_type = int(model.geom_type[geom_id])
            size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            if geom_type == 2:  # sphere
                local_half = np.repeat(size[0], 3)
            elif geom_type in (3, 5):  # capsule / cylinder, local Z axis
                local_half = np.asarray((size[0], size[0], size[1]))
            elif geom_type in (4, 6, 7):  # ellipsoid / box / mesh
                local_half = size[:3]
            else:
                continue
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            world_half = np.abs(rotation) @ local_half
            centre = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            lower.append(centre - world_half)
            upper.append(centre + world_half)
        if not lower:
            return None
        return np.max(np.stack(upper), axis=0) - np.min(np.stack(lower), axis=0)
    except Exception:
        return None


def _dining_orientation_state(env: Any) -> dict[str, bool]:
    """Check flat plate support and front-to-back utensil alignment."""

    extents = {
        name: _object_world_extent(env, name)
        for name in ("plate_1", "knife_1", "fork_1")
    }
    plate = extents["plate_1"]
    knife = extents["knife_1"]
    fork = extents["fork_1"]

    def utensil_is_flat_and_aligned(extent: np.ndarray | None) -> bool:
        return bool(
            extent is not None
            and float(extent[2]) <= 0.040
            and float(extent[1]) >= 0.080
            and float(extent[1]) >= 1.5 * float(extent[0])
        )

    return {
        "plate_flat_and_level": bool(
            plate is not None
            and float(plate[2]) <= 0.018
            and float(plate[0]) >= 0.16
            and float(plate[1]) >= 0.16
        ),
        "knife_flat_and_front_to_back": utensil_is_flat_and_aligned(knife),
        "fork_flat_and_front_to_back": utensil_is_flat_and_aligned(fork),
    }


def _joint_position(env: Any, *logical_names: str) -> float | None:
    model = getattr(getattr(env, "sim", None), "model", None)
    data = getattr(getattr(env, "sim", None), "data", None)
    if model is None or data is None:
        return None
    for logical_name in logical_names:
        for candidate in _joint_name_variants(logical_name):
            try:
                joint_id = int(model.joint_name2id(candidate))
                address = int(model.jnt_qposadr[joint_id])
                return float(data.qpos[address])
            except Exception:
                continue
    return None


def _sim_time(env: Any) -> float | None:
    try:
        return float(env.sim.data.time)
    except Exception:
        return None


def _bodies_in_contact(env: Any, first_name: str, second_name: str) -> bool:
    """Return whether two named MuJoCo bodies currently have a contact."""

    model = getattr(getattr(env, "sim", None), "model", None)
    data = getattr(getattr(env, "sim", None), "data", None)
    first_id = _body_id(env, first_name)
    second_id = _body_id(env, second_name)
    if model is None or data is None or first_id is None or second_id is None:
        return False
    try:
        def descendants(root_id: int) -> set[int]:
            # Asset adapters may put collision geoms below a fixed child
            # frame (for example, the canonical tableware-axis correction).
            # A logical object therefore owns its full body subtree, not only
            # the wrapper body returned by ``body_name2id``.
            owned = {int(root_id)}
            changed = True
            while changed:
                changed = False
                for body_id, parent_id in enumerate(model.body_parentid):
                    if int(parent_id) in owned and int(body_id) not in owned:
                        owned.add(int(body_id))
                        changed = True
            return owned

        first_bodies = descendants(first_id)
        second_bodies = descendants(second_id)
        first_geoms = {
            index
            for index, body_id in enumerate(model.geom_bodyid)
            if int(body_id) in first_bodies
        }
        second_geoms = {
            index
            for index, body_id in enumerate(model.geom_bodyid)
            if int(body_id) in second_bodies
        }
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if (int(contact.geom1) in first_geoms and int(contact.geom2) in second_geoms) or (
                int(contact.geom2) in first_geoms and int(contact.geom1) in second_geoms
            ):
                return True
    except Exception:
        return False
    return False


def _in_region(env: Any, object_name: str, region_name: str) -> bool:
    """Evaluate an ``In`` predicate, with a live-site fallback.

    The VLABench container adapters add a real ``contain_region`` site, so the
    normal BDDL predicate is authoritative in production.  The fallback keeps
    checker behavior deterministic on robosuite minor versions that do not
    expose a site through ``object_states_dict`` after a model merge.
    """

    # Receptacle regions are not just an XY answer zone.  A successful
    # placement must be on the real container, so require a live object /
    # container contact whenever the region denotes a tray ``contain_region``.
    require_container_contact = region_name.endswith("_contain_region")
    container_name = region_name[: -len("_contain_region")]
    for domain in (env, getattr(env, "env", None)):
        evaluator = getattr(domain, "_eval_predicate", None)
        if not callable(evaluator):
            continue
        try:
            contained = bool(evaluator(["in", object_name, region_name]))
            if not contained:
                return False
            if require_container_contact:
                return _bodies_in_contact(env, object_name, container_name)
            return True
        except Exception:
            pass

    object_position = _body_position(env, object_name)
    if object_position is None:
        return False
    model = getattr(getattr(env, "sim", None), "model", None)
    data = getattr(getattr(env, "sim", None), "data", None)
    if model is None or data is None:
        return False
    for candidate in (region_name, f"{region_name}_main"):
        try:
            site_id = int(model.site_name2id(candidate))
            site_position = np.asarray(data.site_xpos[site_id], dtype=np.float64)
            site_matrix = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
            size = np.asarray(model.site_size[site_id], dtype=np.float64)
            local = site_matrix.T @ (object_position - site_position)
            inside = bool(
                np.all(np.abs(local[:2]) < size[:2])
                and -0.08 < float(local[2]) < float(size[2]) + 0.08
            )
            if inside and require_container_contact:
                return _bodies_in_contact(env, object_name, container_name)
            return inside
        except Exception:
            continue
    return False


def _record_stage(
    events: list[dict[str, Any]],
    *,
    index: int,
    name: str,
    control_step: int,
    env: Any,
) -> None:
    event: dict[str, Any] = {
        "stage_index": int(index),
        "stage_name": name,
        "control_step": int(control_step),
    }
    sim_time = _sim_time(env)
    if sim_time is not None:
        event["sim_time_s"] = sim_time
    events.append(event)


class FindUnseenObjectEvaluator:
    """Require active drawer search before accepting the tray placement."""

    schema_version = "libero.vlabench_additional.find_unseen_object.v1"
    task_id = 0
    drawer_open_threshold = 0.14
    drawer_names = (
        "short_cabinet_1_top_drawer",
        "short_cabinet_1_middle_drawer",
        "short_cabinet_1_bottom_drawer",
    )
    drawer_region_names = (
        "short_cabinet_1_top_region",
        "short_cabinet_1_middle_region",
        "short_cabinet_1_bottom_region",
    )

    def __init__(self, env: Any) -> None:
        self.env = env
        self.reset()

    def reset(self) -> None:
        self._control_steps = 0
        self._stage_index = 0
        self._events: list[dict[str, Any]] = []
        self._max_drawer_qpos = float("-inf")
        self._active = True
        self._post_open_target_position: np.ndarray | None = None
        self._drawer_open_control_step: int | None = None
        # The upstream VLABench cabinet asset uses drawer-local placement
        # sites whose frame is not guaranteed to coincide with the merged
        # LIBERO world frame.  Anchor the target at the live pose when the
        # first occluding drawer is opened instead of trusting a site label
        # that could be shifted by a MuJoCo version.  Trusted init states can
        # therefore hide the target in any of the three drawers without
        # changing the public prompt.
        self._stage_names = (
            "01_Open_Occluding_Drawer",
            "02_Extract_Hidden_Target",
            "03_Place_Target_On_Tray",
        )
        self._stage_done = {name: False for name in self._stage_names}

    def _drawer_is_open(self) -> bool:
        return any(
            (value := _joint_position(self.env, name)) is not None
            and value >= self.drawer_open_threshold
            for name in self.drawer_names
        )

    def _target_extracted_after_search(self) -> bool:
        """Require a displacement observed after the drawer-open event.

        A terminal placement that happened before the drawer was opened must
        not be retroactively counted as retrieval.  Anchoring the target pose
        at the first open event makes the stage transition causal even when a
        caller teleports or otherwise moves the body between observations.
        """

        current = _body_position(self.env, "apple_1")
        anchor = self._post_open_target_position
        if current is None or anchor is None or self._drawer_open_control_step is None:
            return False
        return float(np.linalg.norm(current[:2] - anchor[:2])) >= 0.08

    def _target_on_tray(self) -> bool:
        return _in_region(self.env, "apple_1", "tray_1_contain_region")

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        if not self._active:
            return
        self._control_steps += 1
        for drawer_name in self.drawer_names:
            drawer_qpos = _joint_position(self.env, drawer_name)
            if drawer_qpos is not None:
                self._max_drawer_qpos = max(self._max_drawer_qpos, drawer_qpos)

        # Advance at most one stage per observation.  In particular, opening
        # a drawer and finding the apple must be two temporally ordered
        # control steps; a body already moved to the receptacle cannot satisfy
        # all stages in one ``while`` pass.
        if self._stage_index == 0 and self._drawer_is_open():
            name = self._stage_names[0]
            self._stage_done[name] = True
            self._drawer_open_control_step = self._control_steps
            self._post_open_target_position = _body_position(self.env, "apple_1")
            _record_stage(
                self._events,
                index=self._stage_index,
                name=name,
                control_step=self._control_steps,
                env=self.env,
            )
            self._stage_index += 1
            return

        if self._stage_index == 1 and self._target_extracted_after_search():
            name = self._stage_names[1]
            self._stage_done[name] = True
            _record_stage(
                self._events,
                index=self._stage_index,
                name=name,
                control_step=self._control_steps,
                env=self.env,
            )
            self._stage_index += 1
            return

        if self._stage_index == 2 and self._target_on_tray():
            name = self._stage_names[2]
            self._stage_done[name] = True
            _record_stage(
                self._events,
                index=self._stage_index,
                name=name,
                control_step=self._control_steps,
                env=self.env,
            )
            self._stage_index += 1

    def result(self) -> dict[str, Any]:
        drawer_open_now = self._drawer_is_open()
        extracted_now = self._target_extracted_after_search()
        target_on_tray = self._target_on_tray()
        stage_success = self._stage_index == len(self._stage_names)
        terminal_checks = {
            "drawer_open_event_observed": self._stage_done["01_Open_Occluding_Drawer"],
            "target_extracted_from_drawer": self._stage_done[
                "02_Extract_Hidden_Target"
            ]
            and extracted_now,
            "target_in_tray": target_on_tray,
        }
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "success": bool(stage_success and all(terminal_checks.values())),
            "required_stage_count": len(self._stage_names),
            "completed_required_stage_count": self._stage_index,
            "stage_score_percent": 100.0 * self._stage_index / len(self._stage_names),
            "ordered_stage_names": list(self._stage_names),
            "completed_stage_names": list(self._stage_names[: self._stage_index]),
            "terminal_state_success": bool(all(terminal_checks.values())),
            "terminal_state_checks": terminal_checks,
            "drawer_open_now": bool(drawer_open_now),
            "max_drawer_qpos": self._max_drawer_qpos,
            "control_steps_observed": self._control_steps,
            "failure_reason": None
            if stage_success and all(terminal_checks.values())
            else "incomplete_active_search_or_target_placement",
            "stage_events": list(self._events),
        }


class DiningTableEvaluator:
    """Check a real left/right table-setting relation around the plate."""

    schema_version = "libero.vlabench_additional.set_dining_table.v2"
    lateral_gap_m = 0.025
    transverse_tolerance_m = 0.13
    vertical_tolerance_m = 0.14
    movement_threshold_m = 0.04
    # VLABench's expert places the utensils about 15 cm from the plate
    # centre.  The band admits the released mesh's collision settling while
    # rejecting a distant tabletop shortcut.
    relation_max_gap_m = 0.22
    table_support_tolerance_m = 0.055

    def __init__(self, env: Any, *, left_handed: bool = False) -> None:
        self.env = env
        self.left_handed = bool(left_handed)
        self.task_id = 3 if self.left_handed else 1
        self._ordered_placements = (
            (("fork_1", "left"), ("knife_1", "right"))
            if self.left_handed
            else (("knife_1", "left"), ("fork_1", "right"))
        )
        self.reset()

    def reset(self) -> None:
        self._control_steps = 0
        self._events: list[dict[str, Any]] = []
        self._stage_index = 0
        self._stage_names = tuple(
            f"{index:02d}_{name.removesuffix('_1').title()}_{side.title()}_Of_Plate"
            for index, (name, side) in enumerate(self._ordered_placements, start=1)
        )
        self._stage_done = {name: False for name in self._stage_names}
        self._initial_positions = {
            name: _body_position(self.env, name)
            for name in ("plate_1", "knife_1", "fork_1")
        }

    def _table_supported(self, logical_name: str, position: np.ndarray) -> bool:
        # A live contact with the table collision mesh is the primary guard;
        # the z band rejects a stale / suspended body even if contact data is
        # unavailable in a lightweight fake environment.
        table_z = 0.90
        try:
            site_id = self.env.sim.model.site_name2id("table_top")
            table_z = float(self.env.sim.data.site_xpos[site_id][2])
        except Exception:
            pass
        in_support_band = abs(float(position[2]) - table_z) <= self.table_support_tolerance_m
        return bool(in_support_band and _bodies_in_contact(self.env, logical_name, "table"))

    def _placement_state(self) -> dict[str, bool]:
        plate = _body_position(self.env, "plate_1")
        knife = _body_position(self.env, "knife_1")
        fork = _body_position(self.env, "fork_1")
        if plate is None or knife is None or fork is None:
            return {
                "knife_left": False,
                "knife_right": False,
                "fork_left": False,
                "fork_right": False,
                "knife_moved": False,
                "fork_moved": False,
                "knife_supported": False,
                "fork_supported": False,
                **_dining_orientation_state(self.env),
            }

        def near_plate(position: np.ndarray) -> bool:
            return bool(
                abs(float(position[1] - plate[1])) <= self.transverse_tolerance_m
                and abs(float(position[2] - plate[2])) <= self.vertical_tolerance_m
            )

        def moved(name: str, position: np.ndarray) -> bool:
            initial = self._initial_positions.get(name)
            return initial is None or float(np.linalg.norm(position[:2] - initial[:2])) >= self.movement_threshold_m

        def on_side(position: np.ndarray, side: str) -> bool:
            if not near_plate(position):
                return False
            delta = float(position[0] - plate[0])
            if side == "left":
                return -self.relation_max_gap_m <= delta <= -self.lateral_gap_m
            return self.lateral_gap_m <= delta <= self.relation_max_gap_m

        return {
            "knife_left": on_side(knife, "left"),
            "knife_right": on_side(knife, "right"),
            "fork_left": on_side(fork, "left"),
            "fork_right": on_side(fork, "right"),
            "knife_moved": moved("knife_1", knife),
            "fork_moved": moved("fork_1", fork),
            "knife_supported": self._table_supported("knife_1", knife),
            "fork_supported": self._table_supported("fork_1", fork),
            **_dining_orientation_state(self.env),
        }

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        self._control_steps += 1
        state = self._placement_state()
        # The reset layout intentionally already places the utensils on the
        # corresponding sides of the plate, but far away.  A stage is only a
        # real table-setting event after that utensil has been physically
        # relocated into the relation band.
        checks = tuple(
            state[f"{name.removesuffix('_1')}_{side}"]
            and state[f"{name.removesuffix('_1')}_moved"]
            and state[f"{name.removesuffix('_1')}_supported"]
            and state[f"{name.removesuffix('_1')}_flat_and_front_to_back"]
            and state["plate_flat_and_level"]
            for name, side in self._ordered_placements
        )
        while self._stage_index < len(checks) and checks[self._stage_index]:
            name = self._stage_names[self._stage_index]
            self._stage_done[name] = True
            _record_stage(
                self._events,
                index=self._stage_index,
                name=name,
                control_step=self._control_steps,
                env=self.env,
            )
            self._stage_index += 1

    def result(self) -> dict[str, Any]:
        state = self._placement_state()
        terminal_checks = {
            f"{name.removesuffix('_1')}_{side}_of_plate": state[
                f"{name.removesuffix('_1')}_{side}"
            ]
            for name, side in self._ordered_placements
        }
        terminal_checks.update({
            "knife_physically_relocated": state["knife_moved"],
            "fork_physically_relocated": state["fork_moved"],
            "knife_supported_by_table": state["knife_supported"],
            "fork_supported_by_table": state["fork_supported"],
            "plate_flat_and_level": state["plate_flat_and_level"],
            "knife_flat_and_front_to_back": state[
                "knife_flat_and_front_to_back"
            ],
            "fork_flat_and_front_to_back": state[
                "fork_flat_and_front_to_back"
            ],
        })
        success = bool(all(terminal_checks.values()))
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "success": success,
            "required_stage_count": len(self._stage_names),
            "completed_required_stage_count": self._stage_index,
            "stage_score_percent": 100.0 * self._stage_index / len(self._stage_names),
            "ordered_stage_names": list(self._stage_names),
            "completed_stage_names": list(self._stage_names[: self._stage_index]),
            "terminal_state_success": success,
            "terminal_state_checks": terminal_checks,
            "control_steps_observed": self._control_steps,
            "failure_reason": None if success else "dining_relation_or_support_invalid",
            "stage_events": list(self._events),
        }


class ClusterSeriesEvaluator:
    """Accept either tray assignment while enforcing semantic separation."""

    schema_version = "libero.vlabench_additional.cluster_series.v1"
    task_id = 1
    movement_threshold_m = 0.04

    def __init__(self, env: Any) -> None:
        self.env = env
        self.reset()

    def reset(self) -> None:
        self._control_steps = 0
        self._events: list[dict[str, Any]] = []
        self._stage_index = 0
        self._stage_names = ("01_Collect_One_Semantic_Class", "02_Separate_Classes")
        self._stage_done = {name: False for name in self._stage_names}
        self._initial_positions = {
            name: _body_position(self.env, name)
            for name in ("apple_1", "apple_2", "banana_1", "banana_2")
        }

    @staticmethod
    def _class_in_tray(
        env: Any, names: tuple[str, str], tray_name: str
    ) -> bool:
        return all(
            _in_region(env, name, f"{tray_name}_contain_region") for name in names
        )

    def _assignment_state(self) -> dict[str, Any]:
        apples = ("apple_1", "apple_2")
        bananas = ("banana_1", "banana_2")
        a_a = self._class_in_tray(self.env, apples, "tray_a")
        b_b = self._class_in_tray(self.env, bananas, "tray_b")
        a_b = self._class_in_tray(self.env, apples, "tray_b")
        b_a = self._class_in_tray(self.env, bananas, "tray_a")
        return {
            "apple_group_in_tray_a": a_a,
            "banana_group_in_tray_b": b_b,
            "apple_group_in_tray_b": a_b,
            "banana_group_in_tray_a": b_a,
            "canonical_assignment": a_a and b_b,
            "swapped_assignment": a_b and b_a,
            "accepted_assignment": (a_a and b_b) or (a_b and b_a),
        }

    def _all_moved(self) -> bool:
        for name, initial in self._initial_positions.items():
            current = _body_position(self.env, name)
            if current is None:
                return False
            if initial is not None and float(np.linalg.norm(current[:2] - initial[:2])) < self.movement_threshold_m:
                return False
        return True

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        self._control_steps += 1
        state = self._assignment_state()
        stage_checks = (
            state["apple_group_in_tray_a"]
            or state["apple_group_in_tray_b"]
            or state["banana_group_in_tray_a"]
            or state["banana_group_in_tray_b"],
            state["accepted_assignment"],
        )
        while self._stage_index < len(stage_checks) and stage_checks[self._stage_index]:
            name = self._stage_names[self._stage_index]
            self._stage_done[name] = True
            _record_stage(
                self._events,
                index=self._stage_index,
                name=name,
                control_step=self._control_steps,
                env=self.env,
            )
            self._stage_index += 1

    def result(self) -> dict[str, Any]:
        state = self._assignment_state()
        terminal_checks = {
            "apples_co_located": state["apple_group_in_tray_a"]
            or state["apple_group_in_tray_b"],
            "bananas_co_located": state["banana_group_in_tray_a"]
            or state["banana_group_in_tray_b"],
            "classes_in_separate_trays": state["accepted_assignment"],
            "all_objects_physically_relocated": self._all_moved(),
        }
        success = bool(all(terminal_checks.values()))
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "success": success,
            "assignment": state,
            "accepted_assignment": state["accepted_assignment"],
            "required_stage_count": len(self._stage_names),
            "completed_required_stage_count": self._stage_index,
            "stage_score_percent": 100.0 * self._stage_index / len(self._stage_names),
            "ordered_stage_names": list(self._stage_names),
            "completed_stage_names": list(self._stage_names[: self._stage_index]),
            "terminal_state_success": success,
            "terminal_state_checks": terminal_checks,
            "control_steps_observed": self._control_steps,
            "failure_reason": None if success else "semantic_clusters_invalid",
            "stage_events": list(self._events),
        }


def vlabench_additional_private_evaluator(
    env: Any, *, suite: str, task_id: int
) -> FindUnseenObjectEvaluator | DiningTableEvaluator | ClusterSeriesEvaluator | None:
    """Return the private evaluator for the requested additional task."""

    if suite != VLABENCH_ADDITIONAL_SUITE:
        return None
    normalized_task_id = int(task_id)
    evaluators = (FindUnseenObjectEvaluator, ClusterSeriesEvaluator)
    if 0 <= normalized_task_id < len(evaluators):
        return evaluators[normalized_task_id](env)
    return None


__all__ = [
    "ClusterSeriesEvaluator",
    "DiningTableEvaluator",
    "FindUnseenObjectEvaluator",
    "VLABENCH_ADDITIONAL_SUITE",
    "VLABENCH_ADDITIONAL_TASK_NAMES",
    "vlabench_additional_private_evaluator",
]
