"""RoboMemArena task source for the agent benchmark.

A frozen compatibility subset supplies simulation overrides, BDDL, and private
ordered-stage semantics. Public observations, control transport, audit logs,
and success reporting remain owned by this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping

import numpy as np

from libero.libero.envs import OffScreenRenderEnv

from ...runtime.control import OSCControlConfig
from ...runtime.environment import LiberoAgentEnv
from ...runtime.observation import TaskEntitySelection
from ...runtime.profiles import ObservationProfile
from .vendor import UPSTREAM_COMMIT
from .vendor.stage import reference_stage


ROBOMEMARENA_SUITE = "robomemarena"
ROBOMEMARENA_SOURCE_SCHEMA_VERSION = "libero.robomemarena_source.v3"
ROBOMEMARENA_VENDOR_ROOT = (
    Path(__file__).resolve().parent / "vendor"
)


@dataclass(frozen=True)
class RoboMemArenaTaskSpec:
    task_id: int
    instruction: str
    bddl_relative_path: str
    required_stage_names: tuple[str, ...]
    optional_stage_names: tuple[str, ...] = ()


TASK_INSTRUCTIONS = {
    1: "Pick and place cookies into the basket, then pick and place tomato sauce into the same basket.",
    2: "Pick and place butter into the basket, then pick and place popcorn into the same basket.",
    3: "Pick and place cream into the basket, then pick and place chocolate into the same basket.",
    4: "Open and close all drawers in order to check. Put butter into the drawer that already contains an object.",
    5: "Open and close all drawers in order to check. Put butter into the empty drawer.",
    6: "Pour tomato sauce over cookies twice and place the sauce bottle into the bowl drainer.",
    7: "Pour tomato sauce over the frypan twice and place the sauce bottle into the bowl drainer.",
    8: "Pick and place chocolate into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
    9: "Put butter into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
    10: "Pour wine into the mug twice.",
    11: "Put cookies into the top drawer and put butter into another drawer.",
    12: "Put cookies into the middle drawer, then put chocolate into the same drawer, and close it.",
    13: "Put cookies into the middle drawer and then put butter into the same drawer.",
    14: "Put cookies into the top drawer and put chocolate into another drawer.",
    15: "Pick and place butter into the frypan, then pour milk over it twice.",
    16: "Pick milk from the table, pour it into the mug twice, then place the milk container into the bowl drainer.",
    17: "Put butter into the middle drawer and then put chocolate into the same drawer.",
    18: "Pick and place chocolate and butter from cabinet1 to cabinet2, respectively.",
    19: "Pick and place tomato sauce, milk, and orange juice from cabinet1 to cabinet2.",
    20: "Put cookies into the microwave and then put chocolate into the location where the cookies were placed.",
    21: "Put butter into the microwave and then put chocolate into the location where the butter is placed.",
    22: "Pour tomato sauce over cookies twice, then put the cookies into the microwave.",
    23: "Put cream into the microwave and then put popcorn into the location where the cream was placed.",
    24: "Put cookies into the microwave and then put popcorn into the location where the cookies were placed.",
    25: "Pick and place butter and cream from plate1 to plate2, respectively.",
    26: "Pick and place chocolate and cream from plate1 to plate2, respectively.",
}

TASK_BDDL_FILENAMES = {
    1: "1_cookies_tomato_basket.bddl",
    2: "2_butter_popcorn_basket.bddl",
    3: "3_cream_pudding_basket.bddl",
    4: "4_drawer_butter.bddl",
    5: "5_butter_middle_drawer.bddl",
    6: "6_pour_tomato_sauce_into_bowl_drainer.bddl",
    7: "7_pour_tomato_sauce_into_frypan.bddl",
    8: "8_pick_chocolate_in_frypan_pour_tomato_sauce_twice.bddl",
    9: "9_pick_butter_in_frypan_pour_tomato_sauce_twice_and_place_tomato_sauce_in_bowl_drainer.bddl",
    10: "10_pour_wine_bottle_into_mug.bddl",
    11: "11_cookies_top_butter_middle.bddl",
    12: "12_cookies_chocolate_middle_drawer.bddl",
    13: "13_butter_cookies_middle_drawer.bddl",
    14: "14_cookies_chocolate_drawers.bddl",
    15: "15_butter_milk_frypan.bddl",
    16: "16_pour_milk_in_red_coffee_mug_twice_and_place_milk_in_bowl_drainer.bddl",
    17: "17_butter_chocolate_middle_drawer.bddl",
    18: "18_chocolate_butter_cabinet.bddl",
    19: "19_tomato_milk_orange_cabinet.bddl",
    20: "20_cookies_chocolate_microwave.bddl",
    21: "21_butter_chocolate_microwave.bddl",
    22: "22_pour_tomato_cookies_microwave.bddl",
    23: "23_cream_popcorn_microwave.bddl",
    24: "24_cookies_popcorn_microwave.bddl",
    25: "25_butter_cream.bddl",
    26: "26_chocolate_pudding_cream.bddl",
}

TASK4_OCCUPIED_DRAWER_BY_INIT_STATE_ID = {
    0: "top",
    1: "middle",
    2: "bottom",
}
TASK4_INIT_STATE_ID_BY_OCCUPIED_DRAWER = {
    drawer: init_state_id
    for init_state_id, drawer in TASK4_OCCUPIED_DRAWER_BY_INIT_STATE_ID.items()
}


def _build_task_specs() -> dict[int, RoboMemArenaTaskSpec]:
    specs: dict[int, RoboMemArenaTaskSpec] = {}
    for task_id, instruction in TASK_INSTRUCTIONS.items():
        stage_specs = reference_stage._task_specs(task_id)
        optional_stage = reference_stage._optional_final_stage_name(task_id)
        required = tuple(
            stage.name for stage in stage_specs if stage.name != optional_stage
        )
        optional = () if optional_stage is None else (optional_stage,)
        specs[task_id] = RoboMemArenaTaskSpec(
            task_id=task_id,
            instruction=instruction,
            bddl_relative_path=(
                "evaluation_benchmark/bddl/" + TASK_BDDL_FILENAMES[task_id]
            ),
            required_stage_names=required,
            optional_stage_names=optional,
        )
    return specs


TASK_SPECS = _build_task_specs()


def get_robomemarena_task_spec(task_id: int) -> RoboMemArenaTaskSpec:
    try:
        return TASK_SPECS[int(task_id)]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in sorted(TASK_SPECS))
        raise ValueError(
            f"unsupported RoboMemArena task_id {task_id}; supported: {supported}"
        ) from exc


def robomemarena_source_fingerprint(
    checkout_root: str | os.PathLike[str] | None = None,
    *,
    task_id: int,
    init_state_id: int = 0,
) -> dict[str, Any]:
    """Validate and fingerprint the evaluator-private frozen task source."""

    spec = get_robomemarena_task_spec(task_id)
    task_variant = robomemarena_task_variant(
        task_id=task_id, init_state_id=init_state_id
    )
    bddl_path = robomemarena_bddl_path(
        checkout_root,
        task_id=task_id,
        init_state_id=init_state_id,
    )
    if checkout_root is None:
        source_kind = "vendored_compatibility_subset"
        source_commit = UPSTREAM_COMMIT
        cabinet_path = (
            ROBOMEMARENA_VENDOR_ROOT
            / "core/assets/articulated_objects/"
            "wooden_cabinet_tall_bottom.xml"
        )
        upstream_stage_path = None
    else:
        source_kind = "external_clean_checkout"
        root = Path(checkout_root).expanduser().resolve()
        cabinet_path = (
            root
            / "evaluation_benchmark/libero_fork/libero/assets/"
            "articulated_objects/wooden_cabinet_tall_bottom.xml"
        )
        upstream_stage_path = (
            root
            / "evaluation_benchmark/scripts/task2_26_reference_stage.py"
        )
        source_commit = _git_output(root, "rev-parse", "HEAD")
        tracked_status = _git_output(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        )
        if tracked_status:
            raise RuntimeError(
                "RoboMemArena checkout has tracked modifications; use a "
                "clean, versioned task source"
            )
    required_files = {
        "bddl_sha256": bddl_path,
        "cabinet_asset_sha256": cabinet_path,
        "runtime_stage_reference_sha256": (
            ROBOMEMARENA_VENDOR_ROOT / "stage/reference_stage.py"
        ),
    }
    if upstream_stage_path is not None:
        required_files["external_upstream_stage_reference_sha256"] = (
            upstream_stage_path
        )
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "RoboMemArena task source is incomplete: " + ", ".join(missing)
        )
    return {
        "schema_version": ROBOMEMARENA_SOURCE_SCHEMA_VERSION,
        "task_id": spec.task_id,
        "init_state_id": int(init_state_id),
        "task_variant": task_variant,
        "bddl_filename": bddl_path.name,
        "bddl_source_kind": (
            "vendored_initial_state_variant"
            if task_id == 4 and task_variant != "top"
            else source_kind
        ),
        "source_kind": source_kind,
        "source_commit": source_commit,
        **{
            label: _file_sha256(path)
            for label, path in required_files.items()
        },
    }


def robomemarena_bddl_path(
    checkout_root: str | os.PathLike[str] | None,
    *,
    task_id: int,
    init_state_id: int = 0,
) -> Path:
    spec = get_robomemarena_task_spec(task_id)
    task_variant = robomemarena_task_variant(
        task_id=task_id, init_state_id=init_state_id
    )
    filename = Path(spec.bddl_relative_path).name
    if task_id == 4 and task_variant != "top":
        filename = f"4_drawer_butter_{task_variant}.bddl"
        # These two variants repair an upstream packaging omission. They are
        # frozen in this adapter even when an external checkout supplies the
        # simulator implementation and assets.
        return ROBOMEMARENA_VENDOR_ROOT / "bddl" / filename
    if checkout_root is None:
        return ROBOMEMARENA_VENDOR_ROOT / "bddl" / filename
    return Path(checkout_root).expanduser().resolve() / spec.bddl_relative_path


def robomemarena_task_variant(*, task_id: int, init_state_id: int) -> str:
    """Resolve evaluator-private initial-state variants.

    RoboMemArena Task 4 trajectories use all three occupied drawers, while
    upstream publishes only the top-drawer BDDL. The adapter therefore uses
    init-state IDs 0/1/2 for top/middle/bottom. Other tasks currently expose
    their single frozen BDDL as init-state ID 0.
    """

    task_id = int(task_id)
    init_state_id = int(init_state_id)
    if task_id == 4:
        try:
            return TASK4_OCCUPIED_DRAWER_BY_INIT_STATE_ID[init_state_id]
        except KeyError as exc:
            raise ValueError(
                "RoboMemArena Task 4 init_state_id must be 0 (top), "
                "1 (middle), or 2 (bottom)"
            ) from exc
    if init_state_id != 0:
        raise ValueError(
            f"RoboMemArena Task {task_id} supports init_state_id 0 only"
        )
    return "default"


def task4_init_state_id_from_recorded_instruction(instruction: str) -> int:
    """Recover Task 4's omitted occupied-drawer variant from HDF5 metadata."""

    match = re.search(
        r"open the (top|middle|bottom) drawer again",
        " ".join(str(instruction).lower().split()),
    )
    if match is None:
        raise ValueError(
            "Task 4 trajectory instruction does not identify the occupied "
            "drawer"
        )
    return TASK4_INIT_STATE_ID_BY_OCCUPIED_DRAWER[match.group(1)]


class RoboMemArenaTask4Evaluator:
    """Evaluator-private ordered-stage monitor for RoboMemArena Task 4."""

    OPEN_THRESHOLD_M = 0.10
    CLOSED_THRESHOLD_M = 0.08
    BUTTER_HORIZONTAL_THRESHOLD_M = 0.25
    BUTTER_HEIGHT_THRESHOLD_M = 0.15

    def __init__(self, env: Any) -> None:
        self.env = env
        self.spec = get_robomemarena_task_spec(4)
        self._initial_sites: dict[str, np.ndarray] = {}
        self._completed: list[str] = []
        self._events: list[dict[str, Any]] = []
        self._control_step = 0
        self._active = False
        self._target_drawer: str | None = None
        self._checks: tuple[tuple[str, Callable[[], bool]], ...] = ()

    def reset(self) -> None:
        self._initial_sites = {
            drawer: self._site_position(
                f"wooden_cabinet_1_{drawer}_region"
            )
            for drawer in ("top", "middle", "bottom")
        }
        hidden_object = self._body_position("cream_cheese_1")
        self._target_drawer = min(
            self._initial_sites,
            key=lambda drawer: float(
                np.linalg.norm(hidden_object - self._initial_sites[drawer])
            ),
        )
        self._checks = (
            (self.spec.required_stage_names[0], lambda: self._drawer_open("top")),
            (self.spec.required_stage_names[1], lambda: self._drawer_closed("top")),
            (
                self.spec.required_stage_names[2],
                lambda: self._drawer_open("middle"),
            ),
            (
                self.spec.required_stage_names[3],
                lambda: self._drawer_closed("middle"),
            ),
            (
                self.spec.required_stage_names[4],
                lambda: self._drawer_open("bottom"),
            ),
            (
                self.spec.required_stage_names[5],
                lambda: self._drawer_closed("bottom"),
            ),
            (
                self.spec.required_stage_names[6],
                lambda: self._drawer_open(self._required_target_drawer()),
            ),
            (self.spec.required_stage_names[7], self._butter_in_target_drawer),
            (
                self.spec.optional_stage_names[0],
                lambda: self._drawer_closed(self._required_target_drawer()),
            ),
        )
        self._completed = []
        self._events = []
        self._control_step = 0
        self._active = True

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        if not self._active:
            return
        self._control_step += 1
        stage_index = len(self._completed)
        if stage_index >= len(self._checks):
            return
        name, check = self._checks[stage_index]
        if check():
            self._completed.append(name)
            self._events.append(
                {
                    "stage_index": stage_index,
                    "stage_name": name,
                    "control_step": self._control_step,
                    "sim_time_s": float(self.env.sim.data.time),
                }
            )

    def result(self) -> dict[str, Any]:
        required = self.spec.required_stage_names
        required_completed = sum(name in self._completed for name in required)
        terminal_success = self._butter_in_target_drawer()
        success = required_completed == len(required) and terminal_success
        return {
            "schema_version": "libero.robomemarena_private_evaluation.v1",
            "task_id": self.spec.task_id,
            "success": success,
            "required_stage_count": len(required),
            "completed_required_stage_count": required_completed,
            "stage_score_percent": (
                100.0 * required_completed / len(required)
            ),
            "ordered_stage_names": [
                *required,
                *self.spec.optional_stage_names,
            ],
            "completed_stage_names": list(self._completed),
            "optional_stage_names": list(self.spec.optional_stage_names),
            "terminal_state_success": terminal_success,
            "terminal_state_checks": {
                "Terminal_Butter_In_Occupied_Drawer": terminal_success
            },
            "control_steps_observed": self._control_step,
            "stage_events": list(self._events),
        }

    def _drawer_open(self, drawer: str) -> bool:
        return self._drawer_displacement(drawer) > self.OPEN_THRESHOLD_M

    def _drawer_closed(self, drawer: str) -> bool:
        return self._drawer_displacement(drawer) < self.CLOSED_THRESHOLD_M

    def _drawer_displacement(self, drawer: str) -> float:
        current = self._site_position(
            f"wooden_cabinet_1_{drawer}_region"
        )
        initial = self._initial_sites[drawer]
        return abs(float(current[1] - initial[1]))

    def _required_target_drawer(self) -> str:
        if self._target_drawer is None:
            raise RuntimeError("Task 4 checker has not been reset")
        return self._target_drawer

    def _butter_in_target_drawer(self) -> bool:
        butter = self._body_position("butter_1")
        region = self._site_position(
            f"wooden_cabinet_1_{self._required_target_drawer()}_region"
        )
        horizontal = float(np.linalg.norm(butter[:2] - region[:2]))
        height = abs(float(butter[2] - region[2]))
        return (
            horizontal < self.BUTTER_HORIZONTAL_THRESHOLD_M
            and height < self.BUTTER_HEIGHT_THRESHOLD_M
        )

    def _site_position(self, name: str) -> np.ndarray:
        for candidate in (name, f"{name}_main"):
            try:
                site_id = self.env.sim.model.site_name2id(candidate)
            except Exception:
                continue
            return np.asarray(
                self.env.sim.data.site_xpos[site_id], dtype=np.float64
            ).copy()
        raise RuntimeError(f"private checker could not resolve site {name!r}")

    def _body_position(self, name: str) -> np.ndarray:
        for candidate in (name, f"{name}_main"):
            try:
                body_id = self.env.sim.model.body_name2id(candidate)
            except Exception:
                continue
            return np.asarray(
                self.env.sim.data.body_xpos[body_id], dtype=np.float64
            ).copy()
        raise RuntimeError(f"private checker could not resolve body {name!r}")


class RoboMemArenaOrderedStageEvaluator:
    """Evaluator-private ordered checker frozen from RoboMemArena Task 1--26."""

    def __init__(self, env: Any, *, task_id: int) -> None:
        self.env = env
        self.spec = get_robomemarena_task_spec(task_id)
        self._stage_specs: list[Any] = []
        self._terminal_specs: list[Any] = []
        self._stage_done: dict[str, bool] = {}
        self._stage_index = 0
        self._stage_start = 0
        self._state: dict[str, Any] = {}
        self._events: list[dict[str, Any]] = []
        self._control_step = 0
        self._active = False
        self._extra_pour_check: Callable[..., bool] | None = None
        self._extra_pour_detected = False

    def reset(self) -> None:
        self._stage_specs = reference_stage._task_specs(self.spec.task_id)
        self._terminal_specs = reference_stage._terminal_task_specs(
            self.spec.task_id
        )
        self._stage_done = {
            stage.name: False for stage in self._stage_specs
        }
        self._stage_index = 0
        self._state = reference_stage._build_initial_state(self.env)
        self._stage_start = int(self._state["step_idx"])
        self._events = []
        self._control_step = 0
        self._extra_pour_check = reference_stage._extra_pour_check(
            self.spec.task_id
        )
        self._extra_pour_detected = False
        self._active = True

    def observe(self, raw_observation: Mapping[str, Any]) -> None:
        if not self._active:
            return
        self._control_step += 1
        reference_stage._update_state(raw_observation, self._state)

        if self._stage_index < len(self._stage_specs):
            stage = self._stage_specs[self._stage_index]
            if stage.check_fn(self.env, self._state, self._stage_start):
                self._stage_done[stage.name] = True
                self._events.append(
                    {
                        "stage_index": self._stage_index,
                        "stage_name": stage.name,
                        "control_step": self._control_step,
                        "sim_time_s": float(self.env.sim.data.time),
                    }
                )
                self._stage_index += 1
                self._stage_start = int(self._state["step_idx"])

        if (
            self._extra_pour_check is not None
            and any(
                name.endswith("_Pour_Two") and complete
                for name, complete in self._stage_done.items()
            )
            and self._extra_pour_check(
                self.env, self._state, self._stage_start
            )
        ):
            self._extra_pour_detected = True

    def result(self) -> dict[str, Any]:
        required_names = list(self.spec.required_stage_names)
        completed_names = [
            name for name, complete in self._stage_done.items() if complete
        ]
        required_completed = sum(
            self._stage_done.get(name, False) for name in required_names
        )
        required_success = reference_stage._stage_success_from_stage_done(
            self.spec.task_id, self._stage_done
        )
        terminal_state = {
            spec.name: bool(spec.check_fn(self.env, self._state, 0))
            for spec in self._terminal_specs
        }
        terminal_success = all(terminal_state.values())
        success = bool(
            required_success
            and terminal_success
            and not self._extra_pour_detected
        )
        if self._extra_pour_detected:
            failure_reason = "extra_pour"
        elif not required_success:
            failure_reason = "incomplete_stage"
        elif not terminal_success:
            failure_reason = "terminal_state_invalid"
        else:
            failure_reason = None
        return {
            "schema_version": "libero.robomemarena_private_evaluation.v3",
            "task_id": self.spec.task_id,
            "success": success,
            "required_stage_count": len(required_names),
            "completed_required_stage_count": required_completed,
            "stage_score_percent": reference_stage._stage_score_pct(
                self.spec.task_id, self._stage_done
            ),
            "ordered_stage_names": list(self._stage_done),
            "completed_stage_names": completed_names,
            "optional_stage_names": list(self.spec.optional_stage_names),
            "terminal_state_success": terminal_success,
            "terminal_state_checks": terminal_state,
            "control_steps_observed": self._control_step,
            "extra_pour_detected": self._extra_pour_detected,
            "failure_reason": failure_reason,
            "stage_events": list(self._events),
        }


def make_robomemarena_agent_env(
    *,
    checkout_root: str | os.PathLike[str] | None = None,
    task_id: int = 4,
    init_state_id: int = 0,
    profile: ObservationProfile | int | str = ObservationProfile.LEVEL3,
    seed: int = 0,
    camera_height: int = 256,
    camera_width: int = 256,
    task_entities: TaskEntitySelection | None = None,
    control_config: OSCControlConfig | None = None,
    initial_settle_control_steps: int = 10,
    max_agent_steps: int | None = None,
    private_control_step_callback: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    render_gpu_device_id: int = -1,
    **env_kwargs: Any,
) -> LiberoAgentEnv:
    """Create one task using RoboMemArena physics and our public contract."""

    from ...runtime.contracts.release import validate_release_task
    validate_release_task(ROBOMEMARENA_SUITE, task_id)
    spec = get_robomemarena_task_spec(task_id)
    robomemarena_task_variant(
        task_id=task_id, init_state_id=init_state_id
    )
    robomemarena_source_fingerprint(
        checkout_root,
        task_id=task_id,
        init_state_id=init_state_id,
    )
    bddl_path = robomemarena_bddl_path(
        checkout_root,
        task_id=task_id,
        init_state_id=init_state_id,
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
        "control_freq",
    }
    conflicts = reserved.intersection(env_kwargs)
    if conflicts:
        raise ValueError(
            "factory-managed env kwargs cannot be overridden: "
            f"{sorted(conflicts)}"
        )
    # RoboMemArena's official evaluator seeds NumPy as well as the wrapped
    # environment before reset. Keep the same deterministic scene contract.
    np.random.seed(seed)
    env = OffScreenRenderEnv(
        bddl_file_name=os.fspath(bddl_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_depths=True,
        camera_segmentations="instance",
        # RoboMemArena's fork currently assumes these private sensors exist.
        # They stay inside the server; MasterObservationCollector allowlists the
        # public state and never serializes object pose/relative-pose fields.
        use_object_obs=True,
        ignore_done=True,
        initialization_noise=None,
        render_gpu_device_id=render_gpu_device_id,
        horizon=10000,
        control_freq=20,
        **env_kwargs,
    )
    env.seed(seed)
    evaluator = RoboMemArenaOrderedStageEvaluator(env, task_id=task_id)
    return LiberoAgentEnv(
        env,
        profile=profile,
        camera_height=camera_height,
        camera_width=camera_width,
        task_instruction=spec.instruction,
        initial_state=None,
        task_entities=task_entities,
        control_config=control_config,
        initial_settle_control_steps=initial_settle_control_steps,
        max_agent_steps=max_agent_steps,
        private_control_step_callback=private_control_step_callback,
        private_episode_evaluator=evaluator,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", os.fspath(root), *arguments),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
