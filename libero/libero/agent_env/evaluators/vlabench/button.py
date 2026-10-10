"""Private evaluators for the VLABench physical-QA button tasks.

Seven tasks ask world-knowledge property questions and one control task uses
a single button without candidates or alternative targets. Answers are never
exposed as observation metadata. Each property candidate is paired with a real
VLABench-style button assembly. Success
requires sustained MuJoCo touch force and simultaneous button-cap/gripper
contact on the correct button; touching or hovering over a candidate cannot
pass. Pressing a wrong button latches failure for the episode.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import mujoco

from libero.libero.benchmark.physical_property_questions import build_question
from libero.libero.benchmark.benchmark_physical_property import (
    PHYSICAL_PROPERTY_TASKS,
)


VLABENCH_BUTTON_SUITE = "vlabench_button"
VLABENCH_BUTTON_TASK_NAME = "select_highest_density"
VLABENCH_BUTTON_PRESS_TASK_NAME = "press_button"
VLABENCH_BUTTON_INSTRUCTION = (
    "Choose the object with the greatest density and press its button."
)
VLABENCH_BUTTON_PRESS_INSTRUCTION = "Press the red button."
VLABENCH_BUTTON_IDLE_RGBA = np.asarray((0.85, 0.12, 0.12, 1.0))
VLABENCH_BUTTON_ACTIVE_RGBA = np.asarray((0.0, 1.0, 0.0, 1.0))

# Keep the task names stable and human-readable, while making the mapping from
# a benchmark task to its question-bank family an evaluator-side detail.  The
# seed is fixed for the runnable task so the trusted reset state and the
# private answer agree; it is never put in the agent observation.
VLABENCH_BUTTON_TASK_SPECS = tuple(
    (task.name, task.family_id, task.seed) for task in PHYSICAL_PROPERTY_TASKS
)
VLABENCH_BUTTON_TASKS = (
    *(spec[0] for spec in VLABENCH_BUTTON_TASK_SPECS),
    VLABENCH_BUTTON_PRESS_TASK_NAME,
)


def _name_variants(name: str) -> tuple[str, ...]:
    return (name, f"{name}_main", f"{name}_object")


def _body_id(env: Any, name: str) -> int | None:
    for candidate in _name_variants(name):
        try:
            return int(env.sim.model.body_name2id(candidate))
        except Exception:
            continue
    return None


def _button_touch_force(env: Any, button: str) -> float:
    """Read the real touch sensor, fail closed if an asset is incomplete."""

    # The XML sensor is prefixed by MujocoXMLObject.  We deliberately resolve
    # by suffix so the private checker does not depend on a public annotation.
    model = env.sim.model
    data = env.sim.data
    for sensor_id in range(int(model.nsensor)):
        name = mujoco.mj_id2name(
            model._model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id
        )
        if name is not None and str(name).endswith(f"{button}_button_touch"):
            adr = int(model.sensor_adr[sensor_id])
            dim = int(model.sensor_dim[sensor_id])
            return float(np.max(np.abs(data.sensordata[adr : adr + dim])))
    # Contact fallback is only used for MuJoCo builds that drop object
    # sensors while merging an object model; it still requires actual contact.
    body_id = _body_id(env, button)
    if body_id is None:
        return 0.0
    geom_ids = {
        index
        for index, geom_body in enumerate(np.asarray(model.geom_bodyid))
        if int(geom_body) == body_id
    }
    max_force = 0.0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if int(contact.geom1) in geom_ids or int(contact.geom2) in geom_ids:
            max_force = max(max_force, float(max(0.0, -contact.dist)))
    return max_force


def _robot_button_contact(env: Any, button: str) -> bool:
    """Require a live button-cap to robot collision, not table support."""

    model = env.sim.model
    data = env.sim.data
    try:
        button_geom = int(model.geom_name2id(f"{button}_button_cap"))
    except Exception:
        return False
    robot_geoms = set()
    for geom_id in range(int(model.ngeom)):
        name = mujoco.mj_id2name(
            model._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        if name is not None and str(name).startswith("gripper0_") and "collision" in str(name):
            robot_geoms.add(geom_id)
    if not robot_geoms:
        return False
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        if not (
            (int(contact.geom1) == button_geom and int(contact.geom2) in robot_geoms)
            or (int(contact.geom2) == button_geom and int(contact.geom1) in robot_geoms)
        ):
            continue
        try:
            mujoco.mj_contactForce(model._model, data._data, contact_id, force)
            return float(np.linalg.norm(force[:3])) >= 1e-3
        except Exception:
            return float(contact.dist) <= 0.0
    return False


def _set_button_activation_color(env: Any, button: str, active: bool) -> bool:
    """Mirror VLABench's red/green live button feedback.

    VLABench's button is a fixed touch target rather than a translating
    mechanical switch.  Its visible state change is therefore the cap color:
    red while idle and green while its touch sensor is active.  Return False
    when a lightweight unit-test double or an incomplete asset has no cap.
    """

    try:
        model = env.sim.model
        geom_id = int(model.geom_name2id(f"{button}_button_cap"))
        model.geom_rgba[geom_id] = (
            VLABENCH_BUTTON_ACTIVE_RGBA
            if active
            else VLABENCH_BUTTON_IDLE_RGBA
        )
    except Exception:
        return False
    return True


class VLABenchPhysicalPropertyButtonEvaluator:
    """Private contact-gated evaluator shared by all seven property tasks.

    The task is a world-knowledge selection followed by a physical press.  It
    deliberately does not inspect candidate geometry or run a dynamics
    experiment: the question bank supplies the private answer, and the live
    MuJoCo touch sensor plus gripper-cap collision establish the physical
    action.  A contact with either wrong cap permanently fails the episode.
    """

    schema_version = "libero.vlabench_button.physical_property.v1"
    button_names = ("button_1", "button_2", "button_3")
    touch_force_threshold = 0.001
    required_pressed_steps = 11

    def __init__(
        self,
        env: Any,
        *,
        family_id: str,
        seed: int,
        task_id: int = 0,
    ) -> None:
        self.env = env
        self.family_id = str(family_id)
        self.seed = int(seed)
        self.task_id = int(task_id)
        self._question = build_question(self.family_id, self.seed)
        self._target_index = int(self._question.target_index)
        # This identity is host-private.  It is intentionally not included
        # in ``result`` or any agent-visible observation.
        self._target_button = self.button_names[self._target_index]
        self.reset()

    @property
    def target_button(self) -> str:
        """Return the private answer identity for the host-side checker."""

        return self._target_button

    def reset(self) -> None:
        self._control_steps = 0
        self._physics_substeps = 0
        # ``_pressed_steps`` is a diagnostic total.  Success uses the maximum
        # uninterrupted streak so six presses, a release, and six more presses
        # cannot satisfy the sustained-contact requirement.
        self._pressed_steps = {name: 0 for name in self.button_names}
        self._pressed_streak = {name: 0 for name in self.button_names}
        self._max_pressed_streak = {name: 0 for name in self.button_names}
        self._ever_pressed = {name: False for name in self.button_names}
        self._wrong_button_pressed = False
        self._touch_seen = {name: False for name in self.button_names}
        for name in self.button_names:
            _set_button_activation_color(self.env, name, False)

    def observe_physics_substep(self) -> None:
        """Track activation at VLABench's original physics-substep rate."""

        self._physics_substeps += 1
        for name in self.button_names:
            force = _button_touch_force(self.env, name)
            _set_button_activation_color(
                self.env,
                name,
                force >= self.touch_force_threshold,
            )
            pressed = force >= self.touch_force_threshold and _robot_button_contact(
                self.env, name
            )
            if force >= self.touch_force_threshold:
                self._touch_seen[name] = True
            if pressed:
                self._pressed_steps[name] += 1
                self._pressed_streak[name] += 1
                self._max_pressed_streak[name] = max(
                    self._max_pressed_streak[name], self._pressed_streak[name]
                )
                self._ever_pressed[name] = True
                if name != self.target_button:
                    self._wrong_button_pressed = True
            else:
                # A press is a sustained physical event, not an episode-total
                # count.  Any control cycle without simultaneous sensor and
                # gripper-cap contact breaks the streak.
                self._pressed_streak[name] = 0

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        # Keep a control-rate diagnostic alongside the physics-rate press
        # state. VLABench updates Button activation from ``after_substep``;
        # counting only this method would make its >10 condition 25x longer
        # at LIBERO's 20 Hz controller and 2 ms MuJoCo timestep.
        self._control_steps += 1

    def result(self) -> dict[str, Any]:
        target_pressed = (
            self._max_pressed_streak[self.target_button]
            >= self.required_pressed_steps
        )
        success = bool(target_pressed and not self._wrong_button_pressed)
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "success": success,
            "target_button_pressed": target_pressed,
            "wrong_button_pressed": bool(self._wrong_button_pressed),
            "touch_observed_private": dict(self._touch_seen),
            "pressed_steps_private": dict(self._pressed_steps),
            "pressed_streak_private": dict(self._pressed_streak),
            "max_pressed_streak_private": dict(self._max_pressed_streak),
            "control_steps_observed": self._control_steps,
            "physics_substeps_observed": self._physics_substeps,
            "failure_reason": None if success else "target_button_not_sustained_without_wrong_press",
        }


class VLABenchDensityButtonEvaluator(VLABenchPhysicalPropertyButtonEvaluator):
    """Backward-compatible host checker for the original density task."""

    schema_version = "libero.vlabench_button.density.v1"

    def __init__(self, env: Any) -> None:
        super().__init__(env, family_id="density", seed=102, task_id=0)


class VLABenchPressButtonEvaluator(VLABenchPhysicalPropertyButtonEvaluator):
    """Private checker for the isolated single-button control task."""

    schema_version = "libero.vlabench_button.press.v1"
    button_names = ("button_1",)

    def __init__(self, env: Any) -> None:
        # This task deliberately contains no property question or alternative
        # target: it isolates reaching and sustaining a press on button_1.
        self.env = env
        self.family_id = "mechanical_press"
        self.seed = -1
        self.task_id = len(VLABENCH_BUTTON_TASK_SPECS)
        self._question = None
        self._target_index = 0
        self._target_button = "button_1"
        self.reset()


def vlabench_button_private_evaluator(
    env: Any, *, suite: str, task_id: int
) -> VLABenchPhysicalPropertyButtonEvaluator | None:
    if suite != VLABENCH_BUTTON_SUITE:
        return None
    normalized_task_id = int(task_id)
    if normalized_task_id == len(VLABENCH_BUTTON_TASK_SPECS):
        return VLABenchPressButtonEvaluator(env)
    try:
        _task_name, family_id, seed = VLABENCH_BUTTON_TASK_SPECS[
            normalized_task_id
        ]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"unknown {VLABENCH_BUTTON_SUITE} task_id={task_id}"
        ) from exc
    if normalized_task_id == 0:
        # Preserve the original public Python contract for the first task.
        return VLABenchDensityButtonEvaluator(env)
    return VLABenchPhysicalPropertyButtonEvaluator(
        env,
        family_id=family_id,
        seed=seed,
        task_id=normalized_task_id,
    )


__all__ = [
    "VLABENCH_BUTTON_INSTRUCTION",
    "VLABENCH_BUTTON_PRESS_INSTRUCTION",
    "VLABENCH_BUTTON_PRESS_TASK_NAME",
    "VLABENCH_BUTTON_SUITE",
    "VLABENCH_BUTTON_TASK_NAME",
    "VLABENCH_BUTTON_TASK_SPECS",
    "VLABENCH_BUTTON_TASKS",
    "VLABenchPhysicalPropertyButtonEvaluator",
    "VLABenchDensityButtonEvaluator",
    "VLABenchPressButtonEvaluator",
    "vlabench_button_private_evaluator",
]
