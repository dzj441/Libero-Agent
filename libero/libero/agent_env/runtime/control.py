"""High-level EEF deltas backed by LIBERO's normalized OSC_POSE controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np


# Current v2 agent protocol: one submitted sequence may contain up to 50
# native OSC policy intervals.  The episode-level Agent-step budget remains a
# separate limit (and is enforced by ``LiberoAgentEnv`` / the service).
MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION = 50
MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS = 1_000_000


class ActionInterface(str, Enum):
    """Public robot-control condition selected for one benchmark run."""

    METRIC_OSC_STEP = "metric_osc_step"
    NATIVE_OSC_SEQUENCE = "native_osc_sequence"

    @classmethod
    def parse(cls, value: "ActionInterface | str") -> "ActionInterface":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"unknown action interface {value!r}; expected one of {choices}"
            ) from exc

    @property
    def wire_command(self) -> str:
        if self is ActionInterface.METRIC_OSC_STEP:
            return "osc_step"
        return "osc_sequence"

    @property
    def cli_operation(self) -> str:
        return self.wire_command.replace("_", "-")


@dataclass(frozen=True)
class EEFCommand:
    """A robot-base-frame EEF delta expressed in physical units.

    Rotation uses a rotation vector (axis multiplied by angle), not Euler / RPY
    components.  The delta is left-applied in the robot base frame.
    """

    delta_position_m: np.ndarray
    delta_rotation_rotvec_rad: np.ndarray
    delta_gripper_width_m: float = 0.0

    @classmethod
    def create(
        cls,
        delta_position_m: Sequence[float] = (0.0, 0.0, 0.0),
        delta_rotation_rotvec_rad: Sequence[float] = (0.0, 0.0, 0.0),
        delta_gripper_width_m: float = 0.0,
    ) -> "EEFCommand":
        position = _finite_vector(delta_position_m, "delta_position_m")
        rotation = _finite_vector(
            delta_rotation_rotvec_rad, "delta_rotation_rotvec_rad"
        )
        gripper = _finite_scalar(delta_gripper_width_m, "delta_gripper_width_m")
        return cls(position, rotation, gripper)


@dataclass(frozen=True)
class OSCControlConfig:
    """Internal execution limits, tolerances, and native OSC scaling.

    These are substep limits, not public high-level action limits.  A larger
    finite EEF command is accepted and executed over multiple control cycles.
    """

    native_translation_scale_m: float = 0.05
    native_rotation_scale_rad: float = 0.5
    max_translation_substep_m: float = 0.04
    max_rotation_substep_rad: float = 0.35
    position_tolerance_m: float = 0.003
    orientation_tolerance_rad: float = 0.03
    gripper_target_tolerance_m: float = 1.0e-6
    max_motion_control_steps: int = 80
    post_action_settle_steps: int = 2

    def __post_init__(self) -> None:
        positive_floats = (
            "native_translation_scale_m",
            "native_rotation_scale_rad",
            "max_translation_substep_m",
            "max_rotation_substep_rad",
            "position_tolerance_m",
            "orientation_tolerance_rad",
            "gripper_target_tolerance_m",
        )
        for name in positive_floats:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_translation_substep_m > self.native_translation_scale_m:
            raise ValueError("translation substep exceeds native OSC scale")
        if self.max_rotation_substep_rad > self.native_rotation_scale_rad:
            raise ValueError("rotation substep exceeds native OSC scale")
        if self.max_motion_control_steps <= 0:
            raise ValueError("motion control-step limit must be positive")
        if self.post_action_settle_steps < 0:
            raise ValueError("post_action_settle_steps must be non-negative")


@dataclass(frozen=True)
class EEFExecution:
    """Safe execution metadata returned alongside the next observation."""

    command_completed: bool
    target_reached: bool
    gripper_command_completed: bool
    termination_reason: str
    position_error_m: float
    orientation_error_rad: float
    control_steps: int
    motion_control_steps: int
    gripper_control_steps: int
    settle_control_steps: int
    requested_gripper_width_m: float
    commanded_gripper_width_m: float
    actual_gripper_width_m: float

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "command_completed": self.command_completed,
            "target_reached": self.target_reached,
            "gripper_command_completed": self.gripper_command_completed,
            "termination_reason": self.termination_reason,
            "position_error_m": self.position_error_m,
            "orientation_error_rad": self.orientation_error_rad,
            "control_steps": self.control_steps,
            "motion_control_steps": self.motion_control_steps,
            "gripper_control_steps": self.gripper_control_steps,
            "settle_control_steps": self.settle_control_steps,
            "requested_gripper_width_m": self.requested_gripper_width_m,
            "commanded_gripper_width_m": self.commanded_gripper_width_m,
            "actual_gripper_width_m": self.actual_gripper_width_m,
        }


@dataclass(frozen=True)
class NativeOSCSequenceExecution:
    """Safe metadata for one accepted native OSC micro-action sequence."""

    command_completed: bool
    termination_reason: str
    micro_step_count: int
    control_steps: int

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "command_completed": self.command_completed,
            "termination_reason": self.termination_reason,
            "micro_step_count": self.micro_step_count,
            "control_steps": self.control_steps,
        }


class NativeOSCSequenceExecutor:
    """Execute an exact bounded sequence of normalized LIBERO OSC actions."""

    def __init__(
        self,
        env: Any,
        control_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.env = env
        self.control_step_callback = control_step_callback

    def execute(
        self, actions: Sequence[Sequence[float]]
    ) -> tuple[dict[str, Any], NativeOSCSequenceExecution]:
        sequence = validate_native_osc_sequence(actions)
        raw_observation: dict[str, Any] | None = None
        for action in sequence:
            raw_observation, _reward, _done, _info = self.env.step(action)
            if self.control_step_callback is not None:
                self.control_step_callback(raw_observation)
        if raw_observation is None:  # Defensive; validation rejects an empty batch.
            raise RuntimeError("native OSC sequence produced no observation")
        count = int(sequence.shape[0])
        return raw_observation, NativeOSCSequenceExecution(
            command_completed=True,
            termination_reason="sequence_executed",
            micro_step_count=count,
            control_steps=count,
        )


def validate_native_osc_sequence(
    actions: Sequence[Sequence[float]],
) -> np.ndarray:
    """Validate without clipping so accepted actions retain dataset semantics."""

    try:
        sequence = np.asarray(actions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("actions must be a rectangular numeric array") from exc
    if sequence.size == 0:
        raise ValueError(
            "actions must contain between 1 and "
            f"{MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION} micro steps"
        )
    if sequence.ndim != 2 or sequence.shape[1:] != (7,):
        raise ValueError("actions must have shape [N, 7]")
    count = int(sequence.shape[0])
    if not 1 <= count <= MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION:
        raise ValueError(
            "actions must contain between 1 and "
            f"{MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION} micro steps"
        )
    if not np.all(np.isfinite(sequence)):
        raise ValueError("actions must contain only finite numbers")
    if np.any(np.abs(sequence) > 1.0):
        raise ValueError("native OSC action components must be within [-1, 1]")
    return sequence.copy()


class BaseFrameOSCExecutor:
    """Execute physical EEF deltas through ordinary normalized OSC actions."""

    def __init__(
        self,
        env: Any,
        config: OSCControlConfig | None = None,
        control_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.env = env
        self.config = config or OSCControlConfig()
        self.control_step_callback = control_step_callback

    def execute(self, command: EEFCommand) -> tuple[dict[str, Any], EEFExecution]:
        robot = self.env.robots[0]
        controller = robot.controller
        controller.update(force=True)

        rotation_world_from_base = quaternion_xyzw_to_matrix(robot.base_ori)
        position_world_from_base = np.asarray(robot.base_pos, dtype=np.float64)
        current_position_base, current_rotation_base = _controller_pose_in_base(
            controller,
            rotation_world_from_base,
            position_world_from_base,
        )
        target_position_base = current_position_base + command.delta_position_m
        target_rotation_base = (
            rotation_vector_to_matrix(command.delta_rotation_rotvec_rad)
            @ current_rotation_base
        )

        raw_obs: dict[str, Any] | None = None
        control_steps = 0
        motion_steps = 0
        gripper_steps = 0
        actual_gripper_width = _gripper_actual_width_m(robot)
        current_gripper_target = _gripper_internal_target_width_m(robot)
        minimum_gripper_width, maximum_gripper_width = _gripper_width_limits_m(robot)
        requested_gripper_width = (
            current_gripper_target
            if command.delta_gripper_width_m == 0.0
            else actual_gripper_width + command.delta_gripper_width_m
        )
        range_tolerance = 1.0e-9
        if not (
            minimum_gripper_width - range_tolerance
            <= requested_gripper_width
            <= maximum_gripper_width + range_tolerance
        ):
            raise ValueError(
                "delta_gripper_width_m would place the gripper outside its "
                f"physical width range [{minimum_gripper_width:.6f}, "
                f"{maximum_gripper_width:.6f}] metres"
            )

        # A nonzero public delta is relative to measured jaw width.  If contact
        # has left the measured width behind a stronger existing actuator
        # target, never reverse that target for a command in the same direction.
        if command.delta_gripper_width_m > 0.0:
            commanded_gripper_width = max(
                current_gripper_target, requested_gripper_width
            )
        elif command.delta_gripper_width_m < 0.0:
            commanded_gripper_width = min(
                current_gripper_target, requested_gripper_width
            )
        else:
            commanded_gripper_width = current_gripper_target
        _set_gripper_internal_target_width_m(robot, commanded_gripper_width)

        def correction_action(gripper_value: float) -> tuple[np.ndarray, float, float]:
            controller.update(force=True)
            position_base, rotation_base = _controller_pose_in_base(
                controller,
                rotation_world_from_base,
                position_world_from_base,
            )
            position_error_base = target_position_base - position_base
            rotation_error_base = matrix_to_rotation_vector(
                target_rotation_base @ rotation_base.T
            )
            position_substep_base = limit_vector_norm(
                position_error_base, self.config.max_translation_substep_m
            )
            rotation_substep_base = limit_vector_norm(
                rotation_error_base, self.config.max_rotation_substep_rad
            )
            action = normalized_osc_action(
                rotation_world_from_base @ position_substep_base,
                rotation_world_from_base @ rotation_substep_base,
                gripper_value,
                self.config,
            )
            return (
                action,
                float(np.linalg.norm(position_error_base)),
                float(np.linalg.norm(rotation_error_base)),
            )

        while motion_steps < self.config.max_motion_control_steps:
            raw_action, position_error, rotation_error = correction_action(0.0)
            if (
                position_error <= self.config.position_tolerance_m
                and rotation_error <= self.config.orientation_tolerance_rad
            ):
                break
            raw_obs = self._step(raw_action)
            control_steps += 1
            motion_steps += 1

        # A true no-op still advances one policy cycle so that osc_step always
        # has causal observation semantics.
        if raw_obs is None:
            raw_action, _position_error, _rotation_error = correction_action(0.0)
            raw_obs = self._step(raw_action)
            control_steps += 1

        settle_steps = 0
        for _ in range(self.config.post_action_settle_steps):
            raw_action, _position_error, _rotation_error = correction_action(0.0)
            raw_obs = self._step(raw_action)
            control_steps += 1
            settle_steps += 1

        controller.update(force=True)
        final_position_base, final_rotation_base = _controller_pose_in_base(
            controller,
            rotation_world_from_base,
            position_world_from_base,
        )
        position_error = float(np.linalg.norm(target_position_base - final_position_base))
        orientation_error = float(
            np.linalg.norm(
                matrix_to_rotation_vector(target_rotation_base @ final_rotation_base.T)
            )
        )
        target_reached = (
            position_error <= self.config.position_tolerance_m
            and orientation_error <= self.config.orientation_tolerance_rad
        )
        gripper_command_completed = _gripper_internal_target_width_reached(
            robot,
            commanded_gripper_width,
            self.config.gripper_target_tolerance_m,
        )
        command_completed = target_reached and gripper_command_completed
        if command_completed:
            termination_reason = "command_completed"
        elif not target_reached and motion_steps >= self.config.max_motion_control_steps:
            termination_reason = "motion_control_step_budget_exhausted"
        elif not gripper_command_completed:
            termination_reason = "gripper_control_step_budget_exhausted"
        else:
            # The motion loop reached tolerance but the post-action settling
            # cycles moved the measured pose back outside it.
            termination_reason = "post_settle_pose_error"

        execution = EEFExecution(
            command_completed=command_completed,
            target_reached=target_reached,
            gripper_command_completed=gripper_command_completed,
            termination_reason=termination_reason,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            control_steps=control_steps,
            motion_control_steps=motion_steps,
            gripper_control_steps=gripper_steps,
            settle_control_steps=settle_steps,
            requested_gripper_width_m=requested_gripper_width,
            commanded_gripper_width_m=_gripper_internal_target_width_m(robot),
            actual_gripper_width_m=_gripper_actual_width_m(robot),
        )
        return raw_obs, execution

    def _step(self, action: np.ndarray) -> dict[str, Any]:
        observation, _reward, _done, _info = self.env.step(action)
        if self.control_step_callback is not None:
            self.control_step_callback(observation)
        return observation


def normalized_osc_action(
    delta_position_world_m: Sequence[float],
    delta_rotation_world_rotvec_rad: Sequence[float],
    gripper_scalar: float,
    config: OSCControlConfig,
) -> np.ndarray:
    """Convert a bounded physical substep to LIBERO's normalized 7D action."""

    position = _finite_vector(delta_position_world_m, "delta_position_world_m")
    rotation = _finite_vector(
        delta_rotation_world_rotvec_rad, "delta_rotation_world_rotvec_rad"
    )
    action = np.concatenate(
        (
            position / config.native_translation_scale_m,
            rotation / config.native_rotation_scale_rad,
            np.asarray([float(gripper_scalar)], dtype=np.float64),
        )
    )
    # This assertion catches adapter bugs before robosuite silently clips them.
    if np.any(np.abs(action) > 1.0 + 1.0e-9):
        raise ValueError(f"internal OSC substep exceeds normalized range: {action}")
    return np.clip(action, -1.0, 1.0)


def limit_vector_norm(vector: Sequence[float], maximum_norm: float) -> np.ndarray:
    vector = _finite_vector(vector, "vector")
    norm = float(np.linalg.norm(vector))
    if norm <= maximum_norm or norm == 0.0:
        return vector.copy()
    return vector * (maximum_norm / norm)


def rotation_vector_to_matrix(rotation_vector: Sequence[float]) -> np.ndarray:
    """Rodrigues conversion for a rotation vector."""

    vector = _finite_vector(rotation_vector, "rotation_vector")
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.eye(3)
    axis = vector / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_rotation_vector(rotation_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Stable matrix-to-rotation-vector conversion with a pi-angle branch."""

    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation_matrix must be a finite 3x3 matrix")
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1.0e-9:
        return 0.5 * np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
    if np.pi - angle < 1.0e-6:
        diagonal = np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] < 1.0e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            if largest == 0:
                axis[1] = np.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
                axis[2] = np.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
            elif largest == 1:
                axis[0] = np.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
                axis[2] = np.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
            else:
                axis[0] = np.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
                axis[1] = np.copysign(axis[1], matrix[1, 2] + matrix[2, 1])
            axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError("quaternion norm must be nonzero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quaternion_xyzw(rotation_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a rotation matrix to a normalized xyzw quaternion."""

    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation_matrix must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion


def _controller_pose_in_base(
    controller: Any,
    rotation_world_from_base: np.ndarray,
    position_world_from_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_base_from_world = rotation_world_from_base.T
    position_base = rotation_base_from_world @ (
        np.asarray(controller.ee_pos, dtype=np.float64) - position_world_from_base
    )
    rotation_base = rotation_base_from_world @ np.asarray(
        controller.ee_ori_mat, dtype=np.float64
    )
    return position_base, rotation_base


def _gripper_actual_width_m(robot: Any) -> float:
    indexes = list(robot._ref_gripper_joint_pos_indexes)
    if len(indexes) != 2:
        raise ValueError("delta gripper-width control requires a two-finger gripper")
    positions = np.asarray(
        [robot.sim.data.qpos[index] for index in indexes], dtype=np.float64
    )
    return float(positions[0] - positions[1])


def _gripper_actuator_control_ranges(robot: Any) -> np.ndarray:
    actuator_indexes = [
        robot.sim.model.actuator_name2id(name) for name in robot.gripper.actuators
    ]
    ranges = np.asarray(
        robot.sim.model.actuator_ctrlrange[actuator_indexes], dtype=np.float64
    )
    if ranges.shape != (2, 2):
        raise ValueError("delta gripper-width control requires two position actuators")
    return ranges


def _gripper_width_limits_m(robot: Any) -> tuple[float, float]:
    ranges = _gripper_actuator_control_ranges(robot)
    minimum = float(ranges[0, 0] - ranges[1, 1])
    maximum = float(ranges[0, 1] - ranges[1, 0])
    return minimum, maximum


def _gripper_internal_target_width_m(robot: Any) -> float:
    current_action = np.asarray(robot.gripper.current_action, dtype=np.float64)
    if current_action.shape != (2,):
        raise ValueError("Panda gripper internal action must contain two targets")
    ranges = _gripper_actuator_control_ranges(robot)
    bias = 0.5 * (ranges[:, 1] + ranges[:, 0])
    weight = 0.5 * (ranges[:, 1] - ranges[:, 0])
    targets = bias + weight * current_action
    return float(targets[0] - targets[1])


def _gripper_internal_target_width_reached(
    robot: Any, target_width_m: float, tolerance_m: float
) -> bool:
    return bool(
        abs(_gripper_internal_target_width_m(robot) - target_width_m) <= tolerance_m
    )


def _set_gripper_internal_target_width_m(robot: Any, target_width_m: float) -> None:
    """Set a Panda position-actuator target without writing physical qpos.

    PandaGripper's public scalar is sign-only and is integrated once per MuJoCo
    control substep, so one env.step() cannot represent an arbitrary metric
    delta.  Updating its persistent normalized target preserves the ordinary
    robosuite actuator path: subsequent zero gripper actions retain the target,
    and MuJoCo alone determines motion, contact blocking, and applied force.
    """

    minimum, maximum = _gripper_width_limits_m(robot)
    if maximum <= minimum:
        raise ValueError("invalid gripper actuator width range")
    fraction = (float(target_width_m) - minimum) / (maximum - minimum)
    normalized = 2.0 * fraction - 1.0
    robot.gripper.current_action = np.asarray(
        [normalized, -normalized], dtype=np.float64
    )


def _finite_vector(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return array


def _finite_scalar(value: float, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite number")
    return scalar
