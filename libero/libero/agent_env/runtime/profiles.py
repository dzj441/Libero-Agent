"""Observation-profile definitions and strict public projection.

The projector is deliberately an allowlist.  Adding a value to the internal
master observation never makes it public until it is explicitly selected here.
"""

from __future__ import annotations

from copy import deepcopy
from enum import IntEnum
from typing import Any, Mapping

import numpy as np

COORDINATE_CONVENTION_FIELDS = (
    "robot_state_frame",
    "eef_delta_frame",
    "eef_rotation_delta",
    "quaternion_order",
    "camera_frame",
    "camera_extrinsic",
    "length_unit",
)
STATE_FIELDS = (
    "arm_joint_position_rad_7d",
    "gripper_finger_joint_position_m_2d",
    "gripper_width_m",
    "eef_pose_robot_base_xyzw_7d",
)
PROPRIOCEPTION_FIELDS = (
    "arm_joint_velocity_rad_s_7d",
    "gripper_finger_joint_velocity_m_s_2d",
    "gripper_width_velocity_m_s",
    "commanded_arm_joint_torque_nm_7d",
    "commanded_arm_joint_torque_available",
    "eef_force_sensor_n_3d",
    "eef_torque_sensor_nm_3d",
    "eef_twist_robot_base_6d",
    "force_torque_frame",
)

# A task reference specifies the desired arrangement of task objects.  Robot
# pose is deliberately excluded from the goal so that a visible robot in the
# reference image cannot be mistaken for part of the target state.
TASK_REFERENCE_SEMANTICS = "desired_object_arrangement"
# v2 artifacts written before the visual-goal terminology was tightened use
# this value.  Keep it readable during archive re-materialization, while new
# observations always emit TASK_REFERENCE_SEMANTICS above.
LEGACY_TASK_REFERENCE_SEMANTICS = "desired_final_state"
_ACCEPTED_TASK_REFERENCE_SEMANTICS = frozenset(
    (TASK_REFERENCE_SEMANTICS, LEGACY_TASK_REFERENCE_SEMANTICS)
)


def validate_task_reference_rgb(value: Any) -> None:
    """Validate a task-reference image without silently changing its type."""

    if not isinstance(value, np.ndarray):
        raise TypeError("task_reference.rgb must be a numpy.ndarray")
    if value.dtype != np.dtype(np.uint8):
        raise TypeError("task_reference.rgb must have dtype uint8")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("task_reference.rgb must have HWC shape with 3 channels")
    if any(int(size) <= 0 for size in value.shape):
        raise ValueError("task_reference.rgb dimensions must be positive")


def validate_task_reference(value: Any) -> None:
    """Validate the public task-reference contract, including v2 archives."""

    if not isinstance(value, Mapping):
        raise TypeError("task_reference must be a mapping")
    if set(value) != {"semantics", "rgb"}:
        raise ValueError("task_reference only allows semantics and rgb")
    if value["semantics"] not in _ACCEPTED_TASK_REFERENCE_SEMANTICS:
        raise ValueError(
            "task_reference.semantics must be one of "
            "'desired_object_arrangement' or 'desired_final_state'"
        )
    validate_task_reference_rgb(value["rgb"])


class ObservationProfile(IntEnum):
    """Agent-visible observation levels in the current three-level contract."""

    LEVEL1 = 1
    LEVEL2 = 2
    LEVEL3 = 3

    @classmethod
    def parse(cls, value: "ObservationProfile | int | str") -> "ObservationProfile":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        normalized = str(value).strip().lower().replace("_", "")
        aliases = {
            "1": cls.LEVEL1,
            "l1": cls.LEVEL1,
            "level1": cls.LEVEL1,
            "2": cls.LEVEL2,
            "l2": cls.LEVEL2,
            "level2": cls.LEVEL2,
            "3": cls.LEVEL3,
            "l3": cls.LEVEL3,
            "level3": cls.LEVEL3,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown observation profile: {value!r}") from exc

    @property
    def public_name(self) -> str:
        return f"level{int(self)}"


def profile_capabilities(profile: ObservationProfile | int | str) -> dict[str, Any]:
    """Return declarative, non-task-specific capabilities for a profile."""

    profile = ObservationProfile.parse(profile)
    return {
        "head_rgb": True,
        "wrist_rgb": True,
        "kinematic_state": True,
        "dynamic_proprioception": profile >= ObservationProfile.LEVEL2,
        "metric_depth": profile >= ObservationProfile.LEVEL3,
        "camera_calibration": profile >= ObservationProfile.LEVEL3,
    }


def project_public_observation(
    master: Mapping[str, Any],
    profile: ObservationProfile | int | str,
) -> dict[str, Any]:
    """Project an internal master frame through a strict public allowlist.

    ``master`` is host-private and may grow over time.  This function copies
    only the fields defined by the public contract.  In particular, reward,
    checker state, object poses, raw segmentation ids, and task metadata are
    never forwarded implicitly.
    """

    profile = ObservationProfile.parse(profile)
    required = {
        "observation_id",
        "frame_index",
        "sim_time_s",
        "state",
        "cameras",
    }
    missing = required.difference(master)
    if missing:
        raise KeyError(f"master observation missing required fields: {sorted(missing)}")

    public: dict[str, Any] = {
        "schema_version": "libero.agent_observation.v2",
        "observation_id": str(master["observation_id"]),
        "frame_index": int(master["frame_index"]),
        "sim_time_s": float(master["sim_time_s"]),
        "profile": profile.public_name,
        "capabilities": profile_capabilities(profile),
        "coordinate_conventions": _copy_required_fields(
            master["coordinate_conventions"], COORDINATE_CONVENTION_FIELDS
        ),
        "state": _copy_required_fields(master["state"], STATE_FIELDS),
        "cameras": {},
    }

    if profile >= ObservationProfile.LEVEL2:
        if "proprioception" not in master:
            raise KeyError("Level 2+ master observation is missing proprioception")
        public["proprioception"] = _copy_required_fields(
            master["proprioception"], PROPRIOCEPTION_FIELDS
        )

    if "task_reference" in master:
        validate_task_reference(master["task_reference"])
        task_reference = master["task_reference"]
        public["task_reference"] = {
            "semantics": task_reference["semantics"],
            "rgb": np.array(task_reference["rgb"], copy=True),
        }

    for public_camera_name in ("head", "wrist"):
        source_camera = master["cameras"][public_camera_name]
        public_camera: dict[str, Any] = {"rgb": deepcopy(source_camera["rgb"])}
        if profile >= ObservationProfile.LEVEL3:
            public_camera.update(
                {
                    "depth_m": deepcopy(source_camera["depth_m"]),
                    "depth_valid_mask": deepcopy(source_camera["depth_valid_mask"]),
                    "intrinsic_matrix_3x3": deepcopy(
                        source_camera["intrinsic_matrix_3x3"]
                    ),
                    "matrix_T_robot_base_from_camera_opencv_4x4": deepcopy(
                        source_camera[
                            "matrix_T_robot_base_from_camera_opencv_4x4"
                        ]
                    ),
                }
            )
        public["cameras"][public_camera_name] = public_camera

    return public


def _copy_required_fields(
    source: Mapping[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    missing = set(fields).difference(source)
    if missing:
        raise KeyError(f"source missing public contract fields: {sorted(missing)}")
    return {field: deepcopy(source[field]) for field in fields}
