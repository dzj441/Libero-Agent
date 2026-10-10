"""Capture and project verified LIBERO demonstrations for Agent ICL.

The replay master is evaluator-owned provenance.  The projected bundle is a
strict, ordinary-file view that may be placed in an Agent workspace.  Source
paths, MuJoCo state, checker timing, and raw segmentation metadata never enter
the public bundle.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..runtime.contracts.annotation import (
    TASK_ENTITY_ANNOTATION_SCHEMA_VERSION,
    validate_task_entity_mapping,
)
from ..runtime.artifacts import (
    DEPTH_PREVIEW_MAPPING,
    depth_preview_with_metadata,
    write_public_observation,
)
from ..runtime.profiles import (
    COORDINATE_CONVENTION_FIELDS,
    PROPRIOCEPTION_FIELDS,
    STATE_FIELDS,
    ObservationProfile,
    profile_capabilities,
    project_public_observation,
)


P4_MASTER_SCHEMA_VERSION = "libero.fixed_demo_p4_master.v2"
FIXED_DEMO_BUNDLE_SCHEMA_VERSION = "libero.fixed_demo_bundle.v2"
SOURCE_ACTION_SCHEMA_VERSION = "libero.normalized_osc_pose_action.v1"
MAX_CONTACT_SHEET_FRAMES = 12
CAMERA_NAMES = ("head", "wrist")
LEGACY_DEPTH_FIELDS = frozenset(
    {
        "metric_file",
        "preview_file",
        "valid_mask_file",
        "dtype",
        "shape",
        "unit",
    }
)
PREVIEW_DEPTH_FIELDS = frozenset(
    {
        *LEGACY_DEPTH_FIELDS,
        "preview_near_m",
        "preview_far_m",
        "preview_mapping",
        "preview_range_source",
    }
)
PREVIEW_RANGE_SOURCES = frozenset(
    {
        "inverse_depth_percentile_2_98",
        "finite_min_max_fallback",
        "degenerate",
        "no_valid_depth",
    }
)


class FixedDemoError(ValueError):
    """Raised when a replay master or public bundle violates its contract."""


class P4ReplayMasterRecorder:
    """Atomically record full P4 frames from one physical action replay."""

    def __init__(
        self,
        destination: str | Path,
        episode: Any,
        *,
        camera_height: int,
        camera_width: int,
        initial_frame_after_settle: bool = True,
    ) -> None:
        self.destination = Path(destination).expanduser().resolve()
        if self.destination.exists() or self.destination.is_symlink():
            raise FileExistsError(
                f"P4 replay master destination already exists: {self.destination}"
            )
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{self.destination.name}.capture-",
                dir=self.destination.parent,
            )
        ).resolve()
        self.episode = episode
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.initial_frame_after_settle = bool(initial_frame_after_settle)
        self._collector: Any | None = None
        self._frames: list[dict[str, Any]] = []
        self._published = False

    def capture(
        self,
        env: Any,
        raw_observation: Mapping[str, Any],
        frame_index: int,
        source_action_index: int | None,
    ) -> None:
        """Capture the settled initial frame or one actual post-action frame."""

        if self._published:
            raise RuntimeError("P4 replay master has already been published")
        if frame_index != len(self._frames):
            raise FixedDemoError(
                f"non-contiguous replay frame index {frame_index}; "
                f"expected {len(self._frames)}"
            )
        expected_action = None if frame_index == 0 else frame_index - 1
        if source_action_index != expected_action:
            raise FixedDemoError(
                f"frame {frame_index} has source action {source_action_index}; "
                f"expected {expected_action}"
            )
        if self._collector is None:
            # Lazy import keeps the non-rendering launcher from initializing GL.
            from .observation import MasterObservationCollector

            self._collector = MasterObservationCollector(
                env,
                camera_height=self.camera_height,
                camera_width=self.camera_width,
            )

        master = self._collector.collect(raw_observation, frame_index)
        frame_id = f"frame_{frame_index:06d}"
        master["observation_id"] = frame_id
        public = project_public_observation(master, ObservationProfile.LEVEL3)
        frame_relative = Path("frames") / frame_id
        frame_root = self._temporary / frame_relative
        observation_path = write_public_observation(public, frame_root)
        files = _tree_inventory(self._temporary, frame_relative)
        self._frames.append(
            {
                "frame_index": frame_index,
                "observation_id": frame_id,
                "observation": os.fspath(observation_path.relative_to(self._temporary)),
                "source_action_index": source_action_index,
                "files": files,
            }
        )

    def finalize(self, replay_report: Mapping[str, Any]) -> dict[str, Any]:
        """Authenticate, validate, and atomically publish the completed master."""

        if self._published:
            raise RuntimeError("P4 replay master has already been published")
        if replay_report.get("verified_success") is not True:
            raise FixedDemoError("refusing to publish an unverified demonstration")
        actions = np.asarray(self.episode.actions, dtype=np.float64)
        if len(self._frames) != len(actions) + 1:
            raise FixedDemoError(
                f"captured {len(self._frames)} frames for {len(actions)} actions; "
                "expected one initial frame plus one post-action frame per action"
            )

        trajectory_path = self._temporary / "source_trajectory.jsonl"
        transition_records = [
            _source_transition_record(index, action)
            for index, action in enumerate(actions)
        ]
        _write_jsonl(trajectory_path, transition_records)
        init_state = np.asarray(self.episode.init_state, dtype=np.float64)
        manifest = {
            "schema_version": P4_MASTER_SCHEMA_VERSION,
            "task": {
                "instruction": _normalize_instruction(self.episode.task_instruction),
                "problem_name": str(self.episode.problem_name),
                "environment_name": str(self.episode.env_name),
            },
            "source": {
                "dataset_path": os.fspath(self.episode.dataset_path),
                "dataset_sha256": file_sha256(Path(self.episode.dataset_path)),
                "demo_key": str(self.episode.demo_key),
                "bddl_file": os.fspath(self.episode.bddl_file),
                "init_state_source": str(self.episode.init_state_source),
                "init_state_size": int(init_state.size),
                "init_state_sha256": _array_sha256(init_state),
            },
            "capture": {
                "profile": ObservationProfile.LEVEL3.public_name,
                "camera_height": self.camera_height,
                "camera_width": self.camera_width,
                "frame_count": len(self._frames),
                "transition_count": len(actions),
                "initial_frame_after_settle": self.initial_frame_after_settle,
                "post_action_frames_are_causal": True,
                "frames": self._frames,
                "trajectory": _artifact_record(
                    trajectory_path, self._temporary, "application/x-ndjson"
                ),
            },
            "action_semantics": source_action_semantics(),
            "verification": {
                "verified_success": True,
                "final_success": bool(replay_report.get("final_success")),
                "first_success_step": replay_report.get("first_success_step"),
                "final_success_streak": int(
                    replay_report.get("final_success_streak", 0)
                ),
                "required_stable_success_steps": int(
                    replay_report.get("required_stable_success_steps", 0)
                ),
            },
            "visibility_contract": {
                "public_episode_outcome": "verified successful demonstration",
                "source_paths": False,
                "mujoco_state": False,
                "stepwise_checker": False,
                "first_success_step": False,
                "raw_segmentation_ids": False,
                "object_ground_truth_pose": False,
            },
        }
        _write_json(self._temporary / "p4_master_manifest.json", manifest)
        validate_p4_replay_master(self._temporary)
        os.replace(self._temporary, self.destination)
        self._published = True
        return {
            "schema_version": "libero.fixed_demo_master_receipt.v1",
            "master": os.fspath(self.destination),
            "manifest_sha256": file_sha256(
                self.destination / "p4_master_manifest.json"
            ),
            "frame_count": len(self._frames),
            "transition_count": len(actions),
        }

    def abort(self) -> None:
        if not self._published and self._temporary.exists():
            shutil.rmtree(self._temporary)


def source_action_semantics() -> dict[str, Any]:
    """Describe the recorded native controller action without implying ctl parity."""

    return {
        "schema_version": SOURCE_ACTION_SCHEMA_VERSION,
        "representation": "normalized_libero_osc_pose_7d",
        "component_order": [
            "normalized_delta_position_x",
            "normalized_delta_position_y",
            "normalized_delta_position_z",
            "normalized_delta_rotation_vector_x",
            "normalized_delta_rotation_vector_y",
            "normalized_delta_rotation_vector_z",
            "normalized_gripper_command",
        ],
        "normalized_range": [-1.0, 1.0],
        "controller_output_scale": {
            "translation_m_per_unit": 0.05,
            "rotation_vector_rad_per_unit": 0.5,
        },
        "gripper_command": {
            "minus_one": "open",
            "plus_one": "close",
            "metric_width_semantics": False,
        },
        "liberoctl_step_compatible": False,
        "note": (
            "These are the source episode's native per-control-cycle OSC inputs, "
            "not high-level metric liberoctl commands."
        ),
    }


def project_fixed_demo_bundle(
    *,
    master_root: str | Path,
    destination: str | Path,
    profile: ObservationProfile | int | str,
    expected_task_instruction: str,
) -> dict[str, Any]:
    """Project one authenticated P4 replay master into an Agent-visible bundle."""

    master_root = Path(master_root).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    profile = ObservationProfile.parse(profile)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"fixed-demo destination already exists: {destination}")
    manifest = validate_p4_replay_master(master_root)
    source_instruction = _normalize_instruction(manifest["task"]["instruction"])
    expected_instruction = _normalize_instruction(expected_task_instruction)
    if source_instruction != expected_instruction:
        raise FixedDemoError(
            "fixed demonstration task instruction does not match the target task"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.projection-", dir=destination.parent)
    ).resolve()
    try:
        public_frames: list[dict[str, Any]] = []
        overview_sources: dict[str, list[tuple[str, Path]]] = {
            "head_rgb": [],
            "wrist_rgb": [],
        }
        if profile >= ObservationProfile.LEVEL3:
            overview_sources.update({"head_depth": [], "wrist_depth": []})

        for frame in manifest["capture"]["frames"]:
            frame_index = int(frame["frame_index"])
            frame_id = f"frame_{frame_index:06d}"
            source_frame_root = master_root / "frames" / frame_id
            target_frame_root = temporary / "frames" / frame_id
            projected = _project_materialized_frame(
                source_frame_root, target_frame_root, profile
            )
            observation_relative = Path("frames") / frame_id / "observation.json"
            public_frames.append(
                {
                    "frame_index": frame_index,
                    "observation_id": frame_id,
                    "observation": _artifact_record(
                        temporary / observation_relative,
                        temporary,
                        "application/json",
                    ),
                }
            )
            overview_sources["head_rgb"].append(
                (frame_id, target_frame_root / projected["cameras"]["head"]["rgb"]["file"])
            )
            overview_sources["wrist_rgb"].append(
                (frame_id, target_frame_root / projected["cameras"]["wrist"]["rgb"]["file"])
            )
            if profile >= ObservationProfile.LEVEL3:
                overview_sources["head_depth"].append(
                    (
                        frame_id,
                        target_frame_root
                        / projected["cameras"]["head"]["depth"]["preview_file"],
                    )
                )
                overview_sources["wrist_depth"].append(
                    (
                        frame_id,
                        target_frame_root
                        / projected["cameras"]["wrist"]["depth"]["preview_file"],
                    )
                )

        source_trajectory = _read_jsonl(
            master_root / manifest["capture"]["trajectory"]["path"]
        )
        trajectory_records = [_public_transition_record(record) for record in source_trajectory]
        trajectory_path = temporary / "trajectory.jsonl"
        _write_jsonl(trajectory_path, trajectory_records)

        sampled = contact_sheet_indices(len(public_frames))
        contact_sheets: dict[str, Any] = {}
        for name, sources in overview_sources.items():
            selected = [sources[index] for index in sampled]
            sheet_path = temporary / "overview" / "contact_sheets" / f"{name}.png"
            _build_contact_sheet(selected, sheet_path)
            contact_sheets[name] = _artifact_record(
                sheet_path, temporary, "image/png"
            )

        public_manifest = {
            "schema_version": FIXED_DEMO_BUNDLE_SCHEMA_VERSION,
            "task_instruction": expected_instruction,
            "icl_condition": "fixed_demo",
            "observation_profile": profile.public_name,
            "capabilities": profile_capabilities(profile),
            "demonstration": {
                "episode_outcome": "verified successful demonstration",
                "relation_to_target": "same_task_separate_episode",
                "scene_or_object_poses_may_differ": True,
                "frame_count": len(public_frames),
                "transition_count": len(trajectory_records),
                "actions_present": True,
                "action_representation": SOURCE_ACTION_SCHEMA_VERSION,
                "stepwise_success_present": False,
            },
            "action_semantics": source_action_semantics(),
            "trajectory": _artifact_record(
                trajectory_path, temporary, "application/x-ndjson"
            ),
            "frames": public_frames,
            "overview": {
                "sampling": {
                    "rule": "uniform_endpoint_preserving_v1",
                    "maximum_frames_per_sheet": MAX_CONTACT_SHEET_FRAMES,
                    "sampled_frame_indices": sampled,
                },
                "contact_sheets": contact_sheets,
            },
        }
        public_manifest["integrity"] = _bundle_integrity(temporary)
        _write_json(temporary / "manifest.json", public_manifest)
        validate_fixed_demo_bundle(
            temporary,
            expected_profile=profile,
            expected_task_instruction=expected_instruction,
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "schema_version": "libero.fixed_demo_projection_receipt.v1",
        "source_master": os.fspath(master_root),
        "agent_bundle": os.fspath(destination),
        "manifest_sha256": file_sha256(destination / "manifest.json"),
        "frame_count": len(public_frames),
        "transition_count": len(trajectory_records),
    }


def validate_p4_replay_master(master_root: str | Path) -> dict[str, Any]:
    """Validate master provenance and every captured P4 frame hash."""

    root = Path(master_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise FixedDemoError(f"P4 replay master is not a real directory: {root}")
    manifest_path = root / "p4_master_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != P4_MASTER_SCHEMA_VERSION:
        raise FixedDemoError("unsupported P4 replay master schema")
    capture = manifest.get("capture")
    if not isinstance(capture, dict) or capture.get("profile") not in {
        "level3",
        "level4",
    }:
        raise FixedDemoError("P4 replay master capture metadata is invalid")
    frames = capture.get("frames")
    frame_count = capture.get("frame_count")
    transition_count = capture.get("transition_count")
    if (
        not isinstance(frames, list)
        or frame_count != len(frames)
        or transition_count != len(frames) - 1
    ):
        raise FixedDemoError("P4 replay master frame counts are inconsistent")
    if manifest.get("verification", {}).get("verified_success") is not True:
        raise FixedDemoError("P4 replay master is not verified successful")
    semantics = manifest.get("action_semantics")
    if semantics != source_action_semantics():
        raise FixedDemoError("P4 replay master action semantics are invalid")

    for index, frame in enumerate(frames):
        frame_id = f"frame_{index:06d}"
        if (
            frame.get("frame_index") != index
            or frame.get("observation_id") != frame_id
            or frame.get("source_action_index") != (None if index == 0 else index - 1)
        ):
            raise FixedDemoError(f"P4 replay master frame {index} is misaligned")
        observation = _inside(root, frame.get("observation"), "master observation")
        _validate_materialized_observation(
            observation.parent,
            expected_profile=str(capture["profile"]),
            expected_frame_index=index,
        )
        files = frame.get("files")
        actual_files = _tree_inventory(root, Path("frames") / frame_id)
        if files != actual_files:
            raise FixedDemoError(f"P4 replay master frame {index} file hashes differ")

    trajectory_artifact = capture.get("trajectory")
    _validate_artifact(trajectory_artifact, root, "master trajectory")
    records = _read_jsonl(root / trajectory_artifact["path"])
    if len(records) != transition_count:
        raise FixedDemoError("P4 replay master transition count differs")
    for index, record in enumerate(records):
        _validate_source_transition(record, index, frame_count)
    return manifest


def validate_fixed_demo_bundle(
    bundle_root: str | Path,
    *,
    expected_profile: ObservationProfile | int | str,
    expected_task_instruction: str,
) -> dict[str, Any]:
    """Validate exact public fields, paths, action alignment, and integrity."""

    root = Path(bundle_root).expanduser().resolve()
    profile = ObservationProfile.parse(expected_profile)
    if not root.is_dir() or root.is_symlink():
        raise FixedDemoError(f"fixed-demo bundle is not a real directory: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise FixedDemoError(f"fixed-demo bundle contains a symlink: {path}")
    manifest = _read_json(root / "manifest.json")
    expected_keys = {
        "schema_version",
        "task_instruction",
        "icl_condition",
        "observation_profile",
        "capabilities",
        "demonstration",
        "action_semantics",
        "trajectory",
        "frames",
        "overview",
        "integrity",
    }
    if set(manifest) != expected_keys:
        raise FixedDemoError("fixed-demo manifest fields do not match the allowlist")
    if (
        manifest["schema_version"] != FIXED_DEMO_BUNDLE_SCHEMA_VERSION
        or manifest["task_instruction"] != _normalize_instruction(expected_task_instruction)
        or manifest["icl_condition"] != "fixed_demo"
        or manifest["observation_profile"] != profile.public_name
        or manifest["capabilities"] != profile_capabilities(profile)
        or manifest["action_semantics"] != source_action_semantics()
    ):
        raise FixedDemoError("fixed-demo manifest metadata does not match")
    demonstration = manifest["demonstration"]
    if demonstration != {
        "episode_outcome": "verified successful demonstration",
        "relation_to_target": "same_task_separate_episode",
        "scene_or_object_poses_may_differ": True,
        "frame_count": demonstration.get("frame_count"),
        "transition_count": demonstration.get("transition_count"),
        "actions_present": True,
        "action_representation": SOURCE_ACTION_SCHEMA_VERSION,
        "stepwise_success_present": False,
    }:
        raise FixedDemoError("fixed-demo public outcome contract is invalid")
    frames = manifest["frames"]
    frame_count = demonstration["frame_count"]
    transition_count = demonstration["transition_count"]
    if (
        not isinstance(frames, list)
        or frame_count != len(frames)
        or transition_count != frame_count - 1
    ):
        raise FixedDemoError("fixed-demo frame counts are inconsistent")
    for index, frame in enumerate(frames):
        frame_id = f"frame_{index:06d}"
        if set(frame) != {"frame_index", "observation_id", "observation"}:
            raise FixedDemoError("fixed-demo frame record has unexpected fields")
        if frame["frame_index"] != index or frame["observation_id"] != frame_id:
            raise FixedDemoError("fixed-demo frame indices are not contiguous")
        observation = _validate_artifact(
            frame["observation"], root, f"frame {index} observation"
        )
        _validate_materialized_observation(
            observation.parent,
            expected_profile=profile,
            expected_frame_index=index,
        )

    trajectory = _validate_artifact(manifest["trajectory"], root, "trajectory")
    records = _read_jsonl(trajectory)
    if len(records) != transition_count:
        raise FixedDemoError("fixed-demo trajectory count differs")
    for index, record in enumerate(records):
        _validate_source_transition(record, index, frame_count)
    if manifest["integrity"] != _bundle_integrity(root):
        raise FixedDemoError("fixed-demo bundle integrity differs")
    return manifest


def contact_sheet_indices(
    frame_count: int, limit: int = MAX_CONTACT_SHEET_FRAMES
) -> list[int]:
    if frame_count < 1 or limit < 1:
        raise ValueError("frame_count and contact-sheet limit must be positive")
    if frame_count <= limit:
        return list(range(frame_count))
    return [round(index * (frame_count - 1) / (limit - 1)) for index in range(limit)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_transition_record(index: int, action: Sequence[float]) -> dict[str, Any]:
    vector = np.asarray(action, dtype=np.float64)
    if vector.shape != (7,) or not np.isfinite(vector).all():
        raise FixedDemoError(f"source action {index} is not a finite 7D vector")
    if np.any(np.abs(vector) > 1.0 + 1.0e-9):
        raise FixedDemoError(f"source action {index} exceeds the normalized range")
    return {
        "transition_index": index,
        "observation_before": f"frames/frame_{index:06d}/observation.json",
        "source_action": {
            "schema_version": SOURCE_ACTION_SCHEMA_VERSION,
            "normalized_vector_7d": vector.tolist(),
        },
        "observation_after": f"frames/frame_{index + 1:06d}/observation.json",
    }


def _public_transition_record(record: Mapping[str, Any]) -> dict[str, Any]:
    index = int(record.get("transition_index"))
    vector = record.get("source_action", {}).get("normalized_vector_7d")
    return _source_transition_record(index, vector)


def _validate_source_transition(
    record: Mapping[str, Any], index: int, frame_count: int
) -> None:
    if index >= frame_count - 1 or record != _public_transition_record(record):
        raise FixedDemoError(f"source transition {index} violates the public schema")
    if record["transition_index"] != index:
        raise FixedDemoError("source transition indices are not contiguous")


def _project_materialized_frame(
    source_root: Path, destination_root: Path, profile: ObservationProfile
) -> dict[str, Any]:
    source = _validate_materialized_observation(
        source_root,
        expected_profile=_read_json(source_root / "observation.json")["profile"],
        expected_frame_index=int(source_root.name.split("_")[-1]),
    )
    destination_root.mkdir(parents=True)
    projected: dict[str, Any] = {
        "schema_version": source["schema_version"],
        "observation_id": source["observation_id"],
        "frame_index": source["frame_index"],
        "sim_time_s": source["sim_time_s"],
        "profile": profile.public_name,
        "capabilities": profile_capabilities(profile),
        "coordinate_conventions": {
            field: deepcopy(source["coordinate_conventions"][field])
            for field in COORDINATE_CONVENTION_FIELDS
        },
        "state": deepcopy(source["state"]),
        "cameras": {},
    }
    if profile >= ObservationProfile.LEVEL2:
        projected["proprioception"] = deepcopy(source["proprioception"])
    for camera_name in CAMERA_NAMES:
        source_camera = source["cameras"][camera_name]
        camera: dict[str, Any] = {
            "rgb": deepcopy(source_camera["rgb"]),
        }
        _copy_referenced_file(source_root, destination_root, camera["rgb"]["file"])
        if profile >= ObservationProfile.LEVEL3:
            camera.update(
                {
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
            depth = deepcopy(source_camera["depth"])
            for field in ("metric_file", "valid_mask_file"):
                _copy_referenced_file(
                    source_root, destination_root, depth[field]
                )
            source_depth = np.load(
                source_root / depth["metric_file"], allow_pickle=False
            )
            with Image.open(source_root / depth["valid_mask_file"]) as mask_image:
                source_valid = np.asarray(mask_image) > 0
            preview, preview_metadata = depth_preview_with_metadata(
                source_depth, source_valid
            )
            preview_path = destination_root / depth["preview_file"]
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(preview).save(preview_path)
            depth.update(preview_metadata)
            camera["depth"] = depth
        projected["cameras"][camera_name] = camera
    _write_json(destination_root / "observation.json", projected)
    _validate_materialized_observation(
        destination_root,
        expected_profile=profile,
        expected_frame_index=source["frame_index"],
    )
    return projected


def _validate_materialized_observation(
    frame_root: Path,
    *,
    expected_profile: ObservationProfile | str,
    expected_frame_index: int,
) -> dict[str, Any]:
    observation = _read_json(frame_root / "observation.json")
    legacy_level4 = expected_profile == "level4"
    profile = (
        ObservationProfile.LEVEL3
        if legacy_level4
        else ObservationProfile.parse(expected_profile)
    )
    expected_keys = {
        "schema_version",
        "observation_id",
        "frame_index",
        "sim_time_s",
        "profile",
        "capabilities",
        "coordinate_conventions",
        "state",
        "cameras",
    }
    if legacy_level4 or profile >= ObservationProfile.LEVEL2:
        expected_keys.add("proprioception")
    if legacy_level4 and expected_frame_index == 0:
        expected_keys.add("annotations")
    if set(observation) != expected_keys:
        raise FixedDemoError("materialized observation fields do not match the profile")
    frame_id = f"frame_{expected_frame_index:06d}"
    if (
        observation["schema_version"] != "libero.agent_observation.v2"
        or observation["observation_id"] != frame_id
        or observation["frame_index"] != expected_frame_index
        or observation["profile"] != ("level4" if legacy_level4 else profile.public_name)
        or observation["capabilities"]
        != (_legacy_level4_capabilities() if legacy_level4 else profile_capabilities(profile))
        or set(observation["coordinate_conventions"])
        != (
            set(COORDINATE_CONVENTION_FIELDS) | {"bbox_xyxy"}
            if legacy_level4
            else set(COORDINATE_CONVENTION_FIELDS)
        )
        or set(observation["cameras"]) != set(CAMERA_NAMES)
    ):
        raise FixedDemoError("materialized observation metadata is invalid")
    if set(observation["state"]) != set(STATE_FIELDS):
        raise FixedDemoError("materialized robot-state fields are invalid")
    if (legacy_level4 or profile >= ObservationProfile.LEVEL2) and set(
        observation["proprioception"]
    ) != set(PROPRIOCEPTION_FIELDS):
        raise FixedDemoError("materialized proprioception fields are invalid")
    for camera_name in CAMERA_NAMES:
        camera = observation["cameras"][camera_name]
        expected_camera_keys = {"rgb"}
        if legacy_level4 or profile >= ObservationProfile.LEVEL3:
            expected_camera_keys.update(
                {
                    "depth",
                    "intrinsic_matrix_3x3",
                    "matrix_T_robot_base_from_camera_opencv_4x4",
                }
            )
        if set(camera) != expected_camera_keys:
            raise FixedDemoError("materialized camera fields are invalid")
        _validate_artifact_file_field(camera["rgb"], "file", frame_root)
        if legacy_level4 or profile >= ObservationProfile.LEVEL3:
            depth = camera["depth"]
            if set(depth) == LEGACY_DEPTH_FIELDS:
                pass
            elif set(depth) == PREVIEW_DEPTH_FIELDS:
                _validate_depth_preview_metadata(depth)
            else:
                raise FixedDemoError("materialized depth fields are invalid")
            for field in ("metric_file", "preview_file", "valid_mask_file"):
                _inside(frame_root, depth[field], f"{camera_name} {field}")
    if "annotations" in observation:
        annotations = observation["annotations"]
        if (
            set(annotations) != {"schema_version", "schedule", "cameras"}
            or annotations["schema_version"]
            != TASK_ENTITY_ANNOTATION_SCHEMA_VERSION
            or annotations["schedule"] != "initial_observation_only"
            or set(annotations["cameras"]) != set(CAMERA_NAMES)
        ):
            raise FixedDemoError("materialized annotation metadata is invalid")
        expected_entity_ids: tuple[str, ...] | None = None
        for camera_name in CAMERA_NAMES:
            camera_annotations = annotations["cameras"][camera_name]
            if set(camera_annotations) != {"task_entities", "overlay_file"}:
                raise FixedDemoError("materialized task-entity fields are invalid")
            _inside(frame_root, camera_annotations["overlay_file"], "annotation overlay")
            task_entities = camera_annotations["task_entities"]
            try:
                entity_ids = validate_task_entity_mapping(task_entities)
            except ValueError as exc:
                raise FixedDemoError(str(exc)) from exc
            if expected_entity_ids is None:
                expected_entity_ids = entity_ids
            elif entity_ids != expected_entity_ids:
                raise FixedDemoError("task entities differ between cameras")
            for entity_id in entity_ids:
                if set(task_entities[entity_id]) != {
                    "visible",
                    "visible_pixel_count",
                    "bbox_xyxy",
                    "mask_file",
                }:
                    raise FixedDemoError("materialized annotation fields are invalid")
                _inside(frame_root, task_entities[entity_id]["mask_file"], "mask")
    return observation


def _legacy_level4_capabilities() -> dict[str, Any]:
    """Describe archived public P4 masters without re-exposing P4 online."""

    return {
        "head_rgb": True,
        "wrist_rgb": True,
        "kinematic_state": True,
        "initial_task_entity_bbox_and_mask": True,
        "dynamic_proprioception": True,
        "metric_depth": True,
        "camera_calibration": True,
        "annotation_schedule": "initial_observation_only",
    }


def _validate_depth_preview_metadata(depth: Mapping[str, Any]) -> None:
    """Validate the optional depth-preview transfer metadata.

    The six-field legacy depth contract remains accepted above so previously
    captured demonstrations stay readable.  New captures must describe the
    preview mapping and the metric near/far range that was used to render it.
    """

    if depth.get("dtype") != "float32" or depth.get("unit") != "metre":
        raise FixedDemoError("materialized depth base metadata is invalid")
    if depth.get("preview_mapping") != DEPTH_PREVIEW_MAPPING:
        raise FixedDemoError("materialized depth preview mapping is invalid")
    source = depth.get("preview_range_source")
    if source not in PREVIEW_RANGE_SOURCES:
        raise FixedDemoError("materialized depth preview range source is invalid")

    near = depth.get("preview_near_m")
    far = depth.get("preview_far_m")

    def _positive_finite(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
        )

    if source == "no_valid_depth":
        if near is not None or far is not None:
            raise FixedDemoError(
                "no-valid-depth preview must not publish a range"
            )
        return
    if not _positive_finite(near) or not _positive_finite(far):
        raise FixedDemoError("materialized depth preview range is invalid")
    if float(near) > float(far):
        raise FixedDemoError("materialized depth preview near/far order is invalid")
    if source == "degenerate" and not math.isclose(
        float(near), float(far), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise FixedDemoError("degenerate depth preview range is inconsistent")


def _build_contact_sheet(sources: list[tuple[str, Path]], destination: Path) -> None:
    images = [(label, Image.open(path).convert("RGB")) for label, path in sources]
    tile_width = max(image.width for _, image in images)
    tile_height = max(image.height for _, image in images)
    columns = min(4, len(images))
    rows = math.ceil(len(images) / columns)
    label_height = 22
    sheet = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + label_height)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        column = index % columns
        row = index // columns
        x = column * tile_width
        y = row * (tile_height + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 5, y + 4), label, fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def _copy_referenced_file(source_root: Path, destination_root: Path, value: str) -> None:
    source = _inside(source_root, value, "frame artifact")
    relative = Path(value)
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _tree_inventory(root: Path, relative_root: Path) -> list[dict[str, Any]]:
    directory = root / relative_root
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise FixedDemoError(f"asset tree contains a symlink: {path}")
        if path.is_file():
            records.append(
                {
                    "path": os.fspath(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return records


def _bundle_integrity(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FixedDemoError(f"bundle contains a symlink: {path}")
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": os.fspath(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return {"algorithm": "sha256", "files": files}


def _artifact_record(path: Path, root: Path, media_type: str) -> dict[str, Any]:
    path = path.resolve()
    if os.path.commonpath((root, path)) != os.fspath(root):
        raise FixedDemoError("artifact escapes its root")
    return {
        "path": os.fspath(path.relative_to(root)),
        "media_type": media_type,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _validate_artifact(value: Any, root: Path, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "media_type",
        "size_bytes",
        "sha256",
    }:
        raise FixedDemoError(f"{label} artifact fields are invalid")
    path = _inside(root, value["path"], label)
    if path.stat().st_size != value["size_bytes"] or file_sha256(path) != value["sha256"]:
        raise FixedDemoError(f"{label} artifact integrity differs")
    return path


def _validate_artifact_file_field(value: Any, field: str, root: Path) -> Path:
    if not isinstance(value, dict) or field not in value:
        raise FixedDemoError("image artifact metadata is invalid")
    return _inside(root, value[field], "image artifact")


def _inside(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise FixedDemoError(f"{label} path must be non-empty and relative")
    path = (root / value).resolve()
    if os.path.commonpath((root.resolve(), path)) != os.fspath(root.resolve()):
        raise FixedDemoError(f"{label} path escapes its root")
    if not path.is_file() or path.is_symlink():
        raise FixedDemoError(f"{label} file is missing or unsafe: {path}")
    return path


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FixedDemoError(f"JSON object expected: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise FixedDemoError(f"blank JSONL record at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FixedDemoError(f"JSON object expected at {path}:{line_number}")
        records.append(value)
    return records


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _normalize_instruction(value: Any) -> str:
    return " ".join(str(value).split())
