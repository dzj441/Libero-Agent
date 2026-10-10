"""Project a verified replay master as an RGB-video-only ICL bundle.

The full ``fixed_demo`` projection is intentionally rich: it exposes the
public observations, native actions, and all profile-approved camera/state
fields.  This module is the deliberately narrow counterpart used when an
Agent should receive only the two RGB camera streams as a video.  The public
bundle therefore contains no observation JSON, trajectory, depth, calibration,
state, segmentation, or annotation artifact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from .fixed_demo import (
    FixedDemoError,
    contact_sheet_indices,
    file_sha256,
    validate_p4_replay_master,
)


VIDEO_ONLY_BUNDLE_SCHEMA_VERSION = "libero.video_only_demo_bundle.v1"
VIDEO_ONLY_RECEIPT_SCHEMA_VERSION = "libero.video_only_demo_receipt.v1"
VIDEO_ONLY_FPS_HZ = 20.0
VIDEO_ONLY_CONTACT_SHEET_MAX_FRAMES = 12
VIDEO_ONLY_CONTACT_SHEET_GUTTER_PX = 8
VIDEO_ONLY_CONTACT_SHEET_SAMPLING_RULE = "uniform_endpoint_preserving_v1"


def project_video_only_demo_bundle(
    *,
    master_root: str | Path,
    destination: str | Path,
    expected_task_instruction: str,
) -> dict[str, Any]:
    """Publish only head/wrist RGB video from an authenticated replay master.

    The returned receipt is evaluator-private and may retain the source master
    for post-hoc auditing.  ``destination`` is the Agent-visible bundle and is
    intentionally limited to ``manifest.json``, one RGB-only MP4, and two RGB
    contact sheets that remain directly viewable without a video decoder.
    """

    master_root = Path(master_root).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"video-only destination already exists: {destination}")
    master_manifest = validate_p4_replay_master(master_root)
    source_instruction = _normalize_instruction(
        master_manifest.get("task", {}).get("instruction")
    )
    target_instruction = _normalize_instruction(expected_task_instruction)
    if source_instruction != target_instruction:
        raise FixedDemoError(
            "video-only demonstration task instruction does not match the target task"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.projection-", dir=destination.parent
        )
    ).resolve()
    try:
        video_path = temporary / "video" / "head_wrist_rgb.mp4"
        sampled_frame_indices, sampled_rgb = _build_rgb_video(
            master_root=master_root,
            master_manifest=master_manifest,
            destination=video_path,
        )
        contact_sheet_root = temporary / "video" / "contact_sheets"
        contact_sheet_paths = {
            camera_name: contact_sheet_root / f"{camera_name}.png"
            for camera_name in ("head_rgb", "wrist_rgb")
        }
        for camera_name, path in contact_sheet_paths.items():
            _build_contact_sheet(sampled_rgb[camera_name], path)
        frame_count = int(master_manifest["capture"]["frame_count"])
        manifest = {
            "schema_version": VIDEO_ONLY_BUNDLE_SCHEMA_VERSION,
            "task_instruction": target_instruction,
            "icl_condition": "video_only",
            "demonstration": {
                "episode_outcome": "verified successful demonstration",
                "relation_to_target": "same_task_separate_episode",
                "scene_or_object_poses_may_differ": True,
                "modalities": ["head_rgb", "wrist_rgb"],
                "video": {
                    "file": _artifact_record(video_path, temporary),
                    "fps_hz": VIDEO_ONLY_FPS_HZ,
                    "frame_count": int(master_manifest["capture"]["frame_count"]),
                    "layout": "head_rgb_left__wrist_rgb_right",
                    "contact_sheets": {
                        camera_name: _artifact_record(path, temporary, "image/png")
                        for camera_name, path in contact_sheet_paths.items()
                    },
                    "sampling": {
                        "rule": VIDEO_ONLY_CONTACT_SHEET_SAMPLING_RULE,
                        "maximum_frames_per_sheet": VIDEO_ONLY_CONTACT_SHEET_MAX_FRAMES,
                        "source_frame_count": frame_count,
                        "sampled_frame_indices": sampled_frame_indices,
                    },
                },
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        validate_video_only_demo_bundle(
            temporary,
            expected_task_instruction=target_instruction,
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "schema_version": VIDEO_ONLY_RECEIPT_SCHEMA_VERSION,
        "icl_condition": "video_only",
        "source_master": os.fspath(master_root),
        "target_task_instruction": target_instruction,
        "agent_bundle": os.fspath(destination),
        "manifest_sha256": file_sha256(destination / "manifest.json"),
        "frame_count": int(master_manifest["capture"]["frame_count"]),
    }


def validate_video_only_demo_bundle(
    bundle_root: str | Path,
    *,
    expected_task_instruction: str,
) -> dict[str, Any]:
    """Validate the strict Agent-visible RGB-only allowlist."""

    unresolved_root = Path(bundle_root).expanduser()
    if unresolved_root.is_symlink():
        raise FixedDemoError(
            f"video-only bundle root must not be a symlink: {unresolved_root}"
        )
    root = unresolved_root.resolve()
    if not root.is_dir():
        raise FixedDemoError(f"video-only bundle is not a real directory: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise FixedDemoError(f"video-only bundle contains a symlink: {path}")
    manifest = _read_json(root / "manifest.json")
    if set(manifest) != {
        "schema_version",
        "task_instruction",
        "icl_condition",
        "demonstration",
    }:
        raise FixedDemoError("video-only manifest fields do not match the allowlist")
    expected_instruction = _normalize_instruction(expected_task_instruction)
    if (
        manifest["schema_version"] != VIDEO_ONLY_BUNDLE_SCHEMA_VERSION
        or manifest["task_instruction"] != expected_instruction
        or manifest["icl_condition"] != "video_only"
    ):
        raise FixedDemoError("video-only manifest metadata does not match")
    demonstration = manifest["demonstration"]
    if not isinstance(demonstration, Mapping) or set(demonstration) != {
        "episode_outcome",
        "relation_to_target",
        "scene_or_object_poses_may_differ",
        "modalities",
        "video",
    }:
        raise FixedDemoError("video-only demonstration metadata is invalid")
    if (
        demonstration["episode_outcome"] != "verified successful demonstration"
        or demonstration["relation_to_target"] != "same_task_separate_episode"
        or demonstration["scene_or_object_poses_may_differ"] is not True
        or demonstration["modalities"] != ["head_rgb", "wrist_rgb"]
    ):
        raise FixedDemoError("video-only demonstration outcome contract is invalid")
    video = demonstration["video"]
    if not isinstance(video, Mapping) or set(video) != {
        "file",
        "fps_hz",
        "frame_count",
        "layout",
        "contact_sheets",
        "sampling",
    }:
        raise FixedDemoError("video-only video metadata is invalid")
    if (
        video["fps_hz"] != VIDEO_ONLY_FPS_HZ
        or not isinstance(video["frame_count"], int)
        or video["frame_count"] < 1
        or video["layout"] != "head_rgb_left__wrist_rgb_right"
    ):
        raise FixedDemoError("video-only video metadata values are invalid")
    video_path = _validate_artifact(
        video["file"], root, expected_media_type="video/mp4"
    )
    if video_path != root / "video" / "head_wrist_rgb.mp4":
        raise FixedDemoError("video-only video path is non-canonical")

    frame_count = int(video["frame_count"])
    sampling = video["sampling"]
    if not isinstance(sampling, Mapping) or set(sampling) != {
        "rule",
        "maximum_frames_per_sheet",
        "source_frame_count",
        "sampled_frame_indices",
    }:
        raise FixedDemoError(
            "video-only contact-sheet sampling metadata is invalid"
        )
    expected_indices = contact_sheet_indices(
        frame_count,
        limit=min(VIDEO_ONLY_CONTACT_SHEET_MAX_FRAMES, frame_count),
    )
    if (
        sampling["rule"] != VIDEO_ONLY_CONTACT_SHEET_SAMPLING_RULE
        or sampling["maximum_frames_per_sheet"]
        != VIDEO_ONLY_CONTACT_SHEET_MAX_FRAMES
        or sampling["source_frame_count"] != frame_count
        or sampling["sampled_frame_indices"] != expected_indices
    ):
        raise FixedDemoError(
            "video-only contact-sheet sampling metadata differs"
        )
    contact_sheets = video["contact_sheets"]
    if not isinstance(contact_sheets, Mapping) or set(contact_sheets) != {
        "head_rgb",
        "wrist_rgb",
    }:
        raise FixedDemoError("video-only contact-sheet fields differ")
    for camera_name, artifact in contact_sheets.items():
        sheet_path = _validate_artifact(
            artifact, root, expected_media_type="image/png"
        )
        expected_sheet = root / "video" / "contact_sheets" / f"{camera_name}.png"
        if sheet_path != expected_sheet:
            raise FixedDemoError("video-only contact-sheet path is non-canonical")

    # The only materialized payloads are RGB: the combined stream and two
    # decoder-independent contact sheets.
    expected_files = {
        Path("manifest.json"),
        Path("video/head_wrist_rgb.mp4"),
        Path("video/contact_sheets/head_rgb.png"),
        Path("video/contact_sheets/wrist_rgb.png"),
    }
    actual_files = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise FixedDemoError("video-only bundle contains non-RGB artifacts")
    return manifest


def _build_rgb_video(
    *,
    master_root: Path,
    master_manifest: Mapping[str, Any],
    destination: Path,
) -> tuple[list[int], dict[str, list[np.ndarray]]]:
    """Encode the two source RGB streams side by side without other fields."""

    import imageio.v2 as imageio
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        destination,
        fps=VIDEO_ONLY_FPS_HZ,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    frame_count = int(master_manifest["capture"]["frame_count"])
    sampled_frame_indices = contact_sheet_indices(
        frame_count,
        limit=min(VIDEO_ONLY_CONTACT_SHEET_MAX_FRAMES, frame_count),
    )
    sampled_set = set(sampled_frame_indices)
    sampled_rgb: dict[str, list[np.ndarray]] = {
        "head_rgb": [],
        "wrist_rgb": [],
    }
    try:
        for frame_index, frame in enumerate(master_manifest["capture"]["frames"]):
            observation_path = _safe_inside(
                master_root, frame.get("observation"), "master observation"
            )
            observation = _read_json(observation_path)
            images = []
            for camera_name in ("head", "wrist"):
                camera = observation.get("cameras", {}).get(camera_name)
                if not isinstance(camera, Mapping):
                    raise FixedDemoError(
                        f"master frame is missing {camera_name} RGB camera"
                    )
                rgb_file = camera.get("rgb", {}).get("file")
                rgb_path = _safe_inside(
                    observation_path.parent, rgb_file, f"{camera_name} RGB"
                )
                with Image.open(rgb_path) as image:
                    images.append(np.array(image.convert("RGB"), copy=True))
            if images[0].shape != images[1].shape:
                raise FixedDemoError(
                    "head and wrist RGB shapes differ in source demonstration"
                )
            writer.append_data(
                np.ascontiguousarray(np.concatenate(images, axis=1))
            )
            if frame_index in sampled_set:
                sampled_rgb["head_rgb"].append(images[0])
                sampled_rgb["wrist_rgb"].append(images[1])
    finally:
        writer.close()
    return sampled_frame_indices, sampled_rgb


def _build_contact_sheet(images: list[np.ndarray], destination: Path) -> None:
    """Build a clean RGB-only grid with no labels or task annotations."""

    from PIL import Image

    if not images:
        raise FixedDemoError("cannot build an empty RGB contact sheet")
    first = np.asarray(images[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise FixedDemoError("RGB contact-sheet frames must be HxWx3")
    height, width = (int(first.shape[0]), int(first.shape[1]))
    if height < 1 or width < 1:
        raise FixedDemoError("RGB contact-sheet frames must be non-empty")
    columns = max(1, int(np.ceil(np.sqrt(len(images)))))
    rows = int(np.ceil(len(images) / columns))
    gutter = VIDEO_ONLY_CONTACT_SHEET_GUTTER_PX
    canvas = Image.new(
        "RGB",
        (
            columns * width + (columns + 1) * gutter,
            rows * height + (rows + 1) * gutter,
        ),
        color=(255, 255, 255),
    )
    for index, array in enumerate(images):
        value = np.asarray(array)
        if value.shape != first.shape:
            raise FixedDemoError("RGB contact-sheet frame shapes differ")
        image = Image.fromarray(value.astype(np.uint8, copy=False), mode="RGB")
        column = index % columns
        row = index // columns
        canvas.paste(
            image,
            (
                gutter + column * (width + gutter),
                gutter + row * (height + gutter),
            ),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def _artifact_record(
    path: Path, root: Path, media_type: str = "video/mp4"
) -> dict[str, Any]:
    resolved = path.resolve()
    if os.path.commonpath((root.resolve(), resolved)) != os.fspath(root.resolve()):
        raise FixedDemoError("video-only artifact escapes its public root")
    return {
        "path": os.fspath(resolved.relative_to(root.resolve())),
        "media_type": media_type,
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _validate_artifact(
    value: Any, root: Path, *, expected_media_type: str
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "media_type",
        "size_bytes",
        "sha256",
    }:
        raise FixedDemoError("video-only artifact fields differ")
    relative = value["path"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise FixedDemoError("video-only artifact path must be relative")
    relative_path = Path(relative)
    if (
        relative_path.as_posix() != relative
        or any(part in {".", ".."} for part in relative_path.parts)
    ):
        raise FixedDemoError("video-only artifact path must be canonical")
    path = (root / relative_path).resolve()
    if os.path.commonpath((root.resolve(), path)) != os.fspath(root.resolve()):
        raise FixedDemoError("video-only artifact escapes its root")
    if path.is_symlink() or not path.is_file():
        raise FixedDemoError("video-only artifact is missing or unsafe")
    if value["media_type"] != expected_media_type:
        raise FixedDemoError("video-only artifact media type differs")
    if (
        path.stat().st_size != value["size_bytes"]
        or file_sha256(path) != value["sha256"]
    ):
        raise FixedDemoError("video-only artifact integrity differs")
    return path


def _safe_inside(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise FixedDemoError(f"{label} path is invalid")
    path = (root / relative).resolve()
    if os.path.commonpath((root.resolve(), path)) != os.fspath(root.resolve()):
        raise FixedDemoError(f"{label} path escapes its root")
    if not path.is_file() or path.is_symlink():
        raise FixedDemoError(f"{label} file is missing or unsafe")
    return path


def _normalize_instruction(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FixedDemoError("task instruction must be non-empty")
    return " ".join(value.split())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FixedDemoError(f"JSON object expected: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
