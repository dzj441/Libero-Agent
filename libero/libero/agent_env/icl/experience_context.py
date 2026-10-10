"""Project verified embodied experiences into modality-controlled public bundles.

The source specification and relation labels are evaluator-private.  The Agent
sees only public source-task metadata and the explicitly selected modalities.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .fixed_demo import (
    SOURCE_ACTION_SCHEMA_VERSION,
    _validate_materialized_observation,
    file_sha256,
    project_fixed_demo_bundle,
    source_action_semantics,
    validate_p4_replay_master,
)
from ..runtime.profiles import ObservationProfile, profile_capabilities


EXPERIENCE_CONTEXT_SPEC_SCHEMA_VERSION = "libero.experience_context_spec.v1"
EXPERIENCE_CONTEXT_BUNDLE_SCHEMA_VERSION = "libero.experience_context_bundle.v1"
EXPERIENCE_ITEM_SCHEMA_VERSION = "libero.public_experience.v1"
EXPERIENCE_CONTEXT_RECEIPT_SCHEMA_VERSION = (
    "libero.experience_context_projection_receipt.v1"
)

MODALITY_ORDER = ("text", "video", "observations", "actions")
ALLOWED_MODALITIES = frozenset(MODALITY_ORDER)
ALLOWED_TEXT_KINDS = frozenset(
    {"outcome_summary", "procedural_guidance", "source_agent_message"}
)
ALLOWED_SOURCE_TYPES = frozenset({"expert_episode"})
ALLOWED_RELATIONS = frozenset(
    {
        "same_task_separate_episode",
        "same_task_different_variant",
        "compositional_subtask",
        "analogous_task",
        "irrelevant_task",
        "counterfactual_or_misleading",
    }
)
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
VIDEO_FPS_HZ = 20.0


class ExperienceContextError(ValueError):
    """Raised when an experience-context spec or bundle violates its contract."""


def load_experience_context_spec(
    path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and normalize one evaluator-private context specification."""

    source = Path(path).expanduser().resolve()
    value = _read_json(source)
    root = (
        Path(artifact_root).expanduser().resolve()
        if artifact_root is not None
        else source.parent
    )
    normalized = normalize_experience_context_spec(value, artifact_root=root)
    normalized["source_spec"] = os.fspath(source)
    normalized["source_spec_sha256"] = file_sha256(source)
    return normalized


def normalize_experience_context_spec(
    value: Mapping[str, Any],
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Validate private labels and resolve replay-master paths."""

    if value.get("schema_version") != EXPERIENCE_CONTEXT_SPEC_SCHEMA_VERSION:
        raise ExperienceContextError("unsupported experience-context spec schema")
    context_id = _identifier(value.get("context_id"), "context_id")
    experiences = value.get("experiences")
    if not isinstance(experiences, list) or not experiences:
        raise ExperienceContextError("experience-context spec requires items")
    root = Path(artifact_root).expanduser().resolve()
    normalized_items = [
        _normalize_private_item(item, index=index, artifact_root=root)
        for index, item in enumerate(experiences)
    ]
    identifiers = [item["experience_id"] for item in normalized_items]
    if len(set(identifiers)) != len(identifiers):
        raise ExperienceContextError("experience_id values must be unique")
    return {
        "schema_version": EXPERIENCE_CONTEXT_SPEC_SCHEMA_VERSION,
        "context_id": context_id,
        "experiences": normalized_items,
    }


def project_experience_context_bundle(
    *,
    spec: Mapping[str, Any],
    destination: str | Path,
    profile: ObservationProfile | int | str,
    target_task_instruction: str,
) -> dict[str, Any]:
    """Publish one multi-item context bundle with file-level modality pruning."""

    destination = Path(destination).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"experience-context destination already exists: {destination}"
        )
    profile = ObservationProfile.parse(profile)
    target_instruction = _normalize_instruction(target_task_instruction)
    normalized = _coerce_normalized_spec(spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.projection-", dir=destination.parent
        )
    ).resolve()
    private_items: list[dict[str, Any]] = []
    public_items: list[dict[str, Any]] = []
    try:
        for index, private_item in enumerate(normalized["experiences"]):
            master_root = Path(private_item["master_root"])
            master_manifest = validate_p4_replay_master(master_root)
            source_instruction = _normalize_instruction(
                master_manifest["task"]["instruction"]
            )
            if (
                private_item["relation_to_target"]
                == "same_task_separate_episode"
                and source_instruction != target_instruction
            ):
                raise ExperienceContextError(
                    "same-task context item does not match the target instruction"
                )

            legacy_root = temporary / f".legacy_{index:03d}"
            project_fixed_demo_bundle(
                master_root=master_root,
                destination=legacy_root,
                profile=profile,
                expected_task_instruction=source_instruction,
            )
            legacy_manifest = _read_json(legacy_root / "manifest.json")
            item_relative = Path("experiences") / private_item["experience_id"]
            item_root = temporary / item_relative
            item_root.mkdir(parents=True)
            item_manifest = _materialize_public_item(
                private_item=private_item,
                source_instruction=source_instruction,
                legacy_root=legacy_root,
                legacy_manifest=legacy_manifest,
                item_root=item_root,
                profile=profile,
            )
            _write_json(item_root / "manifest.json", item_manifest)
            shutil.rmtree(legacy_root)
            public_items.append(
                {
                    "experience_id": private_item["experience_id"],
                    "source_task_instruction": source_instruction,
                    "episode_outcome": "verified_success",
                    "modalities": list(private_item["modalities"]),
                    "manifest": _artifact_record(
                        item_root / "manifest.json", temporary, "application/json"
                    ),
                }
            )
            private_items.append(
                {
                    "experience_id": private_item["experience_id"],
                    "source_type": private_item["source_type"],
                    "relation_to_target": private_item["relation_to_target"],
                    "modalities": list(private_item["modalities"]),
                    "text_kind": private_item["text_kind"],
                    "source_master": os.fspath(master_root),
                    "source_master_manifest_sha256": file_sha256(
                        master_root / "p4_master_manifest.json"
                    ),
                }
            )

        visibility_contract = {
            "relation_labels_disclosed": False,
            "target_task_or_seed_disclosed": False,
            "source_paths_disclosed": False,
            "stepwise_checker_disclosed": False,
            "only_declared_modalities_materialized": True,
            "scene_or_object_poses_may_differ": True,
        }
        public_manifest = {
            "schema_version": EXPERIENCE_CONTEXT_BUNDLE_SCHEMA_VERSION,
            "observation_profile": profile.public_name,
            "capabilities": profile_capabilities(profile),
            "experience_count": len(public_items),
            "experiences": public_items,
        }
        _write_json(temporary / "manifest.json", public_manifest)
        private_integrity = _context_integrity(temporary)
        private_budget = _context_budget(temporary)
        validate_experience_context_bundle(
            temporary,
            expected_profile=profile,
            expected_experience_count=len(public_items),
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    receipt = {
        "schema_version": EXPERIENCE_CONTEXT_RECEIPT_SCHEMA_VERSION,
        "context_id": normalized["context_id"],
        "target_task_instruction": target_instruction,
        "agent_bundle": os.fspath(destination),
        "manifest_sha256": file_sha256(destination / "manifest.json"),
        "experience_count": len(public_items),
        "experiences": private_items,
        "visibility_contract": visibility_contract,
        "bundle_integrity": private_integrity,
        "budget": private_budget,
    }
    if isinstance(normalized.get("source_spec"), str):
        receipt["source_spec"] = normalized["source_spec"]
        receipt["source_spec_sha256"] = normalized["source_spec_sha256"]
    return receipt


def validate_experience_context_bundle(
    bundle_root: str | Path,
    *,
    expected_profile: ObservationProfile | int | str,
    expected_experience_count: int | None = None,
) -> dict[str, Any]:
    """Validate the public allowlist, artifact hashes, and profile projection."""

    root = Path(bundle_root).expanduser().resolve()
    profile = ObservationProfile.parse(expected_profile)
    if not root.is_dir() or root.is_symlink():
        raise ExperienceContextError(
            f"experience-context bundle is not a real directory: {root}"
        )
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ExperienceContextError(f"context bundle contains a symlink: {path}")
    manifest = _read_json(root / "manifest.json")
    if set(manifest) != {
        "schema_version",
        "observation_profile",
        "capabilities",
        "experience_count",
        "experiences",
    }:
        raise ExperienceContextError("context root manifest fields differ")
    if (
        manifest["schema_version"] != EXPERIENCE_CONTEXT_BUNDLE_SCHEMA_VERSION
        or manifest["observation_profile"] != profile.public_name
        or manifest["capabilities"] != profile_capabilities(profile)
    ):
        raise ExperienceContextError("context root manifest metadata differs")
    items = manifest["experiences"]
    if (
        not isinstance(items, list)
        or not items
        or manifest["experience_count"] != len(items)
        or (
            expected_experience_count is not None
            and len(items) != expected_experience_count
        )
    ):
        raise ExperienceContextError("context item counts differ")
    seen: set[str] = set()
    for item in items:
        if set(item) != {
            "experience_id",
            "source_task_instruction",
            "episode_outcome",
            "modalities",
            "manifest",
        }:
            raise ExperienceContextError("context item summary fields differ")
        experience_id = _identifier(item["experience_id"], "experience_id")
        if experience_id in seen:
            raise ExperienceContextError("context experience IDs are duplicated")
        seen.add(experience_id)
        if item["episode_outcome"] != "verified_success":
            raise ExperienceContextError("unsupported public experience outcome")
        modalities = _normalize_modalities(item["modalities"])
        item_manifest_path = _validate_artifact(
            item["manifest"], root, f"{experience_id} manifest"
        )
        expected_path = root / "experiences" / experience_id / "manifest.json"
        if item_manifest_path != expected_path:
            raise ExperienceContextError("experience manifest path is non-canonical")
        public_item = _read_json(item_manifest_path)
        _validate_public_item(
            public_item,
            item_root=item_manifest_path.parent,
            profile=profile,
            expected_experience_id=experience_id,
            expected_task_instruction=item["source_task_instruction"],
            expected_modalities=modalities,
        )

    _assert_no_private_metadata(root)
    return manifest


def _normalize_private_item(
    value: Any,
    *,
    index: int,
    artifact_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperienceContextError(f"experience {index} must be an object")
    experience_id = _identifier(value.get("experience_id"), "experience_id")
    source_type = str(value.get("source_type", "expert_episode"))
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ExperienceContextError(f"unsupported source_type: {source_type}")
    relation = str(value.get("relation_to_target", ""))
    if relation not in ALLOWED_RELATIONS:
        raise ExperienceContextError(f"unsupported relation_to_target: {relation}")
    modalities = _normalize_modalities(value.get("modalities"))
    master_value = value.get("master_root")
    if not isinstance(master_value, str) or not master_value.strip():
        raise ExperienceContextError("expert_episode requires master_root")
    master_candidate = Path(master_value).expanduser()
    master_root = (
        master_candidate
        if master_candidate.is_absolute()
        else artifact_root / master_candidate
    ).resolve()
    public_text = value.get("public_text")
    text_kind = value.get("text_kind")
    if "text" in modalities:
        text_kind = str(text_kind or "outcome_summary")
        if text_kind not in ALLOWED_TEXT_KINDS:
            raise ExperienceContextError(f"unsupported text_kind: {text_kind}")
    elif text_kind is not None:
        raise ExperienceContextError("text_kind requires the text modality")
    if public_text is not None:
        if "text" not in modalities:
            raise ExperienceContextError("public_text requires the text modality")
        if not isinstance(public_text, str) or not public_text.strip():
            raise ExperienceContextError("public_text must be non-empty")
        if len(public_text.encode("utf-8")) > 64 * 1024:
            raise ExperienceContextError("public_text exceeds the 64 KiB limit")
    if (
        "text" in modalities
        and text_kind != "outcome_summary"
        and public_text is None
    ):
        raise ExperienceContextError(
            f"{text_kind} requires explicit public_text"
        )
    return {
        "experience_id": experience_id,
        "source_type": source_type,
        "master_root": os.fspath(master_root),
        "relation_to_target": relation,
        "modalities": modalities,
        "text_kind": text_kind,
        "public_text": public_text,
    }


def _coerce_normalized_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != EXPERIENCE_CONTEXT_SPEC_SCHEMA_VERSION:
        raise ExperienceContextError("unsupported experience-context spec schema")
    _identifier(spec.get("context_id"), "context_id")
    experiences = spec.get("experiences")
    if not isinstance(experiences, list) or not experiences:
        raise ExperienceContextError("experience-context spec requires items")
    for item in experiences:
        if not isinstance(item, Mapping) or not Path(
            str(item.get("master_root", ""))
        ).is_absolute():
            raise ExperienceContextError(
                "projector requires a normalized spec with absolute master paths"
            )
        _identifier(item.get("experience_id"), "experience_id")
        modalities = _normalize_modalities(item.get("modalities"))
        if item.get("source_type") not in ALLOWED_SOURCE_TYPES:
            raise ExperienceContextError("normalized source_type is invalid")
        if item.get("relation_to_target") not in ALLOWED_RELATIONS:
            raise ExperienceContextError("normalized relation is invalid")
        if "text" in modalities:
            if item.get("text_kind") not in ALLOWED_TEXT_KINDS:
                raise ExperienceContextError("normalized text_kind is invalid")
        elif item.get("text_kind") is not None:
            raise ExperienceContextError("normalized text_kind is invalid")
    return dict(spec)


def _materialize_public_item(
    *,
    private_item: Mapping[str, Any],
    source_instruction: str,
    legacy_root: Path,
    legacy_manifest: Mapping[str, Any],
    item_root: Path,
    profile: ObservationProfile,
) -> dict[str, Any]:
    modalities = tuple(private_item["modalities"])
    content: dict[str, Any] = {}
    if "text" in modalities:
        text_path = item_root / "text" / "guidance.md"
        public_text = private_item.get("public_text")
        lines = [
            "# Embodied experience",
            "",
            f"Source task: {source_instruction}",
            "",
            "Outcome: verified success in a separate episode.",
            "",
            "The current scene configuration and object or goal poses may differ.",
        ]
        if isinstance(public_text, str):
            lines.extend(("", "## Additional guidance", "", public_text.strip()))
        _write_text(text_path, "\n".join(lines) + "\n")
        content["text"] = {
            "kind": private_item["text_kind"],
            "artifact": _artifact_record(text_path, item_root, "text/markdown"),
        }

    if "video" in modalities:
        video_root = item_root / "video"
        video_path = video_root / "head_wrist_rgb.mp4"
        _build_rgb_video(
            legacy_root=legacy_root,
            legacy_manifest=legacy_manifest,
            destination=video_path,
        )
        contact_sheets: dict[str, Any] = {}
        for name, artifact in legacy_manifest["overview"]["contact_sheets"].items():
            if name not in {"head_rgb", "wrist_rgb"}:
                continue
            source = _artifact_path(artifact, legacy_root, f"{name} contact sheet")
            target = video_root / "contact_sheets" / f"{name}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            contact_sheets[name] = _artifact_record(target, item_root, "image/png")
        content["video"] = {
            "rgb_video": _artifact_record(video_path, item_root, "video/mp4"),
            "frame_count": int(
                legacy_manifest["demonstration"]["frame_count"]
            ),
            "fps_hz": VIDEO_FPS_HZ,
            "duration_s": (
                int(legacy_manifest["demonstration"]["frame_count"])
                / VIDEO_FPS_HZ
            ),
            "layout": "head_rgb_left__wrist_rgb_right",
            "contact_sheets": contact_sheets,
            "sampling": legacy_manifest["overview"]["sampling"],
        }

    if "observations" in modalities:
        target_frames = item_root / "frames"
        shutil.copytree(legacy_root / "frames", target_frames)
        frames = []
        for frame in legacy_manifest["frames"]:
            index = int(frame["frame_index"])
            path = item_root / "frames" / f"frame_{index:06d}" / "observation.json"
            frames.append(
                {
                    "frame_index": index,
                    "observation_id": f"frame_{index:06d}",
                    "observation": _artifact_record(
                        path, item_root, "application/json"
                    ),
                }
            )
        content["observations"] = {
            "frame_count": len(frames),
            "frames": frames,
        }

    if "actions" in modalities:
        source_records = _read_jsonl(legacy_root / "trajectory.jsonl")
        records = []
        for index, record in enumerate(source_records):
            vector = record["source_action"]["normalized_vector_7d"]
            records.append(
                {
                    "action_index": index,
                    "schema_version": SOURCE_ACTION_SCHEMA_VERSION,
                    "normalized_vector_7d": vector,
                }
            )
        action_path = item_root / "trajectory" / "actions.jsonl"
        _write_jsonl(action_path, records)
        content["actions"] = {
            "action_count": len(records),
            "trajectory": _artifact_record(
                action_path, item_root, "application/x-ndjson"
            ),
            "semantics": _public_experience_action_semantics(),
        }

    return {
        "schema_version": EXPERIENCE_ITEM_SCHEMA_VERSION,
        "experience_id": private_item["experience_id"],
        "source_type": "verified_expert_episode",
        "source_task_instruction": source_instruction,
        "episode_outcome": "verified_success",
        "scene_or_object_poses_may_differ": True,
        "observation_profile": profile.public_name,
        "modalities": list(modalities),
        "content": content,
    }


def _validate_public_item(
    value: Mapping[str, Any],
    *,
    item_root: Path,
    profile: ObservationProfile,
    expected_experience_id: str,
    expected_task_instruction: str,
    expected_modalities: tuple[str, ...],
) -> None:
    if set(value) != {
        "schema_version",
        "experience_id",
        "source_type",
        "source_task_instruction",
        "episode_outcome",
        "scene_or_object_poses_may_differ",
        "observation_profile",
        "modalities",
        "content",
    }:
        raise ExperienceContextError("public experience fields differ")
    if (
        value["schema_version"] != EXPERIENCE_ITEM_SCHEMA_VERSION
        or value["experience_id"] != expected_experience_id
        or value["source_type"] != "verified_expert_episode"
        or value["source_task_instruction"]
        != _normalize_instruction(expected_task_instruction)
        or value["episode_outcome"] != "verified_success"
        or value["scene_or_object_poses_may_differ"] is not True
        or value["observation_profile"] != profile.public_name
        or _normalize_modalities(value["modalities"]) != expected_modalities
    ):
        raise ExperienceContextError("public experience metadata differs")
    content = value["content"]
    if not isinstance(content, Mapping) or set(content) != set(expected_modalities):
        raise ExperienceContextError("materialized context modalities differ")

    if "text" in content:
        text = content["text"]
        if (
            not isinstance(text, Mapping)
            or set(text) != {"kind", "artifact"}
            or text["kind"] not in ALLOWED_TEXT_KINDS
        ):
            raise ExperienceContextError("text context metadata differs")
        _validate_artifact(text["artifact"], item_root, "text guidance")
    if "video" in content:
        video = content["video"]
        if set(video) != {
            "rgb_video",
            "frame_count",
            "fps_hz",
            "duration_s",
            "layout",
            "contact_sheets",
            "sampling",
        }:
            raise ExperienceContextError("video metadata fields differ")
        _validate_artifact(video["rgb_video"], item_root, "RGB video")
        if (
            not isinstance(video["frame_count"], int)
            or video["frame_count"] < 1
            or video["fps_hz"] != VIDEO_FPS_HZ
            or video["duration_s"]
            != video["frame_count"] / video["fps_hz"]
            or video["layout"] != "head_rgb_left__wrist_rgb_right"
        ):
            raise ExperienceContextError("video metadata is invalid")
        sheets = video["contact_sheets"]
        expected_sheet_names = {"head_rgb", "wrist_rgb"}
        if not isinstance(sheets, Mapping) or set(sheets) != expected_sheet_names:
            raise ExperienceContextError("contact-sheet modalities differ")
        for name, artifact in sheets.items():
            _validate_artifact(artifact, item_root, f"{name} contact sheet")
    if "observations" in content:
        observations = content["observations"]
        if set(observations) != {"frame_count", "frames"}:
            raise ExperienceContextError("observation trajectory fields differ")
        frames = observations["frames"]
        if (
            not isinstance(frames, list)
            or not frames
            or observations["frame_count"] != len(frames)
        ):
            raise ExperienceContextError("observation frame counts differ")
        for index, frame in enumerate(frames):
            if set(frame) != {"frame_index", "observation_id", "observation"}:
                raise ExperienceContextError("observation frame record differs")
            if (
                frame["frame_index"] != index
                or frame["observation_id"] != f"frame_{index:06d}"
            ):
                raise ExperienceContextError("observation frames are not contiguous")
            path = _validate_artifact(
                frame["observation"], item_root, f"observation {index}"
            )
            _validate_materialized_observation(
                path.parent,
                expected_profile=profile,
                expected_frame_index=index,
            )
    if "actions" in content:
        actions = content["actions"]
        if set(actions) != {"action_count", "trajectory", "semantics"}:
            raise ExperienceContextError("action trajectory fields differ")
        if actions["semantics"] != _public_experience_action_semantics():
            raise ExperienceContextError("action semantics differ")
        path = _validate_artifact(actions["trajectory"], item_root, "actions")
        records = _read_jsonl(path)
        if actions["action_count"] != len(records):
            raise ExperienceContextError("action counts differ")
        for index, record in enumerate(records):
            if set(record) != {
                "action_index",
                "schema_version",
                "normalized_vector_7d",
            }:
                raise ExperienceContextError("public action record differs")
            vector = np.asarray(record["normalized_vector_7d"], dtype=np.float64)
            if (
                record["action_index"] != index
                or record["schema_version"] != SOURCE_ACTION_SCHEMA_VERSION
                or vector.shape != (7,)
                or not np.isfinite(vector).all()
                or np.any(np.abs(vector) > 1.0 + 1.0e-9)
            ):
                raise ExperienceContextError("public action values are invalid")


def _public_experience_action_semantics() -> dict[str, Any]:
    semantics = source_action_semantics()
    semantics["native_osc_sequence_compatible"] = True
    semantics["note"] = (
        "These are native per-control-cycle OSC inputs. Their 7D vectors can "
        "be submitted as osc_sequence actions; they are not high-level metric "
        "liberoctl step commands."
    )
    return semantics


def _build_rgb_video(
    *,
    legacy_root: Path,
    legacy_manifest: Mapping[str, Any],
    destination: Path,
) -> None:
    import imageio.v2 as imageio

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        destination,
        fps=VIDEO_FPS_HZ,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame in legacy_manifest["frames"]:
            observation_path = _artifact_path(
                frame["observation"], legacy_root, "legacy observation"
            )
            observation = _read_json(observation_path)
            images = []
            for camera_name in ("head", "wrist"):
                image_path = observation_path.parent / observation["cameras"][
                    camera_name
                ]["rgb"]["file"]
                images.append(np.asarray(Image.open(image_path).convert("RGB")))
            if images[0].shape != images[1].shape:
                raise ExperienceContextError(
                    "head and wrist RGB shapes differ in source experience"
                )
            writer.append_data(np.ascontiguousarray(np.concatenate(images, axis=1)))
    finally:
        writer.close()


def _normalize_modalities(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ExperienceContextError("modalities must be a non-empty list")
    names = [str(item) for item in value]
    if len(set(names)) != len(names) or not set(names).issubset(ALLOWED_MODALITIES):
        raise ExperienceContextError("modalities contain duplicates or unknown values")
    return tuple(name for name in MODALITY_ORDER if name in names)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_IDENTIFIER.fullmatch(value):
        raise ExperienceContextError(f"{label} is not a safe identifier")
    return value


def _normalize_instruction(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperienceContextError("task instruction must be non-empty")
    return " ".join(value.split())


def _context_integrity(root: Path) -> dict[str, Any]:
    files = []
    root_manifest = root / "manifest.json"
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ExperienceContextError(f"context bundle contains a symlink: {path}")
        if path.is_file() and path != root_manifest:
            files.append(
                {
                    "path": os.fspath(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return {"algorithm": "sha256", "files": files}


def _context_budget(root: Path) -> dict[str, Any]:
    manifests = [
        _read_json(path)
        for path in sorted((root / "experiences").glob("*/manifest.json"))
    ]
    budget: dict[str, Any] = {
        "experience_count": len(manifests),
        "file_count": sum(path.is_file() for path in root.rglob("*")),
        "total_bytes": sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        ),
        "observation_frame_count": 0,
        "native_osc_action_count": 0,
        "video_frame_count": 0,
        "video_duration_s": 0.0,
        "text_utf8_bytes": 0,
    }
    for manifest in manifests:
        content = manifest.get("content", {})
        observations = content.get("observations", {})
        actions = content.get("actions", {})
        video = content.get("video", {})
        text = content.get("text", {})
        budget["observation_frame_count"] += int(
            observations.get("frame_count", 0)
        )
        budget["native_osc_action_count"] += int(actions.get("action_count", 0))
        budget["video_frame_count"] += int(video.get("frame_count", 0))
        budget["video_duration_s"] += float(video.get("duration_s", 0.0))
        artifact = text.get("artifact", {})
        budget["text_utf8_bytes"] += int(artifact.get("size_bytes", 0))
    return budget


def _artifact_record(path: Path, root: Path, media_type: str) -> dict[str, Any]:
    resolved = path.resolve()
    if os.path.commonpath((root.resolve(), resolved)) != os.fspath(root.resolve()):
        raise ExperienceContextError("artifact escapes its public root")
    return {
        "path": os.fspath(resolved.relative_to(root.resolve())),
        "media_type": media_type,
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _validate_artifact(value: Any, root: Path, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "media_type",
        "size_bytes",
        "sha256",
    }:
        raise ExperienceContextError(f"{label} artifact fields differ")
    path = _artifact_path(value, root, label)
    if path.stat().st_size != value["size_bytes"] or file_sha256(path) != value["sha256"]:
        raise ExperienceContextError(f"{label} artifact integrity differs")
    return path


def _artifact_path(value: Any, root: Path, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ExperienceContextError(f"{label} artifact is invalid")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value or Path(path_value).is_absolute():
        raise ExperienceContextError(f"{label} path must be relative")
    path = (root / path_value).resolve()
    if os.path.commonpath((root.resolve(), path)) != os.fspath(root.resolve()):
        raise ExperienceContextError(f"{label} path escapes its root")
    if not path.is_file() or path.is_symlink():
        raise ExperienceContextError(f"{label} artifact is missing or unsafe")
    return path


def _assert_no_private_metadata(root: Path) -> None:
    forbidden = (
        "dataset_path",
        "demo_key",
        "bddl_file",
        "mujoco_state",
        "init_state_id",
        "simulator_seed",
        "first_success_step",
        "relation_to_target",
        "same_task_separate_episode",
        "compositional_subtask",
        "irrelevant_task",
        "counterfactual_or_misleading",
    )
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".md"}:
            continue
        rendered = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in rendered:
                raise ExperienceContextError(
                    f"private metadata token {token!r} appears in {path}"
                )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperienceContextError(f"JSON object expected: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ExperienceContextError(f"blank JSONL record at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ExperienceContextError(
                f"JSON object expected at {path}:{line_number}"
            )
        records.append(value)
    return records


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
