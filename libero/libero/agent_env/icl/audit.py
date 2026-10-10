"""Post-hoc audit of Agent context use and source-action similarity."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


CONTEXT_AUDIT_SCHEMA_VERSION = "libero.experience_context_audit.v1"
PUBLIC_PATH_PATTERN = re.compile(
    r"benchmark_inputs/(?:experience_context|expert_demo|current_observation)"
    r"(?:/[A-Za-z0-9_.-]+)*"
)
CONTEXT_REFERENCE_PATTERN = re.compile(
    r"\b(context|experience|demonstration|demo|expert|trajectory)\b", re.IGNORECASE
)


def audit_experience_context_run(run_directory: str | Path) -> dict[str, Any]:
    """Build an evidence report from persisted evaluator-private run artifacts."""

    root = Path(run_directory).expanduser().resolve()
    actions = _read_jsonl(root / "actions.jsonl")
    session = _read_jsonl(root / "codex_session.jsonl")
    result = _read_json(root / "result.json", required=False)
    receipt, context_kind = _load_context_receipt(root)
    first_action_time = _first_robot_action_time(actions)
    activity = _session_activity(session, first_action_time=first_action_time)
    agent_actions = _flatten_agent_actions(actions)
    sources = _source_trajectories(receipt, context_kind=context_kind)
    comparisons = [
        {
            "experience_id": source["experience_id"],
            "source_action_count": len(source["actions"]),
            **compare_action_trajectories(agent_actions, source["actions"]),
        }
        for source in sources
    ]
    return {
        "schema_version": CONTEXT_AUDIT_SCHEMA_VERSION,
        "run_directory": root.as_posix(),
        "run_status": result.get("status"),
        "official_success": result.get("success"),
        "context_kind": context_kind,
        "experience_count": len(sources),
        "agent_native_osc_action_count": len(agent_actions),
        "context_access": activity["context_access"],
        "current_observation_access_before_first_action": activity[
            "current_observation_access_before_first_action"
        ],
        "context_reference_messages": activity["context_reference_messages"],
        "initial_public_alignment": _initial_public_alignment(
            root,
            receipt,
            context_kind=context_kind,
        ),
        "source_action_comparisons": comparisons,
        "interpretation_contract": {
            "file_access_is_observed_not_inferred": True,
            "absence_of_logged_access_does_not_prove_nonuse": True,
            "action_similarity_alone_is_not_success_or_leakage": True,
            "hidden_chain_of_thought_used": False,
        },
    }


def compare_action_trajectories(
    agent_actions: Sequence[Sequence[float]],
    source_actions: Sequence[Sequence[float]],
    *,
    exact_tolerance: float = 1.0e-9,
    near_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare two normalized 7D action streams without assuming alignment."""

    agent = _action_array(agent_actions)
    source = _action_array(source_actions)
    if len(agent) == 0 or len(source) == 0:
        return {
            "agent_action_count": len(agent),
            "exact_source_action_fraction": None,
            "near_source_action_fraction": None,
            "mean_nearest_source_l2": None,
            "median_nearest_source_l2": None,
            "shared_initial_exact_prefix": 0,
            "longest_exact_contiguous_match": 0,
            "longest_near_contiguous_match": 0,
            "exact_contiguous_copy_fraction_min_4": None,
            "near_contiguous_copy_fraction_min_4": None,
            "dtw_mean_l2": None,
            "exact_tolerance_l2": exact_tolerance,
            "near_tolerance_l2": near_tolerance,
        }
    distances = np.linalg.norm(agent[:, None, :] - source[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    prefix = 0
    for agent_vector, source_vector in zip(agent, source):
        if np.linalg.norm(agent_vector - source_vector) > exact_tolerance:
            break
        prefix += 1
    longest_exact = _longest_contiguous_match(distances, exact_tolerance)
    longest_near = _longest_contiguous_match(distances, near_tolerance)
    return {
        "agent_action_count": len(agent),
        "exact_source_action_fraction": float(
            np.mean(nearest <= exact_tolerance)
        ),
        "near_source_action_fraction": float(np.mean(nearest <= near_tolerance)),
        "mean_nearest_source_l2": float(np.mean(nearest)),
        "median_nearest_source_l2": float(np.median(nearest)),
        "shared_initial_exact_prefix": prefix,
        "longest_exact_contiguous_match": longest_exact,
        "longest_exact_contiguous_fraction_of_agent": float(
            longest_exact / len(agent)
        ),
        "longest_exact_contiguous_fraction_of_source": float(
            longest_exact / len(source)
        ),
        "longest_near_contiguous_match": longest_near,
        "longest_near_contiguous_fraction_of_agent": float(
            longest_near / len(agent)
        ),
        "longest_near_contiguous_fraction_of_source": float(
            longest_near / len(source)
        ),
        "exact_contiguous_copy_fraction_min_4": (
            _fraction_in_contiguous_matches(
                distances, exact_tolerance, minimum_run_length=4
            )
        ),
        "near_contiguous_copy_fraction_min_4": (
            _fraction_in_contiguous_matches(
                distances, near_tolerance, minimum_run_length=4
            )
        ),
        "dtw_mean_l2": _dtw_mean_cost(distances),
        "exact_tolerance_l2": exact_tolerance,
        "near_tolerance_l2": near_tolerance,
    }


def _session_activity(
    records: Sequence[Mapping[str, Any]],
    *,
    first_action_time: datetime | None,
) -> dict[str, Any]:
    context_events = []
    observation_before_action = []
    reference_messages = []
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, Mapping) or payload.get("type") != "item_completed":
            continue
        item = payload.get("item")
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        timestamp = str(record.get("timestamp", ""))
        rendered = ""
        output_rendered = ""
        if item_type == "CommandExecution":
            rendered = json.dumps(
                {
                    "command": item.get("command"),
                    "parsed_cmd": item.get("parsed_cmd"),
                    "cwd": item.get("cwd"),
                },
                sort_keys=True,
            )
            output_rendered = "\n".join(
                str(item.get(field, ""))
                for field in ("stdout", "stderr", "formatted_output")
            )
        elif item_type == "ImageView":
            rendered = str(item.get("path", ""))
        elif item_type == "AgentMessage":
            message = _agent_message_text(item)
            if message and CONTEXT_REFERENCE_PATTERN.search(message):
                reference_messages.append(
                    {"timestamp": timestamp, "phase": item.get("phase"), "text": message}
                )
            continue
        else:
            continue
        paths = sorted(set(PUBLIC_PATH_PATTERN.findall(rendered)))
        output_paths = sorted(set(PUBLIC_PATH_PATTERN.findall(output_rendered)))
        context_paths = [
            path
            for path in paths
            if path.startswith("benchmark_inputs/experience_context")
            or path.startswith("benchmark_inputs/expert_demo")
        ]
        if context_paths:
            context_events.append(
                {
                    "timestamp": timestamp,
                    "event_type": item_type,
                    "requested_paths": context_paths,
                    "output_referenced_paths": [
                        path
                        for path in output_paths
                        if path.startswith("benchmark_inputs/experience_context")
                        or path.startswith("benchmark_inputs/expert_demo")
                    ],
                    "command": item.get("command") if item_type == "CommandExecution" else None,
                }
            )
        current_paths = [
            path
            for path in paths
            if path.startswith("benchmark_inputs/current_observation")
        ]
        event_time = _parse_time(timestamp)
        if (
            current_paths
            and first_action_time is not None
            and event_time is not None
            and event_time < first_action_time
        ):
            observation_before_action.append(
                {
                    "timestamp": timestamp,
                    "event_type": item_type,
                    "paths": current_paths,
                }
            )

    unique_context_paths = sorted(
        {path for event in context_events for path in event["requested_paths"]}
    )
    unique_output_references = sorted(
        {
            path
            for event in context_events
            for path in event["output_referenced_paths"]
        }
    )
    modality_counts = {
        "manifest_or_metadata": 0,
        "text": 0,
        "video_or_contact_sheet": 0,
        "observations": 0,
        "actions": 0,
    }
    for event in context_events:
        rendered = " ".join(event["requested_paths"])
        if "manifest.json" in rendered or "guidance.md" in rendered:
            modality_counts["manifest_or_metadata"] += 1
        if "/text/" in rendered or "guidance.md" in rendered:
            modality_counts["text"] += 1
        if "/video/" in rendered or "/overview/" in rendered:
            modality_counts["video_or_contact_sheet"] += 1
        if "/frames/" in rendered:
            modality_counts["observations"] += 1
        if "actions.jsonl" in rendered or "trajectory.jsonl" in rendered:
            modality_counts["actions"] += 1
    return {
        "context_access": {
            "event_count": len(context_events),
            "unique_requested_paths": unique_context_paths,
            "unique_output_referenced_paths": unique_output_references,
            "modality_event_counts": modality_counts,
            "events": context_events,
        },
        "current_observation_access_before_first_action": {
            "observed": bool(observation_before_action),
            "events": observation_before_action,
        },
        "context_reference_messages": reference_messages,
    }


def _flatten_agent_actions(records: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    actions = []
    for record in records:
        request = record.get("request")
        response = record.get("response")
        if not isinstance(request, Mapping) or request.get("command") != "osc_sequence":
            continue
        if isinstance(response, Mapping) and response.get("ok") is False:
            continue
        batch = request.get("actions")
        if not isinstance(batch, list):
            continue
        for action in batch:
            vector = np.asarray(action, dtype=np.float64)
            if vector.shape == (7,) and np.isfinite(vector).all():
                actions.append(vector.tolist())
    return actions


def _source_trajectories(
    receipt: Mapping[str, Any] | None,
    *,
    context_kind: str,
) -> list[dict[str, Any]]:
    if receipt is None:
        return []
    if context_kind == "experience_context":
        items = receipt.get("experiences")
        if not isinstance(items, list):
            return []
        sources = [
            (str(item.get("experience_id")), item.get("source_master"))
            for item in items
            if isinstance(item, Mapping)
        ]
    else:
        sources = [("legacy_fixed_demo", receipt.get("source_master"))]
    output = []
    for experience_id, source in sources:
        if not isinstance(source, str):
            continue
        root = Path(source)
        manifest = _read_json(root / "p4_master_manifest.json", required=False)
        trajectory = manifest.get("capture", {}).get("trajectory", {}).get("path")
        if not isinstance(trajectory, str):
            continue
        records = _read_jsonl(root / trajectory)
        vectors = []
        for record in records:
            vector = record.get("source_action", {}).get("normalized_vector_7d")
            array = np.asarray(vector, dtype=np.float64)
            if array.shape == (7,) and np.isfinite(array).all():
                vectors.append(array.tolist())
        output.append({"experience_id": experience_id, "actions": vectors})
    return output


def _initial_public_alignment(
    run_root: Path,
    receipt: Mapping[str, Any] | None,
    *,
    context_kind: str,
) -> dict[str, Any]:
    """Compare source/query initial states using only public observation fields."""

    query_path = run_root / "private_observations/obs_000000/observation.json"
    if receipt is None or not query_path.is_file():
        return {"available": False, "experiences": []}
    query = _read_json(query_path)
    target_instruction = _normalized_text(receipt.get("target_task_instruction"))
    if context_kind == "experience_context":
        items = receipt.get("experiences")
        if not isinstance(items, list):
            return {"available": False, "experiences": []}
    else:
        items = [
            {
                "experience_id": "legacy_fixed_demo",
                "source_master": receipt.get("source_master"),
            }
        ]
    comparisons = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        experience_id = str(item.get("experience_id", ""))
        source_root = item.get("source_master")
        if not isinstance(source_root, str):
            continue
        manifest = _read_json(
            Path(source_root) / "p4_master_manifest.json", required=False
        )
        frames = manifest.get("capture", {}).get("frames")
        if not isinstance(frames, list) or not frames:
            continue
        source_relative = frames[0].get("observation")
        if not isinstance(source_relative, str):
            continue
        source_path = Path(source_root) / source_relative
        if not source_path.is_file():
            continue
        source = _read_json(source_path)
        source_instruction = _normalized_text(
            manifest.get("task", {}).get("instruction")
        )
        same_task = bool(
            target_instruction
            and source_instruction
            and target_instruction == source_instruction
        )
        record: dict[str, Any] = {
            "experience_id": experience_id,
            "source_task_matches_query": same_task,
        }
        source_pose = np.asarray(
            source.get("state", {}).get("eef_pose_robot_base_xyzw_7d"),
            dtype=np.float64,
        )
        query_pose = np.asarray(
            query.get("state", {}).get("eef_pose_robot_base_xyzw_7d"),
            dtype=np.float64,
        )
        if source_pose.shape == (7,) and query_pose.shape == (7,):
            delta = query_pose[:3] - source_pose[:3]
            record["eef_position_delta_query_minus_source_m_3d"] = delta.tolist()
            record["eef_position_distance_m"] = float(np.linalg.norm(delta))
        if same_task:
            record["task_entity_bbox_center_deltas_px"] = (
                _bbox_center_deltas(source, query)
            )
        comparisons.append(record)
    return {
        "available": bool(comparisons),
        "field_contract": "agent_visible_initial_observation_fields_only",
        "experiences": comparisons,
    }


def _bbox_center_deltas(
    source: Mapping[str, Any], query: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    source_cameras = source.get("annotations", {}).get("cameras", {})
    query_cameras = query.get("annotations", {}).get("cameras", {})
    if not isinstance(source_cameras, Mapping) or not isinstance(
        query_cameras, Mapping
    ):
        return output
    for camera_name in sorted(set(source_cameras) & set(query_cameras)):
        source_entities = source_cameras[camera_name].get("task_entities", {})
        query_entities = query_cameras[camera_name].get("task_entities", {})
        if not isinstance(source_entities, Mapping) or not isinstance(
            query_entities, Mapping
        ):
            continue
        for entity_id in sorted(set(source_entities) & set(query_entities)):
            source_bbox = np.asarray(
                source_entities[entity_id].get("bbox_xyxy"), dtype=np.float64
            )
            query_bbox = np.asarray(
                query_entities[entity_id].get("bbox_xyxy"), dtype=np.float64
            )
            if source_bbox.shape != (4,) or query_bbox.shape != (4,):
                continue
            source_center = (source_bbox[:2] + source_bbox[2:]) / 2.0
            query_center = (query_bbox[:2] + query_bbox[2:]) / 2.0
            delta = query_center - source_center
            output.append(
                {
                    "camera": camera_name,
                    "entity_id": entity_id,
                    "query_minus_source_xy_px": delta.tolist(),
                    "distance_px": float(np.linalg.norm(delta)),
                }
            )
    return output


def _normalized_text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _load_context_receipt(root: Path) -> tuple[dict[str, Any] | None, str]:
    experience = root / "experience_context_projection_receipt.json"
    if experience.is_file():
        return _read_json(experience), "experience_context"
    legacy = root / "icl_projection_receipt.json"
    if legacy.is_file():
        receipt = _read_json(legacy)
        if receipt.get("icl_condition") == "video_only":
            return receipt, "video_only"
        return receipt, "legacy_fixed_demo"
    return None, "none"


def _first_robot_action_time(records: Sequence[Mapping[str, Any]]) -> datetime | None:
    for record in records:
        request = record.get("request")
        if isinstance(request, Mapping) and request.get("command") == "osc_sequence":
            return _parse_time(str(record.get("recorded_at", "")))
    return None


def _longest_contiguous_match(distances: np.ndarray, tolerance: float) -> int:
    previous = np.zeros(distances.shape[1] + 1, dtype=np.int32)
    longest = 0
    for row in distances:
        current = np.zeros_like(previous)
        for source_index, distance in enumerate(row, 1):
            if distance <= tolerance:
                current[source_index] = previous[source_index - 1] + 1
                longest = max(longest, int(current[source_index]))
        previous = current
    return longest


def _fraction_in_contiguous_matches(
    distances: np.ndarray,
    tolerance: float,
    *,
    minimum_run_length: int,
) -> float:
    """Fraction of Agent actions covered by a source-aligned matching run.

    Nearest-action overlap can be high when an action vocabulary contains many
    repeated saturated primitives.  Requiring four or more consecutive actions
    on one source-aligned diagonal is a stronger, still model-agnostic signal of
    trajectory reuse.
    """

    if minimum_run_length <= 0:
        raise ValueError("minimum_run_length must be positive")
    agent_count, source_count = distances.shape
    covered = np.zeros(agent_count, dtype=np.bool_)
    for offset in range(-(agent_count - 1), source_count):
        agent_start = max(0, -offset)
        source_start = max(0, offset)
        diagonal_length = min(
            agent_count - agent_start,
            source_count - source_start,
        )
        run_start = None
        for index in range(diagonal_length + 1):
            matches = (
                index < diagonal_length
                and distances[agent_start + index, source_start + index]
                <= tolerance
            )
            if matches and run_start is None:
                run_start = index
            elif not matches and run_start is not None:
                if index - run_start >= minimum_run_length:
                    covered[
                        agent_start + run_start : agent_start + index
                    ] = True
                run_start = None
    return float(np.mean(covered))


def _dtw_mean_cost(distances: np.ndarray) -> float:
    source_count = distances.shape[1]
    previous_cost = np.full(source_count + 1, np.inf)
    previous_length = np.zeros(source_count + 1, dtype=np.int32)
    previous_cost[0] = 0.0
    for row in distances:
        current_cost = np.full(source_count + 1, np.inf)
        current_length = np.zeros(source_count + 1, dtype=np.int32)
        for source_index, local_cost in enumerate(row, 1):
            candidates = (
                (previous_cost[source_index], previous_length[source_index]),
                (current_cost[source_index - 1], current_length[source_index - 1]),
                (
                    previous_cost[source_index - 1],
                    previous_length[source_index - 1],
                ),
            )
            best_cost, best_length = min(candidates, key=lambda item: item[0])
            current_cost[source_index] = best_cost + float(local_cost)
            current_length[source_index] = best_length + 1
        previous_cost = current_cost
        previous_length = current_length
    length = int(previous_length[-1])
    return float(previous_cost[-1] / length) if length else math.nan


def _action_array(value: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 7), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 7 or not np.isfinite(array).all():
        raise ValueError("action trajectory must be a finite Nx7 array")
    return array


def _agent_message_text(item: Mapping[str, Any]) -> str:
    parts = []
    content = item.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(parts)


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            records.append(value)
    return records
