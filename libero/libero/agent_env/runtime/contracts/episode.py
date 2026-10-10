"""Evaluator-private commitments for one Agent-controlled LIBERO run."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from ..control import (
    MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION,
    ActionInterface,
)
from ..profiles import ObservationProfile


SERVER_READY_SCHEMA_VERSION = "libero.agent_server_ready.v1"
CURRICULUM_SERVER_READY_SCHEMA_VERSION = (
    "libero.agent_curriculum_server_ready.v1"
)


def build_server_ready_contract(
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    task_instruction: str,
    profile: ObservationProfile | int | str,
    seed: int,
    resolution: int,
    render_gpu_device_id: int,
    initial_settle_control_steps: int,
    max_agent_steps: int | None,
    max_wall_time_seconds: float = 30 * 60,
    action_interface: ActionInterface | str,
    task_source_fingerprint: Mapping[str, Any] | None = None,
    max_resets: int = 0,
) -> dict[str, Any]:
    """Build the exact host/server contract checked before Codex is launched."""

    profile = ObservationProfile.parse(profile)
    action_interface = ActionInterface.parse(action_interface)
    if max_resets not in {0, 2}:
        raise ValueError("max_resets must be either 0 or 2")
    max_wall_time_seconds = float(max_wall_time_seconds)
    if not math.isfinite(max_wall_time_seconds) or max_wall_time_seconds <= 0:
        raise ValueError("max_wall_time_seconds must be finite and positive")
    normalized_instruction = " ".join(str(task_instruction).split())
    contract = {
        "schema_version": SERVER_READY_SCHEMA_VERSION,
        "transport": "unix_socket",
        "protocol": "libero.agent_unix_socket.v1",
        "suite": str(suite),
        "task_id": int(task_id),
        "init_state_id": int(init_state_id),
        "task_instruction_sha256": sha256_text(normalized_instruction),
        "observation_profile": profile.public_name,
        "seed": int(seed),
        "resolution": int(resolution),
        "render_gpu_device_id": int(render_gpu_device_id),
        "initial_settle_control_steps": int(initial_settle_control_steps),
        "max_agent_steps": (
            None if max_agent_steps is None else int(max_agent_steps)
        ),
        "action_interface": action_interface.value,
        "accepted_operations": (
            ["start", action_interface.wire_command, "reset", "finish"]
            if max_resets
            else ["start", action_interface.wire_command, "finish"]
        ),
        "max_resets": int(max_resets),
        "max_attempts": int(max_resets) + 1,
        "agent_step_budget_scope": "per_attempt",
        "max_wall_time_seconds": max_wall_time_seconds,
        "wall_time_budget_scope": "per_episode_across_resets",
        "max_native_osc_micro_steps_per_submission": (
            MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION
            if action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
            else None
        ),
        "observation_retention": "current_only",
        "observation_publication": "atomic_replace_before_response",
    }
    if task_source_fingerprint is not None:
        contract["task_source_fingerprint"] = dict(task_source_fingerprint)
    return contract


def validate_server_ready_contract(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Reject any startup configuration drift before an Agent can act."""

    if dict(actual) == dict(expected):
        return
    keys = sorted(set(actual) | set(expected))
    differing = [key for key in keys if actual.get(key) != expected.get(key)]
    raise RuntimeError(
        "LIBERO server ready contract differs from launcher request: "
        + ", ".join(differing)
    )


def build_curriculum_server_ready_contract(
    *,
    episodes: list[Mapping[str, Any]],
    profile: ObservationProfile | int | str,
    resolution: int,
    render_gpu_device_id: int,
    initial_settle_control_steps: int,
    max_agent_steps: int | None,
    max_wall_time_seconds: float = 30 * 60,
    action_interface: ActionInterface | str,
    max_resets: int = 0,
) -> dict[str, Any]:
    """Commit to an ordered multi-episode run before Codex is launched."""

    if not episodes:
        raise ValueError("curriculum must contain at least one episode")
    profile = ObservationProfile.parse(profile)
    action_interface = ActionInterface.parse(action_interface)
    if max_resets not in {0, 2}:
        raise ValueError("max_resets must be either 0 or 2")
    max_wall_time_seconds = float(max_wall_time_seconds)
    if not math.isfinite(max_wall_time_seconds) or max_wall_time_seconds <= 0:
        raise ValueError("max_wall_time_seconds must be finite and positive")
    committed_episodes = []
    for episode_index, episode in enumerate(episodes):
        instruction = " ".join(str(episode["task_instruction"]).split())
        episode_wall_time_seconds = float(
            episode.get("max_wall_time_seconds", max_wall_time_seconds)
        )
        if (
            not math.isfinite(episode_wall_time_seconds)
            or episode_wall_time_seconds <= 0
        ):
            raise ValueError(
                "episode max_wall_time_seconds must be finite and positive"
            )
        committed_episodes.append(
            {
                "episode_index": episode_index,
                "suite": str(episode["suite"]),
                "task_id": int(episode["task_id"]),
                "init_state_id": int(episode["init_state_id"]),
                "seed": int(episode["seed"]),
                "task_instruction_sha256": sha256_text(instruction),
                "max_agent_steps": int(
                    episode.get("max_agent_steps", max_agent_steps)
                ),
                "max_wall_time_seconds": episode_wall_time_seconds,
                "icl_condition": str(episode["icl_condition"]),
                "fixed_demo_master_manifest_sha256": episode.get(
                    "fixed_demo_master_manifest_sha256"
                ),
            }
        )
    return {
        "schema_version": CURRICULUM_SERVER_READY_SCHEMA_VERSION,
        "transport": "unix_socket",
        "protocol": "libero.agent_unix_socket.v1",
        "run_mode": "multi_episode_curriculum",
        "episode_count": len(committed_episodes),
        "episodes": committed_episodes,
        "observation_profile": profile.public_name,
        "resolution": int(resolution),
        "render_gpu_device_id": int(render_gpu_device_id),
        "initial_settle_control_steps": int(initial_settle_control_steps),
        "default_max_agent_steps_per_episode": (
            None if max_agent_steps is None else int(max_agent_steps)
        ),
        "episode_max_agent_steps": [
            episode["max_agent_steps"] for episode in committed_episodes
        ],
        "action_interface": action_interface.value,
        "accepted_operations": (
            ["start", action_interface.wire_command, "reset", "finish"]
            if max_resets
            else ["start", action_interface.wire_command, "finish"]
        ),
        "max_resets": int(max_resets),
        "max_attempts": int(max_resets) + 1,
        "agent_step_budget_scope": "per_attempt",
        "default_max_wall_time_seconds_per_episode": max_wall_time_seconds,
        "episode_max_wall_time_seconds": [
            episode["max_wall_time_seconds"] for episode in committed_episodes
        ],
        "wall_time_budget_scope": "per_episode_across_resets",
        "max_native_osc_micro_steps_per_submission": (
            MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION
            if action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
            else None
        ),
        "observation_retention": "current_only",
        "observation_publication": "atomic_replace_before_response",
        "next_task_disclosure": "start_response_only",
    }


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
