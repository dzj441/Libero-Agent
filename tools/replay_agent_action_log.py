#!/usr/bin/env python3
"""Fresh-replay native OSC batches recorded by an AgentEnv action log."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from libero.libero.agent_env import make_libero_agent_env  # noqa: E402
from libero.libero.agent_env.runtime.control import (  # noqa: E402
    validate_native_osc_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-action-log", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--allow-failure", action="store_true")
    return parser.parse_args()


def load_action_batches(path: str | Path) -> list[np.ndarray]:
    source = Path(path).expanduser().resolve()
    batches = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON at {source}:{line_number}"
            ) from exc
        request = event.get("request")
        if not isinstance(request, dict):
            continue
        if request.get("command") != "osc_sequence":
            continue
        try:
            batch = validate_native_osc_sequence(request.get("actions"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid OSC sequence at {source}:{line_number}: {exc}"
            ) from exc
        batches.append(batch)
    if not batches:
        raise ValueError(f"action log has no osc_sequence requests: {source}")
    return batches


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    source = args.source_action_log.expanduser().resolve()
    batches = load_action_batches(source)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    agent_env = make_libero_agent_env(
        suite=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        profile="level3",
        seed=args.seed,
        camera_height=args.camera_size,
        camera_width=args.camera_size,
        initial_settle_control_steps=10,
        max_agent_steps=None,
        native_sequence_submission_limit=None,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    try:
        episode = agent_env.start_episode()
        initial_state = np.asarray(agent_env.env.sim.get_state().flatten()).copy()
        model_xml = str(agent_env.env.sim.model.get_xml())
        action_count = 0
        for batch in batches:
            agent_env.step_osc_sequence(batch)
            action_count += len(batch)
        result = agent_env.finish_episode()
    finally:
        agent_env.close()

    report: dict[str, Any] = {
        "schema_version": "libero.agent_action_log_replay.v1",
        "verified_success": bool(result["success"]),
        "verification_authority": (
            "private_evaluator"
            if "private_evaluation" in result
            else "bddl_terminal"
        ),
        "suite": args.suite,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "seed": args.seed,
        "instruction": episode["task_instruction"],
        "source_action_log": os.fspath(source),
        "source_action_log_sha256": _sha256_file(source),
        "sequence_submission_count": len(batches),
        "native_osc_action_count": action_count,
        "initial_state_sha256": _sha256_array(initial_state),
        "model_xml_sha256": hashlib.sha256(model_xml.encode("utf-8")).hexdigest(),
        "state_forcing_after_reset": False,
        "result": result,
    }
    report_path = output / "replay_verification.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[report] {report_path}")
    return 0 if report["verified_success"] or args.allow_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
