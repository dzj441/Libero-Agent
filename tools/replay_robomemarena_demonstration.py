#!/usr/bin/env python3
"""Replay one seeded RoboMemArena trajectory and publish a verified P4 master."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--robomemarena-root",
        type=Path,
        help=(
            "Optional external RoboMemArena checkout; the frozen "
            "in-repository compatibility subset is the default"
        ),
    )
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument(
        "--p4-master-dir",
        type=Path,
        help="Optionally publish a verified evaluator-private replay master",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(__file__).resolve().parents[1]
    checkout_root = (
        None
        if args.robomemarena_root is None
        else args.robomemarena_root.expanduser().resolve()
    )

    # RoboMemArena owns the simulator package and assets. It must be activated
    # before importing this repository's agent modules under the shared package.
    bootstrap = (
        source_root
        / "libero"
        / "libero"
        / "agent_env"
        / "integrations"
        / "robomemarena"
        / "bootstrap.py"
    )
    activate_robomemarena_core = runpy.run_path(os.fspath(bootstrap))[
        "activate_robomemarena_core"
    ]

    activate_robomemarena_core(
        checkout_root=checkout_root,
        source_root=source_root,
    )

    from libero.libero.agent_env.icl.fixed_demo import P4ReplayMasterRecorder
    from libero.libero.agent_env.runtime.recording import PrivateRolloutVideoRecorder
    from libero.libero.agent_env.integrations.robomemarena.runtime import (
        RoboMemArenaOrderedStageEvaluator,
        get_robomemarena_task_spec,
        robomemarena_bddl_path,
        robomemarena_source_fingerprint,
        task4_init_state_id_from_recorded_instruction,
    )
    from tools.robomemarena_demo import (
        load_robomemarena_full_trajectory,
    )
    from libero.libero.envs import OffScreenRenderEnv

    trajectory = load_robomemarena_full_trajectory(
        args.dataset, expected_task_id=args.task_id
    )
    init_state_id = (
        task4_init_state_id_from_recorded_instruction(
            trajectory.recorded_instruction
        )
        if args.task_id == 4
        else 0
    )
    spec = get_robomemarena_task_spec(args.task_id)
    task_source = robomemarena_source_fingerprint(
        checkout_root,
        task_id=args.task_id,
        init_state_id=init_state_id,
    )
    bddl_path = robomemarena_bddl_path(
        checkout_root,
        task_id=args.task_id,
        init_state_id=init_state_id,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master_dir = (
        None
        if args.p4_master_dir is None
        else args.p4_master_dir.expanduser().resolve()
    )
    if master_dir is not None and (master_dir.exists() or master_dir.is_symlink()):
        raise FileExistsError(f"P4 replay master already exists: {master_dir}")

    # Official RoboMemArena evaluation seeds NumPy and the environment before
    # reset. The dataset filename records this seed instead of a MuJoCo state.
    np.random.seed(trajectory.seed)
    env = OffScreenRenderEnv(
        bddl_file_name=os.fspath(bddl_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        camera_depths=True,
        camera_segmentations="instance",
        use_object_obs=True,
        ignore_done=True,
        initialization_noise=None,
        render_gpu_device_id=args.render_gpu_device_id,
        horizon=len(trajectory.actions) + 1,
        control_freq=20,
        reward_shaping=True,
    )
    env.seed(trajectory.seed)
    raw_observation = env.reset()
    initial_state = np.asarray(env.get_sim_state(), dtype=np.float64)
    episode = SimpleNamespace(
        dataset_path=trajectory.dataset_path,
        demo_key=trajectory.demo_key,
        bddl_file=bddl_path,
        actions=trajectory.actions,
        init_state=initial_state,
        task_instruction=spec.instruction,
        problem_name=str(
            env.env.parsed_problem.get(
                "problem_name", f"robomemarena_task{args.task_id}"
            )
        ),
        env_name="RoboMemArena-OffScreenRenderEnv",
        robots=("Panda",),
        controller="OSC_POSE",
        control_freq=20,
        init_state_source=(
            f"seeded_reset:{trajectory.seed};init_state_id:{init_state_id}"
        ),
    )
    recorder = (
        None
        if master_dir is None
        else P4ReplayMasterRecorder(
            master_dir,
            episode,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            initial_frame_after_settle=False,
        )
    )
    video = (
        PrivateRolloutVideoRecorder(output_dir / "replay.mp4")
        if args.save_video
        else None
    )
    evaluator = RoboMemArenaOrderedStageEvaluator(
        env, task_id=args.task_id
    )
    evaluator.reset()
    first_success_step: int | None = None
    success_trace: list[bool] = []
    published = recorder is None
    try:
        if recorder is not None:
            recorder.capture(env, raw_observation, 0, None)
        if video is not None:
            video.append_raw_observation(raw_observation)
        for action_index, action in enumerate(trajectory.actions):
            raw_observation, _, _, _ = env.step(action)
            evaluator.observe(raw_observation)
            success = bool(evaluator.result()["success"])
            success_trace.append(success)
            if success and first_success_step is None:
                first_success_step = action_index
            if recorder is not None:
                recorder.capture(
                    env,
                    raw_observation,
                    action_index + 1,
                    action_index,
                )
            if video is not None:
                video.append_raw_observation(raw_observation)

        private_evaluation = evaluator.result()
        bddl_diagnostic_error = None
        try:
            bddl_success = bool(env.check_success())
        except Exception as exc:  # ordered stages are authoritative here
            bddl_success = None
            bddl_diagnostic_error = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        # RoboMemArena's ordered checker is the authoritative task contract.
        # Its counting tasks intentionally define success by completed pour
        # events even when the ordinary BDDL terminal-state proxy is false.
        verified = bool(private_evaluation["success"])
        final_streak = _ending_true_streak(success_trace)
        report: dict[str, Any] = {
            "schema_version": "libero.robomemarena_replay.v1",
            "verified_success": verified,
            "final_success": verified,
            "first_success_step": first_success_step,
            "final_success_streak": final_streak,
            "required_stable_success_steps": 1,
            "verification_authority": "robomemarena_ordered_stage_checker",
            "private_evaluation": {
                **private_evaluation,
                "bddl_final_goal_success": bddl_success,
                **(
                    {
                        "bddl_final_goal_diagnostic_error": (
                            bddl_diagnostic_error
                        )
                    }
                    if bddl_diagnostic_error is not None
                    else {}
                ),
            },
            "source": {
                "dataset": os.fspath(trajectory.dataset_path),
                "dataset_seed": trajectory.seed,
                "init_state_id": init_state_id,
                "task_variant": task_source["task_variant"],
                "dataset_recorded_instruction": trajectory.recorded_instruction,
                "raw_gripper_action_range": list(
                    trajectory.raw_gripper_action_range
                ),
                "gripper_action_clipped_to_contract": (
                    trajectory.gripper_action_clipped_to_contract
                ),
                "task_source": task_source,
            },
            "capture": {
                "profile": "level3",
                "frame_count": len(trajectory.actions) + 1,
                "transition_count": len(trajectory.actions),
                "initial_state_source": episode.init_state_source,
                "video": os.fspath((output_dir / "replay.mp4").resolve())
                if video is not None
                else None,
            },
        }
        # Preserve diagnostics for failed certification as well as success.
        # Without this receipt, ordered-stage failure and a BDDL final-goal
        # mismatch collapse into the same generic exception.
        _write_json(output_dir / "replay_report.json", report)
        if not verified:
            raise RuntimeError(
                "RoboMemArena replay did not pass the ordered-stage checker"
            )
        if recorder is not None:
            master_receipt = recorder.finalize(report)
            published = True
            report["p4_master"] = master_receipt
            _write_json(output_dir / "replay_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        if video is not None:
            video.close()
        env.close()
        if recorder is not None and not published:
            recorder.abort()


def _ending_true_streak(values: list[bool]) -> int:
    count = 0
    for value in reversed(values):
        if not value:
            break
        count += 1
    return count


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
