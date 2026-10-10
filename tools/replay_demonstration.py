#!/usr/bin/env python3
"""Physically replay and verify one episode from a LIBERO HDF5 dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# GL selection must happen before importing LIBERO / robosuite.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from libero.libero.utils.demonstration_replay import (  # noqa: E402
    load_demonstration_episode,
    normalize_demo_key,
    run_action_replay,
    write_replay_report,
)
from libero.libero.agent_env.icl.fixed_demo import P4ReplayMasterRecorder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore one LIBERO HDF5 episode's initial MuJoCo state, execute its "
            "recorded OSC actions, and verify the result with the task checker."
        )
    )
    parser.add_argument("--dataset", required=True, help="Converted LIBERO HDF5 file")
    parser.add_argument(
        "--episode",
        default="demo_0",
        help="Episode index or HDF5 key (default: demo_0)",
    )
    parser.add_argument(
        "--bddl-file",
        help="Override the task BDDL; stale recorded paths are relocated automatically",
    )
    parser.add_argument(
        "--bddl-root",
        help="Alternative local bddl_files directory used for path relocation",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: outputs/replay/<dataset>/<episode>)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=10,
        help="No-motion controller steps after restoring the initial state",
    )
    parser.add_argument(
        "--stable-success-steps",
        type=int,
        default=10,
        help="Required consecutive successful action steps at the trajectory end",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Render a side-by-side agentview / wrist replay.mp4",
    )
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--video-fps", type=float)
    parser.add_argument(
        "--render-gpu-device-id",
        type=int,
        default=-1,
        help="EGL device index; -1 follows CUDA_VISIBLE_DEVICES (default: -1)",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Return exit status 0 even when stable checker verification fails",
    )
    parser.add_argument(
        "--p4-master-dir",
        help=(
            "Atomically publish a verified P4 replay master at this path. "
            "This enables RGB, metric depth, camera calibration, initial "
            "bbox/mask, state, proprioception, and causal post-action capture."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo_key = normalize_demo_key(args.episode)
    dataset = Path(args.dataset).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / "outputs" / "replay" / dataset.stem / demo_key).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    episode = load_demonstration_episode(
        dataset,
        demo_key,
        bddl_file=args.bddl_file,
        bddl_root=args.bddl_root,
    )
    video_path = output_dir / "replay.mp4" if args.save_video else None
    recorder = (
        P4ReplayMasterRecorder(
            args.p4_master_dir,
            episode,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )
        if args.p4_master_dir
        else None
    )
    master_receipt = None
    try:
        report = run_action_replay(
            episode,
            seed=args.seed,
            settle_steps=args.settle_steps,
            stable_success_steps=args.stable_success_steps,
            video_path=video_path,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            video_stride=args.video_stride,
            video_fps=args.video_fps,
            render_gpu_device_id=args.render_gpu_device_id,
            observation_callback=None if recorder is None else recorder.capture,
        )
        if recorder is not None and report["verified_success"]:
            master_receipt = recorder.finalize(report)
    finally:
        if recorder is not None:
            recorder.abort()
    report_path = write_replay_report(report, output_dir / "replay_report.json")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[report] {report_path}")
    if master_receipt is not None:
        print(json.dumps(master_receipt, indent=2, sort_keys=True))
        print(f"[p4-master] {master_receipt['master']}")
    if report["verified_success"]:
        print("[verified] physical action replay passed")
        return 0
    print("[failed] replay did not maintain final checker success")
    return 0 if args.allow_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
