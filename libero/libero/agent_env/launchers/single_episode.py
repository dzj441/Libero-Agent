#!/usr/bin/env python3
"""Run one Codex-controlled LIBERO episode in an isolated workspace."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from libero.libero.benchmark import get_benchmark
from libero.libero.agent_env.runtime.control import (
    MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION,
    MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS,
    ActionInterface,
)
from libero.libero.agent_env.icl.fixed_demo import file_sha256, project_fixed_demo_bundle
from libero.libero.agent_env.icl.video_demo import project_video_only_demo_bundle
from libero.libero.agent_env.icl.experience_context import (
    load_experience_context_spec,
    project_experience_context_bundle,
)
from libero.libero.agent_env.icl.audit import audit_experience_context_run
from libero.libero.agent_env.harness.codex import (
    DEFAULT_CODEX_EFFORT,
    DEFAULT_CODEX_MODEL,
)
from libero.libero.agent_env.runtime.contracts.episode import (
    build_server_ready_contract,
    canonical_json_sha256,
    sha256_text,
    validate_server_ready_contract,
)
from libero.libero.agent_env.runtime.service import (
    DEFAULT_EPISODE_WALL_TIME_SECONDS,
    WALL_TIME_BUDGET_EXHAUSTED,
)
from libero.libero.agent_env.integrations.robomemarena.runtime import (
    ROBOMEMARENA_SUITE,
    get_robomemarena_task_spec,
    robomemarena_source_fingerprint,
)
from libero.libero.agent_env.runtime.contracts.release import validate_release_task


AGENT_ENV_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(__file__).resolve().parents[4]
CONTROL_TRANSPORTS = ("cli", "mcp")
CODEX_EXECUTION_MODES = ("exec", "interactive")
AGENT_ISOLATION_MODES = ("isolated", "debug_full_access")
ISOLATED_PERMISSION_PROFILE = "libero_agent_isolated"
MCP_SERVER_NAME = "libero"
MCP_TOOL_NAMES = (
    "start_episode",
    "osc_sequence",
    "finish_episode",
)
MCP_TOOL_NAMES_WITH_RESET = (
    "start_episode",
    "osc_sequence",
    "reset_episode",
    "finish_episode",
)
ICL_CONDITIONS = ("none", "fixed_demo", "video_only", "experience_context")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--profile", default="level3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--initial-settle-control-steps", type=int, default=10)
    parser.add_argument("--max-agent-steps", type=int, default=50)
    parser.add_argument(
        "--max-wall-time-seconds",
        type=float,
        default=float(DEFAULT_EPISODE_WALL_TIME_SECONDS),
        help=(
            "Evaluator wall-clock budget for the whole query, including all "
            "reset attempts (default: 1800 seconds)"
        ),
    )
    parser.add_argument(
        "--max-resets",
        type=int,
        choices=(0, 2),
        default=0,
        help="Reset condition: 0 disables reset; 2 gives three attempts total",
    )
    parser.add_argument(
        "--action-interface",
        choices=tuple(interface.value for interface in ActionInterface),
        default=ActionInterface.NATIVE_OSC_SEQUENCE.value,
        help="Mutually exclusive public robot-control condition",
    )
    parser.add_argument(
        "--control-transport",
        choices=CONTROL_TRANSPORTS,
        default="mcp",
        help="Agent-facing adapter over the same Unix-socket episode service",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="Parent for the isolated workspace (defaults to the system temp disk)",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Use a named persistent debug workspace instead of a random temp path",
    )
    parser.add_argument("--nvidia-runtime-root", type=Path)
    parser.add_argument("--server-ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--codex-effort", default=DEFAULT_CODEX_EFFORT)
    parser.add_argument(
        "--codex-execution-mode",
        choices=CODEX_EXECUTION_MODES,
        default="exec",
        help=(
            "Use one-shot codex exec, or open the interactive Codex TUI and "
            "let the operator paste the generated task prompt"
        ),
    )
    parser.add_argument(
        "--agent-isolation",
        choices=AGENT_ISOLATION_MODES,
        default="isolated",
        help=(
            "Filesystem/network boundary for the Codex process. The default "
            "isolated mode denies reads outside the generated workspace; "
            "debug_full_access preserves the legacy unrestricted launcher."
        ),
    )
    parser.add_argument(
        "--external-agent",
        action="store_true",
        help=(
            "Start and supervise only the LIBERO episode service. Print the "
            "workspace, prompt, and standard MCP config paths so a separately "
            "launched Agent can connect."
        ),
    )
    parser.add_argument(
        "--https-proxy",
        default=os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy", "")),
    )
    parser.add_argument(
        "--icl",
        choices=ICL_CONDITIONS,
        default="none",
        help="Static in-context demonstration condition",
    )
    parser.add_argument(
        "--fixed-demo-master",
        type=Path,
        help=(
            "Evaluator-private verified P4 replay master for --icl fixed_demo "
            "or --icl video_only"
        ),
    )
    parser.add_argument(
        "--experience-context-spec",
        type=Path,
        help=(
            "Evaluator-private multi-item context spec for "
            "--icl experience_context"
        ),
    )
    parser.add_argument(
        "--robomemarena-root",
        type=Path,
        help=(
            "Optional clean external RoboMemArena checkout for development; "
            "the frozen in-repository compatibility subset is the default"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_release_task(args.suite, args.task_id)
    if (
        not math.isfinite(args.max_wall_time_seconds)
        or args.max_wall_time_seconds <= 0
    ):
        raise ValueError("--max-wall-time-seconds must be finite and positive")
    if (
        args.agent_isolation == "isolated"
        and args.codex_execution_mode == "interactive"
    ):
        raise ValueError(
            "isolated mode requires non-interactive codex exec; use "
            "--agent-isolation debug_full_access for an explicitly manual TUI"
        )
    source_root = SOURCE_ROOT
    action_interface = ActionInterface.parse(args.action_interface)
    if (
        args.control_transport == "mcp"
        and action_interface is not ActionInterface.NATIVE_OSC_SEQUENCE
    ):
        raise ValueError(
            "the MCP adapter currently exposes only native_osc_sequence"
        )
    if args.external_agent and args.control_transport != "mcp":
        raise ValueError(
            "--external-agent currently requires --control-transport mcp"
        )
    if args.icl in {"fixed_demo", "video_only"} and args.fixed_demo_master is None:
        raise ValueError(
            f"--icl {args.icl} requires --fixed-demo-master"
        )
    if args.icl not in {"fixed_demo", "video_only"} and args.fixed_demo_master is not None:
        raise ValueError(
            "--fixed-demo-master is valid only with --icl fixed_demo or video_only"
        )
    if args.icl == "experience_context" and args.experience_context_spec is None:
        raise ValueError(
            "--icl experience_context requires --experience-context-spec"
        )
    if args.icl != "experience_context" and args.experience_context_spec is not None:
        raise ValueError(
            "--experience-context-spec is valid only with --icl experience_context"
        )
    task_source_fingerprint = None
    if args.suite == ROBOMEMARENA_SUITE:
        if args.icl not in {"none", "fixed_demo", "video_only"}:
            raise ValueError(
                "RoboMemArena supports no ICL or one verified fixed/video demo; "
                "experience-context projection is not yet integrated"
            )
        if args.robomemarena_root is not None:
            args.robomemarena_root = (
                args.robomemarena_root.expanduser().resolve()
            )
        task_source_fingerprint = robomemarena_source_fingerprint(
            args.robomemarena_root,
            task_id=args.task_id,
            init_state_id=args.init_state_id,
        )
    elif args.robomemarena_root is not None:
        raise ValueError(
            "--robomemarena-root is valid only for --suite robomemarena"
        )
    if (
        action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
        and args.max_agent_steps > MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS
    ):
        raise ValueError(
            "native_osc_sequence permits at most "
            f"{MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS} accepted submissions"
        )
    canonical_root = _canonical_repository_root(source_root)
    run_root = (args.run_root or canonical_root / "agent_runs").resolve()
    default_runtime_root = canonical_root / "runtime" / "nvidia"
    nvidia_runtime_root = (
        args.nvidia_runtime_root.resolve()
        if args.nvidia_runtime_root is not None
        else default_runtime_root.resolve() if default_runtime_root.is_dir()
        else None
    )
    run_id = args.run_id or _new_run_id()
    _validate_run_id(run_id)
    run_directory = run_root / run_id
    _create_new_directory(run_directory)
    try:
        workspace, system_temp_workspace = _allocate_workspace(
            canonical_root=canonical_root,
            requested_root=args.workspace_root,
            run_id=run_id,
            keep_workspace=args.keep_workspace,
        )
    except Exception:
        run_directory.rmdir()
        raise
    task_instruction = _task_instruction(args.suite, args.task_id)
    prompt = build_task_prompt(
        task_instruction,
        icl_condition=args.icl,
        action_interface=action_interface,
        control_transport=args.control_transport,
        max_agent_steps=args.max_agent_steps,
        max_wall_time_seconds=args.max_wall_time_seconds,
        max_resets=args.max_resets,
    )
    _prepare_workspace(
        workspace,
        prompt,
        icl_condition=args.icl,
        action_interface=action_interface,
        control_transport=args.control_transport,
        max_resets=args.max_resets,
        max_wall_time_seconds=args.max_wall_time_seconds,
    )
    shutil.copy2(workspace / "TASK_PROMPT.txt", run_directory / "agent_prompt.txt")
    shutil.copy2(
        workspace / ".libero" / "episode.json",
        run_directory / "agent_workspace_contract.json",
    )
    mcp_config_path: Path | None = None
    if args.control_transport == "mcp":
        mcp_config = build_external_mcp_config(
            workspace, max_resets=args.max_resets
        )
        mcp_config_path = run_directory / "agent_mcp_config.json"
        _write_json_atomic(workspace / ".libero" / "mcp.json", mcp_config)
        _write_json_atomic(mcp_config_path, mcp_config)
    icl_projection_receipt = None
    experience_context_projection_receipt = None
    if args.icl == "fixed_demo":
        icl_projection_receipt = project_fixed_demo_bundle(
            master_root=args.fixed_demo_master,
            destination=workspace / "benchmark_inputs" / "expert_demo",
            profile=args.profile,
            expected_task_instruction=task_instruction,
        )
        _write_json_atomic(
            run_directory / "icl_projection_receipt.json",
            icl_projection_receipt,
        )
    elif args.icl == "video_only":
        icl_projection_receipt = project_video_only_demo_bundle(
            master_root=args.fixed_demo_master,
            destination=workspace / "benchmark_inputs" / "expert_demo",
            expected_task_instruction=task_instruction,
        )
        _write_json_atomic(
            run_directory / "icl_projection_receipt.json",
            icl_projection_receipt,
        )
    elif args.icl == "experience_context":
        context_spec = load_experience_context_spec(
            args.experience_context_spec,
            artifact_root=source_root,
        )
        experience_context_projection_receipt = project_experience_context_bundle(
            spec=context_spec,
            destination=workspace / "benchmark_inputs" / "experience_context",
            profile=args.profile,
            target_task_instruction=task_instruction,
        )
        _write_json_atomic(
            run_directory / "experience_context_projection_receipt.json",
            experience_context_projection_receipt,
        )
        _archive_experience_context_contract(
            workspace / "benchmark_inputs" / "experience_context",
            run_directory / "experience_context_public_contract",
        )
    socket_path = workspace / ".libero" / "control.sock"
    server_ready_path = run_directory / "server_ready.json"
    server_environment, driver_version = _server_environment(
        source_root, nvidia_runtime_root
    )
    source_commit = _git_value(source_root, "rev-parse", "HEAD", required=False)
    source_status = _git_value(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        required=False,
    ) or ""
    expected_server_ready = build_server_ready_contract(
        suite=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        task_instruction=task_instruction,
        profile=args.profile,
        seed=args.seed,
        resolution=args.resolution,
        render_gpu_device_id=args.render_gpu_device_id,
        initial_settle_control_steps=args.initial_settle_control_steps,
        max_agent_steps=args.max_agent_steps,
        max_wall_time_seconds=args.max_wall_time_seconds,
        action_interface=action_interface,
        task_source_fingerprint=task_source_fingerprint,
        max_resets=args.max_resets,
    )
    run_configuration = {
        "suite": args.suite,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "profile": expected_server_ready["observation_profile"],
        "icl_condition": args.icl,
        "fixed_demo_available": args.icl == "fixed_demo",
        "video_only_demo_available": args.icl == "video_only",
        "experience_context_available": args.icl == "experience_context",
        "experience_context_id": (
            experience_context_projection_receipt["context_id"]
            if experience_context_projection_receipt is not None
            else None
        ),
        "seed": args.seed,
        "resolution": args.resolution,
        "render_gpu_device_id": args.render_gpu_device_id,
        "initial_settle_control_steps": args.initial_settle_control_steps,
        "max_agent_steps": args.max_agent_steps,
        "max_resets": args.max_resets,
        "max_attempts": args.max_resets + 1,
        "agent_step_budget_scope": "per_attempt",
        "max_wall_time_seconds": args.max_wall_time_seconds,
        "wall_time_budget_scope": "per_episode_across_resets",
        "action_interface": action_interface.value,
        "control_transport": args.control_transport,
        "task_source": (
            ROBOMEMARENA_SUITE
            if args.suite == ROBOMEMARENA_SUITE
            else "libero_official"
        ),
        "task_source_fingerprint": task_source_fingerprint,
        "agent_harness": "external" if args.external_agent else "codex",
        "codex_binary": None if args.external_agent else args.codex_bin,
        "codex_model_requested": None if args.external_agent else args.codex_model,
        "codex_effort_requested": None if args.external_agent else args.codex_effort,
        "codex_execution_mode": (
            None if args.external_agent else args.codex_execution_mode
        ),
        "agent_isolation": (
            None if args.external_agent else args.agent_isolation
        ),
        "max_native_osc_micro_steps_per_submission": (
            MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION
            if action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
            else None
        ),
    }
    prompt_sha256 = file_sha256(workspace / "TASK_PROMPT.txt")
    workspace_contract_sha256 = file_sha256(workspace / ".libero" / "episode.json")
    fixed_demo_manifest_sha256 = (
        None
        if args.icl not in {"fixed_demo", "video_only"}
        else file_sha256(
            workspace / "benchmark_inputs" / "expert_demo" / "manifest.json"
        )
    )
    experience_context_manifest_sha256 = (
        file_sha256(
            workspace
            / "benchmark_inputs"
            / "experience_context"
            / "manifest.json"
        )
        if args.icl == "experience_context"
        else None
    )
    configuration_fingerprint = {
        **run_configuration,
        "source_commit": source_commit,
        "render_backend": "egl",
        "nvidia_userspace_driver": driver_version,
        "operator_prompt_sha256": prompt_sha256,
        "workspace_contract_sha256": workspace_contract_sha256,
        "fixed_demo_manifest_sha256": fixed_demo_manifest_sha256,
        "experience_context_manifest_sha256": (
            experience_context_manifest_sha256
        ),
        "server_ready_contract_sha256": canonical_json_sha256(
            expected_server_ready
        ),
    }
    manifest = {
        "schema_version": "libero.agent_run_manifest.v1",
        "run_id": run_id,
        "created_at": _utc_now(),
        **run_configuration,
        "source_checkout": os.fspath(source_root),
        "source_commit": source_commit,
        "source_branch": _git_value(
            source_root, "branch", "--show-current", required=False
        ),
        "source_worktree_dirty": bool(source_status),
        "workspace": os.fspath(workspace),
        "workspace_lifecycle": (
            "system_temporary" if system_temp_workspace else "persistent_debug"
        ),
        "workspace_retained": True,
        "workspace_cleanup_owner": (
            "operating_system" if system_temp_workspace else "evaluator"
        ),
        "episode_resumable": False,
        "transport": "unix_socket",
        "agent_control_adapter": (
            "mcp_stdio" if args.control_transport == "mcp" else "liberoctl_cli"
        ),
        "observation_retention": "current_only",
        "render_backend": "egl",
        "nvidia_userspace_driver": driver_version,
        "server_ready_verified": False,
        "integrity": {
            "algorithm": "sha256",
            "configuration_sha256": canonical_json_sha256(
                configuration_fingerprint
            ),
            "operator_prompt_sha256": prompt_sha256,
            "workspace_contract_sha256": workspace_contract_sha256,
            "fixed_demo_manifest_sha256": fixed_demo_manifest_sha256,
            "experience_context_manifest_sha256": (
                experience_context_manifest_sha256
            ),
            "expected_server_ready_contract_sha256": canonical_json_sha256(
                expected_server_ready
            ),
            "source_worktree_status_sha256": sha256_text(source_status),
            "task_source_fingerprint_sha256": (
                None
                if task_source_fingerprint is None
                else canonical_json_sha256(task_source_fingerprint)
            ),
        },
    }
    if args.suite == ROBOMEMARENA_SUITE:
        manifest["robomemarena_source_kind"] = task_source_fingerprint[
            "source_kind"
        ]
        if args.robomemarena_root is not None:
            manifest["robomemarena_checkout"] = os.fspath(
                args.robomemarena_root
            )
    _write_json_atomic(run_directory / "run_manifest.json", manifest)

    server_command = [
        sys.executable,
        "-u",
        os.fspath(AGENT_ENV_ROOT / "runtime" / "episode_server.py"),
        "--suite",
        args.suite,
        "--task-id",
        str(args.task_id),
        "--init-state-id",
        str(args.init_state_id),
        "--profile",
        args.profile,
        "--seed",
        str(args.seed),
        "--resolution",
        str(args.resolution),
        "--render-gpu-device-id",
        str(args.render_gpu_device_id),
        "--initial-settle-control-steps",
        str(args.initial_settle_control_steps),
        "--max-agent-steps",
        str(args.max_agent_steps),
        "--max-wall-time-seconds",
        str(args.max_wall_time_seconds),
        "--max-resets",
        str(args.max_resets),
        "--action-interface",
        action_interface.value,
        "--workspace",
        os.fspath(workspace),
        "--socket",
        ".libero/control.sock",
        "--run-directory",
        os.fspath(run_directory),
        "--launcher-pid",
        str(os.getpid()),
    ]
    if (
        args.suite == ROBOMEMARENA_SUITE
        and args.robomemarena_root is not None
    ):
        server_command.extend(
            (
                "--robomemarena-root",
                os.fspath(args.robomemarena_root),
            )
        )
    server_log_path = run_directory / "server.log"
    codex_started_at = time.time()
    known_sessions = set() if args.external_agent else _session_files()
    server_log = server_log_path.open("w", encoding="utf-8")
    server_process: subprocess.Popen[Any] | None = None
    codex_process: subprocess.Popen[Any] | None = None
    codex_return_code: int | None = None
    server_return_code: int | None = None
    infrastructure_error: str | None = None
    session_archive_error: str | None = None
    archived_session: Path | None = None
    caught_exception: BaseException | None = None
    try:
        server_process = subprocess.Popen(
            server_command,
            cwd=source_root,
            env=server_environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        actual_server_ready = _wait_for_server_ready(
            socket_path,
            server_ready_path,
            server_process,
            expected_contract=expected_server_ready,
            timeout_s=args.server_ready_timeout_s,
            server_log_path=server_log_path,
        )
        manifest["server_ready_verified"] = True
        manifest["integrity"]["actual_server_ready_contract_sha256"] = (
            canonical_json_sha256(actual_server_ready)
        )
        _write_json_atomic(run_directory / "run_manifest.json", manifest)

        print(f"run_id: {run_id}", flush=True)
        print(f"workspace: {workspace}", flush=True)
        print(f"private_run: {run_directory}", flush=True)
        print(
            f"prompt_file: {run_directory / 'agent_prompt.txt'}",
            flush=True,
        )
        if mcp_config_path is not None:
            print(f"mcp_config_file: {mcp_config_path}", flush=True)
        if args.external_agent:
            print(
                "LIBERO server ready. Keep this launcher running, start the "
                "selected Agent from the workspace above, and load the MCP "
                "config above. The launcher exits after finish_episode; press "
                "Ctrl-C here to abort an unfinished episode.",
                flush=True,
            )
            server_return_code = server_process.wait()
        else:
            codex_environment = os.environ.copy()
            if args.agent_isolation == "isolated":
                # A parent shell (including this benchmark's development
                # environment) may export a permission-profile override such
                # as :danger-full-access.  Do not let it silently defeat the
                # launcher-selected least-privilege profile.
                for variable in (
                    "CODEX_PERMISSION_PROFILE",
                    "CODEX_SANDBOX",
                    "CODEX_APPROVAL_POLICY",
                ):
                    codex_environment.pop(variable, None)
            codex_environment["PATH"] = os.pathsep.join(
                (os.fspath(workspace / "bin"), codex_environment.get("PATH", ""))
            )
            codex_environment["LIBERO_CONTROL_SOCKET"] = ".libero/control.sock"
            codex_environment["LIBERO_AGENT_WORKSPACE"] = os.fspath(workspace)
            codex_environment["LIBERO_ACTION_INTERFACE"] = action_interface.value
            codex_environment["LIBERO_MAX_RESETS"] = str(args.max_resets)
            codex_environment["HTTPS_PROXY"] = args.https_proxy
            codex_command = build_codex_command(
                codex_bin=args.codex_bin,
                prompt=prompt,
                model=args.codex_model,
                effort=args.codex_effort,
                workspace=workspace,
                control_transport=args.control_transport,
                execution_mode=args.codex_execution_mode,
                agent_isolation=args.agent_isolation,
                max_resets=args.max_resets,
            )

            if args.codex_execution_mode == "interactive":
                print("\n----- BEGIN TASK PROMPT -----", flush=True)
                print(prompt.rstrip(), flush=True)
                print("----- END TASK PROMPT -----\n", flush=True)
                print(
                    "Starting interactive Codex CLI. Paste the complete task "
                    "prompt above as the first message.",
                    flush=True,
                )
            else:
                print("starting Codex CLI...", flush=True)
            codex_process = subprocess.Popen(
                codex_command,
                cwd=workspace,
                env=codex_environment,
                stdin=(
                    None
                    if args.codex_execution_mode == "interactive"
                    else subprocess.DEVNULL
                ),
            )

            timed_out_server_exit_at: float | None = None
            while codex_process.poll() is None:
                server_return_code = server_process.poll()
                if server_return_code is not None:
                    result = _read_json(run_directory / "result.json")
                    if result.get("status") != "finished":
                        infrastructure_error = (
                            "LIBERO server exited before finish "
                            f"with code {server_return_code}"
                        )
                        _terminate_process(codex_process)
                        break
                    if (
                        result.get("termination_reason")
                        == WALL_TIME_BUDGET_EXHAUSTED
                    ):
                        if timed_out_server_exit_at is None:
                            timed_out_server_exit_at = time.monotonic()
                        elif time.monotonic() - timed_out_server_exit_at >= 5.0:
                            # The authoritative result is already durable. Give
                            # the Agent a brief chance to consume the timeout
                            # response, then prevent post-episode reasoning from
                            # making a bounded evaluation run unbounded.
                            _terminate_process(codex_process)
                            break
                time.sleep(0.25)

            codex_return_code = codex_process.wait()
            if server_process.poll() is None:
                result = _read_json(run_directory / "result.json")
                if result.get("status") == "finished":
                    try:
                        server_process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        _terminate_process_group(server_process)
                else:
                    _terminate_process_group(server_process)
            server_return_code = server_process.wait()
    except BaseException as exc:
        infrastructure_error = f"{type(exc).__name__}: {exc}"
        caught_exception = exc
        if codex_process is not None and codex_process.poll() is None:
            _terminate_process(codex_process)
        if server_process is not None and server_process.poll() is None:
            _terminate_process_group(server_process)
        codex_return_code = (
            None if codex_process is None else codex_process.poll()
        )
        server_return_code = (
            None if server_process is None else server_process.poll()
        )
    finally:
        server_log.close()
        if not args.external_agent:
            try:
                archived_session = _copy_codex_session(
                    workspace=workspace,
                    run_directory=run_directory,
                    known_sessions=known_sessions,
                    started_at=codex_started_at,
                )
                if archived_session is not None:
                    _archive_viewed_artifacts(
                        session_path=archived_session,
                        workspace=workspace,
                        run_directory=run_directory,
                    )
            except BaseException as exc:
                session_archive_error = f"{type(exc).__name__}: {exc}"
                if infrastructure_error is None:
                    infrastructure_error = (
                        f"session archival failed: {session_archive_error}"
                    )
                if caught_exception is None:
                    caught_exception = exc
    result_path = run_directory / "result.json"
    result = _read_json(result_path)
    codex_session_infrastructure_error = None
    if (
        archived_session is not None
        and codex_return_code not in (None, 0)
        and result.get("status") != "finished"
    ):
        codex_session_infrastructure_error = (
            _codex_infrastructure_error_from_session(archived_session)
        )
        if (
            codex_session_infrastructure_error is not None
            and infrastructure_error is None
        ):
            infrastructure_error = codex_session_infrastructure_error
    if caught_exception is not None and not result:
        result = {
            "schema_version": "libero.agent_run_result.v1",
            "status": "infrastructure_error",
            "reason": infrastructure_error,
            "finished_at": _utc_now(),
        }
    if codex_session_infrastructure_error is not None:
        result["status"] = "infrastructure_error"
        result["reason"] = codex_session_infrastructure_error
    elif (
        not args.external_agent
        and result.get("status") == "aborted"
        and codex_return_code is not None
    ):
        result["reason"] = "codex_process_exited_before_finish"
    result.update(
        {
            "codex_exit_code": codex_return_code,
            "agent_harness": "external" if args.external_agent else "codex",
            "server_exit_code": server_return_code,
            "launcher_finished_at": _utc_now(),
            "infrastructure_error": infrastructure_error,
            "session_archive_error": session_archive_error,
        }
    )
    _write_json_atomic(result_path, result)
    context_audit_error = None
    if args.icl in {"fixed_demo", "video_only", "experience_context"}:
        try:
            context_audit = audit_experience_context_run(run_directory)
            _write_json_atomic(
                run_directory / "experience_context_audit.json", context_audit
            )
        except BaseException as exc:
            context_audit_error = f"{type(exc).__name__}: {exc}"
    result["context_audit_error"] = context_audit_error
    _write_json_atomic(result_path, result)
    if caught_exception is not None:
        print(infrastructure_error, file=sys.stderr, flush=True)
        return 2
    if infrastructure_error is not None or result.get("status") != "finished":
        return 2
    if args.external_agent:
        return 0 if server_return_code == 0 else 2
    if result.get("termination_reason") == WALL_TIME_BUDGET_EXHAUSTED:
        return 0
    return 0 if codex_return_code == 0 else 2


def build_task_prompt(
    task_instruction: str,
    *,
    icl_condition: str = "none",
    action_interface: ActionInterface | str = ActionInterface.METRIC_OSC_STEP,
    control_transport: str = "cli",
    max_agent_steps: int = 50,
    max_wall_time_seconds: float = float(DEFAULT_EPISODE_WALL_TIME_SECONDS),
    max_resets: int = 0,
) -> str:
    instruction = " ".join(str(task_instruction).split())
    action_interface = ActionInterface.parse(action_interface)
    if control_transport not in CONTROL_TRANSPORTS:
        raise ValueError(f"unsupported control transport: {control_transport!r}")
    if (
        control_transport == "mcp"
        and action_interface is not ActionInterface.NATIVE_OSC_SEQUENCE
    ):
        raise ValueError("MCP currently requires native_osc_sequence")
    if icl_condition not in ICL_CONDITIONS:
        raise ValueError(f"unsupported ICL condition: {icl_condition!r}")
    if not 1 <= int(max_agent_steps) <= MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS:
        raise ValueError(
            "max_agent_steps must be between 1 and "
            f"{MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS}"
        )
    if int(max_resets) not in {0, 2}:
        raise ValueError("max_resets must be either 0 or 2")
    max_wall_time_seconds = float(max_wall_time_seconds)
    if not math.isfinite(max_wall_time_seconds) or max_wall_time_seconds <= 0:
        raise ValueError("max_wall_time_seconds must be finite and positive")
    wall_time_notice = (
        f"This query has a {max_wall_time_seconds:g}-second evaluator wall-clock "
        "budget shared by all reset attempts. Exceeding it ends the query as "
        "an official task failure."
    )
    icl_notice = ""
    if icl_condition == "fixed_demo":
        compatibility_notice = ""
        if action_interface is ActionInterface.NATIVE_OSC_SEQUENCE:
            action_reference = (
                "one `osc_sequence` micro action"
                if control_transport == "mcp"
                else "one `osc-sequence` micro action"
            )
            compatibility_notice = (
                " Each source action vector has the same component semantics as "
                f"{action_reference}."
            )
        icl_notice = (
            "\nA verified successful demonstration from a separate episode of "
            "the same task is available at `benchmark_inputs/expert_demo/`. "
            "The current scene configuration and object or goal poses may differ. "
            "The demonstration records the expert's native per-control-cycle "
            "OSC_POSE actions and measured EEF state observations. The measured "
            "EEF poses are observations, not actions."
            f"{compatibility_notice}\n"
        )
    elif icl_condition == "video_only":
        icl_notice = (
            "\nA verified successful demonstration from a separate episode of the "
            "same task is available at `benchmark_inputs/expert_demo/` as RGB "
            "video only (head and wrist views; decoder-free contact sheets are in "
            "`video/contact_sheets/`). The current scene configuration "
            "and object or goal poses may differ. No actions, states, depth, "
            "camera calibration, or annotations are provided.\n"
        )
    elif icl_condition == "experience_context":
        icl_notice = (
            "\nOne or more embodied experiences are available at "
            "`benchmark_inputs/experience_context/`. Public item manifests "
            "describe each source task, outcome, and available modality.\n"
        )
    if control_transport == "mcp":
        start_instruction = (
            "1. Call the `start_episode` robot tool exactly once to begin and "
            "receive the initial observation and `max_agent_steps` budget."
        )
        control_instruction = (
            "2. Control the robot with the `osc_sequence` robot tool. Its "
            "`actions` argument is an array of 1 to "
            f"{MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION} normalized 7D OSC_POSE "
            "micro actions in `[dx, dy, dz, rx, ry, rz, gripper]` order. Every "
            "component must be within [-1, 1]. Each vector executes one LIBERO "
            "policy interval; translation 1.0 corresponds to 0.05 m, rotation "
            "1.0 to a 0.5 rad rotation-vector component, gripper -1 opens, and "
            "+1 closes. One sequence call counts as one Agent action, with at "
            f"most {int(max_agent_steps)} accepted calls."
        )
        if max_resets:
            reset_instruction = (
                "3. If the current attempt is unrecoverably disordered, you may call "
                "the `reset_episode` robot tool at most twice. It destroys the current "
                "rollout, recreates the same task from its initial state, and returns a "
                "new initial observation and id; retain useful knowledge only in your "
                "conversation context."
            )
            inspect_instruction = (
                "4. Wait for each action or reset to complete, then inspect "
                "`benchmark_inputs/current_observation/observation.json` and any "
                "referenced files before issuing another command."
            )
            finish_instruction = (
                "5. When you have completed the task, call the `finish_episode` robot "
                "tool exactly once. Only finish reports official task success."
            )
        else:
            reset_instruction = ""
            inspect_instruction = (
                "3. Wait for each action to complete, then inspect "
                "`benchmark_inputs/current_observation/observation.json` and any "
                "referenced files before issuing another command."
            )
            finish_instruction = (
                "4. When you have completed the task, call the `finish_episode` robot "
                "tool exactly once. Only finish reports official task success."
            )
    elif action_interface is ActionInterface.METRIC_OSC_STEP:
        start_instruction = (
            "1. Run `liberoctl start` exactly once to begin and receive the "
            "initial observation."
        )
        control_instruction = (
            "2. Control the robot with `liberoctl osc-step --position DX DY DZ "
            "--rotation RX RY RZ --gripper-delta-m DG`. Each command specifies "
            "a metric Cartesian target delta executed through LIBERO's OSC_POSE "
            "controller. Position deltas are robot-base-frame metres. Rotation "
            "deltas are robot-base-frame rotation vectors in radians. DG is the "
            "change in total jaw opening width in metres: positive opens, negative "
            "closes, and zero preserves the current gripper target and grip force. "
            "A target outside the physical gripper-width range is rejected."
        )
        reset_instruction, inspect_instruction, finish_instruction = (
            _cli_reset_prompt(max_resets)
        )
    else:
        start_instruction = (
            "1. Run `liberoctl start` exactly once to begin and receive the "
            "initial observation."
        )
        control_instruction = (
            "2. Control the robot with `liberoctl osc-sequence --actions-file "
            "PATH`. PATH must contain a JSON array of 1 to "
            f"{MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION} normalized 7D OSC_POSE "
            "micro actions in `[dx, dy, dz, rx, ry, rz, gripper]` order. Every "
            "component must be within [-1, 1]. Each vector executes one LIBERO "
            "policy interval; translation 1.0 corresponds to 0.05 m, rotation "
            "1.0 to a 0.5 rad rotation-vector component, gripper -1 opens, and "
            "+1 closes. One sequence submission counts as one Agent action, with "
            f"at most {int(max_agent_steps)} accepted submissions."
        )
        reset_instruction, inspect_instruction, finish_instruction = (
            _cli_reset_prompt(max_resets)
        )
    return f"""{instruction}
{icl_notice}

A LIBERO episode has been prepared for you.

{wall_time_notice}

{start_instruction}
{control_instruction}
{reset_instruction + chr(10) if reset_instruction else ""}{inspect_instruction}
{finish_instruction}
"""


def _cli_reset_prompt(max_resets: int) -> tuple[str, str, str]:
    if max_resets:
        return (
            "3. If the current attempt is unrecoverably disordered, you may run "
            "`liberoctl reset` at most twice. It recreates the same task from its "
            "initial state and returns a new initial observation and id; retain "
            "useful knowledge only in your conversation context.",
            "4. Wait for each action or reset to complete, then inspect "
            "`benchmark_inputs/current_observation/observation.json` and any "
            "referenced files before issuing another command.",
            "5. When you have completed the task, run `liberoctl finish` exactly "
            "once. Only finish reports official task success.",
        )
    return (
        "",
        "3. Wait for each action to complete, then inspect "
        "`benchmark_inputs/current_observation/observation.json` and any "
        "referenced files before issuing another command.",
        "4. When you have completed the task, run `liberoctl finish` exactly "
        "once. Only finish reports official task success.",
    )


def codex_native_runtime_paths(codex_bin: str) -> tuple[Path, ...]:
    """Find the native CLI runtime without granting access to its user home."""
    resolved = shutil.which(codex_bin)
    if resolved is None:
        return ()
    executable = Path(resolved).resolve()
    if executable.suffix == ".js":
        paths = executable.parent.parent.glob(
            "node_modules/@openai/codex-linux-*/vendor/*/bin/codex"
        )
        return tuple(path.resolve() for path in paths if path.is_file())
    return (executable,) if executable.is_file() else ()


def build_codex_command(
    *,
    codex_bin: str,
    prompt: str,
    model: str | None = None,
    effort: str | None = None,
    workspace: Path | None = None,
    control_transport: str = "cli",
    execution_mode: str = "exec",
    agent_isolation: str = "isolated",
    max_resets: int = 0,
) -> list[str]:
    """Build a persistent Codex CLI invocation for one episode."""

    if execution_mode not in CODEX_EXECUTION_MODES:
        raise ValueError(
            f"unsupported Codex execution mode: {execution_mode!r}"
        )
    if agent_isolation not in AGENT_ISOLATION_MODES:
        raise ValueError(
            f"unsupported Agent isolation mode: {agent_isolation!r}"
        )

    command = [codex_bin]
    if execution_mode == "exec":
        command.append("exec")
    if agent_isolation == "debug_full_access":
        command.extend(
            (
                "--dangerously-bypass-approvals-and-sandbox",
                "--dangerously-bypass-hook-trust",
            )
        )
    else:
        # The profile is passed as an inline TOML value so each generated
        # workspace is the only runtime workspace root.  `:root=deny` is
        # essential: Codex's ordinary workspace-write mode protects writes,
        # but otherwise leaves unrelated files readable.
        runtime_rules = "".join(
            f'{json.dumps(os.fspath(path))}="read",'
            for path in codex_native_runtime_paths(codex_bin)
        )
        command.extend(
            (
                "--config",
                f'default_permissions="{ISOLATED_PERMISSION_PROFILE}"',
                "--config",
                'approval_policy="never"',
                "--config",
                (
                    f"permissions.{ISOLATED_PERMISSION_PROFILE}.filesystem="
                    '{":root"="deny",":minimal"="read",'
                    + runtime_rules
                    +
                    '":workspace_roots"={"."="write"}}'
                ),
            )
        )
        # Do not load user config or user/project execpolicy rules which can
        # alter command execution. The generated workspace has no project
        # config, and the source checkout containing any such config is hidden
        # by the profile. Auth is intentionally still read from CODEX_HOME by
        # the Codex CLI; only configuration is ignored.
        if execution_mode == "exec":
            # This option is currently exposed by `codex exec`, not by the
            # interactive top-level command. Interactive mode is explicitly
            # operator-facing and keeps the selected profile overrides above.
            command.extend(("--ignore-user-config", "--ignore-rules"))
    if execution_mode == "exec":
        command.extend(("--skip-git-repo-check", "--color", "never"))
    else:
        command.append("--no-alt-screen")
    if model:
        command.extend(("--model", model))
    if effort:
        command.extend(("--config", f'model_reasoning_effort="{effort}"'))
    if control_transport == "mcp":
        if workspace is None:
            raise ValueError("MCP Codex launch requires the Agent workspace")
        mcp_server = (workspace / "bin" / "libero_mcp_server").resolve()
        if agent_isolation == "isolated":
            socket_path = (workspace / ".libero" / "control.sock").resolve()
            # Isolation is deliberately local-filesystem-only. Keep network
            # access (including normal hosted/image tools) available while
            # still recording the evaluator socket explicitly.
            command.extend(
                (
                    "--config",
                    (
                        f"permissions.{ISOLATED_PERMISSION_PROFILE}.network="
                        '{"enabled"=true,"unix_sockets"={'
                        + json.dumps(os.fspath(socket_path))
                        + '="allow"}}'
                    ),
                )
            )
        command.extend(
            (
                "--config",
                f'mcp_servers.{MCP_SERVER_NAME}.command={json.dumps(os.fspath(mcp_server))}',
                "--config",
                f'mcp_servers.{MCP_SERVER_NAME}.cwd={json.dumps(os.fspath(workspace))}',
                "--config",
                f'mcp_servers.{MCP_SERVER_NAME}.required=true',
                "--config",
                f'mcp_servers.{MCP_SERVER_NAME}.startup_timeout_sec=10',
                "--config",
                f'mcp_servers.{MCP_SERVER_NAME}.tool_timeout_sec=600',
                "--config",
                f'mcp_servers.{MCP_SERVER_NAME}.default_tools_approval_mode="auto"',
                "--config",
                (
                    f'mcp_servers.{MCP_SERVER_NAME}.enabled_tools='
                    + json.dumps(list(_mcp_tool_names(max_resets)))
                ),
                "--config",
                (
                    f'mcp_servers.{MCP_SERVER_NAME}.env_vars='
                    + json.dumps(
                        [
                            "LIBERO_AGENT_WORKSPACE",
                            "LIBERO_CONTROL_SOCKET",
                            "LIBERO_MAX_RESETS",
                        ]
                    )
                ),
            )
        )
    elif control_transport != "cli":
        raise ValueError(f"unsupported control transport: {control_transport!r}")
    if execution_mode == "exec":
        command.append(prompt)
    return command


def build_external_mcp_config(
    workspace: Path, *, max_resets: int = 0
) -> dict[str, Any]:
    """Return a client-neutral STDIO MCP config for one prepared episode."""

    _mcp_tool_names(max_resets)
    root = workspace.expanduser().resolve()
    return {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "command": os.fspath(root / "bin" / "libero_mcp_server"),
                "args": [],
                "env": {
                    "LIBERO_AGENT_WORKSPACE": os.fspath(root),
                    "LIBERO_CONTROL_SOCKET": os.fspath(
                        root / ".libero" / "control.sock"
                    ),
                    "LIBERO_MAX_RESETS": str(int(max_resets)),
                },
            }
        }
    }


def _task_instruction(suite: str, task_id: int) -> str:
    if suite == ROBOMEMARENA_SUITE:
        return get_robomemarena_task_spec(task_id).instruction
    benchmark_class = get_benchmark(suite)
    task_suite = benchmark_class()
    if not 0 <= task_id < task_suite.get_num_tasks():
        raise ValueError(
            f"task_id must be in [0, {task_suite.get_num_tasks()}), got {task_id}"
        )
    return " ".join(task_suite.get_task(task_id).language.split())


def _prepare_workspace(
    workspace: Path,
    prompt: str,
    *,
    icl_condition: str,
    action_interface: ActionInterface | str,
    control_transport: str = "cli",
    max_resets: int = 0,
    max_wall_time_seconds: float = float(DEFAULT_EPISODE_WALL_TIME_SECONDS),
) -> None:
    action_interface = ActionInterface.parse(action_interface)
    if int(max_resets) not in {0, 2}:
        raise ValueError("max_resets must be either 0 or 2")
    max_wall_time_seconds = float(max_wall_time_seconds)
    if not math.isfinite(max_wall_time_seconds) or max_wall_time_seconds <= 0:
        raise ValueError("max_wall_time_seconds must be finite and positive")
    if control_transport not in CONTROL_TRANSPORTS:
        raise ValueError(f"unsupported control transport: {control_transport!r}")
    (workspace / ".libero").mkdir(mode=0o700)
    (workspace / "benchmark_inputs").mkdir()
    (workspace / "scratch").mkdir()
    binary_directory = workspace / "bin"
    binary_directory.mkdir()
    if control_transport == "cli":
        client = binary_directory / "liberoctl"
        shutil.copy2(
            AGENT_ENV_ROOT / "runtime" / "cli_client.py",
            client,
        )
        client.chmod(0o755)
    else:
        mcp_server = binary_directory / "libero_mcp_server"
        shutil.copy2(
            AGENT_ENV_ROOT / "runtime" / "mcp_server.py",
            mcp_server,
        )
        mcp_server.chmod(0o755)
    (workspace / "TASK_PROMPT.txt").write_text(prompt, encoding="utf-8")
    _write_json_atomic(
        workspace / ".libero" / "episode.json",
        {
            "schema_version": "libero.agent_workspace.v1",
            "episode_resumable": False,
            "operations": (
                list(_mcp_tool_names(max_resets))
                if control_transport == "mcp"
                else (
                    ["start", action_interface.wire_command, "reset", "finish"]
                    if max_resets
                    else ["start", action_interface.wire_command, "finish"]
                )
            ),
            "action_interface": action_interface.value,
            "control_transport": control_transport,
            "max_native_osc_micro_steps_per_submission": (
                MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION
                if action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
                else None
            ),
            "observation_retention": "current_only",
            "max_resets": int(max_resets),
            "max_attempts": int(max_resets) + 1,
            "agent_step_budget_scope": "per_attempt",
            "max_wall_time_seconds": max_wall_time_seconds,
            "wall_time_budget_scope": "per_episode_across_resets",
            "icl_condition": icl_condition,
            "expert_demo": (
                "benchmark_inputs/expert_demo"
                if icl_condition in {"fixed_demo", "video_only"}
                else None
            ),
            "experience_context": (
                "benchmark_inputs/experience_context"
                if icl_condition == "experience_context"
                else None
            ),
        },
    )


def _mcp_tool_names(max_resets: int) -> tuple[str, ...]:
    if int(max_resets) not in {0, 2}:
        raise ValueError("max_resets must be either 0 or 2")
    return MCP_TOOL_NAMES_WITH_RESET if max_resets else MCP_TOOL_NAMES


def _archive_experience_context_contract(
    bundle_root: Path, destination: Path
) -> None:
    """Preserve public manifests without copying the large context payload."""

    root = bundle_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    destination.mkdir(parents=True)
    shutil.copy2(manifest_path, destination / "manifest.json")
    for item in manifest.get("experiences", []):
        if not isinstance(item, dict):
            raise ValueError("experience-context item summary is invalid")
        experience_id = item.get("experience_id")
        artifact = item.get("manifest")
        relative = artifact.get("path") if isinstance(artifact, dict) else None
        if (
            not isinstance(experience_id, str)
            or not isinstance(relative, str)
            or Path(relative).is_absolute()
        ):
            raise ValueError("experience-context manifest reference is invalid")
        source = (root / relative).resolve()
        if (
            os.path.commonpath((root, source)) != os.fspath(root)
            or not source.is_file()
        ):
            raise ValueError("experience-context manifest escapes its bundle")
        shutil.copy2(source, destination / f"{experience_id}.json")


def _server_environment(
    source_root: Path, nvidia_runtime_root: Path | None
) -> tuple[dict[str, str], str]:
    environment = os.environ.copy()
    try:
        driver_version = subprocess.check_output(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"),
            text=True, stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        driver_version = "system"
    environment["MUJOCO_GL"] = "egl"
    environment["PYOPENGL_PLATFORM"] = "egl"
    if nvidia_runtime_root is not None:
        driver_directory = nvidia_runtime_root / driver_version
        library_directory = driver_directory / "runtime-libs-full"
        egl_manifest = driver_directory / "10_nvidia.local.json"
        if not library_directory.is_dir() or not egl_manifest.is_file():
            raise FileNotFoundError(
                f"matching EGL userspace stack is unavailable for NVIDIA {driver_version}"
            )
        environment["__EGL_VENDOR_LIBRARY_FILENAMES"] = os.fspath(egl_manifest)
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            (os.fspath(library_directory), environment.get("LD_LIBRARY_PATH", ""))
        ).rstrip(os.pathsep)
    environment["PYTHONPATH"] = os.pathsep.join(
        (os.fspath(source_root), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    return environment, driver_version


def _canonical_repository_root(source_root: Path) -> Path:
    value = _git_value(
        source_root, "rev-parse", "--path-format=absolute", "--git-common-dir",
        required=False,
    )
    if value is None:
        return source_root
    common_git = Path(value).resolve()
    return common_git.parent if common_git.name == ".git" else source_root


def _wait_for_server_ready(
    socket_path: Path,
    ready_path: Path,
    process: subprocess.Popen[Any],
    *,
    expected_contract: Mapping[str, Any],
    timeout_s: float,
    server_log_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"LIBERO server exited during startup with code {return_code}; "
                f"see {server_log_path}"
            )
        try:
            socket_ready = stat.S_ISSOCK(socket_path.stat().st_mode)
        except FileNotFoundError:
            socket_ready = False
        if socket_ready and ready_path.is_file():
            actual = _read_json(ready_path)
            validate_server_ready_contract(actual, expected_contract)
            return actual
        time.sleep(0.2)
    raise TimeoutError(
        "LIBERO server did not publish a verified ready contract within "
        f"{timeout_s}s"
    )


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _validate_run_id(run_id: str) -> None:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if not run_id or len(run_id) > 96 or any(character not in allowed for character in run_id):
        raise ValueError("run_id must contain only letters, digits, '-' and '_'")


def _create_new_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()


def _allocate_workspace(
    *,
    canonical_root: Path,
    requested_root: Path | None,
    run_id: str,
    keep_workspace: bool,
) -> tuple[Path, bool]:
    if keep_workspace:
        root = (
            requested_root
            or canonical_root.parent / "agent_workspaces" / "libero"
        ).expanduser().resolve()
        workspace = root / run_id
        _create_new_directory(workspace)
        return workspace, False
    parent = requested_root.expanduser().resolve() if requested_root else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(
            prefix="libero-agent-workspace-",
            dir=os.fspath(parent) if parent is not None else None,
        )
    ).resolve()
    return workspace, True


def _terminate_process(process: subprocess.Popen[Any], timeout_s: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _terminate_process_group(
    process: subprocess.Popen[Any], timeout_s: float = 15.0
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _session_files() -> set[Path]:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    sessions_root = codex_home / "sessions"
    return set(sessions_root.rglob("*.jsonl")) if sessions_root.is_dir() else set()


def _copy_codex_session(
    *,
    workspace: Path,
    run_directory: Path,
    known_sessions: set[Path],
    started_at: float,
) -> Path | None:
    candidates = []
    for path in _session_files():
        try:
            if path not in known_sessions or path.stat().st_mtime >= started_at - 2.0:
                metadata = _session_metadata(path)
                if Path(metadata.get("cwd", "")).resolve() == workspace:
                    candidates.append((path.stat().st_mtime, path, metadata))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    _modified, source, metadata = max(candidates, key=lambda item: item[0])
    shutil.copy2(source, run_directory / "codex_session.jsonl")
    _write_json_atomic(
        run_directory / "codex_session_metadata.json",
        {
            "schema_version": "libero.codex_session_reference.v1",
            "session_id": metadata.get("session_id") or metadata.get("id"),
            "cwd": metadata.get("cwd"),
            "source_file": os.fspath(source),
            "episode_resumable": False,
        },
    )
    return run_directory / "codex_session.jsonl"


def _codex_infrastructure_error_from_session(session_path: Path) -> str | None:
    """Return a public-safe Codex service failure recorded by the CLI session."""

    infrastructure_error_codes = {
        "usage_limit_exceeded",
        "rate_limit_exceeded",
        "stream_disconnected",
        "service_unavailable",
        "server_overloaded",
    }
    infrastructure_message_fragments = (
        "usage limit",
        "rate limit",
        "stream disconnected",
        "error sending request",
        "service unavailable",
        "connection reset",
        "connection timed out",
        "request timed out",
        "temporarily unavailable",
        "model is at capacity",
        "server overloaded",
    )
    detected: str | None = None
    try:
        lines = session_path.open("r", encoding="utf-8")
    except OSError:
        return None
    with lines:
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "event_msg":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict) or payload.get("type") != "task_complete":
                continue
            error = payload.get("error")
            if not isinstance(error, dict):
                continue
            code = str(
                error.get("codex_error_info")
                or error.get("codexErrorInfo")
                or ""
            ).strip()
            message = " ".join(str(error.get("message") or "").split())
            normalized_message = message.lower()
            if code not in infrastructure_error_codes and not any(
                fragment in normalized_message
                for fragment in infrastructure_message_fragments
            ):
                continue
            if code == "usage_limit_exceeded" or "usage limit" in normalized_message:
                detected = "Codex usage limit reached before episode completion"
            elif code == "rate_limit_exceeded" or "rate limit" in normalized_message:
                detected = "Codex rate limit reached before episode completion"
            else:
                detected = "Codex service connection failed before episode completion"
    return detected


def _session_image_view_paths(session_path: Path) -> list[str]:
    paths: list[str] = []
    try:
        lines = session_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return paths
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            continue
        payload = record.get("payload")
        if (
            record.get("type") != "event_msg"
            or not isinstance(payload, dict)
            or payload.get("type") != "item_completed"
        ):
            continue
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") != "ImageView":
            continue
        path = item.get("path")
        if isinstance(path, str) and path and path not in paths:
            paths.append(path)
    return paths


def _image_view_file(value: str, *, base: Path | None = None) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    if not parsed.scheme:
        path = Path(value).expanduser()
        if not path.is_absolute() and base is not None:
            path = base / path
        return path.resolve()
    return None


def _is_current_observation_path(source: Path, workspace: Path) -> bool:
    try:
        relative = source.relative_to(workspace)
    except ValueError:
        return False
    markers = (
        Path("benchmark_inputs/current_observation"),
        Path("benchmark_inputs/live_observation/current"),
    )
    return any(
        relative.parts[: len(marker.parts)] == marker.parts for marker in markers
    )


def _archive_viewed_artifacts(
    *, session_path: Path, workspace: Path, run_directory: Path
) -> dict[str, Any]:
    """Preserve files explicitly viewed by Codex before workspace cleanup."""

    archive_root = run_directory / "viewed_artifacts"
    entries: list[dict[str, Any]] = []
    archived_by_digest: dict[tuple[str, str], Path] = {}
    for source_value in _session_image_view_paths(session_path):
        source = _image_view_file(source_value, base=workspace)
        entry: dict[str, Any] = {
            "source_path": source_value,
            "source_absolute_path": os.fspath(source) if source is not None else None,
        }
        if source is None:
            entry["status"] = "unsupported_uri"
            entries.append(entry)
            continue
        try:
            relative = source.relative_to(workspace)
        except ValueError:
            relative = None
        if relative is not None:
            entry["workspace_relative_path"] = relative.as_posix()
        if _is_current_observation_path(source, workspace):
            entry["status"] = "historical_observation_archive"
            entries.append(entry)
            continue
        try:
            source.relative_to(run_directory)
        except ValueError:
            inside_run = False
        else:
            inside_run = True
        if inside_run:
            entry["status"] = "already_in_private_run"
            entry["archived_file"] = source.relative_to(run_directory).as_posix()
            entries.append(entry)
            continue
        if not source.is_file():
            entry["status"] = "source_missing_at_archive_time"
            entries.append(entry)
            continue
        digest = file_sha256(source)
        suffix = source.suffix.lower()
        if not suffix or len(suffix) > 12 or not suffix[1:].isalnum():
            suffix = ".bin"
        key = (digest, suffix)
        destination = archived_by_digest.get(key)
        if destination is None:
            archive_root.mkdir(parents=True, exist_ok=True)
            destination = archive_root / f"{digest}{suffix}"
            if not destination.is_file():
                shutil.copy2(source, destination)
            archived_by_digest[key] = destination
        entry.update(
            {
                "status": "archived",
                "archived_file": destination.relative_to(run_directory).as_posix(),
                "sha256": digest,
                "size_bytes": destination.stat().st_size,
            }
        )
        entries.append(entry)
    manifest = {
        "schema_version": "libero.viewed_artifacts_archive.v1",
        "created_at": _utc_now(),
        "artifact_count": sum(
            entry.get("status") in {"archived", "already_in_private_run"}
            for entry in entries
        ),
        "artifacts": entries,
    }
    _write_json_atomic(run_directory / "viewed_artifacts_manifest.json", manifest)
    return manifest


def _session_metadata(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        first_record = json.loads(stream.readline())
    if first_record.get("type") != "session_meta":
        raise ValueError("not a Codex session log")
    payload = first_record.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid Codex session metadata")
    return payload


def _git_value(source_root: Path, *arguments: str, required: bool = True) -> str | None:
    try:
        return subprocess.check_output(
            ("git", "-C", os.fspath(source_root), *arguments),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        if required:
            raise
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
