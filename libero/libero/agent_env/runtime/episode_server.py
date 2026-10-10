#!/usr/bin/env python3
"""Host one LIBERO episode behind a workspace-local Unix socket."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import runpy
import signal
import socketserver
import sys
import tempfile
import traceback
from typing import Any, Mapping

# The launcher sets the matching NVIDIA userspace stack before this process is
# created.  These defaults prevent accidental OSMesa fallback when run directly.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

AGENT_ENV_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(__file__).resolve().parents[4]
ROBOMEMARENA_BOOTSTRAP = (
    AGENT_ENV_ROOT / "integrations" / "robomemarena" / "bootstrap.py"
)


def _early_option(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


if _early_option("--suite") == "robomemarena":
    checkout = _early_option("--robomemarena-root")
    activate_robomemarena_core = runpy.run_path(
        os.fspath(ROBOMEMARENA_BOOTSTRAP)
    )["activate_robomemarena_core"]
    activate_robomemarena_core(
        source_root=SOURCE_ROOT,
        checkout_root=checkout,
    )

from libero.libero.agent_env import (  # noqa: E402
    make_libero_agent_env,
    make_robomemarena_agent_env,
    robomemarena_source_fingerprint,
)
from libero.libero.agent_env.runtime.control import ActionInterface  # noqa: E402
from libero.libero.agent_env.runtime.recording import (  # noqa: E402
    PrivateRolloutVideoRecorder,
)
from libero.libero.agent_env.runtime.contracts.episode import (  # noqa: E402
    build_server_ready_contract,
)
from libero.libero.agent_env.runtime.service import AgentEpisodeService  # noqa: E402
from libero.libero.agent_env.runtime.service import (  # noqa: E402
    DEFAULT_EPISODE_WALL_TIME_SECONDS,
)


LOGGER = logging.getLogger("libero-agent-server")
MAX_REQUEST_BYTES = 1024 * 1024


class EpisodeUnixServer(socketserver.UnixStreamServer):
    allow_reuse_address = False

    def __init__(self, socket_path: str, service: AgentEpisodeService) -> None:
        self.service = service
        self.stop_requested = False
        super().__init__(socket_path, EpisodeRequestHandler)


class EpisodeRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        request: object = None
        try:
            line = self.rfile.readline(MAX_REQUEST_BYTES + 1)
            if not line:
                return
            if len(line) > MAX_REQUEST_BYTES:
                raise ValueError("request exceeds protocol size limit")
            request = json.loads(line.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            response = self.server.service.handle(request)  # type: ignore[attr-defined]
            if self.server.service.finished:  # type: ignore[attr-defined]
                self.server.stop_requested = True  # type: ignore[attr-defined]
        except (ValueError, RuntimeError) as exc:
            self.server.service.record_error(request, exc)  # type: ignore[attr-defined]
            response = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        except Exception as exc:
            self.server.service.record_error(request, exc)  # type: ignore[attr-defined]
            LOGGER.error("unexpected command failure\n%s", traceback.format_exc())
            response = {
                "ok": False,
                "error_type": "EnvironmentCommandError",
                "error": "the LIBERO server could not execute this command",
            }
        self.wfile.write((json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))
        self.wfile.flush()


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
    parser.add_argument("--max-agent-steps", type=int)
    parser.add_argument(
        "--max-wall-time-seconds",
        type=float,
        default=float(DEFAULT_EPISODE_WALL_TIME_SECONDS),
    )
    parser.add_argument("--max-resets", type=int, choices=(0, 2), default=0)
    parser.add_argument(
        "--action-interface",
        choices=tuple(interface.value for interface in ActionInterface),
        default=ActionInterface.METRIC_OSC_STEP.value,
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--robomemarena-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    socket_path = (
        args.socket.resolve()
        if args.socket.is_absolute()
        else (workspace / args.socket).resolve()
    )
    run_directory = args.run_directory.resolve()
    if workspace not in socket_path.parents:
        raise ValueError("control socket must be inside the agent workspace")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists() or socket_path.is_symlink():
        socket_path.unlink()

    recorder = PrivateRolloutVideoRecorder(run_directory / "continuous_video.mp4")
    task_source_fingerprint = None

    def make_agent_env() -> Any:
        common_environment_arguments = {
            "task_id": args.task_id,
            "init_state_id": args.init_state_id,
            "profile": args.profile,
            "seed": args.seed,
            "camera_height": args.resolution,
            "camera_width": args.resolution,
            "render_gpu_device_id": args.render_gpu_device_id,
            "initial_settle_control_steps": args.initial_settle_control_steps,
            "max_agent_steps": args.max_agent_steps,
            "private_control_step_callback": recorder.append_raw_observation,
        }
        if args.suite == "robomemarena":
            return make_robomemarena_agent_env(
                checkout_root=args.robomemarena_root,
                **common_environment_arguments,
            )
        if args.robomemarena_root is not None:
            raise ValueError(
                "--robomemarena-root is valid only for --suite robomemarena"
            )
        return make_libero_agent_env(
            suite=args.suite,
            **common_environment_arguments,
        )

    if args.suite == "robomemarena":
        task_source_fingerprint = robomemarena_source_fingerprint(
            args.robomemarena_root,
            task_id=args.task_id,
            init_state_id=args.init_state_id,
        )
    agent_env = make_agent_env()
    service = AgentEpisodeService(
        agent_env,
        workspace_directory=workspace,
        current_observation_directory=(
            workspace / "benchmark_inputs" / "current_observation"
        ),
        private_run_directory=run_directory,
        action_interface=args.action_interface,
        reset_agent_env=make_agent_env,
        max_resets=args.max_resets,
        max_wall_time_seconds=args.max_wall_time_seconds,
    )
    ready_contract = build_server_ready_contract(
        suite=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        task_instruction=agent_env.task_instruction,
        profile=agent_env.profile,
        seed=args.seed,
        resolution=args.resolution,
        render_gpu_device_id=args.render_gpu_device_id,
        initial_settle_control_steps=agent_env.initial_settle_control_steps,
        max_agent_steps=agent_env.max_agent_steps,
        max_wall_time_seconds=args.max_wall_time_seconds,
        action_interface=service.action_interface,
        task_source_fingerprint=task_source_fingerprint,
        max_resets=args.max_resets,
    )
    server: EpisodeUnixServer | None = None
    interrupted_reason = "server_stopped_before_finish"
    try:
        # Linux counts bytes in the address passed to AF_UNIX.bind(), even when
        # the resulting socket lives at a much longer absolute filesystem path.
        # Bind from the workspace using the short relative protocol path.
        original_cwd = Path.cwd()
        os.chdir(workspace)
        try:
            server = EpisodeUnixServer(".libero/control.sock", service)
        finally:
            os.chdir(original_cwd)
        os.chmod(socket_path, 0o600)
        server.timeout = 0.5

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal interrupted_reason
            interrupted_reason = f"server_received_signal_{signum}_before_finish"
            if server is not None:
                server.stop_requested = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        _write_json_atomic(run_directory / "server_ready.json", ready_contract)
        LOGGER.info("ready protocol=libero.agent_unix_socket.v1 socket=%s", socket_path)
        while not server.stop_requested:
            if os.getppid() != args.launcher_pid:
                interrupted_reason = "launcher_process_exited_before_finish"
                break
            if service.expire_wall_time_budget_if_needed() is not None:
                server.stop_requested = True
                continue
            server.handle_request()
        if service.finished:
            return 0
        service.finalize_aborted(interrupted_reason)
        return 2
    finally:
        if server is not None:
            server.server_close()
        if socket_path.exists() or socket_path.is_symlink():
            socket_path.unlink()
        service.close()
        recorder.close()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.error("server startup failed\n%s", traceback.format_exc())
        raise
