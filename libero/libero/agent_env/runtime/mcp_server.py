#!/usr/bin/env python3
"""STDIO MCP adapter for one workspace-local LIBERO episode service."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Mapping


MCP_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "libero-agent-control"
SERVER_VERSION = "0.1.0"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_MICRO_STEPS = 50
SERVER_INSTRUCTIONS = (
    "Control each prepared LIBERO episode only through start_episode, "
    "osc_sequence, and finish_episode. For each episode, call start_episode "
    "once, inspect the atomically updated current observation after every "
    "successful action, and call finish_episode once to obtain the "
    "official checker result. A non-final finish reports that another episode "
    "is available; begin it with start_episode. The start result reports the "
    "active episode's max_agent_steps and max_wall_time_seconds budgets. Each "
    f"osc_sequence call accepts 1 to {MAX_MICRO_STEPS} normalized "
    "OSC_POSE actions."
)
RESET_SERVER_INSTRUCTIONS = (
    "Control each prepared LIBERO episode only through start_episode, "
    "osc_sequence, reset_episode, and finish_episode. For each episode, call "
    "start_episode once and inspect the atomically updated current observation "
    "after every successful action or reset. If an attempt is unrecoverably "
    "disordered, reset_episode may be called at most twice; it destroys and "
    "recreates the same task from its initial state, returns a new observation "
    "id, and does not preserve previous rollout files in the Agent workspace. "
    "The Agent's conversation context is the only memory across lives. Call "
    "finish_episode once to obtain the official checker result. A non-final "
    "finish reports that another episode is available; begin it with "
    f"start_episode. Each osc_sequence call accepts 1 to {MAX_MICRO_STEPS} "
    "normalized "
    "OSC_POSE actions; its action budget resets for each life, while the "
    "wall-clock budget is shared across all lives."
)


def configured_max_resets() -> int:
    raw = os.environ.get("LIBERO_MAX_RESETS", "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("LIBERO_MAX_RESETS must be 0 or 2") from exc
    if value not in {0, 2}:
        raise ValueError("LIBERO_MAX_RESETS must be 0 or 2")
    return value


def tool_definitions(max_resets: int | None = None) -> list[dict[str, Any]]:
    if max_resets is None:
        max_resets = configured_max_resets()
    if max_resets not in {0, 2}:
        raise ValueError("max_resets must be 0 or 2")
    empty_schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    action_vector = {
        "type": "array",
        "minItems": 7,
        "maxItems": 7,
        "items": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "description": (
            "Normalized [dx, dy, dz, rx, ry, rz, gripper]. Translation 1.0 "
            "corresponds to 0.05 m, rotation 1.0 to a 0.5 rad rotation-vector "
            "component, gripper -1 opens, and +1 closes for one policy interval."
        ),
    }
    mutating = {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
    tools = [
        {
            "name": "start_episode",
            "title": "Start LIBERO episode",
            "description": (
                "Start the next prepared episode when none is active, publish its "
                "initial current observation, and return its task instruction and "
                "Agent-step and wall-clock budgets."
            ),
            "inputSchema": empty_schema,
            "annotations": mutating,
        },
        {
            "name": "osc_sequence",
            "title": "Execute LIBERO OSC sequence",
            "description": (
                f"Execute 1 to {MAX_MICRO_STEPS} exact normalized LIBERO "
                "OSC_POSE micro-actions "
                "as one Agent action, then atomically publish the resulting "
                "current observation."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_MICRO_STEPS,
                        "items": action_vector,
                    }
                },
                "required": ["actions"],
                "additionalProperties": False,
            },
            "annotations": mutating,
        },
        {
            "name": "finish_episode",
            "title": "Finish LIBERO episode",
            "description": (
                "Finish the active episode once and return its official task-success "
                "result plus whether another prepared episode is available."
            ),
            "inputSchema": empty_schema,
            "annotations": mutating,
        },
    ]
    if max_resets:
        tools.insert(
            2,
            {
                "name": "reset_episode",
                "title": "Reset LIBERO episode",
                "description": (
                    "Discard the active rollout and recreate the same LIBERO task "
                    "from its initial state. At most two resets are available per "
                    "episode; the result publishes a new initial observation and id."
                ),
                "inputSchema": empty_schema,
                "annotations": mutating,
            },
        )
    return tools


def workspace_root() -> Path:
    configured = os.environ.get("LIBERO_AGENT_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def control_socket_path() -> Path:
    configured = os.environ.get("LIBERO_CONTROL_SOCKET", ".libero/control.sock")
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (workspace_root() / path).resolve()


def current_observation_file() -> Path:
    return (
        workspace_root()
        / "benchmark_inputs"
        / "current_observation"
        / "observation.json"
    )


def bind_current_observation(request: Mapping[str, Any]) -> dict[str, Any]:
    bound = dict(request)
    if bound.get("command") == "start":
        return bound
    value = json.loads(current_observation_file().read_text(encoding="utf-8"))
    observation_id = value.get("observation_id") if isinstance(value, dict) else None
    if not isinstance(observation_id, str) or not observation_id:
        raise ValueError("current observation has no valid observation_id")
    bound["observation_id"] = observation_id
    return bound


def send_service_request(request: Mapping[str, Any]) -> dict[str, Any]:
    encoded = (json.dumps(dict(request), separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("LIBERO service request exceeds protocol size limit")
    path = control_socket_path()
    chunks: list[bytes] = []
    size = 0
    original_cwd = os.open(".", os.O_RDONLY)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        try:
            os.chdir(path.parent)
            connection.connect(path.name)
        finally:
            os.fchdir(original_cwd)
            os.close(original_cwd)
        connection.sendall(encoded)
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise RuntimeError("LIBERO service response exceeds protocol size limit")
            if b"\n" in chunk:
                break
    payload = b"".join(chunks).split(b"\n", 1)[0]
    if not payload:
        raise RuntimeError("LIBERO service closed without a response")
    response = json.loads(payload.decode("utf-8"))
    if not isinstance(response, dict):
        raise RuntimeError("LIBERO service returned a non-object response")
    return response


def request_for_tool(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if name == "start_episode":
        if arguments:
            raise ValueError("start_episode accepts no arguments")
        return {"command": "start"}
    if name == "osc_sequence":
        if set(arguments) != {"actions"}:
            raise ValueError("osc_sequence requires only the actions argument")
        return {"command": "osc_sequence", "actions": arguments["actions"]}
    if name == "reset_episode":
        if arguments:
            raise ValueError("reset_episode accepts no arguments")
        return {"command": "reset"}
    if name == "finish_episode":
        if arguments:
            raise ValueError("finish_episode accepts no arguments")
        return {"command": "finish"}
    raise ValueError(f"unknown LIBERO MCP tool: {name}")


def tool_result(response: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(response)
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, indent=2, sort_keys=True),
            }
        ],
        "structuredContent": payload,
        "isError": payload.get("ok") is not True,
    }


def handle_message(message: Mapping[str, Any]) -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    if request_id is None:
        # Initialized, cancelled, and progress notifications require no reply.
        return None
    try:
        if method == "initialize":
            max_resets = configured_max_resets()
            result = {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": (
                    RESET_SERVER_INSTRUCTIONS if max_resets else SERVER_INSTRUCTIONS
                ),
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": tool_definitions()}
        elif method == "tools/call":
            parameters = message.get("params", {})
            if not isinstance(parameters, Mapping):
                raise ValueError("tools/call params must be an object")
            name = parameters.get("name")
            arguments = parameters.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, Mapping):
                raise ValueError("tools/call requires a tool name and object arguments")
            available = {tool["name"] for tool in tool_definitions()}
            if name not in available:
                raise ValueError(f"unknown or disabled LIBERO MCP tool: {name}")
            request = bind_current_observation(request_for_tool(name, arguments))
            result = tool_result(send_service_request(request))
        else:
            return _jsonrpc_error(request_id, -32601, f"method not found: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        if method == "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": tool_result(
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                ),
            }
        return _jsonrpc_error(request_id, -32602, str(exc))


def _jsonrpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": int(code), "message": str(message)},
    }


def main() -> int:
    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("JSON-RPC message must be an object")
            response = handle_message(value)
        except (ValueError, json.JSONDecodeError) as exc:
            response = _jsonrpc_error(None, -32700, str(exc))
        if response is not None:
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
