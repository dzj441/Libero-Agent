"""Read-only viewer for LIBERO Agent runs and their persisted Codex sessions."""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, unquote, urlsplit


STATIC_FILENAMES = frozenset({"index.html", "app.js", "styles.css"})
OBSERVATION_ID = re.compile(r"(?:obs_\d{6}|reset_\d{2}_obs_\d{6})")
CURRENT_OBSERVATION_MARKERS = (
    Path("benchmark_inputs/current_observation"),
    Path("benchmark_inputs/live_observation/current"),
)
ROBOT_COMMAND = re.compile(
    r"(?:^|\s)liberoctl\s+(start|step|osc-step|osc-sequence|reset|finish)\b"
)
MCP_ROBOT_COMMANDS = {
    "start_episode": "start",
    "osc_sequence": "osc_sequence",
    "reset_episode": "reset",
    "finish_episode": "finish",
}
TEXT_LIMIT = 24_000
COLLECTION_LIMIT = 200

ACTIVITY_LABELS = {
    "reasoning": "公开 reasoning summary",
    "agent_message": "Agent message",
    "command_execution": "Shell / command execution",
    "image_view": "Image view",
    "file_change": "File change",
    "mcp_tool_call": "MCP tool call",
    "collab_tool_call": "Collaboration tool call",
    "sub_agent_activity": "Sub-agent activity",
    "web_search": "Web search",
    "context_compaction": "Context compaction",
    "plan": "Plan update",
    "session_error": "Codex session error",
    "session_event": "Codex session event",
    "task_started": "Codex task started",
    "task_complete": "Codex task completed",
    "turn_aborted": "Codex turn aborted",
    "thread_settings_applied": "Codex thread settings",
}

HIDDEN_SESSION_FIELDS = frozenset({"raw_content", "encrypted_content"})
RUNTIME_USER_CONTEXT_MARKERS = (
    "<environment_context>",
    "<recommended_plugins>",
)


class ViewerDataError(RuntimeError):
    """A malformed or unsafe viewer request."""


class RunNotFound(ViewerDataError):
    """The requested run does not exist below the configured root."""


@dataclass(frozen=True)
class RunFiles:
    run_id: str
    directory: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL records while tolerating a live partial final line."""

    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    output: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            continue
        if isinstance(value, dict):
            output.append(value)
    return output


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _elapsed(value: Any, origin: datetime | None) -> float:
    timestamp = _parse_time(value)
    if timestamp is None or origin is None:
        return 0.0
    return max(0.0, (timestamp - origin).total_seconds())


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _snake_case(value: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _bounded_text(value: str) -> str:
    if len(value) <= TEXT_LIMIT:
        return value
    head = TEXT_LIMIT * 2 // 3
    tail = TEXT_LIMIT - head
    omitted = len(value) - TEXT_LIMIT
    return (
        value[:head]
        + f"\n\n... <viewer omitted {omitted} characters; source log is intact> ...\n\n"
        + value[-tail:]
    )


def _bounded(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _bounded_text(value)
    if depth >= 8:
        return "<viewer nesting limit reached>"
    if isinstance(value, list):
        output = [_bounded(item, depth=depth + 1) for item in value[:COLLECTION_LIMIT]]
        if len(value) > COLLECTION_LIMIT:
            output.append(f"<viewer omitted {len(value) - COLLECTION_LIMIT} items>")
        return output
    if isinstance(value, dict):
        items = list(value.items())
        output = {
            str(key): _bounded(item, depth=depth + 1)
            for key, item in items[:COLLECTION_LIMIT]
        }
        if len(items) > COLLECTION_LIMIT:
            output["<viewer_omitted_fields>"] = len(items) - COLLECTION_LIMIT
        return output
    return value


def _public_session_value(value: Any) -> Any:
    """Return complete public session content while removing hidden reasoning."""

    if isinstance(value, list):
        return [_public_session_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _public_session_value(item)
            for key, item in value.items()
            if key not in HIDDEN_SESSION_FIELDS
        }
    return value


def _thread_settings_summary(value: Any) -> dict[str, Any]:
    settings = value if isinstance(value, dict) else {}
    collaboration = settings.get("collaboration_mode")
    collaboration = collaboration if isinstance(collaboration, dict) else {}
    collaboration_settings = collaboration.get("settings")
    collaboration_settings = (
        collaboration_settings if isinstance(collaboration_settings, dict) else {}
    )
    return {
        key: item
        for key, item in {
            "model": settings.get("model"),
            "model_provider_id": settings.get("model_provider_id"),
            "service_tier": settings.get("service_tier"),
            "reasoning_effort": settings.get("reasoning_effort"),
            "approval_policy": settings.get("approval_policy"),
            "approvals_reviewer": settings.get("approvals_reviewer"),
            "permission_profile": settings.get("permission_profile"),
            "cwd": settings.get("cwd"),
            "personality": settings.get("personality"),
            "collaboration_mode": collaboration.get("mode"),
            "collaboration_model": collaboration_settings.get("model"),
            "collaboration_reasoning_effort": collaboration_settings.get(
                "reasoning_effort"
            ),
        }.items()
        if item is not None
    }


def _runtime_user_context(value: str) -> bool:
    return any(marker in value for marker in RUNTIME_USER_CONTEXT_MARKERS)


def _world_state_summary(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    state = payload.get("state")
    state = state if isinstance(state, dict) else {}
    host_skills = state.get("host_skills")
    host_skills = host_skills if isinstance(host_skills, dict) else {}
    return _public_session_value(
        {
            "full_snapshot": payload.get("full"),
            "agents_md": state.get("agents_md"),
            "apps_instructions": state.get("apps_instructions"),
            "plugins_instructions": state.get("plugins_instructions"),
            "skills": state.get("skills"),
            "host_skills": {
                "include_instructions": host_skills.get("includeInstructions"),
                "body_present": bool(host_skills.get("body")),
            },
            "orchestrator_skills": state.get("orchestrator_skills"),
            "collaboration_mode": state.get("collaboration_mode"),
            "multi_agent_mode": state.get("multi_agent_mode"),
            "multi_agent_usage_hint": state.get("multi_agent_usage_hint"),
            "permissions": state.get("permissions"),
            "environments": state.get("environments"),
            "environments_instructions": state.get("environments_instructions"),
            "model": state.get("model"),
            "personality": state.get("personality"),
            "realtime": state.get("realtime"),
            "git_attribution": state.get("git_attribution"),
        }
    )


def _content_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            output.append(item.strip())
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                output.append(text.strip())
    return output


def _command_text(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[-1])
    if isinstance(value, str):
        return value
    return ""


def _robot_command(value: str) -> str | None:
    match = ROBOT_COMMAND.search(value)
    return match.group(1).replace("-", "_") if match else None


def _mcp_robot_command(item: Mapping[str, Any]) -> str | None:
    """Map only the workspace-local LIBERO MCP server onto wire commands."""

    server = item.get("server") or item.get("server_name")
    tool = item.get("tool") or item.get("tool_name") or item.get("name")
    if not isinstance(tool, str):
        return None
    normalized_tool = tool.rsplit("/", 1)[-1].rsplit(".", 1)[-1]
    if normalized_tool.startswith("mcp__libero__"):
        normalized_tool = normalized_tool.removeprefix("mcp__libero__")
        server = "libero"
    if server != "libero":
        return None
    return MCP_ROBOT_COMMANDS.get(normalized_tool)


def _session_origin(records: Iterable[Mapping[str, Any]]) -> datetime | None:
    for record in records:
        timestamp = _parse_time(record.get("timestamp"))
        if timestamp is not None:
            return timestamp
    return None


def _normalize_activity(
    records: list[dict[str, Any]], origin: datetime | None
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        if record.get("type") != "event_msg":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        event_type = payload.get("type")
        if event_type != "item_completed":
            if event_type == "token_count":
                # The latest aggregate is exposed in the session summary. Repeating
                # every token checkpoint would overwhelm the physical timeline.
                continue
            if not isinstance(event_type, str):
                continue
            kind = event_type if event_type in {
                "task_started",
                "task_complete",
                "turn_aborted",
                "thread_settings_applied",
            } else (
                "session_error"
                if "error" in event_type.lower()
                else "session_event"
            )
            details: Any
            if event_type == "thread_settings_applied":
                details = _thread_settings_summary(payload.get("thread_settings"))
            else:
                details = {
                    key: child
                    for key, child in payload.items()
                    if key != "type"
                }
            output.append(
                {
                    "kind": kind,
                    "label": ACTIVITY_LABELS.get(kind, event_type),
                    "status": (
                        "error"
                        if kind in {"session_error", "turn_aborted"}
                        else "completed"
                    ),
                    "title": event_type,
                    "parts": [],
                    "details": _public_session_value(details),
                    "timestamp_utc": record.get("timestamp"),
                    "elapsed_seconds": _elapsed(record.get("timestamp"), origin),
                    "ordinal": record.get("ordinal"),
                    "robot_command": None,
                    "source_path": None,
                }
            )
            continue
        item = payload.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if not isinstance(item_type, str) or item_type == "UserMessage":
            continue
        kind = _snake_case(item_type)
        parts: list[str] = []
        title: str | None = None
        details: Any = None
        robot_command: str | None = None
        source_path: str | None = None

        if item_type == "Reasoning":
            parts = _content_text(item.get("summary_text") or item.get("summary"))
            # Hidden/raw reasoning is intentionally never exposed.
            if not parts:
                continue
        elif item_type == "AgentMessage":
            parts = _content_text(item.get("content") or item.get("text"))
            if not parts:
                continue
            details = {
                key: item.get(key)
                for key in ("phase", "status")
                if item.get(key) is not None
            }
        elif item_type == "CommandExecution":
            title = _command_text(item.get("command")) or "Command execution"
            robot_command = _robot_command(title)
            details = {
                key: child
                for key, child in item.items()
                if key
                not in {
                    "id",
                    "type",
                    "command",
                    "aggregated_output",
                    "formatted_output",
                    *HIDDEN_SESSION_FIELDS,
                }
                and child not in (None, "", [])
            }
        elif item_type == "ImageView":
            path = item.get("path")
            title = str(path) if path is not None else "Image view"
            source_path = str(path) if isinstance(path, str) else None
            details = {
                key: child
                for key, child in item.items()
                if key not in {"id", "type", "path", *HIDDEN_SESSION_FIELDS}
            }
        elif kind == "mcp_tool_call":
            server = item.get("server") or item.get("server_name")
            tool = item.get("tool") or item.get("tool_name") or item.get("name")
            title = ".".join(str(value) for value in (server, tool) if value) or item_type
            robot_command = _mcp_robot_command(item)
            details = {
                key: child
                for key, child in item.items()
                if key not in {"id", "type", *HIDDEN_SESSION_FIELDS}
            }
        elif item_type == "ContextCompaction":
            parts = ["The Codex context was compacted within this session."]
        else:
            title_value = (
                item.get("tool")
                or item.get("name")
                or item.get("query")
                or item.get("path")
            )
            title = str(title_value) if title_value is not None else item_type
            details = {
                key: child
                for key, child in item.items()
                if key not in {"id", "type", *HIDDEN_SESSION_FIELDS}
            }

        event_metadata = {
            key: child
            for key, child in payload.items()
            if key not in {"type", "item", *HIDDEN_SESSION_FIELDS}
        }
        if event_metadata:
            if not isinstance(details, dict):
                details = {"item_details": details} if details is not None else {}
            details["session_event"] = event_metadata

        output.append(
            {
                "kind": kind,
                "label": ACTIVITY_LABELS.get(kind, item_type),
                "status": str(item.get("status") or "completed"),
                "title": title if title else None,
                "parts": parts,
                "details": _public_session_value(details) if details else None,
                "timestamp_utc": record.get("timestamp"),
                "elapsed_seconds": _elapsed(record.get("timestamp"), origin),
                "ordinal": record.get("ordinal"),
                "robot_command": robot_command,
                "source_path": source_path,
            }
        )
    return output


def _session_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Account for every persisted record without exposing hidden reasoning."""

    top_level_types: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    item_types: Counter[str] = Counter()
    response_item_types: Counter[str] = Counter()
    classifications: Counter[str] = Counter()
    unsupported_types: Counter[str] = Counter()
    hidden_fields: Counter[str] = Counter()

    def count_hidden(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                count_hidden(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key in HIDDEN_SESSION_FIELDS and child not in (None, "", []):
                    hidden_fields[key] += 1
                else:
                    count_hidden(child)

    for record in records:
        record_type = str(record.get("type") or "<missing>")
        top_level_types[record_type] += 1
        count_hidden(record)
        payload = record.get("payload")
        payload = payload if isinstance(payload, dict) else {}

        if record_type in {"session_meta", "world_state", "turn_context"}:
            classifications["summarized"] += 1
        elif record_type == "event_msg":
            event_type = str(payload.get("type") or "<missing>")
            event_types[event_type] += 1
            if event_type == "item_completed":
                item = payload.get("item")
                item = item if isinstance(item, dict) else {}
                item_type = str(item.get("type") or "<missing>")
                item_types[item_type] += 1
                if item_type == "UserMessage":
                    classifications["summarized"] += 1
                elif item_type == "Reasoning" and not _content_text(
                    item.get("summary_text") or item.get("summary")
                ):
                    classifications["deliberately_hidden"] += 1
                elif item_type == "<missing>":
                    classifications["unsupported"] += 1
                    unsupported_types["event_msg/item_completed/<missing>"] += 1
                else:
                    classifications["rendered"] += 1
            elif event_type == "token_count":
                classifications["summarized"] += 1
            elif event_type == "<missing>":
                classifications["unsupported"] += 1
                unsupported_types["event_msg/<missing>"] += 1
            else:
                # Known lifecycle events and future JSON events are shown as
                # sanitized generic cards rather than being silently discarded.
                classifications["rendered"] += 1
        elif record_type == "response_item":
            response_type = str(payload.get("type") or "<missing>")
            response_item_types[response_type] += 1
            if response_type == "message":
                if payload.get("role") in {"developer", "user"}:
                    content = payload.get("content")
                    content = content if isinstance(content, list) else []
                    nontext = Counter(
                        str(item.get("type") or "<missing>")
                        for item in content
                        if not isinstance(item, dict)
                        or not isinstance(item.get("text"), str)
                    )
                    if nontext:
                        classifications["unsupported"] += 1
                        for content_type, count in nontext.items():
                            unsupported_types[
                                f"response_item/message/{payload.get('role')}/"
                                f"{content_type}"
                            ] += count
                    else:
                        classifications["summarized"] += 1
                else:
                    classifications["structural_duplicate"] += 1
            elif response_type in {"custom_tool_call", "custom_tool_call_output"}:
                classifications["structural_duplicate"] += 1
            elif response_type == "reasoning":
                classifications["deliberately_hidden"] += 1
            else:
                classifications["unsupported"] += 1
                unsupported_types[f"response_item/{response_type}"] += 1
        else:
            classifications["unsupported"] += 1
            unsupported_types[record_type] += 1

    return {
        "schema_version": "libero.codex_viewer_coverage.v1",
        "source_record_count": len(records),
        "classification_counts": dict(sorted(classifications.items())),
        "top_level_types": dict(sorted(top_level_types.items())),
        "event_types": dict(sorted(event_types.items())),
        "completed_item_types": dict(sorted(item_types.items())),
        "response_item_types": dict(sorted(response_item_types.items())),
        "hidden_field_counts": dict(sorted(hidden_fields.items())),
        "unsupported_types": dict(sorted(unsupported_types.items())),
        "public_trace_complete": not unsupported_types,
        "policy": {
            "rendered": "Shown in the action-aligned timeline.",
            "summarized": "Shown once in prompt, runtime, completion, or token summaries.",
            "structural_duplicate": "Protocol duplicate of a rendered completed item.",
            "deliberately_hidden": (
                "Raw/encrypted model reasoning is not human-visible Codex output and "
                "is never exposed. Public reasoning summaries remain rendered."
            ),
            "unsupported": "Unknown record shape; surfaced as a coverage warning.",
        },
    }


def _session_context(
    records: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    session_meta = next(
        (
            record.get("payload")
            for record in records
            if record.get("type") == "session_meta"
            and isinstance(record.get("payload"), dict)
        ),
        {},
    )
    base = session_meta.get("base_instructions")
    base_text = base.get("text") if isinstance(base, dict) else None
    messages: dict[str, list[str]] = {"developer": [], "user": []}
    for record in records:
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "message":
            continue
        role = payload.get("role")
        if role not in messages:
            continue
        text = "\n\n".join(_content_text(payload.get("content")))
        if text and text not in messages[role]:
            messages[role].append(text)
    token_usage: dict[str, Any] = {}
    completion: dict[str, Any] = {}
    task_started: dict[str, Any] = {}
    thread_settings: dict[str, Any] = {}
    for record in records:
        if record.get("type") != "event_msg":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "token_count":
            info = payload.get("info")
            if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                token_usage = info["total_token_usage"]
        elif payload.get("type") == "task_complete":
            completion = payload
        elif payload.get("type") == "task_started":
            task_started = payload
        elif payload.get("type") == "thread_settings_applied":
            thread_settings = _thread_settings_summary(payload.get("thread_settings"))

    turn_contexts = [
        record.get("payload")
        for record in records
        if record.get("type") == "turn_context"
        and isinstance(record.get("payload"), dict)
    ]
    latest_turn = turn_contexts[-1] if turn_contexts else {}
    world_states = [
        record.get("payload")
        for record in records
        if record.get("type") == "world_state"
        and isinstance(record.get("payload"), dict)
    ]
    collaboration = latest_turn.get("collaboration_mode")
    collaboration = collaboration if isinstance(collaboration, dict) else {}
    runtime_settings = {
        key: item
        for key, item in {
            "model": latest_turn.get("model") or thread_settings.get("model"),
            "reasoning_effort": latest_turn.get("effort")
            or thread_settings.get("reasoning_effort"),
            "sandbox_policy": latest_turn.get("sandbox_policy"),
            "approval_policy": latest_turn.get("approval_policy")
            or thread_settings.get("approval_policy"),
            "approvals_reviewer": latest_turn.get("approvals_reviewer")
            or thread_settings.get("approvals_reviewer"),
            "permission_profile": latest_turn.get("permission_profile")
            or thread_settings.get("permission_profile"),
            "workspace_roots": latest_turn.get("workspace_roots"),
            "current_date": latest_turn.get("current_date"),
            "timezone": latest_turn.get("timezone"),
            "collaboration_mode": collaboration.get("mode")
            or thread_settings.get("collaboration_mode"),
            "multi_agent_version": latest_turn.get("multi_agent_version"),
            "realtime_active": latest_turn.get("realtime_active"),
            "context_window": task_started.get("model_context_window"),
            "thread_settings": thread_settings or None,
            "world_state": (
                _world_state_summary(world_states[-1]) if world_states else None
            ),
        }.items()
        if item is not None
    }
    task_messages = [
        message for message in messages["user"] if not _runtime_user_context(message)
    ]
    runtime_user_messages = [
        message for message in messages["user"] if _runtime_user_context(message)
    ]
    return {
        "session_id": metadata.get("session_id") or session_meta.get("session_id"),
        "cli_version": session_meta.get("cli_version"),
        "originator": session_meta.get("originator"),
        "source": session_meta.get("source"),
        "thread_source": session_meta.get("thread_source"),
        "session_created_at": session_meta.get("timestamp"),
        "model_provider": session_meta.get("model_provider"),
        "cwd": metadata.get("cwd") or session_meta.get("cwd"),
        "episode_resumable": metadata.get("episode_resumable"),
        "base_instructions": base_text if isinstance(base_text, str) else None,
        "developer_messages": messages["developer"],
        "user_messages": messages["user"],
        "task_user_messages": task_messages,
        "runtime_user_messages": runtime_user_messages,
        "runtime_settings": _public_session_value(runtime_settings),
        "turn_context_count": len(turn_contexts),
        "token_usage": token_usage,
        "completion": _public_session_value(completion),
        "coverage": _session_coverage(records),
    }


def _file_url_path(value: str, *, base: Path | None = None) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    if not parsed.scheme:
        path = Path(value).expanduser()
        if not path.is_absolute() and base is not None:
            path = base / path
        return path.resolve()
    return None


class RunRepository:
    """Discover and normalize persisted LIBERO Agent runs."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise ViewerDataError(f"Run root is not a directory: {self.root}")
        self._detail_cache: dict[str, tuple[tuple[int, ...], dict[str, Any]]] = {}
        self._artifact_cache: dict[str, tuple[tuple[int, ...], set[str]]] = {}

    def _within_root(self, candidate: Path) -> Path:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ViewerDataError("Requested path leaves the configured run root") from exc
        return resolved

    def discover(self) -> list[RunFiles]:
        runs: list[RunFiles] = []
        for marker in self.root.rglob("run_manifest.json"):
            directory = self._within_root(marker.parent)
            if not (directory / "actions.jsonl").is_file():
                continue
            runs.append(RunFiles(directory.relative_to(self.root).as_posix(), directory))
        return sorted(
            runs,
            key=lambda run: (run.directory.stat().st_mtime, run.run_id),
            reverse=True,
        )

    def resolve_run(self, run_id: str) -> RunFiles:
        if not run_id or "\x00" in run_id or Path(run_id).is_absolute():
            raise RunNotFound("Invalid run id")
        directory = self._within_root(self.root / run_id)
        if not directory.is_dir():
            raise RunNotFound(f"Run not found: {run_id}")
        canonical = directory.relative_to(self.root).as_posix()
        if canonical != Path(run_id).as_posix():
            raise RunNotFound("Run id is not canonical")
        if not (directory / "run_manifest.json").is_file():
            raise RunNotFound(f"Not a LIBERO Agent run: {run_id}")
        return RunFiles(canonical, directory)

    @staticmethod
    def _workspace_root(run: RunFiles, manifest: dict[str, Any]) -> Path | None:
        value = manifest.get("workspace")
        if not isinstance(value, str) or not value:
            return None
        # Ephemeral workspaces may later be reclaimed by the operating system.
        # Keep the recorded path for historical ImageView reconstruction.
        return Path(value).expanduser().resolve()

    @staticmethod
    def _viewed_artifact_index(run: RunFiles) -> dict[str, Path]:
        manifest = _read_json(run.directory / "viewed_artifacts_manifest.json")
        entries = manifest.get("artifacts")
        if not isinstance(entries, list):
            return {}
        output: dict[str, Path] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            archived = entry.get("archived_file")
            if not isinstance(archived, str):
                continue
            archived_path = (run.directory / archived).resolve()
            try:
                archived_path.relative_to(run.directory)
            except ValueError:
                continue
            if not archived_path.is_file():
                continue
            for key in ("source_path", "source_absolute_path"):
                source = entry.get(key)
                if isinstance(source, str) and source:
                    output[source] = archived_path
        return output

    def _artifact_token(
        self,
        run: RunFiles,
        manifest: dict[str, Any],
        source: Path,
    ) -> str | None:
        resolved = source.resolve()
        try:
            return "run/" + resolved.relative_to(run.directory).as_posix()
        except ValueError:
            pass
        workspace = self._workspace_root(run, manifest)
        if workspace is not None:
            try:
                return "workspace/" + resolved.relative_to(workspace).as_posix()
            except ValueError:
                pass
        return None

    def _map_image_view(
        self,
        run: RunFiles,
        manifest: dict[str, Any],
        source_path: str | None,
        current_observation_id: str | None,
        episode_index: int | None = None,
    ) -> str | None:
        if source_path is None:
            return None
        workspace = self._workspace_root(run, manifest)
        source = _file_url_path(source_path, base=workspace)
        if source is None:
            return None
        if workspace is not None and current_observation_id is not None:
            try:
                relative = source.relative_to(workspace)
            except ValueError:
                relative = None
            if relative is not None:
                parts = relative.parts
                for marker in CURRENT_OBSERVATION_MARKERS:
                    marker_parts = marker.parts
                    if parts[: len(marker_parts)] != marker_parts:
                        continue
                    tail = Path(*parts[len(marker_parts) :])
                    historical = self._private_observation_root(
                        run, current_observation_id, episode_index
                    ) / tail
                    if historical.is_file():
                        return self._artifact_token(run, manifest, historical)
        viewed_artifacts = self._viewed_artifact_index(run)
        archived = viewed_artifacts.get(source_path) or viewed_artifacts.get(
            os.fspath(source)
        )
        if archived is not None:
            return self._artifact_token(run, manifest, archived)
        if source.is_file():
            return self._artifact_token(run, manifest, source)
        return None

    def _observation(
        self,
        run: RunFiles,
        manifest: dict[str, Any],
        observation_id: Any,
        episode_index: int | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(observation_id, str) or not OBSERVATION_ID.fullmatch(
            observation_id
        ):
            return None
        root = self._private_observation_root(
            run, observation_id, episode_index
        )
        value = _read_json(root / "observation.json")
        if not value:
            return None
        images: list[dict[str, Any]] = []
        downloads: list[dict[str, Any]] = []
        cameras = value.get("cameras")
        if isinstance(cameras, dict):
            for camera_name in ("head", "wrist"):
                camera = cameras.get(camera_name)
                if not isinstance(camera, dict):
                    continue
                rgb = camera.get("rgb")
                if isinstance(rgb, dict) and isinstance(rgb.get("file"), str):
                    path = root / rgb["file"]
                    if path.is_file():
                        images.append(
                            {
                                "label": f"{camera_name} RGB",
                                "artifact": self._artifact_token(run, manifest, path),
                            }
                        )
                depth = camera.get("depth")
                if isinstance(depth, dict):
                    for field, label in (
                        ("preview_file", f"{camera_name} metric depth"),
                        ("valid_mask_file", f"{camera_name} depth valid mask"),
                    ):
                        relative = depth.get(field)
                        if isinstance(relative, str) and (root / relative).is_file():
                            images.append(
                                {
                                    "label": label,
                                    "artifact": self._artifact_token(
                                        run, manifest, root / relative
                                    ),
                                }
                            )
                    metric = depth.get("metric_file")
                    if isinstance(metric, str) and (root / metric).is_file():
                        downloads.append(
                            {
                                "label": f"{camera_name} depth_m.npy",
                                "artifact": self._artifact_token(
                                    run, manifest, root / metric
                                ),
                            }
                        )
        annotations = value.get("annotations")
        annotation_cameras = (
            annotations.get("cameras") if isinstance(annotations, dict) else None
        )
        if isinstance(annotation_cameras, dict):
            for camera_name, camera in annotation_cameras.items():
                if not isinstance(camera, dict):
                    continue
                overlay = camera.get("overlay_file")
                if isinstance(overlay, str) and (root / overlay).is_file():
                    images.append(
                        {
                            "label": f"{camera_name} initial annotations",
                            "artifact": self._artifact_token(
                                run, manifest, root / overlay
                            ),
                        }
                    )
        observation_file = self._artifact_token(run, manifest, root / "observation.json")
        return {
            "observation_id": observation_id,
            "frame_index": value.get("frame_index"),
            "profile": value.get("profile"),
            "images": images,
            "downloads": downloads,
            "observation_file": observation_file,
            "state": _bounded(value.get("state")),
            "proprioception": _bounded(value.get("proprioception")),
            "coordinate_conventions": _bounded(value.get("coordinate_conventions")),
        }

    @staticmethod
    def _private_observation_root(
        run: RunFiles,
        observation_id: str,
        episode_index: int | None,
    ) -> Path:
        if episode_index is None:
            return run.directory / "private_observations" / observation_id
        return (
            run.directory
            / "episodes"
            / f"episode_{episode_index:03d}"
            / "private_observations"
            / observation_id
        )

    def _session(self, run: RunFiles) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        records = _read_jsonl(run.directory / "codex_session.jsonl")
        metadata = _read_json(run.directory / "codex_session_metadata.json")
        origin = _session_origin(records)
        return _normalize_activity(records, origin), _session_context(records, metadata)

    @staticmethod
    def _cache_stamp(run: RunFiles) -> tuple[int, ...]:
        paths = (
            run.directory,
            run.directory / "actions.jsonl",
            run.directory / "codex_session.jsonl",
            run.directory / "viewed_artifacts_manifest.json",
            run.directory / "result.json",
            run.directory / "private_observations",
        )
        return tuple(path.stat().st_mtime_ns if path.exists() else 0 for path in paths)

    def _steps(
        self,
        run: RunFiles,
        manifest: dict[str, Any],
        activity: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        actions = _read_jsonl(run.directory / "actions.jsonl")
        steps: list[dict[str, Any]] = []
        activity_cursor = 0
        current_observation_id: str | None = None
        current_episode_index: int | None = None
        per_episode_action_indices: dict[int, int] = {}
        matched_commands = 0
        for index, action in enumerate(actions):
            request = action.get("request")
            response = action.get("response")
            request = request if isinstance(request, dict) else {}
            response = response if isinstance(response, dict) else {}
            command = str(request.get("command") or "unknown")
            raw_episode_index = action.get("episode_index")
            episode_index = (
                int(raw_episode_index)
                if isinstance(raw_episode_index, int)
                else None
            )
            if episode_index is not None:
                episode_action_index = per_episode_action_indices.get(
                    episode_index, 0
                )
                per_episode_action_indices[episode_index] = episode_action_index + 1
            else:
                episode_action_index = index
            match_index: int | None = None
            for candidate_index in range(activity_cursor, len(activity)):
                if activity[candidate_index].get("robot_command") == command:
                    match_index = candidate_index
                    break
            if match_index is None:
                grouped: list[dict[str, Any]] = []
            else:
                grouped = activity[activity_cursor : match_index + 1]
                activity_cursor = match_index + 1
                matched_commands += 1
            for item in grouped:
                artifact = self._map_image_view(
                    run,
                    manifest,
                    item.pop("source_path", None),
                    current_observation_id,
                    current_episode_index,
                )
                if artifact is not None:
                    item["artifact"] = artifact
            if command == "start" and episode_index != current_episode_index:
                current_observation_id = None
                current_episode_index = episode_index
            prior_id = current_observation_id
            next_id = response.get("observation_id")
            if isinstance(next_id, str):
                current_observation_id = next_id
            steps.append(
                {
                    "index": index,
                    "episode_index": episode_index,
                    "episode_action_index": episode_action_index,
                    "episode_count": action.get("episode_count"),
                    "command": command,
                    "recorded_at": action.get("recorded_at"),
                    "elapsed_seconds": (
                        grouped[-1].get("elapsed_seconds")
                        if grouped
                        else _elapsed(action.get("recorded_at"), None)
                    ),
                    "ok": response.get("ok") is True,
                    "request": _bounded(request),
                    "response": _bounded(response),
                    "prior_observation_id": prior_id,
                    "next_observation_id": next_id,
                    "input_observation": self._observation(
                        run, manifest, prior_id, episode_index
                    ),
                    "output_observation": self._observation(
                        run, manifest, next_id, episode_index
                    ),
                    "agent_activity": grouped,
                }
            )
        tail = activity[activity_cursor:]
        for item in tail:
            artifact = self._map_image_view(
                run,
                manifest,
                item.pop("source_path", None),
                current_observation_id,
                current_episode_index,
            )
            if artifact is not None:
                item["artifact"] = artifact
        return steps, tail, {
            "action_records": len(actions),
            "session_robot_commands": sum(
                item.get("robot_command") is not None for item in activity
            ),
            "matched_robot_commands": matched_commands,
        }

    def summary(self, run: RunFiles) -> dict[str, Any]:
        manifest = _read_json(run.directory / "run_manifest.json")
        result = _read_json(run.directory / "result.json")
        private_evaluation = result.get("private_evaluation")
        if not isinstance(private_evaluation, dict):
            private_evaluation = {}
        session_metadata = _read_json(run.directory / "codex_session_metadata.json")
        experience_context_receipt = _read_json(
            run.directory / "experience_context_projection_receipt.json"
        )
        actions = _read_jsonl(run.directory / "actions.jsonl")
        task_instruction = ""
        archived_prompt = run.directory / "agent_prompt.txt"
        workspace = self._workspace_root(run, manifest)
        prompt_path = workspace / "TASK_PROMPT.txt" if workspace is not None else None
        if archived_prompt.is_file():
            prompt_path = archived_prompt
        if prompt_path is not None and prompt_path.is_file():
            try:
                task_instruction = prompt_path.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, IndexError):
                task_instruction = ""
        return {
            "id": run.run_id,
            "name": run.directory.name,
            "suite": (
                "multi_episode"
                if manifest.get("run_mode") == "multi_episode_curriculum"
                else manifest.get("suite")
            ),
            "task_id": (
                f"{manifest.get('episode_count')} episodes"
                if manifest.get("run_mode") == "multi_episode_curriculum"
                else manifest.get("task_id")
            ),
            "task_instruction": task_instruction,
            "profile": manifest.get("profile"),
            "icl": manifest.get("icl_condition"),
            "experience_context_id": (
                manifest.get("experience_context_id")
                or experience_context_receipt.get("context_id")
            ),
            "action_interface": manifest.get("action_interface"),
            "created_at": manifest.get("created_at"),
            "status": result.get("status") or "in_progress",
            "success": result.get("success"),
            "accepted_agent_steps": result.get("accepted_agent_steps"),
            "completed_required_stage_count": private_evaluation.get(
                "completed_required_stage_count"
            ),
            "required_stage_count": private_evaluation.get("required_stage_count"),
            "stage_score_percent": private_evaluation.get("stage_score_percent"),
            "action_count": len(actions),
            "has_session": (run.directory / "codex_session.jsonl").is_file(),
            "session_id": session_metadata.get("session_id"),
            "has_video": (
                (run.directory / "continuous_video.mp4").is_file()
                or any(
                    (path / "continuous_video.mp4").is_file()
                    for path in (run.directory / "episodes").glob("episode_*")
                )
            ),
            "episode_count": manifest.get("episode_count"),
            "curriculum_name": manifest.get("curriculum_name"),
        }

    def list_runs(self) -> list[dict[str, Any]]:
        return [self.summary(run) for run in self.discover()]

    def detail(self, run_id: str) -> dict[str, Any]:
        run = self.resolve_run(run_id)
        stamp = self._cache_stamp(run)
        cached = self._detail_cache.get(run_id)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        manifest = _read_json(run.directory / "run_manifest.json")
        result = _read_json(run.directory / "result.json")
        activity, session = self._session(run)
        steps, tail, alignment = self._steps(run, manifest, activity)
        normalized_activity = [
            item
            for step in steps
            for item in step.get("agent_activity", [])
        ] + tail
        image_views = [
            item for item in normalized_activity if item.get("kind") == "image_view"
        ]
        available_image_views = sum(
            isinstance(item.get("artifact"), str) for item in image_views
        )
        artifact_coverage = {
            "image_view_events": len(image_views),
            "image_view_artifacts_available": available_image_views,
            "image_view_artifacts_missing": len(image_views) - available_image_views,
        }
        session_coverage = session.get("coverage")
        if isinstance(session_coverage, dict):
            session_coverage["artifact_coverage"] = artifact_coverage
            session_coverage["viewer_complete"] = bool(
                session_coverage.get("public_trace_complete")
                and artifact_coverage["image_view_artifacts_missing"] == 0
            )
        video = None
        if (run.directory / "continuous_video.mp4").is_file():
            video = {
                "label": "Continuous simulator video",
                "artifact": "run/continuous_video.mp4",
            }
        videos = []
        episodes_root = run.directory / "episodes"
        if episodes_root.is_dir():
            for episode_directory in sorted(episodes_root.glob("episode_[0-9][0-9][0-9]")):
                episode_video = episode_directory / "continuous_video.mp4"
                if not episode_video.is_file():
                    continue
                episode_index = int(episode_directory.name.rsplit("_", 1)[1])
                videos.append(
                    {
                        "episode_index": episode_index,
                        "label": f"Episode {episode_index + 1} simulator video",
                        "artifact": self._artifact_token(
                            run, manifest, episode_video
                        ),
                    }
                )
            if videos and video is None:
                video = videos[0]
        detail = {
            "summary": self.summary(run),
            "manifest": _bounded(manifest),
            "result": _bounded(result),
            "session": session,
            "alignment": alignment,
            "steps": steps,
            "tail_activity": tail,
            "video": video,
            "videos": videos,
        }
        self._detail_cache[run_id] = (stamp, detail)
        self._artifact_cache[run_id] = (stamp, self._allowed_artifacts(detail))
        return detail

    @staticmethod
    def _allowed_artifacts(value: Any) -> set[str]:
        output: set[str] = set()
        if isinstance(value, dict):
            artifact = value.get("artifact")
            if isinstance(artifact, str):
                output.add(artifact)
            for child in value.values():
                output.update(RunRepository._allowed_artifacts(child))
        elif isinstance(value, list):
            for child in value:
                output.update(RunRepository._allowed_artifacts(child))
        return output

    def resolve_artifact(self, run_id: str, artifact: str) -> Path:
        run = self.resolve_run(run_id)
        if not artifact or "\x00" in artifact or Path(artifact).is_absolute():
            raise ViewerDataError("Invalid artifact path")
        self.detail(run_id)
        stamp = self._cache_stamp(run)
        cached = self._artifact_cache.get(run_id)
        allowed = cached[1] if cached is not None and cached[0] == stamp else set()
        if artifact not in allowed:
            raise ViewerDataError("Artifact is not part of the normalized viewer record")
        manifest = _read_json(run.directory / "run_manifest.json")
        if artifact.startswith("run/"):
            root = run.directory
            relative = artifact[len("run/") :]
        elif artifact.startswith("workspace/"):
            root = self._workspace_root(run, manifest)
            if root is None:
                raise RunNotFound("Recorded workspace is unavailable")
            relative = artifact[len("workspace/") :]
        else:
            raise ViewerDataError("Unknown artifact namespace")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ViewerDataError("Artifact path leaves its recorded root") from exc
        if not path.is_file():
            raise RunNotFound(f"Artifact not found: {artifact}")
        return path


def public_viewer_url(port: int, template: str | None = None) -> str | None:
    value = (template if template is not None else os.environ.get("VSCODE_PROXY_URI", "")).strip()
    if not value:
        return None
    if "{{port}}" in value:
        value = value.replace("{{port}}", str(port))
    elif "{port}" in value:
        value = value.replace("{port}", str(port))
    else:
        return None
    return value if value.endswith("/") else value + "/"


def _security_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Referrer-Policy", "no-referrer")
    handler.send_header("X-Frame-Options", "SAMEORIGIN")
    handler.send_header(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; media-src 'self'; "
        "style-src 'self'; script-src 'self'; connect-src 'self'; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'self'",
    )


def _route_suffix(path: str, suffix: str) -> bool:
    normalized = path.rstrip("/")
    return normalized == suffix or normalized.endswith(suffix)


def make_handler(
    repository: RunRepository, static_root: Path
) -> type[BaseHTTPRequestHandler]:
    static_root = static_root.resolve()

    class ViewerHandler(BaseHTTPRequestHandler):
        server_version = "LIBEROAgentSessionViewer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[viewer] {self.address_string()} {fmt % args}", flush=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._dispatch(head_only=True)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch(head_only=False)

        def _dispatch(self, *, head_only: bool) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path
            query = parse_qs(parsed.query, keep_blank_values=True)
            try:
                if _route_suffix(path, "/api/health"):
                    self._send_json(
                        {
                            "status": "ok",
                            "run_count": len(repository.discover()),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        },
                        head_only=head_only,
                    )
                elif _route_suffix(path, "/api/runs"):
                    self._send_json({"runs": repository.list_runs()}, head_only=head_only)
                elif _route_suffix(path, "/api/run"):
                    self._send_json(
                        repository.detail(query.get("run", [""])[0]),
                        head_only=head_only,
                    )
                elif _route_suffix(path, "/api/artifact"):
                    self._send_file(
                        repository.resolve_artifact(
                            query.get("run", [""])[0], query.get("path", [""])[0]
                        ),
                        head_only=head_only,
                        download=query.get("download", ["0"])[0] == "1",
                    )
                else:
                    filename = Path(path.rstrip("/")).name
                    if not filename or path.endswith("/"):
                        filename = "index.html"
                    if filename not in STATIC_FILENAMES:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    static_path = (static_root / filename).resolve()
                    if static_path.parent != static_root or not static_path.is_file():
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    self._send_file(static_path, head_only=head_only, download=False)
            except RunNotFound as exc:
                self._send_error_json(HTTPStatus.NOT_FOUND, str(exc), head_only=head_only)
            except ViewerDataError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc), head_only=head_only)
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                self.log_error("Unhandled viewer error: %r", exc)
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "The viewer could not read this run.",
                    head_only=head_only,
                )

        def _send_json(self, value: Any, *, head_only: bool) -> None:
            payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            _security_headers(self)
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)

        def _send_error_json(
            self, status: HTTPStatus, message: str, *, head_only: bool
        ) -> None:
            payload = json.dumps(
                {"error": message, "status": int(status)}, ensure_ascii=False
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            _security_headers(self)
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)

        def _send_file(self, path: Path, *, head_only: bool, download: bool) -> None:
            stat = path.stat()
            size = stat.st_size
            start = 0
            end = max(0, size - 1)
            status = HTTPStatus.OK
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if not match or size == 0:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    _security_headers(self)
                    self.end_headers()
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = int(last) if last else size - 1
                elif last:
                    start = max(0, size - int(last))
                    end = size - 1
                if start >= size or end < start:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    _security_headers(self)
                    self.end_headers()
                    return
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
            length = 0 if size == 0 else end - start + 1
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix in {".json", ".jsonl"}:
                content_type = "application/json; charset=utf-8"
            elif path.suffix in {".txt", ".log", ".md"}:
                content_type = "text/plain; charset=utf-8"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, max-age=60")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            _security_headers(self)
            self.end_headers()
            if head_only or length == 0:
                return
            with path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return ViewerHandler


def create_server(
    host: str, port: int, runs_root: Path, *, static_root: Path | None = None
) -> ThreadingHTTPServer:
    repository = RunRepository(runs_root)
    assets = static_root or Path(__file__).with_name("viewer_static")
    server = ThreadingHTTPServer((host, port), make_handler(repository, assets))
    server.daemon_threads = True
    return server


def describe_server_urls(host: str, port: int) -> list[str]:
    output: list[str] = []
    public = public_viewer_url(port)
    if public:
        output.append(f"Code-server URL: {public}")
    local_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    output.append(f"Local URL: http://{local_host}:{port}/")
    return output
