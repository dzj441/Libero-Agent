#!/usr/bin/env python3
"""Run one LIBERO episode through a non-Codex coding-agent harness.

The LIBERO episode service is deliberately kept harness-neutral.  This
launcher owns the small amount of glue needed by Claude Code, DeepSeek
Harness, and Kimi Code: model-channel probing, MCP configuration, process
supervision, credential isolation, and public run accounting.

No API credential is ever put in an argv list, a run manifest, or a log.  The
Kimi provider config is written only to a short-lived tmpfs directory because
the current Kimi Code CLI reads provider credentials from ``config.toml``.
Claude Code and DeepSeek Harness native session homes are persistent per-run
directories outside the checkout so their own inspection/resume commands have
stable state; they are never included in the source tree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler


SOURCE_ROOT = Path(__file__).resolve().parents[4]
if os.fspath(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(SOURCE_ROOT))

from libero.libero.agent_env.runtime.control import (  # noqa: E402
    MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS,
)


HARNESS_CHOICES = (
    "claude_code",
    "deepseek_harness",
    "kimi_code",
    "qwen_code",
)
DEFAULT_MODELS = {
    # These are the newest model IDs that were independently observed as
    # available through the supplied gateway.  Users can override them when a
    # gateway exposes a newer channel.
    "claude_code": "claude-opus-5",
    "deepseek_harness": "deepseek-v4-pro",
    "kimi_code": "kimi-k3",
    "qwen_code": "qwen3.8-max",
}
# The cross-harness acceptance run is the same one-hour Task 04 experiment
# used for the Astra baseline.  Keep this independent of the ordinary
# single-episode launcher default (30 minutes).
DEFAULT_MULTI_AGENT_WALL_TIME_SECONDS = 3600.0
DEFAULT_PACKAGES = {
    "deepseek_harness": "@deepseek-ai/dsh@0.1.5-rc.2",
    "kimi_code": "@moonshot-ai/kimi-code@0.42.0",
    "qwen_code": "@qwen-code/qwen-code@0.24.0",
}
# External coding-agent CLIs persist their native sessions under a home
# directory.  Keep those homes outside the benchmark checkout, but make them
# stable by default so a finished run can be inspected with the native CLI.
# Each run gets its own child directory; this avoids mixing benchmark state
# with a user's ordinary ~/.claude / ~/.dsh data.
DEFAULT_SESSION_ROOTS = {
    "claude_code": SOURCE_ROOT.parent / ".claude" / "libero-runs",
    "deepseek_harness": SOURCE_ROOT.parent / ".deepseek" / "libero-runs",
    "kimi_code": SOURCE_ROOT.parent / ".kimi" / "libero-runs",
    "qwen_code": SOURCE_ROOT.parent / ".qwen" / "libero-runs",
}


def _default_secret_file(name: str) -> Path:
    """Locate evaluator-local credential files without copying them.

    A git worktree intentionally does not contain the ignored API files.  The
    normal checkout and the sibling ``LIBERO`` checkout are both supported so
    the launcher can be invoked from a clean worktree without putting a key in
    that worktree.
    """

    return SOURCE_ROOT / ".secrets" / name


DEFAULT_API_URL_FILE = _default_secret_file("APIURL")
DEFAULT_API_KEY_FILE = _default_secret_file("APIKEY")
MCP_CONFIG_NAME = "agent_mcp_config.json"
REDACTED = "<redacted>"


class HarnessLaunchError(RuntimeError):
    """A public-safe failure while preparing or supervising a harness."""


@dataclass(frozen=True)
class GatewayProbe:
    protocol: str
    endpoint: str
    model: str
    stream: bool
    state: str
    status_code: int | None
    response_model: str | None
    error_category: str | None
    error_message: str | None
    elapsed_seconds: float
    usage: dict[str, int] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "model": self.model,
            "stream": self.stream,
            "state": self.state,
            "status_code": self.status_code,
            "response_model": self.response_model,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "elapsed_seconds": self.elapsed_seconds,
            "usage": self.usage,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(value: object) -> str:
    """Return a bounded public-safe rendering of a subprocess/API error."""

    text = " ".join(str(value).split())
    # API keys supplied for this project use the sk- prefix.  Also remove
    # bearer values in case a gateway reflects an Authorization header.
    text = re.sub(r"(?i)sk-[A-Za-z0-9._~+/=-]{8,}", REDACTED, text)
    text = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}",
        rf"\1{REDACTED}",
        text,
    )
    return text[:1000]


def _read_url_file(path: Path) -> str:
    """Read an API URL from an assignment file or a plain URL file."""

    text = path.expanduser().read_text(encoding="utf-8")
    assignment_patterns = (
        r"(?im)^\s*(?:OPENAI_API_BASE|OPENAI_API_URL|API_BASE_URL|APIURL)\s*=\s*['\"]([^'\"]+)['\"]",
        r"(?im)^\s*(?:OPENAI_API_BASE|OPENAI_API_URL|API_BASE_URL|APIURL)\s*=\s*([^\s#]+)",
    )
    for pattern in assignment_patterns:
        match = re.search(pattern, text)
        if match:
            return _validate_url(match.group(1).strip())
    for line in text.splitlines():
        candidate = line.strip().strip("'\"")
        if not candidate or candidate.startswith("#"):
            continue
        match = re.search(r"https?://[^\s'\"]+", candidate)
        if match:
            return _validate_url(match.group(0).rstrip("/"))
    raise ValueError(f"no HTTP(S) API URL found in {path}")


def _read_key_file(path: Path) -> str:
    """Read a key without returning its contents in any exception/message."""

    text = path.expanduser().read_text(encoding="utf-8")
    # Support the plain one-line file used by this project and harmless
    # KEY=value variants without echoing the value.
    match = re.search(
        r"(?im)^\s*(?:APIKEY|API_KEY|OPENAI_API_KEY)\s*=\s*([^\s#]+)", text
    )
    if match:
        key = match.group(1).strip().strip("'\"")
    else:
        key = next(
            (
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ),
            "",
        )
    if not key or len(key) < 8:
        raise ValueError(f"API key file {path} is empty or malformed")
    return key


def _validate_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API URL must be an absolute HTTP(S) URL")
    return value.rstrip("/")


def normalize_openai_base(api_url: str) -> str:
    """Normalize a gateway URL for OpenAI-compatible chat-completions."""

    value = _validate_url(api_url)
    return value if value.endswith("/v1") else f"{value}/v1"


def normalize_anthropic_base(api_url: str) -> str:
    """Normalize a gateway URL for Anthropic Messages (without ``/v1``)."""

    value = _validate_url(api_url)
    return value[:-3].rstrip("/") if value.endswith("/v1") else value


def _proxy_opener(proxy: str | None):
    if proxy:
        return build_opener(ProxyHandler({"http": proxy, "https": proxy}))
    # Do not inherit a developer shell's 7890 proxy accidentally.  The
    # supplied gateway is reachable directly on the evaluation machine.
    return build_opener(ProxyHandler({}))


class _GatewayHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(body)
        self.status = status
        self.body = body


def _post_stream_events(
    endpoint: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    timeout: float,
    proxy: str | None,
) -> tuple[int, list[dict[str, Any]], str]:
    """POST and decode a short Server-Sent Events response.

    The gateway's model routes are exercised in the same streaming mode used
    by the agent harnesses.  We keep the response bounded because this is only
    a one-token availability probe, not a completion consumer.
    """

    request = Request(
        endpoint,
        data=json.dumps(dict(payload), separators=(",", ":")).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    events: list[dict[str, Any]] = []
    raw_parts: list[str] = []
    raw_size = 0
    data_lines: list[str] = []

    def consume_event() -> None:
        nonlocal data_lines
        if not data_lines:
            return
        data = "\n".join(data_lines).strip()
        data_lines = []
        if not data or data == "[DONE]":
            return
        try:
            value = json.loads(data)
        except json.JSONDecodeError:
            return
        if isinstance(value, dict):
            events.append(value)

    try:
        with _proxy_opener(proxy).open(request, timeout=timeout) as response:
            status = int(response.status)
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace")
                if raw_size < 4 * 1024 * 1024:
                    remaining = 4 * 1024 * 1024 - raw_size
                    raw_parts.append(line[:remaining])
                    raw_size += len(line)
                stripped = line.rstrip("\r\n")
                if not stripped:
                    consume_event()
                elif stripped.startswith("data:"):
                    data_lines.append(stripped[5:].lstrip())
            consume_event()
            return status, events, "".join(raw_parts)
    except HTTPError as exc:
        body = exc.read(4 * 1024 * 1024).decode("utf-8", "replace")
        raise _GatewayHTTPError(int(exc.code), body) from exc


def _find_mapping_value(value: object, key: str) -> object | None:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_mapping_value(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_mapping_value(child, key)
            if found is not None:
                return found
    return None


def _probe_error_category(status: int | None, message: str) -> str:
    lowered = message.lower()
    if "no available channel" in lowered or "no channel" in lowered:
        return "no_channel"
    if status in {401, 403} or "unauthorized" in lowered or "api key" in lowered:
        return "authentication"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limit"
    if status is not None and status >= 500:
        return "server_error"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    return "network_error"


def probe_model_gateway(
    *,
    api_url: str,
    protocol: str,
    model: str,
    api_key: str,
    timeout: float = 90.0,
    proxy: str | None = None,
) -> GatewayProbe:
    """Actively test one model channel independently of its harness."""

    if protocol not in {"openai_chat", "anthropic_messages"}:
        raise ValueError(f"unsupported probe protocol: {protocol}")
    if protocol == "openai_chat":
        base = normalize_openai_base(api_url)
        endpoint = f"{base}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 1,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "Mozilla/5.0 libero-multi-agent-harness/1",
        }
    else:
        base = normalize_anthropic_base(api_url)
        # Claude Code's Anthropic client appends /v1/messages to the supplied
        # root URL.  Probe that same endpoint explicitly.
        endpoint = f"{base}/v1/messages"
        payload = {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "stream": True,
        }
        headers = {
            "x-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "Mozilla/5.0 libero-multi-agent-harness/1",
        }
    started = time.monotonic()
    try:
        status, events, raw = _post_stream_events(
            endpoint, payload, headers, timeout=timeout, proxy=proxy
        )
    except _GatewayHTTPError as exc:
        message = _redact(exc.body)
        return GatewayProbe(
            protocol=protocol,
            endpoint=endpoint,
            model=model,
            stream=True,
            state="unavailable" if exc.status in {404, 503} else "error",
            status_code=exc.status,
            response_model=None,
            error_category=_probe_error_category(exc.status, message),
            error_message=message,
            elapsed_seconds=round(time.monotonic() - started, 3),
            usage=None,
        )
    except (TimeoutError, URLError, OSError) as exc:
        message = _redact(exc)
        return GatewayProbe(
            protocol=protocol,
            endpoint=endpoint,
            model=model,
            stream=True,
            state="error",
            status_code=None,
            response_model=None,
            error_category=_probe_error_category(None, message),
            error_message=message,
            elapsed_seconds=round(time.monotonic() - started, 3),
            usage=None,
        )
    usage = None
    for event in reversed(events):
        candidate = _numeric_usage(event.get("usage"))
        if candidate:
            usage = candidate
            break
        nested_usage = _find_mapping_value(event, "usage")
        candidate = _numeric_usage(nested_usage)
        if candidate:
            usage = candidate
            break
    response_model_value = _find_mapping_value(events, "model")
    response_model = (
        response_model_value if isinstance(response_model_value, str) else None
    )
    stream_state = 200 <= status < 300 and bool(events)
    return GatewayProbe(
        protocol=protocol,
        endpoint=endpoint,
        model=model,
        stream=True,
        state="available" if stream_state else "error",
        status_code=status,
        response_model=response_model if isinstance(response_model, str) else None,
        error_category=(
            None
            if stream_state
            else _probe_error_category(status, raw or "empty streaming response")
        ),
        error_message=(
            None if stream_state else _redact(raw or "empty streaming response")
        ),
        elapsed_seconds=round(time.monotonic() - started, 3),
        usage=usage,
    )


def _numeric_usage(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    aliases = {
        "input_tokens": (
            "input_tokens",
            "prompt_tokens",
            "inputTokens",
        ),
        "output_tokens": (
            "output_tokens",
            "completion_tokens",
            "outputTokens",
            "output",
        ),
        "cache_read_input_tokens": (
            "cache_read_input_tokens",
            "cache_read_tokens",
            "cached_tokens",
            "cacheReadInputTokens",
            "inputCacheRead",
        ),
        "reasoning_tokens": ("reasoning_tokens", "reasoningTokens"),
        "total_tokens": ("total_tokens", "totalTokens"),
    }
    result: dict[str, int] = {}
    for canonical, names in aliases.items():
        for name in names:
            raw = value.get(name)
            if isinstance(raw, bool):
                continue
            if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
                result[canonical] = int(raw)
                break
    # Kimi Code's wire format reports uncached prompt tokens as
    # ``inputOther`` and cached prompt tokens separately.  Normalize that
    # pair to the same total-input convention used by the other adapters.
    input_other = value.get("inputOther")
    if (
        "input_tokens" not in result
        and isinstance(input_other, (int, float))
        and not isinstance(input_other, bool)
    ):
        if math.isfinite(float(input_other)):
            result["input_tokens"] = int(input_other) + result.get(
                "cache_read_input_tokens", 0
            )
    if "total_tokens" not in result and {
        "input_tokens",
        "output_tokens",
    }.issubset(result):
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result or None


def _package_version(spec: str) -> str | None:
    match = re.search(r"@(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)$", spec)
    return match.group(1) if match else None


def _run_version_command(
    command: Sequence[str], *, env: Mapping[str, str], timeout: float = 90.0
) -> tuple[str | None, str]:
    try:
        completed = subprocess.run(
            list(command),
            env=dict(env),
            cwd=SOURCE_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, _redact(exc)
    output = _redact((completed.stdout or "") + "\n" + (completed.stderr or ""))
    match = re.search(r"(?<![\w.])\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", output)
    return (match.group(0) if match else None), output


def detect_harness_version(
    harness: str,
    *,
    claude_bin: str,
    npx_bin: str,
    dsh_package: str,
    kimi_package: str,
    qwen_package: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    if harness == "claude_code":
        observed, output = _run_version_command([claude_bin, "--version"], env=env)
        return {"requested": None, "observed": observed, "probe_output": output}
    package = {
        "deepseek_harness": dsh_package,
        "kimi_code": kimi_package,
        "qwen_code": qwen_package,
    }[harness]
    command = [npx_bin, "--yes", package, "--version"]
    version_env = dict(env)
    version_env.setdefault("CHOKIDAR_USEPOLLING", "1")
    observed, output = _run_version_command(command, env=version_env)
    return {
        "requested": _package_version(package),
        "observed": observed or _package_version(package),
        "probe_output": output,
    }


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_kimi_config(
    *,
    api_base: str,
    api_key: str,
    model: str,
    max_output_size: int = 8192,
    mcp_only: bool = False,
    thinking_enabled: bool = True,
) -> str:
    """Build Kimi's provider config; callers must keep the result ephemeral."""

    # The key is intentionally accepted only here, at the final materialization
    # boundary.  This function is never used for a manifest or command string.
    lines = [
        'default_model = "libero-kimi"',
        'default_permission_mode = "auto"',
        "merge_all_available_skills = false",
        "builtin_product_skills = false",
        "telemetry = false",
        "",
        "[providers.libero]",
        'type = "openai"',
        f"base_url = {_toml_string(api_base)}",
        f"api_key = {_toml_string(api_key)}",
        "",
        "[models.libero-kimi]",
        'provider = "libero"',
        f"model = {_toml_string(model)}",
        "max_context_size = 200000",
        f"max_output_size = {int(max_output_size)}",
        'capabilities = ["thinking", "image_in", "tool_use"]',
        "",
        "[thinking]",
        f"enabled = {'true' if thinking_enabled else 'false'}",
        *(
            [
                'effort = "high"',
                'keep = "all"',
            ]
            if thinking_enabled
            else []
        ),
        "",
        "[loop_control]",
        "max_attempts_per_step = 10",
        "",
    ]
    if mcp_only:
        # Controlled diagnostics can remove the shell/file-writing tools so a
        # model cannot spend the entire budget building an offline planner.
        # Read and ReadMediaFile remain available for the observation files;
        # all robot control still goes through the evaluator MCP server.
        lines.extend(
            (
                "[tools]",
                'enabled = ["Read", "ReadMediaFile", "mcp__libero__*"]',
                "",
            )
        )
    return "\n".join(lines)


def build_claude_command(
    *,
    claude_bin: str,
    model: str,
    effort: str,
    mcp_config: Path,
    prompt: str,
) -> list[str]:
    return [
        claude_bin,
        "--bare",
        "--strict-mcp-config",
        "--mcp-config",
        os.fspath(mcp_config),
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--effort",
        effort,
        "--permission-mode",
        "auto",
        prompt,
    ]


def build_kimi_command(
    *,
    npx_bin: str,
    package: str,
    model: str,
    prompt: str,
    mcp_config: Path,
    skills_dir: Path,
) -> list[str]:
    # Kimi Code 0.42.0 has no CLI ``--mcp-config-file`` option.  Its supported
    # non-interactive path is the user-level ``$KIMI_CODE_HOME/mcp.json``;
    # _build_harness_runtime materializes that file before this command runs.
    return [
        npx_bin,
        "--yes",
        package,
        "--model",
        "libero-kimi",
        "--prompt",
        prompt,
        "--output-format",
        "stream-json",
        "--skills-dir",
        os.fspath(skills_dir),
    ]


def build_qwen_settings(*, model: str, effort: str) -> dict[str, Any]:
    """Build a credential-free Qwen Code settings document."""

    return {
        "security": {"auth": {"selectedType": "openai"}},
        "model": {
            "name": model,
            "reasoningEffort": effort,
        },
        "privacy": {"usageStatisticsEnabled": False},
        "telemetry": {
            "enabled": False,
            "logPrompts": False,
            "includeSensitiveSpanAttributes": False,
        },
    }


def build_qwen_command(
    *,
    npx_bin: str,
    package: str,
    model: str,
    prompt: str,
    mcp_config: Path,
) -> list[str]:
    """Build Qwen Code's non-interactive streaming MCP invocation."""

    return [
        npx_bin,
        "--yes",
        package,
        "--auth-type",
        "openai",
        "--model",
        model,
        "--mcp-config",
        os.fspath(mcp_config),
        "--allowed-mcp-server-names",
        "libero",
        "--approval-mode",
        "yolo",
        "--chat-recording",
        "--output-format",
        "stream-json",
        prompt,
    ]


def _yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_deepseek_patch(
    *,
    api_base: str,
    model: str,
    mcp_command: Path,
    workspace: Path,
    mcp_config: Path,
    max_resets: int = 0,
    session_root: Path | None = None,
) -> str:
    """Build a public DeepSeek Harness patch with one stdio MCP server."""

    # The patch contains paths and model metadata only.  The credential is
    # supplied through DEEPSEEK_API_KEY in the child environment.
    # dsh treats an explicitly listed model as text-only unless its catalog
    # entry declares image input.  ``deepseek-flash`` is the current V4.1
    # Flash alias and is natively multimodal, so preserve that capability in
    # the generated patch instead of accidentally shadowing dsh's defaults.
    model_catalog_lines = [
        f"      - id: {_yaml_scalar(model)}",
        "        contextWindow: 200000",
    ]
    if model in {"deepseek-flash", "deepseek-v4-flash-vision-exp"}:
        model_catalog_lines.extend(
            (
                "        inputModalities: [text, image]",
                # Formal runs use dsh's normal V4 vision request budget.  The
                # evaluator-side observation is still the canonical 256x256
                # RGB frame; no model-specific low-detail cap is applied.
                "        imagePixelBudget: 640000",
                "        imageMaxBytes: 1048576",
            )
        )
        if model == "deepseek-flash":
            model_catalog_lines.append("        systemPromptUpdate: in-history")
    return "\n".join(
        (
            "# Generated by external_harness.py; no credential material.",
            "- id: llm-deepseek",
            "  config:",
            "    apiKeyEnv: DEEPSEEK_API_KEY",
            f"    baseURL: {_yaml_scalar(api_base)}",
            "    thinking: enabled",
            "    reasoningEffort: high",
            "    maxTokens: 8192",
            "    models:",
            *model_catalog_lines,
            "- id: agent-loop",
            "  config:",
            "    agents:",
            "      - id: main",
            "        provider: deepseek-official",
            f"        model: {_yaml_scalar(model)}",
            "        cwd: !!js process.cwd()",
            "- id: agent-default-model",
            "  config:",
            "    provider: deepseek-official",
            f"    model: {_yaml_scalar(model)}",
            "- id: session-persistence-jsonl",
            "  config:",
            f"    root: {_yaml_scalar(os.fspath(session_root) if session_root is not None else './.sessions')}",
            "    compression: none",
            "- insert:",
            "    - id: libero-mcp",
            "      name: '@deepseek-ai/dsh-mcp-client'",
            "      config:",
            "        serverName: libero",
            "        transport: stdio",
            f"        command: {_yaml_scalar(os.fspath(mcp_command))}",
            "        args: []",
            f"        cwd: {_yaml_scalar(os.fspath(workspace))}",
            "        env:",
            f"          LIBERO_AGENT_WORKSPACE: {_yaml_scalar(os.fspath(workspace))}",
            f"          LIBERO_CONTROL_SOCKET: {_yaml_scalar(os.fspath(workspace / '.libero' / 'control.sock'))}",
            f"          LIBERO_MAX_RESETS: {_yaml_scalar(str(int(max_resets)))}",
            "        toolCallTimeoutMs: 600000",
            "        failOnStartupError: true",
            "",
        )
    )


def _clean_child_env(base: Mapping[str, str], proxy: str | None) -> dict[str, str]:
    env = dict(base)
    for name in list(env):
        if name.lower().endswith("proxy") or name.lower() in {"all_proxy", "no_proxy"}:
            env.pop(name, None)
    if proxy:
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy
    return env


def _remove_ambient_credentials(env: dict[str, str]) -> None:
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "DEEPSEEK_API_KEY",
        "KIMI_API_KEY",
        "KIMI_MODEL_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
    ):
        env.pop(name, None)


def _tmp_parent(requested: Path | None) -> Path | None:
    if requested is not None:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    candidate = Path("/dev/shm")
    return candidate if candidate.is_dir() and os.access(candidate, os.W_OK) else None


def _persistent_session_home(root: Path, run_id: str) -> Path:
    """Create one private, stable native-harness home for a run.

    The parent is intentionally caller-selected (the defaults live in the
    project parent), while the run child is never reused.  We do not remove
    this directory during launcher cleanup: native session files are the
    material required by Claude/DeepSeek's own inspection/resume commands.
    """

    parent = root.expanduser().resolve()
    missing_parents: list[Path] = []
    cursor = parent
    while not cursor.exists() and cursor != cursor.parent:
        missing_parents.append(cursor)
        cursor = cursor.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Do not change permissions on an existing user directory.  The run child
    # is private even when an operator supplied an already-existing parent.
    for directory in missing_parents:
        try:
            directory.chmod(0o700)
        except OSError:
            pass
    home = parent / run_id
    home.mkdir(mode=0o700)
    try:
        home.chmod(0o700)
    except OSError:
        pass
    return home


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{secrets.token_hex(4)}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _new_run_id(harness: str, model: str) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9-]+", "-", model).strip("-")[:32] or "model"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{harness}_{safe_model}_{stamp}_{secrets.token_hex(3)}"


def _launcher_command(args: argparse.Namespace, run_id: str) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "libero.libero.agent_env.launchers.single_episode",
        "--external-agent",
        "--control-transport",
        "mcp",
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
        "--icl",
        args.icl,
        "--run-id",
        run_id,
        "--run-root",
        os.fspath(args.run_root),
        "--server-ready-timeout-s",
        str(args.server_ready_timeout_s),
    ]
    if args.workspace_root is not None:
        command.extend(("--workspace-root", os.fspath(args.workspace_root)))
    if args.keep_workspace:
        command.append("--keep-workspace")
    if args.robomemarena_root is not None:
        command.extend(("--robomemarena-root", os.fspath(args.robomemarena_root)))
    if args.fixed_demo_master is not None:
        command.extend(
            ("--fixed-demo-master", os.fspath(args.fixed_demo_master))
        )
    return command


def _pump_stream(
    stream: Any,
    path: Path,
    first_line_queue: queue.Queue[str | None] | None = None,
) -> None:
    try:
        with path.open("w", encoding="utf-8", errors="replace") as output:
            for line in iter(stream.readline, ""):
                output.write(line)
                output.flush()
                if first_line_queue is not None:
                    first_line_queue.put(line)
    finally:
        if first_line_queue is not None:
            first_line_queue.put(None)


def _parse_launcher_line(line: str, metadata: dict[str, str]) -> None:
    if ":" not in line:
        return
    key, value = line.rstrip("\n").split(":", 1)
    key = key.strip()
    if key in {"run_id", "workspace", "private_run", "prompt_file", "mcp_config_file"}:
        metadata[key] = value.strip()


def _terminate_process(
    process: subprocess.Popen[Any] | None, timeout: float = 10.0
) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _terminate_process_group(
    process: subprocess.Popen[Any] | None, timeout: float = 15.0
) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _terminate_episode_processes(
    run_directory: Path,
    *,
    result_path: Path | None = None,
    graceful_timeout: float = 10.0,
) -> None:
    """Stop a server started in its own session by launch_agent_episode.

    The server owns the authoritative evaluator result.  Give its SIGTERM
    handler a short window to call ``finalize_aborted`` before using SIGKILL;
    otherwise an agent that exits early can leave a video but no result.json.
    """

    target = os.fspath(run_directory)
    candidates: list[int] = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
            if pid == os.getpid():
                continue
            command = (
                (proc_dir / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", "replace")
            )
            if target in command:
                candidates.append(pid)
        except (OSError, ValueError):
            continue
    for pid in candidates:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if candidates and result_path is not None:
        deadline = time.monotonic() + max(0.0, graceful_timeout)
        while time.monotonic() < deadline:
            result_ready = result_path.is_file()
            live_pids: list[int] = []
            for pid in candidates:
                try:
                    state = (
                        Path(f"/proc/{pid}/stat")
                        .read_text(encoding="utf-8", errors="replace")
                        .split()
                    )
                except (OSError, ValueError):
                    continue
                # A zombie has finished all user-space cleanup; its parent
                # will reap it independently and it is safe to exclude here.
                if len(state) < 3 or state[2] != "Z":
                    live_pids.append(pid)
            if result_ready and not live_pids:
                break
            time.sleep(0.1)
    elif candidates:
        time.sleep(0.25)
    for pid in candidates:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _extract_usage_records(value: object) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    if isinstance(value, Mapping):
        numeric = _numeric_usage(value.get("usage"))
        if numeric:
            records.append(numeric)
        for child in value.values():
            records.extend(_extract_usage_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_extract_usage_records(child))
    return records


def _usage_from_jsonl(paths: Iterable[Path]) -> dict[str, Any]:
    records: list[dict[str, int]] = []
    result_records: list[dict[str, int]] = []
    lines_seen = 0
    for path in paths:
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for line in handle:
                lines_seen += 1
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                found = _extract_usage_records(value)
                records.extend(found)
                if isinstance(value, Mapping) and value.get("type") in {
                    "result",
                    "turn.result",
                    "session.result",
                }:
                    result_records.extend(found)
    selected = result_records or records
    # A stream may repeat the same usage object in an envelope and its nested
    # result.  Deduplicate exact adjacent dictionaries while retaining separate
    # request totals.
    unique: list[dict[str, int]] = []
    for record in selected:
        if not unique or record != unique[-1]:
            unique.append(record)
    totals: dict[str, int] = {}
    for record in unique:
        for key, amount in record.items():
            totals[key] = totals.get(key, 0) + amount
    if "total_tokens" not in totals and {"input_tokens", "output_tokens"}.issubset(
        totals
    ):
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return {
        "available": bool(totals),
        "requests_with_usage": len(unique),
        "lines_scanned": lines_seen,
        **totals,
    }


def _session_ids_from_jsonl(paths: Iterable[Path]) -> list[str]:
    found: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, Mapping):
                continue
            for key in ("session_id", "sessionId", "session", "id"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate and len(candidate) <= 160:
                    if candidate not in found:
                        found.append(candidate)
    return found


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return os.fspath(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return os.fspath(path)


def _harness_protocol(harness: str) -> str:
    return "anthropic_messages" if harness == "claude_code" else "openai_chat"


def _build_harness_runtime(
    *,
    harness: str,
    model: str,
    prompt: str,
    run_id: str,
    workspace: Path,
    private_run: Path,
    mcp_config: Path,
    api_url: str,
    api_key: str,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, str], list[Path], dict[str, Any]]:
    """Return command, child env, cleanup paths, and public runtime facts."""

    env = _clean_child_env(os.environ, args.proxy)
    _remove_ambient_credentials(env)
    cleanup_paths: list[Path] = []
    public: dict[str, Any] = {}
    if harness == "claude_code":
        claude_root = getattr(
            args,
            "claude_session_root",
            DEFAULT_SESSION_ROOTS["claude_code"],
        )
        claude_home = _persistent_session_home(claude_root, run_id)
        env.update(
            {
                "ANTHROPIC_BASE_URL": normalize_anthropic_base(api_url),
                "ANTHROPIC_API_KEY": api_key,
                "ANTHROPIC_AUTH_TOKEN": api_key,
                "CLAUDE_CONFIG_DIR": os.fspath(claude_home),
                "NO_COLOR": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        )
        command = build_claude_command(
            claude_bin=args.claude_bin,
            model=model,
            effort=args.claude_effort,
            mcp_config=mcp_config,
            prompt=prompt,
        )
        public.update(
            {
                "credential_transport": "environment_only",
                "config_home_ephemeral": False,
                "session_home": os.fspath(claude_home),
                "session_persistence": "persistent",
                "reasoning_effort": args.claude_effort,
                "reasoning_enabled": True,
            }
        )
        return command, env, cleanup_paths, public

    if harness == "deepseek_harness":
        dsh_root = getattr(
            args,
            "deepseek_session_root",
            DEFAULT_SESSION_ROOTS["deepseek_harness"],
        )
        dsh_home = _persistent_session_home(dsh_root, run_id)
        dsh_session_root = dsh_home / "sessions"
        dsh_session_root.mkdir(mode=0o700)
        patch_path = private_run / "dsh_mcp.patch.yml"
        patch_path.write_text(
            build_deepseek_patch(
                api_base=normalize_openai_base(api_url),
                model=model,
                mcp_command=workspace / "bin" / "libero_mcp_server",
                workspace=workspace,
                mcp_config=mcp_config,
                max_resets=args.max_resets,
                session_root=dsh_session_root,
            ),
            encoding="utf-8",
        )
        env.update(
            {
                "DEEPSEEK_API_KEY": api_key,
                "DEEPSEEK_BASE_URL": normalize_openai_base(api_url),
                "DSH_HOME": os.fspath(dsh_home),
                "DSH_TELEMETRY_DISABLED": "1",
                "DSH_TELEMETRY_MODE": "DISABLED",
                "DSH_TOOLS_MODE": "native",
                "DSH_PERMISSION_MODE": "workspace-write",
                "NO_COLOR": "1",
            }
        )
        command = [
            args.npx_bin,
            "--yes",
            args.dsh_package,
            "--profile",
            "headless",
            "--patch",
            os.fspath(patch_path),
            prompt,
        ]
        public.update(
            {
                "credential_transport": "environment_only",
                "config_home_ephemeral": False,
                "session_home": os.fspath(dsh_home),
                "session_root": os.fspath(dsh_session_root),
                "session_persistence": "persistent",
                "patch_file": _relative_or_absolute(patch_path, private_run),
            }
        )
        return command, env, cleanup_paths, public

    if harness == "kimi_code":
        kimi_root = getattr(
            args,
            "kimi_session_root",
            DEFAULT_SESSION_ROOTS["kimi_code"],
        )
        kimi_home = _persistent_session_home(kimi_root, run_id)
        config_path = kimi_home / "config.toml"
        config_path.write_text(
            build_kimi_config(
                api_base=normalize_openai_base(api_url),
                api_key=api_key,
                model=model,
                mcp_only=args.kimi_mcp_only,
                thinking_enabled=not args.kimi_thinking_off,
            ),
            encoding="utf-8",
        )
        config_path.chmod(0o600)
        # Kimi Code 0.42.0 loads MCP declarations from the user-level
        # ``$KIMI_CODE_HOME/mcp.json``.  The published CLI does not accept the
        # older ``--mcp-config-file`` flag, so copy the evaluator-generated
        # config into this ephemeral home instead of putting it on argv.
        kimi_mcp_path = kimi_home / "mcp.json"
        kimi_mcp = json.loads(mcp_config.read_text(encoding="utf-8"))
        if not isinstance(kimi_mcp, dict):
            raise HarnessLaunchError("generated MCP config is not an object")
        servers = kimi_mcp.get("mcpServers")
        if not isinstance(servers, dict) or "libero" not in servers:
            raise HarnessLaunchError("generated MCP config has no libero server")
        libero_server = servers["libero"]
        if isinstance(libero_server, dict):
            libero_server.setdefault("startupTimeoutMs", 600000)
            libero_server.setdefault("toolTimeoutMs", 600000)
            libero_server.setdefault(
                "enabledTools",
                (
                    [
                        "start_episode",
                        "osc_sequence",
                        "finish_episode",
                    ]
                    if args.max_resets == 0
                    else [
                        "start_episode",
                        "osc_sequence",
                        "reset_episode",
                        "finish_episode",
                    ]
                ),
            )
        kimi_mcp_path.write_text(
            json.dumps(kimi_mcp, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        kimi_mcp_path.chmod(0o600)
        skills_dir = kimi_home / "empty-skills"
        skills_dir.mkdir(mode=0o700)
        env.update(
            {
                "KIMI_CODE_HOME": os.fspath(kimi_home),
                "KIMI_CODE_NO_AUTO_UPDATE": "1",
                "KIMI_CLI_NO_AUTO_UPDATE": "1",
                "KIMI_DISABLE_TELEMETRY": "1",
                "KIMI_CODE_BUILTIN_PRODUCT_SKILLS": "0",
                # The current Kimi Code v2 watcher uses chokidar on Linux.
                # Polling avoids a host-wide inotify quota being mistaken for
                # a model or MCP failure.
                "CHOKIDAR_USEPOLLING": "1",
                "CHOKIDAR_INTERVAL": "1000",
                "NO_COLOR": "1",
            }
        )
        command = build_kimi_command(
            npx_bin=args.npx_bin,
            package=args.kimi_package,
            model=model,
            prompt=prompt,
            mcp_config=mcp_config,
            skills_dir=skills_dir,
        )
        public.update(
            {
                "credential_transport": "tmpfs_config_toml",
                "config_home_ephemeral": False,
                "session_home": os.fspath(kimi_home),
                "session_persistence": "persistent",
                "config_file_removed_after_exit": False,
                "mcp_config_home_ephemeral": False,
                "watcher_mode": "chokidar_polling",
                "tool_policy": ("mcp_read_only" if args.kimi_mcp_only else "default"),
                # Keep the public receipt aligned with build_kimi_config:
                # this benchmark uses enabled Thinking at the controlled
                # ``high`` effort level (not the CLI/provider ``max`` label).
                "thinking_policy": "off" if args.kimi_thinking_off else "high",
            }
        )
        return command, env, cleanup_paths, public

    if harness == "qwen_code":
        qwen_root = getattr(
            args,
            "qwen_session_root",
            DEFAULT_SESSION_ROOTS["qwen_code"],
        )
        qwen_home = _persistent_session_home(qwen_root, run_id)
        settings_path = qwen_home / "settings.json"
        settings_path.write_text(
            json.dumps(
                build_qwen_settings(model=model, effort=args.qwen_effort),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        settings_path.chmod(0o600)
        env.update(
            {
                "OPENAI_API_KEY": api_key,
                "OPENAI_BASE_URL": normalize_openai_base(api_url),
                "OPENAI_MODEL": model,
                "QWEN_MODEL": model,
                "QWEN_HOME": os.fspath(qwen_home),
                "QWEN_RUNTIME_DIR": os.fspath(qwen_home),
                "QWEN_DISABLE_TELEMETRY": "1",
                "QWEN_DISABLE_USAGE_STATISTICS": "1",
                "NO_COLOR": "1",
            }
        )
        command = build_qwen_command(
            npx_bin=args.npx_bin,
            package=args.qwen_package,
            model=model,
            prompt=prompt,
            mcp_config=mcp_config,
        )
        public.update(
            {
                "credential_transport": "environment_only",
                "config_home_ephemeral": False,
                "session_home": os.fspath(qwen_home),
                "session_persistence": "persistent",
                "settings_file": _relative_or_absolute(settings_path, qwen_home),
                "reasoning_effort": args.qwen_effort,
                "reasoning_enabled": True,
                "tool_policy": "qwen_default_builtins_yolo",
                "mcp_server_allowlist": ["libero"],
            }
        )
        return command, env, cleanup_paths, public
    raise ValueError(f"unsupported harness: {harness}")


def _public_command(command: Sequence[str], prompt: str) -> list[str]:
    result = list(command)
    if result and result[-1] == prompt:
        result[-1] = "<task-prompt>"
    # Kimi's prompt follows --prompt rather than being last.
    for index, value in enumerate(result[:-1]):
        if value == "--prompt":
            result[index + 1] = "<task-prompt>"
    return result


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _redacted_file_text(path: Path, secret: str | None) -> str:
    """Read a text artifact and remove credential-shaped material."""

    value = path.read_text(encoding="utf-8", errors="replace")
    if secret:
        value = value.replace(secret, REDACTED)
    # Keep the generic scrub in addition to the exact replacement: a gateway
    # or CLI may render a bearer token in a normalized form.
    value = re.sub(r"(?i)sk-[A-Za-z0-9._~+/=-]{8,}", REDACTED, value)
    value = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}",
        rf"\1{REDACTED}",
        value,
    )
    return value


def _copy_redacted_if_exists(
    source: Path, destination: Path, *, secret: str | None = None
) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_redacted_file_text(source, secret), encoding="utf-8")


def _sanitize_file_in_place(path: Path | None, *, secret: str | None) -> None:
    if path is None or not path.is_file():
        return
    path.write_text(_redacted_file_text(path, secret), encoding="utf-8")


def _archive_ephemeral_jsonl(
    root: Path, destination: Path, *, secret: str | None
) -> list[Path]:
    """Preserve harness session logs before deleting an ephemeral home."""

    if not root.is_dir():
        return []
    archived: list[Path] = []
    for source in sorted(root.rglob("*.jsonl")):
        try:
            relative = source.relative_to(root)
        except ValueError:
            continue
        target = destination / relative
        _copy_redacted_if_exists(source, target, secret=secret)
        if target.is_file():
            archived.append(target)
    return archived


def _validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(float(args.max_wall_time_seconds))
        or args.max_wall_time_seconds <= 0
    ):
        raise ValueError("--max-wall-time-seconds must be finite and positive")
    if not 1 <= int(args.max_agent_steps) <= MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS:
        raise ValueError(
            f"--max-agent-steps must be between 1 and {MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS}"
        )
    if args.max_resets not in {0, 2}:
        raise ValueError("--max-resets must be 0 or 2")
    if (
        args.icl in {"video_only", "video_plus_trajectory"}
        and args.fixed_demo_master is None
    ):
        raise ValueError(f"--icl {args.icl} requires --fixed-demo-master")
    if args.icl == "none" and args.fixed_demo_master is not None:
        raise ValueError(
            "--fixed-demo-master requires --icl video_only or "
            "video_plus_trajectory"
        )
    if args.tmp_root is not None:
        args.tmp_root = args.tmp_root.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    args.claude_session_root = args.claude_session_root.expanduser().resolve()
    args.deepseek_session_root = args.deepseek_session_root.expanduser().resolve()
    args.kimi_session_root = args.kimi_session_root.expanduser().resolve()
    args.qwen_session_root = args.qwen_session_root.expanduser().resolve()
    if args.workspace_root is not None:
        args.workspace_root = args.workspace_root.expanduser().resolve()
    if args.robomemarena_root is not None:
        args.robomemarena_root = args.robomemarena_root.expanduser().resolve()
    if args.fixed_demo_master is not None:
        args.fixed_demo_master = args.fixed_demo_master.expanduser().resolve()
        if not args.fixed_demo_master.is_dir():
            raise FileNotFoundError(
                f"demonstration bundle is missing: {args.fixed_demo_master}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=HARNESS_CHOICES, required=True)
    parser.add_argument("--model")
    parser.add_argument("--suite", default="robomemarena")
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--profile", default="level3")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--initial-settle-control-steps", type=int, default=10)
    parser.add_argument("--max-agent-steps", type=int, default=1_000_000)
    parser.add_argument(
        "--max-wall-time-seconds",
        type=float,
        default=DEFAULT_MULTI_AGENT_WALL_TIME_SECONDS,
    )
    parser.add_argument("--max-resets", type=int, choices=(0, 2), default=0)
    parser.add_argument(
        "--icl",
        choices=("none", "video_only", "video_plus_trajectory"),
        default="none",
    )
    parser.add_argument("--fixed-demo-master", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path, default=SOURCE_ROOT / "agent_runs")
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--robomemarena-root", type=Path)
    parser.add_argument("--server-ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--launcher-grace-s", type=float, default=15.0)
    # The Anthropic-compatible gateway may buffer the first SSE event for
    # roughly 30 seconds; leave headroom while keeping the probe bounded.
    parser.add_argument("--model-probe-timeout-s", type=float, default=90.0)
    parser.add_argument("--api-url-file", type=Path, default=DEFAULT_API_URL_FILE)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--proxy", help="Optional HTTP(S) proxy; direct is the default")
    parser.add_argument(
        "--tmp-root",
        type=Path,
        help="Ephemeral parent (defaults to /dev/shm when writable)",
    )
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument(
        "--claude-effort",
        choices=("low", "medium", "high"),
        default="high",
        help="Claude Code reasoning effort (default: high)",
    )
    parser.add_argument("--npx-bin", default="npx")
    parser.add_argument("--dsh-package", default=DEFAULT_PACKAGES["deepseek_harness"])
    parser.add_argument("--kimi-package", default=DEFAULT_PACKAGES["kimi_code"])
    parser.add_argument("--qwen-package", default=DEFAULT_PACKAGES["qwen_code"])
    parser.add_argument(
        "--claude-session-root",
        type=Path,
        default=DEFAULT_SESSION_ROOTS["claude_code"],
        help=(
            "Persistent parent for Claude Code native sessions "
            "(default: ../.claude/libero-runs)"
        ),
    )
    parser.add_argument(
        "--deepseek-session-root",
        type=Path,
        default=DEFAULT_SESSION_ROOTS["deepseek_harness"],
        help=(
            "Persistent parent for DeepSeek Harness native sessions "
            "(default: ../.deepseek/libero-runs)"
        ),
    )
    parser.add_argument(
        "--kimi-session-root",
        type=Path,
        default=DEFAULT_SESSION_ROOTS["kimi_code"],
        help=(
            "Persistent parent for Kimi Code native sessions "
            "(default: ../.kimi/libero-runs)"
        ),
    )
    parser.add_argument(
        "--qwen-session-root",
        type=Path,
        default=DEFAULT_SESSION_ROOTS["qwen_code"],
        help=(
            "Persistent parent for Qwen Code native sessions "
            "(default: ../.qwen/libero-runs)"
        ),
    )
    parser.add_argument(
        "--qwen-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="high",
        help="Qwen Code reasoning effort (default: high)",
    )
    parser.add_argument("--skip-model-probe", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--kimi-mcp-only",
        action="store_true",
        help="For Kimi diagnostics, allow observation reads and evaluator MCP only",
    )
    parser.add_argument(
        "--kimi-thinking-off",
        action="store_true",
        help="Disable Kimi Thinking for a controlled protocol diagnostic",
    )
    return parser.parse_args(argv)


def _start_launcher(
    args: argparse.Namespace,
    *,
    run_id: str,
    staging: Path,
) -> tuple[
    subprocess.Popen[str],
    dict[str, str],
    threading.Thread,
    threading.Thread,
    queue.Queue[str | None],
]:
    command = _launcher_command(args, run_id)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (os.fspath(SOURCE_ROOT), env.get("PYTHONPATH")) if item
    )
    stdout_path = staging / "launcher_stdout.log"
    stderr_path = staging / "launcher_stderr.log"
    process = subprocess.Popen(
        command,
        cwd=SOURCE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    first_lines: queue.Queue[str | None] = queue.Queue()
    stdout_thread = threading.Thread(
        target=_pump_stream,
        args=(process.stdout, stdout_path, first_lines),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_pump_stream,
        args=(process.stderr, stderr_path),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    return (
        process,
        {"stdout": os.fspath(stdout_path), "stderr": os.fspath(stderr_path)},
        stdout_thread,
        stderr_thread,
        first_lines,
    )


def _wait_for_launcher_metadata(
    process: subprocess.Popen[str],
    first_lines: queue.Queue[str | None],
    *,
    timeout: float,
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = first_lines.get(timeout=0.25)
        except queue.Empty:
            if process.poll() is not None:
                break
            continue
        if line is None:
            break
        _parse_launcher_line(line, metadata)
        # mcp_config_file is printed only after server readiness has been
        # verified, so it is the synchronization point for the Agent.
        if "mcp_config_file" in metadata:
            return metadata
    missing = sorted(
        {"run_id", "workspace", "private_run", "prompt_file", "mcp_config_file"}
        - metadata.keys()
    )
    raise HarnessLaunchError(
        "LIBERO launcher did not become ready; missing " + ", ".join(missing)
    )


def _prepare_manifest(
    *,
    args: argparse.Namespace,
    harness: str,
    model: str,
    run_id: str,
    private_run: Path,
    workspace: Path,
    probe: GatewayProbe | None,
    version: Mapping[str, Any],
    command: Sequence[str],
    runtime_public: Mapping[str, Any],
) -> dict[str, Any]:
    source_commit = "unknown"
    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SOURCE_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "schema_version": "libero.multi_agent_harness_run.v1",
        "run_id": run_id,
        "created_at": _utc_now(),
        "harness": harness,
        "model_requested": model,
        "harness_version": dict(version),
        "api_protocol": _harness_protocol(harness),
        "api_base_public": (
            normalize_anthropic_base(_read_url_file(args.api_url_file))
            if harness == "claude_code"
            else normalize_openai_base(_read_url_file(args.api_url_file))
        ),
        "credential_source": os.fspath(args.api_key_file),
        "credential_material": "never_recorded",
        "model_probe": None if probe is None else probe.as_dict(),
        "source_checkout": os.fspath(SOURCE_ROOT),
        "source_commit": source_commit,
        "workspace": os.fspath(workspace),
        "private_run": os.fspath(private_run),
        "launcher": {
            "suite": args.suite,
            "task_id": args.task_id,
            "init_state_id": args.init_state_id,
            "profile": args.profile,
            "seed": args.seed,
            "resolution": args.resolution,
            "render_gpu_device_id": args.render_gpu_device_id,
            "initial_settle_control_steps": args.initial_settle_control_steps,
            "max_agent_steps": args.max_agent_steps,
            "max_wall_time_seconds": args.max_wall_time_seconds,
            "max_resets": args.max_resets,
            "icl_condition": args.icl,
            "fixed_demo_master": (
                None
                if args.fixed_demo_master is None
                else os.fspath(args.fixed_demo_master)
            ),
            "action_interface": "native_osc_sequence",
            "control_transport": "mcp",
        },
        "command": _public_command(
            command,
            (
                Path(private_run / "agent_prompt.txt").read_text(encoding="utf-8")
                if (private_run / "agent_prompt.txt").is_file()
                else ""
            ),
        ),
        **dict(runtime_public),
    }


def _cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except OSError:
            pass


def _run_episode(
    args: argparse.Namespace,
    *,
    model: str,
    probe: GatewayProbe | None,
    version: Mapping[str, Any],
) -> int:
    run_id = args.run_id or _new_run_id(args.harness, model)
    staging_root = _tmp_parent(args.tmp_root)
    staging = Path(tempfile.mkdtemp(prefix="libero-harness-launch-", dir=staging_root))
    launcher: subprocess.Popen[str] | None = None
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    staging_logs: dict[str, str] = {}
    cleanup_paths: list[Path] = []
    private_run: Path | None = None
    agent: subprocess.Popen[Any] | None = None
    manifest: dict[str, Any] = {}
    infrastructure_error: str | None = None
    result: dict[str, Any] = {}
    harness_stdout: Path | None = None
    harness_stderr: Path | None = None
    api_key: str | None = None
    harness_session_logs: list[Path] = []
    started = time.monotonic()
    try:
        launcher, staging_logs, stdout_thread, stderr_thread, first_lines = (
            _start_launcher(args, run_id=run_id, staging=staging)
        )
        metadata = _wait_for_launcher_metadata(
            launcher, first_lines, timeout=args.server_ready_timeout_s
        )
        private_run = Path(metadata["private_run"]).resolve()
        workspace = Path(metadata["workspace"]).resolve()
        prompt_file = Path(metadata["prompt_file"]).resolve()
        mcp_config = Path(metadata["mcp_config_file"]).resolve()
        if (
            not private_run.is_dir()
            or not workspace.is_dir()
            or not mcp_config.is_file()
        ):
            raise HarnessLaunchError("LIBERO launcher reported invalid runtime paths")
        prompt = prompt_file.read_text(encoding="utf-8")
        api_key = _read_key_file(args.api_key_file)
        command, env, cleanup_paths, runtime_public = _build_harness_runtime(
            harness=args.harness,
            model=model,
            prompt=prompt,
            run_id=run_id,
            workspace=workspace,
            private_run=private_run,
            mcp_config=mcp_config,
            api_url=_read_url_file(args.api_url_file),
            api_key=api_key,
            args=args,
        )
        manifest = _prepare_manifest(
            args=args,
            harness=args.harness,
            model=model,
            run_id=metadata.get("run_id", run_id),
            private_run=private_run,
            workspace=workspace,
            probe=probe,
            version=version,
            command=command,
            runtime_public=runtime_public,
        )
        manifest["launcher_metadata"] = {key: value for key, value in metadata.items()}
        _write_json(private_run / "harness_manifest.json", manifest)
        suffix = (
            "jsonl"
            if args.harness in {"claude_code", "kimi_code", "qwen_code"}
            else "log"
        )
        harness_stdout = private_run / f"{args.harness}_stdout.{suffix}"
        harness_stderr = private_run / f"{args.harness}_stderr.log"
        with harness_stdout.open(
            "w", encoding="utf-8"
        ) as stdout_file, harness_stderr.open("w", encoding="utf-8") as stderr_file:
            agent = subprocess.Popen(
                command,
                cwd=workspace,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            deadline = (
                time.monotonic()
                + args.max_wall_time_seconds
                + args.launcher_grace_s
                + 30.0
            )
            agent_exit_seen_at: float | None = None
            finished_seen_at: float | None = None
            while time.monotonic() < deadline:
                result = _read_json(private_run / "result.json")
                if result.get("status") == "finished":
                    if finished_seen_at is None:
                        finished_seen_at = time.monotonic()
                    if (
                        agent.poll() is not None
                        or time.monotonic() - finished_seen_at >= 5.0
                    ):
                        break
                elif agent.poll() is not None:
                    if agent_exit_seen_at is None:
                        agent_exit_seen_at = time.monotonic()
                    if time.monotonic() - agent_exit_seen_at >= args.launcher_grace_s:
                        infrastructure_error = "harness exited before finish_episode"
                        break
                if launcher.poll() is not None and result.get("status") != "finished":
                    infrastructure_error = (
                        "LIBERO launcher exited before finish_episode"
                    )
                    break
                time.sleep(0.25)
            else:
                infrastructure_error = (
                    "multi-agent rollout wall-clock supervision timeout"
                )
            if agent.poll() is None:
                # npm/npx is a small wrapper around the actual Node process;
                # terminate the whole session so a child cannot outlive the
                # evaluator after a wall-clock timeout.
                _terminate_process_group(agent)
        # Let a normally finished launcher flush its result; otherwise stop
        # both the launcher and its independently-sessioned simulator child.
        if result.get("status") == "finished":
            try:
                launcher.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                _terminate_process_group(launcher)
        else:
            _terminate_process_group(launcher)
            _terminate_episode_processes(
                private_run,
                result_path=private_run / "result.json",
            )
    except BaseException as exc:
        infrastructure_error = _redact(f"{type(exc).__name__}: {exc}")
        if launcher is not None:
            _terminate_process_group(launcher)
        if agent is not None:
            _terminate_process_group(agent)
        if private_run is not None:
            _terminate_episode_processes(
                private_run,
                result_path=private_run / "result.json",
            )
    finally:
        if launcher is not None and launcher.poll() is None:
            _terminate_process_group(launcher)
        if agent is not None and agent.poll() is None:
            _terminate_process_group(agent)
        if stdout_thread is not None:
            stdout_thread.join(timeout=3.0)
        if stderr_thread is not None:
            stderr_thread.join(timeout=3.0)
        if private_run is not None:
            _copy_if_exists(
                Path(staging_logs.get("stdout", "")),
                private_run / "launcher_stdout.log",
            )
            _copy_if_exists(
                Path(staging_logs.get("stderr", "")),
                private_run / "launcher_stderr.log",
            )
            # The harness is deliberately allowed to use the credential in
            # memory/environment, but any reflected value is scrubbed before
            # artifacts become part of the evaluator record.
            _sanitize_file_in_place(harness_stdout, secret=api_key)
            _sanitize_file_in_place(harness_stderr, secret=api_key)
            # Keep a redacted copy in the run directory for portable audit
            # and leave the native home in place when persistence is enabled.
            # Kimi remains ephemeral for now, so this archive runs before its
            # cleanup path is removed as well.
            session_home_value = manifest.get("session_home")
            if isinstance(session_home_value, str) and session_home_value:
                harness_session_logs = _archive_ephemeral_jsonl(
                    Path(session_home_value),
                    private_run / "harness_sessions",
                    secret=api_key,
                )
            result = _read_json(private_run / "result.json")
            output_paths = [
                path for path in (harness_stdout, harness_stderr) if path is not None
            ]
            usage_paths = [path for path in output_paths if path.suffix == ".jsonl"]
            usage_paths.extend(harness_session_logs)
            # DeepSeek's durable session log is also public audit material but
            # lives under the generated Agent workspace rather than private_run.
            if args.harness == "deepseek_harness":
                usage_paths.extend(workspace.rglob("*.jsonl"))
            usage = _usage_from_jsonl(usage_paths)
            _write_json(private_run / "harness_usage.json", usage)
            manifest.update(
                {
                    "finished_at": _utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "agent_exit_code": None if agent is None else agent.returncode,
                    "launcher_exit_code": (
                        None if launcher is None else launcher.returncode
                    ),
                    "infrastructure_error": infrastructure_error,
                    "result_status": result.get("status"),
                    "task_success": result.get("success"),
                    "session_ids": _session_ids_from_jsonl(
                        [*output_paths, *harness_session_logs]
                    ),
                    "artifacts": {
                        "run_manifest": "run_manifest.json",
                        "episode_result": "result.json",
                        "harness_manifest": "harness_manifest.json",
                        "harness_usage": "harness_usage.json",
                        "agent_stdout": (
                            None if harness_stdout is None else harness_stdout.name
                        ),
                        "agent_stderr": (
                            None if harness_stderr is None else harness_stderr.name
                        ),
                        "harness_sessions": (
                            [
                                _relative_or_absolute(path, private_run)
                                for path in harness_session_logs
                            ]
                            or None
                        ),
                        "launcher_stdout": "launcher_stdout.log",
                        "launcher_stderr": "launcher_stderr.log",
                    },
                }
            )
            _write_json(private_run / "harness_manifest.json", manifest)
        _cleanup_paths(cleanup_paths)
        try:
            shutil.rmtree(staging)
        except OSError:
            pass
    if private_run is None:
        print(
            json.dumps(
                {
                    "harness": args.harness,
                    "model": model,
                    "reasoning_effort": (
                        args.claude_effort
                        if args.harness == "claude_code"
                        else args.qwen_effort
                        if args.harness == "qwen_code"
                        else None
                    ),
                    "status": "infrastructure_error",
                    "error": infrastructure_error,
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "harness": args.harness,
                "model": model,
                "run_id": run_id,
                "private_run": os.fspath(private_run),
                "result_status": result.get("status"),
                "task_success": result.get("success"),
                "infrastructure_error": infrastructure_error,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return (
        0 if result.get("status") == "finished" and infrastructure_error is None else 2
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    model = args.model or DEFAULT_MODELS[args.harness]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "harness": args.harness,
                    "model": model,
                    "reasoning_effort": (
                        args.claude_effort
                        if args.harness == "claude_code"
                        else args.qwen_effort
                        if args.harness == "qwen_code"
                        else None
                    ),
                    "suite": args.suite,
                    "task_id": args.task_id,
                    "max_agent_steps": args.max_agent_steps,
                    "max_wall_time_seconds": args.max_wall_time_seconds,
                    "session_root": (
                        os.fspath(args.claude_session_root)
                        if args.harness == "claude_code"
                        else os.fspath(args.deepseek_session_root)
                        if args.harness == "deepseek_harness"
                        else os.fspath(args.kimi_session_root)
                        if args.harness == "kimi_code"
                        else os.fspath(args.qwen_session_root)
                    ),
                    "api_url_file": os.fspath(args.api_url_file),
                    "credential_material": "not_read_in_dry_run",
                },
                ensure_ascii=False,
            )
        )
        return 0
    api_url = _read_url_file(args.api_url_file)
    api_key = _read_key_file(args.api_key_file)
    base_env = _clean_child_env(os.environ, args.proxy)
    _remove_ambient_credentials(base_env)
    version = detect_harness_version(
        args.harness,
        claude_bin=args.claude_bin,
        npx_bin=args.npx_bin,
        dsh_package=args.dsh_package,
        kimi_package=args.kimi_package,
        qwen_package=args.qwen_package,
        env=base_env,
    )
    probe: GatewayProbe | None = None
    if not args.skip_model_probe:
        probe = probe_model_gateway(
            api_url=api_url,
            protocol=_harness_protocol(args.harness),
            model=model,
            api_key=api_key,
            timeout=args.model_probe_timeout_s,
            proxy=args.proxy,
        )
        print(
            json.dumps(
                {
                    "harness": args.harness,
                    "model": model,
                    "harness_version": version,
                    "model_probe": probe.as_dict(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if probe.state != "available":
            # The active probe is intentionally before simulator allocation:
            # a 503/no-channel result is a model routing issue, not a harness
            # or LIBERO failure.
            return 3
    if args.preflight_only:
        return 0
    return _run_episode(args, model=model, probe=probe, version=version)


if __name__ == "__main__":
    raise SystemExit(main())
