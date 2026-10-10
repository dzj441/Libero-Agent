"""One-episode service shared by the stdin and Unix-socket transports."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from .artifacts import replace_current_public_observation
from .control import ActionInterface
from .environment import LiberoAgentEnv


EpisodeServiceFactory = Callable[[int, Path], "AgentEpisodeService"]
EpisodeStartHook = Callable[[int, Path], Mapping[str, Any] | None]
EpisodeCloseHook = Callable[[int], None]
AgentEnvFactory = Callable[[], LiberoAgentEnv]
MonotonicClock = Callable[[], float]


DEFAULT_EPISODE_WALL_TIME_SECONDS = 30 * 60
WALL_TIME_BUDGET_SCOPE = "per_episode_across_resets"
WALL_TIME_BUDGET_EXHAUSTED = "episode_wall_time_budget_exhausted"


class AgentEpisodeService:
    """Expose one selected robot action command and keep private audit data."""

    def __init__(
        self,
        agent_env: LiberoAgentEnv,
        *,
        workspace_directory: str | Path,
        current_observation_directory: str | Path,
        private_run_directory: str | Path | None = None,
        action_interface: ActionInterface | str = ActionInterface.METRIC_OSC_STEP,
        reset_agent_env: AgentEnvFactory | None = None,
        max_resets: int = 0,
        max_wall_time_seconds: float | None = DEFAULT_EPISODE_WALL_TIME_SECONDS,
        monotonic_clock: MonotonicClock = time.monotonic,
    ) -> None:
        self.agent_env = agent_env
        self.action_interface = ActionInterface.parse(action_interface)
        self.workspace_directory = Path(workspace_directory).resolve()
        self.current_observation_directory = Path(
            current_observation_directory
        ).resolve()
        self.private_run_directory = (
            None
            if private_run_directory is None
            else Path(private_run_directory).resolve()
        )
        if max_resets not in {0, 2}:
            raise ValueError("max_resets must be either 0 or 2")
        if max_resets and reset_agent_env is None:
            raise ValueError(
                "positive max_resets requires a reset_agent_env factory"
            )
        self.reset_agent_env = reset_agent_env
        self.max_resets = int(max_resets)
        if max_wall_time_seconds is not None:
            max_wall_time_seconds = float(max_wall_time_seconds)
            if not np.isfinite(max_wall_time_seconds) or max_wall_time_seconds <= 0:
                raise ValueError(
                    "max_wall_time_seconds must be finite and positive when provided"
                )
        self.max_wall_time_seconds = max_wall_time_seconds
        self._monotonic_clock = monotonic_clock
        self._episode_started_at_monotonic: float | None = None
        self._episode_deadline_monotonic: float | None = None
        self._terminal_response: dict[str, Any] | None = None
        self.reset_count = 0
        self._accepted_agent_steps_before_current_attempt = 0
        self._observation_namespace = ""
        self._pending_reset_audit: dict[str, Any] | None = None
        self.state = "ready"
        self.latest_observation_id: str | None = None
        self._event_index = 0
        if self.private_run_directory is not None:
            self.private_run_directory.mkdir(parents=True, exist_ok=True)
            (self.private_run_directory / "private_observations").mkdir(
                parents=True, exist_ok=True
            )

    @property
    def finished(self) -> bool:
        return self.state == "finished"

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expired = self.expire_wall_time_budget_if_needed()
        if expired is not None:
            return expired
        command = request.get("command")
        if command == "start":
            response = self._start()
        elif command == "reset":
            response = self._reset(request)
        elif (
            command == "osc_step"
            and self.action_interface is ActionInterface.METRIC_OSC_STEP
        ):
            response = self._osc_step(request)
        elif (
            command == "osc_sequence"
            and self.action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
        ):
            response = self._osc_sequence(request)
        elif command == "finish":
            response = self._finish(request)
        else:
            raise ValueError(
                "unknown command; expected start, "
                f"{self.action_interface.wire_command}, reset, or finish"
            )
        self._record_event(request=request, response=response)
        return response

    def record_error(self, request: object, exc: BaseException) -> None:
        self._record_event(
            request=request,
            response={
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

    def finalize_aborted(self, reason: str) -> None:
        if self.finished:
            return
        self.state = "aborted"
        result = {
            "schema_version": "libero.agent_run_result.v1",
            "status": "aborted",
            "reason": str(reason),
            "finished_at": _utc_now(),
            **self._wall_time_result_fields(),
        }
        snapshot_method = getattr(
            self.agent_env, "private_evaluation_snapshot", None
        )
        private_evaluation = (
            snapshot_method() if callable(snapshot_method) else None
        )
        if isinstance(private_evaluation, Mapping):
            result["private_evaluation"] = deepcopy(
                dict(private_evaluation)
            )
            self._write_private_stage_events(private_evaluation)
        self._write_result(result)

    def close(self) -> None:
        self.agent_env.close()

    def _start(self) -> dict[str, Any]:
        if self.state != "ready":
            raise RuntimeError("start may be called exactly once")
        result = self.agent_env.start_episode()
        observation = result["observation"]
        observation_file = self._publish(observation)
        self._episode_started_at_monotonic = self._monotonic_clock()
        self._episode_deadline_monotonic = (
            None
            if self.max_wall_time_seconds is None
            else self._episode_started_at_monotonic + self.max_wall_time_seconds
        )
        self.state = "active"
        return {
            "ok": True,
            "observation_id": self.latest_observation_id,
            "observation_file": observation_file,
            "max_agent_steps": self.agent_env.max_agent_steps,
            "attempt_index": 1,
            "reset_count": 0,
            "remaining_resets": self.max_resets,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "wall_time_budget_scope": WALL_TIME_BUDGET_SCOPE,
            "execution": {},
        }

    def _reset(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Discard the active rollout and recreate the same task episode.

        A reset is deliberately a service operation rather than an ordinary
        environment step.  The launcher supplies a factory so the old live
        simulator is closed and a fresh evaluator-owned environment is built
        with the exact same task, init state, seed, and controller contract.
        """

        if self.state != "active":
            raise RuntimeError("reset requires one active episode")
        self._require_latest_observation(request)
        if self.reset_count >= self.max_resets:
            raise RuntimeError(
                f"no resets remaining (maximum {self.max_resets} per episode)"
            )
        if self.reset_agent_env is None:
            raise RuntimeError("reset is unavailable for this episode service")

        old_agent_env = self.agent_env
        snapshot_method = getattr(
            old_agent_env, "private_evaluation_snapshot", None
        )
        previous_private_evaluation = (
            snapshot_method() if callable(snapshot_method) else None
        )
        previous_agent_steps = getattr(old_agent_env, "_agent_step_index", None)
        next_reset_count = self.reset_count + 1
        pending_reset_audit = {
            "reset_count": next_reset_count,
            "previous_attempt_index": next_reset_count,
            "previous_accepted_agent_steps": (
                None
                if previous_agent_steps is None
                else int(previous_agent_steps)
            ),
            "previous_private_evaluation": (
                None
                if not isinstance(previous_private_evaluation, Mapping)
                else deepcopy(dict(previous_private_evaluation))
            ),
        }
        new_agent_env: LiberoAgentEnv | None = None
        try:
            # MuJoCo's EGL renderer is process-global enough that briefly
            # keeping the old and replacement environments alive together is
            # unsafe: closing the old environment can invalidate the new
            # environment's render context.  Reset is explicitly destructive,
            # so retire the old simulator before constructing its replacement.
            # A reconstruction failure therefore aborts the service rather
            # than attempting to continue a partially retired attempt.
            old_agent_env.close()
            new_agent_env = self.reset_agent_env()
            result = new_agent_env.start_episode()
            self._observation_namespace = f"reset_{next_reset_count:02d}_"
            observation_file = self._publish(result["observation"])
        except BaseException:
            self._pending_reset_audit = pending_reset_audit
            if new_agent_env is not None:
                try:
                    new_agent_env.close()
                except BaseException:
                    pass
            self.state = "aborted"
            raise
        self.agent_env = new_agent_env
        if previous_agent_steps is not None:
            self._accepted_agent_steps_before_current_attempt += int(
                previous_agent_steps
            )
        self.reset_count = next_reset_count
        self._pending_reset_audit = pending_reset_audit
        return {
            "ok": True,
            "observation_id": self.latest_observation_id,
            "observation_file": observation_file,
            "attempt_index": self.reset_count + 1,
            "reset_count": self.reset_count,
            "remaining_resets": self.max_resets - self.reset_count,
            "max_agent_steps": self.agent_env.max_agent_steps,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "wall_time_budget_scope": WALL_TIME_BUDGET_SCOPE,
            "execution": {},
        }

    def _osc_step(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self.state != "active":
            raise RuntimeError("osc_step requires one active episode")
        self._require_latest_observation(request)
        result = self.agent_env.step_osc_target(
            delta_position_m=request.get("delta_position_m", (0.0, 0.0, 0.0)),
            delta_rotation_rotvec_rad=request.get(
                "delta_rotation_rotvec_rad", (0.0, 0.0, 0.0)
            ),
            delta_gripper_width_m=request.get("delta_gripper_width_m", 0.0),
        )
        observation = result["observation"]
        observation_file = self._publish(observation)
        return {
            "ok": True,
            "observation_id": self.latest_observation_id,
            "observation_file": observation_file,
            "execution": result["execution"],
        }

    def _osc_sequence(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self.state != "active":
            raise RuntimeError("osc_sequence requires one active episode")
        self._require_latest_observation(request)
        result = self.agent_env.step_osc_sequence(actions=request.get("actions", ()))
        observation = result["observation"]
        observation_file = self._publish(observation)
        return {
            "ok": True,
            "observation_id": self.latest_observation_id,
            "observation_file": observation_file,
            "execution": result["execution"],
        }

    def _finish(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self.state != "active":
            raise RuntimeError("finish requires one active episode")
        self._require_latest_observation(request)
        result = self.agent_env.finish_episode()
        private_evaluation = result.get("private_evaluation")
        self.state = "finished"
        current_attempt_steps = int(result["accepted_agent_steps"])
        total_steps = (
            self._accepted_agent_steps_before_current_attempt
            + current_attempt_steps
        )
        response = {
            "ok": True,
            "status": "finished",
            "success": bool(result["success"]),
            "accepted_agent_steps": total_steps,
            "accepted_agent_steps_current_attempt": current_attempt_steps,
            "attempt_index": self.reset_count + 1,
            "reset_count": self.reset_count,
            "remaining_resets": self.max_resets - self.reset_count,
            **self._wall_time_result_fields(),
        }
        private_result = {
            "schema_version": "libero.agent_run_result.v1",
            **deepcopy(response),
            "finished_at": _utc_now(),
        }
        if isinstance(private_evaluation, Mapping):
            private_result["private_evaluation"] = deepcopy(
                dict(private_evaluation)
            )
            self._write_private_stage_events(private_evaluation)
        self._write_result(private_result)
        self._terminal_response = deepcopy(response)
        return response

    def expire_wall_time_budget_if_needed(self) -> dict[str, Any] | None:
        """Finish an active episode once its evaluator wall-clock budget expires.

        The deadline belongs to the logical query, not to an individual reset
        attempt. Reconstructing the simulator therefore never changes it. This
        method is safe to poll from a socket-server timeout even when the Agent
        sends no further commands.
        """

        if self.state == "finished":
            return None
        if self.state != "active" or self._episode_deadline_monotonic is None:
            return None
        if self._monotonic_clock() < self._episode_deadline_monotonic:
            return None

        current_attempt_steps = int(
            getattr(self.agent_env, "_agent_step_index", 0)
        )
        total_steps = (
            self._accepted_agent_steps_before_current_attempt
            + current_attempt_steps
        )
        response = {
            "ok": True,
            "status": "finished",
            "success": False,
            "termination_reason": WALL_TIME_BUDGET_EXHAUSTED,
            "accepted_agent_steps": total_steps,
            "accepted_agent_steps_current_attempt": current_attempt_steps,
            "attempt_index": self.reset_count + 1,
            "reset_count": self.reset_count,
            "remaining_resets": self.max_resets - self.reset_count,
            **self._wall_time_result_fields(),
        }
        private_result = {
            "schema_version": "libero.agent_run_result.v1",
            **deepcopy(response),
            "finished_at": _utc_now(),
        }
        snapshot_method = getattr(
            self.agent_env, "private_evaluation_snapshot", None
        )
        private_evaluation = (
            snapshot_method() if callable(snapshot_method) else None
        )
        if isinstance(private_evaluation, Mapping):
            private_result["private_evaluation"] = deepcopy(
                dict(private_evaluation)
            )
            self._write_private_stage_events(private_evaluation)
        self.state = "finished"
        self._terminal_response = deepcopy(response)
        self._write_result(private_result)
        self._record_event(
            request={"command": "wall_time_budget_expired"},
            response=response,
        )
        return deepcopy(response)

    def _wall_time_result_fields(self) -> dict[str, Any]:
        elapsed = None
        if self._episode_started_at_monotonic is not None:
            elapsed = max(
                0.0,
                self._monotonic_clock() - self._episode_started_at_monotonic,
            )
        return {
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "wall_time_budget_scope": WALL_TIME_BUDGET_SCOPE,
            "wall_time_seconds": elapsed,
        }

    def _write_private_stage_events(
        self, private_evaluation: Mapping[str, Any]
    ) -> None:
        if self.private_run_directory is None:
            return
        events = private_evaluation.get("stage_events", ())
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            return
        path = self.private_run_directory / "private_stage_events.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for event in events:
                if isinstance(event, Mapping):
                    stream.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _publish(self, observation: Mapping[str, Any]) -> str:
        if self._observation_namespace:
            observation = deepcopy(dict(observation))
            observation["observation_id"] = (
                self._observation_namespace + str(observation["observation_id"])
            )
        json_path = replace_current_public_observation(
            observation, self.current_observation_directory
        ).resolve()
        try:
            relative_path = json_path.relative_to(self.workspace_directory)
        except ValueError as exc:
            raise ValueError(
                "current observation directory must be inside the agent workspace"
            ) from exc

        if self.private_run_directory is not None:
            observation_id = str(observation["observation_id"])
            destination = (
                self.private_run_directory / "private_observations" / observation_id
            )
            if destination.exists():
                raise RuntimeError(f"duplicate observation id: {observation_id}")
            shutil.copytree(self.current_observation_directory, destination)
        self.latest_observation_id = str(observation["observation_id"])
        return relative_path.as_posix()

    def _require_latest_observation(self, request: Mapping[str, Any]) -> None:
        requested = request.get("observation_id")
        if not isinstance(requested, str) or not requested:
            raise ValueError("observation_id must identify the latest observation")
        if requested != self.latest_observation_id:
            raise ValueError(
                "observation_id is stale; latest observation is "
                f"{self.latest_observation_id!r}"
            )

    def _record_event(self, request: object, response: Mapping[str, Any]) -> None:
        if self.private_run_directory is None:
            return
        event = {
            "schema_version": "libero.agent_action_event.v1",
            "event_index": self._event_index,
            "recorded_at": _utc_now(),
            "request": _jsonable(request),
            "response": _jsonable(response),
            "attempt_index": self.reset_count + 1,
        }
        if request_is_reset(request):
            if self._pending_reset_audit is not None:
                event["private_reset_audit"] = _jsonable(
                    self._pending_reset_audit
                )
            self._pending_reset_audit = None
        self._event_index += 1
        path = self.private_run_directory / "actions.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_result(self, result: Mapping[str, Any]) -> None:
        if self.private_run_directory is None:
            return
        _write_json_atomic(self.private_run_directory / "result.json", result)


class MultiEpisodeService:
    """Run multiple isolated LIBERO episodes through one Agent session.

    Every child remains a regular :class:`AgentEpisodeService`. The wrapper
    only owns sequencing, current-task disclosure, root audit aggregation, and
    the final curriculum result. A child environment is created lazily at its
    ``start`` call and closed immediately after its matching ``finish`` call.
    """

    def __init__(
        self,
        *,
        task_instructions: Sequence[str],
        service_factory: EpisodeServiceFactory,
        private_run_directory: str | Path,
        before_episode_start: EpisodeStartHook | None = None,
        after_episode_close: EpisodeCloseHook | None = None,
    ) -> None:
        instructions = [" ".join(str(value).split()) for value in task_instructions]
        if not instructions or any(not value for value in instructions):
            raise ValueError("curriculum requires non-empty task instructions")
        self.task_instructions = tuple(instructions)
        self.service_factory = service_factory
        self.before_episode_start = before_episode_start
        self.after_episode_close = after_episode_close
        self.private_run_directory = Path(private_run_directory).resolve()
        self.private_run_directory.mkdir(parents=True, exist_ok=True)
        (self.private_run_directory / "episodes").mkdir(parents=True, exist_ok=True)
        self.state = "ready"
        self.current_service: AgentEpisodeService | None = None
        self.current_episode_index: int | None = None
        self.next_episode_index = 0
        self.episode_results: list[dict[str, Any]] = []
        self._event_index = 0

    @property
    def finished(self) -> bool:
        return self.state == "finished"

    @property
    def episode_count(self) -> int:
        return len(self.task_instructions)

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expired = self.expire_wall_time_budget_if_needed()
        if expired is not None:
            return expired
        command = request.get("command")
        episode_index = self._request_episode_index(command)
        if command == "start":
            response = self._start_next_episode()
        else:
            if self.current_service is None or self.current_episode_index is None:
                raise RuntimeError("start must begin the next prepared episode first")
            response = self.current_service.handle(request)
            if response.get("termination_reason") == WALL_TIME_BUDGET_EXHAUSTED:
                response = self._complete_timed_out_episode(response)
                self._record_event(episode_index, request, response)
                return response
            response = self._augment_response(response, self.current_episode_index)
            if command == "finish":
                response = self._finish_current_episode(response)
        self._record_event(episode_index, request, response)
        return response

    def record_error(self, request: object, exc: BaseException) -> None:
        if self.current_service is not None:
            self.current_service.record_error(request, exc)
        episode_index = (
            self.current_episode_index
            if self.current_episode_index is not None
            else min(self.next_episode_index, self.episode_count - 1)
        )
        self._record_event(
            episode_index,
            request,
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

    def finalize_aborted(self, reason: str) -> None:
        if self.finished:
            return
        if self.current_service is not None:
            self.current_service.finalize_aborted(reason)
            self._close_current_service()
        self.state = "aborted"
        self._write_root_result(status="aborted", reason=str(reason))

    def close(self) -> None:
        self._close_current_service()

    def expire_wall_time_budget_if_needed(self) -> dict[str, Any] | None:
        """Terminate the curriculum if its active episode exhausts wall time."""

        if self.finished or self.current_service is None:
            return None
        response = self.current_service.expire_wall_time_budget_if_needed()
        if response is None:
            return None
        if response.get("termination_reason") != WALL_TIME_BUDGET_EXHAUSTED:
            return None
        return self._complete_timed_out_episode(response)

    def _complete_timed_out_episode(
        self, response: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self.current_episode_index is None:
            raise RuntimeError("timed-out curriculum has no active episode")
        episode_index = self.current_episode_index
        summary = {
            "episode_index": episode_index,
            "task_instruction": self.task_instructions[episode_index],
            "status": "finished",
            "success": False,
            "termination_reason": WALL_TIME_BUDGET_EXHAUSTED,
            "accepted_agent_steps": int(response["accepted_agent_steps"]),
            "wall_time_seconds": response.get("wall_time_seconds"),
            "max_wall_time_seconds": response.get("max_wall_time_seconds"),
            "wall_time_budget_scope": response.get("wall_time_budget_scope"),
            "directory": self._episode_directory(episode_index).relative_to(
                self.private_run_directory
            ).as_posix(),
        }
        self.episode_results.append(summary)
        self.next_episode_index = episode_index + 1
        self._close_current_service()
        self.state = "finished"
        self._write_progress()
        self._write_root_result(
            status="finished", reason=WALL_TIME_BUDGET_EXHAUSTED
        )
        return {
            **dict(response),
            "episode_index": episode_index,
            "episode_count": self.episode_count,
            "next_episode_available": False,
            "curriculum_complete": True,
        }

    def _request_episode_index(self, command: object) -> int:
        if command == "start":
            if self.current_service is not None:
                raise RuntimeError("finish the active episode before starting another")
            if self.next_episode_index >= self.episode_count:
                raise RuntimeError("all prepared episodes have already finished")
            return self.next_episode_index
        if self.current_episode_index is None:
            return min(self.next_episode_index, self.episode_count - 1)
        return self.current_episode_index

    def _start_next_episode(self) -> dict[str, Any]:
        episode_index = self.next_episode_index
        episode_directory = self._episode_directory(episode_index)
        episode_directory.mkdir(parents=True, exist_ok=False)
        public_inputs: Mapping[str, Any] | None = None
        if self.before_episode_start is not None:
            public_inputs = self.before_episode_start(episode_index, episode_directory)
        service = self.service_factory(episode_index, episode_directory)
        self.current_service = service
        self.current_episode_index = episode_index
        self.state = "active"
        try:
            response = service.handle({"command": "start"})
        except BaseException:
            self._close_current_service()
            raise
        response = self._augment_response(response, episode_index)
        response["task_instruction"] = self.task_instructions[episode_index]
        if public_inputs is not None:
            response.update(_jsonable(public_inputs))
        return response

    def _finish_current_episode(
        self, response: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self.current_episode_index is None:
            raise RuntimeError("no active curriculum episode")
        episode_index = self.current_episode_index
        summary = {
            "episode_index": episode_index,
            "task_instruction": self.task_instructions[episode_index],
            "status": "finished",
            "success": bool(response["success"]),
            "accepted_agent_steps": int(response["accepted_agent_steps"]),
            "termination_reason": response.get("termination_reason"),
            "wall_time_seconds": response.get("wall_time_seconds"),
            "max_wall_time_seconds": response.get("max_wall_time_seconds"),
            "wall_time_budget_scope": response.get("wall_time_budget_scope"),
            "directory": self._episode_directory(episode_index).relative_to(
                self.private_run_directory
            ).as_posix(),
        }
        self.episode_results.append(summary)
        self.next_episode_index = episode_index + 1
        self._close_current_service()
        complete = self.next_episode_index >= self.episode_count
        self.state = "finished" if complete else "between_episodes"
        augmented = dict(response)
        augmented.update(
            {
                "episode_index": episode_index,
                "episode_count": self.episode_count,
                "next_episode_available": not complete,
                "curriculum_complete": complete,
            }
        )
        self._write_progress()
        if complete:
            self._write_root_result(status="finished")
        return augmented

    def _augment_response(
        self, response: Mapping[str, Any], episode_index: int
    ) -> dict[str, Any]:
        return {
            **dict(response),
            "episode_index": int(episode_index),
            "episode_count": self.episode_count,
        }

    def _close_current_service(self) -> None:
        if self.current_service is None:
            return
        episode_index = self.current_episode_index
        try:
            self.current_service.close()
        finally:
            self.current_service = None
            self.current_episode_index = None
            if episode_index is not None and self.after_episode_close is not None:
                self.after_episode_close(episode_index)

    def _episode_directory(self, episode_index: int) -> Path:
        return (
            self.private_run_directory
            / "episodes"
            / f"episode_{episode_index:03d}"
        )

    def _record_event(
        self,
        episode_index: int,
        request: object,
        response: Mapping[str, Any],
    ) -> None:
        event = {
            "schema_version": "libero.agent_curriculum_action_event.v1",
            "event_index": self._event_index,
            "episode_index": int(episode_index),
            "episode_count": self.episode_count,
            "recorded_at": _utc_now(),
            "request": _jsonable(request),
            "response": _jsonable(response),
        }
        self._event_index += 1
        path = self.private_run_directory / "actions.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_progress(self) -> None:
        _write_json_atomic(
            self.private_run_directory / "curriculum_progress.json",
            {
                "schema_version": "libero.agent_curriculum_progress.v1",
                "status": self.state,
                "episode_count": self.episode_count,
                "next_episode_index": self.next_episode_index,
                "episodes": deepcopy(self.episode_results),
                "updated_at": _utc_now(),
            },
        )

    def _write_root_result(self, *, status: str, reason: str | None = None) -> None:
        final_success = (
            bool(self.episode_results[-1]["success"])
            if len(self.episode_results) == self.episode_count
            else False
        )
        result = {
            "schema_version": "libero.agent_curriculum_result.v1",
            "status": status,
            "ok": status == "finished",
            "success": final_success,
            "final_episode_success": final_success,
            "all_episodes_success": (
                len(self.episode_results) == self.episode_count
                and all(item["success"] for item in self.episode_results)
            ),
            "episode_count": self.episode_count,
            "completed_episode_count": len(self.episode_results),
            "accepted_agent_steps": sum(
                int(item["accepted_agent_steps"])
                for item in self.episode_results
            ),
            "episodes": deepcopy(self.episode_results),
            "finished_at": _utc_now(),
        }
        if reason is not None:
            result["reason"] = reason
            if reason == WALL_TIME_BUDGET_EXHAUSTED:
                result["termination_reason"] = reason
        _write_json_atomic(self.private_run_directory / "result.json", result)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def request_is_reset(request: object) -> bool:
    return isinstance(request, Mapping) and request.get("command") == "reset"
