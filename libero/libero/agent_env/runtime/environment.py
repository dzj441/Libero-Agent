"""Stateful AgentEnv lifecycle around a single live LIBERO simulation."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .control import (
    MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS,
    BaseFrameOSCExecutor,
    EEFCommand,
    NativeOSCSequenceExecutor,
    OSCControlConfig,
)
from .observation import MasterObservationCollector, TaskEntitySelection
from .profiles import ObservationProfile, project_public_observation


class PrivateEpisodeEvaluator(Protocol):
    """Host-private ordered-task monitor; its details are never projected."""

    def reset(self) -> None: ...

    def observe(self, raw_observation: Mapping[str, Any]) -> None: ...

    def result(self) -> dict[str, Any]: ...


class LiberoAgentEnv:
    """Expose one stateful Agent action at a time without private task state.

    Reward and task checker outputs are intentionally withheld until
    ``finish_episode``. Every accepted metric target or native OSC sequence
    returns the actual post-execution observation from the same trajectory.
    """

    def __init__(
        self,
        env: Any,
        profile: ObservationProfile | int | str,
        camera_height: int,
        camera_width: int,
        *,
        task_instruction: str,
        initial_state: np.ndarray | None = None,
        task_entities: TaskEntitySelection | None = None,
        task_reference_rgb: np.ndarray | None = None,
        control_config: OSCControlConfig | None = None,
        initial_settle_control_steps: int = 10,
        max_agent_steps: int | None = None,
        native_sequence_submission_limit: int | None = (
            MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS
        ),
        private_control_step_callback: (
            Callable[[Mapping[str, Any]], None] | None
        ) = None,
        private_episode_evaluator: PrivateEpisodeEvaluator | None = None,
        settle_before_initial_state: bool = False,
    ) -> None:
        if initial_settle_control_steps < 0:
            raise ValueError("initial_settle_control_steps must be non-negative")
        if max_agent_steps is not None and max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be positive when provided")
        if (
            native_sequence_submission_limit is not None
            and native_sequence_submission_limit <= 0
        ):
            raise ValueError(
                "native_sequence_submission_limit must be positive when provided"
            )
        self.env = env
        self.profile = ObservationProfile.parse(profile)
        self.task_instruction = str(task_instruction)
        self.initial_state = (
            None if initial_state is None else np.asarray(initial_state).copy()
        )
        self.initial_settle_control_steps = int(initial_settle_control_steps)
        self.max_agent_steps = max_agent_steps
        self.native_sequence_submission_limit = native_sequence_submission_limit
        # Some trusted physical init states deliberately contain an object on
        # an articulated surface.  The requested settle budget still belongs
        # in the launcher/server contract, but applying it *after* restoring
        # such a state can destroy the very reset pose the checker must see.
        # Task factories may therefore spend the budget on the ordinary reset
        # before restoring the trusted state.
        self.settle_before_initial_state = bool(settle_before_initial_state)
        self.collector = MasterObservationCollector(
            env,
            camera_height=camera_height,
            camera_width=camera_width,
            task_entities=task_entities,
            task_reference_rgb=task_reference_rgb,
        )
        self.private_control_step_callback = private_control_step_callback
        self.private_episode_evaluator = private_episode_evaluator
        self._private_physics_substep_callback_registered = False
        physics_substep_observer = getattr(
            private_episode_evaluator,
            "observe_physics_substep",
            None,
        )
        physics_substep_setter = getattr(
            env,
            "set_agent_physics_substep_callback",
            None,
        )
        if callable(physics_substep_observer):
            if not callable(physics_substep_setter):
                raise RuntimeError(
                    "private evaluator requires physics-substep observation, "
                    "but the environment does not expose the callback hook"
                )
            physics_substep_setter(physics_substep_observer)
            self._private_physics_substep_callback_registered = True
        self.executor = BaseFrameOSCExecutor(
            env,
            control_config,
            control_step_callback=self._on_agent_control_step,
        )
        self.native_sequence_executor = NativeOSCSequenceExecutor(
            env,
            control_step_callback=self._on_agent_control_step,
        )
        self._started = False
        self._finished = False
        self._agent_step_index = 0
        self._latest_raw_observation: dict[str, Any] | None = None

    def start_episode(self) -> dict[str, Any]:
        if self._started and not self._finished:
            raise RuntimeError("episode is already active")
        raw_observation = self.env.reset()

        hold_action = np.zeros(7, dtype=np.float64)
        if self.settle_before_initial_state and self.initial_state is not None:
            # This is an internal reset normalization, not an agent-visible
            # trajectory.  Do not publish these transient observations to the
            # recorder; the first public frame comes from the trusted state.
            for _ in range(self.initial_settle_control_steps):
                raw_observation, _reward, _done, _info = self.env.step(hold_action)
            raw_observation = self.env.set_init_state(self.initial_state)
        else:
            if self.initial_state is not None:
                raw_observation = self.env.set_init_state(self.initial_state)
            for _ in range(self.initial_settle_control_steps):
                raw_observation, _reward, _done, _info = self.env.step(hold_action)
                if self.private_control_step_callback is not None:
                    self.private_control_step_callback(raw_observation)

        if self.private_episode_evaluator is not None:
            self.private_episode_evaluator.reset()

        self._started = True
        self._finished = False
        self._agent_step_index = 0
        self._latest_raw_observation = raw_observation
        return {
            "task_instruction": self.task_instruction,
            "observation": self._public_observation(frame_index=0),
        }

    def step_osc_target(
        self,
        delta_position_m: Sequence[float] = (0.0, 0.0, 0.0),
        delta_rotation_rotvec_rad: Sequence[float] = (0.0, 0.0, 0.0),
        delta_gripper_width_m: float = 0.0,
    ) -> dict[str, Any]:
        self._require_active()
        self._require_agent_step_budget(self.max_agent_steps)
        command = EEFCommand.create(
            delta_position_m=delta_position_m,
            delta_rotation_rotvec_rad=delta_rotation_rotvec_rad,
            delta_gripper_width_m=delta_gripper_width_m,
        )
        raw_observation, execution = self.executor.execute(command)
        self._latest_raw_observation = raw_observation
        self._agent_step_index += 1
        return {
            "accepted_agent_step": self._agent_step_index,
            "execution": execution.to_public_dict(),
            "observation": self._public_observation(
                frame_index=self._agent_step_index
            ),
        }

    def step_osc_sequence(
        self,
        actions: Sequence[Sequence[float]],
    ) -> dict[str, Any]:
        """Execute a bounded normalized OSC sequence as one Agent submission."""

        self._require_active()
        configured_limits = (
            self.max_agent_steps,
            self.native_sequence_submission_limit,
        )
        finite_limits = [limit for limit in configured_limits if limit is not None]
        sequence_limit = min(finite_limits) if finite_limits else None
        self._require_agent_step_budget(sequence_limit)
        raw_observation, execution = self.native_sequence_executor.execute(actions)
        self._latest_raw_observation = raw_observation
        self._agent_step_index += 1
        return {
            "accepted_agent_step": self._agent_step_index,
            "execution": execution.to_public_dict(),
            "observation": self._public_observation(
                frame_index=self._agent_step_index
            ),
        }

    def finish_episode(self) -> dict[str, Any]:
        self._require_active()
        private_evaluation = None
        if self.private_episode_evaluator is None:
            bddl_final_goal_success = bool(self.env.check_success())
            success = bddl_final_goal_success
        else:
            private_evaluation = self.private_episode_evaluator.result()
            bddl_diagnostic_error = None
            try:
                bddl_final_goal_success = bool(self.env.check_success())
            except Exception as exc:  # diagnostic must not replace authority
                bddl_final_goal_success = None
                bddl_diagnostic_error = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            private_evaluation["bddl_final_goal_success"] = (
                bddl_final_goal_success
            )
            if bddl_diagnostic_error is not None:
                private_evaluation["bddl_final_goal_diagnostic_error"] = (
                    bddl_diagnostic_error
                )
            success = bool(private_evaluation["success"])
        self._finished = True
        result = {
            "success": success,
            "accepted_agent_steps": self._agent_step_index,
        }
        if private_evaluation is not None:
            result["private_evaluation"] = private_evaluation
        return result

    def close(self) -> None:
        if self._private_physics_substep_callback_registered:
            setter = getattr(
                self.env,
                "set_agent_physics_substep_callback",
                None,
            )
            if callable(setter):
                setter(None)
        self.env.close()

    def private_evaluation_snapshot(self) -> dict[str, Any] | None:
        """Return host-private partial progress for aborted-run audit."""

        if self.private_episode_evaluator is None:
            return None
        return self.private_episode_evaluator.result()

    def _public_observation(self, frame_index: int) -> dict[str, Any]:
        if self._latest_raw_observation is None:
            raise RuntimeError("no observation is available")
        master = self.collector.collect(self._latest_raw_observation, frame_index)
        return project_public_observation(master, self.profile)

    def _require_active(self) -> None:
        if not self._started:
            raise RuntimeError("start_episode must be called first")
        if self._finished:
            raise RuntimeError("episode has already finished")

    def _require_agent_step_budget(self, limit: int | None) -> None:
        if limit is not None and self._agent_step_index >= limit:
            raise RuntimeError(f"agent step limit reached ({limit})")

    def _on_agent_control_step(
        self, raw_observation: Mapping[str, Any]
    ) -> None:
        if self.private_control_step_callback is not None:
            self.private_control_step_callback(raw_observation)
        if self.private_episode_evaluator is not None:
            self.private_episode_evaluator.observe(raw_observation)
