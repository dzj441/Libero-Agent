"""Factory for the official LIBERO suites and deterministic init states."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

import libero.libero as libero_package
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

from .evaluators.long_horizon import long_horizon_v2_private_evaluator
from .evaluators.manipulation import mujoco_adapted_v2_private_evaluator
from .evaluators.perception import perception_v2_private_evaluator
from .evaluators.vlabench.additional import vlabench_additional_private_evaluator
from .evaluators.vlabench.button import vlabench_button_private_evaluator
from .evaluators.vlabench.physical import vlabench_physical_private_evaluator
from .evaluators.vlabench.tube import vlabench_tube_precision_private_evaluator
from .runtime.contracts.release import (
    RELEASE_RESOURCE_DIRECTORY,
    validate_release_task,
)
from .runtime.control import MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS, OSCControlConfig
from .runtime.environment import LiberoAgentEnv
from .runtime.observation import TaskEntitySelection
from .runtime.profiles import ObservationProfile
from .runtime.references import load_task_reference_rgb


def make_libero_agent_env(
    *,
    suite: str = "libero_object",
    task_id: int = 0,
    init_state_id: int = 0,
    profile: ObservationProfile | int | str = ObservationProfile.LEVEL3,
    seed: int = 0,
    camera_height: int = 256,
    camera_width: int = 256,
    task_entities: TaskEntitySelection | None = None,
    control_config: OSCControlConfig | None = None,
    initial_settle_control_steps: int = 10,
    max_agent_steps: int | None = None,
    native_sequence_submission_limit: int | None = (
        MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS
    ),
    private_control_step_callback: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    render_gpu_device_id: int = -1,
    bddl_root: str | os.PathLike[str] | None = None,
    init_states_root: str | os.PathLike[str] | None = None,
    **env_kwargs: Any,
) -> LiberoAgentEnv:
    """Create an agent-safe environment for one official task and init state."""

    validate_release_task(suite, task_id)
    benchmark_class = get_benchmark(suite)
    task_suite = benchmark_class()
    if not 0 <= task_id < task_suite.get_num_tasks():
        raise ValueError(
            f"task_id must be in [0, {task_suite.get_num_tasks()}), got {task_id}"
        )
    task = task_suite.get_task(task_id)
    package_root = Path(libero_package.__file__).resolve().parent
    bddl_root = Path(bddl_root) if bddl_root is not None else package_root / "bddl_files"
    init_states_root = (
        Path(init_states_root)
        if init_states_root is not None
        else package_root / "init_files"
    )
    bddl_path = bddl_root / RELEASE_RESOURCE_DIRECTORY / task.bddl_file
    init_state_path = (
        init_states_root / RELEASE_RESOURCE_DIRECTORY / task.init_states_file
    )
    init_states = _load_trusted_init_states(os.fspath(init_state_path))
    if not 0 <= init_state_id < len(init_states):
        raise ValueError(
            f"init_state_id must be in [0, {len(init_states)}), got {init_state_id}"
        )

    reserved = {
        "bddl_file_name",
        "camera_names",
        "camera_heights",
        "camera_widths",
        "camera_depths",
        "camera_segmentations",
        "use_object_obs",
        "ignore_done",
        "initialization_noise",
        "render_gpu_device_id",
        "horizon",
    }
    conflicts = reserved.intersection(env_kwargs)
    if conflicts:
        raise ValueError(f"factory-managed env kwargs cannot be overridden: {sorted(conflicts)}")

    env = OffScreenRenderEnv(
        bddl_file_name=os.fspath(bddl_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_depths=True,
        camera_segmentations="instance",
        use_object_obs=False,
        ignore_done=True,
        initialization_noise=None,
        render_gpu_device_id=render_gpu_device_id,
        horizon=10000,
        **env_kwargs,
    )
    env.seed(seed)
    instruction = " ".join(task.language.split())
    private_episode_evaluator = perception_v2_private_evaluator(
        env,
        suite=suite,
        task_id=task_id,
    )
    if private_episode_evaluator is None:
        private_episode_evaluator = long_horizon_v2_private_evaluator(
            env,
            suite=suite,
            task_id=task_id,
        )
    if private_episode_evaluator is None:
        private_episode_evaluator = mujoco_adapted_v2_private_evaluator(
            env,
            suite=suite,
            task_id=task_id,
        )
    if private_episode_evaluator is None:
        private_episode_evaluator = vlabench_tube_precision_private_evaluator(
            env,
            suite=suite,
            task_id=task_id,
        )
    if private_episode_evaluator is None:
        private_episode_evaluator = vlabench_physical_private_evaluator(
            env,
            suite=suite,
            task_id=task_id,
        )
    if private_episode_evaluator is None:
        private_episode_evaluator = vlabench_additional_private_evaluator(
            env,
            suite=suite,
            task_id=task_id,
        )
    if private_episode_evaluator is None:
        private_episode_evaluator = vlabench_button_private_evaluator(
            env,
            suite=suite,
            task_id=task_id,
        )
    return LiberoAgentEnv(
        env,
        profile=profile,
        camera_height=camera_height,
        camera_width=camera_width,
        task_instruction=instruction,
        task_reference_rgb=load_task_reference_rgb(suite, task_id),
        initial_state=np.asarray(init_states[init_state_id]),
        task_entities=task_entities,
        control_config=control_config,
        # Keep the caller's settle budget unchanged for the launcher-ready
        # contract.  Physical trusted states are restored after spending that
        # budget on the ordinary reset, so articulated payloads are not
        # displaced before the private checker captures its baseline.
        initial_settle_control_steps=initial_settle_control_steps,
        settle_before_initial_state=(suite == "vlabench_physical"),
        max_agent_steps=max_agent_steps,
        native_sequence_submission_limit=native_sequence_submission_limit,
        private_control_step_callback=private_control_step_callback,
        private_episode_evaluator=private_episode_evaluator,
    )


def _load_trusted_init_states(path: str) -> Any:
    """Load the official local init-state tensor across PyTorch versions."""

    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)
