"""Utilities for physically replaying LIBERO HDF5 demonstrations.

The converted LIBERO datasets contain both an initial flattened MuJoCo state
and the normalized OSC action sequence.  A valid replay restores the initial
state once and then lets the simulator execute the actions; saved states are
never forced between actions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import h5py
import numpy as np


REPLAY_SCHEMA_VERSION = "libero.demonstration_replay.v1"
DEFAULT_CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")


@dataclass
class DemonstrationEpisode:
    """The simulator inputs and public metadata for one HDF5 episode."""

    dataset_path: Path
    demo_key: str
    bddl_file: Path
    actions: np.ndarray
    init_state: np.ndarray
    task_instruction: str
    problem_name: str
    env_name: str
    robots: tuple[str, ...]
    controller: str
    control_freq: int
    init_state_source: str


def normalize_demo_key(episode: str | int) -> str:
    """Accept either ``0`` or ``demo_0`` and return an HDF5 group key."""

    value = str(episode)
    if value.startswith("demo_"):
        suffix = value[5:]
    else:
        suffix = value
    if not suffix.isdigit():
        raise ValueError(
            f"Invalid episode {episode!r}; expected an integer or a key like demo_0"
        )
    return f"demo_{int(suffix)}"


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _json_attr(attrs: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = _decode_attr(attrs.get(key, "{}"))
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"HDF5 attribute {key!r} is not valid JSON") from exc
    elif isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        raise ValueError(f"HDF5 attribute {key!r} must contain a JSON object")
    if not isinstance(parsed, dict):
        raise ValueError(f"HDF5 attribute {key!r} must contain a JSON object")
    return parsed


def default_bddl_root() -> Path:
    """Return the BDDL directory belonging to this LIBERO checkout."""

    return Path(__file__).resolve().parents[1] / "bddl_files"


def resolve_bddl_file(
    recorded_path: str | Path,
    *,
    override: Optional[str | Path] = None,
    bddl_root: Optional[str | Path] = None,
) -> Path:
    """Resolve a recorded (often stale) BDDL path into this checkout.

    Official datasets may store paths from the machine that generated them.
    The path below ``bddl_files/`` is stable and is therefore used to relocate
    the task definition into the current repository.
    """

    if override is not None:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"BDDL override does not exist: {candidate}")
        return candidate

    recorded = Path(str(_decode_attr(recorded_path))).expanduser()
    if recorded.is_file():
        return recorded.resolve()

    root = Path(bddl_root).expanduser().resolve() if bddl_root else default_bddl_root()
    parts = recorded.parts
    if "bddl_files" in parts:
        suffix = Path(*parts[parts.index("bddl_files") + 1 :])
        candidate = root / suffix
        if candidate.is_file():
            return candidate.resolve()

    # A basename fallback makes hand-moved datasets usable while refusing an
    # ambiguous match across suites.
    matches = sorted(root.rglob(recorded.name)) if recorded.name else []
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Recorded BDDL path {recorded_path!r} is stale and its basename is "
            f"ambiguous under {root}: {len(matches)} matches"
        )
    raise FileNotFoundError(
        f"Could not resolve recorded BDDL path {recorded_path!r} under {root}"
    )


def load_demonstration_episode(
    dataset_path: str | Path,
    episode: str | int = "demo_0",
    *,
    bddl_file: Optional[str | Path] = None,
    bddl_root: Optional[str | Path] = None,
) -> DemonstrationEpisode:
    """Load and validate the simulator inputs for one converted LIBERO demo."""

    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Demonstration HDF5 does not exist: {path}")
    demo_key = normalize_demo_key(episode)

    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"{path} has no HDF5 group named 'data'")
        data = handle["data"]
        if demo_key not in data:
            available = sorted(data.keys())
            preview = ", ".join(available[:8])
            raise KeyError(
                f"Episode {demo_key!r} is not present in {path}; available: {preview}"
            )
        demo = data[demo_key]
        if "actions" not in demo:
            raise ValueError(f"{path}:{demo_key} has no actions dataset")

        actions = np.asarray(demo["actions"], dtype=np.float64)
        if "init_state" in demo.attrs:
            init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
            init_state_source = "episode_attribute"
        elif "states" in demo and len(demo["states"]):
            init_state = np.asarray(demo["states"][0], dtype=np.float64)
            init_state_source = "first_recorded_state"
        else:
            raise ValueError(
                f"{path}:{demo_key} has neither an init_state attribute nor states"
            )

        if "bddl_file_name" not in data.attrs and bddl_file is None:
            raise ValueError(
                f"{path} has no data.attrs['bddl_file_name']; pass --bddl-file"
            )
        recorded_bddl = data.attrs.get("bddl_file_name", "")
        resolved_bddl = resolve_bddl_file(
            recorded_bddl, override=bddl_file, bddl_root=bddl_root
        )

        problem_info = _json_attr(data.attrs, "problem_info")
        env_args = _json_attr(data.attrs, "env_args")
        env_kwargs = env_args.get("env_kwargs", {})
        controller_config = env_kwargs.get("controller_configs", {})
        robots_raw = env_kwargs.get("robots", ["Panda"])
        if isinstance(robots_raw, str):
            robots_raw = [robots_raw]

        task_instruction = str(problem_info.get("language_instruction", ""))
        problem_name = str(
            problem_info.get("problem_name", env_args.get("problem_name", ""))
        )
        env_name = str(_decode_attr(data.attrs.get("env_name", "")))
        controller = str(controller_config.get("type", "OSC_POSE"))
        control_freq = int(env_kwargs.get("control_freq", 20))

    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
        raise ValueError(
            f"{path}:{demo_key} actions must have non-empty shape (T, 7), got "
            f"{actions.shape}"
        )
    if init_state.ndim != 1 or init_state.size == 0:
        raise ValueError(
            f"{path}:{demo_key} init_state must be a non-empty vector, got "
            f"{init_state.shape}"
        )
    if not np.isfinite(actions).all() or not np.isfinite(init_state).all():
        raise ValueError(f"{path}:{demo_key} contains NaN or infinite simulator input")

    return DemonstrationEpisode(
        dataset_path=path,
        demo_key=demo_key,
        bddl_file=resolved_bddl,
        actions=actions,
        init_state=init_state,
        task_instruction=task_instruction,
        problem_name=problem_name,
        env_name=env_name,
        robots=tuple(str(robot) for robot in robots_raw),
        controller=controller,
        control_freq=control_freq,
        init_state_source=init_state_source,
    )


def ending_true_streak(values: Sequence[bool]) -> int:
    streak = 0
    for value in reversed(values):
        if not value:
            break
        streak += 1
    return streak


def maximum_true_streak(values: Sequence[bool]) -> int:
    maximum = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _standard_rgb(image: np.ndarray) -> np.ndarray:
    """Convert robosuite's OpenGL-origin RGB image to a normal top-left image."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got {array.shape}")
    return np.ascontiguousarray(array[::-1]).astype(np.uint8, copy=False)


def _compose_camera_frame(obs: Mapping[str, Any]) -> np.ndarray:
    head = _standard_rgb(np.asarray(obs["agentview_image"]))
    wrist = _standard_rgb(np.asarray(obs["robot0_eye_in_hand_image"]))
    if head.shape[0] != wrist.shape[0]:
        raise ValueError(
            f"Camera heights differ: agentview={head.shape}, wrist={wrist.shape}"
        )
    return np.concatenate((head, wrist), axis=1)


def run_action_replay(
    episode: DemonstrationEpisode,
    *,
    seed: int = 0,
    settle_steps: int = 10,
    stable_success_steps: int = 10,
    video_path: Optional[str | Path] = None,
    camera_height: int = 256,
    camera_width: int = 256,
    video_stride: int = 1,
    video_fps: Optional[float] = None,
    render_gpu_device_id: int = -1,
    observation_callback: Optional[
        Callable[[Any, Mapping[str, Any], int, Optional[int]], None]
    ] = None,
) -> dict[str, Any]:
    """Execute a demonstration through LIBERO and return a verification report.

    Only the initial state is restored. Every subsequent state is produced by
    ``env.step(action)``. The environment is deliberately fresh for each call so
    OSC controller state cannot leak between episodes.
    """

    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")
    if stable_success_steps < 1:
        raise ValueError("stable_success_steps must be at least 1")
    if video_stride < 1:
        raise ValueError("video_stride must be at least 1")
    if camera_height < 1 or camera_width < 1:
        raise ValueError("camera dimensions must be positive")
    if render_gpu_device_id < -1:
        raise ValueError("render_gpu_device_id must be -1 or a non-negative index")

    # Importing robosuite initializes its GL backend. Respect an evaluator's
    # explicit backend, otherwise choose the headless default before the lazy
    # import. Keeping this import here also lets schema utilities run without
    # MuJoCo installed.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    from libero.libero.envs.env_wrapper import ControlEnv

    render = video_path is not None
    capture_observations = observation_callback is not None
    use_camera_observations = render or capture_observations
    env = ControlEnv(
        bddl_file_name=str(episode.bddl_file),
        robots=list(episode.robots),
        controller=episode.controller,
        has_renderer=False,
        has_offscreen_renderer=use_camera_observations,
        use_camera_obs=use_camera_observations,
        camera_names=list(DEFAULT_CAMERA_NAMES),
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_depths=capture_observations,
        camera_segmentations="instance" if capture_observations else None,
        use_object_obs=False,
        render_gpu_device_id=render_gpu_device_id,
        ignore_done=True,
        control_freq=episode.control_freq,
        horizon=settle_steps + len(episode.actions) + 1,
    )

    frames: list[np.ndarray] = []
    success_trace: list[bool] = []
    gl_info: dict[str, Optional[str]] = {
        "vendor": None,
        "renderer": None,
        "version": None,
    }
    try:
        # LIBERO placement sampling uses NumPy's global RNG. Seed immediately
        # before reset, then overwrite the sampled state with the recorded one.
        np.random.seed(seed)
        env.reset()
        obs = env.set_init_state(episode.init_state)
        if render:
            from OpenGL import GL

            def decode_gl_string(name: int) -> Optional[str]:
                value = GL.glGetString(name)
                return value.decode("utf-8") if isinstance(value, bytes) else value

            gl_info = {
                "vendor": decode_gl_string(GL.GL_VENDOR),
                "renderer": decode_gl_string(GL.GL_RENDERER),
                "version": decode_gl_string(GL.GL_VERSION),
            }
        settle_action = np.zeros(7, dtype=np.float64)
        settle_action[-1] = episode.actions[0, -1]
        for _ in range(settle_steps):
            obs, _, _, _ = env.step(settle_action)

        # The expert bundle's initial observation is the stable state the
        # policy actually acts from, not the transient returned immediately by
        # set_init_state().  Every later frame is captured after its causal
        # source action has been executed by the same environment.
        if render:
            frames.append(_compose_camera_frame(obs))
        if observation_callback is not None:
            observation_callback(env, obs, 0, None)

        first_success_step: Optional[int] = None
        for step_index, action in enumerate(episode.actions):
            obs, _, _, _ = env.step(action)
            success = bool(env.check_success())
            success_trace.append(success)
            if success and first_success_step is None:
                first_success_step = step_index
            if render and step_index % video_stride == 0:
                frames.append(_compose_camera_frame(obs))
            if observation_callback is not None:
                observation_callback(env, obs, step_index + 1, step_index)
    finally:
        env.close()

    final_success_streak = ending_true_streak(success_trace)
    maximum_success_streak = maximum_true_streak(success_trace)
    final_success = bool(success_trace[-1])
    verified_success = final_success and final_success_streak >= stable_success_steps

    resolved_video: Optional[Path] = None
    if video_path is not None:
        if not frames:
            raise RuntimeError("Video was requested but replay produced no frames")
        import imageio.v2 as imageio

        resolved_video = Path(video_path).expanduser().resolve()
        resolved_video.parent.mkdir(parents=True, exist_ok=True)
        fps = video_fps or (episode.control_freq / video_stride)
        imageio.mimsave(resolved_video, frames, fps=fps, quality=7)

    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "verified_success": verified_success,
        "final_success": final_success,
        "first_success_step": first_success_step,
        "final_success_streak": final_success_streak,
        "maximum_success_streak": maximum_success_streak,
        "required_stable_success_steps": stable_success_steps,
        "episode": {
            "dataset": str(episode.dataset_path),
            "demo_key": episode.demo_key,
            "task_instruction": episode.task_instruction,
            "problem_name": episode.problem_name,
            "env_name": episode.env_name,
            "bddl_file": str(episode.bddl_file),
            "action_count": int(len(episode.actions)),
            "action_shape": list(episode.actions.shape),
            "init_state_size": int(episode.init_state.size),
            "init_state_source": episode.init_state_source,
        },
        "replay": {
            "mode": "actions",
            "state_forcing_after_reset": False,
            "seed": seed,
            "settle_steps": settle_steps,
            "control_frequency_hz": episode.control_freq,
            "controller": episode.controller,
            "robots": list(episode.robots),
            "render_backend": os.environ.get("MUJOCO_GL") if render else "disabled",
            "render_gpu_device_id": render_gpu_device_id if render else None,
            "gl": gl_info if render else None,
        },
        "artifacts": {
            "video": str(resolved_video) if resolved_video is not None else None,
            "video_layout": "agentview_rgb | robot0_eye_in_hand_rgb"
            if resolved_video is not None
            else None,
        },
    }


def write_replay_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
