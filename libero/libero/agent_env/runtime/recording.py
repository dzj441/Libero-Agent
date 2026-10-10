"""Evaluator-private continuous RGB recording for one LIBERO episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
import robosuite.macros as robosuite_macros


class PrivateRolloutVideoRecorder:
    """Record every LIBERO policy interval as head/wrist side-by-side RGB."""

    def __init__(self, output_path: str | Path, fps: float = 20.0) -> None:
        self.output_path = Path(output_path)
        self.fps = float(fps)
        self._writer: Any | None = None
        self.frame_count = 0

    def append_raw_observation(self, observation: Mapping[str, Any]) -> None:
        head = _opencv_rows(np.asarray(observation["agentview_image"], dtype=np.uint8))
        wrist = _opencv_rows(
            np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8)
        )
        if head.shape != wrist.shape or head.ndim != 3 or head.shape[2] != 3:
            raise ValueError("private recorder requires equal-size head/wrist RGB frames")
        frame = np.concatenate((head, wrist), axis=1)
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(
                self.output_path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        self._writer.append_data(frame)
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _opencv_rows(image: np.ndarray) -> np.ndarray:
    if robosuite_macros.IMAGE_CONVENTION == "opengl":
        image = image[::-1]
    elif robosuite_macros.IMAGE_CONVENTION != "opencv":
        raise ValueError(
            "unsupported robosuite image convention for private recording: "
            f"{robosuite_macros.IMAGE_CONVENTION!r}"
        )
    return np.ascontiguousarray(image)
