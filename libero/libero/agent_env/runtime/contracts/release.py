"""The public 30-task catalog and packaged-resource boundary."""

import json
from functools import lru_cache
from pathlib import Path


RELEASE_RESOURCE_DIRECTORY = "libero_agent"


@lru_cache(maxsize=1)
def release_manifest():
    path = Path(__file__).resolve().parents[2] / "release" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def release_task(suite, task_id):
    """Return the frozen catalog entry for a source-suite task identity."""

    identity = (str(suite), int(task_id))
    for task in release_manifest()["tasks"]:
        if (task["suite"], int(task["task_id"])) == identity:
            return task
    raise ValueError(f"Task {suite}:{task_id} is outside this 30-task release")


def validate_release_task(suite, task_id):
    release_task(suite, task_id)


__all__ = [
    "RELEASE_RESOURCE_DIRECTORY",
    "release_manifest",
    "release_task",
    "validate_release_task",
]
