"""Repository-relative benchmark paths with optional explicit configuration."""
import os
from pathlib import Path

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parent
libero_config_path = os.environ.get("LIBERO_CONFIG_PATH", str(_PACKAGE_ROOT.parent))
config_file = os.path.join(libero_config_path, "config.yaml")


def get_default_path_dict(custom_location=None):
    root = (
        _PACKAGE_ROOT
        if custom_location is None
        else Path(custom_location).expanduser().resolve()
    )
    # Keep the legacy custom-location dataset layout for explicit callers.
    datasets = (
        _PACKAGE_ROOT.parents[2] / "dataset/libero_hdf5"
        if custom_location is None
        else root.parent / "datasets"
    )
    if os.environ.get("LIBERO_DATASET_ROOT"):
        datasets = Path(os.environ["LIBERO_DATASET_ROOT"]).expanduser() / "libero_hdf5"
    return {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(datasets),
        "assets": str(root / "assets"),
    }


def get_libero_path(query_key):
    config = get_default_path_dict()
    if os.path.isfile(config_file):
        with open(config_file) as stream:
            config.update(yaml.safe_load(stream) or {})
    elif "LIBERO_CONFIG_PATH" in os.environ:
        raise FileNotFoundError(config_file)
    assert query_key in config, f"Unknown LIBERO path: {query_key}. Available keys: {config.keys()}"
    return config[query_key]


def set_libero_default_path(custom_location=None):
    """Persist an explicitly requested local configuration; imports never write it."""
    os.makedirs(libero_config_path, exist_ok=True)
    with open(config_file, "w") as stream:
        yaml.safe_dump(get_default_path_dict(custom_location), stream)
