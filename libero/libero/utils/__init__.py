"""Compatibility exports for the shared LIBERO path configuration."""
from libero.libero import (
    config_file,
    get_default_path_dict,
    get_libero_path as _get_libero_path,
    libero_config_path,
    set_libero_default_path as set_libero_path,
)


def get_path_dict(root_location=None):
    return get_default_path_dict(root_location)


def get_libero_path(key):
    return _get_libero_path(key)
