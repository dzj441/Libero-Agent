"""Load the frozen RoboMemArena compatibility core before the agent stack.

This module deliberately depends only on the standard library. Entry points
load it directly from its file path before importing ``libero`` because the
compatibility core must become the first provider of that package namespace.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import sys
import tempfile


def activate_robomemarena_core(
    *,
    source_root: str | Path,
    checkout_root: str | Path | None = None,
) -> Path:
    """Use RoboMemArena's simulator assets with this repository's agent_env.

    By default a lightweight merged package is assembled on the system temp
    disk. Unchanged files are linked from this installation when the platform
    permits it, with a real-file copy fallback; the frozen RoboMemArena
    overrides vendored under ``agent_env`` are copied. An external checkout
    remains an explicit development override, but is no longer a runtime
    dependency.

    The server is a fresh process, so it can load the merged fork first and
    then extend the package search path with this repository's ``agent_env``
    modules. The Codex process never imports the fork or receives private raw
    simulator state.
    """

    source = Path(source_root).resolve()
    if checkout_root is None:
        fork_root = _build_vendored_fork(source)
    else:
        checkout = Path(checkout_root).expanduser().resolve()
        fork_root = checkout / "evaluation_benchmark" / "libero_fork"
    if not (fork_root / "libero" / "envs").is_dir():
        raise FileNotFoundError(
            f"RoboMemArena LIBERO fork is unavailable under {fork_root}"
        )
    if "libero" in sys.modules:
        raise RuntimeError(
            "RoboMemArena core must be activated before importing libero"
        )

    sys.path.insert(0, str(fork_root))
    import libero  # pylint: disable=import-outside-toplevel
    import libero.libero as core  # pylint: disable=import-outside-toplevel

    # The upstream checkout contains a compatibility package at
    # libero/libero.  Its benchmark and environment modules expect these path
    # helpers on that compatibility package, while the implementations live on
    # the outer package.
    core.get_libero_path = libero.get_libero_path
    core.get_default_path_dict = libero.get_default_path_dict

    local_outer = str(source / "libero")
    local_inner = str(source / "libero" / "libero")
    if local_outer not in libero.__path__:
        libero.__path__.append(local_outer)
    if local_inner not in core.__path__:
        core.__path__.append(local_inner)
    return fork_root


def _build_vendored_fork(source_root: Path) -> Path:
    base_package = source_root / "libero" / "libero"
    override_package = (
        base_package
        / "agent_env"
        / "integrations"
        / "robomemarena"
        / "vendor"
        / "core"
    )
    if not override_package.is_dir():
        raise FileNotFoundError(
            f"vendored RoboMemArena compatibility core is missing: "
            f"{override_package}"
        )

    fork_root = Path(tempfile.mkdtemp(prefix="libero-robomemarena-core-"))
    atexit.register(shutil.rmtree, fork_root, ignore_errors=True)
    merged_package = fork_root / "libero"
    _materialize_tree(base_package, merged_package)
    _overlay_tree(override_package, merged_package)
    # ``vendor/core/libero/__init__.py`` extends the compatibility package's
    # search path to this merged root. The legacy ``libero.libero.*`` imports
    # therefore resolve against the same canonical assets / benchmark / envs /
    # utils directories without repository symlinks or duplicate source trees.
    return fork_root


def _materialize_tree(source: Path, destination: Path) -> None:
    """Build a temporary package without requiring symlink support."""

    for directory, directory_names, file_names in os.walk(source):
        directory_names[:] = [
            name for name in directory_names if name != "__pycache__"
        ]
        source_directory = Path(directory)
        relative = source_directory.relative_to(source)
        destination_directory = destination / relative
        destination_directory.mkdir(parents=True, exist_ok=True)
        for filename in file_names:
            if filename.endswith((".pyc", ".pyo")):
                continue
            source_file = source_directory / filename
            destination_file = destination_directory / filename
            _link_or_copy_file(source_file.resolve(), destination_file)


def _link_or_copy_file(source: Path, destination: Path) -> None:
    try:
        destination.symlink_to(source)
    except (NotImplementedError, OSError):
        shutil.copy2(source, destination)


def _overlay_tree(source: Path, destination: Path) -> None:
    for directory, directory_names, file_names in os.walk(
        source, followlinks=False
    ):
        source_directory = Path(directory)
        relative = source_directory.relative_to(source)
        destination_directory = destination / relative
        destination_directory.mkdir(parents=True, exist_ok=True)
        traversed_directories: list[str] = []
        for name in directory_names:
            source_child = source_directory / name
            if name == "__pycache__":
                continue
            if source_child.is_symlink():
                destination_child = destination_directory / name
                if (
                    destination_child.exists()
                    or destination_child.is_symlink()
                ):
                    if destination_child.is_dir() and not destination_child.is_symlink():
                        shutil.rmtree(destination_child)
                    else:
                        destination_child.unlink()
                destination_child.symlink_to(os.readlink(source_child))
            else:
                traversed_directories.append(name)
        directory_names[:] = traversed_directories
        for filename in file_names:
            if filename.endswith((".pyc", ".pyo")):
                continue
            source_file = source_directory / filename
            destination_file = destination_directory / filename
            if destination_file.exists() or destination_file.is_symlink():
                destination_file.unlink()
            if source_file.is_symlink():
                destination_file.symlink_to(os.readlink(source_file))
            else:
                shutil.copy2(source_file, destination_file)
