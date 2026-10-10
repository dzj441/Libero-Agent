"""Compatibility namespace for the frozen RoboMemArena core.

RoboMemArena's fork imports modules through ``libero.libero`` even though the
vendored overrides have a single canonical home one directory above this
shim. Extend the package search path instead of maintaining duplicate trees
or repository symlinks.
"""

from pathlib import Path


_CORE_ROOT = str(Path(__file__).resolve().parent.parent)
if _CORE_ROOT not in __path__:
    __path__.append(_CORE_ROOT)
