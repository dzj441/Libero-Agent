"""Relocate known MJCF asset roots without changing any model parameters."""
from pathlib import Path
import re
from typing import Mapping


def relocate_model_xml_asset_paths(xml: str, roots: Mapping[str, Path]) -> str:
    """Rewrite only file attributes with known, existing local asset suffixes.

    Unknown paths and missing resources remain unchanged (fail closed when
    comparing models). This checks relocation equivalence, not historical file
    contents; copied assets must be verified separately.
    """
    def replace(match):
        recorded = match.group(1)
        for marker, root in roots.items():
            if marker not in recorded:
                continue
            suffix = Path(recorded.split(marker, 1)[1])
            if suffix.is_absolute() or '..' in suffix.parts:
                return match.group(0)
            root = root.resolve()
            target = (root / suffix).resolve()
            if target.is_relative_to(root) and target.is_file():
                return f'file="{target}"'
        return match.group(0)

    return re.sub(r'\bfile="([^"]+)"', replace, xml)
