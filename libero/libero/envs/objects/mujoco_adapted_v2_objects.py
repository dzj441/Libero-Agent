"""Small MuJoCo object adapters used by the local V2 benchmark suites."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from robosuite.models.objects import DoorObject, MujocoXMLObject

from libero.libero.envs.base_object import register_object


_ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets" / "external"
_UNSET = object()


class _LocalAdaptedXMLObject(MujocoXMLObject):
    """Load a repository-owned asset through LIBERO's normal object path."""

    relative_xml: str
    default_rotation = (0.0, 0.0)
    default_rotation_axis = "z"

    def __init__(
        self,
        name: str,
        joints: Any = _UNSET,
        *,
        duplicate_collision_geoms: bool = False,
    ) -> None:
        xml_path = (_ASSET_ROOT / self.relative_xml).resolve()
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        joint_specs = (
            [dict(type="free", damping="0.0005")]
            if joints is _UNSET
            else copy.deepcopy(joints)
        )
        super().__init__(
            str(xml_path),
            name=name,
            joints=joint_specs,
            obj_type="all",
            duplicate_collision_geoms=duplicate_collision_geoms,
        )
        self.category_name = type(self).__name__.lower()
        self.rotation = self.default_rotation
        self.rotation_axis = self.default_rotation_axis
        self.object_properties = {"vis_site_names": {}}


@register_object
class RobosuiteLatchedDoor(DoorObject):
    """The stock robosuite 1.4 locked door exposed as a LIBERO fixture."""

    def __init__(
        self,
        name: str = "latched_door",
        joints: Any = None,
    ) -> None:
        if joints is not None:
            raise ValueError("RobosuiteLatchedDoor must remain a fixed fixture")
        super().__init__(name=name, friction=0.0, damping=0.1, lock=True)
        self.category_name = "robosuite_latched_door"
        self.rotation = (-np.pi / 2.0 - 0.25, -np.pi / 2.0)
        self.rotation_axis = "z"
        self.latch_joint = self.naming_prefix + "latch_joint"
        # LIBERO's generic Open predicate iterates ``joints`` and passes one
        # scalar at a time to ``is_open``. Expose only the door hinge there;
        # the spring-loaded handle remains in the compiled MJCF and is still
        # addressable through ``latch_joint``.
        self._joints = ["hinge"]
        self.object_properties = {
            "articulation": {
                "default_open_ranges": [0.31, 0.39],
                "default_close_ranges": [0.0, 0.0],
            },
            "vis_site_names": {},
        }

    @staticmethod
    def is_open(qpos: float) -> bool:
        """Preserve robosuite Door's stock ``hinge_qpos > 0.3`` checker."""

        return bool(qpos > 0.3)

    @staticmethod
    def is_close(qpos: float) -> bool:
        return bool(qpos <= 0.01)


@register_object
class MetaworldHammer(_LocalAdaptedXMLObject):
    relative_xml = "metaworld/hammer.xml"


@register_object
class MetaworldHammerBlock(_LocalAdaptedXMLObject):
    relative_xml = "metaworld/hammer_block.xml"


@register_object
class MetaworldCoffeeMachine(_LocalAdaptedXMLObject):
    """MetaWorld machine geometry with a physical spring-loaded button."""

    relative_xml = "metaworld/coffee_machine.xml"


@register_object
class ComposuiteTargetBin(_LocalAdaptedXMLObject):
    """Green receptacle used as the unambiguous weighing target."""

    relative_xml = "composuite/target_bin.xml"


class _ClierWeighingMug(_LocalAdaptedXMLObject):
    """Visually identical CLIER mug with a hidden inertial mass."""

    relative_xml = "clier/weighing_mug.xml"
    total_mass_kg: float

    def __init__(self, name: str) -> None:
        super().__init__(name)
        collision_geoms = [
            geom
            for geom in self.get_obj().findall(".//geom")
            if geom.get("group") == "0"
        ]
        if not collision_geoms:
            raise RuntimeError("CLIER weighing mug has no collision geometry")
        per_geom_mass = self.total_mass_kg / len(collision_geoms)
        for geom in collision_geoms:
            geom.set("mass", f"{per_geom_mass:.12g}")


@register_object
class ClierLightMug(_ClierWeighingMug):
    total_mass_kg = 0.18


@register_object
class ClierMediumMug(_ClierWeighingMug):
    total_mass_kg = 0.36


@register_object
class ClierHeavyMug(_ClierWeighingMug):
    total_mass_kg = 0.72


@register_object
class RobocasaSponge(_LocalAdaptedXMLObject):
    relative_xml = "robocasa_sponge.xml"


@register_object
class RobocasaSpillPatch(_LocalAdaptedXMLObject):
    relative_xml = "robocasa_spill_patch.xml"


__all__ = [
    "ClierHeavyMug",
    "ClierLightMug",
    "ClierMediumMug",
    "ComposuiteTargetBin",
    "MetaworldCoffeeMachine",
    "MetaworldHammer",
    "MetaworldHammerBlock",
    "RobocasaSpillPatch",
    "RobocasaSponge",
    "RobosuiteLatchedDoor",
]
