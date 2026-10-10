import os
import numpy as np
import re

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from libero.libero.envs.base_object import (
    register_visual_change_object,
    register_object,
)


class GoogleScannedObject(MujocoXMLObject):
    def __init__(self, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(
                str(absolute_path),
                f"assets/stable_scanned_objects/{obj_name}/{obj_name}.xml",
            ),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}


@register_object
class Rack(GoogleScannedObject):
    def __init__(
        self,
        name="simple_rack",
        obj_name="simple_rack",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints=joints)
        self.rotation = (0, 0)
        self.rotation_axis = "x"


@register_object
class WhiteBowl(GoogleScannedObject):
    def __init__(self, name="white_bowl", obj_name="white_bowl"):
        super().__init__(name, obj_name)


@register_object
class AkitaBlackBowl(GoogleScannedObject):
    def __init__(self, name="akita_black_bowl", obj_name="akita_black_bowl"):
        super().__init__(name, obj_name)


@register_object
class Plate(GoogleScannedObject):
    def __init__(self, name="plate", obj_name="plate"):
        super().__init__(name, obj_name)


@register_object
class Basket(GoogleScannedObject):
    def __init__(self, name="basket", obj_name="basket"):
        super().__init__(name, obj_name)


@register_object
class Chefmate8Frypan(GoogleScannedObject):
    def __init__(self, name="chefmate_8_frypan", obj_name="chefmate_8_frypan"):
        super().__init__(name, obj_name)


@register_object
class RobomemarenaFrypan(Chefmate8Frypan):
    """Coffee-table-specific frypan with a collision-aligned support height.

    The scanned pan's generic ``bottom_site`` is 6 cm below its physical base.
    On RoboMemArena's coffee table that makes the pan fall until its rendered
    base is hidden by the tabletop.  Keep the shared Chefmate asset unchanged
    and correct only the task-local variant used by the affected benchmark.
    """

    _BASE_COLLISION_SIZE = np.array([0.00595, 0.05485, 0.05485])
    _BASE_COLLISION_DROP_M = 0.008
    _PLACEMENT_BOTTOM_Z = -0.0112

    def __init__(self, name="robomemarena_frypan"):
        super().__init__(name=name)
        base_geoms = []
        for geom in self.get_obj().findall(".//geom"):
            if geom.get("type") != "box" or geom.get("group") != "0":
                continue
            size = np.fromstring(geom.get("size", ""), sep=" ")
            if size.shape == (3,) and np.allclose(
                size, self._BASE_COLLISION_SIZE, atol=1e-6
            ):
                base_geoms.append(geom)
        if len(base_geoms) != 1:
            raise RuntimeError(
                "expected exactly one Chefmate frypan base collision geom"
            )
        base_position = np.fromstring(base_geoms[0].get("pos", ""), sep=" ")
        if base_position.shape != (3,):
            raise RuntimeError("Chefmate frypan base collision has no 3D position")
        base_position[2] -= self._BASE_COLLISION_DROP_M
        base_geoms[0].set(
            "pos", " ".join(f"{value:.8g}" for value in base_position)
        )

        bottom_sites = [
            site
            for site in self.worldbody.findall(".//site")
            if (site.get("name") or "").endswith("bottom_site")
        ]
        if len(bottom_sites) != 1:
            raise RuntimeError("expected exactly one Chefmate frypan bottom site")
        bottom_sites[0].set("pos", f"0 0 {self._PLACEMENT_BOTTOM_Z:.8g}")


@register_object
class LongHorizonFrypan(GoogleScannedObject):
    """Chefmate pan with its handle facing the robot for the ultra-long task."""

    def __init__(self, name="long_horizon_frypan"):
        super().__init__(name, obj_name="chefmate_8_frypan")
        # The legacy LIBERO table sampler composes this value with the scanned
        # mesh's canonical frame.  Zero gives the stable tabletop pose with
        # the handle pointing toward the robot, instead of behind the bowl.
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"


@register_object
class GlazedRimPorcelainRamekin(GoogleScannedObject):
    def __init__(
        self,
        name="glazed_rim_porcelain_ramekin",
        obj_name="glazed_rim_porcelain_ramekin",
    ):
        super().__init__(name, obj_name)
