"""LIBERO object wrappers for selected VLABench assets.

The classes in this module add only ``vlabench_*`` registry keys.  Existing
LIBERO object names are intentionally untouched, so current benchmark tasks
retain their original XML and behavior.  The external asset directory is
resolved lazily when an object is instantiated; importing LIBERO therefore
does not require a 13 GB download.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import OBJECTS_DICT
from libero.libero.envs.vlabench_assets import (
    _find_nametag_assets,
    VLABenchAssetError,
    VLABenchAssetsUnavailable,
    resolve_vlabench_object_xml,
)


_UNSET = object()


def _append_visual_geom(
    parent: ET.Element,
    *,
    name: str,
    geom_type: str,
    pos: str,
    size: str,
    rgba: tuple[float, float, float, float],
    quat: str | None = None,
) -> ET.Element:
    """Add a collision-free visual cue to a compiled object subtree.

    This is used only by perception wrappers.  The cue is part of the visual
    adapter, has no contact geometry, and therefore cannot affect placement or
    the BDDL terminal checker.  Keeping it in the compiled subtree means the
    cue remains present when a real VLABench mesh (rather than a fallback XML)
    is selected.
    """

    attributes = {
        "name": name,
        "type": geom_type,
        "pos": pos,
        "size": size,
        "group": "1",
        "contype": "0",
        "conaffinity": "0",
        "rgba": " ".join(f"{value:.8g}" for value in rgba),
    }
    if quat is not None:
        attributes["quat"] = quat
    return ET.SubElement(
        parent,
        "geom",
        attributes,
    )


class VLABenchXMLObject(MujocoXMLObject):
    """Generic lazy VLABench-to-LIBERO XML object adapter.

    Subclasses set :attr:`asset_key`; callers may override ``asset_root`` or
    pass an explicit ``xml_path`` for tests and local variants.  By default a
    movable object receives robosuite's free joint.  BDDL fixtures pass
    ``joints=None`` and therefore preserve only joints defined by the asset
    itself (important for articulated fixtures such as the seesaw).  Selected
    fixture subclasses opt into concrete-geom mass repair independently of
    that no-free-joint contract.
    """

    asset_key: str | None = None
    asset_category: str = "object"
    interactive: bool = False
    allow_primitive_fallback: bool = False
    # A few source fixtures have collision-only bodies with mass=0.  Those
    # wrappers explicitly opt into the narrow concrete-geom repair because
    # LIBERO's task compiler requires a positive body mass.  This is an
    # adapter-level inertial approximation, not dynamics equivalence.
    fixture_mass_repair: bool = False
    visual_tag: str | None = None
    visual_rgba: tuple[float, float, float, float] | None = None

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        """Allow a concrete wrapper to add a narrow asset contract."""

        return xml_path

    def __init__(
        self,
        name: str | None = None,
        obj_name: str | None = None,
        joints: Any = _UNSET,
        *,
        asset_root: str | None = None,
        xml_path: str | None = None,
        obj_type: str = "all",
        duplicate_collision_geoms: bool = False,
        ensure_dynamic_mass: bool | None = None,
        repair_static_bodies: bool | None = None,
        visual_rgba: tuple[float, float, float, float] | None = _UNSET,
        visual_tag: str | None | object = _UNSET,
    ) -> None:
        key = obj_name or self.asset_key
        if not key:
            raise VLABenchAssetError(
                f"{type(self).__name__} must define asset_key or receive obj_name"
            )

        if ensure_dynamic_mass is None:
            ensure_dynamic_mass = joints is _UNSET or self.fixture_mass_repair
        if repair_static_bodies is None:
            repair_static_bodies = self.fixture_mass_repair
        if visual_rgba is _UNSET:
            visual_rgba = self.visual_rgba
        tag_content = self.visual_tag if visual_tag is _UNSET else visual_tag
        is_fallback = False
        if xml_path is None:
            xml_path_value, is_fallback = resolve_vlabench_object_xml(
                key,
                asset_root,
                allow_primitive_fallback=self.allow_primitive_fallback,
                # The no-free-joint fixture contract is independent from the
                # explicit, per-wrapper concrete mass repair opt-in.
                ensure_dynamic_mass=ensure_dynamic_mass,
                repair_static_bodies=repair_static_bodies,
                visual_tag=tag_content,
                visual_rgba=visual_rgba,
            )
        else:
            # Explicit paths are still converted, unless they are the
            # repository-owned fallback XML which already follows LIBERO's
            # body contract.  Keeping this branch explicit makes tests and
            # local asset debugging straightforward.
            explicit = Path(xml_path).expanduser().resolve()
            if not explicit.is_file():
                raise VLABenchAssetsUnavailable(f"VLABench XML does not exist: {explicit}")
            repository_support_root = (
                Path(__file__).resolve().parents[2]
                / "assets"
                / "external"
                / "vlabench"
            )
            if (
                explicit.parent == repository_support_root
                and explicit.name.endswith("_primitive.xml")
            ):
                xml_path_value = explicit
                is_fallback = True
            else:
                from libero.libero.envs.vlabench_assets import convert_vlabench_xml

                xml_path_value = convert_vlabench_xml(
                    explicit,
                    ensure_dynamic_mass=ensure_dynamic_mass,
                    repair_static_bodies=repair_static_bodies,
                    visual_tag=tag_content,
                    visual_rgba=visual_rgba,
                )

        xml_path_value = self._postprocess_xml_path(
            Path(xml_path_value), is_fallback
        )

        joint_specs = (
            [dict(type="free", damping="0.0005")]
            if joints is _UNSET
            else copy.deepcopy(joints)
        )
        super().__init__(
            str(xml_path_value),
            name=name or key,
            joints=joint_specs,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )

        self.asset_key = key
        self.asset_path = str(xml_path_value)
        self.using_primitive_fallback = is_fallback
        self.category_name = f"vlabench_{key}"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


class VLABenchOpenCloseMixin:
    """Implement LIBERO's articulated-object Open / Close contract.

    VLABench XML files describe the mechanism joints, but do not provide the
    LIBERO-side state predicates or the ranges consumed by
    ``OpenCloseSampler``.  This mixin adds that narrow bridge only to wrappers
    whose pinned assets have a stable, common joint range.  The source range
    is retained in metadata and the state thresholds are deliberately kept
    away from the endpoints so reset noise cannot make an open object appear
    closed (or vice versa).

    The mixin is intentionally not part of :class:`VLABenchXMLObject`: a
    static object must not accidentally advertise an Open / Close affordance
    merely because it shares the generic XML adapter.
    """

    open_direction = "upper"
    open_fraction = 0.5
    close_fraction = 0.1

    def _configure_open_close_contract(
        self,
        *,
        open_direction: str | None = None,
        open_fraction: float | None = None,
        close_fraction: float | None = None,
    ) -> None:
        direction = open_direction or self.open_direction
        open_fraction = self.open_fraction if open_fraction is None else open_fraction
        close_fraction = self.close_fraction if close_fraction is None else close_fraction
        articulation: dict[str, Any] = {
            "state_contract": "disabled",
            "joint_names": [],
            "joint_ranges": [],
            "default_open_ranges": [],
            "default_close_ranges": [],
        }
        self.object_properties["articulation"] = articulation

        if direction not in {"upper", "lower"}:
            articulation["disabled_reason"] = f"unsupported open direction {direction!r}"
            self.interactive = False
            return
        if not (0.0 < float(open_fraction) <= 1.0) or not (
            0.0 < float(close_fraction) <= 1.0
        ):
            articulation["disabled_reason"] = "invalid open/close threshold fractions"
            self.interactive = False
            return

        declared_joints = set(str(name) for name in getattr(self, "joints", ()))
        joint_ranges: list[tuple[float, float]] = []
        joint_names: list[str] = []
        for joint in self.root.iter("joint"):
            name = joint.get("name")
            # Mujoco's default joint type is hinge when omitted.  Free joints
            # are wrapper-added movable-object joints and are never part of an
            # articulated fixture state contract.
            joint_type = joint.get("type", "hinge")
            if not name or (declared_joints and name not in declared_joints):
                continue
            if joint_type not in {"hinge", "slide"}:
                continue
            values = (joint.get("range") or "").replace(",", " ").split()
            if len(values) < 2:
                continue
            try:
                lower, upper = float(values[0]), float(values[1])
            except (TypeError, ValueError):
                continue
            if not (lower < upper):
                continue
            joint_names.append(name)
            joint_ranges.append((lower, upper))

        # ``OpenCloseSampler`` sets every object joint to one scalar.  Using a
        # single range for mechanisms with heterogeneous ranges would silently
        # violate at least one source joint's limits, so disable the contract
        # instead of pretending it is task-compatible.
        if not joint_ranges:
            articulation["disabled_reason"] = "no ranged hinge/slide joint found"
            self.interactive = False
            return
        reference_lower, reference_upper = joint_ranges[0]
        if any(
            abs(lower - reference_lower) > 1e-6
            or abs(upper - reference_upper) > 1e-6
            for lower, upper in joint_ranges[1:]
        ):
            articulation["disabled_reason"] = (
                "articulated joints do not share one scalar OpenCloseSampler range"
            )
            articulation["joint_names"] = joint_names
            articulation["joint_ranges"] = [list(item) for item in joint_ranges]
            self.interactive = False
            return

        span = reference_upper - reference_lower
        if direction == "upper":
            open_ranges = [
                reference_lower + float(open_fraction) * span,
                reference_upper,
            ]
            close_ranges = [
                reference_lower,
                reference_lower + float(close_fraction) * span,
            ]
        else:
            open_ranges = [
                reference_lower,
                reference_lower + float(open_fraction) * span,
            ]
            close_ranges = [
                reference_upper - float(close_fraction) * span,
                reference_upper,
            ]
        # Keep the public metadata stable for simple source ranges such as
        # [0, 0.6].  This also avoids exposing binary floating-point noise in
        # BDDL/debug reports (e.g. 0.09999999999999999 instead of 0.1).
        open_ranges = [round(float(value), 12) for value in open_ranges]
        close_ranges = [round(float(value), 12) for value in close_ranges]

        articulation.update(
            {
                "state_contract": "open_close",
                "joint_names": joint_names,
                "joint_ranges": [list(item) for item in joint_ranges],
                "source_joint_range": [reference_lower, reference_upper],
                "default_open_ranges": open_ranges,
                "default_close_ranges": close_ranges,
                "open_direction": direction,
                "dynamics_equivalent": not bool(
                    getattr(self, "fixture_mass_repair", False)
                ),
            }
        )

    def _state_contract_range(self, state: str) -> tuple[float, float]:
        articulation = self.object_properties.get("articulation", {})
        if articulation.get("state_contract") != "open_close":
            raise RuntimeError(
                f"{type(self).__name__} does not expose a valid Open/Close contract: "
                f"{articulation.get('disabled_reason', 'disabled')}"
            )
        values = articulation.get(f"default_{state}_ranges")
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise RuntimeError(f"invalid VLABench {state} range: {values!r}")
        return float(values[0]), float(values[1])

    def is_open(self, qpos: float) -> bool:
        """Return whether one mechanism joint is in its configured open band."""

        lower, upper = self._state_contract_range("open")
        value = float(qpos)
        if self.open_direction == "upper":
            return value >= lower
        return value <= upper

    def is_close(self, qpos: float) -> bool:
        """Return whether one mechanism joint is in its configured close band."""

        lower, upper = self._state_contract_range("close")
        value = float(qpos)
        if self.open_direction == "upper":
            return value <= upper
        return value >= lower

    @property
    def provenance(self) -> dict[str, str | bool]:
        """Return source metadata without implying an asset license grant."""

        from libero.libero.envs.vlabench_assets import (
            VLABENCH_ASSET_REPOSITORY,
            VLABENCH_ASSET_REVISION,
            VLABENCH_SOURCE_COMMIT,
            VLABENCH_SOURCE_REPOSITORY,
        )

        return {
            "source_repository": VLABENCH_SOURCE_REPOSITORY,
            "source_commit": VLABENCH_SOURCE_COMMIT,
            "asset_repository": VLABENCH_ASSET_REPOSITORY,
            "asset_revision": VLABENCH_ASSET_REVISION,
            "external_asset": not self.using_primitive_fallback,
            "license_grant": False,
        }


class VLABenchCondiment(VLABenchXMLObject):
    asset_category = "condiment"


class VLABenchContainer(VLABenchXMLObject):
    asset_category = "container"


class VLABenchTool(VLABenchXMLObject):
    asset_category = "tool"


class VLABenchAppearanceMixin:
    """Keep perception appearance changes inside the object adapter.

    VLABench's released bundle carries the real mesh/material variation.  The
    repository-owned primitive is intentionally tiny, so source-only tests
    need a deterministic visual surrogate as well.  The surrogate colours are
    selected from the private instance name while the observation layer still
    projects anonymous ``entity_###`` identifiers.  No appearance label is
    copied into task metadata or the terminal checker.
    """

    appearance_palette: dict[str, tuple[float, float, float, float]] = {}
    default_appearance = (0.35, 0.55, 0.80, 1.0)
    apply_external_appearance = False

    @classmethod
    def _appearance_for_name(
        cls, name: str | None
    ) -> tuple[float, float, float, float]:
        token = str(name or "").lower()
        for key, rgba in cls.appearance_palette.items():
            if key in token:
                return rgba
        # Neutral letter suffixes make matched source-only instances visibly
        # distinct without baking an answer or role into public annotations.
        suffix_palette = {
            "_a": (0.86, 0.20, 0.14, 1.0),
            "_b": (0.16, 0.45, 0.86, 1.0),
            "_c": (0.18, 0.65, 0.26, 1.0),
        }
        for suffix, rgba in suffix_palette.items():
            if token.endswith(suffix) or f"{suffix}_" in token:
                return rgba
        return cls.default_appearance

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        instance_name = kwargs.get("name") or kwargs.get("obj_name")
        super().__init__(*args, **kwargs)
        appearance = self._appearance_for_name(instance_name)
        self.appearance_rgba = tuple(float(value) for value in appearance)
        # Real VLABench XML already has its own texture/material contract.  A
        # fallback has no texture files, so apply the visual-only appearance
        # to its group-1 geoms after the wrapper has loaded the XML.
        if self.using_primitive_fallback or self.apply_external_appearance:
            self._apply_fallback_appearance(self.appearance_rgba)

    def _apply_fallback_appearance(
        self, rgba: tuple[float, float, float, float]
    ) -> None:
        # ``MujocoXMLObject`` deep-copies the object subtree into ``get_obj``
        # before adding LIBERO's naming prefix and free joint.  Mutating
        # ``self.root`` here would make source inspection look correct while
        # leaving the compiled MuJoCo model unchanged.
        for geom in self.get_obj().iter("geom"):
            if geom.get("group", "1") != "1":
                continue
            geom.set("rgba", " ".join(f"{value:.8g}" for value in rgba))


class VLABenchTableware(VLABenchXMLObject):
    """Real VLABench tableware meshes used by ordered table setting."""

    asset_category = "tableware"


class VLABenchTube(VLABenchXMLObject):
    """External VLABench ChemistryTube with task-visible solution colours."""

    asset_key = "tube"
    asset_category = "tube"

    _SOLUTION_RGBA = {
        "blue": (0.0, 0.45, 1.0, 0.4),  # CuSO4
        "green": (0.141, 1.0, 0.174043, 0.4),  # CuCl2
        "purple": (0.5, 0.0, 0.5, 0.4),  # KMnO4
    }

    @classmethod
    def _rgba_for_name(cls, name: str | None) -> tuple[float, float, float, float]:
        token = str(name or "").lower()
        for colour, rgba in cls._SOLUTION_RGBA.items():
            if re.search(rf"(?:^|[_-]){re.escape(colour)}(?:[_-]|$)", token):
                return rgba
        # The target in the precision task is deliberately blue even when a
        # caller uses a neutral instance name.  This keeps the wrapper useful
        # for one-tube smoke tests while distractor colours remain opt-in.
        return cls._SOLUTION_RGBA["blue"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "visual_rgba" not in kwargs:
            kwargs["visual_rgba"] = self._rgba_for_name(
                kwargs.get("name") or kwargs.get("obj_name")
            )
        super().__init__(*args, **kwargs)
        self._reorient_physics_subtree()
        self.solution_rgba = tuple(float(value) for value in kwargs["visual_rgba"])

    def _reorient_physics_subtree(self) -> None:
        """Keep the tube's long axis vertical under a free MuJoCo joint.

        The pinned XML puts the tube geometry under a body with ``euler=1.57
        0 0``.  MuJoCo's free-joint qpos supplies the runtime body rotation,
        and the fixed root rotation is not applied to its direct geoms in the
        native binding used by LIBERO.  A fixed child body preserves that
        source orientation while leaving the free root available to the
        placement sampler.
        """

        root = self.get_obj()
        source_euler = root.get("euler", "1.57 0 0")
        root.attrib.pop("euler", None)
        children = [child for child in list(root) if child.tag != "joint"]
        if not children:
            return
        axis_body = ET.Element(
            "body",
            {"name": f"{self.name}_tube_axis", "euler": source_euler},
        )
        for child in children:
            root.remove(child)
            axis_body.append(child)
        root.append(axis_body)


class VLABenchTubeStand(VLABenchXMLObject):
    """Static external tube rack with one named precision target slot."""

    asset_key = "tube_stand"
    asset_category = "tube_fixture"
    fixture_mass_repair = True

    # World-frame offsets from the stand root.  The source mesh's euler=Z(pi/2)
    # rotates its long axis into world X, matching VLABench's 5-column layout.
    SLOT_WORLD_OFFSETS = {"rear_center_slot": (0.0, 0.05, 0.055)}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # A rack is a fixed fixture; never let the generic wrapper add a free
        # joint when a caller omits ``joints``.
        kwargs.setdefault("joints", None)
        super().__init__(*args, **kwargs)
        self._add_precision_slot_sites()

    @staticmethod
    def _world_to_source_local(offset: tuple[float, float, float]) -> tuple[float, float, float]:
        world_x, world_y, world_z = offset
        return (world_y, -world_x, world_z)

    def _add_precision_slot_sites(self) -> None:
        inner = self.get_obj()
        outer = self.worldbody.find("./body")
        if outer is None:
            raise VLABenchAssetError("tube stand adapter has no outer body")
        source_inner = outer.find("./body")
        if source_inner is None:
            raise VLABenchAssetError("tube stand adapter has no source object body")
        for slot_name, world_offset in self.SLOT_WORLD_OFFSETS.items():
            full_name = f"{self.name}_{slot_name}"
            local_offset = self._world_to_source_local(world_offset)
            attributes = {
                "name": full_name,
                "pos": " ".join(f"{value:.8g}" for value in local_offset),
                "type": "box",
                "size": "0.022 0.022 0.08",
                "quat": "1 0 0 0",
                "rgba": "0 0 0 0",
                "group": "3",
            }
            inner.append(ET.Element("site", attributes))
            # LIBERO discovers BDDL regions from the pre-merge ``worldbody``
            # metadata tree, while MuJoCo receives ``get_obj()`` above.  Keep
            # the same real site in both trees so discovery and simulation
            # refer to one physical slot.
            source_inner.append(ET.Element("site", attributes))
            outer_attributes = dict(attributes)
            outer_attributes["pos"] = " ".join(
                f"{value:.8g}" for value in world_offset
            )
            outer.append(ET.Element("site", outer_attributes))
            # ``MujocoObject.sites`` applies ``self.naming_prefix`` lazily;
            # retain the unprefixed logical token in this bookkeeping list
            # while the XML node itself is already prefixed (we add it after
            # robosuite's normal XML prefix pass).
            self._sites.append(slot_name)


class VLABenchRearrangementTubeStand(VLABenchTubeStand):
    """Two-slot rack variant with visible, non-interactive solution labels.

    The labels are attached to the same static rack body as visual-only
    NameTags.  They are deliberately not BDDL objects: the private checker
    owns the tube-to-slot mapping, while the rendered labels are the only
    semantic cue available to an agent.
    """

    SLOT_WORLD_OFFSETS = {
        "left_slot": (-0.08, 0.05, 0.055),
        "right_slot": (0.08, 0.05, 0.055),
    }
    _SLOT_LABELS = (
        ("left_slot", "cuso4"),
        ("right_slot", "cucl2"),
    )

    def _add_precision_slot_sites(self) -> None:
        super()._add_precision_slot_sites()
        inner = self.get_obj()
        outer = self.worldbody.find("./body")
        if outer is None:
            raise VLABenchAssetError("rearrangement tube stand has no outer body")
        source_inner = outer.find("./body")
        if source_inner is None:
            raise VLABenchAssetError(
                "rearrangement tube stand has no source object body"
            )
        world_offset = (0.0, 0.05, 0.055)
        local_offset = self._world_to_source_local(world_offset)
        attributes = {
            "name": f"{self.name}_rack_region",
            "pos": " ".join(f"{value:.8g}" for value in local_offset),
            "type": "box",
            "size": "0.13 0.12 0.08",
            "quat": "1 0 0 0",
            "rgba": "0 0 0 0",
            "group": "3",
        }
        inner.append(ET.Element("site", attributes))
        source_inner.append(ET.Element("site", attributes))
        outer_attributes = dict(attributes)
        outer_attributes["pos"] = " ".join(
            f"{value:.8g}" for value in world_offset
        )
        outer.append(ET.Element("site", outer_attributes))
        self._sites.append("rack_region")

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        xml_path = super()._postprocess_xml_path(xml_path, is_fallback)
        if is_fallback:
            return xml_path
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            source_inner = root.find("./worldbody/body/body")
            if source_inner is None:
                raise VLABenchAssetError(
                    "rearrangement tube stand adapter has no source object body"
                )
            mesh_hint = None
            for mesh in root.findall("./asset/mesh"):
                file_name = mesh.get("file")
                if file_name:
                    mesh_hint = Path(file_name)
                    break
            if mesh_hint is None:
                raise VLABenchAssetError(
                    "rearrangement tube stand adapter has no source mesh"
                )
            asset = root.find("asset")
            if asset is None:
                asset = ET.SubElement(root, "asset")
            for slot_name, label in self._SLOT_LABELS:
                mesh_path, texture_path = _find_nametag_assets(mesh_hint, label) or (None, None)
                if mesh_path is None or texture_path is None:
                    raise VLABenchAssetsUnavailable(
                        "rearrangement tube rack requires the external NameTag mesh "
                        f"and solution/{label}.png"
                    )
                token = f"rearrange_{label}"
                mesh_name = f"vlabench_{token}_rack_tag_mesh"
                texture_name = f"vlabench_{token}_rack_tag_texture"
                material_name = f"vlabench_{token}_rack_tag_material"
                ET.SubElement(
                    asset,
                    "mesh",
                    {
                        "name": mesh_name,
                        "file": str(mesh_path),
                        "scale": "0.06 0.06 0.06",
                    },
                )
                ET.SubElement(
                    asset,
                    "texture",
                    {
                        "name": texture_name,
                        "type": "2d",
                        "file": str(texture_path),
                    },
                )
                ET.SubElement(
                    asset,
                    "material",
                    {"name": material_name, "texture": texture_name},
                )
                world_offset = self.SLOT_WORLD_OFFSETS[slot_name]
                local_x, local_y, local_z = self._world_to_source_local(
                    (world_offset[0], world_offset[1] - 0.025, 0.145)
                )
                tag_body = ET.SubElement(
                    source_inner,
                    "body",
                    {
                        "name": f"{token}_nametag",
                        "pos": f"{local_x:.8g} {local_y:.8g} {local_z:.8g}",
                        "euler": "0 1.57079632679 1.57079632679",
                    },
                )
                visual = {
                    "type": "mesh",
                    "mesh": mesh_name,
                    "material": material_name,
                    "group": "1",
                    "contype": "0",
                    "conaffinity": "0",
                }
                ET.SubElement(tag_body, "geom", {"name": f"{token}_nametag_front", **visual})
                ET.SubElement(
                    tag_body,
                    "geom",
                    {
                        "name": f"{token}_nametag_back",
                        "pos": "0 -0.00005 0",
                        "quat": "0 1 0 0",
                        **visual,
                    },
                )
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            return xml_path
        except ET.ParseError as exc:
            raise VLABenchAssetError(
                f"unable to parse converted rearrangement rack XML {xml_path}: {exc}"
            ) from exc


class VLABenchBBQSauce(VLABenchCondiment):
    asset_key = "bbq_sauce"


class VLABenchHotSauce(VLABenchCondiment):
    asset_key = "hotsauce"


class VLABenchKetchup(VLABenchCondiment):
    asset_key = "ketchup"


class VLABenchSaladDressing(VLABenchCondiment):
    asset_key = "salad_dressing"


class VLABenchShaker(VLABenchCondiment):
    asset_key = "shaker"


class VLABenchSugarShaker(VLABenchCondiment):
    """VLABench shaker variant used for the normal sugar semantic role."""

    asset_key = "sugar"
    visual_tag = "sugar"


class VLABenchSaltShaker(VLABenchCondiment):
    """VLABench shaker variant used for the normal salt semantic role."""

    asset_key = "salt"
    visual_tag = "salt"


class VLABenchFlowerAssetMixin:
    """Fit large nested flower assemblies into LIBERO tabletop slots.

    The released flower XMLs are authored at a much larger footprint than
    LIBERO's three matched item slots.  Uniformly lying about
    ``horizontal_radius`` would make placement succeed while leaving the
    mesh/collision assembly overlapping its neighbours.  Instead, the
    generated adapter applies a narrow, non-uniform body scale to the real
    mesh assembly and scales its placement sites by the same factors.  The
    source XML and all external mesh/texture files remain unchanged.
    """

    flower_body_scale = (0.35, 0.35, 0.75)

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        xml_path = super()._postprocess_xml_path(xml_path, is_fallback)
        if is_fallback:
            return xml_path

        tree = ET.parse(xml_path)
        root = tree.getroot()
        outer = root.find("./worldbody/body")
        inner = root.find("./worldbody/body/body[@name='object']")
        if outer is None or inner is None:
            raise VLABenchAssetError(
                f"Converted flower has no object wrapper: {xml_path}"
            )

        scale = tuple(float(value) for value in self.flower_body_scale)

        def scaled_attribute(element: ET.Element, attribute: str) -> None:
            values = element.get(attribute, "").replace(",", " ").split()
            try:
                position = tuple(float(value) for value in values)
            except ValueError:
                return
            if len(position) != 3:
                return
            element.set(
                attribute,
                " ".join(
                    f"{position[index] * scale[index]:.8g}"
                    for index in range(3)
                ),
            )

        # MuJoCo does not support a ``scale`` attribute on bodies.  Scale the
        # generated adapter's mesh declarations and concrete local geometry
        # transforms instead; the source tree remains untouched.
        asset = root.find("./asset")
        if asset is not None:
            for mesh in asset.findall("./mesh"):
                values = mesh.get("scale", "1 1 1").replace(",", " ").split()
                try:
                    previous_values = tuple(float(value) for value in values)
                except ValueError:
                    previous_values = (1.0, 1.0, 1.0)
                if len(previous_values) == 3:
                    mesh.set(
                        "scale",
                        " ".join(
                            f"{previous_values[index] * scale[index]:.8g}"
                            for index in range(3)
                        ),
                    )

        for element in inner.iter():
            if element.tag in {"body", "geom", "site", "joint", "inertial"}:
                scaled_attribute(element, "pos")
            if element.tag == "geom":
                scaled_attribute(element, "size")

        # The direct source bodies can carry offsets relative to the adapter
        # outer body; their positions were handled above.  Placement sites are
        # on that outer body and therefore need the same transform explicitly.
        for site in outer.findall("./site"):
            if site.get("name") in {
                "bottom_site",
                "top_site",
                "horizontal_radius_site",
            }:
                scaled_attribute(site, "pos")

        # The source flowers contain dozens of thin, intersecting convex
        # collision pieces for petals and leaves.  At tabletop scale those
        # pieces catch one finger or the table before the parallel jaws can
        # establish a stable stem grasp.  Preserve the full mesh appearance,
        # but expose a conservative physical envelope: a narrow stem capsule
        # plus a flower-head sphere.  Both remain ordinary MuJoCo contacts.
        for geom in inner.iter("geom"):
            if geom.get("contype", "1") != "0":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        ET.SubElement(
            inner,
            "geom",
            {
                "name": "flower_stem_collision_proxy",
                "type": "capsule",
                "fromto": "0 -0.040 0 0 0.020 0",
                "size": "0.008",
                "mass": "0.035",
                "friction": "0.95 0.3 0.1",
                "group": "0",
                "contype": "1",
                "conaffinity": "1",
                "rgba": "0 0 0 0",
            },
        )
        ET.SubElement(
            inner,
            "geom",
            {
                "name": "flower_head_collision_proxy",
                "type": "sphere",
                "pos": "0 0.050 0",
                "size": "0.026",
                "mass": "0.040",
                "friction": "0.95 0.3 0.1",
                "group": "0",
                "contype": "1",
                "conaffinity": "1",
                "rgba": "0 0 0 0",
            },
        )

        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        return xml_path


class VLABenchFruit(VLABenchAppearanceMixin, VLABenchXMLObject):
    """VLABench fruit family used by the common-sense world task."""

    asset_key = "fruit"
    asset_category = "fruit"
    allow_primitive_fallback = True
    # The released fruit meshes are authored at roughly 11 cm diameter for
    # round fruit, wider than Panda's physical 8 cm parallel-jaw opening.
    # A uniform 0.68 adapter scale keeps the recognizable source geometry
    # while restoring a physically graspable household-object size.  This is
    # applied only to the generated adapter, never the pinned VLABench assets.
    fruit_body_scale = 0.68
    appearance_palette = {
        "orange": (0.94, 0.38, 0.06, 1.0),
        "apple": (0.78, 0.08, 0.06, 1.0),
        "banana": (0.96, 0.82, 0.12, 1.0),
        "lemon": (0.98, 0.86, 0.16, 1.0),
        "kiwi": (0.37, 0.64, 0.18, 1.0),
    }

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        xml_path = super()._postprocess_xml_path(xml_path, is_fallback)
        if is_fallback:
            return xml_path

        tree = ET.parse(xml_path)
        root = tree.getroot()
        outer = root.find("./worldbody/body")
        inner = root.find("./worldbody/body/body[@name='object']")
        if outer is None or inner is None:
            raise VLABenchAssetError(
                f"Converted fruit has no object wrapper: {xml_path}"
            )
        scale = float(self.fruit_body_scale)

        def scale_vec3(element: ET.Element, attribute: str) -> None:
            values = element.get(attribute, "").replace(",", " ").split()
            try:
                vector = tuple(float(value) for value in values)
            except ValueError:
                return
            if len(vector) != 3:
                return
            element.set(
                attribute,
                " ".join(f"{value * scale:.8g}" for value in vector),
            )

        asset = root.find("./asset")
        if asset is not None:
            for mesh in asset.findall("./mesh"):
                values = mesh.get("scale", "1 1 1").replace(",", " ").split()
                try:
                    previous = tuple(float(value) for value in values)
                except ValueError:
                    previous = (1.0, 1.0, 1.0)
                if len(previous) == 3:
                    mesh.set(
                        "scale",
                        " ".join(f"{value * scale:.8g}" for value in previous),
                    )
        for element in inner.iter():
            if element.tag in {"body", "geom", "site", "joint", "inertial"}:
                scale_vec3(element, "pos")
            if element.tag == "geom":
                scale_vec3(element, "size")
        for site in outer.findall("./site"):
            if site.get("name") in {
                "bottom_site",
                "top_site",
                "horizontal_radius_site",
            }:
                scale_vec3(site, "pos")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        return xml_path


class VLABenchOrangeFruit(VLABenchFruit):
    asset_key = "orange_fruit"
    appearance_palette = {"orange": (0.94, 0.38, 0.06, 1.0)}


class VLABenchAppleFruit(VLABenchFruit):
    asset_key = "apple_fruit"
    appearance_palette = {"apple": (0.78, 0.08, 0.06, 1.0)}


class VLABenchBananaFruit(VLABenchFruit):
    asset_key = "banana_fruit"
    appearance_palette = {"banana": (0.96, 0.82, 0.12, 1.0)}

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        xml_path = super()._postprocess_xml_path(xml_path, is_fallback)
        if is_fallback:
            return xml_path

        tree = ET.parse(xml_path)
        root = tree.getroot()
        object_body = root.find("./worldbody/body/body[@name='object']")
        if object_body is None:
            raise VLABenchAssetError(
                f"Converted banana has no object body: {xml_path}"
            )
        # The released banana uses one concave mesh as its collision body.
        # It can rest visually on the table, but Panda's parallel fingers pass
        # through the sparse concavity without establishing a two-sided grasp.
        # Keep the source mesh for appearance and replace only its contact
        # envelope with a conservative capsule following the visible fruit.
        for geom in object_body.iter("geom"):
            if geom.get("contype", "1") != "0":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": "banana_collision_proxy",
                "type": "capsule",
                "fromto": "-0.050 0 0 0.050 0 0",
                "size": "0.018",
                "mass": "0.050",
                "friction": "0.95 0.3 0.1",
                "solref": "0.001 1",
                "group": "0",
                "contype": "1",
                "conaffinity": "1",
                "rgba": "0 0 0 0",
            },
        )
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        return xml_path


class VLABenchDrink(VLABenchAppearanceMixin, VLABenchXMLObject):
    """VLABench drink family used by the common-sense world task."""

    asset_key = "drink"
    asset_category = "drink"
    allow_primitive_fallback = True
    appearance_palette = {
        "milk": (0.94, 0.96, 0.90, 1.0),
        "cola": (0.20, 0.07, 0.025, 1.0),
        "juice": (0.94, 0.50, 0.08, 1.0),
        "water": (0.32, 0.70, 0.94, 1.0),
    }


class VLABenchMilkDrink(VLABenchDrink):
    asset_key = "milk_drink"
    appearance_palette = {"milk": (0.94, 0.96, 0.90, 1.0)}


class VLABenchColaDrink(VLABenchDrink):
    asset_key = "cola_drink"
    appearance_palette = {"cola": (0.20, 0.07, 0.025, 1.0)}


class VLABenchJuiceDrink(VLABenchDrink):
    asset_key = "juice_drink"
    appearance_palette = {"juice": (0.94, 0.50, 0.08, 1.0)}


class VLABenchBloomFlower(
    VLABenchFlowerAssetMixin, VLABenchAppearanceMixin, VLABenchXMLObject
):
    """VLABench blooming flower family."""

    asset_key = "bloom_flower"
    asset_category = "flower"
    allow_primitive_fallback = True
    # The released flower is a nested assembly: its petal child bodies carry
    # collision geoms with source ``mass=0`` while a separate stem body holds
    # the only small weight.  LIBERO adds a free joint to the outer object,
    # so repair one concrete collision geom in each otherwise-static child
    # within the generated adapter.  This does not edit or reserialize the
    # downloaded asset tree.
    fixture_mass_repair = True
    appearance_palette = {
        "rose": (0.92, 0.10, 0.30, 1.0),
        "sunflower": (0.96, 0.68, 0.08, 1.0),
        "daisy": (0.95, 0.95, 0.90, 1.0),
        "peony": (0.94, 0.34, 0.62, 1.0),
        "chrysanthemum": (0.96, 0.78, 0.18, 1.0),
    }


class VLABenchRoseFlower(VLABenchBloomFlower):
    asset_key = "rose_flower"
    appearance_palette = {"rose": (0.92, 0.10, 0.30, 1.0)}


class VLABenchSunflower(VLABenchBloomFlower):
    asset_key = "sunflower_flower"
    appearance_palette = {"sunflower": (0.96, 0.68, 0.08, 1.0)}


class VLABenchDaisyFlower(VLABenchBloomFlower):
    asset_key = "daisy_flower"
    appearance_palette = {"daisy": (0.95, 0.95, 0.90, 1.0)}


class VLABenchPeonyFlower(VLABenchBloomFlower):
    """Readable blooming-flower distractor for the romantic-love task."""

    asset_key = "peony_flower"
    appearance_palette = {"peony": (0.94, 0.34, 0.62, 1.0)}


class VLABenchChrysanthemumFlower(VLABenchBloomFlower):
    """Readable chrysanthemum distractor for the romantic-love task."""

    asset_key = "chrysanthemum_flower"
    appearance_palette = {"chrysanthemum": (0.96, 0.78, 0.18, 1.0)}


class VLABenchTulipFlower(VLABenchBloomFlower):
    """Readable tulip distractor candidate for flower-selection scenes."""

    asset_key = "tulip_flower"
    appearance_palette = {"tulip": (0.92, 0.18, 0.12, 1.0)}


class VLABenchWiltedFlower(
    VLABenchFlowerAssetMixin, VLABenchAppearanceMixin, VLABenchXMLObject
):
    """VLABench wilted flower family for matched perception scenes."""

    asset_key = "wilted_flower"
    asset_category = "flower"
    allow_primitive_fallback = True
    fixture_mass_repair = True
    default_appearance = (0.48, 0.43, 0.34, 1.0)


class VLABenchChemistrySolution(VLABenchAppearanceMixin, VLABenchXMLObject):
    """VLABench chemistry-tube family with solution appearance cues."""

    asset_key = "chemistry_solution"
    asset_category = "tube"
    allow_primitive_fallback = True
    appearance_palette = {
        "blue": (0.08, 0.40, 0.92, 1.0),
        "green": (0.10, 0.72, 0.28, 1.0),
        "purple": (0.54, 0.14, 0.72, 1.0),
        "yellow": (0.92, 0.82, 0.12, 1.0),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        name = kwargs.get("name") or kwargs.get("obj_name")
        kwargs.setdefault("visual_rgba", self._appearance_for_name(name))
        super().__init__(*args, **kwargs)


class VLABenchBlueChemistrySolution(VLABenchChemistrySolution):
    asset_key = "chemistry_blue"
    appearance_palette = {"blue": (0.08, 0.40, 0.92, 1.0)}


class VLABenchGreenChemistrySolution(VLABenchChemistrySolution):
    asset_key = "chemistry_green"
    appearance_palette = {"green": (0.10, 0.72, 0.28, 1.0)}


class VLABenchPurpleChemistrySolution(VLABenchChemistrySolution):
    asset_key = "chemistry_purple"
    appearance_palette = {"purple": (0.54, 0.14, 0.72, 1.0)}


class VLABenchPerceptionToken(VLABenchAppearanceMixin, VLABenchXMLObject):
    """One shared mesh surrogate for upright/fallen and instance tasks."""

    asset_key = "perception_token"
    asset_category = "perception"
    allow_primitive_fallback = True

    @classmethod
    def _appearance_for_name(
        cls, _name: str | None
    ) -> tuple[float, float, float, float]:
        # Upright-vs-fallen is intentionally a pose-only cue: all candidates
        # use the same mesh and the same appearance.
        return cls.default_appearance


class VLABenchPerceptionColorToken(VLABenchAppearanceMixin, VLABenchXMLObject):
    """Matched colour discrimination object."""

    asset_key = "perception_color"
    asset_category = "perception"
    allow_primitive_fallback = True
    apply_external_appearance = True
    appearance_palette = {
        "red": (0.90, 0.08, 0.06, 1.0),
        "blue": (0.08, 0.30, 0.90, 1.0),
        "green": (0.08, 0.65, 0.18, 1.0),
    }

    def _apply_fallback_appearance(
        self, rgba: tuple[float, float, float, float]
    ) -> None:
        # Source bottles can carry a texture material.  A colour task's
        # defining cue is the wrapper-selected solid colour, so remove only
        # that visual material while preserving every source mesh and all
        # collision geoms.
        for geom in self.get_obj().iter("geom"):
            if geom.get("group", "1") != "1":
                continue
            geom.attrib.pop("material", None)
            geom.set("rgba", " ".join(f"{value:.8g}" for value in rgba))


class VLABenchPerceptionTextureToken(VLABenchAppearanceMixin, VLABenchXMLObject):
    """Matched texture/stripe discrimination object."""

    asset_key = "perception_texture"
    asset_category = "perception"
    allow_primitive_fallback = True
    apply_external_appearance = True
    appearance_palette = {
        # Matched texture candidates deliberately share one base colour.  The
        # only task-visible distinction is the stripe / spot / plain cue that
        # this adapter adds below, never a palette shortcut.
        "striped": (0.20, 0.56, 0.78, 1.0),
        "spotted": (0.20, 0.56, 0.78, 1.0),
        "plain": (0.20, 0.56, 0.78, 1.0),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The external drink mesh has a deeply transformed bottle body.  A
        # cue appended to its LIBERO wrapper frame therefore floats beside
        # the bottle instead of lying on its surface.  Texture discrimination
        # needs a controlled, shared geometry, so use the repository-owned
        # cuboid for all three candidates and vary only the requested cue.
        local_xml = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "external"
            / "vlabench"
            / "perception_texture_primitive.xml"
        )
        kwargs.setdefault("xml_path", str(local_xml))
        super().__init__(*args, **kwargs)

    def _apply_fallback_appearance(
        self, rgba: tuple[float, float, float, float]
    ) -> None:
        object_subtree = self.get_obj()
        token = str(getattr(self, "name", "")).lower()
        is_striped = "striped" in token
        is_spotted = "spotted" in token
        accent_geoms = []
        for geom in object_subtree.iter("geom"):
            if geom.get("group", "1") != "1":
                continue
            if geom.get("name", "").endswith("texture_accent"):
                accent = (
                    tuple(max(0.02, value * 0.22) for value in rgba[:3])
                    + (rgba[3],)
                    if is_striped
                    else (rgba[0], rgba[1], rgba[2], 0.0)
                )
                geom.set("rgba", " ".join(f"{value:.8g}" for value in accent))
                accent_geoms.append(geom)
            else:
                geom.set("rgba", " ".join(f"{value:.8g}" for value in rgba))

        # The fallback XML already has one accent geom.  Real VLABench bottle
        # XMLs do not, so add the same cue directly to the compiled subtree;
        # this keeps striped / spotted / plain visibly distinct in either
        # asset mode without changing the source download tree.
        if is_striped:
            stripe_rgba = (0.04, 0.04, 0.04, 1.0)
            stripe_positions = (
                "-0.024 0 0.046",
                "0 0 0.046",
                "0.024 0 0.046",
            )
            existing = len(accent_geoms)
            for index, pos in enumerate(stripe_positions[existing:], start=existing):
                _append_visual_geom(
                    object_subtree,
                    name=f"{self.name}_texture_stripe_{index}",
                    geom_type="box",
                    pos=pos,
                    size="0.004 0.035 0.002",
                    rgba=stripe_rgba,
                )
        elif is_spotted:
            # Use an equilateral three-dot mark on the canonical top face.
            # The former asymmetric offsets made the distractor look like a
            # partially occluded or malformed texture in the goal frame.
            for index, pos in enumerate(
                (
                    "-0.017 -0.010 0.047",
                    "0.017 -0.010 0.047",
                    "0 0.0194 0.047",
                )
            ):
                _append_visual_geom(
                    object_subtree,
                    name=f"{self.name}_texture_spot_{index}",
                    geom_type="sphere",
                    pos=pos,
                    size="0.006",
                    rgba=(0.02, 0.02, 0.02, 1.0),
                )


class VLABenchPerceptionInstanceToken(VLABenchAppearanceMixin, VLABenchXMLObject):
    """Same-mesh object family with ordinary visual instance marks."""

    asset_key = "perception_instance"
    asset_category = "perception"
    allow_primitive_fallback = True
    apply_external_appearance = True
    appearance_palette = {
        # The instance family also shares one base colour.  Identity is
        # carried only by the number of small visual marks.
        "unmarked": (0.20, 0.56, 0.78, 1.0),
        "single_marked": (0.20, 0.56, 0.78, 1.0),
        "double_marked": (0.20, 0.56, 0.78, 1.0),
        "marked": (0.20, 0.56, 0.78, 1.0),
        "single": (0.20, 0.56, 0.78, 1.0),
        "double": (0.20, 0.56, 0.78, 1.0),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # As with the texture family, attach marks to a canonical local
        # surface rather than guessing coordinates in a nested source mesh.
        # This makes one/two/no marks visually causal and removes the former
        # hovering-marker artifact.
        local_xml = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "external"
            / "vlabench"
            / "perception_instance_primitive.xml"
        )
        kwargs.setdefault("xml_path", str(local_xml))
        super().__init__(*args, **kwargs)

    def _apply_fallback_appearance(
        self, rgba: tuple[float, float, float, float]
    ) -> None:
        object_subtree = self.get_obj()
        token = str(getattr(self, "name", "")).lower()
        is_unmarked = "unmarked" in token
        is_double = "double" in token
        marker = next(
            (
                geom
                for geom in object_subtree.iter("geom")
                if geom.get("name", "").endswith("instance_mark")
            ),
            None,
        )
        if marker is None and not is_unmarked:
            marker = _append_visual_geom(
                object_subtree,
                name=f"{self.name}_instance_mark",
                geom_type="box",
                pos="0 0.044 0",
                size="0.010 0.003 0.010",
                rgba=(0.96, 0.90, 0.08, 1.0),
            )
        for geom in object_subtree.iter("geom"):
            if geom.get("group", "1") != "1":
                continue
            if geom.get("name", "").endswith("instance_mark"):
                marker = (0.96, 0.90, 0.08, 0.0 if is_unmarked else rgba[3])
                geom.set(
                    "rgba",
                    " ".join(f"{value:.8g}" for value in marker),
                )
                if not is_unmarked:
                    geom.set("pos", "-0.014 0 0.044")
                    geom.set("size", "0.010 0.010 0.002")
            else:
                geom.set("rgba", " ".join(f"{value:.8g}" for value in rgba))
        if is_double and not is_unmarked:
            # The clean-room fallback starts with one ordinary square mark.
            # Add a second mark only for the matched ``double`` candidate;
            # external VLABench instance meshes retain their own texture.
            marker_geom = next(
                (
                    geom
                    for geom in object_subtree.iter("geom")
                    if geom.get("name", "").endswith("instance_mark")
                ),
                None,
            )
            if marker_geom is not None:
                second = copy.deepcopy(marker_geom)
                second.set("name", f"{self.name}_instance_mark_2")
                second.set("pos", "0.014 0 0.044")
                object_subtree.append(second)


class VLABenchPerceptionBin(VLABenchContainer):
    """Receptacle with an external-basket adapter and local test fallback."""

    asset_key = "perception_bin"
    allow_primitive_fallback = True
    external_body_scale = 0.55
    external_mass_kg = 0.50

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        """Give source baskets/trays the neutral BDDL containment contract.

        The pinned basket XML exposes visual/collision geometry and placement
        points but no LIBERO ``contain_region`` site.  Add that site to the
        generated adapter XML only (never to the downloaded asset tree), so
        the normal ``In`` predicate remains available in external-asset mode.
        The repository fallback already declares the same site.
        """

        if is_fallback:
            return xml_path
        tree = ET.parse(xml_path)
        root = tree.getroot()
        object_body = root.find("./worldbody/body/body[@name='object']")
        if object_body is None:
            raise VLABenchAssetError(
                f"Converted perception bin has no object body: {xml_path}"
            )
        scale = float(self.external_body_scale)

        def scale_vec3(element: ET.Element, attribute: str) -> None:
            values = element.get(attribute, "").replace(",", " ").split()
            try:
                vector = tuple(float(value) for value in values)
            except ValueError:
                return
            if len(vector) != 3:
                return
            element.set(
                attribute,
                " ".join(f"{value * scale:.8g}" for value in vector),
            )

        # The selected VLABench basket is approximately 30 cm wide and only
        # 100 g after generic mass repair.  At LIBERO tabletop scale that
        # makes it both disproportionately large and easy for a held object
        # to shove away.  Normalize only the generated adapter to a 16.5 cm,
        # 500 g collection bin; the downloaded source asset stays untouched.
        asset = root.find("./asset")
        if asset is not None:
            for mesh in asset.findall("./mesh"):
                values = mesh.get("scale", "1 1 1").replace(",", " ").split()
                try:
                    previous = tuple(float(value) for value in values)
                except ValueError:
                    previous = (1.0, 1.0, 1.0)
                if len(previous) == 3:
                    mesh.set(
                        "scale",
                        " ".join(f"{value * scale:.8g}" for value in previous),
                    )
        for element in object_body.iter():
            if element.tag in {"body", "geom", "site", "joint", "inertial"}:
                scale_vec3(element, "pos")
            if element.tag == "geom":
                scale_vec3(element, "size")
        for geom in object_body.iter("geom"):
            try:
                mass = float(geom.get("mass", "0"))
            except ValueError:
                mass = 0.0
            if mass > 0.0:
                geom.set("mass", f"{self.external_mass_kg:.8g}")
                break

        if not any(
            site.get("name") == "contain_region"
            for site in object_body.iter("site")
        ):
            ET.SubElement(
                object_body,
                "site",
                {
                    "name": "contain_region",
                    "type": "box",
                    "pos": f"0 0 {0.150 * scale:.8g}",
                    "quat": "1 0 0 0",
                    "size": " ".join(
                        f"{value * scale:.8g}" for value in (0.110, 0.110, 0.120)
                    ),
                    "group": "3",
                    "rgba": "0 0 0 0",
                },
            )
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        return xml_path


class VLABenchNameTag(VLABenchXMLObject):
    """Visual-only VLABench NameTag source for arena presentation bodies.

    The regular condiment wrappers keep their labels in the movable object
    subtree for compatibility with ``MujocoXMLObject``.  The world-knowledge
    tabletop loader additionally instantiates the three classes below and puts
    their object bodies directly in the arena.  Those bodies have no joints,
    are not part of ``objects_dict`` / ``obj_of_interest``, and therefore do
    not participate in task checking or semantic segmentation mappings.
    """

    asset_key = "nametag"
    asset_category = "visual_support"


class VLABenchSugarNameTag(VLABenchNameTag):
    """World-frame sugar NameTag used by the condiment presentation."""

    visual_tag = "sugar"


class VLABenchSaltNameTag(VLABenchNameTag):
    """World-frame salt NameTag used by the condiment presentation."""

    visual_tag = "salt"


class VLABenchHotSauceNameTag(VLABenchNameTag):
    """World-frame hot-sauce NameTag used by the condiment presentation."""

    visual_tag = "hotsauce"


class VLABenchBasket(VLABenchContainer):
    asset_key = "basket"
    # The released basket MJCF uses collision-only fixture bodies without
    # explicit inertials.  LIBERO's merged MuJoCo model still requires every
    # static body to have a positive mass, so opt into the adapter's narrow
    # concrete-geom repair while keeping the asset's real collision geometry.
    fixture_mass_repair = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The basket asset's collision walls include a real base, but the
        # released XML only labels keypoints.  Add a named interior site for
        # BDDL ``In`` predicates without changing the downloaded asset.
        site_name = f"{self.name}_contain_region"
        attributes = {
            "name": site_name,
            "type": "box",
            "pos": "0 0 0.12",
            "quat": "1 0 0 0",
            "size": "0.115 0.115 0.12",
            "rgba": "0 0 0 0",
        }
        for parent in (
            self.get_obj(),
            self.worldbody.find("./body/body[@name='" + f"{self.name}_object" + "']"),
        ):
            if parent is None:
                continue
            if parent.find(f"./site[@name='{site_name}']") is None:
                parent.append(ET.Element("site", attributes))
        if "contain_region" not in self._sites:
            self._sites.append("contain_region")


class VLABenchTray(VLABenchContainer):
    asset_key = "tray"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The released tray mesh has keypoint sites but no semantic
        # receptacle site.  Add one adapter-level site to the derived object
        # subtree (and its unmerged source tree) so BDDL ``In`` predicates
        # and the private checkers can use the real tray geometry.  The
        # external XML itself is never modified.
        site_name = f"{self.name}_contain_region"
        attributes = {
            "name": site_name,
            "type": "box",
            "pos": "0 0 0.045",
            "quat": "1 0 0 0",
            "size": "0.145 0.185 0.045",
            "rgba": "0 0 0 0",
        }
        for parent in (
            self.get_obj(),
            self.worldbody.find("./body/body[@name='" + f"{self.name}_object" + "']"),
        ):
            if parent is None:
                continue
            if parent.find(f"./site[@name='{site_name}']") is None:
                parent.append(ET.Element("site", attributes))
        # ``MujocoModel.sites`` applies ``self.naming_prefix`` lazily.  Keep
        # the logical token here; storing the already-prefixed name would
        # produce ``tray_1_tray_1_contain_region`` during task merge.
        if "contain_region" not in self._sites:
            self._sites.append("contain_region")


class VLABenchCuttingBoard(VLABenchContainer):
    asset_key = "cutting_board"


class VLABenchGiftbox(VLABenchContainer):
    asset_key = "giftbox"


class VLABenchVase(VLABenchContainer):
    asset_key = "vase"


class VLABenchShelf(VLABenchContainer):
    asset_key = "shelf"
    fixture_mass_repair = True


class VLABenchShortCabinet(VLABenchOpenCloseMixin, VLABenchContainer):
    asset_key = "short_cabinet"
    interactive = True
    fixture_mass_repair = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._configure_open_close_contract()


class VLABenchWoodenCabinet(VLABenchOpenCloseMixin, VLABenchContainer):
    asset_key = "wooden_cabinet"
    interactive = True
    fixture_mass_repair = True
    open_direction = "lower"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._configure_open_close_contract()


class VLABenchMicrowave(VLABenchOpenCloseMixin, VLABenchContainer):
    asset_key = "microwave"
    interactive = True
    # ContainerWithDoor in the pinned VLABench source uses pi/3 and pi/10
    # state thresholds.  The source joint range is [0, 1.57], so these
    # fractions retain that contract without hard-coding a particular model
    # variant's rounded upper limit.
    open_fraction = 2.0 / 3.0
    close_fraction = 0.2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._configure_open_close_contract()


class VLABenchBilliardsTable(VLABenchContainer):
    asset_key = "billiards_table"


def _canonical_tableware_xml(
    xml_path: Path,
    *,
    suffix: str,
    bottom_z: float,
    top_z: float,
    horizontal_radius: float,
    inner_euler: str | None = None,
) -> Path:
    """Repair the pinned dining meshes' tabletop frame and placement sites.

    The VLABench source meshes use different authoring axes.  The generic
    converter cannot infer which thin axis is a plate normal or which long
    axis should run front-to-back, and its old generic bounds left the plate
    vertical and the fork partly embedded in the table.  This task-local
    adapter keeps the original meshes while publishing one audited tabletop
    frame for each item.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    outer = root.find("./worldbody/body")
    inner = root.find("./worldbody/body/body[@name='object']")
    if outer is None or inner is None:
        raise VLABenchAssetError(
            f"Converted tableware has no object wrapper: {xml_path}"
        )
    if inner_euler is not None:
        # The BDDL fixture sampler owns the outer body's pose and replaces
        # its quaternion at reset.  Put the asset-axis correction in a child
        # frame so sampling cannot silently erase it.
        correction = ET.Element(
            "body",
            {
                "name": f"canonical_{suffix}_frame",
                "euler": inner_euler,
            },
        )
        for child in list(inner):
            inner.remove(child)
            correction.append(child)
        inner.attrib.pop("quat", None)
        inner.attrib.pop("euler", None)
        inner.append(correction)

    site_values = {
        "bottom_site": f"0 0 {bottom_z:.8g}",
        "top_site": f"0 0 {top_z:.8g}",
        "horizontal_radius_site": f"{horizontal_radius:.8g} 0 0",
    }
    for site in outer.findall("./site"):
        name = site.get("name")
        if name in site_values:
            site.set("pos", site_values[name])

    # The source knife material is effectively black and the fork / plate are
    # very dark.  A neutral metallic / ceramic appearance makes the genuine
    # mesh silhouette legible without adding a semantic marker.
    rgba = "0.70 0.72 0.75 1" if suffix != "plate" else "0.82 0.82 0.80 1"
    for geom in inner.iter("geom"):
        if geom.get("group", "1") == "1":
            geom.attrib.pop("material", None)
            geom.set("rgba", rgba)

    # The released knife/fork meshes are physically valid but only a few
    # millimetres wide in the camera-facing direction.  At the benchmark's
    # 256 px wrist resolution that makes a correctly placed utensil collapse
    # to an ambiguous one-pixel line.  Add a collision-free, neutral silhouette
    # in the same local frame.  It changes neither mass, contacts, placement
    # sites, nor the terminal predicate; it only preserves the identity of a
    # thin tool in the public RGB observation.
    if suffix == "knife":
        # The source mesh has an embedded asset-axis rotation and its visual
        # centroid is about +9.8 cm in the adapter-local Y direction.
        visual_quat = "0.70710678 0.70710678 0 0"
        _append_visual_geom(
            inner,
            name="knife_wrist_handle_visual",
            geom_type="box",
            pos="0 0.098 -0.035",
            size="0.006 0.010 0.028",
            rgba=(0.32, 0.34, 0.37, 1.0),
            quat=visual_quat,
        )
        _append_visual_geom(
            inner,
            name="knife_wrist_blade_visual",
            geom_type="box",
            pos="0 0.098 0.030",
            size="0.008 0.013 0.052",
            rgba=(0.64, 0.66, 0.69, 1.0),
            quat=visual_quat,
        )
    elif suffix == "fork":
        # As with the knife, align the silhouette's long local Z axis with
        # table Y.  The fork mesh centroid is about (+2.6 cm, +0.3 cm) in
        # this adapter-local frame.
        # The fork's sampled outer body is rotated +90 degrees about world Z;
        # use the cyclic local frame that gives the same flat world frame as
        # the corrected physical mesh (rather than the knife's Rx(90) frame).
        visual_quat = "0.5 0.5 0.5 0.5"
        _append_visual_geom(
            inner,
            name="fork_wrist_handle_visual",
            geom_type="box",
            pos="0.026 0.003 -0.030",
            size="0.006 0.009 0.032",
            rgba=(0.38, 0.40, 0.43, 1.0),
            quat=visual_quat,
        )
        _append_visual_geom(
            inner,
            name="fork_wrist_head_visual",
            geom_type="box",
            pos="0.026 0.003 0.033",
            size="0.010 0.013 0.028",
            rgba=(0.64, 0.66, 0.69, 1.0),
            quat=visual_quat,
        )
        for index, local_x in enumerate((-0.0075, -0.0025, 0.0025, 0.0075)):
            _append_visual_geom(
                inner,
                name=f"fork_wrist_tine_{index}_visual",
                geom_type="box",
                pos=f"{local_x + 0.026:.6f} 0.003 0.073",
                size="0.0015 0.008 0.014",
                rgba=(0.64, 0.66, 0.69, 1.0),
                quat=visual_quat,
            )

    destination = xml_path.with_name(
        f"{xml_path.stem}-canonical-{suffix}{xml_path.suffix}"
    )
    # Formal sweeps construct the same task on several GPUs concurrently.
    # Publish the derived adapter atomically so one process cannot parse
    # another process's partially written XML.
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class VLABenchKnife(VLABenchTool):
    asset_key = "knife"

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        xml_path = super()._postprocess_xml_path(xml_path, is_fallback)
        if is_fallback:
            return xml_path
        return _canonical_tableware_xml(
            xml_path,
            suffix="knife",
            bottom_z=-0.0065,
            top_z=0.0195,
            horizontal_radius=0.133,
        )


class VLABenchSpoon(VLABenchTool):
    asset_key = "spoon"


class VLABenchFork(VLABenchTableware):
    asset_key = "fork"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # WorkspaceRegionSampler stores quaternion components in its historic
        # robosuite order.  This fixed x-axis value compiles to the audited
        # tabletop frame in the pinned runtime: the fork's long axis is along
        # table Y and its thin axis is vertical.
        self.rotation = (1.5707963267948966, 1.5707963267948966)
        self.rotation_axis = "x"

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        xml_path = super()._postprocess_xml_path(xml_path, is_fallback)
        if is_fallback:
            return xml_path
        return _canonical_tableware_xml(
            xml_path,
            suffix="fork",
            bottom_z=-0.024,
            top_z=0.024,
            horizontal_radius=0.104,
            # The pinned fork mesh is authored with an approximately five
            # degree roll around its long axis.  Keep the sampled outer pose
            # unchanged, but cancel that asset-local roll in a child frame so
            # the physical fork lies flat on the tabletop.
            inner_euler="-0.00011105 -0.0872948 -0.00923707",
        )


class VLABenchPlate(VLABenchTableware):
    asset_key = "plate"
    fixture_mass_repair = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The dining plate is an immobile fixture; its real VLABench mesh is
        # still merged into the MuJoCo scene and rendered normally.
        kwargs.setdefault("joints", None)
        super().__init__(*args, **kwargs)

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        xml_path = super()._postprocess_xml_path(xml_path, is_fallback)
        if is_fallback:
            return xml_path
        return _canonical_tableware_xml(
            xml_path,
            suffix="plate",
            # LIBERO's fixture sampler applies its standard -4 cm tabletop
            # z offset.  Encode that fixture convention in the placement
            # sites so the corrected plate surface rests at z=table_top.
            bottom_z=-0.0405,
            top_z=-0.0338,
            horizontal_radius=0.114,
            # The plate mesh's local X is its thin surface normal.
            inner_euler="1.5707963267948966 0 0",
        )


class VLABenchApple(VLABenchFruit):
    asset_key = "apple"


class VLABenchBanana(VLABenchBananaFruit):
    """Legacy banana alias with the same graspable physical proxy.

    ``cluster_series`` predates the semantic fruit task family and therefore
    requests ``vlabench_banana`` rather than ``vlabench_banana_fruit``.  Both
    names resolve the same upstream fruit shape and must share its conservative
    capsule collision envelope; otherwise the clustering task retains the
    original concave mesh that Panda cannot grasp reliably.
    """

    asset_key = "banana"


class VLABenchOrange(VLABenchFruit):
    asset_key = "orange"


class VLABenchLemon(VLABenchFruit):
    asset_key = "lemon"


class VLABenchHammer(VLABenchTool):
    asset_key = "hammer"


class VLABenchSeesaw(VLABenchOpenCloseMixin, VLABenchTool):
    """Interactive seesaw wrapper with external-or-primitive resolution."""

    asset_key = "seesaw"
    asset_category = "interactive"
    interactive = True
    allow_primitive_fallback = True
    fixture_mass_repair = True
    # Preserve the original fallback/external seesaw threshold contract:
    # range [0, 0.6] -> open [0.1, 0.6], close [0.0, 0.1].
    open_fraction = 1.0 / 6.0
    close_fraction = 1.0 / 6.0

    @staticmethod
    def _postprocess_external_xml(xml_path: Path) -> Path:
        """Name and anchor the downloaded seesaw without editing its source.

        The pinned XML intentionally leaves its hinge and placement site
        unnamed.  The generic converter gives them generated names, which is
        sufficient for robosuite but not for the task's state checker.  Add
        only the adapter contract (a stable hinge name and board-end sites)
        to a derived XML cache; the downloaded asset is never modified.
        """

        try:
            tree = ET.parse(xml_path)
        except (OSError, ET.ParseError):
            return xml_path
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            return xml_path

        hinge_parent: ET.Element | None = None
        hinge: ET.Element | None = None

        def visit(parent: ET.Element) -> None:
            nonlocal hinge_parent, hinge
            for child in parent:
                if child.tag == "joint" and child.get("type", "hinge") == "hinge":
                    hinge_parent, hinge = parent, child
                    return
                visit(child)
                if hinge is not None:
                    return

        visit(worldbody)
        if hinge is None or hinge_parent is None:
            return xml_path

        changed = False
        if hinge.get("name") != "seesaw_hinge":
            hinge.set("name", "seesaw_hinge")
            changed = True
        # The source hinge is damped but has no restoring torque. Given enough
        # time, any single non-zero block therefore drives it to the joint
        # limit, which makes a two-counterweight task physically meaningless.
        # A modest neutral spring creates a stable intermediate response for
        # one block while two blocks still expose the payload completely.
        for attribute, value in (
            ("damping", "1"),
            ("stiffness", "0.6"),
            ("springref", "0"),
        ):
            if hinge.get(attribute) != value:
                hinge.set(attribute, value)
                changed = True

        existing_sites = {site.get("name") for site in root.iter("site")}
        for name, position in (
            ("seesaw_left_region", "-0.24 0 0.006"),
            ("seesaw_right_region", "0.24 0 0.006"),
        ):
            if name in existing_sites:
                continue
            ET.SubElement(
                hinge_parent,
                "site",
                {
                    "name": name,
                    "type": "box",
                    "pos": position,
                    "size": "0.055 0.040 0.012",
                    "rgba": "0 0 0 0",
                },
            )
            changed = True

        if not changed:
            return xml_path

        destination = xml_path.with_name(
            f"{xml_path.stem}-seesaw-contract{xml_path.suffix}"
        )
        # Re-publish the tiny derived contract on every instantiation.  This
        # also upgrades caches produced before the MuJoCo-3 ``autolimits``
        # compiler compatibility attribute was added, while leaving the
        # downloaded source tree untouched.
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            tree.write(temporary, encoding="utf-8", xml_declaration=True)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        if is_fallback:
            return xml_path
        return self._postprocess_external_xml(xml_path)

    @property
    def horizontal_radius(self) -> float:
        """Return the support footprint used by LIBERO placement samplers.

        The upstream asset advertises a radius covering the full board.  That
        conservative radius is useful for a free-standing fixture, but would
        reject the deliberate target/weight placements on the board before
        MuJoCo can simulate contact.  The base/support footprint is the
        relevant exclusion radius here; board occupancy remains governed by
        collision geometry and the private physical checker.
        """

        # ``0`` lets conditioned BDDL samplers intentionally place a movable
        # body on the articulated board.  Collisions, rather than this broad
        # placement heuristic, remain authoritative for the actual contact.
        return 0.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._configure_open_close_contract()

    @property
    def hinge_joint_name(self) -> str | None:
        """Return the prefixed hinge joint name, if the XML defines one."""

        articulation = self.object_properties.get("articulation", {})
        names = articulation.get("joint_names", [])
        return str(names[0]) if names else None

    @staticmethod
    def is_tilted(qpos: float, threshold: float = 0.1) -> bool:
        """Small task-side predicate useful for a seesaw interaction checker."""

        return abs(float(qpos)) >= float(threshold)


class VLABenchSeesawWeight(VLABenchXMLObject):
    """Task-local counterweight used by the physical seesaw benchmark.

    VLABench's ``simple_seesaw_use`` samples ``RandomGeom`` counterweights
    inside the upstream task generator.  LIBERO BDDL needs a registered,
    movable XML object instead, so this clean-room primitive keeps the
    manipulation contract deterministic while the mechanism itself remains
    the VLABench seesaw asset.
    """

    asset_key = "seesaw_weight"
    asset_category = "counterweight"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from pathlib import Path

        local_xml = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "external"
            / "vlabench"
            / "seesaw_weight.xml"
        )
        kwargs.setdefault("xml_path", str(local_xml))
        super().__init__(*args, **kwargs)


class _VLABenchSeesawMassWeight(VLABenchXMLObject):
    """Neutral task-local weight whose only hidden difference is inertial mass.

    The three XMLs intentionally share the same collision and visual envelope.
    A checker may therefore use the live ``body_mass`` and contact response,
    while an agent must infer the useful configuration from interaction.  The
    XMLs are repository-owned support fixtures; they are not copied into the
    external VLABench download tree.
    """

    asset_category = "counterweight"
    _variant_filename: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not self._variant_filename:
            raise RuntimeError("mass-weight wrapper did not select a variant")
        local_xml = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "external"
            / "vlabench"
            / self._variant_filename
        )
        kwargs.setdefault("xml_path", str(local_xml))
        super().__init__(*args, **kwargs)


class VLABenchSeesawWeightLight(_VLABenchSeesawMassWeight):
    """Light neutral weight used by adaptive and comparison tasks."""

    asset_key = "seesaw_weight_light"
    _variant_filename = "seesaw_weight_light.xml"


class VLABenchSeesawWeightMedium(_VLABenchSeesawMassWeight):
    """Medium neutral weight used by the adaptive configuration task."""

    asset_key = "seesaw_weight_medium"
    _variant_filename = "seesaw_weight_medium.xml"


class VLABenchSeesawWeightHeavy(_VLABenchSeesawMassWeight):
    """Heavy neutral weight used by adaptive and comparison tasks."""

    asset_key = "seesaw_weight_heavy"
    _variant_filename = "seesaw_weight_heavy.xml"


class VLABenchDensityButton(VLABenchXMLObject):
    """LIBERO wrapper for the primitive VLABench physical-QA Button.

    The upstream entity is a fixed cylinder with a touch sensor.  This wrapper
    retains that geometry and candidate appearance/inertial mass
    are selected by the three task aliases below; the public instruction does
    not name any alias.
    """

    asset_key = "density_button"
    asset_category = "interactive"
    _variant_rgba = "0.35 0.35 0.35 1"
    _variant_mass = "0.08"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        local_xml = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "external"
            / "vlabench"
            / "vlabench_density_button.xml"
        )
        kwargs.setdefault("xml_path", str(local_xml))
        # BDDL declares these as fixtures.  Preserve the fixed button assembly
        # from the XML; do not add a free joint around it.
        kwargs.setdefault("joints", None)
        super().__init__(*args, **kwargs)
        self.object_properties["button_variant"] = {
            "material": self._variant_name,
            "density_rank_private": self._density_rank,
        }

    @property
    def _variant_name(self) -> str:
        return "generic"

    @property
    def _density_rank(self) -> int:
        return 0

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        """Write a tiny derived XML with the selected candidate appearance."""

        if is_fallback:
            return xml_path
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for geom in root.iter("geom"):
            if geom.get("name") == "candidate_visual":
                geom.set("rgba", self._variant_rgba)
            elif geom.get("name") == "candidate_collision":
                geom.set("mass", self._variant_mass)
        destination = xml_path.with_name(
            f"{xml_path.stem}-{self._variant_name}{xml_path.suffix}"
        )
        tree.write(destination, encoding="utf-8", xml_declaration=True)
        return destination


class VLABenchDensityButtonWood(VLABenchDensityButton):
    _variant_rgba = "0.55 0.27 0.08 1"
    _variant_mass = "0.04"

    @property
    def _variant_name(self) -> str:
        return "wood"

    @property
    def _density_rank(self) -> int:
        return 0


class VLABenchDensityButtonRubber(VLABenchDensityButton):
    _variant_rgba = "0.10 0.10 0.10 1"
    _variant_mass = "0.09"

    @property
    def _variant_name(self) -> str:
        return "rubber"

    @property
    def _density_rank(self) -> int:
        return 1


class VLABenchDensityButtonMetal(VLABenchDensityButton):
    _variant_rgba = "0.65 0.68 0.72 1"
    _variant_mass = "0.16"

    @property
    def _variant_name(self) -> str:
        return "metal"

    @property
    def _density_rank(self) -> int:
        return 2


class VLABenchSingleButton(VLABenchXMLObject):
    """One VLABench-style touch button without selection distractors."""

    asset_key = "single_button"
    asset_category = "interactive"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        local_xml = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "external"
            / "vlabench"
            / "vlabench_single_button.xml"
        )
        kwargs.setdefault("xml_path", str(local_xml))
        kwargs.setdefault("joints", None)
        super().__init__(*args, **kwargs)


class VLABenchPhysicalPropertyButton(VLABenchXMLObject):
    """Fixed button assembly with a family-specific visible candidate cue.

    Every candidate uses the same inertial envelope and the same contact
    mechanics.  Differences in the pure question bank are world-knowledge
    facts, not simulated experiments.  This class only changes the visible
    color/shape so the three candidates are distinguishable in rendered
    images; it never stores a target or property answer in public metadata.
    """

    asset_key = "physical_property_button"
    asset_category = "interactive"
    _category_name: str | None = None
    _variant_key = ""

    # (short variant name, RGBA, visual geom type, visual geom size)
    _VARIANTS: dict[str, tuple[str, str, str, str]] = {
        "vlabench_property_weight_sphere": (
            "weight_sphere",
            "0.78 0.10 0.08 1",
            "sphere",
            "0.026",
        ),
        "vlabench_property_weight_cylinder": (
            "weight_cylinder",
            "0.78 0.10 0.08 1",
            "cylinder",
            "0.027 0.031",
        ),
        "vlabench_property_weight_cube": (
            "weight_cube",
            "0.78 0.10 0.08 1",
            "box",
            "0.030 0.030 0.030",
        ),
        "vlabench_property_size_small_cylinder": (
            "size_small_cylinder",
            "0.18 0.48 0.82 1",
            "cylinder",
            "0.018 0.022",
        ),
        "vlabench_property_size_medium_cylinder": (
            "size_medium_cylinder",
            "0.18 0.48 0.82 1",
            "cylinder",
            "0.024 0.030",
        ),
        "vlabench_property_size_large_cylinder": (
            "size_large_cylinder",
            "0.18 0.48 0.82 1",
            "cylinder",
            "0.030 0.038",
        ),
        "vlabench_property_magnetism_wood": (
            "magnetism_wood",
            "0.55 0.27 0.08 1",
            "cylinder",
            "0.024 0.026",
        ),
        "vlabench_property_magnetism_steel": (
            "magnetism_low_carbon_steel",
            "0.70 0.73 0.78 1",
            "cylinder",
            "0.024 0.026",
        ),
        "vlabench_property_magnetism_glass": (
            "magnetism_glass",
            "0.20 0.62 0.92 1",
            "cylinder",
            "0.024 0.026",
        ),
        "vlabench_property_density_wood": (
            "density_wood",
            "0.55 0.27 0.08 1",
            "box",
            "0.026 0.026 0.026",
        ),
        "vlabench_property_density_rubber": (
            "density_rubber",
            "0.08 0.08 0.08 1",
            "box",
            "0.026 0.026 0.026",
        ),
        "vlabench_property_density_glass": (
            "density_glass",
            "0.20 0.62 0.92 1",
            "box",
            "0.026 0.026 0.026",
        ),
        "vlabench_property_density_metal": (
            "density_steel",
            "0.82 0.84 0.88 1",
            "box",
            "0.026 0.026 0.026",
        ),
        "vlabench_property_friction_polished_metal": (
            "friction_white_ptfe",
            "0.92 0.92 0.90 1",
            "cylinder",
            "0.024 0.026",
        ),
        "vlabench_property_friction_unfinished_wood": (
            "friction_unfinished_wood",
            "0.55 0.27 0.08 1",
            "cylinder",
            "0.024 0.026",
        ),
        "vlabench_property_friction_rubber": (
            "friction_rubber",
            "0.08 0.08 0.08 1",
            "cylinder",
            "0.024 0.026",
        ),
        "vlabench_property_thermal_fused_silica": (
            "thermal_fused_silica",
            "0.93 0.97 1.00 0.42",
            "box",
            "0.017 0.017 0.034",
        ),
        "vlabench_property_thermal_steel": (
            "thermal_steel",
            "0.28 0.30 0.33 1",
            "box",
            "0.017 0.017 0.034",
        ),
        "vlabench_property_thermal_aluminum": (
            "thermal_aluminum",
            "0.82 0.84 0.88 1",
            "box",
            "0.017 0.017 0.034",
        ),
        "vlabench_property_sound_rubber": (
            "sound_rubber",
            "0.08 0.08 0.08 1",
            "cylinder",
            "0.024 0.026",
        ),
        "vlabench_property_sound_glass": (
            "sound_glass",
            "0.20 0.62 0.92 1",
            "cylinder",
            "0.024 0.026",
        ),
        "vlabench_property_sound_steel": (
            "sound_steel",
            "0.70 0.73 0.78 1",
            "cylinder",
            "0.024 0.026",
        ),
        "vlabench_property_reflection_matte_gray": (
            "reflection_black_rubber",
            "0.055 0.055 0.06 1",
            "box",
            "0.035 0.030 0.006",
        ),
        "vlabench_property_reflection_satin_gray": (
            "reflection_unfinished_wood",
            "0.55 0.27 0.08 1",
            "box",
            "0.035 0.030 0.006",
        ),
        "vlabench_property_reflection_polished_silver": (
            "reflection_polished_steel",
            "0.88 0.90 0.95 1",
            "box",
            "0.035 0.030 0.006",
        ),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        category = str(
            self._category_name or kwargs.get("obj_name") or self.asset_key
        ).lower()
        if category not in self._VARIANTS:
            raise ValueError(f"unknown physical-property button category: {category}")
        self._variant_key = category
        local_xml = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "external"
            / "vlabench"
            / "vlabench_physical_property_button.xml"
        )
        kwargs.setdefault("xml_path", str(local_xml))
        kwargs.setdefault("joints", None)
        super().__init__(*args, **kwargs)
        # This is diagnostic host metadata only.  It contains no answer and is
        # never copied into the agent observation (object observations are off
        # for the benchmark factory).
        self.object_properties["button_variant"] = self._VARIANTS[category][0]

    def _postprocess_xml_path(self, xml_path: Path, is_fallback: bool) -> Path:
        if is_fallback:
            return xml_path
        variant_name, rgba, geom_type, size = self._VARIANTS[self._variant_key]
        tree = ET.parse(xml_path)
        root = tree.getroot()
        asset = root.find("./asset")
        if asset is None:
            asset = ET.Element("asset")
            compiler = root.find("./compiler")
            root.insert(1 if compiler is not None else 0, asset)

        material_kind = "neutral"
        if variant_name.startswith("weight_"):
            material_kind = "painted_wood"
        elif "ptfe" in variant_name:
            material_kind = "ptfe"
        elif "wood" in variant_name:
            material_kind = "wood"
        elif "rubber" in variant_name:
            material_kind = "rubber"
        elif "fused_silica" in variant_name:
            material_kind = "fused_silica"
        elif "glass" in variant_name or "silica" in variant_name:
            material_kind = "glass"
        elif any(
            token in variant_name
            for token in ("metal", "steel", "aluminum", "silver")
        ):
            material_kind = "metal"
        elif "matte" in variant_name:
            material_kind = "matte"
        elif "satin" in variant_name:
            material_kind = "satin"

        material_name = f"candidate_{variant_name}_material"
        material_attributes = {
            "name": material_name,
            "rgba": rgba,
            "specular": "0.20",
            "shininess": "0.20",
            "reflectance": "0.05",
        }
        if material_kind == "wood":
            texture_name = f"candidate_{variant_name}_texture"
            ET.SubElement(
                asset,
                "texture",
                {
                    "name": texture_name,
                    "type": "2d",
                    "builtin": "checker",
                    "rgb1": "0.30 0.10 0.025",
                    "rgb2": "0.72 0.42 0.12",
                    "width": "64",
                    "height": "64",
                },
            )
            material_attributes.update(
                {
                    "texture": texture_name,
                    "texrepeat": "3 3",
                    "texuniform": "true",
                    "specular": "0.02",
                    "shininess": "0.02",
                    "reflectance": "0",
                }
            )
        elif material_kind == "painted_wood":
            material_attributes.update(
                {"rgba": rgba, "specular": "0.08", "shininess": "0.12", "reflectance": "0.01"}
            )
        elif material_kind == "ptfe":
            material_attributes.update(
                {"rgba": rgba, "specular": "0.08", "shininess": "0.08", "reflectance": "0.02"}
            )
        elif material_kind == "rubber":
            material_attributes.update(
                {"rgba": "0.055 0.055 0.06 1", "specular": "0.02", "shininess": "0.02", "reflectance": "0"}
            )
        elif material_kind == "glass":
            material_attributes.update(
                {"rgba": "0.30 0.72 0.94 0.68", "specular": "0.85", "shininess": "0.90", "reflectance": "0.18"}
            )
        elif material_kind == "fused_silica":
            material_attributes.update(
                {"rgba": "0.93 0.97 1.00 0.42", "specular": "0.75", "shininess": "0.85", "reflectance": "0.10"}
            )
        elif material_kind == "metal":
            material_attributes.update(
                {"rgba": rgba, "specular": "0.80", "shininess": "0.85", "reflectance": "0.38"}
            )
        elif material_kind == "matte":
            material_attributes.update(
                {"specular": "0.01", "shininess": "0.01", "reflectance": "0"}
            )
        elif material_kind == "satin":
            material_attributes.update(
                {"specular": "0.35", "shininess": "0.45", "reflectance": "0.12"}
            )
        ET.SubElement(asset, "material", material_attributes)

        size_values = tuple(float(value) for value in size.split())
        if geom_type == "sphere":
            half_height = size_values[0]
        elif geom_type == "cylinder":
            half_height = size_values[1]
        else:
            half_height = size_values[2]
        # LIBERO places tabletop fixtures with their root 4 cm below the
        # visible table surface.  The old 8 mm support-plane assumption left
        # the sphere, cylinder, and box cues roughly half buried.  Put the
        # bottom of every non-reflectance visual exactly on the tabletop; the
        # reflection panels already use the equivalent explicit 0.046 value.
        candidate_z = 0.040 + half_height
        for geom in root.iter("geom"):
            if geom.get("name") == "candidate_visual":
                geom.set("type", geom_type)
                geom.set("size", size)
                # The candidate is on the far side of the shared base and
                # the red button is on the robot-facing side.  Keeping this
                # axis explicit makes the pair readable in both cameras and
                # avoids a perspective overlap in the head view.
                if variant_name.startswith("reflection_"):
                    # Keep all reflectance samples horizontal and coplanar.
                    # Place them farther from the tall button so both public
                    # cameras can see the top face without perspective
                    # occlusion.
                    # The fixture origin sits 4 cm below the table surface in
                    # this scene.  A centre height of 4.6 cm therefore rests
                    # the 6 mm half-height panel on the tabletop instead of
                    # burying it inside the table mesh.
                    geom.set("pos", "0.077 0 0.046")
                    geom.attrib.pop("euler", None)
                else:
                    geom.set("pos", f"0.045 0 {candidate_z:.8g}")
                    geom.attrib.pop("euler", None)
                geom.attrib.pop("rgba", None)
                geom.set("material", material_name)
            elif geom.get("name") == "candidate_collision":
                # Keep all variants mechanically identical: this family is a
                # knowledge question, not a dynamics experiment.
                geom.set("mass", "0.08")
                geom.set("pos", "0.045 0 0.032")
            elif (
                geom.get("name") == "pairing_plate"
                and variant_name.startswith("reflection_")
            ):
                geom.set("size", "0.130 0.045 0.004")
        destination = xml_path.with_name(
            f"{xml_path.stem}-{variant_name}{xml_path.suffix}"
        )
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            tree.write(temporary, encoding="utf-8", xml_declaration=True)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination


VLABENCH_OBJECTS = {
    "vlabench_bbq_sauce": VLABenchBBQSauce,
    "vlabench_hotsauce": VLABenchHotSauce,
    "vlabench_ketchup": VLABenchKetchup,
    "vlabench_salad_dressing": VLABenchSaladDressing,
    "vlabench_shaker": VLABenchShaker,
    "vlabench_sugar": VLABenchSugarShaker,
    "vlabench_salt": VLABenchSaltShaker,
    "vlabench_nametag": VLABenchNameTag,
    "vlabench_basket": VLABenchBasket,
    "vlabench_tray": VLABenchTray,
    "vlabench_cutting_board": VLABenchCuttingBoard,
    "vlabench_giftbox": VLABenchGiftbox,
    "vlabench_vase": VLABenchVase,
    "vlabench_shelf": VLABenchShelf,
    "vlabench_short_cabinet": VLABenchShortCabinet,
    "vlabench_wooden_cabinet": VLABenchWoodenCabinet,
    "vlabench_microwave": VLABenchMicrowave,
    "vlabench_billiards_table": VLABenchBilliardsTable,
    "vlabench_knife": VLABenchKnife,
    "vlabench_spoon": VLABenchSpoon,
    "vlabench_fork": VLABenchFork,
    "vlabench_plate": VLABenchPlate,
    "vlabench_apple": VLABenchApple,
    "vlabench_banana": VLABenchBanana,
    "vlabench_orange": VLABenchOrange,
    "vlabench_lemon": VLABenchLemon,
    "vlabench_hammer": VLABenchHammer,
    "vlabench_seesaw": VLABenchSeesaw,
    "vlabench_tube": VLABenchTube,
    "vlabench_tube_stand": VLABenchTubeStand,
    "vlabench_fruit": VLABenchFruit,
    "vlabench_orange_fruit": VLABenchOrangeFruit,
    "vlabench_apple_fruit": VLABenchAppleFruit,
    "vlabench_banana_fruit": VLABenchBananaFruit,
    "vlabench_drink": VLABenchDrink,
    "vlabench_milk_drink": VLABenchMilkDrink,
    "vlabench_cola_drink": VLABenchColaDrink,
    "vlabench_juice_drink": VLABenchJuiceDrink,
    "vlabench_bloom_flower": VLABenchBloomFlower,
    "vlabench_rose_flower": VLABenchRoseFlower,
    "vlabench_sunflower_flower": VLABenchSunflower,
    "vlabench_daisy_flower": VLABenchDaisyFlower,
    "vlabench_peony_flower": VLABenchPeonyFlower,
    "vlabench_chrysanthemum_flower": VLABenchChrysanthemumFlower,
    "vlabench_tulip_flower": VLABenchTulipFlower,
    "vlabench_wilted_flower": VLABenchWiltedFlower,
    "vlabench_chemistry_solution": VLABenchChemistrySolution,
    "vlabench_chemistry_blue": VLABenchBlueChemistrySolution,
    "vlabench_chemistry_green": VLABenchGreenChemistrySolution,
    "vlabench_chemistry_purple": VLABenchPurpleChemistrySolution,
    "vlabench_perception_token": VLABenchPerceptionToken,
    "vlabench_perception_color": VLABenchPerceptionColorToken,
    "vlabench_perception_texture": VLABenchPerceptionTextureToken,
    "vlabench_perception_instance": VLABenchPerceptionInstanceToken,
    "vlabench_perception_bin": VLABenchPerceptionBin,
}

# Task-local aliases and clean-room support objects are registered alongside,
# but kept out of the provenance manifest mapping above.  Alias categories
# let BDDL keep each candidate on its own line (the pinned BDDL parser stores
# one object list per category) while still resolving to the same source mesh
# where a matched perception task intentionally requires it.
VLABENCH_PROPERTY_BUTTON_CATEGORIES = tuple(
    VLABenchPhysicalPropertyButton._VARIANTS
)

VLABENCH_PROPERTY_BUTTON_CLASSES = {
    category: type(
        "VLABenchPropertyButton_" + category.removeprefix("vlabench_property_"),
        (VLABenchPhysicalPropertyButton,),
        {"_category_name": category},
    )
    for category in VLABENCH_PROPERTY_BUTTON_CATEGORIES
}

VLABENCH_TASK_OBJECTS = {
    # Perception-only condiment aliases intentionally do not match the three
    # source categories that trigger detached world-frame NameTags in the
    # tabletop domain.  They retain the exact same source meshes and object
    # names, while keeping the spatial "between" question free of text cues.
    "vlabench_perception_sugar_unlabeled": VLABenchSugarShaker,
    "vlabench_perception_salt_unlabeled": VLABenchSaltShaker,
    "vlabench_perception_hotsauce_unlabeled": VLABenchHotSauce,
    "vlabench_seesaw_weight": VLABenchSeesawWeight,
    "vlabench_seesaw_weight_light": VLABenchSeesawWeightLight,
    "vlabench_seesaw_weight_medium": VLABenchSeesawWeightMedium,
    "vlabench_seesaw_weight_heavy": VLABenchSeesawWeightHeavy,
    "vlabench_perception_upright_token": VLABenchPerceptionToken,
    "vlabench_perception_fallen_token": VLABenchPerceptionToken,
    "vlabench_perception_fallen_token_2": VLABenchPerceptionToken,
    "vlabench_wilted_flower_2": VLABenchWiltedFlower,
    "vlabench_perception_red_color": VLABenchPerceptionColorToken,
    "vlabench_perception_blue_color": VLABenchPerceptionColorToken,
    "vlabench_perception_green_color": VLABenchPerceptionColorToken,
    "vlabench_perception_striped_texture": VLABenchPerceptionTextureToken,
    "vlabench_perception_spotted_texture": VLABenchPerceptionTextureToken,
    "vlabench_perception_plain_texture": VLABenchPerceptionTextureToken,
    "vlabench_perception_single_marked_instance": VLABenchPerceptionInstanceToken,
    "vlabench_perception_double_marked_instance": VLABenchPerceptionInstanceToken,
    "vlabench_perception_unmarked_instance": VLABenchPerceptionInstanceToken,
    "vlabench_rearrangement_tube_stand": VLABenchRearrangementTubeStand,
    "vlabench_density_button_wood": VLABenchDensityButtonWood,
    "vlabench_density_button_rubber": VLABenchDensityButtonRubber,
    "vlabench_density_button_metal": VLABenchDensityButtonMetal,
    "vlabench_single_button": VLABenchSingleButton,
    **{
        category: target_class
        for category, target_class in VLABENCH_PROPERTY_BUTTON_CLASSES.items()
    },
}


def register_vlabench_objects(*, overwrite: bool = False) -> dict[str, type[VLABenchXMLObject]]:
    """Register only namespaced VLABench object categories.

    The function is idempotent to make interactive notebooks safe.  Existing
    LIBERO names are protected unless ``overwrite=True`` is explicitly used
    by a caller (the package import never does so).
    """

    for key, target_class in {
        **VLABENCH_OBJECTS,
        **VLABENCH_TASK_OBJECTS,
    }.items():
        if overwrite or key not in OBJECTS_DICT:
            OBJECTS_DICT[key] = target_class
    return {key: OBJECTS_DICT[key] for key in VLABENCH_OBJECTS}


def get_vlabench_object_fn(category_name: str) -> type[VLABenchXMLObject]:
    """Resolve a namespaced VLABench category."""

    key = category_name.lower()
    register_vlabench_objects()
    registered = {**VLABENCH_OBJECTS, **VLABENCH_TASK_OBJECTS}
    if key not in registered:
        raise KeyError(f"Unknown VLABench object category: {category_name!r}")
    return registered[key]


register_vlabench_objects()


__all__ = [
    "VLABENCH_OBJECTS",
    "VLABENCH_TASK_OBJECTS",
    "VLABenchBBQSauce",
    "VLABenchBasket",
    "VLABenchBilliardsTable",
    "VLABenchCondiment",
    "VLABenchContainer",
    "VLABenchFruit",
    "VLABenchCuttingBoard",
    "VLABenchChemistrySolution",
    "VLABenchBlueChemistrySolution",
    "VLABenchGreenChemistrySolution",
    "VLABenchPurpleChemistrySolution",
    "VLABenchAppleFruit",
    "VLABenchBananaFruit",
    "VLABenchOrangeFruit",
    "VLABenchDrink",
    "VLABenchMilkDrink",
    "VLABenchColaDrink",
    "VLABenchJuiceDrink",
    "VLABenchFruit",
    "VLABenchGiftbox",
    "VLABenchHammer",
    "VLABenchHotSauce",
    "VLABenchKetchup",
    "VLABenchKnife",
    "VLABenchFork",
    "VLABenchPlate",
    "VLABenchApple",
    "VLABenchBanana",
    "VLABenchOrange",
    "VLABenchLemon",
    "VLABenchMicrowave",
    "VLABenchOpenCloseMixin",
    "VLABenchAppearanceMixin",
    "VLABenchFlowerAssetMixin",
    "VLABenchBloomFlower",
    "VLABenchRoseFlower",
    "VLABenchSunflower",
    "VLABenchDaisyFlower",
    "VLABenchPeonyFlower",
    "VLABenchChrysanthemumFlower",
    "VLABenchTulipFlower",
    "VLABenchWiltedFlower",
    "VLABenchPerceptionBin",
    "VLABenchPerceptionColorToken",
    "VLABenchPerceptionInstanceToken",
    "VLABenchPerceptionTextureToken",
    "VLABenchPerceptionToken",
    "VLABenchSaladDressing",
    "VLABenchSeesaw",
    "VLABenchSeesawWeight",
    "VLABenchSeesawWeightHeavy",
    "VLABenchSeesawWeightLight",
    "VLABenchSeesawWeightMedium",
    "VLABenchShelf",
    "VLABenchSaltShaker",
    "VLABenchNameTag",
    "VLABenchSugarNameTag",
    "VLABenchSaltNameTag",
    "VLABenchHotSauceNameTag",
    "VLABenchShaker",
    "VLABenchShortCabinet",
    "VLABenchSpoon",
    "VLABenchSugarShaker",
    "VLABenchTool",
    "VLABenchTableware",
    "VLABenchTube",
    "VLABenchTubeStand",
    "VLABenchRearrangementTubeStand",
    "VLABenchTray",
    "VLABenchVase",
    "VLABenchWoodenCabinet",
    "VLABenchXMLObject",
    "VLABenchPhysicalPropertyButton",
    "get_vlabench_object_fn",
    "register_vlabench_objects",
]
