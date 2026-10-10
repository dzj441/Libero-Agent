import numpy as np

from robosuite.utils.mjcf_utils import new_site
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange


@register_problem
class Libero_Tabletop_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "main_table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_tabletop_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "light-gray",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Load BDDL fixtures and optional world-frame condiment labels.

        VLABench's upstream ``NameTag`` entities are detached from each
        condiment and attached to the arena after task construction.  LIBERO's
        normal ``MujocoXMLObject`` merge keeps an object's descendants under
        its free joint, so doing the same thing in an object XML would make a
        label rotate with the bottle.  For the condiment family we therefore
        merge sugar, salt, and hot-sauce NameTag bodies directly into the arena
        as mocap bodies.  They are visual support only: they are not in
        ``fixtures_dict``, ``objects_dict``, ``obj_of_interest``, or
        ``ManipulationTask.mujoco_objects`` and cannot affect the checker or
        semantic instance/mask mappings.
        """
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "table":
                continue

            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

        self._vlabench_world_tag_specs = []
        object_categories = set(self.parsed_problem["objects"])
        if {
            "vlabench_sugar",
            "vlabench_salt",
            "vlabench_hotsauce",
        }.issubset(object_categories):
            for content, object_name, tag_class in (
                ("sugar", "sugar_1", VLABenchSugarNameTag),
                ("salt", "salt_1", VLABenchSaltNameTag),
                ("hotsauce", "hotsauce_1", VLABenchHotSauceNameTag),
            ):
                tag = tag_class(
                    name=f"vlabench_{content}_world_label",
                    joints=None,
                )
                tag_body = tag.get_obj()
                tag_body_name = f"vlabench_{content}_world_label_body"
                tag_body.set("name", tag_body_name)
                tag_body.set("mocap", "true")
                tag_body.set("pos", "0 0 0")
                mujoco_arena.merge_assets(tag)
                mujoco_arena.worldbody.append(tag_body)
                self._vlabench_world_tag_specs.append(
                    (content, object_name, tag_body_name)
                )

    def _sync_vlabench_world_tags(self) -> None:
        """Place visual NameTags at the current initial object poses.

        This hook runs after ordinary reset placement and after a launcher
        calls ``set_init_state``.  Once positioned, mocap bodies are not
        updated during manipulation, so moving or rotating a condiment leaves
        its presentation tag fixed in the world frame exactly like upstream
        VLABench's post-build ``detach(); arena.attach(...)`` sequence.
        """

        specs = getattr(self, "_vlabench_world_tag_specs", ())
        if not specs or not hasattr(self, "sim"):
            return
        # The BDDL placement sampler writes free-joint qpos immediately before
        # this hook, without a final forward pass.  Refresh xpos first so the
        # detached tags follow the selected initial state rather than the
        # previous model state (often the origin during a hard reset).
        self.sim.forward()
        # Keep one canonical world pose for every observer.  The source mesh's
        # front is along local +y; this fixed +90 degree world-z rotation keeps
        # the text upright while the adapter supplies a separate back print.
        # No camera-dependent billboard or per-view orientation is involved.
        # Use one world-frame height for the three cards.  The bottle meshes
        # have different root origins (the hot-sauce bottle is taller), so a
        # fixed per-object Z offset makes the labels visibly staggered.  The
        # common height is just above the tabletop and is independent of the
        # selected condiment geometry.
        table_top = float(self.table_offset[2] + self.table_full_size[2] / 2.0)
        presentation_height = table_top + 0.1525
        presentation_quat = np.asarray(
            (0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
            dtype=np.float64,
        )
        for _content, object_name, tag_body_name in specs:
            body_id = int(self.sim.model.body_name2id(tag_body_name))
            mocap_id = int(self.sim.model.body_mocapid[body_id])
            if mocap_id < 0 or object_name not in self.obj_body_id:
                continue
            object_body_id = int(self.obj_body_id[object_name])
            object_position = np.asarray(
                self.sim.data.xpos[object_body_id], dtype=np.float64
            )
            self.sim.data.mocap_pos[mocap_id] = (
                object_position[0], object_position[1], presentation_height
            )
            self.sim.data.mocap_quat[mocap_id] = presentation_quat
        self.sim.forward()

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        has_world_tags = bool(getattr(self, "_vlabench_world_tag_specs", ()))
        # The perception-only condiment aliases deliberately use the same
        # source meshes as the world-knowledge shakers, but must not expose
        # their semantic NameTags.  Passing ``visual_tag=None`` explicitly is
        # important here: the alias classes inherit ``visual_tag`` from
        # ``VLABenchSugarShaker`` / ``VLABenchSaltShaker`` and otherwise the
        # label is still nested under the movable object.
        unlabeled_categories = {
            "vlabench_perception_sugar_unlabeled",
            "vlabench_perception_salt_unlabeled",
            "vlabench_perception_hotsauce_unlabeled",
        }
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                object_kwargs = {"name": object_name}
                if category_name in unlabeled_categories or (
                    has_world_tags
                    and category_name in {"vlabench_sugar", "vlabench_salt"}
                ):
                    # The world-frame copy is the sole presentation label in
                    # this task family.  Disable the normal nested child tag
                    # so a shaker cannot carry a second, movable label.
                    object_kwargs["visual_tag"] = None
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    **object_kwargs
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "main_table" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone

                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.6586131746834771, 0.0, 1.6103500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        # For visualization purpose
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )
