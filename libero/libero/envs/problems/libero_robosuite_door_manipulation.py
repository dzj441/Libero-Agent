"""LIBERO BDDL bridge for robosuite's original Door scene."""

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.problems.libero_tabletop_manipulation import (
    Libero_Tabletop_Manipulation,
)


@register_problem
class Libero_Robosuite_Door_Manipulation(Libero_Tabletop_Manipulation):
    """Run the adapted Door task in robosuite's stock narrow-table scene.

    The generic LIBERO tabletop problem deliberately uses a larger styled
    table and a mounted robot. Door is calibrated against robosuite 1.4's
    0.8 x 0.3 m TableArena, world offset, ordinary Panda, and agent camera, so
    this problem keeps those source settings isolated from all other tasks.
    """

    TABLE_FULL_SIZE = (0.8, 0.3, 0.05)
    TABLE_OFFSET = (-0.2, -0.35, 0.8)
    AGENTVIEW_POS = (
        0.5986131746834771,
        -4.392035683362857e-09,
        1.5903500240372423,
    )
    AGENTVIEW_QUAT = (
        0.6380177736282349,
        0.3048497438430786,
        0.30484986305236816,
        0.6380177736282349,
    )

    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "main_table"
        self.visualization_sites_list = []
        self.table_full_size = self.TABLE_FULL_SIZE
        self.table_offset = self.TABLE_OFFSET

        # robosuite's DoorObject bottom_site is already calibrated for its
        # own sampler, so unlike native LIBERO fixtures it needs no -4 cm
        # fixture-origin correction.
        self.z_offset = 0.0

        # These source settings are part of this dedicated problem rather
        # than caller-tunable variants of LIBERO's generic tabletop scene.
        kwargs.pop("table_full_size", None)
        kwargs.update(
            {
                "workspace_offset": self.table_offset,
                "arena_type": "robosuite_table",
            }
        )
        BDDLBaseDomain.__init__(self, bddl_file_name, *args, **kwargs)

    def _setup_camera(self, mujoco_arena):
        # Match robosuite.environments.manipulation.Door._load_model exactly.
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=self.AGENTVIEW_POS,
            quat=self.AGENTVIEW_QUAT,
        )
        # Retain LIBERO's optional canonical camera name without introducing
        # a second viewpoint for the benchmark-facing head observation.
        mujoco_arena.set_camera(
            camera_name="canonical_agentview",
            pos=self.AGENTVIEW_POS,
            quat=self.AGENTVIEW_QUAT,
        )
