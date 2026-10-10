"""Problem registry for the frozen RoboMemArena compatibility core.

Keep this module independent from local LIBERO-only problem classes. The
bootstrap overlays these four upstream-compatible implementations on a merged
package; importing the local Door bridge here would mix the two registries
before RoboMemArena's core is initialized.
"""

from .libero_coffee_table_manipulation import Libero_Coffee_Table_Manipulation
from .libero_kitchen_tabletop_manipulation import (
    Libero_Kitchen_Tabletop_Manipulation,
)
from .libero_living_room_tabletop_manipulation import (
    Libero_Living_Room_Tabletop_Manipulation,
)
from .libero_study_tabletop_manipulation import Libero_Study_Tabletop_Manipulation


__all__ = [
    "Libero_Coffee_Table_Manipulation",
    "Libero_Kitchen_Tabletop_Manipulation",
    "Libero_Living_Room_Tabletop_Manipulation",
    "Libero_Study_Tabletop_Manipulation",
]
