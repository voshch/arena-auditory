from __future__ import annotations

from arena_simulation_setup.tree.World import LevelDescription
from arena_simulation_setup.utils.cattrs import converter

from arena_auditory.rooms import OPENING_MATERIAL_ID, AcousticRoomSpecBuilder


def _level(**zone_fields: object) -> LevelDescription:
    return converter.structure(
        {"zones": [{"name": "room", "corners": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]], **zone_fields}]},
        LevelDescription,
    )


def test_room_uses_authored_ceiling_height_and_material() -> None:
    (room,) = AcousticRoomSpecBuilder(3.0).from_world(_level(ceiling_height=2.4, ceiling_material="Plaster_Wall"))

    assert room.ceiling_height_m == 2.4
    assert room.ceiling_material_id == "Plaster_Wall"


def test_room_without_authored_ceiling_height_uses_configured_height() -> None:
    (room,) = AcousticRoomSpecBuilder(3.5).from_world(_level())

    assert room.ceiling_height_m == 3.5


def test_open_ceiling_is_fully_absorbing() -> None:
    (room,) = AcousticRoomSpecBuilder(3.0).from_world(_level(ceiling=False, ceiling_material="Plaster_Wall"))

    assert room.ceiling_material_id == OPENING_MATERIAL_ID
