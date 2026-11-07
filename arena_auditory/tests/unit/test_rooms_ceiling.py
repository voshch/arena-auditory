from __future__ import annotations

import attrs
from arena_rclpy_mixins.param_groups import configure
from arena_simulation_setup.tree.World import Level, parse_world

from arena_auditory.params import PortalGroup, WorldGroup
from arena_auditory.rooms import OPENING_MATERIAL_ID, AcousticRoomSpecBuilder
from arena_auditory.world import WorldConfig

CONFIG = configure(WorldConfig, WorldGroup, PortalGroup)


def _level(**zone_fields: object) -> Level:
    return parse_world({"format": "arena-world/2", "levels": [{"id": "0", "zones": [{"name": "room", "polygon": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]], **zone_fields}]}]}).levels["0"]


def _builder(ceiling_height_m: float) -> AcousticRoomSpecBuilder:
    return AcousticRoomSpecBuilder(attrs.evolve(CONFIG, ceiling_height_m=ceiling_height_m))


def test_room_uses_authored_ceiling_height_and_material() -> None:
    (room,) = _builder(3.0).from_world(_level(ceiling={"height": 2.4, "material": "Plaster_Wall"}))

    assert room.ceiling_height_m == 2.4
    assert room.ceiling_material_id == "Plaster_Wall"


def test_room_without_authored_ceiling_height_uses_configured_height() -> None:
    (room,) = _builder(3.5).from_world(_level())

    assert room.ceiling_height_m == 3.5


def test_open_ceiling_is_fully_absorbing() -> None:
    (room,) = _builder(3.0).from_world(_level(ceiling=False))

    assert room.ceiling_material_id == OPENING_MATERIAL_ID
