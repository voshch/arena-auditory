from __future__ import annotations

from pathlib import Path

import attrs
import yaml
from arena_rclpy_mixins.param_groups import configure
from arena_simulation_setup.kg import Scene
from arena_simulation_setup.tree.World import Level, World, WorldIdentifier, WorldView
from shapely.geometry import Polygon

from arena_auditory.params import PortalGroup, WorldGroup
from arena_auditory.propagation.portal import PortalConfig
from arena_auditory.rooms import AcousticPortalRoute, AcousticRoomSpec, AcousticRoomSpecBuilder, AcousticWorldGraph
from arena_auditory.world import AcousticScene, AcousticZone, WorldConfig, compact_authored_world

WORLD_CONFIG = configure(WorldConfig, WorldGroup, PortalGroup)
PORTAL_CONFIG = configure(PortalConfig, PortalGroup)


def _models(world: World | Level, semantics: Scene | None, config: WorldConfig = WORLD_CONFIG) -> tuple[AcousticScene, tuple[AcousticRoomSpec, ...], AcousticWorldGraph]:
    rooms = AcousticRoomSpecBuilder(config).from_world(world)
    graph = AcousticWorldGraph.from_world(world, rooms, config, semantics if semantics is not None else Scene())
    return AcousticScene.from_world(world, ceiling_height_m=config.ceiling_height_m), rooms, graph


def _route(graph: AcousticWorldGraph, source_zone: str, listener_zone: str, *, max_hops: int) -> AcousticPortalRoute | None:
    return graph.find_portal_route(
        source_zone,
        listener_zone,
        source_xy=(13.69, 18.00),
        listener_xy=(23.95, 6.81),
        max_hops=max_hops,
        route_loss_db_per_m=PORTAL_CONFIG.route_loss_db_per_m,
        door_loss_db=PORTAL_CONFIG.door_loss_db,
        opening_loss_db=PORTAL_CONFIG.opening_loss_db,
        corner_reference_hz=PORTAL_CONFIG.corner_reference_hz,
        speed_of_sound_mps=343.0,
    )


def test_world_without_zones_builds_empty_acoustic_models() -> None:
    scene, rooms, graph = _models(World(levels={"0": Level(id="0")}), Scene())

    assert scene.zones == ()
    assert scene.walls == ()
    assert rooms == ()
    assert graph.rooms == ()
    assert graph.portals == ()


def test_zone_lookup_prefers_exact_room_before_nearby_room() -> None:
    scene = AcousticScene(
        zones=(
            AcousticZone(name="left", polygon=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), floor_material_id="default"),
            AcousticZone(name="right", polygon=Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]), floor_material_id="default"),
        ),
        walls=(),
        ceiling_height_m=3.0,
    )

    assert scene.zone_at_xy(1.1, 0.5).name == "right"
    assert scene.zone_at_xy(-0.3, 0.5).name == "left"
    assert scene.zone_at_xy(-0.36, 0.5) is None


def test_hospital_world_merges_the_open_corridors_and_keeps_door_portals() -> None:
    world = WorldIdentifier("hospital_1").resolve_sync().load()

    scene, rooms, graph = _models(world, world.kg)

    assert len(scene.zones) == 13
    assert len(rooms) == 12
    (corridor,) = [room for room in rooms if len(room.zone_names) > 1]
    assert corridor.zone_names == ("central_hallway", "sub_hallway")
    assert scene.zone_at_xy(-1e-10, 0.0).name == "reception"
    assert any(portal.connects(corridor.zone_name, "reception") for portal in graph.portals)
    assert all(portal.portal_kind == "door" for portal in graph.portals)
    route = _route(graph, "operating_room", "waiting_area", max_hops=4)
    assert route is not None
    assert route.zones == ("operating_room", corridor.zone_name, "waiting_area")
    assert route.hop_count == 2
    assert _route(graph, "operating_room", "waiting_area", max_hops=1) is None


def _write_world(root: Path, levels: list[dict]) -> WorldView:
    root.mkdir(parents=True)
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": levels}), encoding="utf-8")
    return WorldView(root)


def _box_zone(name: str, x_min: float, x_max: float) -> dict:
    return {"name": name, "polygon": [[x_min, 0.0], [x_max, 0.0], [x_max, 4.0], [x_min, 4.0]]}


def _doorway_walls() -> list[dict]:
    return [{"path": [[4.0, 0.0], [4.0, 1.5]]}, {"path": [[4.0, 2.5], [4.0, 4.0]]}]


def test_zones_joined_across_a_wall_free_boundary_form_one_room(tmp_path: Path) -> None:
    world_view = _write_world(tmp_path / "open", [{"id": "0", "zones": [_box_zone("lobby", 0.0, 4.0), _box_zone("hall", 4.0, 8.0)]}])
    world = world_view.load()

    _, rooms, graph = _models(world, world.kg)

    (room,) = rooms
    assert room.zone_name == "hall+lobby"
    assert room.zone_names == ("hall", "lobby")
    assert room.corners_xy == ((0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0), (0.0, 4.0))
    assert graph.portals == ()
    assert graph.room("hall") == graph.room("lobby") == room


def test_wall_gap_between_two_zones_is_an_opening_portal(tmp_path: Path) -> None:
    world_view = _write_world(tmp_path / "walled", [{"id": "0", "zones": [_box_zone("lobby", 0.0, 4.0), _box_zone("hall", 4.0, 8.0)], "walls": _doorway_walls()}])
    world = world_view.load()

    _, rooms, graph = _models(world, world.kg)

    assert sorted(room.zone_name for room in rooms) == ["hall", "lobby"]
    (opening,) = graph.portals
    assert opening.portal_kind == "opening"
    assert sorted((opening.start, opening.end)) == [(4.0, 1.5), (4.0, 2.5)]

    _, _, doors_only = _models(world, world.kg, attrs.evolve(WORLD_CONFIG, openings_enabled=False))
    assert doors_only.portals == ()


def test_compacted_two_level_world_builds_rooms_and_level_scoped_portals(tmp_path: Path) -> None:
    world_view = _write_world(
        tmp_path / "two_level",
        [
            {"id": "0", "zones": [_box_zone("lobby", 0.0, 4.0), _box_zone("hall", 4.0, 8.0)], "walls": _doorway_walls()},
            {"id": "1", "zones": [_box_zone("office", 0.0, 4.0), _box_zone("store", 4.0, 8.0)], "walls": _doorway_walls()},
        ],
    )
    description = world_view.load()

    world, origin = compact_authored_world(world_view, description)
    scene, rooms, graph = _models(world, description.kg)

    assert origin is not None
    assert {zone.name for zone in scene.zones} == {"lobby", "hall", "office", "store"}
    assert len(rooms) == 4
    assert any(portal.connects("lobby", "hall") for portal in graph.portals)
    assert any(portal.connects("office", "store") for portal in graph.portals)
    assert not any(portal.connects("lobby", "office") or portal.connects("hall", "store") for portal in graph.portals)


def test_door_portal_joins_the_zones_the_semantics_connects(tmp_path: Path) -> None:
    world_view = _write_world(
        tmp_path / "doors",
        [
            {
                "id": "0",
                "zones": [_box_zone("lobby", 0.0, 4.0), _box_zone("hall", 4.0, 8.0)],
                "walls": [{"path": [[4.0, 0.0], [4.0, 4.0]]}],
                "doors": [
                    {"name": "lobby_door", "span": [[4.0, 1.5], [4.0, 2.5]]},
                    {"name": "exit", "span": [[0.0, 1.5], [0.0, 2.5]]},
                ],
            }
        ],
    )
    world = world_view.load()

    scene, rooms, graph = _models(world, world.kg)

    (portal,) = graph.portals
    assert (portal.portal_kind, portal.door_name) == ("door", "lobby_door")
    assert portal.connects("lobby", "hall")
    assert (portal.start, portal.end) == ((4.0, 1.5), (4.0, 2.5))
    (unpaired,) = graph.unpaired_doors
    assert (unpaired.door_name, unpaired.owner_zone, unpaired.reason) == ("exit", "lobby", "no_adjacent_acoustic_zone")
    lobby = next(room for room in rooms if room.zone_name == "lobby")
    assert [segment.kind for segment in lobby.boundary if segment.start[0] == segment.end[0] == 4.0] == ["wall", "door", "wall"]
    assert len(scene.walls) == 2


def test_second_door_between_the_same_zones_is_its_own_portal(tmp_path: Path) -> None:
    world_view = _write_world(
        tmp_path / "doors",
        [
            {
                "id": "0",
                "zones": [_box_zone("lobby", 0.0, 4.0), _box_zone("hall", 4.0, 8.0)],
                "walls": [{"path": [[4.0, 0.0], [4.0, 4.0]]}],
                "doors": [{"name": "south_door", "span": [[4.0, 0.5], [4.0, 1.5]]}, {"name": "north_door", "span": [[4.0, 2.5], [4.0, 3.5]]}],
            }
        ],
    )
    world = world_view.load()

    _, _, graph = _models(world, world.kg)

    assert sorted(portal.door_name for portal in graph.portals) == ["north_door", "south_door"]
    assert all(portal.connects("lobby", "hall") for portal in graph.portals)
    assert graph.unpaired_doors == ()


def test_door_without_semantics_is_unpaired(tmp_path: Path) -> None:
    world_view = _write_world(
        tmp_path / "doors",
        [{"id": "0", "zones": [_box_zone("lobby", 0.0, 4.0), _box_zone("hall", 4.0, 8.0)], "walls": [{"path": [[4.0, 0.0], [4.0, 4.0]]}], "doors": [{"name": "lobby_door", "span": [[4.0, 1.5], [4.0, 2.5]]}]}],
    )

    world = world_view.load(derive="off")
    _, _, graph = _models(world, world.kg)

    assert graph.portals == ()
    assert [(door.door_name, door.owner_zone) for door in graph.unpaired_doors] == [("lobby_door", "")]
