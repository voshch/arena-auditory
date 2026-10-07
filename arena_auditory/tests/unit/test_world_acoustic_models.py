from __future__ import annotations

from pathlib import Path

import attrs
import yaml
from arena_rclpy_mixins.param_groups import configure
from arena_simulation_setup.tree.World import Level, MultiLevelWorldView, WorldDescription, WorldIdentifier
from shapely.geometry import Polygon

from arena_auditory.params import PortalGroup, WorldGroup
from arena_auditory.propagation.portal import PortalConfig
from arena_auditory.rooms import AcousticPortalRoute, AcousticRoomSpec, AcousticRoomSpecBuilder, AcousticWorldGraph
from arena_auditory.world import AcousticScene, AcousticZone, WorldConfig, compact_authored_world

WORLD_CONFIG = configure(WorldConfig, WorldGroup, PortalGroup)
PORTAL_CONFIG = configure(PortalConfig, PortalGroup)


def _models(world: WorldDescription, config: WorldConfig = WORLD_CONFIG) -> tuple[AcousticScene, tuple[AcousticRoomSpec, ...], AcousticWorldGraph]:
    rooms = AcousticRoomSpecBuilder(config.ceiling_height_m).from_world(world)
    return AcousticScene.from_world(world, ceiling_height_m=config.ceiling_height_m), rooms, AcousticWorldGraph.from_world(world, rooms, config)


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
    )


def test_world_without_zones_builds_empty_acoustic_models() -> None:
    scene, rooms, graph = _models(WorldDescription(levels={"0": Level()}))

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


def test_hospital_world_builds_rooms_and_portal_routes() -> None:
    world = WorldIdentifier("hospital_1").resolve_sync().load()

    scene, rooms, graph = _models(world)

    assert len(scene.zones) == len(rooms) > 1
    assert scene.zone_at_xy(-1e-10, 0.0).name == "reception"
    assert any(portal.connects("central_hallway", "reception") for portal in graph.portals)
    opening = next(portal for portal in graph.portals if portal.connects("central_hallway", "sub_hallway"))
    assert opening.portal_kind == "opening"
    route = _route(graph, "operating_room", "waiting_area", max_hops=4)
    assert route is not None
    assert route.zones == ("operating_room", "central_hallway", "sub_hallway", "waiting_area")
    assert route.hop_count == 3
    assert _route(graph, "operating_room", "waiting_area", max_hops=2) is None

    _, _, doors_only = _models(world, attrs.evolve(WORLD_CONFIG, openings_enabled=False))
    assert not any(portal.connects("central_hallway", "sub_hallway") for portal in doors_only.portals)
    assert _route(doors_only, "operating_room", "waiting_area", max_hops=12) is None


def _write_level(root: Path, level_id: str, zones: list[dict]) -> None:
    level_dir = root / level_id
    level_dir.mkdir(parents=True)
    (level_dir / "world.yaml").write_text(yaml.safe_dump({"zones": zones}), encoding="utf-8")


def _box_zone(name: str, x_min: float, x_max: float) -> dict:
    return {"name": name, "corners": [[x_min, 0.0], [x_max, 0.0], [x_max, 4.0], [x_min, 4.0]]}


def test_compacted_two_level_world_builds_rooms_and_level_scoped_portals(tmp_path: Path) -> None:
    _write_level(tmp_path / "two_level", "0", [_box_zone("lobby", 0.0, 4.0), _box_zone("hall", 4.0, 8.0)])
    _write_level(tmp_path / "two_level", "1", [_box_zone("office", 0.0, 4.0), _box_zone("store", 4.0, 8.0)])
    world_view = MultiLevelWorldView(tmp_path / "two_level")

    world, origin = compact_authored_world(world_view, world_view.load())
    scene, rooms, graph = _models(world)

    assert origin is not None
    assert {zone.name for zone in scene.zones} == {"lobby", "hall", "office", "store"}
    assert len(rooms) == 4
    assert any(portal.connects("lobby", "hall") for portal in graph.portals)
    assert any(portal.connects("office", "store") for portal in graph.portals)
    assert not any(portal.connects("lobby", "office") or portal.connects("hall", "store") for portal in graph.portals)
