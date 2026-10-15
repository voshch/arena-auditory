from __future__ import annotations

import math
from pathlib import Path

import pytest
from arena_rclpy_mixins.param_groups import configure
from shapely.geometry import Polygon

from arena_auditory.materials import AcousticMaterialCatalog
from arena_auditory.params import PortalGroup, RirGroup
from arena_auditory.propagation.portal import MultiPortalRirCoupler, PortalConfig
from arena_auditory.propagation.pyroom_adapter import PyroomacousticsAdapter, RirConfig, direct_arrival
from arena_auditory.rooms import AcousticBoundarySpec, AcousticPortal, AcousticPortalRoute, AcousticRoomSpec, AcousticWorldGraph, Point2D

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "acoustic_materials.yaml"
AB = ((4.0, 0.0), (4.0, 4.0))
BC = ((4.0, 4.0), (8.0, 4.0))
AC = ((0.0, 4.0), (1.0, 4.0))
SOURCE = (1.0, 1.0, 1.5)
LISTENER = (7.0, 7.0, 1.5)


def _on_edge(point: Point2D, start: Point2D, end: Point2D) -> bool:
    return (start[0] == end[0] == point[0] and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])) or (start[1] == end[1] == point[1] and min(start[0], end[0]) <= point[0] <= max(start[0], end[0]))


def _room(name: str, x_min: float, x_max: float, y_min: float, y_max: float, *openings: tuple[Point2D, Point2D]) -> AcousticRoomSpec:
    corners = ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
    boundary: list[AcousticBoundarySpec] = []
    for index, start in enumerate(corners):
        end = corners[(index + 1) % 4]
        cuts = sorted({start, end, *(point for opening in openings for point in opening if _on_edge(point, start, end))}, key=lambda point: math.dist(start, point))
        for first, second in zip(cuts, cuts[1:], strict=False):
            open_span = any(_on_edge((0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1])), *opening) for opening in openings)
            boundary.append(AcousticBoundarySpec(start=first, end=second, material_id="Acoustic_Open" if open_span else "Acoustic_Default_Wall", kind="opening" if open_span else "wall"))
    return AcousticRoomSpec(zone_name=name, boundary=tuple(boundary), floor_material_id="Acoustic_Default_Floor", ceiling_material_id="Acoustic_Default_Ceiling", ceiling_height_m=3.0)


def _corner_graph() -> tuple[AcousticWorldGraph, tuple[AcousticPortal, AcousticPortal, AcousticPortal]]:
    rooms = (_room("a", 0.0, 4.0, 0.0, 4.0, AB, AC), _room("b", 4.0, 8.0, 0.0, 4.0, AB, BC), _room("c", 0.0, 8.0, 4.0, 8.0, AC, BC))
    ab, bc, ac = (AcousticPortal(portal_id=f"opening:{name}", door_name="", zone_a=name[0], zone_b=name[1], start=span[0], end=span[1], height_m=3.0, material_id="Acoustic_Open", portal_kind="opening") for name, span in (("ab", AB), ("bc", BC), ("ac", AC)))
    return AcousticWorldGraph(rooms=rooms, portals=(ab, bc, ac), zone_polygons=tuple((room.zone_name, Polygon(room.corners_xy)) for room in rooms)), (ab, bc, ac)


def test_route_search_prefers_two_shallow_corners_over_one_sharp_corner() -> None:
    pytest.importorskip("pyroomacoustics")
    adapter = PyroomacousticsAdapter(AcousticMaterialCatalog(CATALOG_PATH), configure(RirConfig, RirGroup))
    config = configure(PortalConfig, PortalGroup)
    graph, (ab, bc, ac) = _corner_graph()
    coupler = MultiPortalRirCoupler(adapter, graph, world_name="corner_route", config=config)

    chosen = graph.find_portal_route(
        "a",
        "c",
        source_xy=SOURCE[:2],
        listener_xy=LISTENER[:2],
        max_hops=config.max_hops,
        route_loss_db_per_m=config.route_loss_db_per_m,
        door_loss_db=config.door_loss_db,
        opening_loss_db=config.opening_loss_db,
        corner_reference_hz=config.corner_reference_hz,
        speed_of_sound_mps=adapter.speed_of_sound_mps,
    )
    levels = {}
    for route in (AcousticPortalRoute(zones=("a", "c"), portals=(ac,)), AcousticPortalRoute(zones=("a", "b", "c"), portals=(ab, bc))):
        result = coupler.compute(source_zone="a", listener_zone="c", source_position_m=SOURCE, listener_position_m=LISTENER, route=route)
        levels[route.zones] = direct_arrival(result.rir).gain_db
        print(f"\nroute {route.zones}: direct {levels[route.zones]:.2f} dB, corner loss {result.applied_corner_loss_db:.2f} dB")

    assert chosen is not None
    assert chosen.zones == max(levels, key=levels.__getitem__) == ("a", "b", "c")
    assert levels[("a", "b", "c")] - levels[("a", "c")] > 3.0
