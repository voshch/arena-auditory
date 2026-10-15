from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from arena_rclpy_mixins.param_groups import configure
from scipy.signal import fftconvolve
from shapely.geometry import Polygon

from arena_auditory.materials import AcousticMaterialCatalog
from arena_auditory.params import PortalGroup, RirGroup
from arena_auditory.propagation.portal import MultiPortalRirCoupler, PortalConfig, corner_loss_db
from arena_auditory.propagation.pyroom_adapter import PyroomacousticsAdapter, RirConfig, RoomImpulseResponse, direct_arrival
from arena_auditory.rooms import AcousticBoundarySpec, AcousticPortal, AcousticRoomSpec, AcousticWorldGraph, Point2D, Position3D

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "acoustic_materials.yaml"
EAR_HEIGHT_M = 1.5
U_SOURCE = (1.0, 1.5, EAR_HEIGHT_M)


def _portal_config(**given: object) -> PortalConfig:
    return configure(PortalConfig, PortalGroup, **given)


def _room(name: str, x_min: float, x_max: float, y_max: float = 1.0, material_id: str = "wall") -> AcousticRoomSpec:
    corners = ((x_min, 0.0), (x_max, 0.0), (x_max, y_max), (x_min, y_max))
    boundary = tuple(AcousticBoundarySpec(start=start, end=corners[(index + 1) % len(corners)], material_id=material_id, kind="wall") for index, start in enumerate(corners))
    return AcousticRoomSpec(zone_name=name, boundary=boundary, floor_material_id="floor", ceiling_material_id="ceiling", ceiling_height_m=3.0)


def _route_lengths(graph: AcousticWorldGraph, route_zones: tuple[str, ...], portals: tuple[AcousticPortal, ...], source: Position3D, listener: Position3D, inset_m: float = 0.03) -> tuple[list[float], float]:
    points = [source]
    gap = 0.0
    for index, portal in enumerate(portals):
        height = min(0.5 * portal.height_m, 2.99)
        before = graph.position_inside_portal(portal, route_zones[index], inset_m=inset_m, height_m=height)
        after = graph.position_inside_portal(portal, route_zones[index + 1], inset_m=inset_m, height_m=height)
        points.extend((before, after))
        gap += math.dist(before, after)
    points.append(listener)
    return [math.dist(points[index], points[index + 1]) for index in range(0, len(points), 2)], gap


def _path_scale(graph: AcousticWorldGraph, route_zones: tuple[str, ...], portals: tuple[AcousticPortal, ...], source: Position3D, listener: Position3D) -> float:
    lengths, gap = _route_lengths(graph, route_zones, portals, source, listener)
    return math.prod(lengths) / (sum(lengths) + gap)


def _free_field_db(graph: AcousticWorldGraph, route_zones: tuple[str, ...], portals: tuple[AcousticPortal, ...], source: Position3D, listener: Position3D) -> float:
    lengths, gap = _route_lengths(graph, route_zones, portals, source, listener)
    return 20.0 * math.log10(1.0 / (sum(lengths) + gap))


def _on_edge(point: Point2D, start: Point2D, end: Point2D) -> bool:
    return (start[0] == end[0] == point[0] and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])) or (start[1] == end[1] == point[1] and min(start[0], end[0]) <= point[0] <= max(start[0], end[0]))


def _open_room(name: str, x_min: float, x_max: float, y_min: float, y_max: float, *openings: tuple[Point2D, Point2D], ceiling_material_id: str = "ceiling") -> AcousticRoomSpec:
    corners = ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
    boundary: list[AcousticBoundarySpec] = []
    for index, start in enumerate(corners):
        end = corners[(index + 1) % len(corners)]
        cuts = sorted({start, end, *(point for opening in openings for point in opening if _on_edge(point, start, end))}, key=lambda point: math.dist(start, point))
        for first, second in zip(cuts, cuts[1:], strict=False):
            middle = (0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1]))
            opening = any(_on_edge(middle, *span) for span in openings)
            boundary.append(AcousticBoundarySpec(start=first, end=second, material_id="Acoustic_Open" if opening else "Acoustic_Default_Wall", kind="opening" if opening else "wall"))
    return AcousticRoomSpec(zone_name=name, boundary=tuple(boundary), floor_material_id="floor", ceiling_material_id=ceiling_material_id, ceiling_height_m=3.0)


def _graph(rooms: tuple[AcousticRoomSpec, ...], portals: tuple[AcousticPortal, ...]) -> AcousticWorldGraph:
    return AcousticWorldGraph(rooms=rooms, portals=portals, zone_polygons=tuple((room.zone_name, Polygon(room.corners_xy)) for room in rooms))


def _f_shape_graph() -> tuple[AcousticWorldGraph, tuple[AcousticPortal, AcousticPortal]]:
    lower = ((7.775, 9.0), (10.875, 9.0))
    upper = ((7.775, 11.85), (10.875, 11.85))
    rooms = (
        _open_room("zone_02", 0.85, 10.875, 6.05, 9.0, ((7.75, 6.05), (10.875, 6.05)), lower, ceiling_material_id="Acoustic_Open"),
        _open_room("zone_03", 7.775, 10.875, 9.0, 11.85, lower, upper, ceiling_material_id="Acoustic_Open"),
        _open_room("zone_04", 0.025, 10.875, 11.85, 14.775, upper, ceiling_material_id="Acoustic_Open"),
    )
    portals = (
        AcousticPortal(portal_id="opening:23", door_name="", zone_a="zone_02", zone_b="zone_03", start=lower[0], end=lower[1], height_m=3.0, material_id="Acoustic_Open", portal_kind="opening"),
        AcousticPortal(portal_id="opening:34", door_name="", zone_a="zone_03", zone_b="zone_04", start=upper[0], end=upper[1], height_m=3.0, material_id="Acoustic_Open", portal_kind="opening"),
    )
    return _graph(rooms, portals), portals


def _u_graph() -> tuple[AcousticWorldGraph, tuple[AcousticPortal, AcousticPortal]]:
    ab = ((6.0, 0.0), (6.0, 3.0))
    bc = ((6.0, 6.0), (9.0, 6.0))
    rooms = (_open_room("a", 0.0, 6.0, 0.0, 3.0, ab), _open_room("b", 6.0, 9.0, 0.0, 6.0, ab, bc), _open_room("c", 3.0, 9.0, 6.0, 9.0, bc))
    portals = (
        AcousticPortal(portal_id="opening:ab", door_name="", zone_a="a", zone_b="b", start=ab[0], end=ab[1], height_m=3.0, material_id="Acoustic_Open", portal_kind="opening"),
        AcousticPortal(portal_id="opening:bc", door_name="", zone_a="b", zone_b="c", start=bc[0], end=bc[1], height_m=3.0, material_id="Acoustic_Open", portal_kind="opening"),
    )
    return _graph(rooms, portals), portals


class _FixedRirAdapter:
    speed_of_sound_mps = 343.0

    def __init__(self) -> None:
        self.calls = 0

    def compute_rir(self, room: AcousticRoomSpec, *, source_position_m: Position3D, listener_position_m: Position3D) -> RoomImpulseResponse:
        self.calls += 1
        return RoomImpulseResponse(samples=np.asarray([1.0, 0.5], dtype=np.float64), sample_rate_hz=100, global_delay_samples=2, fallback_material_ids=())


def test_one_door_rir_is_composed_and_endpoint_rirs_are_cached() -> None:
    portal = AcousticPortal(portal_id="door:test:a:b", door_name="test", zone_a="a", zone_b="b", start=(1.0, 0.4), end=(1.0, 0.6), height_m=2.0, material_id="door")
    graph = AcousticWorldGraph(
        rooms=(_room("a", 0.0, 1.0), _room("b", 1.0, 2.0)),
        portals=(portal,),
        zone_polygons=(
            ("a", Polygon(((0, 0), (1, 0), (1, 1), (0, 1)))),
            ("b", Polygon(((1, 0), (2, 0), (2, 1), (1, 1)))),
        ),
    )
    adapter = _FixedRirAdapter()
    coupler = MultiPortalRirCoupler(
        adapter,  # type: ignore[arg-type]
        graph,
        world_name="test",
        config=_portal_config(door_loss_db=6.0, early_window_s=0.02, max_rir_duration_s=1.0),
    )
    source = (0.5, 0.5, 1.6)
    listener = (1.5, 0.5, 0.35)

    first_result = coupler.compute(source_zone="a", listener_zone="b", source_position_m=source, listener_position_m=listener)
    second_result = coupler.compute(source_zone="a", listener_zone="b", source_position_m=source, listener_position_m=listener)

    expected = fftconvolve([1.0, 0.5], [1.0, 0.5]) * 10.0 ** (-6.0 / 20.0) * _path_scale(graph, ("a", "b"), (portal,), source, listener)
    np.testing.assert_allclose(first_result.rir.samples, expected)
    np.testing.assert_allclose(second_result.rir.samples, expected)
    assert first_result.rir.global_delay_samples == 4
    assert first_result.portal.portal_id == "door:test:a:b"
    assert first_result.applied_portal_loss_db == 6.0
    assert adapter.calls == 2
    assert coupler.cache_entries == 2
    assert coupler.cache_hits == 0
    assert coupler.cache_misses == 2
    assert coupler.route_cache_entries == 1
    assert coupler.route_cache_hits == 1
    assert coupler.route_cache_misses == 1


def test_multi_portal_rir_composes_transit_rooms_and_reuses_cache() -> None:
    portals = (
        AcousticPortal(portal_id="door:ab", door_name="ab", zone_a="a", zone_b="b", start=(1.0, 0.4), end=(1.0, 0.6), height_m=2.0, material_id="door"),
        AcousticPortal(portal_id="opening:bc", door_name="", zone_a="b", zone_b="c", start=(2.0, 0.4), end=(2.0, 0.6), height_m=2.5, material_id="open", portal_kind="opening"),
    )
    graph = AcousticWorldGraph(
        rooms=(_room("a", 0.0, 1.0), _room("b", 1.0, 2.0), _room("c", 2.0, 3.0)),
        portals=portals,
        zone_polygons=(
            ("a", Polygon(((0, 0), (1, 0), (1, 1), (0, 1)))),
            ("b", Polygon(((1, 0), (2, 0), (2, 1), (1, 1)))),
            ("c", Polygon(((2, 0), (3, 0), (3, 1), (2, 1)))),
        ),
    )
    adapter = _FixedRirAdapter()
    coupler = MultiPortalRirCoupler(
        adapter,  # type: ignore[arg-type]
        graph,
        world_name="test",
        config=_portal_config(door_loss_db=6.0, opening_loss_db=1.0, early_window_s=0.02, max_rir_duration_s=1.0, max_hops=3),
    )
    source = (0.5, 0.5, 1.6)
    listener = (2.5, 0.5, 0.35)

    first_result = coupler.compute(source_zone="a", listener_zone="c", source_position_m=source, listener_position_m=listener)
    second_result = coupler.compute(source_zone="a", listener_zone="c", source_position_m=source, listener_position_m=listener)

    expected = fftconvolve(fftconvolve([1.0, 0.5], [1.0, 0.5]), [1.0, 0.5]) * 10.0 ** (-7.0 / 20.0) * _path_scale(graph, ("a", "b", "c"), portals, source, listener)
    np.testing.assert_allclose(first_result.rir.samples, expected)
    np.testing.assert_allclose(second_result.rir.samples, expected)
    assert first_result.route is not None
    assert first_result.route.zones == ("a", "b", "c")
    assert first_result.route.hop_count == 2
    assert first_result.applied_portal_loss_db == 7.0
    assert adapter.calls == 3
    assert coupler.cache_entries == 3
    assert coupler.cache_hits == 0
    assert coupler.cache_misses == 3
    assert coupler.route_cache_entries == 1
    assert coupler.route_cache_hits == 1
    assert coupler.route_cache_misses == 1


def _anechoic_adapter() -> PyroomacousticsAdapter:
    pytest.importorskip("pyroomacoustics")
    return PyroomacousticsAdapter(AcousticMaterialCatalog(CATALOG_PATH), configure(RirConfig, RirGroup, max_order=0))


def _two_room_coupler(width: float, adapter: PyroomacousticsAdapter) -> tuple[MultiPortalRirCoupler, AcousticWorldGraph]:
    first = _room("a", 0.0, width, y_max=width, material_id="Acoustic_Default_Wall")
    second = _room("b", width, 2.0 * width, y_max=width, material_id="Acoustic_Default_Wall")
    portal = AcousticPortal(portal_id="door:ab", door_name="ab", zone_a="a", zone_b="b", start=(width, 0.4 * width), end=(width, 0.6 * width), height_m=2.0, material_id="door")
    graph = AcousticWorldGraph(
        rooms=(first, second),
        portals=(portal,),
        zone_polygons=(("a", Polygon(first.corners_xy)), ("b", Polygon(second.corners_xy))),
    )
    coupler = MultiPortalRirCoupler(adapter, graph, world_name="test", config=_portal_config(door_loss_db=0.0, opening_loss_db=0.0, max_rir_duration_s=0.5))
    return coupler, graph


def test_route_split_at_zone_boundary_matches_unsplit_level_and_delay() -> None:
    adapter = _anechoic_adapter()
    coupler, _ = _two_room_coupler(1.0, adapter)
    source = (0.5, 0.5, 1.0)
    listener = (1.5, 0.5, 1.0)

    split = direct_arrival(coupler.compute(source_zone="a", listener_zone="b", source_position_m=source, listener_position_m=listener).rir)
    unsplit = direct_arrival(adapter.compute_rir(_room("ab", 0.0, 2.0, material_id="Acoustic_Default_Wall"), source_position_m=source, listener_position_m=listener))

    assert unsplit.gain_db == pytest.approx(0.0, abs=0.2)
    assert split.gain_db == pytest.approx(unsplit.gain_db, abs=0.3)
    assert split.delay_s == pytest.approx(unsplit.delay_s, abs=1.0 / adapter.config.sample_rate_hz)


def test_route_level_matches_a_straight_path_of_the_same_total_length() -> None:
    adapter = _anechoic_adapter()
    coupler, _ = _two_room_coupler(5.0, adapter)
    source = (1.0, 2.5, 1.0)
    listener = (9.0, 2.5, 1.0)

    arrival = direct_arrival(coupler.compute(source_zone="a", listener_zone="b", source_position_m=source, listener_position_m=listener).rir)
    unsplit = direct_arrival(adapter.compute_rir(_room("ab", 0.0, 10.0, y_max=5.0, material_id="Acoustic_Default_Wall"), source_position_m=source, listener_position_m=listener))

    assert 20.0 * math.log10(1.0 / 8.0) - 1.0 < arrival.gain_db < 20.0 * math.log10(1.0 / 8.0)
    assert arrival.gain_db == pytest.approx(unsplit.gain_db, abs=0.3)
    assert arrival.delay_s == pytest.approx(8.0 / adapter.speed_of_sound_mps, abs=1.0 / adapter.config.sample_rate_hz)


def test_route_cache_hit_returns_exact_position_delay_and_level() -> None:
    adapter = _anechoic_adapter()
    coupler, _ = _two_room_coupler(5.0, adapter)
    source = (1.0, 2.5, 1.0)

    first = direct_arrival(coupler.compute(source_zone="a", listener_zone="b", source_position_m=source, listener_position_m=(8.46, 2.5, 1.0)).rir)
    second = direct_arrival(coupler.compute(source_zone="a", listener_zone="b", source_position_m=source, listener_position_m=(8.54, 2.5, 1.0)).rir)

    assert coupler.route_cache_hits == 1
    assert second.delay_s - first.delay_s == pytest.approx(0.08 / adapter.speed_of_sound_mps, abs=0.6 / adapter.config.sample_rate_hz)
    assert second.gain_db - first.gain_db == pytest.approx(20.0 * math.log10(7.46 / 7.54), abs=0.05)


def test_supplied_route_skips_the_route_search() -> None:
    adapter = _anechoic_adapter()
    coupler, graph = _two_room_coupler(5.0, adapter)
    source = (1.0, 2.5, 1.0)
    listener = (9.0, 2.5, 1.0)
    config = _portal_config()
    route = graph.find_portal_route(
        "a",
        "b",
        source_xy=source[:2],
        listener_xy=listener[:2],
        max_hops=config.max_hops,
        route_loss_db_per_m=config.route_loss_db_per_m,
        door_loss_db=config.door_loss_db,
        opening_loss_db=config.opening_loss_db,
    )

    result = coupler.compute(source_zone="a", listener_zone="b", source_position_m=source, listener_position_m=listener, route=route)

    assert result.route is route


def test_corner_loss_is_zero_without_deflection_and_grows_with_the_angle() -> None:
    angles = np.radians(np.arange(0.0, 181.0, 15.0))
    for incoming_m, outgoing_m in ((5.0, 3.0), (2.0, 8.0), (0.5, 0.5)):
        losses = [corner_loss_db(float(angle), incoming_m, outgoing_m, reference_hz=500.0, speed_of_sound_mps=343.0) for angle in angles]

        assert losses[0] == 0.0
        assert all(later > earlier for earlier, later in zip(losses, losses[1:], strict=False))


def test_straight_route_through_an_opening_has_no_corner_loss() -> None:
    graph, _ = _u_graph()
    coupler = MultiPortalRirCoupler(_FixedRirAdapter(), graph, world_name="u", config=_portal_config())  # type: ignore[arg-type]

    result = coupler.compute(source_zone="a", listener_zone="b", source_position_m=U_SOURCE, listener_position_m=(8.5, 1.5, EAR_HEIGHT_M))

    assert result.applied_corner_loss_db == 0.0


def test_route_corner_loss_grows_with_the_turn_toward_the_listener() -> None:
    graph, _ = _u_graph()
    coupler = MultiPortalRirCoupler(_FixedRirAdapter(), graph, world_name="u", config=_portal_config())  # type: ignore[arg-type]
    losses = []
    for degrees in (0.0, 20.0, 40.0, 60.0, 80.0):
        turn = math.radians(degrees)
        listener = (6.03 + 2.5 * math.cos(turn), 1.5 + 2.5 * math.sin(turn), EAR_HEIGHT_M)
        losses.append(coupler.compute(source_zone="a", listener_zone="b", source_position_m=U_SOURCE, listener_position_m=listener).applied_corner_loss_db)

    assert losses[0] == 0.0
    assert all(later > earlier for earlier, later in zip(losses, losses[1:], strict=False))


def _reverberant_adapter() -> PyroomacousticsAdapter:
    pytest.importorskip("pyroomacoustics")
    return PyroomacousticsAdapter(AcousticMaterialCatalog(CATALOG_PATH), configure(RirConfig, RirGroup))


def test_straight_route_through_an_opening_reads_free_field_less_the_opening_loss() -> None:
    graph, portals = _u_graph()
    coupler = MultiPortalRirCoupler(_reverberant_adapter(), graph, world_name="u", config=_portal_config())
    listener = (8.5, 1.5, EAR_HEIGHT_M)

    result = coupler.compute(source_zone="a", listener_zone="b", source_position_m=U_SOURCE, listener_position_m=listener)

    assert result.applied_corner_loss_db == 0.0
    expected = _free_field_db(graph, ("a", "b"), portals[:1], U_SOURCE, listener) - result.applied_portal_loss_db
    assert expected - 1.0 < direct_arrival(result.rir).gain_db <= expected


def test_two_corner_route_at_12_m_is_quieter_than_one_corner_route_at_8_m() -> None:
    graph, portals = _u_graph()
    coupler = MultiPortalRirCoupler(_reverberant_adapter(), graph, world_name="u", config=_portal_config())
    near, far = (7.5, 4.1, EAR_HEIGHT_M), (5.5, 7.2, EAR_HEIGHT_M)

    one_corner = coupler.compute(source_zone="a", listener_zone="b", source_position_m=U_SOURCE, listener_position_m=near)
    two_corners = coupler.compute(source_zone="a", listener_zone="c", source_position_m=U_SOURCE, listener_position_m=far)

    assert sum(_route_lengths(graph, ("a", "b"), portals[:1], U_SOURCE, near)[0]) == pytest.approx(8.0, abs=0.1)
    assert sum(_route_lengths(graph, ("a", "b", "c"), portals, U_SOURCE, far)[0]) == pytest.approx(12.0, abs=0.1)
    assert direct_arrival(two_corners.rir).gain_db < direct_arrival(one_corner.rir).gain_db


@pytest.mark.parametrize(
    ("listener_zone", "listener"),
    [("b", (8.5, 1.5, EAR_HEIGHT_M)), ("b", (7.5, 4.1, EAR_HEIGHT_M)), ("c", (5.5, 7.2, EAR_HEIGHT_M)), ("c", (8.0, 8.0, EAR_HEIGHT_M))],
)
def test_no_route_is_louder_than_free_field_at_its_unfolded_length(listener_zone: str, listener: Position3D) -> None:
    graph, portals = _u_graph()
    coupler = MultiPortalRirCoupler(_reverberant_adapter(), graph, world_name="u", config=_portal_config())

    result = coupler.compute(source_zone="a", listener_zone=listener_zone, source_position_m=U_SOURCE, listener_position_m=listener)

    assert result.route is not None
    assert direct_arrival(result.rir).gain_db <= _free_field_db(graph, result.route.zones, portals[: result.route.hop_count], U_SOURCE, listener)


def test_f_shape_victim_behind_two_corners_at_12_m_is_quieter_than_behind_one_at_8_m() -> None:
    graph, portals = _f_shape_graph()
    coupler = MultiPortalRirCoupler(_reverberant_adapter(), graph, world_name="f_shape", config=_portal_config())
    near_victim, near_robot = (2.12, 7.32, 1.6), (8.18, 11.02, 0.3)
    far_victim, far_robot = (1.42, 6.72, 1.6), (8.03, 14.27, 0.3)

    near = direct_arrival(coupler.compute(source_zone="zone_02", listener_zone="zone_03", source_position_m=near_victim, listener_position_m=near_robot).rir).gain_db
    far = direct_arrival(coupler.compute(source_zone="zone_02", listener_zone="zone_04", source_position_m=far_victim, listener_position_m=far_robot).rir).gain_db

    assert near <= _free_field_db(graph, ("zone_02", "zone_03"), portals[:1], near_victim, near_robot)
    assert far <= _free_field_db(graph, ("zone_02", "zone_03", "zone_04"), portals, far_victim, far_robot)
    assert far < near


def _doorway_graph(kind: str, material_id: str) -> tuple[AcousticWorldGraph, AcousticPortal]:
    span = ((6.0, 1.5), (6.0, 2.5))
    rooms = []
    for name, x_min, x_max in (("a", 0.0, 6.0), ("b", 6.0, 12.0)):
        corners = ((x_min, 0.0), (x_max, 0.0), (x_max, 4.0), (x_min, 4.0))
        boundary: list[AcousticBoundarySpec] = []
        for index, start in enumerate(corners):
            end = corners[(index + 1) % 4]
            cuts = sorted({start, end, *(point for point in span if _on_edge(point, start, end))}, key=lambda point: math.dist(start, point))
            for first, second in zip(cuts, cuts[1:], strict=False):
                doorway = _on_edge((0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1])), *span)
                boundary.append(AcousticBoundarySpec(start=first, end=second, material_id=material_id if doorway else "Acoustic_Default_Wall", kind=kind if doorway else "wall"))
        rooms.append(AcousticRoomSpec(zone_name=name, boundary=tuple(boundary), floor_material_id="Acoustic_Default_Floor", ceiling_material_id="Acoustic_Default_Ceiling", ceiling_height_m=3.0))
    portal = AcousticPortal(portal_id=f"{kind}:ab", door_name="ab" if kind == "door" else "", zone_a="a", zone_b="b", start=span[0], end=span[1], height_m=2.1, material_id=material_id, portal_kind=kind)
    return _graph(tuple(rooms), (portal,)), portal


@pytest.mark.parametrize(("kind", "material_id"), [("door", "Acoustic_Default_Wall"), ("door", "Oak_Planks"), ("opening", "Acoustic_Open")])
@pytest.mark.parametrize(("source", "listener"), [((2.0, 2.0, EAR_HEIGHT_M), (10.0, 2.0, EAR_HEIGHT_M)), ((3.0, 1.0, EAR_HEIGHT_M), (9.0, 3.0, EAR_HEIGHT_M)), ((5.0, 2.0, EAR_HEIGHT_M), (7.0, 2.0, EAR_HEIGHT_M))])
def test_door_crossing_reads_free_field_less_the_portal_loss_like_an_opening(kind: str, material_id: str, source: Position3D, listener: Position3D) -> None:
    graph, portal = _doorway_graph(kind, material_id)
    coupler = MultiPortalRirCoupler(_reverberant_adapter(), graph, world_name=f"doorway_{kind}_{material_id}", config=_portal_config())

    result = coupler.compute(source_zone="a", listener_zone="b", source_position_m=source, listener_position_m=listener)

    expected = _free_field_db(graph, ("a", "b"), (portal,), source, listener) - result.applied_portal_loss_db - result.applied_corner_loss_db
    excess = direct_arrival(result.rir).gain_db - expected
    print(f"\n{kind} {material_id} {source[:2]}->{listener[:2]}: {excess:+.2f} dB against free field less the portal and corner loss")
    assert -1.0 < excess <= 0.0
