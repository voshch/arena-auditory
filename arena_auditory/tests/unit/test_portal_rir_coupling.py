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
from arena_auditory.propagation.portal import MultiPortalRirCoupler, PortalConfig
from arena_auditory.propagation.pyroom_adapter import PyroomacousticsAdapter, RirConfig, RoomImpulseResponse, direct_arrival
from arena_auditory.rooms import AcousticBoundarySpec, AcousticPortal, AcousticRoomSpec, AcousticWorldGraph, Position3D

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "acoustic_materials.yaml"


def _portal_config(**given: object) -> PortalConfig:
    return configure(PortalConfig, PortalGroup, **given)


def _room(name: str, x_min: float, x_max: float, y_max: float = 1.0, material_id: str = "wall") -> AcousticRoomSpec:
    corners = ((x_min, 0.0), (x_max, 0.0), (x_max, y_max), (x_min, y_max))
    boundary = tuple(AcousticBoundarySpec(start=start, end=corners[(index + 1) % len(corners)], material_id=material_id, kind="wall") for index, start in enumerate(corners))
    return AcousticRoomSpec(zone_name=name, boundary=boundary, floor_material_id="floor", ceiling_material_id="ceiling", ceiling_height_m=3.0)


def _path_scale(graph: AcousticWorldGraph, route_zones: tuple[str, ...], portals: tuple[AcousticPortal, ...], source: Position3D, listener: Position3D, inset_m: float = 0.03) -> float:
    points = [source]
    gap = 0.0
    for index, portal in enumerate(portals):
        height = min(0.5 * portal.height_m, 2.99)
        before = graph.position_inside_portal(portal, route_zones[index], inset_m=inset_m, height_m=height)
        after = graph.position_inside_portal(portal, route_zones[index + 1], inset_m=inset_m, height_m=height)
        points.extend((before, after))
        gap += math.dist(before, after)
    points.append(listener)
    lengths = [math.dist(points[index], points[index + 1]) for index in range(0, len(points), 2)]
    return math.prod(lengths) / (sum(lengths) + gap)


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
