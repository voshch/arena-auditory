from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from arena_rclpy_mixins.param_groups import configure

from arena_auditory.materials import AcousticMaterialCatalog
from arena_auditory.params import RirGroup
from arena_auditory.propagation.pyroom_adapter import PyroomacousticsAdapter, RirConfig, RirUnavailable, direct_arrival
from arena_auditory.rooms import AcousticBoundarySpec, AcousticRoomSpec

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "acoustic_materials.yaml"


def _room(size: float = 2.0) -> AcousticRoomSpec:
    corners = ((0.0, 0.0), (size, 0.0), (size, size), (0.0, size))
    boundary = tuple(
        AcousticBoundarySpec(start=start, end=end, material_id="Acoustic_Default_Wall", kind="wall")
        for start, end in zip(corners, (*corners[1:], corners[0]), strict=True)
    )
    return AcousticRoomSpec(
        zone_name="room",
        boundary=boundary,
        floor_material_id="Acoustic_Default_Floor",
        ceiling_material_id="Acoustic_Default_Ceiling",
        ceiling_height_m=3.0,
    )


def _adapter(**given: object) -> PyroomacousticsAdapter:
    return PyroomacousticsAdapter(AcousticMaterialCatalog(CATALOG_PATH), configure(RirConfig, RirGroup, **given))


def _anechoic_adapter() -> PyroomacousticsAdapter:
    pytest.importorskip("pyroomacoustics")
    return _adapter(max_order=0)


def test_near_boundary_microphone_is_moved_inside_room() -> None:
    adjusted = _adapter()._position_inside_room(_room(), (-0.23, 0.5, 0.22), name="listener_position_m")

    assert adjusted[0] == pytest.approx(0.01)
    assert adjusted[1:] == pytest.approx((0.5, 0.22))


def test_position_well_outside_room_is_rejected() -> None:
    with pytest.raises(RirUnavailable, match="outside acoustic room"):
        _adapter()._position_inside_room(_room(), (-0.36, 0.5, 0.22), name="listener_position_m")


def test_cache_hit_in_same_bucket_returns_exact_position_delay_and_level() -> None:
    adapter = _anechoic_adapter()
    room = _room(5.0)
    source = (1.0, 2.5, 1.0)
    first_listener = (3.46, 2.5, 1.0)
    second_listener = (3.54, 2.5, 1.0)

    first = direct_arrival(adapter.compute_rir(room, source_position_m=source, listener_position_m=first_listener))
    second = direct_arrival(adapter.compute_rir(room, source_position_m=source, listener_position_m=second_listener))

    assert adapter.cache_hits == 1
    speed = adapter.speed_of_sound_mps
    sample_period = 1.0 / adapter.config.sample_rate_hz
    first_distance = math.dist(source, first_listener)
    second_distance = math.dist(source, second_listener)
    assert first.delay_s == pytest.approx(first_distance / speed, abs=0.6 * sample_period)
    assert second.delay_s - first.delay_s == pytest.approx((second_distance - first_distance) / speed, abs=0.6 * sample_period)
    assert second.gain_db - first.gain_db == pytest.approx(20.0 * math.log10(first_distance / second_distance), abs=0.05)


def test_reused_built_room_matches_a_fresh_room() -> None:
    adapter = _anechoic_adapter()
    room = _room(5.0)
    adapter.compute_rir(room, source_position_m=(1.0, 1.0, 1.0), listener_position_m=(4.0, 4.0, 1.5))
    reused = adapter.compute_rir(room, source_position_m=(2.0, 3.0, 1.2), listener_position_m=(4.0, 1.0, 0.5))
    fresh = _anechoic_adapter().compute_rir(room, source_position_m=(2.0, 3.0, 1.2), listener_position_m=(4.0, 1.0, 0.5))

    assert adapter.cache_misses == 2
    np.testing.assert_allclose(reused.samples, fresh.samples)


def test_microphone_at_ceiling_height_is_kept_inside_room() -> None:
    adapter = _anechoic_adapter()
    room = _room(5.0)

    rir = adapter.compute_rir(room, source_position_m=(1.0, 2.5, 1.0), listener_position_m=(4.0, 2.5, room.ceiling_height_m))

    assert direct_arrival(rir).delay_s == pytest.approx(
        math.dist((1.0, 2.5, 1.0), (4.0, 2.5, room.ceiling_height_m)) / adapter.speed_of_sound_mps,
        abs=1.0 / adapter.config.sample_rate_hz,
    )
