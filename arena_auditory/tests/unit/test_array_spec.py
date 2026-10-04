from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from arena_auditory.shared import PRESETS, ArraySpec, geometric_delays_s, load_array_spec, rectangular, transform


def _delays(source: tuple[float, float, float]) -> np.ndarray:
    spec = load_array_spec("four_mic")
    return geometric_delays_s(source, tuple(mic.position_m for mic in spec.mics))


def test_four_mic_geometry_matches_jackal_chassis_inset() -> None:
    spec = load_array_spec("four_mic")
    assert spec.channel_names == ("front_left", "front_right", "rear_left", "rear_right")
    assert spec.mics[0].position_m == pytest.approx((0.19, 0.135, 0.22))
    assert spec.mics[1].position_m == pytest.approx((0.19, -0.135, 0.22))
    assert spec.mics[2].position_m == pytest.approx((-0.19, 0.135, 0.22))
    assert spec.mics[3].position_m == pytest.approx((-0.19, -0.135, 0.22))
    assert [round(math.degrees(mic.yaw_rad)) for mic in spec.mics] == [45, -45, 135, -135]
    assert (spec.sample_rate_hz, spec.block_size, spec.sensitivity_dbfs_at_94_dbspl) == (16000, 320, -26.0)


def test_mono_and_stereo_presets() -> None:
    mono = load_array_spec("mono")
    stereo = load_array_spec("stereo")
    assert mono.channel_names == ("mono",)
    assert mono.mics[0].position_m == (0.0, 0.0, 0.0)
    assert stereo.channel_names == ("left", "right")
    assert [mic.side for mic in stereo.mics] == ["left", "right"]
    assert stereo.mics[0].position_m[1] - stereo.mics[1].position_m[1] == pytest.approx(0.20)
    assert all(mic.position_m[2] == pytest.approx(0.35) for mic in stereo.mics)
    assert {(spec.sample_rate_hz, spec.block_size) for spec in (mono, stereo)} == {(44100, 512)}
    assert set(PRESETS) == {"mono", "stereo", "four_mic"}


def test_a_custom_array_loads_from_a_yaml_path(tmp_path: Path) -> None:
    path = tmp_path / "triangle.yaml"
    path.write_text(
        "name: triangle\n"
        "sample_rate_hz: 48000\n"
        "block_size: 480\n"
        "sensitivity_dbfs_at_94_dbspl: -38.0\n"
        "mics:\n"
        "  - {name: nose, position_m: [0.3, 0.0, 0.5], yaw_deg: 0, group: front}\n"
        "  - {name: port, position_m: [-0.1, 0.2, 0.5], yaw_deg: 90, side: left, group: rear}\n"
        "  - {name: starboard, position_m: [-0.1, -0.2, 0.5], yaw_deg: -90, side: right, group: rear}\n"
    )
    spec = load_array_spec(str(path))
    assert spec.name == "triangle"
    assert spec.channels == 3
    assert spec.mics[1].yaw_rad == pytest.approx(math.pi / 2.0)
    assert spec.mics[0].side == "center"
    assert spec.centroid_m == pytest.approx((0.1 / 3.0, 0.0, 0.5))


def test_array_spec_rejects_duplicate_names_and_unknown_refs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ArraySpec.from_dict(
            {
                "name": "twins",
                "sample_rate_hz": 16000,
                "block_size": 320,
                "sensitivity_dbfs_at_94_dbspl": -26.0,
                "mics": [{"name": "a"}, {"name": "a"}],
            }
        )
    with pytest.raises(FileNotFoundError, match="preset"):
        load_array_spec(str(tmp_path / "missing.yaml"))


def test_rectangular_inset_must_leave_an_aperture() -> None:
    with pytest.raises(ValueError, match="inset"):
        rectangular(width_m=0.3, length_m=0.4, height_m=0.2, corner_inset_m=0.15)


def test_array_moves_rigidly_with_robot() -> None:
    mics = load_array_spec("four_mic").mics
    world = transform(mics, position_m=(2.0, -1.0, 0.1), yaw_rad=math.pi / 2.0)
    assert np.allclose(world[0], (2.0 - 0.135, -1.0 + 0.19, 0.32))
    local = np.asarray([mic.position_m for mic in mics])
    original_distances = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=2)
    moved = np.asarray(world)
    moved_distances = np.linalg.norm(moved[:, None, :] - moved[None, :, :], axis=2)
    assert np.allclose(original_distances, moved_distances)


def test_left_right_and_front_rear_geometric_arrivals() -> None:
    left = _delays((0.0, 5.0, 0.22))
    right = _delays((0.0, -5.0, 0.22))
    front = _delays((5.0, 0.0, 0.22))
    rear = _delays((-5.0, 0.0, 0.22))
    assert max(left[0], left[2]) < min(left[1], left[3])
    assert max(right[1], right[3]) < min(right[0], right[2])
    assert max(front[0], front[1]) < min(front[2], front[3])
    assert max(rear[2], rear[3]) < min(rear[0], rear[1])
    assert (left[1] - left[0]) == pytest.approx(0.000787, abs=2e-6)
    assert (front[2] - front[0]) == pytest.approx(0.001107, abs=2e-6)
