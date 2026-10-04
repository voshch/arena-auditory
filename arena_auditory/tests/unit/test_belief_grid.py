from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy.ndimage import maximum_filter1d

from arena_auditory.assets import SoundLibrary
from arena_auditory.hearing.belief_grid import SPEED_MASK_NO_LIMIT, BeliefConfig, BeliefGrid, dilate_belief, emission_levels, speed_mask_from_belief
from arena_auditory.hearing.policy import PolicyConfig
from arena_auditory.params import BeliefGroup, PolicyGroup, configure


def _config(**given: object) -> BeliefConfig:
    return configure(BeliefConfig, BeliefGroup, emission_db=emission_levels(SoundLibrary.default()), **given)


def _policy(**given: object) -> PolicyConfig:
    return configure(PolicyConfig, PolicyGroup, **given)


def _grid(**given: object) -> BeliefGrid:
    return BeliefGrid(origin_x=-10.0, origin_y=-10.0, resolution=0.1, width=200, height=200, config=_config(**given))


def test_event_paints_mass_along_the_map_frame_bearing() -> None:
    grid = _grid()
    grid.add_event(0.0, 0.0, 0.0, math.pi / 2, sound_type="footstep", bearing_frame="map")
    x, y, mass = grid.argmax_world()
    assert mass > 0.0
    assert abs(x) < 0.5
    assert y > 0.5


def test_robot_frame_bearing_is_rotated_by_yaw() -> None:
    grid = _grid()
    grid.add_event(0.0, 0.0, math.pi / 2, 0.0, sound_type="footstep", bearing_frame="robot")
    x, y, _ = grid.argmax_world()
    assert abs(x) < 0.5
    assert y > 0.5


def test_decay_and_clear_drain_the_grid() -> None:
    grid = _grid(tau_s=1.0)
    grid.add_event(0.0, 0.0, 0.0, 0.0, sound_type="footstep", bearing_frame="map")
    before = grid.mass.sum()
    grid.decay(1.0)
    assert grid.mass.sum() == pytest.approx(before * math.exp(-1.0), rel=1e-4)
    grid.clear()
    assert grid.mass.sum() == 0.0
    assert grid.argmax_world()[2] == 0.0


def test_wall_blocks_mass_but_wedge_crosses_into_the_far_corridor() -> None:
    free = np.ones((40, 40), dtype=bool)
    free[:, 20:23] = False
    grid = BeliefGrid(origin_x=0.0, origin_y=0.0, resolution=0.1, width=40, height=40, config=_config(), free=free)
    grid.add_event(0.5, 2.0, 0.0, 0.0, sound_type="footstep", bearing_frame="map")
    assert grid.mass[:, 20:23].sum() == 0.0
    assert grid.mass[:, :20].sum() > 0.0
    assert grid.mass[:, 23:].sum() > 0.0

    open_grid = BeliefGrid(origin_x=0.0, origin_y=0.0, resolution=0.1, width=40, height=40, config=_config())
    open_grid.add_event(0.5, 2.0, 0.0, 0.0, sound_type="footstep", bearing_frame="map")
    assert grid.mass.sum() < open_grid.mass.sum()


def test_speed_mask_limits_within_the_reaction_radius_only() -> None:
    grid = _grid(event_rate_hz=1.0, tau_s=1.0)
    policy = _policy(reaction_radius_m=1.0, belief_threshold=0.1)
    grid.add_event(0.0, 0.0, 0.0, 0.0, sound_type="footstep", bearing_frame="map", weight=1.0)
    mask = grid.speed_mask_int8(policy)
    px, py, _ = grid.argmax_world()
    row, col = grid.world_to_cell(px, py)
    assert policy.speed_min_pct <= mask[row, col] < policy.speed_free_pct
    far_row, far_col = grid.world_to_cell(px, py - 5.0)
    assert mask[far_row, far_col] == SPEED_MASK_NO_LIMIT
    assert mask.dtype == np.int8


def test_speed_mask_falls_from_free_to_min_percentage_over_the_threshold_span() -> None:
    policy = _policy(reaction_radius_m=0.0, belief_threshold=0.5, speed_min_pct=40, speed_free_pct=100)
    belief = np.array([[0.0, 0.5, 0.75, 1.0]], dtype=np.float32)
    assert speed_mask_from_belief(belief, 0.1, policy).tolist() == [[0, 0, 70, 40]]


def _full_grid_dilation(belief: np.ndarray, resolution: float, reaction_radius_m: float) -> np.ndarray:
    k = int(round(float(reaction_radius_m) / resolution))
    if k <= 0:
        return belief
    out = np.zeros_like(belief)
    h = belief.shape[0]
    for dy in range(-k, k + 1):
        half = int(math.floor(math.sqrt(k * k - dy * dy)))
        row = maximum_filter1d(belief, 2 * half + 1, axis=1, mode="constant", cval=0.0)
        src0, src1 = max(0, -dy), min(h, h - dy)
        if src1 > src0:
            np.maximum(out[src0 + dy : src1 + dy], row[src0:src1], out=out[src0 + dy : src1 + dy])
    return out


@pytest.mark.parametrize("seed", range(8))
def test_dilate_belief_matches_the_full_grid_filter(seed: int) -> None:
    rng = np.random.default_rng(seed)
    h, w = rng.integers(5, 90, size=2)
    belief = np.zeros((h, w), dtype=np.float32)
    n = int(rng.integers(0, 6))
    belief[rng.integers(0, h, n), rng.integers(0, w, n)] = rng.random(n).astype(np.float32)
    if seed % 2:
        belief[rng.random((h, w)) < 0.05] = rng.random() * 0.5
    for radius in (0.0, 0.1, 0.35, 1.0, 2.0):
        np.testing.assert_array_equal(dilate_belief(belief, 0.1, radius), _full_grid_dilation(belief, 0.1, radius))


def test_dilate_belief_of_an_empty_grid_is_zero() -> None:
    belief = np.zeros((30, 40), dtype=np.float32)
    assert not dilate_belief(belief, 0.1, 1.0).any()


def _full_disc_paint(grid: BeliefGrid, robot_x: float, robot_y: float, theta: float, sound_type: str, received_db: float | None) -> tuple[np.ndarray, int]:
    p = grid.config
    rng, level_derived = grid.range_estimate(sound_type, received_db)
    half = math.radians(float(p.wedge_deg)) * 0.5
    tan_half = math.tan(min(half, math.radians(89.0)))
    min_hw = max(float(p.min_half_width_m), 0.0)
    sigma_rad = max(float(p.range_sigma_frac) * rng, grid.resolution)
    r_paint = min(float(p.max_range_m), rng + 2.5 * sigma_rad) if level_derived else rng
    mass = np.zeros_like(grid.mass)
    r0, c0 = grid.world_to_cell(robot_x - r_paint, robot_y - r_paint)
    r1, c1 = grid.world_to_cell(robot_x + r_paint, robot_y + r_paint)
    r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1 + 1, grid.height), min(c1 + 1, grid.width)
    if r1 <= r0 or c1 <= c0:
        return mass, 0
    dx = grid._xs[c0:c1][None, :] - robot_x
    dy = grid._ys[r0:r1][:, None] - robot_y
    rr = np.hypot(dx, dy)
    fwd = dx * math.cos(theta) + dy * math.sin(theta)
    lat = np.abs(-dx * math.sin(theta) + dy * math.cos(theta))
    hw = np.maximum(fwd * tan_half, min_hw)
    inside = (rr <= r_paint) & (fwd >= 0.0) & (lat <= hw)
    if not inside.any():
        return mass, 0
    w_ang = np.exp(-0.5 * (lat / np.maximum(hw * 0.5, 1e-3)) ** 2)
    w_rad = np.exp(-0.5 * ((rr - rng) / sigma_rad) ** 2) + float(p.range_floor_weight) * (rr <= rng) if level_derived else np.ones_like(rr)
    w = np.where(inside, w_ang * w_rad, 0.0)
    w *= float(p.event_mass) / float(w.max())
    w *= grid.free[r0:r1, c0:c1]
    mass[r0:r1, c0:c1] = w.astype(np.float32)
    return mass, int(inside.sum())


@pytest.mark.parametrize("seed", range(12))
def test_add_event_paints_the_same_mass_as_the_full_disc(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    config = _config(
        wedge_deg=float(rng.choice([0.0, 10.0, 45.0, 120.0, 200.0])),
        min_half_width_m=float(rng.choice([0.0, 0.3, 1.0])),
        max_range_m=float(rng.choice([3.0, 15.0])),
        level_range_enabled=bool(seed % 2),
    )
    free = rng.random((120, 160)) > 0.1
    grid = BeliefGrid(origin_x=-6.0, origin_y=-4.0, resolution=0.1, width=160, height=120, config=config, free=free)
    for _ in range(6):
        x, y = float(rng.uniform(-8.0, 12.0)), float(rng.uniform(-6.0, 10.0))
        theta = float(rng.uniform(-math.pi, math.pi))
        if rng.random() < 0.3:
            theta = float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2]))
        received = float(rng.uniform(20.0, 45.0))
        grid.clear()
        info = grid.add_event(x, y, 0.0, theta, sound_type="footstep", received_db=received, bearing_frame="map")
        expected, painted = _full_disc_paint(grid, x, y, theta, "footstep", received)
        assert info["painted"] == painted
        np.testing.assert_allclose(grid.mass, expected, rtol=1e-5, atol=1e-6)


def test_level_range_inverts_the_spreading_law_against_the_kind_emission_level() -> None:
    grid = _grid(level_range_enabled=True, reference_distance_m=1.0, min_range_m=0.5, max_range_m=15.0)
    emission = grid.config.emission_for("footstep")
    assert math.isfinite(emission)
    rng, level_derived = grid.range_estimate("footstep", emission - 20.0)
    assert level_derived
    assert rng == pytest.approx(10.0)
    assert grid.range_estimate("footstep", None) == (15.0, False)
    assert grid.range_estimate("not_a_kind", emission) == (15.0, False)
    assert _grid().range_estimate("footstep", emission - 20.0) == (15.0, False)
