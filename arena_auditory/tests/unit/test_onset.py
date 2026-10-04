from __future__ import annotations

import math

import numpy as np

from arena_auditory.hearing.onset import OnsetDetector
from arena_auditory.params import SrpGroup

FS = 16000
HOP_S = 0.1
HOP_N = int(HOP_S * FS)


def _detector(**given: float) -> OnsetDetector:
    srp = SrpGroup.defaults()
    settings = {"hop_s": HOP_S, "floor_window_s": srp["floor_window_s"], "onset_db": srp["onset_db"], **given}
    return OnsetDetector(FS, **settings)


def _frame(amplitude: float, n: int = HOP_N, channels: int = 1) -> np.ndarray:
    return np.full((n, channels), amplitude, dtype=np.float64)


def test_silence_never_fires() -> None:
    det = _detector()
    for _ in range(30):
        fired, _peak_db, _floor_db = det.step(_frame(0.0))
        assert fired is False


def test_burst_fires_once_floor_established() -> None:
    det = _detector(onset_db=6.0)
    for _ in range(12):
        fired, _, _ = det.step(_frame(1.0))
        assert fired is False
    fired, peak_db, floor_db = det.step(_frame(10.0))
    assert fired is True
    assert peak_db - floor_db >= 6.0


def test_burst_before_floor_established_does_not_fire() -> None:
    det = _detector(onset_db=6.0)
    for _ in range(5):
        fired, _, _ = det.step(_frame(1.0))
        assert fired is False
    fired, _, _ = det.step(_frame(100.0))
    assert fired is False


def test_floor_tracks_level_change() -> None:
    det = _detector(floor_window_s=5.0, onset_db=6.0)
    for _ in range(60):
        _, low_peak_db, floor_db = det.step(_frame(1.0))
    assert abs(floor_db - low_peak_db) < 0.01
    for _ in range(60):
        _, high_peak_db, floor_db = det.step(_frame(3.0))
    assert abs(floor_db - high_peak_db) < 0.01
    assert high_peak_db - low_peak_db > 5.0


def _walking_pedestrian(total_s: float, step_interval_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal((int(total_s * FS), 4)) * 0.002
    burst_n = int(0.05 * FS)
    envelope = np.exp(-np.arange(burst_n) / (0.01 * FS))
    t = 0.3
    while t + 0.05 < total_s:
        start = int(t * FS)
        audio[start : start + burst_n] += (rng.standard_normal(burst_n) * 0.3 * envelope)[:, None]
        t += step_interval_s
    return audio


def test_srp_onsets_of_a_walking_pedestrian_hold_belief_above_the_policy_threshold() -> None:
    from arena_auditory.assets import SoundLibrary
    from arena_auditory.hearing.belief_grid import NOMINAL_EVENT_RATE_HZ, BeliefConfig, BeliefGrid, dilate_belief, emission_levels
    from arena_auditory.params import BeliefGroup, PolicyGroup, configure

    policy = PolicyGroup.defaults()
    total_s = 12.0
    audio = _walking_pedestrian(total_s, 0.45, seed=7)
    det = _detector()
    config = configure(BeliefConfig, BeliefGroup, emission_db=emission_levels(SoundLibrary.default()), event_rate_hz=NOMINAL_EVENT_RATE_HZ["srp"])
    grid = BeliefGrid(-5.0, -5.0, 0.1, 100, 100, config)
    robot_row, robot_col = grid.world_to_cell(0.0, 0.0)
    fired_total = 0
    tail: list[float] = []
    for k in range(int(total_s / HOP_S)):
        fired, _, _ = det.step(audio[k * HOP_N : (k + 1) * HOP_N])
        grid.decay(HOP_S)
        if fired:
            fired_total += 1
            grid.add_event(0.0, 0.0, 0.0, math.radians(30.0), sound_type="onset", bearing_frame="robot")
        if (k + 1) * HOP_S > total_s - 3.0:
            dilated = dilate_belief(grid.normalized(), grid.resolution, float(policy["reaction_radius_m"]))
            tail.append(float(dilated[robot_row, robot_col]))
    rate_hz = fired_total / (total_s - 10 * HOP_S)
    assert 0.8 * NOMINAL_EVENT_RATE_HZ["srp"] <= rate_hz <= 2.5 * NOMINAL_EVENT_RATE_HZ["srp"], rate_hz
    assert min(tail) > float(policy["belief_threshold"]), tail
