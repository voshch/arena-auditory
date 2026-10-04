from __future__ import annotations

import math

import numpy as np
import pytest

from arena_auditory.propagation.pyroom_adapter import RoomImpulseResponse, direct_arrival

SAMPLE_RATE_HZ = 44100
FILTER_TAPS = 81
GLOBAL_DELAY = FILTER_TAPS // 2


def _windowed_sinc(fraction: float) -> np.ndarray:
    return np.hanning(FILTER_TAPS) * np.sinc(np.arange(FILTER_TAPS) - GLOBAL_DELAY - fraction)


def _rir(*arrivals: tuple[float, float]) -> RoomImpulseResponse:
    samples = np.zeros(2048, dtype=np.float64)
    for delay_samples, amplitude in arrivals:
        start = int(math.floor(delay_samples))
        samples[start : start + FILTER_TAPS] += amplitude * _windowed_sinc(delay_samples - start)
    return RoomImpulseResponse(
        samples=samples,
        sample_rate_hz=SAMPLE_RATE_HZ,
        global_delay_samples=GLOBAL_DELAY,
        fallback_material_ids=(),
    )


@pytest.mark.parametrize("delay_samples", [300.0, 300.5])
def test_direct_arrival_delay_excludes_filter_delay(delay_samples: float) -> None:
    arrival = direct_arrival(_rir((delay_samples, 0.5)))

    assert arrival.delay_s * SAMPLE_RATE_HZ == pytest.approx(delay_samples, abs=0.05)


def test_direct_arrival_level_is_independent_of_fractional_delay() -> None:
    on_sample = direct_arrival(_rir((300.0, 0.5)))
    between_samples = direct_arrival(_rir((300.5, 0.5)))

    assert on_sample.gain_db == pytest.approx(20.0 * math.log10(0.5), abs=0.05)
    assert between_samples.gain_db == pytest.approx(on_sample.gain_db, abs=0.2)


def test_direct_arrival_ignores_sinc_pre_sidelobe() -> None:
    arrival = direct_arrival(_rir((300.5, 1.0)))

    assert arrival.index == 300 + GLOBAL_DELAY
    assert arrival.delay_s * SAMPLE_RATE_HZ == pytest.approx(300.5, abs=0.1)


def test_direct_arrival_precedes_louder_reflection() -> None:
    arrival = direct_arrival(_rir((300.0, 0.4), (360.0, 1.0)))

    assert arrival.delay_s * SAMPLE_RATE_HZ == pytest.approx(300.0, abs=0.05)
    assert arrival.gain_db == pytest.approx(20.0 * math.log10(0.4), abs=0.1)
