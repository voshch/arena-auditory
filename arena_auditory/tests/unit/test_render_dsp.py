from __future__ import annotations

import math

import numpy as np
import pytest
from arena_auditory.render.core import prepare_clip
from arena_auditory.render.dsp import calibrate_mems, fractional_delay, interleave, ramped_read, resample_impulse, streaming_fractional_delays
from arena_robots.audio import active_rms, dbfs_from_rms


def _tone() -> np.ndarray:
    return np.sin(np.linspace(0.0, 4.0 * math.pi, 1600, endpoint=False)).astype(np.float32)


def test_mems_calibration_reads_94_db_spl_sine_at_minus_26_dbfs() -> None:
    tone = _tone()
    calibrated = calibrate_mems(tone, 94.0, active_rms(tone, 16000))
    assert dbfs_from_rms(float(np.sqrt(np.mean(calibrated.astype(np.float64) ** 2)))) == pytest.approx(-26.0, abs=0.02)
    assert np.array_equal(calibrate_mems(np.zeros(64, dtype=np.float32), 80.0, 0.0), np.zeros(64, dtype=np.float32))


def test_mems_calibration_peaks_a_120_db_spl_sine_at_full_scale() -> None:
    tone = _tone()
    peak = float(np.max(np.abs(calibrate_mems(tone, 120.0, active_rms(tone, 16000)))))
    assert peak == pytest.approx(1.0, abs=1e-3)
    assert peak <= 1.0 + 1e-6


def test_padding_a_clip_with_silence_keeps_its_calibrated_level() -> None:
    rng = np.random.default_rng(3)
    burst = (rng.standard_normal(700) * np.hanning(700)).astype(np.float32)
    padded = np.concatenate((np.zeros(5000, dtype=np.float32), burst, np.zeros(13000, dtype=np.float32)))
    _, short = prepare_clip(burst, 70.0, -26.0, 0.0, level_rms=active_rms(burst, 16000))
    _, long = prepare_clip(padded, 70.0, -26.0, 0.0, level_rms=active_rms(padded, 16000))
    assert active_rms(padded, 16000) == active_rms(burst, 16000)
    assert np.array_equal(long[5000 : 5000 + burst.size], short)


def test_fractional_delay_is_not_rounded_away() -> None:
    integer, delayed = fractional_delay(np.asarray([1.0, 0.0], dtype=np.float32), 3.25)
    assert integer == 3
    assert delayed[0] == pytest.approx(0.75)
    assert delayed[1] == pytest.approx(0.25)


def test_fractional_delay_rejects_a_negative_delay() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        fractional_delay(np.asarray([1.0], dtype=np.float32), -0.5)


def test_streaming_fractional_delays_retain_history_between_blocks() -> None:
    first, history = streaming_fractional_delays(
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.asarray([0.0, 1.5]),
    )
    np.testing.assert_allclose(first[0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(first[1], [0.0, 0.5, 1.5, 2.5])

    second, _ = streaming_fractional_delays(
        np.asarray([5.0, 6.0], dtype=np.float32),
        np.asarray([0.0, 1.5]),
        history,
    )
    np.testing.assert_allclose(second[0], [5.0, 6.0])
    np.testing.assert_allclose(second[1], [3.5, 4.5])


def test_ramped_read_constant_delay_is_a_plain_shift() -> None:
    loop = np.arange(8, dtype=np.float32)
    out = ramped_read(loop, 0.0, 3.0, 3.0, 8, loop=True)
    np.testing.assert_allclose(out, [5, 6, 7, 0, 1, 2, 3, 4])
    clip = ramped_read(loop, 0.0, 3.0, 3.0, 8, loop=False)
    np.testing.assert_allclose(clip, [0, 0, 0, 0, 1, 2, 3, 4])


def test_ramped_read_is_continuous_across_blocks_and_reaches_the_target() -> None:
    fs = 16000
    tone = np.sin(2.0 * np.pi * 440.0 * np.arange(fs, dtype=np.float64) / fs).astype(np.float32)
    block = 320
    delay, target = 100.0, 103.0
    blocks = []
    for index in range(3):
        blocks.append(ramped_read(tone, index * block, delay, target, block, loop=True))
        delay = target
    signal = np.concatenate(blocks)
    steps = np.abs(np.diff(signal))
    assert steps.max() < 1.05 * np.abs(np.diff(tone[: 2 * block])).max()
    settled = ramped_read(tone, 3 * block, target, target, block, loop=True)
    expected = tone[(np.arange(3 * block, 4 * block) - 103) % tone.size]
    np.testing.assert_allclose(settled, expected, atol=1e-6)


def test_resample_impulse_keeps_the_filter_gain_and_skips_matching_rates() -> None:
    impulse = np.zeros(480, dtype=np.float32)
    impulse[120] = 1.0
    assert np.array_equal(resample_impulse(impulse, 16000, 16000), impulse)
    resampled = resample_impulse(impulse, 48000, 16000)
    assert resampled.size == 160
    assert float(np.sum(resampled)) == pytest.approx(1.0, rel=0.05)
    with pytest.raises(ValueError, match="positive"):
        resample_impulse(impulse, 0, 16000)


def test_interleave_writes_frames_channel_by_channel() -> None:
    channels = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert list(interleave(channels)) == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
