from __future__ import annotations

import numpy as np
import pytest
from arena_auditory.render.core import ContinuousInput, RenderInputs, RenderParams, StreamInput, new_state, render_block
from arena_auditory.render.monitor import (
    MonitorConfig,
    apply_controls,
    hearing_mono,
    monitor_amplify,
    monitor_mix,
    monitor_playback,
    monitor_stereo,
    tdoa_pairs,
)
from arena_auditory.shared import ArraySpec, MicSpec, geometric_delays_s, rectangular
from arena_auditory.sources.drivetrain.program import LEFT_VELOCITY, RIGHT_VELOCITY

FOUR_MIC = ArraySpec(
    name="four_mic",
    sample_rate_hz=16000,
    block_size=320,
    sensitivity_dbfs_at_94_dbspl=-26.0,
    mics=rectangular(width_m=0.31, length_m=0.42, height_m=0.22, corner_inset_m=0.02),
)


def _config(**overrides: object) -> MonitorConfig:
    fields: dict[str, object] = {
        "enabled": True,
        "hearing": False,
        "solo": "",
        "front_gain": 1.0,
        "rear_gain": 1.0,
        "output_gain": 1.0,
        "gain_db": 0.0,
        "limit": 1.0,
    }
    fields.update(overrides)
    return MonitorConfig(**fields)


def test_stereo_fold_keeps_the_left_right_asymmetry_of_a_lateral_source() -> None:
    rate = 16000
    delays = geometric_delays_s((0.0, 5.0, 0.22), tuple(mic.position_m for mic in FOUR_MIC.mics))
    pulse = np.hanning(64).astype(np.float32)
    audio = np.zeros((4, 512), dtype=np.float32)
    for channel, delay in enumerate(delays - delays.min()):
        start = 80 + round(float(delay) * rate)
        audio[channel, start : start + len(pulse)] = pulse * (1.0 - 0.08 * channel)
    stereo = monitor_stereo(audio, FOUR_MIC, _config())
    assert stereo.shape == (2, 512)
    assert not np.array_equal(stereo[0], stereo[1])


def test_stereo_fold_weights_front_and_rear_per_ear() -> None:
    audio = np.asarray([[0.4], [0.2], [0.1], [0.05]], dtype=np.float32)
    stereo = monitor_stereo(audio, FOUR_MIC, _config(front_gain=1.0, rear_gain=0.5))
    assert stereo[0, 0] == pytest.approx((0.4 + 0.5 * 0.1) / 1.5)
    assert stereo[1, 0] == pytest.approx((0.2 + 0.5 * 0.05) / 1.5)


def test_center_microphone_feeds_both_ears() -> None:
    mono = ArraySpec(name="mono", sample_rate_hz=44100, block_size=4, sensitivity_dbfs_at_94_dbspl=-26.0, mics=(MicSpec(name="mono", position_m=(0.0, 0.0, 0.0)),))
    audio = np.asarray([[0.1, -0.2, 0.3, 0.0]], dtype=np.float32)
    stereo = monitor_stereo(audio, mono, _config())
    np.testing.assert_allclose(stereo[0], audio[0])
    np.testing.assert_allclose(stereo[1], audio[0])


def test_detection_fusion_selects_strongest_without_phase_averaging() -> None:
    audio = np.zeros((4, 32), dtype=np.float32)
    audio[0] = 0.1
    audio[2] = -0.8
    assert np.array_equal(hearing_mono(audio), audio[2])


def test_disable_mute_solo_and_reenable_controls() -> None:
    audio = np.arange(32, dtype=np.float32).reshape(4, 8)
    names = FOUR_MIC.channel_names
    assert not np.any(apply_controls(audio, enabled=False))
    assert not np.any(apply_controls(audio, muted=True))
    assert np.array_equal(apply_controls(audio, enabled=True), audio)
    solo = apply_controls(audio, solo="rear_left", names=names)
    assert np.array_equal(solo[2], audio[2])
    assert not np.any(solo[[0, 1, 3]])
    with pytest.raises(ValueError, match="solo"):
        apply_controls(audio, solo="center", names=names)


def test_monitor_gain_does_not_change_calibrated_raw_audio() -> None:
    raw = np.asarray([0.001, -0.001, 0.02], dtype=np.float32)
    untouched = raw.copy()
    monitored = monitor_amplify(raw, gain_db=40.0, limit=0.98)
    np.testing.assert_allclose(monitored[:2], [0.1, -0.1], atol=1e-6)
    assert monitored[2] == pytest.approx(0.98)
    np.testing.assert_array_equal(raw, untouched)


def test_hearing_mode_plays_the_amplified_strongest_microphone_on_both_ears() -> None:
    raw = np.zeros((4, 4), dtype=np.float32)
    raw[1] = [0.001, -0.002, 0.003, 0.0]
    raw[3] = 0.0001
    config = _config(hearing=True, gain_db=20.0, limit=0.98)
    stereo, hearing = monitor_mix(raw, FOUR_MIC, config)
    np.testing.assert_array_equal(hearing, raw[1])
    playback = monitor_playback(stereo, hearing, config)
    np.testing.assert_allclose(playback, np.repeat((raw[1] * 10.0)[None, :], 2, axis=0), atol=1e-7)
    assert not np.any(monitor_playback(*monitor_mix(raw, FOUR_MIC, _config(hearing=True, enabled=False)), _config(hearing=True, enabled=False)))


def test_tdoa_pairs_cover_left_right_then_front_rear() -> None:
    assert tdoa_pairs(FOUR_MIC) == ((0, 1, "FL-FR"), (2, 3, "RL-RR"), (0, 2, "FL-RL"), (1, 3, "FR-RR"))


def test_muted_sample_voice_stays_synchronized() -> None:
    loop = np.asarray([0.1, 0.2], dtype=np.float32)
    params = RenderParams(channels=1, block_size=3, sample_rate=16000, resolve=lambda _key: loop)
    state = new_state(1)
    voice = ContinuousInput(source_id="motor:1", channel=0, asset_key="loop", gain=1.0, delay_target=0.0, loop=True, program_start=0)
    muted = render_block(state, RenderInputs(block_index=0, clips=(), continuous=(voice,), streams=(), channels=1), params)
    np.testing.assert_allclose(apply_controls(muted.raw, muted=True)[0], [0.0] * 3)
    audible = render_block(state, RenderInputs(block_index=1, clips=(), continuous=(voice,), streams=(), channels=1), params)
    np.testing.assert_allclose(apply_controls(audible.raw)[0], [0.2, 0.1, 0.2])


def _motor_stream() -> StreamInput:
    return StreamInput(
        source_id="robot:jackal:motor",
        model="drivetrain",
        seed=3,
        params={"spec": "jackal"},
        state={LEFT_VELOCITY: 0.5, RIGHT_VELOCITY: 0.5},
        tuning={"trim_db": 0.0, "frequency_scale": 1.0, "tonal_gain_db": 0.0, "broadband_gain_db": 0.0, "speed_exponent": 1.0, "velocity_smoothing_s": 0.0},
        gains=(0.5,),
        delay_samples=(0.0,),
        active_channels=(True,),
        active=True,
        source_agent_id=0,
        source_agent_name="jackal",
        kind="motor",
        asset_id="motor",
    )


def test_muted_stream_voice_stays_synchronized() -> None:
    params = RenderParams(channels=1, block_size=64, sample_rate=16000, resolve=lambda _key: np.zeros(1, dtype=np.float32))
    reference_state, state = new_state(1), new_state(1)
    reference = [render_block(reference_state, RenderInputs(block_index=index, clips=(), continuous=(), streams=(_motor_stream(),), channels=1), params) for index in range(2)]
    muted = render_block(state, RenderInputs(block_index=0, clips=(), continuous=(), streams=(_motor_stream(),), channels=1), params)
    assert not np.any(apply_controls(muted.motor, muted=True))
    audible = render_block(state, RenderInputs(block_index=1, clips=(), continuous=(), streams=(_motor_stream(),), channels=1), params)
    assert np.max(np.abs(audible.motor)) > 0.0
    assert np.array_equal(apply_controls(audible.motor), reference[1].motor)
    assert not np.array_equal(audible.motor, reference[0].motor)
