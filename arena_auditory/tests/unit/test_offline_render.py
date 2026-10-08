from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from arena_auditory import offline_render
from arena_auditory.render.core import (
    ClipInput,
    ContinuousInput,
    ImpulseShape,
    RenderInputs,
    RenderParams,
    StreamInput,
    new_state,
    prepare_clip,
    render_block,
    render_inputs_to_json,
)
from arena_robots.audio import active_rms
from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary
from scipy.io import wavfile

CHANNELS = 4
BLOCK_SIZE = 64
SAMPLE_RATE = 16000

TUNING = {
    "trim_db": -20.0,
    "frequency_scale": 1.0,
    "tonal_gain_db": 0.0,
    "broadband_gain_db": -12.0,
    "speed_exponent": 1.5,
    "velocity_smoothing_s": 0.015,
}


def asset() -> np.ndarray:
    rng = np.random.default_rng(20240917)
    return np.ascontiguousarray((rng.standard_normal(SAMPLE_RATE) * 0.1).astype(np.float32))


def level(_key: str) -> float:
    return active_rms(asset(), SAMPLE_RATE)


def params() -> RenderParams:
    samples = asset()
    return RenderParams(channels=CHANNELS, block_size=BLOCK_SIZE, sample_rate=SAMPLE_RATE, resolve=lambda _key: samples)


def block(index: int, *, gain: float = 0.5, velocity: float = 0.3) -> RenderInputs:
    return RenderInputs(
        block_index=index,
        clips=(),
        continuous=(
            ContinuousInput(
                source_id="fan",
                channel=1,
                asset_key="fan_a",
                gain=gain,
                delay_target=2.0 + 0.25 * index,
                loop=True,
                program_start=0,
            ),
        ),
        streams=(
            StreamInput(
                source_id="robot",
                model="drivetrain",
                seed=11,
                params={"spec": "jackal"},
                state={"left_velocity_mps": velocity, "right_velocity_mps": velocity + 0.1},
                tuning=TUNING,
                gains=(0.5, 0.4, 0.3, 0.2),
                delay_samples=(1.5, 2.5, 3.5, 4.5),
                active_channels=(True, True, True, True),
                active=True,
                source_agent_id=1,
                source_agent_name="robot",
                kind="motor",
                asset_id="motor",
            ),
        ),
        channels=CHANNELS,
    )


def trace(indices: list[int], **kwargs: float) -> list[RenderInputs]:
    return [block(index, **kwargs) for index in indices]


def test_thread_pin_is_set_by_import() -> None:
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_replay_matches_a_direct_render_block_loop() -> None:
    blocks = trace(list(range(24)))

    expected_params = params()
    expected_state = new_state(CHANNELS)
    expected = [render_block(expected_state, item, expected_params) for item in blocks]

    replayed = list(offline_render.replay(iter(blocks), params()))

    assert [item.block_index for item, _ in replayed] == [item.block_index for item in blocks]
    for (_, result), reference in zip(replayed, expected, strict=True):
        assert np.array_equal(result.raw, reference.raw)
        assert np.array_equal(result.motor, reference.motor)
        assert result.clipped_samples == reference.clipped_samples


def test_replay_carries_state_between_blocks() -> None:
    blocks = trace(list(range(8)))
    stateless = [render_block(new_state(CHANNELS), item, params()) for item in blocks]
    replayed = [result for _, result in offline_render.replay(iter(blocks), params())]

    assert np.array_equal(replayed[0].raw, stateless[0].raw)
    assert not np.array_equal(replayed[-1].raw, stateless[-1].raw)


def test_replay_reanchors_the_cursor_from_the_block_index() -> None:
    blocks = trace([0, 1, 7, 8])
    state = new_state(CHANNELS)
    replayed = list(offline_render.replay(iter(blocks), params(), state=state))

    assert state.cursor == 9 * BLOCK_SIZE
    assert len(replayed) == 4

    expected_state = new_state(CHANNELS)
    expected = []
    for item in blocks:
        expected_state.cursor = item.block_index * BLOCK_SIZE
        expected.append(render_block(expected_state, item, params()))
    for (_, result), reference in zip(replayed, expected, strict=True):
        assert np.array_equal(result.raw, reference.raw)


def test_replay_starts_fresh_state_at_a_reset_block() -> None:
    blocks = trace([0, 1, 2])
    blocks[2] = RenderInputs(block_index=2, clips=(), continuous=blocks[2].continuous, streams=blocks[2].streams, channels=CHANNELS, reset=True)
    replayed = [result for _, result in offline_render.replay(iter(blocks), params())]

    fresh = new_state(CHANNELS)
    fresh.cursor = 2 * BLOCK_SIZE
    assert np.array_equal(replayed[2].raw, render_block(fresh, blocks[2], params()).raw)


def test_render_results_concatenates_blocks_in_order() -> None:
    blocks = trace(list(range(5)))
    results = list(offline_render.replay(iter(blocks), params()))
    audio = offline_render.render_results(results)

    assert audio.shape == (CHANNELS, 5 * BLOCK_SIZE)
    assert np.array_equal(audio[:, :BLOCK_SIZE], results[0][1].raw)
    assert np.array_equal(audio[:, -BLOCK_SIZE:], results[-1][1].raw)


def test_render_results_places_blocks_across_a_skip_gap() -> None:
    blocks = trace([0, 1, 7, 8])
    results = list(offline_render.replay(iter(blocks), params()))
    audio = offline_render.render_results(results)

    assert audio.shape == (CHANNELS, 9 * BLOCK_SIZE)
    for inputs, result in results:
        start = inputs.block_index * BLOCK_SIZE
        assert np.array_equal(audio[:, start : start + BLOCK_SIZE], result.raw)
    assert not np.any(audio[:, 2 * BLOCK_SIZE : 7 * BLOCK_SIZE])


def test_render_results_starts_at_the_first_traced_block() -> None:
    results = list(offline_render.replay(iter(trace([5, 6, 9])), params()))
    audio = offline_render.render_results(results)

    assert audio.shape == (CHANNELS, 5 * BLOCK_SIZE)
    assert np.array_equal(audio[:, :BLOCK_SIZE], results[0][1].raw)
    assert np.array_equal(audio[:, 4 * BLOCK_SIZE :], results[-1][1].raw)
    assert not np.any(audio[:, 2 * BLOCK_SIZE : 4 * BLOCK_SIZE])


def test_remix_results_places_blocks_across_a_skip_gap() -> None:
    results = list(offline_render.replay(iter(trace([0, 1, 7, 8])), params()))
    audio, clipped = offline_render.remix_results(iter(results), 0.0)

    assert clipped == ()
    assert np.array_equal(audio, offline_render.render_results(results))


def test_verify_results_accepts_an_exact_replay() -> None:
    results = list(offline_render.replay(iter(trace(list(range(6)))), params()))
    raw = [result.raw.copy() for _, result in results]
    motor = [result.motor.copy() for _, result in results]

    assert offline_render.verify_results(iter(results), raw, motor) is None


def test_verify_results_reports_the_first_divergence() -> None:
    results = list(offline_render.replay(iter(trace([4, 5, 6, 7])), params()))
    raw = [result.raw.copy() for _, result in results]
    motor = [result.motor.copy() for _, result in results]
    raw[2][3, 17] += np.float32(0.5)

    divergence = offline_render.verify_results(iter(results), raw, motor)

    assert divergence is not None
    assert divergence.stream == "raw_array"
    assert divergence.block == 6
    assert divergence.channel == 3
    assert divergence.sample == 17
    assert divergence.recorded == pytest.approx(divergence.replayed + 0.5, abs=1e-6)


def test_verify_results_checks_the_motor_stem_too() -> None:
    results = list(offline_render.replay(iter(trace([0, 1])), params()))
    raw = [result.raw.copy() for _, result in results]
    motor = [result.motor.copy() for _, result in results]
    motor[1][0, 5] = np.float32(1.0)

    divergence = offline_render.verify_results(iter(results), raw, motor)

    assert divergence is not None
    assert divergence.stream == "stem_motor"
    assert divergence.block == 1


def test_verify_results_tolerates_a_missing_stream() -> None:
    results = list(offline_render.replay(iter(trace([0, 1])), params()))
    raw = [result.raw.copy() for _, result in results]

    assert offline_render.verify_results(iter(results), raw, []) is None


def test_first_divergence_rejects_a_shape_mismatch() -> None:
    replayed = np.zeros((4, 8), dtype=np.float32)
    recorded = np.zeros((4, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="replayed"):
        offline_render.first_divergence("raw_array", 3, replayed, recorded)


def test_remix_block_leaves_a_motorless_block_untouched() -> None:
    ped = np.linspace(-0.5, 0.5, 4 * 16, dtype=np.float32).reshape(4, 16)
    silent = np.zeros_like(ped)

    remixed = offline_render.remix_block(ped, silent, silent, 12.0)

    assert remixed.dtype == np.float32
    assert np.array_equal(remixed, ped)


def test_remix_block_attenuates_pure_ego_noise() -> None:
    motor = np.full((4, 16), 0.4, dtype=np.float32)
    silent = np.zeros_like(motor)

    assert offline_render.remix_block(silent, silent, motor, 20.0) == pytest.approx(0.04, abs=1e-6)
    assert offline_render.remix_block(silent, silent, motor, 0.0) == pytest.approx(0.4, abs=1e-6)


def test_remix_block_keeps_the_pedestrian_and_ambient_stems() -> None:
    pedestrians = np.full((4, 16), 0.2, dtype=np.float32)
    ambient = np.full((4, 16), 0.05, dtype=np.float32)
    motor = np.full((4, 16), 0.1, dtype=np.float32)

    remixed = offline_render.remix_block(pedestrians, ambient, motor, 6.0206)

    assert remixed == pytest.approx(0.3, abs=1e-5)


def test_remix_at_zero_db_rebuilds_the_live_mix_through_a_clipped_block() -> None:
    blocks = [block(3, gain=0.2), block(4, gain=40.0), block(5, gain=0.2)]
    results = list(offline_render.replay(iter(blocks), params()))
    assert results[1][1].clipped_samples > 0

    audio, _ = offline_render.remix_results(iter(results), 0.0)

    assert np.array_equal(audio, offline_render.render_results(results))


def test_remix_of_a_clipped_block_comes_from_the_unclipped_stems() -> None:
    results = list(offline_render.replay(iter([block(4, gain=40.0)]), params()))
    result = results[0][1]
    gain = np.float32(10.0 ** (-60.0 / 20.0))
    assert result.clipped_samples > 0

    audio, _ = offline_render.remix_results(iter(results), 60.0)

    assert np.array_equal(audio, np.clip(result.ped + result.ambient + gain * result.motor, -1.0, 1.0))
    assert not np.array_equal(audio, np.clip((result.raw - result.motor) + gain * result.motor, -1.0, 1.0))


def test_remix_results_reports_clipped_blocks() -> None:
    blocks = [block(3, gain=0.2), block(4, gain=40.0), block(5, gain=0.2)]
    results = list(offline_render.replay(iter(blocks), params()))
    assert [result.clipped_samples > 0 for _, result in results] == [False, True, False]

    audio, clipped = offline_render.remix_results(iter(results), 12.0)

    assert clipped == (4,)
    assert audio.shape == (CHANNELS, 3 * BLOCK_SIZE)


def test_remix_results_on_a_clean_trace() -> None:
    results = list(offline_render.replay(iter(trace([0, 1, 2])), params()))

    audio, clipped = offline_render.remix_results(iter(results), 6.0)

    assert clipped == ()
    assert audio.shape == (CHANNELS, 3 * BLOCK_SIZE)


def test_write_wav_round_trips_the_samples(tmp_path: Path) -> None:
    audio = np.linspace(-0.9, 0.9, 4 * 100, dtype=np.float32).reshape(4, 100)
    path = tmp_path / "nested" / "episode.wav"

    offline_render.write_wav(path, audio, SAMPLE_RATE)
    rate, read_back = wavfile.read(path)

    assert rate == SAMPLE_RATE
    assert read_back.dtype == np.float32
    assert np.array_equal(read_back.T, audio)


def test_output_name_carries_the_attenuation(tmp_path: Path) -> None:
    job = offline_render.EpisodeJob(path=tmp_path / "episode_007.mcap", mode="remix", rir_crossfade_s=0.1, attenuation_db=12.5)
    render_job = offline_render.EpisodeJob(path=tmp_path / "episode_007.mcap", mode="render", rir_crossfade_s=0.1)

    assert offline_render._output_name(job) == "episode_007.remix_12.5db.wav"
    assert offline_render._output_name(render_job) == "episode_007.render.wav"


def test_audio_block_deinterleaves_the_payload() -> None:
    channels = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)

    block_out = offline_render.audio_block(
        "/robot/audio/raw_array",
        encoding="32FC1",
        interleaved=True,
        channel_count=4,
        frame_count=3,
        data=channels.T.reshape(-1).tolist(),
    )

    assert np.array_equal(block_out, channels)


def test_audio_block_rejects_a_foreign_encoding() -> None:
    with pytest.raises(ValueError, match="32FC1"):
        offline_render.audio_block(
            "/robot/audio/raw_array",
            encoding="16SC1",
            interleaved=True,
            channel_count=4,
            frame_count=1,
            data=[0.0, 0.0, 0.0, 0.0],
        )


def test_audio_block_rejects_a_truncated_payload() -> None:
    with pytest.raises(ValueError, match="malformed"):
        offline_render.audio_block(
            "/robot/audio/raw_array",
            encoding="32FC1",
            interleaved=True,
            channel_count=4,
            frame_count=3,
            data=[0.0] * 8,
        )


def clip_block(index: int, *, anchor: int, impulse: ImpulseShape | None = None, rir_key: str = "") -> RenderInputs:
    delay, samples = prepare_clip(asset(), 62.0, -26.0, 3.4, level_rms=level("bark_a"), impulse=impulse)
    return RenderInputs(
        block_index=index,
        clips=(
            ClipInput(
                channel=2,
                start=anchor + delay,
                asset_key="bark_a",
                samples=samples,
                anchor=anchor,
                received_volume_db=62.0,
                sensitivity_dbfs_at_94_dbspl=-26.0,
                delay_samples=3.4,
                rir_key=rir_key,
            ),
        ),
        continuous=block(index).continuous,
        streams=block(index).streams,
        channels=CHANNELS,
    )


def test_decoded_trace_replays_to_the_same_audio() -> None:
    blocks = [clip_block(2, anchor=3 * BLOCK_SIZE) if index == 2 else block(index) for index in range(12)]

    expected_state = new_state(CHANNELS)
    expected_params = params()
    expected = [render_block(expected_state, item, expected_params) for item in blocks]

    texts = [render_inputs_to_json(item) for item in blocks]
    replay_params = params()
    replayed = list(offline_render.replay(offline_render.decode_blocks(texts, replay_params.resolve, level), replay_params))

    for (_, result), reference in zip(replayed, expected, strict=True):
        assert np.array_equal(result.raw, reference.raw)
        assert np.array_equal(result.motor, reference.motor)


def test_recorded_room_impulses_replay_the_convolved_clip() -> None:
    impulse = np.zeros(96, dtype=np.float32)
    impulse[3] = 1.0
    impulse[40] = 0.3
    recorded = {"room_a": offline_render.RecordedImpulse(samples=impulse, sample_rate_hz=48000, lead_samples=9)}
    resolver = offline_render.impulse_resolver(recorded, SAMPLE_RATE)
    shape = resolver("room_a")
    assert shape is not None
    assert shape.lead_samples == 3
    assert resolver("room_a") is shape
    assert resolver("room_b") is None

    blocks = [clip_block(0, anchor=0, impulse=shape, rir_key="room_a"), block(1)]
    live_params = RenderParams(channels=CHANNELS, block_size=BLOCK_SIZE, sample_rate=SAMPLE_RATE, resolve=params().resolve, impulse=resolver)
    expected = [result for _, result in offline_render.replay(iter(blocks), live_params)]
    texts = [render_inputs_to_json(item) for item in blocks]
    replayed = [result for _, result in offline_render.replay(offline_render.decode_blocks(texts, live_params.resolve, level, resolver), live_params)]

    for result, reference in zip(replayed, expected, strict=True):
        assert np.array_equal(result.raw, reference.raw)
    with pytest.raises(KeyError, match="room_a"):
        list(offline_render.decode_blocks(texts, live_params.resolve, level))


def test_trace_channels_reads_the_block_and_defaults_version_1_to_four() -> None:
    assert offline_render.trace_channels(render_inputs_to_json(RenderInputs(block_index=0, clips=(), continuous=(), streams=(), channels=2))) == 2
    assert offline_render.trace_channels('{"block_index": 0, "clips": [], "continuous": [], "drivetrain": []}') == 4


def _write_recording(path: Path, messages: list[tuple[str, str]]) -> None:
    from mcap_ros2.writer import Writer

    with path.open("wb") as stream:
        writer = Writer(stream)
        schema = writer.register_msgdef("std_msgs/msg/String", "string data")
        for index, (topic, text) in enumerate(messages):
            writer.write_message(topic=topic, schema=schema, message={"data": text}, log_time=index, publish_time=index)
        writer.finish()


def test_a_recording_without_a_render_trace_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "episode.mcap"
    _write_recording(path, [("/arena/env_0/task_generator_node/state/world", "map_empty")])

    with pytest.raises(ValueError, match="no render trace"):
        offline_render.read_episode_trace(path)
    report = offline_render.render_episode(offline_render.EpisodeJob(path=path, mode="render", rir_crossfade_s=0.1, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE))
    assert not report.ok
    assert "no render trace" in report.error


def _library_block(index: int) -> RenderInputs:
    return RenderInputs(
        block_index=index,
        clips=(),
        continuous=(ContinuousInput(source_id="radio", channel=0, asset_key="radio_loop#radio_loop_01", gain=0.25, delay_target=1.5, loop=True, program_start=0, stem="ambient"),),
        streams=(),
        channels=2,
    )


@pytest.mark.usefixtures("default_sounds")
def test_render_episode_writes_the_replayed_trace_beside_the_recording(tmp_path: Path) -> None:
    path = tmp_path / "episode.mcap"
    topic = "/arena/env_0/task_generator_node/jackal/audio/diagnostics/render_inputs"
    _write_recording(path, [(topic, render_inputs_to_json(_library_block(index))) for index in (4, 5, 6)])

    trace_read = offline_render.read_episode_trace(path)
    assert trace_read.robot == "jackal"
    assert trace_read.channels == 2
    assert len(trace_read.blocks) == 3

    report = offline_render.render_episode(offline_render.EpisodeJob(path=path, mode="render", rir_crossfade_s=0.1, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE))
    assert report.ok, report.error
    assert report.output == tmp_path / "episode.render.wav"
    assert report.first_block == 4
    resolve, _ = offline_render.asset_resolver(SAMPLE_RATE)
    expected = offline_render.render_results(list(offline_render.replay((_library_block(index) for index in (4, 5, 6)), RenderParams(channels=2, block_size=BLOCK_SIZE, sample_rate=SAMPLE_RATE, resolve=resolve))))
    rate, written = wavfile.read(report.output)
    assert rate == SAMPLE_RATE
    assert np.max(np.abs(expected)) > 0.0
    assert np.array_equal(written.T, expected)

    verify = offline_render.render_episode(offline_render.EpisodeJob(path=path, mode="verify", rir_crossfade_s=0.1, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE))
    assert not verify.ok
    assert "nothing to verify against" in verify.error


def test_legacy_sample_key_resolves_to_the_default_asset_owning_the_variant(default_sounds: SoundLibrary) -> None:
    default_sounds.use_world(None)
    footstep = default_sounds.default_asset("footstep")
    assert offline_render.legacy_owner(default_sounds, footstep.variants[0].id) == "footstep"


def test_legacy_sample_key_without_an_owner_is_rejected() -> None:
    library = SoundLibrary.default()
    library.use_world(None)
    with pytest.raises(KeyError, match="needs exactly one sound asset with that variant, found \\[\\]"):
        offline_render.legacy_owner(library, "no_such_variant_01")
