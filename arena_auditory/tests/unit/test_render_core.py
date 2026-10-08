from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from arena_auditory.render.core import (
    ClipInput,
    ContinuousInput,
    ImpulseShape,
    RenderInputs,
    RenderParams,
    RenderResult,
    StreamInput,
    new_state,
    prepare_clip,
    render_block,
    render_inputs_from_json,
    render_inputs_to_json,
)
from arena_auditory.sources.drivetrain.program import (
    BROADBAND_GAIN_DB,
    FREQUENCY_SCALE,
    LEFT_VELOCITY,
    RIGHT_VELOCITY,
    SPEED_EXPONENT,
    TONAL_GAIN_DB,
    TRIM_DB,
    VELOCITY_SMOOTHING_S,
)
from arena_robots.audio import active_rms
from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary

CHANNELS = 4
_RENDER_BLOCK = 8
_RENDER_RATE = 16000
_MOTOR_TUNING = {
    TRIM_DB: 0.0,
    FREQUENCY_SCALE: 1.0,
    TONAL_GAIN_DB: 0.0,
    BROADBAND_GAIN_DB: -12.0,
    SPEED_EXPONENT: 1.5,
    VELOCITY_SMOOTHING_S: 0.0,
}


def _asset_samples(asset_key: str) -> np.ndarray:
    """A per-key waveform whose samples are exact in float32."""
    steps = np.arange(48, dtype=np.int64)
    pattern = ((steps * 7 + len(asset_key) * 13) % 17) - 8
    return np.ascontiguousarray((pattern / 256.0).astype(np.float32))


def _asset_level(asset_key: str) -> float:
    return active_rms(_asset_samples(asset_key), _RENDER_RATE)


def _render_params(**overrides: object) -> RenderParams:
    fields: dict[str, object] = {
        "channels": CHANNELS,
        "block_size": _RENDER_BLOCK,
        "sample_rate": _RENDER_RATE,
        "resolve": _asset_samples,
    }
    fields.update(overrides)
    return RenderParams(**fields)


def _drivetrain_input(**overrides: object) -> StreamInput:
    fields: dict[str, object] = {
        "source_id": "jackal",
        "model": "drivetrain",
        "seed": 7,
        "params": {"spec": "jackal"},
        "state": {LEFT_VELOCITY: 0.8, RIGHT_VELOCITY: 0.6},
        "tuning": _MOTOR_TUNING,
        "gains": (0.5, 0.25, 0.125, 0.0625),
        "delay_samples": (0.0, 1.5, 2.0, 3.25),
        "active_channels": (True, True, True, True),
        "active": True,
        "source_agent_id": 0,
        "source_agent_name": "jackal",
        "kind": "motor",
        "asset_id": "drivetrain",
    }
    fields.update(overrides)
    return StreamInput(**fields)


def _continuous_input(gain: float = 0.5) -> ContinuousInput:
    return ContinuousInput(
        source_id="fan",
        channel=1,
        asset_key="fan_loop",
        gain=gain,
        delay_target=1.5,
        loop=True,
        program_start=0,
    )


def _clip(channel: int, start: int, asset_key: str, samples: np.ndarray, stem: str = "pedestrian") -> ClipInput:
    """A clip whose scheduling parameters the accumulation never reads."""
    return ClipInput(
        channel=channel,
        start=start,
        asset_key=asset_key,
        samples=samples,
        anchor=start,
        received_volume_db=94.0,
        sensitivity_dbfs_at_94_dbspl=-26.0,
        delay_samples=0.0,
        stem=stem,
    )


def _clip_input() -> ClipInput:
    return _clip(0, 4, "thud", (_asset_samples("thud")[:9] * 4.0).astype(np.float32))


def _inputs(index: int, clips: tuple[ClipInput, ...] = (), continuous: tuple[ContinuousInput, ...] = (), streams: tuple[StreamInput, ...] = ()) -> RenderInputs:
    return RenderInputs(block_index=index, clips=clips, continuous=continuous, streams=streams, channels=CHANNELS)


def _concat(results: list[RenderResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    return (
        np.concatenate([result.raw for result in results], axis=1),
        np.concatenate([result.ped for result in results], axis=1),
        np.concatenate([result.ambient for result in results], axis=1),
        np.concatenate([result.motor for result in results], axis=1),
        sum(result.clipped_samples for result in results),
    )


def _render_scenario(
    blocks: int,
    *,
    continuous_gain: float = 0.5,
    with_clip: bool = True,
    with_drivetrain: bool = True,
    **drivetrain_overrides: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    state = new_state(CHANNELS)
    params = _render_params()
    results = [
        render_block(
            state,
            _inputs(
                index,
                clips=(_clip_input(),) if with_clip and index == 0 else (),
                continuous=(_continuous_input(continuous_gain),),
                streams=(_drivetrain_input(**drivetrain_overrides),) if with_drivetrain else (),
            ),
            params,
        )
        for index in range(blocks)
    ]
    return _concat(results)


_GOLDEN_RAW = np.array(
    [
        [
            0.0,
            3.738891507509834e-07,
            1.2791963854397181e-05,
            2.986005893035326e-05,
            -0.109336718916893,
            3.869182910420932e-05,
            0.1093854308128357,
            -0.04696095734834671,
            0.062359560281038284,
            -0.09385977685451508,
            0.015570778399705887,
            0.12494441866874695,
            -0.03137093782424927,
            -0.00020738481543958187,
            -0.00023394532036036253,
            -0.00013321504229679704,
        ],
        [
            -0.0068359375,
            -0.005859375,
            -0.004882718902081251,
            0.008792353793978691,
            0.00587003817781806,
            0.0029467223212122917,
            1.924281968967989e-05,
            -0.0029174070805311203,
            -0.0058782570995390415,
            -0.008845661766827106,
            0.004820257890969515,
            0.0019121244549751282,
            -0.0010040130000561476,
            -0.00395037978887558,
            -0.006918018218129873,
            0.006725605111569166,
        ],
        [
            0.0,
            0.0,
            0.0,
            9.347228768774585e-08,
            3.1979909635992954e-06,
            7.465014732588315e-06,
            9.56986150413286e-06,
            9.67295727605233e-06,
            2.6074030756717548e-06,
            -2.1489693608600646e-05,
            -3.5109489544993266e-05,
            -2.744505400187336e-05,
            -1.355547738057794e-05,
            -1.3894976291339844e-05,
            -3.023472527274862e-05,
            -5.184620385989547e-05,
        ],
        [0.0, 0.0, 0.0, 0.0, 3.505210699472627e-08, 1.2109306908314466e-06, 3.1991294235922396e-06, 4.521824848779943e-06, 4.823591552849393e-06, 2.186895926570287e-06, -7.732709491392598e-06, -1.5852270735194907e-05, -1.4680581443826668e-05, -8.513935426890384e-06, -6.9050506681378465e-06, -1.3074894013698213e-05],
    ],
    dtype=np.float32,
)


def test_render_block_reproduces_the_recorded_accumulation() -> None:
    raw, _, _, _, clipped = _render_scenario(2)
    assert clipped == 0
    assert np.array_equal(raw, _GOLDEN_RAW)


def test_stems_add_back_to_the_mix_and_pedestrian_stem_ignores_the_drivetrain() -> None:
    raw, ped, ambient, motor, clipped = _render_scenario(2)
    _, quiet_ped, quiet_ambient, quiet_motor, _ = _render_scenario(2, with_drivetrain=False)
    assert clipped == 0
    assert np.max(np.abs(motor)) > 0.0
    assert not np.any(quiet_motor)
    assert np.array_equal(raw, ped + ambient + motor)
    assert np.array_equal(ped, quiet_ped)
    assert np.array_equal(ambient, quiet_ambient)


def test_pedestrian_stem_carries_clips_and_ambient_stem_the_continuous_voices() -> None:
    _, ped, ambient, _, _ = _render_scenario(2)
    _, clipless_ped, clipless_ambient, _, _ = _render_scenario(2, with_clip=False)
    assert np.max(np.abs(ped)) > 0.0
    assert not np.any(clipless_ped)
    assert np.max(np.abs(ambient)) > 0.0
    assert np.array_equal(ambient, clipless_ambient)
    assert not np.any(ped[1:])
    assert not np.any(ambient[[0, 2, 3]])


def test_motor_stem_is_silent_when_no_drivetrain_channel_is_active() -> None:
    raw, _, _, motor, _ = _render_scenario(
        2,
        gains=(0.0, 0.0, 0.0, 0.0),
        active_channels=(False, False, False, False),
        active=False,
    )
    assert np.array_equal(motor, np.zeros_like(motor))
    assert np.max(np.abs(raw)) > 0.0


def test_motor_trim_change_remixes_as_a_scalar_on_the_stem() -> None:
    attenuation_db = 6.0
    quiet = {**_MOTOR_TUNING, TRIM_DB: _MOTOR_TUNING[TRIM_DB] - attenuation_db}
    raw, _, _, motor, _ = _render_scenario(3, continuous_gain=0.02, with_clip=False)
    remixed_raw, _, _, remixed_motor, _ = _render_scenario(3, continuous_gain=0.02, with_clip=False, tuning=quiet)
    gain = 10.0 ** (-attenuation_db / 20.0)
    assert np.max(np.abs(motor)) > 1e-6
    assert not np.array_equal(remixed_motor, motor)
    np.testing.assert_allclose(remixed_raw, (raw - motor) + gain * motor, rtol=1e-5, atol=1e-8)


def test_clipped_samples_counts_the_clamped_output() -> None:
    over = np.full(_RENDER_BLOCK, 2.0, dtype=np.float32)
    under = np.full(_RENDER_BLOCK, -3.0, dtype=np.float32)
    inside = np.full(_RENDER_BLOCK, 0.5, dtype=np.float32)
    result = render_block(
        new_state(CHANNELS),
        _inputs(0, clips=(_clip(0, 0, "over", over), _clip(1, 0, "under", under), _clip(2, 0, "inside", inside))),
        _render_params(),
    )
    assert result.clipped_samples == 2 * _RENDER_BLOCK
    assert np.array_equal(result.raw[0], np.ones(_RENDER_BLOCK, dtype=np.float32))
    assert np.array_equal(result.raw[1], -np.ones(_RENDER_BLOCK, dtype=np.float32))
    assert np.array_equal(result.raw[2], inside)
    assert np.array_equal(result.motor, np.zeros_like(result.motor))


def _saturating_scenario(blocks: int, *, with_drivetrain: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """A full-scale clip on channel 0 that only the drivetrain pushes past the limit."""
    state = new_state(CHANNELS)
    params = _render_params()
    results = [
        render_block(
            state,
            _inputs(
                index,
                clips=(_clip(0, 0, "rail", np.ones(blocks * _RENDER_BLOCK, dtype=np.float32)),) if index == 0 else (),
                continuous=(_continuous_input(),),
                streams=(_drivetrain_input(),) if with_drivetrain else (),
            ),
            params,
        )
        for index in range(blocks)
    ]
    return _concat(results)


def _pre_clip_mix(blocks: int) -> np.ndarray:
    """The accumulator the renderer clips, rebuilt from a drivetrain-free run of the same scenario."""
    reference, _, _, _, clipped = _saturating_scenario(blocks, with_drivetrain=False)
    assert clipped == 0
    _, _, _, motor, _ = _saturating_scenario(blocks)
    return reference + motor


def test_stems_add_back_to_the_pre_clip_mix_across_a_clipped_block() -> None:
    blocks = 2
    raw, ped, ambient, motor, clipped = _saturating_scenario(blocks)
    _, quiet_ped, quiet_ambient, _, _ = _saturating_scenario(blocks, with_drivetrain=False)
    assert clipped > 0
    assert np.max(np.abs(motor)) > 0.0
    assert np.max(np.abs(ped)) > 0.0
    assert np.max(np.abs(ambient)) > 0.0
    assert np.array_equal(ped + ambient + motor, _pre_clip_mix(blocks))
    assert np.array_equal(raw, np.clip(ped + ambient + motor, -1.0, 1.0))
    assert np.array_equal(ped, quiet_ped)
    assert np.array_equal(ambient, quiet_ambient)


def test_pedestrian_stem_survives_the_clipping_that_breaks_raw_minus_motor() -> None:
    blocks = 2
    raw, ped, ambient, motor, clipped = _saturating_scenario(blocks)
    assert clipped > 0
    assert not np.array_equal(raw - motor, ped + ambient)
    assert np.array_equal(ped + ambient + motor, _pre_clip_mix(blocks))
    assert not np.array_equal((raw - motor) + motor, _pre_clip_mix(blocks))


def test_pedestrian_stem_does_not_move_when_only_the_drivetrain_level_does() -> None:
    attenuation_db = 6.0
    quiet = {**_MOTOR_TUNING, TRIM_DB: _MOTOR_TUNING[TRIM_DB] - attenuation_db}
    _, ped, ambient, motor, _ = _render_scenario(3, continuous_gain=0.02)
    _, quiet_ped, quiet_ambient, quiet_motor, _ = _render_scenario(3, continuous_gain=0.02, tuning=quiet)
    assert np.max(np.abs(motor)) > 1e-6
    assert np.max(np.abs(quiet_motor - motor)) > 1e-6
    assert np.array_equal(quiet_ped, ped)
    assert np.array_equal(quiet_ambient, ambient)
    np.testing.assert_allclose(quiet_motor, 10.0 ** (-attenuation_db / 20.0) * motor, rtol=1e-5, atol=1e-9)


_STEM_BY_KIND = {"footstep": "pedestrian", "speech": "pedestrian", "music": "ambient", "alarm": "ambient", "motor": "motor"}


@pytest.mark.parametrize(("kind", "stem"), sorted(_STEM_BY_KIND.items()))
def test_kinds_table_assigns_each_kind_its_stem(kind: str, stem: str) -> None:
    assert SoundLibrary.default().kind(kind).stem == stem


def _stems(result: RenderResult) -> dict[str, np.ndarray]:
    return {"pedestrian": result.ped, "ambient": result.ambient, "motor": result.motor}


def _voice_of(kind: str, voice: str, channel: int, stem: str) -> RenderInputs:
    match voice:
        case "clip":
            return _inputs(0, clips=(_clip(channel, 2, kind, _asset_samples(kind)[:5].copy(), stem=stem),))
        case "continuous":
            return _inputs(
                0,
                continuous=(ContinuousInput(source_id=kind, channel=channel, asset_key=kind, gain=0.5, delay_target=0.0, loop=True, program_start=0, stem=stem),),
            )
    return _inputs(0, streams=(_drivetrain_input(source_id=kind, kind=kind, stem=stem),))


@pytest.mark.parametrize(
    ("kind", "voice"),
    [
        ("footstep", "clip"),
        ("speech", "clip"),
        ("speech", "continuous"),
        ("speech", "stream"),
        ("music", "continuous"),
        ("music", "clip"),
        ("alarm", "continuous"),
        ("motor", "stream"),
        ("motor", "continuous"),
    ],
)
def test_a_source_renders_into_the_stem_of_its_kind(kind: str, voice: str) -> None:
    stem = SoundLibrary.default().kind(kind).stem
    result = render_block(new_state(CHANNELS), _voice_of(kind, voice, 1, stem), _render_params())
    stems = _stems(result)
    assert np.max(np.abs(stems[stem])) > 0.0
    assert all(not np.any(audio) for name, audio in stems.items() if name != stem)
    assert np.array_equal(result.raw, np.clip(stems[stem], -1.0, 1.0))


def _routed_scene(kinds: tuple[str, ...], library: SoundLibrary) -> RenderInputs:
    clips: list[ClipInput] = []
    continuous: list[ContinuousInput] = []
    streams: list[StreamInput] = []
    if "footstep" in kinds:
        clips.append(_clip(0, 1, "footstep", _asset_samples("footstep")[:6].copy(), stem=library.kind("footstep").stem))
    if "speech" in kinds:
        clips.append(_clip(1, 0, "speech", np.full(_RENDER_BLOCK, 2.0, dtype=np.float32), stem=library.kind("speech").stem))
    for channel, kind in ((2, "music"), (3, "alarm")):
        if kind in kinds:
            continuous.append(ContinuousInput(source_id=kind, channel=channel, asset_key=kind, gain=0.5, delay_target=0.0, loop=True, program_start=0, stem=library.kind(kind).stem))
    if "motor" in kinds:
        streams.append(_drivetrain_input(stem=library.kind("motor").stem))
    return _inputs(0, clips=tuple(clips), continuous=tuple(continuous), streams=tuple(streams))


def test_routed_stems_sum_to_the_pre_clip_mix_whose_clip_is_raw() -> None:
    library = SoundLibrary.default()
    all_kinds = tuple(_STEM_BY_KIND)
    full = render_block(new_state(CHANNELS), _routed_scene(all_kinds, library), _render_params())
    solos = {stem: render_block(new_state(CHANNELS), _routed_scene(tuple(kind for kind, owner in _STEM_BY_KIND.items() if owner == stem), library), _render_params()) for stem in ("pedestrian", "ambient", "motor")}
    for stem, solo in solos.items():
        assert np.max(np.abs(_stems(solo)[stem])) > 0.0
        assert np.array_equal(_stems(full)[stem], _stems(solo)[stem])
        assert all(not np.any(audio) for name, audio in _stems(solo).items() if name != stem)
    pre_clip = solos["pedestrian"].ped + solos["ambient"].ambient + solos["motor"].motor
    assert full.clipped_samples > 0
    assert np.array_equal(full.ped + full.ambient + full.motor, pre_clip)
    assert np.array_equal(full.raw, np.clip(pre_clip, -1.0, 1.0))
    assert not np.array_equal(full.raw, pre_clip)


def test_a_continuous_voice_is_keyed_by_source_and_channel() -> None:
    state = new_state(CHANNELS)
    params = _render_params()
    for index in range(2):
        render_block(state, _inputs(index, continuous=(_continuous_input(), _continuous_input()), streams=(_drivetrain_input(), _drivetrain_input())), params)
    assert list(state.continuous) == [("fan", 1)]
    assert list(state.streams) == ["jackal"]
    other_channel = ContinuousInput(source_id="fan", channel=2, asset_key="fan_loop", gain=0.5, delay_target=1.5, loop=True, program_start=0)
    render_block(state, _inputs(2, continuous=(_continuous_input(), other_channel)), params)
    assert sorted(state.continuous) == [("fan", 1), ("fan", 2)]
    assert list(state.streams) == []


def test_a_looping_voice_repeats_and_stops_at_the_block_boundary() -> None:
    loop = np.asarray([0.1, 0.2], dtype=np.float32)
    params = _render_params(channels=1, block_size=5, resolve=lambda _key: loop)
    state = new_state(1)
    voice = ContinuousInput(source_id="motor:1", channel=0, asset_key="loop", gain=1.0, delay_target=0.0, loop=True, program_start=0)
    first = render_block(state, RenderInputs(block_index=0, clips=(), continuous=(voice,), streams=(), channels=1), params)
    np.testing.assert_allclose(first.raw[0], [0.1, 0.2, 0.1, 0.2, 0.1])
    second = render_block(state, RenderInputs(block_index=1, clips=(), continuous=(), streams=(), channels=1), params)
    np.testing.assert_allclose(second.raw[0], [0.0] * 5)
    assert state.continuous == {}


def test_a_stream_and_a_sample_voice_share_one_mix() -> None:
    state = new_state(CHANNELS)
    result = render_block(state, _inputs(0, continuous=(_continuous_input(),), streams=(_drivetrain_input(),)), _render_params())
    assert np.max(np.abs(result.ambient)) > 0.0
    assert np.max(np.abs(result.motor)) > 0.0
    assert np.array_equal(result.raw, np.clip(result.ambient + result.motor, -1.0, 1.0))
    assert len(state.continuous) == 1
    assert len(state.streams) == 1


def _program_block(program_start: int, block_index: int, *, loop: bool) -> np.ndarray:
    samples = np.arange(1, 11, dtype=np.float32) / 16.0
    params = _render_params(channels=1, block_size=4, resolve=lambda _key: samples)
    state = new_state(1)
    state.cursor = block_index * 4
    voice = ContinuousInput(source_id="radio", channel=0, asset_key="radio", gain=1.0, delay_target=0.0, loop=loop, program_start=program_start)
    return render_block(state, RenderInputs(block_index=block_index, clips=(), continuous=(voice,), streams=(), channels=1), params).raw[0]


def test_a_one_shot_program_started_in_the_past_resumes_at_its_offset() -> None:
    np.testing.assert_allclose(_program_block(-6, 0, loop=False), np.asarray([7, 8, 9, 10], dtype=np.float32) / 16.0)


def test_a_one_shot_program_past_its_end_renders_silence() -> None:
    np.testing.assert_allclose(_program_block(-20, 0, loop=False), [0.0] * 4)


def test_a_looping_program_wraps_its_offset() -> None:
    np.testing.assert_allclose(_program_block(-28, 0, loop=True), np.asarray([9, 10, 1, 2], dtype=np.float32) / 16.0)


def test_a_one_shot_program_starting_in_the_future_starts_at_its_first_frame() -> None:
    np.testing.assert_allclose(_program_block(2, 0, loop=False), np.asarray([0, 0, 1, 2], dtype=np.float32) / 16.0)


def test_a_single_tap_room_impulse_scales_the_voice_and_an_unknown_key_renders_dry() -> None:
    errors: list[str] = []
    shapes = {"half": ImpulseShape(samples=np.asarray([0.5], dtype=np.float32), lead_samples=0)}
    params = _render_params(impulse=shapes.get, log_error=errors.append)
    dry = render_block(new_state(CHANNELS), _inputs(0, continuous=(_continuous_input(),)), params)
    wet = render_block(new_state(CHANNELS), _inputs(0, continuous=(replace(_continuous_input(), rir_key="half"),)), params)
    unknown = render_block(new_state(CHANNELS), _inputs(0, continuous=(replace(_continuous_input(), rir_key="missing"),)), params)
    assert np.max(np.abs(dry.ambient)) > 0.0
    np.testing.assert_allclose(wet.ambient, 0.5 * dry.ambient, rtol=1e-6)
    assert np.array_equal(unknown.ambient, dry.ambient)
    assert errors == ["room impulse 'missing' is unknown, rendering dry"]


def test_render_block_rejects_inputs_of_another_channel_count() -> None:
    with pytest.raises(ValueError, match="channels"):
        render_block(new_state(CHANNELS), RenderInputs(block_index=0, clips=(), continuous=(), streams=(), channels=2), _render_params())


def _traced_clip() -> ClipInput:
    """A clip built the way the live renderer builds one, so the trace can rebuild it."""
    anchor, received_volume_db, sensitivity, delay_samples = 4, 88.0, -26.0, 3.25
    delay, samples = prepare_clip(_asset_samples("thud"), received_volume_db, sensitivity, delay_samples, level_rms=_asset_level("thud"))
    return ClipInput(
        channel=0,
        start=anchor + delay,
        asset_key="thud",
        samples=samples,
        anchor=anchor,
        received_volume_db=received_volume_db,
        sensitivity_dbfs_at_94_dbspl=sensitivity,
        delay_samples=delay_samples,
    )


def _traced_inputs(block_index: int) -> RenderInputs:
    return _inputs(
        block_index,
        clips=(_traced_clip(),) if block_index == 0 else (),
        continuous=(_continuous_input(),),
        streams=(_drivetrain_input(),),
    )


def test_render_inputs_json_round_trip_renders_the_same_blocks() -> None:
    params = _render_params()
    live_state, replay_state = new_state(CHANNELS), new_state(CHANNELS)
    live_results: list[RenderResult] = []
    for index in range(3):
        inputs = _traced_inputs(index)
        replayed = render_inputs_from_json(render_inputs_to_json(inputs), _asset_samples, _asset_level)
        assert replayed.block_index == inputs.block_index
        assert replayed.channels == CHANNELS
        assert replayed.continuous == inputs.continuous
        assert replayed.streams == inputs.streams
        if inputs.clips:
            assert np.max(np.abs(inputs.clips[0].samples)) > 0.0
            assert np.array_equal(replayed.clips[0].samples, inputs.clips[0].samples)
            assert replayed.clips[0].start == inputs.clips[0].start
            assert replayed.clips[0].stem == inputs.clips[0].stem
        live = render_block(live_state, inputs, params)
        replay = render_block(replay_state, replayed, params)
        assert np.array_equal(live.raw, replay.raw)
        assert np.array_equal(live.ped, replay.ped)
        assert np.array_equal(live.ambient, replay.ambient)
        assert np.array_equal(live.motor, replay.motor)
        assert live.clipped_samples == replay.clipped_samples
        live_results.append(live)
    raw, ped, ambient, motor, _ = _concat(live_results)
    assert np.max(np.abs(raw[0])) > 0.0
    assert np.max(np.abs(raw[1])) > 0.0
    assert np.max(np.abs(motor)) > 0.0
    assert np.array_equal(raw, np.clip(ped + ambient + motor, -1.0, 1.0))


def test_render_inputs_json_rejects_a_clip_that_replays_at_another_sample() -> None:
    payload = json.loads(render_inputs_to_json(_traced_inputs(0)))
    payload["clips"][0]["start"] += 1
    with pytest.raises(ValueError, match="trace recorded"):
        render_inputs_from_json(json.dumps(payload), _asset_samples, _asset_level)


def test_render_inputs_json_rejects_an_unknown_version() -> None:
    payload = json.loads(render_inputs_to_json(_traced_inputs(1)))
    payload["version"] = 3
    with pytest.raises(ValueError, match="version"):
        render_inputs_from_json(json.dumps(payload), _asset_samples, _asset_level)


def test_render_inputs_json_requires_the_room_impulses_it_names() -> None:
    payload = json.loads(render_inputs_to_json(_traced_inputs(1)))
    payload["continuous"][0]["rir_key"] = "room_a"
    with pytest.raises(KeyError, match="room_a"):
        render_inputs_from_json(json.dumps(payload), _asset_samples, _asset_level)


def test_render_inputs_json_carries_no_sample_arrays() -> None:
    clip = _traced_clip()
    assert clip.samples.size > 4
    payload = json.loads(render_inputs_to_json(_traced_inputs(0)))
    assert [sorted(entry) for entry in payload["clips"]] == [
        [
            "anchor",
            "asset_key",
            "channel",
            "delay_samples",
            "received_volume_db",
            "rir_key",
            "sensitivity_dbfs_at_94_dbspl",
            "start",
            "stem",
        ]
    ]

    def _no_waveforms(node: object) -> None:
        if isinstance(node, dict):
            for item in node.values():
                _no_waveforms(item)
        elif isinstance(node, list):
            for item in node:
                _no_waveforms(item)
            assert all(isinstance(item, dict) for item in node) or len(node) <= CHANNELS

    _no_waveforms(payload)


def _v1_payload() -> dict[str, object]:
    return {
        "block_index": 2,
        "clips": [],
        "continuous": [
            {"source_id": "fan", "channel": 1, "asset_key": "fan_loop", "gain": 0.5, "delay_target": 1.5, "loop": True, "program_start": 0},
        ],
        "drivetrain": [
            {
                "source_id": "jackal",
                "deterministic_seed": 7,
                "gains": [0.5, 0.25, 0.125, 0.0625],
                "delay_samples": [0.0, 1.5, 2.0, 3.25],
                "active_channels": [True, True, True, True],
                "left_velocity": 0.8,
                "right_velocity": 0.6,
                "active": True,
                "tuning": {
                    "volume_db": 0.0,
                    "frequency_scale": 1.0,
                    "tonal_gain_db": 0.0,
                    "broadband_gain_db": -12.0,
                    "speed_exponent": 1.5,
                    "velocity_smoothing_seconds": 0.0,
                },
                "source_agent_id": 0,
                "source_agent_name": "jackal",
                "sound_type": "motor",
                "asset_id": "drivetrain",
            }
        ],
    }


def test_a_version_1_trace_replays_as_four_channel_drivetrain_streams() -> None:
    replayed = render_inputs_from_json(json.dumps(_v1_payload()), _asset_samples, _asset_level)
    expected = _inputs(2, continuous=(_continuous_input(),), streams=(_drivetrain_input(),))
    assert replayed.channels == CHANNELS
    assert replayed.continuous == expected.continuous
    assert replayed.streams == expected.streams
    params = _render_params()
    from_trace = render_block(new_state(CHANNELS), replayed, params)
    direct = render_block(new_state(CHANNELS), expected, params)
    assert np.array_equal(from_trace.raw, direct.raw)
    assert np.array_equal(from_trace.motor, direct.motor)
