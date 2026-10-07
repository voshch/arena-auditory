from __future__ import annotations

import attrs
import numpy as np
import pytest
from arena_simulation_setup.tree.assets.Sound import SoundIdentifier
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind, parse_manifest

from arena_auditory.params import MotorGroup
from arena_auditory.shared import SourceSpec
from arena_auditory.sources import stream_model, streamed, tuning_of
from arena_auditory.sources.drivetrain import JACKAL, DrivetrainSpec, DrivetrainVoice, cache_bytes, clear_cache
from arena_auditory.sources.drivetrain.program import LEFT_VELOCITY, RIGHT_VELOCITY, DrivetrainModel, DrivetrainProgram, drivetrain_spec, wheel_state


def _tuning(**overrides: float) -> dict[str, float]:
    return {**tuning_of(MotorGroup.defaults()), **overrides}


def _driving(left: float, right: float) -> SourceSpec:
    state = {LEFT_VELOCITY: left, RIGHT_VELOCITY: right}
    return SourceSpec(id="robot:jackal:motor", kind="motor", asset_id="motor", model="drivetrain", agent_kind=AgentKind.ROBOT, position=(0.0, 0.0, 0.0), level_db=45.0, state=state)


def _shared_field_program(seed: int) -> DrivetrainProgram:
    program = DrivetrainProgram(seed=seed, sample_rate_hz=8000, block_size=256, tuning=_tuning(velocity_smoothing_s=0.01))
    program.update(_driving(0.5, 0.5))
    return program


def test_drivetrain_programs_of_different_seeds_share_one_noise_field() -> None:
    clear_cache()
    first = _shared_field_program(seed=1)
    held = cache_bytes()
    second = _shared_field_program(seed=2)

    assert held > 0
    assert cache_bytes() == held
    assert not np.allclose(first.render(256), second.render(256))


def test_drivetrain_stream_renders_mono_blocks_at_the_requested_sample_rate() -> None:
    program = stream_model("drivetrain").stream(seed=3, sample_rate_hz=8000, block_size=32, params={"spec": "jackal"}, tuning=_tuning())
    program.update(_driving(0.5, 0.5))

    block = program.render(32)

    assert isinstance(program, DrivetrainProgram)
    assert program.sample_rate_hz == 8000
    assert drivetrain_spec("jackal", 8000).sample_rate == 8000
    assert block.shape == (32,)
    assert block.dtype == np.float32
    with pytest.raises(ValueError, match="configured for 32 frames"):
        program.render(16)


def test_unknown_drivetrain_spec_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown drivetrain spec 'tracked'"):
        DrivetrainModel.stream(seed=0, sample_rate_hz=8000, block_size=32, params={"spec": "tracked"}, tuning=_tuning())


def test_drivetrain_is_the_registered_continuous_stream_model() -> None:
    assert streamed("drivetrain")
    assert stream_model("drivetrain") is DrivetrainModel
    assert DrivetrainModel.continuous
    assert DrivetrainModel.tuning_group is MotorGroup


def test_wheel_state_splits_the_body_twist_across_the_wheels() -> None:
    assert wheel_state(1.0, 2.0, 0.5) == {LEFT_VELOCITY: 0.5, RIGHT_VELOCITY: 1.5}


def test_stopped_program_falls_silent_and_finishes_after_its_release() -> None:
    program = DrivetrainProgram(seed=0, sample_rate_hz=8000, block_size=256, tuning=_tuning())
    program.update(_driving(1.0, 1.0))
    for _ in range(8):
        program.render(256)
    program.update(attrs.evolve(_driving(1.0, 1.0), active=False))

    blocks = 0
    while not program.finished:
        tail = program.render(256)
        blocks += 1
        assert blocks < 100

    assert np.max(np.abs(program.render(256))) == 0.0
    assert np.max(np.abs(tail)) < 1e-3


def test_drivetrain_runtime_tuning_changes_pitch_and_tonal_level() -> None:
    sample_rate = 8000
    spec = DrivetrainSpec(
        K=2.0 * np.pi * 100.0,
        partials_db=(0.0,),
        n_drivetrains=1,
        v_static=0.0,
        crossfade_s=0.0001,
        sample_rate=sample_rate,
    )
    frames = sample_rate

    baseline = DrivetrainVoice(spec, transfer=False, gain=1.0).render(1.0, frames)
    tuned = DrivetrainVoice(spec, transfer=False, gain=1.0).render(1.0, frames, frequency_scale=1.5, tonal_gain_db=-12.0)

    frequencies = np.fft.rfftfreq(frames, 1.0 / sample_rate)
    baseline_peak = frequencies[np.argmax(np.abs(np.fft.rfft(baseline)))]
    tuned_peak = frequencies[np.argmax(np.abs(np.fft.rfft(tuned)))]
    assert baseline_peak == 100.0
    assert tuned_peak == 150.0
    np.testing.assert_allclose(np.sqrt(np.mean(tuned**2)), np.sqrt(np.mean(baseline**2)) * 10.0 ** (-12.0 / 20.0), rtol=0.01)


@pytest.mark.usefixtures("default_sounds")
def test_motor_asset_level_is_45_db_at_1_m_for_every_variant_at_zero_trim() -> None:
    view = SoundIdentifier.parse("motor").resolve_sync()
    asset, _ = parse_manifest("motor", view.path, view.manifest)
    drivetrain = asset.variant("jackal_drivetrain")

    assert asset.level_db == 45.0
    assert asset.reference_distance_m == 1.0
    assert asset.loop
    assert drivetrain.model == "drivetrain"
    assert drivetrain.params == {"spec": "jackal"}
    assert asset.select(context={"robot_model": "jackal"}, seed=5).id == "jackal_drivetrain"
    assert asset.select(context={"robot_model": "turtlebot3"}, seed=5).id == "motor_loop_01"
    assert MotorGroup.defaults()[MotorGroup.TRIM_DB.field] == 0.0
    assert JACKAL.v_ref == 1.0
