from __future__ import annotations

import numpy as np
import pytest

from arena_auditory.params import MotorGroup
from arena_auditory.render.dsp import calibrate_mems, mems_gain
from arena_auditory.shared import AgentKind, SourceSpec, active_rms, dbfs_from_rms, rms, spl_to_dbfs
from arena_auditory.sources import stream_model, tuning_of
from arena_auditory.sources.drivetrain.program import (
    LEFT_VELOCITY,
    REFERENCE_BLOCK_SIZE,
    REFERENCE_S,
    REFERENCE_WARMUP_S,
    RIGHT_VELOCITY,
    DrivetrainModel,
    drivetrain_spec,
)

SAMPLE_RATE_HZ = 16_000
PARAMS = {"spec": "jackal"}
SENSITIVITY_DBFS = -26.0
MOTOR_LEVEL_DB = 45.0


def _tuning(**overrides: float) -> dict[str, float]:
    return {**tuning_of(MotorGroup.defaults()), **overrides}


def _render_at(speed_mps: float, *, seed: int, tuning: dict[str, float]) -> np.ndarray:
    program = DrivetrainModel.stream(seed=seed, sample_rate_hz=SAMPLE_RATE_HZ, block_size=REFERENCE_BLOCK_SIZE, params=PARAMS, tuning=tuning)
    state = {LEFT_VELOCITY: speed_mps, RIGHT_VELOCITY: speed_mps}
    program.update(SourceSpec(id="robot:jackal:motor", kind="motor", asset_id="motor", model="drivetrain", agent_kind=AgentKind.ROBOT, position=(0.0, 0.0, 0.0), level_db=MOTOR_LEVEL_DB, state=state))
    warmup = -(-int(REFERENCE_WARMUP_S * SAMPLE_RATE_HZ) // REFERENCE_BLOCK_SIZE)
    blocks = -(-int(REFERENCE_S * SAMPLE_RATE_HZ) // REFERENCE_BLOCK_SIZE)
    for _ in range(warmup):
        program.render(REFERENCE_BLOCK_SIZE)
    return np.concatenate([program.render(REFERENCE_BLOCK_SIZE) for _ in range(blocks)])


def _v_ref() -> float:
    return drivetrain_spec("jackal", SAMPLE_RATE_HZ).v_ref


def test_reference_rms_is_the_level_of_the_drivetrain_at_v_ref() -> None:
    reference = stream_model("drivetrain").reference_rms(SAMPLE_RATE_HZ, PARAMS, _tuning())

    assert reference > 0.0
    assert reference == pytest.approx(float(rms(_render_at(_v_ref(), seed=0, tuning=_tuning()))), rel=1e-6)


def test_reference_rms_ignores_trim_and_follows_the_timbre() -> None:
    reference = DrivetrainModel.reference_rms(SAMPLE_RATE_HZ, PARAMS, _tuning())

    assert DrivetrainModel.reference_rms(SAMPLE_RATE_HZ, PARAMS, _tuning(trim_db=-12.0)) == reference
    assert DrivetrainModel.reference_rms(SAMPLE_RATE_HZ, PARAMS, _tuning(broadband_gain_db=-24.0)) < DrivetrainModel.reference_rms(SAMPLE_RATE_HZ, PARAMS, _tuning(broadband_gain_db=0.0))


def test_drivetrain_at_v_ref_calibrates_to_the_asset_level_like_a_clip() -> None:
    target_dbfs = spl_to_dbfs(MOTOR_LEVEL_DB, SENSITIVITY_DBFS)
    reference = DrivetrainModel.reference_rms(SAMPLE_RATE_HZ, PARAMS, _tuning())
    motor = _render_at(_v_ref(), seed=7, tuning=_tuning()) * mems_gain(MOTOR_LEVEL_DB, reference, sensitivity_dbfs_at_94_dbspl=SENSITIVITY_DBFS)
    tone = np.sin(np.linspace(0.0, 4.0 * np.pi, 1600, endpoint=False)).astype(np.float32)
    pedestrian = calibrate_mems(tone, MOTOR_LEVEL_DB, active_rms(tone, SAMPLE_RATE_HZ), sensitivity_dbfs_at_94_dbspl=SENSITIVITY_DBFS)

    assert dbfs_from_rms(float(rms(pedestrian))) == pytest.approx(target_dbfs, abs=0.01)
    assert dbfs_from_rms(float(rms(motor))) == pytest.approx(target_dbfs, abs=1.0)


def test_calibrated_drivetrain_level_holds_across_timbre_tuning() -> None:
    tuned = _tuning(tonal_gain_db=-12.0, broadband_gain_db=3.0)
    default_gain = mems_gain(MOTOR_LEVEL_DB, DrivetrainModel.reference_rms(SAMPLE_RATE_HZ, PARAMS, _tuning()), sensitivity_dbfs_at_94_dbspl=SENSITIVITY_DBFS)
    tuned_gain = mems_gain(MOTOR_LEVEL_DB, DrivetrainModel.reference_rms(SAMPLE_RATE_HZ, PARAMS, tuned), sensitivity_dbfs_at_94_dbspl=SENSITIVITY_DBFS)

    default_level = dbfs_from_rms(float(rms(_render_at(_v_ref(), seed=0, tuning=_tuning()) * default_gain)))
    tuned_level = dbfs_from_rms(float(rms(_render_at(_v_ref(), seed=0, tuning=tuned) * tuned_gain)))

    assert tuned_level == pytest.approx(default_level, abs=1e-4)


def test_trim_offsets_the_calibrated_drivetrain_level() -> None:
    reference = DrivetrainModel.reference_rms(SAMPLE_RATE_HZ, PARAMS, _tuning())
    gain = mems_gain(MOTOR_LEVEL_DB, reference, sensitivity_dbfs_at_94_dbspl=SENSITIVITY_DBFS)

    untrimmed = dbfs_from_rms(float(rms(_render_at(_v_ref(), seed=0, tuning=_tuning()) * gain)))
    trimmed = dbfs_from_rms(float(rms(_render_at(_v_ref(), seed=0, tuning=_tuning(trim_db=-6.0)) * gain)))

    assert trimmed - untrimmed == pytest.approx(-6.0, abs=1e-3)
