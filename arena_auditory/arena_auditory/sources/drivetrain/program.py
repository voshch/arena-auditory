"""The drivetrain source model: a persistent left/right voice pair driven by the source's wheel speeds."""

from __future__ import annotations

import functools
import threading
import typing
from collections.abc import Mapping

import numpy as np
from arena_robots.audio import rms
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind

from arena_auditory.params import MotorGroup
from arena_auditory.shared import SourceSpec
from arena_auditory.sources import tuning_of
from arena_auditory.sources.drivetrain.spec import JACKAL, DrivetrainSpec
from arena_auditory.sources.drivetrain.voice import DrivetrainVoice
from arena_auditory.sources.drivetrain.voice import prewarm as prewarm_field

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.param_groups import ParamGroup
    from arena_simulation_setup.tree.assets.sound_catalog import SoundAsset, Variant

    from arena_auditory.sources import ProgramContext

DRIVETRAIN_FIELD_SEED = 0
RELEASE_S = 0.1
REFERENCE_WARMUP_S = 0.25
REFERENCE_S = 1.0
REFERENCE_BLOCK_SIZE = 1024

SPEC = "spec"
DEFAULT_SPEC = "jackal"
LEFT_VELOCITY = "left_velocity_mps"
RIGHT_VELOCITY = "right_velocity_mps"
SPECS: dict[str, DrivetrainSpec] = {DEFAULT_SPEC: JACKAL}

TRIM_DB = MotorGroup.TRIM_DB.field
FREQUENCY_SCALE = MotorGroup.FREQUENCY_SCALE.field
TONAL_GAIN_DB = MotorGroup.TONAL_GAIN_DB.field
BROADBAND_GAIN_DB = MotorGroup.BROADBAND_GAIN_DB.field
SPEED_EXPONENT = MotorGroup.SPEED_EXPONENT.field
VELOCITY_SMOOTHING_S = MotorGroup.VELOCITY_SMOOTHING_S.field


def wheel_state(linear_mps: float, angular_radps: float, wheel_separation_m: float) -> dict[str, float]:
    """Drivetrain source state of a differential base from its body twist."""
    half = 0.5 * wheel_separation_m
    return {LEFT_VELOCITY: linear_mps - angular_radps * half, RIGHT_VELOCITY: linear_mps + angular_radps * half}


def spec_name(params: Mapping[str, object]) -> str:
    """Drivetrain spec name a sound variant's params select."""
    return str(params.get(SPEC, DEFAULT_SPEC))


def drivetrain_spec(name: str, sample_rate_hz: int) -> DrivetrainSpec:
    """Raises ValueError for an unknown spec name."""
    try:
        spec = SPECS[name]
    except KeyError:
        raise ValueError(f"unknown drivetrain spec {name!r}, expected one of {sorted(SPECS)}") from None
    return spec.replace(sample_rate=int(sample_rate_hz))


def _volume_gain(tuning: Mapping[str, float]) -> float:
    return 10.0 ** (float(tuning[TRIM_DB]) / 20.0)


@functools.lru_cache(maxsize=64)
def _reference_rms(spec: str, sample_rate_hz: int, timbre: tuple[tuple[str, float], ...]) -> float:
    v_ref = drivetrain_spec(spec, sample_rate_hz).v_ref
    program = DrivetrainProgram(seed=0, sample_rate_hz=sample_rate_hz, block_size=REFERENCE_BLOCK_SIZE, tuning={**dict(timbre), TRIM_DB: 0.0}, spec=spec)
    program.update(SourceSpec(id="reference", kind="", asset_id="", model=DrivetrainModel.name, agent_kind=AgentKind.ROBOT, position=(0.0, 0.0, 0.0), level_db=0.0, state={LEFT_VELOCITY: v_ref, RIGHT_VELOCITY: v_ref}))
    warmup = -(-int(REFERENCE_WARMUP_S * sample_rate_hz) // REFERENCE_BLOCK_SIZE)
    blocks = -(-int(REFERENCE_S * sample_rate_hz) // REFERENCE_BLOCK_SIZE)
    for _ in range(warmup):
        program.render(REFERENCE_BLOCK_SIZE)
    return float(rms(np.concatenate([program.render(REFERENCE_BLOCK_SIZE) for _ in range(blocks)])))


class DrivetrainProgram:
    """Mono drivetrain stream on the shared noise field, phase-offset by the source seed."""

    def __init__(self, *, seed: int, sample_rate_hz: int, block_size: int, tuning: Mapping[str, float], spec: str = DEFAULT_SPEC) -> None:
        self.block_size = int(block_size)
        self.sample_rate_hz = int(sample_rate_hz)
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate must be positive")
        self._spec = drivetrain_spec(spec, self.sample_rate_hz)
        phase_index = int(seed) & 0x0FFFFFFF
        prewarm_field(self._spec, seed=DRIVETRAIN_FIELD_SEED)
        self._left = DrivetrainVoice(self._spec, index=phase_index * 2, count=2, seed=DRIVETRAIN_FIELD_SEED, transfer=False)
        self._right = DrivetrainVoice(self._spec, index=phase_index * 2 + 1, count=2, seed=DRIVETRAIN_FIELD_SEED, transfer=False)
        self._lock = threading.Lock()
        self._target_left = 0.0
        self._target_right = 0.0
        self._current_left = 0.0
        self._current_right = 0.0
        self._target_gain = 0.0
        self._current_gain = 0.0
        self._target_volume_gain = _volume_gain(tuning)
        self._current_volume_gain = self._target_volume_gain
        self._frequency_scale = float(tuning[FREQUENCY_SCALE])
        self._tonal_gain_db = float(tuning[TONAL_GAIN_DB])
        self._broadband_gain_db = float(tuning[BROADBAND_GAIN_DB])
        self._speed_exponent = float(tuning[SPEED_EXPONENT])
        self._velocity_smoothing_s = float(tuning[VELOCITY_SMOOTHING_S])
        self._active = False
        self._inactive_frames = 0
        self._release_frames = max(int(self.sample_rate_hz * RELEASE_S), 1)

    def update(self, source: SourceSpec) -> None:
        active = bool(source.active)
        with self._lock:
            self._target_left = source.state_value(LEFT_VELOCITY) if active else 0.0
            self._target_right = source.state_value(RIGHT_VELOCITY) if active else 0.0
            self._target_gain = 1.0 if active else 0.0
            self._active = active
            if active:
                self._inactive_frames = 0

    def tune(self, tuning: Mapping[str, float]) -> None:
        with self._lock:
            self._target_volume_gain = _volume_gain(tuning)
            self._frequency_scale = float(tuning[FREQUENCY_SCALE])
            self._tonal_gain_db = float(tuning[TONAL_GAIN_DB])
            self._broadband_gain_db = float(tuning[BROADBAND_GAIN_DB])
            self._speed_exponent = float(tuning[SPEED_EXPONENT])
            self._velocity_smoothing_s = float(tuning[VELOCITY_SMOOTHING_S])

    def render(self, frames: int) -> np.ndarray:
        if int(frames) != self.block_size:
            raise ValueError(f"drivetrain source configured for {self.block_size} frames, audio callback requested {frames}")
        with self._lock:
            target_left = self._target_left
            target_right = self._target_right
            target_gain = self._target_gain
            target_volume_gain = self._target_volume_gain
            frequency_scale = self._frequency_scale
            tonal_gain_db = self._tonal_gain_db
            broadband_gain_db = self._broadband_gain_db
            speed_exponent = self._speed_exponent
            velocity_smoothing_s = self._velocity_smoothing_s

        if velocity_smoothing_s <= 0.0:
            left_speed = np.full(frames, target_left, dtype=np.float64)
            right_speed = np.full(frames, target_right, dtype=np.float64)
        else:
            decay = np.exp(-np.arange(1, frames + 1, dtype=np.float64) / (self.sample_rate_hz * velocity_smoothing_s))
            left_speed = target_left + (self._current_left - target_left) * decay
            right_speed = target_right + (self._current_right - target_right) * decay
        self._current_left = float(left_speed[-1])
        self._current_right = float(right_speed[-1])
        render_options = {
            "frequency_scale": frequency_scale,
            "tonal_gain_db": tonal_gain_db,
            "broadband_gain_db": broadband_gain_db,
            "speed_exponent": speed_exponent,
        }
        dry = self._left.render(left_speed, **render_options) + self._right.render(right_speed, **render_options)

        gain = np.linspace(self._current_gain, target_gain, frames, dtype=np.float32)
        self._current_gain = target_gain
        dry = np.asarray(dry, dtype=np.float32) * gain
        volume_gain = np.linspace(self._current_volume_gain, target_volume_gain, frames, dtype=np.float32)
        self._current_volume_gain = target_volume_gain
        dry *= volume_gain
        mono = np.asarray(dry, dtype=np.float32)
        with self._lock:
            if not self._active:
                self._inactive_frames += frames
        return mono

    @property
    def finished(self) -> bool:
        with self._lock:
            velocity_tail_frames = int(self.sample_rate_hz * self._velocity_smoothing_s * 5.0)
            return not self._active and self._inactive_frames >= max(self.block_size, velocity_tail_frames) + self._release_frames


class DrivetrainModel:
    """Procedural drivetrain for variants with params {spec: <name>}."""

    name: typing.ClassVar[str] = "drivetrain"
    continuous: typing.ClassVar[bool] = True
    streams: typing.ClassVar[bool] = True
    tuning_group: typing.ClassVar[type[ParamGroup] | None] = MotorGroup

    @classmethod
    def program(cls, source: SourceSpec, asset: SoundAsset, variant: Variant, context: ProgramContext) -> DrivetrainProgram:
        del asset
        return cls.stream(
            seed=source.seed,
            sample_rate_hz=context.sample_rate_hz,
            block_size=context.block_size,
            params=variant.params,
            tuning=tuning_of(MotorGroup.defaults()),
        )

    @classmethod
    def stream(cls, *, seed: int, sample_rate_hz: int, block_size: int, params: Mapping[str, object], tuning: Mapping[str, float]) -> DrivetrainProgram:
        return DrivetrainProgram(seed=seed, sample_rate_hz=sample_rate_hz, block_size=block_size, tuning=tuning, spec=spec_name(params))

    @classmethod
    def prewarm(cls, sample_rate_hz: int, params: Mapping[str, object]) -> None:
        """Build the shared noise field and the default-tuning reference of the params' spec at sample_rate_hz. Raises ValueError for an unknown spec."""
        prewarm_field(drivetrain_spec(spec_name(params), sample_rate_hz), seed=DRIVETRAIN_FIELD_SEED)
        cls.reference_rms(sample_rate_hz, params, tuning_of(MotorGroup.defaults()))

    @classmethod
    def reference_rms(cls, sample_rate_hz: int, params: Mapping[str, object], tuning: Mapping[str, float]) -> float:
        """RMS at both wheels at the spec's v_ref and zero trim, the operating point the asset level_db refers to."""
        return _reference_rms(spec_name(params), int(sample_rate_hz), tuple(sorted((name, float(value)) for name, value in tuning.items() if name != TRIM_DB)))
