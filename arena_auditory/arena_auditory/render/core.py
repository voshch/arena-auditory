"""Block renderer for an N-channel microphone array, free of ROS: resolved source descriptions in, one PCM block per stem out."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np
from scipy.signal import fftconvolve

from arena_auditory.assets import Stem
from arena_auditory.render.dsp import (
    PartitionedConvolver,
    calibrate_mems,
    fractional_delay,
    ramped_read,
    resample_impulse,
    streaming_fractional_delays,
)
from arena_auditory.shared import AgentKind, SourceSpec
from arena_auditory.sources import StreamProgram, stream_model
from arena_auditory.sources.drivetrain.program import (
    BROADBAND_GAIN_DB,
    DEFAULT_SPEC,
    FREQUENCY_SCALE,
    LEFT_VELOCITY,
    RIGHT_VELOCITY,
    SPEC,
    SPEED_EXPONENT,
    TONAL_GAIN_DB,
    TRIM_DB,
    VELOCITY_SMOOTHING_S,
    DrivetrainModel,
)

TRACE_VERSION = 2
V1_CHANNELS = 4
IMPULSE_FLOOR_DB = -60.0

_LOG = logging.getLogger(__name__)


def _log_error(message: str) -> None:
    _LOG.error(message)


@dataclass(frozen=True)
class ImpulseShape:
    """Room response at the render rate, lead_samples before the direct arrival."""

    samples: np.ndarray
    lead_samples: int


def no_impulse(_key: str) -> ImpulseShape | None:
    return None


def trim_impulse(samples: np.ndarray, lead_samples: int, floor_db: float = IMPULSE_FLOOR_DB) -> np.ndarray:
    """The impulse cut where its remaining energy falls floor_db below its total, and no earlier than the direct arrival."""
    energy = np.square(samples, dtype=np.float64)
    remaining = np.cumsum(energy[::-1])[::-1]
    if remaining.size == 0 or remaining[0] <= 0.0:
        return samples
    below = np.flatnonzero(remaining < remaining[0] * 10.0 ** (floor_db / 10.0))
    end = max(int(below[0]) if below.size else samples.size, min(int(lead_samples) + 1, samples.size))
    return samples[:end]


def impulse_shape(samples: np.ndarray, sample_rate_hz: int, lead_samples: int, render_rate_hz: int) -> ImpulseShape:
    """A RoomImpulse payload resampled to the render rate, lead scaled with it, its tail trimmed."""
    lead = round(int(lead_samples) * render_rate_hz / sample_rate_hz)
    return ImpulseShape(samples=trim_impulse(resample_impulse(samples, sample_rate_hz, render_rate_hz), lead), lead_samples=lead)


@dataclass(frozen=True)
class ClipInput:
    """One finite arrival, already calibrated, convolved and fractionally delayed."""

    channel: int
    start: int
    asset_key: str
    samples: np.ndarray
    anchor: int
    received_volume_db: float
    sensitivity_dbfs_at_94_dbspl: float
    delay_samples: float
    rir_key: str = ""
    stem: Stem = "pedestrian"


@dataclass(frozen=True)
class ContinuousInput:
    source_id: str
    channel: int
    asset_key: str
    gain: float
    delay_target: float
    loop: bool
    program_start: int
    rir_key: str = ""
    stem: Stem = "ambient"


@dataclass(frozen=True)
class StreamInput:
    """One streamed source in full: the model's program inputs and its per-channel arrival."""

    source_id: str
    model: str
    seed: int
    params: Mapping[str, object]
    state: Mapping[str, float]
    tuning: Mapping[str, float]
    gains: tuple[float, ...]
    delay_samples: tuple[float, ...]
    active_channels: tuple[bool, ...]
    active: bool
    source_agent_id: int
    source_agent_name: str
    kind: str
    asset_id: str
    rir_keys: tuple[str, ...] = ()
    stem: Stem = "motor"


@dataclass(frozen=True)
class RenderInputs:
    """Everything one block needs: clips starting now, live sources in full."""

    block_index: int
    clips: tuple[ClipInput, ...]
    continuous: tuple[ContinuousInput, ...]
    streams: tuple[StreamInput, ...]
    channels: int = V1_CHANNELS
    reset: bool = False


@dataclass(frozen=True)
class RenderParams:
    channels: int
    block_size: int
    sample_rate: int
    resolve: Callable[[str], np.ndarray]
    impulse: Callable[[str], ImpulseShape | None] = no_impulse
    rir_crossfade_frames: int = 1
    log_error: Callable[[str], None] = _log_error


class RirStage:
    """One voice channel's room response, crossfading with equal power whenever its key changes."""

    def __init__(self, crossfade_frames: int) -> None:
        self.key = ""
        self.lead_samples = 0
        self.tail_frames = 0
        self._total = max(int(crossfade_frames), 1)
        self._remaining = 0
        self._current: Callable[[np.ndarray], np.ndarray] | None = None
        self._previous: Callable[[np.ndarray], np.ndarray] | None = None

    def retarget(self, key: str, params: RenderParams, *, crossfade: bool) -> None:
        if key == self.key:
            return
        shape = params.impulse(key) if key else None
        if key and shape is None:
            params.log_error(f"room impulse {key!r} is unknown, rendering dry")
        self._previous = self._current
        self._current = None if shape is None else _filter(shape.samples, params.block_size)
        self.key = key
        self.lead_samples = 0 if shape is None else int(shape.lead_samples)
        self.tail_frames = max(self.tail_frames, 0 if shape is None else len(shape.samples))
        self._remaining = self._total if crossfade else 0
        if not crossfade:
            self._previous = None

    def process(self, block: np.ndarray) -> np.ndarray:
        wet = block if self._current is None else self._current(block)
        if self._remaining <= 0:
            return wet
        old = block if self._previous is None else self._previous(block)
        elapsed = self._total - self._remaining
        alpha = np.clip((elapsed + np.arange(block.size)) / self._total, 0.0, 1.0).astype(np.float32)
        mixed = old * np.sqrt(1.0 - alpha) + wet * np.sqrt(alpha)
        self._remaining = max(self._remaining - block.size, 0)
        if self._remaining == 0:
            self._previous = None
        return np.asarray(mixed, dtype=np.float32)

    @property
    def release_frames(self) -> int:
        return self.tail_frames + self._total


def _filter(impulse: np.ndarray, block_size: int) -> Callable[[np.ndarray], np.ndarray]:
    taps = np.asarray(impulse, dtype=np.float32).reshape(-1)
    if taps.size == 1:
        tap = taps[0]
        return lambda block: block * tap
    return PartitionedConvolver(taps, block_size).process


def convolve(samples: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """Full linear convolution, a single tap is an exact scalar gain."""
    taps = np.asarray(impulse, dtype=np.float32).reshape(-1)
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if taps.size == 1:
        return np.ascontiguousarray(mono * taps[0])
    return np.ascontiguousarray(fftconvolve(mono, taps, mode="full"), dtype=np.float32)


def _retarget(stage: RirStage | None, key: str, params: RenderParams, *, fresh: bool) -> RirStage | None:
    if stage is None:
        if not key:
            return None
        stage = RirStage(params.rir_crossfade_frames)
        stage.retarget(key, params, crossfade=False)
        return stage
    stage.retarget(key, params, crossfade=not fresh)
    return stage


def _delay(delay: float, stage: RirStage | None) -> float:
    if stage is None or stage.lead_samples <= 0:
        return delay
    return max(delay - stage.lead_samples, 0.0)


@dataclass(slots=True)
class ScheduledClip:
    start: int
    samples: np.ndarray
    stem: Stem = "pedestrian"


@dataclass(slots=True)
class ContinuousVoice:
    samples: np.ndarray
    program_start: int
    delay_samples: float
    delay_target: float
    gain: float
    loop: bool
    active: bool
    stem: Stem = "ambient"
    rir: RirStage | None = None
    released_frames: int = -1
    asset_key: str = ""


@dataclass(slots=True)
class StreamVoice:
    program: StreamProgram
    model: str
    seed: int
    params: Mapping[str, object]
    gains: np.ndarray
    delay_samples: np.ndarray
    active_channels: np.ndarray
    source_id: str
    source_agent_id: int
    source_agent_name: str
    kind: str
    asset_id: str
    stem: Stem = "motor"
    rir: dict[int, RirStage] = field(default_factory=dict)
    history: np.ndarray | None = None
    inactive_frames: int = 0


@dataclass
class RenderState:
    channels: int
    cursor: int = 0
    clips: list[list[ScheduledClip]] = field(init=False)
    continuous: dict[tuple[str, int], ContinuousVoice] = field(default_factory=dict)
    streams: dict[str, StreamVoice] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.clips = [[] for _ in range(self.channels)]


@dataclass
class RenderResult:
    raw: np.ndarray
    ped: np.ndarray
    ambient: np.ndarray
    motor: np.ndarray
    clipped_samples: int


def new_state(channels: int) -> RenderState:
    return RenderState(channels=channels)


def prepare_clip(
    mono: np.ndarray,
    received_volume_db: float,
    sensitivity_dbfs_at_94_dbspl: float,
    delay_samples: float,
    *,
    level_rms: float,
    impulse: ImpulseShape | None = None,
) -> tuple[int, np.ndarray]:
    """Calibrate one arrival, convolve it with its room response, then split the delay left after the impulse lead."""
    calibrated = calibrate_mems(
        mono,
        received_volume_db,
        level_rms,
        sensitivity_dbfs_at_94_dbspl=sensitivity_dbfs_at_94_dbspl,
    )
    if impulse is not None:
        calibrated = convolve(calibrated, impulse.samples)
        if impulse.lead_samples > 0:
            delay_samples = max(delay_samples - impulse.lead_samples, 0.0)
    return fractional_delay(calibrated, delay_samples)


def render_inputs_to_json(inputs: RenderInputs) -> str:
    """Serialize one block's inputs as trace version 2. Samples stay out, a clip is its scheduling parameters."""
    return json.dumps(
        {
            "version": TRACE_VERSION,
            "channels": inputs.channels,
            "block_index": inputs.block_index,
            "reset": inputs.reset,
            "clips": [
                {
                    "channel": clip.channel,
                    "start": clip.start,
                    "asset_key": clip.asset_key,
                    "anchor": clip.anchor,
                    "received_volume_db": clip.received_volume_db,
                    "sensitivity_dbfs_at_94_dbspl": clip.sensitivity_dbfs_at_94_dbspl,
                    "delay_samples": clip.delay_samples,
                    "rir_key": clip.rir_key,
                    "stem": clip.stem,
                }
                for clip in inputs.clips
            ],
            "continuous": [asdict(source) for source in inputs.continuous],
            "streams": [asdict(source) for source in inputs.streams],
        },
        separators=(",", ":"),
    )


def _require_impulse(impulse: Callable[[str], ImpulseShape | None], key: str, owner: str) -> ImpulseShape | None:
    if not key:
        return None
    shape = impulse(key)
    if shape is None:
        raise KeyError(f"{owner} needs room impulse {key!r}, which the trace source does not carry")
    return shape


def render_inputs_from_json(
    text: str,
    resolve: Callable[[str], np.ndarray],
    level: Callable[[str], float],
    impulse: Callable[[str], ImpulseShape | None] = no_impulse,
) -> RenderInputs:
    """Rebuild one block's inputs from a version 1 (4 channels, dry, fixed stems) or version 2 trace. Raises ValueError on a trace that does not replay, KeyError on a room impulse that impulse lacks."""
    payload = json.loads(text)
    version = int(payload.get("version", 1))
    if version not in (1, TRACE_VERSION):
        raise ValueError(f"unknown render trace version {version}")
    channels = int(payload.get("channels", V1_CHANNELS))
    clips: list[ClipInput] = []
    for entry in payload["clips"]:
        rir_key = str(entry.get("rir_key", ""))
        shape = _require_impulse(impulse, rir_key, f"clip {entry['asset_key']!r}")
        delay, samples = prepare_clip(
            resolve(entry["asset_key"]),
            entry["received_volume_db"],
            entry["sensitivity_dbfs_at_94_dbspl"],
            entry["delay_samples"],
            level_rms=level(entry["asset_key"]),
            impulse=shape,
        )
        start = entry["anchor"] + delay
        if start != entry["start"]:
            raise ValueError(f"clip {entry['asset_key']!r} replays at sample {start}, trace recorded {entry['start']}")
        clips.append(
            ClipInput(
                channel=entry["channel"],
                start=start,
                asset_key=entry["asset_key"],
                samples=samples,
                anchor=entry["anchor"],
                received_volume_db=entry["received_volume_db"],
                sensitivity_dbfs_at_94_dbspl=entry["sensitivity_dbfs_at_94_dbspl"],
                delay_samples=entry["delay_samples"],
                rir_key=rir_key,
                stem=cast(Stem, entry.get("stem", "pedestrian")),
            )
        )
    continuous = tuple(ContinuousInput(**entry) for entry in payload["continuous"])
    for source in continuous:
        _require_impulse(impulse, source.rir_key, f"continuous source {source.source_id!r}")
    streams = tuple(_v1_drivetrain(entry) for entry in payload["drivetrain"]) if version == 1 else tuple(_stream_input(entry) for entry in payload["streams"])
    for source in streams:
        for key in source.rir_keys:
            _require_impulse(impulse, key, f"stream {source.source_id!r}")
    return RenderInputs(
        block_index=payload["block_index"],
        clips=tuple(clips),
        continuous=continuous,
        streams=streams,
        channels=channels,
        reset=bool(payload.get("reset", False)),
    )


def _stream_input(entry: Mapping[str, Any]) -> StreamInput:
    return StreamInput(
        source_id=entry["source_id"],
        model=entry["model"],
        seed=entry["seed"],
        params=dict(entry["params"]),
        state={str(name): float(value) for name, value in entry["state"].items()},
        tuning={str(name): float(value) for name, value in entry["tuning"].items()},
        gains=tuple(entry["gains"]),
        delay_samples=tuple(entry["delay_samples"]),
        active_channels=tuple(entry["active_channels"]),
        active=entry["active"],
        source_agent_id=entry["source_agent_id"],
        source_agent_name=entry["source_agent_name"],
        kind=entry["kind"],
        asset_id=entry["asset_id"],
        rir_keys=tuple(str(key) for key in entry["rir_keys"]),
        stem=cast(Stem, entry["stem"]),
    )


def _v1_drivetrain(entry: Mapping[str, Any]) -> StreamInput:
    tuning = entry["tuning"]
    return StreamInput(
        source_id=entry["source_id"],
        model=DrivetrainModel.name,
        seed=entry["deterministic_seed"],
        params={SPEC: DEFAULT_SPEC},
        state={LEFT_VELOCITY: float(entry["left_velocity"]), RIGHT_VELOCITY: float(entry["right_velocity"])},
        tuning={
            TRIM_DB: float(tuning["volume_db"]),
            FREQUENCY_SCALE: float(tuning["frequency_scale"]),
            TONAL_GAIN_DB: float(tuning["tonal_gain_db"]),
            BROADBAND_GAIN_DB: float(tuning["broadband_gain_db"]),
            SPEED_EXPONENT: float(tuning["speed_exponent"]),
            VELOCITY_SMOOTHING_S: float(tuning["velocity_smoothing_seconds"]),
        },
        gains=tuple(entry["gains"]),
        delay_samples=tuple(entry["delay_samples"]),
        active_channels=tuple(entry["active_channels"]),
        active=entry["active"],
        source_agent_id=entry["source_agent_id"],
        source_agent_name=entry["source_agent_name"],
        kind=entry["sound_type"],
        asset_id=entry["asset_id"],
    )


def _ingest_clips(state: RenderState, inputs: RenderInputs) -> None:
    for clip in inputs.clips:
        state.clips[clip.channel].append(ScheduledClip(clip.start, clip.samples, clip.stem))


def _reconcile_continuous(state: RenderState, inputs: RenderInputs, params: RenderParams) -> None:
    live: set[tuple[str, int]] = set()
    for source in inputs.continuous:
        key = (source.source_id, source.channel)
        live.add(key)
        voice = state.continuous.get(key)
        if voice is None or voice.released_frames >= 0 or voice.program_start != source.program_start or voice.asset_key != source.asset_key:
            rir = _retarget(None, source.rir_key, params, fresh=True)
            state.continuous[key] = ContinuousVoice(
                samples=params.resolve(source.asset_key),
                program_start=source.program_start,
                delay_samples=_delay(source.delay_target, rir),
                delay_target=_delay(source.delay_target, rir),
                gain=source.gain,
                loop=source.loop,
                active=True,
                stem=source.stem,
                rir=rir,
                asset_key=source.asset_key,
            )
            continue
        voice.rir = _retarget(voice.rir, source.rir_key, params, fresh=False)
        voice.gain = source.gain
        voice.delay_target = _delay(source.delay_target, voice.rir)
        voice.active = True
    for key, voice in tuple(state.continuous.items()):
        if key in live:
            continue
        if voice.rir is None or voice.released_frames >= voice.rir.release_frames:
            state.continuous.pop(key)
        elif voice.released_frames < 0:
            voice.released_frames = 0
            voice.active = False


def _stream_source(source: StreamInput) -> SourceSpec:
    return SourceSpec(
        id=source.source_id,
        kind=source.kind,
        asset_id=source.asset_id,
        model=source.model,
        agent_kind=AgentKind.ROBOT,
        position=(0.0, 0.0, 0.0),
        level_db=0.0,
        agent_id=source.source_agent_id,
        agent_name=source.source_agent_name,
        active=source.active,
        seed=source.seed,
        state=source.state,
    )


def _reconcile_streams(state: RenderState, inputs: RenderInputs, params: RenderParams) -> None:
    live: set[str] = set()
    for source in inputs.streams:
        live.add(source.source_id)
        voice = state.streams.get(source.source_id)
        fresh = voice is None or voice.seed != source.seed or voice.model != source.model or voice.params != source.params
        if voice is None or fresh:
            voice = StreamVoice(
                program=stream_model(source.model).stream(
                    seed=source.seed,
                    sample_rate_hz=params.sample_rate,
                    block_size=params.block_size,
                    params=source.params,
                    tuning=source.tuning,
                ),
                model=source.model,
                seed=source.seed,
                params=source.params,
                gains=np.zeros(params.channels, dtype=np.float32),
                delay_samples=np.zeros(params.channels, dtype=np.float64),
                active_channels=np.zeros(params.channels, dtype=np.bool_),
                source_id=source.source_id,
                source_agent_id=source.source_agent_id,
                source_agent_name=source.source_agent_name,
                kind=source.kind,
                asset_id=source.asset_id,
                stem=source.stem,
            )
            state.streams[source.source_id] = voice
        for channel in range(params.channels):
            key = source.rir_keys[channel] if channel < len(source.rir_keys) else ""
            stage = _retarget(voice.rir.get(channel), key, params, fresh=fresh)
            if stage is not None:
                voice.rir[channel] = stage
        voice.gains = np.asarray(source.gains, dtype=np.float32)
        voice.delay_samples = np.asarray([_delay(float(delay), voice.rir.get(channel)) for channel, delay in enumerate(source.delay_samples)], dtype=np.float64)
        voice.active_channels = np.asarray(source.active_channels, dtype=np.bool_)
        if source.active:
            voice.inactive_frames = 0
        voice.program.tune(source.tuning)
        voice.program.update(_stream_source(source))
    for source_id in tuple(state.streams):
        if source_id not in live:
            state.streams.pop(source_id)


def _stream_finished(voice: StreamVoice) -> bool:
    if not voice.program.finished:
        return False
    return not voice.rir or voice.inactive_frames >= max(stage.release_frames for stage in voice.rir.values())


def render_block(state: RenderState, inputs: RenderInputs, params: RenderParams) -> RenderResult:
    """Render one block and advance the carried state by block_size samples."""
    if inputs.channels != params.channels or state.channels != params.channels:
        raise ValueError(f"render inputs carry {inputs.channels} channels and state {state.channels}, the renderer {params.channels}")
    _ingest_clips(state, inputs)
    _reconcile_continuous(state, inputs, params)
    _reconcile_streams(state, inputs, params)

    ped = np.zeros((params.channels, params.block_size), dtype=np.float32)
    ambient = np.zeros((params.channels, params.block_size), dtype=np.float32)
    motor = np.zeros((params.channels, params.block_size), dtype=np.float32)
    stems: dict[Stem, np.ndarray] = {"pedestrian": ped, "ambient": ambient, "motor": motor}
    block_start, block_end = state.cursor, state.cursor + params.block_size
    for channel, clips in enumerate(state.clips):
        active: list[ScheduledClip] = []
        for clip in clips:
            clip_end = clip.start + len(clip.samples)
            overlap_start = max(block_start, clip.start)
            overlap_end = min(block_end, clip_end)
            if overlap_start < overlap_end:
                dst = slice(overlap_start - block_start, overlap_end - block_start)
                src = slice(overlap_start - clip.start, overlap_end - clip.start)
                stems[clip.stem][channel, dst] += clip.samples[src]
            if clip_end > block_end:
                active.append(clip)
        state.clips[channel] = active

    for (_, channel), voice in tuple(state.continuous.items()):
        if voice.released_frames >= 0 and voice.rir is not None:
            stems[voice.stem][channel] += voice.rir.process(np.zeros(params.block_size, dtype=np.float32))
            voice.released_frames += params.block_size
            continue
        if not voice.active:
            continue
        contribution = (
            ramped_read(
                voice.samples,
                block_start - voice.program_start,
                voice.delay_samples,
                voice.delay_target,
                params.block_size,
                loop=voice.loop,
            )
            * voice.gain
        )
        if voice.rir is not None:
            contribution = voice.rir.process(contribution)
        stems[voice.stem][channel] += contribution
        voice.delay_samples = voice.delay_target
    for source_id, voice in tuple(state.streams.items()):
        try:
            mono = voice.program.render(params.block_size)
        except (ValueError, FloatingPointError, IndexError) as exc:
            params.log_error(f"{voice.model} stream render failed for {source_id!r}: {exc}")
            state.streams.pop(source_id, None)
            continue
        delayed, voice.history = streaming_fractional_delays(
            mono,
            voice.delay_samples,
            voice.history,
        )
        contribution = delayed * voice.gains[:, None]
        for channel, stage in voice.rir.items():
            contribution[channel] = stage.process(contribution[channel])
        stems[voice.stem] += contribution
        if not voice.active_channels.any():
            voice.inactive_frames += params.block_size
        if _stream_finished(voice):
            state.streams.pop(source_id, None)
    state.cursor = block_end
    output = ped + ambient + motor
    clipped = int(np.count_nonzero((output < -1.0) | (output > 1.0)))
    return RenderResult(
        raw=np.ascontiguousarray(np.clip(output, -1.0, 1.0)),
        ped=ped,
        ambient=ambient,
        motor=motor,
        clipped_samples=clipped,
    )
