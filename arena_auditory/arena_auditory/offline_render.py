"""Offline driver of the array renderer: replays a recorded render trace and its room impulses through render_block, one process per episode."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from arena_robots.audio import ArrayStream
from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary, split_sample_key
from mcap.exceptions import McapError
from scipy.io import wavfile

from arena_auditory.assets import DecodedSample, SampleDecoder
from arena_auditory.constants import ROOM_IMPULSES
from arena_auditory.params import RenderGroup
from arena_auditory.render.core import (
    V1_CHANNELS,
    ImpulseShape,
    RenderInputs,
    RenderParams,
    RenderResult,
    RenderState,
    impulse_shape,
    new_state,
    no_impulse,
    render_block,
    render_inputs_from_json,
)

TRACE_TOPIC_SUFFIX = f"/audio/{ArrayStream.RENDER_INPUTS}"
RAW_TOPIC_SUFFIX = f"/audio/{ArrayStream.RAW}"
MOTOR_TOPIC_SUFFIX = f"/audio/{ArrayStream.STEM_MOTOR}"
IMPULSE_TOPIC_SUFFIX = f"/{ROOM_IMPULSES}"

MODES = ("render", "verify", "remix")

EPISODE_ERRORS = (OSError, LookupError, ValueError, TypeError, yaml.YAMLError, McapError)


@dataclass(frozen=True)
class RecordedImpulse:
    samples: np.ndarray
    sample_rate_hz: int
    lead_samples: int


@dataclass(frozen=True)
class EpisodeTrace:
    """One recording's render trace, its room impulses and the audio the live renderer published."""

    robot: str
    channels: int
    sample_rate: int
    block_size: int
    blocks: tuple[str, ...]
    raw: tuple[np.ndarray, ...]
    motor: tuple[np.ndarray, ...]
    impulses: Mapping[str, RecordedImpulse] = field(default_factory=dict)


@dataclass(frozen=True)
class Divergence:
    stream: str
    block: int
    channel: int
    sample: int
    replayed: float
    recorded: float


@dataclass(frozen=True)
class EpisodeJob:
    path: Path
    mode: str
    rir_crossfade_s: float
    out_dir: Path | None = None
    robot: str = ""
    world: Path | None = None
    attenuation_db: float = 0.0
    sample_rate: int = 0
    block_size: int = 0


@dataclass(frozen=True)
class EpisodeReport:
    path: Path
    mode: str
    blocks: int = 0
    output: Path | None = None
    divergence: Divergence | None = None
    clipped_blocks: tuple[int, ...] = ()
    first_block: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.divergence is None


def audio_block(
    topic: str,
    *,
    encoding: str,
    interleaved: bool,
    channel_count: int,
    frame_count: int,
    data: npt.ArrayLike,
) -> np.ndarray:
    """Deinterleave one recorded AudioFrame payload into a channels-first block."""
    if encoding != "32FC1" or not interleaved:
        raise ValueError(f"{topic}: expected an interleaved 32FC1 AudioFrame, got encoding={encoding!r} interleaved={interleaved!r}")
    values = np.asarray(data, dtype="<f4")
    if channel_count <= 0 or frame_count <= 0 or values.size != channel_count * frame_count:
        raise ValueError(f"{topic}: malformed AudioFrame dimensions ({channel_count} x {frame_count}, {values.size} values)")
    return np.ascontiguousarray(values.reshape(frame_count, channel_count).T)


def trace_channels(text: str) -> int:
    """Channel count of one trace block, 4 for a version 1 trace."""
    return int(json.loads(text).get("channels", V1_CHANNELS))


def read_episode_trace(path: Path, *, robot: str = "", full_audio: bool = True) -> EpisodeTrace:
    """Render trace, room impulses and published audio of one recording. full_audio=False keeps only the first block of each audio stream."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    blocks: list[str] = []
    raw: list[np.ndarray] = []
    motor: list[np.ndarray] = []
    impulses: dict[str, RecordedImpulse] = {}
    robots: set[str] = set()
    sample_rates: set[int] = set()

    with path.open("rb") as source:
        reader = make_reader(source, decoder_factories=[DecoderFactory()])
        for _, channel, _, message in reader.iter_decoded_messages(log_time_order=True):
            topic = "/" + channel.topic.strip("/")
            if topic.endswith(IMPULSE_TOPIC_SUFFIX):
                impulses.setdefault(
                    str(message.key),
                    RecordedImpulse(samples=np.asarray(message.samples, dtype=np.float32), sample_rate_hz=int(message.sample_rate_hz), lead_samples=int(message.lead_samples)),
                )
                continue
            for suffix in (TRACE_TOPIC_SUFFIX, RAW_TOPIC_SUFFIX, MOTOR_TOPIC_SUFFIX):
                if not topic.endswith(suffix):
                    continue
                name = topic[: -len(suffix)].rsplit("/", 1)[-1]
                if robot and name != robot:
                    continue
                robots.add(name)
                if suffix == TRACE_TOPIC_SUFFIX:
                    blocks.append(str(message.data))
                else:
                    sample_rates.add(int(message.sample_rate))
                    stream = raw if suffix == RAW_TOPIC_SUFFIX else motor
                    if full_audio or not stream:
                        stream.append(
                            audio_block(
                                topic,
                                encoding=str(message.encoding),
                                interleaved=bool(message.interleaved),
                                channel_count=int(message.channel_count),
                                frame_count=int(message.frame_count),
                                data=message.data,
                            )
                        )

    if len(robots) > 1:
        raise ValueError(f"{path}: recording carries several robots {sorted(robots)}, select one with --robot")
    if not blocks:
        raise ValueError(f"{path}: no render trace found on */{TRACE_TOPIC_SUFFIX.strip('/')}")
    if len(sample_rates) > 1:
        raise ValueError(f"{path}: recorded audio mixes sample rates {sorted(sample_rates)}")
    channels = trace_channels(blocks[0])
    recorded = raw or motor
    if recorded and recorded[0].shape[0] != channels:
        raise ValueError(f"{path}: the trace renders {channels} channels, the recorded audio carries {recorded[0].shape[0]}")
    return EpisodeTrace(
        robot=next(iter(robots), ""),
        channels=channels,
        sample_rate=next(iter(sample_rates), 0),
        block_size=int(recorded[0].shape[1]) if recorded else 0,
        blocks=tuple(blocks),
        raw=tuple(raw),
        motor=tuple(motor),
        impulses=impulses,
    )


def legacy_owner(library: SoundLibrary, variant_id: str) -> str:
    """The sound asset owning a bare variant id, the sample key of traces recorded before keys carried the asset id. Raises KeyError unless exactly one owns it."""
    owners = []
    for asset_id in library.known_asset_ids():
        try:
            asset = library.asset(asset_id)
        except (KeyError, ValueError, FileNotFoundError):
            continue
        if any(variant.id == variant_id for variant in asset.variants):
            owners.append(asset.id)
    if len(owners) != 1:
        raise KeyError(f"legacy sample key {variant_id!r} needs exactly one sound asset with that variant, found {owners}")
    return owners[0]


def asset_resolver(sample_rate: int, world: Path | None = None) -> tuple[Callable[[str], np.ndarray], Callable[[str], float]]:
    """The live renderer's asset_key -> mono samples and asset_key -> active level mappings, through the sound library."""
    library = SoundLibrary.default()
    library.use_world(world)
    decoder = SampleDecoder(library, sample_rate)
    owners: dict[str, str] = {}

    def sample(asset_key: str) -> DecodedSample:
        try:
            split_sample_key(asset_key)
        except KeyError:
            if asset_key not in owners:
                owners[asset_key] = legacy_owner(library, asset_key)
            return decoder.load(owners[asset_key], asset_key)
        return decoder.by_key(asset_key)

    def resolve(asset_key: str) -> np.ndarray:
        return sample(asset_key).samples

    def level(asset_key: str) -> float:
        return sample(asset_key).active_rms

    return resolve, level


def impulse_resolver(impulses: Mapping[str, RecordedImpulse], sample_rate: int) -> Callable[[str], ImpulseShape | None]:
    """Recorded room impulses at the render rate, shaped the way the live renderer shapes them."""
    shapes: dict[str, ImpulseShape] = {}

    def impulse(key: str) -> ImpulseShape | None:
        shape = shapes.get(key)
        if shape is None:
            recorded = impulses.get(key)
            if recorded is None:
                return None
            shape = shapes[key] = impulse_shape(recorded.samples, recorded.sample_rate_hz, recorded.lead_samples, sample_rate)
        return shape

    return impulse


def decode_blocks(
    texts: Iterable[str],
    resolve: Callable[[str], np.ndarray],
    level: Callable[[str], float],
    impulse: Callable[[str], ImpulseShape | None] = no_impulse,
) -> Iterator[RenderInputs]:
    for text in texts:
        yield render_inputs_from_json(text, resolve, level, impulse)


def replay(
    blocks: Iterable[RenderInputs],
    params: RenderParams,
    *,
    state: RenderState | None = None,
) -> Iterator[tuple[RenderInputs, RenderResult]]:
    """Drive render_block over a trace, carrying state exactly as the live renderer does."""
    carried = new_state(params.channels) if state is None else state
    for block in blocks:
        if block.reset:
            carried = new_state(params.channels)
        carried.cursor = block.block_index * params.block_size
        yield block, render_block(carried, block, params)


def render_results(results: Iterable[tuple[RenderInputs, RenderResult]]) -> np.ndarray:
    """Raw blocks at their block index, sample 0 being the first traced block."""
    return place_blocks([(inputs.block_index, result.raw) for inputs, result in results])


def verify_results(
    results: Iterable[tuple[RenderInputs, RenderResult]],
    raw: Sequence[np.ndarray],
    motor: Sequence[np.ndarray],
) -> Divergence | None:
    """Compare a replay against the recorded streams, first mismatch wins."""
    for index, (block, result) in enumerate(results):
        if index < len(raw):
            divergence = first_divergence(ArrayStream.RAW.value, block.block_index, result.raw, raw[index])
            if divergence is not None:
                return divergence
        if index < len(motor):
            divergence = first_divergence(ArrayStream.STEM_MOTOR.value, block.block_index, result.motor, motor[index])
            if divergence is not None:
                return divergence
    return None


def remix_results(
    results: Iterable[tuple[RenderInputs, RenderResult]],
    attenuation_db: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Re-balance ego noise block by block, listing the blocks whose remix clips."""
    blocks: list[tuple[int, np.ndarray]] = []
    clipped: list[int] = []
    for inputs, result in results:
        mix = remix_block(result.ped, result.ambient, result.motor, attenuation_db)
        if np.any((mix < -1.0) | (mix > 1.0)):
            clipped.append(inputs.block_index)
        blocks.append((inputs.block_index, np.clip(mix, -1.0, 1.0)))
    return place_blocks(blocks), tuple(clipped)


def remix_block(ped: np.ndarray, ambient: np.ndarray, motor: np.ndarray, attenuation_db: float) -> np.ndarray:
    """The renderer's pre-clip sum with the motor stem attenuated by attenuation_db."""
    return ped + ambient + np.float32(10.0 ** (-attenuation_db / 20.0)) * motor


def first_divergence(stream: str, block: int, replayed: np.ndarray, recorded: np.ndarray) -> Divergence | None:
    if replayed.shape != recorded.shape:
        raise ValueError(f"block {block} {stream}: replayed {replayed.shape} against recorded {recorded.shape}")
    if np.array_equal(replayed, recorded):
        return None
    channel, sample = (int(index) for index in np.argwhere(replayed != recorded)[0])
    return Divergence(
        stream=stream,
        block=block,
        channel=channel,
        sample=sample,
        replayed=float(replayed[channel, sample]),
        recorded=float(recorded[channel, sample]),
    )


def write_wav(path: Path, channels: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.ascontiguousarray(channels.T, dtype=np.float32))


def place_blocks(blocks: Sequence[tuple[int, np.ndarray]]) -> np.ndarray:
    """Each block written at its index relative to the first, skipped blocks silent."""
    first = min(index for index, _ in blocks)
    count, block_size = blocks[0][1].shape
    audio = np.zeros((count, (max(index for index, _ in blocks) - first + 1) * block_size), dtype=np.float32)
    for index, block in blocks:
        start = (index - first) * block_size
        audio[:, start : start + block_size] = block
    return audio


def render_episode(job: EpisodeJob) -> EpisodeReport:
    """Worker entry point: one recording in, one report (and maybe a wav) out."""
    try:
        return _render_episode(job)
    except EPISODE_ERRORS as exc:
        return EpisodeReport(path=job.path, mode=job.mode, error=f"{type(exc).__name__}: {exc}")


def _render_episode(job: EpisodeJob) -> EpisodeReport:
    trace = read_episode_trace(job.path, robot=job.robot, full_audio=job.mode == "verify")
    sample_rate = job.sample_rate or trace.sample_rate
    block_size = job.block_size or trace.block_size
    if sample_rate <= 0 or block_size <= 0:
        raise ValueError(f"{job.path}: no recorded audio to read the block geometry from, pass --sample-rate and --block-size")
    resolve, level = asset_resolver(sample_rate, job.world)
    params = RenderParams(
        channels=trace.channels,
        block_size=block_size,
        sample_rate=sample_rate,
        resolve=resolve,
        impulse=impulse_resolver(trace.impulses, sample_rate),
        rir_crossfade_frames=max(int(sample_rate * job.rir_crossfade_s), 1),
    )
    results = replay(decode_blocks(trace.blocks, params.resolve, level, params.impulse), params)
    first_block = int(json.loads(trace.blocks[0])["block_index"])

    if job.mode == "verify":
        if not trace.raw and not trace.motor:
            raise ValueError(f"{job.path}: nothing to verify against, neither {ArrayStream.RAW} nor {ArrayStream.STEM_MOTOR} was recorded")
        for name, recorded in ((ArrayStream.RAW, trace.raw), (ArrayStream.STEM_MOTOR, trace.motor)):
            if recorded and len(recorded) != len(trace.blocks):
                raise ValueError(f"{job.path}: {len(trace.blocks)} trace blocks against {len(recorded)} recorded {name} blocks")
        return EpisodeReport(
            path=job.path,
            mode=job.mode,
            blocks=len(trace.blocks),
            divergence=verify_results(results, trace.raw, trace.motor),
        )

    output = (job.out_dir or job.path.parent) / _output_name(job)
    if job.mode == "remix":
        audio, clipped = remix_results(results, job.attenuation_db)
        write_wav(output, audio, sample_rate)
        return EpisodeReport(path=job.path, mode=job.mode, blocks=len(trace.blocks), output=output, clipped_blocks=clipped, first_block=first_block)
    audio = render_results(results)
    write_wav(output, audio, sample_rate)
    return EpisodeReport(path=job.path, mode=job.mode, blocks=len(trace.blocks), output=output, first_block=first_block)


def _output_name(job: EpisodeJob) -> str:
    if job.mode == "remix":
        return f"{job.path.stem}.remix_{job.attenuation_db:g}db.wav"
    return f"{job.path.stem}.render.wav"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="auditory_offline_render",
        description="Replay recorded episodes through the live array renderer.",
    )
    parser.add_argument(
        "mode",
        choices=MODES,
        help="render: write the replayed array audio to a wav. verify: assert the replay is bit-identical to the recorded raw_array and stem_motor, which holds only for a recording made with array.enabled=true and array.muted=false. remix: write the audio with the motor stem attenuated by --attenuate-db.",
    )
    parser.add_argument("recordings", nargs="+", type=Path, metavar="RECORDING.mcap", help="episode recordings, rendered one per worker process")
    parser.add_argument("--out", type=Path, default=None, help="directory for render/remix wavs (default: beside each recording)")
    parser.add_argument("--attenuate-db", type=float, default=0.0, metavar="DB", help="remix: attenuation applied to the motor stem, in dB (0 reproduces the original mix)")
    parser.add_argument("--robot", default="", help="robot name, needed only when a recording carries more than one")
    parser.add_argument("--world", type=Path, default=None, help="world directory whose world-local sounds the recording used (default: shared and bucket sounds only)")
    parser.add_argument("--sample-rate", type=int, default=0, metavar="HZ", help="override the sample rate read from the recorded audio")
    parser.add_argument("--block-size", type=int, default=0, metavar="FRAMES", help="override the block size read from the recorded audio")
    parser.add_argument("--rir-crossfade-s", type=float, default=float(RenderGroup.RIR_CROSSFADE_S.default), metavar="S", help="render.rir.crossfade_s of the recording, used when room impulses change key (default: the renderer default)")
    parser.add_argument("--jobs", type=int, default=0, metavar="N", help="worker processes (default: one per CPU, capped at the number of recordings)")
    return parser.parse_args()


def _run(jobs: Sequence[EpisodeJob], workers: int) -> list[EpisodeReport]:
    if workers <= 1:
        return [render_episode(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(render_episode, jobs))


def main() -> None:
    args = _parse_args()
    jobs = [
        EpisodeJob(
            path=recording,
            mode=args.mode,
            out_dir=args.out,
            robot=args.robot,
            world=args.world,
            attenuation_db=args.attenuate_db,
            sample_rate=args.sample_rate,
            block_size=args.block_size,
            rir_crossfade_s=args.rir_crossfade_s,
        )
        for recording in args.recordings
    ]
    workers = args.jobs if args.jobs > 0 else min(len(jobs), os.cpu_count() or 1)

    failed = 0
    for report in _run(jobs, workers):
        if report.error:
            failed += 1
            print(f"FAIL   {report.path}: {report.error}", file=sys.stderr)
            continue
        if report.divergence is not None:
            failed += 1
            divergence = report.divergence
            print(f"DIFFER {report.path}: {divergence.stream} block {divergence.block} channel {divergence.channel} sample {divergence.sample} replayed {divergence.replayed!r} recorded {divergence.recorded!r}", file=sys.stderr)
            continue
        detail = f" -> {report.output} (sample 0 is block {report.first_block})" if report.output else ""
        print(f"OK     {report.path}: {report.blocks} blocks{detail}")
        if report.clipped_blocks:
            print(f"       {len(report.clipped_blocks)} block(s) clip in the remix, first {report.clipped_blocks[:8]}", file=sys.stderr)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
