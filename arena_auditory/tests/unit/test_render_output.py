from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
from arena_auditory.render.clock import RenderCursor
from arena_auditory.render.output import AudioOutput, JitterBuffer, output_candidates

RATE = 44100
BLOCK = 512
TARGET = round(0.04 * RATE)


def _device(name: str, outputs: int = 2) -> dict[str, object]:
    return {"name": name, "max_output_channels": outputs}


@pytest.mark.parametrize(
    ("devices", "default_index", "expected"),
    [
        ([_device("hw:0,0"), _device("pulse"), _device("default")], 0, [1, 2, 0]),
        ([_device("hw:0,0"), _device("pipewire")], 0, [1, 0]),
        ([_device("default")], 0, [0]),
        ([_device("hw:0,0"), _device("hw:1,0")], 1, [1]),
        ([_device("pulse")], 0, [0]),
        ([], -1, []),
        ([_device("hw:0,0")], -1, []),
        ([_device("pulse", outputs=0), _device("default")], 0, [1]),
        ([_device("mic", outputs=0)], 0, []),
    ],
)
def test_auto_output_candidates(devices: list[dict[str, object]], default_index: int, expected: list[int]) -> None:
    assert output_candidates("auto", devices, default_index, pulse_server=False) == expected


def test_auto_output_prefers_stereo_pulse_devices_under_a_pulse_server() -> None:
    devices = [_device("hw:0,0"), _device("PulseAudio Sound Server"), _device("pulse mono", outputs=1), _device("default")]
    assert output_candidates("auto", devices, 0, pulse_server=True) == [1, 3, 0]
    assert output_candidates("auto", devices, 0, pulse_server=False) == [3, 0]


@pytest.mark.parametrize(("requested", "expected"), [("3", [3]), (" hw:1,0 ", ["hw:1,0"]), ("none", []), ("", [])])
def test_explicit_and_disabled_output_requests(requested: str, expected: list[int | str]) -> None:
    assert output_candidates(requested, [_device("default")], 0, pulse_server=False) == expected


def test_disabled_output_opens_no_stream_and_drops_pushed_blocks() -> None:
    output = AudioOutput(device="none", sample_rate_hz=16000, channels=2, block_size=320, buffer_s=0.04, push_frames=320)
    assert output.open() == "output disabled"
    output.push(np.zeros((320, 2), dtype=np.float32))
    stats = output.stats
    assert not stats.active
    assert stats.queued == 0
    assert stats.error == "output disabled"
    output.close()


def test_output_push_rejects_a_block_of_the_wrong_layout() -> None:
    output = AudioOutput(device="none", sample_rate_hz=16000, channels=2, block_size=320, buffer_s=0.04, push_frames=320)
    with pytest.raises(ValueError, match="frames, 2"):
        output.push(np.zeros((2, 320), dtype=np.float32))


def test_output_latency_is_the_median_push_to_dac_time_of_played_blocks() -> None:
    output = AudioOutput(device="none", sample_rate_hz=16000, channels=2, block_size=320, buffer_s=0.02, push_frames=320)
    assert math.isnan(output.stats.latency_s)
    out = np.zeros((320, 2), dtype=np.float32)
    output._callback(out, 320, SimpleNamespace(outputBufferDacTime=5.010, currentTime=5.0), "")
    assert math.isnan(output.stats.latency_s)
    for _ in range(3):
        output._buffer.push(np.full((320, 2), 0.5, dtype=np.float32), stamp=time.monotonic() - 0.02)
        output._callback(out, 320, SimpleNamespace(outputBufferDacTime=5.010, currentTime=5.0), "")
    stats = output.stats
    assert stats.latency_s == pytest.approx(0.03, abs=0.01)
    assert (stats.callbacks, stats.underflows, stats.peak) == (4, 0, 0.5)


def _frames(count: int, value: float = 1.0) -> np.ndarray:
    return np.full((count, 2), value, dtype=np.float32)


def _read(buffer: JitterBuffer, frames: int = BLOCK) -> tuple[np.ndarray, float | None]:
    out = np.full((frames, 2), np.nan, dtype=np.float32)
    stamp = buffer.read(out)
    return out, stamp


def test_jitter_buffer_limit_is_target_plus_target_in_whole_push_blocks() -> None:
    assert JitterBuffer(target_frames=TARGET, push_frames=BLOCK).limit_frames == TARGET + 4 * BLOCK
    assert JitterBuffer(target_frames=1024, push_frames=BLOCK).limit_frames == 2048


def test_jitter_buffer_plays_silence_until_target_frames_are_queued() -> None:
    buffer = JitterBuffer(target_frames=TARGET, push_frames=BLOCK)
    for index in range(3):
        buffer.push(_frames(BLOCK), stamp=float(index))
        out, stamp = _read(buffer)
        assert buffer.priming
        assert stamp is None
        assert not out.any()
    assert buffer.queued == 3 * BLOCK
    buffer.push(_frames(BLOCK, 0.5), stamp=3.0)
    assert not buffer.priming
    out, stamp = _read(buffer)
    assert stamp == 0.0
    np.testing.assert_array_equal(out, 1.0)
    assert buffer.queued == 3 * BLOCK
    assert (buffer.underflows, buffer.overflows) == (0, 0)


def test_jitter_buffer_reads_across_pushed_chunks_in_order() -> None:
    buffer = JitterBuffer(target_frames=300, push_frames=200)
    buffer.push(_frames(200, 1.0), stamp=1.0)
    buffer.push(_frames(200, 2.0), stamp=2.0)
    first, first_stamp = _read(buffer, 150)
    second, second_stamp = _read(buffer, 150)
    assert (first_stamp, second_stamp) == (1.0, 1.0)
    np.testing.assert_array_equal(first, 1.0)
    np.testing.assert_array_equal(second[:50], 1.0)
    np.testing.assert_array_equal(second[50:], 2.0)
    assert buffer.queued == 100


def test_jitter_buffer_underflow_plays_the_tail_then_reprimes() -> None:
    buffer = JitterBuffer(target_frames=BLOCK, push_frames=BLOCK)
    buffer.push(_frames(BLOCK + 100), stamp=0.0)
    _read(buffer)
    out, stamp = _read(buffer)
    assert stamp == 0.0
    np.testing.assert_array_equal(out[:100], 1.0)
    assert not out[100:].any()
    assert buffer.underflows == 1
    assert buffer.priming
    assert buffer.queued == 0

    buffer.push(_frames(BLOCK - 1), stamp=1.0)
    out, stamp = _read(buffer)
    assert (stamp, buffer.underflows, buffer.queued) == (None, 1, BLOCK - 1)
    assert not out.any()
    buffer.push(_frames(1, 0.5), stamp=2.0)
    out, stamp = _read(buffer)
    assert stamp == 1.0
    np.testing.assert_array_equal(out[:-1], 1.0)
    np.testing.assert_array_equal(out[-1], 0.5)


def test_jitter_buffer_overflow_drops_the_oldest_frames_down_to_target() -> None:
    buffer = JitterBuffer(target_frames=TARGET, push_frames=BLOCK)
    pushes = buffer.limit_frames // BLOCK
    for index in range(pushes):
        buffer.push(_frames(BLOCK, float(index)), stamp=float(index))
    assert buffer.overflows == 0
    assert buffer.queued == pushes * BLOCK
    buffer.push(_frames(BLOCK, float(pushes)), stamp=float(pushes))
    assert buffer.overflows == 1
    assert buffer.queued == TARGET
    dropped = (pushes + 1) * BLOCK - TARGET
    out, stamp = _read(buffer)
    assert stamp == float(dropped // BLOCK)
    np.testing.assert_array_equal(out[: BLOCK - dropped % BLOCK], float(dropped // BLOCK))


def _drive(buffer: JitterBuffer, bursts: list[tuple[float, int]], duration_s: float) -> tuple[list[float], int]:
    """Steady BLOCK-frame reads against timed bursts of BLOCK-frame pushes: latency of every played read and the reads before the first one."""
    events = [(at, 0, blocks) for at, blocks in bursts if at < duration_s]
    events += [(index * BLOCK / RATE, 1, 0) for index in range(int(duration_s * RATE / BLOCK))]
    latencies: list[float] = []
    silent = 0
    for at, kind, blocks in sorted(events):
        if kind == 0:
            for _ in range(blocks):
                buffer.push(_frames(BLOCK), stamp=at)
            continue
        _, stamp = _read(buffer)
        if stamp is not None:
            latencies.append(at - stamp)
        elif not latencies:
            silent += 1
    return latencies, silent


def test_jitter_buffer_bounds_latency_under_three_block_bursts_every_33_ms() -> None:
    buffer = JitterBuffer(target_frames=TARGET, push_frames=BLOCK)
    latencies, silent = _drive(buffer, [(tick * 0.033, 3) for tick in range(300)], 300 * 0.033)
    assert silent <= 4
    assert buffer.underflows == 0
    assert max(latencies) <= (buffer.limit_frames + BLOCK) / RATE
    assert min(latencies) >= 0.0
    assert buffer.overflows > 0
    assert buffer.queued <= buffer.limit_frames


def test_jitter_buffer_holds_a_30_hz_clock_render_without_underflow_or_overflow() -> None:
    cursor = RenderCursor(block_ns=round(BLOCK * 1e9 / RATE), max_catchup=40)
    bursts = [(tick / 30, cursor.owed(tick * 1_000_000_000 // 30)[0]) for tick in range(600)]
    buffer = JitterBuffer(target_frames=TARGET, push_frames=BLOCK)
    latencies, _ = _drive(buffer, bursts, 600 / 30 - 0.1)
    assert (buffer.underflows, buffer.overflows) == (0, 0)
    assert min(latencies) > 0.035
    assert max(latencies) < 0.075
