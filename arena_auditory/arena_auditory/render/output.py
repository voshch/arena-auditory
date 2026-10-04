"""Workstation audio output: a PortAudio stream fed from a jitter buffer."""

from __future__ import annotations

import math
import os
import statistics
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import attrs
import numpy as np

if TYPE_CHECKING:
    import sounddevice as sd

LATENCY_WINDOW = 64
AUTO_OUTPUT_CHAIN = ("pulse", "pipewire", "default")
DISABLED_DEVICES = ("", "none")


class StreamTime(Protocol):
    """PortAudio callback timestamps in stream seconds."""

    outputBufferDacTime: float
    currentTime: float


@attrs.frozen(kw_only=True)
class OutputStats:
    active: bool = False
    device: str = ""
    queued: int = 0
    latency_s: float = math.nan
    callbacks: int = 0
    underflows: int = 0
    overflows: int = 0
    peak: float = 0.0
    status: str = ""
    error: str = ""


def _outputs(description: Mapping[str, Any]) -> int:
    return int(description["max_output_channels"])


def output_candidates(requested: str, devices: Sequence[Mapping[str, Any]], default_index: int, *, pulse_server: bool) -> list[int | str]:
    """Devices to try in order. auto: pulse-named stereo outputs under PULSE_SERVER, the named chain, then PortAudio's default."""
    requested = requested.strip()
    if requested in DISABLED_DEVICES:
        return []
    if requested != "auto":
        return [int(requested) if requested.isdigit() else requested]
    candidates: list[int | str] = []
    if pulse_server:
        candidates.extend(index for index, description in enumerate(devices) if "pulse" in str(description["name"]).lower() and _outputs(description) >= 2)
    candidates.extend(index for name in AUTO_OUTPUT_CHAIN for index, description in enumerate(devices) if description["name"] == name and _outputs(description) > 0)
    if 0 <= default_index < len(devices) and _outputs(devices[default_index]) > 0:
        candidates.append(default_index)
    return list(dict.fromkeys(candidates))


class JitterBuffer:
    """Frame FIFO from bursty pushes to a steady reader, silent until target_frames are queued, trimmed to target_frames past limit_frames."""

    def __init__(self, *, target_frames: int, push_frames: int) -> None:
        self.target_frames = max(int(target_frames), 1)
        push_frames = max(int(push_frames), 1)
        self.limit_frames = self.target_frames + -(-self.target_frames // push_frames) * push_frames
        self.priming = True
        self.underflows = 0
        self.overflows = 0
        self._chunks: deque[tuple[np.ndarray, float]] = deque()
        self._offset = 0
        self._queued = 0

    @property
    def queued(self) -> int:
        return self._queued

    def clear(self) -> None:
        self._chunks.clear()
        self._offset = 0
        self._queued = 0
        self.priming = True

    def push(self, frames: np.ndarray, *, stamp: float) -> None:
        """Queue a copy of (frames, channels) audio pushed at monotonic time stamp."""
        chunk = np.array(frames, dtype=np.float32)
        if not len(chunk):
            return
        self._chunks.append((chunk, stamp))
        self._queued += len(chunk)
        if self._queued > self.limit_frames:
            self._consume(self._queued - self.target_frames)
            self.overflows += 1
        if self.priming and self._queued >= self.target_frames:
            self.priming = False

    def read(self, out: np.ndarray) -> float | None:
        """Fill out with the oldest frames and return the push stamp of its first frame, None when it stays silent."""
        out.fill(0.0)
        if self.priming:
            return None
        stamp = self._chunks[0][1] if self._chunks else None
        written = 0
        while written < len(out) and self._chunks:
            chunk, _ = self._chunks[0]
            count = min(len(out) - written, len(chunk) - self._offset)
            out[written : written + count] = chunk[self._offset : self._offset + count]
            written += count
            self._consume(count)
        if written < len(out):
            self.underflows += 1
            self.priming = True
        return stamp

    def _consume(self, frames: int) -> None:
        self._queued -= frames
        while frames > 0:
            chunk, _ = self._chunks[0]
            count = min(frames, len(chunk) - self._offset)
            self._offset += count
            frames -= count
            if self._offset == len(chunk):
                self._chunks.popleft()
                self._offset = 0


class AudioOutput:
    """PortAudio sink. push() queues (frames, channels) blocks into a JitterBuffer the stream callback drains."""

    def __init__(self, *, device: str, sample_rate_hz: int, channels: int, block_size: int, buffer_s: float, push_frames: int) -> None:
        self._device = device.strip()
        self._sample_rate_hz = int(sample_rate_hz)
        self._channels = int(channels)
        self._block_size = int(block_size)
        self._lock = threading.Lock()
        self._buffer = JitterBuffer(target_frames=round(buffer_s * self._sample_rate_hz), push_frames=push_frames)
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._stream: sd.OutputStream | None = None
        self._callbacks = 0
        self._peak = 0.0
        self._status = ""
        self._error = ""

    def open(self) -> str:
        """Start the stream on the first candidate that opens. Returns "" on success, else the error text."""
        if self._stream is not None and self._active(self._stream):
            return ""
        self.close()
        if self._device in DISABLED_DEVICES:
            self._error = "output disabled"
            return self._error
        try:
            import sounddevice as sd
        except OSError as exc:
            self._error = f"PortAudio unavailable: {exc}"
            return self._error
        try:
            devices = sd.query_devices()
            default_index = int(sd.default.device[1])
        except sd.PortAudioError as exc:
            self._error = f"cannot query audio devices: {exc}"
            return self._error
        candidates = output_candidates(self._device, devices, default_index, pulse_server=bool(os.environ.get("PULSE_SERVER")))
        tried: list[str] = []
        for candidate in candidates:
            try:
                sd.query_devices(candidate, "output")
                stream = sd.OutputStream(
                    samplerate=self._sample_rate_hz,
                    channels=self._channels,
                    dtype="float32",
                    blocksize=self._block_size,
                    latency="low",
                    device=candidate,
                    callback=self._callback,
                )
            except (ValueError, sd.PortAudioError) as exc:
                tried.append(f"{candidate}: {exc}")
                continue
            with self._lock:
                self._buffer.clear()
                self._latencies.clear()
            try:
                stream.start()
            except sd.PortAudioError as exc:
                stream.close()
                tried.append(f"{candidate}: {exc}")
                continue
            self._stream = stream
            self._error = ""
            return ""
        available = [f"{index}: {description['name']}" for index, description in enumerate(devices) if _outputs(description) > 0] or ["none"]
        self._error = f"cannot open audio output {self._device!r}: tried={tried or ['none']}, available={available}"
        return self._error

    def push(self, frames: np.ndarray) -> None:
        block = np.asarray(frames, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != self._channels:
            raise ValueError(f"expected (frames, {self._channels}) audio, got {block.shape}")
        if self._stream is None:
            return
        stamp = time.monotonic()
        with self._lock:
            self._buffer.push(block, stamp=stamp)

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        import sounddevice as sd

        try:
            stream.stop()
            stream.close()
        except (OSError, sd.PortAudioError) as exc:
            self._error = str(exc)

    def _active(self, stream: sd.OutputStream) -> bool:
        import sounddevice as sd

        try:
            return bool(stream.active)
        except sd.PortAudioError as exc:
            self._error = str(exc)
            return False

    @property
    def stats(self) -> OutputStats:
        """Counters of the stream, latency_s is the running median push-to-DAC latency, NaN before the first played block."""
        stream = self._stream
        with self._lock:
            queued = self._buffer.queued
            underflows = self._buffer.underflows
            overflows = self._buffer.overflows
            latencies = list(self._latencies)
        return OutputStats(
            active=stream is not None and self._active(stream),
            device=str(stream.device) if stream is not None else "",
            queued=queued,
            latency_s=statistics.median(latencies) if latencies else math.nan,
            callbacks=self._callbacks,
            underflows=underflows,
            overflows=overflows,
            peak=self._peak,
            status=self._status,
            error=self._error,
        )

    def _callback(self, outdata: np.ndarray, _frames: int, timing: StreamTime, status: object) -> None:
        now = time.monotonic()
        self._callbacks += 1
        status_text = str(status).strip()
        if status_text:
            self._status = status_text
        with self._lock:
            stamp = self._buffer.read(outdata)
            if stamp is not None:
                self._latencies.append(now - stamp + max(timing.outputBufferDacTime - timing.currentTime, 0.0))
        self._peak = float(np.max(np.abs(outdata))) if outdata.size else 0.0
