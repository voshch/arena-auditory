"""Signal primitives of the renderer, free of ROS."""

from __future__ import annotations

import array
import math

import numpy as np
from arena_robots.audio import rms_from_dbfs, spl_to_dbfs
from numpy.typing import NDArray
from scipy.signal import resample_poly

J0_FIRST_ZERO = 2.404825557695773
DRIFT_COMPONENTS = 3
DRIFT_RATE_HZ = (0.25, 1.0)
HILBERT_S = 0.064


def mems_gain(received_spl_db: float, level_rms: float, *, sensitivity_dbfs_at_94_dbspl: float = -26.0) -> float:
    """Linear gain that brings a signal of active level level_rms to received_spl_db through a fixed MEMS sensitivity, 0 for a silent signal."""
    if level_rms <= 1e-12:
        return 0.0
    return rms_from_dbfs(spl_to_dbfs(received_spl_db, sensitivity_dbfs_at_94_dbspl)) / level_rms


def calibrate_mems(
    samples: NDArray[np.floating],
    received_spl_db: float,
    level_rms: float,
    *,
    sensitivity_dbfs_at_94_dbspl: float = -26.0,
) -> NDArray[np.float32]:
    """Fixed MEMS sensitivity applied to the clip's active level."""
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if level_rms <= 1e-12:
        return np.zeros_like(mono)
    return np.ascontiguousarray(mono * mems_gain(received_spl_db, level_rms, sensitivity_dbfs_at_94_dbspl=sensitivity_dbfs_at_94_dbspl), dtype=np.float32)


def fractional_delay(
    samples: NDArray[np.floating],
    delay_samples: float,
) -> tuple[int, NDArray[np.float32]]:
    """Split a non-negative delay into integer scheduling and a linear-interpolation fractional FIR."""
    if delay_samples < 0.0 or not math.isfinite(delay_samples):
        raise ValueError("delay_samples must be finite and non-negative")
    integer = int(math.floor(delay_samples))
    fraction = delay_samples - integer
    source = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source.size == 0:
        return integer, source
    if fraction <= 1e-9:
        return integer, np.ascontiguousarray(source)
    shifted = np.empty(source.size + 1, dtype=np.float32)
    shifted[0] = (1.0 - fraction) * source[0]
    shifted[1:-1] = (1.0 - fraction) * source[1:] + fraction * source[:-1]
    shifted[-1] = fraction * source[-1]
    return integer, np.ascontiguousarray(shifted)


def streaming_fractional_delays(
    samples: NDArray[np.floating],
    delay_samples: NDArray[np.floating],
    history: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Independent causal delays of one streaming mono block, history carried across calls."""
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    delays = np.asarray(delay_samples, dtype=np.float64).reshape(-1)
    if delays.size == 0:
        raise ValueError("at least one channel delay is required")
    if not np.all(np.isfinite(delays)) or np.any(delays < 0.0):
        raise ValueError("channel delays must be finite and non-negative")
    required = max(int(math.ceil(float(np.max(delays)))) + 1, 1)
    previous = np.zeros(required, dtype=np.float32) if history is None else np.asarray(history, dtype=np.float32).reshape(-1)
    if previous.size < required:
        previous = np.pad(previous, (required - previous.size, 0))
    combined = np.concatenate((previous, mono))
    origin = previous.size
    output = np.zeros((delays.size, mono.size), dtype=np.float32)
    frames = np.arange(mono.size, dtype=np.float64)
    for channel, delay in enumerate(delays):
        positions = origin + frames - delay
        lower = np.floor(positions).astype(np.int64)
        fraction = positions - lower
        lower_valid = (lower >= 0) & (lower < combined.size)
        output[channel, lower_valid] = combined[lower[lower_valid]] * (1.0 - fraction[lower_valid])
        upper = lower + 1
        upper_valid = (fraction > 1e-12) & (upper >= 0) & (upper < combined.size)
        output[channel, upper_valid] += combined[upper[upper_valid]] * fraction[upper_valid]
    return output, np.ascontiguousarray(combined[-required:], dtype=np.float32)


def ramped_read(
    samples: NDArray[np.floating],
    start_position: float,
    delay_from: float,
    delay_to: float,
    frames: int,
    *,
    loop: bool,
) -> NDArray[np.float32]:
    """Read frames samples from start_position while the delay ramps linearly, continuous across calls."""
    source = np.asarray(samples, dtype=np.float32).reshape(-1)
    if frames <= 0 or source.size == 0:
        return np.zeros(max(frames, 0), dtype=np.float32)
    steps = np.arange(frames, dtype=np.float64)
    delay = delay_from + (delay_to - delay_from) * (steps + 1.0) / frames
    positions = start_position + steps - delay
    lower = np.floor(positions)
    fraction = (positions - lower).astype(np.float32)
    lower = lower.astype(np.int64)
    upper = lower + 1
    if loop:
        lower %= source.size
        upper %= source.size
        return (source[lower] * (1.0 - fraction) + source[upper] * fraction).astype(np.float32)
    output = np.zeros(frames, dtype=np.float32)
    valid_lower = (lower >= 0) & (lower < source.size)
    valid_upper = (upper >= 0) & (upper < source.size)
    output[valid_lower] += source[lower[valid_lower]] * (1.0 - fraction[valid_lower])
    output[valid_upper] += source[upper[valid_upper]] * fraction[valid_upper]
    return output


class PartitionedConvolver:
    """Uniform partitioned mono FIR convolution for fixed audio blocks."""

    def __init__(self, impulse: np.ndarray, block_size: int) -> None:
        self.block_size = int(block_size)
        impulse = np.asarray(impulse, dtype=np.float32).reshape(-1)
        if self.block_size <= 0 or impulse.size == 0:
            raise ValueError("block_size and impulse must be non-empty")
        count = (len(impulse) + self.block_size - 1) // self.block_size
        padded = np.pad(
            impulse,
            (0, count * self.block_size - len(impulse)),
        ).reshape(count, self.block_size)
        self._filters = np.fft.rfft(
            np.pad(padded, ((0, 0), (0, self.block_size))),
            axis=1,
        )
        self._history = np.zeros_like(self._filters)
        self._position = 0
        self._overlap = np.zeros(self.block_size, dtype=np.float64)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if len(block) != self.block_size:
            raise ValueError(f"convolver requires {self.block_size} frames, got {len(block)}")
        spectrum = np.fft.rfft(np.pad(block, (0, self.block_size)))
        self._history[self._position] = spectrum
        first_count = self._position + 1
        output_spectrum = np.sum(
            self._filters[:first_count] * self._history[self._position :: -1],
            axis=0,
        )
        if first_count < len(self._filters):
            output_spectrum += np.sum(
                self._filters[first_count:] * self._history[: self._position : -1],
                axis=0,
            )
        rendered = np.fft.irfft(output_spectrum)
        output = rendered[: self.block_size] + self._overlap
        self._overlap = rendered[self.block_size :].copy()
        self._position = (self._position + 1) % len(self._history)
        return output.astype(np.float32)


def hilbert_taps(sample_rate_hz: int) -> NDArray[np.float32]:
    """Blackman-windowed FIR Hilbert transformer spanning HILBERT_S."""
    half = max(round(HILBERT_S * sample_rate_hz / 2.0), 1)
    offsets = np.arange(-half, half + 1)
    taps = np.zeros(offsets.size)
    odd = offsets % 2 != 0
    taps[odd] = 2.0 / (np.pi * offsets[odd])
    return (taps * np.blackman(offsets.size)).astype(np.float32)


class PhaseDrift:
    """Rotates each channel's analytic signal by its own slow seeded phase drift, a mean-zero phasor over time."""

    def __init__(self, channels: int, block_size: int, sample_rate_hz: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        taps = hilbert_taps(sample_rate_hz)
        self._lag = (taps.size - 1) // 2
        self._quadrature = [PartitionedConvolver(taps, block_size) for _ in range(channels)]
        self._history = np.zeros((channels, self._lag), dtype=np.float32)
        self._rates = rng.uniform(*DRIFT_RATE_HZ, (channels, DRIFT_COMPONENTS, 1)) * (2.0 * math.pi / sample_rate_hz)
        self._phases = rng.uniform(0.0, 2.0 * math.pi, (channels, DRIFT_COMPONENTS, 1))
        self._frame = 0

    def process(self, block: NDArray[np.float32]) -> NDArray[np.float32]:
        frames = block.shape[1]
        theta = J0_FIRST_ZERO * np.sin(self._rates * (self._frame + np.arange(frames)) + self._phases).sum(axis=1)
        self._frame += frames
        lagged = np.concatenate((self._history, block), axis=1)
        self._history = lagged[:, frames:]
        quadrature = np.stack([convolver.process(channel) for convolver, channel in zip(self._quadrature, block, strict=True)])
        return (lagged[:, :frames] * np.cos(theta) - quadrature * np.sin(theta)).astype(np.float32)


def resample_impulse(samples: NDArray[np.floating], from_rate_hz: int, to_rate_hz: int) -> NDArray[np.float32]:
    """Polyphase resample of an impulse response that keeps its filter gain, unchanged when the rates match."""
    impulse = np.asarray(samples, dtype=np.float32).reshape(-1)
    if from_rate_hz <= 0 or to_rate_hz <= 0:
        raise ValueError("sample rates must be positive")
    if from_rate_hz == to_rate_hz or impulse.size == 0:
        return np.ascontiguousarray(impulse)
    divisor = math.gcd(from_rate_hz, to_rate_hz)
    resampled = resample_poly(impulse, to_rate_hz // divisor, from_rate_hz // divisor) * (from_rate_hz / to_rate_hz)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def interleave(channels: NDArray[np.floating]) -> array.array:
    audio = np.asarray(channels, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError("audio must be channels-first")
    data = array.array("f")
    data.frombytes(np.ascontiguousarray(audio.T, dtype=np.float32).tobytes())
    return data
