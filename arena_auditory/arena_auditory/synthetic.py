"""Synthesized programs named by their sample key: the diffuse ambient bed."""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

BED_TAG = "bed@pink="
BED_LOOP_S = 60.0
BED_LOW_HZ = 20.0
BED_RMS = 0.1

_BED_KEY = re.compile(r"^bed@pink=(?P<seed>\d+),(?P<channel>\d+)$")


def bed_seed(robot: str) -> int:
    """Seed of a robot's ambient bed, fixed by its name."""
    return int.from_bytes(hashlib.blake2b(f"bed/{robot}".encode(), digest_size=8).digest(), "big")


def bed_key(seed: int, channel: int) -> str:
    return f"{BED_TAG}{int(seed)},{int(channel)}"


def parse_bed_key(key: str) -> tuple[int, int] | None:
    """Seed and channel of a bed key, None for any other key."""
    match = _BED_KEY.match(key)
    return None if match is None else (int(match["seed"]), int(match["channel"]))


def synthetic(key: str) -> bool:
    return parse_bed_key(key) is not None


def a_weighting(frequencies_hz: np.ndarray) -> np.ndarray:
    """IEC 61672 A-weighting as a linear amplitude gain, 1 at 1 kHz."""
    f2 = np.square(np.asarray(frequencies_hz, dtype=np.float64))
    ra = 12194.0**2 * f2**2 / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2))
    return ra * 10.0 ** (2.0 / 20.0)


def a_weighted_rms(samples: np.ndarray, sample_rate_hz: int) -> float:
    """RMS of the circularly A-weighted signal, from its spectrum."""
    samples = np.asarray(samples, dtype=np.float64)
    spectrum = np.fft.rfft(samples) * a_weighting(np.fft.rfftfreq(samples.size, 1.0 / sample_rate_hz))
    weights = np.full(spectrum.size, 2.0)
    weights[0] = 1.0
    if samples.size % 2 == 0:
        weights[-1] = 1.0
    return float(math.sqrt(np.sum(weights * np.abs(spectrum) ** 2)) / samples.size)


def pink_bed(sample_rate_hz: int, seed: int, channel: int, seconds: float = BED_LOOP_S) -> np.ndarray:
    """Seamless loop of pink noise from BED_LOW_HZ up, random phase seeded by seed and channel, BED_RMS unweighted."""
    frames = 2 * round(seconds * sample_rate_hz / 2.0)
    frequencies = np.fft.rfftfreq(frames, 1.0 / sample_rate_hz)
    amplitude = np.where(frequencies >= BED_LOW_HZ, 1.0 / np.sqrt(np.maximum(frequencies, BED_LOW_HZ)), 0.0)
    phase = np.random.default_rng([int(seed), int(channel)]).uniform(0.0, 2.0 * math.pi, frequencies.size)
    samples = np.fft.irfft(amplitude * np.exp(1j * phase), frames)
    return np.ascontiguousarray(samples * (BED_RMS / np.sqrt(np.mean(samples**2))), dtype=np.float32)
