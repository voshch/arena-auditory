"""Sample decoding of sound assets: wav lengths, decoded variants and the per-rate decoder."""

from __future__ import annotations

import math
import threading
from pathlib import Path

import attrs
import numpy as np
from arena_robots.audio import active_rms, rms_from_dbfs
from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary, sample_key, split_sample_key
from scipy.io import wavfile
from scipy.signal import resample_poly

_durations: dict[Path, float] = {}
_durations_lock = threading.Lock()


def wav_duration_s(path: Path) -> float:
    """Length of a wav file from its header and a memory map, without decoding."""
    rate, data = wavfile.read(path, mmap=True)
    return len(data) / rate


@attrs.frozen(kw_only=True, eq=False)
class DecodedSample:
    """Mono float32 samples of one variant, first channel, resampled and normalized to the asset's normalize_dbfs."""

    key: str
    samples: np.ndarray
    sample_rate_hz: int
    duration_s: float
    active_rms: float


def _to_float32(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    info = np.iinfo(data.dtype)
    scale = float(max(abs(info.min), info.max))
    return data.astype(np.float32) / scale


def decode_wav(path: Path, *, key: str, sample_rate_hz: int, normalize_dbfs: float) -> DecodedSample:
    """Decode, keep the first channel, resample polyphase and scale the active RMS to normalize_dbfs. Raises ValueError on an empty file."""
    target_rate = int(sample_rate_hz)
    source_rate, data = wavfile.read(path)
    samples = _to_float32(data)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[1] > 1:
        samples = samples[:, :1]
    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        samples = resample_poly(samples, target_rate // divisor, source_rate // divisor, axis=0).astype(np.float32)
    if samples.size == 0:
        raise ValueError(f"empty WAV file: {path}")
    level = active_rms(samples, target_rate)
    if level > 0.0:
        samples *= rms_from_dbfs(normalize_dbfs) / level
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    return DecodedSample(
        key=key,
        samples=np.ascontiguousarray(samples[:, 0]),
        sample_rate_hz=target_rate,
        duration_s=len(samples) / target_rate,
        active_rms=active_rms(samples, target_rate),
    )


def duration_s(library: SoundLibrary, asset_id: str, variant_id: str) -> float:
    """Variant wav length from the header, 0 for variants without a file."""
    path = library.variant(asset_id, variant_id).path
    if path is None:
        return 0.0
    with _durations_lock:
        cached = _durations.get(path)
    if cached is None:
        cached = wav_duration_s(path)
        with _durations_lock:
            _durations[path] = cached
    return cached


class SampleDecoder:
    """Decodes variants at one sample rate, each once. Thread-safe."""

    def __init__(self, library: SoundLibrary, sample_rate_hz: int) -> None:
        self._library = library
        self._sample_rate_hz = int(sample_rate_hz)
        self._cache: dict[tuple[Path, float], DecodedSample] = {}
        self._lock = threading.Lock()

    @property
    def sample_rate_hz(self) -> int:
        return self._sample_rate_hz

    def load(self, asset_id: str, variant_id: str) -> DecodedSample:
        """Raises KeyError for an unknown asset or variant, ValueError for a variant without a wav."""
        asset = self._library.asset(asset_id)
        variant = asset.variant(variant_id)
        if variant.path is None:
            raise ValueError(f"variant {variant.id!r} of sound asset {asset.id!r} has no wav file")
        key = (variant.path, asset.normalize_dbfs)
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                cached = decode_wav(variant.path, key=sample_key(asset.id, variant.id), sample_rate_hz=self._sample_rate_hz, normalize_dbfs=asset.normalize_dbfs)
                self._cache[key] = cached
            return cached

    def by_key(self, key: str) -> DecodedSample:
        """Decode by sample key. Raises KeyError for a key without an asset id."""
        return self.load(*split_sample_key(key))
