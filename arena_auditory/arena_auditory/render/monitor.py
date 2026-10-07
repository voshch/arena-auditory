"""Workstation monitoring of rendered array audio: controls, stereo fold-down, hearing mono and gain."""

from __future__ import annotations

from collections.abc import Sequence

import attrs
import numpy as np
from arena_robots.audio import ArraySpec, rms
from numpy.typing import NDArray


@attrs.frozen(kw_only=True)
class MonitorConfig:
    enabled: bool
    hearing: bool
    solo: str
    front_gain: float
    rear_gain: float
    output_gain: float
    gain_db: float
    limit: float


def _channels_first(channels: NDArray[np.floating], count: int | None = None) -> NDArray[np.float32]:
    audio = np.asarray(channels, dtype=np.float32)
    if audio.ndim != 2 or (count is not None and audio.shape[0] != count):
        raise ValueError(f"channels must have shape ({count if count is not None else 'channels'}, frames)")
    return audio


def apply_controls(
    channels: NDArray[np.floating],
    *,
    enabled: bool = True,
    muted: bool = False,
    solo: str = "",
    names: Sequence[str] = (),
) -> NDArray[np.float32]:
    """Reversible gating that keeps channel alignment. A solo name keeps only that channel."""
    audio = _channels_first(channels)
    output = np.ascontiguousarray(audio.copy())
    if not enabled or muted:
        output.fill(0.0)
        return output
    if solo:
        if solo not in names:
            raise ValueError(f"solo must be empty or one of {tuple(names)}")
        keep = list(names).index(solo)
        output[np.arange(audio.shape[0]) != keep] = 0.0
    return output


def hearing_mono(channels: NDArray[np.floating]) -> NDArray[np.float32]:
    """The highest-energy channel."""
    audio = _channels_first(channels)
    energies = np.asarray(rms(audio, axis=1))
    return np.ascontiguousarray(audio[int(np.argmax(energies))])


def _weight(group: str, config: MonitorConfig) -> float:
    return config.front_gain if group == "front" else config.rear_gain if group == "rear" else 1.0


def _fold(audio: NDArray[np.float32], weights: Sequence[float], indices: Sequence[int]) -> NDArray[np.float32]:
    if not indices:
        return np.zeros(audio.shape[1], dtype=np.float32)
    total = weights[indices[0]] * audio[indices[0]]
    for index in indices[1:]:
        total = total + weights[index] * audio[index]
    return total / max(sum(abs(weights[index]) for index in indices), 1e-12)


def monitor_stereo(raw: NDArray[np.floating], spec: ArraySpec, config: MonitorConfig) -> NDArray[np.float32]:
    """(2, frames): each ear is the group-weighted mean of its side and the center mics, times output_gain, clipped."""
    audio = _channels_first(raw, spec.channels)
    weights = [_weight(mic.group, config) for mic in spec.mics]
    left = _fold(audio, weights, [index for index, mic in enumerate(spec.mics) if mic.side in ("left", "center")])
    right = _fold(audio, weights, [index for index, mic in enumerate(spec.mics) if mic.side in ("right", "center")])
    stereo = np.stack((left, right), axis=0) * float(config.output_gain)
    return np.ascontiguousarray(np.clip(stereo, -1.0, 1.0), dtype=np.float32)


def monitor_amplify(
    samples: NDArray[np.floating],
    *,
    gain_db: float,
    limit: float,
) -> NDArray[np.float32]:
    """Listening gain on calibrated PCM with a peak limit."""
    audio = np.asarray(samples, dtype=np.float32)
    gain = 10.0 ** (gain_db / 20.0)
    return np.ascontiguousarray(
        np.clip(audio * gain, -limit, limit),
        dtype=np.float32,
    )


def monitor_mix(raw: NDArray[np.floating], spec: ArraySpec, config: MonitorConfig) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """(stereo fold (2, frames), hearing mono (frames,)) of the solo-gated raw channels, stereo silent when disabled."""
    soloed = apply_controls(raw, solo=config.solo, names=spec.channel_names)
    hearing = hearing_mono(soloed)
    stereo = monitor_amplify(monitor_stereo(soloed, spec, config), gain_db=config.gain_db, limit=config.limit)
    if not config.enabled:
        stereo.fill(0.0)
    return np.ascontiguousarray(stereo), hearing


def monitor_playback(stereo: NDArray[np.float32], hearing: NDArray[np.float32], config: MonitorConfig) -> NDArray[np.float32]:
    """(2, frames) the workstation plays: the stereo fold, or the amplified hearing mono on both ears in hearing mode."""
    if not config.hearing:
        return stereo
    playback = np.repeat(monitor_amplify(hearing * config.output_gain, gain_db=config.gain_db, limit=config.limit)[None, :], 2, axis=0)
    if not config.enabled:
        playback.fill(0.0)
    return np.ascontiguousarray(playback)


def tdoa_pairs(spec: ArraySpec) -> tuple[tuple[int, int, str], ...]:
    """Left-right pairs within a group, then front-rear pairs within a side, labeled by name initials."""

    def label(index: int) -> str:
        return "".join(part[:1] for part in spec.mics[index].name.split("_")).upper()

    mics = spec.mics
    pairs: list[tuple[int, int]] = []
    for group in dict.fromkeys(mic.group for mic in mics):
        lefts = [i for i, mic in enumerate(mics) if mic.group == group and mic.side == "left"]
        rights = [i for i, mic in enumerate(mics) if mic.group == group and mic.side == "right"]
        pairs.extend((first, second) for first in lefts for second in rights)
    for side in dict.fromkeys(mic.side for mic in mics if mic.side != "center"):
        fronts = [i for i, mic in enumerate(mics) if mic.side == side and mic.group == "front"]
        rears = [i for i, mic in enumerate(mics) if mic.side == side and mic.group == "rear"]
        pairs.extend((first, second) for first in fronts for second in rears)
    return tuple((first, second, f"{label(first)}-{label(second)}") for first, second in pairs)
