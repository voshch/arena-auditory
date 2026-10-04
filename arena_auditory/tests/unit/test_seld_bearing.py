"""The bearing fit recovers every azimuth on the sim array; the front-end's gcc path does too, with the real checkpoint."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from arena_auditory.hearing.doa import ArrayBearing
from arena_auditory.render.dsp import fractional_delay
from arena_auditory.shared import Vec3, geometric_delays_s, load_array_spec

FS = 16000
BEARINGS_DEG = (0, 45, 90, 180, 225, 270)


def _positions() -> tuple[Vec3, ...]:
    return tuple(mic.position_m for mic in load_array_spec("four_mic").mics)


def _place(clip: np.ndarray, az_deg: float, *, repeats: int, period_s: float, lead_s: float, total_s: float) -> np.ndarray:
    """4-channel (samples, ch) rendering of ``clip`` from ``az_deg`` at 3 m, sim delays, no reverb."""
    az = math.radians(az_deg)
    delays = geometric_delays_s((3.0 * math.cos(az), 3.0 * math.sin(az), 0.0), _positions())
    delays = delays - delays.min()
    out = np.zeros((int(total_s * FS), 4), dtype=np.float32)
    for k in range(repeats):
        t0 = int((lead_s + period_s * k) * FS)
        for ch in range(4):
            i, s = fractional_delay(clip, delays[ch] * FS)
            out[t0 + i : t0 + i + len(s), ch] += s
    return out


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("az_deg", BEARINGS_DEG)
def test_array_bearing_fits_noise_burst(az_deg: int) -> None:
    rng = np.random.default_rng(az_deg)
    burst = (rng.standard_normal(int(0.05 * FS)) * 0.3).astype(np.float32)
    frame = _place(burst, az_deg, repeats=1, period_s=1.0, lead_s=0.02, total_s=0.2)
    theta, residual, valid = ArrayBearing(FS, _positions()).bearing(frame)
    assert valid
    assert abs(_wrap_deg(math.degrees(theta) - az_deg)) <= 3.0
    assert residual < 1e-4


def _ego_noise(seed: int, amp: float, *, total_s: float = 0.2) -> np.ndarray:
    """Identical 4-channel noise, standing in for drivetrain noise rendered from the array center."""
    rng = np.random.default_rng(seed)
    mono = (rng.standard_normal(int(total_s * FS)) * amp).astype(np.float32)
    return np.tile(mono[:, None], (1, 4))


def test_array_bearing_gates_ego_noise() -> None:
    _theta, _residual, valid = ArrayBearing(FS, _positions()).bearing(_ego_noise(1, 0.3))
    assert not valid


def test_array_bearing_valid_when_footstep_above_ego_noise() -> None:
    rng = np.random.default_rng(90)
    burst = (rng.standard_normal(int(0.05 * FS)) * 0.3).astype(np.float32)
    foot = _place(burst, 90, repeats=1, period_s=1.0, lead_s=0.02, total_s=0.2)
    theta, _residual, valid = ArrayBearing(FS, _positions()).bearing(_ego_noise(2, 0.05) + foot)
    assert valid
    assert abs(_wrap_deg(math.degrees(theta) - 90)) <= 5.0


def test_array_bearing_gated_when_footstep_drowned_by_ego_noise() -> None:
    rng = np.random.default_rng(91)
    burst = (rng.standard_normal(int(0.05 * FS)) * 0.02).astype(np.float32)
    foot = _place(burst, 90, repeats=1, period_s=1.0, lead_s=0.02, total_s=0.2)
    _theta, _residual, valid = ArrayBearing(FS, _positions()).bearing(_ego_noise(3, 0.3) + foot)
    assert not valid


def _weights() -> dict[str, str] | None:
    from arena_auditory.hearing import weights

    if not os.environ.get("ARENA_DATA_DIR"):
        return None
    files = {entry["role"]: Path(weights.data_dir()) / entry["dest"] for entry in weights.manifest()}
    return {role: str(path) for role, path in files.items()} if all(p.is_file() for p in files.values()) else None


def _frontend(files: dict[str, str]):
    from arena_auditory.assets import SoundLibrary
    from arena_auditory.hearing import weights
    from arena_auditory.hearing.seld import SeldFrontend
    from arena_auditory.params import SeldGroup

    seld = SeldGroup.defaults()
    return SeldFrontend(files["checkpoint"], files["scaler"], weights.classes(SoundLibrary.default()), device="cpu", det_threshold=seld["det_threshold"], torch_threads=seld["torch_threads"])


def _footstep() -> np.ndarray:
    librosa = pytest.importorskip("librosa")
    soundfile = pytest.importorskip("soundfile")
    from arena_auditory.assets import SoundLibrary

    path = next(variant.path for variant in SoundLibrary.default().default_asset("footstep").variants if variant.default)
    audio, fs = soundfile.read(str(path), dtype="float32", always_2d=True)
    return librosa.resample(audio[:, 0], orig_sr=fs, target_sr=FS).astype(np.float32) * 0.3


@pytest.mark.parametrize("az_deg", BEARINGS_DEG)
def test_frontend_gcc_bearing_with_checkpoint(az_deg: int) -> None:
    pytest.importorskip("torch")
    files = _weights()
    if files is None:
        pytest.skip("SELD weights not fetched (ros2 run arena_auditory hearing_setup)")
    from arena_auditory.hearing import weights
    from arena_auditory.hearing.seld import SeldStream

    fe = _frontend(files)
    stream = SeldStream(fe, lookahead_frames=5)
    doa = ArrayBearing(fe.fs, tuple(mic.position_m for mic in weights.array_spec().mics))
    audio = _place(_footstep(), az_deg, repeats=12, period_s=0.5, lead_s=0.4, total_s=7.0)
    fitted: list[float] = []
    hop = fe.label_hop_len
    for start in range(0, audio.shape[0] - hop + 1, hop):
        stream.push(audio[start : start + hop])
        dets, _end, seg = stream.step()
        if dets:
            fitted.append(math.degrees(doa.bearing(seg)[0]))
    assert len(fitted) >= 10
    errors = np.abs([_wrap_deg(f - az_deg) for f in fitted])
    assert float(np.median(errors)) <= 5.0


def _rolled(history: np.ndarray, block: np.ndarray) -> np.ndarray:
    n = block.shape[0]
    if n >= history.shape[0]:
        history[:] = block[-history.shape[0] :, : history.shape[1]]
    elif n > 0:
        history = np.roll(history, -n, axis=0)
        history[-n:] = block[:, : history.shape[1]]
    return history


def test_sample_ring_window_equals_the_rolled_buffer() -> None:
    from arena_auditory.hearing.seld import SampleRing

    rng = np.random.default_rng(5)
    ring = SampleRing(1000, 4)
    reference = np.zeros((1000, 4), dtype=np.float64)
    for n in (320, 0, 1, 999, 320, 1000, 1500, 7, 320, 320, 320, 640):
        block = rng.standard_normal((n, 5)).astype(np.float32)
        ring.push(block)
        reference = _rolled(reference, block.astype(np.float64))
        np.testing.assert_array_equal(ring.window(), reference)


def test_seld_stream_step_equals_the_rolled_window_path() -> None:
    pytest.importorskip("torch")
    files = _weights()
    if files is None:
        pytest.skip("SELD weights not fetched (ros2 run arena_auditory hearing_setup)")
    from arena_auditory.hearing.seld import SeldStream

    fe = _frontend(files)
    stream = SeldStream(fe, lookahead_frames=5)
    reference = np.zeros((fe.window_samples, fe.nb_raw_ch), dtype=np.float64)
    audio = _place(_footstep(), 90, repeats=6, period_s=0.5, lead_s=0.4, total_s=4.5)
    frame = fe.label_sequence_length - 1 - stream.lookahead_frames
    for start in range(0, audio.shape[0] - 320 + 1, 320):
        block = audio[start : start + 320]
        stream.push(block)
        reference = _rolled(reference, np.asarray(block, dtype=np.float64))
        if stream.ready():
            dets, _end, seg = stream.step()
            accdoa = fe.forward(fe.features(reference))
            assert dets == fe.decode(accdoa, frames=range(frame, frame + 1))
            np.testing.assert_array_equal(seg, reference[max(frame - 1, 0) * fe.label_hop_len : (frame + 1) * fe.label_hop_len])


def test_array_bearing_refuses_an_array_without_a_planar_baseline() -> None:
    with pytest.raises(ValueError, match="two apart in the plane"):
        ArrayBearing(FS, [(0.1, 0.0, 0.2)])
    with pytest.raises(ValueError, match="two apart in the plane"):
        ArrayBearing(FS, [(0.0, 0.0, 0.0), (0.0, 0.0, 0.3)])
