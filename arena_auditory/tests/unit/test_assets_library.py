from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pytest
import yaml
from arena_robots.audio import active_rms, dbfs_from_rms
from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary, kinds_file
from scipy.io import wavfile

from arena_auditory.assets import SampleDecoder, decode_wav


def _write_asset(world: Path, name: str, manifest: Mapping[str, object], wavs: Mapping[str, tuple[int, np.ndarray]]) -> Path:
    directory = world / "assets" / "Common" / "Sound" / name
    directory.mkdir(parents=True)
    for file, (rate, samples) in wavs.items():
        wavfile.write(directory / file, rate, samples)
    (directory / f"{name}.yaml").write_text(yaml.safe_dump(dict(manifest)), encoding="utf-8")
    return directory


def _step_manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "version": 2,
        "kind": "footstep",
        "level_db": 45.0,
        "normalize_dbfs": -6.0,
        "variants": [
            {"id": "step_walnut", "file": "walnut.wav", "match": {"floor": ["walnut"]}, "tags": ["walnut_planks"]},
            {"id": "step_default", "file": "step.wav", "default": True, "tags": ["default"]},
        ],
    }
    manifest.update(overrides)
    return manifest


def _step_wavs() -> dict[str, tuple[int, np.ndarray]]:
    return {
        "step.wav": (22_050, np.asarray([0, 1000, -1000, 0], dtype=np.int16)),
        "walnut.wav": (22_050, np.asarray([0, 500, -500, 0], dtype=np.int16)),
    }


@pytest.fixture
def library() -> Iterator[SoundLibrary]:
    library = SoundLibrary([kinds_file()])
    try:
        yield library
    finally:
        library.use_world(None)


def test_decoder_caches_each_variant(tmp_path: Path, library: SoundLibrary) -> None:
    world = tmp_path / "world"
    _write_asset(world, "tmp_step", _step_manifest(), _step_wavs())
    library.use_world(world)
    asset = library.asset("tmp_step")

    decoder = SampleDecoder(library, 44_100)
    first = decoder.load(asset.id, "step_walnut")
    second = decoder.load(asset.id, "step_walnut")

    assert first is second
    assert first.key == "tmp_step#step_walnut"
    assert first.sample_rate_hz == 44_100
    assert first.samples.ndim == 1
    assert first.samples.dtype == np.float32
    assert first.samples.size == 8
    assert first.duration_s == pytest.approx(8 / 44_100)


def test_decoding_normalizes_the_active_level_so_silence_padding_does_not_change_it(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    burst = (rng.standard_normal(1600) * np.hanning(1600) * 0.5).astype(np.float32)
    padded = np.concatenate((np.zeros(4000, dtype=np.float32), burst, np.zeros(20000, dtype=np.float32)))
    wavfile.write(tmp_path / "short.wav", 16_000, burst)
    wavfile.write(tmp_path / "padded.wav", 16_000, padded)

    short = decode_wav(tmp_path / "short.wav", key="thud#short", sample_rate_hz=16_000, normalize_dbfs=-30.0)
    long = decode_wav(tmp_path / "padded.wav", key="thud#padded", sample_rate_hz=16_000, normalize_dbfs=-30.0)

    assert dbfs_from_rms(short.active_rms) == pytest.approx(-30.0, abs=1e-4)
    assert long.active_rms == short.active_rms
    assert short.active_rms == active_rms(short.samples, 16_000)
    np.testing.assert_array_equal(long.samples[4000 : 4000 + burst.size], short.samples)


def test_decoding_keeps_the_first_channel_and_resamples_to_the_decoder_rate(tmp_path: Path) -> None:
    left = np.sin(2.0 * np.pi * 440.0 * np.arange(8_000) / 8_000).astype(np.float32)
    stereo = np.stack((left, np.zeros_like(left)), axis=1)
    wavfile.write(tmp_path / "stereo.wav", 8_000, stereo)

    decoded = decode_wav(tmp_path / "stereo.wav", key="tone#stereo", sample_rate_hz=16_000, normalize_dbfs=-12.0)

    assert decoded.samples.shape == (16_000,)
    assert decoded.duration_s == pytest.approx(1.0)
    assert dbfs_from_rms(decoded.active_rms) == pytest.approx(-12.0, abs=0.05)
