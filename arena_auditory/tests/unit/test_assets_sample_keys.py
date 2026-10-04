from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import wavfile

from arena_auditory.assets import SampleDecoder, SoundLibrary, sample_key, split_sample_key

KINDS_FILE = Path(__file__).resolve().parents[2] / "config" / "sounds.yaml"


@pytest.fixture
def library(tmp_path: Path) -> Iterator[SoundLibrary]:
    directory = tmp_path / "world" / "assets" / "Common" / "Sound" / "tmp_chirp"
    directory.mkdir(parents=True)
    wavfile.write(directory / "chirp.wav", 16_000, np.sin(np.linspace(0.0, 40.0 * np.pi, 1600)).astype(np.float32))
    manifest = {"version": 2, "kind": "speech", "level_db": 60.0, "normalize_dbfs": -20.0, "variants": [{"id": "chirp_01", "file": "chirp.wav"}]}
    (directory / "tmp_chirp.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    library = SoundLibrary([KINDS_FILE])
    library.use_world(tmp_path / "world")
    try:
        yield library
    finally:
        library.use_world(None)


@pytest.mark.parametrize(("asset_id", "variant_id"), [("footstep", "footstep_default_01"), ("lab/chime", "chime_02"), ("motor", "jackal_drivetrain")])
def test_sample_key_round_trips_asset_and_variant(asset_id: str, variant_id: str) -> None:
    key = sample_key(asset_id, variant_id)

    assert key == f"{asset_id}#{variant_id}"
    assert split_sample_key(key) == (asset_id, variant_id)


@pytest.mark.parametrize("key", ["", "footstep", "#footstep_default_01", "footstep#", "#"])
def test_split_sample_key_rejects_keys_without_asset_or_variant(key: str) -> None:
    with pytest.raises(KeyError, match="is not '<asset id>#<variant id>'"):
        split_sample_key(key)


def test_by_key_resolves_the_named_asset_variant(library: SoundLibrary) -> None:
    decoder = SampleDecoder(library, 16_000)

    by_key = decoder.by_key("tmp_chirp#chirp_01")

    assert by_key is decoder.load("tmp_chirp", "chirp_01")
    assert by_key.key == "tmp_chirp#chirp_01"
    assert by_key.samples.size == 1600


def test_by_key_does_not_search_the_library_for_a_bare_variant_id(library: SoundLibrary) -> None:
    decoder = SampleDecoder(library, 16_000)

    with pytest.raises(KeyError, match="is not '<asset id>#<variant id>'"):
        decoder.by_key("chirp_01")


def test_by_key_rejects_a_variant_the_asset_does_not_have(library: SoundLibrary) -> None:
    decoder = SampleDecoder(library, 16_000)

    with pytest.raises(KeyError, match="has no variant 'chirp_02'"):
        decoder.by_key("tmp_chirp#chirp_02")
