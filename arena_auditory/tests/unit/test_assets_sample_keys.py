from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import yaml
from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary, kinds_file
from scipy.io import wavfile

from arena_auditory.assets import SampleDecoder

@pytest.fixture
def library(tmp_path: Path) -> Iterator[SoundLibrary]:
    directory = tmp_path / "world" / "assets" / "Common" / "Sound" / "tmp_chirp"
    directory.mkdir(parents=True)
    wavfile.write(directory / "chirp.wav", 16_000, np.sin(np.linspace(0.0, 40.0 * np.pi, 1600)).astype(np.float32))
    manifest = {"version": 2, "kind": "speech", "level_db": 60.0, "normalize_dbfs": -20.0, "variants": [{"id": "chirp_01", "file": "chirp.wav"}]}
    (directory / "tmp_chirp.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    library = SoundLibrary([kinds_file()])
    library.use_world(tmp_path / "world")
    try:
        yield library
    finally:
        library.use_world(None)


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
