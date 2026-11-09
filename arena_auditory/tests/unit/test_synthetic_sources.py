from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from arena_robots.audio import rms_from_dbfs, spl_to_dbfs
from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary, kinds_file

from arena_auditory.assets import SampleDecoder
from arena_auditory.offline_render import asset_resolver
from arena_auditory.render.core import ContinuousInput, RenderInputs, RenderParams, new_state, render_block
from arena_auditory.render.dsp import mems_gain
from arena_auditory.synthetic import a_weighted_rms, a_weighting, bed_key, bed_seed, parse_bed_key, synthetic

CHANNELS = 4
SAMPLE_RATE_HZ = 16_000
BLOCK_SIZE = 320
SENSITIVITY_DBFS = -26.0


def test_bed_keys_parse_back_and_only_they_are_synthetic() -> None:
    assert parse_bed_key(bed_key(7, 3)) == (7, 3)
    assert synthetic(bed_key(7, 3))
    assert not synthetic("greeting#greeting_01")
    assert parse_bed_key("greeting#greeting_01") is None


def test_offline_resolver_rebuilds_bed_keys(tmp_path: Path) -> None:
    decoder = SampleDecoder(SoundLibrary([kinds_file()]), SAMPLE_RATE_HZ)
    try:
        resolve, level = asset_resolver(SAMPLE_RATE_HZ, tmp_path / "world")

        np.testing.assert_array_equal(resolve(bed_key(9, 2)), decoder.by_key(bed_key(9, 2)).samples)
        assert level(bed_key(9, 2)) == decoder.by_key(bed_key(9, 2)).active_rms
    finally:
        SoundLibrary.default().use_world(None)


def test_a_weighting_is_unity_at_one_kilohertz_and_cuts_the_low_end() -> None:
    gains_db = 20.0 * np.log10(a_weighting(np.asarray([100.0, 1000.0, 4000.0])))

    np.testing.assert_allclose(gains_db, [-19.1, 0.0, 1.0], atol=0.1)


def _render_bed(level_dba: float, seconds: float) -> np.ndarray:
    decoder = SampleDecoder(SoundLibrary([kinds_file()]), SAMPLE_RATE_HZ)
    seed = bed_seed("jackal")
    samples = {bed_key(seed, channel): decoder.by_key(bed_key(seed, channel)) for channel in range(CHANNELS)}
    params = RenderParams(channels=CHANNELS, block_size=BLOCK_SIZE, sample_rate=SAMPLE_RATE_HZ, resolve=lambda key: samples[key].samples)
    voices = tuple(ContinuousInput(source_id="ambient_bed", channel=channel, asset_key=bed_key(seed, channel), gain=mems_gain(level_dba, samples[bed_key(seed, channel)].active_rms), delay_target=0.0, loop=True, program_start=0, stem="ambient") for channel in range(CHANNELS))
    state = new_state(CHANNELS)
    blocks = [render_block(state, RenderInputs(block_index=index, clips=(), continuous=voices, streams=(), channels=CHANNELS), params).ambient for index in range(round(seconds * SAMPLE_RATE_HZ / BLOCK_SIZE))]
    return np.concatenate(blocks, axis=1)


def test_bed_reaches_its_a_weighted_level_at_every_microphone_and_is_incoherent_across_them() -> None:
    bed = _render_bed(40.0, 60.0)
    target = rms_from_dbfs(spl_to_dbfs(40.0, SENSITIVITY_DBFS))

    levels_db = [20.0 * math.log10(a_weighted_rms(channel, SAMPLE_RATE_HZ) / target) for channel in bed]
    correlation = np.corrcoef(bed)

    np.testing.assert_allclose(levels_db, 0.0, atol=0.05)
    assert np.max(np.abs(correlation[~np.eye(CHANNELS, dtype=bool)])) < 0.05


def test_bed_loop_has_no_seam() -> None:
    bed = SampleDecoder(SoundLibrary([kinds_file()]), SAMPLE_RATE_HZ).by_key(bed_key(3, 0)).samples
    steps = np.abs(np.diff(np.concatenate([bed, bed[:1]])))

    assert steps[-1] <= np.percentile(steps, 99.9)
