from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import yaml
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind, SoundLibrary, kinds_file
from scipy.io import wavfile

from arena_auditory.assets import SampleDecoder
from arena_auditory.render.core import ContinuousInput, ImpulseShape, RenderInputs, RenderParams, new_state, render_block
from arena_auditory.render.dsp import PartitionedConvolver
from arena_auditory.shared import SourceSpec
from arena_auditory.sources import SOURCE_MODELS, BufferProgram, ProgramContext, streamed

@pytest.fixture
def library(tmp_path: Path) -> Iterator[SoundLibrary]:
    directory = tmp_path / "world" / "assets" / "Common" / "Sound" / "tmp_steps"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(3)
    wavfile.write(directory / "step.wav", 22_050, (rng.standard_normal(2205) * 0.2).astype(np.float32))
    wavfile.write(directory / "hum.wav", 22_050, np.sin(np.linspace(0.0, 200.0 * np.pi, 22_050)).astype(np.float32))
    manifest = {
        "version": 2,
        "kind": "footstep",
        "level_db": 45.0,
        "normalize_dbfs": -26.5,
        "variants": [{"id": "step_01", "file": "step.wav"}, {"id": "hum_01", "file": "hum.wav", "model": "wav_loop"}],
    }
    (directory / "tmp_steps.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    library = SoundLibrary([kinds_file()])
    library.use_world(tmp_path / "world")
    try:
        yield library
    finally:
        library.use_world(None)


def _source(model: str, *, loop: bool) -> SourceSpec:
    return SourceSpec(id="human:7:step", kind="footstep", asset_id="tmp_steps", model=model, agent_kind=AgentKind.PEDESTRIAN, position=(1.0, 0.0, 0.0), level_db=50.0, agent_id=7, loop=loop)


def test_wav_event_program_is_the_decoded_one_shot_sample(library: SoundLibrary) -> None:
    decoder = SampleDecoder(library, 48_000)
    context = ProgramContext(decoder=decoder, sample_rate_hz=48_000, block_size=1024)
    asset = library.asset("tmp_steps")
    variant = asset.variant("step_01")

    model = SOURCE_MODELS.get(variant.model)
    program = model.program(_source(variant.model, loop=False), asset, variant, context)

    assert not model.continuous
    assert not streamed(variant.model)
    assert isinstance(program, BufferProgram)
    assert not program.loop
    assert program.sample is decoder.load("tmp_steps", "step_01")
    assert program.sample.key == "tmp_steps#step_01"
    assert program.sample.sample_rate_hz == 48_000
    assert program.sample.samples.size == 4800


@pytest.mark.parametrize("loop", [True, False])
def test_wav_loop_program_loops_as_the_source_says(library: SoundLibrary, loop: bool) -> None:
    decoder = SampleDecoder(library, 48_000)
    context = ProgramContext(decoder=decoder, sample_rate_hz=48_000, block_size=1024)
    asset = library.asset("tmp_steps")
    variant = asset.variant("hum_01")

    model = SOURCE_MODELS.get(variant.model)
    program = model.program(_source(variant.model, loop=loop), asset, variant, context)

    assert model.continuous
    assert isinstance(program, BufferProgram)
    assert program.loop is loop
    assert program.sample is decoder.by_key("tmp_steps#hum_01")


def test_partitioned_convolver_matches_linear_convolution() -> None:
    rng = np.random.default_rng(7)
    block_size = 32
    signal = rng.standard_normal(block_size * 8).astype(np.float32)
    impulse = rng.standard_normal(75).astype(np.float32)
    convolver = PartitionedConvolver(impulse, block_size)

    rendered = np.concatenate([convolver.process(signal[offset : offset + block_size]) for offset in range(0, len(signal), block_size)])
    expected = np.convolve(signal, impulse)[: len(signal)]

    np.testing.assert_allclose(rendered, expected, rtol=2e-5, atol=2e-5)


def _loop_params(samples: np.ndarray, block_size: int, impulses: dict[str, np.ndarray] | None = None) -> RenderParams:
    shapes = {key: ImpulseShape(samples=taps, lead_samples=0) for key, taps in (impulses or {}).items()}
    return RenderParams(channels=2, block_size=block_size, sample_rate=8, resolve={"loop#loop_01": samples}.__getitem__, impulse=shapes.get)


def _loop_input(source_id: str, channel: int, *, program_start: int, rir_key: str = "") -> ContinuousInput:
    return ContinuousInput(source_id=source_id, channel=channel, asset_key="loop#loop_01", gain=1.0, delay_target=0.0, loop=True, program_start=program_start, rir_key=rir_key)


def test_looping_sources_share_phase_but_keep_distinct_rir_delays() -> None:
    samples = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    params = _loop_params(samples, 4, {"direct": np.asarray([1.0], dtype=np.float32), "delayed": np.asarray([0.0, 1.0], dtype=np.float32)})
    inputs = RenderInputs(
        block_index=0,
        clips=(),
        continuous=(_loop_input("direct", 0, program_start=-2, rir_key="direct"), _loop_input("delayed", 1, program_start=-2, rir_key="delayed")),
        streams=(),
        channels=2,
    )

    result = render_block(new_state(2), inputs, params)
    direct_block = result.ambient[0]
    delayed_block = result.ambient[1]

    np.testing.assert_allclose(direct_block, [0.3, 0.4, 0.5, 0.6], rtol=1e-6)
    assert delayed_block[0] == 0.0
    np.testing.assert_allclose(delayed_block[1:], direct_block[:-1], rtol=2e-5, atol=2e-5)


def test_looping_source_keeps_program_phase_while_inaudible() -> None:
    samples = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    params = _loop_params(samples, 2)
    state = new_state(2)
    audible = (_loop_input("hum", 0, program_start=0),)

    first = render_block(state, RenderInputs(block_index=0, clips=(), continuous=audible, streams=(), channels=2), params).ambient[0].copy()
    silent = render_block(state, RenderInputs(block_index=1, clips=(), continuous=(), streams=(), channels=2), params).ambient[0].copy()
    resumed = render_block(state, RenderInputs(block_index=2, clips=(), continuous=audible, streams=(), channels=2), params).ambient[0].copy()

    np.testing.assert_array_equal(first, [1.0, 2.0])
    np.testing.assert_array_equal(silent, [0.0, 0.0])
    np.testing.assert_array_equal(resumed, [5.0, 6.0])
