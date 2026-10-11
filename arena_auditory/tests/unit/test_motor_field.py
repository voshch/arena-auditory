from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pytest
from arena_robots.audio import SPEED_OF_SOUND_MPS, load_array_spec
from scipy.signal import coherence, csd

from arena_auditory.params import MotorGroup
from arena_auditory.propagation.self_path import drive_distances
from arena_auditory.render.core import ClipInput, ContinuousInput, RenderInputs, RenderParams, StreamInput, new_state, prepare_clip, render_block, render_inputs_from_json, render_inputs_to_json
from arena_auditory.render.dsp import streaming_fractional_delays
from arena_auditory.shared import SourceSpec
from arena_auditory.sources import stream_model, tuning_of
from arena_auditory.sources.drivetrain.program import LEFT_VELOCITY, RIGHT_VELOCITY

CHANNELS = 4
SAMPLE_RATE_HZ = 16_000
BLOCK_SIZE = 320
SENSITIVITY_DBFS = -26.0
REFERENCE_SHA256 = {
    "raw": "d73c494d1e364dc88ace223078d3ca51cd408ff73988bdf246c02e9410871b49",
    "ped": "9891178fbd2c5d4fd3c1cc01165cde9f1080a3887f1e697215edd93586cda8dc",
    "ambient": "7e77f4e9510d32f9b4d3f3a56a87d5194907e96b129a3fbba2dd24d251dd9a44",
    "motor": "8eafc7bd411c1f02b9e972a83d2b0a4164eefc5ef51e6b63ad7acc78be4ad44f",
}
FIELD_SECONDS = 16.0
SPEED_MPS = 0.4
BANDS_HZ = ((88.0, 177.0), (707.0, 1414.0))
NPERSEG = 1024
STATE = {LEFT_VELOCITY: SPEED_MPS, RIGHT_VELOCITY: SPEED_MPS * 0.98}


def _footstep() -> np.ndarray:
    frames = np.arange(1600)
    return ((frames * 37 % 61 - 30) / 64.0 * (1600 - frames) / 1600.0).astype(np.float32)


def _fan_loop() -> np.ndarray:
    frames = np.arange(4000)
    return ((frames * 53 % 97 - 48) / 128.0).astype(np.float32)


def _non_robot_scene() -> dict[str, np.ndarray]:
    """A footstep clip and a fan loop rendered on a four-microphone array."""
    samples = {"footstep#a": _footstep(), "fan#loop": _fan_loop()}
    params = RenderParams(channels=CHANNELS, block_size=BLOCK_SIZE, sample_rate=SAMPLE_RATE_HZ, resolve=samples.__getitem__)
    state = new_state(CHANNELS)
    clip_delays = (3.25, 7.5, 11.0, 14.75)
    clip_levels = (62.0, 60.5, 58.0, 57.25)
    clips = []
    for channel, (delay_samples, level) in enumerate(zip(clip_delays, clip_levels, strict=True)):
        delay, delayed = prepare_clip(samples["footstep#a"], level, SENSITIVITY_DBFS, delay_samples, level_rms=0.125)
        anchor = 2 * BLOCK_SIZE
        clips.append(ClipInput(channel=channel, start=anchor + delay, asset_key="footstep#a", samples=delayed, anchor=anchor, received_volume_db=level, sensitivity_dbfs_at_94_dbspl=SENSITIVITY_DBFS, delay_samples=delay_samples))
    stems: dict[str, list[np.ndarray]] = {"raw": [], "ped": [], "ambient": [], "motor": []}
    for block in range(40):
        delay_target = 5.5 if block < 20 else 9.25
        continuous = tuple(ContinuousInput(source_id="fan", channel=channel, asset_key="fan#loop", gain=0.3 + 0.1 * channel, delay_target=delay_target + channel, loop=True, program_start=0, stem="ambient") for channel in range(CHANNELS))
        inputs = RenderInputs(block_index=block, clips=tuple(clips) if block == 2 else (), continuous=continuous, streams=(), channels=CHANNELS)
        result = render_block(state, inputs, params)
        for name in stems:
            stems[name].append(getattr(result, name))
    return {name: np.concatenate(blocks, axis=1) for name, blocks in stems.items()}


def test_non_robot_sources_render_bit_identical_to_the_reference() -> None:
    rendered = _non_robot_scene()

    assert {name: hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest() for name, audio in rendered.items()} == REFERENCE_SHA256


def _drive_paths() -> tuple[np.ndarray, np.ndarray]:
    """Per channel and drivetrain side the inverse-distance gain and the delay in samples, from the four_mic array and the default drive points."""
    points = MotorGroup.defaults()[MotorGroup.DRIVE_POINTS_M.field]
    distances = np.asarray([drive_distances(points, mic.position_m) for mic in load_array_spec("four_mic").mics])
    return 0.05 / distances, distances / SPEED_OF_SOUND_MPS * SAMPLE_RATE_HZ


def _own_motor(share: float, robot: str = "jackal") -> StreamInput:
    gains, delays = _drive_paths()
    return StreamInput(
        source_id=f"robot:{robot}:motor",
        model="drivetrain",
        seed=3,
        params={"spec": "jackal"},
        state=STATE,
        tuning={**tuning_of(MotorGroup.defaults()), MotorGroup.STRUCTURAL_SHARE.field: share},
        gains=tuple(gains.reshape(-1)),
        delay_samples=tuple(delays.reshape(-1)),
        active_channels=(True,) * CHANNELS,
        active=True,
        source_agent_id=-1,
        source_agent_name=robot,
        kind="motor",
        asset_id="motor",
        paths=gains.shape[1],
        own_robot=True,
    )


def _render_motor(stream: StreamInput, seconds: float) -> np.ndarray:
    params = RenderParams(channels=CHANNELS, block_size=BLOCK_SIZE, sample_rate=SAMPLE_RATE_HZ, resolve=lambda key: np.zeros(1, dtype=np.float32))
    state = new_state(CHANNELS)
    blocks = [render_block(state, RenderInputs(block_index=index, clips=(), continuous=(), streams=(stream,), channels=CHANNELS), params).motor for index in range(round(seconds * SAMPLE_RATE_HZ / BLOCK_SIZE))]
    return np.concatenate(blocks, axis=1)


def _band_coherence(motor: np.ndarray, band: tuple[float, float]) -> dict[tuple[int, int], float]:
    result = {}
    for first, second in itertools.combinations(range(CHANNELS), 2):
        frequencies, msc = coherence(motor[first], motor[second], fs=SAMPLE_RATE_HZ, nperseg=NPERSEG)
        selected = (frequencies >= band[0]) & (frequencies < band[1])
        result[first, second] = float(np.mean(msc[selected]))
    return result


def _rank_ratio(motor: np.ndarray, band: tuple[float, float], order: int) -> float:
    """Largest ratio of the order-th to the first eigenvalue of the cross-spectral matrix over the band's bins."""
    frequencies, _ = csd(motor[0], motor[0], fs=SAMPLE_RATE_HZ, nperseg=NPERSEG)
    matrix = np.empty((frequencies.size, CHANNELS, CHANNELS), dtype=np.complex128)
    for first, second in itertools.product(range(CHANNELS), repeat=2):
        matrix[:, first, second] = csd(motor[first], motor[second], fs=SAMPLE_RATE_HZ, nperseg=NPERSEG)[1]
    selected = (frequencies >= band[0]) & (frequencies < band[1])
    eigenvalues = np.linalg.eigvalsh(matrix[selected])
    return float(np.max(eigenvalues[:, -order] / eigenvalues[:, -1]))


@pytest.fixture(scope="module")
def default_field() -> np.ndarray:
    return _render_motor(_own_motor(MotorGroup.defaults()[MotorGroup.STRUCTURAL_SHARE.field]), FIELD_SECONDS)


@pytest.fixture(scope="module")
def airborne_field() -> np.ndarray:
    return _render_motor(_own_motor(0.0), FIELD_SECONDS)


@pytest.mark.parametrize("band", BANDS_HZ)
def test_own_motor_band_coherence_drops_by_one_minus_the_structural_share_squared(default_field: np.ndarray, airborne_field: np.ndarray, band: tuple[float, float]) -> None:
    share = MotorGroup.defaults()[MotorGroup.STRUCTURAL_SHARE.field]
    airborne = _band_coherence(airborne_field, band)

    pairs = _band_coherence(default_field, band)

    assert all(value < 0.8 for value in pairs.values()), pairs
    assert all(abs(pairs[pair] - (1.0 - share) ** 2 * airborne[pair]) < 0.08 for pair in pairs), (pairs, airborne)


@pytest.mark.parametrize("band", BANDS_HZ)
def test_own_motor_airborne_field_alone_has_the_rank_of_its_two_drivetrain_sides(airborne_field: np.ndarray, band: tuple[float, float]) -> None:
    assert _rank_ratio(airborne_field, band, order=3) < 1e-3


def test_structural_share_raises_the_rank_of_the_own_motor_field(default_field: np.ndarray) -> None:
    assert _rank_ratio(default_field, BANDS_HZ[0], order=3) > 0.02


def test_own_motor_airborne_paths_sum_the_drivetrain_sides_at_their_geometric_delays(airborne_field: np.ndarray) -> None:
    gains, delays = _drive_paths()
    stream = _own_motor(0.0)
    program = stream_model("drivetrain").stream(seed=stream.seed, sample_rate_hz=SAMPLE_RATE_HZ, block_size=BLOCK_SIZE, params=stream.params, tuning=stream.tuning)
    program.update(SourceSpec(id=stream.source_id, kind="motor", asset_id="motor", model="drivetrain", agent_kind="robot", position=(0.0, 0.0, 0.0), level_db=0.0, state=STATE))
    blocks = 50
    histories: list[np.ndarray | None] = [None, None]
    expected = []
    for _ in range(blocks):
        block = np.zeros((CHANNELS, BLOCK_SIZE), dtype=np.float32)
        for side, output in enumerate(program.render_sides(BLOCK_SIZE)):
            delayed, histories[side] = streaming_fractional_delays(output, delays[:, side], histories[side])
            block += delayed * gains[:, side, None].astype(np.float32)
        expected.append(block)

    np.testing.assert_array_equal(airborne_field[:, : blocks * BLOCK_SIZE], np.concatenate(expected, axis=1))
    mics = np.asarray([mic.position_m for mic in load_array_spec("four_mic").mics])
    points = np.asarray(MotorGroup.defaults()[MotorGroup.DRIVE_POINTS_M.field])
    np.testing.assert_allclose(delays, np.linalg.norm(mics[:, None, :] - points[None, :, :], axis=2) / SPEED_OF_SOUND_MPS * SAMPLE_RATE_HZ)
    assert delays[0, 0] < delays[0, 1]
    assert delays[1, 0] > delays[1, 1]
    assert not np.allclose(airborne_field[0], airborne_field[1])


def test_structural_layer_is_fixed_by_the_robot_name() -> None:
    first = _render_motor(_own_motor(0.5), 1.0)
    again = _render_motor(_own_motor(0.5), 1.0)
    other = _render_motor(_own_motor(0.5, robot="dingo"), 1.0)

    assert np.array_equal(first, again)
    assert not np.allclose(first, other)


def test_own_motor_paths_survive_the_render_trace() -> None:
    stream = _own_motor(0.2)
    inputs = RenderInputs(block_index=0, clips=(), continuous=(), streams=(stream,), channels=CHANNELS)

    replayed = render_inputs_from_json(render_inputs_to_json(inputs), lambda key: np.zeros(1, dtype=np.float32), lambda key: 1.0)

    assert replayed.streams == (stream,)
