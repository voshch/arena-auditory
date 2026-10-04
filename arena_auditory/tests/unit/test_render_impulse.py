from __future__ import annotations

import numpy as np
import pytest
from arena_auditory.propagation.pyroom_adapter import RoomImpulseResponse
from arena_auditory.propagation.rir import impulse_from_rir
from arena_auditory.render.core import impulse_shape, prepare_clip, trim_impulse

GLOBAL_DELAY = 40
PHYSICAL_DELAY = 100


def _rir() -> RoomImpulseResponse:
    samples = np.zeros(400)
    samples[GLOBAL_DELAY + PHYSICAL_DELAY] = 0.5
    samples[GLOBAL_DELAY + PHYSICAL_DELAY + 60] = 0.2
    return RoomImpulseResponse(samples=samples, sample_rate_hz=16000, global_delay_samples=GLOBAL_DELAY, fallback_material_ids=())


def test_rir_direct_path_lands_at_the_physical_delay() -> None:
    impulse = impulse_from_rir(_rir(), key="room", backend="pyroomacoustics")

    assert impulse.lead_samples == PHYSICAL_DELAY
    assert int(np.argmax(np.abs(impulse.samples))) == PHYSICAL_DELAY
    assert impulse.samples[PHYSICAL_DELAY] == pytest.approx(1.0)
    assert impulse.samples[PHYSICAL_DELAY + 60] == pytest.approx(0.4)


def test_rir_without_energy_is_rejected() -> None:
    rir = RoomImpulseResponse(samples=np.zeros(64), sample_rate_hz=16000, global_delay_samples=40, fallback_material_ids=())

    with pytest.raises(ValueError, match="direct arrival"):
        impulse_from_rir(rir, key="silent", backend="pyroomacoustics")


def test_convolved_clip_puts_the_direct_path_at_the_reception_delay() -> None:
    impulse = impulse_from_rir(_rir(), key="room", backend="pyroomacoustics")
    shape = impulse_shape(impulse.samples, impulse.sample_rate_hz, impulse.lead_samples, 16000)
    dry = np.zeros(32, dtype=np.float32)
    dry[0] = 1.0

    delay, wet = prepare_clip(dry, 94.0, -26.0, float(PHYSICAL_DELAY), level_rms=1.0, impulse=shape)

    assert delay + int(np.argmax(np.abs(wet))) == PHYSICAL_DELAY
    late_delay, late = prepare_clip(dry, 94.0, -26.0, PHYSICAL_DELAY + 7.0, level_rms=1.0, impulse=shape)
    assert late_delay + int(np.argmax(np.abs(late))) == PHYSICAL_DELAY + 7


def test_impulse_shape_scales_the_lead_to_the_render_rate() -> None:
    samples = np.zeros(960, dtype=np.float32)
    samples[300] = 1.0
    samples[600] = 0.3

    shape = impulse_shape(samples, 48000, 300, 16000)

    assert shape.lead_samples == 100
    assert abs(int(np.argmax(np.abs(shape.samples))) - 100) <= 1


def test_trim_impulse_cuts_the_tail_below_the_floor_but_never_before_the_direct_arrival() -> None:
    samples = np.zeros(64, dtype=np.float32)
    samples[0] = 1.0
    samples[50] = 1e-4

    assert trim_impulse(samples, 0).size == 1
    assert trim_impulse(samples, 30).size == 31
    assert trim_impulse(samples, 0, floor_db=-100.0).size == 51
    assert np.array_equal(trim_impulse(np.zeros(8, dtype=np.float32), 3), np.zeros(8, dtype=np.float32))
