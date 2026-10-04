"""Room impulses shared across processes: the Impulse type, its cache key and the RIR conversion."""

from __future__ import annotations

import array
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Sequence

import attrs
import numpy as np
from arena_auditory_msgs.msg import RoomImpulse
from builtin_interfaces.msg import Time
from std_msgs.msg import Header

from arena_auditory.propagation.pyroom_adapter import RoomImpulseResponse, direct_arrival
from arena_auditory.shared import Vec3

IMPULSE_WINDOW = 64


@attrs.frozen
class Impulse:
    """Room impulse whose direct arrival sits at lead_samples with windowed amplitude 1."""

    key: str
    samples: np.ndarray = attrs.field(eq=False)
    sample_rate_hz: int
    lead_samples: int
    backend: str

    def to_msg(self, stamp: Time) -> RoomImpulse:
        return RoomImpulse(
            header=Header(stamp=stamp),
            key=self.key,
            sample_rate_hz=int(self.sample_rate_hz),
            lead_samples=int(self.lead_samples),
            samples=array.array("f", np.ascontiguousarray(self.samples, dtype=np.float32).tobytes()),
            backend=self.backend,
        )

    @classmethod
    def from_msg(cls, msg: RoomImpulse) -> Impulse:
        return cls(
            key=msg.key,
            samples=np.asarray(msg.samples, dtype=np.float32),
            sample_rate_hz=int(msg.sample_rate_hz),
            lead_samples=int(msg.lead_samples),
            backend=msg.backend,
        )


class RirCache:
    """Thread-safe LRU of impulses by key."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(int(capacity), 1)
        self._entries: OrderedDict[str, Impulse] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: str) -> Impulse | None:
        with self._lock:
            impulse = self._entries.get(key)
            if impulse is not None:
                self._entries.move_to_end(key)
            return impulse

    def put(self, impulse: Impulse) -> bool:
        """True when the key is new, the caller publishes it then."""
        with self._lock:
            known = impulse.key in self._entries
            self._entries[impulse.key] = impulse
            self._entries.move_to_end(impulse.key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
            return not known

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def digest(*parts: object) -> str:
    """Stable blake2b hex of the repr of parts."""
    return hashlib.blake2b(repr(parts).encode(), digest_size=8).hexdigest()


def rir_key(
    *,
    world_signature: str,
    backend: str,
    zones: Sequence[str],
    portal_ids: Sequence[str],
    source: Vec3,
    listener: Vec3,
    quantization_m: float,
    rir_digest: str,
) -> str:
    """blake2b hex of the world, route, quantized positions and RIR config, stable across processes."""

    def quantize(position: Vec3) -> tuple[int, ...]:
        return tuple(round(float(value) / quantization_m) for value in position)

    payload = (world_signature, backend, tuple(zones), tuple(portal_ids), quantize(source), quantize(listener), rir_digest)
    return hashlib.blake2b(repr(payload).encode(), digest_size=16).hexdigest()


def impulse_from_rir(rir: RoomImpulseResponse, *, key: str, backend: str) -> Impulse:
    """Drop the fractional-delay filter latency and normalize the direct arrival. Raises ValueError for an RIR without a finite direct arrival."""
    arrival = direct_arrival(rir)
    if not np.isfinite(arrival.amplitude) or arrival.amplitude <= 0.0:
        raise ValueError("RIR has no finite non-zero direct arrival")
    start = max(int(rir.global_delay_samples), 0)
    samples = np.asarray(rir.samples, dtype=np.float64)[start:] / arrival.amplitude
    return Impulse(
        key=key,
        samples=samples.astype(np.float32),
        sample_rate_hz=int(rir.sample_rate_hz),
        lead_samples=max(arrival.index - start, 0),
        backend=backend,
    )
