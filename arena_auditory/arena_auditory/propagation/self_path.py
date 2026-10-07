"""Free-field direct path from a robot's own source to a listener it owns."""

from __future__ import annotations

import math
import typing

from arena_robots.audio import SPEED_OF_SOUND_MPS

from arena_auditory.propagation import Emission, Listener, PropagationConfig, PropagationScene, Reception, bearing_rad

MIN_SELF_DISTANCE_M = 0.01


class SelfPathBackend:
    name: typing.ClassVar[str] = "self_direct_path"

    def __init__(self, config: PropagationConfig) -> None:
        self.config = config

    def propagate(self, emission: Emission, listener: Listener, scene: PropagationScene) -> Reception:
        distance = max(math.dist(emission.position, listener.position), MIN_SELF_DISTANCE_M)
        received = emission.level_db - 20.0 * math.log10(distance)
        return Reception(
            listener=listener,
            distance_m=float(distance),
            bearing_rad=bearing_rad(emission.position, listener.position),
            received_level_db=float(received),
            threshold_db=self.config.threshold_db,
            direct_delay_s=float(distance / SPEED_OF_SOUND_MPS),
            audible=received >= self.config.threshold_db,
            occluded=False,
            backend=self.name,
        )
