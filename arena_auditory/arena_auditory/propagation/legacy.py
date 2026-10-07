"""Distance and occupancy-occlusion propagation, the backend of last resort."""

from __future__ import annotations

import math
import typing

from arena_robots.audio import SPEED_OF_SOUND_MPS

from arena_auditory.propagation import Emission, Listener, PropagationConfig, PropagationScene, Reception, bearing_rad


class LegacyBackend:
    name: typing.ClassVar[str] = "legacy_distance_occlusion"

    def __init__(self, config: PropagationConfig) -> None:
        self.config = config

    def propagate(self, emission: Emission, listener: Listener, scene: PropagationScene) -> Reception:
        source, position = emission.position, listener.position
        geometric_distance = math.hypot(source[0] - position[0], source[1] - position[1])
        effective_distance = max(geometric_distance, self.config.min_distance_m, 1e-3)
        occluded = scene.occupancy is not None and scene.occupancy.occluded(source, position)
        received = emission.level_db - 20.0 * math.log10(effective_distance) - (self.config.occlusion_db if occluded else 0.0)
        return Reception(
            listener=listener,
            distance_m=float(geometric_distance),
            bearing_rad=bearing_rad(source, position),
            received_level_db=float(received),
            threshold_db=self.config.threshold_db,
            direct_delay_s=float(geometric_distance / SPEED_OF_SOUND_MPS),
            audible=received >= self.config.threshold_db,
            occluded=occluded,
            backend=self.name,
        )
