"""Transient pedestrian sounds derived from pedestrian state: footsteps while walking, speech when two pedestrians meet."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import attrs
import numpy as np

from arena_auditory.shared import Vec3

FOOTSTEP = "footstep"
SPEECH = "speech"


@attrs.frozen(kw_only=True)
class PedestrianState:
    """One pedestrian sample. speed_mps is the twist speed from ``arena_peds``, gait_phase 0 means unknown."""

    id: int
    name: str
    position: Vec3
    yaw_rad: float
    speed_mps: float
    gait_phase: float


class PedestrianEventDetector:
    """Emits the kinds footstep and speech from successive pedestrian snapshots."""

    def __init__(
        self,
        emit: Callable[[str, PedestrianState], None],
        *,
        walking_speed_mps: float,
        footstep_interval_s: float,
        greeting_distance_m: float,
        greeting_fov_deg: float,
        greeting_cooldown_s: float,
    ) -> None:
        self._emit = emit
        self._walking_speed_mps = walking_speed_mps
        self._footstep_interval_s = footstep_interval_s
        self._greeting_distance_m = greeting_distance_m
        self._greeting_fov_rad = math.radians(greeting_fov_deg)
        self._greeting_cooldown_s = greeting_cooldown_s

        self._last_footstep_by_ped: dict[int, float] = {}
        self._last_half_step_by_ped: dict[int, int] = {}
        self._last_greeting_by_pair: dict[tuple[int, int], float] = {}
        self._last_pose_by_ped: dict[int, tuple[float, float, float]] = {}

    def reset(self) -> None:
        """Forget every pedestrian, as at an episode change."""
        self._last_footstep_by_ped.clear()
        self._last_half_step_by_ped.clear()
        self._last_greeting_by_pair.clear()
        self._last_pose_by_ped.clear()

    def update(self, pedestrians: Sequence[PedestrianState], now_s: float) -> None:
        peds = list(pedestrians)
        current_ids = {ped.id for ped in peds}

        for ped in peds:
            derived_speed = self._pose_delta_speed(ped, now_s)
            if max(ped.speed_mps, derived_speed) > self._walking_speed_mps:
                self._maybe_emit_footstep(ped, now_s)

        for stale_id in set(self._last_pose_by_ped) - current_ids:
            self._last_pose_by_ped.pop(stale_id, None)
            self._last_footstep_by_ped.pop(stale_id, None)
            self._last_half_step_by_ped.pop(stale_id, None)

        if len(peds) < 2:
            return
        xy = np.array([ped.position[:2] for ped in peds], dtype=np.float64)
        distance = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        for a, b in zip(*np.nonzero(np.triu(distance <= self._greeting_distance_m + 1e-9, k=1)), strict=True):
            self._maybe_emit_greeting(peds[a], peds[b], now_s)

    def _pose_delta_speed(self, ped: PedestrianState, now_s: float) -> float:
        x, y = float(ped.position[0]), float(ped.position[1])
        previous = self._last_pose_by_ped.get(ped.id)
        self._last_pose_by_ped[ped.id] = (x, y, now_s)
        if previous is None:
            return 0.0
        previous_x, previous_y, previous_time = previous
        elapsed = now_s - previous_time
        if elapsed <= 1e-4:
            return 0.0
        return math.hypot(x - previous_x, y - previous_y) / elapsed

    def _maybe_emit_footstep(self, ped: PedestrianState, now_s: float) -> None:
        """One step per half gait cycle when the producer publishes gait_phase, else a fixed interval."""
        if ped.gait_phase > 0.0:
            half_step = int(ped.gait_phase // math.pi)
            previous = self._last_half_step_by_ped.get(ped.id)
            self._last_half_step_by_ped[ped.id] = half_step
            if previous is None or previous == half_step:
                return
        else:
            last = self._last_footstep_by_ped.get(ped.id, -math.inf)
            if now_s - last < self._footstep_interval_s:
                return
            self._last_footstep_by_ped[ped.id] = now_s
        self._emit(FOOTSTEP, ped)

    def _maybe_emit_greeting(self, ped_a: PedestrianState, ped_b: PedestrianState, now_s: float) -> None:
        emitter: PedestrianState | None = None
        if self._sees(ped_a, ped_b):
            emitter = ped_a
        elif self._sees(ped_b, ped_a):
            emitter = ped_b
        if emitter is None:
            return

        low, high = sorted((ped_a.id, ped_b.id))
        last = self._last_greeting_by_pair.get((low, high), -math.inf)
        if now_s - last < self._greeting_cooldown_s:
            return
        self._last_greeting_by_pair[(low, high)] = now_s
        self._emit(SPEECH, emitter)

    def _sees(self, observer: PedestrianState, target: PedestrianState) -> bool:
        dx = target.position[0] - observer.position[0]
        dy = target.position[1] - observer.position[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-6 or distance > self._greeting_distance_m:
            return False
        target_angle = math.atan2(dy, dx)
        angle_error = math.atan2(math.sin(target_angle - observer.yaw_rad), math.cos(target_angle - observer.yaw_rad))
        return abs(angle_error) <= self._greeting_fov_rad / 2.0
