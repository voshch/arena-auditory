"""Decaying directional pedestrian-likelihood grid, in the map frame.

Pure numpy: no ROS imports, so the same update runs inside the belief node and
inside offline replay tooling. That is deliberate: analysis figures and the
live layer must be the same estimator.

Model
-----
The layer consumes only what a sound front-end can produce: an
``AuditoryDetection`` tuple of (kind, bearing, received level, timestamp). It
never reads emitter identity, source position or true range.

Each event paints a wedge from the robot pose along the reported bearing:

* angular extent ``wedge_deg``, weighted by a Gaussian in
  lateral offset with sigma = half-width / 2, where the half-width at forward
  distance ``d`` is ``max(d * tan(wedge / 2), min_half_width_m)``: the wedge
  is never narrower than the array footprint, so the cells
  around the robot carry full weight instead of a one-cell sliver;
* radial extent out to ``R``.  With ``level_range_enabled`` the range is implied by
  inverting the simulator's spreading law
  ``received = emission - 20 log10(max(r, 1 m))`` against a per-kind emission
  level (the level of the kind's default sound asset), clamped to ``[min_range_m, max_range_m]``.  By default (off, the
  corpus finding: level is not a range cue, and a wall costs ~7 dB on the live
  bus so an occluded pedestrian sounds twice as far), or when the event carries
  no usable level, ``R = max_range_m`` (by default the median per-episode
  maximum source range in the recorded corpus) and the radial profile is flat;
* radial weighting when ``R`` is level-derived: a Gaussian bump at ``R`` with
  sigma = ``range_sigma_frac * R`` (the level-to-range inversion is coarse),
  plus a flat floor ``range_floor_weight`` over ``[0, R]`` so the whole wedge
  still carries mass.  Without the bump the argmax along the ray would be
  arbitrary and any attribution rate meaningless.

The wedge is painted **through walls on purpose**: an occluded pedestrian is
audible, and the belief must be able to put mass behind a corner where no
range sensor can see.  It is masked by ``free`` so mass never lands in an
occupied cell itself, only in the free cells the wedge crosses into.

Mass decays as ``exp(-dt / tau_s)``, so a source that stops emitting fades
rather than latching.

Outputs
-------
``normalized()``  belief in [0, 1] = mass / full scale, clipped.  Full scale is
the steady state of one source emitting at ``event_rate_hz``, i.e.
``event_mass * event_rate_hz * tau_s``, so the normalisation follows
the front-end's event rate instead of being retuned by hand.
``belief_int8()``  that, times 100, as an ``OccupancyGrid`` payload for RViz.
``speed_mask_int8()``  the Nav2 SpeedFilter mask.  The filter reads the mask at
the robot's own cell, so the mask is the belief max-filtered over a disc of
``reaction_radius_m`` before thresholding: the robot is slowed while likely
pedestrian mass lies within that radius of it.  Nav2 semantics (see
``nav2_costmap_2d/costmap_filters/filter_values.hpp``): mask value 0 =
``SPEED_MASK_NO_LIMIT`` (no restriction), -1 = unknown, and any other value v
gives ``speed_limit = base + multiplier * v``, which for filter type 1 is a
*percentage of maximum speed*.  So a **high** mask value means **fast**.  Cells
below ``belief_threshold`` therefore get 0 (free), and cells above it get
``speed_free_pct`` linearly pulled down to ``speed_min_pct`` as the belief goes
from the threshold to 1.  The mask never emits a value below ``speed_min_pct``
(and never 0 for a restricted cell), so this layer slows the robot and never
commands a hard stop through the filter.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import attrs
import numpy as np
from scipy.ndimage import maximum_filter1d

if TYPE_CHECKING:
    from nav_msgs.msg import MapMetaData

    from arena_auditory.assets import SoundLibrary
    from arena_auditory.hearing.policy import PolicyConfig

# Nav2 costmap_filters/filter_values.hpp
SPEED_MASK_NO_LIMIT = 0

NOMINAL_EVENT_RATE_HZ: dict[str, float] = {
    "bus": 2.0,
    "srp": 2.0,
    "seld": 10.0,
}


def emission_levels(library: SoundLibrary) -> dict[str, float]:
    """Emission level per detect kind, the level of the kind's default asset."""
    return {name: float(library.default_asset(name).level_db) for name, kind in library.kinds().items() if kind.detect and kind.default_asset}


@attrs.frozen(kw_only=True)
class BeliefConfig:
    """Estimator settings, the belief.* parameters plus the per-kind emission levels."""

    wedge_deg: float
    min_half_width_m: float
    tau_s: float
    max_range_m: float
    min_range_m: float
    reference_distance_m: float
    level_range_enabled: bool
    range_sigma_frac: float
    range_floor_weight: float
    event_mass: float
    event_rate_hz: float
    mass_full_scale: float
    emission_db: dict[str, float]

    def full_scale(self) -> float:
        if self.mass_full_scale > 0.0:
            return float(self.mass_full_scale)
        return max(
            float(self.event_mass) * float(self.event_rate_hz) * float(self.tau_s),
            1e-9,
        )

    def emission_for(self, sound_type: str) -> float:
        return float(self.emission_db.get((sound_type or "").strip().lower(), math.nan))


def wrap_pi(a: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


class BeliefGrid:
    """Peak-normalised likelihood grid over a fixed map-frame rectangle."""

    def __init__(
        self,
        origin_x: float,
        origin_y: float,
        resolution: float,
        width: int,
        height: int,
        config: BeliefConfig,
        free: np.ndarray | None = None,
    ) -> None:
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.resolution = float(resolution)
        self.width = int(width)
        self.height = int(height)
        self.config = config
        self.mass = np.zeros((self.height, self.width), dtype=np.float32)
        self.free = np.ones((self.height, self.width), dtype=bool) if free is None else free
        self._xs = self.origin_x + (np.arange(self.width, dtype=np.float64) + 0.5) * self.resolution
        self._ys = self.origin_y + (np.arange(self.height, dtype=np.float64) + 0.5) * self.resolution
        self.n_events = 0

    @classmethod
    def from_occupancy_info(cls, info: MapMetaData, config: BeliefConfig, free: np.ndarray | None = None) -> BeliefGrid:
        """``info`` is a ``nav_msgs/MapMetaData``."""
        return cls(
            origin_x=info.origin.position.x,
            origin_y=info.origin.position.y,
            resolution=info.resolution,
            width=info.width,
            height=info.height,
            config=config,
            free=free,
        )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.origin_x + (col + 0.5) * self.resolution,
            self.origin_y + (row + 0.5) * self.resolution,
        )

    def decay(self, dt: float) -> None:
        if dt <= 0.0:
            return
        tau = max(float(self.config.tau_s), 1e-6)
        self.mass *= float(math.exp(-dt / tau))

    def clear(self) -> None:
        self.mass[:] = 0.0
        self.n_events = 0

    def range_estimate(self, sound_type: str, received_db: float | None) -> tuple[float, bool]:
        """Return (range_m, level_derived).

        Inverts the simulator's spreading law.  ``level_derived`` False means the
        fallback flat wedge out to ``max_range_m`` is in force.
        """
        p = self.config
        emission = p.emission_for(sound_type)
        if not p.level_range_enabled or received_db is None or not np.isfinite(received_db) or not np.isfinite(emission):
            return float(p.max_range_m), False
        r = float(p.reference_distance_m) * 10.0 ** ((emission - float(received_db)) / 20.0)
        if not np.isfinite(r):
            return float(p.max_range_m), False
        return float(np.clip(r, p.min_range_m, p.max_range_m)), True

    def add_event(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        bearing_rad: float,
        sound_type: str = "",
        received_db: float | None = None,
        weight: float = 1.0,
        bearing_frame: str = "robot",
    ) -> dict:
        """Paint one event.  ``bearing_rad`` is CCW-positive.

        ``bearing_frame='robot'`` (the front-end / label convention, and the
        ``bearing_robot_rad`` column) means the consumer rotates the bearing by
        the robot's yaw into the map frame; ``'map'`` takes the bearing as
        already absolute, which is what a bus detection carries by construction.
        """
        p = self.config
        theta = float(bearing_rad) + (float(robot_yaw) if bearing_frame == "robot" else 0.0)
        rng, level_derived = self.range_estimate(sound_type, received_db)

        half = math.radians(float(p.wedge_deg)) * 0.5
        tan_half = math.tan(min(half, math.radians(89.0)))
        min_hw = max(float(p.min_half_width_m), 0.0)
        sigma_rad = max(float(p.range_sigma_frac) * rng, self.resolution)
        r_paint = min(float(p.max_range_m), rng + 2.5 * sigma_rad) if level_derived else rng

        r0, r1, c0, c1 = self._wedge_box(robot_x, robot_y, theta, min(half, math.radians(89.0)), r_paint, min_hw)
        if r1 <= r0 or c1 <= c0:
            return {"painted": 0, "range_m": rng, "level_derived": level_derived, "theta": theta}

        dx = self._xs[c0:c1][None, :] - robot_x
        dy = self._ys[r0:r1][:, None] - robot_y
        rr = np.hypot(dx, dy)
        ct, st = math.cos(theta), math.sin(theta)
        fwd = dx * ct + dy * st
        lat = np.abs(-dx * st + dy * ct)
        hw = np.maximum(fwd * tan_half, min_hw)

        inside = (rr <= r_paint) & (fwd >= 0.0) & (lat <= hw)
        if not inside.any():
            return {"painted": 0, "range_m": rng, "level_derived": level_derived, "theta": theta}

        lat32 = lat.astype(np.float32)
        hw32 = hw.astype(np.float32)
        w = np.exp(np.float32(-0.5) * (lat32 / np.maximum(hw32 * np.float32(0.5), np.float32(1e-3))) ** 2)
        if level_derived:
            rr32 = rr.astype(np.float32)
            w_rad = np.exp(np.float32(-0.5) * ((rr32 - np.float32(rng)) / np.float32(sigma_rad)) ** 2)
            w_rad += np.float32(p.range_floor_weight) * (rr <= rng)
            w *= w_rad
        w[~inside] = 0.0
        peak = float(w.max())
        if peak <= 0.0:
            return {"painted": 0, "range_m": rng, "level_derived": level_derived, "theta": theta}
        # peak-normalised: one event deposits event_mass at its likeliest cell whatever the wedge size
        w *= np.float32((float(p.event_mass) * float(weight)) / peak)
        w *= self.free[r0:r1, c0:c1]

        self.mass[r0:r1, c0:c1] += w
        self.n_events += 1
        return {
            "painted": int(inside.sum()),
            "range_m": rng,
            "level_derived": level_derived,
            "theta": theta,
        }

    def _wedge_box(self, x: float, y: float, theta: float, half: float, r_paint: float, pad: float) -> tuple[int, int, int, int]:
        """Cell rows [r0, r1) and cols [c0, c1) covering the sector plus ``pad``."""
        angles = [theta - half, theta + half, *(a for a in np.arange(4) * (math.pi / 2.0) if abs(float(wrap_pi(a - theta))) <= half)]
        xs = [x, *(x + r_paint * math.cos(a) for a in angles)]
        ys = [y, *(y + r_paint * math.sin(a) for a in angles)]
        pad += self.resolution
        r0, c0 = self.world_to_cell(min(xs) - pad, min(ys) - pad)
        r1, c1 = self.world_to_cell(max(xs) + pad, max(ys) + pad)
        return max(r0, 0), min(r1 + 1, self.height), max(c0, 0), min(c1 + 1, self.width)

    def normalized(self) -> np.ndarray:
        return np.clip(self.mass / self.config.full_scale(), 0.0, 1.0)

    def belief_int8(self) -> np.ndarray:
        return np.round(self.normalized() * 100.0).astype(np.int8)

    def speed_mask_int8(self, policy: PolicyConfig) -> np.ndarray:
        return speed_mask_from_belief(self.normalized(), self.resolution, policy)

    def argmax_world(self) -> tuple[float, float, float]:
        """(x, y, mass) of the maximum-mass cell.  mass 0.0 if the grid is empty."""
        if not np.isfinite(self.mass).any() or float(self.mass.max()) <= 0.0:
            return float("nan"), float("nan"), 0.0
        idx = int(np.argmax(self.mass))
        row, col = divmod(idx, self.width)
        x, y = self.cell_to_world(row, col)
        return x, y, float(self.mass[row, col])


def dilate_belief(belief: np.ndarray, resolution: float, reaction_radius_m: float) -> np.ndarray:
    """Max filter over the ``reaction_radius_m`` disc, floored at 0, on the box of positive cells."""
    k = int(round(float(reaction_radius_m) / resolution))
    if k <= 0:
        return belief
    out = np.zeros_like(belief)
    positive = belief > 0
    rows = np.flatnonzero(positive.any(axis=1))
    if rows.size == 0:
        return out
    cols = np.flatnonzero(positive.any(axis=0))
    h, w = belief.shape
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = max(int(cols[0]) - k, 0), min(int(cols[-1]) + 1 + k, w)
    src = belief[r0:r1, c0:c1]
    filtered: dict[int, np.ndarray] = {}
    for dy in range(-k, k + 1):
        half = int(math.floor(math.sqrt(k * k - dy * dy)))
        row = filtered.get(half)
        if row is None:
            row = filtered[half] = maximum_filter1d(src, 2 * half + 1, axis=1, mode="constant", cval=0.0)
        d0, d1 = max(r0 + dy, 0), min(r1 + dy, h)
        if d1 > d0:
            np.maximum(out[d0:d1, c0:c1], row[d0 - dy - r0 : d1 - dy - r0], out=out[d0:d1, c0:c1])
    return out


def speed_mask_from_belief(belief: np.ndarray, resolution: float, policy: PolicyConfig) -> np.ndarray:
    """Nav2 SpeedFilter mask from a normalised belief: 0 = no limit, else percentage of max speed.
    Cells below ``belief_threshold`` after dilation are free, above it the percentage falls linearly
    from ``speed_free_pct`` to ``speed_min_pct`` as the belief goes from the threshold to 1."""
    b = dilate_belief(belief, resolution, policy.reaction_radius_m)
    thr = float(np.clip(policy.belief_threshold, 0.0, 0.999))
    span = max(1.0 - thr, 1e-6)
    frac = np.clip((b - thr) / span, 0.0, 1.0)
    pct = float(policy.speed_free_pct) - frac * (float(policy.speed_free_pct) - float(policy.speed_min_pct))
    pct = np.clip(np.round(pct), policy.speed_min_pct, policy.speed_free_pct)
    return np.where(b > thr, pct, SPEED_MASK_NO_LIMIT).astype(np.int8)
