"""Speed-mask composition and the listen-then-yield state machine, pure."""

from __future__ import annotations

import enum
import math

import attrs
import numpy as np

LEVEL_ALPHA = 0.3
SLOPE_ALPHA = 0.3


class State(enum.Enum):
    CRUISE = "cruise"
    LISTEN = "listen"
    YIELD = "yield"
    PASS = "pass"  # released after a contact: through the bend at listen speed


def compose_masks(*layers: np.ndarray) -> np.ndarray:
    """Lowest nonzero percentage per cell. 0 (no limit) never wins over a limit."""
    out = np.zeros_like(layers[0], dtype=np.int16)
    for layer in layers:
        lim = layer.astype(np.int16)
        take = (lim > 0) & ((out == 0) | (lim < out))
        out[take] = lim[take]
    return out.astype(np.int8)


def paint_lane(shape: tuple[int, int], origin: tuple[float, float], resolution: float, points_xy: np.ndarray, radius_m: float, pct: int) -> np.ndarray:
    """``pct`` on every cell within ``radius_m`` of any of ``points_xy``."""
    mask = np.zeros(shape, dtype=np.int8)
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.size == 0:
        return mask
    k = int(np.ceil(radius_m / resolution))
    h, w = shape
    for x, y in pts:
        c, r = int((x - origin[0]) / resolution), int((y - origin[1]) / resolution)
        r0, r1, c0, c1 = max(r - k, 0), min(r + k + 1, h), max(c - k, 0), min(c + k + 1, w)
        if r0 >= r1 or c0 >= c1:
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        cx, cy = origin[0] + (cc + 0.5) * resolution, origin[1] + (rr + 0.5) * resolution
        inside = (cx - x) ** 2 + (cy - y) ** 2 <= radius_m * radius_m
        mask[r0:r1, c0:c1][inside] = pct
    return mask


def mass_split(belief: np.ndarray, origin: tuple[float, float], resolution: float, bend_xy: tuple[float, float], radius_m: float, robot_xy: tuple[float, float], direction: tuple[float, float]) -> tuple[float, float, float]:
    """(mass in the bend disc, mass behind the robot along ``direction``, total mass)."""
    h, w = belief.shape
    cx = origin[0] + (np.arange(w) + 0.5) * resolution
    cy = origin[1] + (np.arange(h) + 0.5) * resolution
    xx, yy = np.meshgrid(cx, cy)
    total = float(belief.sum())
    disc = (xx - bend_xy[0]) ** 2 + (yy - bend_xy[1]) ** 2 <= radius_m * radius_m
    behind = (xx - robot_xy[0]) * direction[0] + (yy - robot_xy[1]) * direction[1] < 0.0
    return float(belief[disc].sum()), float(belief[behind].sum()), total


@attrs.frozen(kw_only=True)
class PolicyConfig:
    """Yield and speed-mask settings, the policy.* parameters the pure functions read."""

    yield_fraction: float
    release_fraction: float
    min_yield_s: float
    yield_timeout_s: float
    recede_s: float
    belief_threshold: float
    speed_min_pct: int
    speed_free_pct: int
    reaction_radius_m: float


@attrs.define
class YieldMachine:
    config: PolicyConfig
    state: State = State.CRUISE
    entered_at: float = 0.0
    contact: bool = False  # a pedestrian was heard at this bend
    yield_count: int = 0

    def step(self, now: float, *, in_approach: bool, past_bend: bool, frac_ahead: float, ahead: float, behind: float, event_age_s: float, receding_s: float) -> State:
        p = self.config
        if past_bend:
            self.state = State.CRUISE
        elif self.state is State.CRUISE and in_approach:
            self.state = State.LISTEN
        elif self.state is State.LISTEN:
            if not in_approach:
                self.state = State.CRUISE
            elif frac_ahead >= p.yield_fraction and ahead > 0.0:
                self.state = State.YIELD
                self.entered_at = now
                self.contact = True
                self.yield_count += 1
        elif self.state is State.YIELD:
            quiet = event_age_s >= p.min_yield_s
            passed = behind > ahead and now - self.entered_at >= p.min_yield_s
            faded = frac_ahead < p.release_fraction and quiet
            receded = receding_s >= p.recede_s and quiet
            if now - self.entered_at >= p.yield_timeout_s or passed or faded or receded:
                self.state = State.PASS
        return self.state

    def new_bend(self) -> None:
        self.contact = False
        if self.state in (State.YIELD, State.PASS):
            self.state = State.CRUISE


@attrs.define
class LevelTrend:
    """Slope in dB/s of an EMA of received levels, relaxing toward 0 with ``level_trend_tau_s`` between events."""

    level_trend_tau_s: float
    ema: float | None = attrs.field(default=None, init=False)
    _slope: float = attrs.field(default=0.0, init=False)
    _slope_t: float = attrs.field(default=0.0, init=False)
    _anchor_ema: float = attrs.field(default=0.0, init=False)
    _anchor_t: float = attrs.field(default=0.0, init=False)

    def observe(self, t: float, level: float) -> None:
        if self.ema is None:
            self.ema = self._anchor_ema = level
            self._slope, self._slope_t, self._anchor_t = 0.0, t, t
            return
        self.ema = (1.0 - LEVEL_ALPHA) * self.ema + LEVEL_ALPHA * level
        if t <= self._anchor_t:
            return
        rate = (self.ema - self._anchor_ema) / (t - self._anchor_t)
        self._slope = (1.0 - SLOPE_ALPHA) * self.slope(t) + SLOPE_ALPHA * rate
        self._slope_t = t
        self._anchor_ema, self._anchor_t = self.ema, t

    def slope(self, now: float) -> float:
        if self.ema is None:
            return 0.0
        return self._slope * math.exp(-max(now - self._slope_t, 0.0) / max(self.level_trend_tau_s, 1e-6))

    def reset(self) -> None:
        self.ema = None
        self._slope = 0.0
