"""Offline replay of the hearing belief layer over one exported episode, run from a source checkout: python3 tools/replay_belief.py EPISODE_DIR.

Feeds the detections of an episode (export_recording layout: <prefix>raw.wav,
<prefix>frame_labels.parquet, <prefix>occupancy_map.npz) through the same
BeliefGrid update the belief node runs, prints the attribution rate and renders
ground truth beside the belief around an occlusion transition.

--frontend seld runs the SELDnet front-end offline on the wav. --frontend labels
is the oracle control with the true bearings, which separates front-end error
from belief-model error.

The received level is the frame level of the wav plus a microphone-gain offset,
fitted with --fit-offset on calibration episodes disjoint from the replayed one.
Without an offset the belief paints the flat max-range wedge.
"""

from __future__ import annotations

import argparse
import math
import wave
from collections.abc import Sequence
from pathlib import Path

import attrs
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from arena_auditory.assets import SoundLibrary
from arena_auditory.hearing import weights
from arena_auditory.hearing.belief_grid import NOMINAL_EVENT_RATE_HZ, BeliefConfig, BeliefGrid, emission_levels
from arena_auditory.hearing.seld import Detection, SeldFrontend, load_wav
from arena_auditory.params import BeliefGroup, Frontend, MapGroup, PolicyGroup, SeldGroup, configure
from arena_auditory.shared import dbfs_from_rms

LABEL_HOP_S = 0.1
MIN_CALIBRATION_ROWS = 50
MIN_EVENT_WEIGHT = 0.05
MIN_DRAWN_BELIEF = 0.02
MIN_PEAK_BELIEF = 0.05
REPORT_TOLERANCES_M = (0.5, 1.0, 1.5, 2.0, 3.0)
BEARING_TOLERANCE_DEG = 10.0
CROP_PAD_M = 3.0

_COLUMNS = (
    "recording_time_seconds",
    "robot_x",
    "robot_y",
    "robot_yaw",
    "pedestrian_sound_active",
    "active_pedestrian_count",
    "active_sound_types",
    "motor_active",
    "pedestrian_x",
    "pedestrian_y",
    "line_of_sight",
    "range_m",
    "bearing_robot_rad",
)


@attrs.frozen
class OccupancySnapshot:
    """Occupancy map of an episode, row-major from the origin cell."""

    occupancy: np.ndarray = attrs.field(eq=False)
    resolution_m: float
    origin_x: float
    origin_y: float
    frame_id: str

    @property
    def width(self) -> int:
        return int(self.occupancy.shape[1])

    @property
    def height(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.origin_x, self.origin_x + self.width * self.resolution_m, self.origin_y, self.origin_y + self.height * self.resolution_m)

    @classmethod
    def load(cls, path: Path) -> OccupancySnapshot:
        with np.load(path) as data:
            origin = np.asarray(data["origin"], dtype=np.float64)
            return cls(
                occupancy=np.asarray(data["occupancy"], dtype=np.int8),
                resolution_m=float(data["resolution"]),
                origin_x=float(origin[0]),
                origin_y=float(origin[1]),
                frame_id=str(data["frame_id"]),
            )


@attrs.frozen
class Episode:
    raw_wav: Path
    labels: pa.Table = attrs.field(eq=False)
    occupancy: OccupancySnapshot
    sample_rate_hz: int


@attrs.frozen
class FrameStats:
    frame: int
    peak_x: float
    peak_y: float
    peak_mass: float
    peak_dist_m: float
    bearing_err_deg: float
    belief_at_truth: float
    n_events: int
    any_los: int
    n_active: int


@attrs.frozen
class ReplayResult:
    grid: BeliefGrid = attrs.field(eq=False)
    snapshots: dict[int, np.ndarray] = attrs.field(eq=False)
    truth: dict[int, list[tuple[float, float, bool]]]
    poses: dict[int, tuple[float, float, float]]
    frames: tuple[FrameStats, ...]
    attribution_rate: float
    attribution_frames: int
    peak_dists: np.ndarray = attrs.field(eq=False)
    median_peak_error_m: float
    tolerance_m: float


def load_episode(directory: Path, prefix: str = "") -> Episode:
    """Raises FileNotFoundError when one of the three export files is missing."""
    raw = directory / f"{prefix}raw.wav"
    labels = directory / f"{prefix}frame_labels.parquet"
    occupancy = directory / f"{prefix}occupancy_map.npz"
    for path in (raw, labels, occupancy):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing")
    with wave.open(str(raw), "rb") as stream:
        sample_rate_hz = int(stream.getframerate())
    return Episode(raw_wav=raw, labels=pq.read_table(labels), occupancy=OccupancySnapshot.load(occupancy), sample_rate_hz=sample_rate_hz)


def _rows(episode: Episode) -> list[dict]:
    table = episode.labels
    rows = table.select([name for name in _COLUMNS if name in table.column_names]).to_pylist()
    for row in rows:
        row["frame"] = int(float(row["recording_time_seconds"]) // LABEL_HOP_S)
    return rows


def _kind(label: str, library: SoundLibrary) -> str:
    """Kind of a recorded sound label, a kind name or an asset id."""
    if label in library.kinds():
        return label
    try:
        return library.asset(label).kind
    except KeyError:
        return label


def _types(row: dict, library: SoundLibrary) -> list[str]:
    return [_kind(str(name), library) for name in row.get("active_sound_types") or ()]


def seld_detections(episode: Episode, frontend: SeldFrontend) -> list[Detection]:
    return frontend.events_from_wav(episode.raw_wav)


def label_detections(episode: Episode, library: SoundLibrary) -> list[Detection]:
    """Oracle detections: the true robot-frame bearing of every sound-active pedestrian row, kind the loudest active detect kind."""
    levels = emission_levels(library)
    detect = {name for name, kind in library.kinds().items() if kind.detect}
    seen: set[tuple[int, str, float]] = set()
    out: list[Detection] = []
    for row in _rows(episode):
        bearing = row.get("bearing_robot_rad")
        if not row.get("pedestrian_sound_active") or bearing is None:
            continue
        candidates = [name for name in _types(row, library) if name in detect]
        kind = max(candidates, key=lambda name: levels.get(name, -math.inf)) if candidates else ""
        key = (row["frame"], kind, float(bearing))
        if key in seen:
            continue
        seen.add(key)
        out.append(Detection(frame=row["frame"], kind=kind, azimuth_rad=float(bearing), elevation_rad=0.0, activity=1.0))
    return out


def frame_level_db(wav: Path, n_frames: int, hop_s: float = LABEL_HOP_S) -> np.ndarray:
    """Level of each label frame over all channels, dBFS re a full-scale sine, -inf past the end."""
    audio, fs = load_wav(wav)
    hop = max(round(hop_s * fs), 1)
    out = np.full(n_frames, -np.inf, dtype=np.float64)
    for k in range(n_frames):
        seg = audio[k * hop : (k + 1) * hop]
        if seg.size == 0:
            break
        out[k] = dbfs_from_rms(float(np.sqrt(np.mean(seg * seg))), floor_db=-np.inf)
    return out


def fit_level_offset(episodes: Sequence[Episode], library: SoundLibrary) -> float:
    """Median of emission - 20 log10(range) - frame level over single-pedestrian line-of-sight frames, nan without any."""
    levels = emission_levels(library)
    values: list[float] = []
    for episode in episodes:
        rows = [
            row
            for row in _rows(episode)
            if row.get("pedestrian_sound_active") and row.get("active_pedestrian_count") == 1 and row.get("line_of_sight") and (row.get("range_m") or 0.0) > 1.0 and not row.get("motor_active")
        ]
        if len(rows) < MIN_CALIBRATION_ROWS:
            continue
        by_frame: dict[int, list[dict]] = {}
        for row in rows:
            by_frame.setdefault(row["frame"], []).append(row)
        level = frame_level_db(episode.raw_wav, max(by_frame) + 1)
        for frame, group in by_frame.items():
            emissions = [levels[name] for name in _types(group[0], library) if name in levels]
            if not emissions or not np.isfinite(level[frame]):
                continue
            distance = float(np.median([float(row["range_m"]) for row in group]))
            values.append(max(emissions) - 20.0 * math.log10(distance) - float(level[frame]))
    return float(np.median(values)) if values else math.nan


def replay(episode: Episode, detections: Sequence[Detection], config: BeliefConfig, level_offset_db: float, *, tolerance_m: float = 1.0) -> ReplayResult:
    occ = episode.occupancy
    grid = BeliefGrid(occ.origin_x, occ.origin_y, occ.resolution_m, occ.width, occ.height, config)
    poses: dict[int, tuple[float, float, float]] = {}
    truth: dict[int, list[tuple[float, float, bool]]] = {}
    for row in _rows(episode):
        frame = row["frame"]
        poses.setdefault(frame, (float(row["robot_x"]), float(row["robot_y"]), float(row["robot_yaw"])))
        if row.get("pedestrian_sound_active") and row.get("pedestrian_x") is not None:
            truth.setdefault(frame, []).append((float(row["pedestrian_x"]), float(row["pedestrian_y"]), bool(row.get("line_of_sight"))))
    by_frame: dict[int, list[Detection]] = {}
    for detection in detections:
        by_frame.setdefault(detection.frame, []).append(detection)
    levels = frame_level_db(episode.raw_wav, max(poses) + 1) if poses and math.isfinite(level_offset_db) else None

    snapshots: dict[int, np.ndarray] = {}
    stats: list[FrameStats] = []
    peak_dists: list[float] = []
    hits = 0
    for frame in sorted(poses):
        grid.decay(LABEL_HOP_S)
        x, y, yaw = poses[frame]
        received = float(levels[frame]) + level_offset_db if levels is not None and np.isfinite(levels[frame]) else None
        events = [d for d in by_frame.get(frame, ()) if math.isfinite(d.azimuth_rad)]
        for d in events:
            grid.add_event(robot_x=x, robot_y=y, robot_yaw=yaw, bearing_rad=d.azimuth_rad, sound_type=d.kind, received_db=received, weight=max(d.activity, MIN_EVENT_WEIGHT), bearing_frame="robot")
        bx, by, bm = grid.argmax_world()
        norm = grid.normalized()
        active = truth.get(frame, [])
        d_pos = bear_err = belief_at_truth = math.nan
        if active and bm > 0.0 and math.isfinite(bx):
            tx = np.array([p[0] for p in active])
            ty = np.array([p[1] for p in active])
            d_pos = float(np.min(np.hypot(tx - bx, ty - by)))
            peak_dists.append(d_pos)
            hits += int(d_pos <= tolerance_m)
            peak_angle = math.atan2(by - y, bx - x)
            truth_angles = np.arctan2(ty - y, tx - x)
            bear_err = float(np.min(np.abs(np.degrees((truth_angles - peak_angle + np.pi) % (2 * np.pi) - np.pi))))
            cells = [grid.world_to_cell(float(px), float(py)) for px, py in zip(tx, ty, strict=True)]
            inside = [float(norm[r, c]) for r, c in cells if 0 <= r < grid.height and 0 <= c < grid.width]
            belief_at_truth = max(inside) if inside else math.nan
        stats.append(
            FrameStats(
                frame=frame,
                peak_x=bx,
                peak_y=by,
                peak_mass=bm,
                peak_dist_m=d_pos,
                bearing_err_deg=bear_err,
                belief_at_truth=belief_at_truth,
                n_events=len(events),
                any_los=int(max(p[2] for p in active)) if active else -1,
                n_active=len(active),
            )
        )
        snapshots[frame] = grid.belief_int8()
    return ReplayResult(
        grid=grid,
        snapshots=snapshots,
        truth=truth,
        poses=poses,
        frames=tuple(stats),
        attribution_rate=hits / len(peak_dists) if peak_dists else math.nan,
        attribution_frames=len(peak_dists),
        peak_dists=np.asarray(peak_dists),
        median_peak_error_m=float(np.median(peak_dists)) if peak_dists else math.nan,
        tolerance_m=tolerance_m,
    )


def pick_timestamps(result: ReplayResult, n: int = 4, span_s: float = 1.5) -> list[int]:
    """Frames spanning span_s around a line-of-sight transition, picked in time so the panels stay contiguous."""
    active = [s for s in result.frames if s.n_active > 0]
    if not active:
        return []
    fr = np.array([s.frame for s in active])
    los = np.array([s.any_los for s in active])
    flips = np.nonzero(np.diff(los) != 0)[0]
    if len(flips) == 0:
        return sorted({int(fr[i]) for i in np.linspace(0, len(fr) - 1, n).astype(int)})
    half = round(span_s / LABEL_HOP_S)
    best, best_score = int(flips[len(flips) // 2]), -1
    for k in flips:
        score = int(((fr >= fr[k] - half) & (fr <= fr[k] + half)).sum())
        if score > best_score:
            best, best_score = int(k), score
    center = int(fr[best])
    lo, hi = center - half, center + half
    if lo < fr.min():
        lo, hi = int(fr.min()), int(fr.min()) + 2 * half
    if hi > fr.max():
        lo, hi = int(fr.max()) - 2 * half, int(fr.max())
    lo = max(lo, int(fr.min()))
    out: list[int] = []
    for want in np.linspace(lo, hi, n).round().astype(int):
        i = int(np.argmin(np.abs(fr - want)))
        if not out or fr[i] != out[-1]:
            out.append(int(fr[i]))
    return out


def render(result: ReplayResult, episode: Episode, stamps: Sequence[int], out_png: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.lines import Line2D

    c_robot, c_ped, c_peak, c_occ = "tab:blue", "tab:red", "tab:green", "0.35"
    cmap_belief = LinearSegmentedColormap.from_list("belief", ["white", c_peak])
    occ = episode.occupancy
    walls = np.ma.masked_where(occ.occupancy < MapGroup.OCCUPIED_THRESHOLD.default, np.ones_like(occ.occupancy, dtype=float))
    unknown = np.ma.masked_where(occ.occupancy >= 0, np.ones_like(occ.occupancy, dtype=float))
    grid = result.grid
    grid_extent = (grid.origin_x, grid.origin_x + grid.width * grid.resolution, grid.origin_y, grid.origin_y + grid.height * grid.resolution)
    stats = {s.frame: s for s in result.frames}

    xs: list[float] = []
    ys: list[float] = []
    for frame in stamps:
        x, y, _ = result.poses[frame]
        xs.append(x)
        ys.append(y)
        xs += [p[0] for p in result.truth.get(frame, [])]
        ys += [p[1] for p in result.truth.get(frame, [])]
    xlim = (min(xs) - CROP_PAD_M, max(xs) + CROP_PAD_M)
    ylim = (min(ys) - CROP_PAD_M, max(ys) + CROP_PAD_M)

    n = len(stamps)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n + 2.4, 6.3), squeeze=False, sharex=True, sharey=True)
    for j, frame in enumerate(stamps):
        x, y, yaw = result.poses[frame]
        active = result.truth.get(frame, [])
        for ax in (axes[0][j], axes[1][j]):
            ax.imshow(unknown, extent=occ.extent, origin="lower", cmap=ListedColormap(["0.88"]), interpolation="nearest")
            ax.imshow(walls, extent=occ.extent, origin="lower", cmap=ListedColormap([c_occ]), alpha=0.55, interpolation="nearest")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.plot([x], [y], "o", ms=8, color=c_robot, zorder=5)
            ax.plot([x, x + 0.9 * math.cos(yaw)], [y, y + 0.9 * math.sin(yaw)], "-", lw=1.6, color=c_robot, zorder=5)

        ax = axes[0][j]
        for px, py, los in active:
            ax.plot([px], [py], "o", ms=8, color=c_ped, zorder=5)
            ax.plot([x, px], [y, py], lw=1.0, color=c_ped, alpha=0.85 if los else 0.35, ls="-" if los else (0, (2, 2)), zorder=4)
        any_los = stats[frame].any_los
        ax.set_title(f"$t$ = {frame * LABEL_HOP_S:.1f} s\n" + ("line of sight" if any_los == 1 else "occluded"), fontsize=10)
        if j == 0:
            ax.set_ylabel("ground truth", fontsize=11)

        ax = axes[1][j]
        belief = result.snapshots[frame].astype(float) / 100.0
        ax.imshow(np.ma.masked_where(belief < MIN_DRAWN_BELIEF, belief), extent=grid_extent, origin="lower", cmap=cmap_belief, vmin=0.0, vmax=1.0, alpha=0.95, zorder=3)
        for px, py, _ in active:
            ax.plot([px], [py], "o", ms=8, color=c_ped, mfc="none", mew=1.8, zorder=6)
        peak = stats[frame]
        if math.isfinite(peak.peak_x) and float(belief.max()) >= MIN_PEAK_BELIEF:
            ax.plot([peak.peak_x], [peak.peak_y], "o", ms=7, color=c_peak, zorder=7)
        if j == 0:
            ax.set_ylabel("acoustic belief", fontsize=11)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=8, color=c_robot, label="robot"),
        Line2D([], [], marker="o", ls="none", ms=8, color=c_ped, label="pedestrian (truth)"),
        Line2D([], [], marker="o", ls="none", ms=8, mfc="none", mew=1.8, color=c_ped, label="pedestrian (truth), belief panel"),
        Line2D([], [], marker="o", ls="none", ms=7, color=c_peak, label="belief maximum"),
        Line2D([], [], ls="-", lw=1.2, color=c_ped, label="direct path, line of sight"),
        Line2D([], [], ls=(0, (2, 2)), lw=1.2, color=c_ped, label="direct path, occluded"),
        Line2D([], [], marker="s", ls="none", ms=9, color=c_occ, alpha=0.55, label="wall"),
        Line2D([], [], marker="s", ls="none", ms=9, color=cmap_belief(0.85), label="belief mass"),
    ]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 0.86), frameon=False, fontsize=9)
    cax = fig.add_axes((1.03, 0.10, 0.012, 0.24))
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0), cmap=cmap_belief), cax=cax)
    colorbar.set_label("pedestrian likelihood", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.outline.set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)
    return out_png


def _summary(result: ReplayResult, threshold: float) -> None:
    print(f"attribution rate (belief maximum within {result.tolerance_m:g} m of a true pedestrian): {result.attribution_rate:.3f} over {result.attribution_frames} frames")
    if len(result.peak_dists):
        print("  at other tolerances: " + "  ".join(f"{t:g} m {100 * float((result.peak_dists <= t).mean()):.1f} %" for t in REPORT_TOLERANCES_M))
    print(f"median belief-peak error: {result.median_peak_error_m:.2f} m")
    slices = (("all", lambda s: s.n_active > 0), ("line of sight", lambda s: s.any_los == 1), ("occluded", lambda s: s.any_los == 0))
    print(f"{'slice':14s} {'n':>5s} {'attrib':>7s} {'med err':>8s} {'bear<tol':>9s} {'med bear':>9s} {'belief@ped':>11s} {'>thr':>6s}")
    for name, keep in slices:
        sub = [s for s in result.frames if keep(s) and s.peak_mass > 0 and math.isfinite(s.peak_dist_m)]
        if not sub:
            continue
        dist = np.array([s.peak_dist_m for s in sub])
        bearing = np.array([s.bearing_err_deg for s in sub])
        belief = np.array([s.belief_at_truth for s in sub])
        print(
            f"  {name:12s} {len(sub):5d} {100 * float((dist <= result.tolerance_m).mean()):6.1f}% {float(np.median(dist)):7.2f}m "
            f"{100 * float((bearing <= BEARING_TOLERANCE_DEG).mean()):8.1f}% {float(np.median(bearing)):8.1f}d {float(np.nanmean(belief)):11.2f} {100 * float((belief > threshold).mean()):5.1f}%"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode_dir", type=Path, metavar="EPISODE_DIR", help="export_recording output directory of the replayed episode")
    parser.add_argument("--prefix", default="", help="file name prefix of the export files, shared by the calibration episodes")
    parser.add_argument("--frontend", choices=[Frontend.SELD.value, "labels"], default=Frontend.SELD.value, help="seld runs SELDnet on the wav, labels replays the true bearings")
    offset = parser.add_mutually_exclusive_group()
    offset.add_argument("--level-offset-db", type=float, default=math.nan, help="wav level to SPL offset, nan paints the flat max-range wedge")
    offset.add_argument("--fit-offset", type=Path, nargs="+", metavar="DIR", default=[], help="fit the level offset on these calibration episodes first")
    parser.add_argument("--out", type=Path, default=Path("belief_replay.png"), help="figure path")
    parser.add_argument("--device", default=SeldGroup.DEVICE.default, help="torch device of the seld front-end")
    parser.add_argument("--tolerance-m", type=float, default=1.0, help="attribution radius around a true pedestrian")
    args = parser.parse_args(argv)

    library = SoundLibrary.default()
    level_offset_db = args.level_offset_db
    if args.fit_offset:
        level_offset_db = fit_level_offset([load_episode(directory, args.prefix) for directory in args.fit_offset], library)
        print(f"level_offset_db {level_offset_db:.2f} from {len(args.fit_offset)} calibration episodes")

    episode = load_episode(args.episode_dir, args.prefix)
    if args.frontend == Frontend.SELD.value:
        files = weights.ensure()
        frontend = SeldFrontend(files["checkpoint"], files["scaler"], weights.classes(library), device=args.device, det_threshold=SeldGroup.DET_THRESHOLD.default, torch_threads=SeldGroup.TORCH_THREADS.default)
        detections = seld_detections(episode, frontend)
    else:
        detections = label_detections(episode, library)
    print(f"{args.frontend}: {len(detections)} detections over {len({d.frame for d in detections})} frames")

    config = configure(BeliefConfig, BeliefGroup, level_range_enabled=math.isfinite(level_offset_db), event_rate_hz=NOMINAL_EVENT_RATE_HZ[Frontend.SELD.value], emission_db=emission_levels(library))
    result = replay(episode, detections, config, level_offset_db, tolerance_m=args.tolerance_m)
    _summary(result, PolicyGroup.BELIEF_THRESHOLD.default)

    stamps = pick_timestamps(result)
    if not stamps:
        print("no sound-active frames, no figure rendered")
        return 0
    title = f"{args.episode_dir.name}  |  attribution {result.attribution_rate:.2f}"
    print(f"wrote {render(result, episode, stamps, args.out, title)} (frames {stamps})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
