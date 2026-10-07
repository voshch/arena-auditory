"""Offline acoustic checks: world coverage and connectivity audit, and the RIR plot of one static room."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import attrs
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_rclpy_mixins.param_groups import configure
from arena_simulation_setup.tree.World import MultiLevelWorldView, WorldIdentifier
from PIL import Image
from shapely.geometry import Point, Polygon

from arena_auditory.materials import default_catalog
from arena_auditory.params import PortalGroup, RirGroup, VizGroup, WorldGroup
from arena_auditory.propagation.plot import AcousticPlotDashboard, AcousticPlotSnapshot
from arena_auditory.propagation.pyroom_adapter import PyroomacousticsAdapter, RirConfig
from arena_auditory.rooms import AcousticBoundarySpec, AcousticRoomSpec, AcousticRoomSpecBuilder, AcousticWorldGraph
from arena_auditory.world import WorldConfig

DEFAULT_ROOM_CORNERS = (0.0, 0.0, 8.0, 0.0, 8.0, 6.0, 0.0, 6.0)
DEFAULT_SOURCE = (2.0, 2.0, 1.60)
DEFAULT_LISTENER = (6.0, 4.0, 0.35)


@attrs.frozen
class AcousticWorldAudit:
    world_name: str
    rooms: int
    door_portals: int
    opening_portals: int
    connected_components: int
    unpaired_doors: int
    overlapping_zone_pairs: tuple[tuple[str, str], ...]
    sampled_traversable_cells: int
    uncovered_traversable_cells: int
    uncovered_examples: tuple[tuple[float, float], ...]
    map_issues: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.uncovered_traversable_cells == 0 and not self.overlapping_zone_pairs and not self.map_issues


def audit_world(world_name: str, *, stride_cells: int) -> AcousticWorldAudit:
    return audit_world_view(world_name, WorldIdentifier(world_name).resolve_sync(), stride_cells=stride_cells)


def audit_world_view(world_name: str, world_view: MultiLevelWorldView, *, stride_cells: int) -> AcousticWorldAudit:
    """Rooms, portals, zone overlaps per level, and free cells of every level map.yaml outside all zones."""
    if stride_cells <= 0:
        raise ValueError("stride_cells must be positive")
    world = world_view.load()
    config = configure(WorldConfig, WorldGroup, PortalGroup)
    rooms = AcousticRoomSpecBuilder(config.ceiling_height_m).from_world(world)
    graph = AcousticWorldGraph.from_world(world, rooms, config)

    overlaps: list[tuple[str, str]] = []
    for level in world.levels.values():
        level_zones = [(str(zone.name), Polygon([(float(c.x), float(c.y)) for c in zone.corners])) for zone in level.zones]
        for index, (first_name, first) in enumerate(level_zones):
            for second_name, second in level_zones[index + 1 :]:
                if first.intersection(second).area > 1e-6:
                    overlaps.append((first_name, second_name))

    traversable = 0
    uncovered = 0
    examples: list[tuple[float, float]] = []
    map_issues: list[str] = []
    for level_id in sorted(world.levels):
        level_zone_polygons = tuple(Polygon([(float(corner.x), float(corner.y)) for corner in level_zone.corners]) for level_zone in world.levels[level_id].zones)
        level_root = Path(world_view.path) / str(level_id)
        map_yaml = level_root / "map.yaml"
        if not map_yaml.exists():
            map_issues.append(f"level {level_id}: missing map.yaml")
            continue
        config = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
        image_path = level_root / str(config["image"])
        if not image_path.exists():
            map_issues.append(f"level {level_id}: missing map image {image_path.name!r}")
            continue
        opened = Image.open(image_path)
        alpha: np.ndarray | None = None
        if "transparency" in opened.info or opened.mode == "RGBA":
            rgba = opened.convert("RGBA")
            alpha = np.asarray(rgba.getchannel("A"))
            opened = rgba
        image = np.asarray(opened.convert("L"))
        resolution = float(config["resolution"])
        origin_x, origin_y, _ = map(float, config.get("origin", (0, 0, 0)))
        free_threshold = float(config.get("free_thresh", 0.196))
        negate = bool(config.get("negate", 0))

        for row in range(0, image.shape[0], stride_cells):
            for column in range(0, image.shape[1], stride_cells):
                if alpha is not None and alpha[row, column] < 255:
                    continue
                intensity = float(image[row, column]) / 255.0
                occupancy = intensity if negate else 1.0 - intensity
                if occupancy >= free_threshold:
                    continue
                x = origin_x + (column + 0.5) * resolution
                y = origin_y + (image.shape[0] - row - 0.5) * resolution
                traversable += 1
                if not any(polygon.covers(Point(x, y)) for polygon in level_zone_polygons):
                    uncovered += 1
                    if len(examples) < 8:
                        examples.append((x, y))

    return AcousticWorldAudit(
        world_name=world_name,
        rooms=len(rooms),
        door_portals=sum(p.portal_kind == "door" for p in graph.portals),
        opening_portals=sum(p.portal_kind == "opening" for p in graph.portals),
        connected_components=len(graph.connected_components()),
        unpaired_doors=len(graph.unpaired_doors),
        overlapping_zone_pairs=tuple(overlaps),
        sampled_traversable_cells=traversable,
        uncovered_traversable_cells=uncovered,
        uncovered_examples=tuple(examples),
        map_issues=tuple(map_issues),
    )


def static_room(corners_xy: Sequence[tuple[float, float]], *, ceiling_height_m: float, wall_material_id: str, floor_material_id: str, ceiling_material_id: str) -> AcousticRoomSpec:
    """A closed room from counter-clockwise corners with one wall material."""
    if len(corners_xy) < 3:
        raise ValueError("a room needs at least three corners")
    corners = tuple(corners_xy)
    boundary = tuple(AcousticBoundarySpec(start=start, end=corners[(index + 1) % len(corners)], material_id=wall_material_id, kind="wall") for index, start in enumerate(corners))
    return AcousticRoomSpec(
        zone_name="static_room",
        boundary=boundary,
        floor_material_id=floor_material_id,
        ceiling_material_id=ceiling_material_id,
        ceiling_height_m=ceiling_height_m,
    )


def plot_room(args: argparse.Namespace) -> int:
    corners = tuple(args.room_corners)
    room = static_room(
        tuple(zip(corners[::2], corners[1::2], strict=True)),
        ceiling_height_m=args.room_height,
        wall_material_id=args.wall_material,
        floor_material_id=args.floor_material,
        ceiling_material_id=args.ceiling_material,
    )
    rir_config = configure(RirConfig, RirGroup, max_order=args.max_order)
    source = tuple(args.source)
    listener = tuple(args.listener)
    rir = PyroomacousticsAdapter(default_catalog(), rir_config).compute_rir(room, source_position_m=source, listener_position_m=listener)
    dashboard = AcousticPlotDashboard(energy_bin_ms=VizGroup.PLOT_ENERGY_BIN_MS.default, early_window_s=VizGroup.PLOT_EARLY_WINDOW_S.default, interactive=False)
    dashboard.update(
        AcousticPlotSnapshot.from_rir(
            rir,
            room_specs=(room,),
            source_position_m=source,
            listener_position_m=listener,
            backend="pyroomacoustics_static",
            source_zone=room.zone_name,
            listener_zone=room.zone_name,
            traversed_zones=(room.zone_name,),
            label="fixed source and listener",
        )
    )
    if args.out is not None:
        dashboard.save(str(args.out))
        print(f"wrote {args.out}")
    else:
        dashboard.show()
    return 0


def audit(names: Sequence[str], *, stride_cells: int) -> int:
    root = Path(get_package_share_directory("arena_simulation_setup")) / "worlds"
    names = list(names) or sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    failed = False
    for name in names:
        report = audit_world(name, stride_cells=stride_cells)
        status = "PASS" if report.complete else "INCOMPLETE"
        failed |= not report.complete
        print(f"{status:10} {name:28} rooms={report.rooms:3} doors={report.door_portals:3} openings={report.opening_portals:3} components={report.connected_components:2} unpaired={report.unpaired_doors:3} uncovered={report.uncovered_traversable_cells}/{report.sampled_traversable_cells}")
        if report.uncovered_examples:
            print(f"  uncovered examples: {report.uncovered_examples}")
        if report.overlapping_zone_pairs:
            print(f"  overlapping zones: {report.overlapping_zone_pairs}")
        if report.map_issues:
            print(f"  map issues: {report.map_issues}")
    return 1 if failed else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="acoustic_world_audit",
        description=(
            "Audit Arena worlds for acoustic coverage and connectivity, or plot the pyroomacoustics RIR of one static room. "
            "Audit mode (default) prints one PASS or INCOMPLETE line per world: rooms, door and opening portals, connected components, "
            "unpaired doors, and how many sampled free map cells lie outside every acoustic zone. "
            "Plot mode (--plot-room) builds one room from --room-corners and --room-height, computes the RIR from --source to --listener "
            "and shows the waveform, Schroeder decay and binned energy next to the 3-D geometry."
        ),
        epilog="Exit status: 0 when every audited world passes or the plot was drawn, 1 when a world is incomplete, 2 on a usage error.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    result.add_argument("worlds", nargs="*", help="world names to audit, empty audits every world under arena_simulation_setup/worlds")
    result.add_argument("--stride-cells", type=int, default=WorldGroup.COVERAGE_STRIDE_CELLS.default, help="audit every n-th map cell in each direction")
    plot = result.add_argument_group("plot mode")
    plot.add_argument("--plot-room", action="store_true", help="plot one static room RIR instead of auditing worlds, world names are then ignored")
    plot.add_argument("--room-corners", type=float, nargs="+", default=list(DEFAULT_ROOM_CORNERS), metavar="X Y", help="counter-clockwise corner coordinates in meters, at least three X Y pairs")
    plot.add_argument("--room-height", type=float, default=WorldGroup.CEILING_HEIGHT_M.default, help="ceiling height in meters")
    plot.add_argument("--source", type=float, nargs=3, default=list(DEFAULT_SOURCE), metavar=("X", "Y", "Z"), help="source position in meters")
    plot.add_argument("--listener", type=float, nargs=3, default=list(DEFAULT_LISTENER), metavar=("X", "Y", "Z"), help="listener position in meters")
    plot.add_argument("--wall-material", default="Acoustic_Default_Wall", help="acoustic_materials.yaml id of every wall")
    plot.add_argument("--floor-material", default="Acoustic_Default_Floor", help="acoustic_materials.yaml id of the floor")
    plot.add_argument("--ceiling-material", default="Acoustic_Default_Ceiling", help="acoustic_materials.yaml id of the ceiling")
    plot.add_argument("--max-order", type=int, default=RirGroup.MAX_ORDER.default, help="image-source reflection order")
    plot.add_argument("--out", type=Path, default=None, help="write the figure to this image file instead of opening a window")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    cli = parser()
    args = cli.parse_args(argv)
    if args.stride_cells <= 0:
        cli.error("--stride-cells must be positive")
    if args.plot_room and (len(args.room_corners) < 6 or len(args.room_corners) % 2):
        cli.error("--room-corners needs at least three X Y pairs")
    raise SystemExit(plot_room(args) if args.plot_room else audit(args.worlds, stride_cells=args.stride_cells))


if __name__ == "__main__":
    main()
