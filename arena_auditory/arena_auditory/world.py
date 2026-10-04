"""Acoustic world: zones, walls, rooms, portals and microphones of an Arena world, realized in the runtime map frame."""

from __future__ import annotations

import hashlib
import math
import subprocess
import typing
from collections.abc import Iterator
from pathlib import Path

import attrs
import numpy as np
import shapely
import shapely.errors
import yaml
from arena_simulation_setup.tree.World import (
    MICROPHONE_PLACEMENT_TOLERANCE_M,
    LevelDescription,
    MultiLevelWorldView,
    WorldDescription,
    WorldIdentifier,
)
from shapely.affinity import translate

from arena_auditory.params import MapSource
from arena_auditory.rooms import (
    AcousticPortal,
    AcousticRoomSpec,
    AcousticRoomSpecBuilder,
    AcousticWorldGraph,
    UnpairedDoor,
    world_zones,
)
from arena_auditory.shared import ListenerId, Vec3

if typing.TYPE_CHECKING:
    from nav_msgs.msg import OccupancyGrid

Offset2D = tuple[float, float]

LOAD_ERRORS = (OSError, LookupError, ValueError, TypeError, yaml.YAMLError, shapely.errors.ShapelyError, subprocess.SubprocessError)
ZONE_LOOKUP_TOLERANCE_M = 0.35


@attrs.frozen
class AcousticWall:
    start: tuple[float, float]
    end: tuple[float, float]
    material_id: str
    geometry: shapely.LineString = attrs.field(
        init=False,
        eq=False,
        repr=False,
        default=attrs.Factory(lambda wall: shapely.LineString([wall.start, wall.end]), takes_self=True),
    )


@attrs.frozen
class AcousticZone:
    name: str
    polygon: shapely.Polygon
    floor_material_id: str


@attrs.frozen
class AcousticScene:
    zones: tuple[AcousticZone, ...]
    walls: tuple[AcousticWall, ...]
    ceiling_height_m: float
    _wall_geometries: np.ndarray = attrs.field(
        init=False,
        eq=False,
        repr=False,
        default=attrs.Factory(lambda scene: np.asarray([wall.geometry for wall in scene.walls], dtype=object), takes_self=True),
    )

    @classmethod
    def from_world(cls, world: WorldDescription | LevelDescription, *, ceiling_height_m: float) -> AcousticScene:
        zones = []
        walls = []

        for zone in world_zones(world):
            polygon = shapely.Polygon([(corner.x, corner.y) for corner in zone.corners])
            zones.append(AcousticZone(name=zone.name, polygon=polygon, floor_material_id=zone.material.name))

            for wall in zone.walls:
                material_id = wall.material.name if wall.material is not None else "default"
                walls.append(AcousticWall(start=(wall.start.x, wall.start.y), end=(wall.end.x, wall.end.y), material_id=material_id))

        return cls(zones=tuple(zones), walls=tuple(walls), ceiling_height_m=ceiling_height_m)

    def zone_at_xy(self, x: float, y: float) -> AcousticZone | None:
        """Exact containment wins, else the nearest zone within ZONE_LOOKUP_TOLERANCE_M."""
        candidate = shapely.Point(float(x), float(y))
        exact = next(
            (zone for zone in self.zones if zone.polygon.covers(candidate)),
            None,
        )
        if exact is not None:
            return exact

        nearby = [(zone.polygon.distance(candidate), index, zone) for index, zone in enumerate(self.zones) if zone.polygon.distance(candidate) <= ZONE_LOOKUP_TOLERANCE_M]
        return min(nearby, default=(0.0, 0, None))[2]

    def intersecting_walls(self, source: Vec3, listener: Vec3) -> list[AcousticWall]:
        if not self.walls:
            return []
        path = shapely.LineString([(source[0], source[1]), (listener[0], listener[1])])
        return [self.walls[index] for index in np.flatnonzero(shapely.crosses(path, self._wall_geometries))]


def compact_authored_world(
    world_view: MultiLevelWorldView,
    world_description: WorldDescription,
    *,
    disk_map: bool = False,
) -> tuple[LevelDescription, Offset2D | None]:
    """Compact a world onto one grid the way the runtime does and return it with the authored map origin."""
    level_origins = world_view.level_origins()
    world_description.apply_elevator_door_sides()
    compacted = world_description.compact_world(level_origins if level_origins is not None else dict.fromkeys(world_description.levels, (0.0, 0.0)))
    if disk_map:
        for level_id in sorted(world_description.levels):
            map_yaml = Path(world_view.path) / str(level_id) / "map.yaml"
            if map_yaml.exists():
                origin = yaml.safe_load(map_yaml.read_text(encoding="utf-8")).get("origin", (0.0, 0.0, 0.0))
                return compacted, (float(origin[0]), float(origin[1]))
        return compacted, None
    if not compacted.zones:
        return compacted, None
    _, origin = compacted.render_grid()
    return compacted, (float(origin[0]), float(origin[1]))


def _translate_xy(point: tuple[float, float], dx: float, dy: float) -> tuple[float, float]:
    return point[0] + dx, point[1] + dy


def _translate_room(room: AcousticRoomSpec, dx: float, dy: float) -> AcousticRoomSpec:
    return attrs.evolve(
        room,
        boundary=tuple(
            attrs.evolve(
                boundary,
                start=_translate_xy(boundary.start, dx, dy),
                end=_translate_xy(boundary.end, dx, dy),
            )
            for boundary in room.boundary
        ),
    )


def _translate_wall(wall: AcousticWall, dx: float, dy: float) -> AcousticWall:
    return attrs.evolve(wall, start=_translate_xy(wall.start, dx, dy), end=_translate_xy(wall.end, dx, dy))


def _translate_portal(portal: AcousticPortal, dx: float, dy: float) -> AcousticPortal:
    return attrs.evolve(portal, start=_translate_xy(portal.start, dx, dy), end=_translate_xy(portal.end, dx, dy))


def _translate_unpaired_door(door: UnpairedDoor, dx: float, dy: float) -> UnpairedDoor:
    return attrs.evolve(door, start=_translate_xy(door.start, dx, dy), end=_translate_xy(door.end, dx, dy))


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"microphone {field} must be a string")
    result = value.strip()
    if not result or ":" in result:
        raise ValueError(f"microphone {field} must be non-empty and contain no ':'")
    return result


@attrs.frozen
class RobotMicrophoneSpec:
    robot: str
    placement: str
    frame: str
    index: int

    @property
    def listener_id(self) -> str:
        return ListenerId.robot_mic(self.robot, self.placement, self.index)

    def resolve_frame(self, robot_frame_prefix: str) -> str:
        prefix = robot_frame_prefix.strip("/")
        frame = self.frame.strip("/")
        if prefix and (frame == prefix or frame.startswith(f"{prefix}/")):
            return frame
        return "/".join(part for part in (prefix, frame) if part)


@attrs.frozen
class WorldMicrophoneSpec:
    listener_id: str
    zone: str
    placement: str
    frame: str
    position: Vec3
    ceiling_height_m: float | None


def parse_robot_microphones(raw: str) -> tuple[RobotMicrophoneSpec, ...]:
    """Parse the microphones param, a YAML list of {owner: robot, robot, placement, frame, index}. Raises ValueError."""
    configured = yaml.safe_load(raw) if raw.strip() else []
    if configured is None:
        configured = []
    if not isinstance(configured, list):
        raise ValueError("microphones must be a YAML list")

    specs: list[RobotMicrophoneSpec] = []
    listener_ids: set[str] = set()
    for item in configured:
        if not isinstance(item, dict):
            raise ValueError("each robot microphone must be a mapping")
        owner = _identifier(item.get("owner", "robot"), "owner")
        if owner != "robot":
            raise ValueError("launch-configured microphones must use owner: robot")
        robot = _identifier(item.get("robot"), "robot")
        placement = _identifier(item.get("placement"), "placement").lower()
        frame = item.get("frame")
        if not isinstance(frame, str) or not frame.strip().strip("/"):
            raise ValueError(f"robot microphone {robot!r}/{placement!r} requires a TF frame")
        index = item.get("index", 1)
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError("microphone index must be a positive integer")
        spec = RobotMicrophoneSpec(robot=robot, placement=placement, frame=frame.strip(), index=index)
        if spec.listener_id in listener_ids:
            raise ValueError(f"duplicate microphone {spec.listener_id!r}")
        listener_ids.add(spec.listener_id)
        specs.append(spec)
    return tuple(specs)


def world_microphones(
    world: WorldDescription,
    ceiling_height_m: float,
    level_origins: dict[str, tuple[float, float]] | None = None,
) -> tuple[WorldMicrophoneSpec, ...]:
    levels = world.levels
    specs = []
    for level_id in sorted(levels):
        zones = {zone.name: zone for zone in levels[level_id].zones}
        for microphone in levels[level_id].microphones:
            zone = zones[microphone.zone]
            frame = microphone.frame.strip().strip("/")
            offset = level_origins.get(str(level_id), (0.0, 0.0)) if level_origins is not None and frame == "map" else (0.0, 0.0)
            ceiling_height = None
            if microphone.placement == "ceiling":
                ceiling_height = float(zone.ceiling_height) if zone.ceiling_height is not None else float(ceiling_height_m)
                if frame == "map" and not math.isclose(float(microphone.position.z), ceiling_height, abs_tol=MICROPHONE_PLACEMENT_TOLERANCE_M):
                    raise ValueError(f"microphone {microphone.listener_id!r} z={microphone.position.z} does not match ceiling height {ceiling_height}")
            specs.append(
                WorldMicrophoneSpec(
                    listener_id=microphone.listener_id,
                    zone=microphone.zone,
                    placement=microphone.placement,
                    frame=frame,
                    position=(
                        float(microphone.position.x) + offset[0],
                        float(microphone.position.y) + offset[1],
                        float(microphone.position.z),
                    ),
                    ceiling_height_m=ceiling_height,
                )
            )
    return tuple(specs)


def _yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))


@attrs.frozen
class OccupancyMap:
    frame_id: str
    resolution_m: float
    origin_xy: tuple[float, float]
    origin_yaw: float
    data: np.ndarray = attrs.field(eq=False)
    occupied_threshold: int

    @classmethod
    def from_msg(cls, msg: OccupancyGrid, occupied_threshold: int) -> OccupancyMap:
        info = msg.info
        orientation = info.origin.orientation
        return cls(
            frame_id=str(msg.header.frame_id).strip(),
            resolution_m=float(info.resolution),
            origin_xy=(float(info.origin.position.x), float(info.origin.position.y)),
            origin_yaw=_yaw(orientation.x, orientation.y, orientation.z, orientation.w),
            data=np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width),
            occupied_threshold=int(occupied_threshold),
        )

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        """Grid column and row of a map-frame point, None outside the grid."""
        dx = x - self.origin_xy[0]
        dy = y - self.origin_xy[1]
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        column = math.floor(local_x / self.resolution_m + 1e-9)
        row = math.floor(local_y / self.resolution_m + 1e-9)
        if column < 0 or row < 0 or column >= self.width or row >= self.height:
            return None
        return column, row

    def occluded(self, a: Vec3, b: Vec3) -> bool:
        """True when the Bresenham line between two in-grid points crosses an occupied cell."""
        start = self.cell(a[0], a[1])
        end = self.cell(b[0], b[1])
        if start is None or end is None:
            return False
        return any(self.data[row, column] >= self.occupied_threshold for column, row in _bresenham(*start, *end))

    def free_points(self, stride: int) -> np.ndarray:
        """Map-frame centers of every stride-th free cell, shape (n, 2)."""
        grid = self.data.astype(np.int16)
        grid_y, grid_x = np.mgrid[0 : self.height : stride, 0 : self.width : stride]
        values = grid[grid_y, grid_x]
        free = (values >= 0) & (values < self.occupied_threshold)
        local_x = (grid_x[free] + 0.5) * self.resolution_m
        local_y = (grid_y[free] + 0.5) * self.resolution_m
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        world_x = self.origin_xy[0] + cos_yaw * local_x - sin_yaw * local_y
        world_y = self.origin_xy[1] + sin_yaw * local_x + cos_yaw * local_y
        return np.column_stack((world_x, world_y))


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> Iterator[tuple[int, int]]:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


@attrs.frozen
class CoverageReport:
    sampled: int
    uncovered: tuple[tuple[float, float], ...]

    @property
    def complete(self) -> bool:
        return not self.uncovered


@attrs.frozen
class WorldConfig:
    """World geometry settings, world.* and the portal.* parameters that shape the graph."""

    ceiling_height_m: float
    map_source: MapSource
    adjacency_tolerance_m: float
    openings_enabled: bool
    min_opening_width_m: float
    door_loss_db: float
    opening_loss_db: float

    @property
    def digest(self) -> str:
        return hashlib.blake2b(repr(attrs.astuple(self)).encode(), digest_size=8).hexdigest()


@attrs.frozen
class AcousticWorld:
    """Acoustic geometry of one world, translated by offset from the authored into the runtime map frame."""

    name: str
    path: Path
    config: WorldConfig
    scene: AcousticScene
    rooms: tuple[AcousticRoomSpec, ...]
    graph: AcousticWorldGraph
    microphones: tuple[WorldMicrophoneSpec, ...]
    authored_origin: Offset2D | None
    offset: Offset2D = (0.0, 0.0)

    @classmethod
    def load(cls, world: str, config: WorldConfig) -> AcousticWorld:
        """Resolve and build a world in its authored frame. Raises one of LOAD_ERRORS."""
        world_view = WorldIdentifier(world).resolve_sync()
        description = world_view.load()
        microphones = world_microphones(description, config.ceiling_height_m, level_origins=world_view.level_origins())
        compacted, authored_origin = compact_authored_world(world_view, description, disk_map=config.map_source is MapSource.DISK)
        rooms = AcousticRoomSpecBuilder(config.ceiling_height_m).from_world(compacted)
        graph = AcousticWorldGraph.from_world(compacted, rooms, config)
        return cls(
            name=world,
            path=Path(world_view.path),
            config=config,
            scene=AcousticScene.from_world(compacted, ceiling_height_m=config.ceiling_height_m),
            rooms=rooms,
            graph=graph,
            microphones=microphones,
            authored_origin=authored_origin,
        )

    def realized(self, map_origin_xy: tuple[float, float]) -> AcousticWorld:
        """Translate into the frame of a runtime map with this origin. Raises ValueError when zones exist without an authored origin."""
        if self.authored_origin is None:
            if self.scene.zones:
                raise ValueError(f"cannot realize acoustic world {self.name!r}: no authored map origin (debug.map_source=disk without a level map.yaml)")
            return self
        offset = (float(map_origin_xy[0]) - self.authored_origin[0], float(map_origin_xy[1]) - self.authored_origin[1])
        dx, dy = offset[0] - self.offset[0], offset[1] - self.offset[1]
        rooms = tuple(_translate_room(room, dx, dy) for room in self.rooms)
        return attrs.evolve(
            self,
            scene=attrs.evolve(
                self.scene,
                zones=tuple(attrs.evolve(zone, polygon=translate(zone.polygon, xoff=dx, yoff=dy)) for zone in self.scene.zones),
                walls=tuple(_translate_wall(wall, dx, dy) for wall in self.scene.walls),
            ),
            rooms=rooms,
            graph=attrs.evolve(
                self.graph,
                rooms=rooms,
                portals=tuple(_translate_portal(portal, dx, dy) for portal in self.graph.portals),
                zone_polygons=tuple((name, translate(polygon, xoff=dx, yoff=dy)) for name, polygon in self.graph.zone_polygons),
                unpaired_doors=tuple(_translate_unpaired_door(door, dx, dy) for door in self.graph.unpaired_doors),
            ),
            microphones=tuple(
                attrs.evolve(microphone, position=(microphone.position[0] + dx, microphone.position[1] + dy, microphone.position[2])) if microphone.frame == "map" else microphone
                for microphone in self.microphones
            ),
            offset=offset,
        )

    def zone_at(self, x: float, y: float) -> AcousticZone | None:
        return self.scene.zone_at_xy(x, y)

    def zone_named(self, name: str) -> AcousticZone | None:
        return next((zone for zone in self.scene.zones if zone.name == name), None)

    def room(self, zone: str) -> AcousticRoomSpec | None:
        return self.graph.room(zone)

    def floor_material(self, x: float, y: float) -> str:
        """Floor material id of the zone at x, y, empty outside every zone."""
        zone = self.zone_at(x, y)
        return zone.floor_material_id if zone is not None else ""

    def coverage(self, occupancy: OccupancyMap, *, stride_cells: int, tolerance_m: float) -> CoverageReport:
        """Free map cells (every stride_cells-th) farther than tolerance_m from every zone."""
        points = occupancy.free_points(max(int(stride_cells), 1))
        if not self.scene.zones:
            return CoverageReport(sampled=len(points), uncovered=tuple((float(x), float(y)) for x, y in points))
        covered = shapely.union_all([zone.polygon for zone in self.scene.zones]).buffer(max(float(tolerance_m), 0.0))
        shapely.prepare(covered)
        outside = ~shapely.intersects_xy(covered, points[:, 0], points[:, 1])
        return CoverageReport(sampled=len(points), uncovered=tuple((float(x), float(y)) for x, y in points[outside]))

    @property
    def signature(self) -> str:
        """Name, offset and config digest, part of every rir_key."""
        return hashlib.blake2b(repr((self.name, self.offset, self.config.digest)).encode(), digest_size=16).hexdigest()
