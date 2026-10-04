"""Acoustic rooms built from world zones and the portal graph that connects them."""

from __future__ import annotations

import heapq
import itertools
import math
import re
from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal

import attrs
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.World import LevelDescription, WorldDescription
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points

if TYPE_CHECKING:
    from arena_auditory.world import WorldConfig

Point2D = tuple[float, float]
Position3D = tuple[float, float, float]

DEFAULT_WALL_MATERIAL_ID = "Acoustic_Default_Wall"
DEFAULT_FLOOR_MATERIAL_ID = "Acoustic_Default_Floor"
OPENING_MATERIAL_ID = "Acoustic_Open"
GEOMETRY_TOLERANCE_M = 1e-5


def world_zones(world: WorldDescription | LevelDescription) -> tuple[LevelDescription.Zone, ...]:
    """Every zone of the world, ordered by level id."""
    return tuple(zone for _, zones in world_zone_groups(world) for zone in zones)


def world_zone_groups(
    world: WorldDescription | LevelDescription,
) -> tuple[tuple[str, tuple[LevelDescription.Zone, ...]], ...]:
    """Level-scoped zones, so overlapping building floors stay uncoupled."""
    if isinstance(world, LevelDescription):
        return (("0", tuple(world.zones)),)
    return tuple((level_id, tuple(level.zones)) for level_id, level in sorted(world.levels.items()))


BoundaryKind = Literal["wall", "door", "opening"]


@attrs.frozen
class AcousticBoundarySpec:
    start: Point2D
    end: Point2D
    material_id: str
    kind: BoundaryKind
    height_m: float | None = None


@attrs.frozen
class AcousticRoomSpec:
    zone_name: str
    boundary: tuple[AcousticBoundarySpec, ...]

    floor_material_id: str
    ceiling_material_id: str
    ceiling_height_m: float

    @property
    def corners_xy(self) -> tuple[Point2D, ...]:
        return tuple(segment.start for segment in self.boundary)

    @property
    def boundary_material_ids(self) -> tuple[str, ...]:
        return tuple(segment.material_id for segment in self.boundary)


def _is_finite_point(point: Point2D) -> bool:
    return math.isfinite(point[0]) and math.isfinite(point[1])


def _normalize_polygon(zone: LevelDescription.Zone) -> Polygon:
    """Validate and orient an Arena zone polygon counter-clockwise."""

    coordinates = [(float(corner.x), float(corner.y)) for corner in zone.corners]

    if len(coordinates) < 3:
        raise ValueError(f"zone {zone.name!r} has fewer than three corners")

    if any(not _is_finite_point(coordinate) for coordinate in coordinates):
        raise ValueError(f"zone {zone.name!r} contains non-finite coordinates")

    polygon = Polygon(coordinates)

    if polygon.is_empty:
        raise ValueError(f"zone {zone.name!r} has an empty polygon")

    if not polygon.is_valid:
        raise ValueError(f"zone {zone.name!r} has invalid polygon geometry")

    if polygon.area <= 1e-8:
        raise ValueError(f"zone {zone.name!r} has zero or negligible area")

    return orient(polygon, sign=1.0)


@attrs.frozen
class _SourceSegment:
    start: Point2D
    end: Point2D
    material_id: str
    kind: BoundaryKind
    height_m: float | None = None


def _projection_parameter(
    point: Point2D,
    edge_start: Point2D,
    edge_end: Point2D,
) -> float:
    px, py = point
    ax, ay = edge_start
    bx, by = edge_end

    dx = bx - ax
    dy = by - ay
    length_squared = dx * dx + dy * dy

    if length_squared <= 1e-12:
        raise ValueError("cannot project onto a zero-length edge")

    return ((px - ax) * dx + (py - ay) * dy) / length_squared


def _point_on_edge(
    point: Point2D,
    edge_start: Point2D,
    edge_end: Point2D,
    tolerance: float,
) -> bool:
    line = LineString([edge_start, edge_end])

    if line.distance(Point(point)) > tolerance:
        return False

    parameter = _projection_parameter(
        point,
        edge_start,
        edge_end,
    )

    return -tolerance <= parameter <= 1.0 + tolerance


def _interpolate(
    start: Point2D,
    end: Point2D,
    parameter: float,
) -> Point2D:
    return (
        start[0] + parameter * (end[0] - start[0]),
        start[1] + parameter * (end[1] - start[1]),
    )


def _points_close(
    first: Point2D,
    second: Point2D,
    tolerance: float,
) -> bool:
    return (
        math.hypot(
            first[0] - second[0],
            first[1] - second[1],
        )
        <= tolerance
    )


def _material_name(
    identifier: MaterialIdentifier | None,
    fallback: str,
) -> str:
    if identifier is None:
        return fallback

    name = str(identifier.name).strip()
    return name or fallback


def _segment_covers_point(
    segment: _SourceSegment,
    point: Point2D,
    tolerance: float,
) -> bool:
    return _point_on_edge(
        point,
        segment.start,
        segment.end,
        tolerance,
    )


def _classify_subsegment(
    midpoint: Point2D,
    doors: list[_SourceSegment],
    walls: list[_SourceSegment],
) -> tuple[str, BoundaryKind, float | None]:

    matching_doors = [
        segment
        for segment in doors
        if _segment_covers_point(
            segment,
            midpoint,
            GEOMETRY_TOLERANCE_M,
        )
    ]

    if matching_doors:
        door = matching_doors[0]
        return door.material_id, "door", door.height_m

    matching_walls = [
        segment
        for segment in walls
        if _segment_covers_point(
            segment,
            midpoint,
            GEOMETRY_TOLERANCE_M,
        )
    ]

    if matching_walls:
        wall = matching_walls[0]
        return wall.material_id, "wall", wall.height_m

    return OPENING_MATERIAL_ID, "opening", None


def _split_polygon_edge(
    *,
    edge_start: Point2D,
    edge_end: Point2D,
    walls: list[_SourceSegment],
    doors: list[_SourceSegment],
) -> list[AcousticBoundarySpec]:

    breakpoints = {0.0, 1.0}

    for segment in [*walls, *doors]:
        for endpoint in (segment.start, segment.end):
            if not _point_on_edge(
                endpoint,
                edge_start,
                edge_end,
                GEOMETRY_TOLERANCE_M,
            ):
                continue

            parameter = _projection_parameter(
                endpoint,
                edge_start,
                edge_end,
            )

            breakpoints.add(min(max(parameter, 0.0), 1.0))

    ordered = sorted(breakpoints)
    output: list[AcousticBoundarySpec] = []

    for start_parameter, end_parameter in zip(
        ordered,
        ordered[1:],
        strict=False,
    ):
        if end_parameter - start_parameter <= 1e-9:
            continue

        start = _interpolate(
            edge_start,
            edge_end,
            start_parameter,
        )
        end = _interpolate(
            edge_start,
            edge_end,
            end_parameter,
        )
        midpoint = _interpolate(
            edge_start,
            edge_end,
            (start_parameter + end_parameter) / 2.0,
        )

        material_id, kind, height_m = _classify_subsegment(
            midpoint,
            doors,
            walls,
        )

        output.append(
            AcousticBoundarySpec(
                start=start,
                end=end,
                material_id=material_id,
                kind=kind,
                height_m=height_m,
            )
        )

    return output


def _validate_spec(
    spec: AcousticRoomSpec,
    *,
    tolerance: float,
) -> None:
    if len(spec.boundary) < 3:
        raise ValueError(f"room {spec.zone_name!r} has fewer than three boundary segments")

    if spec.ceiling_height_m <= 0.0:
        raise ValueError(f"room {spec.zone_name!r} has a non-positive ceiling height")

    if not spec.floor_material_id:
        raise ValueError("floor material ID cannot be empty")

    if not spec.ceiling_material_id:
        raise ValueError("ceiling material ID cannot be empty")

    for index, segment in enumerate(spec.boundary):
        following = spec.boundary[(index + 1) % len(spec.boundary)]

        if _points_close(
            segment.start,
            segment.end,
            tolerance,
        ):
            raise ValueError(f"room {spec.zone_name!r} contains a zero-length boundary segment")

        if not _points_close(
            segment.end,
            following.start,
            tolerance,
        ):
            raise ValueError(f"room {spec.zone_name!r} has a discontinuous boundary")

        if not segment.material_id:
            raise ValueError(f"room {spec.zone_name!r} contains an empty material ID")


class AcousticRoomSpecBuilder:
    def __init__(
        self,
        ceiling_height_m: float,
    ) -> None:
        self._ceiling_height_m = ceiling_height_m

    def from_world(
        self,
        world: WorldDescription | LevelDescription,
    ) -> tuple[AcousticRoomSpec, ...]:
        return tuple(self._from_zone(zone) for zone in world_zones(world))

    def _normalize_walls(self, zone: LevelDescription.Zone) -> list[_SourceSegment]:
        return [
            _SourceSegment(
                start=(
                    float(wall.start.x),
                    float(wall.start.y),
                ),
                end=(
                    float(wall.end.x),
                    float(wall.end.y),
                ),
                material_id=_material_name(
                    wall.material,
                    DEFAULT_WALL_MATERIAL_ID,
                ),
                kind="wall",
            )
            for wall in zone.walls
        ]

    def _normalize_doors(self, zone: LevelDescription.Zone) -> list[_SourceSegment]:
        return [
            _SourceSegment(
                start=(
                    float(door.start.x),
                    float(door.start.y),
                ),
                end=(
                    float(door.end.x),
                    float(door.end.y),
                ),
                material_id=_material_name(
                    door.material,
                    DEFAULT_WALL_MATERIAL_ID,
                ),
                kind="door",
                height_m=float(door.height),
            )
            for door in zone.doors
        ]

    def _from_zone(self, zone: LevelDescription.Zone) -> AcousticRoomSpec:
        polygon = _normalize_polygon(zone)

        corners = list(polygon.exterior.coords)[:-1]

        walls = self._normalize_walls(zone)
        doors = self._normalize_doors(zone)

        boundary: list[AcousticBoundarySpec] = []

        for index, edge_start in enumerate(corners):
            edge_end = corners[(index + 1) % len(corners)]

            boundary.extend(
                _split_polygon_edge(
                    edge_start=edge_start,
                    edge_end=edge_end,
                    walls=walls,
                    doors=doors,
                )
            )

        floor_material_id = _material_name(
            zone.material,
            DEFAULT_FLOOR_MATERIAL_ID,
        )

        ceiling_material_id = str(zone.ceiling_material.name).strip() if zone.ceiling else ""

        spec = AcousticRoomSpec(
            zone_name=str(zone.name),
            boundary=tuple(boundary),
            floor_material_id=floor_material_id,
            ceiling_material_id=ceiling_material_id or OPENING_MATERIAL_ID,
            ceiling_height_m=float(zone.ceiling_height) if zone.ceiling_height is not None else self._ceiling_height_m,
        )

        _validate_spec(
            spec,
            tolerance=GEOMETRY_TOLERANCE_M,
        )

        return spec


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())


@attrs.frozen
class UnpairedDoor:
    door_name: str
    owner_zone: str
    start: Point2D
    end: Point2D
    reason: str


@attrs.frozen
class AcousticPortal:
    portal_id: str
    door_name: str
    zone_a: str
    zone_b: str
    start: Point2D
    end: Point2D
    height_m: float
    material_id: str
    portal_kind: Literal["door", "opening"] = "door"
    loss_db: float | None = None

    @property
    def center_xy(self) -> Point2D:
        return (
            0.5 * (self.start[0] + self.end[0]),
            0.5 * (self.start[1] + self.end[1]),
        )

    def connects(self, first_zone: str, second_zone: str) -> bool:
        return {first_zone, second_zone} == {self.zone_a, self.zone_b}

    def other_zone(self, zone_name: str) -> str:
        if zone_name == self.zone_a:
            return self.zone_b
        if zone_name == self.zone_b:
            return self.zone_a
        raise KeyError(f"portal {self.portal_id!r} is not connected to {zone_name!r}")


@attrs.frozen
class AcousticPortalRoute:
    zones: tuple[str, ...]
    portals: tuple[AcousticPortal, ...]

    @property
    def hop_count(self) -> int:
        return len(self.portals)


def _portal_adjacency(portals: tuple[AcousticPortal, ...]) -> dict[str, tuple[AcousticPortal, ...]]:
    adjacency: dict[str, list[AcousticPortal]] = {}
    for portal in portals:
        adjacency.setdefault(portal.zone_a, []).append(portal)
        adjacency.setdefault(portal.zone_b, []).append(portal)
    return {zone: tuple(zone_portals) for zone, zone_portals in adjacency.items()}


@attrs.frozen
class AcousticWorldGraph:
    """Acoustic rooms connected by authored doors and shared openings."""

    rooms: tuple[AcousticRoomSpec, ...]
    portals: tuple[AcousticPortal, ...]
    zone_polygons: tuple[tuple[str, Polygon], ...]
    unpaired_doors: tuple[UnpairedDoor, ...] = ()
    _adjacency: dict[str, tuple[AcousticPortal, ...]] = attrs.field(
        init=False,
        eq=False,
        repr=False,
        default=attrs.Factory(lambda graph: _portal_adjacency(graph.portals), takes_self=True),
    )

    @classmethod
    def from_world(
        cls,
        world: WorldDescription | LevelDescription,
        rooms: tuple[AcousticRoomSpec, ...],
        config: WorldConfig,
    ) -> AcousticWorldGraph:
        zones = world_zones(world)
        zone_levels = {str(zone.name): level_id for level_id, level_zones in world_zone_groups(world) for zone in level_zones}
        polygons = tuple(
            (
                str(zone.name),
                Polygon([(float(corner.x), float(corner.y)) for corner in zone.corners]),
            )
            for zone in zones
        )
        room_names = {room.zone_name for room in rooms}
        portals: list[AcousticPortal] = []
        unpaired: list[UnpairedDoor] = []
        seen: set[tuple[str, str, tuple[float, ...]]] = set()

        for zone in zones:
            owner = str(zone.name)
            for door in zone.doors:
                name = str(door.name)
                start = (float(door.start.x), float(door.start.y))
                end = (float(door.end.x), float(door.end.y))
                line = LineString([start, end])
                midpoint = line.interpolate(0.5, normalized=True)

                candidates: list[tuple[float, str]] = []
                for candidate_name, polygon in polygons:
                    if candidate_name == owner:
                        continue
                    if zone_levels.get(candidate_name) != zone_levels.get(owner):
                        continue
                    overlap = line.intersection(polygon.boundary.buffer(config.adjacency_tolerance_m)).length
                    distance = polygon.boundary.distance(midpoint)
                    if overlap > 0.5 * line.length or distance <= config.adjacency_tolerance_m:
                        candidates.append((overlap - distance, candidate_name))

                if not candidates:
                    unpaired.append(
                        UnpairedDoor(
                            door_name=name,
                            owner_zone=owner,
                            start=start,
                            end=end,
                            reason="no_adjacent_acoustic_zone",
                        )
                    )
                    continue

                candidates.sort(reverse=True)
                neighbor = candidates[0][1]
                if owner not in room_names or neighbor not in room_names:
                    unpaired.append(
                        UnpairedDoor(
                            door_name=name,
                            owner_zone=owner,
                            start=start,
                            end=end,
                            reason="adjacent_zone_has_no_room_spec",
                        )
                    )
                    continue

                ordered_zones = tuple(sorted((owner, neighbor)))
                endpoints = sorted((start, end))
                geometry_key = tuple(round(v, 4) for point in endpoints for v in point)
                key = (ordered_zones[0], ordered_zones[1], geometry_key)
                if key in seen:
                    continue
                seen.add(key)

                portals.append(
                    AcousticPortal(
                        portal_id=_safe_id(f"door:{name}:{ordered_zones[0]}:{ordered_zones[1]}"),
                        door_name=name,
                        zone_a=owner,
                        zone_b=neighbor,
                        start=start,
                        end=end,
                        height_m=max(float(door.height), 0.1),
                        material_id=_material_name(door.material, DEFAULT_WALL_MATERIAL_ID),
                        portal_kind="door",
                        loss_db=config.door_loss_db,
                    )
                )

        if config.openings_enabled:
            room_by_name = {room.zone_name: room for room in rooms}
            for index, (zone_a, polygon_a) in enumerate(polygons):
                room_a = room_by_name.get(zone_a)
                if room_a is None:
                    continue
                openings_a = [LineString((boundary.start, boundary.end)) for boundary in room_a.boundary if boundary.kind == "opening"]
                for zone_b, polygon_b in polygons[index + 1 :]:
                    if zone_levels.get(zone_b) != zone_levels.get(zone_a):
                        continue
                    if polygon_a.boundary.distance(polygon_b.boundary) > config.adjacency_tolerance_m:
                        continue
                    room_b = room_by_name.get(zone_b)
                    if room_b is None:
                        continue
                    openings_b = [LineString((boundary.start, boundary.end)) for boundary in room_b.boundary if boundary.kind == "opening"]
                    for opening_a in openings_a:
                        for opening_b in openings_b:
                            shared = opening_a.intersection(opening_b)
                            lines = list(shared.geoms) if isinstance(shared, BaseMultipartGeometry) else [shared]
                            for line in lines:
                                if line.geom_type != "LineString" or line.length < config.min_opening_width_m:
                                    continue
                                coordinates = list(line.coords)
                                start = tuple(map(float, coordinates[0]))
                                end = tuple(map(float, coordinates[-1]))
                                ordered_zones = tuple(sorted((zone_a, zone_b)))
                                endpoints = sorted((start, end))
                                geometry_key = tuple(round(value, 4) for point in endpoints for value in point)
                                key = (
                                    ordered_zones[0],
                                    ordered_zones[1],
                                    geometry_key,
                                )
                                if key in seen:
                                    continue
                                candidate_line = LineString((start, end))
                                if any(portal.connects(zone_a, zone_b) and LineString((portal.start, portal.end)).distance(candidate_line) <= config.adjacency_tolerance_m and LineString((portal.start, portal.end)).intersection(candidate_line).length > 0.5 * candidate_line.length for portal in portals):
                                    continue
                                seen.add(key)
                                portals.append(
                                    AcousticPortal(
                                        portal_id=_safe_id(f"opening:{ordered_zones[0]}:{ordered_zones[1]}:" + ":".join(f"{value:.3f}" for value in geometry_key)),
                                        door_name="",
                                        zone_a=zone_a,
                                        zone_b=zone_b,
                                        start=start,
                                        end=end,
                                        height_m=min(
                                            room_a.ceiling_height_m,
                                            room_b.ceiling_height_m,
                                        ),
                                        material_id=OPENING_MATERIAL_ID,
                                        portal_kind="opening",
                                        loss_db=config.opening_loss_db,
                                    )
                                )

        return cls(
            rooms=rooms,
            portals=tuple(portals),
            zone_polygons=polygons,
            unpaired_doors=tuple(unpaired),
        )

    def room(self, zone_name: str) -> AcousticRoomSpec | None:
        return next((room for room in self.rooms if room.zone_name == zone_name), None)

    def zone_at_xy(self, x: float, y: float) -> str | None:
        point = Point(float(x), float(y))
        return next(
            (name for name, polygon in self.zone_polygons if polygon.covers(point)),
            None,
        )

    def find_portal_route(
        self,
        source_zone: str,
        listener_zone: str,
        *,
        source_xy: Point2D,
        listener_xy: Point2D,
        max_hops: int,
        route_loss_db_per_m: float,
        door_loss_db: float,
        opening_loss_db: float,
    ) -> AcousticPortalRoute | None:
        """Return the least-cost simple portal route between two rooms."""
        if source_zone == listener_zone:
            return AcousticPortalRoute(zones=(source_zone,), portals=())
        if max_hops <= 0:
            return None

        # (estimated total cost, sequence, accumulated cost, zone, anchor,
        #  zones, portals)
        queue: list[tuple[Hashable, ...]] = []
        sequence = itertools.count()
        heapq.heappush(
            queue,
            (
                route_loss_db_per_m * math.dist(source_xy, listener_xy),
                next(sequence),
                0.0,
                source_zone,
                source_xy,
                (source_zone,),
                (),
            ),
        )
        while queue:
            (
                _,
                _,
                cost,
                zone,
                anchor,
                zones,
                route,
            ) = heapq.heappop(queue)
            if zone == listener_zone and route:
                return AcousticPortalRoute(zones=zones, portals=route)
            if len(route) >= max_hops:
                continue
            for portal in self._adjacency.get(zone, ()):
                neighbor = portal.other_zone(zone)
                if neighbor in zones:
                    continue
                center = portal.center_xy
                segment_distance = math.dist(anchor, center)
                portal_loss = portal.loss_db if portal.loss_db is not None else (opening_loss_db if portal.portal_kind == "opening" else door_loss_db)
                next_cost = cost + portal_loss + route_loss_db_per_m * segment_distance
                heuristic = route_loss_db_per_m * math.dist(center, listener_xy)
                heapq.heappush(
                    queue,
                    (
                        next_cost + heuristic,
                        next(sequence),
                        next_cost,
                        neighbor,
                        center,
                        (*zones, neighbor),
                        (*route, portal),
                    ),
                )
        return None

    def connected_components(self) -> tuple[tuple[str, ...], ...]:
        remaining = {room.zone_name for room in self.rooms}
        adjacency = {name: set() for name in remaining}
        for portal in self.portals:
            adjacency.setdefault(portal.zone_a, set()).add(portal.zone_b)
            adjacency.setdefault(portal.zone_b, set()).add(portal.zone_a)
        components: list[tuple[str, ...]] = []
        while remaining:
            seed = min(remaining)
            stack = [seed]
            component: set[str] = set()
            while stack:
                zone = stack.pop()
                if zone in component:
                    continue
                component.add(zone)
                stack.extend(adjacency.get(zone, ()) - component)
            remaining -= component
            components.append(tuple(sorted(component)))
        return tuple(components)

    def position_inside_portal(
        self,
        portal: AcousticPortal,
        zone_name: str,
        *,
        inset_m: float,
        height_m: float,
    ) -> Position3D:
        polygon = next(
            (polygon for name, polygon in self.zone_polygons if name == zone_name),
            None,
        )
        if polygon is None:
            raise KeyError(f"unknown acoustic zone {zone_name!r}")

        center = Point(portal.center_xy)
        inner = polygon.buffer(-max(inset_m, 1e-4))
        xy = polygon.representative_point() if inner.is_empty else nearest_points(center, inner)[1]
        room = self.room(zone_name)
        if room is None:
            raise KeyError(f"no acoustic room for zone {zone_name!r}")
        z = min(max(float(height_m), 0.01), room.ceiling_height_m - 0.01)
        return float(xy.x), float(xy.y), z
