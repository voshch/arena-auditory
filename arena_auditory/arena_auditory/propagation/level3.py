"""Level-3 propagation: distance, floor and wall losses, first-order reflections and a Sabine reverb estimate."""

from __future__ import annotations

import math
import typing

import attrs
import shapely

from arena_auditory.materials import AcousticMaterialCatalog
from arena_auditory.propagation import EarlyPath, Emission, Listener, PropagationConfig, PropagationScene, Reception, bearing_rad
from arena_auditory.shared import SPEED_OF_SOUND_MPS, Vec3
from arena_auditory.world import AcousticScene, AcousticWall, AcousticZone


@attrs.frozen
class PropagationPath:
    delay_s: float
    gain_db: float
    bearing_rad: float
    reflection_point: tuple[float, float] | None
    interaction_type: str
    material_id: str = ""


@attrs.frozen
class PropagationResult:
    received_volume_db: float
    direct_delay_s: float
    occluded: bool
    paths: tuple[PropagationPath, ...]
    rt60_s: float
    reverb_gain_db: float
    source_zone: str
    listener_zone: str


def sabine_reverb(materials: AcousticMaterialCatalog, scene: AcousticScene, zone: AcousticZone) -> tuple[float, float]:
    """Sabine RT60 of a zone, walls and ceiling at absorption 0.10, and the reverb gain paired with it."""
    floor_area = max(zone.polygon.area, 1e-6)
    perimeter = zone.polygon.length
    height = scene.ceiling_height_m

    volume = floor_area * height
    floor_material = materials.get(zone.floor_material_id)

    floor_absorption = floor_area * sum(floor_material.absorption) / len(floor_material.absorption)

    wall_absorption = perimeter * height * 0.10
    ceiling_absorption = floor_area * 0.10
    total_absorption = max(
        floor_absorption + wall_absorption + ceiling_absorption,
        1e-6,
    )

    rt60 = min(0.161 * volume / total_absorption, 10.0)
    return rt60, -12.0 if rt60 > 0.0 else -math.inf


class Level3Propagation:
    def __init__(
        self,
        materials: AcousticMaterialCatalog,
        *,
        max_reflections: int,
        reflection_floor_db: float,
    ) -> None:
        self._materials = materials
        self._max_reflections = max_reflections
        self._reflection_floor_db = reflection_floor_db

    def calculate(
        self,
        scene: AcousticScene,
        source: Vec3,
        listener: Vec3,
        source_level_db: float,
        *,
        floor_damped: bool = False,
    ) -> PropagationResult:
        distance = self._distance(source, listener)
        attenuation_distance = max(distance, 1.0)
        direct_loss = 20.0 * math.log10(attenuation_distance)

        source_zone = scene.zone_at_xy(source[0], source[1])
        listener_zone = scene.zone_at_xy(listener[0], listener[1])
        floor_loss_db = 0.0
        if source_zone is not None and not floor_damped:
            floor_loss_db += self._materials.surface_damping_db(
                source_zone.floor_material_id,
                "floor",
            )
        if listener_zone is not None and (source_zone is None or source_zone.name != listener_zone.name):
            floor_loss_db += self._materials.surface_damping_db(
                listener_zone.floor_material_id,
                "floor",
            )

        crossed = scene.intersecting_walls(source, listener)
        transmission_loss = sum(self._materials.surface_damping_db(wall.material_id, "wall") for wall in crossed)

        direct_level = source_level_db - direct_loss - floor_loss_db - transmission_loss

        paths = [
            PropagationPath(
                delay_s=distance / SPEED_OF_SOUND_MPS,
                gain_db=direct_level - source_level_db,
                bearing_rad=self._bearing(source, listener),
                reflection_point=None,
                interaction_type="direct",
            )
        ]

        reflections = self._first_order_reflections(scene, source, listener, source_level_db)
        paths.extend(reflections[: self._max_reflections])

        combined_level = self._sum_decibels([source_level_db + path.gain_db for path in paths])

        reverb_zone = listener_zone or source_zone
        rt60, reverb_gain_db = sabine_reverb(self._materials, scene, reverb_zone) if reverb_zone is not None else (0.0, -math.inf)

        return PropagationResult(
            received_volume_db=combined_level,
            direct_delay_s=distance / SPEED_OF_SOUND_MPS,
            occluded=bool(crossed),
            paths=tuple(paths),
            rt60_s=rt60,
            reverb_gain_db=reverb_gain_db,
            source_zone=source_zone.name if source_zone else "",
            listener_zone=listener_zone.name if listener_zone else "",
        )

    def _first_order_reflections(self, scene: AcousticScene, source: Vec3, listener: Vec3, source_level_db: float) -> list[PropagationPath]:
        paths = []

        for wall in scene.walls:
            image = self._reflect_point(source, wall)
            ray = shapely.LineString([(image[0], image[1]), (listener[0], listener[1])])
            intersection = ray.intersection(wall.geometry)

            if intersection.is_empty or intersection.geom_type != "Point":
                continue

            reflection = (float(intersection.x), float(intersection.y), source[2])

            d1 = self._distance(source, reflection)
            d2 = self._distance(reflection, listener)
            total_distance = d1 + d2
            attenuation_distance = max(total_distance, 1.0)

            material = self._materials.get(wall.material_id)
            absorption = sum(material.absorption) / len(material.absorption)
            reflection_coefficient = math.sqrt(max(1.0 - absorption, 1e-6))

            reflection_loss = -20.0 * math.log10(max(reflection_coefficient, 1e-6))
            crossed = [crossed_wall for leg in ((source, reflection), (reflection, listener)) for crossed_wall in scene.intersecting_walls(*leg) if crossed_wall is not wall]
            transmission_loss = sum(self._materials.surface_damping_db(crossed_wall.material_id, "wall") for crossed_wall in crossed)
            gain_db = -20.0 * math.log10(attenuation_distance) - reflection_loss - transmission_loss

            if source_level_db + gain_db < self._reflection_floor_db:
                continue

            paths.append(
                PropagationPath(
                    delay_s=total_distance / SPEED_OF_SOUND_MPS,
                    gain_db=gain_db,
                    bearing_rad=self._bearing(reflection, listener),
                    reflection_point=(reflection[0], reflection[1]),
                    interaction_type="reflection",
                    material_id=wall.material_id,
                )
            )

        return sorted(paths, key=lambda path: path.delay_s)

    @staticmethod
    def _reflect_point(point: Vec3, wall: AcousticWall) -> Vec3:
        x1, y1 = wall.start
        x2, y2 = wall.end

        dx = x2 - x1
        dy = y2 - y1
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-12:
            return point

        projection = ((point[0] - x1) * dx + (point[1] - y1) * dy) / length_squared

        projected_x = x1 + projection * dx
        projected_y = y1 + projection * dy

        return (2.0 * projected_x - point[0], 2.0 * projected_y - point[1], point[2])

    @staticmethod
    def _distance(a: Vec3, b: Vec3) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        dz = float(a[2] - b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _bearing(source: Vec3, listener: Vec3) -> float:
        dx = float(source[0] - listener[0])
        dy = float(source[1] - listener[1])
        return math.atan2(dy, dx)

    @staticmethod
    def _sum_decibels(levels_db: list[float]) -> float:
        powers = [10.0 ** (level / 10.0) for level in levels_db if math.isfinite(level)]

        if not powers:
            return -math.inf

        return 10.0 * math.log10(sum(powers))


class Level3Backend:
    name: typing.ClassVar[str] = "level3"

    def __init__(self, config: PropagationConfig, materials: AcousticMaterialCatalog) -> None:
        self._materials = materials
        self.config = config

    @property
    def config(self) -> PropagationConfig:
        return self._config

    @config.setter
    def config(self, config: PropagationConfig) -> None:
        self._config = config
        self._model = Level3Propagation(self._materials, max_reflections=config.max_reflections, reflection_floor_db=config.reflection_floor_db)

    def propagate(self, emission: Emission, listener: Listener, scene: PropagationScene) -> Reception:
        if scene.world is None:
            raise ValueError("level3 propagation needs a loaded acoustic world")
        source, position = emission.position, listener.position
        result = self._model.calculate(scene.world.scene, source, position, emission.level_db, floor_damped=emission.surface == "floor")
        return Reception(
            listener=listener,
            distance_m=float(max(math.hypot(source[0] - position[0], source[1] - position[1]), 1.0)),
            bearing_rad=bearing_rad(source, position),
            received_level_db=result.received_volume_db,
            threshold_db=self.config.threshold_db,
            direct_delay_s=result.direct_delay_s,
            audible=result.received_volume_db >= self.config.threshold_db,
            occluded=result.occluded,
            backend=self.name,
            source_zone=result.source_zone,
            listener_zone=result.listener_zone,
            early_paths=tuple(
                EarlyPath(
                    delay_s=path.delay_s,
                    gain_db=path.gain_db,
                    bearing_rad=path.bearing_rad,
                    reflection_point=(path.reflection_point[0], path.reflection_point[1], 0.0) if path.reflection_point is not None else (0.0, 0.0, 0.0),
                    interaction_type=path.interaction_type,
                    material_id=path.material_id,
                )
                for path in result.paths
            ),
            reverb_rt60_s=result.rt60_s,
            reverb_gain_db=result.reverb_gain_db,
        )
