"""pyroomacoustics rooms from Arena room specs, RIR computation and direct-arrival analysis."""

from __future__ import annotations

import functools
import math
from collections import OrderedDict
from collections.abc import Hashable, Sequence
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import attrs
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from arena_auditory.materials import AcousticMaterial, AcousticMaterialCatalog
from arena_auditory.rooms import AcousticRoomSpec

if TYPE_CHECKING:
    import pyroomacoustics


Position3D = tuple[float, float, float]

DIRECT_WINDOW_HALF_SAMPLES = 40
DIRECT_ARRIVAL_RELATIVE_THRESHOLD = 0.25
MAIN_LOBE_SEARCH_SAMPLES = 3
AIR_ABSORPTION = True
RAY_TRACING = False
RANDOMIZED_ISM = False
MAX_RANDOM_DISPLACEMENT_M = 0.08
MINIMUM_PHASE = False
ROOM_BOUNDARY_TOLERANCE_M = 0.35
ROOM_BOUNDARY_INSET_M = 0.01


class PyroomacousticsUnavailableError(RuntimeError):
    """pyroomacoustics is required but not installed."""


class RirUnavailable(Exception):
    """No RIR exists for this room geometry and these positions."""


@attrs.frozen
class RirConfig:
    """Image-source room impulse settings, the rir.* parameters."""

    sample_rate_hz: int
    max_order: int
    temperature_c: float
    relative_humidity_pct: float
    quantization_m: float
    cache_size: int


@attrs.frozen
class BuiltPyroom:
    """A pyroomacoustics room, fallback_material_ids resolved through the catalog default."""

    room: Any
    specification: AcousticRoomSpec
    fallback_material_ids: tuple[str, ...]
    obstructing_walls: tuple[int, ...]


@attrs.frozen
class RoomImpulseResponse:
    """One source-to-listener room impulse response, global_delay_samples is the fractional-delay filter latency."""

    samples: NDArray[np.float64]
    sample_rate_hz: int
    global_delay_samples: int

    fallback_material_ids: tuple[str, ...]


@functools.lru_cache(maxsize=256)
def _room_polygon(specification: AcousticRoomSpec) -> Polygon:
    return Polygon(specification.corners_xy)


@attrs.frozen
class DirectArrival:
    """First significant arrival of an RIR, net of the filter delay."""

    index: int
    delay_s: float
    amplitude: float

    @property
    def gain_db(self) -> float:
        return 20.0 * math.log10(max(self.amplitude, 1e-12))


def direct_arrival(rir: RoomImpulseResponse) -> DirectArrival:
    """Locate the direct arrival and measure its windowed energy amplitude."""
    samples = np.asarray(rir.samples, dtype=np.float64)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("RIR contains no finite samples")
    magnitude = np.abs(samples)
    peak_index = int(np.argmax(magnitude))
    threshold = float(magnitude[peak_index]) * DIRECT_ARRIVAL_RELATIVE_THRESHOLD
    first = int(np.flatnonzero(magnitude[: peak_index + 1] >= threshold)[0])
    lobe_end = min(first + MAIN_LOBE_SEARCH_SAMPLES + 1, samples.size)
    index = first + int(np.argmax(magnitude[first:lobe_end]))
    offset = 0.0
    if 0 < index < samples.size - 1:
        before, center, after = magnitude[index - 1], magnitude[index], magnitude[index + 1]
        curvature = before - 2.0 * center + after
        if curvature < 0.0:
            offset = float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))
    window = samples[max(index - DIRECT_WINDOW_HALF_SAMPLES, 0) : index + DIRECT_WINDOW_HALF_SAMPLES + 1]
    return DirectArrival(
        index=index,
        delay_s=(index + offset - rir.global_delay_samples) / float(rir.sample_rate_hz),
        amplitude=float(np.sqrt(np.sum(window**2))),
    )


def retarget_rir(
    rir: RoomImpulseResponse,
    *,
    from_distance_m: float,
    to_distance_m: float,
    speed_of_sound_mps: float,
) -> RoomImpulseResponse:
    """Move an RIR's arrivals and 1/r level from one path length to another."""
    to_distance_m = max(to_distance_m, 1e-3)
    shift = int(round((to_distance_m - from_distance_m) / speed_of_sound_mps * rir.sample_rate_hz))
    samples = np.asarray(rir.samples, dtype=np.float64) * (from_distance_m / to_distance_m)
    if shift > 0:
        samples = np.concatenate((np.zeros(shift, dtype=np.float64), samples))
    elif shift < 0:
        samples = samples[-shift:]
    return attrs.evolve(rir, samples=samples)


@attrs.frozen
class _CachedRir:
    rir: RoomImpulseResponse
    source: Position3D
    listener: Position3D
    distance_m: float


def _reset_room(built: BuiltPyroom) -> None:
    """Drop sources, microphones and results so a built room can be reused."""
    room = built.room
    room.sources = []
    room.mic_array = None
    room.visibility = None
    room.rir = None
    room.simulator_state.update(ism_done=False, rt_done=False, rir_done=False)
    room._init_room_engine(room.walls, list(built.obstructing_walls))


def _load_pyroomacoustics() -> ModuleType:
    """Import pyroomacoustics on first use. Raises PyroomacousticsUnavailableError."""
    try:
        return import_module("pyroomacoustics")
    except ImportError as exc:
        raise PyroomacousticsUnavailableError("pyroomacoustics is required for acoustic RIR rendering but is not installed in the current Python environment") from exc


def _coefficient_description(
    material: AcousticMaterial,
    property_name: str,
) -> str:
    fallback_note = ", resolved through catalog default" if material.used_default else ""

    return f"Arena material {material.material_id}: {material.canonical_name}, {property_name}{fallback_note}"


class PyroomMaterialAdapter:
    """Arena materials as pyroomacoustics absorption and scattering, transmission loss stays with propagation."""

    def __init__(
        self,
        catalog: AcousticMaterialCatalog,
    ) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> AcousticMaterialCatalog:
        return self._catalog

    def resolve(
        self,
        material_id: str,
    ) -> AcousticMaterial:
        return self._catalog.get(material_id)

    def convert(
        self,
        material_or_id: AcousticMaterial | str,
        *,
        pyroomacoustics_module: ModuleType | None = None,
    ) -> pyroomacoustics.Material:
        """A fresh Material per call, pyroomacoustics may resample band data in place."""
        if isinstance(material_or_id, AcousticMaterial):
            material = material_or_id
        else:
            material = self.resolve(material_or_id)

        pra = pyroomacoustics_module if pyroomacoustics_module is not None else _load_pyroomacoustics()

        frequencies = list(material.center_frequencies_hz)

        energy_absorption = {
            "description": _coefficient_description(
                material,
                "energy absorption",
            ),
            "coeffs": list(material.absorption),
            "center_freqs": frequencies,
        }

        scattering = {
            "description": _coefficient_description(
                material,
                "scattering",
            ),
            "coeffs": list(material.scattering),
            "center_freqs": frequencies,
        }

        return pra.Material(
            energy_absorption=energy_absorption,
            scattering=scattering,
        )


class PyroomacousticsAdapter:
    """Builds pyroomacoustics rooms from isolated Arena room specs."""

    def __init__(
        self,
        material_catalog: AcousticMaterialCatalog,
        config: RirConfig,
    ) -> None:
        self._config = config
        self._materials = PyroomMaterialAdapter(material_catalog)
        self._rir_cache: OrderedDict[
            tuple[Hashable, ...],
            _CachedRir,
        ] = OrderedDict()
        self._rooms: OrderedDict[AcousticRoomSpec, BuiltPyroom] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._speed_of_sound_mps: float | None = None

    @property
    def config(self) -> RirConfig:
        return self._config

    @property
    def materials(self) -> PyroomMaterialAdapter:
        return self._materials

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def cache_entries(self) -> int:
        return len(self._rir_cache)

    @property
    def speed_of_sound_mps(self) -> float:
        if self._speed_of_sound_mps is None:
            self._speed_of_sound_mps = float(self._physics().get_sound_speed())
        return self._speed_of_sound_mps

    def _physics(self) -> pyroomacoustics.parameters.Physics:
        return _load_pyroomacoustics().parameters.Physics(
            temperature=self._config.temperature_c,
            humidity=self._config.relative_humidity_pct,
        )

    def build_room(
        self,
        specification: AcousticRoomSpec,
    ) -> BuiltPyroom:
        """One Arena zone specification as an extruded 3-D room."""
        self._validate_room_specification(specification)

        pra = _load_pyroomacoustics()

        corners = np.asarray(
            specification.corners_xy,
            dtype=np.float64,
        ).T

        resolved_boundary_materials = [self._materials.resolve(material_id) for material_id in specification.boundary_material_ids]

        boundary_materials = [
            self._materials.convert(
                material,
                pyroomacoustics_module=pra,
            )
            for material in resolved_boundary_materials
        ]

        floor_material = self._materials.resolve(specification.floor_material_id)
        ceiling_material = self._materials.resolve(specification.ceiling_material_id)

        surfaces = {
            "floor": self._materials.convert(floor_material, pyroomacoustics_module=pra),
            "ceiling": self._materials.convert(ceiling_material, pyroomacoustics_module=pra),
        }
        try:
            room = pra.Room.from_corners(
                corners,
                fs=self._config.sample_rate_hz,
                max_order=self._config.max_order,
                materials=boundary_materials,
                temperature=self._config.temperature_c,
                humidity=self._config.relative_humidity_pct,
                air_absorption=AIR_ABSORPTION,
                ray_tracing=RAY_TRACING,
                use_rand_ism=RANDOMIZED_ISM,
                max_rand_disp=MAX_RANDOM_DISPLACEMENT_M,
                min_phase=MINIMUM_PHASE,
            )
            room.extrude(specification.ceiling_height_m, materials=surfaces)
        except ValueError as exc:
            raise RirUnavailable(f"pyroomacoustics rejected room {specification.zone_name!r}: {exc}") from exc

        all_resolved_materials = [
            *resolved_boundary_materials,
            floor_material,
            ceiling_material,
        ]

        fallback_material_ids = tuple(dict.fromkeys(material.material_id for material in all_resolved_materials if material.used_default))

        return BuiltPyroom(
            room=room,
            specification=specification,
            fallback_material_ids=(fallback_material_ids),
            obstructing_walls=tuple(int(index) for index in pra.room.find_non_convex_walls(room.walls)),
        )

    def compute_rir(
        self,
        specification: AcousticRoomSpec,
        *,
        source_position_m: Sequence[float],
        listener_position_m: Sequence[float],
    ) -> RoomImpulseResponse:
        """RIR between two positions in the room frame, z the real emitter and listener heights. Cached per quantized pair and retargeted."""
        result = self.compute_rirs(specification, source_position_m=source_position_m, listener_positions_m=(listener_position_m,))[0]
        if isinstance(result, RirUnavailable):
            raise result
        return result

    def compute_rirs(
        self,
        specification: AcousticRoomSpec,
        *,
        source_position_m: Sequence[float],
        listener_positions_m: Sequence[Sequence[float]],
    ) -> list[RoomImpulseResponse | RirUnavailable]:
        """compute_rir for each listener in order, the cache misses from one pyroomacoustics run with one microphone per listener."""
        ceiling_height_m = specification.ceiling_height_m
        source = self._position_xyz(source_position_m, name="source_position_m")
        listeners = [self._position_xyz(position, name="listener_position_m") for position in listener_positions_m]
        try:
            source = self._height_inside_room(source, name="source_position_m", ceiling_height_m=ceiling_height_m)
        except RirUnavailable as exc:
            return [exc] * len(listeners)

        results: list[RoomImpulseResponse | RirUnavailable | None] = [None] * len(listeners)
        keys: list[tuple[Hashable, ...] | None] = [None] * len(listeners)
        planned: dict[tuple[Hashable, ...], tuple[int, Position3D]] = {}
        room_source: Position3D | None = None
        for index, position in enumerate(listeners):
            try:
                listener = listeners[index] = self._height_inside_room(position, name="listener_position_m", ceiling_height_m=ceiling_height_m)
            except RirUnavailable as exc:
                results[index] = exc
                continue
            key = keys[index] = self._rir_key(specification, source, listener)
            if key in planned or key in self._rir_cache:
                continue
            self._cache_misses += 1
            try:
                if room_source is None:
                    room_source = self._position_inside_room(specification, source, name="source_position_m")
                planned[key] = (index, self._position_inside_room(specification, listener, name="listener_position_m"))
            except RirUnavailable as exc:
                results[index] = exc

        if planned and room_source is not None:
            built = self._built_room(specification)
            computed = self._run(built, room_source, [room_listener for _, room_listener in planned.values()])
            for (key, (index, room_listener)), outcome in zip(planned.items(), computed, strict=True):
                results[index] = outcome
                if isinstance(outcome, RirUnavailable):
                    continue
                self._rir_cache[key] = _CachedRir(rir=outcome, source=source, listener=listeners[index], distance_m=math.dist(room_source, room_listener))
                while len(self._rir_cache) > self._config.cache_size:
                    self._rir_cache.popitem(last=False)

        for index, key in enumerate(keys):
            if results[index] is not None or key is None:
                continue
            cached = self._rir_cache.pop(key, None)
            if cached is None:
                results[index] = self.compute_rirs(specification, source_position_m=source, listener_positions_m=(listeners[index],))[0]
                continue
            self._rir_cache[key] = cached
            self._cache_hits += 1
            results[index] = retarget_rir(
                cached.rir,
                from_distance_m=cached.distance_m,
                to_distance_m=cached.distance_m + math.dist(source, listeners[index]) - math.dist(cached.source, cached.listener),
                speed_of_sound_mps=self.speed_of_sound_mps,
            )
        return cast("list[RoomImpulseResponse | RirUnavailable]", results)

    def _rir_key(self, specification: AcousticRoomSpec, source: Position3D, listener: Position3D) -> tuple[Hashable, ...]:
        quantization = self._config.quantization_m
        return (
            specification,
            tuple(int(round(value / quantization)) for value in source),
            tuple(int(round(value / quantization)) for value in listener),
        )

    def _run(self, built: BuiltPyroom, source: Position3D, listeners: Sequence[Position3D]) -> list[RoomImpulseResponse | RirUnavailable]:
        """One pyroomacoustics run with one microphone per listener, retried per listener when the joint run fails."""
        zone_name = built.specification.zone_name
        room = built.room
        _reset_room(built)
        room.add_source(np.asarray(source, dtype=np.float64))
        for listener in listeners:
            room.add_microphone(np.asarray(listener, dtype=np.float64))
        try:
            room.compute_rir()
        except ValueError as exc:
            if len(listeners) > 1:
                return [outcome for listener in listeners for outcome in self._run(built, source, (listener,))]
            return [RirUnavailable(f"pyroomacoustics failed in {zone_name!r}: {exc}")]
        fractional_delay_length = int(_load_pyroomacoustics().constants.get("frac_delay_length"))
        outcomes: list[RoomImpulseResponse | RirUnavailable] = []
        for microphone in range(len(listeners)):
            if not room.rir[microphone]:
                outcomes.append(RirUnavailable(f"pyroomacoustics produced no RIR in {zone_name!r}"))
                continue
            samples = np.asarray(room.rir[microphone][0], dtype=np.float64).copy()
            if samples.size == 0 or not np.isfinite(samples).all():
                outcomes.append(RirUnavailable(f"pyroomacoustics produced a non-finite RIR in {zone_name!r}"))
                continue
            outcomes.append(
                RoomImpulseResponse(
                    samples=samples,
                    sample_rate_hz=self._config.sample_rate_hz,
                    global_delay_samples=fractional_delay_length // 2,
                    fallback_material_ids=built.fallback_material_ids,
                )
            )
        return outcomes

    def _built_room(self, specification: AcousticRoomSpec) -> BuiltPyroom:
        built = self._rooms.pop(specification, None)
        if built is None:
            built = self.build_room(specification)
        self._rooms[specification] = built
        while len(self._rooms) > self._config.cache_size:
            self._rooms.popitem(last=False)
        return built

    def _position_inside_room(
        self,
        specification: AcousticRoomSpec,
        position: Position3D,
        *,
        name: str,
    ) -> Position3D:
        """Return a numerically safe in-room point for a near-edge position."""
        polygon = _room_polygon(specification)
        point = Point(position[0], position[1])
        if polygon.contains(point):
            return position

        distance = float(polygon.distance(point))
        tolerance = ROOM_BOUNDARY_TOLERANCE_M
        if distance > tolerance:
            raise RirUnavailable(f"{name} is {distance:.3f} m outside acoustic room {specification.zone_name!r} (tolerance={tolerance:.3f} m)")

        inset = polygon.buffer(-ROOM_BOUNDARY_INSET_M)
        target = inset if not inset.is_empty else polygon.representative_point()
        adjusted = nearest_points(target, point)[0]
        return float(adjusted.x), float(adjusted.y), position[2]

    @staticmethod
    def _position_xyz(
        value: Sequence[float],
        *,
        name: str,
    ) -> Position3D:
        try:
            raw = tuple(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a 3-element sequence") from exc

        if len(raw) != 3:
            raise ValueError(f"{name} must contain exactly three coordinates")

        try:
            position = (
                float(raw[0]),
                float(raw[1]),
                float(raw[2]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains a non-numeric coordinate") from exc

        if not all(math.isfinite(value) for value in position):
            raise ValueError(f"{name} contains a non-finite coordinate")

        return position

    def _height_inside_room(
        self,
        position: Position3D,
        *,
        name: str,
        ceiling_height_m: float,
    ) -> Position3D:
        """Clamp a near-floor or near-ceiling height strictly inside the room."""
        z = position[2]
        tolerance = ROOM_BOUNDARY_TOLERANCE_M
        if not -tolerance <= z <= ceiling_height_m + tolerance:
            raise RirUnavailable(f"{name} z={z} is outside the room height range 0..{ceiling_height_m} (tolerance={tolerance:.3f} m)")
        inset = min(ROOM_BOUNDARY_INSET_M, 0.5 * ceiling_height_m)
        return position[0], position[1], min(max(z, inset), ceiling_height_m - inset)

    @staticmethod
    def _validate_room_specification(
        specification: AcousticRoomSpec,
    ) -> None:
        corners = np.asarray(
            specification.corners_xy,
            dtype=np.float64,
        )

        if corners.ndim != 2 or corners.shape[1] != 2:
            raise RirUnavailable("room corners must form an N-by-2 array")

        if len(corners) < 3:
            raise RirUnavailable("a room requires at least three corners")

        if not np.isfinite(corners).all():
            raise RirUnavailable("room corners contain non-finite coordinates")

        if len(specification.boundary_material_ids) != len(corners):
            raise RirUnavailable("one boundary material is required per room edge")

        x = corners[:, 0]
        y = corners[:, 1]

        signed_double_area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

        if signed_double_area <= 0.0:
            raise RirUnavailable("room corners must be ordered counter-clockwise")

        if not math.isfinite(specification.ceiling_height_m) or specification.ceiling_height_m <= 0.0:
            raise RirUnavailable("ceiling height must be finite and positive")
