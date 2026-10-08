"""Room-local RIRs composed along a portal route."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Hashable

import attrs
import numpy as np
from scipy.signal import fftconvolve

from arena_auditory.propagation.pyroom_adapter import PyroomacousticsAdapter, RirUnavailable, RoomImpulseResponse, retarget_rir
from arena_auditory.rooms import AcousticPortal, AcousticPortalRoute, AcousticWorldGraph, Position3D


@attrs.frozen
class PortalConfig:
    """Portal coupling settings, the portal.* parameters."""

    inset_m: float
    door_loss_db: float
    opening_loss_db: float
    multi_hop_enabled: bool
    max_hops: int
    route_loss_db_per_m: float
    early_window_s: float
    max_rir_duration_s: float
    quantization_m: float
    cache_size: int

    @property
    def effective_max_hops(self) -> int:
        return self.max_hops if self.multi_hop_enabled else 1


@attrs.frozen
class PortalCouplingResult:
    rir: RoomImpulseResponse
    portal: AcousticPortal
    source_portal_position: Position3D
    listener_portal_position: Position3D
    route: AcousticPortalRoute | None = None
    portal_positions: tuple[Position3D, ...] = ()
    applied_portal_loss_db: float = 0.0


@attrs.frozen
class _CachedSegment:
    rir: RoomImpulseResponse
    source: Position3D
    listener: Position3D


@attrs.frozen
class _CachedRoute:
    result: PortalCouplingResult
    source: Position3D
    listener: Position3D
    path_length_m: float


def _segment_length(first: Position3D, second: Position3D) -> float:
    return max(math.dist(first, second), 1e-3)


class MultiPortalRirCoupler:
    """Compose room-local RIRs along an acoustically weighted portal route."""

    def __init__(
        self,
        adapter: PyroomacousticsAdapter,
        graph: AcousticWorldGraph,
        *,
        world_name: str,
        config: PortalConfig,
    ) -> None:
        self._adapter = adapter
        self._graph = graph
        self._world_name = world_name
        self._config = config
        self._cache: OrderedDict[tuple[Hashable, ...], _CachedSegment] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._route_cache: OrderedDict[
            tuple[Hashable, ...],
            _CachedRoute,
        ] = OrderedDict()
        self._route_cache_hits = 0
        self._route_cache_misses = 0

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def cache_entries(self) -> int:
        return len(self._cache)

    @property
    def route_cache_entries(self) -> int:
        return len(self._route_cache)

    @property
    def route_cache_hits(self) -> int:
        return self._route_cache_hits

    @property
    def route_cache_misses(self) -> int:
        return self._route_cache_misses

    def compute(
        self,
        *,
        source_zone: str,
        listener_zone: str,
        source_position_m: Position3D,
        listener_position_m: Position3D,
        route: AcousticPortalRoute | None = None,
    ) -> PortalCouplingResult:
        if route is None:
            route = self._graph.find_portal_route(
                source_zone,
                listener_zone,
                source_xy=source_position_m[:2],
                listener_xy=listener_position_m[:2],
                max_hops=self._config.effective_max_hops,
                route_loss_db_per_m=self._config.route_loss_db_per_m,
                door_loss_db=self._config.door_loss_db,
                opening_loss_db=self._config.opening_loss_db,
            )
        if route is None or not route.portals:
            raise RirUnavailable(f"no portal route connects {source_zone!r} and {listener_zone!r}")
        route_key = self._route_cache_key(
            route,
            source_position_m,
            listener_position_m,
        )
        cached_route = self._route_cache.pop(route_key, None)
        if cached_route is not None:
            self._route_cache[route_key] = cached_route
            self._route_cache_hits += 1
            result = cached_route.result
            path_length = cached_route.path_length_m + (
                _segment_length(source_position_m, result.source_portal_position) + _segment_length(result.listener_portal_position, listener_position_m) - _segment_length(cached_route.source, result.source_portal_position) - _segment_length(result.listener_portal_position, cached_route.listener)
            )
            return attrs.evolve(
                result,
                rir=retarget_rir(
                    result.rir,
                    from_distance_m=cached_route.path_length_m,
                    to_distance_m=path_length,
                    speed_of_sound_mps=self._adapter.speed_of_sound_mps,
                ),
            )
        self._route_cache_misses += 1

        endpoints: list[tuple[Position3D, Position3D]] = []
        flat_positions: list[Position3D] = []
        for index, portal in enumerate(route.portals):
            before_zone = route.zones[index]
            after_zone = route.zones[index + 1]
            before_room = self._graph.room(before_zone)
            after_room = self._graph.room(after_zone)
            if before_room is None or after_room is None:
                raise RirUnavailable("portal endpoint has no acoustic room specification")
            portal_height = min(
                0.5 * portal.height_m,
                before_room.ceiling_height_m - 0.01,
                after_room.ceiling_height_m - 0.01,
            )
            before = self._graph.position_inside_portal(
                portal,
                before_zone,
                inset_m=self._config.inset_m,
                height_m=portal_height,
            )
            after = self._graph.position_inside_portal(
                portal,
                after_zone,
                inset_m=self._config.inset_m,
                height_m=portal_height,
            )
            endpoints.append((before, after))
            flat_positions.append(before)

        segment_endpoints = [
            (source_position_m, endpoints[0][0]),
            *((endpoints[index - 1][1], endpoints[index][0]) for index in range(1, len(endpoints))),
            (endpoints[-1][1], listener_position_m),
        ]
        room_rirs = [self._cached_room_rir(zone_name, start, end) for zone_name, (start, end) in zip(route.zones, segment_endpoints, strict=True)]
        sample_rates = {rir.sample_rate_hz for rir in room_rirs}
        if len(sample_rates) != 1:
            raise ValueError("portal route RIR sample rates differ")
        sample_rate = room_rirs[0].sample_rate_hz
        early_count = int(math.ceil(self._config.early_window_s * sample_rate))
        segments: list[np.ndarray] = []
        for index, rir in enumerate(room_rirs):
            samples = np.asarray(rir.samples, dtype=np.float64)
            if samples.size == 0:
                raise ValueError("portal route contains an empty room RIR")
            if index < len(room_rirs) - 1:
                peak = int(np.argmax(np.abs(samples)))
                samples = samples[: min(samples.size, max(peak + 1, early_count))]
            segments.append(samples)

        coupled = segments[0]
        maximum_count = int(self._config.max_rir_duration_s * sample_rate)
        for segment in segments[1:]:
            coupled = fftconvolve(coupled, segment, mode="full")
            coupled = coupled[:maximum_count]
        segment_lengths = [_segment_length(start, end) for start, end in segment_endpoints]
        gap_length = sum(math.dist(before, after) for before, after in endpoints)
        path_length = sum(segment_lengths) + gap_length
        coupled *= math.prod(segment_lengths) / path_length
        gap_samples = int(round(gap_length / self._adapter.speed_of_sound_mps * sample_rate))
        coupled = np.concatenate((np.zeros(gap_samples, dtype=np.float64), coupled))
        applied_loss = sum((portal.loss_db if portal.loss_db is not None else (self._config.opening_loss_db if portal.portal_kind == "opening" else self._config.door_loss_db)) for portal in route.portals)
        coupled *= 10.0 ** (-applied_loss / 20.0)
        coupled = np.asarray(coupled[:maximum_count], dtype=np.float64)
        fallback_ids = tuple(dict.fromkeys(material_id for rir in room_rirs for material_id in rir.fallback_material_ids))
        result = PortalCouplingResult(
            rir=RoomImpulseResponse(
                samples=coupled,
                sample_rate_hz=sample_rate,
                global_delay_samples=sum(rir.global_delay_samples for rir in room_rirs),
                fallback_material_ids=fallback_ids,
            ),
            portal=route.portals[0],
            source_portal_position=endpoints[0][0],
            listener_portal_position=endpoints[-1][1],
            route=route,
            portal_positions=tuple(flat_positions),
            applied_portal_loss_db=applied_loss,
        )
        self._route_cache[route_key] = _CachedRoute(
            result=result,
            source=source_position_m,
            listener=listener_position_m,
            path_length_m=path_length,
        )
        while len(self._route_cache) > self._config.cache_size:
            self._route_cache.popitem(last=False)
        return result

    def _route_cache_key(
        self,
        route: AcousticPortalRoute,
        source: Position3D,
        listener: Position3D,
    ) -> tuple[Hashable, ...]:
        quantization = self._config.quantization_m

        def quantize(position: Position3D) -> tuple[int, int, int]:
            return tuple(int(round(value / quantization)) for value in position)

        return (
            self._world_name,
            tuple(portal.portal_id for portal in route.portals),
            quantize(source),
            quantize(listener),
        )

    def _cached_room_rir(
        self,
        zone_name: str,
        source: Position3D,
        listener: Position3D,
    ) -> RoomImpulseResponse:
        quantization = self._config.quantization_m

        def quantize(position: Position3D) -> tuple[int, int, int]:
            return tuple(int(round(value / quantization)) for value in position)

        key = (
            self._world_name,
            zone_name,
            quantize(source),
            quantize(listener),
        )
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            self._cache_hits += 1
            return retarget_rir(
                cached.rir,
                from_distance_m=_segment_length(cached.source, cached.listener),
                to_distance_m=_segment_length(source, listener),
                speed_of_sound_mps=self._adapter.speed_of_sound_mps,
            )

        room = self._graph.room(zone_name)
        if room is None:
            raise RirUnavailable(f"no room specification for zone {zone_name!r}")
        rir = self._adapter.compute_rir(
            room,
            source_position_m=source,
            listener_position_m=listener,
        )
        self._cache[key] = _CachedSegment(rir=rir, source=source, listener=listener)
        self._cache_misses += 1
        while len(self._cache) > self._config.cache_size:
            self._cache.popitem(last=False)
        return rir
