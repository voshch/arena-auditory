"""pyroomacoustics backend: same-room RIRs and RIRs coupled along portal routes."""

from __future__ import annotations

import importlib
import math
import typing
from collections.abc import Iterator, Sequence

import numpy as np

from arena_auditory.materials import AcousticMaterialCatalog
from arena_auditory.propagation import (
    BackendUnavailable,
    EarlyPath,
    Emission,
    Listener,
    PropagationConfig,
    PropagationScene,
    Reception,
    RirCache,
    bearing_rad,
    impulse_from_rir,
    rir_key,
)
from arena_auditory.propagation.portal import MultiPortalRirCoupler, PortalCouplingResult
from arena_auditory.propagation.pyroom_adapter import (
    DIRECT_WINDOW_HALF_SAMPLES,
    DirectArrival,
    PyroomacousticsAdapter,
    PyroomacousticsUnavailableError,
    RirUnavailable,
    RoomImpulseResponse,
    direct_arrival,
)
from arena_auditory.rooms import AcousticPortalRoute
from arena_auditory.world import AcousticWorld

SAME_ROOM = "pyroomacoustics_same_room"
ONE_DOOR = "pyroomacoustics_one_door"
MULTI_PORTAL = "pyroomacoustics_multi_portal"


class PyroomBackend:
    name: typing.ClassVar[str] = "pyroomacoustics"

    def __init__(self, config: PropagationConfig, materials: AcousticMaterialCatalog, cache: RirCache) -> None:
        """Raises BackendUnavailable when pyroomacoustics is missing."""
        try:
            importlib.import_module("pyroomacoustics")
            self._adapter = PyroomacousticsAdapter(materials, config.rir)
        except (ImportError, PyroomacousticsUnavailableError) as exc:
            raise BackendUnavailable(f"pyroomacoustics_initialization_failed:{type(exc).__name__}", str(exc)) from exc
        self.config = config
        self._cache = cache
        self._world_signature = ""
        self._coupler: MultiPortalRirCoupler | None = None
        self._route_verdicts: dict[tuple[str, str, int], str] = {}

    def propagate(self, emission: Emission, listener: Listener, scene: PropagationScene) -> Reception:
        """Reception with its room impulse."""
        _, outcome = next(self.propagate_many(emission, (listener,), scene))
        if isinstance(outcome, BackendUnavailable):
            raise outcome
        return outcome

    def propagate_many(self, emission: Emission, listeners: Sequence[Listener], scene: PropagationScene) -> Iterator[tuple[Listener, Reception | BackendUnavailable]]:
        """propagate for each listener in order, the listeners in the source's zone from one pyroomacoustics run."""
        world = scene.world
        if world is None:
            for listener in listeners:
                yield listener, BackendUnavailable("acoustic_scene_not_loaded")
            return
        coupler = self._sync(world)
        source = emission.position
        source_zone = world.zone_at(source[0], source[1])
        if source_zone is None:
            for listener in listeners:
                yield listener, BackendUnavailable("source_outside_acoustic_zones")
            return
        listener_zones = [world.zone_at(listener.position[0], listener.position[1]) for listener in listeners]
        same_room = [index for index, zone in enumerate(listener_zones) if zone is not None and zone.name == source_zone.name]
        room = world.room(source_zone.name)
        rirs: dict[int, RoomImpulseResponse | RirUnavailable] = {}
        for index, (listener, listener_zone) in enumerate(zip(listeners, listener_zones, strict=True)):
            if listener_zone is None:
                yield listener, BackendUnavailable("listener_outside_acoustic_zones")
                continue
            if listener_zone.name != source_zone.name:
                try:
                    outcome: Reception | BackendUnavailable = self._coupled(emission, listener, scene, world, coupler, source_zone.name, listener_zone.name)
                except BackendUnavailable as exc:
                    outcome = exc
                yield listener, outcome
                continue
            if room is None:
                yield listener, BackendUnavailable("same_zone_has_no_room_spec")
                continue
            if not rirs:
                positions = [listeners[member].position for member in same_room]
                rirs = dict(zip(same_room, self._adapter.compute_rirs(room, source_position_m=source, listener_positions_m=positions), strict=True))
            rir = rirs[index]
            if isinstance(rir, RirUnavailable):
                yield listener, BackendUnavailable("same_room_rir_unavailable", f"no same-room RIR for {emission.agent_name}->{listener.id} in {room.zone_name!r}: {rir}")
                continue
            yield listener, self._reception(emission, listener, scene, world, rir, zones=(room.zone_name, room.zone_name), coupling=None)

    def _coupled(
        self,
        emission: Emission,
        listener: Listener,
        scene: PropagationScene,
        world: AcousticWorld,
        coupler: MultiPortalRirCoupler,
        source_zone: str,
        listener_zone: str,
    ) -> Reception:
        """Reception through the portal route between two zones. Raises BackendUnavailable."""
        source, position = emission.position, listener.position
        route, reason = self._route(world, source_zone, listener_zone, source, position)
        if route is None:
            raise BackendUnavailable(reason)
        try:
            coupling = coupler.compute(
                source_zone=source_zone,
                listener_zone=listener_zone,
                route=route,
                source_position_m=source,
                listener_position_m=position,
            )
        except RirUnavailable as exc:
            raise BackendUnavailable(
                "portal_route_rir_unavailable",
                f"no portal-route RIR for {source_zone!r}->{listener_zone!r} through {[p.portal_id for p in route.portals]!r}: {exc}",
            ) from exc
        return self._reception(emission, listener, scene, world, coupling.rir, zones=(source_zone, listener_zone), coupling=coupling)

    def cache_summary(self) -> str:
        """Entries, hits and misses of the room RIR, portal segment and portal route caches."""
        adapter, coupler = self._adapter, self._coupler
        summary = f"room RIR cache {adapter.cache_entries} entries, {adapter.cache_hits} hits, {adapter.cache_misses} misses"
        if coupler is None:
            return summary
        return f"{summary}, portal segments {coupler.cache_entries} entries, {coupler.cache_hits} hits, {coupler.cache_misses} misses, portal routes {coupler.route_cache_entries} entries, {coupler.route_cache_hits} hits, {coupler.route_cache_misses} misses"

    def _sync(self, world: AcousticWorld) -> MultiPortalRirCoupler:
        if self._coupler is None or world.signature != self._world_signature:
            self._world_signature = world.signature
            self._coupler = MultiPortalRirCoupler(self._adapter, world.graph, world_name=world.name, config=self.config.portal)
            self._route_verdicts.clear()
        return self._coupler

    def _route(
        self,
        world: AcousticWorld,
        source_zone: str,
        listener_zone: str,
        source: tuple[float, float, float],
        listener: tuple[float, float, float],
    ) -> tuple[AcousticPortalRoute | None, str]:
        """The portal route between two zones, or the reason there is none."""
        portal = self.config.portal
        max_hops = portal.effective_max_hops
        zones = (source_zone, listener_zone, max_hops)
        verdict = self._route_verdicts.get(zones)
        if verdict is not None and verdict != "routable":
            return None, verdict
        route = world.graph.find_portal_route(
            source_zone,
            listener_zone,
            source_xy=(source[0], source[1]),
            listener_xy=(listener[0], listener[1]),
            max_hops=max_hops,
            route_loss_db_per_m=portal.route_loss_db_per_m,
            door_loss_db=portal.door_loss_db,
            opening_loss_db=portal.opening_loss_db,
        )
        if route is not None:
            self._route_verdicts[zones] = "routable"
            return route, ""
        unrestricted = world.graph.find_portal_route(
            source_zone,
            listener_zone,
            source_xy=(source[0], source[1]),
            listener_xy=(listener[0], listener[1]),
            max_hops=max(len(world.rooms) - 1, 1),
            route_loss_db_per_m=portal.route_loss_db_per_m,
            door_loss_db=portal.door_loss_db,
            opening_loss_db=portal.opening_loss_db,
        )
        verdict = "portal_route_exceeds_max_hops" if unrestricted is not None else "no_portal_route_between_zones"
        self._route_verdicts[zones] = verdict
        return None, verdict

    def _reception(
        self,
        emission: Emission,
        listener: Listener,
        scene: PropagationScene,
        world: AcousticWorld,
        rir: RoomImpulseResponse,
        *,
        zones: tuple[str, str],
        coupling: PortalCouplingResult | None,
    ) -> Reception:
        arrival = direct_arrival(rir)
        gain_db = arrival.gain_db
        delay_s = arrival.delay_s
        source, position = emission.position, listener.position
        occluded = coupling is None and scene.occupancy is not None and scene.occupancy.occluded(source, position)
        received = emission.level_db + gain_db - (self.config.occlusion_db if occluded else 0.0)

        portal_ids: tuple[str, ...] = ()
        traversed: tuple[str, ...] = (zones[0],)
        portal_positions: tuple[tuple[float, float, float], ...] = ()
        early_paths: tuple[EarlyPath, ...] = ()
        route_loss_db = 0.0
        backend = SAME_ROOM
        if coupling is not None:
            portals = coupling.route.portals if coupling.route is not None else (coupling.portal,)
            traversed = coupling.route.zones if coupling.route is not None else zones
            portal_ids = tuple(portal.portal_id for portal in portals)
            portal_positions = tuple((portal.center_xy[0], portal.center_xy[1], 0.5 * portal.height_m) for portal in portals)
            route_loss_db = float(coupling.applied_portal_loss_db)
            backend = ONE_DOOR if len(portals) == 1 else MULTI_PORTAL
            early_paths = tuple(
                EarlyPath(
                    delay_s=float(delay_s),
                    gain_db=float(gain_db),
                    bearing_rad=math.atan2(portal.center_xy[1] - position[1], portal.center_xy[0] - position[0]),
                    reflection_point=point,
                    interaction_type=f"portal_{portal.portal_kind}",
                    material_id=portal.material_id,
                )
                for portal, point in zip(portals, portal_positions, strict=True)
            )

        key = rir_key(
            world_signature=world.signature,
            backend=backend,
            zones=traversed,
            portal_ids=portal_ids,
            source=source,
            listener=position,
            quantization_m=self.config.rir.quantization_m,
            rir_digest=self.config.rir_digest,
        )
        return Reception(
            listener=listener,
            distance_m=float(max(math.hypot(source[0] - position[0], source[1] - position[1]), 1.0)),
            bearing_rad=bearing_rad(source, position),
            received_level_db=float(received),
            threshold_db=self.config.threshold_db,
            direct_delay_s=float(delay_s),
            audible=received >= self.config.threshold_db,
            occluded=occluded,
            backend=backend,
            source_zone=zones[0],
            listener_zone=zones[1],
            portal_ids=portal_ids,
            traversed_zones=tuple(traversed) if coupling is not None else (),
            portal_positions=portal_positions,
            route_loss_db=route_loss_db,
            early_paths=early_paths,
            reverb_rt60_s=0.0,
            reverb_gain_db=float(_late_gain_db(rir, arrival)),
            rir_key=key,
            impulse=self._cache.get(key) or impulse_from_rir(rir, key=key, backend=backend),
        )


def _late_gain_db(rir: RoomImpulseResponse, arrival: DirectArrival) -> float:
    """RMS of the RIR after the direct window relative to the direct amplitude."""
    samples = np.asarray(rir.samples, dtype=np.float64)
    direct_end = min(arrival.index + DIRECT_WINDOW_HALF_SAMPLES + 1, samples.size)
    late_energy = float(np.sqrt(np.mean(samples[direct_end:] ** 2))) if direct_end < samples.size else 0.0
    return 20.0 * math.log10(max(late_energy / max(arrival.amplitude, 1e-12), 1e-12)) if late_energy > 0.0 else -120.0
