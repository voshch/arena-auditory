"""Sound propagation from one emission to one listener: receptions, backends and room impulses."""

from __future__ import annotations

import math
import typing
from collections.abc import Callable, Iterable, Iterator

import attrs
from arena_auditory_msgs.msg import AcousticPath, ContinuousHeardSoundState, HeardSoundEvent, SoundReception, SoundSource
from arena_rclpy_mixins.registry import FactoryRegistry
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from std_msgs.msg import Header

from arena_auditory.assets import SoundLibrary
from arena_auditory.materials import AcousticMaterialCatalog
from arena_auditory.params import BackendName
from arena_auditory.propagation.portal import PortalConfig
from arena_auditory.propagation.pyroom_adapter import RirConfig
from arena_auditory.propagation.rir import IMPULSE_WINDOW, Impulse, RirCache, digest, impulse_from_rir, rir_key
from arena_auditory.shared import NS_PER_S, AgentKind, ListenerId, ListenerKind, SourceSpec, Vec3, point_vec3

if typing.TYPE_CHECKING:
    from arena_auditory.propagation.pyroom import PyroomBackend
    from arena_auditory.world import AcousticWorld, OccupancyMap

__all__ = [
    "IMPULSE_WINDOW",
    "LISTENER_ORDER",
    "PROPAGATION_BACKENDS",
    "BackendName",
    "BackendUnavailable",
    "EarlyPath",
    "Emission",
    "Impulse",
    "Listener",
    "PortalConfig",
    "PropagationBackend",
    "PropagationConfig",
    "PropagationScene",
    "Propagator",
    "Reception",
    "RirCache",
    "RirConfig",
    "impulse_from_rir",
    "rir_key",
    "to_point",
]

LISTENER_ORDER = (ListenerKind.ARRAY, ListenerKind.ROBOT, ListenerKind.MICROPHONE, ListenerKind.AGENT)


def to_point(value: Vec3) -> Point:
    return Point(x=float(value[0]), y=float(value[1]), z=float(value[2]))


@attrs.frozen
class Emission:
    source_id: str
    agent_kind: AgentKind
    agent_name: str
    position: Vec3
    level_db: float
    surface: str = ""

    @classmethod
    def from_source(cls, source: SourceSpec, position: Vec3) -> Emission:
        """Emission of a source at a position in the acoustic frame, level referred to 1 m."""
        try:
            surface = SoundLibrary.default().asset(source.asset_id).surface
        except (KeyError, ValueError, FileNotFoundError):
            surface = ""
        return cls(source_id=source.id, agent_kind=source.agent_kind, agent_name=source.agent_name, position=position, level_db=source.level_at_1m_db, surface=surface)


@attrs.frozen
class Listener:
    id: str
    kind: ListenerKind
    owner: str
    position: Vec3

    @classmethod
    def at(cls, listener_id: str, position: Vec3) -> Listener:
        """Raises ValueError on a malformed listener id."""
        parsed = ListenerId.parse(listener_id)
        return cls(id=listener_id, kind=parsed.kind, owner=parsed.robot_name or "", position=position)

    def hears_itself(self, emission: Emission) -> bool:
        return self.kind in (ListenerKind.ROBOT, ListenerKind.ARRAY) and emission.agent_kind is AgentKind.ROBOT and self.owner == emission.agent_name


@attrs.frozen
class EarlyPath:
    delay_s: float
    gain_db: float
    bearing_rad: float
    reflection_point: Vec3
    interaction_type: str
    material_id: str = ""

    def to_msg(self) -> AcousticPath:
        return AcousticPath(
            delay=Duration(sec=int(self.delay_s), nanosec=int((self.delay_s % 1.0) * NS_PER_S)),
            gain_db=float(self.gain_db),
            bearing_rad=float(self.bearing_rad),
            reflection_point=to_point(self.reflection_point),
            interaction_type=self.interaction_type,
            material_id=self.material_id,
        )

    @classmethod
    def from_msg(cls, msg: AcousticPath) -> EarlyPath:
        return cls(
            delay_s=msg.delay.sec + msg.delay.nanosec / NS_PER_S,
            gain_db=float(msg.gain_db),
            bearing_rad=float(msg.bearing_rad),
            reflection_point=point_vec3(msg.reflection_point),
            interaction_type=msg.interaction_type,
            material_id=msg.material_id,
        )


@attrs.frozen(kw_only=True)
class Reception:
    """Propagation result for one listener, positions in the acoustic frame."""

    listener: Listener
    distance_m: float
    bearing_rad: float
    received_level_db: float
    threshold_db: float
    direct_delay_s: float
    audible: bool
    occluded: bool
    backend: str
    source_zone: str = ""
    listener_zone: str = ""
    used_fallback: bool = False
    fallback_reason: str = ""
    portal_ids: tuple[str, ...] = ()
    traversed_zones: tuple[str, ...] = ()
    portal_positions: tuple[Vec3, ...] = ()
    route_loss_db: float = 0.0
    early_paths: tuple[EarlyPath, ...] = ()
    reverb_rt60_s: float = 0.0
    reverb_gain_db: float = 0.0
    rir_key: str = ""
    impulse: Impulse | None = attrs.field(default=None, eq=False)

    def to_msg(self) -> SoundReception:
        return SoundReception(
            listener_id=self.listener.id,
            listener_position=to_point(self.listener.position),
            distance_m=float(self.distance_m),
            bearing_rad=float(self.bearing_rad),
            received_level_db=float(self.received_level_db),
            threshold_db=float(self.threshold_db),
            direct_delay_s=float(self.direct_delay_s),
            audible=bool(self.audible),
            occluded=bool(self.occluded),
            source_zone=self.source_zone,
            listener_zone=self.listener_zone,
            backend=self.backend,
            used_fallback=bool(self.used_fallback),
            fallback_reason=self.fallback_reason,
            portal_ids=list(self.portal_ids),
            traversed_zones=list(self.traversed_zones),
            portal_positions=[to_point(position) for position in self.portal_positions],
            route_loss_db=float(self.route_loss_db),
            rir_key=self.rir_key,
        )

    @classmethod
    def from_msg(
        cls,
        msg: SoundReception,
        *,
        early_paths: tuple[EarlyPath, ...] = (),
        reverb_rt60_s: float = 0.0,
        reverb_gain_db: float = 0.0,
        impulse: Impulse | None = None,
    ) -> Reception:
        """Raises ValueError on a malformed listener id."""
        return cls(
            listener=Listener.at(msg.listener_id, point_vec3(msg.listener_position)),
            distance_m=float(msg.distance_m),
            bearing_rad=float(msg.bearing_rad),
            received_level_db=float(msg.received_level_db),
            threshold_db=float(msg.threshold_db),
            direct_delay_s=float(msg.direct_delay_s),
            audible=bool(msg.audible),
            occluded=bool(msg.occluded),
            backend=msg.backend,
            source_zone=msg.source_zone,
            listener_zone=msg.listener_zone,
            used_fallback=bool(msg.used_fallback),
            fallback_reason=msg.fallback_reason,
            portal_ids=tuple(msg.portal_ids),
            traversed_zones=tuple(msg.traversed_zones),
            portal_positions=tuple(point_vec3(point) for point in msg.portal_positions),
            route_loss_db=float(msg.route_loss_db),
            early_paths=early_paths,
            reverb_rt60_s=reverb_rt60_s,
            reverb_gain_db=reverb_gain_db,
            rir_key=msg.rir_key,
            impulse=impulse,
        )

    def heard_msg(self, header: Header, source: SoundSource) -> HeardSoundEvent:
        return HeardSoundEvent(
            header=header,
            source=source,
            reception=self.to_msg(),
            early_paths=[path.to_msg() for path in self.early_paths],
            reverb_rt60_s=float(self.reverb_rt60_s),
            reverb_gain_db=float(self.reverb_gain_db),
        )

    def continuous_msg(self, header: Header, source: SoundSource) -> ContinuousHeardSoundState:
        return ContinuousHeardSoundState(header=header, source=source, reception=self.to_msg())


@attrs.frozen(kw_only=True)
class PropagationConfig:
    backend: BackendName
    threshold_db: float
    min_distance_m: float
    occlusion_db: float
    max_reflections: int
    reflection_floor_db: float
    rir: RirConfig
    portal: PortalConfig

    @property
    def rir_digest(self) -> str:
        """Digest of every RIR and portal setting, part of every rir_key."""
        return digest(attrs.astuple(self.rir), attrs.astuple(self.portal))


@attrs.frozen
class PropagationScene:
    world: AcousticWorld | None = None
    occupancy: OccupancyMap | None = None


class BackendUnavailable(Exception):
    """The backend cannot serve this pair, reason is the published fallback_reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class PropagationBackend(typing.Protocol):
    name: typing.ClassVar[str]
    config: PropagationConfig

    def propagate(self, emission: Emission, listener: Listener, scene: PropagationScene) -> Reception: ...


PROPAGATION_BACKENDS: FactoryRegistry[str, PropagationBackend] = FactoryRegistry()


@PROPAGATION_BACKENDS.register(BackendName.LEGACY.value)
def _legacy(config: PropagationConfig) -> PropagationBackend:
    from arena_auditory.propagation.legacy import LegacyBackend

    return LegacyBackend(config)


@PROPAGATION_BACKENDS.register("self")
def _self(config: PropagationConfig) -> PropagationBackend:
    from arena_auditory.propagation.self_path import SelfPathBackend

    return SelfPathBackend(config)


@PROPAGATION_BACKENDS.register(BackendName.LEVEL3.value)
def _level3(config: PropagationConfig, materials: AcousticMaterialCatalog) -> PropagationBackend:
    from arena_auditory.propagation.level3 import Level3Backend

    return Level3Backend(config, materials)


@PROPAGATION_BACKENDS.register(BackendName.PYROOMACOUSTICS.value)
def _pyroomacoustics(config: PropagationConfig, materials: AcousticMaterialCatalog, cache: RirCache) -> PropagationBackend:
    """Raises BackendUnavailable when pyroomacoustics is missing."""
    from arena_auditory.propagation.pyroom import PyroomBackend

    return PyroomBackend(config, materials, cache)


class Propagator:
    """Self path for a robot's own sources, else requested backend first, then level3, then legacy, recording the fallback reason."""

    def __init__(self, config: PropagationConfig, materials: AcousticMaterialCatalog, *, warn: Callable[[str], None] | None = None) -> None:
        self._config = config
        self._warn = warn or (lambda _message: None)
        self.impulses = RirCache(config.rir.cache_size)
        self._self = PROPAGATION_BACKENDS.get("self", config)
        self._legacy = PROPAGATION_BACKENDS.get(BackendName.LEGACY.value, config)
        self._level3 = PROPAGATION_BACKENDS.get(BackendName.LEVEL3.value, config, materials)
        self._pyroom: PyroomBackend | None = None
        self.init_fallback_reason = ""
        if config.backend is BackendName.PYROOMACOUSTICS:
            try:
                self._pyroom = typing.cast("PyroomBackend", PROPAGATION_BACKENDS.get(BackendName.PYROOMACOUSTICS.value, config, materials, self.impulses))
            except BackendUnavailable as exc:
                self.init_fallback_reason = exc.reason
                self._warn(f"pyroomacoustics backend could not be initialized, using Level-3 propagation instead: {exc.detail or exc.reason}")
        self._scene = PropagationScene()

    @property
    def config(self) -> PropagationConfig:
        return self._config

    @config.setter
    def config(self, config: PropagationConfig) -> None:
        """Retune levels and level3 at run time, RIR and portal settings keep their construction values."""
        self._config = config
        for backend in (self._self, self._legacy, self._level3, self._pyroom):
            if backend is not None:
                backend.config = attrs.evolve(config, rir=backend.config.rir, portal=backend.config.portal)

    @property
    def scene(self) -> PropagationScene:
        return self._scene

    def set_scene(self, scene: PropagationScene) -> None:
        self._scene = scene

    def cache_summary(self) -> str:
        return self._pyroom.cache_summary() if self._pyroom is not None else "no pyroomacoustics caches"

    def propagate(self, emission: Emission, listener: Listener) -> Reception:
        """Reception of emission at listener, with a room impulse when the pyroomacoustics backend serves the pair."""
        return next(self.propagate_many(emission, (listener,)))[1]

    def propagate_many(self, emission: Emission, listeners: Iterable[Listener]) -> Iterator[tuple[Listener, Reception]]:
        """propagate for every listener in LISTENER_ORDER, the pyroomacoustics listeners in the source's zone from one run."""
        scene = self._scene
        ordered = sorted(listeners, key=lambda listener: LISTENER_ORDER.index(listener.kind))
        backend = self._config.backend
        requested_pyroom = backend is BackendName.PYROOMACOUSTICS
        world = scene.world
        zoned = world is not None and bool(world.scene.zones)

        def pyroom_serves(listener: Listener) -> bool:
            return self._pyroom is not None and zoned and backend is not BackendName.LEGACY and listener.kind is not ListenerKind.AGENT and not listener.hears_itself(emission)

        served = [listener for listener in ordered if pyroom_serves(listener)]
        pyroom = self._pyroom.propagate_many(emission, served, scene) if self._pyroom is not None and served else iter(())
        for listener in ordered:
            if listener.hears_itself(emission):
                yield listener, self._self.propagate(emission, listener, scene)
                continue
            if backend is BackendName.LEGACY:
                yield listener, self._legacy.propagate(emission, listener, scene)
                continue
            if not zoned:
                reason = ("acoustic_scene_not_loaded" if world is None else "acoustic_scene_has_no_zones") if requested_pyroom else ""
                yield listener, attrs.evolve(self._legacy.propagate(emission, listener, scene), used_fallback=requested_pyroom, fallback_reason=reason)
                continue
            pedestrian = listener.kind is ListenerKind.AGENT
            reason = "" if pedestrian else self.init_fallback_reason
            if pyroom_serves(listener):
                _, outcome = next(pyroom)
                if isinstance(outcome, Reception):
                    yield listener, outcome
                    continue
                reason = outcome.reason
                if outcome.detail:
                    self._warn(outcome.detail)
            reception = self._level3.propagate(emission, listener, scene)
            yield listener, attrs.evolve(reception, used_fallback=requested_pyroom and not pedestrian, fallback_reason=reason)


def bearing_rad(source: Vec3, listener: Vec3) -> float:
    """atan2 of source minus listener in the xy plane."""
    return math.atan2(source[1] - listener[1], source[0] - listener[0])
