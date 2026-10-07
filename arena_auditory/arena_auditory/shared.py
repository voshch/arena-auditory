"""Types shared by every auditory node: sources and listener ids."""

from __future__ import annotations

import enum
import math
import typing
from collections.abc import Iterable, Mapping

import attrs
from arena_auditory_msgs.msg import SoundSource
from arena_robots.audio import NS_PER_S, Vec3
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Point

INACTIVE_REPEATS = 5


def point_vec3(point: Point) -> Vec3:
    return (float(point.x), float(point.y), float(point.z))


def _vec3(value: Iterable[float]) -> Vec3:
    x, y, z = (float(v) for v in value)
    return (x, y, z)


def _split_ns(ns: int) -> tuple[int, int]:
    return ns // NS_PER_S, ns % NS_PER_S


def _state(values: Mapping[str, float] | Iterable[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(name), float(value)) for name, value in dict(values).items()))


def _content(values: Mapping[str, str] | Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(name), str(value)) for name, value in dict(values).items()))


@attrs.frozen(kw_only=True)
class SourceSpec:
    """One emitter and its emission, the python side of SoundSource.msg."""

    id: str
    kind: str
    asset_id: str
    model: str
    agent_kind: AgentKind = attrs.field(converter=AgentKind)
    position: Vec3 = attrs.field(converter=_vec3)
    level_db: float
    group_id: str = ""
    variant_id: str = ""
    agent_id: int = -1
    agent_name: str = ""
    tags: tuple[str, ...] = attrs.field(default=(), converter=tuple)
    yaw_rad: float = 0.0
    reference_distance_m: float = 1.0
    duration_ns: int = 0
    loop: bool = False
    active: bool = True
    program_start_ns: int = 0
    seed: int = 0
    state: tuple[tuple[str, float], ...] = attrs.field(default=(), converter=_state)
    content: tuple[tuple[str, str], ...] = attrs.field(default=(), converter=_content)

    @property
    def group(self) -> str:
        return self.group_id or self.id

    def state_value(self, name: str) -> float:
        """A streamed model input, 0 when the source does not carry it."""
        return dict(self.state).get(name, 0.0)

    def content_value(self, name: str) -> str:
        """A content entry such as speech text, empty when the source does not carry it."""
        return dict(self.content).get(name, "")

    @property
    def duration_s(self) -> float:
        return self.duration_ns / NS_PER_S

    @property
    def level_at_1m_db(self) -> float:
        """Level referred to 1 m, a reference distance <= 0 reads as 1 m."""
        reference = self.reference_distance_m if self.reference_distance_m > 0.0 else 1.0
        return self.level_db + 20.0 * math.log10(reference)

    def to_msg(self) -> SoundSource:
        duration_sec, duration_nanosec = _split_ns(self.duration_ns)
        start_sec, start_nanosec = _split_ns(self.program_start_ns)
        x, y, z = self.position
        return SoundSource(
            id=self.id,
            group_id=self.group_id,
            kind=self.kind,
            asset_id=self.asset_id,
            variant_id=self.variant_id,
            model=self.model,
            agent_kind=self.agent_kind.value,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tags=list(self.tags),
            position=Point(x=x, y=y, z=z),
            yaw_rad=self.yaw_rad,
            level_db=self.level_db,
            reference_distance_m=self.reference_distance_m,
            duration=Duration(sec=duration_sec, nanosec=duration_nanosec),
            loop=self.loop,
            active=self.active,
            program_start=Time(sec=start_sec, nanosec=start_nanosec),
            seed=self.seed,
            state_names=[name for name, _ in self.state],
            state_values=[value for _, value in self.state],
            content_names=[name for name, _ in self.content],
            content_values=[value for _, value in self.content],
        )

    @classmethod
    def from_msg(cls, msg: SoundSource) -> SourceSpec:
        return cls(
            id=msg.id,
            group_id=msg.group_id,
            kind=msg.kind,
            asset_id=msg.asset_id,
            variant_id=msg.variant_id,
            model=msg.model,
            agent_kind=AgentKind(msg.agent_kind),
            agent_id=msg.agent_id,
            agent_name=msg.agent_name,
            tags=tuple(msg.tags),
            position=(msg.position.x, msg.position.y, msg.position.z),
            yaw_rad=msg.yaw_rad,
            level_db=msg.level_db,
            reference_distance_m=msg.reference_distance_m,
            duration_ns=msg.duration.sec * NS_PER_S + msg.duration.nanosec,
            loop=msg.loop,
            active=msg.active,
            program_start_ns=msg.program_start.sec * NS_PER_S + msg.program_start.nanosec,
            seed=msg.seed,
            state=zip(msg.state_names, msg.state_values, strict=True),
            content=zip(msg.content_names, msg.content_values, strict=True),
        )


class ListenerKind(enum.StrEnum):
    ROBOT = "robot"
    AGENT = "agent"
    ARRAY = "array"
    MICROPHONE = "microphone"


class MicrophoneScope(enum.StrEnum):
    ROBOT = "robot"
    ZONE = "zone"
    RUNTIME = "runtime"
    VIEWPORT = "viewport"


VIEWPORT_VIEWS = ("projective_center", "down_projection")


@attrs.frozen(kw_only=True)
class ListenerId:
    """Parsed listener id. String forms: robot:<robot>, agent:<id>, array:<robot>:<mic>, microphone:<scope>:..."""

    kind: ListenerKind
    owner: str = ""
    name: str = ""
    scope: MicrophoneScope | None = None

    def __str__(self) -> str:
        match self.kind, self.scope:
            case ListenerKind.ROBOT | ListenerKind.AGENT, _:
                return f"{self.kind}:{self.owner}"
            case ListenerKind.ARRAY, _:
                return f"{self.kind}:{self.owner}:{self.name}"
            case ListenerKind.MICROPHONE, MicrophoneScope.ROBOT | MicrophoneScope.ZONE:
                return f"{self.kind}:{self.scope}:{self.owner}:{self.name}"
            case ListenerKind.MICROPHONE, MicrophoneScope.RUNTIME | MicrophoneScope.VIEWPORT:
                return f"{self.kind}:{self.scope}:{self.name}"
        raise ValueError(f"inconsistent listener id {self!r}")

    @property
    def robot_name(self) -> str | None:
        """Owning robot of robot, array and robot-mounted microphone listeners."""
        if self.kind in (ListenerKind.ROBOT, ListenerKind.ARRAY) or self.scope is MicrophoneScope.ROBOT:
            return self.owner
        return None

    @classmethod
    def parse(cls, listener_id: str) -> ListenerId:
        """Raises ValueError on a malformed id."""
        head, sep, rest = listener_id.partition(":")
        if not sep or not rest:
            raise ValueError(f"malformed listener id {listener_id!r}")
        try:
            kind = ListenerKind(head)
        except ValueError as exc:
            raise ValueError(f"unknown listener kind in {listener_id!r}") from exc
        match kind:
            case ListenerKind.ROBOT:
                return cls(kind=kind, owner=rest)
            case ListenerKind.AGENT:
                int(rest)
                return cls(kind=kind, owner=rest)
            case ListenerKind.ARRAY:
                robot, sep, mic = rest.rpartition(":")
                if not sep or not robot or not mic:
                    raise ValueError(f"malformed array listener id {listener_id!r}")
                return cls(kind=kind, owner=robot, name=mic)
            case ListenerKind.MICROPHONE:
                scope_text, sep, tail = rest.partition(":")
                try:
                    scope = MicrophoneScope(scope_text)
                except ValueError as exc:
                    raise ValueError(f"unknown microphone scope in {listener_id!r}") from exc
                if not sep or not tail:
                    raise ValueError(f"malformed microphone listener id {listener_id!r}")
                if scope in (MicrophoneScope.RUNTIME, MicrophoneScope.VIEWPORT):
                    if scope is MicrophoneScope.RUNTIME:
                        int(tail)
                    elif tail not in VIEWPORT_VIEWS:
                        raise ValueError(f"unknown viewport view in {listener_id!r}")
                    return cls(kind=kind, scope=scope, name=tail)
                owner, placement, index = (tail.rsplit(":", 2) + ["", ""])[:3]
                if not owner or not placement or not index:
                    raise ValueError(f"malformed microphone listener id {listener_id!r}")
                int(index)
                return cls(kind=kind, scope=scope, owner=owner, name=f"{placement}:{index}")
        raise ValueError(f"malformed listener id {listener_id!r}")

    @staticmethod
    def robot(robot: str) -> str:
        """Array-centroid listener of a robot, the bus listener."""
        return str(ListenerId(kind=ListenerKind.ROBOT, owner=robot))

    @staticmethod
    def agent(agent_id: int) -> str:
        return str(ListenerId(kind=ListenerKind.AGENT, owner=str(int(agent_id))))

    @staticmethod
    def array_mic(robot: str, mic: str) -> str:
        return str(ListenerId(kind=ListenerKind.ARRAY, owner=robot, name=mic))

    @staticmethod
    def robot_mic(robot: str, placement: str, index: int) -> str:
        return str(ListenerId(kind=ListenerKind.MICROPHONE, scope=MicrophoneScope.ROBOT, owner=robot, name=f"{placement}:{int(index)}"))

    @staticmethod
    def runtime_mic(index: int) -> str:
        return str(ListenerId(kind=ListenerKind.MICROPHONE, scope=MicrophoneScope.RUNTIME, name=str(int(index))))

    @staticmethod
    def viewport_mic(view: typing.Literal["projective_center", "down_projection"]) -> str:
        if view not in VIEWPORT_VIEWS:
            raise ValueError(f"unknown viewport view {view!r}")
        return str(ListenerId(kind=ListenerKind.MICROPHONE, scope=MicrophoneScope.VIEWPORT, name=view))

    @staticmethod
    def owner_robot(listener_id: str) -> str | None:
        """Owning robot of a listener id, None for other or malformed ids."""
        try:
            return ListenerId.parse(listener_id).robot_name
        except ValueError:
            return None
