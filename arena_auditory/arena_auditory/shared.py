"""Types shared by every auditory node: sources, listener ids, microphone arrays, robot bindings and level helpers."""

from __future__ import annotations

import enum
import functools
import math
import subprocess
import typing
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_path
from arena_auditory_msgs.msg import SoundSource
from arena_robots.Robot import RobotIdentifier
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Point
from numpy.typing import NDArray

if typing.TYPE_CHECKING:
    from task_generator_msgs.msg import RobotFleet

type Vec3 = tuple[float, float, float]

NS_PER_S = 1_000_000_000
SPEED_OF_SOUND_MPS = 343.0
INACTIVE_REPEATS = 5


def point_vec3(point: Point) -> Vec3:
    return (float(point.x), float(point.y), float(point.z))


def _vec3(value: Iterable[float]) -> Vec3:
    x, y, z = (float(v) for v in value)
    return (x, y, z)


def _join_frame(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _split_ns(ns: int) -> tuple[int, int]:
    return ns // NS_PER_S, ns % NS_PER_S


class AgentKind(enum.StrEnum):
    PEDESTRIAN = "pedestrian"
    ROBOT = "robot"
    ENVIRONMENT = "environment"
    EXTERNAL = "external"


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


@attrs.frozen(kw_only=True)
class MicSpec:
    name: str
    position_m: Vec3 = attrs.field(converter=_vec3)
    yaw_rad: float = 0.0
    side: typing.Literal["left", "right", "center"] = attrs.field(default="center", validator=attrs.validators.in_(("left", "right", "center")))
    group: typing.Literal["front", "rear", ""] = attrs.field(default="", validator=attrs.validators.in_(("front", "rear", "")))


PRESETS: tuple[str, ...] = ("mono", "stereo", "four_mic")


@attrs.frozen(kw_only=True)
class ArraySpec:
    """Microphone array geometry, positions in the robot mount frame."""

    name: str
    sample_rate_hz: int
    block_size: int
    sensitivity_dbfs_at_94_dbspl: float
    mics: tuple[MicSpec, ...] = attrs.field(converter=tuple)

    @property
    def channels(self) -> int:
        return len(self.mics)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(mic.name for mic in self.mics)

    @property
    def centroid_m(self) -> Vec3:
        return _vec3(np.mean(np.asarray([mic.position_m for mic in self.mics], dtype=np.float64), axis=0))

    @classmethod
    def from_dict(cls, data: Mapping[str, typing.Any]) -> ArraySpec:
        """Raises ValueError on an invalid layout."""
        if "rectangular" in data:
            mics = rectangular(**{key: float(value) for key, value in data["rectangular"].items()})
        else:
            mics = tuple(
                MicSpec(
                    name=str(mic["name"]),
                    position_m=mic.get("position_m", (0.0, 0.0, 0.0)),
                    yaw_rad=math.radians(float(mic.get("yaw_deg", 0.0))),
                    side=mic.get("side", "center"),
                    group=mic.get("group", ""),
                )
                for mic in data["mics"]
            )
        if not mics:
            raise ValueError("an array needs at least one microphone")
        names = [mic.name for mic in mics]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate microphone names {names}")
        return cls(
            name=str(data["name"]),
            sample_rate_hz=int(data["sample_rate_hz"]),
            block_size=int(data["block_size"]),
            sensitivity_dbfs_at_94_dbspl=float(data["sensitivity_dbfs_at_94_dbspl"]),
            mics=mics,
        )


def rectangular(*, width_m: float, length_m: float, height_m: float, corner_inset_m: float) -> tuple[MicSpec, ...]:
    """REP-103 four-microphone rectangle in channel order FL FR RL RR."""
    values = (width_m, length_m, height_m, corner_inset_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("microphone geometry must be finite")
    if width_m <= 0.0 or length_m <= 0.0 or height_m < 0.0:
        raise ValueError("width/length must be positive and height non-negative")
    if corner_inset_m < 0.0 or 2.0 * corner_inset_m >= min(width_m, length_m):
        raise ValueError("corner inset must leave a positive rectangular aperture")
    x = length_m / 2.0 - corner_inset_m
    y = width_m / 2.0 - corner_inset_m
    return (
        MicSpec(name="front_left", position_m=(x, y, height_m), yaw_rad=math.radians(45.0), side="left", group="front"),
        MicSpec(name="front_right", position_m=(x, -y, height_m), yaw_rad=math.radians(-45.0), side="right", group="front"),
        MicSpec(name="rear_left", position_m=(-x, y, height_m), yaw_rad=math.radians(135.0), side="left", group="rear"),
        MicSpec(name="rear_right", position_m=(-x, -y, height_m), yaw_rad=math.radians(-135.0), side="right", group="rear"),
    )


def presets_dir() -> Path:
    return get_package_share_path("arena_auditory") / "config" / "arrays"


def load_array_spec(ref: str) -> ArraySpec:
    """A preset name from PRESETS or a yaml path. Raises FileNotFoundError or ValueError."""
    path = presets_dir() / f"{ref}.yaml" if ref in PRESETS else Path(ref).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"array spec {ref!r} is neither a preset {PRESETS} nor a file")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError(f"array spec {path} must be a mapping")
    return ArraySpec.from_dict(data)


def transform(mics: Sequence[MicSpec], *, position_m: Vec3, yaw_rad: float) -> tuple[Vec3, ...]:
    """Rigidly transform mount-frame microphone positions into the frame of position_m."""
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    rx, ry, rz = position_m
    return tuple(
        (
            rx + cosine * mic.position_m[0] - sine * mic.position_m[1],
            ry + sine * mic.position_m[0] + cosine * mic.position_m[1],
            rz + mic.position_m[2],
        )
        for mic in mics
    )


def geometric_delays_s(source_m: Vec3, mics_m: Sequence[Vec3], *, speed_of_sound_mps: float = SPEED_OF_SOUND_MPS) -> NDArray[np.float64]:
    if speed_of_sound_mps <= 0.0 or not math.isfinite(speed_of_sound_mps):
        raise ValueError("speed of sound must be finite and positive")
    source = np.asarray(source_m, dtype=np.float64)
    positions = np.asarray(mics_m, dtype=np.float64)
    return np.linalg.norm(positions - source[None, :], axis=1) / speed_of_sound_mps


DEFAULT_ODOM_TOPIC_TEMPLATE = "{namespace}/{name}_velocity_controller/odom"

_MODEL_ERRORS = (OSError, LookupError, ValueError, TypeError, yaml.YAMLError, subprocess.CalledProcessError)


@attrs.frozen(kw_only=True)
class _ModelFacts:
    base_frame: str
    odom_topic: str
    wheel_separation_m: float | None
    max_linear_mps: float


def _wheel_separation(control: Mapping[str, typing.Any]) -> float | None:
    for name, value in control.items():
        if name == "controller_manager" or not isinstance(value, Mapping):
            continue
        params = value.get("ros__parameters")
        if not isinstance(params, Mapping) or "wheel_separation" not in params:
            continue
        effective = float(params["wheel_separation"]) * float(params.get("wheel_separation_multiplier", 1.0))
        return effective if effective > 0.0 and math.isfinite(effective) else None
    return None


@functools.cache
def _model_facts(model: str) -> _ModelFacts:
    view = RobotIdentifier(model).resolve_sync()
    control = view.model_params.control
    mobile = view.mobile
    try:
        separation = _wheel_separation(view.control)
    except _MODEL_ERRORS:
        separation = None
    return _ModelFacts(
        base_frame=view.model_params.base_frame.strip("/"),
        odom_topic=control.odom_topic.strip("/") if control is not None else "",
        wheel_separation_m=separation,
        max_linear_mps=float(mobile.velocity_limits.linear.max) if mobile is not None and mobile.velocity_limits is not None else 0.0,
    )


@attrs.frozen(kw_only=True)
class RobotBinding:
    """A fleet robot with its model-derived frames, odometry topics and drivetrain facts."""

    name: str
    model: str
    namespace: str
    frame_prefix: str
    model_base_frame: str
    base_frame: str
    odom_topics: tuple[str, ...]
    wheel_separation_m: float | None
    max_linear_mps: float
    error: str = ""

    def frame(self, leaf: str) -> str:
        return _join_frame(self.frame_prefix, leaf)

    def mount(self, template: str) -> str:
        """Array mount frame of an array.mount_frame template: empty = base frame, {prefix} and {base_frame} expand, a bare leaf joins the prefix."""
        configured = template.format(prefix=self.frame_prefix, base_frame=self.model_base_frame).strip("/")
        if not configured:
            return self.base_frame
        if "/" in configured:
            return configured
        return self.frame(configured)

    @classmethod
    def resolve(cls, *, name: str, model: str, namespace: str, frame_prefix: str, odom_topic_template: str = DEFAULT_ODOM_TOPIC_TEMPLATE) -> RobotBinding:
        """Model lookups that fail fall back to base_link, no model odometry, no wheel separation and 0 m/s, with error set."""
        namespace = namespace.rstrip("/")
        prefix = frame_prefix.strip("/")
        error = ""
        try:
            facts = _model_facts(model)
        except _MODEL_ERRORS as exc:
            error = f"could not resolve robot model {model!r}: {exc}"
            facts = _ModelFacts(base_frame="base_link", odom_topic="", wheel_separation_m=None, max_linear_mps=0.0)
        topics = [odom_topic_template.format(namespace=namespace, name=name), f"{namespace}/odom"]
        if facts.odom_topic:
            topics.append(f"{namespace}/{facts.odom_topic}")
        return cls(
            name=name,
            model=model,
            namespace=namespace,
            frame_prefix=prefix,
            model_base_frame=facts.base_frame,
            base_frame=_join_frame(prefix, facts.base_frame),
            odom_topics=tuple(dict.fromkeys(topic for topic in topics if topic)),
            wheel_separation_m=facts.wheel_separation_m,
            max_linear_mps=facts.max_linear_mps,
            error=error,
        )


def robot_bindings(fleet: RobotFleet, *, odom_topic_template: str = DEFAULT_ODOM_TOPIC_TEMPLATE) -> tuple[RobotBinding, ...]:
    """Every named fleet robot, in fleet order."""
    return tuple(
        RobotBinding.resolve(
            name=str(state.descriptor.name).strip(),
            model=str(state.descriptor.model),
            namespace=str(state.descriptor.ns),
            frame_prefix=str(state.descriptor.frame),
            odom_topic_template=odom_topic_template,
        )
        for state in fleet.robots
        if str(state.descriptor.name).strip()
    )


def bind_robot(fleet: RobotFleet, wanted: str = "", *, odom_topic_template: str = DEFAULT_ODOM_TOPIC_TEMPLATE) -> RobotBinding | None:
    """The fleet robot named wanted, or the first one when empty. None until the fleet carries it."""
    for state in fleet.robots:
        name = str(state.descriptor.name).strip()
        if name and (not wanted or name == wanted):
            return RobotBinding.resolve(
                name=name,
                model=str(state.descriptor.model),
                namespace=str(state.descriptor.ns),
                frame_prefix=str(state.descriptor.frame),
                odom_topic_template=odom_topic_template,
            )
    return None


FULL_SCALE_SINE_RMS = 1.0 / math.sqrt(2.0)
ACTIVE_LEVEL_WINDOW_S = 0.01
ACTIVE_LEVEL_GATE_DB = -20.0
REFERENCE_SPL_DB = 94.0


def rms(samples: NDArray[np.floating], *, axis: int | None = None) -> NDArray[np.float64] | float:
    values = np.asarray(samples, dtype=np.float64)
    return np.sqrt(np.mean(values * values, axis=axis)) if values.size else 0.0


def dbfs_from_rms(value: float, *, floor_db: float = -120.0) -> float:
    """Level in dBFS where a full-scale sine reads 0 dBFS."""
    return max(20.0 * math.log10(max(float(value), 1e-12) / FULL_SCALE_SINE_RMS), floor_db)


def rms_from_dbfs(level_dbfs: float) -> float:
    """RMS of a sine at level_dbfs."""
    return FULL_SCALE_SINE_RMS * 10.0 ** (float(level_dbfs) / 20.0)


def active_rms(samples: NDArray[np.floating], sample_rate: int) -> float:
    """RMS of the 10 ms windows within 20 dB of the loudest, framed from the first non-silent frame."""
    audio = np.asarray(samples, dtype=np.float64)
    power = audio * audio if audio.ndim == 1 else np.mean(audio * audio, axis=1)
    sounding = np.flatnonzero(power)
    if sounding.size == 0:
        return 0.0
    power = power[sounding[0] : sounding[-1] + 1]
    window = max(round(sample_rate * ACTIVE_LEVEL_WINDOW_S), 1)
    starts = np.arange(0, power.size, window)
    sums = np.add.reduceat(power, starts)
    lengths = np.diff(np.append(starts, power.size))
    gated = sums / lengths >= np.max(sums / lengths) * 10.0 ** (ACTIVE_LEVEL_GATE_DB / 10.0)
    return float(np.sqrt(np.sum(sums[gated]) / np.sum(lengths[gated])))


def spl_to_dbfs(spl_db: float, sensitivity_dbfs_at_94_dbspl: float) -> float:
    """Digital level of a sound pressure level through a fixed MEMS sensitivity."""
    return float(spl_db) - REFERENCE_SPL_DB + float(sensitivity_dbfs_at_94_dbspl)


def dbfs_to_spl(level_dbfs: float, sensitivity_dbfs_at_94_dbspl: float) -> float:
    return float(level_dbfs) + REFERENCE_SPL_DB - float(sensitivity_dbfs_at_94_dbspl)
