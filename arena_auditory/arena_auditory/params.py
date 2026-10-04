"""Every auditory node parameter, grouped. Defaults live only here."""

from __future__ import annotations

import enum
import functools
import math
import typing
from collections.abc import Callable, Iterable, Mapping

import attrs
import yaml

from arena_auditory.shared import DEFAULT_ODOM_TOPIC_TEMPLATE

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer, ROSParamT


class RenderRole(enum.StrEnum):
    ARRAY = "array"
    LISTENER = "listener"


class MapSource(enum.StrEnum):
    COMPUTE = "compute"
    DISK = "disk"


class BackendName(enum.StrEnum):
    PYROOMACOUSTICS = "pyroomacoustics"
    LEVEL3 = "level3"
    LEGACY = "legacy"


class MotorModel(enum.StrEnum):
    PROCEDURAL = "procedural"
    WAV = "wav"


class MonitorMode(enum.StrEnum):
    STEREO = "stereo"
    HEARING = "hearing"


class PlotMode(enum.StrEnum):
    OFF = "off"
    LIVE = "live"


class Frontend(enum.StrEnum):
    BUS = "bus"
    SRP = "srp"
    SELD = "seld"

    @property
    def array_spec(self) -> str:
        """Array spec the front-end runs on unless auditory.array.spec names one, empty for the bus."""
        return "" if self is Frontend.BUS else "four_mic"


class BearingSource(enum.StrEnum):
    GCC = "gcc"
    SELD = "seld"


@attrs.frozen
class PerRole[T]:
    """Default that differs between the array and the listener renderer."""

    array: T
    listener: T

    def pick(self, role: RenderRole) -> T:
        return self.array if role is RenderRole.ARRAY else self.listener


def _finite(value: object) -> float:
    result = float(typing.cast(float, value))
    if not math.isfinite(result):
        raise ValueError(f"{value!r} is not finite")
    return result


def _within(lo: float, hi: float, *, lo_open: bool = False) -> Callable[[object], float]:
    def parse(value: object) -> float:
        result = _finite(value)
        if result < lo or result > hi or (lo_open and result == lo):
            raise ValueError(f"{result} is outside {'(' if lo_open else '['}{lo}, {hi}]")
        return result

    return parse


def _positive(value: object) -> float:
    result = _finite(value)
    if result <= 0.0:
        raise ValueError(f"{result} must be positive")
    return result


def _non_negative(value: object) -> float:
    result = _finite(value)
    if result < 0.0:
        raise ValueError(f"{result} must not be negative")
    return result


def _count(value: object) -> int:
    result = int(typing.cast(int, value))
    if result < 0:
        raise ValueError(f"{result} must not be negative")
    return result


def _positive_count(value: object) -> int:
    result = _count(value)
    if result == 0:
        raise ValueError(f"{result} must be positive")
    return result


def _floats(value: object) -> tuple[float, ...]:
    return tuple(_finite(v) for v in typing.cast(Iterable[float], value))


def _names(value: object) -> tuple[str, ...]:
    return tuple(str(v).strip().lower() for v in typing.cast(Iterable[str], value) if str(v).strip())


def _frame_template(value: object) -> str:
    template = str(value).strip().strip("/")
    try:
        template.format(prefix="", base_frame="")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"frame template {value!r} may only use {{prefix}} and {{base_frame}}") from exc
    return template


def _coerce_value(default: object, raw: str) -> object:
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    loaded = yaml.safe_load(raw)
    items = loaded if isinstance(loaded, list) else [loaded]
    element = type(default[0]) if default else str
    return [element(item) for item in items]


class Param[T]:
    """Parameter declaration. On a group class it is the declaration, on a group instance the live ROSParam."""

    def __init__(self, name: str, default: T | PerRole[T], *, parse: Callable[[object], T] | None = None) -> None:
        self.name = name
        self.default = default
        self.parse = parse
        self.attr = ""

    def __set_name__(self, owner: type, attr: str) -> None:
        self.attr = attr

    @property
    def field(self) -> str:
        """Config field name, the lowercase declaration name."""
        return self.attr.lower()

    def coerce(self, raw: str) -> object:
        """Launch string as this parameter's type, list elements as the default's element type."""
        default = self.default.array if isinstance(self.default, PerRole) else self.default
        match default:
            case bool():
                value = yaml.safe_load(raw)
                if not isinstance(value, bool):
                    raise ValueError(f"{self.name} expects true or false, got {raw!r}")
                return value
            case int() | float() | list() | tuple():
                try:
                    return _coerce_value(default, raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{self.name} expects {type(default).__name__}, got {raw!r}") from exc
            case _:
                return raw

    @typing.overload
    def __get__(self, instance: None, owner: type) -> Param[T]: ...

    @typing.overload
    def __get__(self, instance: ParamGroup, owner: type) -> ROSParamT[T]: ...

    def __get__(self, instance: ParamGroup | None, owner: type) -> Param[T] | ROSParamT[T]:
        if instance is None:
            return self
        return instance._live[self.attr]


class ParamGroup:
    """Parameter group, declares its parameters on the node at construction."""

    def __init__(self, server: ROSParamServer, *, role: RenderRole | None = None) -> None:
        self._server = server
        self._role = role
        self._live: dict[str, ROSParamT[object]] = {}
        for param in type(self).params():
            default = param.default
            if isinstance(default, PerRole):
                default = default.pick(self.role())
            self._live[param.attr] = server.ROSParam(param.name, default, parse=param.parse)

    def role(self) -> RenderRole:
        if self._role is None:
            raise TypeError(f"{type(self).__name__} has role-dependent defaults and needs a render role")
        return self._role

    @classmethod
    def params(cls) -> tuple[Param[object], ...]:
        return tuple(value for klass in reversed(cls.__mro__) for value in vars(klass).values() if isinstance(value, Param))

    def values(self) -> dict[str, object]:
        """Live value of every parameter by field name."""
        return {param.field: self._live[param.attr].value for param in type(self).params()}

    @classmethod
    def defaults(cls) -> dict[str, object]:
        """Parsed default of every parameter by field name, role-dependent ones left out."""
        return {param.field: param.parse(param.default) if param.parse is not None else param.default for param in cls.params() if not isinstance(param.default, PerRole)}


class EnvGroup(ParamGroup):
    NS = Param[str]("env.ns", "")


class WorldGroup(ParamGroup):
    CEILING_HEIGHT_M = Param[float]("world.ceiling_height_m", 3.0, parse=_positive)
    COVERAGE_ENABLED = Param[bool]("world.coverage.enabled", True)
    COVERAGE_STRIDE_CELLS = Param[int]("world.coverage.stride_cells", 10, parse=_count)
    COVERAGE_TOLERANCE_M = Param[float]("world.coverage.tolerance_m", 0.35, parse=_non_negative)
    MAP_SOURCE = Param[MapSource]("debug.map_source", MapSource.COMPUTE.value, parse=MapSource)


class MapGroup(ParamGroup):
    FRAME = Param[str]("map.frame", "map")
    OCCUPIED_THRESHOLD = Param[int]("map.occupied_threshold", 50, parse=_count)


class RirGroup(ParamGroup):
    SAMPLE_RATE_HZ = Param[int]("rir.sample_rate_hz", 44100, parse=_positive_count)
    MAX_ORDER = Param[int]("rir.max_order", 3, parse=_count)
    TEMPERATURE_C = Param[float]("rir.temperature_c", 20.0, parse=_within(-273.15, math.inf, lo_open=True))
    RELATIVE_HUMIDITY_PCT = Param[float]("rir.relative_humidity_pct", 50.0, parse=_within(0.0, 100.0))
    QUANTIZATION_M = Param[float]("rir.quantization_m", 0.10, parse=_positive)
    CACHE_SIZE = Param[int]("rir.cache_size", 512, parse=_positive_count)


class PortalGroup(ParamGroup):
    ADJACENCY_TOLERANCE_M = Param[float]("portal.adjacency_tolerance_m", 0.2, parse=_positive)
    INSET_M = Param[float]("portal.inset_m", 0.03, parse=_positive)
    DOOR_LOSS_DB = Param[float]("portal.door_loss_db", 3.0, parse=_non_negative)
    OPENING_LOSS_DB = Param[float]("portal.opening_loss_db", 0.5, parse=_non_negative)
    OPENINGS_ENABLED = Param[bool]("portal.openings.enabled", True)
    MIN_OPENING_WIDTH_M = Param[float]("portal.min_opening_width_m", 0.2, parse=_positive)
    MULTI_HOP_ENABLED = Param[bool]("portal.multi_hop.enabled", True)
    MAX_HOPS = Param[int]("portal.max_hops", 4, parse=_positive_count)
    ROUTE_LOSS_DB_PER_M = Param[float]("portal.route_loss_db_per_m", 0.05, parse=_non_negative)
    EARLY_WINDOW_S = Param[float]("portal.early_window_s", 0.08, parse=_positive)
    MAX_RIR_DURATION_S = Param[float]("portal.max_rir_duration_s", 2.0, parse=_positive)
    QUANTIZATION_M = Param[float]("portal.quantization_m", 0.10, parse=_positive)
    CACHE_SIZE = Param[int]("portal.cache_size", 256, parse=_positive_count)


class Level3Group(ParamGroup):
    MAX_REFLECTIONS = Param[int]("level3.max_reflections", 8, parse=_count)
    REFLECTION_FLOOR_DB = Param[float]("level3.reflection_floor_db", -60.0, parse=_finite)


class PropagationGroup(ParamGroup):
    ENABLED = Param[bool]("propagation.enabled", True)
    BACKEND = Param[BackendName]("propagation.backend", BackendName.PYROOMACOUSTICS.value, parse=BackendName)
    THRESHOLD_DB = Param[float]("propagation.threshold_db", 20.0, parse=_finite)
    MIN_DISTANCE_M = Param[float]("propagation.min_distance_m", 1.0, parse=_positive)
    OCCLUSION_DB = Param[float]("propagation.occlusion_db", 20.0, parse=_finite)
    INAUDIBLE_ENABLED = Param[bool]("propagation.inaudible.enabled", True)
    SELF_HEARING_ENABLED = Param[bool]("propagation.self_hearing.enabled", True)
    BUFFER_ENABLED = Param[bool]("propagation.buffer.enabled", True)
    BUFFER_SIZE = Param[int]("propagation.buffer.size", 128, parse=_count)
    BUFFER_MAX_AGE_S = Param[float]("propagation.buffer.max_age_s", 1.0, parse=_positive)
    PEDESTRIAN_LISTENERS_ENABLED = Param[bool]("pedestrian_listeners.enabled", False)
    PEDESTRIAN_LISTENERS_DISCRETE_ENABLED = Param[bool]("pedestrian_listeners.discrete.enabled", False)
    PEDESTRIAN_EAR_HEIGHT_M = Param[float]("pedestrian_listeners.ear_height_m", 1.6, parse=_non_negative)
    MICROPHONES = Param[str]("microphones", "[]")
    VIEWPORT_HEIGHT_M = Param[float]("viewport.height_m", 1.6, parse=_non_negative)


class ListenerGroup(ParamGroup):
    ID = Param[str]("listener.id", "")


class MotorGroup(ParamGroup):
    ENABLED = Param[bool]("motor.enabled", True)
    MODEL = Param[MotorModel]("motor.model", MotorModel.PROCEDURAL.value, parse=MotorModel)
    TRIM_DB = Param[float]("motor.trim_db", 0.0, parse=_finite)
    FREQUENCY_SCALE = Param[float]("motor.frequency_scale", 1.0, parse=_positive)
    TONAL_GAIN_DB = Param[float]("motor.tonal_gain_db", 0.0, parse=_finite)
    BROADBAND_GAIN_DB = Param[float]("motor.broadband_gain_db", -12.0, parse=_finite)
    SPEED_EXPONENT = Param[float]("motor.speed_exponent", 1.0, parse=_finite)
    VELOCITY_SMOOTHING_S = Param[float]("motor.velocity_smoothing_s", 0.015, parse=_non_negative)


class OutputGroup(ParamGroup):
    DEVICE = Param[str]("output.device", "auto")
    BLOCK_SIZE = Param[int]("output.block_size", 512, parse=_count)
    BUFFER_S = Param[float]("output.buffer_s", 0.04, parse=_positive)
    RETRY_PERIOD_S = Param[float]("output.retry_period_s", 2.0, parse=_positive)
    ENABLED = Param[bool]("output.enabled", True)
    MOTOR_ENABLED = Param[bool]("output.motor.enabled", True)
    AMBIENT_ENABLED = Param[bool]("output.ambient.enabled", True)


class MonitorGroup(ParamGroup):
    ENABLED = Param[bool]("monitor.enabled", True)
    MASTER_GAIN_DB = Param[float]("monitor.master_gain_db", PerRole(array=20.0 * math.log10(0.8), listener=0.0), parse=_finite)
    GAIN_DB = Param[float]("monitor.gain_db", 36.0, parse=_within(0.0, 60.0))
    LIMIT = Param[float]("monitor.limit", 0.98, parse=_within(0.0, 1.0, lo_open=True))
    FRONT_GAIN = Param[float]("monitor.front_gain", 1.0, parse=_within(0.0, 4.0))
    REAR_GAIN = Param[float]("monitor.rear_gain", PerRole(array=0.75, listener=1.0), parse=_within(0.0, 4.0))
    SOLO = Param[str]("monitor.solo", "")
    MODE = Param[MonitorMode]("monitor.mode", MonitorMode.STEREO.value, parse=MonitorMode)


class RenderGroup(ParamGroup):
    ROLE = Param[RenderRole]("render.role", RenderRole.ARRAY.value, parse=RenderRole)
    MAX_CATCHUP_BLOCKS = Param[int]("render.max_catchup_blocks", 40, parse=_count)
    RIR_ENABLED = Param[bool]("render.rir.enabled", PerRole(array=False, listener=True))
    RIR_CROSSFADE_S = Param[float]("render.rir.crossfade_s", 0.1, parse=_non_negative)
    RIR_DRY_FALLBACK_ENABLED = Param[bool]("render.rir.dry_fallback.enabled", True)
    INAUDIBLE_ENABLED = Param[bool]("render.inaudible.enabled", PerRole(array=False, listener=True))
    MIN_LEVEL_DB = Param[float]("render.min_level_db", PerRole(array=-120.0, listener=-20.0), parse=_finite)
    LOCKSTEP_ENABLED = Param[bool]("render.lockstep.enabled", PerRole(array=True, listener=False))

    def role(self) -> RenderRole:
        return self.ROLE.value


class ArrayGroup(ParamGroup):
    SPEC = Param[str]("array.spec", "stereo")
    MOUNT_FRAME = Param[str]("array.mount_frame", "", parse=_frame_template)
    ROBOT = Param[str]("array.robot", "")
    ENABLED = Param[bool]("array.enabled", True)
    MUTED = Param[bool]("array.muted", False)


class TdoaGroup(ParamGroup):
    ENABLED = Param[bool]("tdoa.enabled", True)
    MAX_LAG_S = Param[float]("tdoa.max_lag_s", 0.002, parse=_positive)


class DiagnosticsGroup(ParamGroup):
    PERIOD_S = Param[float]("diagnostics.period_s", 5.0, parse=_positive)
    REPORT_PERIOD_S = Param[float]("diagnostics.report_period_s", 0.5, parse=_positive)
    ACTIVITY_THRESHOLD_DBFS = Param[float]("diagnostics.activity_threshold_dbfs", -70.0, parse=_finite)


class HumanGroup(ParamGroup):
    WALKING_SPEED_MPS = Param[float]("human.walking_speed_mps", 0.05, parse=_non_negative)
    FOOTSTEP_INTERVAL_S = Param[float]("human.footstep_interval_s", 0.45, parse=_positive)
    GREETING_DISTANCE_M = Param[float]("human.greeting.distance_m", 1.5, parse=_non_negative)
    GREETING_FOV_DEG = Param[float]("human.greeting.fov_deg", 90.0, parse=_within(0.0, 360.0))
    GREETING_COOLDOWN_S = Param[float]("human.greeting.cooldown_s", 5.0, parse=_non_negative)


class DrivetrainGroup(ParamGroup):
    PERIOD_S = Param[float]("drivetrain.period_s", 0.05, parse=_positive)
    ODOM_TOPIC_TEMPLATE = Param[str]("drivetrain.odom_topic_template", DEFAULT_ODOM_TOPIC_TEMPLATE)
    ANGULAR_SCALE_M = Param[float]("drivetrain.angular_scale_m", 0.25, parse=_positive)
    MOTION_GATE_ENABLED = Param[bool]("drivetrain.motion_gate.enabled", True)
    MOTION_GATE_START_MPS = Param[float]("drivetrain.motion_gate.start_mps", 0.05, parse=_non_negative)
    MOTION_GATE_STOP_MPS = Param[float]("drivetrain.motion_gate.stop_mps", 0.03, parse=_non_negative)
    MARKERS_ENABLED = Param[bool]("drivetrain.markers.enabled", True)
    MARKERS_LIFETIME_S = Param[float]("drivetrain.markers.lifetime_s", 0.8, parse=_positive)
    MARKERS_Z_M = Param[float]("drivetrain.markers.z_m", 0.16, parse=_finite)
    MARKERS_LINE_WIDTH_M = Param[float]("drivetrain.markers.line_width_m", 0.055, parse=_positive)
    MARKERS_CONE_DEG = Param[float]("drivetrain.markers.cone_deg", 70.0, parse=_within(10.0, 180.0))
    MARKERS_RANGE_M = Param[float]("drivetrain.markers.range_m", 1.25, parse=_positive)


class BusGroup(ParamGroup):
    IGNORE_SELF_ENABLED = Param[bool]("bus.ignore_self.enabled", True)
    MIN_SNR_DB = Param[float]("bus.min_snr_db", -5.0, parse=_finite)
    DELAY_ENABLED = Param[bool]("bus.delay.enabled", True)
    MARKERS_LIFETIME_S = Param[float]("bus.markers.lifetime_s", 1.5, parse=_positive)
    MARKERS_Z_M = Param[float]("bus.markers.z_m", 1.2, parse=_finite)
    MARKERS_TEXT_HEIGHT_M = Param[float]("bus.markers.text_height_m", 0.35, parse=_positive)


class VizGroup(ParamGroup):
    ENABLED = Param[bool]("viz.enabled", False)
    LIFETIME_S = Param[float]("viz.lifetime_s", 5.0, parse=_positive)
    CONTINUOUS_LIFETIME_S = Param[float]("viz.continuous_lifetime_s", 0.5, parse=_positive)
    LISTENER_ID = Param[str]("viz.listener_id", "")
    PLOT_MODE = Param[PlotMode]("viz.plot.mode", PlotMode.OFF.value, parse=PlotMode)
    PLOT_LISTENER_ID = Param[str]("viz.plot.listener_id", "")
    PLOT_RATE_HZ = Param[float]("viz.plot.rate_hz", 2.0, parse=_positive)
    PLOT_QUANTIZATION_M = Param[float]("viz.plot.quantization_m", 0.10, parse=_positive)
    PLOT_ENERGY_BIN_MS = Param[float]("viz.plot.energy_bin_ms", 5.0, parse=_positive)
    PLOT_EARLY_WINDOW_S = Param[float]("viz.plot.early_window_s", 0.08, parse=_positive)


class HearingGroup(ParamGroup):
    FRONTEND = Param[Frontend]("hearing.frontend", Frontend.BUS.value, parse=Frontend)
    TG_NODE = Param[str]("hearing.tg_node", "task_generator_node")


class BeliefGroup(ParamGroup):
    MARKERS_ENABLED = Param[bool]("belief.markers.enabled", True)
    MARKERS_RANGE_M = Param[float]("belief.markers.range_m", 4.0, parse=_positive)
    MARKERS_MAX_COUNT = Param[int]("belief.markers.max_count", 12, parse=_count)
    PUBLISH_RATE_HZ = Param[float]("belief.publish_rate_hz", 5.0, parse=_positive)
    RESOLUTION_M = Param[float]("belief.resolution_m", 0.0, parse=_non_negative)
    STANDALONE_ENABLED = Param[bool]("belief.standalone.enabled", False)
    STANDALONE_ORIGIN_M = Param[tuple[float, ...]]("belief.standalone.origin_m", [-10.0, -10.0], parse=_floats)
    STANDALONE_SIZE_M = Param[tuple[float, ...]]("belief.standalone.size_m", [20.0, 20.0], parse=_floats)
    WEDGE_DEG = Param[float]("belief.wedge_deg", 10.0, parse=_positive)
    MIN_HALF_WIDTH_M = Param[float]("belief.min_half_width_m", 0.3, parse=_non_negative)
    TAU_S = Param[float]("belief.tau_s", 2.0, parse=_positive)
    MAX_RANGE_M = Param[float]("belief.max_range_m", 15.0, parse=_positive)
    MIN_RANGE_M = Param[float]("belief.min_range_m", 0.5, parse=_non_negative)
    REFERENCE_DISTANCE_M = Param[float]("belief.reference_distance_m", 1.0, parse=_positive)
    LEVEL_RANGE_ENABLED = Param[bool]("belief.level_range.enabled", False)
    RANGE_SIGMA_FRAC = Param[float]("belief.range_sigma_frac", 0.35, parse=_positive)
    RANGE_FLOOR_WEIGHT = Param[float]("belief.range_floor_weight", 0.15, parse=_non_negative)
    EVENT_MASS = Param[float]("belief.event_mass", 1.0, parse=_positive)
    EVENT_RATE_HZ = Param[float]("belief.event_rate_hz", 2.0, parse=_positive)
    MASS_FULL_SCALE = Param[float]("belief.mass_full_scale", 0.0, parse=_non_negative)
    KINDS = Param[tuple[str, ...]]("belief.kinds", ["footstep", "speech"], parse=_names)
    MIN_LEVEL_DB = Param[float]("belief.min_level_db", -1e9, parse=_finite)

    EMISSION_DB_PREFIX: typing.ClassVar[str] = "belief.emission_db."

    def emission_db(self, levels: Mapping[str, float]) -> dict[str, ROSParamT[float]]:
        """Declare belief.emission_db.<kind> per kind, defaulting to the level of the kind's default asset."""
        return {kind: self._server.ROSParam(f"{self.EMISSION_DB_PREFIX}{kind}", float(level), parse=_finite) for kind, level in levels.items()}


class PolicyGroup(ParamGroup):
    PUBLISH_RATE_HZ = Param[float]("policy.publish_rate_hz", 10.0, parse=_positive)
    MAX_LINEAR_MPS = Param[float]("policy.max_linear_mps", 0.0, parse=_non_negative)
    LISTEN_ENABLED = Param[bool]("policy.listen.enabled", True)
    YIELD_ENABLED = Param[bool]("policy.yield.enabled", True)
    LISTEN_MPS = Param[float]("policy.listen_mps", 0.2, parse=_non_negative)
    HOLD_MPS = Param[float]("policy.hold_mps", 0.03, parse=_non_negative)
    LOOKAHEAD_M = Param[float]("policy.lookahead_m", 4.0, parse=_non_negative)
    APPROACH_M = Param[float]("policy.approach_m", 4.0, parse=_non_negative)
    HOLD_LEN_M = Param[float]("policy.hold_len_m", 1.5, parse=_non_negative)
    HOLD_OFFSET_M = Param[float]("policy.hold_offset_m", 1.0, parse=_non_negative)
    LANE_RADIUS_M = Param[float]("policy.lane_radius_m", 0.6, parse=_non_negative)
    CORNER_RADIUS_M = Param[float]("policy.corner_radius_m", 2.0, parse=_non_negative)
    BEND_HYSTERESIS_M = Param[float]("policy.bend_hysteresis_m", 1.0, parse=_non_negative)
    REARM_AFTER_M = Param[float]("policy.rearm_after_m", 3.0, parse=_non_negative)
    YIELD_FRACTION = Param[float]("policy.yield_fraction", 0.5, parse=_within(0.0, 1.0))
    RELEASE_FRACTION = Param[float]("policy.release_fraction", 0.25, parse=_within(0.0, 1.0))
    MIN_YIELD_S = Param[float]("policy.min_yield_s", 3.0, parse=_non_negative)
    YIELD_TIMEOUT_S = Param[float]("policy.yield_timeout_s", 15.0, parse=_non_negative)
    RECEDE_S = Param[float]("policy.recede_s", 2.0, parse=_non_negative)
    LEVEL_TREND_DB_PER_S = Param[float]("policy.level_trend_db_per_s", -1.0, parse=_finite)
    LEVEL_TREND_TAU_S = Param[float]("policy.level_trend_tau_s", 5.0, parse=_positive)
    BELIEF_THRESHOLD = Param[float]("policy.belief_threshold", 0.6, parse=_within(0.0, 1.0))
    SPEED_MIN_PCT = Param[int]("policy.speed_min_pct", 40, parse=_count)
    SPEED_FREE_PCT = Param[int]("policy.speed_free_pct", 100, parse=_count)
    REACTION_RADIUS_M = Param[float]("policy.reaction_radius_m", 2.0, parse=_non_negative)


class AudioGroup(ParamGroup):
    RELIABLE_ENABLED = Param[bool]("audio.reliable.enabled", False)


class SeldGroup(ParamGroup):
    CHECKPOINT = Param[str]("seld.checkpoint", "")
    SCALER = Param[str]("seld.scaler", "")
    DEVICE = Param[str]("seld.device", "cuda")
    TORCH_THREADS = Param[int]("seld.torch_threads", 2, parse=_count)
    DET_THRESHOLD = Param[float]("seld.det_threshold", 0.5, parse=_within(0.0, 1.0))
    LOOKAHEAD_FRAMES = Param[int]("seld.lookahead_frames", 5, parse=_count)
    BEARING_SOURCE = Param[BearingSource]("seld.bearing_source", BearingSource.GCC.value, parse=BearingSource)


class SrpGroup(ParamGroup):
    HOP_S = Param[float]("srp.hop_s", 0.1, parse=_positive)
    FLOOR_WINDOW_S = Param[float]("srp.floor_window_s", 5.0, parse=_positive)
    ONSET_DB = Param[float]("srp.onset_db", 6.0, parse=_finite)


GROUPS: tuple[type[ParamGroup], ...] = (EnvGroup, WorldGroup, MapGroup, RirGroup, PortalGroup, Level3Group, PropagationGroup, ListenerGroup, MotorGroup, OutputGroup, MonitorGroup, RenderGroup, ArrayGroup, TdoaGroup, DiagnosticsGroup, HumanGroup, DrivetrainGroup, BusGroup, VizGroup, HearingGroup, BeliefGroup, PolicyGroup, AudioGroup, SeldGroup, SrpGroup)


def all_params() -> dict[str, Param[object]]:
    """Every declared parameter by full name, belief.emission_db.<kind> excluded."""
    return {param.name: param for group in GROUPS for param in group.params()}


def configure[C](config: type[C], *groups: ParamGroup | type[ParamGroup], **given: object) -> C:
    """An attrs config with every field not given taken from the group parameter of that field name, live from a group instance, the default from a group class."""
    values: dict[str, object] = {}
    for group in groups:
        values.update(group.values() if isinstance(group, ParamGroup) else group.defaults())
    values.update(given)
    return config(**{field.name: values[field.name] for field in attrs.fields(config) if field.init})


class Configuration:
    """Node parameters by group. A group declares its parameters on first access, so a node touches each group it uses in __init__."""

    def __init__(self, server: ROSParamServer) -> None:
        self._server = server

    def group[G: ParamGroup](self, cls: type[G]) -> G:
        """The instance of cls behind its named accessor, MotorGroup behind Motor."""
        group = getattr(self, cls.__name__.removesuffix("Group"))
        if not isinstance(group, cls):
            raise TypeError(f"{cls.__name__} has no accessor on {type(self).__name__}")
        return group

    @functools.cached_property
    def Env(self) -> EnvGroup:
        return EnvGroup(self._server)

    @functools.cached_property
    def World(self) -> WorldGroup:
        return WorldGroup(self._server)

    @functools.cached_property
    def Map(self) -> MapGroup:
        return MapGroup(self._server)

    @functools.cached_property
    def Rir(self) -> RirGroup:
        return RirGroup(self._server)

    @functools.cached_property
    def Portal(self) -> PortalGroup:
        return PortalGroup(self._server)

    @functools.cached_property
    def Level3(self) -> Level3Group:
        return Level3Group(self._server)

    @functools.cached_property
    def Propagation(self) -> PropagationGroup:
        return PropagationGroup(self._server)

    @functools.cached_property
    def Listener(self) -> ListenerGroup:
        return ListenerGroup(self._server)

    @functools.cached_property
    def Motor(self) -> MotorGroup:
        return MotorGroup(self._server)

    @functools.cached_property
    def Output(self) -> OutputGroup:
        return OutputGroup(self._server)

    @functools.cached_property
    def Monitor(self) -> MonitorGroup:
        return MonitorGroup(self._server, role=self.Render.ROLE.value)

    @functools.cached_property
    def Render(self) -> RenderGroup:
        return RenderGroup(self._server)

    @functools.cached_property
    def Array(self) -> ArrayGroup:
        return ArrayGroup(self._server)

    @functools.cached_property
    def Tdoa(self) -> TdoaGroup:
        return TdoaGroup(self._server)

    @functools.cached_property
    def Diagnostics(self) -> DiagnosticsGroup:
        return DiagnosticsGroup(self._server)

    @functools.cached_property
    def Human(self) -> HumanGroup:
        return HumanGroup(self._server)

    @functools.cached_property
    def Drivetrain(self) -> DrivetrainGroup:
        return DrivetrainGroup(self._server)

    @functools.cached_property
    def Bus(self) -> BusGroup:
        return BusGroup(self._server)

    @functools.cached_property
    def Viz(self) -> VizGroup:
        return VizGroup(self._server)

    @functools.cached_property
    def Hearing(self) -> HearingGroup:
        return HearingGroup(self._server)

    @functools.cached_property
    def Belief(self) -> BeliefGroup:
        return BeliefGroup(self._server)

    @functools.cached_property
    def Policy(self) -> PolicyGroup:
        return PolicyGroup(self._server)

    @functools.cached_property
    def Audio(self) -> AudioGroup:
        return AudioGroup(self._server)

    @functools.cached_property
    def Seld(self) -> SeldGroup:
        return SeldGroup(self._server)

    @functools.cached_property
    def Srp(self) -> SrpGroup:
        return SrpGroup(self._server)
