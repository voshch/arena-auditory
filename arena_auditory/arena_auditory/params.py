"""Every auditory node parameter, grouped. Defaults live only here."""

from __future__ import annotations

import enum
import functools
import math
import typing

from arena_rclpy_mixins.param_groups import Param, ParamGroup, PerRole, count, finite, non_negative, positive, positive_count, within
from arena_robots.fleet import DEFAULT_ODOM_TOPIC_TEMPLATE

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer


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


def _frame_template(value: object) -> str:
    template = str(value).strip().strip("/")
    try:
        template.format(prefix="", base_frame="")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"frame template {value!r} may only use {{prefix}} and {{base_frame}}") from exc
    return template


class EnvGroup(ParamGroup):
    NS = Param[str]("env.ns", "")


class WorldGroup(ParamGroup):
    CEILING_HEIGHT_M = Param[float]("world.ceiling_height_m", 3.0, parse=positive)
    COVERAGE_ENABLED = Param[bool]("world.coverage.enabled", True)
    COVERAGE_STRIDE_CELLS = Param[int]("world.coverage.stride_cells", 10, parse=count)
    COVERAGE_TOLERANCE_M = Param[float]("world.coverage.tolerance_m", 0.35, parse=non_negative)
    MAP_SOURCE = Param[MapSource]("debug.map_source", MapSource.COMPUTE.value, parse=MapSource)


class MapGroup(ParamGroup):
    FRAME = Param[str]("map.frame", "map")
    OCCUPIED_THRESHOLD = Param[int]("map.occupied_threshold", 50, parse=count)


class RirGroup(ParamGroup):
    SAMPLE_RATE_HZ = Param[int]("rir.sample_rate_hz", 44100, parse=positive_count)
    MAX_ORDER = Param[int]("rir.max_order", 3, parse=count, description="Image-source reflection order of every RIR.")
    TEMPERATURE_C = Param[float]("rir.temperature_c", 20.0, parse=within(-273.15, math.inf, lo_open=True))
    RELATIVE_HUMIDITY_PCT = Param[float]("rir.relative_humidity_pct", 50.0, parse=within(0.0, 100.0))
    QUANTIZATION_M = Param[float]("rir.quantization_m", 0.10, parse=positive)
    CACHE_SIZE = Param[int]("rir.cache_size", 512, parse=positive_count)


class PortalGroup(ParamGroup):
    ADJACENCY_TOLERANCE_M = Param[float]("portal.adjacency_tolerance_m", 0.2, parse=positive)
    INSET_M = Param[float]("portal.inset_m", 0.03, parse=positive)
    DOOR_LOSS_DB = Param[float]("portal.door_loss_db", 3.0, parse=non_negative)
    OPENING_LOSS_DB = Param[float]("portal.opening_loss_db", 0.5, parse=non_negative)
    OPENINGS_ENABLED = Param[bool]("portal.openings.enabled", True)
    MIN_OPENING_WIDTH_M = Param[float]("portal.min_opening_width_m", 0.2, parse=positive)
    MULTI_HOP_ENABLED = Param[bool]("portal.multi_hop.enabled", True, description="Allow pyroomacoustics RIR rendering across multi-hop door and opening portal routes.")
    MAX_HOPS = Param[int]("portal.max_hops", 4, parse=positive_count)
    ROUTE_LOSS_DB_PER_M = Param[float]("portal.route_loss_db_per_m", 0.05, parse=non_negative)
    EARLY_WINDOW_S = Param[float]("portal.early_window_s", 0.08, parse=positive)
    MAX_RIR_DURATION_S = Param[float]("portal.max_rir_duration_s", 2.0, parse=positive)
    QUANTIZATION_M = Param[float]("portal.quantization_m", 0.10, parse=positive)
    CACHE_SIZE = Param[int]("portal.cache_size", 256, parse=positive_count)


class Level3Group(ParamGroup):
    MAX_REFLECTIONS = Param[int]("level3.max_reflections", 8, parse=count)
    REFLECTION_FLOOR_DB = Param[float]("level3.reflection_floor_db", -60.0, parse=finite)


class PropagationGroup(ParamGroup):
    ENABLED = Param[bool]("propagation.enabled", True)
    BACKEND = Param[BackendName]("propagation.backend", BackendName.PYROOMACOUSTICS.value, parse=BackendName, description="Propagation backend, pyroomacoustics, level3 or legacy.")
    THRESHOLD_DB = Param[float]("propagation.threshold_db", 20.0, parse=finite)
    MIN_DISTANCE_M = Param[float]("propagation.min_distance_m", 1.0, parse=positive)
    OCCLUSION_DB = Param[float]("propagation.occlusion_db", 20.0, parse=finite)
    INAUDIBLE_ENABLED = Param[bool]("propagation.inaudible.enabled", True)
    SELF_HEARING_ENABLED = Param[bool]("propagation.self_hearing.enabled", True)
    BUFFER_ENABLED = Param[bool]("propagation.buffer.enabled", True)
    BUFFER_SIZE = Param[int]("propagation.buffer.size", 128, parse=count)
    BUFFER_MAX_AGE_S = Param[float]("propagation.buffer.max_age_s", 1.0, parse=positive)
    PEDESTRIAN_LISTENERS_ENABLED = Param[bool]("pedestrian_listeners.enabled", False, description="Pedestrians are propagation listeners and receive sound stimuli through the human simulator.")
    PEDESTRIAN_LISTENERS_DISCRETE_ENABLED = Param[bool]("pedestrian_listeners.discrete.enabled", False)
    PEDESTRIAN_EAR_HEIGHT_M = Param[float]("pedestrian_listeners.ear_height_m", 1.6, parse=non_negative)
    MICROPHONES = Param[str]("microphones", "[]", description="YAML list of robot microphone mappings (owner, robot, placement, frame, index).")
    VIEWPORT_HEIGHT_M = Param[float]("viewport.height_m", 1.6, parse=non_negative, description="Listening height of the viewport camera's down-projection microphone.")


class ListenerGroup(ParamGroup):
    ID = Param[str]("listener.id", "", description="Microphone listener id that feeds the listener renderer, the RViz auditory panel switches it at run time.")


class MotorGroup(ParamGroup):
    ENABLED = Param[bool]("motor.enabled", True, description="Let robots emit drivetrain audio. Robots stay listeners regardless.")
    MODEL = Param[MotorModel]("motor.model", MotorModel.PROCEDURAL.value, parse=MotorModel, description="Robot motor audio, calibrated procedural synthesis (Jackal, other models use WAVs) or WAV loops.")
    TRIM_DB = Param[float]("motor.trim_db", 0.0, parse=finite, description="Live offset in dB on the motor asset level, lower it to attenuate ego-noise.")
    FREQUENCY_SCALE = Param[float]("motor.frequency_scale", 1.0, parse=positive)
    TONAL_GAIN_DB = Param[float]("motor.tonal_gain_db", 0.0, parse=finite)
    BROADBAND_GAIN_DB = Param[float]("motor.broadband_gain_db", -12.0, parse=finite)
    SPEED_EXPONENT = Param[float]("motor.speed_exponent", 1.0, parse=finite)
    VELOCITY_SMOOTHING_S = Param[float]("motor.velocity_smoothing_s", 0.015, parse=non_negative)


class OutputGroup(ParamGroup):
    DEVICE = Param[str]("output.device", "auto", description="PortAudio output device for workstation playback. auto tries pulse, pipewire, default, then the PortAudio default. none starts no listener renderer.")
    BLOCK_SIZE = Param[int]("output.block_size", 512, parse=count, description="Workstation audio callback block size.")
    BUFFER_S = Param[float]("output.buffer_s", 0.04, parse=positive, description="Workstation jitter buffer target in seconds, raise it on repeated underflows.")
    RETRY_PERIOD_S = Param[float]("output.retry_period_s", 2.0, parse=positive)
    ENABLED = Param[bool]("output.enabled", True)
    MOTOR_ENABLED = Param[bool]("output.motor.enabled", True, description="Play robot motor audio on the workstation.")
    AMBIENT_ENABLED = Param[bool]("output.ambient.enabled", True, description="Play environment audio on the workstation. Emission and robot hearing continue when false.")


class MonitorGroup(ParamGroup):
    ENABLED = Param[bool]("monitor.enabled", True)
    MASTER_GAIN_DB = Param[float]("monitor.master_gain_db", PerRole(array=20.0 * math.log10(0.8), listener=0.0), parse=finite)
    GAIN_DB = Param[float]("monitor.gain_db", 36.0, parse=within(0.0, 60.0))
    LIMIT = Param[float]("monitor.limit", 0.98, parse=within(0.0, 1.0, lo_open=True))
    FRONT_GAIN = Param[float]("monitor.front_gain", 1.0, parse=within(0.0, 4.0))
    REAR_GAIN = Param[float]("monitor.rear_gain", PerRole(array=0.75, listener=1.0), parse=within(0.0, 4.0))
    SOLO = Param[str]("monitor.solo", "")
    MODE = Param[MonitorMode]("monitor.mode", MonitorMode.STEREO.value, parse=MonitorMode)


class RenderGroup(ParamGroup):
    ROLE = Param[RenderRole]("render.role", RenderRole.ARRAY.value, parse=RenderRole)
    MAX_CATCHUP_BLOCKS = Param[int]("render.max_catchup_blocks", 40, parse=count)
    RIR_ENABLED = Param[bool]("render.rir.enabled", PerRole(array=False, listener=True))
    RIR_CROSSFADE_S = Param[float]("render.rir.crossfade_s", 0.1, parse=non_negative)
    RIR_DRY_FALLBACK_ENABLED = Param[bool]("render.rir.dry_fallback.enabled", True)
    INAUDIBLE_ENABLED = Param[bool]("render.inaudible.enabled", PerRole(array=False, listener=True))
    MIN_LEVEL_DB = Param[float]("render.min_level_db", PerRole(array=-120.0, listener=-20.0), parse=finite)
    LOCKSTEP_ENABLED = Param[bool]("render.lockstep.enabled", PerRole(array=True, listener=False))

    def role(self) -> RenderRole:
        return self.ROLE.value


class ArrayGroup(ParamGroup):
    SPEC = Param[str]("array.spec", "stereo", description="Robot microphone array, stereo, four_mic, mono or a yaml path. Empty is four_mic when robot.hearing is srp or seld.")
    MOUNT_FRAME = Param[str]("array.mount_frame", "", parse=_frame_template, description="TF frame the robot microphone array is mounted on, {prefix} and {base_frame} expand, a bare leaf joins the robot prefix. Empty uses the robot base frame.")
    ROBOT = Param[str]("array.robot", "")
    ENABLED = Param[bool]("array.enabled", True)
    MUTED = Param[bool]("array.muted", False)


class TdoaGroup(ParamGroup):
    ENABLED = Param[bool]("tdoa.enabled", True)
    MAX_LAG_S = Param[float]("tdoa.max_lag_s", 0.002, parse=positive)


class DiagnosticsGroup(ParamGroup):
    PERIOD_S = Param[float]("diagnostics.period_s", 5.0, parse=positive)
    REPORT_PERIOD_S = Param[float]("diagnostics.report_period_s", 0.5, parse=positive)
    ACTIVITY_THRESHOLD_DBFS = Param[float]("diagnostics.activity_threshold_dbfs", -70.0, parse=finite)


class HumanGroup(ParamGroup):
    WALKING_SPEED_MPS = Param[float]("human.walking_speed_mps", 0.05, parse=non_negative)
    FOOTSTEP_INTERVAL_S = Param[float]("human.footstep_interval_s", 0.45, parse=positive)
    GREETING_DISTANCE_M = Param[float]("human.greeting.distance_m", 1.5, parse=non_negative)
    GREETING_FOV_DEG = Param[float]("human.greeting.fov_deg", 90.0, parse=within(0.0, 360.0))
    GREETING_COOLDOWN_S = Param[float]("human.greeting.cooldown_s", 5.0, parse=non_negative)


class DrivetrainGroup(ParamGroup):
    PERIOD_S = Param[float]("drivetrain.period_s", 0.05, parse=positive)
    ODOM_TOPIC_TEMPLATE = Param[str]("drivetrain.odom_topic_template", DEFAULT_ODOM_TOPIC_TEMPLATE)
    ANGULAR_SCALE_M = Param[float]("drivetrain.angular_scale_m", 0.25, parse=positive)
    MOTION_GATE_ENABLED = Param[bool]("drivetrain.motion_gate.enabled", True)
    MOTION_GATE_START_MPS = Param[float]("drivetrain.motion_gate.start_mps", 0.05, parse=non_negative)
    MOTION_GATE_STOP_MPS = Param[float]("drivetrain.motion_gate.stop_mps", 0.03, parse=non_negative)
    MARKERS_ENABLED = Param[bool]("drivetrain.markers.enabled", True)
    MARKERS_LIFETIME_S = Param[float]("drivetrain.markers.lifetime_s", 0.8, parse=positive)
    MARKERS_Z_M = Param[float]("drivetrain.markers.z_m", 0.16, parse=finite)
    MARKERS_LINE_WIDTH_M = Param[float]("drivetrain.markers.line_width_m", 0.055, parse=positive)
    MARKERS_CONE_DEG = Param[float]("drivetrain.markers.cone_deg", 70.0, parse=within(10.0, 180.0))
    MARKERS_RANGE_M = Param[float]("drivetrain.markers.range_m", 1.25, parse=positive)


class BusGroup(ParamGroup):
    IGNORE_SELF_ENABLED = Param[bool]("bus.ignore_self.enabled", True)
    MIN_SNR_DB = Param[float]("bus.min_snr_db", -5.0, parse=finite)
    DELAY_ENABLED = Param[bool]("bus.delay.enabled", True)
    MARKERS_LIFETIME_S = Param[float]("bus.markers.lifetime_s", 1.5, parse=positive)
    MARKERS_Z_M = Param[float]("bus.markers.z_m", 1.2, parse=finite)
    MARKERS_TEXT_HEIGHT_M = Param[float]("bus.markers.text_height_m", 0.35, parse=positive)


class VizGroup(ParamGroup):
    ENABLED = Param[bool]("viz.enabled", False, description="Publish source, portal and listener propagation markers.")
    LIFETIME_S = Param[float]("viz.lifetime_s", 5.0, parse=positive)
    CONTINUOUS_LIFETIME_S = Param[float]("viz.continuous_lifetime_s", 0.5, parse=positive)
    LISTENER_ID = Param[str]("viz.listener_id", "")
    PLOT_MODE = Param[PlotMode]("viz.plot.mode", PlotMode.OFF.value, parse=PlotMode)
    PLOT_LISTENER_ID = Param[str]("viz.plot.listener_id", "")
    PLOT_RATE_HZ = Param[float]("viz.plot.rate_hz", 2.0, parse=positive)
    PLOT_QUANTIZATION_M = Param[float]("viz.plot.quantization_m", 0.10, parse=positive)
    PLOT_ENERGY_BIN_MS = Param[float]("viz.plot.energy_bin_ms", 5.0, parse=positive)
    PLOT_EARLY_WINDOW_S = Param[float]("viz.plot.early_window_s", 0.08, parse=positive)


GROUPS: tuple[type[ParamGroup], ...] = (EnvGroup, WorldGroup, MapGroup, RirGroup, PortalGroup, Level3Group, PropagationGroup, ListenerGroup, MotorGroup, OutputGroup, MonitorGroup, RenderGroup, ArrayGroup, TdoaGroup, DiagnosticsGroup, HumanGroup, DrivetrainGroup, BusGroup, VizGroup)


def all_params() -> dict[str, Param[object]]:
    """Every declared parameter by full name."""
    return {param.name: param for group in GROUPS for param in group.params()}


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
