"""Every auditory node parameter, grouped. Defaults live only here."""

from __future__ import annotations

import enum
import functools
import math
import typing

from arena_rclpy_mixins.params import Param, ParamGroup, PerRole, Range, floats
from arena_robots.audio import Vec3
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


def _sides(value: object) -> tuple[Vec3, Vec3]:
    flat = floats(value)
    if len(flat) != 6:
        raise ValueError(f"{value!r} is not two x y z points, left then right")
    return (flat[0], flat[1], flat[2]), (flat[3], flat[4], flat[5])


class EnvGroup(ParamGroup):
    NS = Param[str]("env.ns", "", "Env namespace of the pedestrian topics (empty = none).")


class WorldGroup(ParamGroup):
    CEILING_HEIGHT_M = Param[float]("world.ceiling_height_m", 3.0, "Ceiling height in m of rooms that carry no authored height.", constraint=Range(lo=0.0, lo_open=True))
    COVERAGE_ENABLED = Param[bool]("world.coverage.enabled", True, "Audit the acoustic zone coverage of the world map at startup.")
    COVERAGE_STRIDE_CELLS = Param[int]("world.coverage.stride_cells", 10, "Coverage audit samples every this many free map cells.", constraint=Range(lo=0))
    COVERAGE_TOLERANCE_M = Param[float]("world.coverage.tolerance_m", 0.35, "Distance in m a sampled free cell may be from every zone before it counts as uncovered.", constraint=Range(lo=0.0))
    MAP_SOURCE = Param[MapSource]("debug.map_source", MapSource.COMPUTE, "Occupancy map the acoustic world is built from, computed in process or read from the level preview map on disk.")


class MapGroup(ParamGroup):
    FRAME = Param[str]("map.frame", "map", "Frame of the occupancy map.")
    OCCUPIED_THRESHOLD = Param[int]("map.occupied_threshold", 50, "Occupancy grid value from which a cell counts as occupied.", constraint=Range(lo=0))


class RirGroup(ParamGroup):
    SAMPLE_RATE_HZ = Param[int]("rir.sample_rate_hz", 44100, "Sample rate in Hz of the rendered room impulse responses.", constraint=Range(lo=1))
    MAX_ORDER = Param[int]("rir.max_order", 3, "Image-source reflection order of every RIR.", constraint=Range(lo=0))
    TEMPERATURE_C = Param[float]("rir.temperature_c", 20.0, "Air temperature in degrees Celsius that sets the speed of sound.", constraint=Range(lo=-273.15, lo_open=True))
    RELATIVE_HUMIDITY_PCT = Param[float]("rir.relative_humidity_pct", 50.0, "Relative humidity in percent used for air absorption.", constraint=Range(lo=0.0, hi=100.0))
    QUANTIZATION_M = Param[float]("rir.quantization_m", 0.10, "Grid in m that source and listener positions snap to for the RIR cache key.", constraint=Range(lo=0.0, lo_open=True))
    CACHE_SIZE = Param[int]("rir.cache_size", 512, "Number of room impulse responses the cache keeps.", constraint=Range(lo=1))


class PortalGroup(ParamGroup):
    ADJACENCY_TOLERANCE_M = Param[float]("portal.adjacency_tolerance_m", 0.2, "Largest gap in m between two room outlines that still counts as adjacent.", constraint=Range(lo=0.0, lo_open=True))
    INSET_M = Param[float]("portal.inset_m", 0.03, "Distance in m a room outline is shrunk inward when testing portal endpoints.", constraint=Range(lo=0.0, lo_open=True))
    DOOR_LOSS_DB = Param[float]("portal.door_loss_db", 3.0, "Transmission loss in dB of a closed door portal.", constraint=Range(lo=0.0))
    OPENING_LOSS_DB = Param[float]("portal.opening_loss_db", 0.5, "Transmission loss in dB of an open wall opening.", constraint=Range(lo=0.0))
    CORNER_REFERENCE_HZ = Param[float]("portal.corner_reference_hz", 500.0, "Reference frequency in Hz of the diffraction loss at a route corner.", constraint=Range(lo=0.0, lo_open=True))
    OPENINGS_ENABLED = Param[bool]("portal.openings.enabled", True, "Treat wall openings without a door as portals.")
    MIN_OPENING_WIDTH_M = Param[float]("portal.min_opening_width_m", 0.2, "Narrowest wall opening in m that counts as a portal.", constraint=Range(lo=0.0, lo_open=True))
    MULTI_HOP_ENABLED = Param[bool]("portal.multi_hop.enabled", True, "Allow pyroomacoustics RIR rendering across multi-hop door and opening portal routes.")
    MAX_HOPS = Param[int]("portal.max_hops", 4, "Most portals a composed route may cross.", constraint=Range(lo=1))
    ROUTE_LOSS_DB_PER_M = Param[float]("portal.route_loss_db_per_m", 0.05, "Extra loss in dB per m of route length beyond the direct distance.", constraint=Range(lo=0.0))
    EARLY_WINDOW_S = Param[float]("portal.early_window_s", 0.08, "Duration in s of the early reflections that rooms along a route add to a composed RIR.", constraint=Range(lo=0.0, lo_open=True))
    MAX_RIR_DURATION_S = Param[float]("portal.max_rir_duration_s", 2.0, "Longest composed room impulse response in s.", constraint=Range(lo=0.0, lo_open=True))
    QUANTIZATION_M = Param[float]("portal.quantization_m", 0.10, "Grid in m that positions snap to for the portal route cache key.", constraint=Range(lo=0.0, lo_open=True))
    CACHE_SIZE = Param[int]("portal.cache_size", 256, "Number of portal routes the cache keeps.", constraint=Range(lo=1))


class Level3Group(ParamGroup):
    MAX_REFLECTIONS = Param[int]("level3.max_reflections", 8, "Most reflection paths the level3 backend traces per source and listener pair.", constraint=Range(lo=0))
    REFLECTION_FLOOR_DB = Param[float]("level3.reflection_floor_db", -60.0, "Level in dB below which the level3 backend drops a reflection.")


class PropagationGroup(ParamGroup):
    ENABLED = Param[bool]("propagation.enabled", True, "Run sound propagation, off silences every reception.")
    BACKEND = Param[BackendName]("propagation.backend", BackendName.PYROOMACOUSTICS, "Propagation backend, pyroomacoustics, level3 or legacy.")
    THRESHOLD_DB = Param[float]("propagation.threshold_db", 20.0, "Level in dB a reception must reach to count as audible.")
    MIN_DISTANCE_M = Param[float]("propagation.min_distance_m", 1.0, "Distance in m below which the spreading loss stops growing.", constraint=Range(lo=0.0, lo_open=True))
    OCCLUSION_DB = Param[float]("propagation.occlusion_db", 20.0, "Attenuation in dB of a source with no line of sight to the listener, legacy backend.")
    INAUDIBLE_ENABLED = Param[bool]("propagation.inaudible.enabled", True, "Publish receptions below the audibility threshold as inaudible instead of dropping them.")
    SELF_HEARING_ENABLED = Param[bool]("propagation.self_hearing.enabled", True, "Let robot listeners receive their own emissions.")
    BUFFER_ENABLED = Param[bool]("propagation.buffer.enabled", True, "Buffer emissions that arrive before their listener is known.")
    BUFFER_SIZE = Param[int]("propagation.buffer.size", 128, "Most emissions the buffer holds (0 = none).", constraint=Range(lo=0))
    BUFFER_MAX_AGE_S = Param[float]("propagation.buffer.max_age_s", 1.0, "Age in s after which a buffered emission is dropped.", constraint=Range(lo=0.0, lo_open=True))
    PEDESTRIAN_LISTENERS_ENABLED = Param[bool]("pedestrian_listeners.enabled", False, "Pedestrians are propagation listeners and receive sound stimuli through the human simulator.")
    PEDESTRIAN_LISTENERS_DISCRETE_ENABLED = Param[bool]("pedestrian_listeners.discrete.enabled", False, "Publish discrete pedestrian sound stimuli, defaults to the value of viz.enabled in the launch file.")
    PEDESTRIAN_EAR_HEIGHT_M = Param[float]("pedestrian_listeners.ear_height_m", 1.6, "Ear height in m of pedestrian listeners.", constraint=Range(lo=0.0))
    MICROPHONES = Param[str]("microphones", "[]", "YAML list of robot microphone mappings (owner, robot, placement, frame, index).")
    VIEWPORT_HEIGHT_M = Param[float]("viewport.height_m", 1.6, "Listening height of the viewport camera's down-projection microphone.", constraint=Range(lo=0.0))


class ListenerGroup(ParamGroup):
    ID = Param[str]("listener.id", "", "Microphone listener id that feeds the listener renderer, the RViz auditory panel switches it at run time.")


class MotorGroup(ParamGroup):
    ENABLED = Param[bool]("motor.enabled", True, "Let robots emit drivetrain audio. Robots stay listeners regardless.")
    MODEL = Param[MotorModel]("motor.model", MotorModel.PROCEDURAL, "Robot motor audio, calibrated procedural synthesis (Jackal, other models use WAVs) or WAV loops.")
    TRIM_DB = Param[float]("motor.trim_db", 0.0, "Live offset in dB on the motor asset level, lower it to attenuate ego-noise.")
    FREQUENCY_SCALE = Param[float]("motor.frequency_scale", 1.0, "Factor on the pitch of the procedural motor sound.", constraint=Range(lo=0.0, lo_open=True))
    TONAL_GAIN_DB = Param[float]("motor.tonal_gain_db", 0.0, "Gain in dB on the tonal part of the procedural motor sound.")
    BROADBAND_GAIN_DB = Param[float]("motor.broadband_gain_db", -12.0, "Gain in dB on the broadband part of the procedural motor sound.")
    SPEED_EXPONENT = Param[float]("motor.speed_exponent", 1.0, "Exponent on the speed ratio that scales the motor sound level.")
    VELOCITY_SMOOTHING_S = Param[float]("motor.velocity_smoothing_s", 0.015, "Time constant in s that smooths the robot velocity driving the motor sound.", constraint=Range(lo=0.0))
    STRUCTURAL_SHARE = Param[float]("motor.structural_share", 0.2, "Share of a robot's own motor power at each of its array microphones that arrives structure-borne, incoherent across the microphones. 0 leaves the airborne field from the drive points alone.", constraint=Range(lo=0.0, hi=1.0))
    DRIVE_POINTS_M = Param[tuple[Vec3, Vec3]]("motor.drive_points_m", [0.0, 0.1878, 0.0345, 0.0, -0.1878, 0.0345], "Points the left and right drivetrain of a robot radiate from to its own array, x y z left then right in the array mount frame. The default is the Jackal's mid-wheel points.", parse=_sides)


class OutputGroup(ParamGroup):
    DEVICE = Param[str]("output.device", "auto", "PortAudio output device for workstation playback. auto tries pulse, pipewire, default, then the PortAudio default. none starts no listener renderer.")
    BLOCK_SIZE = Param[int]("output.block_size", 512, "Workstation audio callback block size.", constraint=Range(lo=0))
    BUFFER_S = Param[float]("output.buffer_s", 0.04, "Workstation jitter buffer target in seconds, raise it on repeated underflows.", constraint=Range(lo=0.0, lo_open=True))
    RETRY_PERIOD_S = Param[float]("output.retry_period_s", 2.0, "Seconds between attempts to open the output device after a failure.", constraint=Range(lo=0.0, lo_open=True))
    ENABLED = Param[bool]("output.enabled", True, "Play audio on the workstation, off starts no listener renderer output.")
    MOTOR_ENABLED = Param[bool]("output.motor.enabled", True, "Play robot motor audio on the workstation.")
    AMBIENT_ENABLED = Param[bool]("output.ambient.enabled", True, "Play environment audio on the workstation. Emission and robot hearing continue when false.")


class MonitorGroup(ParamGroup):
    ENABLED = Param[bool]("monitor.enabled", True, "Mix the array microphones down to the monitor stereo stream.")
    MASTER_GAIN_DB = Param[float]("monitor.master_gain_db", PerRole(array=20.0 * math.log10(0.8), listener=0.0), "Master gain in dB on the monitor mix, defaults per render role.")
    GAIN_DB = Param[float]("monitor.gain_db", 36.0, "Gain in dB on the monitor mix before limiting.", constraint=Range(lo=0.0, hi=60.0))
    LIMIT = Param[float]("monitor.limit", 0.98, "Peak limit of the monitor mix as a fraction of full scale.", constraint=Range(lo=0.0, hi=1.0, lo_open=True))
    FRONT_GAIN = Param[float]("monitor.front_gain", 1.0, "Linear gain of the front microphones in the stereo monitor mix.", constraint=Range(lo=0.0, hi=4.0))
    REAR_GAIN = Param[float]("monitor.rear_gain", PerRole(array=0.75, listener=1.0), "Linear gain of the rear microphones in the stereo monitor mix, defaults per render role.", constraint=Range(lo=0.0, hi=4.0))
    SOLO = Param[str]("monitor.solo", "", "Microphone label routed alone to the monitor (empty = all).")
    MODE = Param[MonitorMode]("monitor.mode", MonitorMode.STEREO, "Monitor mix, stereo downmix of the array or the hearing mix.")


class RoleGroup(ParamGroup):
    ROLE = Param[RenderRole]("render.role", RenderRole.ARRAY, "Renderer role, array renders robot microphones and listener renders the workstation playback.")


class RenderGroup(ParamGroup):
    MAX_CATCHUP_BLOCKS = Param[int]("render.max_catchup_blocks", 40, "Most audio blocks the renderer produces at once to catch up after a stall.", constraint=Range(lo=0))
    RIR_ENABLED = Param[bool]("render.rir.enabled", PerRole(array=False, listener=True), "Convolve sources with their room impulse response, defaults per render role.")
    RIR_CROSSFADE_S = Param[float]("render.rir.crossfade_s", 0.1, "Crossfade in s between two room impulse responses of one source.", constraint=Range(lo=0.0))
    RIR_DRY_FALLBACK_ENABLED = Param[bool]("render.rir.dry_fallback.enabled", True, "Play the dry source while its room impulse response is not ready.")
    INAUDIBLE_ENABLED = Param[bool]("render.inaudible.enabled", PerRole(array=False, listener=True), "Render receptions flagged inaudible, defaults per render role.")
    MIN_LEVEL_DB = Param[float]("render.min_level_db", PerRole(array=-120.0, listener=-20.0), "Level in dB below which the renderer skips a reception, defaults per render role.")
    LOCKSTEP_ENABLED = Param[bool]("render.lockstep.enabled", PerRole(array=True, listener=False), "Render in lockstep with the simulator clock, defaults per render role.")
    CONTINUOUS_MAX_AGE_S = Param[float]("render.continuous.max_age_s", 2.0, "Age in s after which a continuous source without new receptions is dropped.", constraint=Range(lo=0.0, lo_open=True))
    BED_ENABLED = Param[bool]("render.bed.enabled", False, "Add a diffuse ambient bed to every robot array: seamless pink noise from 20 Hz, independent per microphone, in stem_ambient.")
    BED_LEVEL_DBA = Param[float]("render.bed.level_dba", 40.0, "A-weighted level of the ambient bed at each array microphone, dB SPL through the array sensitivity.")


class ArrayGroup(ParamGroup):
    SPEC = Param[str]("array.spec", "stereo", "Robot microphone array, stereo, four_mic, mono or a yaml path. Empty is four_mic when robot.hearing is srp or seld.")
    MOUNT_FRAME = Param[str]("array.mount_frame", "", "TF frame the robot microphone array is mounted on, {prefix} and {base_frame} expand, a bare leaf joins the robot prefix. Empty uses the robot base frame.", parse=_frame_template)
    ROBOT = Param[str]("array.robot", "", "Robot the array belongs to (empty = all robots).")
    ENABLED = Param[bool]("array.enabled", True, "Render the robot microphone array.")
    MUTED = Param[bool]("array.muted", False, "Mute the array monitor output, the receptions still feed hearing.")


class TdoaGroup(ParamGroup):
    ENABLED = Param[bool]("tdoa.enabled", True, "Estimate TDOA bearings between the array microphone pairs.")
    MAX_LAG_S = Param[float]("tdoa.max_lag_s", 0.002, "Largest inter-microphone lag in s the GCC-PHAT search considers.", constraint=Range(lo=0.0, lo_open=True))


class DiagnosticsGroup(ParamGroup):
    PERIOD_S = Param[float]("diagnostics.period_s", 5.0, "Seconds between diagnostics log lines.", constraint=Range(lo=0.0, lo_open=True))
    REPORT_PERIOD_S = Param[float]("diagnostics.report_period_s", 0.5, "Seconds between diagnostics reports on the topic.", constraint=Range(lo=0.0, lo_open=True))
    ACTIVITY_THRESHOLD_DBFS = Param[float]("diagnostics.activity_threshold_dbfs", -70.0, "Level in dBFS above which a channel counts as active.")


class HumanGroup(ParamGroup):
    WALKING_SPEED_MPS = Param[float]("human.walking_speed_mps", 0.05, "Speed in m/s from which a pedestrian emits footsteps.", constraint=Range(lo=0.0))
    FOOTSTEP_INTERVAL_S = Param[float]("human.footstep_interval_s", 0.45, "Seconds between footsteps of a walking pedestrian.", constraint=Range(lo=0.0, lo_open=True))
    GREETING_DISTANCE_M = Param[float]("human.greeting.distance_m", 1.5, "Distance in m within which a pedestrian greets a robot.", constraint=Range(lo=0.0))
    GREETING_FOV_DEG = Param[float]("human.greeting.fov_deg", 90.0, "Field of view in degrees in which a pedestrian greets a robot.", constraint=Range(lo=0.0, hi=360.0))
    GREETING_COOLDOWN_S = Param[float]("human.greeting.cooldown_s", 5.0, "Seconds before a pedestrian greets the same robot again.", constraint=Range(lo=0.0))


class DrivetrainGroup(ParamGroup):
    PERIOD_S = Param[float]("drivetrain.period_s", 0.05, "Seconds between drivetrain emission updates.", constraint=Range(lo=0.0, lo_open=True))
    ODOM_TOPIC_TEMPLATE = Param[str]("drivetrain.odom_topic_template", DEFAULT_ODOM_TOPIC_TEMPLATE, "Odometry topic of a robot, {namespace} and {name} expand.")
    ANGULAR_SCALE_M = Param[float]("drivetrain.angular_scale_m", 0.25, "Lever arm in m that turns angular velocity into equivalent wheel speed.", constraint=Range(lo=0.0, lo_open=True))
    MOTION_GATE_ENABLED = Param[bool]("drivetrain.motion_gate.enabled", True, "Gate the drivetrain sound on robot motion.")
    MOTION_GATE_START_MPS = Param[float]("drivetrain.motion_gate.start_mps", 0.05, "Speed in m/s above which the drivetrain sound starts.", constraint=Range(lo=0.0))
    MOTION_GATE_STOP_MPS = Param[float]("drivetrain.motion_gate.stop_mps", 0.03, "Speed in m/s below which the drivetrain sound stops.", constraint=Range(lo=0.0))
    MARKERS_ENABLED = Param[bool]("drivetrain.markers.enabled", True, "Publish drivetrain sound markers.")
    MARKERS_LIFETIME_S = Param[float]("drivetrain.markers.lifetime_s", 0.8, "Lifetime in s of a drivetrain marker.", constraint=Range(lo=0.0, lo_open=True))
    MARKERS_Z_M = Param[float]("drivetrain.markers.z_m", 0.16, "Height in m of the drivetrain markers.")
    MARKERS_LINE_WIDTH_M = Param[float]("drivetrain.markers.line_width_m", 0.055, "Line width in m of the drivetrain markers.", constraint=Range(lo=0.0, lo_open=True))
    MARKERS_CONE_DEG = Param[float]("drivetrain.markers.cone_deg", 70.0, "Opening angle in degrees of the drivetrain marker cone.", constraint=Range(lo=10.0, hi=180.0))
    MARKERS_RANGE_M = Param[float]("drivetrain.markers.range_m", 1.25, "Length in m of the drivetrain marker cone.", constraint=Range(lo=0.0, lo_open=True))


class BusGroup(ParamGroup):
    IGNORE_SELF_ENABLED = Param[bool]("bus.ignore_self.enabled", True, "Drop receptions of a robot's own emissions on the bus.")
    MIN_SNR_DB = Param[float]("bus.min_snr_db", -5.0, "Level in dB above the audibility threshold a reception needs to reach the bus.")
    DELAY_ENABLED = Param[bool]("bus.delay.enabled", True, "Delay bus receptions by the propagation time of the sound.")
    CONTINUOUS_GAP_S = Param[float]("bus.continuous.gap_s", 0.5, "A continuous source silent this long ends its audible run, so its next reception is a new detection.", constraint=Range(lo=0.0, lo_open=True))
    MARKERS_LIFETIME_S = Param[float]("bus.markers.lifetime_s", 1.5, "Lifetime in s of a bus reception marker.", constraint=Range(lo=0.0, lo_open=True))
    MARKERS_Z_M = Param[float]("bus.markers.z_m", 1.2, "Height in m of the bus reception markers.")
    MARKERS_TEXT_HEIGHT_M = Param[float]("bus.markers.text_height_m", 0.35, "Text height in m of the bus reception markers.", constraint=Range(lo=0.0, lo_open=True))


class VizGroup(ParamGroup):
    ENABLED = Param[bool]("viz.enabled", False, "Publish source, portal and listener propagation markers.")
    LIFETIME_S = Param[float]("viz.lifetime_s", 5.0, "Lifetime in s of a propagation marker.", constraint=Range(lo=0.0, lo_open=True))
    CONTINUOUS_LIFETIME_S = Param[float]("viz.continuous_lifetime_s", 0.5, "Lifetime in s of a continuous source marker.", constraint=Range(lo=0.0, lo_open=True))
    LISTENER_ID = Param[str]("viz.listener_id", "", "Listener whose receptions the markers show (empty = all).")
    PLOT_MODE = Param[PlotMode]("viz.plot.mode", PlotMode.OFF, "Live acoustic plot dashboard, off or live.")
    PLOT_LISTENER_ID = Param[str]("viz.plot.listener_id", "", "Listener the plot shows (empty = viz.listener_id).")
    PLOT_RATE_HZ = Param[float]("viz.plot.rate_hz", 2.0, "Plot refresh rate in Hz.", constraint=Range(lo=0.0, lo_open=True))
    PLOT_QUANTIZATION_M = Param[float]("viz.plot.quantization_m", 0.10, "Grid in m that positions snap to for the plot RIR cache key.", constraint=Range(lo=0.0, lo_open=True))
    PLOT_ENERGY_BIN_MS = Param[float]("viz.plot.energy_bin_ms", 5.0, "Width in ms of the plot energy bins.", constraint=Range(lo=0.0, lo_open=True))
    PLOT_EARLY_WINDOW_S = Param[float]("viz.plot.early_window_s", 0.08, "Duration in s of the early window of the plot.", constraint=Range(lo=0.0, lo_open=True))


GROUPS: tuple[type[ParamGroup], ...] = (EnvGroup, WorldGroup, MapGroup, RirGroup, PortalGroup, Level3Group, PropagationGroup, ListenerGroup, MotorGroup, OutputGroup, MonitorGroup, RoleGroup, RenderGroup, ArrayGroup, TdoaGroup, DiagnosticsGroup, HumanGroup, DrivetrainGroup, BusGroup, VizGroup)


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
        return MonitorGroup(self._server, role=self.Role.ROLE.value)

    @functools.cached_property
    def Role(self) -> RoleGroup:
        return RoleGroup(self._server)

    @functools.cached_property
    def Render(self) -> RenderGroup:
        return RenderGroup(self._server, role=self.Role.ROLE.value)

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
