"""Propagates sound events and continuous sources to every listener and publishes receptions, and room impulses for the listeners some node renders with them."""

from __future__ import annotations

import copy
import functools
import json
import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Hashable, Iterable, Iterator

import attrs
import rclpy.time
import shapely
import tf2_ros
from arena_auditory_msgs.msg import (
    ContinuousAudioSourceState,
    ContinuousHeardSoundState,
    HeardSoundEvent,
    RoomImpulse,
    SoundEvent,
    SoundSource,
)
from arena_auditory_msgs.srv import RemoveMicrophone, SpawnMicrophone
from arena_people_msgs.msg import Pedestrians
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.param_groups import Param, configure
from arena_rclpy_mixins.qos import best_effort, latched, reliable
from arena_rclpy_mixins.transforms import ThreadedTransformListener
from arena_robots.audio import NS_PER_S, ArraySpec, Vec3, load_array_spec
from arena_robots.fleet import RobotBinding, robot_bindings
from arena_simulation_setup.tree.World import MICROPHONE_PLACEMENT_TOLERANCE_M
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, PoseStamped, Transform
from std_msgs.msg import ColorRGBA, Header, String
from task_generator_msgs.msg import RobotFleet
from visualization_msgs.msg import Marker, MarkerArray

from arena_auditory.constants import (
    ARENA_PEDS,
    CONTINUOUS_AUDIO_SOURCES,
    CONTINUOUS_HEARD_SOUNDS,
    HEARD_SOUND_EVENTS,
    MICROPHONE_LISTENERS,
    MICROPHONE_MARKERS,
    REMOVE_MICROPHONE,
    ROOM_IMPULSES,
    SOUND_EVENTS,
    SPAWN_MICROPHONE,
    STATE_ROBOTS,
    VIEWPORT_CAMERA_POSE,
    env_topic,
)
from arena_auditory.materials import default_catalog
from arena_auditory.params import BackendName, Configuration, Level3Group, PropagationGroup
from arena_auditory.propagation import IMPULSE_WINDOW, Emission, Listener, PortalConfig, PropagationConfig, PropagationScene, Propagator, Reception, RirConfig, to_point
from arena_auditory.shared import ListenerId, ListenerKind, MicrophoneScope, SourceSpec
from arena_auditory.world import AcousticWorld, WorldMicrophoneSpec, parse_robot_microphones
from arena_auditory.world_tracker import WorldTracker

CONTINUOUS_FLUSH_PERIOD_S = 0.05
CONTINUOUS_FLUSH_BUDGET_S = 0.01
MARKER_PERIOD_S = 0.25
WARNING_PERIOD_S = 5.0

DOWN_PROJECTION = ListenerId.viewport_mic("down_projection")
PROJECTIVE_CENTER = ListenerId.viewport_mic("projective_center")
VIEWPORT_IDS = (DOWN_PROJECTION, PROJECTIVE_CENTER)

MIC_COLORS = {
    "front_left": ColorRGBA(r=0.05, g=0.75, b=1.0, a=0.95),
    "front_right": ColorRGBA(r=1.0, g=0.65, b=0.05, a=0.95),
    "rear_left": ColorRGBA(r=0.35, g=0.45, b=1.0, a=0.95),
    "rear_right": ColorRGBA(r=1.0, g=0.25, b=0.45, a=0.95),
    "left": ColorRGBA(r=0.05, g=0.65, b=1.0, a=0.9),
    "right": ColorRGBA(r=1.0, g=0.55, b=0.05, a=0.9),
}
DEFAULT_MIC_COLOR = ColorRGBA(r=0.12, g=0.95, b=0.45, a=0.85)

LIVE_LEVEL_PARAMS: tuple[Param[object], ...] = (
    PropagationGroup.THRESHOLD_DB,
    PropagationGroup.MIN_DISTANCE_M,
    PropagationGroup.OCCLUSION_DB,
    Level3Group.MAX_REFLECTIONS,
    Level3Group.REFLECTION_FLOOR_DB,
)


@attrs.frozen
class _MarkerPose:
    position: Vec3
    frame: str
    yaw_rad: float = 0.0
    color: ColorRGBA = DEFAULT_MIC_COLOR


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _apply_transform(point: Vec3, transform_: Transform) -> Vec3:
    rotation = transform_.rotation
    translation = transform_.translation
    qx, qy, qz, qw = float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm > 0.0:
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm
    x, y, z = (float(value) for value in point)
    uv_x = qy * z - qz * y
    uv_y = qz * x - qx * z
    uv_z = qx * y - qy * x
    uuv_x = qy * uv_z - qz * uv_y
    uuv_y = qz * uv_x - qx * uv_z
    uuv_z = qx * uv_y - qy * uv_x
    return (
        x + 2.0 * (qw * uv_x + uuv_x) + float(translation.x),
        y + 2.0 * (qw * uv_y + uuv_y) + float(translation.y),
        z + 2.0 * (qw * uv_z + uuv_z) + float(translation.z),
    )


class SoundPropagationNode(ArenaMixinNode):
    def __init__(self) -> None:
        super().__init__("sound_propagation_node")
        self.conf = Configuration(self)
        self._params = self.conf.Propagation
        self._listener_id = self.conf.Listener.ID
        self._ceiling_height = self.conf.World.CEILING_HEIGHT_M
        self._quantization = self.conf.Rir.QUANTIZATION_M
        env_ns = self.conf.Env.NS.value.strip().rstrip("/")
        self._array: ArraySpec = load_array_spec(self.conf.Array.SPEC.value)
        self._mount_template = self.conf.Array.MOUNT_FRAME
        self._configured_microphones = parse_robot_microphones(self._params.MICROPHONES.value)
        config = configure(PropagationConfig, self.conf.Propagation, self.conf.Level3, rir=configure(RirConfig, self.conf.Rir), portal=configure(PortalConfig, self.conf.Portal))
        self._propagator = Propagator(config, default_catalog(), warn=lambda message: self.get_logger().warning(message))
        if self._propagator.init_fallback_reason:
            self.get_logger().warning(f"propagation falls back to level3: {self._propagator.init_fallback_reason}")

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = ThreadedTransformListener(self._tf_buffer)
        self._warning_times: dict[tuple[str, ...], float] = {}

        self._robots: tuple[RobotBinding, ...] = ()
        self._missing_microphone_robots: set[str] = set()
        self._peds: Pedestrians | None = None
        self._world_microphones: dict[str, WorldMicrophoneSpec] = {}
        self._spawned_microphones: dict[str, tuple[Vec3, str]] = {}
        self._spawned_index = 0
        self._viewport_pose: PoseStamped | None = None
        self._viewport_microphones: dict[str, tuple[Vec3, str]] = {}
        self._last_continuous: dict[tuple[str, str], ContinuousHeardSoundState] = {}
        self._continuous_cache: dict[tuple[str, str], tuple[tuple[Hashable, ...], Reception]] = {}
        self._pending_continuous: dict[str, ContinuousAudioSourceState] = {}
        self._pending_events: deque[tuple[SoundEvent, dict[str, Listener] | None]] = deque()
        self._reported_routes: set[tuple[str, str, str, str]] = set()

        self._heard_pub = self.create_publisher(HeardSoundEvent, HEARD_SOUND_EVENTS, reliable(50))
        self._continuous_pub = self.create_publisher(ContinuousHeardSoundState, CONTINUOUS_HEARD_SOUNDS, best_effort(64))
        self._impulse_pub = self.create_publisher(RoomImpulse, ROOM_IMPULSES, latched(256))
        self._announced_impulses: OrderedDict[str, None] = OrderedDict()
        self._registry_pub = self.create_publisher(String, MICROPHONE_LISTENERS, latched(1))
        self._marker_pub = self.create_publisher(MarkerArray, MICROPHONE_MARKERS, latched(1))
        self.create_service(SpawnMicrophone, SPAWN_MICROPHONE, self._spawn_microphone)
        self.create_service(RemoveMicrophone, REMOVE_MICROPHONE, self._remove_microphone)

        self._tracker = WorldTracker(self, on_ready=self._on_world_ready, on_clear=self._on_world_clear, on_episode=self._on_episode, conf=self.conf)

        self.create_subscription(SoundEvent, SOUND_EVENTS, self._on_sound_event, reliable(50))
        self.create_subscription(ContinuousAudioSourceState, CONTINUOUS_AUDIO_SOURCES, self._on_continuous_source, best_effort(64))
        self.create_subscription(Pedestrians, env_topic(env_ns, ARENA_PEDS), self._on_peds, 10)
        self.create_subscription(PoseStamped, VIEWPORT_CAMERA_POSE, self._on_viewport_pose, 10)
        self.create_subscription(RobotFleet, STATE_ROBOTS, self._on_fleet, latched(1))
        self.create_timer(CONTINUOUS_FLUSH_PERIOD_S, self._flush_continuous)
        self.create_timer(MARKER_PERIOD_S, self._publish_markers)
        self.create_timer(self.conf.Diagnostics.PERIOD_S.value, lambda: self.get_logger().debug(self._propagator.cache_summary()))

        self.add_param_callback(self._params.ENABLED.name, self._on_enabled)
        self.add_param_callback(self._listener_id.name, self._on_listener_selected)
        for param in LIVE_LEVEL_PARAMS:
            self.add_param_callback(param.name, functools.partial(self._retune, param))

        self._on_listener_selected(self._listener_id.value)
        self._publish_registry()

    def destroy_node(self) -> bool:
        self._tracker.destroy()
        self._tf_listener.close()
        return super().destroy_node()

    def _retune(self, param: Param[object], value: object) -> bool:
        parsed = param.parse(value) if param.parse is not None else value
        self._propagator.config = attrs.evolve(self._propagator.config, **{param.field: parsed})
        self._continuous_cache.clear()
        return True

    def _on_enabled(self, value: object) -> bool:
        if not bool(value):
            self._stop_outputs(lambda _listener_id: False, forget=False)
        return True

    def _on_listener_selected(self, value: object) -> bool:
        selected = str(value).strip()
        leaving_viewport = selected not in VIEWPORT_IDS
        self._stop_outputs(
            lambda listener_id: ListenerId.parse(listener_id).kind is not ListenerKind.MICROPHONE or listener_id == selected or (leaving_viewport and listener_id in VIEWPORT_IDS),
            forget=False,
        )
        if selected in VIEWPORT_IDS:
            if self._viewport_pose is not None:
                self._viewport_microphones = self._build_viewport_microphones(self._viewport_pose)
                self._publish_registry()
        else:
            self._stop_outputs(lambda listener_id: listener_id not in VIEWPORT_IDS, forget=True)
            if self._viewport_microphones:
                self._viewport_microphones = {}
                self._publish_registry()
        return True

    def _stopped(self, previous: ContinuousHeardSoundState) -> ContinuousHeardSoundState:
        stopped = copy.deepcopy(previous)
        stopped.header.stamp = self.get_clock().now().to_msg()
        stopped.source.active = False
        stopped.reception.audible = False
        return stopped

    def _stop_outputs(self, keep: Callable[[str], bool], *, forget: bool) -> None:
        """Publish an inactive, inaudible copy of every continuous output whose listener keep rejects."""
        for key, previous in tuple(self._last_continuous.items()):
            if keep(key[1]):
                continue
            stopped = self._stopped(previous)
            self._continuous_cache.pop(key, None)
            if forget:
                self._last_continuous.pop(key)
            else:
                self._last_continuous[key] = stopped
            self._continuous_pub.publish(stopped)

    def _on_world_ready(self, world: AcousticWorld) -> None:
        self._world_microphones = {microphone.listener_id: microphone for microphone in world.microphones}
        self._continuous_cache.clear()
        self._publish_registry()
        now_ns = self.get_clock().now().nanoseconds
        max_age_ns = round(self._params.BUFFER_MAX_AGE_S.value * NS_PER_S)
        stale = 0
        while self._pending_events:
            msg, listeners = self._pending_events.popleft()
            if now_ns - rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds > max_age_ns:
                stale += 1
                continue
            self._publish_event(msg, listeners)
        if stale:
            self.get_logger().info(f"dropped {stale} buffered sound events older than {self._params.BUFFER_MAX_AGE_S.value} s at world ready")

    def _on_world_clear(self) -> None:
        self._announced_impulses.clear()
        self._world_microphones.clear()
        self._spawned_microphones.clear()
        self._spawned_index = 0
        self._last_continuous.clear()
        self._continuous_cache.clear()
        self._publish_registry()

    def _on_episode(self, _episode_id: int) -> None:
        self._announced_impulses.clear()
        self._spawned_microphones.clear()
        self._spawned_index = 0
        self._last_continuous.clear()
        self._continuous_cache.clear()
        self._publish_registry()

    def _scene(self) -> PropagationScene:
        scene = self._propagator.scene
        if scene.world is not self._tracker.world or scene.occupancy is not self._tracker.occupancy:
            scene = PropagationScene(world=self._tracker.world, occupancy=self._tracker.occupancy)
            self._propagator.set_scene(scene)
        return scene

    def _acoustic_frame(self) -> str | None:
        occupancy = self._tracker.occupancy
        return occupancy.frame_id if occupancy is not None else None

    def _warn_throttled(self, key: tuple[str, ...], message: str) -> None:
        now = time.monotonic()
        if now - self._warning_times.get(key, -math.inf) < WARNING_PERIOD_S:
            return
        self._warning_times[key] = now
        self.get_logger().warning(message)

    def _transform(self, source_frame: str, target_frame: str, entity: str) -> Transform | None:
        source_frame = source_frame.strip().lstrip("/")
        target_frame = target_frame.strip().lstrip("/")
        if not source_frame or not target_frame:
            self._warn_throttled((entity, source_frame, target_frame), f"cannot transform acoustic position for {entity!r} from {source_frame!r} to {target_frame!r}: missing frame ID")
            return None
        if source_frame == target_frame:
            return Transform()
        try:
            return self._tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time()).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            self._warn_throttled((entity, source_frame, target_frame), f"cannot transform acoustic position for {entity!r} from {source_frame!r} to runtime map frame {target_frame!r}: {exc}")
            return None

    def _between(self, point: Vec3, source_frame: str, target_frame: str, entity: str) -> Vec3 | None:
        if source_frame.strip().lstrip("/") == target_frame.strip().lstrip("/") and source_frame.strip():
            return (float(point[0]), float(point[1]), float(point[2]))
        transform_ = self._transform(source_frame, target_frame, entity)
        return _apply_transform(point, transform_) if transform_ is not None else None

    def _in_acoustic_frame(self, point: Vec3, source_frame: str, entity: str) -> Vec3 | None:
        frame = self._acoustic_frame()
        return self._between(point, source_frame, frame, entity) if frame is not None else None

    def _mount_frame(self, binding: RobotBinding) -> str:
        return binding.mount(self._mount_template.value)

    def _robot_listeners(self) -> dict[str, Listener]:
        """robot:<r> at the array centroid and array:<r>:<mic> for every fleet robot, via TF."""
        frame = self._acoustic_frame()
        if frame is None:
            return {}
        listeners: dict[str, Listener] = {}
        for binding in self._robots:
            robot_id = ListenerId.robot(binding.name)
            transform_ = self._transform(self._mount_frame(binding), frame, robot_id)
            if transform_ is None:
                continue
            listeners[robot_id] = Listener(id=robot_id, kind=ListenerKind.ROBOT, owner=binding.name, position=_apply_transform(self._array.centroid_m, transform_))
            for mic in self._array.mics:
                mic_id = ListenerId.array_mic(binding.name, mic.name)
                listeners[mic_id] = Listener(id=mic_id, kind=ListenerKind.ARRAY, owner=binding.name, position=_apply_transform(mic.position_m, transform_))
        return listeners

    def _robot_frame_prefixes(self) -> dict[str, str]:
        return {binding.name: binding.frame_prefix for binding in self._robots}

    def _microphone_position(self, listener_id: str) -> Vec3 | None:
        """Acoustic-frame position of a configured, world, spawned or viewport microphone."""
        if listener_id in self._spawned_microphones:
            position, frame = self._spawned_microphones[listener_id]
            return self._in_acoustic_frame(position, frame, listener_id)
        if listener_id in self._viewport_microphones:
            position, frame = self._viewport_microphones[listener_id]
            return self._in_acoustic_frame(position, frame, listener_id)
        if listener_id in self._world_microphones:
            return self._world_microphone_position(self._world_microphones[listener_id])
        prefixes = self._robot_frame_prefixes()
        for spec in self._configured_microphones:
            if spec.listener_id == listener_id and spec.robot in prefixes:
                return self._in_acoustic_frame((0.0, 0.0, 0.0), spec.resolve_frame(prefixes[spec.robot]), listener_id)
        return None

    def _world_microphone_position(self, microphone: WorldMicrophoneSpec) -> Vec3 | None:
        world = self._tracker.world
        frame = self._acoustic_frame()
        if world is None or frame is None:
            return None
        position = microphone.position if microphone.frame == "map" else self._between(microphone.position, microphone.frame, frame, microphone.listener_id)
        if position is None:
            return None
        zone = world.zone_named(microphone.zone)
        if zone is None or not zone.polygon.buffer(MICROPHONE_PLACEMENT_TOLERANCE_M).covers(shapely.Point(position[0], position[1])):
            self._warn_throttled((microphone.listener_id, "zone"), f"microphone {microphone.listener_id!r} resolved outside declared zone {microphone.zone!r}")
            return None
        if microphone.ceiling_height_m is not None and not math.isclose(position[2], microphone.ceiling_height_m, abs_tol=MICROPHONE_PLACEMENT_TOLERANCE_M):
            self._warn_throttled((microphone.listener_id, "ceiling"), f"microphone {microphone.listener_id!r} resolved height does not match the declared zone ceiling")
            return None
        return position

    def _listeners_for(self, source: SourceSpec, *, include_peds: bool) -> dict[str, Listener]:
        listeners: dict[str, Listener] = {}
        if include_peds and self._params.PEDESTRIAN_LISTENERS_ENABLED.value and self._peds is not None:
            ear_height = self._params.PEDESTRIAN_EAR_HEIGHT_M.value
            peds_frame = str(self._peds.header.frame_id).strip() or "map"
            for ped in self._peds.pedestrians:
                if int(ped.id) == source.agent_id:
                    continue
                listener_id = ListenerId.agent(int(ped.id))
                ground = self._in_acoustic_frame((ped.pose.position.x, ped.pose.position.y, ped.pose.position.z), peds_frame, listener_id)
                if ground is not None:
                    listeners[listener_id] = Listener(id=listener_id, kind=ListenerKind.AGENT, owner="", position=(ground[0], ground[1], ear_height))
        listeners.update(self._robot_listeners())
        selected = self._listener_id.value.strip()
        if selected and selected not in listeners:
            try:
                parsed = ListenerId.parse(selected)
            except ValueError:
                parsed = None
            position = self._microphone_position(selected) if parsed is not None and parsed.kind is ListenerKind.MICROPHONE else None
            if position is not None:
                listeners[selected] = Listener.at(selected, position)
        if not self._params.SELF_HEARING_ENABLED.value:
            listeners = {listener_id: listener for listener_id, listener in listeners.items() if ListenerId.owner_robot(listener_id) != source.agent_name}
        return listeners

    def _source_in_frame(self, header: Header, source_msg: SoundSource, source: SourceSpec) -> tuple[Vec3, SoundSource, str] | None:
        """Source position in the acoustic frame, the source message with that position, and the frame."""
        frame = self._acoustic_frame()
        if frame is None:
            return None
        position = self._between(source.position, str(header.frame_id), frame, source.id or source.agent_name or "sound source")
        if position is None:
            return None
        if position != source.position:
            source_msg = copy.deepcopy(source_msg)
            source_msg.position = to_point(position)
        return position, source_msg, frame

    def _propagate_many(self, emission: Emission, listeners: Iterable[Listener]) -> Iterator[tuple[Listener, Reception]]:
        self._scene()
        for listener, reception in self._propagator.propagate_many(emission, listeners):
            self._report_route(emission, listener, reception)
            yield listener, reception

    def _report_route(self, emission: Emission, listener: Listener, reception: Reception) -> None:
        route = (reception.backend, reception.fallback_reason, reception.source_zone, reception.listener_zone)
        if route not in self._reported_routes:
            self._reported_routes.add(route)
            detail = f", fallback={reception.fallback_reason!r}" if reception.used_fallback else ""
            portal = f", portal={reception.portal_ids[0]!r}" if reception.portal_ids else ""
            message = (
                f"actual propagation backend={reception.backend!r} for {reception.source_zone!r}->{reception.listener_zone!r}{portal}{detail}, "
                f"source=({emission.position[0]:.2f},{emission.position[1]:.2f}) name={emission.agent_name!r}, "
                f"listener={listener.id!r}@({listener.position[0]:.2f},{listener.position[1]:.2f})"
            )
            if reception.used_fallback:
                self.get_logger().warning(message)
            else:
                self.get_logger().info(message)

    def _publish_impulse(self, reception: Reception, stamp: Time) -> None:
        impulse = reception.impulse
        if impulse is None:
            return
        self._propagator.impulses.put(impulse)
        if impulse.key in self._announced_impulses:
            return
        self._announced_impulses[impulse.key] = None
        while len(self._announced_impulses) > IMPULSE_WINDOW:
            self._announced_impulses.popitem(last=False)
        self._impulse_pub.publish(impulse.to_msg(stamp))

    def _source_spec(self, msg: SoundSource) -> SourceSpec | None:
        if not msg.kind.strip():
            return None
        try:
            return SourceSpec.from_msg(msg)
        except ValueError as exc:
            self._warn_throttled(("source", msg.id), f"dropping sound source {msg.id!r}: {exc}")
            return None

    def _on_sound_event(self, msg: SoundEvent) -> None:
        if not self._params.ENABLED.value:
            return
        source = self._source_spec(msg.source)
        if source is None:
            return
        if self._tracker.world is None and self._propagator.config.backend is BackendName.PYROOMACOUSTICS and self._params.BUFFER_ENABLED.value:
            maximum = max(self._params.BUFFER_SIZE.value, 1)
            if len(self._pending_events) >= maximum:
                dropped, _ = self._pending_events.popleft()
                self.get_logger().warning(f"acoustic scene event buffer full, dropping {dropped.source.id!r}")
            listeners = self._listeners_for(source, include_peds=self._params.PEDESTRIAN_LISTENERS_DISCRETE_ENABLED.value) if self._acoustic_frame() is not None else None
            self._pending_events.append((msg, listeners))
            return
        self._publish_event(msg, self._listeners_for(source, include_peds=self._params.PEDESTRIAN_LISTENERS_DISCRETE_ENABLED.value))

    def _publish_event(self, msg: SoundEvent, listeners: dict[str, Listener] | None) -> None:
        source = self._source_spec(msg.source)
        if source is None:
            return
        located = self._source_in_frame(msg.header, msg.source, source)
        if located is None:
            return
        position, source_msg, frame = located
        if listeners is None:
            listeners = self._listeners_for(source, include_peds=self._params.PEDESTRIAN_LISTENERS_DISCRETE_ENABLED.value)
        emission = Emission.from_source(source, position)
        header = Header(stamp=msg.header.stamp, frame_id=frame)
        for _, reception in self._propagate_many(emission, listeners.values()):
            if reception.audible or self._params.INAUDIBLE_ENABLED.value:
                self._publish_impulse(reception, msg.header.stamp)
                self._heard_pub.publish(reception.heard_msg(header, source_msg))

    def _on_continuous_source(self, msg: ContinuousAudioSourceState) -> None:
        if not self._params.ENABLED.value or not msg.source.kind.strip():
            return
        self._pending_continuous[msg.source.id] = msg

    def _flush_continuous(self) -> None:
        """Propagate pending continuous sources until CONTINUOUS_FLUSH_BUDGET_S has passed, the rest stay pending."""
        pending = self._pending_continuous
        if not pending:
            return
        self._pending_continuous = {}
        if not self._params.ENABLED.value:
            return
        if self._propagator.config.backend is BackendName.PYROOMACOUSTICS and self._tracker.world is None:
            return
        deadline = time.monotonic() + CONTINUOUS_FLUSH_BUDGET_S
        for flushed, (source_id, state) in enumerate(tuple(pending.items())):
            if flushed and time.monotonic() >= deadline:
                self._pending_continuous = pending | self._pending_continuous
                return
            del pending[source_id]
            self._flush_source(state)

    def _flush_source(self, state: ContinuousAudioSourceState) -> None:
        source = self._source_spec(state.source)
        if source is None:
            return
        located = self._source_in_frame(state.header, state.source, source)
        if located is None:
            return
        position, source_msg, frame = located
        emission = Emission.from_source(source, position)
        header = Header(stamp=state.header.stamp, frame_id=frame)
        quantum = self._quantization.value
        source_signature = (
            source.agent_id,
            source.agent_name,
            source.model,
            source.kind,
            source.asset_id,
            source.tags,
            round(float(source.level_db), 1),
            round(float(source.reference_distance_m), 2),
            *(round(float(value) / quantum) for value in position),
        )

        def signature(listener: Listener) -> tuple[Hashable, ...]:
            return (*source_signature, *(round(float(value) / quantum) for value in listener.position))

        misses: list[Listener] = []
        for listener in self._listeners_for(source, include_peds=True).values():
            cached = self._continuous_cache.get((source.id, listener.id))
            if cached is not None and cached[0] == signature(listener):
                self._publish_continuous(source.id, attrs.evolve(cached[1], listener=listener), state, header, source_msg)
            else:
                misses.append(listener)
        for listener, reception in self._propagate_many(emission, misses):
            self._continuous_cache[(source.id, listener.id)] = (signature(listener), reception)
            self._publish_continuous(source.id, reception, state, header, source_msg)

    def _publish_continuous(self, source_id: str, reception: Reception, state: ContinuousAudioSourceState, header: Header, source_msg: SoundSource) -> None:
        key = (source_id, reception.listener.id)
        if not reception.audible and not self._params.INAUDIBLE_ENABLED.value:
            previous = self._last_continuous.pop(key, None)
            if previous is not None:
                self._continuous_pub.publish(self._stopped(previous))
            return
        self._publish_impulse(reception, state.header.stamp)
        output = reception.continuous_msg(header, source_msg)
        self._last_continuous[key] = output
        self._continuous_pub.publish(output)

    def _on_peds(self, msg: Pedestrians) -> None:
        self._peds = msg

    def _on_fleet(self, msg: RobotFleet) -> None:
        self._robots = robot_bindings(msg)
        for binding in self._robots:
            if binding.error:
                self._warn_throttled(("robot", binding.name), binding.error)
        configured = {spec.robot for spec in self._configured_microphones}
        missing = configured - {binding.name for binding in self._robots}
        if missing != self._missing_microphone_robots:
            self._missing_microphone_robots = missing
            if missing:
                self.get_logger().warning(f"configured microphones reference robots absent from state/robots: {sorted(missing)}")
        self._publish_registry()

    def _build_viewport_microphones(self, pose: PoseStamped) -> dict[str, tuple[Vec3, str]]:
        frame = str(pose.header.frame_id).strip().lstrip("/")
        position = pose.pose.position
        height = self.conf.Propagation.VIEWPORT_HEIGHT_M.value
        return {
            DOWN_PROJECTION: ((float(position.x), float(position.y), height), frame),
            PROJECTIVE_CENTER: ((float(position.x), float(position.y), float(position.z)), frame),
        }

    def _on_viewport_pose(self, msg: PoseStamped) -> None:
        position = msg.pose.position
        if not str(msg.header.frame_id).strip().lstrip("/") or not _finite(position.x, position.y, position.z):
            return
        first = self._viewport_pose is None
        self._viewport_pose = msg
        if self._viewport_microphones or self._listener_id.value.strip() in VIEWPORT_IDS:
            first = first or not self._viewport_microphones
            self._viewport_microphones = self._build_viewport_microphones(msg)
        if first:
            self._publish_registry()

    def _microphone_ids(self) -> set[str]:
        prefixes = self._robot_frame_prefixes()
        ids = {ListenerId.array_mic(binding.name, mic.name) for binding in self._robots for mic in self._array.mics}
        ids |= {spec.listener_id for spec in self._configured_microphones if spec.robot in prefixes}
        ids |= set(self._world_microphones) | set(self._spawned_microphones) | set(self._viewport_microphones)
        if self._viewport_pose is not None:
            ids |= set(VIEWPORT_IDS)
        return ids

    def _publish_registry(self) -> None:
        self._registry_pub.publish(String(data=json.dumps(sorted(self._microphone_ids()), separators=(",", ":"))))
        self._publish_markers()

    def _marker_poses(self) -> dict[str, _MarkerPose]:
        poses: dict[str, _MarkerPose] = {}
        for binding in self._robots:
            frame = self._mount_frame(binding)
            for mic in self._array.mics:
                poses[ListenerId.array_mic(binding.name, mic.name)] = _MarkerPose(position=mic.position_m, frame=frame, yaw_rad=mic.yaw_rad, color=MIC_COLORS.get(mic.name, DEFAULT_MIC_COLOR))
        prefixes = self._robot_frame_prefixes()
        for spec in self._configured_microphones:
            if spec.robot in prefixes:
                poses[spec.listener_id] = _MarkerPose(position=(0.0, 0.0, 0.0), frame=spec.resolve_frame(prefixes[spec.robot]))
        for listener_id, (position, frame) in (self._spawned_microphones | self._viewport_microphones).items():
            poses[listener_id] = _MarkerPose(position=position, frame=frame)
        frame = self._acoustic_frame() or "map"
        for listener_id, microphone in self._world_microphones.items():
            position = self._world_microphone_position(microphone)
            if position is not None:
                poses[listener_id] = _MarkerPose(position=position, frame=frame)
        return poses

    def _publish_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        markers = []
        for index, (listener_id, pose) in enumerate(sorted(self._marker_poses().items())):
            x, y, z = pose.position
            forward_x, forward_y = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
            left_x, left_y = -forward_y, forward_x
            apex = Point(x=x + 0.20 * forward_x, y=y + 0.20 * forward_y, z=z)
            base_a = Point(x=x - 0.08 * forward_x - 0.08 * left_x, y=y - 0.08 * forward_y - 0.08 * left_y, z=z - 0.07)
            base_b = Point(x=x - 0.08 * forward_x + 0.08 * left_x, y=y - 0.08 * forward_y + 0.08 * left_y, z=z - 0.07)
            base_c = Point(x=x - 0.08 * forward_x, y=y - 0.08 * forward_y, z=z + 0.10)
            marker = Marker(header=Header(frame_id=pose.frame, stamp=stamp), ns="acoustic_microphones", id=index * 2, type=Marker.TRIANGLE_LIST, action=Marker.ADD)
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            marker.color = pose.color
            marker.lifetime.sec = 1
            marker.points = [apex, base_a, base_b, apex, base_b, base_c, apex, base_c, base_a, base_a, base_c, base_b]
            label = Marker(header=marker.header, ns="acoustic_microphone_labels", id=index * 2 + 1, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            label.pose.position = Point(x=x, y=y, z=z + 0.35)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18
            label.color = ColorRGBA(r=pose.color.r * 0.45, g=pose.color.g * 0.45, b=pose.color.b * 0.45, a=1.0)
            label.lifetime = marker.lifetime
            label.text = listener_id
            markers.extend((marker, label))
        self._marker_pub.publish(MarkerArray(markers=markers))

    def _spawn_microphone(self, request: SpawnMicrophone.Request, response: SpawnMicrophone.Response) -> SpawnMicrophone.Response:
        placement = str(request.placement).strip().lower()
        if not placement or ":" in placement:
            response.error_msg = "placement must be non-empty and contain no ':'"
            return response
        point = request.position.point
        if not _finite(point.x, point.y, point.z):
            response.error_msg = "microphone position must be finite"
            return response
        source_frame = str(request.position.header.frame_id).strip().lstrip("/")
        if not source_frame:
            response.error_msg = "microphone position requires a frame ID"
            return response
        clicked = (float(point.x), float(point.y), float(point.z))
        attached_frame = str(request.attached_frame).strip().lstrip("/")
        stored: tuple[Vec3, str] = (clicked, source_frame)
        if attached_frame:
            attached = self._between(clicked, source_frame, attached_frame, "spawned microphone attachment")
            if attached is None:
                response.error_msg = f"cannot transform clicked position into TF frame {attached_frame!r}"
                return response
            stored = (attached, attached_frame)
        transformed = self._in_acoustic_frame(clicked, source_frame, "spawned microphone")
        world = self._tracker.world
        zone = (
            next((candidate for candidate in world.scene.zones if candidate.polygon.buffer(MICROPHONE_PLACEMENT_TOLERANCE_M).covers(shapely.Point(transformed[0], transformed[1]))), None)
            if world is not None and transformed is not None
            else None
        )
        room = world.room(zone.name) if world is not None and zone is not None else None
        check = transformed or stored[0]
        if check[2] < -MICROPHONE_PLACEMENT_TOLERANCE_M:
            response.error_msg = "microphone height cannot be below the floor"
            return response
        if room is not None and zone is not None and check[2] > room.ceiling_height_m + MICROPHONE_PLACEMENT_TOLERANCE_M:
            response.error_msg = f"microphone height exceeds zone {zone.name!r} ceiling at {room.ceiling_height_m:.2f} m"
            return response
        if room is None and check[2] > self._ceiling_height.value + MICROPHONE_PLACEMENT_TOLERANCE_M:
            response.error_msg = "microphone height exceeds the default ceiling"
            return response
        existing = self._microphone_ids()
        while True:
            self._spawned_index += 1
            listener_id = ListenerId.runtime_mic(self._spawned_index)
            if listener_id not in existing:
                break
        self._spawned_microphones[listener_id] = stored
        self._publish_registry()
        response.listener_id = listener_id
        response.zone = zone.name if zone is not None else ""
        response.attached_frame = attached_frame
        response.success = True
        self.get_logger().info(f"spawned microphone {listener_id!r} at ({check[0]:.2f}, {check[1]:.2f}, {check[2]:.2f})")
        return response

    def _remove_microphone(self, request: RemoveMicrophone.Request, response: RemoveMicrophone.Response) -> RemoveMicrophone.Response:
        listener_id = str(request.listener_id).strip()
        if listener_id not in self._spawned_microphones:
            try:
                parsed = ListenerId.parse(listener_id)
            except ValueError:
                parsed = None
            if listener_id in self._world_microphones:
                response.error_msg = "world-authored microphones cannot be removed"
            elif parsed is not None and (parsed.kind is ListenerKind.ARRAY or parsed.scope is MicrophoneScope.ROBOT):
                response.error_msg = "robot-attached microphones cannot be removed"
            elif listener_id in VIEWPORT_IDS:
                response.error_msg = "viewport microphones cannot be removed"
            else:
                response.error_msg = f"unknown runtime microphone {listener_id!r}"
            return response
        self._spawned_microphones.pop(listener_id)
        self._stop_outputs(lambda candidate: candidate != listener_id, forget=True)
        self._publish_registry()
        response.success = True
        return response


def main() -> None:
    SoundPropagationNode.run_main()
