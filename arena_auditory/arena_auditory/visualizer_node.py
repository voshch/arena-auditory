"""RViz markers for propagated sounds, environment sources and acoustic rooms, plus an optional live RIR plot."""

from __future__ import annotations

import asyncio
import math
import threading
import time
import zlib

import attrs
from arena_auditory_msgs.msg import ContinuousAudioSourceState, ContinuousHeardSoundState, HeardSoundEvent, RoomImpulse, SoundReception, SoundSource
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_rclpy_mixins.qos import best_effort, latched, reliable
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from arena_auditory.constants import (
    CONTINUOUS_AUDIO_SOURCES,
    CONTINUOUS_HEARD_SOUNDS,
    ENVIRONMENT_SOURCE_MARKERS,
    HEARD_SOUND_EVENTS,
    PEDESTRIAN_PROPAGATION_MARKERS,
    ROBOT_PROPAGATION_MARKERS,
    ROOM_IMPULSES,
    ROOM_MARKERS,
)
from arena_auditory.params import Configuration, PlotMode
from arena_auditory.propagation import Impulse, RirCache
from arena_auditory.propagation.plot import AcousticPlotDashboard, AcousticPlotSnapshot
from arena_auditory.shared import NS_PER_S, AgentKind, ListenerId, ListenerKind, Vec3, point_vec3
from arena_auditory.world import AcousticWorld
from arena_auditory.world_tracker import WorldTracker

PLOT_PUMP_PERIOD_S = 0.05

PEDESTRIAN_COLOR = ColorRGBA(r=0.10, g=0.45, b=1.0, a=0.92)
ROBOT_COLOR = ColorRGBA(r=0.65, g=0.20, b=1.0, a=0.92)
MICROPHONE_COLOR = ColorRGBA(r=0.10, g=0.85, b=0.55, a=0.92)


@attrs.frozen
class _PlotRequest:
    key: str
    source_position_m: Vec3
    listener_position_m: Vec3
    source_zone: str
    listener_zone: str
    traversed_zones: tuple[str, ...]
    portal_positions_m: tuple[Vec3, ...]
    backend: str
    label: str


def _stable_marker_base(key: str, width: int) -> int:
    maximum = max((2**31 - 1) // max(width, 1), 1)
    return (zlib.crc32(key.encode()) % maximum) * max(width, 1)


def _source_local_point(origin: Point, yaw: float, local_x: float, local_y: float, local_z: float) -> Point:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return Point(
        x=float(origin.x) + cos_yaw * local_x - sin_yaw * local_y,
        y=float(origin.y) + sin_yaw * local_x + cos_yaw * local_y,
        z=float(origin.z) + local_z,
    )


class SoundPropagationVisualizer(ArenaMixinNode):
    def __init__(self) -> None:
        super().__init__("sound_propagation_visualizer")
        self.conf = Configuration(self)
        self._viz = self.conf.Viz
        self._previous_portal_count = {"pedestrian": 0, "robot": 0}
        self._pedestrian_publisher: LazyPublisher[MarkerArray] = LazyPublisher(self.create_publisher(MarkerArray, PEDESTRIAN_PROPAGATION_MARKERS, reliable(10)))
        self._robot_publisher: LazyPublisher[MarkerArray] = LazyPublisher(self.create_publisher(MarkerArray, ROBOT_PROPAGATION_MARKERS, reliable(10)))
        self._room_publisher = self.create_publisher(MarkerArray, ROOM_MARKERS, latched(1))
        self._environment_source_publisher = self.create_publisher(MarkerArray, ENVIRONMENT_SOURCE_MARKERS, latched(32))
        self._room_marker_frame = ""
        self._tracker = WorldTracker(self, on_ready=self._on_world_ready, on_clear=self._on_world_clear, conf=self.conf)

        self.create_subscription(HeardSoundEvent, HEARD_SOUND_EVENTS, self._on_heard, reliable(50))
        self.create_subscription(ContinuousAudioSourceState, CONTINUOUS_AUDIO_SOURCES, self._on_continuous_source, best_effort(64))
        self.create_subscription(ContinuousHeardSoundState, CONTINUOUS_HEARD_SOUNDS, self._on_continuous_heard, best_effort(64))

        self._plot_mode = self._viz.PLOT_MODE.value
        self._impulses = RirCache(self.conf.Rir.CACHE_SIZE.value)
        self._dashboard: AcousticPlotDashboard | None = None
        self._plot_lock = threading.Lock()
        self._pending_plot: _PlotRequest | None = None
        self._last_plot_signature: tuple[object, ...] | None = None
        self._last_plot_submit = 0.0
        self._plot_listener_id = ""
        self._continuous_listener_id = ""
        self._reported_plot_errors: set[str] = set()
        if self._plot_mode is PlotMode.LIVE:
            self.create_subscription(RoomImpulse, ROOM_IMPULSES, self._on_impulse, latched(256))

    async def setup(self) -> None:
        if self._plot_mode is not PlotMode.LIVE:
            return
        self._dashboard = AcousticPlotDashboard(energy_bin_ms=self._viz.PLOT_ENERGY_BIN_MS.value, early_window_s=self._viz.PLOT_EARLY_WINDOW_S.value)
        while True:
            self._dashboard.pump_events()
            self._submit_plot()
            await asyncio.sleep(PLOT_PUMP_PERIOD_S)

    async def teardown(self) -> None:
        if self._dashboard is not None:
            self._dashboard.close()

    def destroy_node(self) -> bool:
        self._tracker.destroy()
        return super().destroy_node()

    def _on_world_ready(self, world: AcousticWorld) -> None:
        occupancy = self._tracker.occupancy
        frame = occupancy.frame_id if occupancy is not None else "map"
        self._clear_rooms()
        self._publish_rooms(world, frame)
        with self._plot_lock:
            self._last_plot_signature = None

    def _on_world_clear(self) -> None:
        self._clear_rooms()
        with self._plot_lock:
            self._pending_plot = None
            self._last_plot_signature = None

    def _on_impulse(self, msg: RoomImpulse) -> None:
        self._impulses.put(Impulse.from_msg(msg))

    def _marker(self, frame: str, marker_id: int, marker_type: int, lifetime_s: float) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        seconds = max(lifetime_s, 0.0)
        marker.lifetime.sec = int(seconds)
        marker.lifetime.nanosec = int((seconds % 1.0) * NS_PER_S)
        return marker

    def _listener_output(self, listener_id: str) -> tuple[LazyPublisher[MarkerArray], str, ColorRGBA] | None:
        try:
            kind = ListenerId.parse(listener_id).kind
        except ValueError:
            return None
        match kind:
            case ListenerKind.AGENT:
                return self._pedestrian_publisher, "pedestrian", PEDESTRIAN_COLOR
            case ListenerKind.ROBOT:
                return self._robot_publisher, "robot", ROBOT_COLOR
            case ListenerKind.ARRAY | ListenerKind.MICROPHONE:
                return self._robot_publisher, listener_id.replace(":", "_").replace("/", "_"), MICROPHONE_COLOR
        return None

    def _on_heard(self, msg: HeardSoundEvent) -> None:
        reception = msg.reception
        if reception.used_fallback and reception.fallback_reason.strip() == "acoustic_scene_not_loaded":
            return
        output = self._listener_output(reception.listener_id)
        if output is None:
            self.get_logger().warning(f"cannot visualize sound for unknown listener {reception.listener_id!r}")
            return
        publisher, listener_kind, color = output
        frame = str(msg.header.frame_id).strip()
        if not frame:
            self.get_logger().warning("cannot visualize sound without a frame ID")
            return
        if not publisher.wanted:
            self._previous_portal_count[listener_kind] = 0
            self._consider_plot(msg.header, msg.source, reception)
            return
        lifetime = self._viz.LIFETIME_S.value
        source = Point(x=float(msg.source.position.x), y=float(msg.source.position.y), z=float(msg.source.position.z))
        listener = Point(x=float(reception.listener_position.x), y=float(reception.listener_position.y), z=float(reception.listener_position.z))
        portal_points = [Point(x=float(point.x), y=float(point.y), z=float(point.z)) for point in reception.portal_positions]

        path = self._marker(frame, 0, Marker.LINE_STRIP, lifetime)
        path.ns = f"{listener_kind}_sound_propagation_path"
        path.scale.x = 0.07
        path.color = color
        path.points = [source, *portal_points, listener]

        source_marker = self._marker(frame, 1, Marker.SPHERE, lifetime)
        source_marker.ns = f"{listener_kind}_sound_source"
        source_marker.pose.position = source
        source_marker.scale.x = source_marker.scale.y = source_marker.scale.z = 0.22
        source_marker.color = color

        listener_marker = self._marker(frame, 2, Marker.SPHERE, lifetime)
        listener_marker.ns = f"{listener_kind}_sound_listener"
        listener_marker.pose.position = listener
        listener_marker.scale.x = listener_marker.scale.y = listener_marker.scale.z = 0.22
        listener_marker.color = color

        text = self._marker(frame, 3, Marker.TEXT_VIEW_FACING, lifetime)
        text.ns = f"{listener_kind}_sound_propagation_backend"
        text.pose.position = Point(x=listener.x, y=listener.y, z=listener.z + 0.35)
        text.scale.z = 0.24
        text.color = color
        text.text = reception.backend.strip() or "unknown"
        if reception.portal_ids:
            text.text += f"\n{len(reception.portal_ids)} portal(s), loss={float(reception.route_loss_db):.1f} dB"
            text.text += "\n" + " -> ".join(reception.portal_ids)
        if reception.used_fallback:
            text.text += f"\nfallback: {reception.fallback_reason}"

        markers = [path, source_marker, listener_marker, text]
        for index, portal_point in enumerate(portal_points):
            portal = self._marker(frame, 4 + index, Marker.CUBE, lifetime)
            portal.ns = f"{listener_kind}_acoustic_portal"
            portal.pose.position = portal_point
            portal.scale.x = portal.scale.y = portal.scale.z = 0.28
            portal.color = color
            markers.append(portal)
        for index in range(len(portal_points), self._previous_portal_count.get(listener_kind, 0)):
            stale = self._marker(frame, 4 + index, Marker.CUBE, 0.0)
            stale.ns = f"{listener_kind}_acoustic_portal"
            stale.action = Marker.DELETE
            markers.append(stale)
        self._previous_portal_count[listener_kind] = len(portal_points)
        publisher.publish(lambda: MarkerArray(markers=markers))
        self._consider_plot(msg.header, msg.source, reception)

    def _on_continuous_heard(self, msg: ContinuousHeardSoundState) -> None:
        if not msg.source.active:
            return
        self._publish_continuous_path(msg)
        self._consider_plot(msg.header, msg.source, msg.reception)

    def _publish_continuous_path(self, msg: ContinuousHeardSoundState) -> None:
        reception = msg.reception
        listener_id = reception.listener_id
        configured = self._viz.LISTENER_ID.value.strip()
        if configured:
            if listener_id != configured:
                return
        else:
            if not self._continuous_listener_id:
                self._continuous_listener_id = listener_id
            if listener_id != self._continuous_listener_id:
                return
        output = self._listener_output(listener_id)
        frame = str(msg.header.frame_id).strip()
        if output is None or not frame:
            return
        publisher, listener_kind, color = output
        if not publisher.wanted:
            return
        lifetime = self._viz.CONTINUOUS_LIFETIME_S.value
        base_id = _stable_marker_base(f"{msg.source.id}|{listener_id}", 4 + len(reception.portal_positions))
        source = Point(x=float(msg.source.position.x), y=float(msg.source.position.y), z=float(msg.source.position.z))
        listener = Point(x=float(reception.listener_position.x), y=float(reception.listener_position.y), z=float(reception.listener_position.z))
        portals = [Point(x=float(point.x), y=float(point.y), z=float(point.z)) for point in reception.portal_positions]

        path = self._marker(frame, base_id, Marker.LINE_STRIP, lifetime)
        path.ns = f"{listener_kind}_continuous_audio_paths"
        path.scale.x = 0.07
        path.color = color
        path.points = [source, *portals, listener]

        source_marker = self._marker(frame, base_id + 1, Marker.SPHERE, lifetime)
        source_marker.ns = f"{listener_kind}_continuous_audio_sources"
        source_marker.pose.position = source
        source_marker.scale.x = source_marker.scale.y = source_marker.scale.z = 0.22
        source_marker.color = color

        listener_marker = self._marker(frame, base_id + 2, Marker.SPHERE, lifetime)
        listener_marker.ns = f"{listener_kind}_continuous_audio_listeners"
        listener_marker.pose.position = listener
        listener_marker.scale.x = listener_marker.scale.y = listener_marker.scale.z = 0.22
        listener_marker.color = color

        label = self._marker(frame, base_id + 3, Marker.TEXT_VIEW_FACING, lifetime)
        label.ns = f"{listener_kind}_continuous_audio_labels"
        label.pose.position = Point(x=listener.x, y=listener.y, z=listener.z + 0.35)
        label.scale.z = 0.22
        label.color = color
        label.text = f"{msg.source.kind}: {reception.backend}\n{float(reception.direct_delay_s) * 1000.0:.1f} ms, {len(portals)} portal(s)"

        markers = [path, source_marker, listener_marker, label]
        for index, point in enumerate(portals):
            portal = self._marker(frame, base_id + 4 + index, Marker.CUBE, lifetime)
            portal.ns = f"{listener_kind}_continuous_audio_portals"
            portal.pose.position = point
            portal.scale.x = portal.scale.y = portal.scale.z = 0.28
            portal.color = color
            markers.append(portal)
        publisher.publish(lambda: MarkerArray(markers=markers))

    def _on_continuous_source(self, msg: ContinuousAudioSourceState) -> None:
        source = msg.source
        if source.agent_kind != AgentKind.ENVIRONMENT.value:
            return
        frame = str(msg.header.frame_id).strip()
        if not frame:
            return
        lifetime = self._viz.CONTINUOUS_LIFETIME_S.value
        base_id = _stable_marker_base(source.id, 6)
        kind = source.kind.lower()
        is_alarm = "alarm" in kind or "siren" in kind
        if not source.active:
            color = ColorRGBA(r=0.45, g=0.45, b=0.45, a=0.65)
        elif is_alarm:
            color = ColorRGBA(r=1.0, g=0.08, b=0.05, a=0.95)
        else:
            color = ColorRGBA(r=0.05, g=0.75, b=0.95, a=0.92)
        yaw = float(source.yaw_rad)
        position = Point(x=float(source.position.x), y=float(source.position.y), z=float(source.position.z))
        markers = self._alarm_markers(frame, base_id, lifetime, position, yaw, color) if is_alarm else self._radio_markers(frame, base_id, lifetime, position, yaw, color)
        label = self._marker(frame, base_id + 5, Marker.TEXT_VIEW_FACING, lifetime)
        label.ns = "environment_audio_source_labels"
        label.pose.position = Point(x=position.x, y=position.y, z=position.z + 0.32)
        label.scale.z = 0.22
        label.color = color
        label.text = f"{source.kind or source.group_id} / {source.agent_name} [{'ACTIVE' if source.active else 'OFF'}]"
        self._environment_source_publisher.publish(MarkerArray(markers=[*markers, label]))

    def _radio_markers(self, frame: str, base_id: int, lifetime: float, position: Point, yaw: float, color: ColorRGBA) -> list[Marker]:
        body = self._marker(frame, base_id, Marker.CUBE, lifetime)
        body.ns = "environment_audio_radio_body"
        body.pose.position = position
        body.pose.orientation.z = math.sin(yaw / 2.0)
        body.pose.orientation.w = math.cos(yaw / 2.0)
        body.scale.x, body.scale.y, body.scale.z = 0.48, 0.20, 0.30
        body.color = color

        speaker = self._marker(frame, base_id + 1, Marker.CYLINDER, lifetime)
        speaker.ns = "environment_audio_radio_speaker"
        speaker.pose.position = _source_local_point(position, yaw, -0.11, -0.115, 0.0)
        roll = math.pi / 2.0
        speaker.pose.orientation.x = math.sin(roll / 2.0) * math.cos(yaw / 2.0)
        speaker.pose.orientation.y = math.sin(roll / 2.0) * math.sin(yaw / 2.0)
        speaker.pose.orientation.z = math.cos(roll / 2.0) * math.sin(yaw / 2.0)
        speaker.pose.orientation.w = math.cos(roll / 2.0) * math.cos(yaw / 2.0)
        speaker.scale.x, speaker.scale.y, speaker.scale.z = 0.18, 0.18, 0.035
        speaker.color = ColorRGBA(r=0.04, g=0.04, b=0.05, a=color.a)

        display = self._marker(frame, base_id + 2, Marker.CUBE, lifetime)
        display.ns = "environment_audio_radio_display"
        display.pose.position = _source_local_point(position, yaw, 0.115, -0.116, 0.035)
        display.pose.orientation.z = math.sin(yaw / 2.0)
        display.pose.orientation.w = math.cos(yaw / 2.0)
        display.scale.x, display.scale.y, display.scale.z = 0.15, 0.025, 0.065
        display.color = ColorRGBA(r=0.95, g=0.82, b=0.16, a=color.a)

        antenna = self._marker(frame, base_id + 3, Marker.LINE_STRIP, lifetime)
        antenna.ns = "environment_audio_radio_antenna"
        antenna.scale.x = 0.018
        antenna.color = ColorRGBA(r=0.12, g=0.12, b=0.14, a=color.a)
        antenna.points = [_source_local_point(position, yaw, 0.17, 0.0, 0.14), _source_local_point(position, yaw, 0.29, 0.0, 0.52)]
        return [body, speaker, display, antenna]

    def _alarm_markers(self, frame: str, base_id: int, lifetime: float, position: Point, yaw: float, color: ColorRGBA) -> list[Marker]:
        body = self._marker(frame, base_id, Marker.CUBE, lifetime)
        body.ns = "environment_audio_alarm_body"
        body.pose.position = position
        body.pose.orientation.z = math.sin(yaw / 2.0)
        body.pose.orientation.w = math.cos(yaw / 2.0)
        body.scale.x, body.scale.y, body.scale.z = 0.36, 0.28, 0.20
        body.color = ColorRGBA(r=0.16, g=0.16, b=0.18, a=color.a)

        beacon = self._marker(frame, base_id + 1, Marker.CYLINDER, lifetime)
        beacon.ns = "environment_audio_alarm_beacon"
        beacon.pose.position = _source_local_point(position, yaw, 0.0, 0.0, 0.17)
        beacon.scale.x, beacon.scale.y, beacon.scale.z = 0.22, 0.22, 0.18
        beacon.color = color
        return [body, beacon]

    def _publish_rooms(self, world: AcousticWorld, frame: str) -> None:
        markers: list[Marker] = []
        marker_id = 0
        for room in world.rooms:
            floor = self._marker(frame, marker_id, Marker.LINE_STRIP, 0.0)
            marker_id += 1
            floor.ns = "acoustic_zone_floor"
            floor.scale.x = 0.045
            floor.color = ColorRGBA(r=0.15, g=0.55, b=0.85, a=0.75)
            floor.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in (*room.corners_xy, room.corners_xy[0])]
            markers.append(floor)

            walls = self._marker(frame, marker_id, Marker.LINE_LIST, 0.0)
            marker_id += 1
            walls.ns = "acoustic_room_walls"
            walls.scale.x = 0.035
            walls.color = ColorRGBA(r=0.55, g=0.60, b=0.65, a=0.65)
            height = room.ceiling_height_m
            for boundary in room.boundary:
                sx, sy = boundary.start
                ex, ey = boundary.end
                walls.points.extend(
                    [
                        Point(x=sx, y=sy, z=0.0),
                        Point(x=ex, y=ey, z=0.0),
                        Point(x=sx, y=sy, z=height),
                        Point(x=ex, y=ey, z=height),
                        Point(x=sx, y=sy, z=0.0),
                        Point(x=sx, y=sy, z=height),
                    ]
                )
            markers.append(walls)

            label = self._marker(frame, marker_id, Marker.TEXT_VIEW_FACING, 0.0)
            marker_id += 1
            label.ns = "acoustic_zone_names"
            corners = room.corners_xy
            label.pose.position = Point(x=sum(point[0] for point in corners) / len(corners), y=sum(point[1] for point in corners) / len(corners), z=0.12)
            label.scale.z = 0.22
            label.color = ColorRGBA(r=0.10, g=0.25, b=0.45, a=0.90)
            label.text = room.zone_name
            markers.append(label)
        self._room_publisher.publish(MarkerArray(markers=markers))
        self._room_marker_frame = frame

    def _clear_rooms(self) -> None:
        if not self._room_marker_frame:
            return
        marker = self._marker(self._room_marker_frame, 0, Marker.DELETEALL, 0.0)
        marker.action = Marker.DELETEALL
        self._room_publisher.publish(MarkerArray(markers=[marker]))
        self._room_marker_frame = ""

    def _report_plot_once(self, key: str, message: str) -> None:
        if key not in self._reported_plot_errors:
            self._reported_plot_errors.add(key)
            self.get_logger().warning(message)

    def _consider_plot(self, header: Header, source: SoundSource, reception: SoundReception) -> None:
        if self._plot_mode is not PlotMode.LIVE or self._tracker.world is None or not str(header.frame_id).strip():
            return
        listener_id = reception.listener_id
        configured = self._viz.PLOT_LISTENER_ID.value.strip()
        if configured:
            if listener_id != configured:
                return
        else:
            if not listener_id.startswith(f"{ListenerKind.ROBOT}:"):
                return
            if not self._plot_listener_id:
                self._plot_listener_id = listener_id
                self.get_logger().info(f"live RIR plot selected listener {listener_id!r}")
            if listener_id != self._plot_listener_id:
                return
        if reception.used_fallback:
            reason = reception.fallback_reason.strip() or "unknown"
            self._report_plot_once(f"fallback:{reason}", f"live RIR plot skipped propagation fallback: {reason}")
            return
        if not reception.source_zone or not reception.listener_zone or not reception.rir_key:
            self._report_plot_once("no_impulse", "live RIR plot skipped a reception without a room impulse")
            return
        source_position = point_vec3(source.position)
        listener_position = point_vec3(reception.listener_position)
        quantization = max(self._viz.PLOT_QUANTIZATION_M.value, 1e-6)
        signature = (
            reception.source_zone,
            reception.listener_zone,
            tuple(round(value / quantization) for value in source_position),
            tuple(round(value / quantization) for value in listener_position),
        )
        with self._plot_lock:
            if signature == self._last_plot_signature:
                return
            self._last_plot_signature = signature
            self._pending_plot = _PlotRequest(
                key=reception.rir_key,
                source_position_m=source_position,
                listener_position_m=listener_position,
                source_zone=reception.source_zone,
                listener_zone=reception.listener_zone,
                traversed_zones=tuple(reception.traversed_zones) or (reception.source_zone,),
                portal_positions_m=tuple(point_vec3(point) for point in reception.portal_positions),
                backend=reception.backend,
                label=listener_id,
            )

    def _submit_plot(self) -> None:
        world = self._tracker.world
        if self._dashboard is None or world is None:
            return
        now = time.monotonic()
        if now - self._last_plot_submit < 1.0 / self._viz.PLOT_RATE_HZ.value:
            return
        with self._plot_lock:
            request = self._pending_plot
            if request is None:
                return
            impulse = self._impulses.get(request.key)
            if impulse is None:
                return
            self._pending_plot = None
        self._last_plot_submit = now
        try:
            self._dashboard.update(self._snapshot(request, impulse, world))
        except ValueError as exc:
            with self._plot_lock:
                self._last_plot_signature = None
            self._report_plot_once(f"{type(exc).__name__}: {exc}", f"RIR plot update failed: {exc}")

    @staticmethod
    def _snapshot(request: _PlotRequest, impulse: Impulse, world: AcousticWorld) -> AcousticPlotSnapshot:
        return AcousticPlotSnapshot.from_impulse(
            impulse,
            room_specs=world.rooms,
            source_position_m=request.source_position_m,
            listener_position_m=request.listener_position_m,
            backend=request.backend,
            source_zone=request.source_zone,
            listener_zone=request.listener_zone,
            traversed_zones=request.traversed_zones,
            portal_positions_m=request.portal_positions_m,
            label=request.label,
        )


def main() -> None:
    SoundPropagationVisualizer.run_main()
