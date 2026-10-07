"""Simulator-bus hearing: robot listener receptions become heard sounds, bus detections and text markers."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import attrs
from arena_auditory_msgs.msg import HeardSoundEvent
from arena_rclpy_mixins import ArenaMixinNode, qos
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_robots.fleet import robot_bindings
from arena_robots_msgs.msg import SoundDetection
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind, SoundLibrary
from rclpy.duration import Duration
from rclpy.publisher import Publisher
from rclpy.time import Time
from std_msgs.msg import ColorRGBA, Header
from task_generator_msgs.msg import RobotFleet
from visualization_msgs.msg import Marker

from arena_auditory.constants import BUS_FRONTEND, HEARD_SOUND_EVENTS, STATE_ROBOTS, detections, heard_sound, heard_sound_marker
from arena_auditory.params import Configuration
from arena_auditory.shared import ListenerId, ListenerKind
from arena_auditory.world_tracker import follow_world_sounds

RELEASE_PERIOD_S = 0.01


@attrs.frozen
class _RobotOutputs:
    base_frame: str
    heard: Publisher
    detections: Publisher
    markers: Publisher


@attrs.frozen
class _Pending:
    release: Time
    robot: str
    msg: HeardSoundEvent


class BusNode(ArenaMixinNode):
    def __init__(self) -> None:
        super().__init__("robot_hearing_node")
        self.conf = Configuration(self)
        self._bus = self.conf.Bus
        self._library = SoundLibrary.default()
        self._loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bus_world")
        follow_world_sounds(self, self._library, self._loader)
        self._robots: dict[str, _RobotOutputs] = {}
        self._pending: list[_Pending] = []
        self.create_subscription(RobotFleet, STATE_ROBOTS, self._cb_fleet, qos.latched())
        self.create_subscription(HeardSoundEvent, HEARD_SOUND_EVENTS, self._cb_heard, qos.reliable(50))
        self.create_timer(RELEASE_PERIOD_S, self._release)
        self.get_logger().info(f"robot hearing node listening on {self.resolve_topic_name(HEARD_SOUND_EVENTS)!r}, waiting for the robot fleet")

    def destroy_node(self) -> bool:
        self._loader.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def _cb_fleet(self, msg: RobotFleet) -> None:
        for binding in robot_bindings(msg):
            if binding.name in self._robots:
                continue
            if binding.error:
                self.get_logger().warning(f"{binding.error}, markers use {binding.base_frame!r}")
            self._robots[binding.name] = _RobotOutputs(
                base_frame=binding.base_frame,
                heard=self.create_publisher(HeardSoundEvent, heard_sound(binding.name), qos.reliable(50)),
                detections=self.create_publisher(SoundDetection, detections(binding.name, BUS_FRONTEND), qos.reliable(50)),
                markers=LazyPublisher(self.create_publisher(Marker, heard_sound_marker(binding.name), qos.reliable(10))),
            )
            self.get_logger().info(f"registered robot hearing outputs for {binding.name!r} in frame {binding.base_frame!r}")

    def _cb_heard(self, msg: HeardSoundEvent) -> None:
        try:
            listener = ListenerId.parse(msg.reception.listener_id)
        except ValueError:
            return
        if listener.kind is not ListenerKind.ROBOT or listener.owner not in self._robots:
            return
        bus = self._bus
        if bus.IGNORE_SELF_ENABLED.value and msg.source.agent_kind == AgentKind.ROBOT and msg.source.agent_name == listener.owner:
            return
        if not msg.reception.audible:
            return
        if float(msg.reception.received_level_db - msg.reception.threshold_db) < bus.MIN_SNR_DB.value:
            return
        if bus.DELAY_ENABLED.value:
            release = Time.from_msg(msg.header.stamp) + Duration(seconds=float(msg.reception.direct_delay_s))
        else:
            release = self.get_clock().now()
        self._pending.append(_Pending(release=release, robot=listener.owner, msg=msg))

    def _release(self) -> None:
        if not self._pending:
            return
        now = self.get_clock().now()
        due = [item for item in self._pending if item.release <= now]
        self._pending = [item for item in self._pending if item.release > now]
        for item in due:
            outputs = self._robots[item.robot]
            outputs.heard.publish(item.msg)
            outputs.detections.publish(self._detection(item))
            self._publish_marker(item.robot, outputs, item.msg)

    @staticmethod
    def _detection(item: _Pending) -> SoundDetection:
        msg = item.msg
        return SoundDetection(
            header=Header(stamp=item.release.to_msg(), frame_id=msg.header.frame_id),
            robot=item.robot,
            frontend=BUS_FRONTEND,
            kind=msg.source.kind,
            event_id=msg.source.id,
            azimuth_rad=float(msg.reception.bearing_rad),
            elevation_rad=math.nan,
            level_db=float(msg.reception.received_level_db),
            confidence=1.0,
        )

    def _publish_marker(self, robot: str, outputs: _RobotOutputs, msg: HeardSoundEvent) -> None:
        kind = self._library.kinds().get(msg.source.kind)
        if kind is None or not kind.marker or not outputs.markers.wanted:
            return
        bus = self._bus
        r, g, b = kind.color
        marker = Marker()
        marker.header.frame_id = outputs.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot}_heard_sound"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = bus.MARKERS_Z_M.value
        marker.pose.orientation.w = 1.0
        marker.scale.z = bus.MARKERS_TEXT_HEIGHT_M.value
        marker.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)
        marker.text = f"HEARD {kind.name.upper()}\n{msg.reception.received_level_db:.1f} dB"
        marker.lifetime = Duration(seconds=bus.MARKERS_LIFETIME_S.value).to_msg()
        marker.frame_locked = True
        outputs.markers.publish(lambda: marker)


def main() -> None:
    BusNode.run_main()
