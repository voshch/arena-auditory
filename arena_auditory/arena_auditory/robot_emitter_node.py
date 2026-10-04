"""Robot noise producer: the continuous motor source of every fleet robot, driven by odometry."""

from __future__ import annotations

import hashlib
import itertools
import math
from concurrent.futures import ThreadPoolExecutor

import attrs
from arena_auditory_msgs.msg import ContinuousAudioSourceState
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.qos import best_effort, latched, reliable
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from std_msgs.msg import ColorRGBA
from task_generator_msgs.msg import EpisodeRecord, RobotFleet
from visualization_msgs.msg import Marker, MarkerArray

from arena_auditory.assets import WAV_MODELS, SoundAsset, SoundLibrary, Variant, selection_seed
from arena_auditory.constants import CONTINUOUS_AUDIO_SOURCES, STATE_EPISODE, STATE_ROBOTS, motor_markers
from arena_auditory.params import Configuration, MotorModel
from arena_auditory.shared import INACTIVE_REPEATS, NS_PER_S, AgentKind, RobotBinding, SourceSpec, robot_bindings
from arena_auditory.sources.drivetrain.program import wheel_state
from arena_auditory.world_tracker import follow_world_sounds

MOTOR = "motor"
MIN_WHEEL_SEPARATION_M = 1e-3


def robot_agent_id(robot: str) -> int:
    """Negative agent id of a robot, disjoint from pedestrian ids."""
    digest = hashlib.blake2b(robot.encode(), digest_size=4).digest()
    return -(int.from_bytes(digest, "big") & 0x7FFFFFFF) - 1


def marker_base_id(robot: str) -> int:
    digest = hashlib.blake2b(robot.encode(), digest_size=4).digest()
    return (int.from_bytes(digest, "big") & 0x0FFFFFFF) * 8


@attrs.define
class _Robot:
    binding: RobotBinding
    wheel_separation_m: float
    marker_publisher: Publisher
    odom_subscriptions: list[Subscription]
    stamp: Time | None = None
    state: dict[str, float] = attrs.field(factory=dict)
    speed_mps: float = 0.0
    moving: bool = False
    program_start_ns: int = 0
    inactive_left: int = 0


class RobotEmitterNode(ArenaMixinNode):
    """Publishes ContinuousAudioSourceState per robot: drivetrain for robot models with a drivetrain variant, the motor wav loop otherwise."""

    def __init__(self) -> None:
        super().__init__("robot_emitter")
        self.conf = Configuration(self)
        self._motor = self.conf.Motor
        self._drivetrain = self.conf.Drivetrain
        self._library = SoundLibrary.default()
        self._robots: dict[str, _Robot] = {}
        self._episode_seed = 0
        self._loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot_emitter_world")
        self._source_publisher = self.create_publisher(ContinuousAudioSourceState, CONTINUOUS_AUDIO_SOURCES, best_effort(64))
        self.create_subscription(RobotFleet, STATE_ROBOTS, self._on_fleet, latched(1))
        self.create_subscription(EpisodeRecord, STATE_EPISODE, self._on_episode, latched(20))
        follow_world_sounds(self, self._library, self._loader)
        self.create_timer(self._drivetrain.PERIOD_S.value, self._publish)

    def destroy_node(self) -> bool:
        self._loader.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def _on_episode(self, msg: EpisodeRecord) -> None:
        self._episode_seed = int(msg.seed)

    def _on_fleet(self, msg: RobotFleet) -> None:
        for binding in robot_bindings(msg, odom_topic_template=self._drivetrain.ODOM_TOPIC_TEMPLATE.value):
            if binding.name in self._robots:
                continue
            if binding.error:
                self.get_logger().warning(f"robot {binding.name!r}: {binding.error}")
            separation = binding.wheel_separation_m
            if separation is None:
                separation = max(2.0 * self._drivetrain.ANGULAR_SCALE_M.value, MIN_WHEEL_SEPARATION_M)
                self.get_logger().warning(f"robot {binding.name!r}: no wheel separation for model {binding.model!r}, using {separation:.5f} m")
            subscriptions = [self.create_subscription(Odometry, topic, lambda odom, name=binding.name: self._on_odom(name, odom), best_effort(10)) for topic in binding.odom_topics]
            self._robots[binding.name] = _Robot(
                binding=binding,
                wheel_separation_m=separation,
                marker_publisher=self.create_publisher(MarkerArray, motor_markers(binding.name), reliable(10)),
                odom_subscriptions=subscriptions,
            )
            self.get_logger().info(f"robot {binding.name!r} motor motion sources: {', '.join(binding.odom_topics)}")

    def _on_odom(self, name: str, msg: Odometry) -> None:
        robot = self._robots.get(name)
        if robot is None:
            return
        linear = float(msg.twist.twist.linear.x)
        lateral = float(msg.twist.twist.linear.y)
        angular = float(msg.twist.twist.angular.z)
        robot.stamp = msg.header.stamp
        robot.state = wheel_state(linear, angular, robot.wheel_separation_m)
        robot.speed_mps = max(math.hypot(linear, lateral), abs(angular) * self._drivetrain.ANGULAR_SCALE_M.value)

    def _select(self, robot: _Robot) -> tuple[SoundAsset, Variant]:
        asset = self._library.default_asset(MOTOR)
        models = WAV_MODELS if self._motor.MODEL.value is MotorModel.WAV else None
        return asset, asset.select(context={"robot_model": robot.binding.model}, seed=selection_seed(self._episode_seed, robot.binding.name, MOTOR), models=models)

    def _moving(self, robot: _Robot) -> bool:
        if not self._motor.ENABLED.value:
            return False
        if not self._drivetrain.MOTION_GATE_ENABLED.value:
            return True
        if robot.moving:
            return robot.speed_mps > self._drivetrain.MOTION_GATE_STOP_MPS.value
        return robot.speed_mps >= self._drivetrain.MOTION_GATE_START_MPS.value

    def _publish(self) -> None:
        for name, robot in self._robots.items():
            if robot.stamp is None:
                continue
            moving = self._moving(robot)
            if moving != robot.moving:
                robot.moving = moving
                self.get_logger().info(f"robot {name!r} motor {'started' if moving else 'stopped'} at {robot.speed_mps:.3f} m/s")
                if moving:
                    robot.program_start_ns = self.get_clock().now().nanoseconds
                else:
                    robot.inactive_left = INACTIVE_REPEATS
                    self._clear_markers(robot)
            if moving:
                self._publish_source(robot, active=True)
                self._publish_markers(robot)
            elif robot.inactive_left > 0:
                robot.inactive_left -= 1
                self._publish_source(robot, active=False)

    def _publish_source(self, robot: _Robot, *, active: bool) -> None:
        try:
            asset, variant = self._select(robot)
            kind = self._library.kind(MOTOR)
        except (LookupError, ValueError, FileNotFoundError) as exc:
            self.get_logger().error(f"no motor sound for robot {robot.binding.name!r}: {exc}", throttle_duration_sec=5.0)
            return
        source = SourceSpec(
            id=f"robot:{robot.binding.name}:{MOTOR}",
            kind=MOTOR,
            asset_id=asset.id,
            variant_id=variant.id,
            model=variant.model,
            agent_kind=AgentKind.ROBOT,
            agent_id=robot_agent_id(robot.binding.name),
            agent_name=robot.binding.name,
            tags=variant.tags,
            position=(0.0, 0.0, kind.height_m),
            level_db=asset.level_db,
            reference_distance_m=asset.reference_distance_m,
            loop=asset.loop,
            active=active,
            program_start_ns=robot.program_start_ns,
            seed=selection_seed(self._episode_seed, robot.binding.name, MOTOR),
            state=robot.state,
        )
        msg = ContinuousAudioSourceState(source=source.to_msg())
        msg.header.stamp = robot.stamp
        msg.header.frame_id = robot.binding.base_frame
        self._source_publisher.publish(msg)

    def _publish_markers(self, robot: _Robot) -> None:
        if not self._drivetrain.MARKERS_ENABLED.value:
            return
        try:
            r, g, b = self._library.kind(MOTOR).color
        except KeyError:
            return
        cone = math.radians(self._drivetrain.MARKERS_CONE_DEG.value)
        reach = self._drivetrain.MARKERS_RANGE_M.value
        z = self._drivetrain.MARKERS_Z_M.value
        lifetime_s = max(self._drivetrain.MARKERS_LIFETIME_S.value, self._drivetrain.PERIOD_S.value * 1.5)
        lifetime_ns = round(lifetime_s * NS_PER_S)
        apex = Point(x=0.15, y=0.0, z=z)
        arc = [Point(x=math.cos(-cone / 2.0 + cone * i / 16) * reach, y=math.sin(-cone / 2.0 + cone * i / 16) * reach, z=z) for i in range(17)]
        base_id = marker_base_id(robot.binding.name)

        fill = Marker()
        fill.header.frame_id = robot.binding.base_frame
        fill.header.stamp = self.get_clock().now().to_msg()
        fill.ns = f"motor_sound_{robot.binding.name}"
        fill.id = base_id
        fill.type = Marker.TRIANGLE_LIST
        fill.action = Marker.ADD
        fill.pose.orientation.w = 1.0
        fill.scale.x = fill.scale.y = fill.scale.z = 1.0
        fill.color = ColorRGBA(r=r, g=g, b=b, a=0.28)
        fill.lifetime = Duration(sec=lifetime_ns // NS_PER_S, nanosec=lifetime_ns % NS_PER_S)
        for left, right in itertools.pairwise(arc):
            fill.points.extend([apex, left, right])

        outline = Marker()
        outline.header = fill.header
        outline.ns = f"motor_sound_{robot.binding.name}_outline"
        outline.id = base_id + 1
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = self._drivetrain.MARKERS_LINE_WIDTH_M.value
        outline.color = ColorRGBA(r=r, g=g, b=b, a=0.95)
        outline.lifetime = fill.lifetime
        outline.points = [apex, *arc, apex]
        robot.marker_publisher.publish(MarkerArray(markers=[fill, outline]))

    def _clear_markers(self, robot: _Robot) -> None:
        stamp = self.get_clock().now().to_msg()
        base_id = marker_base_id(robot.binding.name)
        markers = MarkerArray()
        for index, namespace in enumerate((f"motor_sound_{robot.binding.name}", f"motor_sound_{robot.binding.name}_outline")):
            marker = Marker()
            marker.header.frame_id = robot.binding.base_frame
            marker.header.stamp = stamp
            marker.ns = namespace
            marker.id = base_id + index
            marker.action = Marker.DELETE
            markers.markers.append(marker)
        robot.marker_publisher.publish(markers)


def main() -> None:
    RobotEmitterNode.run_main()
