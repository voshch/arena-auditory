"""Listen-then-yield: the single writer of the Nav2 SpeedFilter mask of every fleet robot.

Per robot, composes three layers by lowest nonzero percentage:

  1. belief: the robot's decaying pedestrian belief from ``hearing_belief``, dilated and thresholded
     (``belief_grid.speed_mask_from_belief``), so the robot slows near likely mass.
  2. listen: ``policy.listen_mps`` on the approach to a blind bend along the global plan, so the
     robot's own drivetrain noise drops enough to hear around the corner before it reaches it.
  3. hold: ``policy.hold_mps`` on a band before the bend while yielding, a creep rather than a stop
     since Nav2 reads a 0 % mask as "no limit".

Yield engages when the fraction of belief mass inside a disc around the bend exceeds
``policy.yield_fraction`` while the robot is on the approach, and releases once the mass has moved
behind the robot (after ``policy.min_yield_s`` of holding), once it has faded or the detection level
has been falling for ``policy.recede_s`` (each only after ``policy.min_yield_s`` of silence, a pedestrian
who stopped is still there), or on ``policy.yield_timeout_s``. After a release the robot goes
through the bend at listen speed, never full speed, so a pedestrian who stopped to yield in
turn is not driven into. Bends are map-frame points with hysteresis, so a replanned path does
not re-arm the same corner. Every output lives below ``<tg_node>/<robot>``: the mask, its
``CostmapFilterInfo`` for the robot's SpeedFilter, and the state as JSON on ``hearing/policy_state``.
"""

from __future__ import annotations

import array
import functools
import json
import math

import attrs
import numpy as np
import tf2_ros
from arena_auditory_msgs.msg import AuditoryDetection
from arena_rclpy_mixins import ArenaMixinNode, qos
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_runtime_msgs.msg import LockstepChannel
from geometry_msgs.msg import Point
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.time import Time
from std_msgs.msg import Bool, ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from arena_auditory.constants import BELIEF_GRID, COSTMAP_FILTER_INFO, MAP, POLICY_MARKERS, POLICY_STATE, SPEED_FILTER_MASK, STATE_RESETTING, detections, plan
from arena_auditory.hearing.belief_grid import speed_mask_from_belief
from arena_auditory.hearing.belief_node import yaw_from_quat
from arena_auditory.hearing.corners import BlindBend, find_blind_bend
from arena_auditory.hearing.fleet import FleetRobots
from arena_auditory.hearing.policy import LevelTrend, PolicyConfig, State, YieldMachine, compose_masks, mass_split, paint_lane
from arena_auditory.params import Configuration, configure
from arena_auditory.shared import RobotBinding
from arena_auditory.transforms import ThreadedTransformListener

PAST_BEND_M = 0.3
SPEED_FILTER_PERCENT = 1


@attrs.define(eq=False)
class _Robot:
    binding: RobotBinding
    max_mps: float
    machine: YieldMachine
    trend: LevelTrend
    pub_mask: Publisher
    pub_state: Publisher
    pub_markers: LazyPublisher[MarkerArray]
    pub_filter_info: Publisher
    subs: list[Subscription] = attrs.Factory(list)
    belief: OccupancyGrid | None = None
    plan_xy: np.ndarray | None = None
    plan_dirty: bool = False
    bend: BlindBend | None = None
    consumed_xy: tuple[float, float] | None = None
    consumed_at: tuple[float, float] | None = None
    last_event_t: float = -1e9
    receding_since: float | None = None
    time_yielding: float = 0.0
    last_tick: float | None = None

    def pct(self, mps: float) -> int:
        if self.max_mps <= 0.0:
            return 0
        return int(max(1, min(100, round(100.0 * mps / self.max_mps))))


class PolicyNode(ArenaMixinNode):
    def __init__(self) -> None:
        super().__init__("hearing_policy")
        self.conf = Configuration(self)
        self._map_conf = self.conf.Map
        p = self.conf.Policy
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = ThreadedTransformListener(self._tf_buffer)
        self._tg = self.conf.Hearing.TG_NODE.value
        self._frontend = self.conf.Hearing.FRONTEND.value
        self._map: OccupancyGrid | None = None
        self._occupied: np.ndarray | None = None
        self.create_subscription(OccupancyGrid, f"{self._tg}/{MAP}", self._cb_map, qos.latched())
        self.create_subscription(Bool, f"{self._tg}/{STATE_RESETTING}", self._cb_reset, qos.latched())
        self._rate = p.PUBLISH_RATE_HZ.value
        self._robots = FleetRobots(self, self._tg, start=self._start, stop=self._stop, channel=self._channel)
        self.create_timer(1.0 / self._rate, self._on_timer)
        self.get_logger().info(f"hearing_policy up: listen={p.LISTEN_ENABLED.value} yield={p.YIELD_ENABLED.value}")

    def destroy_node(self) -> bool:
        self._tf_listener.close()
        return super().destroy_node()

    def _start(self, binding: RobotBinding) -> _Robot:
        if binding.error:
            self.get_logger().warning(f"{binding.error}, poses use {binding.base_frame!r}")
        p = self.conf.Policy
        scope = f"{self._tg}/{binding.name}"
        robot = _Robot(
            binding=binding,
            max_mps=p.MAX_LINEAR_MPS.value if p.MAX_LINEAR_MPS.value > 0.0 else binding.max_linear_mps,
            machine=YieldMachine(configure(PolicyConfig, p)),
            trend=LevelTrend(level_trend_tau_s=p.LEVEL_TREND_TAU_S.value),
            pub_mask=self.create_publisher(OccupancyGrid, f"{scope}/{SPEED_FILTER_MASK}", qos.latched()),
            pub_state=self.create_publisher(String, f"{scope}/{POLICY_STATE}", qos.latched()),
            pub_markers=LazyPublisher(self.create_publisher(MarkerArray, f"{scope}/{POLICY_MARKERS}", qos.reliable(1))),
            pub_filter_info=self.create_publisher(CostmapFilterInfo, f"{scope}/{COSTMAP_FILTER_INFO}", qos.latched()),
        )
        robot.subs = [
            self.create_subscription(OccupancyGrid, f"{scope}/{BELIEF_GRID}", functools.partial(self._cb_belief, robot), qos.reliable(1)),
            self.create_subscription(Path, f"{self._tg}/{plan(binding.name)}", functools.partial(self._cb_plan, robot), qos.reliable(1)),
            self.create_subscription(AuditoryDetection, f"{self._tg}/{detections(binding.name, self._frontend)}", functools.partial(self._cb_detection, robot), qos.reliable(50)),
        ]
        info = CostmapFilterInfo(type=SPEED_FILTER_PERCENT, filter_mask_topic=robot.pub_mask.topic_name, base=0.0, multiplier=1.0)
        info.header.stamp = self.get_clock().now().to_msg()
        robot.pub_filter_info.publish(info)
        self.get_logger().info(f"hearing_policy bound to {binding.name}: base {binding.base_frame}, max {robot.max_mps:.2f} m/s, mask {robot.pub_mask.topic_name!r}")
        return robot

    def _stop(self, robot: _Robot) -> None:
        for sub in robot.subs:
            self.destroy_subscription(sub)
        for pub in (robot.pub_mask, robot.pub_state, robot.pub_markers.publisher, robot.pub_filter_info):
            self.destroy_publisher(pub)

    def _channel(self, robot: _Robot) -> LockstepChannel:
        return LockstepChannel(name=f"policy/{robot.binding.name}", topic=robot.pub_mask.topic_name, type="nav_msgs/msg/OccupancyGrid", period_s=1.1 / self._rate, hard=True)

    def _cb_map(self, msg: OccupancyGrid) -> None:
        self._map = msg
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self._occupied = data >= self._map_conf.OCCUPIED_THRESHOLD.value

    @staticmethod
    def _cb_belief(robot: _Robot, msg: OccupancyGrid) -> None:
        robot.belief = msg

    @staticmethod
    def _cb_plan(robot: _Robot, msg: Path) -> None:
        robot.plan_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=np.float64) if msg.poses else None
        robot.plan_dirty = True

    def _cb_reset(self, msg: Bool) -> None:
        if not msg.data:
            return
        for robot in self._robots:
            robot.bend = None
            robot.consumed_xy = None
            robot.machine.new_bend()
            robot.machine.state = State.CRUISE
            robot.trend.reset()
            robot.receding_since = None
            robot.time_yielding = 0.0

    def _cb_detection(self, robot: _Robot, msg: AuditoryDetection) -> None:
        level = float(msg.level_db)
        if not math.isfinite(level):
            return
        t = self._now()
        robot.trend.observe(t, level)
        robot.last_event_t = t

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _robot_pose(self, robot: _Robot) -> tuple[float, float, float] | None:
        try:
            tf = self._tf_buffer.lookup_transform(self._map_conf.FRAME.value, robot.binding.base_frame, Time())
        except tf2_ros.TransformException:
            return None
        t = tf.transform.translation
        return float(t.x), float(t.y), yaw_from_quat(tf.transform.rotation)

    def _update_bend(self, robot: _Robot, robot_xy: tuple[float, float]) -> None:
        """Recompute on a new plan only, a bend within the hysteresis of a consumed one stays consumed."""
        p = self.conf.Policy
        if robot.plan_xy is None or self._occupied is None or self._map is None:
            robot.bend = None
            return
        if not robot.plan_dirty:
            return
        robot.plan_dirty = False
        info = self._map.info
        found = find_blind_bend(
            robot.plan_xy,
            self._occupied,
            (info.origin.position.x, info.origin.position.y),
            float(info.resolution),
            robot_xy,
            lookahead_m=p.LOOKAHEAD_M.value,
            approach_m=p.APPROACH_M.value,
            hold_len_m=p.HOLD_LEN_M.value,
            hold_offset_m=p.HOLD_OFFSET_M.value,
        )
        if found is None:
            robot.bend = None
            return
        hyst = p.BEND_HYSTERESIS_M.value
        if robot.consumed_xy is not None and math.dist(found.bend_xy, robot.consumed_xy) <= hyst:
            if robot.consumed_at is not None and math.dist(robot_xy, robot.consumed_at) >= p.REARM_AFTER_M.value:
                robot.consumed_xy = None
            else:
                robot.bend = None
                return
        if robot.bend is None or math.dist(found.bend_xy, robot.bend.bend_xy) > hyst:
            robot.machine.new_bend()
            robot.trend.reset()
            robot.receding_since = None
        robot.bend = found

    def _on_timer(self) -> None:
        if self._map is None:
            return
        for robot in self._robots:
            self._tick(robot)

    def _tick(self, robot: _Robot) -> None:
        if robot.belief is None:
            return
        pose = self._robot_pose(robot)
        if pose is None:
            return
        now = self._now()
        dt = 0.0 if robot.last_tick is None else max(now - robot.last_tick, 0.0)
        robot.last_tick = now
        p = self.conf.Policy
        robot_xy = (pose[0], pose[1])
        self._update_bend(robot, robot_xy)

        info = robot.belief.info
        origin = (info.origin.position.x, info.origin.position.y)
        res = float(info.resolution)
        belief = np.asarray(robot.belief.data, dtype=np.float32).reshape(info.height, info.width) / 100.0
        layers = [speed_mask_from_belief(belief, res, configure(PolicyConfig, self.conf.Policy))]
        shape = belief.shape

        state = State.CRUISE
        frac = ahead = behind = total = 0.0
        dist = float("nan")
        bend = robot.bend
        if bend is not None:
            dist = bend.dist_from(robot.plan_xy, robot_xy) if robot.plan_xy is not None else float("nan")
            in_approach = dist <= p.APPROACH_M.value
            past = dist <= PAST_BEND_M
            ahead, behind, total = mass_split(belief, origin, res, bend.bend_xy, p.CORNER_RADIUS_M.value, robot_xy, bend.direction)
            frac = ahead / total if total > 0.0 else 0.0
            if robot.trend.slope(now) < p.LEVEL_TREND_DB_PER_S.value:
                if robot.receding_since is None:
                    robot.receding_since = now
            else:
                robot.receding_since = None
            receding_s = now - robot.receding_since if robot.receding_since is not None else 0.0
            state = robot.machine.step(now, in_approach=in_approach, past_bend=past, frac_ahead=frac, ahead=ahead, behind=behind, event_age_s=now - robot.last_event_t, receding_s=receding_s)
            if not p.YIELD_ENABLED.value and state is State.YIELD:
                state = robot.machine.state = State.LISTEN
            if past:
                robot.consumed_xy, robot.consumed_at = bend.bend_xy, robot_xy
                robot.bend = None
            elif p.LISTEN_ENABLED.value and state in (State.LISTEN, State.YIELD, State.PASS):
                layers.append(paint_lane(shape, origin, res, bend.approach_xy, p.LANE_RADIUS_M.value, robot.pct(p.LISTEN_MPS.value)))
            if state is State.YIELD:
                layers.append(paint_lane(shape, origin, res, bend.hold_xy, p.LANE_RADIUS_M.value, robot.pct(p.HOLD_MPS.value)))
                robot.time_yielding += dt

        mask = compose_masks(*layers)
        out = OccupancyGrid()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._map_conf.FRAME.value
        out.info = info
        out.data = array.array("b", np.ascontiguousarray(mask, dtype=np.int8).tobytes())
        robot.pub_mask.publish(out)
        binding = np.zeros(shape, dtype=np.int8)
        if len(layers) > 1:
            binding = np.argmin(np.where(np.stack(layers) > 0, np.stack(layers), 127), axis=0).astype(np.int8)
        r, c = int((robot_xy[1] - origin[1]) / res), int((robot_xy[0] - origin[0]) / res)
        inside = 0 <= r < shape[0] and 0 <= c < shape[1]
        at_robot = int(mask[r, c]) if inside else 0
        layer_at_robot = ("belief", "listen", "hold")[int(binding[r, c])] if at_robot and inside else "none"
        robot.pub_state.publish(
            String(
                data=json.dumps(
                    {
                        "state": state.value,
                        "dist_to_bend_m": None if math.isnan(dist) else round(dist, 2),
                        "frac_ahead": round(frac, 3),
                        "mass_ahead": round(ahead, 3),
                        "mass_behind": round(behind, 3),
                        "mass_total": round(total, 3),
                        "level_slope_db_s": round(robot.trend.slope(now), 2),
                        "limit_pct": at_robot,
                        "binding_layer": layer_at_robot,
                        "yield_count": robot.machine.yield_count,
                        "time_yielding_s": round(robot.time_yielding, 2),
                    }
                )
            )
        )
        self._publish_markers(robot, bend, state)

    def _publish_markers(self, robot: _Robot, bend: BlindBend | None, state: State) -> None:
        if not robot.pub_markers.wanted:
            return
        arr = MarkerArray()
        frame = self._map_conf.FRAME.value
        stamp = self.get_clock().now().to_msg()
        for mid, (pts, color) in enumerate(((None if bend is None else bend.approach_xy, ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.6)), (None if bend is None else bend.hold_xy, ColorRGBA(r=1.0, g=0.3, b=0.2, a=0.8)))):
            m = Marker()
            m.header.frame_id, m.header.stamp, m.ns, m.id = frame, stamp, "hearing_policy", mid
            if pts is None or len(pts) == 0 or (mid == 1 and state is not State.YIELD):
                m.action = Marker.DELETE
            else:
                m.type, m.action = Marker.LINE_STRIP, Marker.ADD
                m.scale.x = 0.08
                m.color = color
                m.pose.orientation.w = 1.0
                m.points = [Point(x=float(x), y=float(y), z=0.08) for x, y in pts]
            arr.markers.append(m)
        robot.pub_markers.publish(lambda: arr)


def main() -> None:
    PolicyNode.run_main()
