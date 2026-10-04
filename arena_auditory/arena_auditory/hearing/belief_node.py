"""Directional belief of every fleet robot, from its auditory detections.

Per robot, subscribes to ``<tg_node>/<robot>/hearing/<frontend>/detections``,
maintains a decaying pedestrian-likelihood grid in the map frame
(``belief_grid.BeliefGrid``), and publishes below ``<tg_node>/<robot>``

  1. the grid as a 0..100 ``OccupancyGrid`` (``hearing/belief_grid``), consumed by
     ``hearing_policy``, which turns it into the robot's Nav2 ``SpeedFilter`` mask, and by RViz,
  2. optionally, a ``MarkerArray`` drawing the wedge of each detection, fading
     with the grid's own decay so the fan shows what still carries mass.

Only (kind, azimuth, level, stamp) are read off a detection, so the layer is a
fair stand-in for one fed by a real front-end.

Bearing convention
------------------
``AuditoryDetection.azimuth_rad`` is CCW from +x of ``header.frame_id``, and
the consumer is the side that rotates. Bus detections are map-frame by
construction, front-end detections are in the array mount frame, so the
azimuth is rotated by the TF yaw of the detection frame in the map frame.
"""

from __future__ import annotations

import array
import functools
import math
from collections import deque

import attrs
import numpy as np
import tf2_ros
from arena_auditory_msgs.msg import AuditoryDetection
from arena_rclpy_mixins import ArenaMixinNode, qos
from arena_runtime_msgs.msg import LockstepChannel
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.time import Time
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from arena_auditory.assets import SoundLibrary
from arena_auditory.constants import BELIEF_GRID, BELIEF_WEDGES, MAP, STATE_RESETTING, detections
from arena_auditory.hearing.belief_grid import BeliefConfig, BeliefGrid, emission_levels
from arena_auditory.hearing.fleet import FleetRobots
from arena_auditory.params import Configuration, configure
from arena_auditory.shared import RobotBinding
from arena_auditory.transforms import ThreadedTransformListener

DEFAULT_WEDGE_COLOR = (0.9, 0.9, 0.9)
WEDGE_EXPIRE_TAU = 1.5
WEDGE_STEPS = 8


def yaw_from_quat(q: Quaternion) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


@attrs.frozen
class _Wedge:
    id: int
    t_ns: int
    x: float
    y: float
    theta: float
    range_m: float
    kind: str


@attrs.frozen(eq=False)
class _Layout:
    """Map-frame rectangle and free cells every robot's grid covers."""

    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int
    free: np.ndarray | None = None


@attrs.define(eq=False)
class _Robot:
    binding: RobotBinding
    pub_belief: Publisher
    pub_markers: Publisher | None
    sub: Subscription = attrs.field(init=False)
    grid: BeliefGrid | None = None
    wedges: deque[_Wedge] = attrs.Factory(deque)
    next_wedge_id: int = 0
    retired: list[int] = attrs.Factory(list)
    dropped_no_pose: int = 0
    dropped_no_bearing: int = 0


class BeliefNode(ArenaMixinNode):
    def __init__(self) -> None:
        super().__init__("hearing_belief")
        self.conf = Configuration(self)
        self._map_conf = self.conf.Map
        library = SoundLibrary.default()
        self._kinds = library.kinds()
        self._emission_db = self.conf.Belief.emission_db(emission_levels(library))
        self._config = self._belief_config()
        self._layout: _Layout | None = None
        self._map_info: MapMetaData | None = None
        self._last_update = self.get_clock().now()
        self._tg = self.conf.Hearing.TG_NODE.value
        self._frontend = self.conf.Hearing.FRONTEND.value
        self._markers_enabled = self.conf.Belief.MARKERS_ENABLED.value

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = ThreadedTransformListener(self._tf_buffer)

        if self.conf.Belief.STANDALONE_ENABLED.value:
            self._build_standalone_layout()
        self._rate = self.conf.Belief.PUBLISH_RATE_HZ.value
        self.create_subscription(OccupancyGrid, f"{self._tg}/{MAP}", self._cb_map, qos.latched())
        self.create_subscription(Bool, f"{self._tg}/{STATE_RESETTING}", self._cb_reset, qos.latched())
        self._robots = FleetRobots(self, self._tg, start=self._start, stop=self._stop, channel=self._channel)
        self.create_timer(1.0 / self._rate, self._on_timer)
        self.get_logger().info(f"hearing_belief up: frontend {self._frontend}, kinds {list(self.conf.Belief.KINDS.value)}")

    def destroy_node(self) -> bool:
        self._tf_listener.close()
        return super().destroy_node()

    def _start(self, binding: RobotBinding) -> _Robot:
        if binding.error:
            self.get_logger().warning(f"{binding.error}, poses use {binding.base_frame!r}")
        robot = _Robot(
            binding=binding,
            pub_belief=self.create_publisher(OccupancyGrid, f"{self._tg}/{binding.name}/{BELIEF_GRID}", qos.reliable(1)),
            pub_markers=self.create_publisher(MarkerArray, f"{self._tg}/{binding.name}/{BELIEF_WEDGES}", qos.reliable(1)) if self._markers_enabled else None,
            grid=self._new_grid(),
        )
        robot.sub = self.create_subscription(AuditoryDetection, f"{self._tg}/{detections(binding.name, self._frontend)}", functools.partial(self._cb_detection, robot), qos.reliable(50))
        self.get_logger().info(f"hearing_belief: detections on {robot.sub.topic_name!r} -> {robot.pub_belief.topic_name!r}, robot frame {binding.base_frame!r}")
        return robot

    def _stop(self, robot: _Robot) -> None:
        self.destroy_subscription(robot.sub)
        self.destroy_publisher(robot.pub_belief)
        if robot.pub_markers is not None:
            self.destroy_publisher(robot.pub_markers)

    def _channel(self, robot: _Robot) -> LockstepChannel:
        return LockstepChannel(name=f"belief/{robot.binding.name}", topic=robot.pub_belief.topic_name, type="nav_msgs/msg/OccupancyGrid", period_s=1.1 / self._rate, hard=True)

    def _belief_config(self) -> BeliefConfig:
        return configure(BeliefConfig, self.conf.Belief, emission_db={kind: param.value for kind, param in self._emission_db.items()})

    def _new_grid(self) -> BeliefGrid | None:
        layout = self._layout
        if layout is None:
            return None
        return BeliefGrid(layout.origin_x, layout.origin_y, layout.resolution, layout.width, layout.height, self._config, free=layout.free)

    def _build_standalone_layout(self) -> None:
        b = self.conf.Belief
        res = b.RESOLUTION_M.value or 0.1
        ox, oy = b.STANDALONE_ORIGIN_M.value[:2]
        sx, sy = b.STANDALONE_SIZE_M.value[:2]
        self._layout = _Layout(ox, oy, res, int(sx / res), int(sy / res))
        self.get_logger().info(f"standalone belief grid {self._layout.width}x{self._layout.height} @ {res} m")

    def _cb_reset(self, msg: Bool) -> None:
        if not msg.data:
            return
        for robot in self._robots:
            if robot.grid is not None:
                robot.grid.clear()
                robot.retired.extend(w.id for w in robot.wedges)
                robot.wedges.clear()

    def _cb_map(self, msg: OccupancyGrid) -> None:
        if self.conf.Belief.STANDALONE_ENABLED.value:
            return
        info = msg.info
        last = self._map_info
        if (
            last is not None
            and info.width == last.width
            and info.height == last.height
            and abs(info.resolution - last.resolution) < 1e-9
            and abs(info.origin.position.x - last.origin.position.x) < 1e-9
            and abs(info.origin.position.y - last.origin.position.y) < 1e-9
        ):
            return
        self._map_info = info
        data = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        native_free = data < self._map_conf.OCCUPIED_THRESHOLD.value
        res = self.conf.Belief.RESOLUTION_M.value
        if res <= 0.0:
            self._layout = _Layout(info.origin.position.x, info.origin.position.y, info.resolution, info.width, info.height, native_free)
        else:
            w = max(round(info.width * info.resolution / res), 1)
            h = max(round(info.height * info.resolution / res), 1)
            scale = res / info.resolution
            rows = np.minimum((np.arange(h) * scale).astype(int), info.height - 1)
            cols = np.minimum((np.arange(w) * scale).astype(int), info.width - 1)
            self._layout = _Layout(info.origin.position.x, info.origin.position.y, res, w, h, native_free[rows][:, cols])
        for robot in self._robots:
            robot.grid = self._new_grid()
        layout = self._layout
        self.get_logger().info(f"belief grid {layout.width}x{layout.height} @ {layout.resolution} m origin ({layout.origin_x:.2f}, {layout.origin_y:.2f}) frame {msg.header.frame_id!r}")

    def _lookup(self, frame: str, stamp: TimeMsg) -> tuple[float, float, float] | None:
        """(x, y, yaw) of frame in the map frame at stamp, else at the latest transform."""
        map_frame = self._map_conf.FRAME.value
        for when in (Time.from_msg(stamp), Time()):
            try:
                tf = self._tf_buffer.lookup_transform(map_frame, frame, when)
            except tf2_ros.TransformException:
                continue
            t = tf.transform.translation
            return float(t.x), float(t.y), yaw_from_quat(tf.transform.rotation)
        return None

    def _cb_detection(self, robot: _Robot, msg: AuditoryDetection) -> None:
        if robot.grid is None:
            return
        kinds = self.conf.Belief.KINDS.value
        kind = msg.kind.strip().lower()
        if kinds and kind not in kinds:
            return
        azimuth = float(msg.azimuth_rad)
        if not math.isfinite(azimuth):
            robot.dropped_no_bearing += 1
            return
        level = float(msg.level_db)
        if math.isfinite(level) and level < self.conf.Belief.MIN_LEVEL_DB.value:
            return
        base_frame = robot.binding.base_frame
        pose = self._lookup(base_frame, msg.header.stamp)
        frame = msg.header.frame_id.strip("/")
        if pose is None:
            frame_yaw = None
        elif frame in ("", self._map_conf.FRAME.value):
            frame_yaw = 0.0
        elif frame == base_frame:
            frame_yaw = pose[2]
        else:
            frame_pose = self._lookup(frame, msg.header.stamp)
            frame_yaw = None if frame_pose is None else frame_pose[2]
        if pose is None or frame_yaw is None:
            robot.dropped_no_pose += 1
            if robot.dropped_no_pose % 50 == 1:
                self.get_logger().warning(f"no {self._map_conf.FRAME.value} -> {base_frame} or {frame} transform, dropped {robot.dropped_no_pose} detections")
            return

        x, y, _ = pose
        info = robot.grid.add_event(
            robot_x=x,
            robot_y=y,
            robot_yaw=0.0,
            bearing_rad=azimuth + frame_yaw,
            sound_type=kind,
            received_db=level if math.isfinite(level) else None,
            bearing_frame="map",
        )
        robot.wedges.append(_Wedge(id=robot.next_wedge_id, t_ns=self.get_clock().now().nanoseconds, x=x, y=y, theta=info["theta"], range_m=info["range_m"], kind=kind))
        robot.next_wedge_id += 1

    def _on_timer(self) -> None:
        self._config = self._belief_config()
        now = self.get_clock().now()
        dt = (now - self._last_update).nanoseconds * 1e-9
        self._last_update = now
        stamp = now.to_msg()
        frame = self._map_conf.FRAME.value
        for robot in self._robots:
            if robot.grid is None:
                continue
            robot.grid.config = self._config
            robot.grid.decay(dt)
            robot.pub_belief.publish(self._as_grid(robot.grid, stamp, frame))
            if robot.pub_markers is not None:
                robot.pub_markers.publish(self._wedge_markers(robot, now.nanoseconds, stamp, frame))

    @staticmethod
    def _as_grid(grid: BeliefGrid, stamp: TimeMsg, frame: str) -> OccupancyGrid:
        g = OccupancyGrid()
        g.header.stamp = stamp
        g.header.frame_id = frame
        g.info.resolution = grid.resolution
        g.info.width = grid.width
        g.info.height = grid.height
        g.info.origin.position.x = grid.origin_x
        g.info.origin.position.y = grid.origin_y
        g.info.origin.orientation.w = 1.0
        g.data = array.array("b", np.ascontiguousarray(grid.belief_int8(), dtype=np.int8).tobytes())
        return g

    def _wedge_color(self, kind: str, alpha: float) -> ColorRGBA:
        entry = self._kinds.get(kind)
        r, g, b = entry.color if entry is not None else DEFAULT_WEDGE_COLOR
        return ColorRGBA(r=float(r), g=float(g), b=float(b), a=alpha)

    def _wedge_markers(self, robot: _Robot, now_ns: int, stamp: TimeMsg, frame: str) -> MarkerArray:
        """Every live wedge under a stable id, faded by exp(-age / tau), DELETE for the ones that expired."""
        arr = MarkerArray()
        tau = max(float(self._config.tau_s), 1e-6)
        half = math.radians(float(self._config.wedge_deg)) * 0.5
        keep = max(self.conf.Belief.MARKERS_MAX_COUNT.value, 1)
        while robot.wedges and ((now_ns - robot.wedges[0].t_ns) * 1e-9 > WEDGE_EXPIRE_TAU * tau or len(robot.wedges) > keep):
            robot.retired.append(robot.wedges.popleft().id)
        draw_range = self.conf.Belief.MARKERS_RANGE_M.value
        for wid in robot.retired:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = "hearing_wedge"
            m.id = wid
            m.action = Marker.DELETE
            arr.markers.append(m)
        robot.retired.clear()
        for wedge in robot.wedges:
            rng = min(wedge.range_m, draw_range)
            fade = math.exp(-max(now_ns - wedge.t_ns, 0) * 1e-9 / tau)
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = "hearing_wedge"
            m.id = wedge.id
            m.type = Marker.TRIANGLE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color = self._wedge_color(wedge.kind, float(0.35 * fade))
            for k in range(WEDGE_STEPS):
                a0 = wedge.theta - half + 2.0 * half * k / WEDGE_STEPS
                a1 = wedge.theta - half + 2.0 * half * (k + 1) / WEDGE_STEPS
                m.points.append(Point(x=wedge.x, y=wedge.y, z=0.05))
                m.points.append(Point(x=wedge.x + rng * math.cos(a0), y=wedge.y + rng * math.sin(a0), z=0.05))
                m.points.append(Point(x=wedge.x + rng * math.cos(a1), y=wedge.y + rng * math.sin(a1), z=0.05))
            arr.markers.append(m)
        return arr


def main() -> None:
    BeliefNode.run_main()
