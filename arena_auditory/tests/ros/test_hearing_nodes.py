"""hearing_belief and hearing_policy as real processes, bound to a published fleet in a private namespace."""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

TG = "task_generator_node"
MODEL = "jackal"
STARTUP_S = 60.0


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("scipy")
    pytest.importorskip("geometry_msgs.msg")
    pytest.importorskip("nav_msgs.msg")
    pytest.importorskip("std_msgs.msg")
    pytest.importorskip("tf2_msgs.msg")
    pytest.importorskip("arena_auditory_msgs.msg")
    pytest.importorskip("task_generator_msgs.msg")
    pytest.importorskip("visualization_msgs.msg")


def _param_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_param_text(item) for item in value) + "]"
    return str(value)


@contextlib.contextmanager
def _hearing_process(module: str, ns: str, params: dict[str, object], log: Path) -> Iterator[subprocess.Popen]:
    args = [sys.executable, "-c", f"from arena_auditory.hearing.{module} import main; main()", "--ros-args", "-r", f"__ns:={ns}", "-r", f"/tf:={ns}/tf", "-r", f"/tf_static:={ns}/tf_static"]
    for name, value in params.items():
        args += ["-p", f"{name}:={_param_text(value)}"]
    with log.open("w") as out:
        proc = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            yield proc
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)


class _Rig:
    """Test-side node in the namespace of one hearing process."""

    def __init__(self, ns: str, proc: subprocess.Popen, log: Path) -> None:
        import rclpy.node

        self.ns = ns
        self.tg = f"{ns}/{TG}"
        self.proc = proc
        self.log = log
        self.node = rclpy.node.Node(f"probe_{uuid.uuid4().hex[:8]}", namespace=ns)

    def spin_until(self, predicate: Callable[[], bool], timeout_s: float, what: str, tick: Callable[[], None] | None = None) -> None:
        import rclpy

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if tick is not None:
                tick()
            if predicate():
                return
            assert self.proc.poll() is None, f"node exited with {self.proc.returncode}:\n{self.log.read_text()[-4000:]}"
        raise AssertionError(f"timed out waiting for {what}:\n{self.log.read_text()[-4000:]}")

    def spin_for(self, duration_s: float) -> None:
        import rclpy

        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def destroy(self) -> None:
        self.node.destroy_node()


@contextlib.contextmanager
def _rig(module: str, params: dict[str, object], tmp_path: Path) -> Iterator[_Rig]:
    ns = f"/t_{uuid.uuid4().hex[:8]}"
    log = tmp_path / f"{module}.log"
    with _hearing_process(module, ns, params, log) as proc:
        rig = _Rig(ns, proc, log)
        try:
            yield rig
        finally:
            rig.destroy()


def _prefix(rig: _Rig, name: str) -> str:
    return f"{rig.ns.strip('/')}/{name}"


def _fleet(rig: _Rig, names: tuple[str, ...]):
    from task_generator_msgs.msg import RobotFleet, RobotState

    fleet = RobotFleet()
    for name in names:
        state = RobotState()
        state.descriptor.name = name
        state.descriptor.model = MODEL
        state.descriptor.ns = f"{rig.ns}/{name}"
        state.descriptor.frame = _prefix(rig, name)
        fleet.robots.append(state)
    return fleet


def _base_frame(rig: _Rig, name: str) -> str:
    from arena_auditory.shared import RobotBinding

    return RobotBinding.resolve(name=name, model=MODEL, namespace=f"{rig.ns}/{name}", frame_prefix=_prefix(rig, name)).base_frame


def _publish_fleet(rig: _Rig, names: tuple[str, ...]) -> None:
    from arena_rclpy_mixins import qos
    from task_generator_msgs.msg import RobotFleet

    pub = rig.node.create_publisher(RobotFleet, f"{rig.tg}/state/robots", qos.latched())
    pub.publish(_fleet(rig, names))


def _publish_tf(rig: _Rig, child: str, x: float, y: float, yaw: float = 0.0) -> None:
    from arena_rclpy_mixins import qos
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage

    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = child
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.z = math.sin(yaw / 2.0)
    transform.transform.rotation.w = math.cos(yaw / 2.0)
    pub = rig.node.create_publisher(TFMessage, f"{rig.ns}/tf_static", qos.latched())
    pub.publish(TFMessage(transforms=[transform]))


def _detection(frame_id: str, azimuth_rad: float, level_db: float = 50.0):
    from arena_auditory_msgs.msg import AuditoryDetection

    msg = AuditoryDetection()
    msg.header.frame_id = frame_id
    msg.frontend = "bus"
    msg.kind = "footstep"
    msg.azimuth_rad = azimuth_rad
    msg.elevation_rad = math.nan
    msg.level_db = level_db
    msg.confidence = 1.0
    return msg


def _detection_publisher(rig: _Rig, name: str):
    from arena_auditory_msgs.msg import AuditoryDetection
    from arena_rclpy_mixins import qos

    pub = rig.node.create_publisher(AuditoryDetection, f"{rig.tg}/{name}/hearing/bus/detections", qos.reliable(50))
    rig.spin_until(lambda: pub.get_subscription_count() > 0, STARTUP_S, f"hearing_belief to bind {name}")
    return pub


def _grids(rig: _Rig, name: str) -> list:
    from arena_rclpy_mixins import qos
    from nav_msgs.msg import OccupancyGrid

    received: list[OccupancyGrid] = []
    rig.node.create_subscription(OccupancyGrid, f"{rig.tg}/{name}/hearing/belief_grid", received.append, qos.reliable(10))
    return received


def _as_array(grid) -> np.ndarray:
    return np.asarray(grid.data, dtype=np.int8).reshape(grid.info.height, grid.info.width)


def _belief_params(**given: object) -> dict[str, object]:
    return {
        "belief.standalone.enabled": True,
        "belief.standalone.origin_m": [-10.0, -10.0],
        "belief.standalone.size_m": [20.0, 20.0],
        "belief.markers.enabled": False,
        **given,
    }


def test_belief_node_paints_the_wedge_of_each_robot_along_its_bearing_and_not_behind(tmp_path: Path) -> None:
    with _rig("belief_node", _belief_params(), tmp_path) as rig:
        _publish_fleet(rig, ("r0", "r1"))
        _publish_tf(rig, _base_frame(rig, "r0"), 0.0, 0.0)
        _publish_tf(rig, _base_frame(rig, "r1"), 3.0, 3.0)
        r0_grids, r1_grids = _grids(rig, "r0"), _grids(rig, "r1")
        pub = _detection_publisher(rig, "r0")
        _detection_publisher(rig, "r1")
        pub.publish(_detection("map", 0.0))
        rig.spin_until(lambda: bool(r0_grids) and int(_as_array(r0_grids[-1]).sum()) > 0, 10.0, "a painted r0 belief grid")
        rig.spin_until(lambda: len(r1_grids) >= 3, 10.0, "r1 belief grids")

        grid_msg = r0_grids[-1]
        info = grid_msg.info
        grid = _as_array(grid_msg)
        origin_col = int(round((0.0 - info.origin.position.x) / info.resolution))
        assert grid[:, :origin_col].sum() == 0
        assert grid[:, origin_col:].sum() > 0
        assert all(int(_as_array(msg).sum()) == 0 for msg in r1_grids)


def test_belief_node_rotates_an_array_frame_azimuth_by_the_frame_yaw(tmp_path: Path) -> None:
    with _rig("belief_node", _belief_params(), tmp_path) as rig:
        _publish_fleet(rig, ("r0",))
        base = _base_frame(rig, "r0")
        _publish_tf(rig, base, 0.0, 0.0, yaw=math.pi / 2)
        grids = _grids(rig, "r0")
        pub = _detection_publisher(rig, "r0")
        rig.spin_until(lambda: bool(grids) and int(_as_array(grids[-1]).sum()) > 0, 10.0, "a painted belief grid", tick=lambda: pub.publish(_detection(base, 0.0)))

        info = grids[-1].info
        grid = _as_array(grids[-1])
        origin_row = int(round((0.0 - info.origin.position.y) / info.resolution))
        assert grid[:origin_row, :].sum() == 0
        assert grid[origin_row:, :].sum() > 0


def test_belief_node_drops_a_detection_with_nan_azimuth(tmp_path: Path) -> None:
    with _rig("belief_node", _belief_params(), tmp_path) as rig:
        _publish_fleet(rig, ("r0",))
        _publish_tf(rig, _base_frame(rig, "r0"), 0.0, 0.0)
        grids = _grids(rig, "r0")
        pub = _detection_publisher(rig, "r0")
        pub.publish(_detection("map", math.nan))
        rig.spin_for(1.0)
        settled = len(grids)
        assert settled > 0
        assert all(int(_as_array(msg).sum()) == 0 for msg in grids)

        pub.publish(_detection("map", 0.0))
        rig.spin_until(lambda: len(grids) > settled and int(_as_array(grids[-1]).sum()) > 0, 10.0, "the valid detection painted")


def test_belief_node_drops_detections_until_the_robot_transform_exists(tmp_path: Path) -> None:
    with _rig("belief_node", _belief_params(), tmp_path) as rig:
        _publish_fleet(rig, ("r0",))
        base = _base_frame(rig, "r0")
        grids = _grids(rig, "r0")
        pub = _detection_publisher(rig, "r0")
        pub.publish(_detection("map", 0.0))
        pub.publish(_detection(base, 0.0))
        rig.spin_for(1.0)
        settled = len(grids)
        assert settled > 0
        assert all(int(_as_array(msg).sum()) == 0 for msg in grids)

        _publish_tf(rig, base, 2.0, 1.0)
        rig.spin_for(0.5)
        pub.publish(_detection("map", 0.0))
        rig.spin_until(lambda: len(grids) > settled and int(_as_array(grids[-1]).sum()) > 0, 10.0, "a painted belief grid once the transform exists")
        info = grids[-1].info
        grid = _as_array(grids[-1])
        rows, cols = np.nonzero(grid == grid.max())
        xs = info.origin.position.x + (cols + 0.5) * info.resolution
        ys = info.origin.position.y + (rows + 0.5) * info.resolution
        assert float(xs.min()) > 2.0
        assert float(np.abs(ys - 1.0).max()) < 0.5


def test_belief_node_wedge_markers_add_then_expire(tmp_path: Path) -> None:
    from arena_rclpy_mixins import qos
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    from visualization_msgs.msg import Marker, MarkerArray

    with _rig("belief_node", _belief_params(**{"belief.markers.enabled": True}), tmp_path) as rig:
        _publish_fleet(rig, ("r0",))
        _publish_tf(rig, _base_frame(rig, "r0"), 0.0, 0.0)
        rig.spin_for(1.0)
        received: list[MarkerArray] = []
        rig.node.create_subscription(MarkerArray, f"{rig.tg}/r0/hearing/belief_wedges", received.append, qos.reliable(10))
        pub = _detection_publisher(rig, "r0")
        pub.publish(_detection("map", 0.0))
        pub.publish(_detection("map", 0.3))

        def added() -> set[int]:
            return {m.id for m in received[-1].markers if m.action == Marker.ADD} if received else set()

        rig.spin_until(lambda: len(added()) == 2, 10.0, "two wedge markers")
        added_ids = added()

        client = rig.node.create_client(SetParameters, f"{rig.ns}/hearing_belief/set_parameters")
        rig.spin_until(client.service_is_ready, 10.0, "the set_parameters service")
        request = SetParameters.Request(parameters=[Parameter(name="belief.tau_s", value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=0.001))])
        future = client.call_async(request)
        rig.spin_until(future.done, 10.0, "set belief.tau_s")
        assert all(result.successful for result in future.result().results)

        deleted: set[int] = set()

        def collect_deleted() -> bool:
            for array in received:
                deleted.update(m.id for m in array.markers if m.action == Marker.DELETE)
            received.clear()
            return deleted >= added_ids

        rig.spin_until(collect_deleted, 10.0, "the wedge markers to expire")
        assert deleted == added_ids


def _grid_msg(size: int, resolution: float, origin: float, data: np.ndarray):
    from nav_msgs.msg import OccupancyGrid

    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.info.resolution = resolution
    msg.info.width = size
    msg.info.height = size
    msg.info.origin.position.x = origin
    msg.info.origin.position.y = origin
    msg.info.origin.orientation.w = 1.0
    msg.data = np.asarray(data, dtype=np.int8).reshape(-1).tolist()
    return msg


def _l_shaped_occupancy(size: int, resolution: float) -> np.ndarray:
    """True = occupied. Free legs: a 0.5-10.0 x-leg at y in [0.4, 1.6], a 1.0-10.0 y-leg at x in [9.4, 10.6]."""
    rows = (np.arange(size) + 0.5) * resolution
    cols = (np.arange(size) + 0.5) * resolution
    xx, yy = np.meshgrid(cols, rows)
    occupied = np.ones((size, size), dtype=bool)
    leg_a = (xx >= 0.5) & (xx <= 10.0) & (yy >= 0.4) & (yy <= 1.6)
    leg_b = (yy >= 1.0) & (yy <= 10.0) & (xx >= 9.4) & (xx <= 10.6)
    occupied[leg_a | leg_b] = False
    return occupied


def _l_shaped_plan() -> np.ndarray:
    xs = np.round(np.arange(1.0, 10.0 + 1e-9, 0.1), 6)
    leg_a = np.stack([xs, np.full_like(xs, 1.0)], axis=1)
    ys = np.round(np.arange(1.1, 9.5 + 1e-9, 0.1), 6)
    leg_b = np.stack([np.full_like(ys, 10.0), ys], axis=1)
    return np.concatenate([leg_a, leg_b], axis=0)


class _PolicyIo:
    """Map, belief and plan in, mask and state out, for robot r0."""

    def __init__(self, rig: _Rig) -> None:
        from arena_rclpy_mixins import qos
        from nav_msgs.msg import OccupancyGrid, Path
        from std_msgs.msg import String

        self.rig = rig
        self.masks: list[OccupancyGrid] = []
        self.states: list[dict] = []
        self.map_pub = rig.node.create_publisher(OccupancyGrid, f"{rig.tg}/map", qos.latched())
        self.belief_pub = rig.node.create_publisher(OccupancyGrid, f"{rig.tg}/r0/hearing/belief_grid", qos.reliable(1))
        self.plan_pub = rig.node.create_publisher(Path, f"{rig.tg}/r0/plan", qos.reliable(1))
        rig.node.create_subscription(OccupancyGrid, f"{rig.tg}/r0/hearing/speed_filter_mask", self.masks.append, qos.latched())
        rig.node.create_subscription(String, f"{rig.tg}/r0/hearing/policy_state", lambda msg: self.states.append(json.loads(msg.data)), qos.latched())
        rig.spin_until(lambda: self.belief_pub.get_subscription_count() > 0 and self.plan_pub.get_subscription_count() > 0, STARTUP_S, "hearing_policy to bind r0")


def test_policy_node_cruise_speed_mask_around_blob(tmp_path: Path) -> None:
    params = {"policy.max_linear_mps": 1.0, "policy.listen.enabled": False, "policy.yield.enabled": False}
    with _rig("policy_node", params, tmp_path) as rig:
        _publish_fleet(rig, ("r0",))
        _publish_tf(rig, _base_frame(rig, "r0"), 0.0, 0.0)
        io = _PolicyIo(rig)

        size, resolution, origin = 100, 0.1, -5.0
        blob_xy = (2.0, 0.0)
        centers = origin + (np.arange(size) + 0.5) * resolution
        xx, yy = np.meshgrid(centers, centers)
        belief = _grid_msg(size, resolution, origin, np.where(np.hypot(xx - blob_xy[0], yy - blob_xy[1]) <= 0.3, 100, 0))
        io.map_pub.publish(_grid_msg(size, resolution, origin, np.zeros((size, size))))
        rig.spin_until(lambda: bool(io.masks) and bool(io.states), 10.0, "a mask and a state", tick=lambda: io.belief_pub.publish(belief))

        mask = _as_array(io.masks[-1])

        def cell(x: float, y: float) -> tuple[int, int]:
            return int((y - origin) / resolution), int((x - origin) / resolution)

        assert mask[cell(*blob_xy)] == 40
        assert mask[cell(-4.0, -4.0)] == 0
        assert io.states[-1]["state"] == "cruise"


def test_policy_node_listen_lane_on_approach_to_blind_bend(tmp_path: Path) -> None:
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path as PathMsg

    with _rig("policy_node", {"policy.max_linear_mps": 1.0, "policy.listen.enabled": True}, tmp_path) as rig:
        robot_xy = (7.0, 1.0)
        _publish_fleet(rig, ("r0",))
        _publish_tf(rig, _base_frame(rig, "r0"), *robot_xy)
        io = _PolicyIo(rig)

        size, resolution = 200, 0.1
        io.map_pub.publish(_grid_msg(size, resolution, 0.0, np.where(_l_shaped_occupancy(size, resolution), 100, 0)))
        belief = _grid_msg(size, resolution, 0.0, np.zeros((size, size)))
        path = PathMsg()
        path.header.frame_id = "map"
        for x, y in _l_shaped_plan():
            pose = PoseStamped()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        io.plan_pub.publish(path)

        rig.spin_until(lambda: bool(io.states) and io.states[-1]["state"] == "listen", 10.0, "the listen state", tick=lambda: io.belief_pub.publish(belief))
        payload = io.states[-1]
        assert payload["dist_to_bend_m"] is not None
        assert math.isfinite(payload["dist_to_bend_m"])
        assert payload["binding_layer"] == "listen"

        seen = len(io.masks)
        rig.spin_until(lambda: len(io.masks) > seen, 10.0, "a listen mask", tick=lambda: io.belief_pub.publish(belief))
        mask = _as_array(io.masks[-1])
        row, col = int(robot_xy[1] / resolution), int(robot_xy[0] / resolution)
        assert mask[row, col] == 20
