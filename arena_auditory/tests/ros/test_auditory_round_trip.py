from __future__ import annotations

import asyncio
import math
import time
import uuid

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_auditory_msgs.msg")
    pytest.importorskip("arena_people_msgs.msg")
    pytest.importorskip("geometry_msgs.msg")
    pytest.importorskip("nav_msgs.msg")
    pytest.importorskip("task_generator_msgs.msg")
    pytest.importorskip("tf2_ros")
    pytest.importorskip("visualization_msgs.msg")


@pytest.fixture
def loop():
    event_loop = asyncio.new_event_loop()
    try:
        yield event_loop
    finally:
        event_loop.close()


def _build(event_loop, node_cls, namespace: str, params: dict[str, object] | None = None):
    import rclpy
    from arena_rclpy_mixins import ArenaMixinNode

    overrides = [rclpy.Parameter(name, value=value) for name, value in (params or {}).items()]

    class _Scope(ArenaMixinNode):
        def __init__(self, name: str, **kwargs: object) -> None:
            super().__init__(name, namespace=namespace, parameter_overrides=overrides, **kwargs)

    scoped = type(node_cls.__name__, (node_cls, _Scope), {})

    async def build():
        return scoped()

    return event_loop.run_until_complete(build())


def _spin_until(rclpy, nodes, predicate, timeout_sec: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.01)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS round-trip")


def _robot_fleet(robot_name: str, namespace: str, frame: str):
    from task_generator_msgs.msg import RobotDescriptor, RobotFleet, RobotState

    robot = RobotDescriptor()
    robot.name = robot_name
    robot.model = "jackal"
    robot.ns = namespace
    robot.frame = frame
    state = RobotState()
    state.descriptor = robot
    fleet = RobotFleet()
    fleet.robots.append(state)
    return fleet


def _free_map(size: int = 20):
    from nav_msgs.msg import OccupancyGrid

    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.resolution = 1.0
    grid.info.width = size
    grid.info.height = size
    grid.info.origin.position.x = -size / 2.0
    grid.info.origin.position.y = -size / 2.0
    grid.info.origin.orientation.w = 1.0
    grid.data = [0] * (size * size)
    return grid


def _base_transform(child_frame: str, x: float, y: float):
    from geometry_msgs.msg import TransformStamped

    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = child_frame
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.w = 1.0
    return transform


def _pedestrian(ped_id: int, x: float, y: float, yaw_rad: float):
    from arena_people_msgs.msg import Pedestrian

    ped = Pedestrian()
    ped.id = ped_id
    ped.name = f"ped_{ped_id}"
    ped.pose.position.x = x
    ped.pose.position.y = y
    ped.pose.orientation.z = math.sin(yaw_rad / 2.0)
    ped.pose.orientation.w = math.cos(yaw_rad / 2.0)
    return ped


class _Scene:
    """A propagation node with a free map, one fleet robot placed by static TF and a driver node in a fresh namespace."""

    def __init__(self, event_loop, *, robot: str, robot_xy: tuple[float, float], array_spec: str) -> None:
        import rclpy
        import tf2_ros
        from arena_rclpy_mixins.qos import latched
        from nav_msgs.msg import OccupancyGrid
        from task_generator_msgs.msg import RobotFleet

        from arena_auditory.sound_propagation_node import SoundPropagationNode

        self.suffix = uuid.uuid4().hex[:8]
        self.namespace = f"/test_{self.suffix}"
        self.robot = robot
        self.base_frame = f"{robot}_{self.suffix}/base_link"
        self.nodes: list = []
        self.driver = rclpy.create_node(f"round_trip_driver_{self.suffix}", namespace=self.namespace)
        self.nodes.append(self.driver)
        self.propagation = _build(event_loop, SoundPropagationNode, self.namespace, {"propagation.backend": "legacy", "propagation.threshold_db": 10.0, "array.spec": array_spec})
        self.nodes.append(self.propagation)
        self._fleet_pub = self.driver.create_publisher(RobotFleet, f"{self.namespace}/state/robots", latched(1))
        self._map_pub = self.driver.create_publisher(OccupancyGrid, f"{self.namespace}/map", latched(1))
        self._tf = tf2_ros.StaticTransformBroadcaster(self.driver)
        self._robot_xy = robot_xy

    def add(self, node):
        self.nodes.append(node)
        return node

    def start(self) -> None:
        import rclpy
        import rclpy.time

        self._tf.sendTransform(_base_transform(self.base_frame, *self._robot_xy))
        self._map_pub.publish(_free_map())
        self._fleet_pub.publish(_robot_fleet(self.robot, f"{self.namespace}/{self.robot}", f"{self.robot}_{self.suffix}"))
        self.spin_until(
            lambda: self.propagation._tracker.occupancy is not None and bool(self.propagation._robots) and self.propagation._tf_buffer.can_transform("map", self.base_frame, rclpy.time.Time()),
        )

    def spin_until(self, predicate, timeout_sec: float = 10.0) -> None:
        import rclpy

        _spin_until(rclpy, self.nodes, predicate, timeout_sec)

    def destroy(self) -> None:
        for node in reversed(self.nodes):
            node.destroy_node()


def test_auditory_round_trip_greeting_reaches_robot_marker(rclpy_context, default_sounds, loop) -> None:
    from arena_auditory_msgs.msg import HeardSoundEvent
    from arena_people_msgs.msg import Pedestrians
    from arena_rclpy_mixins.qos import reliable
    from visualization_msgs.msg import Marker

    from arena_auditory.bus_node import BusNode
    from arena_auditory.human_emitter_node import HumanEmitterNode

    scene = _Scene(loop, robot="robot1", robot_xy=(1.0, 0.0), array_spec="stereo")
    try:
        emitter = scene.add(_build(loop, HumanEmitterNode, scene.namespace))
        bus = scene.add(_build(loop, BusNode, scene.namespace, {"bus.delay.enabled": False, "bus.min_snr_db": -15.0}))
        heard_by_robot: list[HeardSoundEvent] = []
        propagated: list[HeardSoundEvent] = []
        markers: list[Marker] = []
        driver = scene.driver
        ns = scene.namespace
        driver.create_subscription(HeardSoundEvent, f"{ns}/robot1/heard_sound", heard_by_robot.append, reliable(50))
        driver.create_subscription(HeardSoundEvent, f"{ns}/heard_sound_events", propagated.append, reliable(50))
        driver.create_subscription(Marker, f"{ns}/robot1/heard_sound_marker", markers.append, reliable(10))
        peds_pub = driver.create_publisher(Pedestrians, f"{ns}/arena_peds", 10)
        scene.start()
        scene.spin_until(
            lambda: "robot1" in bus._robots and bus._robots["robot1"].heard.get_subscription_count() > 0 and bus._robots["robot1"].markers.wanted and emitter._sound_publisher.get_subscription_count() > 0 and peds_pub.get_subscription_count() > 0,
        )

        pedestrians = Pedestrians()
        pedestrians.header.frame_id = "map"
        pedestrians.pedestrians.extend([_pedestrian(1, 0.0, 1.0, 0.0), _pedestrian(2, 1.0, 1.0, math.pi)])
        peds_pub.publish(pedestrians)
        mics = {"array:robot1:left", "array:robot1:right", "robot:robot1"}
        scene.spin_until(lambda: bool(heard_by_robot) and bool(markers) and mics.issubset({message.reception.listener_id for message in propagated}))

        heard = heard_by_robot[0]
        assert heard.source.id.startswith("human:1:")
        assert heard.source.kind == "speech"
        assert heard.reception.listener_id == "robot:robot1"
        assert heard.reception.audible is True

        receptions = {message.reception.listener_id: message.reception for message in propagated if message.reception.listener_id in mics}
        left, right, centroid = receptions["array:robot1:left"], receptions["array:robot1:right"], receptions["robot:robot1"]
        assert (left.listener_position.x, left.listener_position.y, left.listener_position.z) == pytest.approx((1.0, 0.1, 0.35))
        assert (right.listener_position.x, right.listener_position.y, right.listener_position.z) == pytest.approx((1.0, -0.1, 0.35))
        assert (centroid.listener_position.x, centroid.listener_position.y, centroid.listener_position.z) == pytest.approx((1.0, 0.0, 0.35))
        assert left.direct_delay_s < right.direct_delay_s

        marker = markers[0]
        assert marker.ns == "robot1_heard_sound"
        assert marker.header.frame_id == scene.base_frame
        assert "HEARD SPEECH" in marker.text
        assert "dB" in marker.text
        assert marker.type == Marker.TEXT_VIEW_FACING
        assert marker.action == Marker.ADD
        assert marker.pose.position.z == pytest.approx(1.2)
        assert marker.scale.z == pytest.approx(0.35)
        assert marker.lifetime.sec == 1
        assert marker.lifetime.nanosec == 500_000_000
    finally:
        scene.destroy()


def test_four_mic_array_renders_and_robot_hearing_hears_the_centroid(rclpy_context, default_sounds, loop) -> None:
    from arena_auditory_msgs.msg import HeardSoundEvent, SoundEvent
    from arena_rclpy_mixins.qos import reliable
    from arena_robots.audio import ArrayStream, load_array_spec
    from arena_simulation_setup.tree.assets.sound_catalog import AgentKind
    from std_msgs.msg import Float32MultiArray

    from arena_auditory.bus_node import BusNode
    from arena_auditory.renderer_node import RendererNode
    from arena_auditory.shared import SourceSpec

    scene = _Scene(loop, robot="jackal", robot_xy=(1.0, 0.0), array_spec="four_mic")
    try:
        renderer = scene.add(
            _build(
                loop,
                RendererNode,
                scene.namespace,
                {"array.spec": "four_mic", "output.enabled": False, "output.device": "none", "motor.enabled": False},
            )
        )
        bus = scene.add(_build(loop, BusNode, scene.namespace, {"bus.delay.enabled": False}))
        driver = scene.driver
        ns = scene.namespace
        levels: list[Float32MultiArray] = []
        propagated: list[HeardSoundEvent] = []
        robot_events: list[HeardSoundEvent] = []
        driver.create_subscription(Float32MultiArray, f"{ns}/jackal/audio/hearing/energy", levels.append, reliable(10))
        driver.create_subscription(HeardSoundEvent, f"{ns}/heard_sound_events", propagated.append, reliable(50))
        driver.create_subscription(HeardSoundEvent, f"{ns}/jackal/heard_sound", robot_events.append, reliable(50))
        sound_pub = driver.create_publisher(SoundEvent, f"{ns}/sound_events", reliable(50))
        scene.start()
        scene.spin_until(
            lambda: "jackal" in renderer._targets and "jackal" in bus._robots and bus._robots["jackal"].heard.get_subscription_count() > 0 and sound_pub.get_subscription_count() > 0 and renderer._targets["jackal"].publishers[ArrayStream.ENERGY].get_subscription_count() > 0,
        )

        event = SoundEvent()
        event.header.frame_id = "map"
        event.header.stamp = driver.get_clock().now().to_msg()
        event.source = SourceSpec(
            id="human:1:four-mic-regression",
            kind="speech",
            asset_id="greeting",
            variant_id="greeting_01",
            model="wav",
            agent_kind=AgentKind.PEDESTRIAN,
            agent_id=1,
            agent_name="ped_1",
            position=(1.5, 0.5, 1.6),
            level_db=94.0,
            duration_ns=1_000_000_000,
        ).to_msg()
        sound_pub.publish(event)
        mic_ids = {f"array:jackal:{name}" for name in ("front_left", "front_right", "rear_left", "rear_right")}
        scene.spin_until(
            lambda: bool(robot_events) and mic_ids.issubset({message.reception.listener_id for message in propagated}) and any(message.data and max(message.data) > 1e-3 for message in levels),
            timeout_sec=15.0,
        )

        receptions = {message.reception.listener_id: message.reception for message in propagated}
        spec = load_array_spec("four_mic")
        for mic in spec.mics:
            position = receptions[f"array:jackal:{mic.name}"].listener_position
            assert (position.x, position.y, position.z) == pytest.approx((1.0 + mic.position_m[0], mic.position_m[1], mic.position_m[2]))
        centroid = receptions["robot:jackal"]
        assert (centroid.listener_position.x, centroid.listener_position.y, centroid.listener_position.z) == pytest.approx((1.0, 0.0, 0.22))
        assert centroid.audible is True
        delays = {name: receptions[f"array:jackal:{name}"].direct_delay_s for name in spec.channel_names}
        assert min(delays, key=delays.get) == "front_left"
        assert robot_events[0].source.id == "human:1:four-mic-regression"
        assert robot_events[0].reception.listener_id == "robot:jackal"
    finally:
        scene.destroy()


def test_static_sound_keeps_its_authored_height_through_propagation(rclpy_context, loop) -> None:
    from arena_auditory_msgs.msg import ContinuousAudioSourceState, ContinuousHeardSoundState
    from arena_rclpy_mixins.qos import best_effort
    from arena_simulation_setup.tree.assets.sound_catalog import AgentKind

    from arena_auditory.shared import SourceSpec

    source_height = 2.6
    scene = _Scene(loop, robot="robot1", robot_xy=(1.0, 0.0), array_spec="stereo")
    try:
        driver = scene.driver
        ns = scene.namespace
        heard: list[ContinuousHeardSoundState] = []
        driver.create_subscription(ContinuousHeardSoundState, f"{ns}/continuous_heard_sounds", heard.append, best_effort(64))
        source_pub = driver.create_publisher(ContinuousAudioSourceState, f"{ns}/continuous_audio_sources", best_effort(64))
        scene.start()
        scene.spin_until(lambda: source_pub.get_subscription_count() > 0 and scene.propagation._continuous_pub.get_subscription_count() > 0)

        state = ContinuousAudioSourceState()
        state.header.frame_id = "map"
        state.source = SourceSpec(
            id="environment:hall_siren",
            kind="alarm",
            asset_id="alarm_loop",
            model="wav_loop",
            agent_kind=AgentKind.ENVIRONMENT,
            agent_name="hall_siren",
            position=(0.0, 2.0, source_height),
            level_db=88.0,
            loop=True,
        ).to_msg()

        def received() -> bool:
            state.header.stamp = driver.get_clock().now().to_msg()
            source_pub.publish(state)
            return any(message.reception.listener_id == "robot:robot1" for message in heard)

        scene.spin_until(received)
        robot = next(message for message in heard if message.reception.listener_id == "robot:robot1")
        assert robot.source.id == "environment:hall_siren"
        assert robot.source.position.z == pytest.approx(source_height)
        assert robot.source.active is True
        assert robot.header.frame_id == "map"
    finally:
        scene.destroy()
