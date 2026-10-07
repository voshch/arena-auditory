from __future__ import annotations

import asyncio
import time
import uuid

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_auditory_msgs.msg")
    pytest.importorskip("task_generator_msgs.msg")
    pytest.importorskip("nav_msgs.msg")
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


def _spin_until(rclpy, nodes, predicate, timeout_sec: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS round-trip")


def _settle(rclpy, nodes, duration_sec: float = 0.3) -> None:
    end = time.monotonic() + duration_sec
    while time.monotonic() < end:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)


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


def _odom(linear_mps: float = 0.0, angular_rps: float = 0.0):
    from nav_msgs.msg import Odometry

    odom = Odometry()
    odom.header.stamp.sec = 1
    odom.pose.pose.orientation.w = 1.0
    odom.twist.twist.linear.x = linear_mps
    odom.twist.twist.angular.z = angular_rps
    return odom


@pytest.fixture
def robot_emitter(rclpy_context, loop, request):
    import rclpy
    from arena_auditory_msgs.msg import ContinuousAudioSourceState
    from arena_rclpy_mixins.qos import best_effort

    from arena_auditory.robot_emitter_node import RobotEmitterNode

    suffix = uuid.uuid4().hex[:8]
    namespace = f"/test_{suffix}"
    params: dict[str, object] = {"drivetrain.period_s": 10.0, "motor.model": request.param}
    motor = _build(loop, RobotEmitterNode, namespace, params)
    consumer = rclpy.create_node(f"robot_emitter_consumer_{suffix}", namespace=namespace)
    states: list[ContinuousAudioSourceState] = []
    consumer.create_subscription(ContinuousAudioSourceState, f"{namespace}/continuous_audio_sources", states.append, best_effort(64))
    motor._on_fleet(_robot_fleet("robot1", f"{namespace}/robot1", f"robot1_{suffix}"))
    _spin_until(rclpy, [motor, consumer], lambda: motor._source_publisher.get_subscription_count() > 0)

    def tick(linear_mps: float) -> None:
        motor._on_odom("robot1", _odom(linear_mps))
        motor._publish()

    try:
        yield motor, consumer, namespace, suffix, tick, states
    finally:
        consumer.destroy_node()
        motor.destroy_node()


def test_robot_agent_id_fits_int32_and_stays_negative() -> None:
    from arena_auditory_msgs.msg import SoundSource

    from arena_auditory.robot_emitter_node import robot_agent_id

    for name in ("robot_0", "robot1", "jackal"):
        agent_id = robot_agent_id(name)
        assert -(2**31) <= agent_id < 0
        assert robot_agent_id(name) == agent_id
        SoundSource().agent_id = agent_id


@pytest.mark.parametrize("robot_emitter", ["procedural"], indirect=True)
@pytest.mark.usefixtures("default_sounds")
def test_procedural_motor_repeats_its_inactive_state_until_active_again(robot_emitter) -> None:
    import rclpy

    from arena_auditory.shared import INACTIVE_REPEATS

    motor, consumer, _namespace, suffix, tick, states = robot_emitter
    tick(0.5)
    tick(0.5)
    for _ in range(INACTIVE_REPEATS + 3):
        tick(0.0)
    _settle(rclpy, [motor, consumer])
    assert [state.source.active for state in states] == [True, True] + [False] * INACTIVE_REPEATS
    first = states[0]
    assert first.header.frame_id == f"robot1_{suffix}/base_link"
    assert first.source.kind == "motor"
    assert first.source.model == "drivetrain"
    assert first.source.variant_id == "jackal_drivetrain"
    assert first.source.agent_kind == "robot"
    assert first.source.agent_name == "robot1"
    assert first.source.position.z == pytest.approx(0.25)
    assert len(first.source.state_names) == len(first.source.state_values) > 0
    assert len({state.source.program_start.sec * 1_000_000_000 + state.source.program_start.nanosec for state in states}) == 1

    states.clear()
    tick(0.5)
    tick(0.0)
    tick(0.0)
    _settle(rclpy, [motor, consumer])
    assert [state.source.active for state in states] == [True, False, False]


@pytest.mark.parametrize("robot_emitter", ["wav"], indirect=True)
@pytest.mark.usefixtures("default_sounds")
def test_wav_motor_publishes_the_motor_loop(robot_emitter) -> None:
    import rclpy

    motor, consumer, _namespace, _suffix, tick, states = robot_emitter
    tick(0.5)
    tick(0.0)
    _settle(rclpy, [motor, consumer])
    assert [state.source.active for state in states][:2] == [True, False]
    assert {state.source.model for state in states} == {"wav_loop"}
    assert {state.source.variant_id for state in states} == {"motor_loop_01"}
    assert all(state.source.loop for state in states)


@pytest.mark.parametrize("robot_emitter", ["procedural"], indirect=True)
def test_motor_sound_publishes_cone_and_clears_it(robot_emitter) -> None:
    import rclpy
    from arena_rclpy_mixins.qos import reliable
    from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary
    from rclpy.parameter import Parameter
    from visualization_msgs.msg import Marker, MarkerArray

    motor, consumer, namespace, suffix, _tick, _states = robot_emitter
    base_frame = f"robot1_{suffix}/base_link"
    marker_topic = f"{namespace}/robot1/motor_sound_markers"
    received: list[MarkerArray] = []
    consumer.create_subscription(MarkerArray, marker_topic, received.append, reliable(10))
    robot = motor._robots["robot1"]
    assert robot.marker_publisher.publisher.topic_name == marker_topic
    assert robot.binding.base_frame == base_frame
    assert f"{namespace}/robot1/odom" in robot.binding.odom_topics
    _spin_until(rclpy, [motor, consumer], lambda: robot.marker_publisher.wanted)

    def actions(action: int) -> list[Marker]:
        return [marker for message in received for marker in message.markers if marker.action == action]

    motor._on_odom("robot1", _odom(0.2))
    motor._publish()
    _spin_until(rclpy, [motor, consumer], lambda: bool(actions(Marker.ADD)))
    added = actions(Marker.ADD)
    assert len(added) == 2
    assert {marker.type for marker in added} == {Marker.TRIANGLE_LIST, Marker.LINE_STRIP}
    assert all(marker.header.frame_id == base_frame for marker in added)
    assert all(len(marker.points) > 12 for marker in added)
    assert max(abs(point.x) for marker in added for point in marker.points) <= 1.25
    color = SoundLibrary.default().kind("motor").color
    assert all((marker.color.r, marker.color.g, marker.color.b) == pytest.approx(color) for marker in added)

    received.clear()
    motor._on_odom("robot1", _odom(0.0))
    motor._publish()
    _spin_until(rclpy, [motor, consumer], lambda: bool(actions(Marker.DELETE)))
    assert len(actions(Marker.DELETE)) == 2

    received.clear()
    motor._on_odom("robot1", _odom(0.0, 1.0))
    assert robot.speed_mps == pytest.approx(0.25)
    motor._publish()
    _spin_until(rclpy, [motor, consumer], lambda: bool(actions(Marker.ADD)))

    received.clear()
    motor.set_parameters([Parameter("motor.enabled", Parameter.Type.BOOL, False)])
    motor._publish()
    _spin_until(rclpy, [motor, consumer], lambda: bool(actions(Marker.DELETE)))
    assert robot.moving is False
