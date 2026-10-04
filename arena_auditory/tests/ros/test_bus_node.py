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


def _heard(listener_id: str, *, event_id: str = "human:1:bus", audible: bool = True, received_db: float = 45.0, threshold_db: float = 20.0):
    from arena_auditory_msgs.msg import HeardSoundEvent

    from arena_auditory.shared import AgentKind, SourceSpec

    event = HeardSoundEvent()
    event.header.frame_id = "map"
    event.source = SourceSpec(
        id=event_id,
        kind="speech",
        asset_id="greeting",
        model="wav",
        agent_kind=AgentKind.PEDESTRIAN,
        agent_id=1,
        agent_name="ped_1",
        position=(0.0, 1.0, 1.6),
        level_db=60.0,
    ).to_msg()
    event.reception.listener_id = listener_id
    event.reception.listener_position.x = 1.0
    event.reception.listener_position.z = 0.22
    event.reception.distance_m = 1.4
    event.reception.bearing_rad = 2.356
    event.reception.received_level_db = received_db
    event.reception.threshold_db = threshold_db
    event.reception.audible = audible
    return event


@pytest.fixture
def bus(rclpy_context, loop):
    import rclpy
    from arena_auditory_msgs.msg import AuditoryDetection, HeardSoundEvent
    from arena_rclpy_mixins.qos import reliable
    from visualization_msgs.msg import Marker

    from arena_auditory.hearing.bus_node import BusNode

    suffix = uuid.uuid4().hex[:8]
    namespace = f"/test_{suffix}"
    node = _build(loop, BusNode, namespace, {"bus.delay.enabled": False})
    consumer = rclpy.create_node(f"bus_consumer_{suffix}", namespace=namespace)
    heard: list[HeardSoundEvent] = []
    found: list[AuditoryDetection] = []
    markers: list[Marker] = []
    consumer.create_subscription(HeardSoundEvent, f"{namespace}/jackal/heard_sound", heard.append, reliable(50))
    consumer.create_subscription(AuditoryDetection, f"{namespace}/jackal/hearing/bus/detections", found.append, reliable(50))
    consumer.create_subscription(Marker, f"{namespace}/jackal/heard_sound_marker", markers.append, reliable(10))
    node._cb_fleet(_robot_fleet("jackal", f"{namespace}/jackal", f"jackal_{suffix}"))
    outputs = node._robots["jackal"]
    _spin_until(
        rclpy,
        [node, consumer],
        lambda: outputs.heard.get_subscription_count() > 0 and outputs.detections.get_subscription_count() > 0 and outputs.markers.get_subscription_count() > 0,
    )
    try:
        yield node, consumer, suffix, heard, found, markers
    finally:
        consumer.destroy_node()
        node.destroy_node()


def test_robot_listener_reception_becomes_heard_sound_detection_and_marker(bus) -> None:
    import rclpy
    from visualization_msgs.msg import Marker

    node, consumer, suffix, heard, found, markers = bus
    node._cb_heard(_heard("robot:jackal", event_id="human:1:four-mic-regression"))
    _spin_until(rclpy, [node, consumer], lambda: bool(heard and found and markers))

    assert heard[0].source.id == "human:1:four-mic-regression"
    assert heard[0].reception.listener_id == "robot:jackal"
    assert heard[0].reception.audible is True

    detection = found[0]
    assert detection.robot == "jackal"
    assert detection.frontend == "bus"
    assert detection.kind == "speech"
    assert detection.event_id == "human:1:four-mic-regression"
    assert detection.listener_id == "robot:jackal"
    assert detection.azimuth_rad == pytest.approx(2.356, abs=1e-5)
    assert detection.level_db == pytest.approx(45.0)

    marker = markers[0]
    assert marker.ns == "jackal_heard_sound"
    assert marker.header.frame_id == f"jackal_{suffix}/base_link"
    assert marker.type == Marker.TEXT_VIEW_FACING
    assert marker.action == Marker.ADD
    assert "HEARD SPEECH" in marker.text
    assert "dB" in marker.text


def test_inaudible_weak_and_non_robot_receptions_are_dropped(bus) -> None:
    import rclpy

    node, consumer, _suffix, heard, found, markers = bus
    node._cb_heard(_heard("robot:jackal", event_id="inaudible", audible=False))
    node._cb_heard(_heard("robot:jackal", event_id="below_snr", received_db=10.0, threshold_db=20.0))
    node._cb_heard(_heard("array:jackal:front_left", event_id="array_mic"))
    node._cb_heard(_heard("robot:other", event_id="unknown_robot"))
    assert node._pending == []
    _settle(rclpy, [node, consumer])
    assert heard == []
    assert found == []
    assert markers == []
