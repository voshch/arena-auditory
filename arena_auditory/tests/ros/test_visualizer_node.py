from __future__ import annotations

import asyncio
import time
import uuid

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_auditory_msgs.msg")
    pytest.importorskip("geometry_msgs.msg")
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


def _heard(listener_id: str, *, source_x: float = 0.0):
    from arena_auditory_msgs.msg import HeardSoundEvent
    from arena_simulation_setup.tree.assets.sound_catalog import AgentKind

    from arena_auditory.shared import SourceSpec

    event = HeardSoundEvent()
    event.header.frame_id = "map"
    event.source = SourceSpec(
        id="human:1:viz",
        kind="speech",
        asset_id="greeting",
        model="wav",
        agent_kind=AgentKind.PEDESTRIAN,
        agent_id=1,
        agent_name="ped_1",
        position=(source_x, 0.0, 0.0),
        level_db=60.0,
    ).to_msg()
    event.reception.listener_id = listener_id
    event.reception.listener_position.x = 1.0
    event.reception.received_level_db = 45.0
    event.reception.threshold_db = 20.0
    event.reception.audible = True
    return event


def test_propagation_visualizer_splits_pedestrian_and_robot_markers(rclpy_context, loop) -> None:
    import rclpy
    from arena_rclpy_mixins.qos import reliable
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray

    from arena_auditory.visualizer_node import SoundPropagationVisualizer

    suffix = uuid.uuid4().hex[:8]
    namespace = f"/test_{suffix}"
    visualizer = _build(loop, SoundPropagationVisualizer, namespace)
    consumer = rclpy.create_node(f"propagation_marker_consumer_{suffix}", namespace=namespace)
    pedestrian_markers: list[MarkerArray] = []
    robot_markers: list[MarkerArray] = []
    consumer.create_subscription(MarkerArray, f"{namespace}/pedestrian_sound_propagation_markers", pedestrian_markers.append, reliable(10))
    consumer.create_subscription(MarkerArray, f"{namespace}/robot_sound_propagation_markers", robot_markers.append, reliable(10))

    def named(message: MarkerArray, ns: str) -> list[Marker]:
        return [marker for marker in message.markers if marker.ns == ns]

    try:
        _spin_until(
            rclpy,
            [visualizer, consumer],
            lambda: visualizer._pedestrian_publisher.wanted and visualizer._robot_publisher.wanted,
        )
        not_ready = _heard("agent:3")
        not_ready.reception.used_fallback = True
        not_ready.reception.fallback_reason = "acoustic_scene_not_loaded"
        visualizer._on_heard(not_ready)
        for _ in range(3):
            rclpy.spin_once(consumer, timeout_sec=0.02)
        assert pedestrian_markers == []
        assert robot_markers == []

        visualizer._on_heard(_heard("agent:3"))
        _spin_until(rclpy, [visualizer, consumer], lambda: bool(pedestrian_markers))
        assert robot_markers == []
        pedestrian_color = pedestrian_markers[-1].markers[0].color
        assert pedestrian_color.b == pytest.approx(1.0)
        assert pedestrian_color.r == pytest.approx(0.10)

        robot_event = _heard("robot:jackal")
        robot_event.reception.backend = "pyroomacoustics_multi_portal"
        robot_event.reception.portal_ids = ["door:a", "opening:b"]
        robot_event.reception.traversed_zones = ["room_a", "hall", "room_b"]
        robot_event.reception.route_loss_db = 3.5
        robot_event.reception.portal_positions = [Point(x=0.25, y=0.0, z=1.0), Point(x=0.75, y=0.0, z=1.0)]
        visualizer._on_heard(robot_event)
        _spin_until(rclpy, [visualizer, consumer], lambda: bool(robot_markers))
        robot_color = robot_markers[-1].markers[0].color
        assert robot_color.r == pytest.approx(0.65)
        assert robot_color.b == pytest.approx(1.0)
        (path,) = named(robot_markers[-1], "robot_sound_propagation_path")
        assert len(path.points) == 4
        assert len(named(robot_markers[-1], "robot_acoustic_portal")) == 2
        (text,) = named(robot_markers[-1], "robot_sound_propagation_backend")
        assert "2 portal(s)" in text.text
        assert "3.5 dB" in text.text

        visualizer._on_heard(_heard("robot:jackal", source_x=2.0))
        _spin_until(rclpy, [visualizer, consumer], lambda: len(robot_markers) >= 2)
        (latest_path,) = named(robot_markers[-1], "robot_sound_propagation_path")
        assert latest_path.id == path.id
        assert latest_path.points[0].x == pytest.approx(2.0)
        assert len(latest_path.points) == 2
        stale = [marker for marker in named(robot_markers[-1], "robot_acoustic_portal") if marker.action == Marker.DELETE]
        assert len(stale) == 2

        marker_count = len(robot_markers)
        array_event = _heard("array:jackal:front_left")
        array_event.reception.listener_position.z = 0.22
        visualizer._on_heard(array_event)
        _spin_until(rclpy, [visualizer, consumer], lambda: len(robot_markers) > marker_count)
        (listener,) = named(robot_markers[-1], "array_jackal_front_left_sound_listener")
        assert listener.pose.position.z == pytest.approx(0.22)
    finally:
        consumer.destroy_node()
        visualizer.destroy_node()


def test_environment_source_marker_sits_at_the_authored_height(rclpy_context, loop) -> None:
    import rclpy
    from arena_auditory_msgs.msg import ContinuousAudioSourceState
    from arena_rclpy_mixins.qos import latched
    from arena_simulation_setup.tree.assets.sound_catalog import AgentKind
    from visualization_msgs.msg import MarkerArray

    from arena_auditory.shared import SourceSpec
    from arena_auditory.visualizer_node import SoundPropagationVisualizer

    suffix = uuid.uuid4().hex[:8]
    namespace = f"/test_{suffix}"
    visualizer = _build(loop, SoundPropagationVisualizer, namespace)
    consumer = rclpy.create_node(f"environment_marker_consumer_{suffix}", namespace=namespace)
    markers: list[MarkerArray] = []
    consumer.create_subscription(MarkerArray, f"{namespace}/environment_audio_source_markers", markers.append, latched(32))
    state = ContinuousAudioSourceState()
    state.header.frame_id = "map"
    state.source = SourceSpec(
        id="environment:hall_siren",
        kind="alarm",
        asset_id="alarm_loop",
        model="wav_loop",
        agent_kind=AgentKind.ENVIRONMENT,
        agent_name="hall_siren",
        position=(4.0, 2.3, 2.6),
        level_db=88.0,
        loop=True,
    ).to_msg()
    try:
        _spin_until(rclpy, [visualizer, consumer], lambda: visualizer._environment_source_publisher.get_subscription_count() > 0)
        visualizer._on_continuous_source(state)
        _spin_until(rclpy, [visualizer, consumer], lambda: bool(markers))
        body = next(marker for marker in markers[-1].markers if marker.ns == "environment_audio_alarm_body")
        assert body.header.frame_id == "map"
        assert (body.pose.position.x, body.pose.position.y, body.pose.position.z) == pytest.approx((4.0, 2.3, 2.6))
        label = next(marker for marker in markers[-1].markers if marker.ns == "environment_audio_source_labels")
        assert label.pose.position.z == pytest.approx(2.6 + 0.32)
        assert "[ACTIVE]" in label.text
    finally:
        consumer.destroy_node()
        visualizer.destroy_node()
