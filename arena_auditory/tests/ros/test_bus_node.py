from __future__ import annotations

import time
import uuid

import pytest

_spin_until = pytest.importorskip("arena_rclpy_mixins.testing").spin_until


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_auditory_msgs.msg")
    pytest.importorskip("arena_runtime_msgs.msg")
    pytest.importorskip("visualization_msgs.msg")


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


def _settle(rclpy, nodes, duration_sec: float = 0.3) -> None:
    end = time.monotonic() + duration_sec
    while time.monotonic() < end:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)


def _robot_fleet(robot_name: str, namespace: str, frame: str):
    from arena_runtime_msgs.msg import RobotDescriptor, RobotFleet, RobotState

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
    from arena_simulation_setup.tree.assets.sound_catalog import AgentKind

    from arena_auditory.shared import SourceSpec

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


def _continuous(listener_id: str, *, source_id: str = "caller:1", audible: bool = True, active: bool = True, received_db: float = 45.0, threshold_db: float = 20.0, agent_robot: str = ""):
    from arena_auditory_msgs.msg import ContinuousHeardSoundState
    from arena_simulation_setup.tree.assets.sound_catalog import AgentKind

    event = _heard(listener_id, event_id=source_id, audible=audible, received_db=received_db, threshold_db=threshold_db)
    if agent_robot:
        event.source.agent_kind = AgentKind.ROBOT
        event.source.agent_name = agent_robot
    event.source.active = active
    return ContinuousHeardSoundState(header=event.header, source=event.source, reception=event.reception)


@pytest.fixture
def bus(rclpy_context, loop):
    import rclpy
    from arena_auditory_msgs.msg import HeardSoundEvent
    from arena_rclpy_mixins.qos import reliable
    from arena_robots_msgs.msg import SoundDetection
    from visualization_msgs.msg import Marker

    from arena_auditory.bus_node import BusNode

    suffix = uuid.uuid4().hex[:8]
    namespace = f"/test_{suffix}"
    node = _build(loop, BusNode, namespace, {"bus.delay.enabled": False, "bus.continuous.gap_s": 0.2})
    consumer = rclpy.create_node(f"bus_consumer_{suffix}", namespace=namespace)
    heard: list[HeardSoundEvent] = []
    found: list[SoundDetection] = []
    markers: list[Marker] = []
    consumer.create_subscription(HeardSoundEvent, f"{namespace}/jackal/heard_sound", heard.append, reliable(50))
    consumer.create_subscription(SoundDetection, f"{namespace}/jackal/hearing/bus/detections", found.append, reliable(50))
    consumer.create_subscription(Marker, f"{namespace}/jackal/heard_sound_marker", markers.append, reliable(10))
    node._cb_fleet(_robot_fleet("jackal", f"{namespace}/jackal", f"jackal_{suffix}"))
    outputs = node._robots["jackal"]
    _spin_until(
        rclpy,
        [node, consumer],
        lambda: outputs.heard.get_subscription_count() > 0 and outputs.detections.get_subscription_count() > 0 and outputs.markers.wanted,
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


def test_continuous_source_gives_one_detection_while_it_stays_audible(bus) -> None:
    import rclpy

    node, consumer, _suffix, heard, found, markers = bus
    for _ in range(6):
        node._cb_continuous(_continuous("robot:jackal"))
        _settle(rclpy, [node, consumer], 0.03)
    _spin_until(rclpy, [node, consumer], lambda: bool(heard and found and markers))
    _settle(rclpy, [node, consumer])
    assert len(heard) == 1
    assert len(found) == 1
    assert found[0].event_id == "caller:1"
    assert found[0].robot == "jackal"
    assert found[0].level_db == pytest.approx(45.0)
    assert heard[0].source.id == "caller:1"


def test_continuous_source_gives_a_second_detection_after_an_inaudible_state(bus) -> None:
    import rclpy

    node, consumer, _suffix, _heard_events, found, _markers = bus
    node._cb_continuous(_continuous("robot:jackal"))
    node._cb_continuous(_continuous("robot:jackal", audible=False, active=False))
    node._cb_continuous(_continuous("robot:jackal"))
    _spin_until(rclpy, [node, consumer], lambda: len(found) >= 2)
    _settle(rclpy, [node, consumer])
    assert len(found) == 2


def test_continuous_source_gives_a_second_detection_after_the_stream_goes_silent(bus) -> None:
    import rclpy

    node, consumer, _suffix, _heard_events, found, _markers = bus
    node._cb_continuous(_continuous("robot:jackal"))
    _settle(rclpy, [node, consumer], 0.4)
    node._cb_continuous(_continuous("robot:jackal"))
    _spin_until(rclpy, [node, consumer], lambda: len(found) >= 2)
    _settle(rclpy, [node, consumer])
    assert len(found) == 2


def test_continuous_sources_have_independent_edges_per_source_id(bus) -> None:
    import rclpy

    node, consumer, _suffix, _heard_events, found, _markers = bus
    node._cb_continuous(_continuous("robot:jackal", source_id="caller:1"))
    node._cb_continuous(_continuous("robot:jackal", source_id="caller:2"))
    _spin_until(rclpy, [node, consumer], lambda: len(found) >= 2)
    assert sorted(detection.event_id for detection in found) == ["caller:1", "caller:2"]


def test_fleet_change_restarts_continuous_edges(bus) -> None:
    import rclpy

    node, consumer, suffix, _heard_events, found, _markers = bus
    node._cb_continuous(_continuous("robot:jackal"))
    node._cb_fleet(_robot_fleet("other", f"/test_{suffix}/other", f"other_{suffix}"))
    node._cb_continuous(_continuous("robot:jackal"))
    _spin_until(rclpy, [node, consumer], lambda: len(found) >= 2)
    assert len(found) == 2


def test_an_audible_environment_source_published_inactive_is_heard(bus) -> None:
    node, *_ = bus
    node._cb_continuous(_continuous("robot:jackal", source_id="environment:env_0/hall_siren/0", active=False))
    assert len(node._pending) == 1


def test_inaudible_weak_self_and_unknown_robot_continuous_states_are_dropped(bus) -> None:
    import rclpy

    node, consumer, _suffix, heard, found, markers = bus
    node._cb_continuous(_continuous("robot:jackal", source_id="inaudible", audible=False))
    node._cb_continuous(_continuous("robot:jackal", source_id="below_snr", received_db=10.0, threshold_db=20.0))
    node._cb_continuous(_continuous("robot:jackal", source_id="stopped", audible=False, active=False))
    node._cb_continuous(_continuous("robot:jackal", source_id="own_motor", agent_robot="jackal"))
    node._cb_continuous(_continuous("array:jackal:front_left", source_id="array_mic"))
    node._cb_continuous(_continuous("robot:other", source_id="unknown_robot"))
    assert node._pending == []
    _settle(rclpy, [node, consumer])
    assert heard == []
    assert found == []
    assert markers == []
