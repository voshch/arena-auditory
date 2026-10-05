from __future__ import annotations

import asyncio
import math
import time
import uuid
from pathlib import Path

import pytest

WORLD = "three_storied_residential"
TILE_ZONE = "1_bathroom"


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_auditory_msgs.msg")
    pytest.importorskip("arena_people_msgs.msg")
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


def _wav_seconds(asset: str, name: str) -> float:
    from ament_index_python.packages import get_package_share_directory
    from scipy.io import wavfile

    rate, data = wavfile.read(Path(get_package_share_directory("arena_auditory")) / "sounds" / "Common" / "Sound" / asset / name)
    return data.shape[0] / rate


def _seconds(duration) -> float:
    return duration.sec + duration.nanosec * 1e-9


def _pedestrian(ped_id: int, x: float, y: float, *, yaw_rad: float = 0.0, speed_mps: float = 0.0):
    from arena_people_msgs.msg import Pedestrian

    ped = Pedestrian()
    ped.id = ped_id
    ped.name = f"ped_{ped_id}"
    ped.pose.position.x = x
    ped.pose.position.y = y
    ped.pose.orientation.z = math.sin(yaw_rad / 2.0)
    ped.pose.orientation.w = math.cos(yaw_rad / 2.0)
    ped.twist.linear.x = speed_mps
    return ped


def _pedestrians(*peds):
    from arena_people_msgs.msg import Pedestrians

    msg = Pedestrians()
    msg.header.frame_id = "map"
    msg.pedestrians.extend(peds)
    return msg


@pytest.fixture
def human_emitter(rclpy_context, loop):
    import rclpy
    from arena_auditory_msgs.msg import SoundEvent
    from arena_rclpy_mixins.qos import reliable

    from arena_auditory.human_emitter_node import HumanEmitterNode

    namespace = f"/test_{uuid.uuid4().hex[:8]}"
    producer = _build(loop, HumanEmitterNode, namespace, {"world.coverage.enabled": False})
    consumer = rclpy.create_node(f"human_emitter_consumer_{uuid.uuid4().hex[:8]}", namespace=namespace)
    received: list[SoundEvent] = []
    consumer.create_subscription(SoundEvent, f"{namespace}/sound_events", received.append, reliable(50))
    _spin_until(rclpy, [producer, consumer], lambda: producer._sound_publisher.get_subscription_count() > 0)

    def emit(*peds) -> SoundEvent:
        count = len(received)
        producer._on_pedestrians(_pedestrians(*peds))
        _spin_until(rclpy, [producer, consumer], lambda: len(received) > count)
        return received[-1]

    try:
        yield producer, consumer, namespace, emit
    finally:
        consumer.destroy_node()
        producer.destroy_node()


def test_pedestrian_state_takes_yaw_from_orientation_and_speed_from_twist() -> None:
    from arena_auditory.human_emitter_node import pedestrian_state

    ped = _pedestrian(4, 1.0, -2.0, yaw_rad=math.pi / 2.0, speed_mps=0.3)
    ped.twist.linear.y = 0.4
    ped.gait_phase = 2.5
    state = pedestrian_state(ped)
    assert state.id == 4
    assert state.name == "ped_4"
    assert state.position == pytest.approx((1.0, -2.0, 0.0))
    assert state.yaw_rad == pytest.approx(math.pi / 2.0)
    assert state.speed_mps == pytest.approx(0.5)
    assert state.gait_phase == pytest.approx(2.5)


def test_footstep_on_a_ceramic_tile_zone_is_tagged_ceramic_tile(human_emitter) -> None:
    import rclpy
    from arena_rclpy_mixins.qos import latched
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import String

    from arena_auditory.assets import SoundLibrary
    from arena_auditory.materials import default_catalog
    from arena_auditory.params import PortalGroup, WorldGroup, configure
    from arena_auditory.world import AcousticWorld, WorldConfig

    producer, consumer, namespace, emit = human_emitter
    authored = AcousticWorld.load(WORLD, configure(WorldConfig, WorldGroup, PortalGroup))
    zone = authored.zone_named(TILE_ZONE)
    assert zone is not None
    assert "ceramic" in zone.floor_material_id.lower()
    x, y = zone.polygon.representative_point().coords[0]

    world_pub = consumer.create_publisher(String, f"{namespace}/state/world", latched(1))
    map_pub = consumer.create_publisher(OccupancyGrid, f"{namespace}/map", latched(1))
    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.resolution = 1.0
    grid.info.width = 1
    grid.info.height = 1
    grid.info.origin.position.x, grid.info.origin.position.y = authored.authored_origin
    grid.info.origin.orientation.w = 1.0
    grid.data = [0]
    world_pub.publish(String(data=WORLD))
    map_pub.publish(grid)
    _spin_until(rclpy, [producer, consumer], lambda: producer._tracker.world is not None, timeout_sec=20.0)

    event = emit(_pedestrian(3, x, y, speed_mps=0.4))

    asset = SoundLibrary.default().asset("footstep")
    assert event.source.kind == "footstep"
    assert event.source.variant_id == "footstep_ceramic_tile_01"
    assert list(event.source.tags) == ["walk", "ceramic_tile"]
    assert _seconds(event.source.duration) == pytest.approx(_wav_seconds("footstep", "footstep_ceramic_tile.wav"), abs=1e-3)
    assert event.source.level_db == pytest.approx(asset.level_db - default_catalog().surface_damping_db(zone.floor_material_id, "floor"), abs=1e-4)
    assert (event.source.position.x, event.source.position.y, event.source.position.z) == pytest.approx((x, y, 0.05), abs=1e-6)


def test_speech_duration_is_the_greeting_wav_length(human_emitter) -> None:
    _producer, _consumer, _namespace, emit = human_emitter
    event = emit(_pedestrian(4, 0.0, 0.0), _pedestrian(5, 1.0, 0.0, yaw_rad=math.pi))
    assert event.source.kind == "speech"
    assert event.source.asset_id == "greeting"
    assert event.source.agent_kind == "pedestrian"
    assert event.source.agent_id == 4
    assert event.source.position.z == pytest.approx(1.6)
    assert _seconds(event.source.duration) == pytest.approx(_wav_seconds("greeting", "greeting.wav"), abs=1e-3)


def test_cones_are_not_built_without_a_marker_subscriber(human_emitter) -> None:
    producer, _consumer, _namespace, emit = human_emitter
    emit(_pedestrian(5, 0.0, 0.0, speed_mps=0.4))
    assert next(producer._marker_counter) == 0


def test_walking_pedestrian_publishes_default_footstep_and_cone(human_emitter) -> None:
    import rclpy
    from arena_rclpy_mixins.qos import reliable
    from visualization_msgs.msg import Marker, MarkerArray

    producer, consumer, namespace, emit = human_emitter
    markers: list[MarkerArray] = []
    consumer.create_subscription(MarkerArray, f"{namespace}/pedestrian_markers/extra", markers.append, reliable(10))
    _spin_until(rclpy, [producer, consumer], lambda: producer._marker_publisher.wanted)

    event = emit(_pedestrian(7, 1.0, 2.0, speed_mps=0.4))
    _spin_until(rclpy, [producer, consumer], lambda: bool(markers))

    assert event.source.kind == "footstep"
    assert list(event.source.tags) == ["walk", "default"]
    assert {marker.type for marker in markers[-1].markers} == {Marker.TRIANGLE_LIST, Marker.LINE_STRIP}
    assert all(marker.header.frame_id == "map" for marker in markers[-1].markers)
