"""SoundPropagationNode on the installed demo world: world switches, runtime microphones, backends, self path and room impulses."""

from __future__ import annotations

import asyncio
import math
import time
import uuid

import pytest

WORLD_LOAD_TIMEOUT_S = 120.0


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_auditory_msgs.msg")
    pytest.importorskip("arena_people_msgs.msg")
    pytest.importorskip("geometry_msgs.msg")
    pytest.importorskip("nav_msgs.msg")
    pytest.importorskip("task_generator_msgs.msg")


def _launch(parameters: dict[str, object] | None = None):
    """SoundPropagationNode below a unique namespace with parameters given the way a launch file passes them."""
    from arena_auditory.sound_propagation_node import SoundPropagationNode
    from arena_rclpy_mixins import ArenaMixinNode
    from rclpy.parameter import Parameter

    namespace = f"/test_{uuid.uuid4().hex[:8]}"
    overrides = [Parameter(name, value=value) for name, value in (parameters or {}).items()]

    class LaunchArguments(ArenaMixinNode):
        def __init__(self, node_name: str) -> None:
            super().__init__(node_name, namespace=namespace, parameter_overrides=overrides)

    class LaunchedPropagation(SoundPropagationNode, LaunchArguments):
        pass

    async def construct() -> LaunchedPropagation:
        return LaunchedPropagation()

    return asyncio.run(construct())


def _spin_until(rclpy, nodes, predicate, timeout_sec: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS round-trip")


def _robot_name() -> str:
    return f"r{uuid.uuid4().hex[:6]}"


def _load_map(propagation, *, size_m: float = 14.0, resolution: float = 0.1, origin: tuple[float, float] = (-2.0, -2.0)) -> None:
    from nav_msgs.msg import OccupancyGrid

    cells = round(size_m / resolution)
    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.resolution = resolution
    grid.info.width = cells
    grid.info.height = cells
    grid.info.origin.position.x = origin[0]
    grid.info.origin.position.y = origin[1]
    grid.info.origin.orientation.w = 1.0
    grid.data = [0] * (cells * cells)
    propagation._tracker._on_map(grid)


def _request_world(propagation, world: str) -> None:
    from std_msgs.msg import String

    propagation._tracker._on_world_msg(String(data=world))


def _await_world(rclpy, propagation, world: str):
    _spin_until(rclpy, [propagation], lambda: propagation._tracker.world is not None, timeout_sec=WORLD_LOAD_TIMEOUT_S)
    realized = propagation._tracker.world
    assert realized.name == world
    return realized


def _demo(rclpy, propagation):
    """Load the demo world on a free map and return the lower-left corner of its realized demo_zone."""
    _load_map(propagation)
    _request_world(propagation, "demo")
    world = _await_world(rclpy, propagation, "demo")
    min_x, min_y, _, _ = world.zone_named("demo_zone").polygon.bounds
    return world, (min_x, min_y)


def _fleet(*names: str):
    from task_generator_msgs.msg import RobotDescriptor, RobotFleet, RobotState

    fleet = RobotFleet()
    for name in names:
        fleet.robots.append(RobotState(descriptor=RobotDescriptor(name=name, model="jackal", ns=f"/{name}", frame=name)))
    return fleet


def _place(propagation, child_frame: str, x: float, y: float = 0.0, z: float = 0.0) -> None:
    from geometry_msgs.msg import TransformStamped

    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = child_frame
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.translation.z = z
    transform.transform.rotation.w = 1.0
    propagation._tf_buffer.set_transform_static(transform, "test")


def _pedestrians(*peds: tuple[int, float, float]):
    from arena_people_msgs.msg import Pedestrian, Pedestrians

    msg = Pedestrians()
    msg.header.frame_id = "map"
    for ped_id, x, y in peds:
        ped = Pedestrian(id=ped_id, name=f"ped_{ped_id}")
        ped.pose.position.x = x
        ped.pose.position.y = y
        ped.pose.orientation.w = 1.0
        msg.pedestrians.append(ped)
    return msg


def _event(
    source_id: str,
    *,
    frame: str = "map",
    kind: str = "speech",
    agent_kind: str = "pedestrian",
    agent_id: int = -1,
    agent_name: str = "",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    level_db: float = 60.0,
):
    from arena_auditory_msgs.msg import SoundEvent, SoundSource
    from geometry_msgs.msg import Point

    event = SoundEvent(
        source=SoundSource(
            id=source_id,
            kind=kind,
            asset_id=kind,
            model="wav",
            agent_kind=agent_kind,
            agent_id=agent_id,
            agent_name=agent_name,
            position=Point(x=position[0], y=position[1], z=position[2]),
            level_db=level_db,
            active=True,
        )
    )
    event.header.frame_id = frame
    return event


def _spawn(propagation, x: float, y: float, z: float, *, frame: str = "map", placement: str = "placed", attached_frame: str = ""):
    from arena_auditory_msgs.srv import SpawnMicrophone

    request = SpawnMicrophone.Request(placement=placement, attached_frame=attached_frame)
    request.position.header.frame_id = frame
    request.position.point.x = x
    request.position.point.y = y
    request.position.point.z = z
    return propagation._spawn_microphone(request, SpawnMicrophone.Response())


def _remove(propagation, listener_id: str):
    from arena_auditory_msgs.srv import RemoveMicrophone

    return propagation._remove_microphone(RemoveMicrophone.Request(listener_id=listener_id), RemoveMicrophone.Response())


def _select(propagation, listener_id: str) -> bool:
    from rclpy.parameter import Parameter

    return propagation.set_parameters([Parameter("listener.id", value=listener_id)])[0].successful


def _heard_collector(rclpy, propagation):
    """Peer node collecting heard_sound_events and acoustic/impulses of propagation, ready once both are matched."""
    from arena_auditory_msgs.msg import HeardSoundEvent, RoomImpulse
    from arena_rclpy_mixins.qos import latched, reliable

    namespace = propagation.get_namespace()
    peer = rclpy.create_node(f"propagation_peer_{uuid.uuid4().hex[:8]}", namespace=namespace)
    heard: list[HeardSoundEvent] = []
    impulses: list[RoomImpulse] = []
    peer.create_subscription(HeardSoundEvent, "heard_sound_events", heard.append, reliable(256))
    peer.create_subscription(RoomImpulse, "acoustic/impulses", impulses.append, latched(256))
    _spin_until(
        rclpy,
        [peer, propagation],
        lambda: propagation.count_subscribers(f"{namespace}/heard_sound_events") > 0 and propagation.count_subscribers(f"{namespace}/acoustic/impulses") > 0,
    )
    return peer, heard, impulses


def test_world_switch_clears_scene_and_loads_the_latest_request(rclpy_context):
    import rclpy
    from arena_auditory.world import AcousticWorld

    propagation = _launch()

    try:
        _load_map(propagation, origin=(-1.5, -2.5))
        assert propagation._tracker.world is None
        assert _spawn(propagation, 1.0, 1.0, 1.0).success is True

        _request_world(propagation, "map_empty")
        assert propagation._spawned_microphones == {}
        _request_world(propagation, "demo")
        assert propagation._tracker.world is None

        world = _await_world(rclpy, propagation, "demo")
        assert {zone.name for zone in world.scene.zones} == {"demo_zone"}
        assert world.authored_origin is not None
        assert world.offset == pytest.approx((-1.5 - world.authored_origin[0], -2.5 - world.authored_origin[1]))
        authored = AcousticWorld.load("demo", world.config).zone_named("demo_zone").polygon.bounds
        min_x, min_y, max_x, max_y = world.zone_named("demo_zone").polygon.bounds
        assert (min_x, min_y, max_x, max_y) == pytest.approx((authored[0] + world.offset[0], authored[1] + world.offset[1], authored[2] + world.offset[0], authored[3] + world.offset[1]))
        assert (max_x - min_x, max_y - min_y) == pytest.approx((10.0, 10.0))

        assert _spawn(propagation, min_x + 1.0, min_y + 1.0, 1.0).zone == "demo_zone"
        _request_world(propagation, "map_empty")
        assert propagation._tracker.world is None
        assert propagation._spawned_microphones == {}

        empty = _await_world(rclpy, propagation, "map_empty")
        assert {zone.name for zone in empty.scene.zones} == {"empty_zone"}
    finally:
        propagation.destroy_node()


def test_spawn_microphone_assigns_zone_and_next_index(rclpy_context):
    import attrs
    import rclpy
    from arena_auditory.world import WorldMicrophoneSpec
    from task_generator_msgs.msg import EpisodeRecord

    propagation = _launch({"propagation.backend": "legacy"})
    robot = _robot_name()

    try:
        without_map = _spawn(propagation, 4.0, 0.0, 1.5, frame="rviz_map")
        assert without_map.success is True
        assert without_map.zone == ""
        assert without_map.listener_id == "microphone:runtime:1"
        marker = propagation._marker_poses()[without_map.listener_id]
        assert marker.frame == "rviz_map"
        assert marker.position[0] == 4.0

        world, (min_x, min_y) = _demo(rclpy, propagation)
        assert propagation._spawned_microphones == {}

        first = _spawn(propagation, min_x + 2.0, min_y + 3.0, 1.5)
        assert first.success is True
        assert first.zone == "demo_zone"
        assert first.listener_id == "microphone:runtime:1"
        assert propagation._spawned_microphones[first.listener_id] == ((min_x + 2.0, min_y + 3.0, 1.5), "map")

        second = _spawn(propagation, min_x + 2.0, min_y + 3.0, 1.5)
        assert second.success is True
        assert second.listener_id == "microphone:runtime:2"

        _place(propagation, f"{robot}/base_link", min_x + 1.0, min_y + 1.0)
        attached = _spawn(propagation, min_x + 2.0, min_y + 3.0, 1.5, placement="body", attached_frame=f"{robot}/base_link")
        assert attached.success is True
        assert attached.listener_id == "microphone:runtime:3"
        assert attached.attached_frame == f"{robot}/base_link"
        local_position, attached_frame = propagation._spawned_microphones[attached.listener_id]
        assert attached_frame == f"{robot}/base_link"
        assert local_position == pytest.approx((1.0, 2.0, 1.5))
        assert propagation._microphone_position(attached.listener_id) == pytest.approx((min_x + 2.0, min_y + 3.0, 1.5))

        outside_zone = _spawn(propagation, min_x + 12.0, min_y + 3.0, 1.5)
        assert outside_zone.success is True
        assert outside_zone.zone == ""
        assert outside_zone.listener_id == "microphone:runtime:4"

        ceiling = world.room("demo_zone").ceiling_height_m
        above_ceiling = _spawn(propagation, min_x + 2.0, min_y + 3.0, ceiling + 0.5)
        assert above_ceiling.success is False
        assert above_ceiling.error_msg == f"microphone height exceeds zone 'demo_zone' ceiling at {ceiling:.2f} m"
        below_floor = _spawn(propagation, min_x + 2.0, min_y + 3.0, -0.5)
        assert below_floor.success is False
        assert below_floor.error_msg == "microphone height cannot be below the floor"
        assert _spawn(propagation, min_x + 2.0, min_y + 3.0, 1.5, placement="ceiling:1").error_msg == "placement must be non-empty and contain no ':'"
        assert _spawn(propagation, min_x + 2.0, min_y + 3.0, 1.5, frame="").error_msg == "microphone position requires a frame ID"

        fifth = _spawn(propagation, min_x + 4.0, min_y + 4.0, 1.5)
        assert fifth.listener_id == "microphone:runtime:5"
        assert len(propagation._spawned_microphones) == 5

        removed = _remove(propagation, outside_zone.listener_id)
        assert removed.success is True
        assert outside_zone.listener_id not in propagation._spawned_microphones

        authored_id = "microphone:zone:demo_zone:ceiling:1"
        authored = WorldMicrophoneSpec(listener_id=authored_id, zone="demo_zone", placement="ceiling", frame="map", position=(min_x + 1.0, min_y + 1.0, ceiling), ceiling_height_m=ceiling)
        propagation._on_world_ready(attrs.evolve(world, microphones=(authored,)))
        assert authored_id in propagation._microphone_ids()
        assert propagation._microphone_position(authored_id) == pytest.approx((min_x + 1.0, min_y + 1.0, ceiling))
        assert _remove(propagation, authored_id).error_msg == "world-authored microphones cannot be removed"
        assert _remove(propagation, f"array:{robot}:left").error_msg == "robot-attached microphones cannot be removed"
        assert _remove(propagation, "microphone:viewport:projective_center").error_msg == "viewport microphones cannot be removed"
        assert _remove(propagation, "microphone:runtime:99").error_msg == "unknown runtime microphone 'microphone:runtime:99'"

        propagation._tracker._on_episode_msg(EpisodeRecord(episode_id=7, world="demo"))
        assert propagation._spawned_microphones == {}
        assert propagation._tracker.world is not None
        assert _spawn(propagation, min_x + 2.0, min_y + 3.0, 1.5).listener_id == "microphone:runtime:1"
    finally:
        propagation.destroy_node()


def test_pedestrian_listener_uses_level3_under_pyroomacoustics_backend(rclpy_context):
    pytest.importorskip("pyroomacoustics")
    import rclpy

    propagation = _launch({"pedestrian_listeners.enabled": True, "pedestrian_listeners.discrete.enabled": True, "rir.max_order": 1, "rir.sample_rate_hz": 16000})
    robot = _robot_name()
    peer = None

    try:
        _, (min_x, min_y) = _demo(rclpy, propagation)
        _place(propagation, f"{robot}/base_link", min_x + 4.0, min_y + 2.0)
        propagation._on_fleet(_fleet(robot))
        propagation._on_peds(_pedestrians((1, min_x + 1.0, min_y + 2.0), (2, min_x + 4.0, min_y + 2.0)))
        peer, heard, _ = _heard_collector(rclpy, propagation)

        propagation._on_sound_event(_event("greeting_1", agent_id=1, agent_name="ped_1", position=(min_x + 1.0, min_y + 2.0, 1.6)))
        _spin_until(rclpy, [peer, propagation], lambda: {msg.reception.listener_id for msg in heard} >= {"agent:2", f"robot:{robot}"}, timeout_sec=10.0)
        receptions = {msg.reception.listener_id: msg.reception for msg in heard}

        assert receptions["agent:2"].backend == "level3"
        assert receptions["agent:2"].used_fallback is False
        assert receptions["agent:2"].rir_key == ""
        assert receptions[f"robot:{robot}"].backend == "pyroomacoustics_same_room"
        assert receptions[f"robot:{robot}"].used_fallback is False
        assert receptions[f"robot:{robot}"].source_zone == receptions[f"robot:{robot}"].listener_zone == "demo_zone"
        assert receptions[f"robot:{robot}"].rir_key != ""
    finally:
        if peer is not None:
            peer.destroy_node()
        propagation.destroy_node()


def test_launch_propagation_settings_reach_level3_model(rclpy_context):
    import rclpy
    from rclpy.parameter import Parameter

    propagation = _launch({"propagation.backend": "level3", "level3.max_reflections": 2, "level3.reflection_floor_db": -60.0})
    peer = None

    def paths_of_next_event(source_id: str) -> list[str]:
        count = len(heard)
        propagation._on_sound_event(_event(source_id, position=(min_x + 3.0, min_y + 4.0, 1.5)))
        _spin_until(rclpy, [peer, propagation], lambda: len(heard) > count)
        reception = heard[-1]
        assert reception.reception.backend == "level3"
        return [path.interaction_type for path in reception.early_paths]

    try:
        _, (min_x, min_y) = _demo(rclpy, propagation)
        spawned = _spawn(propagation, min_x + 6.0, min_y + 5.0, 1.5)
        assert _select(propagation, spawned.listener_id)
        peer, heard, _ = _heard_collector(rclpy, propagation)

        assert paths_of_next_event("speech_1") == ["direct", "reflection", "reflection"]

        assert propagation.set_parameters([Parameter("level3.reflection_floor_db", value=55.0)])[0].successful
        assert paths_of_next_event("speech_2") == ["direct"]

        assert propagation.set_parameters([Parameter("level3.reflection_floor_db", value=-60.0), Parameter("level3.max_reflections", value=8)])[0].successful
        assert paths_of_next_event("speech_3") == ["direct"] + ["reflection"] * 4

        assert propagation.set_parameters([Parameter("level3.max_reflections", value=0)])[0].successful
        assert paths_of_next_event("speech_4") == ["direct"]
    finally:
        if peer is not None:
            peer.destroy_node()
        propagation.destroy_node()


def test_robot_own_source_reaches_its_listeners_through_the_self_path(rclpy_context):
    pytest.importorskip("pyroomacoustics")
    import rclpy

    propagation = _launch({"rir.max_order": 1, "rir.sample_rate_hz": 16000})
    emitter, other = _robot_name(), _robot_name()
    peer = None

    try:
        _, (min_x, min_y) = _demo(rclpy, propagation)
        _place(propagation, f"{emitter}/base_link", min_x + 3.0, min_y + 3.0)
        _place(propagation, f"{other}/base_link", min_x + 7.0, min_y + 6.0)
        propagation._on_fleet(_fleet(emitter, other))
        peer, heard, impulses = _heard_collector(rclpy, propagation)

        propagation._on_sound_event(_event("motor_1", frame=f"{emitter}/base_link", kind="motor", agent_kind="robot", agent_name=emitter, position=(0.0, 0.0, 0.1)))
        expected = {f"robot:{emitter}", f"array:{emitter}:left", f"array:{emitter}:right", f"robot:{other}", f"array:{other}:left", f"array:{other}:right"}
        _spin_until(rclpy, [peer, propagation], lambda: {msg.reception.listener_id for msg in heard} >= expected, timeout_sec=10.0)
        receptions = {msg.reception.listener_id: msg.reception for msg in heard}

        own = {f"robot:{emitter}": 0.25, f"array:{emitter}:left": math.hypot(0.1, 0.25), f"array:{emitter}:right": math.hypot(0.1, 0.25)}
        for listener_id, distance in own.items():
            reception = receptions[listener_id]
            assert reception.backend == "self_direct_path"
            assert reception.rir_key == ""
            assert reception.occluded is False
            assert reception.distance_m == pytest.approx(distance, abs=1e-4)
            assert reception.received_level_db == pytest.approx(60.0 - 20.0 * math.log10(distance), abs=1e-3)

        other_reception = receptions[f"robot:{other}"]
        assert other_reception.backend == "pyroomacoustics_same_room"
        assert other_reception.rir_key != ""
        _spin_until(rclpy, [peer, propagation], lambda: {impulse.key for impulse in impulses} >= {receptions[f"robot:{other}"].rir_key})
        own_keys = {receptions[listener_id].rir_key for listener_id in own}
        assert own_keys == {""}
        assert "" not in {impulse.key for impulse in impulses}
    finally:
        if peer is not None:
            peer.destroy_node()
        propagation.destroy_node()


def test_room_impulse_is_republished_once_impulse_window_newer_keys_went_out(rclpy_context):
    pytest.importorskip("pyroomacoustics")
    import collections

    import rclpy
    from arena_auditory.propagation import IMPULSE_WINDOW

    propagation = _launch({"rir.max_order": 1, "rir.sample_rate_hz": 8000})
    peer = None

    try:
        _, (min_x, min_y) = _demo(rclpy, propagation)
        spawned = _spawn(propagation, min_x + 5.0, min_y + 8.5, 1.5)
        assert _select(propagation, spawned.listener_id)
        peer, heard, impulses = _heard_collector(rclpy, propagation)
        positions = [(min_x + 0.75 + 0.5 * (index % 17), min_y + 0.75 + 0.5 * (index // 17), 1.2) for index in range(IMPULSE_WINDOW + 1)]

        def emit(index: int, label: str) -> str:
            count = len(heard)
            propagation._on_sound_event(_event(f"speech_{label}", position=positions[index]))
            _spin_until(rclpy, [peer, propagation], lambda: len(heard) > count, timeout_sec=10.0)
            reception = heard[-1].reception
            assert reception.listener_id == spawned.listener_id
            assert reception.backend == "pyroomacoustics_same_room"
            return reception.rir_key

        first_key = emit(0, "first")
        keys = [first_key] + [emit(index, f"{index}") for index in range(1, IMPULSE_WINDOW)]
        assert len(set(keys)) == IMPULSE_WINDOW
        assert emit(0, "within_window") == first_key
        keys.append(emit(IMPULSE_WINDOW, "evicting"))
        assert emit(0, "after_window") == first_key

        _spin_until(rclpy, [peer, propagation], lambda: len(impulses) >= IMPULSE_WINDOW + 2, timeout_sec=10.0)
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            rclpy.spin_once(peer, timeout_sec=0.02)
        published = collections.Counter(impulse.key for impulse in impulses)
        assert published[first_key] == 2
        assert all(published[key] == 1 for key in keys[1:])
        assert sum(published.values()) == IMPULSE_WINDOW + 2
        assert {msg.reception.rir_key for msg in heard} == set(published)
        assert all(impulse.samples and impulse.sample_rate_hz == 8000 for impulse in impulses)
    finally:
        if peer is not None:
            peer.destroy_node()
        propagation.destroy_node()
