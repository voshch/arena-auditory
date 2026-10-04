"""SoundPropagationNode without an acoustic world: events, listeners, robot arrays, microphones and continuous sources."""

from __future__ import annotations

import asyncio
import json
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


def _spin_until(rclpy, nodes, predicate, timeout_sec: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS round-trip")


def _robot_name() -> str:
    return f"r{uuid.uuid4().hex[:6]}"


def _load_map(propagation, *, width: int = 20, height: int = 20, resolution: float = 1.0, origin: tuple[float, float] = (0.0, 0.0)) -> None:
    from nav_msgs.msg import OccupancyGrid

    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.resolution = resolution
    grid.info.width = width
    grid.info.height = height
    grid.info.origin.position.x = origin[0]
    grid.info.origin.position.y = origin[1]
    grid.info.origin.orientation.w = 1.0
    grid.data = [0] * (width * height)
    propagation._tracker._on_map(grid)


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


def _source(
    source_id: str,
    *,
    kind: str = "speech",
    agent_kind: str = "pedestrian",
    agent_id: int = -1,
    agent_name: str = "",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    level_db: float = 60.0,
    reference_distance_m: float = 0.0,
    tags: tuple[str, ...] = (),
):
    from arena_auditory_msgs.msg import SoundSource
    from geometry_msgs.msg import Point

    return SoundSource(
        id=source_id,
        kind=kind,
        asset_id=kind,
        model="wav",
        agent_kind=agent_kind,
        agent_id=agent_id,
        agent_name=agent_name,
        tags=list(tags),
        position=Point(x=position[0], y=position[1], z=position[2]),
        level_db=level_db,
        reference_distance_m=reference_distance_m,
        active=True,
    )


def _continuous(source_id: str, *, frame: str = "map", position: tuple[float, float, float] = (0.0, 1.0, 1.0), level_db: float = 70.0, reference_distance_m: float = 0.0):
    from arena_auditory_msgs.msg import ContinuousAudioSourceState

    state = ContinuousAudioSourceState()
    state.header.frame_id = frame
    state.source = _source(source_id, kind="music", agent_kind="environment", position=position, level_db=level_db, reference_distance_m=reference_distance_m)
    state.source.model = "wav_loop"
    state.source.loop = True
    return state


def _spec(source):
    from arena_auditory.shared import SourceSpec

    return SourceSpec.from_msg(source)


def _spawn(propagation, x: float, y: float, z: float, *, frame: str = "map", placement: str = "placed", attached_frame: str = ""):
    from arena_auditory_msgs.srv import SpawnMicrophone

    request = SpawnMicrophone.Request(placement=placement, attached_frame=attached_frame)
    request.position.header.frame_id = frame
    request.position.point.x = x
    request.position.point.y = y
    request.position.point.z = z
    return propagation._spawn_microphone(request, SpawnMicrophone.Response())


def _select(propagation, listener_id: str) -> bool:
    from rclpy.parameter import Parameter

    return propagation.set_parameters([Parameter("listener.id", value=listener_id)])[0].successful


def _robot_set(robot: str, mics: tuple[str, ...] = ("left", "right")) -> set[str]:
    return {f"robot:{robot}"} | {f"array:{robot}:{mic}" for mic in mics}


def test_sound_event_round_trips_to_heard_sound_event(rclpy_context):
    import rclpy
    from arena_auditory_msgs.msg import HeardSoundEvent, SoundEvent
    from arena_rclpy_mixins.qos import reliable
    from rclpy.qos import DurabilityPolicy

    propagation = _launch({"propagation.backend": "legacy", "pedestrian_listeners.enabled": True, "pedestrian_listeners.discrete.enabled": True})
    namespace = propagation.get_namespace()
    peer = rclpy.create_node(f"sound_event_peer_{uuid.uuid4().hex[:8]}", namespace=namespace)
    received: list[HeardSoundEvent] = []
    publisher = peer.create_publisher(SoundEvent, "sound_events", reliable(50))
    peer.create_subscription(HeardSoundEvent, "heard_sound_events", received.append, reliable(50))

    try:
        world_subscription = next(subscription for subscription in propagation.subscriptions if subscription.topic_name == f"{namespace}/state/world")
        assert world_subscription.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL

        _load_map(propagation)
        propagation._on_peds(_pedestrians((1, 0.0, 0.0), (2, 3.0, 4.0)))
        _spin_until(
            rclpy,
            [peer, propagation],
            lambda: publisher.get_subscription_count() > 0 and propagation.count_subscribers(f"{namespace}/heard_sound_events") > 0,
        )

        event = SoundEvent(source=_source("roundtrip_001", agent_id=1, agent_name="ped_1", tags=("human", "greeting")))
        event.header.frame_id = "map"
        publisher.publish(event)
        _spin_until(rclpy, [peer, propagation], lambda: len(received) >= 1)
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            rclpy.spin_once(peer, timeout_sec=0.02)

        assert len(received) == 1
        heard = received[0]
        assert heard.header.frame_id == "map"
        assert heard.source.id == "roundtrip_001"
        assert heard.source.kind == "speech"
        assert list(heard.source.tags) == ["human", "greeting"]
        reception = heard.reception
        assert reception.listener_id == "agent:2"
        assert reception.listener_position.z == pytest.approx(1.6)
        assert reception.distance_m == pytest.approx(5.0)
        assert reception.occluded is False
        assert reception.audible is True
        assert reception.backend == "legacy_distance_occlusion"
        assert reception.used_fallback is False
        assert reception.fallback_reason == ""
        assert list(reception.portal_ids) == []
        assert list(reception.traversed_zones) == []
        assert reception.rir_key == ""
        assert reception.received_level_db == pytest.approx(60.0 - 20.0 * math.log10(5.0), abs=1e-3)

        _load_map(propagation, width=510, height=695, resolution=0.05, origin=(4.75, 4.70))
        assert propagation._tracker.occupancy.cell(5.0, 4.95) == (5, 5)
    finally:
        peer.destroy_node()
        propagation.destroy_node()


def test_robot_only_policy_excludes_pedestrian_listeners(rclpy_context):
    from rclpy.parameter import Parameter

    propagation = _launch({"propagation.backend": "legacy"})
    robot = _robot_name()
    source = _spec(_source("greeting_1", agent_id=1, agent_name="ped_1"))

    try:
        _load_map(propagation)
        propagation._on_fleet(_fleet(robot))
        _place(propagation, f"{robot}/base_link", 3.0, 3.0)
        propagation._on_peds(_pedestrians((1, 1.0, 1.0), (2, 2.0, 2.0)))

        assert set(propagation._listeners_for(source, include_peds=True)) == _robot_set(robot)

        assert propagation.set_parameters([Parameter("pedestrian_listeners.enabled", value=True)])[0].successful
        assert set(propagation._listeners_for(source, include_peds=False)) == _robot_set(robot)
        assert set(propagation._listeners_for(source, include_peds=True)) == _robot_set(robot) | {"agent:2"}
    finally:
        propagation.destroy_node()


def test_propagation_runtime_toggle_stops_continuous_outputs(rclpy_context):
    from rclpy.parameter import Parameter

    propagation = _launch({"propagation.backend": "legacy"})
    robot = _robot_name()
    state = _continuous("environment:runtime_music_1:radio", position=(5.0, 5.0, 1.0))
    robot_key = (state.source.id, f"robot:{robot}")

    try:
        _load_map(propagation)
        propagation._on_fleet(_fleet(robot))
        _place(propagation, f"{robot}/base_link", 8.0, 8.0)
        first = _spawn(propagation, 2.0, 2.0, 1.0)
        second = _spawn(propagation, 3.0, 3.0, 1.0)
        assert (first.listener_id, second.listener_id) == ("microphone:runtime:1", "microphone:runtime:2")
        first_key = (state.source.id, first.listener_id)
        second_key = (state.source.id, second.listener_id)

        assert _select(propagation, second.listener_id)
        propagation._on_continuous_source(state)
        propagation._flush_continuous()
        assert propagation._last_continuous[second_key].source.active is True

        assert _select(propagation, first.listener_id)
        assert set(propagation._listeners_for(_spec(state.source), include_peds=True)) == _robot_set(robot) | {first.listener_id}
        propagation._on_continuous_source(state)
        propagation._flush_continuous()
        assert propagation._last_continuous[first_key].source.active is True
        assert propagation._last_continuous[second_key].source.active is False
        assert propagation._last_continuous[second_key].reception.audible is False
        assert propagation._last_continuous[robot_key].source.active is True

        results = propagation.set_parameters([Parameter("propagation.enabled", value=False)])
        assert results[0].successful is True
        assert propagation.get_parameter("propagation.enabled").value is False
        for key in (first_key, robot_key, (state.source.id, f"array:{robot}:left")):
            assert propagation._last_continuous[key].source.active is False
            assert propagation._last_continuous[key].reception.audible is False

        propagation._on_continuous_source(state)
        propagation._flush_continuous()
        assert propagation._pending_continuous == {}
        assert all(output.source.active is False for output in propagation._last_continuous.values())
    finally:
        propagation.destroy_node()


def test_viewport_camera_registers_selectable_microphones(rclpy_context):
    import rclpy
    from arena_rclpy_mixins.qos import latched
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String

    propagation = _launch({"propagation.backend": "legacy", "viewport.height_m": 1.7})
    peer = rclpy.create_node(f"registry_peer_{uuid.uuid4().hex[:8]}", namespace=propagation.get_namespace())
    registries: list[list[str]] = []
    peer.create_subscription(String, "microphone_listeners", lambda msg: registries.append(json.loads(msg.data)), latched(1))
    projective = "microphone:viewport:projective_center"
    down = "microphone:viewport:down_projection"
    source = _spec(_source("speech_1", position=(1.0, 1.0, 1.0)))
    camera_pose = PoseStamped()
    camera_pose.header.frame_id = "map"
    camera_pose.pose.position.x = 3.0
    camera_pose.pose.position.y = 4.0
    camera_pose.pose.position.z = 8.0

    try:
        _load_map(propagation)
        propagation._on_viewport_pose(camera_pose)
        assert propagation._viewport_microphones == {}
        _spin_until(rclpy, [peer, propagation], lambda: bool(registries) and {projective, down} <= set(registries[-1]))
        assert not {projective, down} & set(propagation._listeners_for(source, include_peds=False))

        assert _select(propagation, projective)
        assert set(propagation._viewport_microphones) == {projective, down}
        listeners = propagation._listeners_for(source, include_peds=False)
        assert listeners[projective].position == (3.0, 4.0, 8.0)
        assert down not in listeners
        assert propagation._microphone_position(down) == (3.0, 4.0, 1.7)

        assert _select(propagation, "")
        assert propagation._viewport_microphones == {}
        assert not {projective, down} & set(propagation._listeners_for(source, include_peds=False))
        assert {projective, down} <= propagation._microphone_ids()

        assert _select(propagation, down)
        assert propagation._listeners_for(source, include_peds=False)[down].position == (3.0, 4.0, 1.7)
    finally:
        peer.destroy_node()
        propagation.destroy_node()


def test_propagation_reconciles_robot_fleet_listeners(rclpy_context):
    import rclpy
    from arena_rclpy_mixins.qos import latched
    from std_msgs.msg import String

    propagation = _launch({"propagation.backend": "legacy"})
    peer = rclpy.create_node(f"registry_peer_{uuid.uuid4().hex[:8]}", namespace=propagation.get_namespace())
    registries: list[list[str]] = []
    peer.create_subscription(String, "microphone_listeners", lambda msg: registries.append(json.loads(msg.data)), latched(1))
    robot = _robot_name()
    fleet = _fleet(robot)
    source = _spec(_source("speech_1", position=(1.0, 1.0, 0.0)))
    left, right = f"array:{robot}:left", f"array:{robot}:right"

    try:
        _load_map(propagation)
        _place(propagation, f"{robot}/base_link", 4.0, 2.0)
        propagation._on_fleet(fleet)
        _spin_until(rclpy, [peer, propagation], lambda: bool(registries) and set(registries[-1]) == {left, right})

        listeners = propagation._listeners_for(source, include_peds=False)
        assert set(listeners) == _robot_set(robot)
        assert listeners[f"robot:{robot}"].position == pytest.approx((4.0, 2.0, 0.35))
        assert listeners[left].position == pytest.approx((4.0, 2.1, 0.35))
        assert listeners[right].position == pytest.approx((4.0, 1.9, 0.35))
        poses = propagation._marker_poses()
        assert poses[left].frame == poses[right].frame == f"{robot}/base_link"
        assert poses[left].position == pytest.approx((0.0, 0.1, 0.35))
        assert poses[right].position == pytest.approx((0.0, -0.1, 0.35))
        assert poses[left].color.b > poses[left].color.r
        assert poses[right].color.r > poses[right].color.b

        bindings = propagation._robots
        propagation._on_fleet(fleet)
        assert propagation._robots == bindings
        assert set(propagation._listeners_for(source, include_peds=False)) == _robot_set(robot)

        propagation._on_fleet(_fleet())
        assert propagation._robots == ()
        assert propagation._listeners_for(source, include_peds=False) == {}
        assert propagation._marker_poses() == {}
        _spin_until(rclpy, [peer, propagation], lambda: bool(registries) and registries[-1] == [])
    finally:
        peer.destroy_node()
        propagation.destroy_node()


def test_propagation_registers_four_mic_jackal_receivers(rclpy_context):
    propagation = _launch({"propagation.backend": "legacy", "array.spec": "four_mic"})
    robot = _robot_name()
    expected = {
        "front_left": ((0.19, 0.135, 0.22), math.pi / 4.0),
        "front_right": ((0.19, -0.135, 0.22), -math.pi / 4.0),
        "rear_left": ((-0.19, 0.135, 0.22), 3.0 * math.pi / 4.0),
        "rear_right": ((-0.19, -0.135, 0.22), -3.0 * math.pi / 4.0),
    }

    try:
        _load_map(propagation)
        _place(propagation, f"{robot}/base_link", 5.0, 5.0)
        propagation._on_fleet(_fleet(robot))

        poses = propagation._marker_poses()
        assert set(poses) == {f"array:{robot}:{mic}" for mic in expected}
        for mic, (position, yaw) in expected.items():
            pose = poses[f"array:{robot}:{mic}"]
            assert pose.position == pytest.approx(position)
            assert pose.yaw_rad == pytest.approx(yaw)
            assert pose.frame == f"{robot}/base_link"

        listeners = propagation._listeners_for(_spec(_source("speech_1", position=(1.0, 1.0, 0.0))), include_peds=False)
        assert set(listeners) == _robot_set(robot, tuple(expected))
        assert listeners[f"robot:{robot}"].position == pytest.approx((5.0, 5.0, 0.22))
        assert listeners[f"array:{robot}:front_left"].position == pytest.approx((5.19, 5.135, 0.22))
    finally:
        propagation.destroy_node()


def test_continuous_source_on_moving_frame_relocalizes_per_update(rclpy_context):
    propagation = _launch({"propagation.backend": "legacy"})
    robot = _robot_name()
    rider = f"rider_{uuid.uuid4().hex[:6]}/base_link"
    state = _continuous("environment:horn", frame=rider, position=(1.0, 0.0, 0.5), level_db=80.0)
    key = (state.source.id, f"robot:{robot}")

    try:
        _load_map(propagation, width=40, height=40)
        _place(propagation, f"{robot}/base_link", 0.0, 0.0)
        propagation._on_fleet(_fleet(robot))

        _place(propagation, rider, 10.0)
        propagation._on_continuous_source(state)
        propagation._flush_continuous()
        assert propagation._last_continuous[key].header.frame_id == "map"
        assert propagation._last_continuous[key].source.position.x == pytest.approx(11.0)
        assert propagation._last_continuous[key].reception.distance_m == pytest.approx(11.0)

        _place(propagation, rider, 20.0)
        propagation._on_continuous_source(state)
        propagation._flush_continuous()
        assert propagation._last_continuous[key].source.position.x == pytest.approx(21.0)
        assert propagation._last_continuous[key].reception.distance_m == pytest.approx(21.0)
    finally:
        propagation.destroy_node()


def test_sound_propagation_uses_base_frame_when_mount_frame_is_empty(rclpy_context):
    propagation = _launch({"propagation.backend": "legacy", "array.mount_frame": ""})
    robot = _robot_name()

    try:
        _load_map(propagation)
        _place(propagation, f"{robot}/base_link", 2.0, 3.0)
        propagation._on_fleet(_fleet(robot))
        assert propagation._marker_poses()[f"array:{robot}:left"].frame == f"{robot}/base_link"
        listeners = propagation._listeners_for(_spec(_source("speech_1")), include_peds=False)
        assert listeners[f"robot:{robot}"].position == pytest.approx((2.0, 3.0, 0.35))
    finally:
        propagation.destroy_node()


def test_sound_propagation_resolves_relative_mount_frame_override(rclpy_context):
    propagation = _launch({"propagation.backend": "legacy", "array.mount_frame": "oakd_rgb_camera_optical_frame"})
    robot = _robot_name()
    mount = f"{robot}/oakd_rgb_camera_optical_frame"

    try:
        _load_map(propagation)
        _place(propagation, f"{robot}/base_link", 2.0, 3.0)
        _place(propagation, mount, 6.0, 7.0, 1.0)
        propagation._on_fleet(_fleet(robot))
        assert propagation._marker_poses()[f"array:{robot}:left"].frame == mount
        listeners = propagation._listeners_for(_spec(_source("speech_1")), include_peds=False)
        assert listeners[f"robot:{robot}"].position == pytest.approx((6.0, 7.0, 1.35))
    finally:
        propagation.destroy_node()


def test_sound_propagation_uses_tf_height_for_microphones(rclpy_context):
    propagation = _launch({"propagation.backend": "legacy", "pedestrian_listeners.enabled": True})
    robot = _robot_name()

    try:
        _load_map(propagation)
        _place(propagation, f"{robot}/base_link", 4.0, 4.0, 0.1)
        propagation._on_fleet(_fleet(robot))
        propagation._on_peds(_pedestrians((2, 6.0, 6.0)))
        spawned = _spawn(propagation, 2.0, 2.0, 2.4)
        assert spawned.success is True
        assert _select(propagation, spawned.listener_id)

        listeners = propagation._listeners_for(_spec(_source("speech_1", agent_id=1)), include_peds=True)
        assert listeners[spawned.listener_id].position[2] == pytest.approx(2.4)
        assert listeners[f"robot:{robot}"].position[2] == pytest.approx(0.45)
        assert listeners[f"array:{robot}:left"].position[2] == pytest.approx(0.45)
        assert listeners["agent:2"].position[2] == pytest.approx(1.6)
    finally:
        propagation.destroy_node()


def test_self_hearing_disabled_drops_every_listener_of_the_emitting_robot(rclpy_context):
    from rclpy.parameter import Parameter

    propagation = _launch({"propagation.backend": "legacy", "propagation.self_hearing.enabled": False})
    emitter, other = _robot_name(), _robot_name()
    source = _spec(_source("motor_1", kind="motor", agent_kind="robot", agent_name=emitter))

    try:
        _load_map(propagation)
        _place(propagation, f"{emitter}/base_link", 2.0)
        _place(propagation, f"{other}/base_link", 6.0)
        propagation._on_fleet(_fleet(emitter, other))

        assert set(propagation._listeners_for(source, include_peds=False)) == _robot_set(other)

        assert propagation.set_parameters([Parameter("propagation.self_hearing.enabled", value=True)])[0].successful
        assert set(propagation._listeners_for(source, include_peds=False)) == _robot_set(emitter) | _robot_set(other)
    finally:
        propagation.destroy_node()


def test_continuous_reference_distance_offsets_source_level(rclpy_context):
    propagation = _launch({"propagation.backend": "legacy"})
    robot = _robot_name()
    key = ("environment:radio", f"robot:{robot}")

    try:
        _load_map(propagation)
        _place(propagation, f"{robot}/base_link", 10.0, 1.0)
        propagation._on_fleet(_fleet(robot))

        propagation._on_continuous_source(_continuous("environment:radio", position=(4.0, 1.0, 1.0), reference_distance_m=2.0))
        propagation._flush_continuous()
        assert propagation._last_continuous[key].reception.received_level_db == pytest.approx(70.0 + 20.0 * math.log10(2.0) - 20.0 * math.log10(6.0), abs=1e-3)

        propagation._on_continuous_source(_continuous("environment:radio", position=(4.0, 1.0, 1.0), reference_distance_m=0.0))
        propagation._flush_continuous()
        assert propagation._last_continuous[key].reception.received_level_db == pytest.approx(70.0 - 20.0 * math.log10(6.0), abs=1e-3)
    finally:
        propagation.destroy_node()


def test_continuous_updates_coalesce_to_the_latest_state_per_source(rclpy_context):
    propagation = _launch({"propagation.backend": "legacy"})
    robot = _robot_name()
    key = ("environment:radio", f"robot:{robot}")

    try:
        _load_map(propagation)
        _place(propagation, f"{robot}/base_link", 10.0, 1.0)
        propagation._on_fleet(_fleet(robot))

        propagation._on_continuous_source(_continuous("environment:radio", position=(2.0, 1.0, 1.0)))
        propagation._on_continuous_source(_continuous("environment:radio", position=(4.0, 1.0, 1.0)))
        assert list(propagation._pending_continuous) == ["environment:radio"]

        propagation._flush_continuous()

        assert propagation._pending_continuous == {}
        assert propagation._last_continuous[key].source.position.x == pytest.approx(4.0)
        assert propagation._last_continuous[key].reception.received_level_db == pytest.approx(70.0 - 20.0 * math.log10(6.0), abs=1e-3)
    finally:
        propagation.destroy_node()


def test_continuous_propagation_is_reused_within_the_position_quantum(rclpy_context):
    propagation = _launch({"propagation.backend": "legacy"})
    robot = _robot_name()
    key = ("environment:radio", f"robot:{robot}")

    try:
        _load_map(propagation)
        _place(propagation, f"{robot}/base_link", 10.0, 1.0)
        propagation._on_fleet(_fleet(robot))

        propagation._on_continuous_source(_continuous("environment:radio", position=(4.0, 1.0, 1.0)))
        propagation._flush_continuous()
        first = propagation._continuous_cache[key][1]

        propagation._on_continuous_source(_continuous("environment:radio", position=(4.02, 1.0, 1.0)))
        propagation._flush_continuous()
        assert propagation._continuous_cache[key][1] is first
        assert propagation._last_continuous[key].source.position.x == pytest.approx(4.02)

        propagation._on_continuous_source(_continuous("environment:radio", position=(5.0, 1.0, 1.0)))
        propagation._flush_continuous()
        assert propagation._continuous_cache[key][1] is not first
        assert propagation._last_continuous[key].reception.received_level_db == pytest.approx(70.0 - 20.0 * math.log10(5.0), abs=1e-3)
    finally:
        propagation.destroy_node()


def test_continuous_flush_stops_at_its_budget_and_keeps_the_rest_pending(rclpy_context):
    propagation = _launch({"propagation.backend": "legacy"})
    robot = _robot_name()
    source_ids = [f"environment:radio_{index}" for index in range(400)]

    try:
        _load_map(propagation)
        _place(propagation, f"{robot}/base_link", 10.0, 1.0)
        propagation._on_fleet(_fleet(robot))
        for index, source_id in enumerate(source_ids):
            propagation._on_continuous_source(_continuous(source_id, position=(1.0 + 0.01 * index, 1.0, 1.0)))

        propagation._flush_continuous()
        left = list(propagation._pending_continuous)
        assert 0 < len(left) < len(source_ids)
        assert left == source_ids[len(source_ids) - len(left) :]
        assert {source_id for source_id, _ in propagation._last_continuous} == set(source_ids[: len(source_ids) - len(left)])

        propagation._on_continuous_source(_continuous(left[-1], position=(9.0, 1.0, 1.0)))
        for _ in range(len(source_ids)):
            if not propagation._pending_continuous:
                break
            propagation._flush_continuous()

        assert propagation._pending_continuous == {}
        assert {source_id for source_id, _ in propagation._last_continuous} == set(source_ids)
        assert propagation._last_continuous[(left[-1], f"robot:{robot}")].source.position.x == pytest.approx(9.0)
    finally:
        propagation.destroy_node()
