from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Iterator

import numpy as np
import pytest

ROBOT = "probe"
STEMS = ("raw_array", "stem_motor", "stem_pedestrian", "stem_ambient")


def _spin_until(nodes: list[object], predicate: Callable[[], bool], timeout_sec: float = 10.0) -> None:
    import rclpy

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.01)
        if predicate():
            return
    raise AssertionError("timed out waiting for the renderer")


def _fleet(namespace: str) -> object:
    from task_generator_msgs.msg import RobotDescriptor, RobotFleet, RobotState

    fleet = RobotFleet()
    fleet.robots.append(RobotState(descriptor=RobotDescriptor(name=ROBOT, model="jackal", ns=f"{namespace}/{ROBOT}", frame=ROBOT)))
    return fleet


class Harness:
    """A renderer node built inside an event loop, in a unique namespace with its own /clock topic."""

    def __init__(self, *, role: str = "array", sim_time: bool = True, extra: dict[str, object] | None = None) -> None:
        from arena_auditory.renderer_node import RendererNode
        from rclpy.parameter import Parameter

        suffix = uuid.uuid4().hex[:8]
        self.namespace = f"/renderer_test_{suffix}"
        self.clock_topic = f"{self.namespace}/clock"
        overrides = {
            "use_sim_time": sim_time,
            "render.role": role,
            "render.lockstep.enabled": False,
            "array.spec": "four_mic",
            "output.device": "none",
            "output.enabled": False,
            "tdoa.enabled": False,
            **(extra or {}),
        }
        self.loop = asyncio.new_event_loop()

        async def build() -> RendererNode:
            return RendererNode(
                namespace=self.namespace,
                parameter_overrides=[Parameter(name, value=value) for name, value in overrides.items()],
                cli_args=["--ros-args", "-r", f"/clock:={self.clock_topic}"],
            )

        self.node = self.loop.run_until_complete(build())

    def close(self) -> None:
        self.node.destroy_node()
        self.loop.close()


@pytest.fixture
def array_harness(rclpy_context: object) -> Iterator[Harness]:
    harness = Harness()
    harness.node._on_fleet(_fleet(harness.namespace))
    yield harness
    harness.close()


@pytest.fixture
def spec_harness(request: pytest.FixtureRequest, rclpy_context: object) -> Iterator[Harness]:
    harness = Harness(extra={"array.spec": request.param})
    harness.node._on_fleet(_fleet(harness.namespace))
    yield harness
    harness.close()


def test_sim_time_rendering_rides_the_node_clock_without_a_second_clock_subscription(array_harness: Harness) -> None:
    from arena_rclpy_mixins.node import ArenaMixinNode
    from rclpy.parameter import Parameter

    async def build() -> ArenaMixinNode:
        return ArenaMixinNode(
            "renderer_base",
            namespace=array_harness.namespace,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
            cli_args=["--ros-args", "-r", f"/clock:={array_harness.clock_topic}"],
        )

    base = array_harness.loop.run_until_complete(build())
    try:
        base_topics = [subscription.topic_name for subscription in base.subscriptions]
    finally:
        base.destroy_node()
    topics = [subscription.topic_name for subscription in array_harness.node.subscriptions]
    assert topics.count(array_harness.clock_topic) == base_topics.count(array_harness.clock_topic)
    assert "/clock" not in topics


@pytest.mark.parametrize(
    ("spec_harness", "block_size", "sample_rate", "names"),
    [
        ("four_mic", 320, 16000, ["front_left", "front_right", "rear_left", "rear_right"]),
        ("stereo", 512, 44100, ["left", "right"]),
        ("mono", 512, 44100, ["mono"]),
    ],
    indirect=["spec_harness"],
)
def test_each_clock_step_renders_one_block_on_raw_and_every_stem(spec_harness: Harness, block_size: int, sample_rate: int, names: list[str]) -> None:
    import rclpy
    from arena_robots_msgs.msg import AudioFrame
    from rosgraph_msgs.msg import Clock

    node = spec_harness.node
    driver = rclpy.create_node(f"renderer_clock_driver_{uuid.uuid4().hex[:8]}", namespace=spec_harness.namespace)
    clock_pub = driver.create_publisher(Clock, spec_harness.clock_topic, 10)
    frames: dict[str, list[AudioFrame]] = {name: [] for name in STEMS}
    topics = {name: f"{spec_harness.namespace}/{ROBOT}/audio/{name}" for name in STEMS}
    for name, received in frames.items():
        driver.create_subscription(AudioFrame, topics[name], received.append, 10)
    try:
        _spin_until([driver, node], lambda: clock_pub.get_subscription_count() > 0 and all(driver.count_publishers(topic) > 0 for topic in topics.values()))
        for step in range(3):
            stamp = Clock()
            stamp.clock.sec = 10
            stamp.clock.nanosec = step * block_size * 1_000_000_000 // sample_rate
            clock_pub.publish(stamp)
            _spin_until([driver, node], lambda count=step + 1: all(len(received) >= count for received in frames.values()))
    finally:
        driver.destroy_node()

    raw = frames["raw_array"]
    assert len(raw) == 3
    assert [frame.header.stamp.nanosec for frame in raw] == [round(step * block_size * 1e9 / sample_rate) for step in range(3)]
    assert all(frame.header.stamp.sec == 10 for frame in raw)
    assert all(len(frame.data) == len(names) * block_size and frame.frame_count == block_size for frame in raw)
    assert all(list(frame.channel_names) == names for frame in raw)
    assert [len(received) for received in frames.values()] == [3, 3, 3, 3]


def _source(**fields: object) -> object:
    from arena_auditory_msgs.msg import SoundSource

    source = SoundSource(reference_distance_m=1.0, active=True, level_db=60.0)
    for name, value in fields.items():
        setattr(source, name, value)
    return source


def _reception(listener_id: str, level_db: float = 50.0) -> object:
    from arena_auditory_msgs.msg import SoundReception

    return SoundReception(listener_id=listener_id, received_level_db=level_db, direct_delay_s=0.001, audible=True)


def _drivetrain_state(seed: int, *, kind: str = "motor") -> object:
    from arena_auditory_msgs.msg import ContinuousHeardSoundState

    return ContinuousHeardSoundState(
        source=_source(
            id="robot:other",
            kind=kind,
            asset_id="motor",
            variant_id="jackal_drivetrain",
            model="drivetrain",
            agent_kind="robot",
            agent_name="other",
            seed=seed,
            state_names=["left_velocity_mps", "right_velocity_mps"],
            state_values=[0.5, 0.5],
        ),
        reception=_reception(f"array:{ROBOT}:front_left"),
    )


@pytest.mark.usefixtures("default_sounds")
def test_drivetrain_seed_change_rebinds_the_source_on_the_prewarmed_field(array_harness: Harness) -> None:
    from arena_auditory.sources.drivetrain import cache_bytes

    node = array_harness.node
    held = cache_bytes()
    node._on_continuous(_drivetrain_state(910001))
    target = node._targets[ROBOT]
    assert target.streams["robot:other"].seed == 910001

    node._on_continuous(_drivetrain_state(910002))

    assert target.streams["robot:other"].seed == 910002
    assert held > 0
    assert cache_bytes() == held


def test_output_switches_gate_their_stems_in_the_workstation_mix(array_harness: Harness) -> None:
    from arena_auditory.render.core import RenderResult
    from rclpy.parameter import Parameter

    node = array_harness.node
    ped = np.full((4, 8), 0.5, dtype=np.float32)
    ambient = np.full((4, 8), 0.25, dtype=np.float32)
    motor = np.full((4, 8), 0.5, dtype=np.float32)
    result = RenderResult(raw=np.clip(ped + ambient + motor, -1.0, 1.0), ped=ped, ambient=ambient, motor=motor, clipped_samples=32)

    assert node._audible_mix(result) is result.raw
    node.set_parameters([Parameter("output.motor.enabled", value=False)])
    np.testing.assert_allclose(node._audible_mix(result), 0.75)
    node.set_parameters([Parameter("output.ambient.enabled", value=False)])
    np.testing.assert_allclose(node._audible_mix(result), 0.5)
    node.set_parameters([Parameter("output.motor.enabled", value=True)])
    np.testing.assert_allclose(node._audible_mix(result), 1.0)


@pytest.fixture
def steady_harness(rclpy_context: object, default_sounds: object) -> Iterator[Harness]:
    harness = Harness(sim_time=False)
    harness.node._on_fleet(_fleet(harness.namespace))
    yield harness
    harness.close()


def _heard(kind: str, asset_id: str, variant_id: str, agent_kind: str, event_id: str) -> object:
    from arena_auditory_msgs.msg import HeardSoundEvent

    return HeardSoundEvent(
        source=_source(id=event_id, kind=kind, asset_id=asset_id, variant_id=variant_id, model="wav", agent_kind=agent_kind, agent_name="emitter"),
        reception=_reception(f"array:{ROBOT}:rear_left"),
    )


def _loop_state(kind: str, asset_id: str, variant_id: str, source_id: str) -> object:
    from arena_auditory_msgs.msg import ContinuousHeardSoundState

    return ContinuousHeardSoundState(
        source=_source(id=source_id, kind=kind, asset_id=asset_id, variant_id=variant_id, model="wav_loop", agent_kind="environment", loop=True),
        reception=_reception(f"array:{ROBOT}:front_right"),
    )


def _wait_for_loads(harness: Harness) -> None:
    target = harness.node._targets[ROBOT]
    futures = [load.future for load in target.event_loads.values() if load.future is not None]
    futures += [pending.future for pending in target.continuous_pending.values()]
    deadline = time.monotonic() + 30.0
    while not all(future.done() for future in futures):
        assert time.monotonic() < deadline, "sample decode timed out"
        time.sleep(0.05)
    harness.node._poll_loads(target)


def test_renderer_routes_each_source_to_the_stem_of_its_library_kind(steady_harness: Harness) -> None:
    node = steady_harness.node
    node._on_heard(_heard("footstep", "footstep", "footstep_default_01", "pedestrian", "footstep:1"))
    node._on_heard(_heard("speech", "greeting", "greeting_01", "pedestrian", "speech:pedestrian"))
    node._on_heard(_heard("speech", "greeting", "greeting_01", "robot", "speech:robot"))
    node._on_continuous(_loop_state("music", "radio_loop", "radio_loop_01", "radio"))
    node._on_continuous(_loop_state("alarm", "alarm_loop", "alarm_loop_01", "alarm"))
    node._on_continuous(_drivetrain_state(5))
    _wait_for_loads(steady_harness)

    target = node._targets[ROBOT]
    clip_stems = sorted((clip.asset_key, clip.stem) for clip in target.pending_clips)
    assert len(clip_stems) == 3
    assert {stem for _, stem in clip_stems} == {"pedestrian"}
    assert {key.split("#")[0] for key, _ in clip_stems} == {"footstep", "greeting"}
    assert target.continuous[("radio", 1)].stem == "ambient"
    assert target.continuous[("alarm", 1)].stem == "ambient"
    assert target.streams["robot:other"].stem == "motor"


def test_continuous_source_without_updates_stops_after_the_stale_window(steady_harness: Harness) -> None:
    from arena_auditory.renderer_node import CONTINUOUS_STALE_S

    node = steady_harness.node
    node._on_continuous(_drivetrain_state(5))
    target = node._targets[ROBOT]
    node._expire_gates(target)
    assert target.streams["robot:other"].active

    time.sleep(CONTINUOUS_STALE_S + 0.1)
    node._expire_gates(target)
    assert not target.streams["robot:other"].active
    assert "robot:other" not in target.gates


def test_heard_event_starts_one_block_past_the_next_unrendered_block(steady_harness: Harness) -> None:
    node = steady_harness.node
    node._skip(5)
    node._on_heard(_heard("footstep", "footstep", "footstep_default_01", "pedestrian", "footstep:anchor"))
    anchor = node._sample_index + node.block_size
    assert node._targets[ROBOT].event_loads["footstep:anchor"].anchor == anchor
    _wait_for_loads(steady_harness)

    clips = node._targets[ROBOT].pending_clips
    assert [clip.anchor for clip in clips] == [anchor]
    assert clips[0].start >= anchor


def test_event_decoded_after_blocks_rendered_is_rescheduled_one_block_past_the_render_cursor(steady_harness: Harness) -> None:
    node = steady_harness.node
    node._on_heard(_heard("footstep", "footstep", "footstep_default_01", "pedestrian", "footstep:cold"))
    node._skip(4)
    _wait_for_loads(steady_harness)

    clips = node._targets[ROBOT].pending_clips
    assert [clip.anchor for clip in clips] == [node._sample_index + node.block_size]
    assert clips[0].start >= node._sample_index + node.block_size


def _impulse(key: str) -> object:
    from arena_auditory_msgs.msg import RoomImpulse

    return RoomImpulse(key=key, sample_rate_hz=16000, lead_samples=0, samples=[1.0, 0.5, 0.25, 0.125])


@pytest.fixture
def rir_harness(rclpy_context: object, default_sounds: object) -> Iterator[Harness]:
    harness = Harness(sim_time=False, extra={"render.rir.enabled": True})
    harness.node._on_fleet(_fleet(harness.namespace))
    yield harness
    harness.close()


def test_event_impulse_arriving_one_poll_late_is_applied_to_a_reanchored_clip(rir_harness: Harness) -> None:
    node = rir_harness.node
    target = node._targets[ROBOT]
    event = _heard("footstep", "footstep", "footstep_default_01", "pedestrian", "footstep:late_room")
    event.reception.rir_key = "room:late"
    node._on_heard(event)
    _wait_for_loads(rir_harness)
    assert target.pending_clips == []

    node._skip(1)
    node._on_impulse(_impulse("room:late"))
    node._poll_loads(target)

    clips = target.pending_clips
    assert [clip.rir_key for clip in clips] == ["room:late"]
    assert clips[0].anchor == node._sample_index + node.block_size


def test_event_without_its_impulse_renders_dry_once_the_wait_block_passed(rir_harness: Harness) -> None:
    from arena_auditory.renderer_node import IMPULSE_WAIT_BLOCKS

    node = rir_harness.node
    target = node._targets[ROBOT]
    event = _heard("footstep", "footstep", "footstep_default_01", "pedestrian", "footstep:lost_room")
    event.reception.rir_key = "room:lost"
    node._on_heard(event)
    _wait_for_loads(rir_harness)
    assert target.pending_clips == []

    node._skip(IMPULSE_WAIT_BLOCKS)
    node._poll_loads(target)

    clips = target.pending_clips
    assert [clip.rir_key for clip in clips] == [""]
    assert clips[0].anchor == node._sample_index + node.block_size


def test_continuous_voice_waits_for_the_impulse_of_its_first_room(rir_harness: Harness) -> None:
    node = rir_harness.node
    target = node._targets[ROBOT]
    state = _loop_state("music", "radio_loop", "radio_loop_01", "radio")
    state.reception.rir_key = "room:radio"
    node._on_continuous(state)
    _wait_for_loads(rir_harness)
    assert ("radio", 1) in target.continuous_pending
    assert target.continuous == {}

    node._skip(1)
    node._on_impulse(_impulse("room:radio"))
    node._poll_loads(target)

    assert target.continuous_pending == {}
    assert target.continuous[("radio", 1)].rir_key == "room:radio"


def test_robot_speech_streamed_as_a_robot_source_still_takes_the_speech_stem(steady_harness: Harness) -> None:
    node = steady_harness.node
    node._on_continuous(_drivetrain_state(6, kind="speech"))
    assert node._targets[ROBOT].streams["robot:other"].stem == "pedestrian"


def test_listener_renderer_wants_output_only_once_a_listener_is_selected(rclpy_context: object) -> None:
    from rclpy.parameter import Parameter

    harness = Harness(role="listener")
    node = harness.node
    try:
        assert node._idle()
        assert not node._wants_output("auto")
        assert node._routes == {}
        assert node._output is None

        node.set_parameters([Parameter("listener.id", value="microphone:runtime:1")])

        assert not node._idle()
        assert node._wants_output("auto")
        assert not node._wants_output("none")
        assert list(node._routes) == ["microphone:runtime:1"]
        assert node._output is None
    finally:
        harness.close()
