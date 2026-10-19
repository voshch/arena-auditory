from __future__ import annotations

import array
import asyncio
import dataclasses
import io
import math
import threading
import time
import uuid
import wave

import pytest

pytest.importorskip("arena_mcp")
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("arena_auditory_msgs.msg")
pytest.importorskip("arena_robots_msgs.msg")
pytest.importorskip("arena_runtime_msgs.msg")
pytest.importorskip("task_generator_msgs.msg")

from arena_mcp.core import ROS, ToolError

from arena_auditory import mcp as acoustics

RATE = 48000
BLOCK = 960
IMPULSE_KEY = "impulse_under_test"
IMPULSE_RATE = 16000
IMPULSE_RT60_S = 0.4


@dataclasses.dataclass
class Rig:
    loop: asyncio.AbstractEventLoop
    tg: str
    peer: object
    publishers: dict

    def run(self, coro):
        return self.loop.run_until_complete(coro)


def _propagation(tg: str):
    from arena_rclpy_mixins import ArenaMixinNode
    from rclpy.parameter import Parameter

    from arena_auditory.sound_propagation_node import SoundPropagationNode

    overrides = [Parameter("propagation.backend", value="legacy")]

    class Scoped(ArenaMixinNode):
        def __init__(self, node_name: str) -> None:
            super().__init__(node_name, namespace=tg, parameter_overrides=overrides)

    class Propagation(SoundPropagationNode, Scoped):
        pass

    async def construct():
        return Propagation()

    return asyncio.run(construct())


def _impulse():
    import numpy as np
    from arena_auditory_msgs.msg import RoomImpulse

    rng = np.random.default_rng(0)
    t = np.arange(IMPULSE_RATE) / IMPULSE_RATE
    samples = 0.05 * rng.standard_normal(t.size) * np.exp(-3.0 * math.log(10.0) * t / IMPULSE_RT60_S)
    samples[10] = 1.0
    return RoomImpulse(key=IMPULSE_KEY, sample_rate_hz=IMPULSE_RATE, lead_samples=10, samples=array.array("f", samples.astype(np.float32).tobytes()), backend="test")


def _audio_frame(index: int):
    import numpy as np
    from arena_robots_msgs.msg import AudioFrame

    t = (index * BLOCK + np.arange(BLOCK)) / RATE
    left = 0.5 * np.sin(2.0 * math.pi * 440.0 * t)
    right = 0.25 * np.sin(2.0 * math.pi * 440.0 * t)
    frame = AudioFrame(sample_rate=RATE, channel_count=2, frame_count=BLOCK, encoding="32FC1", interleaved=True, channel_names=["left", "right"])
    frame.data = array.array("f", np.stack([left, right], axis=1).astype(np.float32).tobytes())
    return frame


@pytest.fixture(scope="module")
def rig():
    from arena_auditory_msgs.msg import HeardSoundEvent, RoomImpulse, SoundEvent
    from arena_rclpy_mixins.qos import latched, reliable
    from arena_robots_msgs.msg import AudioFrame
    from arena_runtime_msgs.msg import EnvRecord, EnvRegistry
    from nav_msgs.msg import OccupancyGrid
    from rclpy.executors import SingleThreadedExecutor
    from task_generator_msgs.msg import RobotDescriptor, RobotFleet, RobotState

    tg = f"/mcp_{uuid.uuid4().hex[:8]}/env_0/task_generator_node"
    propagation = _propagation(tg)
    peer = rclpy.create_node(f"acoustics_peer_{uuid.uuid4().hex[:8]}")
    publishers = {
        "envs": peer.create_publisher(EnvRegistry, "/arena/state/envs", latched(1)),
        "robots": peer.create_publisher(RobotFleet, f"{tg}/state/robots", latched(1)),
        "map": peer.create_publisher(OccupancyGrid, f"{tg}/map", latched(1)),
        "impulses": peer.create_publisher(RoomImpulse, f"{tg}/acoustic/impulses", latched(256)),
        "heard": peer.create_publisher(HeardSoundEvent, f"{tg}/heard_sound_events", reliable(50)),
        "sounds": peer.create_publisher(SoundEvent, f"{tg}/sound_events", reliable(50)),
        "audio": peer.create_publisher(AudioFrame, f"{tg}/jackal/audio/raw_array", reliable(10)),
    }
    publishers["envs"].publish(EnvRegistry(envs=[EnvRecord(env_id=0, fqn=tg, ready=True)]))
    publishers["robots"].publish(RobotFleet(robots=[RobotState(descriptor=RobotDescriptor(name="jackal", model="jackal", ns=f"{tg}/jackal", frame="env_0/jackal", base_frame="env_0/jackal/base_link"))]))
    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.resolution = 1.0
    grid.info.width = grid.info.height = 20
    grid.info.origin.orientation.w = 1.0
    grid.data = [0] * 400
    publishers["map"].publish(grid)
    publishers["impulses"].publish(_impulse())
    blocks = iter(range(1 << 30))
    peer.create_timer(BLOCK / RATE, lambda: publishers["audio"].publish(_audio_frame(next(blocks))))
    executor = SingleThreadedExecutor()
    executor.add_node(peer)
    executor.add_node(propagation)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    loop = asyncio.new_event_loop()
    try:
        yield Rig(loop=loop, tg=tg, peer=peer, publishers=publishers)
    finally:
        executor.shutdown()
        peer.destroy_node()
        propagation.destroy_node()
        loop.run_until_complete(ROS.shutdown())
        loop.close()
        if not rclpy.ok():
            rclpy.init()


def _sound_event(rig: Rig, x: float, y: float, z: float):
    from arena_auditory_msgs.msg import SoundEvent, SoundSource
    from geometry_msgs.msg import Point

    event = SoundEvent(source=SoundSource(id=f"speech_{uuid.uuid4().hex[:6]}", kind="speech", asset_id="speech", model="wav", agent_kind="environment", agent_id=-1, position=Point(x=x, y=y, z=z), level_db=60.0, active=True))
    event.header.frame_id = "map"
    event.header.stamp = rig.peer.get_clock().now().to_msg()
    return event


async def _later(delay_s: float, action) -> None:
    await asyncio.sleep(delay_s)
    action()


def test_acoustics_domain_is_served_from_its_entry_point():
    from arena_mcp.server import discover

    (loaded,) = discover(["acoustics"])
    assert loaded.served, loaded.reason
    assert {spec.name for spec in loaded.domain.tools} == {
        "acoustics_microphones",
        "acoustics_spawn_microphone",
        "acoustics_remove_microphone",
        "acoustics_heard",
        "acoustics_sound_events",
        "acoustics_impulse",
        "acoustics_audio",
    }


def test_spawned_microphone_hears_a_sound_event_and_is_removed(rig: Rig):
    async def scenario():
        spawned = await acoustics.acoustics_spawn_microphone(3.0, 4.0, 0.0)
        listener = spawned["listener_id"]
        deadline = time.monotonic() + 5.0
        listed: list[str] = []
        while listener not in listed and time.monotonic() < deadline:
            microphones = await acoustics.acoustics_microphones()
            listed = [entry["id"] for entry in microphones["listeners"]]
            await asyncio.sleep(0.1)
        await asyncio.gather(acoustics.acoustics_heard(window_s=0.5), acoustics.acoustics_sound_events(window_s=0.5))
        event = _sound_event(rig, 0.0, 0.0, 0.0)
        rig.publishers["sounds"].publish(event)
        deadline = time.monotonic() + 20.0
        heard = await acoustics.acoustics_heard(window_s=5.0, listener=listener)
        while listener not in heard["events"] and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
            heard = await acoustics.acoustics_heard(window_s=5.0, listener=listener)
        emitted = await acoustics.acoustics_sound_events(window_s=5.0)
        removed = await acoustics.acoustics_remove_microphone(listener)
        with pytest.raises(ToolError, match="unknown runtime microphone"):
            await acoustics.acoustics_remove_microphone(listener)
        with pytest.raises(ToolError, match="below the floor"):
            await acoustics.acoustics_spawn_microphone(1.0, 1.0, -1.0)
        return spawned, microphones, listed, heard, emitted, removed, event

    spawned, microphones, listed, heard, emitted, removed, event = rig.run(scenario())
    listener = spawned["listener_id"]
    assert listener.startswith("microphone:runtime:")
    assert spawned["frame"] == "map"
    assert listener in listed
    assert microphones["selected"] == listener
    assert next(entry for entry in microphones["listeners"] if entry["id"] == listener)["removable"] is True
    receptions = heard["events"][listener]
    assert [entry["source"]["id"] for entry in receptions] == [event.source.id]
    assert receptions[0]["distance_m"] == pytest.approx(5.0, abs=1e-3)
    assert receptions[0]["audible"] is True
    assert [entry["id"] for entry in emitted["events"]] == [event.source.id]
    assert removed == {"removed": listener}


def test_impulse_is_found_through_the_latest_reception_of_its_listener(rig: Rig):
    from arena_auditory_msgs.msg import HeardSoundEvent

    listener = "microphone:runtime:99"
    heard = HeardSoundEvent()
    heard.header.stamp = rig.peer.get_clock().now().to_msg()
    heard.reception.listener_id = listener
    heard.reception.rir_key = IMPULSE_KEY

    async def scenario():
        by_listener, _ = await asyncio.gather(acoustics.acoustics_impulse(listener=listener), _later(0.5, lambda: rig.publishers["heard"].publish(heard)))
        listed = await acoustics.acoustics_impulse()
        with pytest.raises(ToolError, match="no room impulse"):
            await acoustics.acoustics_impulse(key="missing")
        with pytest.raises(ToolError, match="references a room impulse"):
            await acoustics.acoustics_impulse(listener="microphone:runtime:404")
        return by_listener, listed

    by_listener, listed = rig.run(scenario())
    assert by_listener["key"] == IMPULSE_KEY
    assert by_listener["listener"] == listener
    assert by_listener["sample_rate_hz"] == IMPULSE_RATE
    assert by_listener["length_s"] == pytest.approx(1.0)
    assert by_listener["rt60_s"] == pytest.approx(IMPULSE_RT60_S, rel=0.15)
    assert by_listener["drr_db"] is not None
    assert IMPULSE_KEY in [entry["key"] for entry in listed["impulses"]]


def test_audio_capture_returns_a_wav_of_the_robot_stream(rig: Rig):
    async def scenario():
        captured = await acoustics.acoustics_audio(seconds=0.5)
        with pytest.raises(ToolError, match="unknown stream"):
            await acoustics.acoustics_audio(stream="nope")
        with pytest.raises(ToolError, match="is not published"):
            await acoustics.acoustics_audio(stream="stem_motor")
        return captured

    summary, audio = rig.run(scenario())
    assert summary["topic"] == f"{rig.tg}/jackal/audio/raw_array"
    assert summary["complete"] is True
    assert summary["channel_names"] == ["left", "right"]
    assert summary["level_dbfs"] == [pytest.approx(-6.0, abs=0.3), pytest.approx(-12.0, abs=0.3)]
    content = audio.to_audio_content()
    assert content.mime_type == "audio/wav"
    with wave.open(io.BytesIO(audio.data)) as stream:
        assert (stream.getnchannels(), stream.getframerate(), stream.getnframes()) == (2, RATE, RATE // 2)


def test_env_scoped_tools_fail_fast_on_an_unknown_env(rig: Rig):
    async def scenario():
        start = time.monotonic()
        with pytest.raises(ToolError, match="no env 7"):
            await acoustics.acoustics_microphones(env_id=7)
        return time.monotonic() - start

    assert rig.run(scenario()) < 3.0
