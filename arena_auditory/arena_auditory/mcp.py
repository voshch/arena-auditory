from __future__ import annotations

import asyncio
import io
import json
import math
import time
import wave
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from arena_auditory.constants import CONTINUOUS_AUDIO_SOURCES, CONTINUOUS_HEARD_SOUNDS, HEARD_SOUND_EVENTS, MAP, MICROPHONE_LISTENERS, REMOVE_MICROPHONE, ROOM_IMPULSES, SOUND_EVENTS, SPAWN_MICROPHONE
from arena_mcp.core import LATCHED, RELIABLE, ROS, SENSOR, Domain, ToolError, msgs, robot_ns, tg

if TYPE_CHECKING:
    import numpy as np
    from arena_auditory_msgs.msg import ContinuousHeardSoundState, HeardSoundEvent, RoomImpulse
    from mcp.server.mcpserver import Audio

    from arena_mcp.core.ros import TopicCache

domain = Domain(
    "acoustics",
    """Arena auditory simulation (acoustics:=arena): microphones, propagated sound receptions, sound events, room impulses and robot audio.
    Every tool takes env_id. Event tools watch their topics from the first call on, so a first call waits up to window_s while it collects.
    Sound sources are spawned by entities_spawn_sound.""",
    requires=("arena_auditory_msgs", "arena_robots_msgs"),
)

NO_AUDITORY = "the env runs without acoustics:=arena, or its sound propagation node is not up"
PROPAGATION_NODE = "sound_propagation_node"
DEPTH = 4096
MAX_WAIT_S = 10.0
MAX_AUDIO_S = 10.0

_watched: dict[str, float] = {}


async def _watch(msg_type: type, topic: str, qos: str) -> tuple[TopicCache, float]:
    cache = await ROS.cache(msg_type, topic, qos=qos, depth=DEPTH)
    return cache, _watched.setdefault(topic, time.monotonic())


async def _collect(caches: list[tuple[TopicCache, float]], window_s: float) -> float:
    began = max(start for _, start in caches)
    waited = time.monotonic() - began
    if waited < window_s:
        await asyncio.sleep(min(window_s - waited, MAX_WAIT_S))
    return round(time.monotonic() - min(start for _, start in caches), 1)


async def _now() -> float:
    return (await ROS.node()).sim_time.to_seconds()


def _recent[MsgT](received: list[MsgT], window_s: float, now: float) -> list[MsgT]:
    stamps = [msgs.seconds(msg.header.stamp) for msg in received]
    reference = now if now > 0.0 else max(stamps, default=0.0)
    return [msg for msg, stamp in zip(received, stamps, strict=True) if reference - window_s <= stamp <= reference + 1.0]


def _listener(listener_id: str) -> dict[str, Any]:
    from arena_auditory.shared import ListenerId, MicrophoneScope

    try:
        parsed = ListenerId.parse(listener_id)
    except ValueError:
        return {"id": listener_id}
    out: dict[str, Any] = {"id": listener_id, "kind": str(parsed.kind), "removable": parsed.scope is MicrophoneScope.RUNTIME}
    if parsed.scope is not None:
        out["scope"] = str(parsed.scope)
    if parsed.owner:
        out["owner"] = parsed.owner
    if parsed.name:
        out["name"] = parsed.name
    if parsed.robot_name is not None:
        out["robot"] = parsed.robot_name
    return out


def _heard(msg: HeardSoundEvent | ContinuousHeardSoundState, detail: bool) -> dict[str, Any]:
    if detail:
        return msgs.to_dict(msg)
    source, reception = msg.source, msg.reception
    out = {
        "stamp": msgs.seconds(msg.header.stamp),
        "source": {"id": source.id, "kind": source.kind, "asset_id": source.asset_id, "agent_kind": source.agent_kind, "agent_name": source.agent_name, "level_db": source.level_db},
        "distance_m": reception.distance_m,
        "bearing_rad": reception.bearing_rad,
        "received_level_db": reception.received_level_db,
        "threshold_db": reception.threshold_db,
        "audible": reception.audible,
        "occluded": reception.occluded,
        "backend": reception.backend,
        "rir_key": reception.rir_key,
    }
    if reception.used_fallback:
        out["fallback_reason"] = reception.fallback_reason
    return out


def _decay_time(edc_db: np.ndarray, rate: int) -> float | None:
    import numpy as np

    if edc_db.size == 0 or edc_db.min() > -25.0:
        return None
    start = int(np.argmax(edc_db <= -5.0))
    end = int(np.argmax(edc_db <= -25.0))
    return 3.0 * (end - start) / rate if end > start else None


def _impulse(msg: RoomImpulse) -> dict[str, Any]:
    import numpy as np

    samples = np.asarray(msg.samples, dtype=np.float64)
    rate = int(msg.sample_rate_hz)
    out: dict[str, Any] = {"key": msg.key, "backend": msg.backend, "stamp": msgs.seconds(msg.header.stamp), "sample_rate_hz": rate, "lead_samples": int(msg.lead_samples), "samples": int(samples.size)}
    if samples.size == 0 or rate <= 0:
        return out
    energy = samples * samples
    total = float(energy.sum())
    out["length_s"] = round(samples.size / rate, 4)
    out["peak"] = float(np.abs(samples).max())
    if total <= 0.0:
        return out
    edc = np.cumsum(energy[::-1])[::-1]
    out["rt60_s"] = _decay_time(10.0 * np.log10(np.maximum(edc / edc[0], 1e-12)), rate)
    half = max(1, round(0.0025 * rate))
    lead = int(msg.lead_samples)
    direct = float(energy[max(0, lead - half) : lead + half + 1].sum())
    out["drr_db"] = round(10.0 * math.log10(direct / (total - direct)), 2) if 0.0 < direct < total else None
    return out


@domain.tool(read_only=True)
async def acoustics_microphones(env_id: int | None = None) -> dict[str, Any]:
    """Every listener the propagation node knows and the selected one. Robots and pedestrians always hear, a microphone only while selected."""
    from std_msgs.msg import String

    node = await tg(env_id)
    msg = await ROS.latest(String, f"{node}/{MICROPHONE_LISTENERS}", qos=LATCHED, timeout=3.0, hint=NO_AUDITORY)
    selected = (await ROS.get_params(f"{node}/{PROPAGATION_NODE}", ["listener.id"]))["listener.id"]
    return {"task_generator": node, "selected": selected or None, "listeners": [_listener(listener_id) for listener_id in json.loads(msg.data)]}


@domain.tool()
async def acoustics_spawn_microphone(
    x: float,
    y: float,
    z: Annotated[float, Field(description="height above the floor in m, at most the zone ceiling")] = 1.5,
    env_id: int | None = None,
    frame: Annotated[str | None, Field(description="frame of x, y, z, default the env map frame")] = None,
    attached_frame: Annotated[str, Field(description="TF frame the microphone rides along with, empty for a fixed microphone")] = "",
    placement: Annotated[str, Field(description="placement label, no ':'")] = "placed",
    select: Annotated[bool, Field(description="make it the selected listener, the one microphone that is propagated and rendered")] = True,
) -> dict[str, Any]:
    """Spawn a runtime microphone listener, cleared on episode reset, and by default select it so it hears."""
    from arena_auditory_msgs.srv import SpawnMicrophone
    from nav_msgs.msg import OccupancyGrid

    node = await tg(env_id)
    if frame is None:
        grid = await ROS.latest(OccupancyGrid, f"{node}/{MAP}", qos=LATCHED, timeout=3.0, hint="the env map is not published yet, pass frame")
        frame = grid.header.frame_id
    request = SpawnMicrophone.Request(placement=placement, attached_frame=attached_frame)
    request.position.header.frame_id = frame
    request.position.point.x, request.position.point.y, request.position.point.z = float(x), float(y), float(z)
    response = await ROS.call(SpawnMicrophone, f"{node}/{SPAWN_MICROPHONE}", request, hint=NO_AUDITORY)
    if not response.success:
        raise ToolError(f"spawn_microphone rejected: {response.error_msg}")
    if select:
        result = (await ROS.set_params(f"{node}/{PROPAGATION_NODE}", {"listener.id": response.listener_id}))["listener.id"]
        if not result["ok"]:
            raise ToolError(f"spawned {response.listener_id} but could not select it: {result['reason']}")
    return {"listener_id": response.listener_id, "zone": response.zone, "attached_frame": response.attached_frame, "frame": frame, "selected": select}


@domain.tool(destructive=True, idempotent=True)
async def acoustics_remove_microphone(listener_id: str, env_id: int | None = None) -> dict[str, Any]:
    """Remove a runtime microphone (microphone:runtime:<n>), world, robot and viewport microphones stay."""
    from arena_auditory_msgs.srv import RemoveMicrophone

    node = await tg(env_id)
    response = await ROS.call(RemoveMicrophone, f"{node}/{REMOVE_MICROPHONE}", RemoveMicrophone.Request(listener_id=listener_id), hint=NO_AUDITORY)
    if not response.success:
        raise ToolError(f"remove_microphone rejected: {response.error_msg}", hint="acoustics_microphones lists the removable ones")
    return {"removed": listener_id}


@domain.tool(read_only=True)
async def acoustics_heard(
    env_id: int | None = None,
    window_s: Annotated[float, Field(description="sim seconds back from now")] = 5.0,
    listener: Annotated[str | None, Field(description="only this listener id")] = None,
    continuous: Annotated[bool, Field(description="also the latest reception of every continuous source")] = True,
    detail: Annotated[bool, Field(description="full messages including early reflection paths")] = False,
) -> dict[str, Any]:
    """Propagated receptions per listener over the last window_s, one-shot events and continuous sources. A microphone hears only while selected."""
    from arena_auditory_msgs.msg import ContinuousHeardSoundState, HeardSoundEvent

    if window_s <= 0.0:
        raise ToolError("window_s must be positive")
    node = await tg(env_id)
    watches = [await _watch(HeardSoundEvent, f"{node}/{HEARD_SOUND_EVENTS}", RELIABLE)]
    if continuous:
        watches.append(await _watch(ContinuousHeardSoundState, f"{node}/{CONTINUOUS_HEARD_SOUNDS}", SENSOR))
    watched = await _collect(watches, window_s)
    now = await _now()
    events: dict[str, list[dict[str, Any]]] = {}
    for msg in _recent(list(watches[0][0].history), window_s, now):
        if listener is None or msg.reception.listener_id == listener:
            events.setdefault(msg.reception.listener_id, []).append(_heard(msg, detail))
    out: dict[str, Any] = {"window_s": window_s, "watched_s": watched, "events": events}
    if continuous:
        latest: dict[str, dict[str, Any]] = {}
        for msg in _recent(list(watches[1][0].history), window_s, now):
            if listener is None or msg.reception.listener_id == listener:
                latest.setdefault(msg.reception.listener_id, {})[msg.source.id] = _heard(msg, detail)
        out["continuous"] = latest
    return out


@domain.tool(read_only=True)
async def acoustics_sound_events(
    env_id: int | None = None,
    window_s: Annotated[float, Field(description="sim seconds back from now")] = 5.0,
) -> dict[str, Any]:
    """Sounds emitted over the last window_s: one-shot sound events and the latest state of each continuous source."""
    from arena_auditory_msgs.msg import ContinuousAudioSourceState, SoundEvent

    if window_s <= 0.0:
        raise ToolError("window_s must be positive")
    node = await tg(env_id)
    events = await _watch(SoundEvent, f"{node}/{SOUND_EVENTS}", RELIABLE)
    sources = await _watch(ContinuousAudioSourceState, f"{node}/{CONTINUOUS_AUDIO_SOURCES}", SENSOR)
    watched = await _collect([events, sources], window_s)
    now = await _now()
    latest: dict[str, dict[str, Any]] = {}
    for msg in _recent(list(sources[0].history), window_s, now):
        latest[msg.source.id] = {"stamp": msgs.seconds(msg.header.stamp), "frame": msg.header.frame_id, **msgs.to_dict(msg.source)}
    return {
        "window_s": window_s,
        "watched_s": watched,
        "events": [{"stamp": msgs.seconds(msg.header.stamp), "frame": msg.header.frame_id, **msgs.to_dict(msg.source)} for msg in _recent(list(events[0].history), window_s, now)],
        "continuous": latest,
    }


@domain.tool(read_only=True)
async def acoustics_impulse(
    env_id: int | None = None,
    listener: Annotated[str | None, Field(description="listener id, picks the impulse of its latest reception")] = None,
    key: Annotated[str | None, Field(description="impulse key, as in a reception's rir_key")] = None,
) -> dict[str, Any]:
    """Room impulse summary (length, RT60 from the Schroeder decay, direct-to-reverberant ratio) by listener or key, or the known impulses."""
    from arena_auditory_msgs.msg import ContinuousHeardSoundState, HeardSoundEvent, RoomImpulse

    node = await tg(env_id)
    impulses = await _watch(RoomImpulse, f"{node}/{ROOM_IMPULSES}", LATCHED)
    if listener is not None and key is None:
        receptions = [await _watch(HeardSoundEvent, f"{node}/{HEARD_SOUND_EVENTS}", RELIABLE), await _watch(ContinuousHeardSoundState, f"{node}/{CONTINUOUS_HEARD_SOUNDS}", SENSOR)]
        await _collect([impulses, *receptions], 2.0)
        candidates = [msg for cache, _ in receptions for msg in list(cache.history) if msg.reception.listener_id == listener and msg.reception.rir_key]
        if not candidates:
            raise ToolError(f"no reception of {listener!r} references a room impulse", hint="impulses exist only for listeners a renderer uses and with an RIR propagation backend, acoustics_heard shows the receptions")
        heard = max(candidates, key=lambda msg: msgs.seconds(msg.header.stamp))
        key = heard.reception.rir_key
    else:
        await _collect([impulses], 0.5)
    known = {msg.key: msg for msg in list(impulses[0].history)}
    if key is None:
        return {"impulses": [_impulse(msg) for msg in sorted(known.values(), key=lambda msg: msgs.seconds(msg.header.stamp), reverse=True)[:20]]}
    if key not in known:
        raise ToolError(f"no room impulse {key!r} was published", hint=f"{len(known)} impulses known, call without listener and key to list them")
    out = _impulse(known[key])
    if listener is not None:
        out["listener"] = listener
    return out


@domain.tool(read_only=True)
async def acoustics_audio(
    env_id: int | None = None,
    robot: str | None = None,
    seconds: Annotated[float, Field(description=f"capture length, at most {MAX_AUDIO_S:g}")] = 2.0,
    stream: Annotated[str, Field(description="array stream below <robot>/audio, e.g. raw_array, stem_motor, headphones/stereo, hearing/mono")] = "raw_array",
) -> list[Any]:
    """A short capture of a robot's audio stream as 16 bit WAV, with per-channel levels."""
    import numpy as np
    from arena_robots.audio import ArrayStream, array_stream, dbfs_from_rms, rms
    from arena_robots_msgs.msg import AudioFrame

    if not 0.0 < seconds <= MAX_AUDIO_S:
        raise ToolError(f"seconds must be in (0, {MAX_AUDIO_S:g}]")
    try:
        selected = ArrayStream(stream)
    except ValueError as exc:
        raise ToolError(f"unknown stream {stream!r}", hint=f"streams: {', '.join(item.value for item in ArrayStream)}") from exc
    topic = array_stream(await robot_ns(env_id, robot), selected)
    node = await ROS.node()
    types = dict(node.get_topic_names_and_types())
    if topic not in types:
        raise ToolError(f"{topic} is not published", hint=f"{NO_AUDITORY}, or the robot has no microphone array")
    if "arena_robots_msgs/msg/AudioFrame" not in types[topic]:
        raise ToolError(f"{topic} carries {', '.join(types[topic])}, not audio frames")
    cache = await ROS.cache(AudioFrame, topic, qos=RELIABLE, depth=DEPTH)
    first = await ROS.next(AudioFrame, topic, qos=RELIABLE, timeout=5.0)
    frames = [first]
    rate = int(first.sample_rate)
    needed = int(seconds * rate)
    deadline = time.monotonic() + 10.0 + 4.0 * seconds
    while sum(frame.frame_count for frame in frames) < needed and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        history = list(cache.history)
        start = next((index for index, frame in enumerate(history) if frame is first), None)
        frames = history[start:] if start is not None else frames
    blocks = []
    for frame in frames:
        values = np.asarray(frame.data, dtype=np.float32)
        if frame.encoding != "32FC1" or not frame.interleaved or values.size != frame.channel_count * frame.frame_count or frame.channel_count != first.channel_count:
            raise ToolError(f"{topic}: malformed or changing AudioFrame ({frame.encoding}, {frame.channel_count} x {frame.frame_count})")
        blocks.append(values.reshape(frame.frame_count, frame.channel_count))
    audio = np.concatenate(blocks)[:needed]
    return [
        {
            "topic": topic,
            "sample_rate": rate,
            "channels": int(first.channel_count),
            "channel_names": list(first.channel_names),
            "seconds": round(audio.shape[0] / rate, 3),
            "complete": audio.shape[0] >= needed,
            "level_dbfs": [round(dbfs_from_rms(float(rms(audio[:, channel]))), 1) for channel in range(audio.shape[1])],
        },
        _wav(audio, rate),
    ]


def _wav(audio: np.ndarray, rate: int) -> Audio:
    import numpy as np
    from mcp.server.mcpserver import Audio

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(int(audio.shape[1]))
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    return Audio(data=buffer.getvalue(), format="wav")
