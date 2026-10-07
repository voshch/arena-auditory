"""Public surface of arena_auditory for the task_generator acoustics backend."""

from __future__ import annotations

from collections.abc import Mapping

from arena_robots.audio import ArrayStream, array_stream

from arena_auditory.constants import (
    CONTINUOUS_AUDIO_SOURCES,
    CONTINUOUS_HEARD_SOUNDS,
    ENVIRONMENT_SOURCE_MARKERS,
    HEARD_SOUND_EVENTS,
    MICROPHONE_MARKERS,
    PEDESTRIAN_PROPAGATION_MARKERS,
    ROBOT_PROPAGATION_MARKERS,
    ROOM_IMPULSES,
    motor_markers,
)
from arena_auditory.shared import INACTIVE_REPEATS, ListenerId, ListenerKind, SourceSpec

__all__ = [
    "CONTINUOUS_AUDIO_SOURCES",
    "CONTINUOUS_HEARD_SOUNDS",
    "ENVIRONMENT_SOURCE_MARKERS",
    "INACTIVE_REPEATS",
    "MICROPHONE_MARKERS",
    "PEDESTRIAN_PROPAGATION_MARKERS",
    "ROBOT_PROPAGATION_MARKERS",
    "ListenerId",
    "ListenerKind",
    "SourceSpec",
    "motor_markers",
    "recorded_topics",
    "rviz_plugins",
]

type RecordedTopicRow = tuple[str, str, str, bool, bool, bool, bool, int, bool]
type RvizPluginRow = tuple[str, str, str, Mapping[str, object]]

_AUDIO_FRAME = "arena_robots_msgs/msg/AudioFrame"
_AUDIO_DEPTH = 1000


def _audio(key: str, stream: ArrayStream, msg_type: str = _AUDIO_FRAME) -> RecordedTopicRow:
    return (key, array_stream("{ns}", stream), msg_type, True, False, False, False, _AUDIO_DEPTH, True)


def recorded_topics() -> tuple[RecordedTopicRow, ...]:
    """Recordable topics as (key, topic template, msg type, robot scoped, throttled, transient local, reliable, depth, recorded)."""
    return (
        _audio("audio_raw", ArrayStream.RAW),
        _audio("audio_stem_motor", ArrayStream.STEM_MOTOR),
        _audio("audio_stem_pedestrian", ArrayStream.STEM_PEDESTRIAN),
        _audio("audio_stem_ambient", ArrayStream.STEM_AMBIENT),
        _audio("audio_rendered", ArrayStream.MONITOR),
        _audio("audio_render_inputs", ArrayStream.RENDER_INPUTS, "std_msgs/msg/String"),
        ("heard_sound_events", f"{{tg}}/{HEARD_SOUND_EVENTS}", "arena_auditory_msgs/msg/HeardSoundEvent", False, False, False, True, 0, True),
        ("continuous_heard_sounds", f"{{tg}}/{CONTINUOUS_HEARD_SOUNDS}", "arena_auditory_msgs/msg/ContinuousHeardSoundState", False, False, False, False, _AUDIO_DEPTH, True),
        ("room_impulses", f"{{tg}}/{ROOM_IMPULSES}", "arena_auditory_msgs/msg/RoomImpulse", False, False, True, True, 0, False),
    )


def rviz_plugins(tg_node: str) -> tuple[RvizPluginRow, ...]:
    """The auditory panel and tools as (role, class name, panel name, properties), each targeting tg_node."""
    return (
        ("panel", "arena_auditory_viz::AuditoryPanel", "AuditoryPanel", {"Target": tg_node}),
        ("tool", "arena_auditory_viz::SpawnMicrophoneTool", "", {"Target": tg_node, "Height": 1.5, "Attach TF Frame": ""}),
        (
            "tool",
            "arena_auditory_viz::SpawnSoundTool",
            "",
            {
                "Target": tg_node,
                "Kind": "music",
                "Height": 1.2,
                "Custom Playback": False,
                "Asset ID": "",
                "Source Volume": 62.0,
                "Loop": True,
                "Start Immediately": True,
            },
        ),
    )
