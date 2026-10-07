"""Topic and service names of the auditory stack. Names resolve below the task generator node, ARENA_PEDS and PEDESTRIAN_MARKERS_EXTRA below the env namespace."""

from __future__ import annotations

SOUND_EVENTS = "sound_events"
HEARD_SOUND_EVENTS = "heard_sound_events"
CONTINUOUS_AUDIO_SOURCES = "continuous_audio_sources"
CONTINUOUS_HEARD_SOUNDS = "continuous_heard_sounds"
ROOM_IMPULSES = "acoustic/impulses"
MICROPHONE_LISTENERS = "microphone_listeners"
MICROPHONE_MARKERS = "microphone_markers"
ENVIRONMENT_SOURCE_MARKERS = "environment_audio_source_markers"
PEDESTRIAN_PROPAGATION_MARKERS = "pedestrian_sound_propagation_markers"
ROBOT_PROPAGATION_MARKERS = "robot_sound_propagation_markers"
ROOM_MARKERS = "acoustic_room_markers"
LISTENER_MONITOR = "audio/listener/monitor"
SPAWN_MICROPHONE = "runtime/spawn_microphone"
REMOVE_MICROPHONE = "runtime/remove_microphone"
STATE_WORLD = "state/world"
STATE_EPISODE = "state/episode"
STATE_ROBOTS = "state/robots"
MAP = "map"

VIEWPORT_CAMERA_POSE = "/arena/viewport/camera_pose"

ARENA_PEDS = "arena_peds"
PEDESTRIAN_MARKERS_EXTRA = "pedestrian_markers/extra"
BUS_FRONTEND = "bus"


def heard_sound(robot: str) -> str:
    return f"{robot}/heard_sound"


def heard_sound_marker(robot: str) -> str:
    return f"{robot}/heard_sound_marker"


def motor_markers(robot: str) -> str:
    return f"{robot}/motor_sound_markers"


def detections(robot: str, frontend: str) -> str:
    return f"{robot}/hearing/{frontend}/detections"


def env_topic(namespace: str, name: str) -> str:
    """name below the env namespace, relative when the namespace is empty."""
    namespace = namespace.strip().rstrip("/")
    return f"{namespace}/{name}" if namespace else name
