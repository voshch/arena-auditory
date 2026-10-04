"""Topic and service names of the auditory stack. Names resolve below the task generator node, BELIEF_GRID to COSTMAP_FILTER_INFO below <tg>/<robot>, ARENA_PEDS and PEDESTRIAN_MARKERS_EXTRA below the env namespace."""

from __future__ import annotations

import enum

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
SPAWN_SOUND = "runtime/spawn_sound"
REMOVE_SOUND = "runtime/remove_sound"
STATE_WORLD = "state/world"
STATE_EPISODE = "state/episode"
STATE_ROBOTS = "state/robots"
STATE_RESETTING = "state/resetting"
MAP = "map"

VIEWPORT_CAMERA_POSE = "/arena/viewport/camera_pose"

ARENA_PEDS = "arena_peds"
PEDESTRIAN_MARKERS_EXTRA = "pedestrian_markers/extra"
BELIEF_GRID = "hearing/belief_grid"
BELIEF_WEDGES = "hearing/belief_wedges"
SPEED_FILTER_MASK = "hearing/speed_filter_mask"
POLICY_STATE = "hearing/policy_state"
POLICY_MARKERS = "hearing/policy_markers"
COSTMAP_FILTER_INFO = "hearing/costmap_filter_info"


class ArrayStream(enum.StrEnum):
    RAW = "raw_array"
    STEM_MOTOR = "stem_motor"
    STEM_PEDESTRIAN = "stem_pedestrian"
    STEM_AMBIENT = "stem_ambient"
    MONITOR = "headphones/stereo"
    HEARING_MONO = "hearing/mono"
    ENERGY = "hearing/energy"
    TDOA = "diagnostics/tdoa"
    RENDER_INPUTS = "diagnostics/render_inputs"
    ACTIVITY = "rendered_sound_activity"
    LEVELS = "diagnostics/levels"


def array_stream(robot: str, stream: ArrayStream) -> str:
    return f"{robot}/audio/{stream}"


def heard_sound(robot: str) -> str:
    return f"{robot}/heard_sound"


def heard_sound_marker(robot: str) -> str:
    return f"{robot}/heard_sound_marker"


def motor_markers(robot: str) -> str:
    return f"{robot}/motor_sound_markers"


def plan(robot: str) -> str:
    return f"{robot}/plan"


def detections(robot: str, frontend: str) -> str:
    return f"{robot}/hearing/{frontend}/detections"


def hearing_heartbeat(robot: str) -> str:
    return f"{robot}/lockstep/hearing"


def env_topic(namespace: str, name: str) -> str:
    """name below the env namespace, relative when the namespace is empty."""
    namespace = namespace.strip().rstrip("/")
    return f"{namespace}/{name}" if namespace else name
