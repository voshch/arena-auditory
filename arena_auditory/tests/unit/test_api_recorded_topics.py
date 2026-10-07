from __future__ import annotations

import importlib

import pytest

from arena_auditory.api import recorded_topics

_AUDIO = (True, False, False, False, 1000, True)
_ROWS = [
    ("audio_raw", "{ns}/audio/raw_array", "arena_robots_msgs/msg/AudioFrame", *_AUDIO),
    ("audio_stem_motor", "{ns}/audio/stem_motor", "arena_robots_msgs/msg/AudioFrame", *_AUDIO),
    ("audio_stem_pedestrian", "{ns}/audio/stem_pedestrian", "arena_robots_msgs/msg/AudioFrame", *_AUDIO),
    ("audio_stem_ambient", "{ns}/audio/stem_ambient", "arena_robots_msgs/msg/AudioFrame", *_AUDIO),
    ("audio_rendered", "{ns}/audio/headphones/stereo", "arena_robots_msgs/msg/AudioFrame", *_AUDIO),
    ("audio_render_inputs", "{ns}/audio/diagnostics/render_inputs", "std_msgs/msg/String", *_AUDIO),
    ("heard_sound_events", "{tg}/heard_sound_events", "arena_auditory_msgs/msg/HeardSoundEvent", False, False, False, True, 0, True),
    ("continuous_heard_sounds", "{tg}/continuous_heard_sounds", "arena_auditory_msgs/msg/ContinuousHeardSoundState", False, False, False, False, 1000, True),
    ("room_impulses", "{tg}/acoustic/impulses", "arena_auditory_msgs/msg/RoomImpulse", False, False, True, True, 0, False),
]


def test_recorded_topics_declares_the_nine_keys_in_order() -> None:
    assert [row[0] for row in recorded_topics()] == [row[0] for row in _ROWS]


@pytest.mark.parametrize("row", _ROWS, ids=[row[0] for row in _ROWS])
def test_recorded_topic_row_keeps_its_topic_type_and_qos(row: tuple) -> None:
    assert {declared[0]: declared for declared in recorded_topics()}[row[0]] == row


@pytest.mark.parametrize("msg_type", sorted({row[2] for row in _ROWS}))
def test_recorded_topic_type_imports(msg_type: str) -> None:
    package, _, name = msg_type.split("/")
    assert isinstance(getattr(importlib.import_module(f"{package}.msg"), name), type)
