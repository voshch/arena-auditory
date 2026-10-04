from __future__ import annotations

import pytest
from arena_auditory.world import parse_robot_microphones


def test_robot_microphones_require_robot_and_keep_stable_indices() -> None:
    specs = parse_robot_microphones(
        """
        - owner: robot
          robot: jackal_1
          placement: front
          frame: microphone_link
          index: 1
        - owner: robot
          robot: jackal_1
          placement: front
          frame: microphone_link_2
          index: 2
        """
    )

    assert [spec.listener_id for spec in specs] == [
        "microphone:robot:jackal_1:front:1",
        "microphone:robot:jackal_1:front:2",
    ]
    assert specs[0].resolve_frame("env_0/jackal_1") == "env_0/jackal_1/microphone_link"
    assert specs[0].resolve_frame("env_0/jackal_1/") == "env_0/jackal_1/microphone_link"


def test_robot_microphone_frame_already_below_the_prefix_is_kept() -> None:
    (spec,) = parse_robot_microphones("[{robot: jackal_1, placement: Rear, frame: /env_0/jackal_1/mic}]")

    assert spec.placement == "rear"
    assert spec.index == 1
    assert spec.resolve_frame("env_0/jackal_1") == "env_0/jackal_1/mic"


def test_robot_microphones_reject_missing_robot() -> None:
    with pytest.raises(ValueError, match="robot must be a string"):
        parse_robot_microphones("[{owner: robot, placement: front, frame: microphone_link}]")


def test_robot_microphones_reject_duplicate_indexed_id() -> None:
    with pytest.raises(ValueError, match="duplicate microphone"):
        parse_robot_microphones(
            """
            - {robot: jackal_1, placement: front, frame: mic_a, index: 1}
            - {robot: jackal_1, placement: front, frame: mic_b, index: 1}
            """
        )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("[{owner: zone, robot: jackal_1, placement: front, frame: mic}]", "owner: robot"),
        ("[{robot: jackal_1, placement: front}]", "requires a TF frame"),
        ("[{robot: jackal_1, placement: front, frame: mic, index: 0}]", "positive integer"),
        ("{robot: jackal_1}", "YAML list"),
    ],
)
def test_robot_microphones_reject_malformed_entries(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_robot_microphones(raw)


def test_empty_microphones_param_yields_none() -> None:
    assert parse_robot_microphones("") == ()
    assert parse_robot_microphones("[]") == ()
