from __future__ import annotations

import math

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("arena_auditory_msgs.msg")


def _pedestrian(ped_id: int, x: float, y: float, *, yaw_rad: float = 0.0, speed_mps: float = 0.0):
    from arena_auditory.sources import PedestrianState

    return PedestrianState(id=ped_id, name=f"ped_{ped_id}", position=(x, y, 0.0), yaw_rad=yaw_rad, speed_mps=speed_mps, gait_phase=0.0)


def _detector(emitted: list[tuple[str, int]]):
    from arena_auditory.params import HumanGroup
    from arena_auditory.sources import PedestrianEventDetector

    return PedestrianEventDetector(lambda kind, ped: emitted.append((kind, ped.id)), **HumanGroup.defaults())


def test_speech_fires_once_per_facing_pair_within_greeting_distance() -> None:
    emitted: list[tuple[str, int]] = []
    detector = _detector(emitted)
    pedestrians = [_pedestrian(3, 5.0, 0.0), _pedestrian(1, 0.0, 0.0), _pedestrian(2, 1.0, 0.0, yaw_rad=math.pi)]
    detector.update(pedestrians, 0.0)
    detector.update(pedestrians, 1.0)
    assert [event for event in emitted if event[0] == "speech"] == [("speech", 1)]


def test_footstep_uses_pose_delta_when_reported_speed_is_zero() -> None:
    emitted: list[tuple[str, int]] = []
    detector = _detector(emitted)
    detector.update([_pedestrian(9, 1.0, 2.0)], 1.0)
    assert emitted == []
    detector.update([_pedestrian(9, 1.2, 2.0)], 1.5)
    assert emitted == [("footstep", 9)]


def test_single_pedestrian_never_speaks() -> None:
    emitted: list[tuple[str, int]] = []
    detector = _detector(emitted)
    for now_s in range(20):
        detector.update([_pedestrian(9, 1.0, 2.0)], float(now_s))
    assert all(kind != "speech" for kind, _ in emitted)
