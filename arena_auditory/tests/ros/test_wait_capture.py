"""CaptureWaiter built in-process: audio coverage on sim time, frame validation, episode endings, the watchdog and the reference sound."""

from __future__ import annotations

import pytest

NS = 1_000_000_000
START_NS = 40 * NS
FS = 16000


@pytest.fixture
def waiter(rclpy_context):
    for module in ("arena_auditory_msgs.msg", "arena_robots_msgs.msg", "task_generator_msgs.msg", "task_generator_msgs.action"):
        pytest.importorskip(module)
    from arena_auditory.dataset.wait_capture import CaptureWaiter

    nodes = []

    def make(duration: float = 1.0, wall_timeout: float = 60.0, **kwargs: object):
        node = CaptureWaiter(duration, wall_timeout, "/arena/env_0", **kwargs)
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()


def _fleet(*names: str):
    from task_generator_msgs.msg import RobotFleet, RobotState

    fleet = RobotFleet()
    for name in names:
        state = RobotState()
        state.descriptor.name = name
        fleet.robots.append(state)
    return fleet


def _episode(state: int, info: str = ""):
    from task_generator_msgs.msg import EpisodeRecord

    msg = EpisodeRecord()
    msg.outcome_state = state
    msg.outcome_info = info
    msg.start_time.sec = START_NS // NS
    return msg


def _running():
    from task_generator_msgs.msg import EpisodeRecord

    return _episode(EpisodeRecord.RUNNING)


def _frame(stamp_ns: int, *, frames: int = 1600, channels: int = 4, fs: int = FS, encoding: str = "32FC1", interleaved: bool = True, samples: int | None = None):
    from arena_robots_msgs.msg import AudioFrame

    msg = AudioFrame()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, NS)
    msg.encoding = encoding
    msg.interleaved = interleaved
    msg.channel_count = channels
    msg.frame_count = frames
    msg.sample_rate = fs
    msg.data = [0.0] * (frames * channels if samples is None else samples)
    return msg


def _cover(node, robot: str, end_ns: int) -> None:
    """Both streams of ``robot`` reach ``end_ns`` with a chunk of 0.1 s."""
    coverage = node.robots[robot]
    node._raw(coverage, _frame(end_ns - NS // 10))
    node._rendered(coverage, _frame(end_ns - NS // 10))


def test_coverage_completes_only_when_both_streams_reach_the_requested_end(waiter) -> None:
    node = waiter(duration=1.0)
    node._fleet(_fleet("r0"))
    node._episode(_running())
    coverage = node.robots["r0"]
    node._raw(coverage, _frame(START_NS + 9 * NS // 10))
    assert coverage.raw_end_ns == START_NS + NS
    assert not node.coverage_complete
    node._rendered(coverage, _frame(START_NS + 8 * NS // 10))
    assert not node.coverage_complete
    assert not node.done
    node._rendered(coverage, _frame(START_NS + 9 * NS // 10))
    assert node.coverage_complete
    assert node.done
    assert (coverage.raw_chunks, coverage.rendered_chunks) == (1, 2)


def test_coverage_needs_the_episode_start_and_every_robot(waiter) -> None:
    node = waiter(duration=1.0)
    node._fleet(_fleet("r0", "r1"))
    assert not node.coverage_complete
    node._episode(_running())
    assert node.start_ns == START_NS
    _cover(node, "r0", START_NS + NS)
    assert not node.coverage_complete
    _cover(node, "r1", START_NS + NS)
    assert node.coverage_complete


def test_coverage_does_not_shrink_on_an_older_chunk(waiter) -> None:
    node = waiter(duration=1.0)
    node._fleet(_fleet("r0"))
    coverage = node.robots["r0"]
    node._raw(coverage, _frame(START_NS))
    node._raw(coverage, _frame(START_NS - NS))
    assert coverage.raw_end_ns == START_NS + NS // 10


def test_fleet_skips_blank_and_repeated_names_and_derives_the_stream_topics(waiter) -> None:
    node = waiter()
    node._fleet(_fleet("r0", "  ", "r0"))
    first = node.robots["r0"]
    node._fleet(_fleet("r0", "r1"))
    assert list(node.robots) == ["r0", "r1"]
    assert node.robots["r0"] is first
    assert first.raw_topic.startswith("/arena/env_0/")
    assert first.raw_topic != first.rendered_topic


@pytest.mark.parametrize(
    ("given", "match"),
    [
        ({"encoding": "16SC1"}, "invalid encoding"),
        ({"interleaved": False}, "invalid encoding"),
        ({"channels": 0, "samples": 0}, "invalid encoding"),
        ({"frames": 0, "samples": 0}, "invalid encoding"),
        ({"fs": 0}, "invalid encoding"),
        ({"samples": 7}, "does not match frame_count"),
    ],
)
def test_an_invalid_frame_stops_the_capture_with_an_error(waiter, given: dict, match: str) -> None:
    node = waiter()
    node._fleet(_fleet("r0"))
    node._raw(node.robots["r0"], _frame(START_NS, **given))
    assert node.done
    assert match in node.error


@pytest.mark.parametrize("state", ["SUCCESS", "FAILED", "SKIPPED", "FATAL"])
def test_episode_ending_before_coverage_is_an_error(waiter, state: str) -> None:
    from task_generator_msgs.msg import EpisodeRecord

    node = waiter()
    node._fleet(_fleet("r0"))
    node._episode(_running())
    value = getattr(EpisodeRecord, state)
    node._episode(_episode(value, "over"))
    assert node.done
    assert node.terminal_event_state == value
    assert "episode ended before audio coverage completed" in node.error


def test_episode_ending_after_coverage_is_not_an_error(waiter) -> None:
    from task_generator_msgs.msg import EpisodeRecord

    node = waiter(duration=1.0)
    node._fleet(_fleet("r0"))
    node._episode(_running())
    _cover(node, "r0", START_NS + NS)
    node._episode(_episode(EpisodeRecord.SUCCESS, "reached goal"))
    assert node.error is None
    assert node.terminal_event_info == "reached goal"


def test_action_capture_finishes_only_with_both_the_result_and_the_terminal_event(waiter) -> None:
    from task_generator_msgs.msg import EpisodeRecord

    node = waiter(duration=1.0, run_episode_action="/arena/env_0/run_episode_unserved")
    node._fleet(_fleet("r0"))
    node._episode(_running())
    _cover(node, "r0", START_NS + NS)
    assert node.coverage_complete
    assert not node.done
    node._episode(_episode(EpisodeRecord.SUCCESS))
    assert not node.done
    node.action_result_state = int(EpisodeRecord.SUCCESS)
    node._maybe_finish()
    assert node.done


def test_watchdog_stops_the_wait_at_the_wall_deadline(waiter) -> None:
    patient = waiter(wall_timeout=60.0)
    patient._watchdog()
    assert not patient.done
    expired = waiter(wall_timeout=0.0)
    expired._watchdog()
    assert expired.done
    assert "wall-time watchdog expired" in expired.error


def test_reference_sound_is_injected_once_per_robot_after_the_episode_starts(waiter) -> None:
    node = waiter(inject_reference_sound=True)
    node._fleet(_fleet("r0", "r1"))
    assert node.injected_reference_events == 0
    node._episode(_running())
    assert node.injected_reference_events == 2
    assert all(coverage.injected_reference for coverage in node.robots.values())
    node._publish_reference_sound()
    assert node.injected_reference_events == 2


def test_reference_sound_stays_off_unless_requested(waiter) -> None:
    node = waiter()
    node._fleet(_fleet("r0"))
    node._episode(_running())
    assert node.sound_publisher is None
    assert node.injected_reference_events == 0
