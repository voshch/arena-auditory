"""Wait for a requested amount of timestamped audio in ROS simulation time."""

from __future__ import annotations

import argparse
import functools
import json
import time
from dataclasses import dataclass
from typing import Any

import rclpy
from arena_auditory_msgs.msg import HeardSoundEvent
from arena_robots.audio import ArrayStream, array_stream
from arena_robots_msgs.msg import AudioFrame
from geometry_msgs.msg import Point
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.task import Future
from rosgraph_msgs.msg import Clock
from task_generator_msgs.action import RunEpisode
from task_generator_msgs.msg import EpisodeRecord, RobotFleet

from arena_auditory.constants import HEARD_SOUND_EVENTS, STATE_ROBOTS
from arena_auditory.shared import ListenerId


@dataclass
class RobotCoverage:
    """Audio seen so far on one robot's raw array and headphone monitor streams."""

    raw_topic: str
    rendered_topic: str
    raw_end_ns: int | None = None
    rendered_end_ns: int | None = None
    raw_chunks: int = 0
    rendered_chunks: int = 0
    injected_reference: bool = False


class CaptureWaiter(Node):
    def __init__(
        self,
        duration: float,
        wall_timeout: float,
        namespace: str,
        *,
        run_episode_action: str | None = None,
        world: str = "",
        inject_reference_sound: bool = False,
    ):
        super().__init__("arena_acoustics_capture_waiter")
        self.duration_ns = round(duration * 1_000_000_000)
        self.wall_deadline = time.monotonic() + wall_timeout
        self.clock_ns: int | None = None
        self.start_ns: int | None = None
        self.robots: dict[str, RobotCoverage] = {}
        self.done = False
        self.error: str | None = None
        self.action_name = run_episode_action
        self.world = world
        self.goal_handle = None
        self.goal_accepted: bool | None = None
        self.cancel_requested = False
        self.action_result_state: int | None = None
        self.action_result_info: str | None = None
        self.action_episode_id: int | None = None
        self.terminal_event_state: int | None = None
        self.terminal_event_info: str | None = None
        self.inject_reference_sound = inject_reference_sound
        self.injected_reference_events = 0
        self.audio_qos = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Clock, "/clock", self._on_clock, self.audio_qos)
        self.prefix = "/" + namespace.strip("/") if namespace.strip("/") else ""
        self.episode_topic = f"{self.prefix}/state/episode"
        self.fleet_topic = f"{self.prefix}/{STATE_ROBOTS}"
        episode_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(EpisodeRecord, self.episode_topic, self._episode, episode_qos)
        self.create_subscription(RobotFleet, self.fleet_topic, self._fleet, QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        sound_qos = QoSProfile(
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.sound_publisher = self.create_publisher(HeardSoundEvent, f"{self.prefix}/{HEARD_SOUND_EVENTS}", sound_qos) if inject_reference_sound else None
        self.create_timer(2.0, self._publish_reference_sound)
        self.create_timer(0.25, self._watchdog)
        self.action_client = ActionClient(self, RunEpisode, run_episode_action) if run_episode_action else None

    def start_episode(self) -> None:
        if self.action_client is None:
            return
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.error = f"episode action is unavailable: {self.action_name}"
            self.done = True
            return
        goal = RunEpisode.Goal()
        goal.world = self.world
        goal.seed = -1
        self.action_client.send_goal_async(goal).add_done_callback(self._goal_response)

    def _goal_response(self, future: Future[Any]) -> None:
        try:
            self.goal_handle = future.result()
        except Exception as exc:
            self.error = f"episode goal failed: {exc}"
            self.done = True
            return
        self.goal_accepted = bool(self.goal_handle.accepted)
        if not self.goal_accepted:
            self.error = "episode goal was rejected"
            self.done = True
            return
        self.goal_handle.get_result_async().add_done_callback(self._action_result)

    def _action_result(self, future: Future[Any]) -> None:
        try:
            result = future.result().result
            self.action_result_state = int(result.state)
            self.action_result_info = str(result.info)
            self.action_episode_id = int(result.episode_id)
        except Exception as exc:
            self.error = f"episode result failed: {exc}"
        self._maybe_finish()

    def _request_cancel(self) -> None:
        if self.action_client is None or self.cancel_requested or self.goal_handle is None:
            return
        self.cancel_requested = True
        self.goal_handle.cancel_goal_async().add_done_callback(self._cancel_response)

    def _cancel_response(self, future: Future[Any]) -> None:
        try:
            response = future.result()
            if not response.goals_canceling:
                self.error = "episode server did not accept capture-complete cancellation"
                self.done = True
        except Exception as exc:
            self.error = f"episode cancellation failed: {exc}"
            self.done = True

    @staticmethod
    def _stamp(msg: AudioFrame) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _on_clock(self, msg: Clock) -> None:
        self.clock_ns = int(msg.clock.sec) * 1_000_000_000 + int(msg.clock.nanosec)

    def _fleet(self, msg: RobotFleet) -> None:
        for state in msg.robots:
            robot = str(state.descriptor.name).strip()
            if not robot or robot in self.robots:
                continue
            coverage = RobotCoverage(raw_topic=f"{self.prefix}/{array_stream(robot, ArrayStream.RAW)}", rendered_topic=f"{self.prefix}/{array_stream(robot, ArrayStream.MONITOR)}")
            self.robots[robot] = coverage
            self.create_subscription(AudioFrame, coverage.raw_topic, functools.partial(self._raw, coverage), self.audio_qos)
            self.create_subscription(AudioFrame, coverage.rendered_topic, functools.partial(self._rendered, coverage), self.audio_qos)
        self._publish_reference_sound()

    def _raw(self, coverage: RobotCoverage, msg: AudioFrame) -> None:
        coverage.raw_chunks += 1
        chunk_end_ns = self._chunk_end(msg)
        coverage.raw_end_ns = chunk_end_ns if coverage.raw_end_ns is None else max(coverage.raw_end_ns, chunk_end_ns)
        self._maybe_finish()

    def _rendered(self, coverage: RobotCoverage, msg: AudioFrame) -> None:
        coverage.rendered_chunks += 1
        chunk_end_ns = self._chunk_end(msg)
        coverage.rendered_end_ns = chunk_end_ns if coverage.rendered_end_ns is None else max(coverage.rendered_end_ns, chunk_end_ns)
        self._maybe_finish()

    def _episode(self, msg: EpisodeRecord) -> None:
        state = int(msg.outcome_state)
        if state == int(EpisodeRecord.RUNNING):
            self.start_ns = int(msg.start_time.sec) * 1_000_000_000 + int(msg.start_time.nanosec)
            self._publish_reference_sound()
        elif state in (
            int(EpisodeRecord.SUCCESS),
            int(EpisodeRecord.FAILED),
            int(EpisodeRecord.SKIPPED),
            int(EpisodeRecord.FATAL),
        ):
            self.terminal_event_state = state
            self.terminal_event_info = str(msg.outcome_info)
            if not self.coverage_complete:
                self.error = f"episode ended before audio coverage completed: state={state} info={msg.outcome_info!r}"
                self.done = True
        self._maybe_finish()

    def _publish_reference_sound(self) -> None:
        """Inject one deterministic calibration greeting through the normal propagation and render path."""
        if self.sound_publisher is None or self.start_ns is None or self.coverage_complete:
            return
        for robot, coverage in self.robots.items():
            if coverage.injected_reference:
                continue
            stamp = self.get_clock().now().to_msg()
            event_id = f"dataset-reference:{self.injected_reference_events}:{stamp.sec}:{stamp.nanosec}"
            for index, channel in enumerate(("front_left", "front_right", "rear_left", "rear_right")):
                msg = HeardSoundEvent()
                msg.header.stamp = stamp
                msg.header.frame_id = "map"
                msg.source.id = event_id
                msg.source.kind = "speech"
                msg.source.asset_id = "greeting"
                msg.source.model = "wav"
                msg.source.agent_kind = "external"
                msg.source.agent_id = -10_001
                msg.source.agent_name = "dataset_reference_source"
                msg.source.position = Point(x=0.0, y=0.0, z=1.6)
                msg.source.level_db = 85.0
                msg.source.reference_distance_m = 1.0
                msg.source.active = True
                msg.reception.listener_id = ListenerId.array_mic(robot, channel)
                msg.reception.listener_position = Point(x=1.5, y=4.7, z=0.3)
                msg.reception.distance_m = 4.93
                msg.reception.bearing_rad = -1.88
                msg.reception.received_level_db = 74.0 - 0.25 * index
                msg.reception.threshold_db = 20.0
                msg.reception.direct_delay_s = 0.014 + 0.0001 * index
                msg.reception.audible = True
                msg.reception.backend = "dataset_reference"
                self.sound_publisher.publish(msg)
            coverage.injected_reference = True
            self.injected_reference_events += 1

    def _chunk_end(self, msg: AudioFrame) -> int:
        channels = int(msg.channel_count)
        frames = int(msg.frame_count)
        if str(msg.encoding) != "32FC1" or not bool(msg.interleaved) or channels <= 0 or frames <= 0 or int(msg.sample_rate) <= 0:
            self.error = "audio stream contains an invalid encoding, channel count, or sample rate"
            self.done = True
            return self._stamp(msg)
        if len(msg.data) != frames * channels:
            self.error = "AudioFrame.data length does not match frame_count * channel_count"
            self.done = True
            return self._stamp(msg)
        return self._stamp(msg) + round(frames * 1_000_000_000 / int(msg.sample_rate))

    def _maybe_finish(self) -> None:
        if not self.coverage_complete:
            return
        if self.action_client is None:
            self.done = True
            return
        if self.action_result_state is None and self.terminal_event_state is None:
            self._request_cancel()
        if self.action_result_state is not None and self.terminal_event_state is not None:
            self.done = True

    @property
    def coverage_complete(self) -> bool:
        if self.start_ns is None or not self.robots:
            return False
        requested_end_ns = self.start_ns + self.duration_ns
        return all(coverage.raw_end_ns is not None and coverage.rendered_end_ns is not None and coverage.raw_end_ns >= requested_end_ns and coverage.rendered_end_ns >= requested_end_ns for coverage in self.robots.values())

    def _watchdog(self) -> None:
        if time.monotonic() >= self.wall_deadline:
            self.error = "wall-time watchdog expired before the requested simulation-time audio duration"
            self.done = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--wall-timeout", type=float, required=True)
    parser.add_argument("--namespace", required=True, help="environment namespace, for example /arena/env_0")
    parser.add_argument("--run-episode-action", help="send and cleanly cancel this RunEpisode action after capture")
    parser.add_argument("--world", default="", help="world passed to --run-episode-action")
    parser.add_argument("--inject-reference-sound", action="store_true", help="publish an audible calibration event every two seconds during the episode")
    args = parser.parse_args(argv)
    rclpy.init()
    node = CaptureWaiter(
        args.duration,
        args.wall_timeout,
        args.namespace,
        run_episode_action=args.run_episode_action,
        world=args.world,
        inject_reference_sound=args.inject_reference_sound,
    )
    try:
        node.start_episode()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.25)
        action_complete = node.action_client is None or (node.action_result_state is not None and node.terminal_event_state is not None and node.action_result_state == node.terminal_event_state)
        result = {
            "valid": node.error is None and node.coverage_complete and action_complete,
            "coverage_complete": node.coverage_complete,
            "start_timestamp_ns": node.start_ns,
            "end_clock_ns": node.clock_ns,
            "robots": {
                robot: {
                    "raw_topic": coverage.raw_topic,
                    "rendered_topic": coverage.rendered_topic,
                    "raw_end_timestamp_ns": coverage.raw_end_ns,
                    "rendered_end_timestamp_ns": coverage.rendered_end_ns,
                    "raw_chunks_seen": coverage.raw_chunks,
                    "rendered_chunks_seen": coverage.rendered_chunks,
                }
                for robot, coverage in node.robots.items()
            },
            "injected_reference_events": node.injected_reference_events,
            "episode_topic": node.episode_topic,
            "fleet_topic": node.fleet_topic,
            "goal_accepted": node.goal_accepted,
            "cancel_requested": node.cancel_requested,
            "action_result_state": node.action_result_state,
            "action_result_info": node.action_result_info,
            "action_episode_id": node.action_episode_id,
            "terminal_event_state": node.terminal_event_state,
            "terminal_event_info": node.terminal_event_info,
            "error": node.error,
        }
        print(json.dumps(result))
        return 0 if result["valid"] else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
