"""Untrained hearing front-end of every fleet robot: energy-onset detection plus GCC-PHAT bearing over the array each AudioFrame declares, drop-in sibling of seld_frontend_node."""

from __future__ import annotations

import functools
import math

import attrs
import numpy as np
from arena_auditory_msgs.msg import AudioFrame, AuditoryDetection
from arena_rclpy_mixins import ArenaMixinNode, qos
from arena_runtime_msgs.msg import LockstepChannel, LockstepHeartbeat
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from arena_auditory.constants import ArrayStream, array_stream, detections, hearing_heartbeat
from arena_auditory.hearing.doa import ArrayBearing, frame_positions
from arena_auditory.hearing.fleet import HEARTBEAT_TYPE, REFUSAL_LOG_EVERY, FleetRobots, heartbeat
from arena_auditory.hearing.onset import OnsetDetector
from arena_auditory.hearing.timeline import AudioTimeline
from arena_auditory.params import Configuration, Frontend
from arena_auditory.shared import NS_PER_S, ListenerId, RobotBinding, Vec3, dbfs_from_rms, dbfs_to_spl

ONSET_KIND = "onset"


def onset_confidence(margin_db: float) -> float:
    """Squashes the onset margin over the noise floor into [0, 1), 6 dB reads 0.5."""
    return 1.0 - 10.0 ** (-max(margin_db, 0.0) / 20.0)


@attrs.define(eq=False)
class _Pipeline:
    """Onset detector, bearing fit and time base of one stream layout."""

    fs: int
    hop_samples: int
    onset: OnsetDetector
    doa: ArrayBearing
    timeline: AudioTimeline
    buf: np.ndarray
    prev_hop: np.ndarray
    sensitivity_dbfs_at_94_dbspl: float
    consumed: int = 0


@attrs.define(eq=False)
class _Robot:
    name: str
    listener_id: str
    pub: Publisher
    tick_pub: Publisher
    sub: Subscription = attrs.field(init=False)
    layout: tuple[int, int, tuple[Vec3, ...], float] | None = None
    pipeline: _Pipeline | None = None
    refusal: str = ""
    frame_id: str = ""
    n_hops: int = 0
    n_events: int = 0
    bad_frames: int = 0
    refused: int = 0
    n_no_bearing: int = 0


class SrpFrontendNode(ArenaMixinNode):
    def __init__(self) -> None:
        super().__init__("srp_frontend")
        self.conf = Configuration(self)
        self._audio_qos = qos.reliable(10) if self.conf.Audio.RELIABLE_ENABLED.value else qos.best_effort(10)
        s = self.conf.Srp
        self._hop_s = s.HOP_S.value
        self._floor_window_s = s.FLOOR_WINDOW_S.value
        self._onset_db = s.ONSET_DB.value
        self._tg = self.conf.Hearing.TG_NODE.value
        self._robots = FleetRobots(self, self._tg, start=self._start, stop=self._stop, channel=self._channel)
        self.create_timer(self.conf.Diagnostics.PERIOD_S.value, self._diag)
        self.get_logger().info(f"srp_frontend up: hop {self._hop_s:.3f} s, floor window {self._floor_window_s:.1f} s, onset {self._onset_db:.1f} dB")

    def _start(self, binding: RobotBinding) -> _Robot:
        robot = _Robot(
            name=binding.name,
            listener_id=ListenerId.robot(binding.name),
            pub=self.create_publisher(AuditoryDetection, f"{self._tg}/{detections(binding.name, Frontend.SRP)}", qos.reliable(50)),
            tick_pub=self.create_publisher(LockstepHeartbeat, f"{self._tg}/{hearing_heartbeat(binding.name)}", qos.reliable(10)),
        )
        robot.sub = self.create_subscription(AudioFrame, f"{self._tg}/{array_stream(binding.name, ArrayStream.RAW)}", functools.partial(self._cb_audio, robot), self._audio_qos)
        self.get_logger().info(f"srp_frontend: audio {robot.sub.topic_name!r} -> {robot.pub.topic_name!r}")
        return robot

    def _stop(self, robot: _Robot) -> None:
        self.destroy_subscription(robot.sub)
        self.destroy_publisher(robot.pub)
        self.destroy_publisher(robot.tick_pub)

    def _channel(self, robot: _Robot) -> LockstepChannel:
        return LockstepChannel(name=f"hearing/{robot.name}", topic=robot.tick_pub.topic_name, type=HEARTBEAT_TYPE, period_s=self._hop_s, hard=True)

    def _pipeline(self, fs: int, channels: int, positions: tuple[Vec3, ...], sensitivity_dbfs_at_94_dbspl: float) -> _Pipeline:
        """Raises ValueError when the layout yields no bearing or no level."""
        if len(positions) != channels:
            raise ValueError(f"frame declares {len(positions)} microphone positions for {channels} channels")
        if sensitivity_dbfs_at_94_dbspl == 0.0:
            raise ValueError("frame declares no microphone sensitivity")
        doa = ArrayBearing(fs, positions)
        hop = max(round(self._hop_s * fs), 1)
        return _Pipeline(
            fs=fs,
            hop_samples=hop,
            onset=OnsetDetector(fs, hop_s=self._hop_s, floor_window_s=self._floor_window_s, onset_db=self._onset_db),
            doa=doa,
            timeline=AudioTimeline(fs, max_gap_samples=hop * 10),
            buf=np.zeros((0, channels), dtype=np.float32),
            prev_hop=np.zeros((hop, channels), dtype=np.float32),
            sensitivity_dbfs_at_94_dbspl=sensitivity_dbfs_at_94_dbspl,
        )

    def _cb_audio(self, robot: _Robot, msg: AudioFrame) -> None:
        fs, ch, n = int(msg.sample_rate), int(msg.channel_count), int(msg.frame_count)
        data = np.asarray(msg.data, dtype=np.float32)
        if fs <= 0 or data.size != ch * n:
            robot.bad_frames += 1
            return
        robot.frame_id = msg.header.frame_id
        stamp_ns = msg.header.stamp.sec * NS_PER_S + msg.header.stamp.nanosec
        layout = (fs, ch, frame_positions(msg), float(msg.sensitivity_dbfs_at_94_dbspl))
        if layout != robot.layout:
            robot.layout, robot.refused = layout, 0
            try:
                robot.pipeline, robot.refusal = self._pipeline(*layout), ""
                self.get_logger().info(f"srp_frontend {robot.name}: {ch} microphones at {fs} Hz, {len(robot.pipeline.doa.pairs)} pairs")
            except ValueError as exc:
                robot.pipeline, robot.refusal = None, str(exc)
        pipe = robot.pipeline
        if pipe is None:
            robot.refused += 1
            if robot.refused % REFUSAL_LOG_EVERY == 1:
                self.get_logger().error(f"srp_frontend {robot.name}: {robot.refusal}, {robot.refused} frames refused")
            robot.tick_pub.publish(heartbeat(stamp_ns + n * NS_PER_S // fs, robot.frame_id))
            return
        block = data.reshape(n, ch) if msg.interleaved else data.reshape(ch, n).T
        fill = pipe.timeline.observe(stamp_ns, n)
        if fill:
            pipe.buf = np.concatenate([pipe.buf, np.zeros((fill, ch), dtype=np.float32)])
        pipe.buf = np.concatenate([pipe.buf, block])
        while pipe.buf.shape[0] >= pipe.hop_samples:
            hop = pipe.buf[: pipe.hop_samples]
            pipe.buf = pipe.buf[pipe.hop_samples :]
            self._step(robot, pipe, hop)
        robot.tick_pub.publish(heartbeat(pipe.timeline.time_ns(pipe.timeline.samples), robot.frame_id))

    def _step(self, robot: _Robot, pipe: _Pipeline, hop: np.ndarray) -> None:
        robot.n_hops += 1
        fired, peak_db, floor_db = pipe.onset.step(hop)
        hop_end = pipe.consumed + hop.shape[0]
        center_ns = pipe.timeline.time_ns(hop_end) - pipe.hop_samples * (NS_PER_S // 2) // pipe.fs
        pipe.consumed = hop_end
        if fired:
            fitted, _residual, valid = pipe.doa.bearing(np.concatenate([pipe.prev_hop, hop]))
            if not valid:
                robot.n_no_bearing += 1
            msg = AuditoryDetection()
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(center_ns, NS_PER_S)
            msg.header.frame_id = robot.frame_id
            msg.robot = robot.name
            msg.listener_id = robot.listener_id
            msg.frontend = Frontend.SRP.value
            msg.kind = ONSET_KIND
            msg.event_id = f"srp:{robot.n_hops}"
            msg.azimuth_rad = fitted if valid else math.nan
            msg.elevation_rad = math.nan
            msg.level_db = dbfs_to_spl(dbfs_from_rms(float(np.sqrt(np.mean(hop**2)))), pipe.sensitivity_dbfs_at_94_dbspl)
            msg.confidence = onset_confidence(peak_db - floor_db)
            robot.pub.publish(msg)
            robot.n_events += 1
        pipe.prev_hop = hop

    def _diag(self) -> None:
        for robot in self._robots:
            gaps = robot.pipeline.timeline.gaps if robot.pipeline is not None else 0
            self.get_logger().debug(f"{robot.name}: hops {robot.n_hops} detections {robot.n_events} bad frames {robot.bad_frames} refused {robot.refused} no-bearing fits {robot.n_no_bearing} gaps {gaps}")


def main() -> None:
    SrpFrontendNode.run_main()
