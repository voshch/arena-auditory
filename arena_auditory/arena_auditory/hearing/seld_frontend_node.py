"""Live SELDnet front-end of every fleet robot: ``AudioFrame`` stream -> ``AuditoryDetection`` at label-frame rate.

Subscribes to each robot's ``<robot>/audio/raw_array`` from the array renderer,
runs SALSA-Lite + SELDnet over a sliding 5 s window once per 100 ms label frame,
and publishes one detection per active class on
``<robot>/hearing/seld/detections``. The network only reads the array its
weights were trained on (``array`` in weights.yaml): a stream whose sample
rate, channel names or microphone positions differ is refused with an error.
Azimuths are CCW from +x of the audio frame's ``frame_id``.
``seld.bearing_source`` picks where the azimuth comes from: ``gcc`` (default)
fits the array geometry to GCC-PHAT pair delays of the detection frame,
``seld`` takes the model's azimuth. The checkpoint's DOA head is front-biased
(sources beside the robot come back near 0 deg), the fit is not, and with two
simultaneous sources the fit follows the louder one. The level is the frame
RMS through the array's MEMS sensitivity, the confidence the class activity.
"""

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

from arena_auditory.assets import SoundLibrary
from arena_auditory.constants import ArrayStream, array_stream, detections, hearing_heartbeat
from arena_auditory.hearing import weights
from arena_auditory.hearing.doa import ArrayBearing, frame_positions
from arena_auditory.hearing.fleet import HEARTBEAT_TYPE, REFUSAL_LOG_EVERY, FleetRobots, heartbeat
from arena_auditory.hearing.seld import SeldFrontend, SeldStream
from arena_auditory.hearing.timeline import AudioTimeline
from arena_auditory.params import BearingSource, Configuration, Frontend
from arena_auditory.shared import NS_PER_S, ListenerId, RobotBinding, Vec3, dbfs_from_rms, dbfs_to_spl

LAYOUT_TOLERANCE_M = 1e-3


@attrs.define(eq=False)
class _Robot:
    name: str
    listener_id: str
    pub: Publisher
    tick_pub: Publisher
    stream: SeldStream
    timeline: AudioTimeline
    sub: Subscription = attrs.field(init=False)
    layout: tuple[int, int, tuple[str, ...], tuple[Vec3, ...], float] | None = None
    sensitivity_dbfs_at_94_dbspl: float = 0.0
    refusal: str = ""
    doa: ArrayBearing | None = None
    frame_id: str = ""
    n_frames: int = 0
    n_events: int = 0
    bad_frames: int = 0
    refused: int = 0
    n_no_bearing: int = 0


class SeldFrontendNode(ArenaMixinNode):
    def __init__(self) -> None:
        super().__init__("seld_frontend")
        self.conf = Configuration(self)
        self._audio_qos = qos.reliable(10) if self.conf.Audio.RELIABLE_ENABLED.value else qos.best_effort(10)
        s = self.conf.Seld
        files = weights.ensure()
        self._fe = SeldFrontend(
            checkpoint=s.CHECKPOINT.value or files["checkpoint"],
            scaler=s.SCALER.value or files["scaler"],
            classes=weights.classes(SoundLibrary.default()),
            device=s.DEVICE.value,
            det_threshold=s.DET_THRESHOLD.value,
            torch_threads=s.TORCH_THREADS.value,
        )
        if s.DEVICE.value != "cpu" and self._fe.device.type == "cpu":
            self.get_logger().warning(f"torch device {s.DEVICE.value!r} unavailable, running SELDnet on the CPU")
        self._array = weights.array_spec()
        if self._array.channels != self._fe.nb_raw_ch:
            raise ValueError(f"weights.yaml array {self._array.name!r} has {self._array.channels} microphones, the SELDnet model reads {self._fe.nb_raw_ch}")
        self._lookahead_frames = s.LOOKAHEAD_FRAMES.value
        self._gcc = s.BEARING_SOURCE.value is BearingSource.GCC
        self._tg = self.conf.Hearing.TG_NODE.value
        self._robots = FleetRobots(self, self._tg, start=self._start, stop=self._stop, channel=self._channel)
        self.create_timer(self.conf.Diagnostics.PERIOD_S.value, self._diag)
        self.get_logger().info(f"seld_frontend up on {self._fe.device}: {self._array.name} array, window {self._fe.window_samples / self._fe.fs:.1f} s, lookahead {self._lookahead_frames} frames")

    def _start(self, binding: RobotBinding) -> _Robot:
        robot = _Robot(
            name=binding.name,
            listener_id=ListenerId.robot(binding.name),
            pub=self.create_publisher(AuditoryDetection, f"{self._tg}/{detections(binding.name, Frontend.SELD)}", qos.reliable(50)),
            tick_pub=self.create_publisher(LockstepHeartbeat, f"{self._tg}/{hearing_heartbeat(binding.name)}", qos.reliable(10)),
            stream=SeldStream(self._fe, lookahead_frames=self._lookahead_frames),
            timeline=AudioTimeline(self._fe.fs, max_gap_samples=self._fe.window_samples),
        )
        robot.sub = self.create_subscription(AudioFrame, f"{self._tg}/{array_stream(binding.name, ArrayStream.RAW)}", functools.partial(self._cb_audio, robot), self._audio_qos)
        self.get_logger().info(f"seld_frontend: audio {robot.sub.topic_name!r} -> {robot.pub.topic_name!r}")
        return robot

    def _stop(self, robot: _Robot) -> None:
        self.destroy_subscription(robot.sub)
        self.destroy_publisher(robot.pub)
        self.destroy_publisher(robot.tick_pub)

    def _channel(self, robot: _Robot) -> LockstepChannel:
        return LockstepChannel(name=f"hearing/{robot.name}", topic=robot.tick_pub.topic_name, type=HEARTBEAT_TYPE, period_s=self._fe.label_hop_len / self._fe.fs, hard=True)

    def _refusal(self, fs: int, channels: int, names: tuple[str, ...], positions: tuple[Vec3, ...], sensitivity_dbfs_at_94_dbspl: float) -> str:
        """Why the SELDnet weights cannot read a stream of this layout, empty when they can."""
        array = self._array
        if sensitivity_dbfs_at_94_dbspl == 0.0:
            return "frame declares no microphone sensitivity"
        if fs != self._fe.fs:
            return f"{fs} Hz, the SELDnet weights need {self._fe.fs} Hz"
        if channels != array.channels or names != array.channel_names:
            return f"channels {list(names)}, the SELDnet weights need {list(array.channel_names)} of the {array.name} array"
        expected = [mic.position_m for mic in array.mics]
        if len(positions) != len(expected) or not np.allclose(positions, expected, atol=LAYOUT_TOLERANCE_M):
            return f"microphone positions {[tuple(round(c, 3) for c in p) for p in positions]} differ from the {array.name} array the SELDnet weights need"
        return ""

    def _cb_audio(self, robot: _Robot, msg: AudioFrame) -> None:
        fs, ch, n = int(msg.sample_rate), int(msg.channel_count), int(msg.frame_count)
        data = np.asarray(msg.data, dtype=np.float32)
        if fs <= 0 or data.size != ch * n:
            robot.bad_frames += 1
            return
        robot.frame_id = msg.header.frame_id
        stamp_ns = msg.header.stamp.sec * NS_PER_S + msg.header.stamp.nanosec
        layout = (fs, ch, tuple(msg.channel_names), frame_positions(msg), float(msg.sensitivity_dbfs_at_94_dbspl))
        if layout != robot.layout:
            robot.layout, robot.refused = layout, 0
            robot.refusal = self._refusal(*layout)
            robot.sensitivity_dbfs_at_94_dbspl = layout[4]
            robot.doa = ArrayBearing(fs, layout[3]) if self._gcc and not robot.refusal else None
            if not robot.refusal:
                self.get_logger().info(f"seld_frontend {robot.name}: {self._array.name} array at {fs} Hz")
        if robot.refusal:
            robot.refused += 1
            if robot.refused % REFUSAL_LOG_EVERY == 1:
                self.get_logger().error(f"seld_frontend {robot.name}: {robot.refusal}, {robot.refused} frames refused")
            robot.tick_pub.publish(heartbeat(stamp_ns + n * NS_PER_S // fs, robot.frame_id))
            return
        block = data.reshape(n, ch) if msg.interleaved else data.reshape(ch, n).T
        fill = robot.timeline.observe(stamp_ns, n)
        if fill:
            robot.stream.push(np.zeros((fill, ch), dtype=np.float32))
        robot.stream.push(block)
        while robot.stream.ready():
            self._step(robot)
        robot.tick_pub.publish(heartbeat(robot.timeline.time_ns(robot.timeline.samples), robot.frame_id))

    def _step(self, robot: _Robot) -> None:
        dets, end, seg = robot.stream.step()
        robot.n_frames += 1
        if not dets:
            return
        fe = self._fe
        center_ns = robot.timeline.time_ns(end) - fe.label_hop_len * (NS_PER_S // 2) // fe.fs
        rms = float(np.sqrt(np.mean(seg[-fe.label_hop_len :] ** 2))) if seg.size else 0.0
        level_db = dbfs_to_spl(dbfs_from_rms(rms), robot.sensitivity_dbfs_at_94_dbspl)
        fitted, valid = None, True
        if robot.doa is not None:
            fitted, _residual, valid = robot.doa.bearing(seg)
            if not valid:
                robot.n_no_bearing += 1
        for d in dets:
            msg = AuditoryDetection()
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(center_ns, NS_PER_S)
            msg.header.frame_id = robot.frame_id
            msg.robot = robot.name
            msg.listener_id = robot.listener_id
            msg.frontend = Frontend.SELD.value
            msg.kind = d.kind
            msg.event_id = f"seld:{robot.n_frames}:{d.kind}"
            if fitted is None:
                msg.azimuth_rad = float(d.azimuth_rad)
            else:
                msg.azimuth_rad = fitted if valid else math.nan
            msg.elevation_rad = float(d.elevation_rad)
            msg.level_db = float(level_db)
            msg.confidence = float(min(max(d.activity, 0.0), 1.0))
            robot.pub.publish(msg)
            robot.n_events += 1

    def _diag(self) -> None:
        for robot in self._robots:
            self.get_logger().debug(
                f"{robot.name}: frames {robot.n_frames} detections {robot.n_events} audio samples {robot.stream.samples_seen} gaps {robot.timeline.gaps} rewinds {robot.timeline.rewinds} "
                f"bad frames {robot.bad_frames} refused {robot.refused} no-bearing fits {robot.n_no_bearing}"
            )


def main() -> None:
    SeldFrontendNode.run_main()
