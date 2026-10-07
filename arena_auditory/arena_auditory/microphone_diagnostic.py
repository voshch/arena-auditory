"""Console diagnostics of a robot's raw array stream: channel levels, GCC-PHAT delays and coarse spatial evidence."""

from __future__ import annotations

import numpy as np
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.qos import latched, reliable
from arena_robots.audio import ArrayStream, array_stream, dbfs_from_rms, gcc_phat, load_array_spec, rms
from arena_robots.fleet import bind_robot
from arena_robots_msgs.msg import AudioFrame
from task_generator_msgs.msg import RobotFleet

from arena_auditory.constants import STATE_ROBOTS
from arena_auditory.params import Configuration
from arena_auditory.render.monitor import tdoa_pairs


def _evidence(first: float, second: float) -> tuple[float, float]:
    total = first + second
    return (0.5, 0.5) if total <= 1e-12 else (first / total, second / total)


class MicrophoneDiagnostic(ArenaMixinNode):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("microphone_diagnostic", **kwargs)
        self.conf = Configuration(self)
        self._array_conf = self.conf.Array
        self._tdoa_conf = self.conf.Tdoa
        self._diagnostics_conf = self.conf.Diagnostics
        self.spec = load_array_spec(self._array_conf.SPEC.value)
        self._pairs = tdoa_pairs(self.spec)
        self._latest: tuple[np.ndarray, int] | None = None
        self._robot = ""
        robot = self._array_conf.ROBOT.value.strip()
        if robot:
            self._listen(robot)
        else:
            self.create_subscription(RobotFleet, STATE_ROBOTS, self._on_fleet, latched(1))
        self.create_timer(self._diagnostics_conf.REPORT_PERIOD_S.value, self._report)

    def _on_fleet(self, msg: RobotFleet) -> None:
        binding = bind_robot(msg)
        if binding is not None and not self._robot:
            self._listen(binding.name)

    def _listen(self, robot: str) -> None:
        self._robot = robot
        topic = array_stream(robot, ArrayStream.RAW)
        self.create_subscription(AudioFrame, topic, self._on_audio, reliable(10))
        self.get_logger().info(f"listening to {topic} as a {self.spec.name} array {self.spec.channel_names}")

    def _on_audio(self, msg: AudioFrame) -> None:
        if msg.channel_count != self.spec.channels or msg.frame_count == 0:
            self.get_logger().warning(f"expected a non-empty {self.spec.channels}-channel AudioFrame, got {msg.channel_count} x {msg.frame_count}", throttle_duration_sec=5.0)
            return
        data = np.asarray(msg.data, dtype=np.float32)
        expected = int(msg.channel_count * msg.frame_count)
        if data.size != expected:
            self.get_logger().warning(f"malformed AudioFrame: {data.size} values, expected {expected}", throttle_duration_sec=5.0)
            return
        self._latest = (data.reshape(msg.frame_count, msg.channel_count).T.copy(), int(msg.sample_rate))

    def _side_energy(self, audio: np.ndarray, *, side: str = "", group: str = "") -> float:
        rows = [index for index, mic in enumerate(self.spec.mics) if (not side or mic.side == side) and (not group or mic.group == group)]
        return float(rms(audio[rows])) if rows else 0.0

    def _report(self) -> None:
        if self._latest is None:
            return
        audio, rate = self._latest
        levels = [dbfs_from_rms(float(value)) for value in np.atleast_1d(rms(audio, axis=1))]
        max_lag_s = self._tdoa_conf.MAX_LAG_S.value
        estimates = [(label, *gcc_phat(audio[first], audio[second], sample_rate_hz=rate, max_tau_s=max_lag_s)) for first, second, label in self._pairs]
        left, right = _evidence(self._side_energy(audio, side="left"), self._side_energy(audio, side="right"))
        front, rear = _evidence(self._side_energy(audio, group="front"), self._side_energy(audio, group="rear"))
        lines = [
            "Source activity detected" if max(levels) >= self._diagnostics_conf.ACTIVITY_THRESHOLD_DBFS.value else "No source activity",
            *(f"{name.replace('_', ' ').upper()}: {level:.1f} dBFS" for name, level in zip(self.spec.channel_names, levels, strict=True)),
            "GCC-PHAT:",
            *(f"  {label}: {delay * 1e6:+.0f} us (confidence {confidence:.3f})" for label, delay, confidence in estimates),
            "coarse spatial evidence (energy heuristic):",
            f"  left: {left:.2f}  right: {right:.2f}",
            f"  front: {front:.2f}  rear: {rear:.2f}",
        ]
        self.get_logger().info("\n" + "\n".join(lines))


def main() -> None:
    MicrophoneDiagnostic.run_main()
