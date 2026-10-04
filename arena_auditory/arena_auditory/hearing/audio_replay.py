"""Publish a wav as ``AudioFrame`` blocks on a robot's raw array stream, the renderer's wire format, for front-end tests.

    ros2 run arena_auditory hearing_audio_replay <wav> <robot> [--array four_mic] [--tg task_generator_node] [--block 320] [--speed 1.0]

Every channel of the wav goes out. ``--array`` names the array the recording
was made on (a preset or a yaml path), its channel names and microphone
positions ride on every frame, which the front-ends need. ``--speed 0``
publishes as fast as the subscriber side can take (no pacing). Stamps come
from the node clock at the start of each block.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rclpy
from arena_auditory_msgs.msg import AudioFrame
from geometry_msgs.msg import Point
from rclpy.node import Node

from arena_auditory.constants import ArrayStream, array_stream
from arena_auditory.hearing.seld import load_wav
from arena_auditory.params import HearingGroup
from arena_auditory.shared import ArraySpec, load_array_spec


class AudioReplay(Node):
    def __init__(self, wav: str, topic: str, block: int, speed: float, frame_id: str, loop: bool, array: ArraySpec | None) -> None:
        super().__init__("audio_replay")
        audio, self.fs = load_wav(wav)
        self.audio = np.asarray(audio, dtype=np.float32).reshape(len(audio), -1)
        channels = self.audio.shape[1]
        if array is not None and array.channels != channels:
            raise SystemExit(f"{wav}: {channels} channels, the {array.name} array has {array.channels}")
        self.array = array
        self.block = int(block)
        self.speed = float(speed)
        self.frame_id = frame_id
        self.loop = loop
        self.pub = self.create_publisher(AudioFrame, topic, 64)
        self.cursor = 0
        self.sent = 0
        self.get_logger().info(f"{wav}: {channels} channels at {self.fs} Hz on {self.pub.topic_name!r}, array {array.name if array is not None else 'undeclared'}")

    def run(self) -> None:
        period = self.block / self.fs
        next_at = time.monotonic()
        while rclpy.ok():
            if self.cursor >= len(self.audio):
                if not self.loop:
                    break
                self.cursor = 0
            chunk = self.audio[self.cursor : self.cursor + self.block]
            self.cursor += self.block
            msg = AudioFrame()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.sample_rate = int(self.fs)
            msg.channel_count = int(chunk.shape[1])
            msg.frame_count = int(chunk.shape[0])
            msg.encoding = "32FC1"
            msg.interleaved = True
            if self.array is not None:
                msg.channel_names = list(self.array.channel_names)
                msg.microphone_positions = [Point(x=mic.position_m[0], y=mic.position_m[1], z=mic.position_m[2]) for mic in self.array.mics]
                msg.microphone_yaw_rad = [mic.yaw_rad for mic in self.array.mics]
            msg.data = chunk.reshape(-1).tolist()
            self.pub.publish(msg)
            self.sent += 1
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.speed > 0.0:
                next_at += period / self.speed
                delay = next_at - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
        self.get_logger().info(f"published {self.sent} blocks ({self.sent * self.block / self.fs:.1f} s)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("robot", help="fleet robot whose raw array stream to publish")
    ap.add_argument("--array", default="", help="array preset or yaml path the recording was made on, empty declares no geometry")
    ap.add_argument("--tg", default=HearingGroup.TG_NODE.default, help="task generator node the robot topics live below")
    ap.add_argument("--block", type=int, default=320)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--frame-id", default="base_link")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--settle", type=float, default=1.0, help="seconds to wait for subscribers before the first block")
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])
    array = load_array_spec(args.array) if args.array else None
    rclpy.init(args=ros_args)
    node = AudioReplay(args.wav, f"{args.tg}/{array_stream(args.robot, ArrayStream.RAW)}", args.block, args.speed, args.frame_id, args.loop, array)
    try:
        time.sleep(args.settle)
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
