"""Per-robot state of a hearing node, kept in step with the robot fleet, with one hard lockstep channel per robot."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator

from arena_rclpy_mixins import ArenaMixinNode, qos
from arena_runtime.lockstep import register_channels
from arena_runtime_msgs.msg import LockstepChannel, LockstepHeartbeat
from task_generator_msgs.msg import RobotFleet

from arena_auditory.constants import STATE_ROBOTS
from arena_auditory.shared import NS_PER_S, RobotBinding, robot_bindings

HEARTBEAT_TYPE = "arena_runtime_msgs/msg/LockstepHeartbeat"
REFUSAL_LOG_EVERY = 100


def heartbeat(end_ns: int, frame_id: str) -> LockstepHeartbeat:
    """Front-end lockstep heartbeat: the stream is consumed up to end_ns."""
    beat = LockstepHeartbeat()
    beat.header.stamp.sec, beat.header.stamp.nanosec = divmod(end_ns, NS_PER_S)
    beat.header.frame_id = frame_id
    return beat


class FleetRobots[R]:
    """Fleet robots in fleet order, each with the state ``start`` returns. ``stop`` releases a robot that left the fleet."""

    def __init__(self, node: ArenaMixinNode, tg: str, *, start: Callable[[RobotBinding], R], stop: Callable[[R], None], channel: Callable[[R], LockstepChannel]) -> None:
        self._node = node
        self._env = node.resolve_topic_name(tg)
        self._start = start
        self._stop = stop
        self._channel = channel
        self._robots: dict[str, R] = {}
        self._lockstep = bool(node.get_parameter("use_sim_time").value)
        self._lock = asyncio.Lock()
        node.create_subscription(RobotFleet, f"{tg}/{STATE_ROBOTS}", self._cb_fleet, qos.latched())

    def __iter__(self) -> Iterator[R]:
        return iter(tuple(self._robots.values()))

    def _cb_fleet(self, msg: RobotFleet) -> None:
        bindings = {binding.name: binding for binding in robot_bindings(msg)}
        for name in self._robots.keys() - bindings.keys():
            self._stop(self._robots[name])
        robots = {name: self._robots[name] if name in self._robots else self._start(binding) for name, binding in bindings.items()}
        changed = list(robots) != list(self._robots)
        self._robots = robots
        if changed and self._lockstep:
            asyncio.run_coroutine_threadsafe(self._register([self._channel(robot) for robot in robots.values()]), self._node.event_loop)

    async def _register(self, channels: list[LockstepChannel]) -> None:
        async with self._lock:
            await register_channels(self._node, channels, env=self._env)
