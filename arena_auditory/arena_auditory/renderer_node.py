"""Renders propagated receptions into calibrated PCM: the robot microphone array (role array) or one selected microphone (role listener)."""

from __future__ import annotations

import asyncio
import functools
import itertools
import json
import math
import typing
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import attrs
import numpy as np
import yaml
from arena_auditory_msgs.msg import (
    AudioFrame,
    ContinuousHeardSoundState,
    HeardSoundEvent,
    RenderedSoundActivity,
    RoomImpulse,
    SoundReception,
    SoundSource,
)
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_rclpy_mixins.qos import best_effort, latched, reliable
from arena_rclpy_mixins.Time import Time
from arena_runtime.lockstep import register_channels
from arena_runtime_msgs.msg import LockstepChannel
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point
from rclpy.clock import Clock, ClockType, JumpThreshold, TimeJump
from rclpy.duration import Duration
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from std_msgs.msg import ColorRGBA, Float32MultiArray, Header, String
from task_generator_msgs.msg import EpisodeRecord, RobotFleet
from visualization_msgs.msg import Marker, MarkerArray

from arena_auditory.assets import DecodedSample, SampleDecoder, SoundAsset, SoundLibrary, Variant
from arena_auditory.constants import (
    CONTINUOUS_HEARD_SOUNDS,
    HEARD_SOUND_EVENTS,
    LISTENER_MONITOR,
    ROOM_IMPULSES,
    STATE_EPISODE,
    STATE_ROBOTS,
    ArrayStream,
    array_stream,
)
from arena_auditory.params import Configuration, MonitorMode, Param, ParamGroup, RenderRole
from arena_auditory.propagation import IMPULSE_WINDOW
from arena_auditory.render.clock import RenderCursor
from arena_auditory.render.core import (
    ClipInput,
    ContinuousInput,
    ImpulseShape,
    RenderInputs,
    RenderParams,
    RenderResult,
    RenderState,
    StreamInput,
    impulse_shape,
    new_state,
    prepare_clip,
    render_block,
    render_inputs_to_json,
)
from arena_auditory.render.dsp import gcc_phat, interleave, mems_gain
from arena_auditory.render.monitor import MonitorConfig, apply_controls, monitor_mix, monitor_playback, tdoa_pairs
from arena_auditory.render.output import DISABLED_DEVICES, AudioOutput
from arena_auditory.shared import ListenerId, RobotBinding, SourceSpec, dbfs_from_rms, load_array_spec, rms, robot_bindings
from arena_auditory.sources import SOURCE_MODELS, BufferProgram, ProgramContext, StreamModel, StreamProgram, stream_model, stream_models, streamed, tuning_of
from arena_auditory.world_tracker import follow_world_sounds

EVENT_QOS = reliable(50)
CONTINUOUS_QOS = best_effort(64)
METADATA_QOS = latched(1)
IMPULSE_QOS = latched(256)
AUDIO_QOS = reliable(10)
LAZY_STREAMS = (ArrayStream.STEM_MOTOR, ArrayStream.STEM_PEDESTRIAN, ArrayStream.STEM_AMBIENT, ArrayStream.MONITOR, ArrayStream.HEARING_MONO, ArrayStream.ENERGY, ArrayStream.TDOA, ArrayStream.RENDER_INPUTS, ArrayStream.LEVELS)

LISTENER_SPEC = "mono"
AUDIO_FRAME_TYPE = "arena_auditory_msgs/msg/AudioFrame"
EVENT_GATHER_S = 1.0
CENTROID = -1
STEREO_NAMES = ("left", "right")
LEVEL_LABEL_PERIOD_S = 0.25
CONTINUOUS_STALE_S = 0.5
IMPULSE_WAIT_BLOCKS = 1

LOAD_ERRORS = (OSError, LookupError, ValueError, TypeError, yaml.YAMLError)

type ProgramKey = tuple[int, str, str, int]


class Room(NamedTuple):
    impulse: ImpulseShape | None
    key: str
    silent: bool


def program_key(source: SourceSpec) -> ProgramKey:
    """Identity of one continuous program: its start and the variant it plays."""
    return source.program_start_ns, source.asset_id, source.variant_id, 0 if source.variant_id else source.seed


@dataclass(slots=True)
class EventLoad:
    source: SourceSpec
    receptions: dict[int, SoundReception]
    scheduled: set[int]
    anchor: int
    updated_at: float
    impulse_deadline: int
    future: Future[DecodedSample] | None = None


@dataclass(slots=True)
class ContinuousLoad:
    future: Future[DecodedSample]
    source: SourceSpec
    reception: SoundReception
    impulse_deadline: int


@dataclass(slots=True)
class Gate:
    """Latest reception per channel of one continuous source and the channels that pass, CENTROID for the robot listener."""

    source: SourceSpec
    updated_ns: int
    receptions: dict[int, SoundReception] = field(default_factory=dict)
    passing: set[int] = field(default_factory=set)


@dataclass(slots=True)
class RenderTarget:
    """One rendered stream and its voices: a fleet robot's array, or the selected listener microphone."""

    robot: str
    state: RenderState
    last_levels: np.ndarray
    binding: RobotBinding | None = None
    publishers: dict[ArrayStream, Publisher] = field(default_factory=dict)
    lazy: dict[ArrayStream, LazyPublisher] = field(default_factory=dict)
    pending_clips: list[ClipInput] = field(default_factory=list)
    reset_pending: bool = False
    event_loads: dict[str, EventLoad] = field(default_factory=dict)
    gates: dict[str, Gate] = field(default_factory=dict)
    continuous: dict[tuple[str, int], ContinuousInput] = field(default_factory=dict)
    continuous_programs: dict[tuple[str, int], ProgramKey] = field(default_factory=dict)
    continuous_pending: dict[tuple[str, int], ContinuousLoad] = field(default_factory=dict)
    streams: dict[str, StreamInput] = field(default_factory=dict)
    stream_sources: dict[str, SourceSpec] = field(default_factory=dict)
    reported_activity: dict[tuple[str, int], bool] = field(default_factory=dict)

    def rir_keys(self) -> set[str]:
        """Room impulse keys its voices and pending receptions reference."""
        keys = {voice.rir_key for voice in self.continuous.values()}
        keys.update(key for stream in self.streams.values() for key in stream.rir_keys)
        keys.update(reception.rir_key for load in self.event_loads.values() for reception in load.receptions.values())
        keys.update(load.reception.rir_key for load in self.continuous_pending.values())
        keys.discard("")
        return keys


class RendererNode(ArenaMixinNode):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("renderer", **kwargs)
        self.conf = Configuration(self)
        conf = self.conf
        self._render_conf = conf.Render
        self._monitor_conf = conf.Monitor
        self._output_conf = conf.Output
        self._motor_conf = conf.Motor
        self._diagnostics_conf = conf.Diagnostics
        self._role = self._render_conf.ROLE.value
        if self._role is RenderRole.ARRAY:
            self._array_conf = conf.Array
            self._tdoa_conf = conf.Tdoa
            self.spec = load_array_spec(self._array_conf.SPEC.value)
        else:
            self._listener_conf = conf.Listener
            self.spec = load_array_spec(LISTENER_SPEC)
        self.sample_rate = self.spec.sample_rate_hz
        self.block_size = self.spec.block_size

        self._library = SoundLibrary.default()
        self._decoder = SampleDecoder(self._library, self.sample_rate)
        self._program_context = ProgramContext(decoder=self._decoder, sample_rate_hz=self.sample_rate, block_size=self.block_size)
        self._tuning: dict[type[ParamGroup], dict[str, float]] = {}
        for model in stream_models():
            group = model.tuning_group
            if group is not None and group not in self._tuning:
                self._tuning[group] = tuning_of(conf.group(group).values())
            if self._motor_conf.ENABLED.value:
                model.prewarm(self.sample_rate, {})

        self._sim_time = bool(self.get_parameter("use_sim_time").value)
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="renderer_loader")
        self._episode_id: int | None = None
        self._world = ""
        self._listener_id = ""
        self._targets: dict[str, RenderTarget] = {}
        self._routes: dict[str, tuple[RenderTarget, int]] = {}
        self._streaming = False
        self._sample_index = 0
        self._lockstep_lock = asyncio.Lock()
        self._params = RenderParams(
            channels=self.spec.channels,
            block_size=self.block_size,
            sample_rate=self.sample_rate,
            resolve=self._resolve,
            impulse=self._impulse,
            rir_crossfade_frames=max(int(self.sample_rate * self._render_conf.RIR_CROSSFADE_S.value), 1),
            log_error=self.get_logger().error,
        )
        self._cursor = RenderCursor(
            block_ns=round(self.block_size * 1_000_000_000 / self.sample_rate),
            max_catchup=max(self._render_conf.MAX_CATCHUP_BLOCKS.value, 1),
        )
        self._stream_start_ns: int | None = None
        self._reported_skipped = 0
        self._render_behind = False
        self._samples: dict[str, DecodedSample] = {}
        self._impulses: OrderedDict[str, ImpulseShape] = OrderedDict()
        self._impulse_sub: Subscription | None = None
        self._clipped_samples = 0
        self._heard = 0
        self._accepted = 0
        self._output: AudioOutput | None = None
        self._output_error = ""
        self._output_degraded = False
        self._reported_underflows = 0
        self._reported_overflows = 0
        self._listener_pub: LazyPublisher[AudioFrame] | None = None

        self.create_subscription(HeardSoundEvent, HEARD_SOUND_EVENTS, self._on_heard, EVENT_QOS)
        self.create_subscription(ContinuousHeardSoundState, CONTINUOUS_HEARD_SOUNDS, self._on_continuous, CONTINUOUS_QOS)
        self._follow_impulses(self._render_conf.RIR_ENABLED.value)
        self.add_param_callback(self._render_conf.RIR_ENABLED.name, self._on_rir_enabled)
        follow_world_sounds(self, self._library, self._loader, self._on_world)
        self.create_subscription(EpisodeRecord, STATE_EPISODE, self._on_episode, latched(20))

        for group in self._tuning:
            for param in group.params():
                if isinstance(param.default, float):
                    self.add_param_callback(param.name, functools.partial(self._on_tuning_param, group, param))
        self.add_param_callback(self._output_conf.ENABLED.name, self._on_output_enabled)
        self.add_param_callback(self._output_conf.DEVICE.name, self._on_output_device)

        if self._role is RenderRole.ARRAY:
            self.create_subscription(RobotFleet, STATE_ROBOTS, self._on_fleet, METADATA_QOS)
        else:
            self._targets[""] = self._new_target("")
            self._listener_pub = LazyPublisher(self.create_publisher(AudioFrame, LISTENER_MONITOR, AUDIO_QOS))
            self.add_param_callback(self._listener_conf.ID.name, self._on_listener_id)
            self._streaming = True
            if not self._sim_time:
                self._stream_start_ns = self._steady_clock.now().nanoseconds
            self._select_listener(self._listener_conf.ID.value)

        if self._sim_time:
            self._clock_jump = self.get_clock().create_jump_callback(
                JumpThreshold(min_forward=Duration(nanoseconds=1), min_backward=None, on_clock_change=False),
                post_callback=self._on_clock,
            )
        else:
            self.create_timer(self.block_size / self.sample_rate, self._on_steady, clock=self._steady_clock)
        self.create_timer(self._output_conf.RETRY_PERIOD_S.value, self._retry_output, clock=self._steady_clock)
        self.create_timer(self._diagnostics_conf.PERIOD_S.value, self._publish_diagnostics, clock=self._steady_clock)
        self.create_timer(LEVEL_LABEL_PERIOD_S, self._publish_levels, clock=self._steady_clock)
        self.get_logger().info(f"{self._role} renderer ready: {self.spec.name} array, {self.spec.channels} channels {self.spec.channel_names}, {self.sample_rate} Hz, {self.block_size} frames")

    def _model_tuning(self, model: type[StreamModel]) -> dict[str, float]:
        return {} if model.tuning_group is None else self._tuning[model.tuning_group]

    def _on_tuning_param(self, group: type[ParamGroup], param: Param[float], value: object) -> bool:
        try:
            parsed = param.parse(value) if param.parse is not None else value
        except (TypeError, ValueError):
            return False
        tuning = tuning_of({**self.conf.group(group).values(), param.field: float(typing.cast(float, parsed))})
        self._tuning[group] = tuning
        for target in self._targets.values():
            for source_id, source in tuple(target.streams.items()):
                if stream_model(source.model).tuning_group is group:
                    target.streams[source_id] = replace(source, tuning=tuning)
        return True

    def _on_episode(self, msg: EpisodeRecord) -> None:
        episode_id = int(msg.episode_id)
        if episode_id == self._episode_id:
            return
        self._episode_id = episode_id
        self._reset_voices()

    def _on_fleet(self, msg: RobotFleet) -> None:
        bindings = {binding.name: binding for binding in robot_bindings(msg)}
        for robot in self._targets.keys() - bindings.keys():
            self._stop_array(self._targets[robot])
        targets: dict[str, RenderTarget] = {}
        for robot, binding in bindings.items():
            target = self._targets.get(robot)
            if target is None:
                target = self._start_array(robot)
                if binding.error:
                    self.get_logger().warning(binding.error)
            target.binding = binding
            targets[robot] = target
        changed = list(targets) != list(self._targets)
        self._targets = targets
        if not changed:
            return
        routes: dict[str, tuple[RenderTarget, int]] = {}
        for robot, target in targets.items():
            routes[ListenerId.robot(robot)] = (target, CENTROID)
            routes.update({ListenerId.array_mic(robot, mic.name): (target, index) for index, mic in enumerate(self.spec.mics)})
        self._routes = routes
        self._streaming = bool(targets)
        if self._streaming and self._stream_start_ns is None and not self._sim_time:
            self._stream_start_ns = self._steady_clock.now().nanoseconds
        if self._streaming and self._output_conf.ENABLED.value and self._output is None:
            self._open_output(self._output_conf.DEVICE.value)
        elif not self._streaming:
            self._close_output()
        if self._sim_time and self._render_conf.LOCKSTEP_ENABLED.value:
            asyncio.run_coroutine_threadsafe(self._register_lockstep(self._lockstep_channels()), self.event_loop)

    def _new_target(self, robot: str) -> RenderTarget:
        state = new_state(self.spec.channels)
        state.cursor = self._sample_index
        return RenderTarget(robot=robot, state=state, last_levels=np.zeros(self.spec.channels + 3, dtype=np.float32))

    def _start_array(self, robot: str) -> RenderTarget:
        target = self._new_target(robot)
        for stream in (ArrayStream.RAW, ArrayStream.STEM_MOTOR, ArrayStream.STEM_PEDESTRIAN, ArrayStream.STEM_AMBIENT, ArrayStream.MONITOR, ArrayStream.HEARING_MONO, ArrayStream.RENDER_INPUTS):
            target.publishers[stream] = self.create_publisher(String if stream is ArrayStream.RENDER_INPUTS else AudioFrame, array_stream(robot, stream), AUDIO_QOS)
        target.publishers[ArrayStream.ENERGY] = self.create_publisher(Float32MultiArray, array_stream(robot, ArrayStream.ENERGY), reliable(10))
        target.publishers[ArrayStream.TDOA] = self.create_publisher(String, array_stream(robot, ArrayStream.TDOA), reliable(10))
        target.publishers[ArrayStream.ACTIVITY] = self.create_publisher(RenderedSoundActivity, array_stream(robot, ArrayStream.ACTIVITY), EVENT_QOS)
        target.publishers[ArrayStream.LEVELS] = self.create_publisher(MarkerArray, array_stream(robot, ArrayStream.LEVELS), reliable(1))
        target.lazy = {stream: LazyPublisher(target.publishers[stream]) for stream in LAZY_STREAMS}
        return target

    def _stop_array(self, target: RenderTarget) -> None:
        self._reset_target(target)
        for publisher in target.publishers.values():
            self.destroy_publisher(publisher)
        target.publishers.clear()
        target.lazy.clear()

    def _lockstep_channels(self) -> list[LockstepChannel]:
        return [
            LockstepChannel(
                name=f"audio/{robot}",
                topic=target.publishers[ArrayStream.RAW].topic_name,
                type=AUDIO_FRAME_TYPE,
                period_s=self.block_size / self.sample_rate,
                hard=True,
            )
            for robot, target in self._targets.items()
        ]

    async def _register_lockstep(self, channels: list[LockstepChannel]) -> None:
        async with self._lockstep_lock:
            await register_channels(self, channels, env=self.get_namespace())

    def _output_target(self) -> RenderTarget | None:
        """The robot the workstation plays: array.robot when set, else the first fleet robot."""
        robot = self._array_conf.ROBOT.value.strip()
        if robot:
            return self._targets.get(robot)
        return next(iter(self._targets.values()), None)

    def _on_listener_id(self, value: object) -> bool:
        self._select_listener(str(value))
        return True

    def _select_listener(self, listener_id: str) -> None:
        listener_id = listener_id.strip()
        if listener_id == self._listener_id:
            return
        self._listener_id = listener_id
        target = self._targets[""]
        self._reset_target(target)
        self._routes = {listener_id: (target, 0)} if listener_id else {}
        if listener_id:
            self._open_output(self._output_conf.DEVICE.value)
        else:
            self._close_output()

    def _reset_voices(self) -> None:
        for target in self._targets.values():
            self._reset_target(target)
        self._evict_impulses()

    def _reset_target(self, target: RenderTarget) -> None:
        if self._role is RenderRole.ARRAY:
            target.streams.clear()
            self._publish_stream_activity(target, self._sample_index)
        target.reset_pending = True
        for load in target.event_loads.values():
            if load.future is not None:
                load.future.cancel()
        for pending in target.continuous_pending.values():
            pending.future.cancel()
        target.event_loads.clear()
        target.gates.clear()
        target.continuous_pending.clear()
        target.continuous.clear()
        target.continuous_programs.clear()
        target.stream_sources.clear()
        target.pending_clips.clear()
        target.reported_activity.clear()
        target.state = new_state(self.spec.channels)
        target.state.cursor = self._sample_index

    def _idle(self) -> bool:
        return self._role is RenderRole.LISTENER and not self._listener_id

    def _wants_output(self, device: str) -> bool:
        if device.strip() in DISABLED_DEVICES:
            return False
        if self._role is RenderRole.ARRAY:
            return self._streaming and self._output_conf.ENABLED.value
        return bool(self._listener_id)

    def _open_output(self, device: str) -> None:
        if device.strip() in DISABLED_DEVICES:
            return
        if self._output is None:
            self._output = AudioOutput(
                device=device,
                sample_rate_hz=self.sample_rate,
                channels=len(STEREO_NAMES),
                block_size=self._output_conf.BLOCK_SIZE.value,
                buffer_s=self._output_conf.BUFFER_S.value,
                push_frames=self.block_size,
            )
        error = self._output.open()
        if error and error != self._output_error:
            self.get_logger().warning(f"workstation output unavailable: {error}, retrying")
        elif not error:
            self.get_logger().info(f"workstation output active on device {self._output.stats.device}")
        self._output_error = error

    def _close_output(self) -> None:
        if self._output is not None:
            self._output.close()
        self._output = None
        self._output_error = ""

    def _on_output_enabled(self, value: object) -> bool:
        if bool(value) and self._streaming and self._role is RenderRole.ARRAY:
            self._open_output(self._output_conf.DEVICE.value)
        elif self._role is RenderRole.ARRAY:
            self._close_output()
        return True

    def _on_output_device(self, value: object) -> bool:
        self._close_output()
        if self._wants_output(str(value)):
            self._open_output(str(value))
        return True

    def _retry_output(self) -> None:
        device = self._output_conf.DEVICE.value
        if self._wants_output(device) and (self._output is None or not self._output.stats.active):
            self._open_output(device)

    def _on_clock(self, _jump: TimeJump) -> None:
        if not self._streaming:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self._stream_start_ns is None:
            self._stream_start_ns = now_ns
            self._cursor.start_ns = now_ns
        render, skip = self._cursor.owed(now_ns)
        if self._idle():
            self._skip(render + skip)
            return
        for _ in range(render):
            self._render_block()
        if skip:
            self._skip(skip)
        self._report_skips()

    def _on_steady(self) -> None:
        if not self._streaming:
            return
        if self._idle():
            self._skip(1)
            return
        self._render_block()

    def _skip(self, blocks: int) -> None:
        self._sample_index += blocks * self.block_size
        for target in self._targets.values():
            target.state.cursor = self._sample_index

    def _report_skips(self) -> None:
        new_skipped = self._cursor.skipped - self._reported_skipped
        self._reported_skipped = self._cursor.skipped
        if new_skipped > 0 and not self._render_behind:
            self.get_logger().error(f"render fell behind /clock, skipped {new_skipped} block(s) ({new_skipped * self.block_size / self.sample_rate:.2f} s of audio) to stay current: those blocks carry no render_inputs trace and cannot be re-rendered offline")
        elif self._render_behind and new_skipped == 0:
            self.get_logger().info("render caught up with /clock")
        self._render_behind = new_skipped > 0

    def _render_time(self) -> float:
        return self._sample_index / self.sample_rate

    def _accepts(self, reception: SoundReception) -> bool:
        render = self._render_conf
        return (reception.audible or render.INAUDIBLE_ENABLED.value) and reception.received_level_db >= render.MIN_LEVEL_DB.value

    def _delay_samples(self, reception: SoundReception) -> float:
        delay = float(reception.direct_delay_s) * self.sample_rate
        return delay if math.isfinite(delay) and delay > 0.0 else 0.0

    def _room(self, rir_key: str, current: str = "") -> Room:
        """The room a reception renders in. A voice keeps its current impulse until the impulse of a new key arrives."""
        render = self._render_conf
        if not rir_key or not render.RIR_ENABLED.value:
            return Room(None, "", False)
        for key in (rir_key, current):
            shape = self._impulses.get(key) if key else None
            if shape is not None:
                if key != rir_key:
                    self.get_logger().warning(f"room impulse {rir_key!r} not received, keeping {key!r}", throttle_duration_sec=5.0)
                return Room(shape, key, False)
        fallback = render.RIR_DRY_FALLBACK_ENABLED.value
        self.get_logger().warning(f"room impulse {rir_key!r} not received, rendering {'dry' if fallback else 'silence'}", throttle_duration_sec=5.0)
        return Room(None, "", not fallback)

    def _awaits_impulse(self, rir_key: str, deadline: int) -> bool:
        """Whether a reception still waits, until the deadline sample, for its room impulse to arrive."""
        return self._sample_index < deadline and bool(rir_key) and self._render_conf.RIR_ENABLED.value and rir_key not in self._impulses

    def _impulse_deadline(self) -> int:
        return self._sample_index + IMPULSE_WAIT_BLOCKS * self.block_size

    def _on_rir_enabled(self, value: object) -> bool:
        self._follow_impulses(bool(value))
        return True

    def _follow_impulses(self, enabled: bool) -> None:
        if enabled and self._impulse_sub is None:
            self._impulse_sub = self.create_subscription(RoomImpulse, ROOM_IMPULSES, self._on_impulse, IMPULSE_QOS)
        elif not enabled and self._impulse_sub is not None:
            self.destroy_subscription(self._impulse_sub)
            self._impulse_sub = None
            self._impulses.clear()

    def _on_world(self, world: str) -> None:
        if world == self._world:
            return
        if self._world:
            self._impulses.clear()
        self._world = world

    def _on_impulse(self, msg: RoomImpulse) -> None:
        if msg.key in self._impulses:
            self._impulses.move_to_end(msg.key)
            return
        try:
            shape = impulse_shape(np.asarray(msg.samples, dtype=np.float32), int(msg.sample_rate_hz), int(msg.lead_samples), self.sample_rate)
        except ValueError as exc:
            self.get_logger().warning(f"dropping room impulse {msg.key!r}: {exc}")
            return
        self._impulses[msg.key] = shape
        self._evict_impulses()

    def _evict_impulses(self) -> None:
        """Drop the impulses no voice references outside the IMPULSE_WINDOW most recently announced."""
        excess = len(self._impulses) - IMPULSE_WINDOW
        if excess <= 0:
            return
        live = {key for target in self._targets.values() for key in target.rir_keys()}
        for key in [key for key in itertools.islice(self._impulses, excess) if key not in live]:
            del self._impulses[key]

    def _impulse(self, key: str) -> ImpulseShape | None:
        return self._impulses.get(key)

    def _resolve(self, asset_key: str) -> np.ndarray:
        return self._samples[asset_key].samples

    def _variant(self, source: SourceSpec) -> tuple[SoundAsset, Variant]:
        asset = self._library.asset(source.asset_id)
        return asset, self._library.variant(source.asset_id, source.variant_id) if source.variant_id else asset.select(context={}, seed=source.seed)

    def _program(self, source: SourceSpec) -> BufferProgram | StreamProgram:
        asset, variant = self._variant(source)
        return SOURCE_MODELS.get(variant.model).program(source, asset, variant, self._program_context)

    def _stream_params(self, source: SourceSpec, model: type[StreamModel]) -> dict[str, object] | None:
        """The params of the source's variant with its model prewarmed for them, None when they do not resolve."""
        try:
            _, variant = self._variant(source)
            params = dict(variant.params)
            model.prewarm(self.sample_rate, params)
        except LOAD_ERRORS as exc:
            self.get_logger().error(f"cannot render {source.model} stream {source.id!r} of {source.asset_id!r}: {exc}", throttle_duration_sec=5.0)
            return None
        return params

    def _sample(self, source: SourceSpec) -> DecodedSample:
        program = self._program(source)
        if not isinstance(program, BufferProgram):
            raise ValueError(f"source {source.id!r} of model {source.model!r} streams, the renderer needs a decoded sample")
        return program.sample

    def _submit[T](self, work: Callable[[], T]) -> Future[T]:
        """Decode off-thread in real time, inline under sim time."""
        if not self._sim_time:
            return self._loader.submit(work)
        future: Future[T] = Future()
        try:
            future.set_result(work())
        except LOAD_ERRORS as exc:
            future.set_exception(exc)
        return future

    def _source(self, msg: SoundSource) -> SourceSpec | None:
        try:
            return SourceSpec.from_msg(msg)
        except ValueError as exc:
            self.get_logger().warning(f"ignoring sound source {msg.id!r}: {exc}", throttle_duration_sec=5.0)
            return None

    def _on_heard(self, msg: HeardSoundEvent) -> None:
        reception = msg.reception
        route = self._routes.get(reception.listener_id)
        if route is None:
            return
        target, channel = route
        self._heard += 1
        source = self._source(msg.source)
        if source is None:
            return
        passes = self._accepts(reception)
        self._accepted += passes
        key = source.id or f"{source.agent_id}:{source.asset_id}:{msg.header.stamp.sec}:{msg.header.stamp.nanosec}"
        load = target.event_loads.get(key)
        if load is None:
            if not passes and channel == CENTROID:
                return
            load = EventLoad(
                source=source,
                receptions={},
                scheduled=set(),
                anchor=self._sample_index + self.block_size,
                updated_at=self._render_time(),
                impulse_deadline=self._impulse_deadline(),
            )
            target.event_loads[key] = load
        if channel != CENTROID:
            load.receptions[channel] = reception
        load.updated_at = self._render_time()
        if passes and load.future is None:
            load.future = self._submit(functools.partial(self._sample, source))

    def _on_continuous(self, msg: ContinuousHeardSoundState) -> None:
        reception = msg.reception
        route = self._routes.get(reception.listener_id)
        if route is None:
            return
        target, channel = route
        source = self._source(msg.source)
        if source is None:
            return
        self._update_gate(target, source, reception, channel)

    def _update_gate(self, target: RenderTarget, source: SourceSpec, reception: SoundReception, channel: int) -> None:
        now_ns = self._steady_clock.now().nanoseconds
        gate = target.gates.setdefault(source.id, Gate(source=source, updated_ns=now_ns))
        gate.source = source
        gate.updated_ns = now_ns
        was_admitted = bool(gate.passing)
        if source.active and self._accepts(reception):
            gate.passing.add(channel)
        else:
            gate.passing.discard(channel)
        if channel != CENTROID:
            gate.receptions[channel] = reception
        admitted = bool(gate.passing)
        changed = tuple(gate.receptions) if admitted != was_admitted else (channel,) if channel != CENTROID else ()
        for each in changed:
            if streamed(source.model):
                self._on_stream(target, source, gate.receptions[each], each, admitted)
            else:
                self._continuous_channel(target, source, gate.receptions[each], each, admitted)
        if not source.active:
            gate.receptions.pop(channel, None)
            if not gate.receptions and not gate.passing:
                target.gates.pop(source.id, None)

    def _expire_gates(self, target: RenderTarget) -> None:
        """Stop every continuous source without an update for CONTINUOUS_STALE_S of wall time."""
        now_ns = self._steady_clock.now().nanoseconds
        for source_id, gate in tuple(target.gates.items()):
            if now_ns - gate.updated_ns <= CONTINUOUS_STALE_S * 1e9:
                continue
            stopped = attrs.evolve(gate.source, active=False)
            for channel, reception in tuple(gate.receptions.items()):
                self._update_gate(target, stopped, reception, channel)
            target.gates.pop(source_id, None)

    def _continuous_channel(self, target: RenderTarget, source: SourceSpec, reception: SoundReception, channel: int, admitted: bool) -> None:
        key = (source.id, channel)
        program = program_key(source)
        existing = target.continuous.get(key)
        if existing is not None and target.continuous_programs.get(key) != program:
            self._drop_continuous(target, key)
            existing = None
        room = self._room(reception.rir_key, existing.rir_key if existing is not None else "") if source.active else Room(None, "", False)
        if not source.active or not admitted or room.silent:
            self._drop_continuous(target, key)
            return
        if existing is not None:
            target.continuous[key] = replace(
                existing,
                gain=self._spl_gain(float(reception.received_level_db), self._samples[existing.asset_key]),
                delay_target=self._delay_samples(reception),
                rir_key=room.key,
            )
            return
        pending = target.continuous_pending.get(key)
        if pending is not None and program_key(pending.source) == program:
            pending.source = source
            pending.reception = reception
            return
        if pending is not None:
            pending.future.cancel()
        target.continuous_pending[key] = ContinuousLoad(
            future=self._submit(functools.partial(self._sample, source)),
            source=source,
            reception=reception,
            impulse_deadline=self._impulse_deadline(),
        )

    def _drop_continuous(self, target: RenderTarget, key: tuple[str, int]) -> None:
        target.continuous.pop(key, None)
        target.continuous_programs.pop(key, None)
        pending = target.continuous_pending.pop(key, None)
        if pending is not None:
            pending.future.cancel()

    def _on_stream(self, target: RenderTarget, source: SourceSpec, reception: SoundReception, channel: int, admitted: bool) -> None:
        channels = self.spec.channels
        current = target.streams.get(source.id)
        if current is not None and (current.seed != source.seed or current.model != source.model):
            target.streams.pop(source.id, None)
            current = None
        if current is None:
            if not source.active:
                return
            model = stream_model(source.model)
            params = self._stream_params(source, model)
            if params is None:
                return
            try:
                stem = self._library.kind(source.kind).stem
            except KeyError as exc:
                self.get_logger().error(f"cannot render {source.model} stream {source.id!r}: {exc}", throttle_duration_sec=5.0)
                return
            current = StreamInput(
                source_id=source.id,
                model=source.model,
                seed=source.seed,
                params=params,
                state={},
                tuning=self._model_tuning(model),
                gains=(0.0,) * channels,
                delay_samples=(0.0,) * channels,
                active_channels=(False,) * channels,
                active=False,
                source_agent_id=source.agent_id,
                source_agent_name=source.agent_name,
                kind=source.kind,
                asset_id=source.asset_id,
                rir_keys=("",) * channels,
                stem=stem,
            )
        target.stream_sources[source.id] = source
        room = self._room(reception.rir_key, current.rir_keys[channel])
        active = source.active and admitted and not room.silent
        active_channels = list(current.active_channels)
        active_channels[channel] = active
        gains = list(current.gains)
        gains[channel] = self._stream_gain(float(reception.received_level_db), current) if active else 0.0
        delay_samples = list(current.delay_samples)
        delay_samples[channel] = self._delay_samples(reception)
        rir_keys = list(current.rir_keys)
        rir_keys[channel] = room.key
        target.streams[source.id] = replace(
            current,
            state=dict(source.state),
            gains=tuple(gains),
            delay_samples=tuple(delay_samples),
            active_channels=tuple(active_channels),
            active=any(active_channels),
            rir_keys=tuple(rir_keys),
        )

    def _stream_gain(self, received_spl_db: float, stream: StreamInput) -> float:
        reference = stream_model(stream.model).reference_rms(self.sample_rate, stream.params, stream.tuning)
        return mems_gain(received_spl_db, reference, sensitivity_dbfs_at_94_dbspl=self.spec.sensitivity_dbfs_at_94_dbspl)

    def _spl_gain(self, received_spl_db: float, sample: DecodedSample) -> float:
        return mems_gain(received_spl_db, sample.active_rms, sensitivity_dbfs_at_94_dbspl=self.spec.sensitivity_dbfs_at_94_dbspl)

    def _poll_loads(self, target: RenderTarget) -> None:
        now = self._render_time()
        sensitivity = self.spec.sensitivity_dbfs_at_94_dbspl
        for key, load in tuple(target.event_loads.items()):
            if load.future is None:
                if now - load.updated_at > EVENT_GATHER_S:
                    target.event_loads.pop(key, None)
                continue
            if not load.future.done():
                continue
            try:
                sample = load.future.result()
                stem = self._library.kind(load.source.kind).stem
            except LOAD_ERRORS as exc:
                self.get_logger().error(f"cannot render {load.source.asset_id!r} for {key!r}: {exc}")
                target.event_loads.pop(key, None)
                continue
            self._samples[sample.key] = sample
            if not load.scheduled:
                load.anchor = max(load.anchor, self._sample_index + self.block_size)
            for channel, reception in load.receptions.items():
                if channel in load.scheduled or self._awaits_impulse(reception.rir_key, load.impulse_deadline):
                    continue
                load.scheduled.add(channel)
                room = self._room(reception.rir_key)
                if room.silent:
                    continue
                delay_samples = self._delay_samples(reception)
                delay, delayed = prepare_clip(
                    sample.samples,
                    float(reception.received_level_db),
                    sensitivity,
                    delay_samples,
                    level_rms=sample.active_rms,
                    impulse=room.impulse,
                )
                start = load.anchor + delay
                target.pending_clips.append(
                    ClipInput(
                        channel=channel,
                        start=start,
                        asset_key=sample.key,
                        samples=delayed,
                        anchor=load.anchor,
                        received_volume_db=float(reception.received_level_db),
                        sensitivity_dbfs_at_94_dbspl=sensitivity,
                        delay_samples=delay_samples,
                        rir_key=room.key,
                        stem=stem,
                    )
                )
                if self._role is RenderRole.ARRAY:
                    self._publish_activity(target, load.source, channel, start, start + len(delayed), continuous=False, active=True)
            if len(load.scheduled) == self.spec.channels or now - load.updated_at > EVENT_GATHER_S:
                target.event_loads.pop(key, None)

        for key, pending in tuple(target.continuous_pending.items()):
            future, source, reception = pending.future, pending.source, pending.reception
            if not future.done() or self._awaits_impulse(reception.rir_key, pending.impulse_deadline):
                continue
            target.continuous_pending.pop(key, None)
            if future.cancelled():
                continue
            try:
                sample = future.result()
                stem = self._library.kind(source.kind).stem
            except LOAD_ERRORS as exc:
                self.get_logger().error(f"cannot render continuous {source.asset_id!r} for {source.id!r}: {exc}")
                continue
            self._samples[sample.key] = sample
            room = self._room(reception.rir_key)
            if room.silent:
                continue
            elapsed = max(self.get_clock().now().nanoseconds - source.program_start_ns, 0)
            source_id, channel = key
            target.continuous_programs[key] = program_key(source)
            target.continuous[key] = ContinuousInput(
                source_id=source_id,
                channel=channel,
                asset_key=sample.key,
                gain=self._spl_gain(float(reception.received_level_db), sample),
                delay_target=self._delay_samples(reception),
                loop=source.loop,
                program_start=self._sample_index - round(elapsed * self.sample_rate / 1e9),
                rir_key=room.key,
                stem=stem,
            )

    def _collect_inputs(self, target: RenderTarget) -> RenderInputs:
        clips = tuple(target.pending_clips)
        target.pending_clips.clear()
        reset, target.reset_pending = target.reset_pending, False
        return RenderInputs(
            block_index=target.state.cursor // self.block_size,
            clips=clips,
            continuous=tuple(target.continuous.values()),
            streams=tuple(target.streams.values()),
            channels=self.spec.channels,
            reset=reset,
        )

    def _render_block(self) -> None:
        block_start = self._sample_index
        stamp = self._sample_time(block_start)
        output = self._output_target() if self._role is RenderRole.ARRAY else None
        for target in tuple(self._targets.values()):
            self._expire_gates(target)
            self._poll_loads(target)
            if self._role is RenderRole.ARRAY:
                self._publish_stream_activity(target, block_start)
            inputs = self._collect_inputs(target)
            if self._role is RenderRole.ARRAY:
                target.lazy[ArrayStream.RENDER_INPUTS].publish(lambda inputs=inputs: String(data=render_inputs_to_json(inputs)))
            result = render_block(target.state, inputs, self._params)
            self._clipped_samples += result.clipped_samples
            for source_id in tuple(target.streams):
                if source_id not in target.state.streams:
                    target.streams.pop(source_id)
            if self._role is RenderRole.ARRAY:
                self._publish_array(target, result, stamp, playback=target is output)
            else:
                self._publish_listener(result, stamp)
        self._sample_index += self.block_size

    def _audible_mix(self, result: RenderResult) -> np.ndarray:
        """The rendered mix the workstation hears, output.motor and output.ambient gate their stems."""
        output = self._output_conf
        if output.MOTOR_ENABLED.value and output.AMBIENT_ENABLED.value:
            return result.raw
        mix = result.ped
        if output.AMBIENT_ENABLED.value:
            mix = mix + result.ambient
        if output.MOTOR_ENABLED.value:
            mix = mix + result.motor
        return np.clip(mix, -1.0, 1.0)

    def _monitor_settings(self) -> MonitorConfig:
        monitor = self._monitor_conf
        solo = monitor.SOLO.value.strip()
        return MonitorConfig(
            enabled=monitor.ENABLED.value,
            hearing=monitor.MODE.value is MonitorMode.HEARING,
            solo=solo if solo in self.spec.channel_names else "",
            front_gain=monitor.FRONT_GAIN.value,
            rear_gain=monitor.REAR_GAIN.value,
            output_gain=10.0 ** (monitor.MASTER_GAIN_DB.value / 20.0),
            gain_db=monitor.GAIN_DB.value,
            limit=monitor.LIMIT.value,
        )

    def _publish_array(self, target: RenderTarget, result: RenderResult, stamp: TimeMsg, *, playback: bool) -> None:
        array = self._array_conf
        enabled, muted = array.ENABLED.value, array.MUTED.value
        names = self.spec.channel_names
        raw = apply_controls(result.raw, enabled=enabled, muted=muted)
        stems = {
            ArrayStream.STEM_MOTOR: result.motor,
            ArrayStream.STEM_PEDESTRIAN: result.ped,
            ArrayStream.STEM_AMBIENT: result.ambient,
        }
        settings = self._monitor_settings()
        stereo, hearing = monitor_mix(raw, self.spec, settings)
        lazy = target.lazy
        target.publishers[ArrayStream.RAW].publish(self._frame(target, raw, stamp, names, spatial=True))
        for stream, audio in stems.items():
            lazy[stream].publish(lambda audio=audio: self._frame(target, apply_controls(audio, enabled=enabled, muted=muted), stamp, names, spatial=True))
        lazy[ArrayStream.MONITOR].publish(lambda: self._frame(target, stereo, stamp, STEREO_NAMES, spatial=False))
        lazy[ArrayStream.HEARING_MONO].publish(lambda: self._frame(target, hearing[None, :], stamp, ("hearing",), spatial=False))
        target.last_levels = np.concatenate((np.asarray(rms(raw, axis=1)), [rms(hearing), rms(stereo[0]), rms(stereo[1])])).astype(np.float32)
        lazy[ArrayStream.ENERGY].publish(lambda: Float32MultiArray(data=target.last_levels.tolist()))
        if self._tdoa_conf.ENABLED.value:
            lazy[ArrayStream.TDOA].publish(lambda: self._tdoa(raw, stamp))
        if playback and self._output is not None and self._output_conf.ENABLED.value:
            mix = self._audible_mix(result)
            if mix is not result.raw:
                stereo, hearing = monitor_mix(apply_controls(mix, enabled=enabled, muted=muted), self.spec, settings)
            self._output.push(monitor_playback(stereo, hearing, settings).T)

    def _publish_listener(self, result: RenderResult, stamp: TimeMsg) -> None:
        settings = self._monitor_settings()
        playback = monitor_playback(*monitor_mix(self._audible_mix(result), self.spec, settings), settings)
        if self._listener_pub is not None:
            self._listener_pub.publish(lambda: self._frame(self._targets[""], playback, stamp, STEREO_NAMES, spatial=False))
        if self._output is not None:
            self._output.push(playback.T)

    def _sample_time(self, sample_index: int) -> TimeMsg:
        if self._stream_start_ns is None:
            raise RuntimeError("audio sample clock is not initialized")
        return Time.from_nanoseconds(self._stream_start_ns + round(sample_index * 1_000_000_000 / self.sample_rate)).to_msg()

    def _mount_frame(self, target: RenderTarget) -> str:
        if self._role is not RenderRole.ARRAY:
            return ""
        return target.binding.mount(self._array_conf.MOUNT_FRAME.value) if target.binding is not None else ""

    @staticmethod
    def _mic_frame(target: RenderTarget, name: str) -> str:
        leaf = f"mic_{name}"
        return target.binding.frame(leaf) if target.binding is not None else leaf

    def _frame(self, target: RenderTarget, audio: np.ndarray, stamp: TimeMsg, names: tuple[str, ...], *, spatial: bool) -> AudioFrame:
        msg = AudioFrame()
        msg.header.stamp = stamp
        msg.header.frame_id = self._mount_frame(target)
        msg.sample_rate = self.sample_rate
        msg.channel_count = audio.shape[0]
        msg.frame_count = audio.shape[1]
        msg.encoding = "32FC1"
        msg.interleaved = True
        msg.channel_names = list(names)
        if spatial:
            msg.frame_ids = [self._mic_frame(target, mic.name) for mic in self.spec.mics]
            msg.microphone_positions = [Point(x=mic.position_m[0], y=mic.position_m[1], z=mic.position_m[2]) for mic in self.spec.mics]
            msg.microphone_yaw_rad = [mic.yaw_rad for mic in self.spec.mics]
            msg.sensitivity_dbfs_at_94_dbspl = self.spec.sensitivity_dbfs_at_94_dbspl
        msg.data = interleave(audio)
        return msg

    def _publish_levels(self) -> None:
        """Label each array microphone with its last block level in dBFS."""
        for target in self._targets.values():
            publisher = target.lazy.get(ArrayStream.LEVELS)
            frame = self._mount_frame(target)
            if publisher is None or not frame or not publisher.wanted:
                continue
            header = Header(frame_id=frame, stamp=self.get_clock().now().to_msg())
            markers = []
            for index, mic in enumerate(self.spec.mics):
                label = Marker(header=header, ns="microphone_levels", id=index, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
                label.pose.position = Point(x=mic.position_m[0], y=mic.position_m[1], z=mic.position_m[2] + 0.10)
                label.pose.orientation.w = 1.0
                label.scale.z = 0.075
                label.color = ColorRGBA(r=0.95, g=0.95, b=0.95, a=1.0)
                label.lifetime.sec = 1
                label.text = f"{dbfs_from_rms(float(target.last_levels[index])):.1f} dBFS"
                markers.append(label)
            publisher.publish(lambda markers=markers: MarkerArray(markers=markers))

    def _tdoa(self, raw: np.ndarray, stamp: TimeMsg) -> String:
        estimates: dict[str, dict[str, float]] = {}
        for first, second, label in tdoa_pairs(self.spec):
            delay, confidence = gcc_phat(raw[first], raw[second], sample_rate_hz=self.sample_rate, max_tau_s=self._tdoa_conf.MAX_LAG_S.value)
            estimates[label] = {"delay_us": delay * 1e6, "confidence": confidence}
        estimates["stamp"] = {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}
        return String(data=json.dumps(estimates, separators=(",", ":")))

    def _publish_activity(self, target: RenderTarget, source: SourceSpec, channel: int, start_sample: int, end_sample: int, *, continuous: bool, active: bool) -> None:
        msg = RenderedSoundActivity()
        msg.header.stamp = self._sample_time(start_sample)
        msg.header.frame_id = self._mount_frame(target)
        msg.stream_id = array_stream(target.robot, ArrayStream.RAW)
        msg.source = source.to_msg()
        msg.channel_name = self.spec.channel_names[channel]
        msg.continuous = continuous
        msg.active = active
        msg.start_sample_index = start_sample
        msg.end_sample_index = end_sample
        msg.start_time = self._sample_time(start_sample)
        msg.end_time = self._sample_time(end_sample)
        target.publishers[ArrayStream.ACTIVITY].publish(msg)

    def _publish_stream_activity(self, target: RenderTarget, sample_index: int) -> None:
        current: dict[tuple[str, int], bool] = {}
        for source_id, source in target.streams.items():
            for channel in range(self.spec.channels):
                key = (source_id, channel)
                active = bool(source.active_channels[channel])
                current[key] = active
                if target.reported_activity.get(key, False) != active:
                    self._publish_activity(target, target.stream_sources[source_id], channel, sample_index, sample_index, continuous=True, active=active)
        for key, was_active in target.reported_activity.items():
            if was_active and key not in current:
                source_id, channel = key
                self._publish_activity(target, target.stream_sources[source_id], channel, sample_index, sample_index, continuous=True, active=False)
                current[key] = False
        for source_id in tuple(target.stream_sources):
            if source_id not in target.streams:
                target.stream_sources.pop(source_id)
        target.reported_activity = current

    def _publish_diagnostics(self) -> None:
        stats = self._output.stats if self._output is not None else None
        underflows = stats.underflows if stats is not None else 0
        overflows = stats.overflows if stats is not None else 0
        new_underflows = underflows - self._reported_underflows
        new_overflows = overflows - self._reported_overflows
        self._reported_underflows, self._reported_overflows = underflows, overflows
        active = stats is not None and stats.active
        degraded = self._wants_output(self._output_conf.DEVICE.value) and (not active or new_underflows > 0 or new_overflows > 0)
        if degraded and not self._output_degraded:
            self.get_logger().warning(f"workstation output degraded (active={active}, underflows={underflows}, overflows={overflows}), monitoring only, rendered streams are unaffected")
        elif self._output_degraded and not degraded:
            self.get_logger().info("workstation output recovered")
        self._output_degraded = degraded
        self.get_logger().debug(
            f"{self._role} renderer diagnostics: robots={list(self._targets) if self._role is RenderRole.ARRAY else None}, listener={self._listener_id or None}, streaming={self._streaming}, "
            f"heard={self._heard}, accepted={self._accepted}, pending_events={sum(len(target.event_loads) for target in self._targets.values())}, "
            f"wav_voices={sum(len(target.state.continuous) for target in self._targets.values())}, stream_voices={sum(len(target.state.streams) for target in self._targets.values())}, impulses={len(self._impulses)}, "
            f"clipped_samples={self._clipped_samples}, rendered={self._cursor.rendered}, skipped={self._cursor.skipped}, output={stats}"
        )

    def destroy_node(self) -> None:
        self._loader.shutdown(wait=False, cancel_futures=True)
        self._close_output()
        super().destroy_node()


def main() -> None:
    RendererNode.run_main()
