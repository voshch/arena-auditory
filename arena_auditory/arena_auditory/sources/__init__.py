"""Source models: how a SoundSource becomes mono audio, as a decoded buffer or a block stream."""

from __future__ import annotations

import typing
from collections.abc import Mapping

import attrs
import numpy as np
from arena_rclpy_mixins.registry import ClassRegistry

from arena_auditory.assets import DecodedSample, SampleDecoder, SoundAsset, Variant
from arena_auditory.shared import SourceSpec
from arena_auditory.sources.events import PedestrianEventDetector, PedestrianState

if typing.TYPE_CHECKING:
    from arena_auditory.params import ParamGroup


@attrs.frozen
class BufferProgram:
    """A decoded sample played once or looped from the source's program start."""

    sample: DecodedSample
    loop: bool


@typing.runtime_checkable
class StreamProgram(typing.Protocol):
    """A stateful generator rendered block by block."""

    def update(self, source: SourceSpec) -> None: ...

    def tune(self, tuning: Mapping[str, float]) -> None: ...

    def render(self, frames: int) -> np.ndarray: ...

    @property
    def finished(self) -> bool: ...


@attrs.frozen(kw_only=True)
class ProgramContext:
    decoder: SampleDecoder
    sample_rate_hz: int
    block_size: int


class SourceModel(typing.Protocol):
    name: typing.ClassVar[str]
    continuous: typing.ClassVar[bool]
    streams: typing.ClassVar[bool]
    tuning_group: typing.ClassVar[type[ParamGroup] | None]

    @classmethod
    def program(cls, source: SourceSpec, asset: SoundAsset, variant: Variant, context: ProgramContext) -> BufferProgram | StreamProgram: ...


class StreamModel(SourceModel, typing.Protocol):
    @classmethod
    def stream(cls, *, seed: int, sample_rate_hz: int, block_size: int, params: Mapping[str, object], tuning: Mapping[str, float]) -> StreamProgram: ...

    @classmethod
    def prewarm(cls, sample_rate_hz: int, params: Mapping[str, object]) -> None: ...

    @classmethod
    def reference_rms(cls, sample_rate_hz: int, params: Mapping[str, object], tuning: Mapping[str, float]) -> float: ...


SOURCE_MODELS: ClassRegistry[str, type[SourceModel]] = ClassRegistry()


@SOURCE_MODELS.register("wav")
def _load_wav() -> type[SourceModel]:
    from arena_auditory.sources.wav import WavModel

    return WavModel


@SOURCE_MODELS.register("wav_loop")
def _load_wav_loop() -> type[SourceModel]:
    from arena_auditory.sources.wav import WavLoopModel

    return WavLoopModel


@SOURCE_MODELS.register("drivetrain")
def _load_drivetrain() -> type[SourceModel]:
    from arena_auditory.sources.drivetrain.program import DrivetrainModel

    return DrivetrainModel


def streamed(name: str) -> bool:
    """Whether the registered model of that name streams, False for an unknown name."""
    return name in SOURCE_MODELS and SOURCE_MODELS.get(name).streams


def stream_model(name: str) -> type[StreamModel]:
    """Raises KeyError for an unknown model, ValueError for a buffered one."""
    model = SOURCE_MODELS.get(name)
    if not model.streams:
        raise ValueError(f"source model {name!r} does not stream")
    return typing.cast("type[StreamModel]", model)


def stream_models() -> tuple[type[StreamModel], ...]:
    """Every registered streamed model."""
    return tuple(stream_model(name) for name in SOURCE_MODELS.keys() if streamed(name))


def tuning_of(values: Mapping[str, object]) -> dict[str, float]:
    """A streamed model's tuning, the float parameters of its tuning group."""
    return {name: value for name, value in values.items() if isinstance(value, float)}


__all__ = [
    "SOURCE_MODELS",
    "BufferProgram",
    "PedestrianEventDetector",
    "PedestrianState",
    "ProgramContext",
    "SourceModel",
    "StreamModel",
    "StreamProgram",
    "stream_model",
    "stream_models",
    "streamed",
    "tuning_of",
]
