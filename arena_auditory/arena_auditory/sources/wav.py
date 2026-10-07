"""Decoded wav source models."""

from __future__ import annotations

import typing

from arena_auditory.sources import BufferProgram

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.param_groups import ParamGroup
    from arena_simulation_setup.tree.assets.sound_catalog import SoundAsset, Variant

    from arena_auditory.shared import SourceSpec
    from arena_auditory.sources import ProgramContext


class WavModel:
    """One-shot playback of the variant's wav."""

    name: typing.ClassVar[str] = "wav"
    continuous: typing.ClassVar[bool] = False
    streams: typing.ClassVar[bool] = False
    tuning_group: typing.ClassVar[type[ParamGroup] | None] = None

    @classmethod
    def program(cls, source: SourceSpec, asset: SoundAsset, variant: Variant, context: ProgramContext) -> BufferProgram:
        del source
        return BufferProgram(sample=context.decoder.load(asset.id, variant.id), loop=False)


class WavLoopModel:
    """Continuous playback of the variant's wav from the source's program start, looped when the source loops."""

    name: typing.ClassVar[str] = "wav_loop"
    continuous: typing.ClassVar[bool] = True
    streams: typing.ClassVar[bool] = False
    tuning_group: typing.ClassVar[type[ParamGroup] | None] = None

    @classmethod
    def program(cls, source: SourceSpec, asset: SoundAsset, variant: Variant, context: ProgramContext) -> BufferProgram:
        return BufferProgram(sample=context.decoder.load(asset.id, variant.id), loop=source.loop)
