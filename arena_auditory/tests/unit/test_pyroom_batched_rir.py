from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from arena_rclpy_mixins.param_groups import configure
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind

from arena_auditory.materials import AcousticMaterialCatalog, default_catalog
from arena_auditory.params import BackendName, Level3Group, PortalGroup, PropagationGroup, RirGroup, WorldGroup
from arena_auditory.propagation import LISTENER_ORDER, Emission, Listener, PortalConfig, PropagationConfig, PropagationScene, Propagator, RirConfig
from arena_auditory.propagation.pyroom_adapter import PyroomacousticsAdapter, RirUnavailable, RoomImpulseResponse
from arena_auditory.rooms import AcousticBoundarySpec, AcousticRoomSpec
from arena_auditory.shared import ListenerId, ListenerKind

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "acoustic_materials.yaml"

SOURCES = ((1.0, 1.0, 0.05), (4.2, 1.3, 1.6), (1.2, 4.1, 0.9))
LISTENERS = ((1.6, 1.2, 0.4), (1.6, 1.4, 0.4), (4.5, 1.6, 0.4), (1.5, 4.5, 2.2), (0.6, 3.3, 0.4))


def _l_room() -> AcousticRoomSpec:
    corners = ((0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (2.0, 2.0), (2.0, 5.0), (0.0, 5.0))
    materials = ("Acoustic_Default_Wall", "Plaster_Wall", "Acoustic_Default_Wall", "Concrete_Smooth", "Acoustic_Default_Wall", "Oak_Planks")
    boundary = tuple(AcousticBoundarySpec(start=start, end=end, material_id=material, kind="wall") for start, end, material in zip(corners, (*corners[1:], corners[0]), materials, strict=True))
    return AcousticRoomSpec(zone_name="l_room", boundary=boundary, floor_material_id="Acoustic_Default_Floor", ceiling_material_id="Acoustic_Default_Ceiling", ceiling_height_m=3.0)


def _adapter() -> PyroomacousticsAdapter:
    pytest.importorskip("pyroomacoustics")
    return PyroomacousticsAdapter(AcousticMaterialCatalog(CATALOG_PATH), configure(RirConfig, RirGroup))


def _fresh_single_listener_samples(room: AcousticRoomSpec, source: tuple[float, float, float], listener: tuple[float, float, float]) -> np.ndarray:
    built = _adapter().build_room(room).room
    built.add_source(np.asarray(source, dtype=np.float64))
    built.add_microphone(np.asarray(listener, dtype=np.float64))
    built.compute_rir()
    return np.asarray(built.rir[0][0], dtype=np.float64)


def test_batched_rirs_on_a_reused_room_equal_fresh_single_listener_runs() -> None:
    room = _l_room()
    adapter = _adapter()
    assert adapter.build_room(room).obstructing_walls
    adapter.compute_rir(room, source_position_m=(0.5, 0.5, 1.0), listener_position_m=(0.5, 4.5, 1.0))

    for source in SOURCES:
        batched = adapter.compute_rirs(room, source_position_m=source, listener_positions_m=LISTENERS)
        for listener, rir in zip(LISTENERS, batched, strict=True):
            assert isinstance(rir, RoomImpulseResponse)
            reference = _fresh_single_listener_samples(room, source, listener)
            assert rir.samples.shape == reference.shape
            np.testing.assert_allclose(rir.samples, reference, rtol=0.0, atol=1e-9)
    assert adapter.cache_misses == 1 + len(SOURCES) * len(LISTENERS)


def test_compute_rirs_keeps_the_per_pair_cache_of_sequential_compute_rir() -> None:
    room = _l_room()
    source = SOURCES[0]
    listeners = (LISTENERS[0], LISTENERS[2], (1.62, 1.21, 0.4), LISTENERS[3])
    sequential = _adapter()
    expected = [sequential.compute_rir(room, source_position_m=source, listener_position_m=listener) for listener in listeners]
    batched = _adapter()

    results = batched.compute_rirs(room, source_position_m=source, listener_positions_m=listeners)

    assert (batched.cache_hits, batched.cache_misses, batched.cache_entries) == (sequential.cache_hits, sequential.cache_misses, sequential.cache_entries) == (1, 3, 3)
    for rir, reference in zip(results, expected, strict=True):
        assert isinstance(rir, RoomImpulseResponse)
        np.testing.assert_allclose(rir.samples, reference.samples, rtol=0.0, atol=1e-9)
    for listener, rir in zip(listeners, results, strict=True):
        np.testing.assert_array_equal(batched.compute_rir(room, source_position_m=source, listener_position_m=listener).samples, rir.samples)
    assert batched.cache_hits == 1 + len(listeners)


def test_listener_outside_the_room_fails_alone_in_a_batch() -> None:
    room = _l_room()
    adapter = _adapter()

    results = adapter.compute_rirs(room, source_position_m=SOURCES[0], listener_positions_m=(LISTENERS[0], (4.0, 4.0, 0.4), LISTENERS[2]))

    assert isinstance(results[0], RoomImpulseResponse)
    assert isinstance(results[1], RirUnavailable)
    assert isinstance(results[2], RoomImpulseResponse)
    np.testing.assert_allclose(results[2].samples, _fresh_single_listener_samples(room, SOURCES[0], LISTENERS[2]), rtol=0.0, atol=1e-9)


def _config(backend: BackendName) -> PropagationConfig:
    return configure(PropagationConfig, PropagationGroup, Level3Group, backend=backend, rir=configure(RirConfig, RirGroup), portal=configure(PortalConfig, PortalGroup))


def _listeners(x: float, y: float) -> list[Listener]:
    return [
        Listener.at(ListenerId.agent(7), (x - 1.0, y, 1.6)),
        Listener.at(ListenerId.viewport_mic("down_projection"), (x, y + 1.0, 2.0)),
        Listener.at(ListenerId.robot("jackal"), (x, y, 0.4)),
        Listener.at(ListenerId.agent(3), (x + 1.0, y, 1.6)),
        Listener.at(ListenerId.array_mic("jackal", "left"), (x, y + 0.1, 0.4)),
        Listener.at(ListenerId.array_mic("jackal", "right"), (x, y - 0.1, 0.4)),
    ]


def test_propagate_many_yields_array_robot_microphone_then_agent_listeners() -> None:
    propagator = Propagator(_config(BackendName.LEGACY), AcousticMaterialCatalog(CATALOG_PATH))
    emission = Emission(source_id="ped/footstep", agent_kind=AgentKind.PEDESTRIAN, agent_name="ped_1", position=(0.0, 0.0, 0.05), level_db=60.0)
    listeners = _listeners(3.0, 2.0)

    yielded = list(propagator.propagate_many(emission, listeners))

    assert [listener.id for listener, _ in yielded] == [
        ListenerId.array_mic("jackal", "left"),
        ListenerId.array_mic("jackal", "right"),
        ListenerId.robot("jackal"),
        ListenerId.viewport_mic("down_projection"),
        ListenerId.agent(7),
        ListenerId.agent(3),
    ]
    assert [LISTENER_ORDER.index(listener.kind) for listener, _ in yielded] == sorted(LISTENER_ORDER.index(listener.kind) for listener in listeners)
    assert [reception for _, reception in yielded] == [propagator.propagate(emission, listener) for listener, _ in yielded]


def test_propagate_many_in_a_pyroom_world_matches_propagate_per_listener() -> None:
    pytest.importorskip("pyroomacoustics")
    from arena_auditory.world import AcousticWorld, WorldConfig

    world = AcousticWorld.load("demo", configure(WorldConfig, WorldGroup, PortalGroup))
    min_x, min_y, _, _ = world.zone_named("demo_zone").polygon.bounds
    emission = Emission(source_id="ped/footstep", agent_kind=AgentKind.PEDESTRIAN, agent_name="ped_1", position=(min_x + 2.0, min_y + 2.5, 0.05), level_db=60.0)
    listeners = _listeners(min_x + 6.0, min_y + 4.0)
    batched = Propagator(_config(BackendName.PYROOMACOUSTICS), default_catalog())
    batched.set_scene(PropagationScene(world=world))
    single = Propagator(_config(BackendName.PYROOMACOUSTICS), default_catalog())
    single.set_scene(PropagationScene(world=world))

    yielded = list(batched.propagate_many(emission, listeners))

    rendered = [(listener, reception) for listener, reception in yielded if listener.kind is not ListenerKind.AGENT]
    assert {reception.backend for _, reception in rendered} == {"pyroomacoustics_same_room"}
    for listener, reception in yielded:
        reference = single.propagate(emission, listener)
        assert reception == reference
        if reference.impulse is not None:
            assert reception.impulse is not None
            np.testing.assert_allclose(reception.impulse.samples, reference.impulse.samples, rtol=0.0, atol=1e-9)
