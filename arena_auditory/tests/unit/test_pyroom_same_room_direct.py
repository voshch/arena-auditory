from __future__ import annotations

import math
from pathlib import Path

import attrs
import pytest
from arena_rclpy_mixins.param_groups import configure
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind
from shapely.geometry import Polygon

from arena_auditory.materials import AcousticMaterialCatalog
from arena_auditory.params import BackendName, Level3Group, PortalGroup, PropagationGroup, RirGroup, WorldGroup
from arena_auditory.propagation import Emission, Listener, PortalConfig, PropagationConfig, PropagationScene, RirCache, RirConfig
from arena_auditory.propagation.pyroom import PyroomBackend
from arena_auditory.propagation.pyroom_adapter import PyroomacousticsAdapter, direct_arrival
from arena_auditory.rooms import AcousticBoundarySpec, AcousticRoomSpec, AcousticWorldGraph
from arena_auditory.shared import ListenerId
from arena_auditory.world import AcousticScene, AcousticWorld, AcousticZone, WorldConfig

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "acoustic_materials.yaml"
AXIS_SOURCE = (2.0, 1.0, 1.5)


def _corridor(material_id: str) -> AcousticRoomSpec:
    corners = ((0.0, 0.0), (24.0, 0.0), (24.0, 2.0), (0.0, 2.0))
    boundary = tuple(AcousticBoundarySpec(start=start, end=corners[(index + 1) % 4], material_id=material_id, kind="wall") for index, start in enumerate(corners))
    return AcousticRoomSpec(zone_name="corridor", boundary=boundary, floor_material_id=material_id, ceiling_material_id=material_id, ceiling_height_m=3.0)


def _world(room: AcousticRoomSpec) -> AcousticWorld:
    polygon = Polygon(room.corners_xy)
    return AcousticWorld(
        name="corridor",
        path=Path(),
        config=configure(WorldConfig, WorldGroup, PortalGroup),
        scene=AcousticScene(zones=(AcousticZone(name=room.zone_name, polygon=polygon, floor_material_id=room.floor_material_id),), walls=(), ceiling_height_m=room.ceiling_height_m),
        rooms=(room,),
        graph=AcousticWorldGraph(rooms=(room,), portals=(), zone_polygons=((room.zone_name, polygon),)),
        microphones=(),
        authored_origin=None,
    )


def _backend() -> PyroomBackend:
    pytest.importorskip("pyroomacoustics")
    config = configure(PropagationConfig, PropagationGroup, Level3Group, backend=BackendName.PYROOMACOUSTICS, rir=configure(RirConfig, RirGroup), portal=configure(PortalConfig, PortalGroup))
    return PyroomBackend(config, AcousticMaterialCatalog(CATALOG_PATH), RirCache(16))


@pytest.mark.parametrize("material_id", ["Acoustic_Default_Wall", "Concrete_Smooth"])
@pytest.mark.parametrize("length_m", [4.0, 8.0, 12.0, 16.0, 20.0])
def test_same_room_reception_on_a_symmetric_corridor_axis_reads_the_anechoic_direct(material_id: str, length_m: float) -> None:
    backend = _backend()
    emission = Emission(source_id="ped/voice", agent_kind=AgentKind.PEDESTRIAN, agent_name="ped_1", position=AXIS_SOURCE, level_db=60.0)
    position = (AXIS_SOURCE[0] + length_m, AXIS_SOURCE[1], AXIS_SOURCE[2])
    anechoic = PyroomacousticsAdapter(AcousticMaterialCatalog(CATALOG_PATH), configure(RirConfig, RirGroup, max_order=0))
    free = direct_arrival(attrs.evolve(anechoic.compute_rir(_corridor(material_id), source_position_m=AXIS_SOURCE, listener_position_m=position), direct=None))

    reception = backend.propagate(emission, Listener.at(ListenerId.robot("jackal"), position), PropagationScene(world=_world(_corridor(material_id))))

    level_error_db = reception.received_level_db - emission.level_db - free.gain_db
    delay_error_samples = (reception.direct_delay_s - free.delay_s) * anechoic.config.sample_rate_hz
    print(f"\n{material_id} {length_m:4.1f} m: level {level_error_db:+.2f} dB, delay {delay_error_samples:+.2f} samples against the anechoic direct, which is {free.gain_db - 20.0 * math.log10(1.0 / length_m):+.2f} dB against 1/r")
    assert abs(level_error_db) <= 0.2
    assert abs(delay_error_samples) <= 0.3


@pytest.mark.parametrize("length_m", [0.5, 2.0, 8.0, 20.0])
@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5])
def test_known_delay_arrival_matches_the_windowed_level_of_a_lone_direct(length_m: float, fraction: float) -> None:
    pytest.importorskip("pyroomacoustics")
    adapter = PyroomacousticsAdapter(AcousticMaterialCatalog(CATALOG_PATH), configure(RirConfig, RirGroup, max_order=0))
    sample_period_m = adapter.speed_of_sound_mps / adapter.config.sample_rate_hz
    distance = (round(length_m / sample_period_m) + fraction) * sample_period_m
    listener = (AXIS_SOURCE[0] + distance, AXIS_SOURCE[1], AXIS_SOURCE[2])

    rir = adapter.compute_rir(_corridor("Acoustic_Default_Wall"), source_position_m=AXIS_SOURCE, listener_position_m=listener)

    assert rir.direct is not None
    windowed = direct_arrival(attrs.evolve(rir, direct=None))
    print(f"\n{length_m:4.1f} m + {fraction:.2f} sample: main lobe {rir.direct.gain_db - windowed.gain_db:+.3f} dB against the windowed level")
    assert rir.direct.delay_s == pytest.approx(distance / adapter.speed_of_sound_mps, rel=1e-12)
    assert rir.direct.delay_s == pytest.approx(windowed.delay_s, abs=0.3 / adapter.config.sample_rate_hz)
    assert rir.direct.gain_db == pytest.approx(windowed.gain_db, abs=0.15)


def test_direct_hidden_behind_a_room_corner_is_located_in_the_impulse() -> None:
    pytest.importorskip("pyroomacoustics")
    corners = ((0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (2.0, 2.0), (2.0, 5.0), (0.0, 5.0))
    boundary = tuple(AcousticBoundarySpec(start=start, end=end, material_id="Acoustic_Default_Wall", kind="wall") for start, end in zip(corners, (*corners[1:], corners[0]), strict=True))
    room = AcousticRoomSpec(zone_name="l_room", boundary=boundary, floor_material_id="Acoustic_Default_Floor", ceiling_material_id="Acoustic_Default_Ceiling", ceiling_height_m=3.0)
    adapter = PyroomacousticsAdapter(AcousticMaterialCatalog(CATALOG_PATH), configure(RirConfig, RirGroup))

    hidden, visible = adapter.compute_rirs(room, source_position_m=(1.2, 4.1, 0.9), listener_positions_m=((4.5, 1.6, 0.4), (1.0, 1.0, 0.4)))

    assert not isinstance(hidden, Exception)
    assert not isinstance(visible, Exception)
    assert hidden.direct is None
    assert direct_arrival(hidden).amplitude > 0.0
    assert visible.direct is not None
