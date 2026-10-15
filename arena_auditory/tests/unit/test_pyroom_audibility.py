from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from arena_rclpy_mixins.param_groups import configure
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind
from shapely.geometry import Polygon

from arena_auditory.materials import AcousticMaterialCatalog
from arena_auditory.params import BackendName, Level3Group, PortalGroup, PropagationGroup, RirGroup, WorldGroup
from arena_auditory.propagation import Emission, Listener, PortalConfig, PropagationConfig, PropagationScene, RirCache, RirConfig
from arena_auditory.propagation.pyroom import PyroomBackend
from arena_auditory.rooms import AcousticBoundarySpec, AcousticPortal, AcousticRoomSpec, AcousticWorldGraph, Point2D
from arena_auditory.shared import ListenerId
from arena_auditory.world import AcousticScene, AcousticWorld, AcousticZone, WorldConfig

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "acoustic_materials.yaml"
OPENING = ((4.0, 0.0), (4.0, 4.0))


def _on_edge(point: Point2D, start: Point2D, end: Point2D) -> bool:
    return (start[0] == end[0] == point[0] and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])) or (start[1] == end[1] == point[1] and min(start[0], end[0]) <= point[0] <= max(start[0], end[0]))


def _room(name: str, x_min: float, x_max: float, y_max: float) -> AcousticRoomSpec:
    corners = ((x_min, 0.0), (x_max, 0.0), (x_max, y_max), (x_min, y_max))
    boundary: list[AcousticBoundarySpec] = []
    for index, start in enumerate(corners):
        end = corners[(index + 1) % 4]
        cuts = sorted({start, end, *(point for point in OPENING if _on_edge(point, start, end))}, key=lambda point: math.dist(start, point))
        for first, second in zip(cuts, cuts[1:], strict=False):
            open_span = _on_edge((0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1])), *OPENING)
            boundary.append(AcousticBoundarySpec(start=first, end=second, material_id="Acoustic_Open" if open_span else "Acoustic_Default_Wall", kind="opening" if open_span else "wall"))
    return AcousticRoomSpec(zone_name=name, boundary=tuple(boundary), floor_material_id="Acoustic_Default_Floor", ceiling_material_id="Acoustic_Default_Ceiling", ceiling_height_m=3.0)


def _world() -> AcousticWorld:
    rooms = (_room("a", 0.0, 4.0, 4.0), _room("b", 4.0, 8.0, 10.0))
    portal = AcousticPortal(portal_id="opening:ab", door_name="", zone_a="a", zone_b="b", start=OPENING[0], end=OPENING[1], height_m=3.0, material_id="Acoustic_Open", portal_kind="opening")
    polygons = tuple((room.zone_name, Polygon(room.corners_xy)) for room in rooms)
    return AcousticWorld(
        name="corner",
        path=Path(),
        config=configure(WorldConfig, WorldGroup, PortalGroup),
        scene=AcousticScene(zones=tuple(AcousticZone(name=name, polygon=polygon, floor_material_id="Acoustic_Default_Floor") for name, polygon in polygons), walls=(), ceiling_height_m=3.0),
        rooms=rooms,
        graph=AcousticWorldGraph(rooms=rooms, portals=(portal,), zone_polygons=polygons),
        microphones=(),
        authored_origin=None,
    )


def test_audibility_follows_the_energy_the_impulse_renders_behind_a_corner() -> None:
    pytest.importorskip("pyroomacoustics")
    config = configure(PropagationConfig, PropagationGroup, Level3Group, backend=BackendName.PYROOMACOUSTICS, rir=configure(RirConfig, RirGroup), portal=configure(PortalConfig, PortalGroup))
    backend = PyroomBackend(config, AcousticMaterialCatalog(CATALOG_PATH), RirCache(16))
    scene = PropagationScene(world=_world())
    listener = Listener.at(ListenerId.robot("jackal"), (6.0, 9.0, 0.4))
    receptions = [backend.propagate(Emission(source_id="ped/voice", agent_kind=AgentKind.PEDESTRIAN, agent_name="ped_1", position=(1.0, 2.0, 1.6), level_db=float(level)), listener, scene) for level in range(0, 80, 5)]

    rendered = []
    for reception in receptions:
        assert reception.impulse is not None
        rendered.append(reception.received_level_db + 10.0 * math.log10(float(np.sum(reception.impulse.samples.astype(np.float64) ** 2))))
        print(f"\ndirect {reception.received_level_db:6.2f} dB, rendered {rendered[-1]:6.2f} dB, corner {reception.route_loss_db:5.2f} dB, audible {reception.audible}")
    threshold = config.threshold_db
    assert [reception.audible for reception in receptions] == [level >= threshold - 0.1 for level in rendered]
    assert any(reception.audible and reception.received_level_db < threshold for reception in receptions)
    assert not receptions[0].audible
