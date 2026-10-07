from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from arena_rclpy_mixins.param_groups import configure
from arena_robots.audio import SPEED_OF_SOUND_MPS
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind

from arena_auditory.materials import AcousticMaterialCatalog
from arena_auditory.params import BackendName, Level3Group, PortalGroup, PropagationGroup, RirGroup
from arena_auditory.propagation import Emission, Listener, PortalConfig, PropagationConfig, PropagationScene, Propagator, RirConfig
from arena_auditory.propagation.self_path import MIN_SELF_DISTANCE_M, SelfPathBackend
from arena_auditory.shared import ListenerId
from arena_auditory.world import OccupancyMap

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "acoustic_materials.yaml"


def _config(backend: BackendName = BackendName.LEGACY) -> PropagationConfig:
    return configure(
        PropagationConfig,
        PropagationGroup,
        Level3Group,
        backend=backend,
        rir=configure(RirConfig, RirGroup),
        portal=configure(PortalConfig, PortalGroup),
    )


def _wall_between_x_1_and_x_1_1() -> OccupancyMap:
    data = np.zeros((20, 30), dtype=np.int8)
    data[:, 10] = 100
    return OccupancyMap(frame_id="map", resolution_m=0.1, origin_xy=(0.0, 0.0), origin_yaw=0.0, data=data, occupied_threshold=50)


def _motor(robot: str = "jackal", position: tuple[float, float, float] = (0.5, 0.5, 0.3), level_db: float = 70.0) -> Emission:
    return Emission(source_id=f"{robot}/motor", agent_kind=AgentKind.ROBOT, agent_name=robot, position=position, level_db=level_db)


def test_self_path_level_is_inverse_distance_from_the_1m_level() -> None:
    config = _config()
    listener = Listener.at(ListenerId.array_mic("jackal", "front_left"), (2.5, 0.5, 0.3))

    reception = SelfPathBackend(config).propagate(_motor(), listener, PropagationScene())

    assert reception.backend == "self_direct_path"
    assert reception.distance_m == pytest.approx(2.0)
    assert reception.received_level_db == pytest.approx(70.0 - 20.0 * math.log10(2.0))
    assert reception.direct_delay_s == pytest.approx(2.0 / SPEED_OF_SOUND_MPS)
    assert reception.threshold_db == config.threshold_db
    assert reception.audible is (reception.received_level_db >= config.threshold_db)
    assert reception.occluded is False
    assert reception.rir_key == ""
    assert reception.impulse is None
    assert reception.early_paths == ()


def test_self_path_distance_is_clamped_at_the_minimum_self_distance() -> None:
    listener = Listener.at(ListenerId.robot("jackal"), (0.5, 0.5, 0.3))

    reception = SelfPathBackend(_config()).propagate(_motor(), listener, PropagationScene())

    assert reception.distance_m == pytest.approx(MIN_SELF_DISTANCE_M)
    assert reception.received_level_db == pytest.approx(70.0 - 20.0 * math.log10(MIN_SELF_DISTANCE_M))
    assert reception.direct_delay_s == pytest.approx(MIN_SELF_DISTANCE_M / SPEED_OF_SOUND_MPS)


def test_self_path_ignores_occupancy_between_source_and_microphone() -> None:
    listener = Listener.at(ListenerId.array_mic("jackal", "rear_left"), (1.5, 0.5, 0.3))
    scene = PropagationScene(occupancy=_wall_between_x_1_and_x_1_1())

    reception = SelfPathBackend(_config()).propagate(_motor(), listener, scene)

    assert scene.occupancy is not None
    assert scene.occupancy.occluded((0.5, 0.5, 0.3), (1.5, 0.5, 0.3))
    assert reception.occluded is False
    assert reception.received_level_db == pytest.approx(70.0)


@pytest.mark.parametrize("backend", [BackendName.LEGACY, BackendName.LEVEL3])
@pytest.mark.parametrize("listener_id", [ListenerId.robot("jackal"), ListenerId.array_mic("jackal", "front_left")])
def test_propagator_routes_a_robots_own_source_to_its_listeners_through_the_self_path(backend: BackendName, listener_id: str) -> None:
    propagator = Propagator(_config(backend), AcousticMaterialCatalog(CATALOG_PATH))
    propagator.set_scene(PropagationScene(occupancy=_wall_between_x_1_and_x_1_1()))

    reception = propagator.propagate(_motor(), Listener.at(listener_id, (1.5, 0.5, 0.3)))

    assert reception.backend == "self_direct_path"
    assert reception.occluded is False
    assert reception.used_fallback is False
    assert reception.received_level_db == pytest.approx(70.0)
    assert reception.impulse is None


def test_propagator_uses_the_backend_for_another_robots_source() -> None:
    config = _config()
    propagator = Propagator(config, AcousticMaterialCatalog(CATALOG_PATH))
    propagator.set_scene(PropagationScene(occupancy=_wall_between_x_1_and_x_1_1()))

    reception = propagator.propagate(_motor("other"), Listener.at(ListenerId.array_mic("jackal", "front_left"), (1.5, 0.5, 0.3)))

    assert reception.backend == "legacy_distance_occlusion"
    assert reception.occluded is True
    assert reception.received_level_db == pytest.approx(70.0 - config.occlusion_db)


def test_propagator_uses_the_backend_for_a_pedestrian_source_at_a_robot_listener() -> None:
    propagator = Propagator(_config(), AcousticMaterialCatalog(CATALOG_PATH))
    emission = Emission(source_id="ped/footstep", agent_kind=AgentKind.PEDESTRIAN, agent_name="jackal", position=(0.5, 0.5, 0.3), level_db=70.0)

    reception = propagator.propagate(emission, Listener.at(ListenerId.robot("jackal"), (2.5, 0.5, 0.3)))

    assert reception.backend == "legacy_distance_occlusion"
