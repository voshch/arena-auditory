from __future__ import annotations

import math
from pathlib import Path

import pytest
from arena_robots.audio import SPEED_OF_SOUND_MPS

from arena_auditory.materials import AcousticMaterialCatalog
from arena_auditory.propagation.level3 import Level3Propagation
from arena_auditory.world import AcousticScene, AcousticWall

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "acoustic_materials.yaml"


def _level3(catalog: AcousticMaterialCatalog) -> Level3Propagation:
    return Level3Propagation(catalog, max_reflections=8, reflection_floor_db=-60.0)


def test_direct_delay_uses_physical_distance_below_attenuation_floor() -> None:
    scene = AcousticScene(zones=(), walls=(), ceiling_height_m=3.0)

    result = _level3(AcousticMaterialCatalog(CATALOG_PATH)).calculate(scene, (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), 60.0)

    assert result.direct_delay_s == pytest.approx(0.2 / SPEED_OF_SOUND_MPS)
    assert result.paths[0].delay_s == pytest.approx(0.2 / SPEED_OF_SOUND_MPS)
    assert result.paths[0].gain_db == pytest.approx(0.0)


def test_reflection_behind_wall_pays_wall_transmission_loss() -> None:
    catalog = AcousticMaterialCatalog(CATALOG_PATH)
    transmission_loss = catalog.surface_damping_db("Acoustic_Default_Wall", "wall")
    scene = AcousticScene(
        zones=(),
        walls=(
            AcousticWall(start=(5.0, 0.0), end=(5.0, 10.0), material_id="Acoustic_Default_Wall"),
            AcousticWall(start=(0.0, 0.0), end=(0.0, 10.0), material_id="Acoustic_Default_Wall"),
        ),
        ceiling_height_m=3.0,
    )

    result = _level3(catalog).calculate(scene, (2.0, 5.0, 0.0), (8.0, 6.0, 0.0), 60.0)

    free_field_level = 60.0 - 20.0 * math.log10(math.dist((2.0, 5.0), (8.0, 6.0)))
    reflection = next(path for path in result.paths if path.interaction_type == "reflection")
    assert reflection.reflection_point == pytest.approx((0.0, 5.0 + 2.0 / 10.0))
    assert free_field_level - result.received_volume_db >= transmission_loss - 2.0
