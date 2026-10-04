from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from arena_auditory.assets import SoundAsset, parse_manifest, pattern_matches
from arena_auditory.materials import AcousticMaterialCatalog

PACKAGE = Path(__file__).resolve().parents[2]
FOOTSTEP_DIR = PACKAGE / "sounds" / "Common" / "Sound" / "footstep"


def _footstep() -> SoundAsset:
    asset, kinds = parse_manifest("footstep", FOOTSTEP_DIR, yaml.safe_load((FOOTSTEP_DIR / "footstep.yaml").read_text(encoding="utf-8")))
    assert kinds == {}
    return asset


@pytest.mark.parametrize(
    ("material_name", "variant_id"),
    [
        ("Common/Material/Walnut_Planks", "footstep_walnut_planks_01"),
        ("oak_planks", "footstep_oak_planks_01"),
        ("Common/Material/Oak_Planks", "footstep_oak_planks_01"),
        ("Marble_Tile_18", "footstep_marble_tile_01"),
        ("Concrete_Smooth", "footstep_smooth_concrete_01"),
        ("smooth concrete", "footstep_smooth_concrete_01"),
        ("Common/Material/Ceramic_Tile_6", "footstep_ceramic_tile_01"),
        ("CERAMIC_TILE_12", "footstep_ceramic_tile_01"),
        ("Ceramic_Tile_18", "footstep_ceramic_tile_01"),
        ("Common/Material/Porcelain_Tile_4", "footstep_ceramic_tile_01"),
        ("Carpet_Cream", "footstep_default_01"),
        ("Acoustic_Default_Floor", "footstep_default_01"),
        ("Cloak_Room_Floor", "footstep_default_01"),
        ("unknown_floor", "footstep_default_01"),
        ("", "footstep_default_01"),
    ],
)
def test_bundled_footstep_matches_the_floor_material(material_name: str, variant_id: str) -> None:
    footstep = _footstep()

    for seed in range(4):
        assert footstep.select(context={"floor": material_name}, seed=seed).id == variant_id


def test_footstep_without_a_floor_in_context_plays_the_default() -> None:
    assert _footstep().select(context={}, seed=11).id == "footstep_default_01"


def test_bundled_footstep_is_a_floor_surface_pedestrian_clip() -> None:
    footstep = _footstep()

    assert footstep.kind == "footstep"
    assert footstep.surface == "floor"
    assert not footstep.loop
    assert footstep.level_db == 45.0
    assert {variant.model for variant in footstep.variants} == {"wav"}
    assert all(variant.path is not None and variant.path.is_file() for variant in footstep.variants)


@pytest.mark.parametrize(
    ("pattern", "value", "matches"),
    [
        ("oak", "Common/Material/Oak_Planks", True),
        ("oak", "Cloak_Room_Floor", False),
        ("ceramic_*", "Common/Material/Ceramic_Tile_6", True),
        ("ceramic_?", "Ceramic_Tile_6", False),
        ("oak", "", False),
    ],
)
def test_pattern_matches_whole_words_or_globs_on_the_leaf(pattern: str, value: str, matches: bool) -> None:
    assert pattern_matches(pattern, value) is matches


def test_floor_surface_damping_follows_the_material_absorption(tmp_path: Path) -> None:
    catalog_path = tmp_path / "acoustic_materials.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "octave_bands_hz": [125, 250, 500, 1000, 2000, 4000],
                "defaults": {
                    "canonical_name": "default",
                    "absorption": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
                    "transmission_loss_db": [12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
                    "scattering": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "materials": {
                    "Concrete_Smooth": {
                        "canonical_name": "concrete_floor_smooth",
                        "absorption": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
                        "transmission_loss_db": [18.0, 18.0, 18.0, 18.0, 18.0, 18.0],
                        "scattering": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                    "Walnut_Planks": {
                        "canonical_name": "walnut_planks",
                        "absorption": [0.14, 0.12, 0.10, 0.09, 0.08, 0.07],
                        "transmission_loss_db": [14.0, 14.0, 14.0, 14.0, 14.0, 14.0],
                        "scattering": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    catalog = AcousticMaterialCatalog(catalog_path)

    concrete_floor = catalog.surface_damping_db("Concrete_Smooth", "floor")
    walnut_floor = catalog.surface_damping_db("Walnut_Planks", "floor")
    concrete_wall = catalog.surface_damping_db("Concrete_Smooth", "wall")

    assert 0.0 < concrete_floor <= 6.0
    assert walnut_floor > concrete_floor
    assert concrete_wall > concrete_floor
    assert concrete_wall > 0.0
