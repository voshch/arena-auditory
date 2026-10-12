from __future__ import annotations

from pathlib import Path

import yaml
from arena_auditory.acoustic_audit import audit_world, audit_world_view
from arena_simulation_setup.tree.World import WorldView
from PIL import Image


def _write_level(world: Path, zones: list[dict[str, object]], *, with_map: bool = True) -> WorldView:
    world.mkdir(parents=True)
    (world / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}), encoding="utf-8")
    if with_map:
        (world / "map").mkdir()
        Image.new("L", (120, 80), color=255).save(world / "map" / "0.png")
        (world / "map" / "0.yaml").write_text(yaml.safe_dump({"image": "0.png", "resolution": 0.05, "origin": [0.0, 0.0, 0.0]}), encoding="utf-8")
    return WorldView(world)


def test_audit_reports_coverage_and_portal_connectivity() -> None:
    airport = audit_world("airport", stride_cells=10)
    hospital = audit_world("hospital_1", stride_cells=10)

    assert airport.complete
    assert airport.sampled_traversable_cells > 0
    assert hospital.rooms == 13
    assert hospital.door_portals == 12
    assert hospital.opening_portals == 1
    assert hospital.connected_components == 1


def test_audit_reads_maps_from_the_resolved_world_path(tmp_path: Path) -> None:
    zone = {"name": "room", "polygon": [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]}
    view = _write_level(tmp_path / "outside_share", [zone])

    report = audit_world_view("outside_share", view, stride_cells=10)

    assert report.map_issues == ()
    assert report.sampled_traversable_cells == 96
    assert report.uncovered_traversable_cells == 0
    assert report.complete


def test_audit_flags_free_cells_outside_zones_and_overlapping_zones(tmp_path: Path) -> None:
    left = {"name": "left", "polygon": [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0]]}
    middle = {"name": "middle", "polygon": [[2.0, 0.0], [4.0, 0.0], [4.0, 4.0], [2.0, 4.0]]}
    view = _write_level(tmp_path / "partial", [left, middle])

    report = audit_world_view("partial", view, stride_cells=10)

    assert report.overlapping_zone_pairs == (("left", "middle"),)
    assert report.sampled_traversable_cells == 96
    assert report.uncovered_traversable_cells == 32
    assert all(x > 4.0 for x, _ in report.uncovered_examples)
    assert not report.complete


def test_audit_reports_a_level_without_preview_map(tmp_path: Path) -> None:
    zone = {"name": "room", "polygon": [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]}
    view = _write_level(tmp_path / "unmapped", [zone], with_map=False)

    report = audit_world_view("unmapped", view, stride_cells=10)

    assert report.map_issues == ("level 0: missing preview map '0.yaml'",)
    assert report.sampled_traversable_cells == 0
    assert not report.complete
