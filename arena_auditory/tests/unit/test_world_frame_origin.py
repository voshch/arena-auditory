from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.tree.World import MultiLevelWorldView

from arena_auditory.world import compact_authored_world


def _single_level_world(root: Path) -> MultiLevelWorldView:
    level_dir = root / "single" / "0"
    level_dir.mkdir(parents=True)
    zone = {"name": "room", "corners": [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]}
    (level_dir / "world.yaml").write_text(yaml.safe_dump({"zones": [zone]}), encoding="utf-8")
    (level_dir / "map.yaml").write_text(yaml.safe_dump({"image": "map.png", "resolution": 0.05, "origin": [-99.0, -42.0, 0.0]}), encoding="utf-8")
    return MultiLevelWorldView(root / "single")


def test_single_level_origin_follows_the_runtime_rendered_map(tmp_path: Path) -> None:
    world_view = _single_level_world(tmp_path)

    world, origin = compact_authored_world(world_view, world_view.load())

    assert origin == pytest.approx(world.render_grid()[1])
    assert origin != pytest.approx((-99.0, -42.0))


def test_disk_map_source_uses_the_level_map_yaml_origin(tmp_path: Path) -> None:
    world_view = _single_level_world(tmp_path)

    _, origin = compact_authored_world(world_view, world_view.load(), disk_map=True)

    assert origin == pytest.approx((-99.0, -42.0))
