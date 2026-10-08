from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from PIL import Image, ImageDraw

TOOLS = Path(__file__).parents[2] / "tools"


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scenario_generator = _load_tool("scenario_generator")
generate_acoustics_scenarios = _load_tool("generate_acoustics_scenarios")

RESOLUTION = 0.05
MAP_SIZE_M = 12.0
CORRIDOR_LEGS = ((1.0, 1.0, 11.0, 3.2), (8.8, 1.0, 11.0, 11.0))


@pytest.fixture
def world_dir(tmp_path: Path) -> Path:
    level = tmp_path / "l_corridor" / "0"
    level.mkdir(parents=True)
    pixels = int(MAP_SIZE_M / RESOLUTION)
    image = Image.new("L", (pixels, pixels), 0)
    draw = ImageDraw.Draw(image)
    zones = []
    for x0, y0, x1, y1 in CORRIDOR_LEGS:
        draw.rectangle(
            [int(x0 / RESOLUTION), pixels - int(y1 / RESOLUTION), int(x1 / RESOLUTION) - 1, pixels - int(y0 / RESOLUTION) - 1],
            fill=255,
        )
        zones.append({"corners": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}]})
    image.save(level / "map.png")
    map_yaml = {"image": "map.png", "resolution": RESOLUTION, "origin": [0.0, 0.0, 0.0], "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196}
    (level / "map.yaml").write_text(yaml.safe_dump(map_yaml), encoding="utf-8")
    (level / "world.yaml").write_text(yaml.safe_dump({"zones": zones}), encoding="utf-8")
    return tmp_path / "l_corridor"


def _matrix(world_dir: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    scenarios = {scenario.name: scenario.data for scenario in scenario_generator.plan_world(world_dir)}
    return {(state, count, direction): scenarios[scenario_generator.scenario_name(world_dir.name, state, count, direction)] for state in scenario_generator.ROBOT_STATES for count in scenario_generator.PEDESTRIAN_COUNTS for direction in scenario_generator.END_DIRECTIONS}


def test_direction_changes_pedestrians_without_moving_robot(world_dir: Path) -> None:
    route = scenario_generator.derive_world_route(world_dir).points_a_to_b
    matrix = _matrix(world_dir)

    assert matrix["idle", 3, "a-to-b"]["robots"][0] == matrix["idle", 3, "b-to-a"]["robots"][0]
    assert matrix["moving", 3, "a-to-b"]["robots"][0] == matrix["moving", 3, "b-to-a"]["robots"][0]
    assert math.dist(matrix["idle", 3, "a-to-b"]["dynamic"][0]["pose"][:2], route[0]) < 2.0
    assert math.dist(matrix["idle", 3, "b-to-a"]["dynamic"][0]["pose"][:2], route[-1]) < 1.0


def test_three_pedestrians_are_separate_and_have_immediate_moving_targets(world_dir: Path) -> None:
    people = _matrix(world_dir)["moving", 3, "a-to-b"]["dynamic"]
    clearance = 2 * scenario_generator.PEDESTRIAN_CLEARANCE_M

    assert len({tuple(person["pose"][:2]) for person in people}) == 3
    assert all(math.dist(left["pose"][:2], right["pose"][:2]) >= clearance for left, right in zip(people, people[1:], strict=False))
    assert all(math.dist(left["waypoints"][-1][:2], right["waypoints"][-1][:2]) >= clearance for left, right in zip(people, people[1:], strict=False))
    assert all(person["velocity"] > 0.0 for person in people)
    assert all(person["agent"]["desired_velocity"] > 0.0 for person in people)
    assert all(person["waypoint_mode"] == "reverse" for person in people)
    assert all(math.dist(person["pose"][:2], person["waypoints"][0][:2]) >= 0.10 for person in people)


def test_regenerating_written_scenarios_is_stable(world_dir: Path) -> None:
    assert scenario_generator.write_scenarios(scenario_generator.plan_world(world_dir)) == (12, 0)
    assert scenario_generator.write_scenarios(scenario_generator.plan_world(world_dir)) == (0, 12)


def test_all_generated_pedestrians_use_bundled_model(world_dir: Path) -> None:
    models = {person["model"] for scenario in _matrix(world_dir).values() for person in scenario["dynamic"]}
    assert models == {"arenian"}


def test_generator_cli_dry_run_writes_nothing_and_write_creates_the_matrix(world_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = str(world_dir.parent)

    assert generate_acoustics_scenarios.main(["--worlds-root", root]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert not (world_dir / "scenarios").exists()
    assert generate_acoustics_scenarios.main(["--worlds-root", root, "--world", "l_*", "--write"]) == 0
    written = json.loads(capsys.readouterr().out)

    assert (dry_run["mode"], dry_run["scenario_count"], dry_run["written"]) == ("dry-run", 12, 0)
    assert (written["mode"], written["worlds"], written["written"], written["unchanged"]) == ("write", ["l_corridor"], 12, 0)
    assert len(list((world_dir / "scenarios").glob("*/scenario.yaml"))) == 12


def test_generator_cli_rejects_unmatched_world_glob(world_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert generate_acoustics_scenarios.main(["--worlds-root", str(world_dir.parent), "--world", "missing_*"]) == 2
    assert "no acoustics worlds matched: missing_*" in capsys.readouterr().err


def test_writing_refuses_to_overwrite_a_changed_scenario(world_dir: Path) -> None:
    scenarios = scenario_generator.plan_world(world_dir)
    scenario_generator.write_scenarios(scenarios)
    scenarios[0].path.write_text("edited: true\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        scenario_generator.write_scenarios(scenarios)
