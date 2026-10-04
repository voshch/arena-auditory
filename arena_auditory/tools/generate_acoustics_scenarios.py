"""Generate the acoustics suite scenarios, run from a source checkout: python3 tools/generate_acoustics_scenarios.py --worlds-root <dir>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scenario_generator import plan, write_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_acoustics_scenarios",
        description=("Generate exactly 12 scenarios per world: idle/moving robot, 1/2/3 pedestrians, and both pedestrian walking directions with the robot at end A."),
    )
    parser.add_argument(
        "--worlds-root",
        type=Path,
        required=True,
        help="acoustics worlds directory, e.g. the acoustics suite bundle's worlds/",
    )
    parser.add_argument(
        "--world",
        action="append",
        default=[],
        metavar="NAME_OR_GLOB",
        help="generate only matching worlds; repeatable (does not add a scenario variant)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write scenario.yaml files; without this flag the command is a dry run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenarios = plan(args.worlds_root, args.world)
        written, unchanged = write_scenarios(scenarios) if args.write else (0, 0)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    worlds = sorted({scenario.world_dir.name for scenario in scenarios})
    print(
        json.dumps(
            {
                "mode": "write" if args.write else "dry-run",
                "world_count": len(worlds),
                "scenario_count": len(scenarios),
                "scenarios_per_world": 12,
                "worlds": worlds,
                "written": written,
                "unchanged": unchanged,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
