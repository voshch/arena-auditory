from __future__ import annotations

import multiprocessing
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from arena_rclpy_mixins.params import configure
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind

from arena_auditory.materials import default_catalog
from arena_auditory.params import BackendName, Level3Group, PortalGroup, PropagationGroup, RirGroup
from arena_auditory.propagation import Emission, Listener, PortalConfig, PropagationConfig, PropagationScene, Propagator, RirConfig
from arena_auditory.propagation.process import Status, Task, run, serve
from arena_auditory.shared import ListenerId
from arena_auditory.world import OccupancyMap

ORPHAN_SCRIPT = """
import multiprocessing, os, pickle, sys

from arena_auditory.propagation.process import serve

if __name__ == "__main__":
    config, task = pickle.loads(bytes.fromhex(sys.stdin.read()))
    context = multiprocessing.get_context("spawn")
    tasks, outcomes = context.Queue(), context.Queue()
    now_ns, generation = context.Value("q", 0), context.Value("q", 0)
    child = context.Process(target=serve, args=(config, tasks, outcomes, now_ns, generation), daemon=True)
    child.start()
    tasks.put(task)
    outcomes.get(timeout=60.0)
    print(child.pid, flush=True)
    os._exit(0)
"""


def _running(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def _config() -> PropagationConfig:
    return configure(PropagationConfig, PropagationGroup, Level3Group, backend=BackendName.LEGACY, rir=configure(RirConfig, RirGroup), portal=configure(PortalConfig, PortalGroup))


def _task(job_id: int = 0, *, generation: int = 0, stamp_ns: int = 0, scene: PropagationScene | None = None) -> Task:
    emission = Emission(source_id="ped", agent_kind=AgentKind.PEDESTRIAN, agent_name="ped", position=(1.0, 1.0, 1.6), level_db=70.0)
    listener = Listener.at(ListenerId.robot("jackal"), (3.0, 1.0, 0.4))
    return Task(job_id=job_id, emission=emission, listeners=(listener,), config=_config(), generation=generation, scene=scene, stamp_ns=stamp_ns, max_age_ns=1_000_000_000)


def _scene() -> PropagationScene:
    return PropagationScene(occupancy=OccupancyMap(frame_id="map", resolution_m=0.1, origin_xy=(0.0, 0.0), origin_yaw=0.0, data=np.zeros((20, 30), dtype=np.int8), occupied_threshold=50))


def test_run_propagates_skips_an_earlier_generation_and_drops_a_late_event() -> None:
    propagator = Propagator(_config(), default_catalog())

    done = run(propagator, _task(7), now_ns=0, generation=0, warnings=[])
    assert (done.job_id, done.status) == (7, Status.DONE)
    assert [listener.id for listener, _ in done.receptions] == [ListenerId.robot("jackal")]

    assert run(propagator, _task(generation=1), now_ns=0, generation=2, warnings=[]).status is Status.SKIPPED
    assert run(propagator, _task(stamp_ns=1_000_000_000), now_ns=2_500_000_000, generation=0, warnings=[]).status is Status.STALE
    assert run(propagator, _task(stamp_ns=2_000_000_000), now_ns=2_500_000_000, generation=0, warnings=[]).status is Status.DONE


def test_run_takes_a_changed_scene_even_for_a_skipped_pass() -> None:
    propagator = Propagator(_config(), default_catalog())
    scene = _scene()

    assert run(propagator, _task(generation=0, scene=scene), now_ns=0, generation=1, warnings=[]).status is Status.SKIPPED
    assert propagator.scene is scene


def test_serve_answers_a_pickled_task_and_stops_at_the_sentinel() -> None:
    context = multiprocessing.get_context("spawn")
    tasks, outcomes = context.Queue(), context.Queue()
    now_ns, generation = context.Value("q", 0), context.Value("q", 0)
    child = context.Process(target=serve, args=(_config(), tasks, outcomes, now_ns, generation), daemon=True)
    child.start()
    try:
        tasks.put(pickle.dumps(_task(3)))
        outcome = pickle.loads(outcomes.get(timeout=60.0))
        assert (outcome.job_id, outcome.status) == (3, Status.DONE)
        tasks.put(None)
        child.join(timeout=10.0)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
        tasks.cancel_join_thread()


def test_serve_exits_when_its_parent_dies_without_cleanup(tmp_path: Path) -> None:
    script = tmp_path / "parent.py"
    script.write_text(ORPHAN_SCRIPT)
    parent = subprocess.run([sys.executable, str(script)], input=pickle.dumps((_config(), pickle.dumps(_task()))).hex(), capture_output=True, text=True, timeout=120, check=False)
    assert parent.returncode == 0, parent.stderr
    pid = int(parent.stdout.split()[0])
    deadline = time.monotonic() + 10.0
    while _running(pid):
        assert time.monotonic() < deadline, "the propagation process outlived its parent"
        time.sleep(0.1)
