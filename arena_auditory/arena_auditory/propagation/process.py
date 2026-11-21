"""Propagation in processes of their own, so room impulse responses never wait on the node's GIL."""

from __future__ import annotations

import enum
import multiprocessing
import pickle
import queue
import signal
from multiprocessing.queues import Queue
from multiprocessing.sharedctypes import Synchronized

import attrs

from arena_auditory.materials import default_catalog
from arena_auditory.propagation import Emission, Listener, PropagationConfig, PropagationScene, Propagator, Reception

PARENT_POLL_S = 1.0


class Status(enum.Enum):
    DONE = "done"
    SKIPPED = "skipped"
    STALE = "stale"


@attrs.frozen
class Task:
    """One propagation pass, scene set only when it changed since the previous task of the same process."""

    job_id: int
    emission: Emission
    listeners: tuple[Listener, ...]
    config: PropagationConfig
    generation: int
    scene: PropagationScene | None = None
    stamp_ns: int = 0
    max_age_ns: int = 0


@attrs.frozen
class Outcome:
    job_id: int
    status: Status
    receptions: tuple[tuple[Listener, Reception], ...] = ()
    warnings: tuple[str, ...] = ()
    cache_summary: str = ""


def run(propagator: Propagator, task: Task, *, now_ns: int, generation: int, warnings: list[str]) -> Outcome:
    """Receptions of task, skipped for an earlier generation, stale for a sound event older than its age limit."""
    if task.scene is not None:
        propagator.set_scene(task.scene)
    if task.generation != generation:
        return Outcome(task.job_id, Status.SKIPPED)
    if task.stamp_ns and now_ns - task.stamp_ns > task.max_age_ns:
        return Outcome(task.job_id, Status.STALE)
    if propagator.config != task.config:
        propagator.config = task.config
    receptions = tuple(propagator.propagate_many(task.emission, task.listeners))
    for _, reception in receptions:
        if reception.impulse is not None:
            propagator.impulses.put(reception.impulse)
    drained = tuple(warnings)
    warnings.clear()
    return Outcome(task.job_id, Status.DONE, receptions, drained, propagator.cache_summary())


def serve(config: PropagationConfig, tasks: Queue[bytes | None], outcomes: Queue[bytes], now_ns: Synchronized[int], generation: Synchronized[int]) -> None:
    """Child process loop: one pickled outcome per pickled task until the None sentinel or the parent's exit."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    outcomes.cancel_join_thread()
    parent = multiprocessing.parent_process()
    warnings: list[str] = []
    propagator = Propagator(config, default_catalog(), warn=warnings.append)
    while True:
        try:
            blob = tasks.get(timeout=PARENT_POLL_S)
        except queue.Empty:
            if parent is None or not parent.is_alive():
                return
            continue
        if blob is None:
            return
        outcome = run(propagator, pickle.loads(blob), now_ns=now_ns.value, generation=generation.value, warnings=warnings)
        outcomes.put(pickle.dumps(outcome))
