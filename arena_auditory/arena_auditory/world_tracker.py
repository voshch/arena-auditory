"""ROS glue of the acoustic world: follows state/world, state/episode and map and realizes the world in the map frame."""

from __future__ import annotations

import subprocess
import typing
from collections.abc import Callable, Hashable
from concurrent.futures import Executor, Future, ThreadPoolExecutor

from arena_rclpy_mixins.param_groups import configure
from arena_rclpy_mixins.qos import latched
from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import String
from task_generator_msgs.msg import EpisodeRecord

from arena_auditory.constants import MAP, STATE_EPISODE, STATE_WORLD
from arena_auditory.params import Configuration
from arena_auditory.world import LOAD_ERRORS, AcousticWorld, OccupancyMap, WorldConfig

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins import ArenaMixinNode

WORLD_RESOLVE_ERRORS = (OSError, LookupError, ValueError, subprocess.SubprocessError)


def follow_world_sounds(node: Node, library: SoundLibrary, loader: Executor, on_world: Callable[[str], None] | None = None) -> None:
    """Keep library's world-local sounds on the world STATE_WORLD names, resolved on loader, calling on_world with each new name first."""

    def use(world: str) -> None:
        try:
            library.use_world_named(world)
        except WORLD_RESOLVE_ERRORS as exc:
            node.get_logger().error(f"world-local sounds of {world!r} unavailable: {exc!r}")

    def follow(msg: String) -> None:
        world = str(msg.data).strip()
        if not world:
            return
        if on_world is not None:
            on_world(world)
        loader.submit(use, world)

    node.create_subscription(String, STATE_WORLD, follow, latched(1))


class WorldTracker:
    """Follows state/world, state/episode and map, loads the world on a worker thread and realizes it in the map frame."""

    def __init__(
        self,
        node: ArenaMixinNode,
        *,
        on_ready: Callable[[AcousticWorld], None],
        on_clear: Callable[[], None],
        on_episode: Callable[[int], None] | None = None,
        conf: Configuration | None = None,
    ) -> None:
        self._node = node
        self._conf = conf if conf is not None else Configuration(node)
        self._world_config()
        self._occupied_threshold = self._conf.Map.OCCUPIED_THRESHOLD
        self._on_ready = on_ready
        self._on_clear = on_clear
        self._on_episode = on_episode
        self._loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="acoustic_world")
        self._future: Future[AcousticWorld] | None = None
        self._loading = ""
        self._pending = ""
        self._loaded: AcousticWorld | None = None
        self._world: AcousticWorld | None = None
        self._occupancy: OccupancyMap | None = None
        self._episode_id = -1
        self._episode_seed = 0
        self._coverage_signature: tuple[Hashable, ...] | None = None
        node.create_subscription(String, STATE_WORLD, self._on_world_msg, latched(1))
        node.create_subscription(EpisodeRecord, STATE_EPISODE, self._on_episode_msg, latched(20))
        node.create_subscription(OccupancyGrid, MAP, self._on_map, latched(1))
        node.create_timer(0.1, self._poll)

    @property
    def world(self) -> AcousticWorld | None:
        """The world realized in the runtime map frame, None until world and map are known."""
        return self._world

    @property
    def occupancy(self) -> OccupancyMap | None:
        return self._occupancy

    @property
    def episode_seed(self) -> int:
        return self._episode_seed

    @property
    def episode_id(self) -> int:
        return self._episode_id

    def destroy(self) -> None:
        self._loader.shutdown(wait=False, cancel_futures=True)

    def _world_config(self) -> WorldConfig:
        return configure(WorldConfig, self._conf.World, self._conf.Portal)

    def _on_world_msg(self, msg: String) -> None:
        self._request(str(msg.data).strip())

    def _on_episode_msg(self, msg: EpisodeRecord) -> None:
        episode_id = int(msg.episode_id)
        self._episode_seed = int(msg.seed)
        if episode_id != self._episode_id:
            self._episode_id = episode_id
            if self._on_episode is not None:
                self._on_episode(episode_id)
        self._request(str(msg.world).strip())

    def _request(self, world: str) -> None:
        if not world:
            return
        self._pending = world
        if self._future is not None and not self._future.done():
            return
        current = self._loaded.name if self._loaded is not None else ""
        if world == current:
            return
        self._start(world)

    def _start(self, world: str) -> None:
        self._node.get_logger().info(f"loading acoustic world {world!r}")
        self._loaded = None
        self._world = None
        self._coverage_signature = None
        self._loading = world
        self._on_clear()
        self._future = self._loader.submit(AcousticWorld.load, world, self._world_config())

    def _poll(self) -> None:
        if self._future is None or not self._future.done():
            return
        future = self._future
        self._future = None
        loading = self._loading
        self._loading = ""
        try:
            loaded = future.result()
        except LOAD_ERRORS as exc:
            self._node.get_logger().error(f"failed to load acoustic world {loading!r}: {exc!r}")
            if self._pending and self._pending != loading:
                self._start(self._pending)
            return
        if self._pending and self._pending != loaded.name:
            self._start(self._pending)
            return
        if not loaded.scene.zones:
            self._node.get_logger().warning(f"world {loaded.name!r} has no authored acoustic zones, using map-based distance and occlusion propagation")
        try:
            SoundLibrary.default().use_world(loaded.path)
        except WORLD_RESOLVE_ERRORS as exc:
            self._node.get_logger().error(f"world-local sounds of {loaded.name!r} unavailable: {exc!r}")
        self._loaded = loaded
        self._realize()

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._occupancy = OccupancyMap.from_msg(msg, self._occupied_threshold.value)
        self._realize()

    def _realize(self) -> None:
        if self._loaded is None or self._occupancy is None:
            return
        try:
            world = self._loaded.realized(self._occupancy.origin_xy)
        except ValueError as exc:
            self._node.get_logger().error(str(exc))
            return
        if self._world is not None and self._world.signature == world.signature:
            self._validate_coverage()
            return
        self._world = world
        self._coverage_signature = None
        graph = world.graph
        self._node.get_logger().info(
            f"realized acoustic world {world.name!r} in runtime map frame {self._occupancy.frame_id!r} with "
            f"offset=({world.offset[0]:.2f},{world.offset[1]:.2f}), rooms={len(world.rooms)}, "
            f"door_portals={sum(p.portal_kind == 'door' for p in graph.portals)}, "
            f"opening_portals={sum(p.portal_kind == 'opening' for p in graph.portals)}, "
            f"components={len(graph.connected_components())}, unpaired_doors={len(graph.unpaired_doors)}"
        )
        for door in graph.unpaired_doors:
            self._node.get_logger().info(f"acoustic door {door.door_name!r} in {door.owner_zone!r} was not paired: {door.reason}")
        for portal in graph.portals:
            self._node.get_logger().info(f"acoustic portal {portal.portal_id}: {portal.zone_a!r} <-> {portal.zone_b!r}, kind={portal.portal_kind!r}, material={portal.material_id!r}, loss={portal.loss_db} dB")
        self._validate_coverage()
        self._on_ready(world)

    def _validate_coverage(self) -> None:
        world, occupancy = self._world, self._occupancy
        if world is None or occupancy is None or not world.scene.zones or not self._conf.World.COVERAGE_ENABLED.value:
            return
        stride = max(self._conf.World.COVERAGE_STRIDE_CELLS.value, 1)
        tolerance = self._conf.World.COVERAGE_TOLERANCE_M.value
        signature = (world.signature, occupancy.width, occupancy.height, occupancy.resolution_m, occupancy.origin_xy, stride, tolerance)
        if signature == self._coverage_signature:
            return
        self._coverage_signature = signature
        report = world.coverage(occupancy, stride_cells=stride, tolerance_m=tolerance)
        if report.complete:
            self._node.get_logger().info(f"acoustic zone coverage validated on {report.sampled} sampled traversable cells (stride={stride})")
            return
        examples = ", ".join(f"({x:.2f},{y:.2f})" for x, y in report.uncovered[:8])
        zone_bounds = ", ".join(f"{zone.name}={tuple(round(v, 2) for v in zone.polygon.bounds)}" for zone in world.scene.zones)
        self._node.get_logger().warning(
            f"{len(report.uncovered)} of {report.sampled} sampled traversable map cells lie outside all acoustic zones, "
            f"map(frame={occupancy.frame_id!r}, cells={occupancy.width}x{occupancy.height}, "
            f"size={occupancy.width * occupancy.resolution_m:.2f}x{occupancy.height * occupancy.resolution_m:.2f} m, "
            f"resolution={occupancy.resolution_m:.3f}, origin=({occupancy.origin_xy[0]:.2f},{occupancy.origin_xy[1]:.2f}), "
            f"yaw={occupancy.origin_yaw:.3f}). Zone bounds: {zone_bounds}. First samples: {examples}. "
            "These events will log an explicit backend fallback reason."
        )
