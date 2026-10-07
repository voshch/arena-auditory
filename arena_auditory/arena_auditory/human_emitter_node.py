"""Pedestrian sound producer: footsteps and speech from the pedestrian stream, with floor-matched footstep variants."""

from __future__ import annotations

import collections
import itertools
import math

from arena_auditory_msgs.msg import SoundEvent
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_rclpy_mixins.qos import reliable
from arena_robots.audio import NS_PER_S
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind, SoundLibrary, selection_seed
from arena_simulation_setup.utils.geometry import Orientation
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from arena_auditory.assets import duration_s
from arena_auditory.constants import ARENA_PEDS, PEDESTRIAN_MARKERS_EXTRA, SOUND_EVENTS, env_topic
from arena_auditory.materials import default_catalog
from arena_auditory.params import Configuration
from arena_auditory.shared import SourceSpec
from arena_auditory.sources import PedestrianEventDetector, PedestrianState
from arena_auditory.world import AcousticWorld
from arena_auditory.world_tracker import WorldTracker

FRAME = "map"
MARKER_Z_M = 0.08
MARKER_RANGE_M = 1.25
MARKER_CONE_DEG = 70.0
MARKER_LIFETIME = Duration(sec=1, nanosec=200_000_000)


def pedestrian_state(ped: Pedestrian) -> PedestrianState:
    return PedestrianState(
        id=int(ped.id),
        name=str(ped.name),
        position=(float(ped.pose.position.x), float(ped.pose.position.y), float(ped.pose.position.z)),
        yaw_rad=Orientation.from_msg(ped.pose.orientation).to_yaw(),
        speed_mps=math.hypot(ped.twist.linear.x, ped.twist.linear.y),
        gait_phase=float(ped.gait_phase),
    )


class HumanEmitterNode(ArenaMixinNode):
    """Publishes one SoundEvent per pedestrian footstep or greeting, variant and level chosen here."""

    def __init__(self) -> None:
        super().__init__("human_emitter")
        self.conf = Configuration(self)
        human = self.conf.Human
        env_ns = self.conf.Env.NS.value
        self._library = SoundLibrary.default()
        self._materials = default_catalog()
        self._tracker = WorldTracker(self, on_ready=self._on_world_ready, on_clear=self._on_world_clear, on_episode=self._on_episode, conf=self.conf)
        self._event_counter = itertools.count()
        self._marker_counter = itertools.count()
        self._occurrences: collections.Counter[tuple[int, str]] = collections.Counter()
        self._detector = PedestrianEventDetector(
            self._emit,
            walking_speed_mps=human.WALKING_SPEED_MPS.value,
            footstep_interval_s=human.FOOTSTEP_INTERVAL_S.value,
            greeting_distance_m=human.GREETING_DISTANCE_M.value,
            greeting_fov_deg=human.GREETING_FOV_DEG.value,
            greeting_cooldown_s=human.GREETING_COOLDOWN_S.value,
        )
        self._sound_publisher = self.create_publisher(SoundEvent, SOUND_EVENTS, reliable(50))
        self._marker_publisher: LazyPublisher[MarkerArray] = LazyPublisher(self.create_publisher(MarkerArray, env_topic(env_ns, PEDESTRIAN_MARKERS_EXTRA), reliable(10)))
        self.create_subscription(Pedestrians, env_topic(env_ns, ARENA_PEDS), self._on_pedestrians, 10)

    def destroy_node(self) -> bool:
        self._tracker.destroy()
        return super().destroy_node()

    def _on_world_ready(self, world: AcousticWorld) -> None:
        self.get_logger().info(f"footstep floor materials from acoustic world {world.name!r}, {len(world.scene.zones)} zones")

    def _on_world_clear(self) -> None:
        self.get_logger().info("acoustic world cleared, footsteps use the default variant until it is realized")

    def _on_episode(self, _episode_id: int) -> None:
        self._occurrences.clear()
        self._detector.reset()

    def _on_pedestrians(self, msg: Pedestrians) -> None:
        now_s = self.get_clock().now().nanoseconds / NS_PER_S
        self._detector.update([pedestrian_state(ped) for ped in msg.pedestrians], now_s)

    def _occurrence(self, agent_id: int, kind: str) -> int:
        key = (agent_id, kind)
        occurrence = self._occurrences[key]
        self._occurrences[key] += 1
        return occurrence

    def _emit(self, kind_name: str, ped: PedestrianState) -> None:
        world = self._tracker.world
        x, y, _ = ped.position
        floor = world.floor_material(x, y) if world is not None else ""
        try:
            kind = self._library.kind(kind_name)
            asset = self._library.default_asset(kind_name)
            seed = selection_seed(self._tracker.episode_seed, ped.id, kind_name, self._occurrence(ped.id, kind_name))
            variant = asset.select(context={"floor": floor}, seed=seed)
            length_s = duration_s(self._library, asset.id, variant.id)
        except (LookupError, ValueError, OSError) as exc:
            self.get_logger().error(f"no {kind_name} sound: {exc}", throttle_duration_sec=5.0)
            return
        level_db = asset.level_db
        if asset.surface == "floor" and floor:
            level_db -= self._materials.surface_damping_db(floor, "floor")
        stamp = self.get_clock().now().to_msg()
        source = SourceSpec(
            id=f"human:{ped.id}:{stamp.sec}:{stamp.nanosec}:{next(self._event_counter)}",
            kind=kind_name,
            asset_id=asset.id,
            variant_id=variant.id,
            model=variant.model,
            agent_kind=AgentKind.PEDESTRIAN,
            agent_id=ped.id,
            agent_name=ped.name,
            tags=variant.tags,
            position=(x, y, kind.height_m),
            yaw_rad=ped.yaw_rad,
            level_db=level_db,
            reference_distance_m=asset.reference_distance_m,
            duration_ns=round(length_s * NS_PER_S),
            loop=asset.loop,
            seed=seed,
        )
        msg = SoundEvent(source=source.to_msg())
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME
        self._sound_publisher.publish(msg)
        self._marker_publisher.publish(lambda: self._cone(kind_name, kind.color, ped, stamp))

    def _cone(self, kind_name: str, color: tuple[float, float, float], ped: PedestrianState, stamp: Time) -> MarkerArray:
        yaw = ped.yaw_rad
        source_x, source_y = ped.position[0], ped.position[1]
        apex = Point(x=source_x + math.cos(yaw) * 0.15, y=source_y + math.sin(yaw) * 0.15, z=MARKER_Z_M)
        cone = math.radians(MARKER_CONE_DEG)
        start = yaw - cone / 2.0
        arc = [Point(x=source_x + math.cos(start + cone * i / 16) * MARKER_RANGE_M, y=source_y + math.sin(start + cone * i / 16) * MARKER_RANGE_M, z=MARKER_Z_M) for i in range(17)]
        r, g, b = color

        fill = Marker()
        fill.header.frame_id = FRAME
        fill.header.stamp = stamp
        fill.ns = f"human_sound_{kind_name}"
        fill.id = next(self._marker_counter)
        fill.type = Marker.TRIANGLE_LIST
        fill.action = Marker.ADD
        fill.pose.orientation.w = 1.0
        fill.scale.x = fill.scale.y = fill.scale.z = 1.0
        fill.lifetime = MARKER_LIFETIME
        fill.color = ColorRGBA(r=r, g=g, b=b, a=0.3)
        for left, right in itertools.pairwise(arc):
            fill.points.extend([apex, left, right])

        outline = Marker()
        outline.header = fill.header
        outline.ns = f"human_sound_{kind_name}_outline"
        outline.id = next(self._marker_counter)
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.035
        outline.lifetime = MARKER_LIFETIME
        outline.color = ColorRGBA(r=r, g=g, b=b, a=0.95)
        outline.points = [apex, *arc, apex]
        return MarkerArray(markers=[fill, outline])


def main() -> None:
    HumanEmitterNode.run_main()
