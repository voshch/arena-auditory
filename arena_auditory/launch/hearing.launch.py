"""Robot hearing layer of one env: belief grid, speed-filter policy and the srp or seld front-end, ``robot.hearing.<key>`` forwarded as ``<key>``."""

import launch
import launch.actions
from launch_ros.actions import Node

from arena_auditory.hearing.belief_grid import NOMINAL_EVENT_RATE_HZ
from arena_auditory.params import BeliefGroup, Frontend, Param, all_params

_PREFIX = "robot.hearing."
_POLICY_KEY = f"{_PREFIX}policy"
_POLICIES = ("belief", "listen", "full")
_FRONTEND_NODES = {Frontend.SRP: ("srp_frontend", "hearing_srp_frontend"), Frontend.SELD: ("seld_frontend", "hearing_seld_frontend")}


def _hearing(context: launch.LaunchContext) -> list[launch.LaunchDescriptionEntity]:
    configs = context.launch_configurations
    params = all_params()
    frontend = Frontend(configs["frontend"])
    policy = configs["policy"]
    derived: dict[str, object] = {
        "hearing.frontend": frontend.value,
        "hearing.tg_node": configs["tg_node"],
        "belief.event_rate_hz": NOMINAL_EVENT_RATE_HZ[frontend.value],
        "policy.listen.enabled": policy != "belief",
        "policy.yield.enabled": policy == "full",
    }
    if frontend is Frontend.SRP:
        derived["belief.kinds"] = ["onset"]
    forwarded: dict[str, object] = {}
    for key, raw in configs.items():
        if not key.startswith(_PREFIX) or key == _POLICY_KEY or not raw:
            continue
        name = key.removeprefix(_PREFIX)
        param = Param[float](name, 0.0) if name.startswith(BeliefGroup.EMISSION_DB_PREFIX) else params.get(name)
        if param is None:
            raise ValueError(f"{key} is not a hearing node parameter")
        forwarded[name] = param.coerce(raw)
    hearing_params = {"use_sim_time": True, **derived, **forwarded}
    env_ns = "/" + configs["env.ns"].strip("/")

    def node(name: str, executable: str) -> Node:
        return Node(package="arena_auditory", executable=executable, name=name, namespace=env_ns, output="screen", parameters=[hearing_params])

    actions: list[launch.LaunchDescriptionEntity] = [node("hearing_belief", "hearing_belief_node"), node("hearing_policy", "hearing_policy")]
    if frontend in _FRONTEND_NODES:
        actions.append(node(*_FRONTEND_NODES[frontend]))
    return actions


def generate_launch_description() -> launch.LaunchDescription:
    return launch.LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument("env.ns", default_value="/arena/env_0", description="Env namespace, the hearing nodes live in it."),
            launch.actions.DeclareLaunchArgument("tg_node", default_value="task_generator_node", description="Task generator node name, the auditory topics live below it."),
            launch.actions.DeclareLaunchArgument("frontend", default_value=Frontend.BUS.value, choices=[frontend.value for frontend in Frontend], description="Detection source: bus is the simulator bus, srp the onset + GCC-PHAT front-end on any array, seld the SELDnet front-end on the array its weights were trained on."),
            launch.actions.DeclareLaunchArgument("policy", default_value="full", choices=list(_POLICIES), description="Mask layers: belief only, plus the corner listen cap, plus yield."),
            launch.actions.OpaqueFunction(function=_hearing),
        ]
    )
