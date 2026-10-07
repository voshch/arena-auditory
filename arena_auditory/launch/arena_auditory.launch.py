"""Auditory simulator stack of one env. Every other launch argument is forwarded as a node parameter of the same name."""

import launch
import launch.actions
from arena_rclpy_mixins.param_groups import declare_launch_arguments
from launch_ros.actions import Node

from arena_auditory.params import MotorGroup, MotorModel, RenderRole, all_params

_RESERVED = ("namespace", "hearing", "use_sim_time", "ros_remaps")
_HEARING_ARRAY_SPEC = {"bus": "", "srp": "four_mic", "seld": "four_mic"}


def _stack(context: launch.LaunchContext) -> list[launch.LaunchDescriptionEntity]:
    configs = context.launch_configurations
    params = all_params()
    namespace = configs.get("namespace", "")
    overrides = {key: params[key].coerce(value) if key in params else value for key, value in configs.items() if key not in _RESERVED and isinstance(value, str) and value}
    hearing = configs.get("hearing", "none")
    if _HEARING_ARRAY_SPEC.get(hearing):
        overrides.setdefault("array.spec", _HEARING_ARRAY_SPEC[hearing])
    viz_enabled = bool(overrides.get("viz.enabled", params["viz.enabled"].default))
    overrides.setdefault("pedestrian_listeners.discrete.enabled", viz_enabled)
    device = str(overrides.get("output.device", params["output.device"].default))

    def node(name: str, executable: str, role: RenderRole | None = None, env: dict[str, str] | None = None) -> Node:
        derived = {"render.role": role.value} if role is not None else {}
        return Node(
            package="arena_auditory",
            executable=executable,
            name=name,
            namespace=namespace,
            output="screen",
            parameters=[{"use_sim_time": True, **overrides, **derived}],
            additional_env=env,
        )

    nodes = [
        node("sound_propagation_node", "sound_propagation_node"),
        node("human_emitter", "human_emitter"),
        node("robot_emitter", "robot_emitter"),
        node("array_renderer", "renderer", RenderRole.ARRAY, {"OMP_NUM_THREADS": "1"}),
        node("robot_hearing_node", "robot_hearing_node"),
    ]
    if device != "none":
        nodes.append(node("listener_renderer", "renderer", RenderRole.LISTENER))
    if viz_enabled:
        nodes.append(node("sound_propagation_visualizer", "sound_propagation_visualizer"))
    return nodes


def generate_launch_description() -> launch.LaunchDescription:
    return launch.LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument("namespace", default_value="", description="Task generator node namespace, the stack nodes live below it."),
            launch.actions.DeclareLaunchArgument("env.ns", default_value="", description="Env namespace of the pedestrian topics."),
            launch.actions.DeclareLaunchArgument("hearing", default_value="none", choices=["none", *_HEARING_ARRAY_SPEC], description="Robot hearing frontend, srp and seld default array.spec to four_mic."),
            *declare_launch_arguments(all_params(), reserved=(MotorGroup.MODEL.name,)),
            launch.actions.DeclareLaunchArgument(MotorGroup.MODEL.name, default_value="", choices=["", *(model.value for model in MotorModel)], description=f"{MotorGroup.MODEL.description} Empty = node default."),
            launch.actions.OpaqueFunction(function=_stack),
        ]
    )
