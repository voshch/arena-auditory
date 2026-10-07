from __future__ import annotations

import os

import pytest
import yaml


@pytest.fixture(scope="module", autouse=True)
def _require_ament() -> None:
    if "AMENT_PREFIX_PATH" not in os.environ:
        pytest.skip("AMENT_PREFIX_PATH unset: source install/setup.bash")


def _launch_path() -> str:
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(get_package_share_directory("arena_auditory"), "launch", "arena_auditory.launch.py")


def _stack_nodes() -> dict:
    import launch
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.actions import Node

    ctx = launch.LaunchContext()
    ctx.launch_configurations.update({"namespace": "env7/task_generator_node", "env.ns": "env7"})
    nodes: dict = {}

    def walk(entity) -> None:
        if isinstance(entity, Node):
            entity._perform_substitutions(ctx)
            nodes[entity.node_name.rsplit("/", 1)[-1]] = entity
        elif isinstance(entity, launch.LaunchDescription):
            for child in entity.entities:
                walk(child)
        elif isinstance(entity, launch.Action):
            for child in entity.execute(ctx) or []:
                walk(child)

    walk(PythonLaunchDescriptionSource(_launch_path()).get_launch_description(ctx))
    return nodes


def _parameters(node) -> dict:
    params: dict = {}
    for argument, is_file in node._Node__expanded_parameter_arguments:
        if is_file:
            with open(argument) as f:
                for section in yaml.load(f, Loader=yaml.FullLoader).values():
                    params.update(section["ros__parameters"])
    return params


def _topic(node, name: str) -> str:
    from rclpy.expand_topic_name import expand_topic_name

    return expand_topic_name(name, node.node_name.rsplit("/", 1)[-1], node.expanded_node_namespace)


def test_auditory_playback_nodes_run_on_sim_time() -> None:
    nodes = _stack_nodes()
    for name in ("array_renderer", "listener_renderer", "robot_emitter"):
        assert _parameters(nodes[name])["use_sim_time"] is True, name


def test_discrete_sound_events_share_the_sound_events_topic() -> None:
    from arena_auditory.constants import HEARD_SOUND_EVENTS, SOUND_EVENTS

    nodes = _stack_nodes()
    assert _topic(nodes["human_emitter"], SOUND_EVENTS) == _topic(nodes["sound_propagation_node"], SOUND_EVENTS) == "/env7/task_generator_node/sound_events"
    heard = _topic(nodes["sound_propagation_node"], HEARD_SOUND_EVENTS)
    for name in ("array_renderer", "listener_renderer", "robot_hearing_node"):
        assert _topic(nodes[name], HEARD_SOUND_EVENTS) == heard, name
