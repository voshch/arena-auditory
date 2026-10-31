from __future__ import annotations

import asyncio
import importlib.util

import pytest


def pytest_configure(config: pytest.Config) -> None:
    if importlib.util.find_spec("arena_simulation_setup") is not None:
        from arena_simulation_setup.testing import install_fake_assets

        install_fake_assets()


_ROS_SKIP_REASON = "ROS2 not discoverable: source install/setup.bash to enable"


def _ros_available() -> bool:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _ros_available():
        return
    skip = pytest.mark.skip(reason=_ROS_SKIP_REASON)
    for item in items:
        path = str(item.path)
        if "/tests/ros/" in path or "/tests/integration/" in path:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def rclpy_context():
    try:
        import rclpy
        import rclpy.node
    except ImportError:
        yield None
        return
    rclpy.init()
    node = rclpy.node.Node("pytest_host")
    try:
        yield rclpy
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.fixture(scope="session")
def default_sounds():
    from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary

    library = SoundLibrary.default()
    try:
        for kind in library.kinds().values():
            if kind.default_asset:
                library.asset(kind.default_asset)
    except KeyError as exc:
        pytest.skip(f"default sound assets do not resolve: {exc}")
    return library


@pytest.fixture
def loop():
    event_loop = asyncio.new_event_loop()
    try:
        yield event_loop
    finally:
        event_loop.close()
