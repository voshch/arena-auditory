import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_auditory'


def _tree(directory: str) -> list[tuple[str, list[str]]]:
    files: dict[str, list[str]] = {}
    for root, _, names in os.walk(directory):
        for name in sorted(names):
            files.setdefault(os.path.join('share', package_name, root), []).append(os.path.join(root, name))
    return sorted(files.items())


setup(
    name=package_name,
    packages=find_packages(where='.', include=[f'{package_name}*']),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *_tree('config'),
        *_tree('sounds'),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='Arena auditory simulator',
    license='TODO',
    entry_points={
        'console_scripts': [
            'sound_propagation_node = arena_auditory.sound_propagation_node:main',
            'sound_propagation_visualizer = arena_auditory.visualizer_node:main',
            'renderer = arena_auditory.renderer_node:main',
            'human_emitter = arena_auditory.human_emitter_node:main',
            'robot_emitter = arena_auditory.robot_emitter_node:main',
            'robot_hearing_node = arena_auditory.hearing.bus_node:main',
            'hearing_belief_node = arena_auditory.hearing.belief_node:main',
            'hearing_policy = arena_auditory.hearing.policy_node:main',
            'hearing_seld_frontend = arena_auditory.hearing.seld_frontend_node:main',
            'hearing_srp_frontend = arena_auditory.hearing.srp_frontend_node:main',
            'hearing_setup = arena_auditory.hearing.weights:main',
            'hearing_audio_replay = arena_auditory.hearing.audio_replay:main',
            'auditory_offline_render = arena_auditory.offline_render:main',
            'acoustic_world_audit = arena_auditory.acoustic_audit:main',
            'microphone_diagnostic = arena_auditory.microphone_diagnostic:main',
            'export_acoustics_recording = arena_auditory.dataset.export_recording:main',
            'wait_acoustics_capture = arena_auditory.dataset.wait_capture:main',
        ]
    },
)
