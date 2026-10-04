# Automated acoustic dataset recording

`record_acoustics_dataset.sh` launches each selected `scenario.yaml` in a fresh
Arena process and records ROS simulation-time data. It records every robot's
four-microphone raw float stream and rendered stereo headphone stream together
with `/clock`, episode state, robot odometry, pedestrian states, TF, the
occupancy map, and the door mask.

The script is source-only and runs inside the Arena environment (the container
shell). It launches through `python3 -m arena_bringup.supervisor`, the same
entry `arena launch` uses, and calls `ros2` and the
`arena_auditory.dataset` modules directly.

The default output is `$ARENA_DATA_DIR/audio_train_set`:

```text
audio_train_set/
|-- failed_scenarios.tsv
`-- <scenario-name>/
    |-- <scenario-name>_0001_recording.wav
    |-- <scenario-name>_0001_meta.csv
    |-- <scenario-name>_0001_validation.json
    |-- <scenario-name>_0001_manifest.yaml
    |-- <scenario-name>_0001_audio_timing.parquet
    |-- <scenario-name>_0001_robot_positions.parquet
    |-- <scenario-name>_0001_pedestrian_positions.parquet
    |-- <scenario-name>_0001_frame_labels.parquet
    |-- <scenario-name>_0001_tf_transforms.parquet
    |-- <scenario-name>_0001_episode_events.parquet
    |-- <scenario-name>_0001_sound_events.parquet
    |-- <scenario-name>_0001_continuous_sound_states.parquet
    |-- <scenario-name>_0001_sound_activity.parquet
    |-- <scenario-name>_0001_occupancy_map.npz
    |-- <scenario-name>_0001_door_mask.npz
    |-- scenario.yaml
    `-- episode_000/episode_000.mcap
```

A scenario with several robots exports one complete set per robot, the robot
name joining the prefix (`<scenario-name>_0001_<robot>_recording.wav`,
`<scenario-name>_0001_<robot>_meta.csv`, and so on). Each robot's set holds
only its own audio, odometry, and activity annotations, and its labels are
relative to that robot. `<scenario-name>_0001_validation.json` then indexes the
per-robot validation files. Robots are discovered from the recorded
`<robot>/audio/raw_array` topics, so a single-robot scenario keeps the layout
above.

The WAV contains the robot's final left/right hearing signal without requiring
an external encoder. Optional FLAC is compact and lossless for its integer PCM
representation. The MCAP remains the source
of truth for the raw propagated float32 microphone values. Converting those
arbitrary floats to FLAC would not preserve them exactly.

Audio and labels use the same ROS simulation clock. `AudioFrame.header.stamp`
is set by the array renderer to the simulation time of the first sample in the
block. A sample at offset `i` therefore has time
`header.stamp + i/sample_rate`. The exporter creates one metadata row per
20 ms audio window and interpolates robot and pedestrian poses at that exact
timestamp. Each row includes sample offset, absolute simulation timestamp,
robot/source pose and velocity, relative Cartesian position, range, bearing,
elevation, radial velocity, occupancy line-of-sight labels, and audio RMS/peak
features. Robot, pedestrian, and TF trajectory exports retain normalized
quaternion `qx,qy,qz,qw` fields in addition to convenient planar yaw values.

`sound_activity.parquet` contains the actual half-open sample intervals placed
on the raw microphone-array clock, grouped across microphone channels. Frame
labels identify active pedestrian/event IDs, sound types, motor activity,
single- versus multiple-pedestrian overlap, and clean single-source windows.
The source `SoundEvent` and continuous propagated state streams are retained in
their own Parquet files for provenance. Dataset recording rejects a new run if
the sample-clock activity annotations are absent.

## Prerequisites

The auditory feature installs `arena_auditory`. FFmpeg is only needed for
optional FLAC export:

```bash
arena feature auditory install
sudo apt install ffmpeg
```

The acoustics worlds ship as an external suite bundle.
Pass its `worlds/` directory as `--worlds-root`. The recorder prepends it to
`ARENA_WORLD_PATH` so the launched runtime resolves the same worlds.
`generate_acoustics_scenarios.py` in this directory writes the scenario matrix
into that bundle. The listener robot stays at end A for both direction
variants. In `a-to-b`, pedestrians spawn as a separated left/center/right
cluster near the robot and initially walk toward B. In `b-to-a`, they spawn as
a separated cluster at B and initially walk toward the robot. Routes reverse at
the endpoints so humans keep walking throughout long captures. The recorder
enables scenario lingering, which keeps a moving-robot episode alive after the
robot reaches B until the requested audio window ends.

## Running

From a source checkout, through the Arena environment:

```bash
cd ~/arena_ws
source arena -c 'src/Arena/arena_auditory/arena_auditory/tools/record_acoustics_dataset.sh --worlds-root <bundle>/worlds --robot jackal --list'
```

`--list` prints the deterministic execution order without launching anything.
Drop it to record all worlds and scenarios. Small test runs can select a world,
scenario-name pattern, and count:

```bash
record_acoustics_dataset.sh \
  --worlds-root <bundle>/worlds \
  --robot jackal \
  --world-glob 'straight_corridor_O' \
  --scenario-glob '*robot-idle*pedestrians-1*' \
  --max-scenarios 1 \
  --duration 5 \
  --output "$ARENA_DATA_DIR/audio_train_set_test"
```

`--output DIR` replaces the default output directory and is required when
`ARENA_DATA_DIR` is unset. Arguments after `--` are passed to the launch
unchanged. The arguments the recorder sets itself (simulator, world, robot,
task modes, auditory array, playback and motor, recording, display) are
refused there.

Use `--force` to move an incomplete/existing case to a timestamped backup and
repeat it. Completed cases with a valid validation JSON are skipped, so a full
run is resumable. With `--sim gazebo`, Gazebo is visible by default. Pass
`--headless` to suppress the Gazebo GUI while keeping the auditory pipeline,
workstation playback, MCAP recording, and export active. For unattended
restart loops, pass `--recover-incomplete`: validated scenarios remain
untouched, while the interrupted scenario directory is moved to a timestamped
backup before that scenario is attempted again. A failed scenario is logged to
`failed_scenarios.tsv` and the run continues with the next one.

Playback is enabled by default with `auditory.output.device:=auto`, which
selects a PulseAudio/PipeWire-compatible PortAudio output when available. Use
`--playback-device pulse` (or another PortAudio device name) to select one
explicitly, or `none` for no workstation playback. The saved WAV comes from the
exact stereo `AudioFrame` the array renderer publishes, rather than a
sound-card loopback, so it excludes desktop audio and stays aligned to
simulation time even when real-time factor varies.

## Validation

Validation is performed by the Python waiter/exporter invoked by the Bash
orchestrator. A case is accepted only when every
robot's raw and stereo streams cover the requested simulation-time interval,
the rendered stream is stereo and non-silent, sample timing is contiguous,
robot/pedestrian poses can be aligned, all four raw microphone channels and
their geometry are present, the episode has a terminal state, and the
occupancy map belongs to the same environment. The waiter owns the episode
action, starts the requested 30-second window only on the episode's `RUNNING`
event, and cancels it cleanly only after the requested
simulation-time coverage is reached, so the MCAP contains both RUNNING and
terminal episode events. Bash exits on any failed preflight, action, capture,
export, or missing validation file.
