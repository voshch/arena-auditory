# arena_auditory

The auditory simulator is the `arena` backend of the acoustics axis next to the
human simulator: robot microphone arrays, sound propagation, the simulator bus
and workstation playback. `acoustics:=none` (the default) runs without the
auditory nodes. `acoustics:=arena` launches this stack per env and makes the
task generator bring up the map server it reads. Pedestrian footsteps and speech are produced from the human
simulator's `arena_peds` topic, whichever backend publishes it.

This repository is the optional auditory feature, a submodule of Arena. Install
it with `arena feature auditory install`, which checks it out, installs its
system dependencies through rosdep (PortAudio for sounddevice), syncs its Python
dependencies (pyroomacoustics, sounddevice, polars, pyarrow) and rebuilds.

Robot hearing (belief grid, policy, srp and SELDnet front-ends, the Nav2
speed-filter overlay) is the separate package `arena_hearing`, installed with
`arena feature hearing install`. This package keeps the simulator bus that
feeds it.

## Layout

```
arena_auditory/                 this repository
|-- arena_auditory/             ament_python package
|   |-- arena_auditory/         python package
|   |   |-- api.py              the names the task_generator acoustics backend imports
|   |   |-- shared.py           SourceSpec, ListenerId
|   |   |-- constants.py        every topic and service name
|   |   |-- params.py           every node parameter and its default
|   |   |-- world.py rooms.py materials.py   acoustic world, rooms, materials
|   |   |-- world_tracker.py    world and map subscriptions, realizes the world in the map frame
|   |   |-- assets.py           sample decoding of sound assets
|   |   |-- propagation/        backends, portal routing, RIR cache, impulses
|   |   |-- sources/            source models (wav, wav_loop, drivetrain), pedestrian events
|   |   |-- render/             block renderer, DSP, monitor, workstation output
|   |   |-- dataset/            acoustics recording export and capture wait
|   |   `-- *_node.py, bus_node.py, offline_render.py, acoustic_audit.py, microphone_diagnostic.py
|   |-- config/                 acoustic_materials.yaml
|   |-- launch/                 arena_auditory.launch.py
|   `-- tools/                  source-only scripts, not installed: scenario generation, dataset recording
|-- arena_auditory_msgs/        messages and services
`-- arena_auditory_viz/         RViz panel and tools
```

## Nodes

`arena_auditory.launch.py` starts these nodes below the task generator node of
the env (`/arena/env_0/task_generator_node/<node>` for env 0):

| Node | Executable | Role |
|---|---|---|
| `sound_propagation_node` | `sound_propagation_node` | Turns `SoundEvent` and `ContinuousAudioSourceState` into one `HeardSoundEvent` or `ContinuousHeardSoundState` per listener. Owns every room impulse response and publishes it on `acoustic/impulses`. Serves `runtime/spawn_microphone` and `runtime/remove_microphone`. |
| `human_emitter` | `human_emitter` | Emits `footstep` and `speech` sound events from `arena_peds`. |
| `robot_emitter` | `robot_emitter` | Publishes the continuous noise sources of every robot, the motor driven by its odometry. |
| `array_renderer` | `renderer` (`render.role: array`) | Renders the microphone array (`array.spec`) of every fleet robot to PCM, the stems, the monitor and the diagnostics. Plays one robot's monitor on the workstation while `output.enabled`. Runs with `OMP_NUM_THREADS=1`. |
| `listener_renderer` | `renderer` (`render.role: listener`) | Renders the single microphone `listener.id` to the workstation. Not launched with `auditory.output.device:=none`. |
| `robot_hearing_node` | `robot_hearing_node` | The simulator bus: republishes what each robot's `robot:<r>` listener hears as `<r>/heard_sound`, `arena_robots_msgs/SoundDetection` on `<r>/hearing/bus/detections` and a text marker. |
| `sound_propagation_visualizer` | `sound_propagation_visualizer` | Propagation paths, rooms, portals and environment sources in RViz. Launched with `auditory.viz.enabled:=true`. |

The robot hearing nodes come from `arena_hearing`, see [Robot hearing](#robot-hearing).

Further executables:

| Executable | Purpose |
|---|---|
| `auditory_offline_render` | Re-renders a recorded `diagnostics/render_inputs` trace (v1 and v2) offline. |
| `acoustic_world_audit` | Audits acoustic zone coverage and portals of installed worlds, `--plot-room` plots a room impulse response. |
| `microphone_diagnostic` | Console levels and GCC-PHAT estimates of a `raw_array` topic. |
| `export_acoustics_recording`, `wait_acoustics_capture` | Acoustics dataset pipeline, see [tools/README.md](arena_auditory/tools/README.md). |

## Parameters

Every node parameter and its default is declared once in
[`params.py`](arena_auditory/arena_auditory/params.py), grouped by prefix. A
node declares only the groups it uses. Names that appear in several nodes
(`world.*`, `map.*`, `rir.*`, `portal.*`, `motor.*`, `output.*`, `listener.id`,
`array.spec`) mean the same thing everywhere.

| Prefix | Read by |
|---|---|
| `env.ns`, `world.*`, `map.*`, `debug.map_source`, `rir.*`, `portal.*` | propagation, emitters, visualizer |
| `propagation.*`, `level3.*`, `pedestrian_listeners.*`, `microphones`, `viewport.height_m` | `sound_propagation_node` |
| `human.*` | `human_emitter` |
| `drivetrain.*`, `motor.enabled`, `motor.model` | `robot_emitter` |
| `render.*`, `array.*`, `monitor.*`, `tdoa.*`, `diagnostics.*`, `output.*`, `motor.*` | renderers, `array.*` and `diagnostics.period_s` also propagation |
| `bus.*` | `robot_hearing_node` |
| `viz.*` | `sound_propagation_visualizer` |

The two renderers share one executable. These defaults depend on `render.role`:

| Parameter | array | listener |
|---|---|---|
| `monitor.master_gain_db` | -1.94 | 0.0 |
| `monitor.rear_gain` | 0.75 | 1.0 |
| `render.rir.enabled` | false | true |
| `render.inaudible.enabled` | false | true |
| `render.min_level_db` | -120.0 | -20.0 |
| `render.lockstep.enabled` | true | false |

Parameters the RViz panel drives are live, for example:

```bash
ros2 param set /arena/env_0/task_generator_node/array_renderer array.muted true
ros2 param set /arena/env_0/task_generator_node/array_renderer monitor.solo front_left
ros2 param set /arena/env_0/task_generator_node/listener_renderer listener.id microphone:runtime:1
ros2 param set /arena/env_0/task_generator_node/sound_propagation_node propagation.enabled false
```

## Launch arguments

`arena launch ... acoustics:=arena` passes every `auditory.<param>:=<value>` to
every stack node as ROS parameter `<param>`, coerced to the type of its default
in `params.py`. Parameters a node does not declare are ignored. An empty value
keeps the node default. Each argument's description lives on its `Param` in
`params.py`, and `launch/arena_auditory.launch.py` declares every described
parameter as a launch argument. The task generator declares only `acoustics`
and `auditory.static_sounds`. The arguments are:

| Argument | Node default | Effect |
|---|---|---|
| `acoustics` | `none` | `arena` starts the stack |
| `auditory.output.device` | `auto` | PortAudio device, `auto` tries `pulse`, `pipewire`, `default`, then the PortAudio default. `none` starts no listener renderer |
| `auditory.output.block_size` | `512` | Workstation callback block size. Repeated underflows call for a larger `output.buffer_s` |
| `auditory.output.motor.enabled`, `auditory.output.ambient.enabled` | `true` | Play motor or environment audio on the workstation |
| `auditory.viz.enabled` | `false` | Start the visualizer and draw discrete pedestrian listeners |
| `auditory.propagation.backend` | `pyroomacoustics` | `pyroomacoustics`, `level3` or `legacy` |
| `auditory.portal.multi_hop.enabled` | `true` | RIRs across multi-hop portal routes |
| `auditory.rir.max_order` | `3` | Image-source order of every RIR, `1` is the cheap fallback |
| `auditory.pedestrian_listeners.enabled` | `false` | Pedestrians are listeners and receive stimuli through the human simulator |
| `auditory.motor.enabled` | `true` | Robots emit drivetrain audio |
| `auditory.motor.model` | `procedural` | `procedural` (Jackal drivetrain synthesis) or `wav` |
| `auditory.motor.trim_db` | `0.0` | Live offset on the motor level. The motor asset `level_db` (45 dB at 1 m) is the level of every motor variant, the procedural drivetrain at the spec `v_ref` (1 m/s) |
| `auditory.listener.id` | empty | Microphone of the listener renderer |
| `auditory.viewport.height_m` | `1.6` | Height of the viewport down-projection microphone |
| `auditory.array.spec` | `stereo` | `mono`, `stereo`, `four_mic` or a yaml path. `four_mic` when `robot.hearing` is `srp` or `seld` |
| `auditory.array.mount_frame` | empty | TF frame of the array, `{prefix}` and `{base_frame}` expand, a bare leaf joins the robot prefix, empty uses the base frame |
| `auditory.microphones` | `[]` | Extra robot-mounted microphones |
| `auditory.static_sounds` | `[]` | Launch-defined sound entities |

The launch also derives `env.ns`, `render.role` per renderer and
`pedestrian_listeners.discrete.enabled` (from `viz.enabled`). The old flat
names (`auditory.playback`, `auditory.viz`, `microphone_mode`, ...) still work
with a deprecation warning, see
[BRINGUP.md](../arena_bringup/BRINGUP.md#deprecated-launch-args).

## Messages

`arena_auditory_msgs` holds the simulator interfaces:

| Type | Purpose |
|---|---|
| `SoundSource` | One emitter and its emission: id, kind, asset and variant, model, agent, position, level, seed, streamed model state (`state_names`, `state_values`), content such as speech text (`content_names`, `content_values`) |
| `SoundReception` | One listener's result: level, delay, bearing, audibility, occlusion, backend, portal route, `rir_key` |
| `SoundEvent`, `HeardSoundEvent` | Finite sound, before and after propagation |
| `ContinuousAudioSourceState`, `ContinuousHeardSoundState` | Continuous source, before and after propagation |
| `RoomImpulse` | Room impulse response by key, direct arrival normalized to 1 |
| `AcousticPath` | Early reflection of a reception |
| `RenderedSoundActivity` | Start and end sample of every rendered source |
| `SpawnMicrophone`, `RemoveMicrophone` | Runtime microphones |

The audio signal `AudioFrame` (interleaved float32 PCM with channel names,
frames, positions and yaws) and `SoundDetection` (bearing, level and kind of a
detected sound) are in `arena_robots_msgs`. The runtime sound services
`SpawnSound` and `RemoveSound` are in `task_generator_msgs`, served by the
task generator's sounds module.

## Topics

Names below the task generator node, `<r>` is a robot name:

| Topic | Type | Publisher |
|---|---|---|
| `sound_events` | `SoundEvent` | emitters, sounds module |
| `continuous_audio_sources` | `ContinuousAudioSourceState` | `robot_emitter`, sounds module |
| `heard_sound_events` | `HeardSoundEvent` | propagation |
| `continuous_heard_sounds` | `ContinuousHeardSoundState` | propagation |
| `acoustic/impulses` | `RoomImpulse`, latched, depth 256 | propagation, before the first reception that references the key and again once `IMPULSE_WINDOW` newer keys went out |
| `microphone_listeners` | `std_msgs/String` JSON list, latched | propagation |
| `microphone_markers` | `MarkerArray` | propagation |
| `<r>/audio/raw_array` | `AudioFrame` | `array_renderer` |
| `<r>/audio/stem_motor`, `stem_pedestrian`, `stem_ambient` | `AudioFrame` | `array_renderer`, only while subscribed |
| `<r>/audio/headphones/stereo` | `AudioFrame` | `array_renderer`, only while subscribed |
| `<r>/audio/hearing/mono` | `AudioFrame` | `array_renderer`, only while subscribed |
| `<r>/audio/hearing/energy` | `std_msgs/Float32MultiArray` | `array_renderer`, only while subscribed |
| `<r>/audio/diagnostics/tdoa` | `std_msgs/String` JSON | `array_renderer`, only while subscribed |
| `<r>/audio/diagnostics/levels` | `MarkerArray`, per-microphone dBFS text in the mount frame | `array_renderer`, only while subscribed |
| `<r>/audio/diagnostics/render_inputs` | `std_msgs/String` JSON | `array_renderer`, only while subscribed |
| `<r>/audio/rendered_sound_activity` | `RenderedSoundActivity` | `array_renderer` |
| `audio/listener/monitor` | `AudioFrame` | `listener_renderer`, only while subscribed |
| `<r>/heard_sound`, `<r>/heard_sound_marker` | `HeardSoundEvent`, `Marker` | `robot_hearing_node`, the marker only while subscribed |
| `<r>/hearing/bus/detections` | `arena_robots_msgs/SoundDetection` | `robot_hearing_node` |
| `<r>/motor_sound_markers` | `MarkerArray` | `robot_emitter`, only while subscribed |
| `pedestrian_sound_propagation_markers`, `robot_sound_propagation_markers` | `MarkerArray` | visualizer, only while subscribed |
| `acoustic_room_markers`, `environment_audio_source_markers` | `MarkerArray`, latched | visualizer |

`<env ns>/pedestrian_markers/extra` carries the pedestrian footstep and speech
cones.

## Listeners

Every listener has a string id:

| Id | Listener |
|---|---|
| `robot:<r>` | Array centroid of robot `<r>`, always propagated, the input of the bus |
| `array:<r>:<mic>` | One microphone of the robot array, for every fleet robot |
| `microphone:robot:<r>:<placement>:<index>` | `auditory.microphones` entry |
| `microphone:zone:<zone>:<placement>:<index>` | World microphone from `world.yaml` |
| `microphone:runtime:<n>` | Spawned with the RViz tool, cleared on episode reset |
| `microphone:viewport:projective_center`, `microphone:viewport:down_projection` | Viewport camera, propagated only while selected |
| `agent:<id>` | Pedestrian, with `pedestrian_listeners.enabled` |

The selected `listener.id` microphone is propagated in addition.

### Robot arrays

`array.spec` picks a preset from `arena_robots/config/audio/arrays/` or a yaml path:

| Preset | Microphones | Rate | Block |
|---|---|---|---|
| `mono` | `mono` at the base frame | 44100 Hz | 512 |
| `stereo` | `left`, `right`, 0.20 m apart at 0.35 m | 44100 Hz | 512 |
| `four_mic` | `front_left`, `front_right`, `rear_left`, `rear_right` | 16000 Hz | 320 |

`four_mic` is the Jackal rectangle: each microphone sits 0.020 m in from the
two nearest edges of the 0.420 m by 0.310 m chassis, at z = 0.220 m. That
gives 0.380 m front to back, 0.270 m left to right, and inlet yaws of +-45 and
+-135 deg. With 343 m/s the far-field delays reach 1108 us (17.7 samples)
front to back and 787 us (12.6 samples) left to right. Sensitivity is -26 dBFS
at 94 dB SPL, after the Seeed reSpeaker XMOS XVF3800 reference (raw channels,
no onboard AEC, AGC or beamforming). A custom array is a yaml with `name`,
`sample_rate_hz`, `block_size`,
`sensitivity_dbfs_at_94_dbspl` and either `mics` (each `name`, `position_m`,
`yaw_deg`, `side`, `group`) or `rectangular`.

The robot's own sources (the motor) reach its array microphones through the
fixed free-field direct path from the source to each microphone, so the
ego-noise level stays constant in rooms, doorways and zone gaps alike.

The array renderer renders every robot of the fleet in one node, each with its
own `<r>/audio/*` topics, render state and lockstep channel. The workstation
output plays one of them: `array.robot` when set, else the first fleet robot.

### Extra microphones

Robot-mounted microphones come from `auditory.microphones`:

```bash
arena launch acoustics:=arena \
  auditory.microphones:='[{owner: robot, robot: jackal, placement: front, frame: front_laser, index: 1}]'
```

A relative frame resolves below the robot's frame prefix. The listener is
inactive while the robot is absent or TF cannot resolve the frame.

World microphones are authored per level in `world.yaml` beside `zones`:

```yaml
microphones:
  - {zone: reception, placement: ceiling, frame: map, position: [4.2, 3.1, 2.9], index: 1}
```

World loading rejects missing zones, duplicate ids, positions outside the zone,
ceiling placements in zones without a ceiling and heights more than 5 cm off an
explicit `ceiling_height`.

## Rendering

The renderers are driven by `/clock`. Every clock message renders the blocks
it covers, so sample zero stays anchored to the sim time of the first block.
A heard event is anchored one block past the next unrendered block.
When the render falls more than `render.max_catchup_blocks` behind, the
surplus blocks are skipped and counted. Under `arena lockstep`, the array
renderer registers one hard channel `audio/<r>` per rendered robot with one
block per window, so the skip path never fires.

Both renderers select the source program through the source models of
`sources/`: `wav` for one-shot clips, `wav_loop` for loops and `drivetrain`
for the procedural Jackal motor. A model is buffered (a decoded sample) or
streamed (a block generator built by its `stream()` from the seed, the
variant's `params` and the live tuning of its parameter group, then driven
each block by the source's `state`). A new streamed source is one synth class,
one `SOURCE_MODELS` line and a sound manifest variant naming the model.
Neither renderer touches the world or pyroomacoustics. A source with a `rir_key` is convolved with the impulse from
`acoustic/impulses` when `render.rir.enabled` is set, and a key change
crossfades over `render.rir.crossfade_s`. The array renderer defaults to dry
rendering. A renderer subscribes to `acoustic/impulses` only while
`render.rir.enabled` is set, keeps the impulses its voices reference plus the
`IMPULSE_WINDOW` most recently announced, drops them all on a world change and cuts each impulse where
its remaining energy falls 60 dB below its total, and no earlier than the direct
arrival.

The array renderer gates a source per robot: an event or continuous source
plays on all of a robot's microphones when any of its `array:<r>:<mic>` or
`robot:<r>` receptions passes `render.inaudible.enabled` and
`render.min_level_db`, else on none. Near the threshold or at a wall edge a
sound therefore never reaches only a subset of the microphones, which would
fake level and delay differences.

### Stems

`raw_array` is the clipped mix. The three stems split the same pre-clip mix by
the `stem` of each source's sound kind (`SoundSource.kind`), whoever emits it:

| Stem | Kinds |
|---|---|
| `stem_pedestrian` | `footstep`, `speech`, also when a robot speaks |
| `stem_ambient` | `music`, `alarm` |
| `stem_motor` | `motor`, the ego-noise |

All four share channel order, stamp and sample clock, so an episode can be
remixed at another ego-noise attenuation offline: weight the stems, add them
and clip the result to [-1, 1] the way the renderer does. The diagnostics count
clipped samples, which tells whether `raw_array` itself was clamped.

`diagnostics/render_inputs` carries one JSON object per block with the
complete input state of the block (version 2: channels, stems, RIR keys and
streamed sources with their model, params, state and tuning). Clips and
continuous sources name their sample by its key `<asset id>#<variant id>`.
`auditory_offline_render` reproduces the recorded audio from that trace alone,
bit for bit. Traces recorded before keys carried the asset id hold the bare
variant id, which the offline replay alone resolves by searching the local
sound assets for the one asset with that variant.

### Monitor and output

The headphone monitor is
`L = (front_gain * front-left + rear_gain * rear-left) / gain_sum` and the
mirrored right side, center microphones feeding both ears.
`monitor.gain_db` (36 dB) is a workstation preamp that leaves `raw_array`
untouched, `monitor.limit` bounds the peaks and `monitor.solo` routes one
microphone to both ears. `monitor.mode: hearing` previews the
highest-energy microphone instead. Docker playback uses the host
PulseAudio or PipeWire socket forwarded as `/tmp/pulse/native`.

Rendered blocks reach the device through a jitter buffer. It plays silence
until `output.buffer_s` (0.04 s) of audio is queued, at start and after every
underflow, and drops the oldest audio back to that level once the queue
exceeds it by more than its own length rounded up to whole render blocks.
`output.buffer_s` has to cover the `/clock` period, raise it on repeated
underflows. The `latency_s` of the output diagnostics is the running median
from push to the DAC.

Every reception is an SPL at the microphone. Both renderers apply the
`sensitivity_dbfs_at_94_dbspl` calibration of their array spec, the listener
renderer that of `mono`.

## Sound library

Sounds are assets of the `Sound` kind. The catalog (kinds table, manifests,
variants, selection) is `arena_simulation_setup.tree.assets.sound_catalog`,
its kinds table and manifest schema are documented in
[configs/sounds/README.md](../arena_simulation_setup/configs/sounds/README.md).
This package only decodes samples ([assets.py](arena_auditory/arena_auditory/assets.py)):
`SampleDecoder` loads a variant's wav, resamples it to the render rate and
normalizes it to the manifest's `normalize_dbfs`.

A kind's `stem` names the renderer stem its sources land in, see
[Stems](#stems).

## World and launch-defined sounds

The `sounds` task module is added to `task.modules` whenever `acoustics` is not
`none` or `auditory.static_sounds` is non-empty. It renders every `sound`
entity declared in the loaded world, the active scenario and the launch
configuration.

A sound is a standalone `sound` semantic entity. The semantics engine owns
whether it plays (`sounding`) and how loud (`volume_db`). The sounds module
resolves its placement and publishes its live state as
`ContinuousAudioSourceState`. A `sound` entry takes:

- `name`: unique among world, scenario and launch sounds.
- `asset_id`: a `Sound` asset.
- exactly one of `position`, `entity_ref` or `frame`. `entity_ref` names one
  static world entity, `offset` then rotates with its yaw. A direct `position`
  is level-local and needs `level` in a multi-level world. `frame` names a TF
  frame and `offset` is local to it, so a sound on `jackal/base_link` moves
  with the robot. Until the frame exists the source is skipped with a warning.
- `loop` (default `true`) and `reference_distance_m` (default `1.0`).
- `semantics`: the `sound` preset, which expands to `sounding` and `volume_db`.

World sounds are authored per zone:

```yaml
zones:
- name: hallway
  corners: [...]
  sounds:
  - name: hall_siren
    asset_id: alarm_loop
    position: [4.0, 2.3]
    semantics:
    - {preset: sound, params: {sound_on: alarm}}
```

`sound_on` names a regime, like a gate's `unlock_on` (see
[AUTHORING.md](../arena_simulation_setup/AUTHORING.md)). Sounds sharing one
`sound_on` toggle together and share their variant selection but stay separate
physical sources, each with its own route, delay and impulse response. A
world or scenario sound without `sound_on` plays only when told to. An
always-on radio is authored as
`{preset: sound, params: {sounding: true, volume_db: 62.0}}`. A different
starting volume is a separate `{state: volume_db, value: 62.0}` entry. A sound without a starting volume plays at its
asset's level, which is also what a timeline `when:` on its `volume_db` reads.
Toggle a sound from a scenario timeline or live:

```bash
ros2 service call /arena/env_0/task_generator_node/semantics/set \
  task_generator_msgs/srv/SetSemantic \
  "{entity: env_0/lobby_radio/1, field: sounding, value: 'true'}"
```

`entity` is the realized name published on `state/semantics`. Scenario sounds
live in the scenario's own `sounds:` list and last one episode. Launch sounds
use the same schema as a flat list and play from the start unless they name a
`sound_on` or set `sounding`:

```bash
arena launch world:=demo acoustics:=arena \
  auditory.static_sounds:='[{name: room_radio, asset_id: radio_loop, position: [5.0, 5.0, 1.2], level: level_1, semantics: [{preset: sound, params: {volume_db: 62.0}}]}]'
```

`sounding` controls simulated emission. `output.ambient.enabled` only mutes
the workstation, propagation and robot hearing continue.

## Propagation and portal routing

On world load, Arena pairs each authored door with the zone on its other side
and derives an opening portal where two adjacent rooms agree that a shared
boundary span is open. Each authored zone is one pyroomacoustics room.
Same-zone sounds use one room-local impulse response. Cross-zone sounds follow
a door or opening route up to `portal.max_hops`, composed from room-local
segments. Without a connected route, propagation falls back to `level3`, then
`legacy`, and records the reason. A robot hears its own noise sources at every
`robot:<r>` and `array:<r>:<mic>` listener it owns through a fixed free-field
direct path (1/d from the 1 m level, no occlusion, no impulse, so it renders
dry), whatever the backend and wherever the robot drives.

The result is on every `SoundReception`: `backend`
(`pyroomacoustics_same_room`, `pyroomacoustics_one_door`,
`pyroomacoustics_multi_portal`, `level3`, `legacy_distance_occlusion`,
`self_direct_path`), `used_fallback`, `fallback_reason`, `portal_ids`,
`portal_positions`, `traversed_zones`, `route_loss_db` and `rir_key`.
Propagation is the only owner of impulse responses. Every pyroomacoustics
reception takes its level and delay from its RIR and carries its `rir_key`.
Propagation quantizes source and listener by
`rir.quantization_m` (0.10 m), caches by key and publishes a key on
`acoustic/impulses` before the first reception that references it and again
once 64 newer keys went out.
Recording that topic makes an offline re-render with impulses
reproducible. Dynamic door state is not published, so doors use
`portal.door_loss_db` and derived openings `portal.opening_loss_db`.

With `pedestrian_listeners.enabled` (default false), every pedestrian other
than the source is an `agent:<id>` listener whose receptions reach
`BaseHumanSimulator.notify_stimulus`, edge-triggered on audibility, with the
kind as stimulus name.

Audit installed worlds before relying on RIR coverage:

```bash
ros2 run arena_auditory acoustic_world_audit --stride-cells 10
```

It lists missing maps, traversable cells outside zones, overlapping zones,
portals, unpaired doors and graph components, and returns non-zero when
coverage is incomplete.

## RViz plugins

`arena_auditory_viz` ships an RViz panel and two tools. The arena acoustics
backend declares them in its viz manifest (`rviz_plugins` in `api.py`), so
`arena viz` adds them to the generated configuration of an `acoustics:=arena`
env.

**AuditoryPanel** (`arena_auditory_viz::AuditoryPanel`, config key `Target`,
default `/task_generator_node`) talks to `<Target>/array_renderer`,
`<Target>/listener_renderer` and `<Target>/sound_propagation_node`:

| Control | Parameter |
|---|---|
| Enable sound propagation | `propagation.enabled` on propagation |
| Play environment audio on this workstation | `output.ambient.enabled` on both renderers |
| Robot Microphone Array group | `array.enabled`, `monitor.enabled`, `array.muted`, `monitor.master_gain_db`, `monitor.gain_db`, `monitor.front_gain`, `monitor.rear_gain`, `monitor.solo`, `monitor.mode`, `tdoa.enabled` on `array_renderer` |
| Play robot motor audio on this workstation | `output.motor.enabled` on both renderers |
| Motor Sound Tuning | `motor.trim_db`, `motor.frequency_scale`, `motor.tonal_gain_db`, `motor.broadband_gain_db`, `motor.speed_exponent`, `motor.velocity_smoothing_s` on both renderers |

**Workstation Listener** lists the ids of `microphone_listeners`. An
`array:<r>:<mic>` id sets `output.enabled=true` on `array_renderer` and
`listener.id=""` on `listener_renderer`, so the array monitor plays. Any other
id sets `output.enabled=false` on `array_renderer` and `listener.id` on
`listener_renderer` and propagation. **Left microphone** and **Right
microphone** pick `array:<r>:left` and `array:<r>:right` when the stereo array
is active. **Sound Entities** lists every `kind == "sound"` entity of
`state/semantics`. A check toggles `sounding` through `semantics/set`, and
runtime sources can be removed through `runtime/remove_sound`.

**SpawnSoundTool** (shortcut `r`) places an environment source through
`runtime/spawn_sound`. `Kind` is an environment sound kind (default `music`),
a rejected kind logs the accepted ones. `Height` sets z. Without `Custom
Playback` the kind's default asset plays at its own level, with it the asset
id, source level, loop flag and initial state are yours to set. Click and drag
sets the pose in the RViz fixed frame.

**SpawnMicrophoneTool** (shortcut `m`) places a microphone through
`runtime/spawn_microphone` at the clicked point and `Height`. `Attach TF
Frame` makes it follow a frame. Runtime microphones are
`microphone:runtime:<n>` and are cleared on the next episode or world change.

## Robot hearing

Robot hearing is the package `arena_hearing`, selected per env with
`robot.hearing:=bus|srp|seld`. This package keeps the `bus` front end:
`robot_hearing_node` publishes the propagated `robot:<r>` receptions as
`arena_robots_msgs/SoundDetection` on `<r>/hearing/bus/detections` with
map-frame bearings, the input of the `arena_hearing` belief node under
`robot.hearing:=bus`. `srp` and `seld` read `<r>/audio/raw_array` and default
`auditory.array.spec` to `four_mic`.

```bash
export ARENA_WORLD_PATH=$ARENA_WS_DIR/src/Arena/_assets/arena-benchmarks-prod-public/suites/acoustics/worlds
arena launch sim:=gazebo robot:=jackal world:=acoustics_bend_narrow_O \
    task.robots:=scenario task.obstacles:=scenario \
    task.scenario.file:=hearing__world-acoustics_bend_narrow_O__robot-moving__pedestrians-1__ends-a-to-b \
    acoustics:=arena robot.hearing:=bus
```

The belief grid, the policy, the front-ends, their parameters and the Nav2
overlay are documented in `arena_hearing`.

## Tests

```bash
python3 -m pytest arena_auditory/tests/unit -q
python3 -m pytest arena_auditory/tests/ros/test_auditory_round_trip.py -q
```

The ROS round trip publishes a speech `SoundEvent`, expects a reception for
`robot:<r>`, the republished `<r>/heard_sound` and its text marker.

## Configuration

| File | Content |
|---|---|
| `config/acoustic_materials.yaml` | absorption and floor damping per material |
| `arena_auditory/params.py` | every node parameter, default and launch description |
| `arena_simulation_setup/configs/sounds/kinds.yaml` (Arena) | kinds table |
| `arena_robots/config/audio/arrays/*.yaml` (Arena) | microphone array presets |
| `task_generator/launch/acoustics/arena/` (Arena) | install check and the include of `launch/arena_auditory.launch.py` |
| `task_generator/task_generator/simulators/acoustics/arena/` (Arena) | node-side backend, the single gateway into this package |
