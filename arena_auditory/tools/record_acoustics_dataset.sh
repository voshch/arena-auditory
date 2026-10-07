#!/usr/bin/env bash
# Record every acoustics scenario with simulation-time-synchronized ROS audio.

set -Eeo pipefail

WORLDS_ROOT=''
OUTPUT_ROOT=''
ROBOT=''
SCENARIO_GLOB='*'
WORLD_GLOB='*'
MAX_SCENARIOS=0
DURATION=30
WALL_TIMEOUT=90
READY_TIMEOUT=120
EPISODE_FINALIZE_TIMEOUT=15
SIMULATOR='gazebo'
PLAYBACK_DEVICE='auto'
INJECT_REFERENCE_SOUND=0
FORCE=0
RECOVER_INCOMPLETE=0
LIST_ONLY=0
HEADLESS=0
LAUNCH_CLIENT_PID=''
CAPTURE_PID=''
SCENARIO_FAIL=0
FAILED_SCENARIOS_FILE=''

usage() {
    cat <<'EOF'
Usage: record_acoustics_dataset.sh --worlds-root DIR --robot NAME [options] [-- extra launch arguments]

Runs inside the Arena environment and records every selected scenario to:
  <output>/<scenario>/

The selected PortAudio device plays the robot's stereo headphone stream while
the same signal is captured from its ROS AudioFrame source. This excludes
launch/setup noise and preserves exact episode timing. The MCAP also contains
the raw four-microphone stream, /clock, maps, poses, TF, and episode events.
The exporter writes <scenario>_<index>_recording.wav (or FLAC when explicitly
requested) and matching metadata. A scenario with several robots gets one set
per robot, <scenario>_<index>_<robot>_recording.wav and so on, indexed by
<scenario>_<index>_validation.json.

Options:
  --worlds-root DIR       Acoustics worlds directory, e.g. the acoustics suite bundle's worlds/ (required)
  --robot NAME            Arena robot that records every scenario (required)
  --output DIR            Output directory (default: $ARENA_DATA_DIR/audio_train_set)
  --scenario-glob GLOB    Scenario-directory glob (default: *, all scenarios)
  --world-glob GLOB       World-directory glob (default: *, all worlds)
  --max-scenarios N       Stop after N selected cases (default: 0, unlimited)
  --duration SEC          Required simulation-time audio duration (default: 30)
  --wall-timeout SEC      Per-capture wall-time watchdog (default: 90)
  --ready-timeout SEC     Simulator/audio startup timeout (default: 120)
  --episode-finalize-timeout SEC  Wait for clean episode cancellation/finalization (default: 15)
  --sim NAME              Arena simulator backend (default: gazebo)
  --playback-device DEV   PortAudio output: auto, none, or a device name (default: auto)
  --reference-sound       Inject an audible calibration event (diagnostics only)
  --headless              Run without the Gazebo GUI (audio capture and playback remain enabled)
  --recover-incomplete    Back up and retry only an incomplete scenario, keep validated runs
  --force                 Move an existing run to a timestamped backup and repeat it
  --list                  List selected index/world/scenario triples
  -h, --help              Show this help
EOF
}

die() { printf 'record_acoustics_dataset: ERROR: %s\n' "$*" >&2; exit 1; }
note() { printf 'record_acoustics_dataset: %s\n' "$*" >&2; }
is_number() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }

EXTRA_LAUNCH_ARGS=()
while (($#)); do
    case "$1" in
        --worlds-root) WORLDS_ROOT="${2:?missing value}"; shift 2 ;;
        --robot) ROBOT="${2:?missing value}"; shift 2 ;;
        --output) OUTPUT_ROOT="${2:?missing value}"; shift 2 ;;
        --scenario-glob) SCENARIO_GLOB="${2:?missing value}"; shift 2 ;;
        --world-glob) WORLD_GLOB="${2:?missing value}"; shift 2 ;;
        --max-scenarios) MAX_SCENARIOS="${2:?missing value}"; shift 2 ;;
        --duration) DURATION="${2:?missing value}"; shift 2 ;;
        --wall-timeout) WALL_TIMEOUT="${2:?missing value}"; shift 2 ;;
        --ready-timeout) READY_TIMEOUT="${2:?missing value}"; shift 2 ;;
        --episode-finalize-timeout) EPISODE_FINALIZE_TIMEOUT="${2:?missing value}"; shift 2 ;;
        --sim) SIMULATOR="${2:?missing value}"; shift 2 ;;
        --playback-device) PLAYBACK_DEVICE="${2:?missing value}"; shift 2 ;;
        --reference-sound) INJECT_REFERENCE_SOUND=1; shift ;;
        --headless) HEADLESS=1; shift ;;
        --recover-incomplete) RECOVER_INCOMPLETE=1; shift ;;
        --force) FORCE=1; shift ;;
        --list) LIST_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_LAUNCH_ARGS=("$@"); break ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$WORLDS_ROOT" ]] || die '--worlds-root DIR is required, e.g. the acoustics suite bundle worlds/ directory'
[[ -d "$WORLDS_ROOT" ]] || die "worlds directory not found: $WORLDS_ROOT"
WORLDS_ROOT="$(cd -- "$WORLDS_ROOT" && pwd)"
[[ -n "$ROBOT" ]] || die '--robot NAME is required, e.g. --robot jackal'
[[ "$MAX_SCENARIOS" =~ ^[0-9]+$ ]] || die '--max-scenarios must be a non-negative integer'
is_number "$DURATION" || die '--duration must be numeric'
is_number "$WALL_TIMEOUT" || die '--wall-timeout must be numeric'
is_number "$READY_TIMEOUT" || die '--ready-timeout must be numeric'
is_number "$EPISODE_FINALIZE_TIMEOUT" || die '--episode-finalize-timeout must be numeric'
[[ -n "$PLAYBACK_DEVICE" ]] || die '--playback-device must be auto, none, or a device name'
[[ -n "$SIMULATOR" ]] || die '--sim must not be empty'
((RECOVER_INCOMPLETE == 0 || FORCE == 0)) || die '--recover-incomplete and --force cannot be used together'

for arg in "${EXTRA_LAUNCH_ARGS[@]}"; do
    case "$arg" in
        sim:=*|world:=*|robot:=*|human:=*|acoustics:=*|auditory.output.device:=*|auditory.array.spec:=*|auditory.motor.enabled:=*|auditory.motor.model:=*|task.robots:=*|task.obstacles:=*|task.scenario:=*|task.scenario.file:=*|task.scenario.linger_after_completion:=*|task.episode.auto_reset:=*|env.n:=*|viz:=*|headless:=*|record.dir:=*|record.auto:=*)
            die "the script owns launch argument '$arg'"
            ;;
    esac
done

SCENARIO_FILES=()
note "scanning worlds='${WORLD_GLOB}' scenarios='${SCENARIO_GLOB}' under $WORLDS_ROOT"
mapfile -d '' -t scenario_candidates < <(
    find "$WORLDS_ROOT" -type f \
        -path "$WORLDS_ROOT/$WORLD_GLOB/scenarios/$SCENARIO_GLOB/scenario.yaml" \
        -print0 | sort -z
)
for scenario_file in "${scenario_candidates[@]}"; do
    SCENARIO_FILES+=("$scenario_file")
    if ((MAX_SCENARIOS > 0 && ${#SCENARIO_FILES[@]} >= MAX_SCENARIOS)); then break; fi
done
unset scenario_candidates
((${#SCENARIO_FILES[@]})) || die "no scenario.yaml matched world '${WORLD_GLOB}' and scenario '${SCENARIO_GLOB}'"

if ((LIST_ONLY)); then
    execution_number=0
    for scenario_file in "${SCENARIO_FILES[@]}"; do
        execution_number=$((execution_number + 1))
        printf -v run_index '%04d' "$execution_number"
        scenario_dir="$(dirname -- "$scenario_file")"
        world_dir="$(dirname -- "$(dirname -- "$scenario_dir")")"
        printf '%s\t%s\t%s\n' "$run_index" "$(basename -- "$world_dir")" "$(basename -- "$scenario_dir")"
    done
    exit 0
fi

command -v ros2 >/dev/null || die 'ROS 2 is unavailable, run this inside the Arena environment, e.g. source arena -c "..."'
if [[ -z "$OUTPUT_ROOT" ]]; then
    [[ -n "${ARENA_DATA_DIR:-}" ]] || die 'ARENA_DATA_DIR is unset, pass --output DIR or run inside the Arena environment'
    OUTPUT_ROOT="${ARENA_DATA_DIR}/audio_train_set"
fi
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")"
export ARENA_WORLD_PATH="${WORLDS_ROOT}${ARENA_WORLD_PATH:+:${ARENA_WORLD_PATH}}"

refresh_ros_discovery() {
    ros2 daemon stop >/dev/null 2>&1 || true
    sleep 1
}

episode_actions() {
    ros2 action list 2>/dev/null | grep '/lifecycle/run_episode$' || true
}

wait_for_environment_shutdown() {
    local actions=''
    refresh_ros_discovery
    for _ in {1..20}; do
        actions="$(episode_actions)"
        [[ -n "$actions" ]] || return 0
        sleep 1
    done
    die "Arena environment survived recorder shutdown: $actions"
}

stop_launch() {
    [[ -n "$LAUNCH_CLIENT_PID" ]] || return 0
    kill -INT -- "-${LAUNCH_CLIENT_PID}" 2>/dev/null || true
    for _ in {1..40}; do kill -0 "$LAUNCH_CLIENT_PID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$LAUNCH_CLIENT_PID" 2>/dev/null; then
        note 'graceful shutdown timed out, sending SIGTERM'
        kill -TERM -- "-${LAUNCH_CLIENT_PID}" 2>/dev/null || true
    fi
    wait "$LAUNCH_CLIENT_PID" 2>/dev/null || true
    LAUNCH_CLIENT_PID=''
}

cancel_and_finalize_episode() {
    local action_name="$1"
    local cancel_service="${action_name}/_action/cancel_goal"
    local cancel_type='action_msgs/srv/CancelGoal'
    local cancel_request
    local cancel_response=''
    local deadline

    cancel_request='{goal_info: {goal_id: {uuid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, stamp: {sec: 0, nanosec: 0}}}'

    note "finalizing episode: cancelling active RunEpisode goal"
    cancel_response="$(
        timeout 10 ros2 service call \
            "$cancel_service" \
            "$cancel_type" \
            "$cancel_request" 2>&1 || true
    )"

    # CancelGoal return_code=0 means ERROR_NONE
    if [[ "$cancel_response" != *'return_code=0'* && "$cancel_response" != *'return_code: 0'* ]]; then
        printf '%s\n' "$cancel_response" >&2
        printf 'record_acoustics_dataset: visible action info follows:\n' >&2
        ros2 action info "$action_name" 2>&1 >&2 || true
        printf 'record_acoustics_dataset: hidden cancel endpoints follow:\n' >&2
        ros2 service list --include-hidden-services 2>/dev/null \
            | grep -F "${action_name}/_action/" >&2 || true
        die 'RunEpisode cancellation was not accepted, local MCAP retained for inspection'
    fi

    deadline=$((SECONDS + ${EPISODE_FINALIZE_TIMEOUT%.*}))
    while ((SECONDS <= deadline)); do
        if [[ -s "$launch_log" ]] && grep -Eqi 'terminal EpisodeRecord|EpisodeRecord.*(cancel|canceled|cancelled|complete|completed|terminal)|episode.*(cancel|canceled|cancelled|complete|completed).*record' "$launch_log"; then
            note 'episode terminal record observed before shutdown'
            return 0
        fi
        sleep 0.25
    done

    note "episode cancellation accepted, finalization grace period (${EPISODE_FINALIZE_TIMEOUT}s) elapsed"
}

cleanup() {
    local status=$?
    if [[ -n "$CAPTURE_PID" ]] && kill -0 "$CAPTURE_PID" 2>/dev/null; then kill "$CAPTURE_PID" 2>/dev/null || true; fi
    stop_launch
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/.failed_scenarios"
FAILED_SCENARIOS_FILE="$OUTPUT_ROOT/failed_scenarios.tsv"
if [[ ! -s "$FAILED_SCENARIOS_FILE" ]]; then
    printf 'timestamp\texecution_index\tworld\tscenario\texit_code\tlog\n' >"$FAILED_SCENARIOS_FILE"
fi
note "robot=$ROBOT scenarios=${#SCENARIO_FILES[@]} duration=${DURATION}s simulation time output=$OUTPUT_ROOT"

refresh_ros_discovery

execution_number=0

for scenario_file in "${SCENARIO_FILES[@]}"; do
    execution_number=$((execution_number + 1))
    printf -v run_index '%04d' "$execution_number"
    scenario_dir="$(dirname -- "$scenario_file")"
    scenario_name="$(basename -- "$scenario_dir")"
    world_dir="$(dirname -- "$(dirname -- "$scenario_dir")")"
    world_name="$(basename -- "$world_dir")"
    run_name="${scenario_name}"
    artifact_prefix="${scenario_name}_${run_index}"
    run_dir="${OUTPUT_ROOT}/${run_name}"
    validation="${run_dir}/${artifact_prefix}_validation.json"
    export GZ_PARTITION="arena_acoustics_${BASHPID}_${run_index}"

    if [[ -s "$validation" && "$FORCE" -eq 0 ]] && grep -q '"valid": true' "$validation"; then
        note "SKIP $run_name (validated)"
        continue
    fi
    failure_log="${OUTPUT_ROOT}/.failed_scenarios/${world_name}__${artifact_prefix}.log"
    set +e
    (
    set -Eeo pipefail
    trap cleanup EXIT

    existing_actions="$(episode_actions)"
    [[ -z "$existing_actions" ]] || die "another Arena environment is already running, stop it before recording: $existing_actions"
    if [[ -e "$run_dir" && ("$FORCE" -eq 1 || "$RECOVER_INCOMPLETE" -eq 1) ]]; then
        backup="${run_dir}.backup.$(date +%Y%m%d-%H%M%S)"
        mv -- "$run_dir" "$backup"
        note "moved previous incomplete/repeated run to $backup"
    elif [[ -e "$run_dir" ]]; then
        die "incomplete existing run at $run_dir (use --force after inspecting it)"
    fi
    mkdir -p "$run_dir"
    cp -- "$scenario_file" "${run_dir}/scenario.yaml"
    launch_log="${run_dir}/launch.log"
    waiter_log="${run_dir}/capture_wait.json"
    note "START $run_name world=$world_name"

    if [[ "$SIMULATOR" == gazebo && "$HEADLESS" -eq 0 ]]; then
        display_args=('viz:=false' 'headless:=false')
    else
        display_args=('viz:=false' 'headless:=true')
    fi
    launch_args=(
        "sim:=${SIMULATOR}" "world:=${world_name}" "robot:=${ROBOT}" 'human:=arena' 'acoustics:=arena'
        'auditory.array.spec:=four_mic' "auditory.output.device:=${PLAYBACK_DEVICE}"
        'auditory.motor.enabled:=true' 'auditory.motor.model:=procedural'
        'task.robots:=scenario' 'task.obstacles:=scenario' "task.scenario.file:=${scenario_name}"
        'task.scenario.linger_after_completion:=true'
        'task.episode.auto_reset:=false' 'env.n:=1' "${display_args[@]}"
        "record.dir:=${run_dir}" 'record.auto:=true'
        "${EXTRA_LAUNCH_ARGS[@]}"
    )
    setsid python3 -m arena_bringup.supervisor "${launch_args[@]}" >"$launch_log" 2>&1 &
    LAUNCH_CLIENT_PID=$!

    deadline=$((SECONDS + ${READY_TIMEOUT%.*}))
    episode_action=''
    env_namespace=''
    robots=()
    map_type=''
    map_state=''
    renderer_node=''
    declare -A raw_live=() rendered_live=()
    while ((SECONDS <= deadline)); do
        kill -0 "$LAUNCH_CLIENT_PID" 2>/dev/null || { tail -n 80 "$launch_log" >&2; die 'Arena exited during startup'; }
        episode_action="$(ros2 action list 2>/dev/null | grep '/lifecycle/run_episode$' | head -n 1 || true)"
        audio_ready=0
        if [[ -n "$episode_action" ]]; then
            env_namespace="${episode_action%/lifecycle/run_episode}"
            mapfile -t robots < <(ros2 topic list 2>/dev/null \
                | sed -n "s|^${env_namespace}/\([^/]*\)/audio/raw_array\$|\1|p" | sort -u || true)
            map_type="$(ros2 topic type "${env_namespace}/map" 2>/dev/null || true)"
            map_state="$(ros2 lifecycle get "${env_namespace}/map_server" 2>/dev/null || true)"
            renderer_node="$(ros2 node list 2>/dev/null | grep -x "${env_namespace}/array_renderer" || true)"
            audio_ready=$((${#robots[@]} > 0))
            for robot in "${robots[@]}"; do
                for stream in raw_array headphones/stereo; do
                    topic="${env_namespace}/${robot}/audio/${stream}"
                    if [[ "$stream" == raw_array ]]; then live="${raw_live[$robot]:-0}"; else live="${rendered_live[$robot]:-0}"; fi
                    if ((live == 0)) \
                        && [[ "$(ros2 topic type "$topic" 2>/dev/null || true)" == 'arena_robots_msgs/msg/AudioFrame' ]] \
                        && timeout 3 ros2 topic echo --once "$topic" >/dev/null 2>&1; then
                        live=1
                        if [[ "$stream" == raw_array ]]; then raw_live[$robot]=1; else rendered_live[$robot]=1; fi
                    fi
                    ((live)) || audio_ready=0
                done
            done
        fi
        if ((audio_ready)) \
           && [[ "$map_type" == 'nav_msgs/msg/OccupancyGrid' ]] \
           && [[ "$map_state" == *active* ]] \
           && [[ -n "$renderer_node" ]]; then
            break
        fi
        sleep 1
    done
    [[ -n "$episode_action" ]] || die "episode action not ready, see $launch_log"
    [[ -n "$renderer_node" ]] || die "${env_namespace}/array_renderer is not running"
    [[ "$map_type" == 'nav_msgs/msg/OccupancyGrid' && "$map_state" == *active* ]] \
        || die "environment map server is not active at ${env_namespace}/map"
    ((${#robots[@]})) || die "no robot publishes ${env_namespace}/<robot>/audio/raw_array"
    for robot in "${robots[@]}"; do
        ((${raw_live[$robot]:-0})) || die "${env_namespace}/${robot}/audio/raw_array is missing, has the wrong type, or did not publish a live frame"
        ((${rendered_live[$robot]:-0})) || die "${env_namespace}/${robot}/audio/headphones/stereo is missing, has the wrong type, or did not publish a live frame"
    done

    waiter_args=(
        --namespace "$env_namespace" \
        --duration "$DURATION" --wall-timeout "$WALL_TIMEOUT" \
        --run-episode-action "$episode_action" --world "$world_name"
    )
    if ((INJECT_REFERENCE_SOUND)); then waiter_args+=(--inject-reference-sound); fi
    python3 -m arena_auditory.dataset.wait_capture "${waiter_args[@]}" >"$waiter_log" &
    CAPTURE_PID=$!
    capture_ready=0
    for _ in {1..20}; do
        kill -0 "$CAPTURE_PID" 2>/dev/null || { wait "$CAPTURE_PID" 2>/dev/null || true; die "capture waiter exited before subscribing, see $waiter_log"; }
        if ros2 node list 2>/dev/null | grep -qx '/arena_acoustics_capture_waiter'; then
            capture_ready=1
            break
        fi
        sleep 0.25
    done
    ((capture_ready)) || die 'capture waiter did not become ready before the episode action'
    if ! wait "$CAPTURE_PID"; then
        CAPTURE_PID=''
        [[ ! -s "$waiter_log" ]] || tail -n 20 "$waiter_log" >&2
        die "capture did not reach ${DURATION}s of synchronized raw and rendered audio"
    fi
    CAPTURE_PID=''

    cancel_and_finalize_episode "$episode_action"

    stop_launch
    wait_for_environment_shutdown

    export_args=(
        "$run_dir" --output "$run_dir" --force --expected-duration "$DURATION"
        --require-activity-annotations
        --artifact-prefix "$artifact_prefix" --execution-index "$execution_number"
        --world-name "$world_name" --scenario-name "$scenario_name"
    )
    if [[ "$SIMULATOR" == dummy ]]; then export_args+=(--basic-audio-pedestrians); fi
    python3 -m arena_auditory.dataset.export_recording "${export_args[@]}"
    [[ -s "$validation" ]] || die "export did not create $validation"
    ) > >(tee "$failure_log") 2>&1
    scenario_status=$?
    set -e
    LAUNCH_CLIENT_PID=''
    CAPTURE_PID=''
    if ((scenario_status != 0)); then
        SCENARIO_FAIL=$((SCENARIO_FAIL + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "$run_index" "$world_name" \
            "$scenario_name" "$scenario_status" "$failure_log" \
            >>"$FAILED_SCENARIOS_FILE"
        note "FAILED $run_name (exit=$scenario_status), continuing. Log: $failure_log"
        continue
    fi
    rm -f -- "$failure_log"

    note "DONE  $run_dir"
done

((SCENARIO_FAIL == 0)) || die "$SCENARIO_FAIL scenario(s) failed, see $FAILED_SCENARIOS_FILE"
trap - EXIT INT TERM
note 'all selected scenarios have valid synchronized recordings'
