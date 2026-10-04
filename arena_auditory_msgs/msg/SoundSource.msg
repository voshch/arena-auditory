string id                              # event id, or source id if continuous
string group_id                        # sources toggled together, else id
string kind                            # kinds-table name
string asset_id                        # sound asset id, [domain/]name
string variant_id                      # empty: the renderer selects by seed
string model                           # source model, e.g. wav, wav_loop, drivetrain
string agent_kind                      # pedestrian robot environment external
int32 agent_id                         # -1 when not an agent
string agent_name
string[] tags                          # free-form, never interpreted downstream
geometry_msgs/Point position           # in header.frame_id, z is emitter height
float32 yaw_rad
float32 level_db                       # SPL at reference_distance_m
float32 reference_distance_m           # > 0, 0 reads as 1 m
builtin_interfaces/Duration duration   # one-shot length, zero for continuous
bool loop
bool active                            # continuous: sounding, one-shot: true
builtin_interfaces/Time program_start  # start of the current continuous program
uint64 seed                            # deterministic variant and phase seed
string[] state_names                   # streamed model inputs, e.g. wheel speeds
float32[] state_values                 # one value per state_names entry
string[] content_names                 # content, e.g. speech text, language
string[] content_values                # one value per content_names entry
