std_msgs/Header header
string stream_id
SoundSource source
string channel_name
bool continuous            # true: active edge, end equals start
bool active
uint64 start_sample_index  # raw-array sample clock, half-open
uint64 end_sample_index
builtin_interfaces/Time start_time
builtin_interfaces/Time end_time
