string listener_id
geometry_msgs/Point listener_position
float32 distance_m
float32 bearing_rad  # map frame, atan2 of source - listener
float32 received_level_db
float32 threshold_db
float32 direct_delay_s
bool audible
bool occluded
string source_zone
string listener_zone
string backend       # propagation backend that produced it
bool used_fallback
string fallback_reason
string[] portal_ids
string[] traversed_zones
geometry_msgs/Point[] portal_positions
float32 route_loss_db
string rir_key       # RoomImpulse key, empty without impulse
