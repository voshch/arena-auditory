cmake_minimum_required(VERSION 3.5)
project(arena_auditory_msgs)

find_package(ament_cmake_auto REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/AcousticPath.msg
  msg/AudioFrame.msg
  msg/AuditoryDetection.msg
  msg/ContinuousAudioSourceState.msg
  msg/ContinuousHeardSoundState.msg
  msg/HeardSoundEvent.msg
  msg/RenderedSoundActivity.msg
  msg/RoomImpulse.msg
  msg/SoundEvent.msg
  msg/SoundReception.msg
  msg/SoundSource.msg
  srv/RemoveMicrophone.srv
  srv/RemoveSound.srv
  srv/SpawnMicrophone.srv
  srv/SpawnSound.srv
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_auto_package()
