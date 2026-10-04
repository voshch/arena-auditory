cmake_minimum_required(VERSION 3.8)
project(arena_auditory_viz)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(arena_auditory_msgs REQUIRED)
find_package(task_generator_msgs REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

set(CMAKE_AUTOMOC ON)
qt5_wrap_cpp(MOC_FILES
  include/arena_auditory_viz/auditory_panel.hpp
  include/arena_auditory_viz/spawn_sound_tool.hpp
  include/arena_auditory_viz/spawn_microphone_tool.hpp
)

add_library(arena_auditory_viz SHARED
  src/auditory_panel.cpp
  src/spawn_sound_tool.cpp
  src/spawn_microphone_tool.cpp
  ${MOC_FILES})
target_include_directories(arena_auditory_viz PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(arena_auditory_viz
  rclcpp
  std_msgs
  pluginlib
  rviz_common
  rviz_default_plugins
  geometry_msgs
  rcl_interfaces
  arena_auditory_msgs
  task_generator_msgs
)
target_link_libraries(arena_auditory_viz Qt5::Widgets)

install(TARGETS arena_auditory_viz
        EXPORT export_arena_auditory_viz
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
install(DIRECTORY include/
        DESTINATION include
)
install(FILES rviz_common_plugins.xml
        DESTINATION share/${PROJECT_NAME}
)
ament_export_include_directories(include)
ament_export_targets(export_arena_auditory_viz)
pluginlib_export_plugin_description_file(rviz_common rviz_common_plugins.xml)

ament_package()
