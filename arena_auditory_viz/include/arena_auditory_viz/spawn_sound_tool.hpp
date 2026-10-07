#ifndef ARENA_AUDITORY_VIZ_SPAWN_SOUND_TOOL_HPP
#define ARENA_AUDITORY_VIZ_SPAWN_SOUND_TOOL_HPP

#include <memory>

#include <QObject>

#include <rclcpp/rclcpp.hpp>
#include <rviz_default_plugins/tools/pose/pose_tool.hpp>

#include "task_generator_msgs/srv/spawn_sound.hpp"

namespace rviz_common
{
namespace properties
{
class BoolProperty;
class FloatProperty;
class StringProperty;
}
}  // namespace rviz_common

namespace arena_auditory_viz
{
class SpawnSoundTool : public rviz_default_plugins::tools::PoseTool
{
  Q_OBJECT

public:
  SpawnSoundTool();
  ~SpawnSoundTool() override;

  void onInitialize() override;

protected:
  void onPoseSet(double x, double y, double theta) override;

private Q_SLOTS:
  void updateClient();

private:
  rviz_common::properties::StringProperty * target_node_property_;
  rviz_common::properties::StringProperty * kind_property_;
  rviz_common::properties::FloatProperty * height_property_;
  rviz_common::properties::BoolProperty * customize_property_;
  rviz_common::properties::StringProperty * asset_property_;
  rviz_common::properties::FloatProperty * volume_property_;
  rviz_common::properties::BoolProperty * loop_property_;
  rviz_common::properties::BoolProperty * initially_active_property_;

  std::shared_ptr<rclcpp::Node> service_node_;
  rclcpp::Client<task_generator_msgs::srv::SpawnSound>::SharedPtr client_;
};
}  // namespace arena_auditory_viz

#endif  // ARENA_AUDITORY_VIZ_SPAWN_SOUND_TOOL_HPP
