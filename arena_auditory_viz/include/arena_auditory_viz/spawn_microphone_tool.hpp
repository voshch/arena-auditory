#ifndef ARENA_AUDITORY_VIZ_SPAWN_MICROPHONE_TOOL_HPP
#define ARENA_AUDITORY_VIZ_SPAWN_MICROPHONE_TOOL_HPP

#include <memory>
#include <string>

#include <QObject>

#include <rclcpp/rclcpp.hpp>
#include <rviz_default_plugins/tools/pose/pose_tool.hpp>

#include "arena_auditory_msgs/srv/spawn_microphone.hpp"

namespace rviz_common
{
namespace properties
{
class FloatProperty;
class StringProperty;
}
}  // namespace rviz_common

namespace arena_auditory_viz
{
class SpawnMicrophoneTool : public rviz_default_plugins::tools::PoseTool
{
  Q_OBJECT

public:
  SpawnMicrophoneTool();
  ~SpawnMicrophoneTool() override;

  void onInitialize() override;

protected:
  void onPoseSet(double x, double y, double theta) override;

private Q_SLOTS:
  void updateClient();

private:
  rviz_common::properties::StringProperty * target_node_property_;
  rviz_common::properties::FloatProperty * height_property_;
  rviz_common::properties::StringProperty * attached_frame_property_;

  std::shared_ptr<rclcpp::Node> service_node_;
  rclcpp::Client<arena_auditory_msgs::srv::SpawnMicrophone>::SharedPtr client_;
};
}  // namespace arena_auditory_viz

#endif  // ARENA_AUDITORY_VIZ_SPAWN_MICROPHONE_TOOL_HPP
