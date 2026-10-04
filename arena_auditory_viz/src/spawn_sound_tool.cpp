#include "arena_auditory_viz/spawn_sound_tool.hpp"

#include <cmath>
#include <exception>
#include <memory>

#include <rviz_common/display_context.hpp>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/string_property.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include <pluginlib/class_list_macros.hpp>

namespace arena_auditory_viz
{
SpawnSoundTool::SpawnSoundTool()
{
  shortcut_key_ = 'r';

  target_node_property_ = new rviz_common::properties::StringProperty(
    "Target", "/task_generator_node",
    "Node providing runtime/spawn_sound.",
    getPropertyContainer(), SLOT(updateClient()), this);

  kind_property_ = new rviz_common::properties::StringProperty(
    "Kind", "music",
    "Environment sound kind of the spawned source. A rejected kind logs the accepted ones.",
    getPropertyContainer());

  height_property_ = new rviz_common::properties::FloatProperty(
    "Height", 1.2, "Source Z coordinate in the RViz Fixed Frame, in meters.",
    getPropertyContainer());
  height_property_->setMin(0.0);

  customize_property_ = new rviz_common::properties::BoolProperty(
    "Custom Playback", false,
    "Use the asset, volume, loop, and initial-state properties below.",
    getPropertyContainer());
  asset_property_ = new rviz_common::properties::StringProperty(
    "Asset ID", "",
    "Sound asset ID. Empty uses the default asset of the kind.",
    getPropertyContainer());
  volume_property_ = new rviz_common::properties::FloatProperty(
    "Source Volume", 0.0,
    "Source level in dB. Without Custom Playback the asset level applies.",
    getPropertyContainer());
  volume_property_->setMin(-120.0);
  volume_property_->setMax(160.0);
  loop_property_ = new rviz_common::properties::BoolProperty(
    "Loop", true, "Loop the selected WAV.", getPropertyContainer());
  initially_active_property_ = new rviz_common::properties::BoolProperty(
    "Start Immediately", true,
    "Start emission and playback when the source is spawned.",
    getPropertyContainer());
}

SpawnSoundTool::~SpawnSoundTool() = default;

void SpawnSoundTool::onInitialize()
{
  PoseTool::onInitialize();
  setName("Spawn Radio");

  auto node_abstraction = context_->getRosNodeAbstraction().lock();
  service_node_ = node_abstraction->get_raw_node();
  updateClient();
}

void SpawnSoundTool::updateClient()
{
  if (!service_node_) {
    return;
  }
  client_ = service_node_->create_client<arena_auditory_msgs::srv::SpawnSound>(
    target_node_property_->getStdString() + "/runtime/spawn_sound");
}

void SpawnSoundTool::onPoseSet(double x, double y, double theta)
{
  if (!client_) {
    updateClient();
  }

  auto request = std::make_shared<arena_auditory_msgs::srv::SpawnSound::Request>();
  request->pose.header.frame_id = context_->getFixedFrame().toStdString();
  request->pose.header.stamp = service_node_->now();
  request->pose.pose.position.x = x;
  request->pose.pose.position.y = y;
  request->pose.pose.position.z = height_property_->getFloat();
  request->pose.pose.orientation.z = std::sin(theta / 2.0);
  request->pose.pose.orientation.w = std::cos(theta / 2.0);
  request->kind = kind_property_->getStdString();
  request->customize_playback = customize_property_->getBool();
  request->asset_id = asset_property_->getStdString();
  request->source_volume_db = volume_property_->getFloat();
  request->loop = loop_property_->getBool();
  request->initially_active = initially_active_property_->getBool();

  if (!client_->service_is_ready()) {
    RCLCPP_WARN(
      service_node_->get_logger(),
      "spawn_sound service not available at %s",
      client_->get_service_name());
    return;
  }

  const auto logger = service_node_->get_logger();
  client_->async_send_request(
    request,
    [logger, kind = request->kind](
      rclcpp::Client<arena_auditory_msgs::srv::SpawnSound>::SharedFuture future)
    {
      try {
        const auto response = future.get();
        if (!response->success) {
          RCLCPP_WARN(
            logger, "spawn_sound rejected: %s",
            response->error_msg.c_str());
          return;
        }
        RCLCPP_INFO(
          logger, "spawned %s source %s",
          kind.c_str(), response->entity.c_str());
      } catch (const std::exception & exception) {
        RCLCPP_ERROR(logger, "spawn_sound failed: %s", exception.what());
      }
    });
}
}  // namespace arena_auditory_viz

PLUGINLIB_EXPORT_CLASS(arena_auditory_viz::SpawnSoundTool, rviz_common::Tool)
