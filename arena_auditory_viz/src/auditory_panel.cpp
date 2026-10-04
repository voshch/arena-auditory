#include "arena_auditory_viz/auditory_panel.hpp"
#include "rviz_common/display_context.hpp"

#include <QHBoxLayout>
#include <QJsonArray>

#include <algorithm>
#include <array>
#include <exception>
#include <memory>

namespace
{
    struct MotorControlSpec
    {
        const char *name;
        const char *label;
        const char *suffix;
        double minimum;
        double maximum;
        double step;
        int decimals;
        const char *tooltip;
    };

    constexpr std::array<MotorControlSpec, 6> kMotorControlSpecs{{
        {"motor.trim_db", "Volume", " dB", -40.0, 6.0, 0.5, 2,
         "Overall motor level. -6 dB is half amplitude."},
        {"motor.frequency_scale", "Frequency", " x", 0.25, 4.0, 0.05, 2,
         "Pitch multiplier. Velocity still controls the pitch trajectory."},
        {"motor.tonal_gain_db", "Gear tone", " dB", -24.0, 12.0, 0.5, 1,
         "Level of the periodic gear-mesh tones."},
        {"motor.broadband_gain_db", "Mechanical noise", " dB", -40.0, 6.0, 0.5, 1,
         "Level of the broadband mechanical-noise layer."},
        {"motor.speed_exponent", "Velocity response", "", 0.25, 3.0, 0.05, 2,
         "Higher values make motor level change more strongly with wheel speed."},
        {"motor.velocity_smoothing_s", "Response smoothing", " s", 0.0, 0.5, 0.005, 3,
         "Time used to smooth wheel-velocity changes. Zero is immediate."},
    }};

    constexpr const char *kArrayPrefix = "array:";
    constexpr const char *kRobotMicrophonePrefix = "microphone:robot:";
    constexpr const char *kListenerPrefix = "listener.";
    constexpr const char *kListenerId = "listener.id";
    constexpr const char *kOutputEnabled = "output.enabled";
    constexpr const char *kOutputMotorEnabled = "output.motor.enabled";
    constexpr const char *kOutputAmbientEnabled = "output.ambient.enabled";
    constexpr const char *kPropagationEnabled = "propagation.enabled";

    using SetParametersFuture =
        std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>>;

    bool isMotorTuningParameter(const std::string &name)
    {
        for (const auto &spec : kMotorControlSpecs)
            if (name == spec.name)
                return true;
        return false;
    }

    bool isArrayListener(const QString &listener_id)
    {
        return listener_id.startsWith(kArrayPrefix);
    }

    bool splitArrayListener(const QString &listener_id, QString &robot, QString &microphone)
    {
        if (!isArrayListener(listener_id))
            return false;
        const QString rest = listener_id.mid(QString(kArrayPrefix).size());
        const int split = rest.lastIndexOf(':');
        if (split <= 0 || split == rest.size() - 1)
            return false;
        robot = rest.left(split);
        microphone = rest.mid(split + 1);
        return true;
    }

    QString listenerRobot(const QString &listener_id)
    {
        QString robot;
        QString microphone;
        if (splitArrayListener(listener_id, robot, microphone))
            return robot;
        if (!listener_id.startsWith(kRobotMicrophonePrefix))
            return {};
        const QStringList parts =
            listener_id.mid(QString(kRobotMicrophonePrefix).size()).split(':');
        if (parts.size() < 3)
            return {};
        return parts.mid(0, parts.size() - 2).join(':');
    }

    QString arrayListener(const QString &robot, const QString &microphone)
    {
        return QString(kArrayPrefix) + robot + ":" + microphone;
    }

    std::string normalizeNodePath(const std::string &path)
    {
        std::string normalized;
        normalized.reserve(path.size() + 1);
        normalized.push_back('/');
        bool previous_slash = true;
        for (const char character : path)
        {
            if (character == '/')
            {
                if (!previous_slash)
                    normalized.push_back(character);
                previous_slash = true;
                continue;
            }
            normalized.push_back(character);
            previous_slash = false;
        }
        if (normalized.size() > 1 && normalized.back() == '/')
            normalized.pop_back();
        return normalized;
    }

    bool serviceReady(const std::shared_ptr<rclcpp::AsyncParametersClient> &client)
    {
        return client && client->service_is_ready();
    }

    void applyParameters(
        const std::shared_ptr<rclcpp::AsyncParametersClient> &client,
        const std::vector<rclcpp::Parameter> &parameters,
        const rclcpp::Logger &logger,
        const std::string &target,
        std::function<void()> done)
    {
        if (!serviceReady(client))
        {
            if (done)
                done();
            return;
        }
        client->set_parameters(
            parameters,
            [logger, target, done](SetParametersFuture future)
            {
                try
                {
                    for (const auto &result : future.get())
                    {
                        if (result.successful)
                            continue;
                        RCLCPP_WARN(
                            logger,
                            "setting parameters on %s failed: %s",
                            target.c_str(),
                            result.reason.c_str());
                        break;
                    }
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        logger,
                        "setting parameters on %s failed: %s",
                        target.c_str(),
                        exception.what());
                }
                if (done)
                    done();
            });
    }

}
namespace arena_auditory_viz
{
    AuditoryPanel::AuditoryPanel(QWidget *parent) : Panel(parent)
    {
        root_layout = new QVBoxLayout(this);
    }

    AuditoryPanel::~AuditoryPanel() = default;

    void AuditoryPanel::onInitialize()
    {
        node_ptr = getDisplayContext()->getRosNodeAbstraction().lock();
        node = node_ptr->get_raw_node();
        node->get_logger().set_level(rclcpp::Logger::Level::Warn);
    }

    void AuditoryPanel::load(const rviz_common::Config &config)
    {
        rviz_common::Panel::load(config);

        QString result;
        if (config.mapGetString("Target", &result))
            task_generator_node = normalizeNodePath(result.toStdString());
        else
            task_generator_node = "/task_generator_node";

        array_renderer_node = normalizeNodePath(
            task_generator_node + "/array_renderer");
        listener_renderer_node = normalizeNodePath(
            task_generator_node + "/listener_renderer");
        propagation_node = normalizeNodePath(
            task_generator_node + "/sound_propagation_node");

        array_renderer_parameters_client =
            std::make_shared<rclcpp::AsyncParametersClient>(
                node,
                array_renderer_node);
        listener_renderer_parameters_client =
            std::make_shared<rclcpp::AsyncParametersClient>(
                node,
                listener_renderer_node);
        propagation_parameters_client =
            std::make_shared<rclcpp::AsyncParametersClient>(
                node,
                propagation_node);
        set_semantic_client =
            node->create_client<task_generator_msgs::srv::SetSemantic>(
                task_generator_node + "/semantics/set");
        remove_microphone_client =
            node->create_client<arena_auditory_msgs::srv::RemoveMicrophone>(
                task_generator_node + "/runtime/remove_microphone");
        remove_sound_client =
            node->create_client<arena_auditory_msgs::srv::RemoveSound>(
                task_generator_node + "/runtime/remove_sound");
        {
            rclcpp::QoS qos(rclcpp::KeepLast(1));
            qos.transient_local();
            microphone_listeners_sub =
                node->create_subscription<std_msgs::msg::String>(
                    task_generator_node + "/microphone_listeners",
                    qos,
                    [this](const std_msgs::msg::String::SharedPtr msg)
                    {
                        QMetaObject::invokeMethod(this, [this, data = msg->data]()
                        {
                            updateMicrophoneListeners(data);
                        }, Qt::QueuedConnection);
                    });
        }

        {
            rclcpp::QoS qos(rclcpp::KeepLast(1));
            qos.transient_local();
            semantic_snapshot_sub = node->create_subscription<
                task_generator_msgs::msg::SemanticSnapshot>(
                task_generator_node + "/state/semantics",
                qos,
                [this](
                    const task_generator_msgs::msg::SemanticSnapshot::SharedPtr msg)
                {
                    QMetaObject::invokeMethod(
                        this,
                        [this, msg]()
                        {
                            if (!sounds_tree || !sounds_group)
                                return;
                            QSignalBlocker blocker(sounds_tree);
                            QString selected_entity;
                            if (!sounds_tree->selectedItems().isEmpty())
                                selected_entity = sounds_tree->selectedItems()
                                                      .front()
                                                      ->data(0, Qt::UserRole)
                                                      .toString();
                            sounds_tree->clear();
                            for (const auto &entity : msg->entities)
                            {
                                if (entity.kind != "sound")
                                    continue;
                                bool sounding = false;
                                for (size_t index = 0;
                                     index < entity.predicate_names.size();
                                     ++index)
                                {
                                    if (entity.predicate_names[index] == "sounding")
                                    {
                                        sounding = entity.predicate_values[index];
                                        break;
                                    }
                                }
                                double volume_db = 0.0;
                                for (size_t index = 0;
                                     index < entity.continuous_names.size();
                                     ++index)
                                {
                                    if (entity.continuous_names[index] == "volume_db")
                                    {
                                        volume_db = entity.continuous_values[index];
                                        break;
                                    }
                                }
                                auto *item = new QTreeWidgetItem(sounds_tree);
                                item->setData(
                                    0,
                                    Qt::UserRole,
                                    QString::fromStdString(entity.entity));
                                item->setFlags(
                                    item->flags() | Qt::ItemIsUserCheckable);
                                item->setText(
                                    0,
                                    QString::fromStdString(entity.entity));
                                item->setText(
                                    1,
                                    QString::number(volume_db, 'f', 1) + " dB");
                                item->setCheckState(
                                    0,
                                    sounding ? Qt::Checked : Qt::Unchecked);
                                if (!selected_entity.isEmpty() &&
                                    item->data(0, Qt::UserRole).toString() == selected_entity)
                                    item->setSelected(true);
                            }
                            sounds_group->setEnabled(
                                sounds_tree->topLevelItemCount() > 0);
                        },
                        Qt::QueuedConnection);
                });
        }


        {
            rclcpp::QoS qos(rclcpp::KeepLast(1));
            qos.transient_local();
            episode_sub = node->create_subscription<task_generator_msgs::msg::EpisodeRecord>(
                task_generator_node + "/state/episode",
                qos,
                [this](const task_generator_msgs::msg::EpisodeRecord::SharedPtr)
                {
                    QMetaObject::invokeMethod(this, [this]()
                    {
                        refreshMotorPlayback();
                    }, Qt::QueuedConnection);
                });
        }

        setupUi();

        param_events_sub = node->create_subscription<rcl_interfaces::msg::ParameterEvent>(
            "/parameter_events",
            rclcpp::QoS(10),
            [this](const rcl_interfaces::msg::ParameterEvent::SharedPtr msg)
            {
                const bool from_array = msg->node == array_renderer_node;
                const bool from_listener = msg->node == listener_renderer_node;
                const bool from_propagation = msg->node == propagation_node;
                if (!from_array && !from_listener && !from_propagation)
                    return;
                auto touches = [&msg](const auto &predicate)
                {
                    for (const auto *parameters : {
                             &msg->new_parameters,
                             &msg->changed_parameters,
                             &msg->deleted_parameters})
                        for (const auto &parameter : *parameters)
                            if (predicate(parameter.name))
                                return true;
                    return false;
                };
                if (touches([](const std::string &name)
                    {
                        return name.rfind(kListenerPrefix, 0) == 0
                            || name == kOutputEnabled;
                    }))
                    refreshAudioListenerRouting();
                if (touches([](const std::string &name)
                    {
                        return name == kOutputAmbientEnabled
                            || name == kPropagationEnabled;
                    }))
                    refreshAuditoryControls();
                if (!from_array)
                    return;
                if (touches([](const std::string &name)
                    {
                        return name == kOutputMotorEnabled
                            || isMotorTuningParameter(name);
                    }))
                    refreshMotorPlayback();
                refreshArrayControls();
            });
        whenReady(
            [client = array_renderer_parameters_client]()
            {
                return client->service_is_ready();
            },
            [this]()
            {
                refreshMotorPlayback();
                refreshArrayControls();
                refreshAuditoryControls();
                refreshAudioListenerRouting();
            });
        whenReady(
            [client = listener_renderer_parameters_client]()
            {
                return client->service_is_ready();
            },
            [this]()
            {
                refreshAuditoryControls();
                refreshAudioListenerRouting();
            });
        whenReady(
            [client = propagation_parameters_client]()
            {
                return client->service_is_ready();
            },
            [this]()
            {
                if (audio_listener_selection_pending_)
                    setAudioListenerRouting();
                else
                    refreshAudioListenerRouting();
                refreshAuditoryControls();
            });
    }

    void AuditoryPanel::setArrayParameters(
        const std::vector<rclcpp::Parameter> &parameters)
    {
        if (!serviceReady(array_renderer_parameters_client))
        {
            refreshArrayControls();
            return;
        }
        array_renderer_parameters_client->set_parameters(
            parameters,
            [this](auto) { refreshArrayControls(); });
    }

    void AuditoryPanel::setRendererParameters(
        const std::vector<rclcpp::Parameter> &parameters)
    {
        applyParameters(
            array_renderer_parameters_client,
            parameters,
            node->get_logger(),
            "array_renderer",
            [this]() { refreshMotorPlayback(); });
        applyParameters(
            listener_renderer_parameters_client,
            parameters,
            node->get_logger(),
            "listener_renderer",
            {});
    }

    void AuditoryPanel::refreshArrayControls()
    {
        if (!serviceReady(array_renderer_parameters_client))
        {
            QMetaObject::invokeMethod(this, [this]()
            {
                if (microphone_array_group)
                    microphone_array_group->setEnabled(false);
            }, Qt::QueuedConnection);
            return;
        }
        array_renderer_parameters_client->get_parameters(
            {"array.enabled", "monitor.enabled", "array.muted",
             "monitor.master_gain_db", "monitor.gain_db", "monitor.front_gain",
             "monitor.rear_gain", "monitor.solo", "monitor.mode",
             "tdoa.enabled"},
            [this](std::shared_future<std::vector<rclcpp::Parameter>> future)
            {
                try
                {
                    const auto values = future.get();
                    if (values.size() != 10)
                        return;
                    QMetaObject::invokeMethod(this, [this, values]()
                    {
                        if (!microphone_array_group)
                            return;
                        microphone_array_group->setEnabled(true);
                        if (array_startup.empty())
                            array_startup.assign(values.begin() + 2, values.begin() + 9);
                        const std::array<QCheckBox *, 4> boxes{
                            array_enabled_checkbox, headphones_enabled_checkbox,
                            array_mute_checkbox, array_tdoa_checkbox};
                        const std::array<bool, 4> states{
                            values[0].as_bool(), values[1].as_bool(),
                            values[2].as_bool(), values[9].as_bool()};
                        for (std::size_t index = 0; index < boxes.size(); ++index)
                        {
                            QSignalBlocker blocker(boxes[index]);
                            boxes[index]->setChecked(states[index]);
                        }
                        {
                            QSignalBlocker blocker(array_master_gain_spinbox);
                            array_master_gain_spinbox->setValue(values[3].as_double());
                        }
                        {
                            QSignalBlocker blocker(array_monitor_gain_spinbox);
                            array_monitor_gain_spinbox->setValue(values[4].as_double());
                        }
                        {
                            QSignalBlocker blocker(array_front_gain_spinbox);
                            array_front_gain_spinbox->setValue(values[5].as_double());
                        }
                        {
                            QSignalBlocker blocker(array_rear_gain_spinbox);
                            array_rear_gain_spinbox->setValue(values[6].as_double());
                        }
                        {
                            QSignalBlocker blocker(array_solo_combobox);
                            const auto text = QString::fromStdString(values[7].as_string());
                            const int index = array_solo_combobox->findData(text);
                            if (index >= 0) array_solo_combobox->setCurrentIndex(index);
                        }
                        {
                            QSignalBlocker blocker(array_monitor_combobox);
                            const auto text = QString::fromStdString(values[8].as_string());
                            const int index = array_monitor_combobox->findData(text);
                            if (index >= 0) array_monitor_combobox->setCurrentIndex(index);
                        }
                    }, Qt::QueuedConnection);
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(node->get_logger(), "reading microphone array controls failed: %s", exception.what());
                }
            });
    }

    void AuditoryPanel::whenReady(std::function<bool()> ready_check,
                                  std::function<void()> action,
                                  std::chrono::milliseconds period)
    {
        if (ready_check()) { action(); return; }
        auto holder = std::make_shared<rclcpp::TimerBase::SharedPtr>();
        std::function<void()> tick =
            [holder, check = std::move(ready_check), act = std::move(action)]() mutable
            {
                if (!check()) return;
                if (*holder) (*holder)->cancel();
                holder->reset();
                act();
            };
        *holder = node->create_wall_timer(period, std::move(tick));
    }

    void AuditoryPanel::refreshMotorPlayback()
    {
        if (!serviceReady(array_renderer_parameters_client))
        {
            QMetaObject::invokeMethod(this, [this]()
            {
                syncMotorPlaybackCheckbox(false, false);
                syncMotorTuningControls({}, false);
            }, Qt::QueuedConnection);
            return;
        }

        std::vector<std::string> names{kOutputMotorEnabled};
        for (const auto &spec : kMotorControlSpecs)
            names.emplace_back(spec.name);
        array_renderer_parameters_client->get_parameters(
            names,
            [this](std::shared_future<std::vector<rclcpp::Parameter>> future)
            {
                bool available = false;
                bool enabled = false;
                std::vector<rclcpp::Parameter> parameters;
                try
                {
                    parameters = future.get();
                    if (!parameters.empty()
                        && parameters.front().get_type()
                            == rclcpp::ParameterType::PARAMETER_BOOL)
                    {
                        enabled = parameters.front().as_bool();
                        available = true;
                    }
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "reading motor playback parameter failed: %s",
                        exception.what());
                }
                QMetaObject::invokeMethod(
                    this,
                    [this, enabled, available, parameters]()
                {
                    syncMotorPlaybackCheckbox(enabled, available);
                    syncMotorTuningControls(parameters, available);
                },
                    Qt::QueuedConnection);
            });
    }

    void AuditoryPanel::setMotorPlaybackEnabled(bool enabled)
    {
        if (!serviceReady(array_renderer_parameters_client))
        {
            syncMotorPlaybackCheckbox(false, false);
            return;
        }
        motor_playback_checkbox->setEnabled(false);
        setRendererParameters({rclcpp::Parameter(kOutputMotorEnabled, enabled)});
    }

    void AuditoryPanel::syncMotorPlaybackCheckbox(
        bool enabled,
        bool available)
    {
        if (!motor_playback_checkbox)
            return;
        QSignalBlocker blocker(motor_playback_checkbox);
        motor_playback_checkbox->setChecked(enabled);
        motor_playback_checkbox->setEnabled(available);
        motor_playback_checkbox->setToolTip(
            available
                ? "Mutes only workstation motor audio. Propagation and robot hearing continue."
                : "Waiting for array_renderer.");
    }

    void AuditoryPanel::setMotorTuningParameter(
        const std::string &name,
        double value)
    {
        if (!serviceReady(array_renderer_parameters_client))
        {
            refreshMotorPlayback();
            return;
        }
        setRendererParameters({rclcpp::Parameter(name, value)});
    }

    void AuditoryPanel::syncMotorTuningControls(
        const std::vector<rclcpp::Parameter> &parameters,
        bool available)
    {
        if (motor_tuning_group)
            motor_tuning_group->setEnabled(available);
        for (const auto &parameter : parameters)
        {
            const auto found = motor_tuning_spinboxes.find(parameter.get_name());
            if (found == motor_tuning_spinboxes.end()
                || parameter.get_type()
                    != rclcpp::ParameterType::PARAMETER_DOUBLE)
            {
                continue;
            }
            motor_tuning_startup.try_emplace(parameter.get_name(), parameter.as_double());
            QSignalBlocker blocker(found->second);
            found->second->setValue(parameter.as_double());
        }
    }

    void AuditoryPanel::resetMotorTuning()
    {
        if (!serviceReady(array_renderer_parameters_client))
        {
            refreshMotorPlayback();
            return;
        }
        if (motor_tuning_startup.empty())
        {
            refreshMotorPlayback();
            return;
        }
        std::vector<rclcpp::Parameter> parameters;
        parameters.reserve(motor_tuning_startup.size());
        for (const auto &[name, value] : motor_tuning_startup)
            parameters.emplace_back(name, value);
        setRendererParameters(parameters);
    }

    void AuditoryPanel::refreshAudioListenerRouting()
    {
        if (audio_listener_selection_pending_
            && serviceReady(propagation_parameters_client))
        {
            setAudioListenerRouting();
            return;
        }
        auto disable = [this]()
        {
            QMetaObject::invokeMethod(this, [this]()
            {
                if (audio_listener_group)
                    audio_listener_group->setEnabled(false);
            }, Qt::QueuedConnection);
        };
        if (!serviceReady(propagation_parameters_client)
            || !serviceReady(array_renderer_parameters_client))
        {
            disable();
            return;
        }
        array_renderer_parameters_client->get_parameters(
            {kOutputEnabled},
            [this, disable](std::shared_future<std::vector<rclcpp::Parameter>> future)
            {
                bool array_output = false;
                try
                {
                    const auto parameters = future.get();
                    if (parameters.size() != 1)
                    {
                        disable();
                        return;
                    }
                    array_output = parameters.front().as_bool();
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "reading array renderer routing failed: %s",
                        exception.what());
                    disable();
                    return;
                }
                if (array_output)
                {
                    QMetaObject::invokeMethod(this, [this]()
                    {
                        syncAudioListenerRouting(true, "");
                    }, Qt::QueuedConnection);
                    return;
                }
                const auto source = serviceReady(listener_renderer_parameters_client)
                    ? listener_renderer_parameters_client
                    : propagation_parameters_client;
                source->get_parameters(
                    {kListenerId},
                    [this, disable](std::shared_future<std::vector<rclcpp::Parameter>> listener_future)
                    {
                        std::string listener_id;
                        try
                        {
                            const auto parameters = listener_future.get();
                            if (parameters.size() != 1)
                            {
                                disable();
                                return;
                            }
                            listener_id = parameters.front().as_string();
                        }
                        catch (const std::exception &exception)
                        {
                            RCLCPP_WARN(
                                node->get_logger(),
                                "reading audio listener routing failed: %s",
                                exception.what());
                            disable();
                            return;
                        }
                        QMetaObject::invokeMethod(this, [this, listener_id]()
                        {
                            syncAudioListenerRouting(false, listener_id);
                        }, Qt::QueuedConnection);
                    });
            });
    }

    void AuditoryPanel::updateMicrophoneListeners(
        const std::string &data)
    {
        microphone_listener_registry_ = data;
        if (!audio_listener_id_combobox)
            return;
        const auto document = QJsonDocument::fromJson(
            QByteArray::fromStdString(data));
        if (!document.isArray())
        {
            RCLCPP_WARN(
                node->get_logger(),
                "ignoring invalid microphone listener registry");
            return;
        }

        const QString selected = audio_listener_id_combobox->currentText();
        QStringList listener_items;
        array_microphones_.clear();
        for (const auto &value : document.array())
        {
            if (!value.isString())
                continue;
            const QString listener_id = value.toString();
            if (listener_id.trimmed().isEmpty())
                continue;
            listener_items.append(listener_id);
            QString robot;
            QString microphone;
            if (!splitArrayListener(listener_id, robot, microphone))
                continue;
            const std::string name = microphone.toStdString();
            if (std::find(array_microphones_.begin(), array_microphones_.end(), name)
                == array_microphones_.end())
                array_microphones_.push_back(name);
        }
        updateSoloChoices();

        {
            QSignalBlocker blocker(audio_listener_id_combobox);
            audio_listener_id_combobox->clear();
            audio_listener_id_combobox->addItems(listener_items);
            const int selected_index =
                audio_listener_id_combobox->findText(selected);
            if (selected_index >= 0 || selected.isEmpty())
                audio_listener_id_combobox->setCurrentIndex(selected_index);
        }
        syncSideMicrophoneButtons();
        if (selected.isEmpty())
            refreshAudioListenerRouting();
        else if (audio_listener_id_combobox->currentText() != selected)
            setAudioListenerRouting();
    }

    void AuditoryPanel::updateSoloChoices()
    {
        if (!array_solo_combobox)
            return;
        QSignalBlocker blocker(array_solo_combobox);
        const QVariant selected = array_solo_combobox->currentData();
        array_solo_combobox->clear();
        array_solo_combobox->addItem("Listen to all microphones", "");
        for (const auto &microphone : array_microphones_)
        {
            const QString name = QString::fromStdString(microphone);
            array_solo_combobox->addItem("Solo " + name, name);
        }
        const int index = array_solo_combobox->findData(selected);
        array_solo_combobox->setCurrentIndex(index >= 0 ? index : 0);
    }

    void AuditoryPanel::selectSideMicrophone(
        const std::string &listener_id)
    {
        if (!audio_listener_id_combobox || listener_id.empty())
            return;
        const int index = audio_listener_id_combobox->findText(
            QString::fromStdString(listener_id));
        if (index < 0)
            return;
        if (audio_listener_id_combobox->currentIndex() == index)
        {
            syncSideMicrophoneButtons();
            return;
        }
        audio_listener_id_combobox->setCurrentIndex(index);
    }

    void AuditoryPanel::syncSideMicrophoneButtons()
    {
        if (!audio_listener_id_combobox
            || !left_microphone_button
            || !right_microphone_button)
        {
            return;
        }

        const QString selected = audio_listener_id_combobox->currentText();
        auto pair_exists = [this](const QString &robot)
        {
            return !robot.isEmpty()
                && audio_listener_id_combobox->findText(
                    arrayListener(robot, "left")) >= 0
                && audio_listener_id_combobox->findText(
                    arrayListener(robot, "right")) >= 0;
        };

        QString preferred_robot = listenerRobot(selected);
        if (!pair_exists(preferred_robot))
            preferred_robot = listenerRobot(
                QString::fromStdString(left_microphone_listener_id_));
        if (!pair_exists(preferred_robot))
        {
            preferred_robot.clear();
            for (int index = 0;
                 index < audio_listener_id_combobox->count();
                 ++index)
            {
                QString robot;
                QString microphone;
                if (!splitArrayListener(
                        audio_listener_id_combobox->itemText(index),
                        robot,
                        microphone)
                    || microphone != "left"
                    || !pair_exists(robot))
                    continue;
                preferred_robot = robot;
                break;
            }
        }

        if (pair_exists(preferred_robot))
        {
            left_microphone_listener_id_ =
                arrayListener(preferred_robot, "left").toStdString();
            right_microphone_listener_id_ =
                arrayListener(preferred_robot, "right").toStdString();
        }
        else
        {
            left_microphone_listener_id_.clear();
            right_microphone_listener_id_.clear();
        }

        const bool pair_available = !left_microphone_listener_id_.empty()
            && !right_microphone_listener_id_.empty();
        {
            QSignalBlocker blocker(left_microphone_button);
            left_microphone_button->setEnabled(pair_available);
            left_microphone_button->setChecked(
                selected.toStdString() == left_microphone_listener_id_);
            left_microphone_button->setToolTip(
                pair_available
                    ? QString::fromStdString(left_microphone_listener_id_)
                    : "Waiting for a robot left microphone.");
        }
        {
            QSignalBlocker blocker(right_microphone_button);
            right_microphone_button->setEnabled(pair_available);
            right_microphone_button->setChecked(
                selected.toStdString() == right_microphone_listener_id_);
            right_microphone_button->setToolTip(
                pair_available
                    ? QString::fromStdString(right_microphone_listener_id_)
                    : "Waiting for a robot right microphone.");
        }
    }

    void AuditoryPanel::setAudioListenerRouting()
    {
        if (!audio_listener_id_combobox)
            return;
        if (!serviceReady(propagation_parameters_client))
        {
            audio_listener_selection_pending_ = true;
            refreshAudioListenerRouting();
            return;
        }
        audio_listener_selection_pending_ = false;
        const QString selected = audio_listener_id_combobox->currentText();
        const std::string listener_id = selected.toStdString();
        audio_listener_group->setEnabled(false);
        const auto logger = node->get_logger();
        auto refresh = [this]() { refreshAudioListenerRouting(); };
        if (isArrayListener(selected))
        {
            applyParameters(
                listener_renderer_parameters_client,
                {rclcpp::Parameter(kListenerId, std::string())},
                logger,
                "listener_renderer",
                {});
            applyParameters(
                array_renderer_parameters_client,
                {rclcpp::Parameter(kOutputEnabled, true)},
                logger,
                "array_renderer",
                refresh);
            return;
        }
        applyParameters(
            array_renderer_parameters_client,
            {rclcpp::Parameter(kOutputEnabled, false)},
            logger,
            "array_renderer",
            {});
        applyParameters(
            listener_renderer_parameters_client,
            {rclcpp::Parameter(kListenerId, listener_id)},
            logger,
            "listener_renderer",
            {});
        applyParameters(
            propagation_parameters_client,
            {rclcpp::Parameter(kListenerId, listener_id)},
            logger,
            "sound_propagation_node",
            refresh);
    }

    void AuditoryPanel::syncAudioListenerRouting(
        bool array_output,
        const std::string &listener_id)
    {
        if (!audio_listener_group)
            return;
        audio_listener_group->setEnabled(true);
        QSignalBlocker blocker(audio_listener_id_combobox);
        QString target = QString::fromStdString(listener_id);
        if (array_output)
        {
            target = audio_listener_id_combobox->currentText();
            for (int index = 0;
                 !isArrayListener(target)
                 && index < audio_listener_id_combobox->count();
                 ++index)
                target = audio_listener_id_combobox->itemText(index);
        }
        const int selected_index =
            audio_listener_id_combobox->findText(target);
        if (selected_index >= 0 && (!array_output || isArrayListener(target)))
            audio_listener_id_combobox->setCurrentIndex(selected_index);
        syncSideMicrophoneButtons();
    }

    void AuditoryPanel::refreshAuditoryControls()
    {
        const bool propagation_available = serviceReady(propagation_parameters_client);
        const auto playback_client = serviceReady(listener_renderer_parameters_client)
            ? listener_renderer_parameters_client
            : array_renderer_parameters_client;
        const bool playback_available = serviceReady(playback_client);
        syncAuditoryControls(
            propagation_checkbox && propagation_checkbox->isChecked(),
            ambient_playback_checkbox
                && ambient_playback_checkbox->isChecked(),
            propagation_available,
            playback_available);
        if (propagation_available)
        {
            propagation_parameters_client->get_parameters(
                {kPropagationEnabled},
                [this](
                    std::shared_future<std::vector<rclcpp::Parameter>> future)
                {
                    bool enabled = false;
                    bool available = false;
                    try
                    {
                        const auto parameters = future.get();
                        available = parameters.size() == 1;
                        enabled = available && parameters.front().as_bool();
                    }
                    catch (const std::exception &exception)
                    {
                        RCLCPP_WARN(
                            node->get_logger(),
                            "reading propagation state failed: %s",
                            exception.what());
                    }
                    QMetaObject::invokeMethod(
                        this,
                        [this, enabled, available]()
                        {
                            if (!propagation_checkbox)
                                return;
                            QSignalBlocker blocker(propagation_checkbox);
                            propagation_checkbox->setChecked(enabled);
                            propagation_checkbox->setEnabled(available);
                        },
                        Qt::QueuedConnection);
                });
        }
        if (playback_available)
        {
            playback_client->get_parameters(
                {kOutputAmbientEnabled},
                [this](
                    std::shared_future<std::vector<rclcpp::Parameter>> future)
                {
                    bool enabled = false;
                    bool available = false;
                    try
                    {
                        const auto parameters = future.get();
                        available = parameters.size() == 1;
                        enabled = available && parameters.front().as_bool();
                    }
                    catch (const std::exception &exception)
                    {
                        RCLCPP_WARN(
                            node->get_logger(),
                            "reading ambient playback state failed: %s",
                            exception.what());
                    }
                    QMetaObject::invokeMethod(
                        this,
                        [this, enabled, available]()
                        {
                            if (!ambient_playback_checkbox)
                                return;
                            QSignalBlocker blocker(
                                ambient_playback_checkbox);
                            ambient_playback_checkbox->setChecked(enabled);
                            ambient_playback_checkbox->setEnabled(available);
                        },
                        Qt::QueuedConnection);
                });
        }
    }

    void AuditoryPanel::setPropagationEnabled(bool enabled)
    {
        if (!serviceReady(propagation_parameters_client))
        {
            refreshAuditoryControls();
            return;
        }
        propagation_checkbox->setEnabled(false);
        propagation_parameters_client->set_parameters(
            {rclcpp::Parameter(kPropagationEnabled, enabled)},
            [this](auto) { refreshAuditoryControls(); });
    }

    void AuditoryPanel::setAmbientPlaybackEnabled(bool enabled)
    {
        if (!serviceReady(array_renderer_parameters_client)
            && !serviceReady(listener_renderer_parameters_client))
        {
            refreshAuditoryControls();
            return;
        }
        ambient_playback_checkbox->setEnabled(false);
        const std::vector<rclcpp::Parameter> parameters{
            rclcpp::Parameter(kOutputAmbientEnabled, enabled)};
        applyParameters(
            array_renderer_parameters_client,
            parameters,
            node->get_logger(),
            "array_renderer",
            {});
        applyParameters(
            listener_renderer_parameters_client,
            parameters,
            node->get_logger(),
            "listener_renderer",
            [this]() { refreshAuditoryControls(); });
    }

    void AuditoryPanel::syncAuditoryControls(
        bool propagation_enabled,
        bool playback_enabled,
        bool propagation_available,
        bool playback_available)
    {
        if (propagation_checkbox)
        {
            QSignalBlocker blocker(propagation_checkbox);
            propagation_checkbox->setChecked(propagation_enabled);
            propagation_checkbox->setEnabled(propagation_available);
        }
        if (ambient_playback_checkbox)
        {
            QSignalBlocker blocker(ambient_playback_checkbox);
            ambient_playback_checkbox->setChecked(playback_enabled);
            ambient_playback_checkbox->setEnabled(playback_available);
        }
    }

    void AuditoryPanel::removeSelectedSound()
    {
        if (!sounds_tree
            || !remove_sound_client
            || !remove_sound_client->service_is_ready())
            return;
        const auto items = sounds_tree->selectedItems();
        if (items.isEmpty())
            return;
        const std::string entity =
            items.front()->data(0, Qt::UserRole).toString().toStdString();
        auto request = std::make_shared<
            arena_auditory_msgs::srv::RemoveSound::Request>();
        request->entity = entity;
        remove_sound_client->async_send_request(
            request,
            [this, entity]( rclcpp::Client<arena_auditory_msgs::srv::RemoveSound>::SharedFuture future)
            {
                try
                {
                    const auto response = future.get();
                    if (!response->success)
                        RCLCPP_WARN(
                            node->get_logger(),
                            "removing sound %s failed: %s",
                            entity.c_str(), response->error_msg.c_str());
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "removing sound %s failed: %s",
                        entity.c_str(), exception.what());
                }
            });
    }

    void AuditoryPanel::setSoundSounding(
        const std::string &entity,
        bool sounding)
    {
        if (entity.empty()
            || !set_semantic_client
            || !set_semantic_client->service_is_ready())
        {
            RCLCPP_WARN(
                node->get_logger(),
                "semantics service is unavailable, cannot set sounding for %s",
                entity.c_str());
            return;
        }
        auto request =
            std::make_shared<task_generator_msgs::srv::SetSemantic::Request>();
        request->entity = entity;
        request->field = "sounding";
        request->value = sounding ? "true" : "false";
        set_semantic_client->async_send_request(
            request,
            [this, entity](
                rclcpp::Client<task_generator_msgs::srv::SetSemantic>::SharedFuture
                    future)
            {
                try
                {
                    const auto response = future.get();
                    if (!response->success)
                    {
                        RCLCPP_WARN(
                            node->get_logger(),
                            "setting sounding on %s failed: %s",
                            entity.c_str(),
                            response->error_msg.c_str());
                    }
                }
                catch (const std::exception &exception)
                {
                    RCLCPP_WARN(
                        node->get_logger(),
                        "setting sounding on %s failed: %s",
                        entity.c_str(),
                        exception.what());
                }
            });
    }

    void AuditoryPanel::setupUi()
    {
        auto auditory_controls_group = new QGroupBox("Auditory Runtime");
        auto auditory_controls_layout = new QVBoxLayout();
        propagation_checkbox = new QCheckBox(
            "Enable sound propagation");
        propagation_checkbox->setEnabled(false);
        propagation_checkbox->setToolTip(
            "Stops or resumes listener-specific simulated sound propagation.");
        connect(
            propagation_checkbox,
            &QCheckBox::toggled,
            this,
            &AuditoryPanel::setPropagationEnabled);
        ambient_playback_checkbox = new QCheckBox(
            "Play environment audio on this workstation");
        ambient_playback_checkbox->setEnabled(false);
        ambient_playback_checkbox->setToolTip(
            "Mutes only local output. Propagation and robot hearing continue.");
        connect(
            ambient_playback_checkbox,
            &QCheckBox::toggled,
            this,
            &AuditoryPanel::setAmbientPlaybackEnabled);
        auditory_controls_layout->addWidget(propagation_checkbox);
        auditory_controls_layout->addWidget(ambient_playback_checkbox);
        auditory_controls_group->setLayout(auditory_controls_layout);
        root_layout->addWidget(auditory_controls_group);

        microphone_array_group = new QGroupBox("Robot Microphone Array");
        microphone_array_group->setEnabled(false);
        auto array_layout = new QFormLayout();
        array_enabled_checkbox = new QCheckBox("Enable microphone array");
        headphones_enabled_checkbox = new QCheckBox("Enable headphone playback");
        headphones_enabled_checkbox->setToolTip(
            "Enables the two-channel monitor derived from the raw array microphones.");
        array_mute_checkbox = new QCheckBox("Mute all array outputs");
        array_mute_checkbox->setToolTip(
            "Silences every array microphone and every product derived from them.");
        array_tdoa_checkbox = new QCheckBox("Enable TDoA diagnostics");
        connect(array_enabled_checkbox, &QCheckBox::toggled, this,
            [this](bool value) { setArrayParameters({rclcpp::Parameter("array.enabled", value)}); });
        connect(headphones_enabled_checkbox, &QCheckBox::toggled, this,
            [this](bool value) { setArrayParameters({rclcpp::Parameter("monitor.enabled", value)}); });
        connect(array_mute_checkbox, &QCheckBox::toggled, this,
            [this](bool value) { setArrayParameters({rclcpp::Parameter("array.muted", value)}); });
        connect(array_tdoa_checkbox, &QCheckBox::toggled, this,
            [this](bool value) { setArrayParameters({rclcpp::Parameter("tdoa.enabled", value)}); });
        array_layout->addRow(array_enabled_checkbox);
        array_layout->addRow(headphones_enabled_checkbox);
        array_layout->addRow(array_mute_checkbox);

        array_master_gain_spinbox = new QDoubleSpinBox();
        array_monitor_gain_spinbox = new QDoubleSpinBox();
        array_front_gain_spinbox = new QDoubleSpinBox();
        array_rear_gain_spinbox = new QDoubleSpinBox();
        for (auto spinbox : {array_front_gain_spinbox, array_rear_gain_spinbox})
        {
            spinbox->setRange(0.0, 4.0);
            spinbox->setSingleStep(0.05);
            spinbox->setDecimals(2);
        }
        array_master_gain_spinbox->setRange(-60.0, 12.0);
        array_master_gain_spinbox->setSingleStep(0.5);
        array_master_gain_spinbox->setDecimals(2);
        array_master_gain_spinbox->setSuffix(" dB");
        array_monitor_gain_spinbox->setRange(0.0, 60.0);
        array_monitor_gain_spinbox->setSingleStep(1.0);
        array_monitor_gain_spinbox->setDecimals(1);
        connect(array_master_gain_spinbox, &QDoubleSpinBox::editingFinished, this,
            [this]() { setArrayParameters({rclcpp::Parameter("monitor.master_gain_db", array_master_gain_spinbox->value())}); });
        connect(array_monitor_gain_spinbox, &QDoubleSpinBox::editingFinished, this,
            [this]() { setArrayParameters({rclcpp::Parameter("monitor.gain_db", array_monitor_gain_spinbox->value())}); });
        connect(array_front_gain_spinbox, &QDoubleSpinBox::editingFinished, this,
            [this]() { setArrayParameters({rclcpp::Parameter("monitor.front_gain", array_front_gain_spinbox->value())}); });
        connect(array_rear_gain_spinbox, &QDoubleSpinBox::editingFinished, this,
            [this]() { setArrayParameters({rclcpp::Parameter("monitor.rear_gain", array_rear_gain_spinbox->value())}); });
        array_layout->addRow("Master gain", array_master_gain_spinbox);
        array_layout->addRow("Monitor preamp (dB)", array_monitor_gain_spinbox);
        array_layout->addRow("Front contribution", array_front_gain_spinbox);
        array_layout->addRow("Rear contribution", array_rear_gain_spinbox);

        array_solo_combobox = new QComboBox();
        updateSoloChoices();
        array_solo_combobox->setToolTip(
            "Diagnostic monitor routing only, the raw array always keeps every microphone.");
        connect(array_solo_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) { setArrayParameters({rclcpp::Parameter("monitor.solo", array_solo_combobox->itemData(index).toString().toStdString())}); });
        array_layout->addRow("Channels", array_solo_combobox);

        array_monitor_combobox = new QComboBox();
        array_monitor_combobox->addItem("Spatial stereo (normal)", "stereo");
        array_monitor_combobox->addItem("Mono detection preview", "hearing");
        array_monitor_combobox->setToolTip(
            "Spatial stereo maps left-side microphones to the left ear and right-side ones to the right. "
            "Mono preview duplicates the highest-energy microphone into both ears for diagnostics.");
        connect(array_monitor_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) { setArrayParameters({rclcpp::Parameter("monitor.mode", array_monitor_combobox->itemData(index).toString().toStdString())}); });
        array_layout->addRow("Headphone output", array_monitor_combobox);
        array_layout->addRow(array_tdoa_checkbox);
        auto reset_array_button = new QPushButton("Reset gains and routing");
        connect(reset_array_button, &QPushButton::clicked, this, [this]()
        {
            if (array_startup.empty())
            {
                refreshArrayControls();
                return;
            }
            setArrayParameters(array_startup);
        });
        array_layout->addRow(reset_array_button);
        microphone_array_group->setLayout(array_layout);
        root_layout->addWidget(microphone_array_group);

        audio_listener_group = new QGroupBox("Workstation Listener");
        audio_listener_group->setEnabled(false);
        auto audio_listener_layout = new QFormLayout();
        audio_listener_id_combobox = new QComboBox();
        audio_listener_id_combobox->setToolTip(
            "An array microphone plays the robot array monitor. Any other listener "
            "plays through the listener renderer.");
        connect(
            audio_listener_id_combobox,
            &QComboBox::currentTextChanged,
            this,
            [this](const QString &)
            {
                syncSideMicrophoneButtons();
                setAudioListenerRouting();
            });
        audio_listener_layout->addRow(
            "Listen through",
            audio_listener_id_combobox);
        auto side_microphone_layout = new QHBoxLayout();
        left_microphone_button = new QPushButton("Left microphone");
        left_microphone_button->setCheckable(true);
        left_microphone_button->setEnabled(false);
        connect(
            left_microphone_button,
            &QPushButton::clicked,
            this,
            [this]()
            {
                selectSideMicrophone(left_microphone_listener_id_);
            });
        right_microphone_button = new QPushButton("Right microphone");
        right_microphone_button->setCheckable(true);
        right_microphone_button->setEnabled(false);
        connect(
            right_microphone_button,
            &QPushButton::clicked,
            this,
            [this]()
            {
                selectSideMicrophone(right_microphone_listener_id_);
            });
        side_microphone_layout->addWidget(left_microphone_button);
        side_microphone_layout->addWidget(right_microphone_button);
        audio_listener_layout->addRow(
            "Robot side",
            side_microphone_layout);
        if (!microphone_listener_registry_.empty())
            updateMicrophoneListeners(microphone_listener_registry_);
        audio_listener_group->setLayout(audio_listener_layout);
        root_layout->addWidget(audio_listener_group);

        sounds_group = new QGroupBox("Sound Entities");
        sounds_group->setEnabled(false);
        auto sounds_layout = new QVBoxLayout();
        sounds_tree = new QTreeWidget();
        sounds_tree->setColumnCount(2);
        sounds_tree->setHeaderLabels(
            QStringList{"Active / entity", "Volume"});
        sounds_tree->setRootIsDecorated(false);
        sounds_tree->setToolTip(
            "Check a sound to start it. Volume reflects the current "
            "semantic volume_db.");
        connect(
            sounds_tree,
            &QTreeWidget::itemChanged,
            this,
            [this](QTreeWidgetItem *item, int column)
            {
                if (column != 0)
                    return;
                setSoundSounding(
                    item->data(0, Qt::UserRole).toString().toStdString(),
                    item->checkState(0) == Qt::Checked);
            });
        sounds_layout->addWidget(sounds_tree);
        remove_sound_button = new QPushButton(
            "Remove selected runtime source");
        remove_sound_button->setEnabled(false);
        connect(
            remove_sound_button,
            &QPushButton::clicked,
            this,
            &AuditoryPanel::removeSelectedSound);
        connect(
            sounds_tree,
            &QTreeWidget::itemSelectionChanged,
            this,
            [this]()
            {
                remove_sound_button->setEnabled(
                    !sounds_tree->selectedItems().isEmpty());
            });
        sounds_layout->addWidget(remove_sound_button);
        sounds_group->setLayout(sounds_layout);
        root_layout->addWidget(sounds_group);

        motor_playback_checkbox = new QCheckBox(
            "Play robot motor audio on this workstation");
        motor_playback_checkbox->setEnabled(false);
        motor_playback_checkbox->setToolTip(
            "Waiting for array_renderer.");
        connect(
            motor_playback_checkbox,
            &QCheckBox::toggled,
            this,
            &AuditoryPanel::setMotorPlaybackEnabled);
        root_layout->addWidget(motor_playback_checkbox);

        motor_tuning_group = new QGroupBox("Motor Sound Tuning");
        motor_tuning_group->setEnabled(false);
        auto motor_tuning_layout = new QFormLayout();
        for (const auto &spec : kMotorControlSpecs)
        {
            auto spinbox = new QDoubleSpinBox();
            spinbox->setRange(spec.minimum, spec.maximum);
            spinbox->setSingleStep(spec.step);
            spinbox->setDecimals(spec.decimals);
            spinbox->setSuffix(spec.suffix);
            spinbox->setToolTip(spec.tooltip);
            motor_tuning_spinboxes.emplace(spec.name, spinbox);
            connect(
                spinbox,
                &QDoubleSpinBox::editingFinished,
                this,
                [this, name = std::string(spec.name), spinbox]()
                {
                    setMotorTuningParameter(name, spinbox->value());
                });
            motor_tuning_layout->addRow(spec.label, spinbox);
        }
        auto reset_motor_tuning_button = new QPushButton("Reset motor tuning");
        connect(
            reset_motor_tuning_button,
            &QPushButton::clicked,
            this,
            &AuditoryPanel::resetMotorTuning);
        motor_tuning_layout->addRow(reset_motor_tuning_button);
        motor_tuning_group->setLayout(motor_tuning_layout);
        root_layout->addWidget(motor_tuning_group);
    }
} // namespace arena_auditory_viz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(arena_auditory_viz::AuditoryPanel, rviz_common::Panel)
