"""Field-level schema validation for arena_auditory_msgs types."""


def test_audio_frame_msg():
    from arena_auditory_msgs.msg import AudioFrame
    from geometry_msgs.msg import Point

    msg = AudioFrame()
    msg.header.frame_id = "jackal/base_link"
    msg.sample_rate = 16000
    msg.channel_count = 4
    msg.frame_count = 2
    msg.encoding = "32FC1"
    msg.interleaved = True
    msg.channel_names = ["front_left", "front_right", "rear_left", "rear_right"]
    msg.frame_ids = ["jackal/mic_front_left", "jackal/mic_front_right", "jackal/mic_rear_left", "jackal/mic_rear_right"]
    msg.microphone_positions = [Point(x=0.19, y=0.135, z=0.22)] * 4
    msg.microphone_yaw_rad = [0.785398, -0.785398, 2.356194, -2.356194]
    msg.data = [0.0] * 8

    assert msg.sample_rate == 16000
    assert msg.channel_count == 4
    assert msg.frame_count == 2
    assert len(msg.data) == msg.channel_count * msg.frame_count


def test_spawn_sound_srv():
    from arena_auditory_msgs.srv import SpawnSound

    req = SpawnSound.Request()
    req.pose.header.frame_id = "map"
    req.kind = "music"
    req.customize_playback = True
    req.asset_id = "custom_radio"
    req.source_volume_db = 70.0
    req.loop = True
    req.initially_active = False
    assert req.pose.header.frame_id == "map"
    assert req.kind == "music"
    assert req.asset_id == "custom_radio"
    assert req.initially_active is False

    res = SpawnSound.Response()
    res.entity = "runtime_music_1"
    res.success = True
    assert res.entity == "runtime_music_1"
    assert res.success is True


def test_spawn_microphone_srv():
    from arena_auditory_msgs.srv import SpawnMicrophone
    from geometry_msgs.msg import PointStamped

    req = SpawnMicrophone.Request()
    req.position = PointStamped()
    req.position.header.frame_id = "map"
    req.position.point.x = 2.0
    req.position.point.y = 3.0
    req.position.point.z = 1.5
    req.placement = "placed"
    req.attached_frame = "robot/base_link"

    assert req.position.header.frame_id == "map"
    assert req.position.point.z == 1.5
    assert req.placement == "placed"
    assert req.attached_frame == "robot/base_link"

    res = SpawnMicrophone.Response()
    res.listener_id = "microphone1"
    res.zone = "reception"
    res.attached_frame = "robot/base_link"
    res.success = True
    res.error_msg = ""

    assert res.listener_id == "microphone1"
    assert res.zone == "reception"
    assert res.attached_frame == "robot/base_link"
    assert res.success is True


def test_remove_sound_srv():
    from arena_auditory_msgs.srv import RemoveSound

    req = RemoveSound.Request()
    req.entity = "runtime_music_1"
    res = RemoveSound.Response()
    res.success = True

    assert req.entity == "runtime_music_1"
    assert res.success is True


def test_remove_microphone_srv():
    from arena_auditory_msgs.srv import RemoveMicrophone

    req = RemoveMicrophone.Request()
    req.listener_id = "microphone:map:placed:1"
    res = RemoveMicrophone.Response()
    res.success = True

    assert req.listener_id == "microphone:map:placed:1"
    assert res.success is True
