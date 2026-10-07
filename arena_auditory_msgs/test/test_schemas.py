"""Field-level schema validation for arena_auditory_msgs types."""


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


def test_remove_microphone_srv():
    from arena_auditory_msgs.srv import RemoveMicrophone

    req = RemoveMicrophone.Request()
    req.listener_id = "microphone:map:placed:1"
    res = RemoveMicrophone.Response()
    res.success = True

    assert req.listener_id == "microphone:map:placed:1"
    assert res.success is True
