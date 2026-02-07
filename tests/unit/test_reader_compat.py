from __future__ import annotations

from types import SimpleNamespace

from egologqa.io.reader import _extract_record_fields


def test_extract_record_fields_from_tuple_shape() -> None:
    schema = SimpleNamespace(name="sensor_msgs/msg/CompressedImage")
    channel = SimpleNamespace(topic="/rgb")
    message = SimpleNamespace(log_time=123, publish_time=124)
    ros_msg = object()
    out = _extract_record_fields((schema, channel, message, ros_msg))
    assert out[0] is schema
    assert out[1] is channel
    assert out[2] == 123
    assert out[3] == 124
    assert out[4] is ros_msg


def test_extract_record_fields_from_object_shape() -> None:
    schema = SimpleNamespace(name="sensor_msgs/msg/Imu")
    channel = SimpleNamespace(topic="/imu")
    ros_msg = object()
    item = SimpleNamespace(
        schema=schema,
        channel=channel,
        ros_msg=ros_msg,
        log_time_ns=111,
        publish_time_ns=222,
    )
    out = _extract_record_fields(item)
    assert out[0] is schema
    assert out[1] is channel
    assert out[2] == 111
    assert out[3] == 222
    assert out[4] is ros_msg
