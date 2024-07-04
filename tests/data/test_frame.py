def test_is_any_object_detected_true(frame_with_detected_object):
    assert frame_with_detected_object.is_any_object_detected() is True


def test_is_any_object_detected_false(frame_without_detected_object):
    assert frame_without_detected_object.is_any_object_detected() is False


def test_detected_object(frame_with_and_without_detected_object):
    obj = frame_with_and_without_detected_object.detected_object
    assert isinstance(obj, list)
    assert obj == ["object 1"]
