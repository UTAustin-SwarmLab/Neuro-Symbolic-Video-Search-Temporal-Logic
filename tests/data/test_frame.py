import numpy as np
import pytest

from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.data.frame import Frame


@pytest.fixture
def detected_object():
    return DetectedObject(
        name="object a",
        confidence=0.6,
        probability=0.8,
        confidence_of_all_obj=[0.3, 0.4, 0.6],
        probability_of_all_obj=[0.6, 0.7, 0.8],
        bounding_box_of_all_obj=None,
        all_obj_detected=None,
        number_of_detection=3,
        is_detected=True,
        model_name="yolo",
        supervision_detections=None,
    )


@pytest.fixture
def non_detected_object():
    return DetectedObject(
        name="object b",
        confidence=0.0,
        probability=0.0,
        confidence_of_all_obj=[],
        probability_of_all_obj=[],
        bounding_box_of_all_obj=None,
        all_obj_detected=None,
        number_of_detection=0,
        is_detected=False,
        model_name="yolo",
        supervision_detections=None,
    )


@pytest.fixture
def frame_with_detected_object(detected_object):
    return Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.zeros((100, 100, 3), dtype=np.uint8),
        annotated_image={},
        object_of_interest={
            "object a": detected_object,
        },
        activity_of_interest=None,
    )


@pytest.fixture
def frame_without_detected_object(non_detected_object):
    return Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.zeros((100, 100, 3), dtype=np.uint8),
        annotated_image={},
        object_of_interest={
            "object b": non_detected_object,
        },
        activity_of_interest=None,
    )


@pytest.fixture
def frame_with_and_without_detected_object(detected_object, non_detected_object):
    return Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.zeros((100, 100, 3), dtype=np.uint8),
        annotated_image={},
        object_of_interest={
            "object a": detected_object,
            "object b": non_detected_object,
        },
        activity_of_interest=None,
    )


def test_is_any_object_detected_true(frame_with_detected_object):
    assert frame_with_detected_object.is_any_object_detected() is True


def test_is_any_object_detected_false(frame_without_detected_object):
    assert frame_without_detected_object.is_any_object_detected() is False


def test_detected_object(frame_with_and_without_detected_object):
    obj = frame_with_and_without_detected_object.detected_object
    assert isinstance(obj, list)
    assert obj == ["object a"]
