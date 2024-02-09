import numpy as np
import pytest

from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.data.frame import Frame


@pytest.fixture
def prob_1_detected_object():
    return DetectedObject(
        name="object 1",
        confidence=0.8,
        probability=1.0,
        confidence_of_all_obj=[0.3, 0.4, 0.8],
        probability_of_all_obj=[0.6, 0.7, 1.0],
        bounding_box_of_all_obj=None,
        all_obj_detected=None,
        number_of_detection=3,
        is_detected=True,
        model_name="yolo",
        supervision_detections=None,
    )


@pytest.fixture
def prob_0_8_detected_object():
    return DetectedObject(
        name="object 0.8",
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
def prob_0_3_detected_object():
    return DetectedObject(
        name="object 0.3",
        confidence=0.1,
        probability=0.3,
        confidence_of_all_obj=[0.1],
        probability_of_all_obj=[0.3],
        bounding_box_of_all_obj=None,
        all_obj_detected=None,
        number_of_detection=0,
        is_detected=False,
        model_name="yolo",
        supervision_detections=None,
    )


@pytest.fixture
def prob_0_detected_object():
    return DetectedObject(
        name="object 0",
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
def frame_with_detected_object(prob_1_detected_object):
    return Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.zeros((100, 100, 3), dtype=np.uint8),
        annotated_image={},
        object_of_interest={
            prob_1_detected_object.name: prob_1_detected_object,
        },
        activity_of_interest=None,
    )


@pytest.fixture
def frame_without_detected_object(prob_0_detected_object):
    return Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.zeros((100, 100, 3), dtype=np.uint8),
        annotated_image={},
        object_of_interest={
            prob_0_detected_object.name: prob_0_detected_object,
        },
        activity_of_interest=None,
    )


@pytest.fixture
def frame_with_and_without_detected_object(
    prob_1_detected_object, prob_0_detected_object
):
    return Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.zeros((100, 100, 3), dtype=np.uint8),
        annotated_image={},
        object_of_interest={
            prob_1_detected_object.name: prob_1_detected_object,
            prob_0_detected_object.name: prob_0_detected_object,
        },
        activity_of_interest=None,
    )
