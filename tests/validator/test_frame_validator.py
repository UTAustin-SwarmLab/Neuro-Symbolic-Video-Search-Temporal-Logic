import pytest

from ns_vfs.data.frame import Frame


class MockFrame:
    def __init__(self, detected_objects=None, propositional_probabilities=None):
        self.detected_objects = detected_objects or []
        self.propositional_probabilities = propositional_probabilities or {}

    def is_any_object_detected(self):
        return bool(self.detected_objects)

    def __getitem__(self, item):
        return self.propositional_probabilities.get(item, 0)


from ns_vfs.validator.frame_validator import FrameValidator


@pytest.fixture
def frame_validator():
    return FrameValidator('!("bad_object") & "good_object1" & "good_object2"')


@pytest.fixture
def mock_frame_good():
    return Frame()
