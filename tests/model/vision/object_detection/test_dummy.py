from ns_vfs.model.vision.object_detection.dummy import DummyVisionModel


def test_dummy_vision_model_default(test_np_ndraary_image):
    dummy = DummyVisionModel(detection_probability=0.8)
    detected_object = dummy.detect(
        frame_img=test_np_ndraary_image, classes=["object 1"]
    )
    assert detected_object.probability == 0.8


def test_dummy_vision_model_default_greater_than(test_np_ndraary_image):
    dummy = DummyVisionModel(detection_probability=0.4, random_greater_than=0.9)
    detected_object = dummy.detect(
        frame_img=test_np_ndraary_image, classes=["object 1"]
    )
    assert detected_object.probability >= 0.9


def test_dummy_vision_model_default_less_than(test_np_ndraary_image):
    dummy = DummyVisionModel(
        detection_probability=0.8, random_prob_less_than=0.3
    )
    detected_object = dummy.detect(
        frame_img=test_np_ndraary_image, classes=["object 1"]
    )
    assert detected_object.probability <= 0.3
