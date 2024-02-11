import numpy as np
import pytest

from ns_vfs.data.frame import Frame
from ns_vfs.enums.symbolic_filter_rule import SymbolicFilterRule
from ns_vfs.validator.frame_validator import FrameValidator


@pytest.fixture
def all_below_threshold_frame(prob_0_detected_object, prob_0_3_detected_object):
    return Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.zeros((100, 100, 3), dtype=np.uint8),
        annotated_image={},
        object_of_interest={
            prob_0_detected_object.name: prob_0_detected_object,
            prob_0_3_detected_object.name: prob_0_3_detected_object,
        },
        activity_of_interest=None,
    )


@pytest.fixture
def frame_validator_threshold_0_5():
    return FrameValidator(ltl_formula="", threshold_of_probability=0.5)


def test_validate_frame_all_below(
    frame_validator_threshold_0_5, all_below_threshold_frame
):
    assert (
        frame_validator_threshold_0_5.validate_frame(
            all_below_threshold_frame, is_symbolic_verification=False
        )
        is False
    )


def test_validate_frame_one_above(
    frame_validator_threshold_0_5, frame_with_and_without_detected_object
):
    assert (
        frame_validator_threshold_0_5.validate_frame(
            frame_with_and_without_detected_object,
            is_symbolic_verification=False,
        )
        is True
    )


def test_get_symbolic_rule_from_ltl_formula_avoid_props(
    frame_validator_threshold_0_5,
):
    validator = frame_validator_threshold_0_5
    ltl_formula = 'P>=0.80 ["prop1" & "prop2" & !"prop3"]'
    symbolic_verification_rule = validator.get_symbolic_rule_from_ltl_formula(
        ltl_formula
    )
    avoid_props = symbolic_verification_rule.get(
        SymbolicFilterRule.AVOID_PROPOSITION
    )
    assert avoid_props == "prop3"


def test_get_symbolic_rule_from_ltl_formula_and_associated_props_I(
    frame_validator_threshold_0_5,
):
    validator = frame_validator_threshold_0_5
    ltl_formula = 'P>=0.80 [("prop1" & "prop2")]'
    symbolic_verification_rule = validator.get_symbolic_rule_from_ltl_formula(
        ltl_formula
    )
    associated_props = symbolic_verification_rule.get(
        SymbolicFilterRule.AND_ASSOCIATED_PROPS
    )
    assert associated_props == ["prop1", "prop2"]


def test_get_symbolic_rule_from_ltl_formula_and_associated_props_II(
    frame_validator_threshold_0_5,
):
    validator = frame_validator_threshold_0_5
    ltl_formula = 'P>=0.80 [("prop1" & "prop2") U "prop3"]'
    symbolic_verification_rule = validator.get_symbolic_rule_from_ltl_formula(
        ltl_formula
    )
    associated_props = symbolic_verification_rule.get(
        SymbolicFilterRule.AND_ASSOCIATED_PROPS
    )
    assert associated_props == ["prop1", "prop2"]
