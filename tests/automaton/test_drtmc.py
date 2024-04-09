import numpy as np
import pytest

from ns_vfs.automaton.drtmc import DRTMarkovChain
from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.data.frame import Frame


@pytest.mark.parametrize(
    "tl_specification, expected_or_error",
    [
        # Two propositions scenarios
        ("P>=0.80 G [prop1 & prop2]", [{"prop1": True, "prop2": True}]),
        (
            "P>=0.80 G [prop1 | prop2]",
            [
                {"prop1": True, "prop2": False},
                {"prop1": False, "prop2": True},
                {"prop1": True, "prop2": True},
            ],
        ),
        ("P>=0.80 G [~prop1]", [{"prop1": False}]),
        # Use "~prop1 | prop2" instead of "prop1 -> prop2"
        (
            "P>=0.80 G [~prop1 | prop2]",
            [
                {"prop1": False, "prop2": False},
                {"prop1": False, "prop2": True},
                {"prop1": True, "prop2": True},
            ],
        ),
        ("", "Expecting ValueError for invalid temporal logic specification"),
        # Three propositions scenarios
        (
            "P>=0.80 G [prop1 & prop2 & prop3]",
            [{"prop1": True, "prop2": True, "prop3": True}],
        ),
        (
            "P>=0.80 G [prop1 | prop2 | prop3]",
            [
                {"prop1": True, "prop2": False, "prop3": False},
                {"prop1": False, "prop2": True, "prop3": False},
                {"prop1": False, "prop2": False, "prop3": True},
                {"prop1": True, "prop2": True, "prop3": False},
                {"prop1": True, "prop2": False, "prop3": True},
                {"prop1": False, "prop2": True, "prop3": True},
                {"prop1": True, "prop2": True, "prop3": True},
            ],
        ),
        # Converted implications for three propositions
        # For "prop1 & (prop2 | prop3)"
        (
            "P>=0.80 G [prop1 & (~prop2 | prop3)]",
            [
                {"prop1": True, "prop2": False, "prop3": True},
                {"prop1": True, "prop2": False, "prop3": False},
                {"prop1": True, "prop2": True, "prop3": True},
            ],
        ),
        # For "(prop1 & prop2) -> prop3" use "~(prop1 & prop2) | prop3"
        (
            "P>=0.80 G [~(prop1 & prop2) | prop3]",
            [
                {"prop1": False, "prop2": False, "prop3": False},
                {"prop1": False, "prop2": True, "prop3": False},
                {"prop1": True, "prop2": False, "prop3": False},
                {"prop1": False, "prop2": False, "prop3": True},
                {"prop1": False, "prop2": True, "prop3": True},
                {"prop1": True, "prop2": False, "prop3": True},
                {"prop1": True, "prop2": True, "prop3": True},
            ],
        ),
        # For "prop1 -> (prop2 & ~prop3)" use "~prop1 | (prop2 & ~prop3)"
        (
            "P>=0.80 G [~prop1 | (prop2 & ~prop3)]",
            [
                {"prop1": False, "prop2": False, "prop3": True},
                {"prop1": False, "prop2": True, "prop3": True},
                {"prop1": False, "prop2": False, "prop3": False},
                {"prop1": False, "prop2": True, "prop3": False},
                {"prop1": True, "prop2": True, "prop3": False},
            ],
        ),
    ],
)
def test_get_satisfiable_propositions(tl_specification, expected_or_error):
    automaton = DRTMarkovChain(tl_specification)
    if "Expecting ValueError" in expected_or_error:
        with pytest.raises(ValueError):
            automaton.get_satisfiable_propositions(tl_specification)
    else:
        result = automaton.get_satisfiable_propositions(tl_specification)
        assert sorted(
            result,
            key=lambda x: (
                x.get("prop1", False),
                x.get("prop2", False),
                x.get("prop3", False),
            ),
        ) == sorted(
            expected_or_error,
            key=lambda x: (
                x.get("prop1", False),
                x.get("prop2", False),
                x.get("prop3", False),
            ),
        ), f"Failed on spec: {tl_specification}"


@pytest.fixture
def drtmc_with_two_props_with_or_relation():
    tl_specification = "P>=0.80 G [prop1 | prop2]"
    proposition_set = ["prop1", "prop2"]
    automaton = DRTMarkovChain(
        tl_specification=tl_specification, proposition_set=proposition_set
    )
    automaton.set_up(proposition_set)
    return automaton


@pytest.fixture
def drtmc_with_two_props_with_and_relation():
    tl_specification = "P>=0.80 G [prop1 & prop2]"
    proposition_set = ["prop1", "prop2"]
    automaton = DRTMarkovChain(
        tl_specification=tl_specification, proposition_set=proposition_set
    )
    automaton.set_up(proposition_set)
    return automaton


@pytest.fixture
def frame_with_two_props():
    detected_objects = {
        "prop1": DetectedObject(name="prop1", probability=0.8, confidence=0.8),
        "prop2": DetectedObject(name="prop2", probability=0.7, confidence=0.7),
    }
    frame = Frame(
        frame_idx=1,
        timestamp=123,  # Example timestamp
        frame_image=np.random.rand(100, 100, 3),
        object_of_interest=detected_objects,
        activity_of_interest=None,
    )
    return frame


def test_add_frame_to_automaton_first_frame_added_with_two_props_or(
    drtmc_with_two_props_with_or_relation, frame_with_two_props
):
    drtmc = drtmc_with_two_props_with_or_relation
    drtmc.add_frame_to_automaton(frame_with_two_props)
    assert len(drtmc.states) == 6
    assert len(drtmc.transitions) == 8
    assert drtmc.states[-1].current_proposition_combination == "terminal"
    assert drtmc.states[-2].current_proposition_combination == "terminal"
    assert drtmc.probability_of_safety == 0.94
    assert drtmc.probability_of_unsafety == 0.06


def test_add_frame_to_automaton_second_frame_added_with_two_props_or(
    drtmc_with_two_props_with_or_relation, frame_with_two_props
):
    drtmc = drtmc_with_two_props_with_or_relation
    drtmc.add_frame_to_automaton(frame_with_two_props)

    assert drtmc.probability_of_safety == 0.94
    assert drtmc.probability_of_unsafety == 0.06

    drtmc.add_frame_to_automaton(frame_with_two_props)

    assert len(drtmc.states) == 8
    assert len(drtmc.transitions) == 12
    assert drtmc.states[0].current_proposition_combination == "init"
    assert drtmc.states[1].current_proposition_combination == "init"
    assert drtmc.states[-1].current_proposition_combination == "terminal"
    assert drtmc.states[-2].current_proposition_combination == "terminal"

    # TODO: Do we need to test the transition probabilities?
    # for transitions in drtmc.transitions:
    #     start_state = transitions[0]
    #     end_state = transitions[1]
    #     transition_probability = transitions[2]
    #     if start_state == 0:
    #         assert end_state in [2, 3, 4, 5]
    #         assert transition_probability == round(
    #             drtmc.states[end_state].probability * probability_of_safety,
    #             4,
    #         )
    #     elif start_state == 1:
    #         assert end_state in [2, 3, 4, 5]
    #         assert transition_probability == round(
    #             drtmc.states[end_state].probability * probability_of_unsafety,
    #             4,
    #         )


def test_add_frame_to_automaton_third_frame_added_with_two_props_or(
    drtmc_with_two_props_with_or_relation, frame_with_two_props
):
    drtmc = drtmc_with_two_props_with_or_relation
    drtmc.add_frame_to_automaton(frame_with_two_props)
    drtmc.add_frame_to_automaton(frame_with_two_props)

    assert drtmc.probability_of_safety == 0.8836  # .94 * (.56 + .24 + .14)
    assert drtmc.probability_of_unsafety == 0.1164  # .94 * .06

    assert len(drtmc.states) == 8
    assert len(drtmc.transitions) == 12
    assert drtmc.states[0].current_proposition_combination == "init"
    assert drtmc.states[1].current_proposition_combination == "init"
    assert drtmc.states[-1].current_proposition_combination == "terminal"
    assert drtmc.states[-2].current_proposition_combination == "terminal"


def test_add_frame_to_automaton_first_frame_added_with_two_props_and(
    drtmc_with_two_props_with_and_relation, frame_with_two_props
):
    drtmc = drtmc_with_two_props_with_and_relation
    drtmc.add_frame_to_automaton(frame_with_two_props)
    assert len(drtmc.states) == 6
    assert len(drtmc.transitions) == 8
    assert drtmc.states[-1].current_proposition_combination == "terminal"
    assert drtmc.states[-2].current_proposition_combination == "terminal"
