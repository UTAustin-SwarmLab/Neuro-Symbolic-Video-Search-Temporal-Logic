import numpy as np
import pytest

from ns_vfs.automaton.probabilistic import ProbabilisticAutomaton
from ns_vfs.data.frame import Frame


@pytest.fixture
def probabilistic_automaton_with_initial_state():
    return ProbabilisticAutomaton(
        include_initial_state=True, proposition_set=["object 1", "object 0"]
    )


@pytest.fixture
def probabilistic_automaton_without_initial_state():
    return ProbabilisticAutomaton(
        include_initial_state=False, proposition_set=["object 1", "object 0"]
    )


def test_probabilistic_automaton_with_initial_state(
    probabilistic_automaton_with_initial_state,
):
    automaton = probabilistic_automaton_with_initial_state
    assert len(automaton.previous_states) == 1
    assert len(automaton.states) == 1
    assert len(automaton.transitions) == 0
    assert automaton.proposition_set == ["object 1", "object 0"]
    assert automaton.probability_of_propositions == [[], []]


def test_probabilistic_automaton_without_initial_state(
    probabilistic_automaton_without_initial_state,
):
    automaton = probabilistic_automaton_without_initial_state
    assert len(automaton.previous_states) == 0
    assert len(automaton.states) == 0
    assert len(automaton.transitions) == 0
    assert automaton.proposition_set == ["object 1", "object 0"]
    assert automaton.probability_of_propositions == [[], []]


def test_create_proposition_combinations(probabilistic_automaton_without_initial_state):
    automaton = probabilistic_automaton_without_initial_state
    assert automaton._create_proposition_combinations(
        len(automaton.proposition_set)
    ) == ["TT", "TF", "FT", "FF"]


def test_update_probability_of_propositions(
    probabilistic_automaton_without_initial_state,
    frame_with_and_without_detected_object,
):
    frame = frame_with_and_without_detected_object
    automaton = probabilistic_automaton_without_initial_state
    automaton._update_probability_of_propositions(frame)
    assert automaton.probability_of_propositions == [[1.0], [0.0]]


def test_add_frame_to_automaton_1(
    prob_1_detected_object,
    prob_0_8_detected_object,
):
    # prob 1 and prob 0.8 are detected
    automaton_no_init = ProbabilisticAutomaton(
        include_initial_state=False,
        proposition_set=[prob_1_detected_object.name, prob_0_8_detected_object.name],
    )

    frame_1 = Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.zeros((100, 100, 3), dtype=np.uint8),
        annotated_image={},
        object_of_interest={
            prob_1_detected_object.name: prob_1_detected_object,
            prob_0_8_detected_object.name: prob_0_8_detected_object,
        },
        activity_of_interest=None,
    )
    automaton_no_init.add_frame_to_automaton(frame_1)
    first_state_prop = automaton_no_init.transitions[0][2]
    second_state_prop = automaton_no_init.transitions[1][2]
    assert len(automaton_no_init.transitions) == 2
    assert automaton_no_init.transitions[0][2] == 0.8
    assert automaton_no_init.transitions[1][2] == 0.2
    assert first_state_prop + second_state_prop == 1.0


def test_add_frame_to_automaton_2(
    prob_0_8_detected_object,
    prob_0_3_detected_object,
):
    # test case 1: prob 0.8 and prob 0.3 are detected
    # test case 2: add frame twice
    automaton_no_init = ProbabilisticAutomaton(
        include_initial_state=False,
        proposition_set=[prob_0_3_detected_object.name, prob_0_8_detected_object.name],
    )
    frame_2 = Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.zeros((100, 100, 3), dtype=np.uint8),
        annotated_image={},
        object_of_interest={
            prob_0_3_detected_object.name: prob_0_3_detected_object,
            prob_0_8_detected_object.name: prob_0_8_detected_object,
        },
        activity_of_interest=None,
    )

    automaton_no_init.add_frame_to_automaton(frame_2)

    assert len(automaton_no_init.transitions) == 4
    assert automaton_no_init.transitions[0][2] == 0.24
    assert automaton_no_init.transitions[1][2] == 0.06
    assert automaton_no_init.transitions[2][2] == 0.56
    assert automaton_no_init.transitions[3][2] == 0.14

    automaton_no_init.add_frame_to_automaton(frame_2)
    print(automaton_no_init.states)
    print(automaton_no_init.previous_states)
    print(automaton_no_init.transitions)

    assert len(automaton_no_init.transitions) == 20
