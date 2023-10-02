from __future__ import annotations

import copy

import numpy as np

from ns_vfs.common.ltl_utility import get_not_operator_mapping, verification_result_eval
from ns_vfs.data.frame import Frame, FramesofInterest
from ns_vfs.state import State
from ns_vfs.verification import check_automaton


def _create_proposition_status(num_props):
    """Create all possible combinations of T and F for the number of propositions.

    Args:
        num_props (int): Number of propositions.


    Returns:
    list: List of all possible combinations of T and F.
    """
    label_list = []

    def add_labels(num_props, label, label_list):
        if len(label) == num_props:
            label_list.append(label)
            return
        add_labels(num_props, label + "T", label_list)
        add_labels(num_props, label + "F", label_list)

    add_labels(num_props, "", label_list)
    return label_list


def update_frame_of_interest(frame_set: list[Frame], frame_of_interest: FramesofInterest) -> FramesofInterest:
    if len(frame_set) > 1:
        frame_interval = list()
        for frame in frame_set:
            frame_interval.append(frame.frame_index)
            frame_of_interest.frame_idx_to_real_idx[frame.frame_index] = frame.real_frame_idx
            frame_of_interest.frame_images.append(frame.frame_image)
            frame_of_interest.save_annotated_images(frame.annotated_image)
        frame_of_interest.foi_list.append(frame_interval)
    else:
        for frame in frame_set:
            frame_of_interest.foi_list.append([frame.frame_index])
            frame_of_interest.frame_idx_to_real_idx[frame.frame_index] = frame.real_frame_idx
            frame_of_interest.frame_images.append(frame.frame_image)
            frame_of_interest.save_annotated_images(frame.annotated_image)

    return frame_of_interest


def _reset_frame_set_and_confidence(proposition_set: list) -> tuple[list, list]:
    """Reset frame set and confidence.

    Args:
        proposition_set (list): List of propositions.

    Returns:
        tuple[list, list]: Frame set and confidence.
    """
    frame_set = list()
    interim_confidence_set = [[] for _ in range(len(proposition_set))]
    return frame_set, interim_confidence_set


def build_automaton(
    frame_set: list[Frame],
    propositional_confidence: list[list[float]],
    proposition_combinations,
    proposition_set: list,
    include_initial_state: bool = True,
) -> (list, list):
    """Build automaton.

    Args:
        frame_set (list[Frame]): List of frames.
        propositional_confidence (list[list[float]]): List of propositional confidence.

    Returns:
        States: List of states.
        Transitions: List of transitions.
    """
    # Initialize required variables
    state_idx = 0
    states = list()
    prev_states = list()
    transitions = list()

    if include_initial_state:
        state = State(state_idx, -1, "init", proposition_set)
        states.append(copy.deepcopy(state))
        prev_states.append(copy.deepcopy(state))

    for i in range(len(frame_set)):
        current_state = list()
        for prop_comb in proposition_combinations:
            if len(states) == 0:
                state = State(
                    index=0,
                    frame_index=i,
                    proposition_status_set=prop_comb,
                    proposition_set=proposition_set,
                )
            else:
                state.update(
                    frame_index=i,
                    proposition_combinations=prop_comb,
                )
            state.compute_probability(probabilities=propositional_confidence)

            if state.probability > 0:
                if len(prev_states) == 0:
                    # prev_states.append(copy.deepcopy(state))
                    states.append(copy.deepcopy(state))
                    current_state.append(copy.deepcopy(state))
                    state.state_index += 1
                else:
                    if include_initial_state:
                        state.state_index += 1
                    states.append(copy.deepcopy(state))
                    current_state.append(copy.deepcopy(state))
                    if not include_initial_state:
                        state.state_index += 1

        if len(prev_states) == 0:
            prev_states = current_state.copy()
            for prev_state in prev_states:
                transition = (
                    prev_state.state_index,
                    prev_state.state_index,
                    0,
                )
                transitions.append(transition)
        else:
            for cur_state in current_state:
                for prev_state in prev_states:
                    transition = (
                        prev_state.state_index,
                        cur_state.state_index,
                        cur_state.probability,
                    )
                    transitions.append(transition)
            prev_states = current_state.copy()

    return states, transitions


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
label_set = [
    "prop1",
    "prop1",
    "random_label",
    "random_label",
    "prop2",
]
proposition_set = ["prop1", "prop2"]
ltl_formula = 'P>=0.90 [G "prop2"]'
# 'P>=0.90 [!F ("prop2" & !F "prop1")]'

# Creating a list of 32x32 black images
black_images = [np.zeros((32, 32), dtype=np.uint8) for _ in label_set]


if "!" in ltl_formula:
    not_operation_list = get_not_operator_mapping(ltl_formula)
else:
    not_operation_list = []

manual_confidence_probability = 1
frame_idx = 0
frame_set, interim_confidence_set = _reset_frame_set_and_confidence(proposition_set)
frame_of_interest = FramesofInterest(ltl_formula=ltl_formula)
propositional_probability = {}
proposition_combinations = _create_proposition_status(len(proposition_set))
test_result = []
for idx in range(len(label_set)):
    frame_img = black_images[idx]
    frame: Frame = Frame(
        frame_index=frame_idx,
        frame_image=frame_img,
    )
    for proposition in proposition_set:
        if isinstance(label_set[idx], list):
            for prop in label_set[idx]:
                if prop == proposition:
                    propositional_confidence = manual_confidence_probability
        else:
            if label_set[idx] == proposition:
                # if label_set[idx] == "prop2":
                #     manual_confidence_probability = 0
                propositional_confidence = manual_confidence_probability
            else:
                propositional_confidence = 1 - manual_confidence_probability
        frame.propositional_probability[str(proposition)] = propositional_confidence

    propositional_confidence_of_frame = frame.propositional_confidence
    proposition_condition = sum(propositional_confidence_of_frame)

    if proposition_condition > 0:
        frame_set.append(frame)
        # if len(not_operation_list) > 0:
        #     for prop in not_operation_list:
        #         if frame.propositional_probability[prop] > 0.5:
        #             frame_set.pop()

        for i in range(len(proposition_set)):
            interim_confidence_set[i].append(propositional_confidence_of_frame[i])

        states, transitions = build_automaton(
            frame_set,
            interim_confidence_set,
            proposition_combinations,
            proposition_set,
            include_initial_state=False,
        )
        verification_result = check_automaton(
            transitions=transitions,
            states=states,
            proposition_set=proposition_set,
            ltl_formula=ltl_formula,
            verbose=True,
            is_filter=False,
        )

        verification_result_str = str(verification_result)

        string_result = verification_result_str.split("{")[-1].split("}")[0]
        # string result is "true" when is absolutely true
        # but it returns "trus, false" when we have some true and false
        result = verification_result_eval(verification_result)

        if result == "PartialTrue":
            frame_set = [frame_set[-1]]
            result = True

        test_result.append(result)
        if result:
            # 2.1 Save result
            frame_of_interest = update_frame_of_interest(
                frame_set=frame_set, frame_of_interest=frame_of_interest
            )
            # 2.2 Reset frame set
            frame_set, interim_confidence_set = _reset_frame_set_and_confidence(proposition_set)
    frame_idx += 1
frame_of_interest.reorder_frame_of_interest()
print(test_result)
print(frame_of_interest.foi_list)
