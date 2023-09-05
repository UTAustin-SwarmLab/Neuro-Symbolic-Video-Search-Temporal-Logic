import numpy as np
import stormpy


def build_trans_matrix(transitions, states):
    matrix = np.zeros((len(states), len(states)))
    for t in transitions:
        matrix[t[0], t[1]] = t[2]
    trans_matrix = stormpy.build_sparse_matrix(matrix, list(range(len(states))))
    return trans_matrix


def build_label_func(states, props):
    state_labeling = stormpy.storage.StateLabeling(len(states))
    for label in props:
        state_labeling.add_label(label)
    for state in states:
        state_labeling.add_label_to_state(
            state.current_proposition_combination, state.state_index
        )

    state_labeling
