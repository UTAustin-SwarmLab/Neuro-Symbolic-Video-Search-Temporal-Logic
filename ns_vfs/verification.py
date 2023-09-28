from __future__ import annotations

import numpy as np
import stormpy
import stormpy.examples.files

from ns_vfs.state import State


def check_automaton(
    transitions: list[tuple[int, int, float]],
    states: list[State],
    proposition_set: list[str],
    ltl_formula: str,
    verbose: bool = False,
) -> any:
    """Check automaton.

    Args:
        transitions (list[tuple[int, int, float]]): List of transitions.
        states (list[State]): List of states.
        proposition_set (list[str]): List of propositions.
        ltl_formula (str): LTL formula.
    """
    transition_matrix = build_trans_matrix(transitions=transitions, states=states)

    state_labeling = build_label_func(states, proposition_set)

    markovian_states = stormpy.BitVector(len(states), list(range(len(states))))

    components = stormpy.SparseModelComponents(
        transition_matrix=transition_matrix,
        state_labeling=state_labeling,
        markovian_states=markovian_states,
    )
    components.exit_rates = [0.0 for i in range(len(states))]
    # [0.0 if i != len(states) - 1 else 1 for i in range(len(states))]

    # Markov Automaton
    markov_automata = stormpy.storage.SparseMA(components)

    formula_str = ltl_formula

    if verbose:
        print(transition_matrix)
        print(markov_automata)

    # Check the model (Markov Automata)
    return model_checking(markov_automata, formula_str)


def model_checking(model: stormpy.storage.SparseMA, formula_str: str) -> any:
    """Model checking.

    Args:
        model (stormpy.storage.SparseMA): Markov Automata.
        formula_str (str): Formula string.

    Returns:
    any: Result.
    """
    # Initialize Prism Program
    path = stormpy.examples.files.prism_dtmc_die  #  prism_mdp_maze
    prism_program = stormpy.parse_prism_program(path)

    # Define Properties
    properties = stormpy.parse_properties(formula_str, prism_program)

    # Get Result and Filter it
    result = stormpy.model_checking(model, properties[0])
    filter = stormpy.create_filter_initial_states_sparse(model)
    result.filter(filter)
    return result


def build_trans_matrix(transitions: list[tuple[int, int, float]], states: list[State]):
    """Build transition matrix.

    Args:
        transitions (list[tuple[int, int, float]]): List of transitions.
        states (list[State]): List of states.
    """
    matrix = np.zeros((len(states), len(states)))
    for t in transitions:
        matrix[int(t[0]), int(t[1])] = float(t[2])
    trans_matrix = stormpy.build_sparse_matrix(matrix, list(range(len(states))))
    return trans_matrix


def build_label_func(states: list[State], props: list[str]) -> stormpy.storage.StateLabeling:
    """Build label function.

    Args:
        states (list[State]): List of states.
        props (list[str]): List of propositions.


    Returns:
        stormpy.storage.StateLabeling: State labeling.
    """
    state_labeling = stormpy.storage.StateLabeling(len(states))
    state_labeling.add_label("init")

    for label in props:
        state_labeling.add_label(label)

    for state in states:
        for label in state.current_descriptive_label:
            state_labeling.add_label_to_state(label, state.state_index)
            if label == "init":
                state_labeling.add_label_to_state(label, state.state_index)
        # if state.state_index == 0:
        #     state_labeling.add_label("init")
        #     state_labeling.add_label_to_state("init", state.state_index)
        # else:
        #     for label in state.current_descriptive_label:
        #         state_labeling.add_label_to_state(label, state.state_index)

    return state_labeling
