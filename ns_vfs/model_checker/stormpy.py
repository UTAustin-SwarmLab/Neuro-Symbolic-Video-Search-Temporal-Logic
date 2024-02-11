from __future__ import annotations

import numpy as np
import stormpy
import stormpy.examples.files
from stormpy.core import ExplicitQualitativeCheckResult

from ns_vfs.automaton.state import State


class StormModelChecker:
    def __init__(
        self,
        proposition_set: list[str],
        ltl_formula: str,
        verbose: bool = False,
        is_filter: bool = False,
    ) -> None:
        self.proposition_set = proposition_set
        self.ltl_formula = ltl_formula
        self.verbose = verbose
        self.is_filter = is_filter

    def check_automaton(
        self,
        transitions: list[tuple[int, int, float]],
        states: list[State],
        verbose: bool = False,
        is_filter: bool = False,
    ) -> any:
        """Check automaton.

        Args:
            transitions (list[tuple[int, int, float]]): List of transitions.
            states (list[State]): List of states.
            proposition_set (list[str]): List of propositions.
            ltl_formula (str): LTL formula.
        """
        transition_matrix = self._build_trans_matrix(
            transitions=transitions, states=states
        )

        state_labeling = self._build_label_func(states, self.proposition_set)

        markovian_states = stormpy.BitVector(
            len(states), list(range(len(states)))
        )

        components = stormpy.SparseModelComponents(
            transition_matrix=transition_matrix,
            state_labeling=state_labeling,
            markovian_states=markovian_states,
        )
        components.exit_rates = [0.0 for i in range(len(states))]
        # [0.0 if i != len(states) - 1 else 1 for i in range(len(states))]

        # Markov Automaton
        markov_automata = stormpy.storage.SparseMA(components)

        if verbose:
            print(transition_matrix)
            print(markov_automata)

        # Check the model (Markov Automata)
        result = self._model_checking(
            markov_automata, self.ltl_formula, is_filter
        )
        return self._verification_result_eval(verification_result=result)

    def _verification_result_eval(
        self, verification_result: ExplicitQualitativeCheckResult
    ):
        # string result is "true" when is absolutely true
        # but it returns "true, false" when we have some true and false
        verification_result_str = str(verification_result)
        string_result = verification_result_str.split("{")[-1].split("}")[0]
        if len(string_result) == 4:
            if string_result[0] == "t":  # 0,6
                result = True
        elif len(string_result) > 5:
            # "true, false" -> some true and some false
            result = True
        else:
            result = False
        return result

    def _model_checking(
        self,
        model: stormpy.storage.SparseMA,
        formula_str: str,
        is_filter: bool = False,
    ) -> any:
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

        if is_filter:
            filter = stormpy.create_filter_initial_states_sparse(model)
            result.filter(filter)

        return result

    def _build_trans_matrix(
        self, transitions: list[tuple[int, int, float]], states: list[State]
    ):
        """Build transition matrix.

        Args:
            transitions (list[tuple[int, int, float]]): List of transitions.
            states (list[State]): List of states.
        """
        matrix = np.zeros((len(states), len(states)))
        for t in transitions:
            matrix[int(t[0]), int(t[1])] = float(t[2])
        trans_matrix = stormpy.build_sparse_matrix(
            matrix, list(range(len(states)))
        )
        return trans_matrix

    def _build_label_func(
        self, states: list[State], props: list[str]
    ) -> stormpy.storage.StateLabeling:
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
            # if state.state_index == 0:
            #     state_labeling.add_label("init")
            #     state_labeling.add_label_to_state("init", state.state_index)
            # else:
            #     for label in state.current_descriptive_label:
            #         state_labeling.add_label_to_state(label, state.state_index)

        return state_labeling
