import logging
import math

import numpy as np
import stormpy
import stormpy.examples.files
from stormpy.core import ExplicitQualitativeCheckResult

from neus_v.model_checking.proposition import process_proposition_set
from neus_v.model_checking.video_state import VideoState


class StormModelChecker:
    """Model Checker using Stormpy for verifying properties."""

    def __init__(
        self,
        proposition_set: list[str],
        ltl_formula: str,
    ) -> None:
        """Initialize the StormModelChecker.

        Args:
            proposition_set: List of propositions.
            ltl_formula: LTL formula to check.
            verbose: Enable verbose output.
            is_filter: Apply filtering to results.
        """
        self.proposition_set = process_proposition_set(proposition_set)
        self.ltl_formula = ltl_formula

    def create_model(
        self,
        transitions: list[tuple[int, int, float]],
        states: list[VideoState],
        model_type: str = "sparse_ma",
    ) -> any:
        """Create model.

        Args:
            transitions (list[tuple[int, int, float]]): List of transitions.
            states (list[VideoState]): List of states.
            model_type (str): Type of model to create ("sparse_ma" or "dtmc").
            verbose (bool): Whether to print verbose output.
        """
        state_labeling = self._build_label_func(states, self.proposition_set)
        if model_type in ["sparse_ma", "mdp"]:
            transition_matrix = self._build_trans_matrix(
                transitions=transitions,
                states=states,
                model_type="nondeterministic",
            )
        else:
            transition_matrix = self._build_trans_matrix(
                transitions=transitions,
                states=states,
                model_type="deterministic",
            )
        components = stormpy.SparseModelComponents(
            transition_matrix=transition_matrix,
            state_labeling=state_labeling,
        )
        if model_type == "sparse_ma":
            markovian_states = stormpy.BitVector(len(states), list(range(len(states))))
            components.markovian_states = markovian_states
            components.exit_rates = [1.0 for _ in range(len(states))]
            model = stormpy.SparseMA(components)
        elif model_type == "dtmc":
            model = stormpy.storage.SparseDtmc(components)
        elif model_type == "mdp":
            model = stormpy.storage.SparseMdp(components)
        else:
            msg = f"Unsupported model type: {model_type}"
            raise ValueError(msg)
        return model

    def check_automaton(
        self,
        transitions: list[tuple[int, int, float]],
        states: list[VideoState],
        model_type: str = "sparse_ma",
        use_filter: bool = False,
    ) -> any:
        """Check automaton.

        Args:
            transitions: List of transitions.
            states: List of states.
            verbose: Enable verbose output.
            use_filter: Apply filtering to results.
        """
        model = self.create_model(
            transitions=transitions,
            states=states,
            model_type=model_type,
        )
        # Check the model
        # Initialize Prism Program
        path = stormpy.examples.files.prism_dtmc_die  #  prism_mdp_maze
        prism_program = stormpy.parse_prism_program(path)

        # Define Properties
        properties = stormpy.parse_properties(self.ltl_formula, prism_program)

        # Get Result and Filter it
        result = stormpy.model_checking(model, properties[0])

        if use_filter:
            # The final result will only consider paths starting from the initial states of the automaton.  # noqa: E501
            filtered_result = stormpy.create_filter_initial_states_sparse(model)
            result.filter(filtered_result)
        return result

    def qualitative_result_eval(self, verification_result: ExplicitQualitativeCheckResult) -> bool:
        if isinstance(verification_result, ExplicitQualitativeCheckResult):
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
        msg = "Model Checking is not qualitative"
        raise ValueError(msg)

    def _build_trans_matrix(
        self,
        transitions: list[tuple[int, int, float]],
        states: list[VideoState],
        model_type: str = "nondeterministic",
    ) -> stormpy.storage.SparseMatrix:
        """Build transition matrix.

        Args:
            transitions: List of transitions.
            states: List of states.
            model_type: Type of model ("nondeterministic" or "deterministic").
        """
        if model_type not in ["nondeterministic", "deterministic"]:
            msg = "Invalid model_type. Must be 'nondeterministic' or 'deterministic'"  # noqa: E501
            raise ValueError(msg)

        if model_type == "nondeterministic":
            matrix = np.zeros((len(states), len(states)))
            for t in transitions:
                matrix[int(t[0]), int(t[1])] = float(t[2])
            trans_matrix = stormpy.build_sparse_matrix(matrix, list(range(len(states))))

        elif model_type == "deterministic":
            num_states = len(states)
            builder = stormpy.SparseMatrixBuilder(
                rows=num_states,
                columns=num_states,
                entries=len(transitions),
                force_dimensions=False,
            )
            states_with_transitions = set(src for src, _, _ in transitions)
            outgoing_probs = {i: 0.0 for i in range(num_states)}

            for src, dest, prob in transitions:
                builder.add_next_value(src, dest, prob)
                outgoing_probs[src] += prob

            for state in range(num_states):
                if state not in states_with_transitions:
                    builder.add_next_value(state, state, 1.0)
                    outgoing_probs[state] = 1.0

            # Check probabilities
            for state, prob_sum in outgoing_probs.items():
                # if not math.isclose(prob_sum, 1.0, rel_tol=1e-9):
                if not math.isclose(prob_sum, 1.0, abs_tol=1e-2):
                    logging.warning(f"State {state} has outgoing probability sum of {prob_sum}, not 1.0")

            # ... (existing logging code) ...
            trans_matrix = builder.build()
        return trans_matrix

    def _build_label_func(
        self,
        states: list[VideoState],
        props: list[str],
        model_type: str = "nondeterministic",
    ) -> stormpy.storage.StateLabeling:
        """Build label function.

        Args:
            states (list[State]): List of states.
            props (list[str]): List of propositions.
            model_type (str): Type of model
                ("nondeterministic" or "deterministic").

        Returns:
            stormpy.storage.StateLabeling: State labeling.
        """
        state_labeling = stormpy.storage.StateLabeling(len(states))
        state_labeling.add_label("init")
        state_labeling.add_label("terminal")
        for label in props:
            state_labeling.add_label(label)

        if model_type == "nondeterministic":
            for state in states:
                for label in state.descriptive_label:
                    state_labeling.add_label_to_state(label, state.state_index)
        else:
            for i, state in enumerate(states):
                for prop in state.props:
                    if prop in props:
                        state_labeling.add_label_to_state(prop, i)
        return state_labeling

    def validate_tl_specification(self, ltl_formula: str) -> bool:
        """Validate LTL specification.

        Args:
            ltl_formula: LTL formula to validate.
        """
        path = stormpy.examples.files.prism_dtmc_die  #  prism_mdp_maze
        prism_program = stormpy.parse_prism_program(path)
        # Define Properties
        try:
            stormpy.parse_properties(ltl_formula, prism_program)
        except Exception as e:
            msg = f"Error validating LTL specification: {e}"
            logging.exception(msg)
            return False
        else:
            return True
