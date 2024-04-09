from __future__ import annotations

import re

from sympy import satisfiable, symbols
from sympy.parsing.sympy_parser import parse_expr

from ns_vfs.automaton._base import Automaton
from ns_vfs.automaton.state import State
from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.data.frame import Frame
from ns_vfs.enums.status import Status

Status.UNKNOWN
"""
Discrete Real Time (DRT) Markov Chain (DRTMC).
"""


class DRTMarkovChain(Automaton):
    def __init__(
        self,
        tl_specification: str,
        proposition_set: list[str] = None,
    ) -> None:
        """Discrete Real Time Markov Chain (DRTMC) Automaton.
        Args:
            tl_specification (str): Temporal logic specification.
            proposition_set (list[str], optional): List of propositions. Defaults to None.
        """
        self.tl_specification = tl_specification
        self.proposition_set = proposition_set
        self.safe_state = None
        self.unsafe_state = None
        self.safety_threshold = self._get_safety_threshold(tl_specification)
        # variables below are set in set_up method
        self.satisfiable_propositions = dict()
        self.previous_states = list()
        self.states = list()
        self.transitions = list()
        self.frame_index_in_automaton = 0

    def _get_safety_threshold(self, tl_specification: str) -> float:
        """Get the safety threshold from the temporal logic specification.

        Args:
            tl_specification (str): Temporal logic specification.

        Returns:
            float: Safety threshold.
        """
        match = re.search(r"P>=(\d+\.\d+)", tl_specification)
        if not match:
            raise ValueError("Temporal logic specification is not valid.")
        safety_threshold = float(match.group(1))
        return safety_threshold

    def set_up(self, proposition_set: list[str]):
        self.proposition_combinations = self._create_proposition_combinations(
            len(proposition_set)
        )
        self.satisfiable_propositions = self.get_satisfiable_propositions(
            tl_specification=self.tl_specification
        )
        self.proposition_set = proposition_set
        self.probability_of_propositions = [
            [] for _ in range(len(proposition_set))
        ]

    def reset(self) -> None:
        """Reset automaton with initial safe/unsafe states."""
        self.safe_state = State(
            state_index=0,
            frame_index_in_automaton="init",
            proposition_status_set="init",
            proposition_set=["safe"],
            probability=self.safe_state.probability,
        )
        self.unsafe_state = State(
            state_index=1,
            frame_index_in_automaton="init",
            proposition_status_set="init",
            proposition_set=["unsafe"],
            probability=self.unsafe_state.probability,
        )
        self.previous_states = [self.safe_state, self.unsafe_state]
        self.states = [self.safe_state, self.unsafe_state]
        self.transitions = list()
        self.probability_of_propositions = [
            [] for _ in range(len(self.proposition_set))
        ]
        self.frame_index_in_automaton = 0

    def get_satisfiable_propositions(
        self, tl_specification: str
    ) -> list[dict[str, bool]]:
        """Extract and evaluate first order logic from a given specification to find all satisfiable propositions.

        Args:
            tl_specification (str): Temporal logic specification including the first order logic part.

        Returns:
            A list of dictionaries, each representing a satisfiable assignment of truth values to propositions.
        """
        # Extract the first order logic part from the specification
        match = re.search(r"\[(.*?)\]", tl_specification)
        if not match:
            raise ValueError("Temporal logic specification is not valid.")
        first_order_logic = match.group(1)

        # Identify all unique propositions within the first order logic
        prop_names = set(re.findall(r"\b(\w+)\b", first_order_logic))
        # Dynamically create symbols for each proposition
        symbols_dict = {name: symbols(name, bool=True) for name in prop_names}
        # Parse the logical expression using the created symbols
        expression = parse_expr(first_order_logic, local_dict=symbols_dict)

        # Attempt to find all satisfying models for the expression
        all_models = list(satisfiable(expression, all_models=True))

        # Convert models to a list of dictionaries with proposition names as keys
        satisfying_assignments = [
            {str(k): v for k, v in model.items()}
            for model in all_models
            if model != False
        ]

        return satisfying_assignments

    def add_frame_to_automaton(self, frame: Frame) -> None:
        """Add frame to automaton."""
        if (
            self.previous_states
            and self.previous_states[0].current_proposition_combination
            == "terminal"
        ):
            self.reset()
        self._update_probability_of_propositions(frame)
        current_states = []
        for prop_comb in self.proposition_combinations:
            # iterate through all possible combinations of T and F
            self._current_state = State(
                state_index=len(self.states),
                frame_index_in_automaton=self.frame_index_in_automaton,
                proposition_status_set=prop_comb,
                proposition_set=self.proposition_set,
            )
            # TODO: Make a method for update and compute probability
            self._current_state.update(
                frame_index_in_automaton=self.frame_index_in_automaton,
                proposition_combinations=prop_comb,
            )
            self._current_state.compute_probability(
                probabilities=self.probability_of_propositions
            )
            if self._current_state.probability > 0:
                self.states.append(self._current_state)
                current_states.append(self._current_state)

        # Build transitions from previous states to current states
        if self.previous_states:
            for prev_state in self.previous_states:
                for cur_state in current_states:
                    transition = (
                        prev_state.state_index,
                        cur_state.state_index,
                        cur_state.probability,
                    )
                    self.transitions.append(transition)
        else:
            # No init state, so add transitions from current states to current states
            for cur_state in current_states:
                transition = (
                    cur_state.state_index,
                    cur_state.state_index,
                    cur_state.probability,
                )
                self.transitions.append(transition)

        self.previous_states = (
            current_states if current_states else self.previous_states
        )
        self.frame_index_in_automaton += 1
        self.add_safety_states()

    def _update_probability_of_propositions(self, frame: Frame):
        """Update the probability of propositions."""
        for i, prop in enumerate(self.proposition_set):
            self.probability_of_propositions[i].append(
                frame.object_of_interest[prop].probability
            )

    def _create_proposition_combinations(self, num_props: int):
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

    def add_safety_states(self):
        # Check if the automaton satisfies the specification.
        probability_of_safety = 0.0
        safe_state_indices = []
        safe_state = State(
            state_index=len(self.states),
            frame_index_in_automaton=self.frame_index_in_automaton,
            proposition_status_set="terminal",
            proposition_set=["safe"],
        )
        unsafe_state = State(
            state_index=len(self.states) + 1,
            frame_index_in_automaton=self.frame_index_in_automaton,
            proposition_status_set="terminal",
            proposition_set=["unsafe"],
        )
        # Check if there exist safe / unsafe states
        if self.safe_state:
            prev_safe_probability = self.safe_state.probability
            prev_unsafe_probability = self.unsafe_state.probability
            for i, transitions in enumerate(self.transitions):
                if transitions[0] == 0:
                    new_transition_probability = (
                        transitions[2] * prev_safe_probability
                    )

                elif transitions[0] == 1:
                    new_transition_probability = (
                        transitions[2] * prev_safe_probability
                    )
                transitions = (
                    transitions[0],
                    transitions[1],
                    Status.UNKNOWN,  # round(new_transition_probability, 4),
                )
                self.transitions[i] = transitions
        else:
            prev_safe_probability = 1
            prev_unsafe_probability = 1

        for satisfiable_proposition in self.satisfiable_propositions:
            sorted_values = [
                satisfiable_proposition[prop] for prop in self.proposition_set
            ]
            boolean_combination_str = "".join(
                ["T" if value else "F" for value in sorted_values]
            )
            for state in self.states:
                if (
                    state.current_proposition_combination
                    == boolean_combination_str
                ):
                    new_safe_probability = (
                        state.probability * prev_safe_probability
                    )
                    probability_of_safety += new_safe_probability
                    safe_state_indices.append(state.state_index)
                    self.transitions.append(
                        (
                            state.state_index,
                            safe_state.state_index,
                            Status.UNKNOWN,  # round(new_safe_probability, 4),
                        )
                    )

        # Add transitions from unsafe state to all states
        for state in self.states:
            if (
                state.state_index not in safe_state_indices
                and state.current_proposition_combination != "init"
            ):
                new_unsafe_probability = (
                    state.probability * prev_unsafe_probability
                )
                self.transitions.append(
                    (
                        state.state_index,
                        unsafe_state.state_index,
                        Status.UNKNOWN,  # round(new_unsafe_probability, 4),
                    )
                )

        # update probability of safe and unsafe states
        safe_state.probability = round(probability_of_safety, 4)
        unsafe_state.probability = round(1 - probability_of_safety, 4)

        self.states.append(safe_state)
        self.states.append(unsafe_state)
        self.previous_states = [safe_state, unsafe_state]
        self.safe_state = safe_state
        self.unsafe_state = unsafe_state

    def check_automaton(self):
        if self.probability_of_safety > self.safety_threshold:
            return True
        else:
            return False

    @property
    def probability_of_safety(self):
        return self.safe_state.probability

    @property
    def probability_of_unsafety(self):
        return self.unsafe_state.probability


if __name__ == "__main__":
    import numpy as np

    # Set Up
    first_order_logic = "prop1 | ~prop2"
    tl_specification = f"P>=0.80 G [{first_order_logic}]"
    proposition_set = ["prop1", "prop2"]
    automaton = DRTMarkovChain(
        tl_specification=tl_specification, proposition_set=proposition_set
    )
    automaton.set_up(["prop1", "prop2"])

    # Frame 1
    detected_objects = {
        "prop1": DetectedObject(name="prop1", probability=0.8, confidence=0.8),
        "prop2": DetectedObject(name="prop2", probability=0.7, confidence=0.7),
    }
    activity_of_interest = None
    frame = Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.random.rand(100, 100, 3),
        object_of_interest=detected_objects,
        activity_of_interest=activity_of_interest,
    )
    automaton.add_frame_to_automaton(frame=frame)

    # Frame 2
    detected_objects = {
        "prop1": DetectedObject(name="prop1", probability=0.8, confidence=0.8),
        "prop2": DetectedObject(name="prop2", probability=0.7, confidence=0.7),
    }
    activity_of_interest = None
    frame = Frame(
        frame_idx=0,
        timestamp=0,
        frame_image=np.random.rand(100, 100, 3),
        object_of_interest=detected_objects,
        activity_of_interest=activity_of_interest,
    )
    automaton.add_frame_to_automaton(frame=frame)

    breakpoint()
    automaton.add_frame_to_automaton(frame=frame)
    breakpoint()
