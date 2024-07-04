from __future__ import annotations

from ns_vfs.automaton._base import Automaton
from ns_vfs.automaton.state import State
from ns_vfs.data.frame import Frame


class ProbabilisticAutomaton(Automaton):
    def __init__(
        self,
        include_initial_state: bool = False,
        proposition_set: list[str] = None,
    ) -> None:
        self.proposition_set = proposition_set
        self.previous_states = list()
        self.states = list()
        self.transitions = list()
        self.frame_index_in_automaton = 0
        self.include_initial_state = include_initial_state

    def set_up(self, proposition_set: list[str]):
        self.frame_index_in_automaton = 0
        self.proposition_combinations = self._create_proposition_combinations(
            len(proposition_set)
        )
        self.proposition_set = proposition_set
        self.probability_of_propositions = [
            [] for _ in range(len(proposition_set))
        ]
        if self.include_initial_state:
            self._current_state = State(
                state_index=0,
                frame_index_in_automaton=-1,
                proposition_status_set="init",
                proposition_set=proposition_set,
            )

            self.previous_states.append(self._current_state)
            self.states.append(self._current_state)

    def reset(self):
        """Reset automaton."""
        self.previous_states = list()
        self.states = list()
        self.transitions = list()
        self.probability_of_propositions = [
            [] for _ in range(len(self.proposition_set))
        ]
        self.frame_index_in_automaton = 0

        if self.include_initial_state:
            self._current_state = State(
                state_index=0,
                frame_index_in_automaton=-1,
                proposition_status_set="init",
                proposition_set=self.proposition_set,
            )

            self.previous_states.append(self._current_state)
            self.states.append(self._current_state)

    def add_frame_to_automaton(self, frame: Frame):
        """Add frame to automaton."""
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
