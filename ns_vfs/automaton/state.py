class State:
    # label is a string with characters T or F indicating True or False
    def __init__(
        self,
        state_index,
        frame_index_in_automaton,
        proposition_status_set,
        proposition_set,
    ) -> None:
        """State class.

        Args:
            state_index (int): state_index.
            frame_index_in_automaton (int): Frame index.
            proposition_status_set (str): Proposition status set.
            proposition_set (list[str]): Proposition set.
        """
        self.state_index = state_index
        self.frame_index_in_automaton = frame_index_in_automaton
        self.proposition_set = proposition_set
        self.current_proposition_combination = (
            proposition_status_set  # "initial", TTT, TFT, FTT, etc.
        )
        self.current_descriptive_label = self._get_descriptive_label(
            label=proposition_status_set
        )
        self.probability = 1

    def __repr__(self):
        """Representation of state."""
        return f"{self.state_index} {self.current_descriptive_label} {self.frame_index_in_automaton} {self.probability}"

    def __str__(self):
        """String of state."""
        return f"{self.__repr__()}"

    def _get_descriptive_label(self, label):
        """Get descriptive label.

        Args:
        label (str): Label.
        """
        labels = []
        for i in range(len(self.proposition_set)):
            if label[i] == "T":
                labels.append(self.proposition_set[i])
        if self.state_index == 0 and len(labels) == 0:
            labels.append("init")
        return labels

    def update(self, frame_index_in_automaton, proposition_combinations):
        """Update state.

        Args:
            frame_index_in_automaton (int): Frame index.
            proposition_combinations (str): Proposition combinations.
        """
        self.frame_index_in_automaton = frame_index_in_automaton
        self.current_proposition_combination = (
            proposition_combinations  # TTT, TFT, FTT, etc.
        )
        self.current_descriptive_label = self._get_descriptive_label(
            label=proposition_combinations
        )
        self.probability = 1

    def compute_probability(self, probabilities):
        """Compute probability of the state given the probabilities of the propositions.

        Args:
            probabilities (list): list of probabilities of the propositions
                e.g. two propositions with three frames
                -> [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]].
        """
        probability = 1
        for i in range(len(self.current_proposition_combination)):
            if self.current_proposition_combination[i] == "T":
                probability *= probabilities[i][self.frame_index_in_automaton]
            else:
                probability *= 1 - probabilities[i][self.frame_index_in_automaton]
        self.probability = round(probability, 2)
