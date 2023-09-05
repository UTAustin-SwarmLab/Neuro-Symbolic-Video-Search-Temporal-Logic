class State:
    # label is a string with characters T or F indicating True or False
    def __init__(
        self, index, frame_index, proposition_status_set, proposition_set
    ):
        self.state_index = index
        self.frame_index = frame_index
        self.proposition_set = proposition_set
        self.current_proposition_combination = (
            proposition_status_set  # "initial", TTT, TFT, FTT, etc.
        )
        self.current_descriptive_label = self._get_descriptive_label(
            label=proposition_status_set
        )
        self.probability = 1

    def __repr__(self):
        return f"{self.state_index} {self.current_descriptive_label} {self.frame_index} {self.probability}"

    def __str__(self):
        return f"{self.__repr__()}"

    def _get_descriptive_label(self, label):
        labels = []
        for i in range(len(self.proposition_set)):
            if label[i] == "T":
                labels.append(self.proposition_set[i])
        return labels

    def update(self, frame_index, proposition_combinations):
        self.frame_index = frame_index
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
                probability *= probabilities[i][self.frame_index]
            else:
                probability *= 1 - probabilities[i][self.frame_index]
        self.probability = round(probability, 2)
