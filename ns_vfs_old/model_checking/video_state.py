class VideoState:
    """Video state class."""

    def __init__(
        self,
        state_index: int,
        frame_index: int,
        label: str,
        proposition_set: list[str],
        probability: float = 1.0,
    ) -> None:
        """State class.

        Args:
            state_index (int): state_index.
            frame_index (int): Frame index.
            label (str): Label set. :abel is a string with characters T or F
                indicating True or False
            proposition_set (list[str]): Proposition set.
            probability (float): Probability of the state.
        """
        self.state_index = state_index
        self.frame_index = frame_index
        self.proposition_set = proposition_set
        self.label = label  # "init", "terminal", TTT, TFT, FTT, etc.
        self.descriptive_label = self._get_descriptive_label(label=label)
        self.probability = probability

    def __repr__(self) -> str:
        """Representation of state."""
        return f"{self.state_index} {self.descriptive_label} {self.frame_index} {self.probability}"  # noqa: E501

    def __str__(self) -> str:
        """String of state."""
        return f"{self.__repr__()}"

    def _get_descriptive_label(self, label: str) -> list:
        """Get descriptive label.

        Args:
        label (str): Label.
        """
        labels = []
        if label == "init":
            labels.append("init")
        elif label == "terminal":
            labels.append("terminal")
        else:
            for i in range(len(self.proposition_set)):
                if label[i] == "T":
                    labels.append(self.proposition_set[i])
        return labels

    def update(self, frame_index: int, target_label: str) -> None:
        """Update state to the new state..

        Args:
            frame_index (int): Frame index.
            target_label (str): Target label for the new state.
        """
        self.frame_index = frame_index
        self.label = target_label  # TTT, TFT, FTT, etc.
        self.descriptive_label = self._get_descriptive_label(label=target_label)
        self.probability = 1.0

    def compute_probability(self, probabilities: list[list[float]]) -> None:
        """Compute probability of the state given the probabilities of the propositions.

        Args:
            probabilities (list): list of probabilities of the propositions
                e.g. two propositions with three frames
                -> [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]].
        """  # noqa: E501
        probability = 1.0
        for i in range(len(self.label)):
            if self.label[i] == "T":
                probability *= probabilities[i][self.frame_index]
            else:
                probability *= 1 - probabilities[i][self.frame_index]
        self.probability = round(probability, 3)
