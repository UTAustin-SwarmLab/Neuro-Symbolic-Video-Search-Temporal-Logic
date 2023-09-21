from __future__ import annotations

import abc
import random

from ns_vfs.data.frame import BenchmarkLTLFrame
from ns_vfs.loader.benchmark_image import BenchmarkImageLoader


class DataGenerator(abc.ABC):
    """Data generator."""

    @abc.abstractmethod
    def generate(self) -> any:
        """Generate data."""


class BenchmarkVideoGenerator(DataGenerator):
    """Benchmark video generator."""

    def __init__(
        self,
        image_data_loader: BenchmarkImageLoader,
    ):
        """Benchmark video generator.

        Args:
            image_data_loader (BenchmarkImageLoader): Image data loader.
        """
        self._data_loader = image_data_loader
        self.data = self._data_loader.data

        print("d")

    def sample_proposition(self, class_list: list[str]) -> str:
        """Sample proposition from class list."""
        return random.sample(class_list, 2)

    def generate(
        self,
        max_number_frame: int = 200,
        number_video_per_set_of_frame: int = 3,
        increase_rate: int = 50,
    ) -> any:
        """Generate data."""
        number_frame = 25
        while number_frame <= max_number_frame:
            for video_idx in range(number_video_per_set_of_frame):
                proposition = self.sample_proposition(class_list=self.data.unique_labels)
                ltl_frame = self.ltl_function(
                    temporal_property="F",
                    proposition_1=proposition[0],
                    number_of_frame=number_frame,
                )
                (
                    ltl_frame.labels_of_frames,
                    ltl_frame.images_of_frames,
                ) = self.data.sample_image_from_label(
                    labels=ltl_frame.labels_of_frames, proposition=ltl_frame.proposition
                )

            number_frame += increase_rate

    def ltl_function(
        self,
        proposition_1: str,
        temporal_property: str,
        number_of_frame: int,
        proposition_2: str | None = None,
        conditional_property: str = "",
    ) -> BenchmarkLTLFrame:
        """LTL function.

        Args:
            proposition_1 (str): Proposition 1.
            temporal_property (str): Temporal property.
                - F: eventuallyProperty
                - G: alwaysProperty
                - U: untilProperty
            number_of_frame (int): Number of frame.
            proposition_2 (str, optional): Proposition 2. Defaults to None.
            conditional_property (str): Conditional property.
                - &: andProperty
                - |: orProperty
                - !: notProperty.
        """
        labels_of_frame = [None] * number_of_frame

        # Single Rule
        # E.G: F "cat" or G "dog"
        if proposition_2 is not None:
            ltl_formula = f'{temporal_property} "{proposition_1}" {conditional_property} "{proposition_2}"'
        else:
            ltl_formula = f'{temporal_property} "{proposition_1}"'

        # F "prop1"
        frames_of_interest = []
        for _ in range(0, int(number_of_frame / 5)):
            frame_index = random.randint(0, number_of_frame - 1)
            labels_of_frame[frame_index] = proposition_1
            frames_of_interest.append([frame_index])
        # TODO: Make a false case
        ltl_frame = BenchmarkLTLFrame(
            ground_truth=True,
            ltl_formula=ltl_formula,
            proposition=[proposition_1, proposition_2],
            number_of_frame=number_of_frame,
            frames_of_interest=frames_of_interest,
            labels_of_frames=labels_of_frame,
        )

        return ltl_frame
