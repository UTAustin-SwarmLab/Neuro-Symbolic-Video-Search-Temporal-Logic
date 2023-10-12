from __future__ import annotations

import abc
import random
import re
from pathlib import Path

from ns_vfs.data.frame import BenchmarkLTLFrame
from ns_vfs.loader.benchmark_cifar import BenchmarkImageLoader


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
        artificat_dir: str,
    ):
        """Benchmark video generator.

        Args:
            image_data_loader (BenchmarkImageLoader): Image data loader.
        """
        self._data_loader = image_data_loader
        self.data = self._data_loader.data
        self._artificat_dir = Path(artificat_dir)
        self._artificat_dir.mkdir(parents=True, exist_ok=True)

    def extract_properties(self, s):
        # Extract properties, the U operator, and the F operator
        properties = re.findall(r'"([^"]*)"', s)
        operators = re.findall(r"\bF\b|\bU\b|\bG\b\b!\b", s)

        result = []
        for i, prop in enumerate(properties):
            if i < len(operators):
                result.append(operators[i])
            result.append(prop)
        return "".join(result)

    def sample_proposition(
        self,
        class_list: list[str],
        is_prop_2: bool = False,
        is_prop_3: bool = False,
        is_and_conditional_op=False,
    ) -> str:
        """Sample proposition from class list."""
        if is_and_conditional_op:
            assert is_prop_2 is True
            while True:
                sample = random.sample(self.data.labels, 1)[0]
                if len(sample) > 1:
                    if not is_prop_3:
                        return random.sample(sample, 2) + [None]
                    else:
                        prop_1_and_2 = random.sample(sample, 2)
                        prop_3 = [random.sample(class_list, 2)[0]]
                        if prop_3[0] not in prop_1_and_2:
                            return prop_1_and_2 + prop_3
                        else:
                            continue
        if is_prop_2 and is_prop_3 is False:
            return random.sample(class_list, 2) + [None]
        elif is_prop_2 is False and is_prop_3:
            return random.sample(class_list, 3)
        else:
            return [random.sample(class_list, 2)[0], None, None]

    def generate(
        self,
        initial_number_of_frame=25,
        max_number_frame: int = 200,
        number_video_per_set_of_frame: int = 3,
        increase_rate: int = 25,
        ltl_logic: str = "F prop1",
        temporal_property: str = "",
        conditional_property: str = "",
        save_frames: bool = False,
    ) -> any:
        """Generate data."""
        number_frame = initial_number_of_frame
        is_prop_2 = False
        is_prop_3 = False
        is_and_conditional_op = False
        for logic_component in ltl_logic.split(" "):
            if logic_component in ["F", "G", "U"]:
                temporal_property = logic_component
            elif logic_component in ["prop2", "prop2)"]:
                is_prop_2 = True
            elif logic_component == "prop3":
                is_prop_3 = True
            elif logic_component in ["!", "&", "|"]:
                conditional_property = logic_component
                if conditional_property == "&":
                    is_and_conditional_op = True

        while number_frame < max_number_frame + 1:
            for video_idx in range(number_video_per_set_of_frame):
                proposition = self.sample_proposition(
                    class_list=self.data.unique_labels,
                    is_prop_2=is_prop_2,
                    is_prop_3=is_prop_3,
                    is_and_conditional_op=is_and_conditional_op,
                )
                ltl_frame = self.ltl_function(
                    logic_component=ltl_logic.split(" "),
                    temporal_property=temporal_property,
                    proposition_1=proposition[0],
                    proposition_2=proposition[1],
                    proposition_3=proposition[2],
                    conditional_property=conditional_property,
                    number_of_frame=number_frame,
                )
                (
                    ltl_frame.labels_of_frames,
                    ltl_frame.images_of_frames,
                ) = self.data.sample_image_from_label(
                    labels=ltl_frame.labels_of_frames, proposition=ltl_frame.proposition
                )
                self.extract_properties(ltl_frame.ltl_formula)
                ltl_frame.save(
                    save_path=self._artificat_dir
                    / f"benchmark_{self._data_loader.name}_ltl_{ltl_frame.ltl_formula}_{number_frame}_{video_idx}.pkl"
                )

            number_frame += increase_rate
        if save_frames:
            ltl_frame.save_frames()

    def ltl_function(
        self,
        logic_component: list[str],
        proposition_1: str,
        temporal_property: str,
        number_of_frame: int,
        proposition_2: str | None = None,
        proposition_3: str | None = None,
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
        temp_frames_of_interest = []
        proposition_set = []
        random_frame_idx_selection = sorted(
            list(set([random.randint(0, number_of_frame - 3) for _ in range(int(number_of_frame / 5))]))
        )  # number_of_frame - 3 to avoid random.range(same_idx,same_idx)
        # Single Rule
        if proposition_2 is not None:
            if proposition_3 is not None:
                proposition_set = [proposition_1, proposition_2, proposition_3]
            else:
                proposition_set = [proposition_1, proposition_2]
            if temporal_property == "U":
                assert proposition_2 is not None, "proposition 2 must be not None"
                u_index = logic_component.index("U")
                pre_u_index = logic_component[u_index - 1]
                post_u_index = logic_component[u_index + 1]
                if post_u_index == "prop2":
                    # TODO: F & G...
                    ltl_formula = f'"{proposition_1}" {temporal_property} "{proposition_2}"'
                    post_u_label_idx = []
                    for idx in list(set(random_frame_idx_selection)):
                        temp_frames_of_interest.append(idx)
                        labels_of_frame[idx] = proposition_1
                        if len(post_u_label_idx) > 0:
                            if idx >= post_u_label_idx[-1]:
                                prop2_idx = random.randrange(idx + 1, number_of_frame - 1)
                                temp_frames_of_interest.append(prop2_idx)
                                post_u_label_idx.append(prop2_idx)
                                labels_of_frame[prop2_idx] = proposition_2
                        else:
                            prop2_idx = random.randrange(idx + 1, number_of_frame - 1)
                            temp_frames_of_interest.append(prop2_idx)
                            post_u_label_idx.append(prop2_idx)
                            labels_of_frame[prop2_idx] = proposition_2

                    temp_frames_of_interest = list(set(temp_frames_of_interest))
                    for i, temp_label in enumerate(temp_frames_of_interest):
                        if temp_label not in post_u_label_idx:
                            if temp_label > max(post_u_label_idx):
                                labels_of_frame[temp_label] = None
                                temp_frames_of_interest.pop(temp_frames_of_interest.index(temp_label))

                    if proposition_2 not in labels_of_frame:
                        labels_of_frame[-1] = proposition_2
                        temp_frames_of_interest.append(len(labels_of_frame) - 1)
                    temp_frames_of_interest = list(set(temp_frames_of_interest))
                    # # select multiple random prop1
                    # for i in range(0, int(number_of_frame / 10)):
                    #     range_of_frame = int(number_of_frame / int(number_of_frame / 10))

                    #     num_prop1 = random.randrange(1, int(range_of_frame / 4))

                    #     prop1_idx = []
                    #     for _ in range(num_prop1):
                    #         random_prop1 = random.randrange(i * range_of_frame, (i + 1) * range_of_frame - 1)
                    #         if random_prop1 not in prop1_idx:
                    #             prop1_idx.append(random_prop1)

                    #     curr_frames_interest = list()
                    #     for p1_idx in prop1_idx:
                    #         labels_of_frame[p1_idx] = proposition_1
                    #         curr_frames_interest.append(p1_idx)
                    #     try:
                    #         prop2_idx = random.randrange(
                    #             int(max(prop1_idx) + 1), (i + 1) * range_of_frame - 1
                    #         )
                    #     except:
                    #         prop2_idx = (i + 1) * range_of_frame - 1

                    #     labels_of_frame[prop2_idx] = proposition_2
                    #     curr_frames_interest.sort()
                    #     curr_frames_interest.append(prop2_idx)
                    #     temp_frames_of_interest.append(curr_frames_interest)
                elif pre_u_index == "prop2" and post_u_index == "prop3":
                    # TODO
                    pass
                elif pre_u_index == "prop2)" and post_u_index == "prop3":
                    ltl_formula = f'("{proposition_1}" {logic_component[logic_component.index(pre_u_index)-1]} "{proposition_2}") U "{proposition_3}"'
                    post_u_label_idx = []
                    for idx in list(set(random_frame_idx_selection)):
                        temp_frames_of_interest.append(idx)
                        labels_of_frame[idx] = [proposition_1, proposition_2]
                        if len(post_u_label_idx) > 0:
                            if idx >= post_u_label_idx[-1]:
                                prop3_idx = random.randrange(idx + 1, number_of_frame - 1)
                                temp_frames_of_interest.append(prop3_idx)
                                post_u_label_idx.append(prop3_idx)
                                labels_of_frame[prop3_idx] = proposition_3
                        else:
                            prop3_idx = random.randrange(idx + 1, number_of_frame - 1)
                            temp_frames_of_interest.append(prop3_idx)
                            post_u_label_idx.append(prop3_idx)
                            labels_of_frame[prop3_idx] = proposition_3

                    temp_frames_of_interest = list(set(temp_frames_of_interest))
                    for i, temp_label in enumerate(temp_frames_of_interest):
                        if temp_label not in post_u_label_idx:
                            if temp_label > max(post_u_label_idx):
                                labels_of_frame[temp_label] = None
                                temp_frames_of_interest.pop(temp_frames_of_interest.index(temp_label))

                    if proposition_3 not in labels_of_frame:
                        labels_of_frame[-1] = proposition_3
                        temp_frames_of_interest.append(len(labels_of_frame) - 1)
                    temp_frames_of_interest = list(set(temp_frames_of_interest))

            else:
                assert conditional_property is not None, "conditional_property must be not None"
                ltl_formula = (
                    f'{temporal_property} "{proposition_1}" {conditional_property} "{proposition_2}"'
                )

                for idx in list(set(random_frame_idx_selection)):
                    temp_frames_of_interest.append([idx])
                    labels_of_frame[idx] = proposition_set

        else:
            proposition_set.append(proposition_1)
            if conditional_property == "":
                ltl_formula = f'{temporal_property} "{proposition_1}"'
            else:
                assert conditional_property == "!", "conditional_property must be ! with one proposition"
                ltl_formula = f'{temporal_property} {conditional_property} "{proposition_1}"'
            # 1. F "prop1"
            frame_index = []
            if temporal_property == "F":
                frame_index = [
                    random.randint(0, number_of_frame - 1) for _ in range(int(number_of_frame / 5))
                ]
                for idx in list(set(frame_index)):
                    temp_frames_of_interest.append([idx])
                    labels_of_frame[idx] = proposition_1
            elif temporal_property == "G":
                frame_index = [
                    random.randint(0, number_of_frame - 1) for _ in range(int(number_of_frame / 10))
                ]
                for idx in list(set(frame_index)):
                    min_idx = idx - int(number_of_frame / 10)
                    if min_idx < 0:
                        min_idx = 0
                    if min_idx == idx:
                        idx = min_idx + 1
                    for sub_idx in list(range(min_idx, idx)):
                        if sub_idx not in temp_frames_of_interest:
                            temp_frames_of_interest.append(sub_idx)
                            labels_of_frame[sub_idx] = proposition_1
                if conditional_property == "!":
                    temp_frames_of_interest = [
                        x for x in range(len(labels_of_frame)) if x not in temp_frames_of_interest
                    ]

            # 2. G "prop1"

        # TODO: Make a false case
        return BenchmarkLTLFrame(
            ground_truth=True,
            ltl_formula=ltl_formula,
            proposition=proposition_set,
            number_of_frame=number_of_frame,
            frames_of_interest=temp_frames_of_interest,
            labels_of_frames=labels_of_frame,
        )
