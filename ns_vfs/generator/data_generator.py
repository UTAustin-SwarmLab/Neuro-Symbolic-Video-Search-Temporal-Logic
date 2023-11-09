from __future__ import annotations

import abc
import copy
import random
import re
from collections import Counter
from pathlib import Path

from ns_vfs.common.utility import load_pickle_to_dict, save_dict_to_pickle
from ns_vfs.data.frame import BenchmarkLTLFrame
from ns_vfs.loader.benchmark_cifar import BenchmarkImageLoader


class DataGenerator(abc.ABC):
    """Data generator."""

    @abc.abstractmethod
    def generate(self) -> any:
        """Generate data."""


class WaymoLTLGroundTruthGenerator(DataGenerator):
    def __init__(self, benchmark_ltl_frame_dir: str, save_dir: str):
        """LTL ground truth generator.

        Args:
            image_data_loader (BenchmarkImageLoader): Image data loader.
        """
        # BenchmarkLTLFrame
        self._data_dir = Path(benchmark_ltl_frame_dir)
        self._save_dir = save_dir
        self._all_files = list(self._data_dir.glob("*.pkl"))

    def generate(self):
        """Run."""
        print(f"Total number of files: {len(self._all_files)}")
        for file in self._all_files:
            benchmark_frame: BenchmarkLTLFrame = load_pickle_to_dict(file)
            self.generate_ltl_ground_truth(benchmark_frame)

    def get_label_count(self, lst):
        output = Counter()

        for item in lst:
            if isinstance(item, list):
                # If the item is a list, sort it, convert it to a tuple, and count occurrences
                item_tuple = tuple(sorted(item))
                output[item_tuple] += 1
            else:
                # If the item is not a list, simply count occurrences
                output[item] += 1

        # Convert Counter object to dictionary (optional)
        return dict(output)

    def evaluate_unique_prop(self, unique_prop, lst):
        for item in lst:
            if isinstance(item, list):
                if unique_prop in item:
                    return True
            else:
                if unique_prop == item:
                    return True
                else:
                    return False

    def _class_map(self, prop):
        map_dict = {
            "vehicle": "car",
            "pedestrian": "person",
            "cyclist": "bicycle",
            "sign": "traffic_sign",
        }
        return map_dict[prop]

    def f_prop1(self, prop1: str, lst: list[list]):
        ltl_formula = f'F "{self._class_map(prop1)}"'
        new_prop_set = [self._class_map(prop1)]
        ltl_ground_truth = []
        tmp_ground_truth = []
        for idx, label in enumerate(lst):
            if isinstance(label, list):
                if prop1 in label:
                    tmp_ground_truth.append(idx)
                    ltl_ground_truth.append(list(set(tmp_ground_truth)))
                    tmp_ground_truth = []
        return ltl_formula, new_prop_set, ltl_ground_truth

    def prop1_u_prop2(self, prop1: str, prop2: str, lst: list[list]):
        ltl_formula = f'"{self._class_map(prop1)}" U "{self._class_map(prop2)}"'
        new_prop_set = [self._class_map(prop1), self._class_map(prop2)]
        ltl_ground_truth = []
        tmp_ground_truth = []
        for idx, label in enumerate(lst):
            if isinstance(label, list):
                if prop1 in label:
                    if len(tmp_ground_truth) == 0:
                        tmp_ground_truth.append(idx)
                    else:
                        if prop2 not in label:
                            tmp_ground_truth.append(idx)
            if isinstance(label, list):
                if prop2 in label:
                    tmp_ground_truth.append(idx)
                    ltl_ground_truth.append(list(set(tmp_ground_truth)))
                    tmp_ground_truth = []

        return ltl_formula, new_prop_set, ltl_ground_truth

    def prop1_and_prop2_u_prop3(self, prop1: str, prop2: str, prop3: str, lst: list[list]):
        ltl_formula = (
            f'("{self._class_map(prop1)}" & "{self._class_map(prop2)}") U "{self._class_map(prop3)}"'
        )
        new_prop_set = [self._class_map(prop1), self._class_map(prop2), self._class_map(prop3)]
        ltl_ground_truth = []
        tmp_ground_truth = []
        for idx, label in enumerate(lst):
            if isinstance(label, list):
                if all(elem in label for elem in [prop1, prop2]):
                    if len(tmp_ground_truth) == 0:
                        tmp_ground_truth.append(idx)
                    else:
                        if prop3 not in label:
                            tmp_ground_truth.append(idx)

            if isinstance(label, list):
                if prop3 in label:
                    tmp_ground_truth.append(idx)
                    ltl_ground_truth.append(list(set(tmp_ground_truth)))
                    tmp_ground_truth = []

                    # return here since we don't
        return ltl_formula, new_prop_set, ltl_ground_truth

    def generate_ltl_ground_truth(self, benchmark_frame: BenchmarkLTLFrame):
        ltl_formula = {}
        unique_props = ["sign", "cyclist"]
        label_counter = self.get_label_count(benchmark_frame.labels_of_frames)
        if any(unique_prop in benchmark_frame.proposition for unique_prop in unique_props):
            unique_prop = next(
                (unique_prop for unique_prop in unique_props if unique_prop in benchmark_frame.proposition),
                None,
            )
            if self.evaluate_unique_prop(unique_prop, benchmark_frame.labels_of_frames[:25]):
                ltl_formula, new_prop_set, frames_of_interest = self.f_prop1(
                    prop1=unique_prop, lst=benchmark_frame.labels_of_frames
                )
                self.update_and_save_benchmark_frame(
                    ltl_formula, new_prop_set, frames_of_interest, benchmark_frame, self._save_dir
                )
            else:
                ltl_formula, new_prop_set, frames_of_interest = self.prop1_u_prop2(
                    prop1=random.choice(["vehicle", "pedestrian"]),
                    prop2=unique_prop,
                    lst=benchmark_frame.labels_of_frames,
                )
                self.update_and_save_benchmark_frame(
                    ltl_formula, new_prop_set, frames_of_interest, benchmark_frame, self._save_dir
                )
                ltl_formula, new_prop_set, frames_of_interest = self.prop1_and_prop2_u_prop3(
                    prop1="vehicle",
                    prop2="pedestrian",
                    prop3=unique_prop,
                    lst=benchmark_frame.labels_of_frames,
                )
                self.update_and_save_benchmark_frame(
                    ltl_formula, new_prop_set, frames_of_interest, benchmark_frame, self._save_dir
                )

    def label_mapping_function(self, lst):
        new_label = []
        try:
            for item in lst:
                if isinstance(item, list):
                    multi_label = []
                    for item_ in item:
                        multi_label.append(self._class_map(item_))
                    new_label.append(multi_label)
                else:
                    new_label.append(self._class_map(item))
        except KeyError:
            return lst
        return new_label

    def update_and_save_benchmark_frame(
        self, ltl_formula, new_prop_set, frames_of_interest, benchmark_frame: BenchmarkLTLFrame, save_dir
    ):
        file_name = f"benchmark_waymo_ltl_{ltl_formula}_{len(benchmark_frame.images_of_frames)}_0.pkl"
        benchmark_frame_ = copy.deepcopy(benchmark_frame)
        benchmark_frame_.frames_of_interest = frames_of_interest
        benchmark_frame_.ltl_formula = ltl_formula
        benchmark_frame_.proposition = new_prop_set
        benchmark_frame_.labels_of_frames = self.label_mapping_function(benchmark_frame.labels_of_frames)
        save_dict_to_pickle(dict_obj=benchmark_frame_, path=save_dir, file_name=file_name)


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

    def generate_until_time_delta(
        self,
        initial_number_of_frame=5,
        max_number_frame: int = 200,
        number_video_per_set_of_frame: int = 3,
        increase_rate: int = 1,
        ltl_logic: str = "prop1 U prop2",
        temporal_property: str = "U",
        conditional_property: str = "",
        present_prop1_till_prop2: bool = False,
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
                ltl_frame = self.until_time_delta_ltl_function(
                    logic_component=ltl_logic.split(" "),
                    temporal_property=temporal_property,
                    proposition_1=proposition[0],
                    proposition_2=proposition[1],
                    proposition_3=proposition[2],
                    conditional_property=conditional_property,
                    number_of_frame=number_frame,
                    present_prop1_till_prop2=present_prop1_till_prop2,
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
                    / f"timedelta_{number_frame - (initial_number_of_frame - 1)}_benchmark_{self._data_loader.name}_ltl_{ltl_frame.ltl_formula}_{number_frame}_{video_idx}.pkl"
                )

            number_frame += increase_rate
        if save_frames:
            ltl_frame.save_frames()

    def generate_unique_random_indices_within_range(self, start, end):
        """Generate a random sequence of unique indices between 'start' and 'end'.
        The length of the sequence will be random.
        """
        # Determine the length of the sequence (at most the number of elements in the range)
        sequence_length = random.randint(1, end - start + 1)
        # Generate the sequence
        return random.sample(range(start, end + 1), sequence_length)

    def until_time_delta_ltl_function(
        self,
        logic_component: list[str],
        proposition_1: str,
        temporal_property: str,
        number_of_frame: int,
        proposition_2: str | None = None,
        proposition_3: str | None = None,
        conditional_property: str = "",
        present_prop1_till_prop2: bool = False,
        initial_number_of_frame=5,
    ) -> BenchmarkLTLFrame:
        labels_of_frame = [None] * number_of_frame
        time_delta = number_of_frame - (initial_number_of_frame - 1)
        temp_frames_of_interest = []
        proposition_set = [proposition_1, proposition_2]
        ltl_formula = f'"{proposition_1}" U "{proposition_2}"'

        prop2_placeholder = 0 + time_delta
        # temp_frames_of_interest.append(0)
        labels_of_frame[0] = proposition_1
        # temp_frames_of_interest.append(prop2_placeholder)
        labels_of_frame[prop2_placeholder] = proposition_2

        prop1_range = self.generate_unique_random_indices_within_range(start=0, end=prop2_placeholder - 1)
        # labels_of_frame[0] = proposition_1
        if present_prop1_till_prop2:
            prop1_range = range(0, prop2_placeholder - 1)
        prop2_range = self.generate_unique_random_indices_within_range(
            start=prop2_placeholder + 1, end=number_of_frame - 1
        )

        p1_indexes = [0]
        for p1 in prop1_range:
            p1_indexes.append(p1)
            labels_of_frame[p1] = proposition_1

        p2_indexes = [prop2_placeholder]
        for p2 in prop2_range:
            p2_indexes.append(p2)
            labels_of_frame[p2] = proposition_2

        # p1_random_idx = sorted(p1_random_idx)
        p1_indexes = sorted(list(set(p1_indexes)))
        temp_frames_of_interest.append(p1_indexes + [prop2_placeholder])
        for p2_idx in p2_indexes:
            if p2_idx not in temp_frames_of_interest[0]:
                temp_frames_of_interest.append([p2_idx])

        return BenchmarkLTLFrame(
            ground_truth=True,
            ltl_formula=ltl_formula,
            proposition=proposition_set,
            number_of_frame=number_of_frame,
            frames_of_interest=temp_frames_of_interest,
            labels_of_frames=labels_of_frame,
        )

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
