from __future__ import annotations

import copy

import cv2
from nuscenes.nuscenes import NuScenes

from ns_vfs.common.utility import save_dict_to_pickle
from ns_vfs.data.frame import BenchmarkLTLFrame

LABEL_MAPPING = {
    "animal": None,  # Assuming a generic animal category
    "human.pedestrian.adult": "person",
    "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.personal_mobility": "person",
    "human.pedestrian.police_officer": "person",
    "human.pedestrian.stroller": "person",
    "human.pedestrian.wheelchair": "person",
    "movable_object.barrier": None,  # No direct equivalent
    "movable_object.debris": None,  # No direct equivalent
    "movable_object.pushable_pullable": None,  # No direct equivalent
    "movable_object.trafficcone": None,  # No direct equivalent
    "static_object.bicycle_rack": None,  # Possibly 'bicycle' if the bikes are significant in the image
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "truck",  # Broadly categorizing as 'truck'
    "vehicle.emergency.ambulance": "truck",  # Or 'car', depending on size/shape
    "vehicle.emergency.police": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "truck",
    "vehicle.truck": "truck",
    # Additional mappings for other labels
    "flat.driveable_surface": None,  # No direct equivalent
    "flat.other": None,  # No direct equivalent
    "flat.sidewalk": None,  # No direct equivalent
    "flat.terrain": None,  # No direct equivalent
    "static.manmade": None,  # No direct equivalent, but could consider 'building' if part of your labels
    "static.other": None,  # No direct equivalent
    "static.vegetation": None,  # Could be 'tree' or 'plant' if those are part of your labels
    "vehicle.ego": "car",  # Assuming this is the ego vehicle (the car with the sensor/camera)
    "noise": None,  # No direct equivalent
}


class NuSceneImageLoader:
    """https://www.nuscenes.org/nuscenes?tutorial=nuscenes"""

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        verbose: bool = False,
        save_dir: str = "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/nuscene_video",
    ):
        self.name = "NuScene"
        self._nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self._nuscene: list = self._nusc.scene
        self._save_dir = save_dir
        self._scene_data = self.loading_data(self._nuscene)

    def get_label_count(self, lst):
        output = {}

        for item in lst:
            if isinstance(item, list):
                # If the item is a list, sort it, convert it to a tuple, and count occurrences
                for i in item:
                    if i not in output.keys():
                        output[i] = 1
                    else:
                        output[i] += 1
            else:
                # If the item is not a list, simply count occurrences
                output[item] += 1

        # Convert Counter object to dictionary (optional)
        return output

    def map_label(self, nuscenes_label):
        return LABEL_MAPPING.get(nuscenes_label, None)  # Returns None if no mapping exists

    def parse_object_class(self, annotation_tokens: list):
        labels = []
        for annotation_token in annotation_tokens:
            annotation = self._nusc.get("sample_annotation", annotation_token)
            label = self.map_label(annotation["category_name"])
            if label is not None:
                labels.append(label)
        return list(set(labels))

    def loading_data(self, nuscene_list: list):
        scene_data = {}
        for scene in nuscene_list:
            scene_token = scene["token"]
            scene_data[scene_token] = {
                "images_of_frame": [],
                "labels_of_frame": [],
            }
            data_validation = True
            for data in self._nusc.sample:
                if scene_token == data["scene_token"]:
                    front_cam_frame = cv2.imread(self._nusc.get_sample_data_path(data["data"]["CAM_FRONT"]))
                    # print(len(data["anns"]))
                    labels = self.parse_object_class(data["anns"])
                    # cv2.imwrite("test__.png", front_cam_frame)
                    if front_cam_frame is not None:
                        scene_data[scene_token]["images_of_frame"].append(front_cam_frame)
                        scene_data[scene_token]["labels_of_frame"].append(labels)
                    else:
                        data_validation = False
            if data_validation:
                self.generate_ltl_ground_truth(scene_data[scene_token])

            # save_dict_to_pickle(
            #     dict_obj=scene_data[scene_token],
            #     path=self._save_dir,
            #     file_name=f"nuscene_{scene_token}.pkl",
            # )

        return scene_data

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

    def prop1_u_prop2(self, prop1: str, prop2: str, lst: list[list]):
        ltl_formula = f'"{str(prop1)}" U "{str(prop2)}"'
        new_prop_set = [prop1, prop2]
        ltl_ground_truth = []
        tmp_ground_truth = []
        for idx, label in enumerate(lst):
            if isinstance(label, list):
                if prop1 in label:
                    tmp_ground_truth.append(idx)

            if isinstance(label, list):
                if prop2 in label:
                    tmp_ground_truth.append(idx)
                    ltl_ground_truth.append(list(set(tmp_ground_truth)))
                    tmp_ground_truth = []

        return ltl_formula, new_prop_set, ltl_ground_truth

    def f_prop1(self, prop1: str, lst: list[list]):
        ltl_formula = f'F "{str(prop1)}"'
        new_prop_set = [prop1]
        ltl_ground_truth = []
        tmp_ground_truth = []
        for idx, label in enumerate(lst):
            if isinstance(label, list):
                if prop1 in label:
                    tmp_ground_truth.append(idx)
                    ltl_ground_truth.append(list(set(tmp_ground_truth)))
                    tmp_ground_truth = []
            else:
                if prop1 == label:
                    tmp_ground_truth.append(idx)
                    ltl_ground_truth.append(list(set(tmp_ground_truth)))
                    tmp_ground_truth = []
        return ltl_formula, new_prop_set, ltl_ground_truth

    def prop1_and_prop2_u_prop3(self, prop1: str, prop2: str, prop3: str, lst: list[list]):
        ltl_formula = f'("{str(prop1)}" & "{str(prop2)}") U "{str(prop3)}"'
        new_prop_set = [prop1, prop2, prop3]
        ltl_ground_truth = []
        tmp_ground_truth = []
        for idx, label in enumerate(lst):
            if isinstance(label, list):
                if all(elem in label for elem in [prop1, prop2]):
                    tmp_ground_truth.append(idx)

            if isinstance(label, list):
                if prop3 in label:
                    tmp_ground_truth.append(idx)
                    ltl_ground_truth.append(list(set(tmp_ground_truth)))
                    tmp_ground_truth = []

        return ltl_formula, new_prop_set, ltl_ground_truth

    def generate_ltl_ground_truth(self, scene_data: dict):
        benchmark_frame = BenchmarkLTLFrame(
            ground_truth=True,
            ltl_formula="",
            proposition=[],
            number_of_frame=len(scene_data["images_of_frame"]),
            frames_of_interest=[],
            labels_of_frames=scene_data["labels_of_frame"],
            images_of_frames=scene_data["images_of_frame"],
        )

        label_count = self.get_label_count(benchmark_frame.labels_of_frames)
        sorted_items = sorted(label_count.items(), key=lambda x: x[1])

        # Ensure exactly two labels for lowest and highest
        try:
            if len(sorted_items) == 3:
                unique_prpos = [sorted_items[0][0]]  # First two unique values
                highest_count_props = [
                    sorted_items[1][0],
                    sorted_items[2][0],
                ]  # [sorted_items[-1][0], sorted_items[-2][0]]  # First two unique values
            elif len(sorted_items) > 3:
                unique_prpos = [sorted_items[0][0], sorted_items[1][0]]
                highest_count_props = [sorted_items[2][0], sorted_items[3][0]]
            elif len(sorted_items) < 3:
                unique_prpos = [sorted_items[0][0]]
                highest_count_props = [sorted_items[1][0]]
            else:
                unique_prpos = [sorted_items[0][0]]
                highest_count_props = [sorted_items[0][0]]

            for unique_prop in unique_prpos:
                if self.evaluate_unique_prop(unique_prop, benchmark_frame.labels_of_frames[:5]):
                    ltl_formula, new_prop_set, frames_of_interest = self.f_prop1(
                        prop1=str(unique_prop), lst=benchmark_frame.labels_of_frames
                    )
                    self.update_and_save_benchmark_frame(
                        ltl_formula, new_prop_set, frames_of_interest, benchmark_frame, self._save_dir
                    )
                else:
                    for prop in highest_count_props:
                        if prop != unique_prop:
                            ltl_formula, new_prop_set, frames_of_interest = self.prop1_u_prop2(
                                prop1=prop,
                                prop2=unique_prop,
                                lst=benchmark_frame.labels_of_frames,
                            )
                            self.update_and_save_benchmark_frame(
                                ltl_formula, new_prop_set, frames_of_interest, benchmark_frame, self._save_dir
                            )
                    if unique_prop not in highest_count_props:
                        ltl_formula, new_prop_set, frames_of_interest = self.prop1_and_prop2_u_prop3(
                            prop1=highest_count_props[0],
                            prop2=highest_count_props[1],
                            prop3=unique_prop,
                            lst=benchmark_frame.labels_of_frames,
                        )
                        self.update_and_save_benchmark_frame(
                            ltl_formula, new_prop_set, frames_of_interest, benchmark_frame, self._save_dir
                        )
        except IndexError:
            unique_prop = sorted_items[0][0]
            ltl_formula, new_prop_set, frames_of_interest = self.f_prop1(
                prop1=unique_prop, lst=benchmark_frame.labels_of_frames
            )
            self.update_and_save_benchmark_frame(
                ltl_formula, new_prop_set, frames_of_interest, benchmark_frame, self._save_dir
            )

    def update_and_save_benchmark_frame(
        self, ltl_formula, new_prop_set, frames_of_interest, benchmark_frame: BenchmarkLTLFrame, save_dir
    ):
        file_name = f"benchmark_nuscene_ltl_{ltl_formula}_{len(benchmark_frame.images_of_frames)}_0.pkl"
        benchmark_frame_ = copy.deepcopy(benchmark_frame)
        benchmark_frame_.frames_of_interest = frames_of_interest
        benchmark_frame_.ltl_formula = ltl_formula
        benchmark_frame_.proposition = new_prop_set
        # print(len(benchmark_frame_.images_of_frames))
        assert len(benchmark_frame_.images_of_frames) == len(benchmark_frame_.images_of_frames)
        assert benchmark_frame_.images_of_frames[0] is not None
        assert len(benchmark_frame_.frames_of_interest) > 0
        save_dict_to_pickle(dict_obj=benchmark_frame_, path=save_dir, file_name=file_name)


if __name__ == "__main__":
    test = NuSceneImageLoader(
        version="v1.0-trainval",
        dataroot="/opt/Neuro-Symbolic-Video-Frame-Search/store/datasets/NUSCENES/train",
        save_dir="/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/nuscene_video",
    )
