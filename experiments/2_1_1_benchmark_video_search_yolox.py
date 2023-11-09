from pathlib import Path

from _common import get_available_benchmark_video, write_to_csv_from_dict
from _metrics import classification_metrics

from ns_vfs.common.utility import get_file_or_dir_with_datetime, list_flatten
from ns_vfs.config.loader import load_config
from ns_vfs.data.frame import BenchmarkLTLFrame
from ns_vfs.frame_searcher import FrameSearcher
from ns_vfs.model.vision.yolo_x import YoloX
from ns_vfs.processor.benchmark_video_processor import BenchmarkVideoFrameProcessor
from ns_vfs.video_to_automaton import VideotoAutomaton

config = load_config()
benchmark_frame_video_root_dir = Path(
    "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/_validated_waymo_video"
)  # waymo_banchmark_with_ltl_label # _validated_benchmark_video
benchmark_image_set_dir = [x for x in benchmark_frame_video_root_dir.iterdir() if x.is_dir()]


# ltl_video_dir_set = [x for x in benchmark_image_set_dir[0].iterdir() if x.is_dir()]

csv_result = {}

############### Variable
root_path = Path("/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/yolo")
weight_path = Path("/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/weights/")
weight = "yolox_x.pth"
cv_model_list = ["yolo"]
mapping_threshold = (0.45, 0.60)  # (0.10, 0.58)
mapping_param_x0 = 0.25  # 0.10
mapping_param_k = 50
save_csv_file_name = get_file_or_dir_with_datetime(base_name="yolo_benchmark_search_result", ext=".csv")
###############

for image_set in benchmark_image_set_dir:
    ltl_video_dir_set = [x for x in image_set.iterdir() if x.is_dir()]
    for ltl_spec in ltl_video_dir_set:
        dataset = ltl_spec.parent.name
        csv_result["dataset"] = dataset
        csv_result["ltl_group"] = ltl_spec.name
        benchmark_video_file_list = get_available_benchmark_video(ltl_spec)
        ####################################################
        for benchmark_video_file in benchmark_video_file_list:
            ltl_formula = benchmark_video_file.name.split(".")[0].split("_ltl_")[-1]
            csv_result["ltl_formula"] = "".join(ltl_formula.split('"')[:-1]).replace(" ", "")
            csv_result["number_of_frame"] = int(ltl_formula.split('"')[-1].split("_")[1])

            # result[ltl_formula] = {}
            cv_detection_model = YoloX(
                config=config.YOLO,
                weight_path=weight_path / weight,
            )
            csv_result["cv_model"] = "yolo-x"
            csv_result["cv_model_weight"] = str(weight)
            benchmark_video_processor = BenchmarkVideoFrameProcessor(
                video_path=benchmark_video_file, artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH
            )

            benchmark_video: BenchmarkLTLFrame = benchmark_video_processor.benchmark_image_frames

            video_automata_builder = VideotoAutomaton(
                detector=cv_detection_model,
                video_processor=benchmark_video_processor,
                artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
                proposition_set=benchmark_video.proposition,
                save_annotation=False,  # TODO: Debug only
                save_image=False,  # TODO: Debug only
                ltl_formula=f"P>=0.80 [{benchmark_video.ltl_formula}]",
                mapping_threshold=mapping_threshold,
                verbose=False,
            )
            frame_sercher = FrameSearcher(
                video_automata_builder=video_automata_builder,
                video_processor=benchmark_video_processor,
            )

            frame_of_interest = frame_sercher.search()

            # writing result as csv

            actual_result = set(list_flatten(benchmark_video.frames_of_interest))
            predictive_result = set(list_flatten(frame_of_interest.foi_list))

            accuracy, precision, recall, f1 = classification_metrics(
                actual_result=actual_result, predicted_result=predictive_result
            )
            csv_result["accuracy"] = round(float(accuracy), 4)
            csv_result["precision"] = round(float(precision), 4)
            csv_result["recall"] = round(float(recall), 4)
            csv_result["f1_score"] = round(float(f1), 4)
            csv_result["mapping_false_threshold"] = mapping_threshold[0]
            csv_result["mapping_true_threshold"] = mapping_threshold[1]
            csv_result["mapping_param_x0"] = mapping_param_x0
            csv_result["mapping_param_k"] = mapping_param_k

            # save as csv
            write_to_csv_from_dict(
                dict_data=csv_result, csv_file_path=root_path, file_name=save_csv_file_name
            )

            # store results
            result_file = get_file_or_dir_with_datetime(
                f"result_{ltl_formula}_{benchmark_video_file.name}", ".pkl"
            )
