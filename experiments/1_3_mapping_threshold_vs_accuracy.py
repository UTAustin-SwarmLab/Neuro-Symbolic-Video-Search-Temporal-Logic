import logging
from pathlib import Path

import matplotlib.pyplot as plt
from _common import get_classification_score, write_to_csv_from_dict
from _metrics import classification_metrics

from ns_vfs.common.utility import get_file_or_dir_with_datetime, save_dict_to_pickle
from ns_vfs.config.loader import load_config
from ns_vfs.data.frame import BenchmarkLTLFrame
from ns_vfs.frame_searcher import FrameSearcher
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.model.vision.yolo import Yolo
from ns_vfs.processor.benchmark_video_processor import BenchmarkVideoFrameProcessor
from ns_vfs.video_to_automaton import VideotoAutomaton

# Set the logging level to INFO
logging.basicConfig(level=logging.ERROR)
config = load_config()
ROOTDIR = Path("/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/exp_1_3_dir")
DATADIR = ROOTDIR / "data"
LTL_DIR = DATADIR  # DATADIR / "Fprop1_25"
LTL_FILE_LIST = [file for file in LTL_DIR.glob("*") if file.is_file()]
CV_MODEL_LIST = ["yolo"]


"""
CV MODEL = YOLO
MANUAL_CONFIDENCE = RANDOM in RANGE
"""
result = {}
result_file = get_file_or_dir_with_datetime("result", ".csv")
# for benchmark_name_dir in LTL_FILE_LIST:
#     ltl_video_dir_set = [x for x in benchmark_name_dir.iterdir() if x.is_dir()]
#     if len(ltl_video_dir_set) > 0:
#         print(f"--processing {benchmark_name_dir.name}--")
#         print(f"number of ltl rule: {len(ltl_video_dir_set)}")
#         result[benchmark_name_dir.name] = {}
#         for ltl_video_dir in ltl_video_dir_set:
#             result[benchmark_name_dir.name][ltl_video_dir] = {}
#             benchmark_video_file_list = get_available_benchmark_video(ltl_video_dir)
#             print(f"number of examples of {ltl_video_dir.name}: {len(benchmark_video_file_list)}")

scaler = 0.1
true_threshold = 0.5
final_result = {}
while true_threshold <= 1.0:
    final_result[str(true_threshold)] = []
    for benchmark_video_file in LTL_FILE_LIST:
        # ** #
        print(f"Current threshold: {true_threshold}")
        result["true_threshold"] = true_threshold
        # ** #
        search_result_per_video = {}
        ltl_specification = "Fprop1"
        ltl_formula = (
            benchmark_video_file.name.split(".")[0].split("_ltl_")[-1].split("_")[0]
        )
        # result[benchmark_name_dir.name][ltl_video_dir.name][ltl_formula] = {}
        result["ltl_specification"] = ltl_specification
        result["ltl_formula"] = ltl_formula

        for cv_model in CV_MODEL_LIST:
            if cv_model == "yolo":
                cv_detection_model = Yolo(
                    config=config.YOLO, weight_path=config.YOLO.YOLO_CHECKPOINT_PATH
                )
            elif cv_model == "grounding_dino":
                cv_detection_model = GroundingDino(
                    config=config.GROUNDING_DINO,
                    weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
                    config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
                )
            benchmark_video_processor = BenchmarkVideoFrameProcessor(
                video_path=benchmark_video_file,
                artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
            )

            benchmark_video: BenchmarkLTLFrame = (
                benchmark_video_processor.benchmark_image_frames
            )

            video_automata_builder = VideotoAutomaton(
                detector=cv_detection_model,
                video_processor=benchmark_video_processor,
                artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
                proposition_set=benchmark_video.proposition,
                save_annotation=False,  # TODO: Debug only
                save_image=False,  # TODO: Debug only
                ltl_formula=f"P>=0.80 [{benchmark_video.ltl_formula}]",
                mapping_threshold=(0.10, true_threshold),
                verbose=False,
            )
            frame_sercher = FrameSearcher(
                video_automata_builder=video_automata_builder,
                video_processor=benchmark_video_processor,
            )

            frame_of_interest = frame_sercher.search()
            # search_result_per_video
            search_result_per_video["benchmark_video"] = benchmark_video
            search_result_per_video[cv_model] = frame_of_interest

            true_foi_list = benchmark_video.frames_of_interest
            # matching_accuracy
            flattened_true_foi = set(
                [item for sublist in true_foi_list for item in sublist]
            )
            flattened_predicted_foi = set(
                [item for sublist in frame_of_interest.foi_list for item in sublist]
            )

            # classification_metrics
            search_result_per_video = get_classification_score(search_result_per_video)
            accuracy, precision, recall, f1 = classification_metrics(
                actual_result=flattened_true_foi,
                predicted_result=flattened_predicted_foi,
            )
            result["accuracy"] = accuracy
            result["precision"] = precision
            result["recall"] = recall
            result["f1"] = f1
            write_to_csv_from_dict(
                dict_data=result,
                csv_file_path=str(ROOTDIR / "results"),
                file_name=result_file,
            )

            final_result[str(true_threshold)].append(result.copy())
            print(result)
    true_threshold += scaler
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #

threshold_data = []
precision_data = []
accuracy_data = []
recall_data = []
for threshold in final_result.keys():
    for result in final_result[threshold]:
        threshold_data.append(float(threshold))
        accuracy_data.append(result["accuracy"])
        precision_data.append(result["precision"])
        recall_data.append(result["recall"])

# Plotting
import seaborn as sns

ax = sns.lineplot(x=threshold_data, y=threshold_data, label="Mapping Estimation")
sns.lineplot(
    x=threshold_data, y=precision_data, ax=ax, label="precision", linestyle="dashed"
)
sns.lineplot(x=threshold_data, y=recall_data, label="racall", linestyle="dashed")
sns.lineplot(x=threshold_data, y=accuracy_data, label="accuracy", linestyle="dashed")
ax.set_xlabel("confidence")
ax.set_ylabel("accuracy")

plt.savefig(ROOTDIR / "plot.png", dpi=300)  # Adjust filename, format, and DPI as needed

# Show the plot (optional, especially if you're running this in a Jupyter notebook)
plt.show()

save_dict_to_pickle(
    final_result,
    path="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/",
    file_name="exp1_3_result_v1.pkl",
)
