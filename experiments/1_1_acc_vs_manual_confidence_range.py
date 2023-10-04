import random
from pathlib import Path

from _common import get_classification_score

from ns_vfs.common.utility import save_dict_to_pickle
from ns_vfs.config.loader import load_config
from ns_vfs.data.frame import BenchmarkLTLFrame
from ns_vfs.frame_searcher import FrameSearcher
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.model.vision.yolo import Yolo
from ns_vfs.processor.benchmark_video_processor import BenchmarkVideoFrameProcessor
from ns_vfs.video_to_automaton import VideotoAutomaton

config = load_config()
ROOTDIR = Path("/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/exp_1_1_dir")
DATADIR = ROOTDIR / "data"
LTL_DIR = DATADIR / "Fprop1_25"
LTL_FILE_LIST = [file for file in LTL_DIR.glob("*") if file.is_file()]
CV_MODEL_LIST = ["yolo"]


"""
CV MODEL = YOLO
MANUAL_CONFIDENCE = RANDOM in RANGE
"""
result = {}
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
random_probability = random.uniform(0, 0.1)
for path in LTL_FILE_LIST:
    for benchmark_video_file in LTL_FILE_LIST:
        search_result_per_video = {}
        ltl_specification = "Fprop1"
        ltl_formula = benchmark_video_file.name.split(".")[0].split("_ltl_")[-1].split("_")[0]
        # result[benchmark_name_dir.name][ltl_video_dir.name][ltl_formula] = {}
        result["ltl_specification"] = ltl_specification
        result["ltl_formula"] = ltl_formula

        for cv_model in CV_MODEL_LIST:
            if cv_model == "yolo":
                cv_detection_model = Yolo(config=config.YOLO, weight_path=config.YOLO.YOLO_CHECKPOINT_PATH)
            elif cv_model == "grounding_dino":
                cv_detection_model = GroundingDino(
                    config=config.GROUNDING_DINO,
                    weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
                    config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
                )
            benchmark_video_processor = BenchmarkVideoFrameProcessor(
                video_path=benchmark_video_file,
                artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
                manual_confidence_probability=1.0,
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

        # classification_metrics
        search_result_per_video = get_classification_score(search_result_per_video)
        result[ltl_formula] = search_result_per_video
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
save_dict_to_pickle(
    result, path="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/", file_name="all_test_result.pkl"
)
