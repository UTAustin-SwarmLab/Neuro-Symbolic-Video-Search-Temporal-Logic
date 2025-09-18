# -------------------------------
# Preprocess: per-frame dicts {class: prob}
# -------------------------------
from ultralytics import YOLO
import numpy as np
import warnings
import tqdm
import os
import pickle
import re
from typing import Dict, List, Literal, Tuple

from ns_vfs.model_checker.property_checker import PropertyChecker
from ns_vfs.model_checker.video_automaton import VideoAutomaton
from ns_vfs.vlm.obj import DetectedObject
from ns_vfs.vlm.vllm_client import VLLMClient
from ns_vfs.video.frame import FramesofInterest, VideoFrame

PRINT_ALL = True
warnings.filterwarnings("ignore")


def preprocess_yolo(
    frames: List[np.ndarray],
    model_weights: str = "yolov8n.pt",
    device: str | int = "cuda:0",
    batch_size: int = 16,
    out_path: str = "yolo_det_cache.pkl",
    aggregation: Literal["max", "one_minus_prod", "mean"] = "max",
    conf_threshold: float = 0.001,
    iou: float = 0.7,
) -> str:
    """
    Run YOLOv8 detection on every frame and save a list of dicts:
      detections: List[Dict[str, float]] (one dict per frame).
      Each dict maps class_name (lowercase, COCO, with spaces) -> aggregated confidence in [0,1].
    """
    model = YOLO(model_weights)
    id_to_name: Dict[int, str] = {int(k): str(v).lower() for k, v in model.names.items()}

    def _aggregate(vals: List[float]) -> float:
        if not vals:
            return 0.0
        if aggregation == "max":
            return float(max(vals))
        if aggregation == "mean":
            return float(sum(vals) / len(vals))
        if aggregation == "one_minus_prod":
            prod = 1.0
            for p in vals:
                prod *= (1.0 - p)
            return float(1.0 - prod)
        raise ValueError(f"Unknown aggregation '{aggregation}'")

    detections: List[Dict[str, float]] = []

    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        results = model.predict(
            batch,
            device=device,
            conf=conf_threshold,
            iou=iou,
            verbose=False,
        )

        for r in results:
            frame_dict: Dict[str, float] = {}
            if r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int).tolist()
                confs   = r.boxes.conf.detach().cpu().numpy().astype(float).tolist()

                per_class: Dict[int, List[float]] = {}
                for cid, c in zip(cls_ids, confs):
                    per_class.setdefault(cid, []).append(c)

                for cid, vals in per_class.items():
                    name = id_to_name.get(int(cid), str(cid))  # e.g., "traffic light"
                    frame_dict[name] = _aggregate(vals)

            detections.append(frame_dict)

    assert len(detections) == len(frames), f"expected {len(frames)} dicts, got {len(detections)}"

    with open(out_path, "wb") as f:
        pickle.dump(detections, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path


# -------------------------------
# NSVS using cached YOLO dicts; 1 frame per step
# -------------------------------

# NEW: normalize props to YOLO label style (spaces, lowercase, collapsed whitespace)
_WS = re.compile(r"\s+")
def normalize_label_for_yolo(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ").replace("–", " ").replace("—", " ")
    s = _WS.sub(" ", s)
    return s

def run_nsvs_yolo(
    frames: List[np.ndarray],
    proposition: List[str],
    specification: str,
    *,
    yolo_cache_path: str = "yolo_det_cache.pkl",
    model_type: str = "dtmc",
    tl_satisfaction_threshold: float = 0.6,
    detection_threshold: float = 0.5,
    vlm_detection_threshold: float = 0.35,   # used as 'false_threshold' in calibrate()
    image_output_dir: str = "output",
) -> Tuple[List[VideoFrame], Dict[str, List[int]]]:
    """
    Replaces vlm.detect with cached YOLO (detection) per frame.
    Always uses 1-frame sequences.

    Returns:
      foi: List[VideoFrame]
      object_frame_dict: Dict[str, List[int]] mapping ORIGINAL prop -> list of frame indices where it was detected
    """
    if not os.path.exists(yolo_cache_path):
        raise FileNotFoundError(
            f"YOLO cache not found at '{yolo_cache_path}'. "
            f"Call preprocess_yolo(frames, out_path='yolo_det_cache.pkl') first."
        )

    with open(yolo_cache_path, "rb") as f:
        yolo_dets: List[Dict[str, float]] = pickle.load(f)

    if len(yolo_dets) != len(frames):
        raise ValueError(f"cache length {len(yolo_dets)} != frames length {len(frames)}")

    # Build a normalized lookup for each prop specifically for YOLO label matching
    # e.g., "traffic_light" -> "traffic light"
    prop_lookup: Dict[str, str] = {prop_raw: normalize_label_for_yolo(prop_raw) for prop_raw in proposition}

    automaton = VideoAutomaton(include_initial_state=True)
    automaton.set_up(proposition_set=proposition)   # keep original props for TL side

    checker = PropertyChecker(
        proposition=proposition,
        specification=specification,
        model_type=model_type,
        tl_satisfaction_threshold=tl_satisfaction_threshold,
        detection_threshold=detection_threshold,
    )

    frame_of_interest = FramesofInterest(1)  # fixed to 1-frame sequences
    object_frame_dict: Dict[str, List[int]] = {}

    calibrator = VLLMClient()

    def _mk_detected_object(name: str, confidence: float) -> DetectedObject:
        probability = calibrator.calibrate(confidence=confidence, false_threshold=vlm_detection_threshold)
        return DetectedObject(
            name=name,
            is_detected=confidence >= vlm_detection_threshold,
            confidence=confidence,
            probability=probability,
        )

    looper = range(len(frames)) if PRINT_ALL else tqdm.tqdm(range(len(frames)))
    for i in looper:
        if PRINT_ALL:
            print("\n" + "*" * 50 + f" {i}/{len(frames) - 1} " + "*" * 50)
            print("Detections:")

        det_dict = yolo_dets[i]  # {class (lowercase COCO, with spaces): aggregated_confidence}
        object_of_interest = {}

        for prop_raw in proposition:
            prop_yolo = prop_lookup[prop_raw]  # normalized to YOLO style
            conf = float(det_dict.get(prop_yolo, 0.0))  # 0.0 if YOLO label not present in this frame
            det = _mk_detected_object(prop_raw, conf)   # keep original prop name for downstream
            object_of_interest[prop_raw] = det

            if det.is_detected:
                object_frame_dict.setdefault(prop_raw, []).append(i)

            if PRINT_ALL:
                print(f"\t{prop_raw} (yolo='{prop_yolo}'): conf={det.confidence:.3f} -> prob={det.probability:.3f}"
                      + (" [DETECTED]" if det.is_detected else ""))

        frame = VideoFrame(
            frame_idx=i,
            frame_images=[frames[i]],    # single-frame sequence
            object_of_interest=object_of_interest,
        )

        if checker.validate_frame(frame_of_interest=frame):
            automaton.add_frame(frame=frame)
            frame_of_interest.frame_buffer.append(frame)
            model_check = checker.check_automaton(automaton=automaton)
            if model_check:
                automaton.reset()
                frame_of_interest.flush_frame_buffer()

    foi = frame_of_interest.foi_list

    if PRINT_ALL:
        print("\n" + "-" * 107)
        print("Detected frames of interest:")
        print(foi)

    return foi, object_frame_dict

