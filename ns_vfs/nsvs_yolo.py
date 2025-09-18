# -------------------------------
# Preprocess: per-frame dicts {class: List[(conf, (x1,y1,x2,y2))]}
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
    conf_threshold: float = 0.001,
    iou: float = 0.7,
) -> str:
    """
    Run YOLOv8 detection on every frame and save a list of dicts.
      cache format:
        yolo_dets: List[ Dict[str, List[Tuple[float, Tuple[float,float,float,float]]]] ]
        # one item per frame
        # each frame dict maps: class_name (lowercase, spaces) ->
        #     list of (confidence, (x1, y1, x2, y2)) in pixel coordinates
    """
    model = YOLO(model_weights)
    id_to_name: Dict[int, str] = {int(k): str(v).lower() for k, v in model.names.items()}

    yolo_dets: List[Dict[str, List[Tuple[float, Tuple[float, float, float, float]]]]] = []

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
            frame_dict: Dict[str, List[Tuple[float, Tuple[float, float, float, float]]]] = {}
            if r.boxes is not None and len(r.boxes) > 0:
                # xyxy in pixels, conf, and class ids
                xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(float)
                confs = r.boxes.conf.detach().cpu().numpy().astype(float)
                cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
                    name = id_to_name.get(int(cid), str(cid))  # e.g., "traffic light"
                    frame_dict.setdefault(name, []).append(
                        (float(conf), (float(x1), float(y1), float(x2), float(y2)))
                    )

            yolo_dets.append(frame_dict)

    assert len(yolo_dets) == len(frames), f"expected {len(frames)} dicts, got {len(yolo_dets)}"

    with open(out_path, "wb") as f:
        pickle.dump(yolo_dets, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path


# -------------------------------
# NSVS using cached YOLO dicts; 1 frame per step
# -------------------------------

# normalize props to YOLO label style (spaces, lowercase, collapsed whitespace)
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
) -> Tuple[List[VideoFrame], Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]]]:
    """
    Replaces vlm.detect with cached YOLO per frame (1-frame sequences).
    Returns:
      foi: List[VideoFrame]
      object_frame_bounding_boxes:
         Dict[str, List[(frame_index, (x1, y1, x2, y2))]]
         # one bbox per frame (the highest-confidence bbox for that class in that frame)
    """
    if not os.path.exists(yolo_cache_path):
        raise FileNotFoundError(
            f"YOLO cache not found at '{yolo_cache_path}'. "
            f"Call preprocess_yolo(frames, out_path='yolo_det_cache.pkl') first."
        )

    with open(yolo_cache_path, "rb") as f:
        # List[Dict[str, List[(conf, (x1,y1,x2,y2))]]]
        yolo_dets: List[Dict[str, List[Tuple[float, Tuple[float, float, float, float]]]]] = pickle.load(f)

    if len(yolo_dets) != len(frames):
        raise ValueError(f"cache length {len(yolo_dets)} != frames length {len(frames)}")

    # Build normalized lookup (e.g., "traffic_light" -> "traffic light")
    prop_lookup: Dict[str, str] = {prop_raw: normalize_label_for_yolo(prop_raw) for prop_raw in proposition}

    automaton = VideoAutomaton(include_initial_state=True)
    automaton.set_up(proposition_set=proposition)   # original props for TL

    checker = PropertyChecker(
        proposition=proposition,
        specification=specification,
        model_type=model_type,
        tl_satisfaction_threshold=tl_satisfaction_threshold,
        detection_threshold=detection_threshold,
    )

    frame_of_interest = FramesofInterest(1)  # 1-frame sequences
    object_frame_bounding_boxes: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]] = {}

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

        # Per-frame dict: class -> List[(conf, (x1,y1,x2,y2))]
        det_dict = yolo_dets[i]
        object_of_interest = {}

        for prop_raw in proposition:
            yolo_label = prop_lookup[prop_raw]
            dets_for_class = det_dict.get(yolo_label, [])

            # confidence for decision = max conf for that class in this frame (0 if none)
            if dets_for_class:
                confs = [c for c, _ in dets_for_class]
                max_idx = int(np.argmax(confs))
                best_conf, best_bbox = dets_for_class[max_idx]
            else:
                best_conf, best_bbox = 0.0, None

            det = _mk_detected_object(prop_raw, float(best_conf))
            object_of_interest[prop_raw] = det

            if det.is_detected and best_bbox is not None:
                # one bbox per frame (highest-confidence one)
                object_frame_bounding_boxes.setdefault(prop_raw, []).append((i, best_bbox))

            if PRINT_ALL:
                if best_bbox is not None:
                    x1, y1, x2, y2 = best_bbox
                    print(f"\t{prop_raw} (yolo='{yolo_label}'): conf={det.confidence:.3f} "
                          f"-> prob={det.probability:.3f} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
                          + (" [DETECTED]" if det.is_detected else ""))
                else:
                    print(f"\t{prop_raw} (yolo='{yolo_label}'): conf=0.000 -> prob={det.probability:.3f}")

        frame = VideoFrame(
            frame_idx=i,
            frame_images=[frames[i]],    # single-frame
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

    # NOTE: replaced the old object_frame_dict return
    return foi, object_frame_bounding_boxes

