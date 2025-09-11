import json
import os
import uuid
import cv2
import subprocess
import numpy as np
import gradio as gr
import tempfile
from typing import Dict, List, Iterable, Tuple

from ns_vfs.video.read_mp4 import Mp4Reader
from execute_with_mp4 import process_entry


def _load_entry_from_reader(video_path, query_text):
    reader = Mp4Reader(
        [{"path": video_path, "query": query_text}],
        openai_save_path="",
        sampling_rate_fps=0.5
    )
    data = reader.read_video()
    if not data:
        raise RuntimeError("No data returned by Mp4Reader (check video path)")
    return data[0]


def _make_empty_video(path, width=320, height=240, fps=1.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    writer.write(frame)
    writer.release()
    return path


def _crop_video_ffmpeg(input_path, output_path, frame_indices, prop_matrix):
    if len(frame_indices) == 0:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        _make_empty_video(output_path, width, height, fps=1.0)
        return

    def group_into_ranges(frames):
        if not frames:
            return []
        frames = sorted(set(frames))
        ranges = []
        start = prev = frames[0]
        for f in frames[1:]:
            if f == prev + 1:
                prev = f
            else:
                ranges.append((start, prev + 1))  # end-exclusive
                start = prev = f
        ranges.append((start, prev + 1))
        return ranges

    ranges = group_into_ranges(frame_indices)
    filters = []
    labels = []
    for i, (start, end) in enumerate(ranges):
        filters.append(
            f"[0:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[v{i}]"
        )
        labels.append(f"[v{i}]")
    filters.append(f"{''.join(labels)}concat=n={len(ranges)}:v=1:a=0[outv]")

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex", "; ".join(filters),
        "-map", "[outv]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def _crop_video(input_path: str, output_path: str, frame_indices: List[int], prop_matrix: Dict[str, List[int]]):
    input_path = str(input_path)
    output_path = str(output_path)

    # Probe width/height/fps
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    cap.release()
    if fps <= 0:
        fps = 30.0

    # If nothing to write, emit a 1-frame empty video
    if not frame_indices:
        from numpy import zeros, uint8
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 1.0, (width, height))
        out.write(zeros((height, width, 3), dtype=uint8))
        out.release()
        return

    # Helper: group consecutive integers into (start, end_exclusive)
    def _group_ranges(frames: Iterable[int]) -> List[Tuple[int, int]]:
        f = sorted(set(int(x) for x in frames))
        if not f:
            return []
        out = []
        s = p = f[0]
        for x in f[1:]:
            if x == p + 1:
                p = x
            else:
                out.append((s, p + 1))
                s = p = x
        out.append((s, p + 1))
        return out

    # Invert prop_matrix to {frame_idx: sorted [props]}
    props_by_frame: Dict[int, List[str]] = {}
    for prop, frames in (prop_matrix or {}).items():
        for fi in frames:
            fi = int(fi)
            props_by_frame.setdefault(fi, []).append(prop)
    for fi in list(props_by_frame.keys()):
        props_by_frame[fi] = sorted(set(props_by_frame[fi]))

    # Only subtitle frames we will output
    fi_set = set(int(x) for x in frame_indices)
    frames_with_labels = sorted(fi for fi in fi_set if props_by_frame.get(fi))

    # Compress consecutive frames that share the same label set
    grouped_label_spans: List[Tuple[int, int, Tuple[str, ...]]] = []
    prev_f = None
    prev_labels: Tuple[str, ...] = ()
    span_start = None
    for f in frames_with_labels:
        labels = tuple(props_by_frame.get(f, []))
        if prev_f is None:
            span_start, prev_f, prev_labels = f, f, labels
        elif (f == prev_f + 1) and (labels == prev_labels):
            prev_f = f
        else:
            grouped_label_spans.append((span_start, prev_f + 1, prev_labels))
            span_start, prev_f, prev_labels = f, f, labels
    if prev_f is not None and prev_labels:
        grouped_label_spans.append((span_start, prev_f + 1, prev_labels))

    # Build ASS subtitle file (top-right)
    def ass_time(t_sec: float) -> str:
        cs = int(round(t_sec * 100))
        h = cs // (100 * 3600)
        m = (cs // (100 * 60)) % 60
        s = (cs // 100) % 60
        cs = cs % 100
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def make_ass(width: int, height: int) -> str:
        lines = []
        lines.append("[Script Info]")
        lines.append("ScriptType: v4.00+")
        lines.append("ScaledBorderAndShadow: yes")
        lines.append(f"PlayResX: {width}")
        lines.append(f"PlayResY: {height}")
        lines.append("")
        lines.append("[V4+ Styles]")
        lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                     "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
                     "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
        # Font size 18 per your request; Alignment=9 (top-right)
        lines.append("Style: Default,DejaVu Sans,18,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
                     "0,0,0,0,100,100,0,0,1,2,0.8,9,16,16,16,1")
        lines.append("")
        lines.append("[Events]")
        lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

        for start_f, end_f, labels in grouped_label_spans:
            if not labels:
                continue
            start_t = ass_time(start_f / fps)
            end_t   = ass_time(end_f   / fps)
            text = r"\N".join(labels)  # stacked lines
            lines.append(f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{text}")

        return "\n".join(lines)

    tmp_dir = tempfile.mkdtemp(prefix="props_ass_")
    ass_path = os.path.join(tmp_dir, "props.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(make_ass(width, height))

    # Build trim/concat ranges from requested frame_indices
    ranges = _group_ranges(frame_indices)

    # Filtergraph with burned subtitles then trim/concat
    split_labels = [f"[s{i}]" for i in range(len(ranges))] if ranges else []
    out_labels   = [f"[v{i}]" for i in range(len(ranges))] if ranges else []

    filters = []
    ass_arg = ass_path.replace("\\", "\\\\")
    filters.append(f"[0:v]subtitles='{ass_arg}'[sub]")

    if len(ranges) == 1:
        s0, e0 = ranges[0]
        filters.append(f"[sub]trim=start_frame={s0}:end_frame={e0},setpts=PTS-STARTPTS[v0]")
    else:
        if ranges:
            filters.append(f"[sub]split={len(ranges)}{''.join(split_labels)}")
            for i, (s, e) in enumerate(ranges):
                filters.append(f"{split_labels[i]}trim=start_frame={s}:end_frame={e},setpts=PTS-STARTPTS{out_labels[i]}")

    if ranges:
        filters.append(f"{''.join(out_labels)}concat=n={len(ranges)}:v=1:a=0[outv]")

    filter_complex = "; ".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]" if ranges else "[sub]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.remove(ass_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


def _format_prop_ranges(prop_matrix: Dict[str, List[int]]) -> str:
    def group_into_ranges(frames: Iterable[int]) -> List[Tuple[int, int]]:
        f = sorted(set(int(x) for x in frames))
        if not f:
            return []
        ranges: List[Tuple[int, int]] = []
        s = p = f[0]
        for x in f[1:]:
            if x == p + 1:
                p = x
            else:
                ranges.append((s, p))   # inclusive end for display
                s = p = x
        ranges.append((s, p))
        return ranges

    if not prop_matrix:
        return "No propositions detected."

    lines = []
    for prop, frames in prop_matrix.items():
        ranges = group_into_ranges(frames)
        pretty = prop.replace("_", " ").title()
        if not ranges:
            lines.append(f"{pretty}: —")
            continue
        parts = [f"{a}" if a == b else f"{a}-{b}" for (a, b) in ranges]
        lines.append(f"{pretty}: {', '.join(parts)}")
    return "\n".join(lines)


# -----------------------------
# Gradio handler
# -----------------------------
def run_pipeline(input_video, mode, query_text, propositions_json, specification_text):
    """
    Returns: (cropped_video_path, prop_ranges_text, tl_text)
    """

    def _err(msg, width=320, height=240):  # keep outputs shape consistent
        tmp_out = os.path.join("/tmp", f"empty_{uuid.uuid4().hex}.mp4")
        _make_empty_video(tmp_out, width=width, height=height, fps=1.0)
        return (
            tmp_out,
            "No propositions detected.",
            f"Error: {msg}"
        )

    # Resolve video path
    if isinstance(input_video, dict) and "name" in input_video:
        video_path = input_video["name"]
    elif isinstance(input_video, str):
        video_path = input_video
    else:
        return _err("Please provide a video.")

    # Build entry
    if mode == "Natural language query":
        if not query_text or not query_text.strip():
            return _err("Please enter a query.")
        entry = _load_entry_from_reader(video_path, query_text)
    else:
        if not (propositions_json and propositions_json.strip()) or not (specification_text and specification_text.strip()):
            return _err("Please provide both Propositions (array) and Specification.")
        entry = _load_entry_from_reader(video_path, "dummy-query")
        try:
            props = json.loads(propositions_json)
            if not isinstance(props, list):
                return _err("Propositions must be a JSON array.")
        except Exception as e:
            return _err(f"Failed to parse propositions JSON: {e}")
        entry["tl"] = {
            "propositions": props,
            "specification": specification_text
        }

    # Compute FOI
    try:
        foi, prop_matrix = process_entry(entry)  # list of frame indices & {prop: [frames]}
        print(foi)
        print(prop_matrix)
    except Exception as e:
        return _err(f"Processing error: {e}")

    # Write cropped video
    try:
        out_path = os.path.join("/tmp", f"cropped_{uuid.uuid4().hex}.mp4")
        _crop_video(video_path, out_path, foi, prop_matrix)
        print(f"Wrote cropped video to: {out_path}")
    except Exception as e:
        return _err(f"Failed to write cropped video: {e}")

    # Build right-side text sections
    prop_ranges_text = _format_prop_ranges(prop_matrix)
    tl_text = (
        f"Propositions: {json.dumps(entry['tl']['propositions'], ensure_ascii=False)}\n"
        f"Specification: {entry['tl']['specification']}"
    )
    return out_path, prop_ranges_text, tl_text


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(css="""
#io-col {display: flex; gap: 1rem;}
#left {flex: 1;}
#right {flex: 1;}
""", title="NSVS-TL") as demo:

    gr.Markdown("# Neuro-Symbolic Visual Search with Temporal Logic")
    gr.Markdown(
        "Upload a video and either provide a natural-language **Query** *or* directly supply **Propositions** (array) + **Specification**. "
        "On the right, you'll get a **cropped video** containing only the frames of interest, a **Propositions by Frames** summary, and the combined TL summary."
    )

    with gr.Row(elem_id="io-col"):
        with gr.Column(elem_id="left"):
            mode = gr.Radio(
                choices=["Natural language query", "Props/Spec"],
                value="Natural language query",
                label="Input mode"
            )
            video = gr.Video(label="Upload Video")

            query = gr.Textbox(
                label="Query (natural language)",
                placeholder="e.g., a man is jumping and panting until he falls down"
            )

            propositions = gr.Textbox(
                label="Propositions (JSON array)",
                placeholder='e.g., ["man_jumps", "man_pants", "man_falls_down"]',
                lines=4,
                visible=False
            )
            specification = gr.Textbox(
                label="Specification",
                placeholder='e.g., ("woman_jumps" & "woman_claps") U "candle_is_blown"',
                visible=False
            )

            def _toggle_fields(m):
                if m == "Natural language query":
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

            mode.change(_toggle_fields, inputs=[mode], outputs=[query, propositions, specification])

            run_btn = gr.Button("Run", variant="primary")

            gr.Examples(
                label="Examples (dummy paths + queries)",
                examples=[
                    ["demo_videos/dog_jump.mp4", "a dog jumps until a red tube is in view"],
                    ["demo_videos/blue_shirt.mp4", "a girl in a green shirt until a candle is blown"],
                    ["demo_videos/car.mp4", "red car until a truck"]
                ],
                inputs=[video, query],
                cache_examples=False
            )

        with gr.Column(elem_id="right"):
            cropped_video = gr.Video(label="Cropped Video (Frames of Interest Only)")

            prop_ranges_out = gr.Textbox(
                label="Propositions by Frames",
                lines=6,
                interactive=False
            )

            tl_out = gr.Textbox(
                label="TL (Propositions & Specification)",
                lines=8,
                interactive=False
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[video, mode, query, propositions, specification],
        outputs=[cropped_video, prop_ranges_out, tl_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

