import json
import os
import uuid
import cv2
import subprocess
import numpy as np
import gradio as gr

from ns_vfs.nsvs import run_nsvs
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

def _write_cropped_video(input_path, frame_indices, output_path):
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

# -----------------------------
# Gradio handler
# -----------------------------
def run_pipeline(input_video, mode, query_text, propositions_json, specification_text):
    """
    Returns: (cropped_video_path, tl_text)
    """

    def _err(msg, width=320, height=240):
        # Return a tiny blank "error" video so the UI still shows something
        tmp_out = os.path.join("/tmp", f"empty_{uuid.uuid4().hex}.mp4")
        _make_empty_video(tmp_out, width=width, height=height, fps=1.0)
        return (
            tmp_out,
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
        foi = process_entry(entry)  # expected list/array of frame indices
    except Exception as e:
        return _err(f"Processing error: {e}")

    # Write cropped video
    try:
        out_path = os.path.join("/tmp", f"cropped_{uuid.uuid4().hex}.mp4")
        _write_cropped_video(video_path, foi, out_path)
        print(f"Wrote cropped video to: {out_path}")
    except Exception as e:
        return _err(f"Failed to write cropped video: {e}")

    # Combined TL text
    tl_text = f"Propositions: {json.dumps(entry['tl']['propositions'], ensure_ascii=False)}\n" \
              f"Specification: {entry['tl']['specification']}"
    return out_path, tl_text

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
        "On the right, you'll get a **cropped video** containing only the frames of interest and the combined TL summary."
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
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible(False))
                else:
                    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

            mode.change(_toggle_fields, inputs=[mode], outputs=[query, propositions, specification])

            run_btn = gr.Button("Run", variant="primary")

            gr.Examples(
                label="Examples (dummy paths + queries)",
                examples=[
                    ["demo_videos/dog_jump.mp4", "a dog jumps until a red tube is in view"],
                    ["demo_videos/blue_shirt.mp4", "a girl in a green shirt until a candle is blown"]
                ],
                inputs=[video, query],
                cache_examples=False
            )

        with gr.Column(elem_id="right"):
            cropped_video = gr.Video(label="Cropped Video (Frames of Interest Only)")
            tl_out = gr.Textbox(
                label="TL (Propositions & Specification)",
                lines=8,
                interactive=False
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[video, mode, query, propositions, specification],
        outputs=[cropped_video, tl_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

