import json
import gradio as gr

from ns_vfs.nsvs import run_nsvs
from ns_vfs.video.read_mp4 import Mp4Reader
from execute_with_mp4 import process_entry

def _load_entry_from_reader(video_path, query_text):
    reader = Mp4Reader(
        [{"path": video_path, "query": query_text}],
        openai_save_path="",
        sampling_rate_fps=1
    )
    data = reader.read_video()
    if not data:
        raise RuntimeError("No data returned by Mp4Reader (check video path)")
    return data[0]

def _format_tl_text(tl: dict) -> str:
    # Directly dump the propositions exactly as they are (JSON array format)
    props_str = json.dumps(tl["propositions"], ensure_ascii=False)
    spec_str = tl["specification"]
    return f"Propositions: {props_str}\nSpecification: {spec_str}"

# -----------------------------
# Gradio handler
# -----------------------------
def run_pipeline(input_video, mode, query_text, propositions_json, specification_text):
    """
    Returns: (foi_text, tl_text)
    """

    def _err(msg):
        return (
            gr.update(value=f"Error: {msg}"),  # FOI textbox
            gr.update(value="")                # combined TL textbox
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
        if not query_text.strip():
            return _err("Please enter a query.")
        entry = _load_entry_from_reader(video_path, query_text)
    else:
        if not propositions_json.strip() or not specification_text.strip():
            return _err("Please provide both Propositions (array) and Specification.")

        entry = _load_entry_from_reader(video_path, "dummy-query")
        try:
            print(propositions_json)
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

    # FOI as a single string (JSON-style)
    foi_text = ", ".join(map(str, foi))

    # Combined TL text
    tl_text = _format_tl_text(entry["tl"])

    return foi_text, tl_text

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
        "On the right, you'll get the **frames of interest** (as a string) followed by the combined TL summary."
    )

    with gr.Row(elem_id="io-col"):
        with gr.Column(elem_id="left"):
            mode = gr.Radio(
                choices=["Natural language query", "Props/Spec"],
                value="Natural language query",
                label="Input mode"
            )
            # video = gr.File(label="MP4 video", file_count="single", file_types=[".mp4"])
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
                    ["demo_videos/blue_shirt.mp4", "a woman in a blue shirt claps until a candle is blown"]
                    # ["demo_videos/teaser-gen3.mp4", "waves until storm"],
                    # ["demo_videos/teaser-pika.mp4", "waves until storm"]
                ],
                inputs=[video, query],
                cache_examples=False
            )

        with gr.Column(elem_id="right"):
            foi_out = gr.Textbox(label="Frames of interest", lines=6, interactive=False)
            tl_out = gr.Textbox(
                label="TL (Propositions & Specification)",
                lines=8,
                interactive=False
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[video, mode, query, propositions, specification],
        outputs=[foi_out, tl_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

