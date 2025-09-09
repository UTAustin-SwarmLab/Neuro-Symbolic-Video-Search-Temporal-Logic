import json
import itertools
import gradio as gr

from ns_vfs.nsvs import run_nsvs
from ns_vfs.video.read_mp4 import Mp4Reader

# -----------------------------
# Core logic (same as your program’s path)
# -----------------------------
def _frames_of_interest(entry):
    """Takes a single entry returned by Mp4Reader.read_video() and returns real frame indices (foi)."""
    tl = entry["tl"]
    # Be flexible: accept both 'propositions' and 'proposition'
    props_for_model = tl.get("propositions", tl.get("proposition", []))

    foi = run_nsvs(
        frames=entry['images'],
        proposition=props_for_model,
        specification=tl['specification'],
        model_name="InternVL2-8B",
        device=7
    )

    # Flatten list of lists
    foi = [i for sub in foi for i in sub]

    # Convert sampled frame indices to real frame indices
    scale = (entry["video_info"].fps) / (entry["metadata"]["sampling_rate_fps"])

    runs = []
    for _, grp in itertools.groupby(
        sorted(foi),
        key=lambda x, c=[0]: (x - (c.__setitem__(0, c[0] + 1) or c[0]))
    ):
        g = list(grp)
        runs.append((g[0], g[-1]))

    real = []
    for start_i, end_i in runs:
        a = int(round(start_i * scale))
        b = int(round(end_i * scale))
        if real and a <= real[-1]:
            a = real[-1] + 1
        real.extend(range(a, b + 1))
    return real


def _load_entry_from_reader(video_path, query_text):
    """
    Uses Mp4Reader to load a single entry.
    The reader expects a list of dicts: [{'path': <video_path>, 'query': <query_text>}]
    """
    reader = Mp4Reader(
        [{"path": video_path, "query": query_text}],
        openai_save_path="",
        sampling_rate_fps=0.1
    )
    data = reader.read_video()
    if not data:
        raise RuntimeError("No data returned by Mp4Reader (check video path)")
    return data[0]  # only one


# -----------------------------
# Gradio handler
# -----------------------------
def run_pipeline(input_video, mode, query_text, propositions_json, specification_text):
    """
    Gradio callback:
      - input_video: gr.File (single file) or a string path (when using examples)
      - mode: 'Natural language query' | 'Props/Spec'
      - query_text: string
      - propositions_json: string (JSON array)
      - specification_text: string
    Returns: frames-of-interest as a Python list (shown in gr.JSON)
    """

    # Handle when examples provide a path string vs real upload
    if isinstance(input_video, dict) and "name" in input_video:
        video_path = input_video["name"]
    elif isinstance(input_video, str):
        video_path = input_video
    else:
        return gr.update(value={"error": "Please provide a video."})

    if mode == "Natural language query":
        if not query_text.strip():
            return gr.update(value={"error": "Please enter a query."})

        # Let Mp4Reader build tl from the natural language query
        entry = _load_entry_from_reader(video_path, query_text)

    else:  # Props/Spec mode
        if not propositions_json.strip() or not specification_text.strip():
            return gr.update(value={"error": "Please provide both Propositions (JSON array) and Specification."})

        # Still use Mp4Reader to load frames/metadata/video_info, but query can be a dummy
        entry = _load_entry_from_reader(video_path, query_text or "dummy-query")

        # Parse propositions JSON (expects an array)
        try:
            props = json.loads(propositions_json)
            if not isinstance(props, list):
                return gr.update(value={"error": "Propositions must be a JSON array."})
        except Exception as e:
            return gr.update(value={"error": f"Failed to parse propositions JSON: {e}"})

        # Overwrite tl in the entry to use user-provided props/spec
        # Keep both keys for downstream flexibility
        entry["tl"] = {
            "proposition": props,
            "propositions": props,
            "specification": specification_text
        }

    # Compute frames of interest
    try:
        foi = _frames_of_interest(entry)
    except Exception as e:
        return gr.update(value={"error": f"Processing error: {e}"})

    # Return ONLY the frames of interest for the right panel
    return foi


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(css="""
#io-col {display: flex; gap: 1rem;}
#left {flex: 1;}
#right {flex: 1;}
""", title="Video FOI Finder") as demo:

    gr.Markdown("# Frames of Interest (FOI) Finder")
    gr.Markdown(
        "Upload a video and either provide a natural-language **Query** *or* directly supply **Propositions** (array) + **Specification**. "
        "On the right, you'll get the **frames of interest (foi)**."
    )

    with gr.Row(elem_id="io-col"):
        with gr.Column(elem_id="left"):
            mode = gr.Radio(
                choices=["Natural language query", "Props/Spec"],
                value="Natural language query",
                label="Input mode"
            )
            video = gr.File(label="MP4 video", file_count="single", file_types=[".mp4"])

            # Query
            query = gr.Textbox(
                label="Query (natural language)",
                placeholder="e.g., a unicorn is prancing until a pizza eats a strawberry"
            )

            # Props/Spec
            propositions = gr.Textbox(
                label="Propositions (JSON array)",
                placeholder='e.g., ["unicorn prancing", "pizza eats strawberry"]',
                lines=4,
                visible=False
            )
            specification = gr.Textbox(
                label="Specification",
                placeholder="e.g., detect when the unicorn is prancing until the pizza eats the strawberry",
                visible=False
            )

            # Show/hide inputs based on mode
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
                    ["/nas/mars/dataset/LongVideoBench/burn-subtitles/zVudr8cxHRE.mp4", "a unicorn is prancing until a pizza eats a strawberry"],
                    ["/nas/mars/dataset/LongVideoBench/burn-subtitles/zVudr8cxHRE.mp4", "the dog chases the ball and then sits by the tree"],
                ],
                inputs=[video, query],
                cache_examples=False
            )

        with gr.Column(elem_id="right"):
            foi_out = gr.JSON(label="Frames of interest (foi)")

    # Wire up
    run_btn.click(
        fn=run_pipeline,
        inputs=[video, mode, query, propositions, specification],
        outputs=[foi_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

