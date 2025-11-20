import os
import time
import random
import gradio as gr

MOCK_DATA = {
    "demo_videos/dog_jump.mp4": {
        "question": "What is the dog doing?",
        "caption": "A dog is repeatedly jumping in a grassy area, seemingly trying to catch something.",
        "tl_spec": 'F ("dog_jumps")',
        "cropped_video_path": "demo_videos/dog_jump.mp4"
    },
    "demo_videos/blue_shirt.mp4": {
        "question": "What happens with the candle?",
        "caption": "A person in a green shirt is lighting a candle, which is then blown out.",
        "tl_spec": 'F ("candle_lit" & F "candle_blown")',
        "cropped_video_path": "demo_videos/blue_shirt.mp4"
    },
    "demo_videos/car.mp4": {
        "question": "Describe the vehicles in the video.",
        "caption": "A red car is shown, followed by a large truck appearing in the frame.",
        "tl_spec": 'F ("red_car" & F "truck")',
        "cropped_video_path": "demo_videos/car.mp4"
    },
    "demo_videos/newyork_1.mp4": {
        "question": "What landmarks are visible?",
        "caption": "The video shows a yellow taxi driving with the Empire State Building visible in the background.",
        "tl_spec": 'F ("taxi" & F "empire_state_building")',
        "cropped_video_path": "demo_videos/newyork_1.mp4"
    },
    "demo_videos/chicago_2.mp4": {
        "question": "What is happening on the water?",
        "caption": "A boat is moving across the water, with a large Ferris wheel visible in the background.",
        "tl_spec": 'F ("boat" & F "ferris_wheel")',
        "cropped_video_path": "demo_videos/chicago_2.mp4"
    }
}

def run_mock_pipeline(video_path, question_text):
    def _get_output_for_video(path):
        return MOCK_DATA.get(path, {
            "caption": "This is a default caption for the processed video.",
            "tl_spec": "G (some_property)",
            "cropped_video_path": path
        })

    processing_time = random.uniform(5, 10)
    time.sleep(processing_time)

    if not video_path:
         return None, "No video provided.", "Please upload a video first."

    outputs = _get_output_for_video(video_path)

    cropped_video_path = outputs.get("cropped_video_path", video_path)
    caption_text = outputs["caption"]
    tl_spec_text = outputs["tl_spec"]

    return cropped_video_path, caption_text, tl_spec_text

def update_example(video_path):
    return MOCK_DATA.get(video_path, {}).get("question", "")

with gr.Blocks(css="""
#io-col {display: flex; gap: 1rem;}
#left {flex: 1;}
#right {flex: 1;}
""", title="Video QA") as demo:

    gr.Markdown("# NeuS-QA: Grounding Long-Form Video Understanding in Temporal Logic and Neuro-Symbolic Reasoning")
    gr.Markdown("Upload a video, ask a question, and see the analysis.")

    with gr.Row(elem_id="io-col"):
        with gr.Column(elem_id="left"):
            video = gr.Video(label="Upload Video")

            question = gr.Textbox(
                label="Question",
                placeholder="e.g., What is the dog doing?"
            )

            run_btn = gr.Button("Run", variant="primary")

            gr.Examples(
                label="Examples",
                examples=[[k, v['question']] for k, v in MOCK_DATA.items()],
                inputs=[video, question],
                cache_examples=False,
            )

        with gr.Column(elem_id="right"):
            frames_of_interest_video = gr.Video(label="Frames of Interest")
            caption_out = gr.Textbox(label="Caption", lines=4, interactive=False)
            tl_out = gr.Textbox(label="TL (Proposition & Specification)", lines=4, interactive=False)

    video.change(fn=update_example, inputs=[video], outputs=[question], queue=False)

    run_btn.click(
        fn=run_mock_pipeline,
        inputs=[video, question],
        outputs=[frames_of_interest_video, caption_out, tl_out]
    )

if __name__ == "__main__":
    print("Starting Video QA App on http://localhost:8080/")
    demo.launch(server_name="0.0.0.0", server_port=8888)
