"""Ask an off-the-shelf open-source VLM (Qwen3-VL, local MLX) the handoff question.

Run with the separate .venv-vlm environment. Model weights are read-only.
"""
import argparse
import json
from pathlib import Path
import time
import mlx.core as mx
from mlx_vlm import generate, load
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS = Path(__file__).resolve().parents[1] / "active_verification" / "data" / "models"
VERDICTS = ("all_arrived", "vehicle_missing", "unexpected_vehicle", "cannot_confirm")

INSTRUCTIONS = """You are the perimeter-security assistant for a military base.
Two fixed overhead cameras watch the same perimeter road. In every image the road runs horizontally and traffic moves left to right.
CAM-W watches the west road. CAM-N watches the north road. Between them is an unobserved 1,050 m stretch of road with no camera.
A vehicle that passes CAM-W normally reaches CAM-N 70 to 130 seconds later. Each image is labeled with its camera and time in seconds.
The images are still frames sampled from the video, a few seconds apart.

Decide whether every vehicle that passed CAM-W later passed CAM-N.
Use exactly one verdict:
- "all_arrived": every vehicle that passed CAM-W also passed CAM-N, and no other vehicle appeared at CAM-N.
- "vehicle_missing": at least one vehicle that passed CAM-W never reached CAM-N.
- "unexpected_vehicle": a vehicle reached CAM-N that never passed CAM-W.
- "cannot_confirm": the CAM-N view was too degraded to confirm the arrivals.
Return only JSON: {"w_count": <vehicles that passed CAM-W>, "n_count": <vehicles that passed CAM-N>, "verdict": "<one verdict>", "missing_or_unexpected": "<short description, or none>"}
"""


def parse(text):
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
    data = json.loads(clean)
    assert data["verdict"] in VERDICTS
    return data


def main(args):
    src = ROOT / "results" / args.inputs
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((src / "manifest.json").read_text())
    truth = json.loads((src / "truth.json").read_text())
    mx.set_memory_limit(16 * 1024**3)
    model, processor = load(str(MODELS / args.model), trust_remote_code=False, local_files_only=True)
    rows = []
    for item in manifest:
        content = [{"type": "text", "text": INSTRUCTIONS}]
        images = []
        for f in item["frames"]:
            content.append({"type": "text", "text": f"\n{f['cam']} at t={f['t']:.1f} s:\n"})
            content.append({"type": "image"})
            images.append(Image.open(ROOT / f["file"]).convert("RGB"))
        content.append({"type": "text", "text": "\nReturn the JSON now.\n"})
        prompt = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                               add_generation_prompt=True)
        assert prompt.count("<|image_pad|>") == len(images)
        t0 = time.perf_counter()
        mx.random.seed(0)
        result = generate(model, processor, prompt, image=images, max_tokens=200, temperature=0.0, seed=0,
                          prefill_step_size=512, verbose=False)
        elapsed = time.perf_counter() - t0
        row = dict(key=item["key"], images=len(images), seconds=elapsed, text=result.text,
                   prompt_tokens=result.prompt_tokens, **truth[item["key"]])
        try:
            p = parse(result.text)
            row.update(parsed=p, parse_ok=True, verdict=p["verdict"],
                       correct=p["verdict"] in row["expected"].split("|"),
                       w_count_ok=p.get("w_count") == row["passed_w"],
                       n_count_ok=p.get("n_count") == row["passed_n"])
        except Exception as e:
            row.update(parse_ok=False, error=repr(e), correct=False, verdict=None)
        rows.append(row)
        print(item["key"], row.get("verdict"), "expected", row["expected"], "ok" if row["correct"] else "WRONG",
              "%.1fs" % elapsed, flush=True)
        (out / "rows.json").write_text(json.dumps(rows, indent=1))
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r["correct"])
    summary = dict(model=args.model, episodes=len(rows), correct=sum(r["correct"] for r in rows),
                   by_kind={k: [sum(v), len(v)] for k, v in by.items()},
                   parse_failures=sum(not r["parse_ok"] for r in rows),
                   w_count_ok=sum(bool(r.get("w_count_ok")) for r in rows),
                   n_count_ok=sum(bool(r.get("n_count_ok")) for r in rows),
                   verdict_counts={v: sum(r.get("verdict") == v for r in rows) for v in VERDICTS},
                   mean_seconds=sum(r["seconds"] for r in rows) / len(rows),
                   mean_images=sum(r["images"] for r in rows) / len(rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default="vlm_inputs_v1")
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="Qwen3-VL-4B-Instruct-4bit")
    main(p.parse_args())
