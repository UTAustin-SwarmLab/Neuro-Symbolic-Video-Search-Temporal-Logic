# WinTAK-style multi-camera demo checkpoint

This is Claude's existing tactical map demo in `research/multicam_demo/`.
Its application source and generated pages are preserved without modification.

## Run

From the repository root, with Python 3.8 or newer:

```sh
python3 demo/wintak/server.py
```

Open **http://127.0.0.1:8766/corridor_handoff_monitor.html** for the real-footage
four-camera corridor replay. The synthetic version is
**http://127.0.0.1:8766/perimeter_handoff_monitor.html**.
Use `--port 8764` to select another local port. Ctrl+C stops the server.
No Python packages, model weights or dataset downloads are required for replay.
The local server supports MP4 byte ranges for video seeking.
The existing page requests Google Fonts; system fonts remain available offline.

## Included state

- Real and synthetic generated pages, their templates and original Python sources.
- Four 10-minute, downscaled recorded camera videos under `research/multicam_demo/video/`.
- Embedded map layers, vehicle crops, tracks, alerts, UAS search-plan display and saved results.
- Build input JSON, evaluation summaries/rows, real monitor report and the author's research notes.

The archive, original 4K footage, feature caches, model weights, virtual environments,
intermediate crops are kept locally and excluded from this commit.
Rebuilding the real page needs only the included build input:

```sh
python3 research/multicam_demo/build_real_demo.py
```

The page is a recorded-data replay. Per-camera tracks are dataset annotations,
and the UAS plan is simulated. See `research/multicam_demo/NOTES.md` for the
original results and limitations. The separate Codex TRACE demo and research
video-agent services are outside this checkpoint.

## Verification

`qa/checkpoint.json` records checks against a separately extracted Git snapshot.
`qa/corridor.png` shows the preserved real-footage page.
The original page retains an 8px body margin; its full document height is 908px
at a 1440×900 viewport. This existing layout issue is recorded, with the first
failed layout assertion retained in `qa/checkpoint-first.json`.

Source attribution: RoundaboutHD (`yl4300/RoundaboutHD`) and OpenStreetMap.
Video hashes and byte sizes are recorded in `assets-manifest.json`.
