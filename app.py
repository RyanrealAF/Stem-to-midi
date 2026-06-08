"""
SongToMIDI — Full Song → Per-Stem MIDI Transcription
Pipeline: Demucs stem separation → Basic-Pitch MIDI transcription per stem
Background processing: job submitted to thread, UI polls via gr.Timer
Output: Individual .mid files + zipped bundle
"""

import os
import gc
import uuid
import zipfile
import tempfile
import shutil
import threading
import gradio as gr
from pathlib import Path
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
import demucs.separate
import torch

# ─── CONFIG ──────────────────────────────────────────────────────────────────

STEMS = ["vocals", "drums", "bass", "other"]

STEM_SETTINGS = {
    "vocals": {
        "onset_threshold": 0.6,
        "frame_threshold": 0.3,
        "minimum_note_length": 100,
        "minimum_frequency": 80,
        "maximum_frequency": 1100,
        "multiple_pitch_bends": False,
        "melodia_trick": True,
    },
    "drums": {
        "onset_threshold": 0.3,
        "frame_threshold": 0.2,
        "minimum_note_length": 50,
        "minimum_frequency": 30,
        "maximum_frequency": 8000,
        "multiple_pitch_bends": False,
        "melodia_trick": False,
    },
    "bass": {
        "onset_threshold": 0.5,
        "frame_threshold": 0.25,
        "minimum_note_length": 80,
        "minimum_frequency": 30,
        "maximum_frequency": 300,
        "multiple_pitch_bends": False,
        "melodia_trick": True,
    },
    "other": {
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length": 80,
        "minimum_frequency": 40,
        "maximum_frequency": 4000,
        "multiple_pitch_bends": True,
        "melodia_trick": True,
    },
}

DEMUCS_MODEL = "htdemucs"
POLL_INTERVAL = 3  # seconds between UI status checks

# ─── JOB REGISTRY ────────────────────────────────────────────────────────────
# Shared dict keyed by job_id. Thread-safe for simple read/write on CPython.
# Structure per job:
#   status: "running" | "done" | "error"
#   stage:  human-readable current step string
#   result: dict with midi paths + zip path (populated on done)
#   error:  error message string (populated on error)
#   work_dir: tempdir to clean up

JOBS: dict[str, dict] = {}


# ─── CORE PIPELINE ───────────────────────────────────────────────────────────

def _set_stage(job_id: str, stage: str):
    JOBS[job_id]["stage"] = stage


def separate_stems(audio_path: str, out_dir: str, job_id: str) -> dict[str, str]:
    _set_stage(job_id, "Separating stems via Demucs…")
    demucs.separate.main([
        "--mp3" if audio_path.endswith(".mp3") else "--wav",
        "-n", DEMUCS_MODEL,
        "--out", out_dir,
        audio_path,
    ])
    track_name = Path(audio_path).stem
    stem_dir = Path(out_dir) / DEMUCS_MODEL / track_name
    stem_paths = {}
    for stem in STEMS:
        candidate = stem_dir / f"{stem}.wav"
        if not candidate.exists():
            raise FileNotFoundError(f"Demucs did not produce {stem}.wav in {stem_dir}")
        stem_paths[stem] = str(candidate)
    return stem_paths


def transcribe_stem(stem_name: str, wav_path: str, out_dir: str, bp_model: Model) -> str:
    cfg = STEM_SETTINGS[stem_name]
    midi_out = Path(out_dir) / f"{stem_name}.mid"
    _, midi_data, _ = predict(
        wav_path,
        bp_model,
        onset_threshold=cfg["onset_threshold"],
        frame_threshold=cfg["frame_threshold"],
        minimum_note_length=cfg["minimum_note_length"],
        minimum_frequency=cfg["minimum_frequency"],
        maximum_frequency=cfg["maximum_frequency"],
        multiple_pitch_bends=cfg["multiple_pitch_bends"],
        melodia_trick=cfg["melodia_trick"],
    )
    midi_data.write(str(midi_out))
    return str(midi_out)


def bundle_midis(midi_paths: dict[str, str], out_dir: str, track_name: str) -> str:
    zip_path = Path(out_dir) / f"{track_name}_midi.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem, path in midi_paths.items():
            zf.write(path, arcname=f"{stem}.mid")
    return str(zip_path)


def _run_pipeline(job_id: str, audio_path: str):
    """Runs in a background thread. Updates JOBS[job_id] throughout."""
    job = JOBS[job_id]
    work_dir = job["work_dir"]

    try:
        track_name = Path(audio_path).stem

        # 1. Stem separation
        stem_dir = os.path.join(work_dir, "stems")
        os.makedirs(stem_dir, exist_ok=True)
        stem_paths = separate_stems(audio_path, stem_dir, job_id)

        # 2. Load model once
        _set_stage(job_id, "Loading Basic-Pitch model…")
        bp_model = Model(ICASSP_2022_MODEL_PATH)

        # 3. Transcribe per stem
        midi_dir = os.path.join(work_dir, "midi")
        os.makedirs(midi_dir, exist_ok=True)
        midi_paths = {}

        for stem in STEMS:
            _set_stage(job_id, f"Transcribing {stem}…")
            midi_paths[stem] = transcribe_stem(stem, stem_paths[stem], midi_dir, bp_model)

        # 4. Cleanup GPU
        del bp_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5. Bundle
        _set_stage(job_id, "Packaging ZIP…")
        zip_path = bundle_midis(midi_paths, work_dir, track_name)

        job["result"] = {**midi_paths, "zip": zip_path, "track_name": track_name}
        job["status"] = "done"
        job["stage"] = f"✓ Done — {track_name}"

    except Exception as e:
        job["status"] = "error"
        job["stage"] = "Error"
        job["error"] = str(e)
        shutil.rmtree(work_dir, ignore_errors=True)


# ─── GRADIO HANDLERS ─────────────────────────────────────────────────────────

def submit_job(audio_file):
    """Called on button click. Spawns thread, returns job_id to gr.State."""
    if audio_file is None:
        raise gr.Error("Upload an audio file first.")

    job_id = str(uuid.uuid4())
    work_dir = tempfile.mkdtemp(prefix="song2midi_")

    JOBS[job_id] = {
        "status": "running",
        "stage": "Queued…",
        "result": None,
        "error": None,
        "work_dir": work_dir,
    }

    thread = threading.Thread(target=_run_pipeline, args=(job_id, audio_file), daemon=True)
    thread.start()

    return (
        job_id,
        "⏳ Processing in background — outputs will appear when ready.",
        gr.update(interactive=False),  # disable button while running
    )


def poll_job(job_id):
    """Called by gr.Timer every POLL_INTERVAL seconds."""
    # No job yet (page just loaded)
    if not job_id:
        return (
            gr.update(),  # status unchanged
            gr.update(),  # vocals
            gr.update(),  # drums
            gr.update(),  # bass
            gr.update(),  # other
            gr.update(),  # zip
            gr.update(),  # button
        )

    job = JOBS.get(job_id)
    if job is None:
        return (
            "⚠ Job not found.",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(interactive=True),
        )

    if job["status"] == "running":
        return (
            f"⏳ {job['stage']}",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(interactive=False),
        )

    if job["status"] == "error":
        return (
            f"❌ Error: {job['error']}",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(interactive=True),
        )

    # done
    r = job["result"]
    return (
        f"✓ Processed: **{r['track_name']}** → 4 MIDI stems ready for download",
        r["vocals"],
        r["drums"],
        r["bass"],
        r["other"],
        r["zip"],
        gr.update(interactive=True),
    )


# ─── UI ──────────────────────────────────────────────────────────────────────

CSS = """
body { font-family: 'IBM Plex Mono', monospace; background: #0a0a0a; color: #e0e0e0; }
.gradio-container { max-width: 860px; margin: 0 auto; }
h1 { font-size: 2rem; letter-spacing: 0.08em; color: #f5f5f5; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
.status-box { background: #111; border: 1px solid #2a2a2a; border-radius: 4px; padding: 0.75rem 1rem; font-size: 0.85rem; color: #8aff8a; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="SongToMIDI") as demo:
    gr.Markdown(
        """
# SongToMIDI
**Upload a full song → get back individual MIDI files per stem.**

Pipeline: `Demucs htdemucs` stem separation → `Basic-Pitch` per-stem MIDI transcription

Stems: `vocals` · `drums` · `bass` · `other`

_GPU recommended. Processing runs in the background — close the tab and come back; files persist until the Space restarts._
        """
    )

    # Hidden state: stores current job_id
    job_state = gr.State(value=None)

    with gr.Row():
        audio_input = gr.Audio(
            label="Input Audio",
            type="filepath",
            sources=["upload"],
        )

    run_btn = gr.Button("▶  Transcribe", variant="primary", size="lg")
    status  = gr.Markdown(value="", elem_classes=["status-box"])

    gr.Markdown("### Output MIDI Files")

    with gr.Row():
        out_vocals = gr.File(label="vocals.mid", interactive=False)
        out_drums  = gr.File(label="drums.mid",  interactive=False)

    with gr.Row():
        out_bass  = gr.File(label="bass.mid",  interactive=False)
        out_other = gr.File(label="other.mid", interactive=False)

    out_zip = gr.File(label="📦 All stems (ZIP)", interactive=False)

    # Timer fires every POLL_INTERVAL seconds, calls poll_job with current job_id
    timer = gr.Timer(value=POLL_INTERVAL, active=False)

    # Submit: spawn thread, store job_id, activate timer
    run_btn.click(
        fn=submit_job,
        inputs=[audio_input],
        outputs=[job_state, status, run_btn],
    ).then(
        fn=lambda: gr.Timer(active=True),
        outputs=[timer],
    )

    # Poll: check job status, populate outputs when done, deactivate timer on finish
    timer.tick(
        fn=poll_job,
        inputs=[job_state],
        outputs=[status, out_vocals, out_drums, out_bass, out_other, out_zip, run_btn],
    )

    gr.Markdown(
        """
---
**Notes on MIDI accuracy:**
- Drums → onset-heavy settings; pitched notes require General MIDI ch10 remapping in DAW
- Bass → frequency capped at 300 Hz
- Vocals → melodia trick on; monophonic bias
- Other → multi-pitch-bends enabled for chords/pads/guitars

Built with [Demucs](https://github.com/facebookresearch/demucs) + [Basic-Pitch](https://github.com/spotify/basic-pitch)
        """
    )

if __name__ == "__main__":
    demo.launch()
