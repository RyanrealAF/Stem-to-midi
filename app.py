"""
SongToMIDI — Full Song → Per-Stem MIDI Transcription with Verification
Pipeline: Demucs stem separation → Per-stem analysis → Basic-Pitch transcription
          → Audio verification (MIDI vs stem) → Adaptive retry if low score
Background processing: job submitted to thread, UI polls via gr.Timer
Output: Individual .mid files + verification report + zipped bundle
"""

import os
import gc
import uuid
import zipfile
import tempfile
import shutil
import threading
import gradio as gr
import numpy as np
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

STEMS = ["vocals", "drums", "bass", "other"]

# Base settings per stem (adapted dynamically based on analysis)
STEM_SETTINGS = {
    "vocals": {
        "onset_threshold": 0.6, "frame_threshold": 0.3,
        "minimum_note_length": 100,
        "minimum_frequency": 80, "maximum_frequency": 1100,
        "multiple_pitch_bends": False, "melodia_trick": True,
    },
    "drums": {
        "onset_threshold": 0.3, "frame_threshold": 0.2,
        "minimum_note_length": 50,
        "minimum_frequency": 30, "maximum_frequency": 8000,
        "multiple_pitch_bends": False, "melodia_trick": False,
    },
    "bass": {
        "onset_threshold": 0.5, "frame_threshold": 0.25,
        "minimum_note_length": 80,
        "minimum_frequency": 30, "maximum_frequency": 300,
        "multiple_pitch_bends": False, "melodia_trick": True,
    },
    "other": {
        "onset_threshold": 0.5, "frame_threshold": 0.3,
        "minimum_note_length": 80,
        "minimum_frequency": 40, "maximum_frequency": 4000,
        "multiple_pitch_bends": True, "melodia_trick": True,
    },
}

DEMUCS_MODEL = "htdemucs"
POLL_INTERVAL = 3
VERIFY_THRESHOLD = 0.65  # minimum verification score to accept
MAX_RETRIES = 3

JOBS: dict[str, dict] = {}


# ─── PER-STEM ANALYSIS ───────────────────────────────────────────────────────

def analyze_stem(wav_path: str, stem_name: str) -> dict:
    """Analyze a stem to guide adaptive transcription parameters."""
    import librosa

    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    duration = len(y) / sr

    # BPM
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
    except Exception:
        bpm = 0.0

    # Key via chroma + Krumhansl
    key = "unknown"
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        # Krumhansl major/minor profiles
        major_prof = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor_prof = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        best_score, best_key = -1, "unknown"
        for i in range(12):
            for prof, suffix in [(major_prof, " major"), (minor_prof, " minor")]:
                rotated = np.roll(prof, i)
                score = np.corrcoef(chroma_mean, rotated)[0, 1]
                if score > best_score:
                    best_score, best_key = score, names[i] + suffix
        key = best_key
    except Exception:
        pass

    # Pitch range via pyin (monophonic estimate)
    pitch_min_hz, pitch_max_hz = 0.0, 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=30, fmax=4000, sr=sr)
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) > 10:
            pitch_min_hz = float(np.percentile(voiced_f0, 5))
            pitch_max_hz = float(np.percentile(voiced_f0, 95))
    except Exception:
        pass

    # Energy
    rms = librosa.feature.rms(y=y)
    mean_energy_db = float(20 * np.log10(rms.mean() + 1e-10))

    # Onset density
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        onset_density = len(onsets) / max(duration, 0.1)  # onsets per second
    except Exception:
        onset_density = 0.0

    # Spectral centroid (brightness)
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness = float(cent.mean())
    except Exception:
        brightness = 0.0

    return {
        "bpm": round(bpm, 1),
        "key": key,
        "pitch_min_hz": round(pitch_min_hz, 1),
        "pitch_max_hz": round(pitch_max_hz, 1),
        "energy_db": round(mean_energy_db, 1),
        "onset_density": round(onset_density, 2),
        "brightness_hz": round(brightness, 0),
        "duration_sec": round(duration, 2),
    }


def adapt_params(base: dict, analysis: dict, stem_name: str, attempt: int) -> dict:
    """Adapt Basic-Pitch params based on stem analysis and retry attempt."""
    p = dict(base)

    # Tighten frequency range if we detected a clear pitch range
    if analysis["pitch_min_hz"] > 0 and stem_name != "drums":
        p["minimum_frequency"] = max(20, analysis["pitch_min_hz"] * 0.8)
        p["maximum_frequency"] = min(8000, analysis["pitch_max_hz"] * 1.25)

    # High onset density → more sensitive onset detection
    if analysis["onset_density"] > 4.0:
        p["onset_threshold"] = max(0.15, p["onset_threshold"] - 0.15)

    # Low energy → more sensitive frame detection
    if analysis["energy_db"] < -30:
        p["frame_threshold"] = max(0.1, p["frame_threshold"] - 0.1)

    # On retry, be more permissive
    if attempt > 0:
        p["onset_threshold"] = max(0.15, p["onset_threshold"] - 0.1 * attempt)
        p["frame_threshold"] = max(0.1, p["frame_threshold"] - 0.05 * attempt)

    return p


# ─── VERIFICATION ────────────────────────────────────────────────────────────

def verify_midi(midi_path: str, stem_wav_path: str) -> dict:
    """
    Verify a MIDI file matches its source stem audio.
    Synthesizes the MIDI and compares chroma + onset features.
    Returns {'score': 0-1, 'pitch_score': 0-1, 'timing_score': 0-1, 'details': str}
    """
    import librosa
    import pretty_midi

    try:
        # Load original stem
        y_orig, sr = librosa.load(stem_wav_path, sr=22050, mono=True)

        # Synthesize MIDI
        pm = pretty_midi.PrettyMIDI(midi_path)
        y_synth = pm.synthesize(sr=sr)
        # Match lengths
        min_len = min(len(y_orig), len(y_synth))
        if min_len < sr * 1.0:  # less than 1 second — can't verify
            return {"score": 0.5, "pitch_score": 0.5, "timing_score": 0.5,
                    "details": "audio too short for verification"}
        y_orig = y_orig[:min_len]
        y_synth = y_synth[:min_len]

        # Pitch score: chroma correlation
        chroma_orig = librosa.feature.chroma_cqt(y=y_orig, sr=sr)
        chroma_synth = librosa.feature.chroma_cqt(y=y_synth, sr=sr)
        # Align frame counts
        fc = min(chroma_orig.shape[1], chroma_synth.shape[1])
        if fc < 10:
            pitch_score = 0.5
        else:
            c1 = chroma_orig[:, :fc].flatten()
            c2 = chroma_synth[:, :fc].flatten()
            corr = np.corrcoef(c1, c2)[0, 1]
            pitch_score = float(max(0, corr)) if not np.isnan(corr) else 0.5

        # Timing score: onset envelope correlation
        onset_orig = librosa.onset.onset_strength(y=y_orig, sr=sr)
        onset_synth = librosa.onset.onset_strength(y=y_synth, sr=sr)
        fo = min(len(onset_orig), len(onset_synth))
        if fo < 10:
            timing_score = 0.5
        else:
            o1, o2 = onset_orig[:fo], onset_synth[:fo]
            corr = np.corrcoef(o1, o2)[0, 1]
            timing_score = float(max(0, corr)) if not np.isnan(corr) else 0.5

        # Combined (pitch weighted higher for melodic, timing for drums)
        score = 0.6 * pitch_score + 0.4 * timing_score

        return {
            "score": round(score, 3),
            "pitch_score": round(pitch_score, 3),
            "timing_score": round(timing_score, 3),
            "details": f"pitch={pitch_score:.2f} timing={timing_score:.2f}",
        }
    except Exception as e:
        return {"score": 0.0, "pitch_score": 0.0, "timing_score": 0.0,
                "details": f"verification error: {e}"}


# ─── CORE PIPELINE ───────────────────────────────────────────────────────────

def _set_stage(job_id: str, stage: str):
    JOBS[job_id]["stage"] = stage


def separate_stems(audio_path: str, out_dir: str, job_id: str) -> dict[str, str]:
    _set_stage(job_id, "Separating stems via Demucs…")
    import demucs.separate
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


def transcribe_with_verification(stem_name: str, wav_path: str, out_dir: str,
                                  bp_model, job_id: str) -> tuple[str, dict, dict]:
    """
    Transcribe a stem with analysis + verification loop.
    Returns (midi_path, analysis, verification).
    """
    from basic_pitch.inference import predict

    # 1. Analyze
    _set_stage(job_id, f"Analyzing {stem_name}…")
    analysis = analyze_stem(wav_path, stem_name)

    # 2. Transcribe + verify loop
    best_midi, best_verify, best_score = None, None, -1
    base_params = STEM_SETTINGS[stem_name]

    for attempt in range(MAX_RETRIES):
        _set_stage(job_id, f"Transcribing {stem_name} (attempt {attempt + 1})…")
        params = adapt_params(base_params, analysis, stem_name, attempt)

        midi_out = Path(out_dir) / f"{stem_name}{'_retry'+str(attempt) if attempt else ''}.mid"
        _, midi_data, _ = predict(wav_path, bp_model, **params)
        midi_data.write(str(midi_out))

        # 3. Verify
        _set_stage(job_id, f"Verifying {stem_name} MIDI vs stem audio…")
        verify = verify_midi(str(midi_out), wav_path)

        if verify["score"] > best_score:
            best_score = verify["score"]
            best_midi, best_verify = str(midi_out), verify

        if verify["score"] >= VERIFY_THRESHOLD:
            break

    # Rename best to canonical name
    final_path = Path(out_dir) / f"{stem_name}.mid"
    if best_midi != str(final_path):
        shutil.copy(best_midi, final_path)

    return str(final_path), analysis, best_verify


def bundle_midis(midi_paths: dict[str, str], out_dir: str, track_name: str) -> str:
    zip_path = Path(out_dir) / f"{track_name}_midi.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem, path in midi_paths.items():
            zf.write(path, arcname=f"{stem}.mid")
    return str(zip_path)


def _run_pipeline(job_id: str, audio_path: str):
    job = JOBS[job_id]
    work_dir = job["work_dir"]

    try:
        import torch
        track_name = Path(audio_path).stem

        # 1. Separate
        stem_dir = os.path.join(work_dir, "stems")
        os.makedirs(stem_dir, exist_ok=True)
        stem_paths = separate_stems(audio_path, stem_dir, job_id)

        # 2. Load model
        _set_stage(job_id, "Loading Basic-Pitch model…")
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        bp_model = Model(ICASSP_2022_MODEL_PATH)

        # 3. Analyze + transcribe + verify per stem
        midi_dir = os.path.join(work_dir, "midi")
        os.makedirs(midi_dir, exist_ok=True)
        midi_paths, analyses, verifications = {}, {}, {}

        for stem in STEMS:
            midi_p, analysis, verify = transcribe_with_verification(
                stem, stem_paths[stem], midi_dir, bp_model, job_id)
            midi_paths[stem] = midi_p
            analyses[stem] = analysis
            verifications[stem] = verify

        # 4. Cleanup
        del bp_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5. Verification report
        report_lines = [f"# Verification Report: {track_name}", ""]
        for stem in STEMS:
            a, v = analyses[stem], verifications[stem]
            report_lines += [
                f"## {stem}",
                f"- BPM: {a['bpm']} | Key: {a['key']} | Energy: {a['energy_db']}dB",
                f"- Pitch range: {a['pitch_min_hz']}-{a['pitch_max_hz']}Hz | Onsets/sec: {a['onset_density']}",
                f"- **Verification score: {v['score']}** ({v['details']})",
                f"- {'✓ PASS' if v['score'] >= VERIFY_THRESHOLD else '⚠ LOW — review manually'}",
                "",
            ]
        report_path = os.path.join(work_dir, "verification_report.md")
        Path(report_path).write_text("\n".join(report_lines))

        # 6. Bundle
        _set_stage(job_id, "Packaging ZIP…")
        zip_path = bundle_midis(midi_paths, work_dir, track_name)

        job["result"] = {
            **midi_paths, "zip": zip_path, "report": report_path,
            "track_name": track_name,
            "analyses": analyses, "verifications": verifications,
        }
        job["status"] = "done"
        job["stage"] = "✓ Done"

    except Exception as e:
        import traceback
        job["status"] = "error"
        job["stage"] = "Error"
        job["error"] = f"{e}\n{traceback.format_exc()[-500:]}"
        shutil.rmtree(work_dir, ignore_errors=True)


# ─── GRADIO UI ───────────────────────────────────────────────────────────────

CSS = """
body { font-family: 'IBM Plex Mono', monospace; background: #0a0a0a; color: #e0e0e0; }
.gradio-container { max-width: 900px; margin: 0 auto; }
h1 { font-size: 2rem; letter-spacing: 0.08em; color: #f5f5f5; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
.status-box { background: #111; border: 1px solid #2a2a2a; border-radius: 4px; padding: 0.75rem 1rem; font-size: 0.85rem; color: #8aff8a; }
.verify-pass { color: #8aff8a; } .verify-warn { color: #ffcc44; }
footer { display: none !important; }
"""

def submit_job(audio_file):
    if audio_file is None:
        raise gr.Error("Upload an audio file first.")
    job_id = str(uuid.uuid4())
    work_dir = tempfile.mkdtemp(prefix="song2midi_")
    JOBS[job_id] = {"status": "running", "stage": "Queued…",
                     "result": None, "error": None, "work_dir": work_dir}
    thread = threading.Thread(target=_run_pipeline, args=(job_id, audio_file), daemon=True)
    thread.start()
    return (job_id, "⏳ Processing in background — analysis → transcription → verification.",
            gr.update(interactive=False))


def poll_job(job_id):
    empty = (gr.update(), gr.update(), gr.update(), gr.update(),
             gr.update(), gr.update(), gr.update())
    if not job_id:
        return ("", *empty)
    job = JOBS.get(job_id)
    if job is None:
        return ("⚠ Job not found.", *empty[:6], gr.update(interactive=True))
    if job["status"] == "running":
        return (f"⏳ {job['stage']}", *empty[:6], gr.update(interactive=False))
    if job["status"] == "error":
        return (f"❌ Error: {job['error']}", *empty[:6], gr.update(interactive=True))

    r = job["result"]
    # Build verification summary
    lines = []
    for stem in STEMS:
        v = r["verifications"][stem]
        a = r["analyses"][stem]
        icon = "✓" if v["score"] >= VERIFY_THRESHOLD else "⚠"
        lines.append(f"{icon} **{stem}**: score {v['score']} ({v['details']}) — {a['bpm']} BPM, {a['key']}")
    summary = "\n".join(lines)

    return (
        f"✓ **{r['track_name']}** complete\n\n{summary}",
        r["vocals"], r["drums"], r["bass"], r["other"],
        r["zip"], r["report"],
        gr.update(interactive=True),
    )


with gr.Blocks(css=CSS, title="SongToMIDI") as demo:
    gr.Markdown("""
# SongToMIDI
**Upload a full song → get verified MIDI files per stem.**

Pipeline: `Demucs` separation → per-stem **analysis** (BPM, key, pitch range)
→ `Basic-Pitch` transcription (adaptive params) → **audio verification**
(MIDI synthesized back and compared to stem — retries if score < 0.65)

Stems: `vocals` · `drums` · `bass` · `other`
""")
    job_state = gr.State(value=None)
    with gr.Row():
        audio_input = gr.Audio(label="Input Audio", type="filepath", sources=["upload"])
    run_btn = gr.Button("▶  Transcribe + Verify", variant="primary", size="lg")
    status = gr.Markdown(value="", elem_classes=["status-box"])
    gr.Markdown("### Output MIDI Files")
    with gr.Row():
        out_vocals = gr.File(label="vocals.mid", interactive=False)
        out_drums = gr.File(label="drums.mid", interactive=False)
    with gr.Row():
        out_bass = gr.File(label="bass.mid", interactive=False)
        out_other = gr.File(label="other.mid", interactive=False)
    with gr.Row():
        out_zip = gr.File(label="📦 All stems (ZIP)", interactive=False)
        out_report = gr.File(label="📋 Verification report", interactive=False)

    timer = gr.Timer(value=POLL_INTERVAL, active=False)
    run_btn.click(fn=submit_job, inputs=[audio_input],
                  outputs=[job_state, status, run_btn]
    ).then(fn=lambda: gr.Timer(active=True), outputs=[timer])
    timer.tick(fn=poll_job, inputs=[job_state],
               outputs=[status, out_vocals, out_drums, out_bass, out_other,
                        out_zip, out_report, run_btn])

    gr.Markdown("""
---
**How verification works:** Each transcribed MIDI is synthesized back to audio,
then compared against the original stem using chroma correlation (pitch) and
onset envelope correlation (timing). If the combined score is below 0.65,
transcription retries with more sensitive parameters (up to 3 attempts).
The best-scoring MIDI is kept.

Built with [Demucs](https://github.com/facebookresearch/demucs) +
[Basic-Pitch](https://github.com/spotify/basic-pitch) + [librosa](https://librosa.org)
""")

if __name__ == "__main__":
    demo.launch()
