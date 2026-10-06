---
title: SongToMIDI
emoji: 🎵
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# SongToMIDI

Converts a full mixed song into **4 individual MIDI files** — one per stem — with **audio verification**.

## Pipeline

```
Input Audio (mp3/wav/flac)
    │
    ▼
Demucs htdemucs (stem separation)
    ├── vocals.wav
    ├── drums.wav
    ├── bass.wav
    └── other.wav
         │
         ▼ (per-stem)
    ┌─────────────┐
    │   ANALYZE   │  BPM, key, pitch range, energy, onset density
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ TRANSCRIBE  │  Basic-Pitch with adaptive parameters
    └──────┬──────┘  (tuned from analysis)
           ▼
    ┌─────────────┐
    │   VERIFY    │  Synthesize MIDI → compare chroma + onsets
    └──────┬──────┘  vs original stem audio
           │
     score < 0.65? ──→ retry with more sensitive params (max 3)
           │
           ▼
    Best-scoring MIDI kept + verification report
```

## Verification

Each MIDI is synthesized back to audio and compared against its source stem:
- **Pitch score**: chroma feature correlation (0-1)
- **Timing score**: onset envelope correlation (0-1)
- **Combined**: 0.6×pitch + 0.4×timing

If below 0.65, transcription retries with adjusted thresholds. The UI shows
per-stem scores and a downloadable verification report.

## Outputs

- `vocals.mid`, `drums.mid`, `bass.mid`, `other.mid`
- `{track}_midi.zip` — all stems bundled
- `verification_report.md` — per-stem analysis + scores
