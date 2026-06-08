its not pushing to hugging face 
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

Converts a full mixed song into **4 individual MIDI files** — one per stem.

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
Basic-Pitch MIDI Transcription
(stem-tuned onset/frame/frequency thresholds)
    ├── vocals.mid
    ├── drums.mid
    ├── bass.mid
    └── other.mid
         │
         ▼
ZIP bundle download
```

## Stem-specific tuning

| Stem | Freq Range | Notes |
|------|-----------|-------|
| Vocals | 80–1100 Hz | Melodia trick on; monophonic bias |
| Drums | 30–8000 Hz | Low onset threshold; no melodia |
| Bass | 30–300 Hz | Low-frequency locked |
| Other | 40–4000 Hz | Multi-pitch-bends; chord-aware |

## Hardware

Runs on **T4 GPU** (recommended). CPU fallback works but expect 3–5× longer processing.

## Limitations

- Drum MIDI uses pitched notes — requires General MIDI drum channel remapping in your DAW
- Polyphonic guitar/piano accuracy depends on arrangement density
- Very busy mixes may produce note density artifacts in `other.mid`
- Max recommended file: ~6 minutes / 100MB

## Built with

- [Demucs](https://github.com/facebookresearch/demucs) — htdemucs model
- [Basic-Pitch](https://github.com/spotify/basic-pitch) — Spotify ICASSP 2022 model
