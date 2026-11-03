# AI Agent Guide for SonicPure DSP

> This repository includes an exhaustive, machine-optimized technical reference manual in [README_AI.md](file:///c:/Users/ayush/Desktop/DTSP/README_AI.md).
> Please read [README_AI.md](file:///c:/Users/ayush/Desktop/DTSP/README_AI.md) for full architecture blueprints, mathematical formulas, data contracts, REST API endpoints, state management, and extension patterns.

## Quick Summary for Agents

- **Project**: SonicPure DSP — ML-Assisted Audio Noise Reduction and Signal Processing Workstation (Python + FastAPI + Vanilla JS + MATLAB).
- **Audio Standards**: $16,000\text{ Hz}$ mono, 1D `np.float32` arrays normalized to $[-1.0, 1.0]$.
- **Hardware**: 100% software-based. No microcontrollers or DSP boards.
- **Filtering Invariant**: Always use zero-phase filtering (`filtfilt` / `sosfiltfilt`) to prevent formant phase distortion and temporal delay.
- **Spectral Restoration**: Always use 2D Gaussian/uniform time-frequency smoothing in STFT spectral gating to eliminate musical noise artifacts.
- **ML Pipeline**: Random Forest (100 trees, max depth 12) on 20 acoustic features; 98.08% test accuracy across 5 classes (`electrical_hum`, `low_frequency_rumble`, `high_frequency_hiss`, `broadband_noise`, `clean_speech`).
- **Web Workstation**: FastAPI server in `server.py` serving REST endpoints and the static UI in `web/` at `http://localhost:5000`.
- **Test Command**: `pytest tests/ -v` (all 26 tests passing).
- **Run Command**: `python server.py` or `run.bat`.

Refer to [README_AI.md](file:///c:/Users/ayush/Desktop/DTSP/README_AI.md) for the complete manual.
