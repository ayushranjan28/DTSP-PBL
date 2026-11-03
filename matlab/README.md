# MATLAB DSP & ML Audio Denoising Suite

This directory contains pure MATLAB / GNU Octave implementations of all the classical digital filters, spectral denoising algorithms, acoustic feature extractors, and adaptive ML dispatchers implemented in the Python project.

---

## Files Overview

| Script / Function | Description |
| :--- | :--- |
| [`main_audio_denoising_demo.m`](file:///c:/Users/ayush/Desktop/DTSP/matlab/main_audio_denoising_demo.m) | Main demo script: loads/synthesizes speech, simulates calibrated noise at target SNR, benchmarks all 6 filtering methods, generates waveform/spectrogram plots, and computes objective SNR/MSE metrics. |
| [`apply_fir_filter.m`](file:///c:/Users/ayush/Desktop/DTSP/matlab/apply_fir_filter.m) | Windowed linear-phase FIR filter using `fir1` with causal (`filter`) and zero-phase (`filtfilt`) options. |
| [`apply_butterworth_filter.m`](file:///c:/Users/ayush/Desktop/DTSP/matlab/apply_butterworth_filter.m) | IIR Butterworth filter designed in Second-Order Sections (`sos`) using `butter` and `sosfilt`. |
| [`apply_notch_filter.m`](file:///c:/Users/ayush/Desktop/DTSP/matlab/apply_notch_filter.m) | Cascaded harmonic IIR notch filter using `iirnotch` to eliminate 50 Hz/60 Hz hum and harmonics. |
| [`spectral_gating.m`](file:///c:/Users/ayush/Desktop/DTSP/matlab/spectral_gating.m) | STFT spectral noise gating with over-subtraction, gain flooring, and 2D Gaussian smoothing. |
| [`wiener_filter_audio.m`](file:///c:/Users/ayush/Desktop/DTSP/matlab/wiener_filter_audio.m) | True frequency-domain audio Wiener filter with Ephraim-Malah Decision-Directed a priori SNR tracking. |
| [`extract_audio_features.m`](file:///c:/Users/ayush/Desktop/DTSP/matlab/extract_audio_features.m) | Extracts physical and spectral acoustic features (RMS, ZCR, Centroid, Flatness, Sub-band energies, Hum peaks). |
| [`adaptive_filter_dispatch.m`](file:///c:/Users/ayush/Desktop/DTSP/matlab/adaptive_filter_dispatch.m) | Intelligent classification engine that determines the noise class and dispatches the optimal DSP filter. |

---

## How to Run in MATLAB

1. Open **MATLAB** (compatible with R2020a through R2024b, or GNU Octave).
2. Set your current MATLAB folder to this directory (`DTSP/matlab`).
3. Run the master script in the MATLAB Command Window:
   ```matlab
   main_audio_denoising_demo
   ```
4. The script will:
   - Load or synthesize a clean vocal speech signal.
   - Mix calibrated acoustic noise at your specified SNR level (default: 5 dB).
   - Process the audio through all 6 filtering methods.
   - Display publication-quality figures (waveforms, STFT spectrograms, frequency responses).
   - Print a complete objective evaluation table (Input SNR, Output SNR, $\Delta\text{SNR}$, RMSE, Execution Time).
   - Save processed `.wav` files into `matlab/outputs/`.
