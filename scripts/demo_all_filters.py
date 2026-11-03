"""
Multi-Filter Denoising Showcase Script
Demonstrates noisy audio generation and runs all 6 DSP & ML filtering topologies
to produce clean unnoised audio files for side-by-side comparison.

Outputs:
  outputs/denoised_showcase/
    ├── 00_clean_reference.wav
    ├── 01_noisy_input.wav
    ├── 02_unnoised_fir_bandpass.wav
    ├── 03_unnoised_butterworth_highpass.wav
    ├── 04_unnoised_harmonic_notch.wav
    ├── 05_unnoised_spectral_gating.wav
    ├── 06_unnoised_wiener_filter.wav
    ├── 07_unnoised_ml_adaptive.wav
    └── comparison_report.json
"""

import os
import sys
import argparse
import time
import json
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.audio_io import load_audio, save_audio, generate_clean_speech_synthetic
from src.noise_generation import (
    generate_electrical_hum,
    generate_low_frequency_rumble,
    generate_high_frequency_hiss,
    generate_broadband_noise,
    generate_mixed_noise,
    mix_speech_and_noise
)
import src.filters as dsp_filters
import src.spectral_denoising as spec_denoising
from src.ml_model import NoiseClassifier
from src.evaluation import compute_reference_based_metrics, compute_unpaired_metrics


def run_all_filters_showcase(
    sample_wav: str = None,
    noisy_wav: str = None,
    noise_type: str = "hum",
    snr_db: float = 5.0,
    output_dir: str = "outputs/denoised_showcase",
    sr: int = 16000
):
    print("=" * 78)
    print("        SonicPure DSP - Multi-Filter Unnoising Showcase")
    print("=" * 78)

    os.makedirs(output_dir, exist_ok=True)

    clean_audio = None
    if noisy_wav and os.path.exists(noisy_wav):
        print(f"\n[1/3] Loading input noisy audio: {noisy_wav}")
        noisy_audio, _ = load_audio(noisy_wav, target_sr=sr)
        duration_s = len(noisy_audio) / sr
        print(f"      Duration: {duration_s:.2f}s ({len(noisy_audio)} samples @ {sr} Hz)")

        if sample_wav and os.path.exists(sample_wav):
            clean_audio, _ = load_audio(sample_wav, target_sr=sr)
            save_audio(os.path.join(output_dir, "00_clean_reference.wav"), clean_audio, sr)

        save_audio(os.path.join(output_dir, "01_noisy_input.wav"), noisy_audio, sr)
    else:
        # 1. Load or synthesize clean audio
        if sample_wav and os.path.exists(sample_wav):
            print(f"\n[1/4] Loading clean reference: {sample_wav}")
            clean_audio, _ = load_audio(sample_wav, target_sr=sr)
        else:
            default_sample = os.path.join(PROJECT_ROOT, "data", "samples", "clean_male_sentence_01.wav")
            if not os.path.exists(default_sample):
                default_sample = os.path.join(PROJECT_ROOT, "data", "samples", "clean_male_speech_01.wav")
            if os.path.exists(default_sample):
                print(f"\n[1/4] Loading bundled clean sample: {default_sample}")
                clean_audio, _ = load_audio(default_sample, target_sr=sr)
            else:
                print("\n[1/4] Synthesizing academic clean formant speech...")
                clean_audio = generate_clean_speech_synthetic(duration=4.5, sr=sr)

        n_samples = len(clean_audio)
        duration_s = n_samples / sr
        print(f"      Duration: {duration_s:.2f}s ({n_samples} samples @ {sr} Hz)")

        # 2. Synthesize calibrated noise and mix
        print(f"\n[2/4] Injecting calibrated '{noise_type}' noise at target SNR = {snr_db:.1f} dB...")
        if noise_type == "hum" or "50" in noise_type:
            noise = generate_electrical_hum(n_samples, sr=sr, base_freq=50.0)
        elif "60" in noise_type:
            noise = generate_electrical_hum(n_samples, sr=sr, base_freq=60.0)
        elif noise_type == "rumble":
            noise = generate_low_frequency_rumble(n_samples, sr=sr, cutoff_hz=130.0)
        elif noise_type == "hiss":
            noise = generate_high_frequency_hiss(n_samples, sr=sr, cutoff_hz=3600.0)
        elif noise_type == "pink" or noise_type == "broadband":
            noise = generate_broadband_noise(n_samples, sr=sr, noise_type="pink")
        else:
            noise = generate_mixed_noise(n_samples, sr=sr)

        noisy_audio, _, noise_meta = mix_speech_and_noise(clean_audio, noise, target_snr_db=snr_db)
        print(f"      Measured Initial SNR: {noise_meta['measured_snr_db']:.2f} dB")

        # Save reference and noisy audio
        clean_path = os.path.join(output_dir, "00_clean_reference.wav")
        noisy_path = os.path.join(output_dir, "01_noisy_input.wav")
        save_audio(clean_path, clean_audio, sr)
        save_audio(noisy_path, noisy_audio, sr)

    # 3. Load ML model for adaptive strategy
    classifier = NoiseClassifier(random_state=42)
    model_path = os.path.join(PROJECT_ROOT, "models", "noise_classifier.joblib")
    if os.path.exists(model_path):
        classifier.load(model_path)

    # 4. Execute all filtering algorithms
    print("\n[3/4] Processing noisy audio through ALL filtering topologies...")

    filters_config = [
        {
            "id": "fir_bandpass",
            "file": "02_unnoised_fir_bandpass.wav",
            "name": "FIR Bandpass (300-3400 Hz)",
            "fn": lambda sig: dsp_filters.apply_fir_filter(
                sig, cutoff=(300.0, 3400.0), num_taps=101, btype="bandpass", sr=sr, zero_phase=True
            )[0]
        },
        {
            "id": "butterworth_highpass",
            "file": "03_unnoised_butterworth_highpass.wav",
            "name": "Butterworth Highpass (150 Hz SOS)",
            "fn": lambda sig: dsp_filters.apply_butterworth_filter(
                sig, cutoff=150.0, order=4, btype="highpass", sr=sr, zero_phase=True
            )[0]
        },
        {
            "id": "harmonic_notch",
            "file": "04_unnoised_harmonic_notch.wav",
            "name": "Harmonic Notch Filter (50 Hz + Harmonics)",
            "fn": lambda sig: dsp_filters.apply_notch_filter(
                sig, notch_freq=50.0, q=35.0, include_harmonics=True, max_harmonics=4, sr=sr, zero_phase=True
            )[0]
        },
        {
            "id": "spectral_gating",
            "file": "05_unnoised_spectral_gating.wav",
            "name": "STFT Spectral Gating (2D Smoothed)",
            "fn": lambda sig: spec_denoising.spectral_gating(
                sig, sr=sr, suppression_db=-18.0, over_subtraction=1.5, time_smoothing=2, freq_smoothing=2
            )[0]
        },
        {
            "id": "wiener_filter",
            "file": "06_unnoised_wiener_filter.wav",
            "name": "Audio Wiener Filter (Ephraim-Malah DD)",
            "fn": lambda sig: spec_denoising.wiener_filter_audio(
                sig, sr=sr, alpha=0.95, overestimation=1.25, gain_floor_db=-18.0
            )[0]
        },
        {
            "id": "ml_adaptive",
            "file": "07_unnoised_ml_adaptive.wav",
            "name": "ML-Assisted Content-Adaptive Filter",
            "fn": lambda sig: classifier.apply_adaptive_filter(sig, sr=sr)[0]
        }
    ]

    results = []

    # Baseline evaluation
    if clean_audio is not None:
        base_metrics = compute_reference_based_metrics(
            clean_ref=clean_audio,
            noisy_input=noisy_audio,
            processed=noisy_audio,
            sr=sr,
            processing_time_sec=0.0
        )
    else:
        base_metrics = compute_unpaired_metrics(
            noisy_input=noisy_audio,
            processed=noisy_audio,
            sr=sr,
            processing_time_sec=0.0
        )
        base_metrics["snr_improvement_db"] = 0.0
        base_metrics["output_snr_db"] = 0.0
    base_metrics["filter_name"] = "0. Noisy Audio (Unfiltered)"
    base_metrics["file_saved"] = "01_noisy_input.wav"
    results.append(base_metrics)

    for cfg in filters_config:
        t0 = time.perf_counter()
        unnoised_sig = cfg["fn"](noisy_audio)
        t_sec = time.perf_counter() - t0

        out_file_path = os.path.join(output_dir, cfg["file"])
        save_audio(out_file_path, unnoised_sig, sr)

        if clean_audio is not None:
            metrics = compute_reference_based_metrics(
                clean_ref=clean_audio,
                noisy_input=noisy_audio,
                processed=unnoised_sig,
                sr=sr,
                processing_time_sec=t_sec
            )
            delta_disp = f"Delta-SNR: {metrics['snr_improvement_db']:+.2f} dB"
        else:
            metrics = compute_unpaired_metrics(
                noisy_input=noisy_audio,
                processed=unnoised_sig,
                sr=sr,
                processing_time_sec=t_sec
            )
            metrics["snr_improvement_db"] = metrics.get("total_attenuation_db", 0.0)
            metrics["output_snr_db"] = metrics.get("output_rms_dbfs", 0.0)
            delta_disp = f"Attenuation: {metrics.get('total_attenuation_db', 0.0):.2f} dB"

        metrics["filter_name"] = cfg["name"]
        metrics["file_saved"] = cfg["file"]
        metrics["exec_time_ms"] = round(t_sec * 1000, 2)
        results.append(metrics)
        print(f"  [+] {cfg['name']:<42} -> Saved {cfg['file']} ({delta_disp}, Time: {metrics['exec_time_ms']} ms)")

    # Save summary report
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    # 5. Display formatted comparison table
    print("\n" + "=" * 78)
    print(f"{'Filter Topology':<38} | {'Delta-SNR':<10} | {'Output SNR':<11} | {'RMSE':<8} | {'RTF':<8}")
    print("-" * 78)

    for r in results:
        delta_str = f"{r['snr_improvement_db']:+.2f} dB" if 'snr_improvement_db' in r else "0.00 dB"
        out_snr = f"{r['output_snr_db']:.2f} dB" if 'output_snr_db' in r else f"{snr_db:.2f} dB"
        rmse_str = f"{r['rmse']:.4f}" if 'rmse' in r else "0.0000"
        rtf_str = f"{r.get('real_time_factor', 0.0):.3f}x"
        print(f"{r['filter_name']:<38} | {delta_str:<10} | {out_snr:<11} | {rmse_str:<8} | {rtf_str:<8}")

    print("=" * 78)
    print(f"\n[4/4] All unnoised audio files successfully generated in:")
    print(f"      Folder: {os.path.abspath(output_dir)}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Filter Audio Denoising Showcase")
    parser.add_argument("--noisy_wav", type=str, default=None, help="Path to input noisy WAV audio file to unnoise directly")
    parser.add_argument("--sample_wav", type=str, default=None, help="Path to optional clean WAV reference (for ground-truth SNR metrics)")
    parser.add_argument("--noise_type", type=str, default="hum", choices=["hum", "rumble", "hiss", "pink", "mixed"])
    parser.add_argument("--snr", type=float, default=5.0, help="Target Signal-to-Noise Ratio (dB) when mixing noise")
    parser.add_argument("--output_dir", type=str, default="outputs/denoised_showcase", help="Output directory")
    args = parser.parse_args()

    run_all_filters_showcase(
        sample_wav=args.sample_wav,
        noisy_wav=args.noisy_wav,
        noise_type=args.noise_type,
        snr_db=args.snr,
        output_dir=args.output_dir
    )
