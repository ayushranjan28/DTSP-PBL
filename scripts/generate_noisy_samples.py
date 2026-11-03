"""
Generate pre-mixed, calibrated noisy demonstration audio files.
Creates ready-to-listen WAV files in data/noisy_samples/ covering various real-world
noise scenarios (hum, rumble, hiss, pink noise, mixed noise) at standard SNRs.
"""

import os
import sys
import json
import soundfile as sf
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.audio_io import load_audio
from src.noise_generation import (
    generate_electrical_hum,
    generate_low_frequency_rumble,
    generate_high_frequency_hiss,
    generate_broadband_noise,
    generate_mixed_noise,
    mix_speech_and_noise
)

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "noisy_samples")

# Selected sentences for demo suite
SENTENCE_TARGETS = [
    {
        "clean_id": "clean_female_sentence_01.wav",
        "prefix": "female01",
        "sentence": "For the twentieth time that evening the two men shook hands.",
        "speaker": "Female (slt)"
    },
    {
        "clean_id": "clean_female_sentence_02.wav",
        "prefix": "female02",
        "sentence": "He turned sharply, and faced Gregson across the table.",
        "speaker": "Female (slt)"
    },
    {
        "clean_id": "clean_male_sentence_01.wav",
        "prefix": "male01",
        "sentence": "For the twentieth time that evening the two men shook hands.",
        "speaker": "Male (bdl)"
    },
    {
        "clean_id": "clean_male_sentence_02.wav",
        "prefix": "male02",
        "sentence": "He turned sharply, and faced Gregson across the table.",
        "speaker": "Male (bdl)"
    }
]

NOISE_PROFILES = [
    {
        "suffix": "50hz_mains_hum_5db",
        "label": "50 Hz Power-line Hum",
        "snr_db": 5.0,
        "gen_fn": lambda n, sr, seed: generate_electrical_hum(n, sr=sr, base_freq=50.0, seed=seed),
        "recommended_filter": "Harmonic Notch Filter (50 Hz + Harmonics)",
        "desc": "Ground-loop electrical hum with 50 Hz fundamental and odd harmonics (150, 250, 350 Hz)."
    },
    {
        "suffix": "hvac_rumble_3db",
        "label": "HVAC Low-Frequency Rumble",
        "snr_db": 3.0,
        "gen_fn": lambda n, sr, seed: generate_low_frequency_rumble(n, sr=sr, cutoff_hz=120.0, seed=seed),
        "recommended_filter": "Butterworth Highpass (130-150 Hz SOS)",
        "desc": "Acoustic rumble from air conditioning, wind buffeting, and heavy traffic vibrations."
    },
    {
        "suffix": "tape_hiss_6db",
        "label": "Pre-amp & Tape Hiss",
        "snr_db": 6.0,
        "gen_fn": lambda n, sr, seed: generate_high_frequency_hiss(n, sr=sr, cutoff_hz=3500.0, seed=seed),
        "recommended_filter": "FIR Lowpass (3800 Hz) or STFT Spectral Gating",
        "desc": "High-frequency thermal noise and analogue tape flutter concentrated above 3.5 kHz."
    },
    {
        "suffix": "ambient_pink_5db",
        "label": "Ambient Pink Noise",
        "snr_db": 5.0,
        "gen_fn": lambda n, sr, seed: generate_broadband_noise(n, sr=sr, noise_type="pink", seed=seed),
        "recommended_filter": "Audio Wiener Filter (Ephraim-Malah DD)",
        "desc": "1/f spectral noise modeling crowd chatter, rainfall, and room reverberation floor."
    },
    {
        "suffix": "harsh_mixed_0db",
        "label": "Severe Mixed Noise (Challenging)",
        "snr_db": 0.0,
        "gen_fn": lambda n, sr, seed: generate_mixed_noise(n, sr=sr, seed=seed),
        "recommended_filter": "ML-Assisted Multi-stage Pipeline (Notch + Wiener)",
        "desc": "Harsh 0 dB SNR corruption combining simultaneous 50 Hz hum, low rumble, and broadband hiss."
    }
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    sr = 16000

    print("==================================================================")
    print("      SonicPure DSP - Pre-Mixed Noisy Sample Generator            ")
    print("==================================================================")

    seed = 100
    for s_info in SENTENCE_TARGETS:
        clean_path = os.path.join(SAMPLES_DIR, s_info["clean_id"])
        if not os.path.exists(clean_path):
            print(f"Warning: Clean audio {clean_path} not found. Skipping.")
            continue

        clean_audio, _ = load_audio(clean_path, target_sr=sr)
        n_samples = len(clean_audio)
        dur = n_samples / sr

        print(f"\nProcessing {s_info['speaker']} ({dur:.2f}s): \"{s_info['sentence']}\"")

        for p_info in NOISE_PROFILES:
            seed += 7
            noise = p_info["gen_fn"](n_samples, sr, seed)
            noisy_audio, _, meta = mix_speech_and_noise(clean_audio, noise, target_snr_db=p_info["snr_db"])

            # Peak normalize with headroom
            peak = np.max(np.abs(noisy_audio))
            if peak > 0.95:
                noisy_audio = (noisy_audio / peak) * 0.95

            out_filename = f"{s_info['prefix']}_{p_info['suffix']}.wav"
            out_path = os.path.join(OUT_DIR, out_filename)
            sf.write(out_path, noisy_audio.astype(np.float32), sr, subtype="PCM_16")

            print(f"  [+] Saved {out_filename} -> Measured SNR: {meta['measured_snr_db']:.2f} dB")

            manifest.append({
                "filename": out_filename,
                "clean_reference": s_info["clean_id"],
                "speaker": s_info["speaker"],
                "transcript": s_info["sentence"],
                "noise_type": p_info["label"],
                "target_snr_db": p_info["snr_db"],
                "measured_snr_db": round(meta["measured_snr_db"], 2),
                "recommended_filter": p_info["recommended_filter"],
                "description": p_info["desc"]
            })

    # Save manifest.json
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Generate Markdown documentation for data/noisy_samples/README.md
    readme_path = os.path.join(OUT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Noisy Audio Demonstration Library\n\n")
        f.write("This directory contains pre-mixed, calibrated noisy audio files (16 kHz, 16-bit PCM WAV) ")
        f.write("ready for direct offline listening, A/B comparisons, and DSP filter evaluation.\n\n")
        f.write("## Catalog\n\n")
        f.write("| Filename | Speaker | Spoken Sentence | Noise Condition | Target SNR | Recommended Filter |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for item in manifest:
            f.write(f"| `{item['filename']}` | {item['speaker']} | \"{item['transcript']}\" | {item['noise_type']} | {item['target_snr_db']} dB | {item['recommended_filter']} |\n")

    print("\n==================================================================")
    print(f"Successfully generated {len(manifest)} noisy WAV files in:")
    print(f"Folder: {OUT_DIR}")
    print("==================================================================")


if __name__ == "__main__":
    main()

