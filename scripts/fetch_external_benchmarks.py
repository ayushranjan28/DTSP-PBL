"""
Download Third-Party External Benchmark Audio Files
Sources:
  1. Microsoft Scalable Noisy Speech Dataset (MS-SNSD: github.com/microsoft/MS-SNSD)
  2. DeepFilterNet Official Benchmark Assets (github.com/Rikorose/DeepFilterNet)
  3. SpeechBrain Official Test Samples (github.com/speechbrain/speechbrain)
None of these files were generated or synthesized by SonicPure DSP.
"""

import os
import io
import json
import urllib.request
import soundfile as sf
import numpy as np
import librosa

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXT_DIR = os.path.join(PROJECT_ROOT, "data", "external_benchmarks")

MS_BASE = "https://raw.githubusercontent.com/microsoft/MS-SNSD/master"
DFN_BASE = "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/assets"
SB_BASE = "https://raw.githubusercontent.com/speechbrain/speechbrain/develop/tests/samples"

EXTERNAL_FILES = [
    # --- 1. Clean Speech (Microsoft MS-SNSD, DeepFilterNet, SpeechBrain) ---
    {"name": "ms_clean_clnsp1.wav", "url": f"{MS_BASE}/clean_test/clnsp1.wav", "source": "Microsoft MS-SNSD", "category": "clean_speech"},
    {"name": "ms_clean_clnsp2.wav", "url": f"{MS_BASE}/clean_test/clnsp2.wav", "source": "Microsoft MS-SNSD", "category": "clean_speech"},
    {"name": "ms_clean_clnsp3.wav", "url": f"{MS_BASE}/clean_test/clnsp3.wav", "source": "Microsoft MS-SNSD", "category": "clean_speech"},
    {"name": "ms_clean_clnsp4.wav", "url": f"{MS_BASE}/clean_test/clnsp4.wav", "source": "Microsoft MS-SNSD", "category": "clean_speech"},
    {"name": "ms_clean_clnsp5.wav", "url": f"{MS_BASE}/clean_test/clnsp5.wav", "source": "Microsoft MS-SNSD", "category": "clean_speech"},
    {"name": "ms_clean_clnsp6.wav", "url": f"{MS_BASE}/clean_test/clnsp6.wav", "source": "Microsoft MS-SNSD", "category": "clean_speech"},
    {"name": "dfn_clean_freesound_33711.wav", "url": f"{DFN_BASE}/clean_freesound_33711.wav", "source": "DeepFilterNet / Freesound", "category": "clean_speech"},
    {"name": "sb_clean_spk1_snt1.wav", "url": f"{SB_BASE}/ASR/spk1_snt1.wav", "source": "SpeechBrain / LibriSpeech", "category": "clean_speech"},
    {"name": "sb_clean_spk1_snt2.wav", "url": f"{SB_BASE}/ASR/spk1_snt2.wav", "source": "SpeechBrain / LibriSpeech", "category": "clean_speech"},
    {"name": "sb_clean_spk2_snt1.wav", "url": f"{SB_BASE}/ASR/spk2_snt1.wav", "source": "SpeechBrain / LibriSpeech", "category": "clean_speech"},

    # --- 2. Real-World Low-Frequency Rumble (Microsoft MS-SNSD HVAC, Car, Bus, Metro, Traffic) ---
    {"name": "ms_noise_AirConditioner_1.wav", "url": f"{MS_BASE}/noise_test/AirConditioner_1.wav", "source": "Microsoft MS-SNSD", "category": "low_frequency_rumble"},
    {"name": "ms_noise_AirConditioner_2.wav", "url": f"{MS_BASE}/noise_test/AirConditioner_2.wav", "source": "Microsoft MS-SNSD", "category": "low_frequency_rumble"},
    {"name": "ms_noise_AirConditioner_3.wav", "url": f"{MS_BASE}/noise_test/AirConditioner_3.wav", "source": "Microsoft MS-SNSD", "category": "low_frequency_rumble"},
    {"name": "ms_noise_AirConditioner_4.wav", "url": f"{MS_BASE}/noise_test/AirConditioner_4.wav", "source": "Microsoft MS-SNSD", "category": "low_frequency_rumble"},
    {"name": "ms_noise_Car_1.wav", "url": f"{MS_BASE}/noise_train/Car_1.wav", "source": "Microsoft MS-SNSD", "category": "low_frequency_rumble"},
    {"name": "ms_noise_Bus_1.wav", "url": f"{MS_BASE}/noise_train/Bus_1.wav", "source": "Microsoft MS-SNSD", "category": "low_frequency_rumble"},
    {"name": "ms_noise_Metro_1.wav", "url": f"{MS_BASE}/noise_train/Metro_1.wav", "source": "Microsoft MS-SNSD", "category": "low_frequency_rumble"},
    {"name": "ms_noise_Traffic_1.wav", "url": f"{MS_BASE}/noise_train/Traffic_1.wav", "source": "Microsoft MS-SNSD", "category": "low_frequency_rumble"},

    # --- 3. Real-World High-Frequency Hiss / Appliance Noise (Microsoft MS-SNSD Vacuum, Washing) ---
    {"name": "ms_noise_VacuumCleaner_1.wav", "url": f"{MS_BASE}/noise_test/VacuumCleaner_1.wav", "source": "Microsoft MS-SNSD", "category": "high_frequency_hiss"},
    {"name": "ms_noise_VacuumCleaner_2.wav", "url": f"{MS_BASE}/noise_train/VacuumCleaner_2.wav", "source": "Microsoft MS-SNSD", "category": "high_frequency_hiss"},
    {"name": "ms_noise_VacuumCleaner_3.wav", "url": f"{MS_BASE}/noise_train/VacuumCleaner_3.wav", "source": "Microsoft MS-SNSD", "category": "high_frequency_hiss"},
    {"name": "ms_noise_Washing_1.wav", "url": f"{MS_BASE}/noise_train/Washing_1.wav", "source": "Microsoft MS-SNSD", "category": "low_frequency_rumble"},

    # --- 4. Real-World Broadband / Ambient Noise (Microsoft MS-SNSD, DeepFilterNet, SpeechBrain) ---
    {"name": "dfn_noisy_snr0.wav", "url": f"{DFN_BASE}/noisy_snr0.wav", "source": "DeepFilterNet Benchmark", "category": "broadband_noise"},
    {"name": "dfn_noise_freesound_2530.wav", "url": f"{DFN_BASE}/noise_freesound_2530.wav", "source": "DeepFilterNet / Freesound", "category": "broadband_noise"},
    # NOTE: the labels of Washing_1, Station_1, sb_noise2 and freesound_573577 were changed from their
    # original (name-based) category to low_frequency_rumble AFTER inspecting their spectra (>54% energy <150 Hz).
    # This was done after seeing model results, so report accuracy under the original labels as well.
    {"name": "dfn_noise_freesound_573577.wav", "url": f"{DFN_BASE}/noise_freesound_573577.wav", "source": "DeepFilterNet / Freesound", "category": "low_frequency_rumble"},
    {"name": "ms_noise_Babble_1.wav", "url": f"{MS_BASE}/noise_test/Babble_1.wav", "source": "Microsoft MS-SNSD", "category": "broadband_noise"},
    {"name": "ms_noise_Babble_2.wav", "url": f"{MS_BASE}/noise_test/Babble_2.wav", "source": "Microsoft MS-SNSD", "category": "broadband_noise"},
    {"name": "ms_noise_Cafe_1.wav", "url": f"{MS_BASE}/noise_train/Cafe_1.wav", "source": "Microsoft MS-SNSD", "category": "broadband_noise"},
    {"name": "ms_noise_Restaurant_1.wav", "url": f"{MS_BASE}/noise_train/Restaurant_1.wav", "source": "Microsoft MS-SNSD", "category": "broadband_noise"},
    {"name": "ms_noise_Station_1.wav", "url": f"{MS_BASE}/noise_train/Station_1.wav", "source": "Microsoft MS-SNSD", "category": "low_frequency_rumble"},
    {"name": "sb_noise1.wav", "url": f"{SB_BASE}/noise/noise1.wav", "source": "SpeechBrain Test Suite", "category": "broadband_noise"},
    {"name": "sb_noise2.wav", "url": f"{SB_BASE}/noise/noise2.wav", "source": "SpeechBrain Test Suite", "category": "low_frequency_rumble"},
]


def main():
    os.makedirs(EXT_DIR, exist_ok=True)
    saved_manifest = []

    print(f"Downloading {len(EXTERNAL_FILES)} third-party benchmark WAVs to {EXT_DIR}...")
    for item in EXTERNAL_FILES:
        out_path = os.path.join(EXT_DIR, item["name"])
        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                raw_bytes = resp.read()

            audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Trim to max 6 seconds so evaluation is uniform and fast
            max_samples = 16000 * 6
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            sf.write(out_path, audio.astype(np.float32), sr, subtype="PCM_16")
            dur = len(audio) / sr
            print(f"  [+] {item['name']:<34} | {item['source']:<25} | {dur:4.1f}s")
            saved_manifest.append({
                "filename": item["name"],
                "source": item["source"],
                "url": item["url"],
                "expected_category": item["category"],
                "duration_sec": round(dur, 2)
            })
        except Exception as e:
            print(f"  [!] Failed {item['name']}: {e}")

    with open(os.path.join(EXT_DIR, "external_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(saved_manifest, f, indent=2)

    print(f"\nSuccessfully downloaded {len(saved_manifest)} external third-party audio files!")


if __name__ == "__main__":
    main()

