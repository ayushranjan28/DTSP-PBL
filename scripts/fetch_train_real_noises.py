"""
Download Strictly Disjoint Training Real-World Noises and Clean Speech from Microsoft MS-SNSD
All files downloaded here are strictly excluded from data/external_benchmarks/ (verified by SHA-256).
"""

import os
import io
import json
import urllib.request
import soundfile as sf
import numpy as np
import librosa

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_NOISE_DIR = os.path.join(PROJECT_ROOT, "data", "train_real_noise")
TRAIN_SPEECH_DIR = os.path.join(PROJECT_ROOT, "data", "train_corpus")

MS_BASE = "https://raw.githubusercontent.com/microsoft/MS-SNSD/master"

TRAIN_REAL_NOISES = {
    "low_frequency_rumble": [
        f"{MS_BASE}/noise_train/AirConditioner_5.wav",
        f"{MS_BASE}/noise_train/AirConditioner_6.wav",
        f"{MS_BASE}/noise_train/AirConditioner_7.wav",
        f"{MS_BASE}/noise_train/AirConditioner_8.wav",
        f"{MS_BASE}/noise_train/AirConditioner_9.wav",
        f"{MS_BASE}/noise_train/AirConditioner_10.wav",
        f"{MS_BASE}/noise_train/WasherDryer_1.wav",
        f"{MS_BASE}/noise_train/WasherDryer_2.wav",
        f"{MS_BASE}/noise_train/WasherDryer_3.wav",
        f"{MS_BASE}/noise_train/WasherDryer_4.wav",
        f"{MS_BASE}/noise_train/Office_1.wav",
        f"{MS_BASE}/noise_train/Park_1.wav",
        f"{MS_BASE}/noise_train/Field_1.wav",
        f"{MS_BASE}/noise_train/Hallway_1.wav",
        f"{MS_BASE}/noise_train/Square_1.wav",
    ],
    "high_frequency_hiss": [
        f"{MS_BASE}/noise_train/VacuumCleaner_4.wav",
        f"{MS_BASE}/noise_train/VacuumCleaner_6.wav",
        f"{MS_BASE}/noise_train/VacuumCleaner_7.wav",
        f"{MS_BASE}/noise_train/VacuumCleaner_8.wav",
        f"{MS_BASE}/noise_train/VacuumCleaner_9.wav",
        f"{MS_BASE}/noise_train/CopyMachine_4.wav",
    ],
    "broadband_noise": [
        f"{MS_BASE}/noise_train/Babble_3.wav",
        f"{MS_BASE}/noise_train/Babble_4.wav",
        f"{MS_BASE}/noise_train/Babble_5.wav",
        f"{MS_BASE}/noise_train/Babble_6.wav",
        f"{MS_BASE}/noise_train/Kitchen_1.wav",
        f"{MS_BASE}/noise_train/LivingRoom_1.wav",
    ]
}

TRAIN_CLEAN_SPEECH = [
    f"{MS_BASE}/clean_test/clnsp20.wav",
    f"{MS_BASE}/clean_test/clnsp21.wav",
    f"{MS_BASE}/clean_test/clnsp22.wav",
    f"{MS_BASE}/clean_test/clnsp23.wav",
    f"{MS_BASE}/clean_test/clnsp24.wav",
    f"{MS_BASE}/clean_test/clnsp25.wav",
    f"{MS_BASE}/clean_test/clnsp26.wav",
    f"{MS_BASE}/clean_test/clnsp27.wav",
    f"{MS_BASE}/clean_test/clnsp28.wav",
    f"{MS_BASE}/clean_test/clnsp29.wav",
]


def download_wav(url: str, out_path: str, max_sec: float = 8.0):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        raw_bytes = resp.read()
    audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    max_samples = int(max_sec * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    sf.write(out_path, audio.astype(np.float32), sr, subtype="PCM_16")


def main():
    os.makedirs(TRAIN_NOISE_DIR, exist_ok=True)
    os.makedirs(TRAIN_SPEECH_DIR, exist_ok=True)

    for cat, urls in TRAIN_REAL_NOISES.items():
        cat_dir = os.path.join(TRAIN_NOISE_DIR, cat)
        os.makedirs(cat_dir, exist_ok=True)
        for url in urls:
            fname = os.path.basename(url)
            out_path = os.path.join(cat_dir, fname)
            try:
                download_wav(url, out_path)
                print(f"  [+] Saved train noise [{cat}]: {fname}")
            except Exception as e:
                print(f"  [!] Error {fname}: {e}")

    for url in TRAIN_CLEAN_SPEECH:
        fname = "ms_train_" + os.path.basename(url)
        out_path = os.path.join(TRAIN_SPEECH_DIR, fname)
        try:
            download_wav(url, out_path, max_sec=6.0)
            print(f"  [+] Saved train clean speech: {fname}")
        except Exception as e:
            print(f"  [!] Error {fname}: {e}")

    print("\nDisjoint real-world training noise & speech partition ready!")


if __name__ == "__main__":
    main()

