"""
Fetch dedicated training speech corpus from CMU ARCTIC (sentences a0020 - a0045).
Keeps data/samples/ clean sentences 01-04 completely held-out for testing.
"""

import os
import io
import urllib.request
import tarfile
import soundfile as sf
import numpy as np
import librosa

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train_corpus")


def fetch_speaker_sentences(speaker: str, start_idx: int = 20, end_idx: int = 42):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    target_names = {f"arctic_a{i:04d}.wav" for i in range(start_idx, end_idx + 1)}
    
    url = f"http://festvox.org/cmu_arctic/packed/cmu_us_{speaker}_arctic.tar.bz2"
    print(f"Fetching {len(target_names)} sentences for speaker '{speaker}' from {url}...")
    
    saved = 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        with tarfile.open(fileobj=resp, mode="r|bz2") as tar:
            for member in tar:
                base = os.path.basename(member.name)
                if base in target_names:
                    ext = tar.extractfile(member)
                    audio_bytes = ext.read()
                    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    if sr != 16000:
                        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                        sr = 16000

                    # Peak normalize with safe headroom
                    peak = np.max(np.abs(data))
                    if peak > 0:
                        data = (data / peak) * 0.92

                    out_path = os.path.join(TRAIN_DIR, f"{speaker}_{base}")
                    sf.write(out_path, data.astype(np.float32), sr, subtype="PCM_16")
                    target_names.remove(base)
                    saved += 1
                    if not target_names:
                        break

    print(f"Saved {saved} training files for {speaker}.")


def main():
    # Fetch 20 female sentences (slt)
    fetch_speaker_sentences("slt", start_idx=20, end_idx=40)
    # Fetch 20 male sentences (bdl)
    fetch_speaker_sentences("bdl", start_idx=20, end_idx=40)
    print("\nTraining corpus download complete!")


if __name__ == "__main__":
    main()

