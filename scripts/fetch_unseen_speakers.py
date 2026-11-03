"""
Fetch completely unseen speakers from CMU ARCTIC (clb, rms, ksp)
to serve as a strict Speaker-Independent (Out-of-Speaker) Validation Corpus.
"""

import os
import io
import urllib.request
import tarfile
import soundfile as sf
import numpy as np
import librosa

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNSEEN_DIR = os.path.join(PROJECT_ROOT, "data", "unseen_speakers_test")

UNSEEN_SPEAKERS = {
    "clb": "US Female Speaker (CLB)",
    "rms": "US Male Baritone Speaker (RMS)",
    "ksp": "Indian-English Male Speaker (KSP)"
}


def fetch_unseen_speaker(speaker: str, start_idx: int = 50, end_idx: int = 55):
    os.makedirs(UNSEEN_DIR, exist_ok=True)
    target_names = {f"arctic_a{i:04d}.wav" for i in range(start_idx, end_idx + 1)}

    url = f"http://festvox.org/cmu_arctic/packed/cmu_us_{speaker}_arctic.tar.bz2"
    print(f"Fetching {len(target_names)} unseen sentences for speaker '{speaker}' ({UNSEEN_SPEAKERS[speaker]})...")

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

                    peak = np.max(np.abs(data))
                    if peak > 0:
                        data = (data / peak) * 0.92

                    out_path = os.path.join(UNSEEN_DIR, f"{speaker}_{base}")
                    sf.write(out_path, data.astype(np.float32), sr, subtype="PCM_16")
                    target_names.remove(base)
                    saved += 1
                    if not target_names:
                        break

    print(f"  -> Saved {saved} files for unseen speaker {speaker}.")


def main():
    for spk in UNSEEN_SPEAKERS:
        fetch_unseen_speaker(spk, start_idx=50, end_idx=54)
    print("\nSpeaker-independent unseen test corpus ready in data/unseen_speakers_test/!")


if __name__ == "__main__":
    main()

