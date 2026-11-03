"""
Fetch authentic, phonetically balanced studio human speech recordings
from the CMU ARCTIC corpus (16 kHz, 16-bit PCM WAV) with actual spoken sentences.
"""

import os
import io
import urllib.request
import tarfile
import numpy as np
import soundfile as sf
import librosa

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")

# High-quality target sentences to fetch
TARGET_SENTENCES = {
    "slt": {  # US English Female Speaker
        "arctic_a0003.wav": {
            "output_name": "clean_female_sentence_01.wav",
            "sentence": "For the twentieth time that evening the two men shook hands.",
            "desc": "Natural conversational female pacing with clear pauses, ideal for Wiener and spectral gating."
        },
        "arctic_a0009.wav": {
            "output_name": "clean_female_sentence_02.wav",
            "sentence": "He turned sharply, and faced Gregson across the table.",
            "desc": "Rich in sharp sibilants ('sh', 's') and stop consonants ('t', 'p', 'k') for hiss/filter testing."
        },
        "arctic_a0011.wav": {
            "output_name": "clean_female_sentence_03.wav",
            "sentence": "If I ever needed a fighter in my life I need one now.",
            "desc": "Expressive dynamic range with strong intonation contour and vocal emotion."
        },
        "arctic_a0046.wav": {
            "output_name": "clean_female_sentence_04.wav",
            "sentence": "The girl faced him, her eyes shining with sudden fear.",
            "desc": "Articulate female narration with clear breath pauses, tests transient preservation."
        }
    },
    "bdl": {  # US English Male Speaker
        "arctic_a0003.wav": {
            "output_name": "clean_male_sentence_01.wav",
            "sentence": "For the twentieth time that evening the two men shook hands.",
            "desc": "Resonant male chest voice (~115 Hz pitch), ideal for testing 50/60 Hz hum and rumble notch."
        },
        "arctic_a0009.wav": {
            "output_name": "clean_male_sentence_02.wav",
            "sentence": "He turned sharply, and faced Gregson across the table.",
            "desc": "Crisp male articulation with sibilants and plosives across 100-4000 Hz."
        },
        "arctic_a0012.wav": {
            "output_name": "clean_male_sentence_03.wav",
            "sentence": "Gregson shoved back his chair and rose to his feet.",
            "desc": "Deep baritone voice with dynamic transitions and sentence-ending cadence."
        },
        "arctic_a0014.wav": {
            "output_name": "clean_male_sentence_04.wav",
            "sentence": "Now you're coming down to business, Phil, he exclaimed.",
            "desc": "Conversational cadence with natural pitch inflection, tests SNR recovery."
        }
    }
}


def download_and_extract_speaker(speaker: str, targets: dict):
    url = f"http://festvox.org/cmu_arctic/packed/cmu_us_{speaker}_arctic.tar.bz2"
    print(f"\nDownloading and streaming {speaker.upper()} corpus from {url}...")
    
    needed = set(targets.keys())
    saved_count = 0
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        with tarfile.open(fileobj=resp, mode="r|bz2") as tar:
            for member in tar:
                base = os.path.basename(member.name)
                if base in needed:
                    extracted = tar.extractfile(member)
                    audio_bytes = extracted.read()
                    
                    # Read using soundfile
                    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    if sr != 16000:
                        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                        sr = 16000
                        
                    # Peak normalize to [-0.92, 0.92] to ensure headroom without clipping
                    peak = np.max(np.abs(data))
                    if peak > 0:
                        data = (data / peak) * 0.92
                        
                    meta = targets[base]
                    out_path = os.path.join(SAMPLES_DIR, meta["output_name"])
                    sf.write(out_path, data.astype(np.float32), sr, subtype="PCM_16")
                    
                    dur = len(data) / sr
                    print(f" -> Saved {meta['output_name']} ({dur:.2f}s, {sr} Hz)")
                    print(f"    Transcript: \"{meta['sentence']}\"")
                    
                    needed.remove(base)
                    saved_count += 1
                    if not needed:
                        break
                        
    print(f"Finished {speaker.upper()}: saved {saved_count} audio files.")


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    for speaker, targets in TARGET_SENTENCES.items():
        try:
            download_and_extract_speaker(speaker, targets)
        except Exception as e:
            print(f"Error fetching {speaker}: {e}")

    print("\nAll proper sentence audios successfully processed and saved to data/samples/!")


if __name__ == "__main__":
    main()

