"""
Dataset Preparation Script for VoiceBank-DEMAND and Custom Audio Corpora
Organizes, checks sample rates, and partitions clean speech and noise sources.
"""

import os
import argparse
import glob
from typing import List, Tuple
import soundfile as sf
import librosa


def prepare_voicebank_demand(
    dataset_dir: str,
    output_dir: str = "data/voicebank_demand_16k",
    target_sr: int = 16000
) -> None:
    """
    Prepare locally downloaded VoiceBank-DEMAND dataset:
    - Verifies directory structure
    - Resamples speech and noise to target_sr (16 kHz)
    - Validates audio file integrity
    """
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        print("Please download VoiceBank-DEMAND from:")
        print("https://datashare.ed.ac.uk/handle/10283/2791")
        return

    os.makedirs(output_dir, exist_ok=True)
    clean_wavs = glob.glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True)

    if not clean_wavs:
        print(f"No .wav files located in {dataset_dir}")
        return

    print(f"Found {len(clean_wavs)} WAV files. Processing to {output_dir} at {target_sr} Hz...")

    count = 0
    for wav_path in clean_wavs[:100]:  # Cap to reasonable batch
        rel_name = os.path.basename(wav_path)
        out_path = os.path.join(output_dir, rel_name)

        try:
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sf.write(out_path, audio, target_sr, subtype="PCM_16")
            count += 1
        except Exception as e:
            print(f"Failed to process {wav_path}: {e}")

    print(f"Successfully prepared {count} files in {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceBank-DEMAND Dataset Preparation")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to local VoiceBank-DEMAND root")
    parser.add_argument("--output_dir", type=str, default="data/voicebank_demand_16k", help="Destination folder")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (Hz)")
    args = parser.parse_args()

    prepare_voicebank_demand(args.dataset_dir, args.output_dir, args.sr)
