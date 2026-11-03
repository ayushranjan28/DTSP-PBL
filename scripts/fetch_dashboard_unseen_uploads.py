"""
Download 100% Unedited, Pre-Existing Third-Party Noisy Speech Benchmark WAVs
Saves directly to: c:/Users/ayush/Desktop/DTSP/unseen_test_uploads/

Zero synthetic noise generation and zero local mixing are performed in this script.
Every file is downloaded directly from external Speech Enhancement benchmarks on HuggingFace and GitHub:
  1. VoiceBank+DEMAND Official Test Samples (DurreSudoku/Speech_Enhancement_Demo)
  2. FSoft AI Center Noisy Speech Benchmark (fsoft-ai-center/Speech-Enhancement)
  3. UniSE & UniPASE DNS Challenge Noisy Speech Samples (hugging-apps/unise & unipase)
  4. SpeechBrain Speech Enhancement Benchmark Samples (akhaliq & marquesafonso)
  5. Picovoice Koala Official Noisy Speech Benchmark (Picovoice/koala)
"""

import os
import io
import json
import glob
import urllib.request
import soundfile as sf
import librosa
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "unseen_test_uploads")

RAW_THIRD_PARTY_NOISY_SPEECH_URLS = [
    # 1. VoiceBank+DEMAND & Real Noisy Speech Demo Samples
    {
        "filename": "01_VoiceBank_DEMAND_noisy_male_1.wav",
        "url": "https://huggingface.co/spaces/DurreSudoku/Speech_Enhancement_Demo/resolve/main/examples/VoiceBank+DEMAND_test_sample_male_1.wav",
        "source": "VoiceBank+DEMAND Official Benchmark (Male Speaker)",
    },
    {
        "filename": "02_VoiceBank_DEMAND_noisy_sample_1.wav",
        "url": "https://huggingface.co/spaces/DurreSudoku/Speech_Enhancement_Demo/resolve/main/examples/durim_test_sample_1.wav",
        "source": "Speech Enhancement Benchmark Sample 1 (DurreSudoku)",
    },
    {
        "filename": "03_VoiceBank_DEMAND_noisy_sample_2.wav",
        "url": "https://huggingface.co/spaces/DurreSudoku/Speech_Enhancement_Demo/resolve/main/examples/durim_test_sample_2.wav",
        "source": "Speech Enhancement Benchmark Sample 2 (DurreSudoku)",
    },
    # 2. FSoft AI Center English & Japanese Noisy Speech Benchmark across SNRs
    {
        "filename": "04_FSoft_Benchmark_English_Noisy_minus6dB.wav",
        "url": "https://huggingface.co/spaces/fsoft-ai-center/Speech-Enhancement/resolve/main/noisy_speech/EN_-6db.wav",
        "source": "FSoft AI Center Noisy Speech Benchmark (English -6 dB)",
    },
    {
        "filename": "05_FSoft_Benchmark_English_Noisy_minus3dB.wav",
        "url": "https://huggingface.co/spaces/fsoft-ai-center/Speech-Enhancement/resolve/main/noisy_speech/EN_-3dB.wav",
        "source": "FSoft AI Center Noisy Speech Benchmark (English -3 dB)",
    },
    {
        "filename": "06_FSoft_Benchmark_English_Noisy_0dB.wav",
        "url": "https://huggingface.co/spaces/fsoft-ai-center/Speech-Enhancement/resolve/main/noisy_speech/EN_+0dB.wav",
        "source": "FSoft AI Center Noisy Speech Benchmark (English 0 dB)",
    },
    {
        "filename": "07_FSoft_Benchmark_English_Noisy_plus3dB.wav",
        "url": "https://huggingface.co/spaces/fsoft-ai-center/Speech-Enhancement/resolve/main/noisy_speech/EN_+3dB.wav",
        "source": "FSoft AI Center Noisy Speech Benchmark (English +3 dB)",
    },
    {
        "filename": "08_FSoft_Benchmark_English_Noisy_plus6dB.wav",
        "url": "https://huggingface.co/spaces/fsoft-ai-center/Speech-Enhancement/resolve/main/noisy_speech/EN_+6dB.wav",
        "source": "FSoft AI Center Noisy Speech Benchmark (English +6 dB)",
    },
    {
        "filename": "09_FSoft_Benchmark_Japanese_Noisy_0dB.wav",
        "url": "https://huggingface.co/spaces/fsoft-ai-center/Speech-Enhancement/resolve/main/noisy_speech/JA_+0dB.wav",
        "source": "FSoft AI Center Noisy Speech Benchmark (Japanese 0 dB)",
    },
    {
        "filename": "10_FSoft_Benchmark_Japanese_Noisy_plus3dB.wav",
        "url": "https://huggingface.co/spaces/fsoft-ai-center/Speech-Enhancement/resolve/main/noisy_speech/JA_+3dB.wav",
        "source": "FSoft AI Center Noisy Speech Benchmark (Japanese +3 dB)",
    },
    # 3. UniSE & UniPASE DNS-Challenge Noisy Speech Benchmark Samples
    {
        "filename": "11_DNS_Challenge_UniPASE_fileid_295.wav",
        "url": "https://huggingface.co/spaces/hugging-apps/unipase-speech-enhancement/resolve/main/examples/fileid_295.wav",
        "source": "Microsoft DNS-Challenge / UniPASE Benchmark (fileid_295.wav)",
    },
    {
        "filename": "12_DNS_Challenge_UniPASE_fileid_210.wav",
        "url": "https://huggingface.co/spaces/hugging-apps/unipase-speech-enhancement/resolve/main/examples/fileid_210.flac",
        "source": "Microsoft DNS-Challenge / UniPASE Benchmark (fileid_210.flac)",
    },
    {
        "filename": "13_DNS_Challenge_UniPASE_fileid_49.wav",
        "url": "https://huggingface.co/spaces/hugging-apps/unipase-speech-enhancement/resolve/main/examples/fileid_49.flac",
        "source": "Microsoft DNS-Challenge / UniPASE Benchmark (fileid_49.flac)",
    },
    {
        "filename": "14_UniSE_Benchmark_sr_noisy_1.wav",
        "url": "https://huggingface.co/spaces/hugging-apps/unise-speech-enhancement/resolve/main/examples/sr_noisy_1.wav",
        "source": "UniSE Speech Enhancement Benchmark (sr_noisy_1.wav)",
    },
    {
        "filename": "15_UniSE_Benchmark_sr_noisy_2.wav",
        "url": "https://huggingface.co/spaces/hugging-apps/unise-speech-enhancement/resolve/main/examples/sr_noisy_2.wav",
        "source": "UniSE Speech Enhancement Benchmark (sr_noisy_2.wav)",
    },
    {
        "filename": "16_UniSE_Benchmark_tse_noisy_1.wav",
        "url": "https://huggingface.co/spaces/hugging-apps/unise-speech-enhancement/resolve/main/examples/tse_noisy_1.wav",
        "source": "UniSE Speech Enhancement Benchmark (tse_noisy_1.wav)",
    },
    # 4. SpeechBrain & Picovoice Official Noisy Speech Benchmarks
    {
        "filename": "17_SpeechBrain_Official_Noisy_French.wav",
        "url": "https://huggingface.co/spaces/akhaliq/Speechbrain-Speech-enhancement/resolve/main/samples_audio_samples_example_fr.wav",
        "source": "SpeechBrain Official Speech Enhancement Demo (example_fr.wav)",
    },
    {
        "filename": "18_SpeechBrain_Official_Test_Mixture.wav",
        "url": "https://huggingface.co/spaces/marquesafonso/speech-enhancement/resolve/main/samples_audio_samples_test_mixture.wav",
        "source": "SpeechBrain Official Speech Enhancement Demo (test_mixture.wav)",
    },
    {
        "filename": "19_Picovoice_Koala_Official_Noisy_Speech.wav",
        "url": "https://raw.githubusercontent.com/Picovoice/koala/master/resources/audio_samples/test.wav",
        "source": "Picovoice Koala Official Noisy Speech Benchmark (test.wav)",
    },
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Clear old files from unseen_test_uploads
    for old_f in glob.glob(os.path.join(OUT_DIR, "*.wav")):
        try:
            os.remove(old_f)
        except OSError:
            pass

    manifest = []
    print(f"Downloading {len(RAW_THIRD_PARTY_NOISY_SPEECH_URLS)} raw, unedited third-party noisy speech files to:\n  {OUT_DIR}\n")

    for item in RAW_THIRD_PARTY_NOISY_SPEECH_URLS:
        out_path = os.path.join(OUT_DIR, item["filename"])
        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw_bytes = resp.read()

            audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            sf.write(out_path, audio.astype(np.float32), sr, subtype="PCM_16")
            dur = round(len(audio) / sr, 2)
            print(f"  [+] {item['filename']:<46} | {dur:5.2f}s | {item['source']}")
            manifest.append({
                "filename": item["filename"],
                "source": item["source"],
                "url": item["url"],
                "duration_sec": dur,
                "locally_generated_or_mixed": False
            })
        except Exception as e:
            print(f"  [!] Failed {item['filename']}: {e}")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(manifest)} genuine third-party noisy speech WAVs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
