"""
Machine Learning Training Pipeline (Hybrid Synthetic + Real-World Environmental Acoustics)
Trains the Random Forest Noise Classifier on:
  1. Real human speech (CMU ARCTIC + Microsoft MS-SNSD clean_train partition)
  2. Synthetic formant speech
  3. Both calibrated synthetic noises AND real-world recorded environmental noises
     from data/train_real_noise/ (strictly disjoint from data/external_benchmarks/).
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.audio_io import generate_clean_speech_synthetic, load_audio
from src.noise_generation import (
    generate_electrical_hum,
    generate_low_frequency_rumble,
    generate_high_frequency_hiss,
    generate_broadband_noise,
    mix_speech_and_noise
)
from src.features import extract_features_from_audio
from src.ml_model import NoiseClassifier, CLASSES


def load_base_utterances(
    train_corpus_dir: str = "data/train_corpus",
    n_synthetic: int = 18,
    sr: int = 16000,
    random_seed: int = 42
):
    rng = np.random.default_rng(random_seed)
    utterances = []

    real_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, train_corpus_dir, "*.wav")))
    print(f"Found {len(real_files)} real human speech WAVs in {train_corpus_dir}...")
    for f in real_files:
        try:
            audio, _ = load_audio(f, target_sr=sr)
            if len(audio) >= sr * 1.5:
                utterances.append(audio)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    for i in range(n_synthetic):
        dur = float(rng.uniform(1.8, 4.2))
        seed = random_seed + i * 13
        synth = generate_clean_speech_synthetic(duration=dur, sr=sr, seed=seed)
        utterances.append(synth)

    print(f"Total base speech pool: {len(utterances)} utterances")
    return utterances


def load_real_noise_pool(train_noise_dir: str = "data/train_real_noise", sr: int = 16000):
    pool = {
        "low_frequency_rumble": [],
        "high_frequency_hiss": [],
        "broadband_noise": []
    }
    for cat in pool.keys():
        files = sorted(glob.glob(os.path.join(PROJECT_ROOT, train_noise_dir, cat, "*.wav")))
        for f in files:
            try:
                aud, _ = load_audio(f, target_sr=sr)
                if len(aud) >= sr * 1.5:
                    pool[cat].append(aud)
            except Exception as e:
                print(f"Skipping noise {f}: {e}")
        print(f"Loaded {len(pool[cat])} real-world training noises for [{cat}]")
    return pool


def random_slice(audio: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) <= target_len:
        repeats = int(np.ceil(target_len / len(audio)))
        return np.tile(audio, repeats)[:target_len]
    start = int(rng.integers(0, len(audio) - target_len))
    return audio[start:start + target_len]


def build_augmented_dataset(
    utterances,
    real_noise_pool,
    train_ratio: float = 0.82,
    sr: int = 16000,
    random_seed: int = 42
):
    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(utterances))
    rng.shuffle(indices)

    n_train = int(len(utterances) * train_ratio)
    train_utterances = [utterances[i] for i in indices[:n_train]]
    test_utterances = [utterances[i] for i in indices[n_train:]]

    # Partition real noises into train and internal validation slices (first 75% vs last 25% of each recording)
    real_noise_train = {k: [a[:int(len(a)*0.75)] for a in v] for k, v in real_noise_pool.items()}
    real_noise_test = {k: [a[int(len(a)*0.75):] for a in v] for k, v in real_noise_pool.items()}

    snr_levels = [-6.0, -3.0, 0.0, 3.0, 5.0, 8.0, 12.0, 16.0]

    def create_samples(base_list, r_noises, samples_per_class: int = 6):
        X = []
        y = []

        # Also include standalone real noise slices so raw unmixed noise files classify accurately
        for cat, noise_list in r_noises.items():
            for n_aud in noise_list:
                for _ in range(3):
                    sl = random_slice(n_aud, sr * 3, rng)
                    feat_vec, _ = extract_features_from_audio(sl, sr=sr)
                    X.append(feat_vec)
                    y.append(cat)

        for speech in base_list:
            n_samples = len(speech)

            # 1. Clean Speech
            for _ in range(samples_per_class):
                gain = float(10.0 ** (rng.uniform(-6.0, 3.0) / 20.0))
                speech_aug = np.clip(speech * gain, -0.95, 0.95)
                feat_vec, _ = extract_features_from_audio(speech_aug, sr=sr)
                X.append(feat_vec)
                y.append("clean_speech")

            for k in range(samples_per_class):
                snr = float(rng.choice(snr_levels))
                cur_seed = int(rng.integers(1000, 9999999))

                # 2. Electrical Hum (50Hz or 60Hz)
                base_f = 50.0 if rng.random() > 0.35 else 60.0
                n_harm = 4 if rng.random() > 0.4 else int(rng.integers(3, 7))
                hum = generate_electrical_hum(n_samples, sr=sr, base_freq=base_f, n_harmonics=n_harm, seed=cur_seed)
                noisy_hum, _, _ = mix_speech_and_noise(speech, hum, target_snr_db=snr)
                feat, _ = extract_features_from_audio(noisy_hum, sr=sr)
                X.append(feat)
                y.append("electrical_hum")

                # 3. Low-Frequency Rumble (50% synthetic Butterworth, 50% real HVAC/Vehicle rumble)
                if k % 2 == 0 or not r_noises["low_frequency_rumble"]:
                    cutoff_r = float(rng.uniform(70.0, 140.0))
                    rumble = generate_low_frequency_rumble(n_samples, sr=sr, cutoff_hz=cutoff_r, seed=cur_seed + 1)
                else:
                    r_src = r_noises["low_frequency_rumble"][int(rng.integers(0, len(r_noises["low_frequency_rumble"])))]
                    rumble = random_slice(r_src, n_samples, rng)
                noisy_rumble, _, _ = mix_speech_and_noise(speech, rumble, target_snr_db=snr)
                feat, _ = extract_features_from_audio(noisy_rumble, sr=sr)
                X.append(feat)
                y.append("low_frequency_rumble")

                # 4. High-Frequency Hiss (50% synthetic highpass, 50% real Vacuum/Appliance hiss)
                if k % 2 == 0 or not r_noises["high_frequency_hiss"]:
                    cutoff_h = float(rng.uniform(3000.0, 5500.0))
                    hiss = generate_high_frequency_hiss(n_samples, sr=sr, cutoff_hz=cutoff_h, seed=cur_seed + 2)
                else:
                    h_src = r_noises["high_frequency_hiss"][int(rng.integers(0, len(r_noises["high_frequency_hiss"])))]
                    hiss = random_slice(h_src, n_samples, rng)
                noisy_hiss, _, _ = mix_speech_and_noise(speech, hiss, target_snr_db=snr)
                feat, _ = extract_features_from_audio(noisy_hiss, sr=sr)
                X.append(feat)
                y.append("high_frequency_hiss")

                # 5. Broadband Noise (50% synthetic white/pink, 50% real Babble/Cafe/Office/Park)
                if k % 2 == 0 or not r_noises["broadband_noise"]:
                    ntype = "pink" if rng.random() > 0.5 else "white"
                    broadband = generate_broadband_noise(n_samples, sr=sr, noise_type=ntype, seed=cur_seed + 3)
                else:
                    b_src = r_noises["broadband_noise"][int(rng.integers(0, len(r_noises["broadband_noise"])))]
                    broadband = random_slice(b_src, n_samples, rng)
                noisy_bb, _, _ = mix_speech_and_noise(speech, broadband, target_snr_db=snr)
                feat, _ = extract_features_from_audio(noisy_bb, sr=sr)
                X.append(feat)
                y.append("broadband_noise")

        return np.array(X, dtype=np.float32), np.array(y)

    print("Generating hybrid (synthetic + real-world) training features...")
    X_train, y_train = create_samples(train_utterances, real_noise_train, samples_per_class=6)
    print(f"X_train shape: {X_train.shape}, classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    print("Generating held-out test features...")
    X_test, y_test = create_samples(test_utterances, real_noise_test, samples_per_class=4)
    print(f"X_test shape: {X_test.shape}, classes: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, y_train, X_test, y_test


def run_training(
    model_output_path: str = "models/noise_classifier.joblib",
    meta_output_path: str = "models/metadata.json"
) -> None:
    print("\n=======================================================")
    print("      SonicPure DSP - Hybrid Model Training Pipeline   ")
    print("=======================================================")

    model_path = os.path.join(PROJECT_ROOT, model_output_path)
    meta_path = os.path.join(PROJECT_ROOT, meta_output_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    utterances = load_base_utterances()
    real_noise_pool = load_real_noise_pool()
    X_train, y_train, X_test, y_test = build_augmented_dataset(utterances, real_noise_pool)

    classifier = NoiseClassifier(random_state=42)
    metrics = classifier.train(
        X_train, y_train,
        X_test, y_test,
        provenance="Hybrid Real Human Speech (CMU + MS-SNSD) & Real Environmental Noise + Synthetic DSP Filters"
    )

    print(f"\n--- Model Training Results ---")
    print(f"Train Accuracy: {metrics['train_accuracy'] * 100:.2f}% ({metrics['n_train_samples']} samples)")
    print(f"Test Accuracy:  {metrics['test_accuracy'] * 100:.2f}% ({metrics['n_test_samples']} samples)")

    classifier.save(model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSuccessfully serialized improved model to:\n  {model_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Improved ML Noise Classifier")
    parser.add_argument("--model_path", type=str, default="models/noise_classifier.joblib")
    parser.add_argument("--meta_path", type=str, default="models/metadata.json")
    args = parser.parse_args()

    run_training(model_output_path=args.model_path, meta_output_path=args.meta_path)
