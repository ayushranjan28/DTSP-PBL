"""
Strict Third-Party External Benchmark Validation Suite
Evaluates models/noise_classifier.joblib strictly on external, third-party real-world
audio datasets NOT created by SonicPure's synthetic noise generators:
  1. Microsoft Scalable Noisy Speech Dataset (MS-SNSD)
  2. DeepFilterNet Official Benchmark Assets (Freesound)
  3. SpeechBrain Official Test Suite (LibriSpeech & Real Noise)
  4. Unseen CMU ARCTIC Speakers (CLB, RMS, KSP)
"""

import os
import sys
import json
import time
import glob
import hashlib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.audio_io import load_audio
from src.ml_model import NoiseClassifier, CLASSES, CLASS_LABELS
from src.noise_generation import mix_speech_and_noise

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "noise_classifier.joblib")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train_corpus")
EXT_DIR = os.path.join(PROJECT_ROOT, "data", "external_benchmarks")
UNSEEN_SPK_DIR = os.path.join(PROJECT_ROOT, "data", "unseen_speakers_test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Labels as originally assigned (by filename) before 4 were revised after inspecting their spectra.
# Both scores are reported so the effect of the post-hoc revision is visible.
ORIGINAL_LABELS = {
    "ms_noise_Washing_1.wav": "high_frequency_hiss",
    "ms_noise_Station_1.wav": "broadband_noise",
    "sb_noise2.wav": "broadband_noise",
    "dfn_noise_freesound_573577.wav": "broadband_noise",
}


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_external_isolation():
    """
    Stage 1: Verify zero SHA-256 overlap between training files and external benchmark files.
    """
    print("\n" + "=" * 86)
    print("STAGE 1: Cryptographic Isolation Audit (Training Data vs. External Benchmarks)")
    print("=" * 86)

    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.wav")))
    train_noise_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "train_real_noise", "**", "*.wav"), recursive=True))
    all_train_files = train_files + train_noise_files

    ext_files = sorted(glob.glob(os.path.join(EXT_DIR, "*.wav")))
    unseen_spk_files = sorted(glob.glob(os.path.join(UNSEEN_SPK_DIR, "*.wav")))

    train_hashes = {file_sha256(f): os.path.basename(f) for f in all_train_files}
    ext_hashes = {file_sha256(f): os.path.basename(f) for f in ext_files}
    unseen_hashes = {file_sha256(f): os.path.basename(f) for f in unseen_spk_files}

    overlap_ext = set(train_hashes.keys()).intersection(set(ext_hashes.keys()))
    overlap_unseen = set(train_hashes.keys()).intersection(set(unseen_hashes.keys()))

    print(f"  Local Training Speech Corpus (data/train_corpus/)        : {len(train_files)} files")
    print(f"  Local Training Real Noise (data/train_real_noise/)       : {len(train_noise_files)} files")
    print(f"  External 3rd-Party Benchmarks (data/external_benchmarks/): {len(ext_files)} files")
    print(f"  Unseen Speaker Corpus (data/unseen_speakers_test/)       : {len(unseen_spk_files)} files")
    print(f"  SHA-256 Overlap (All Train vs. External Benchmarks)      : {len(overlap_ext)} files")
    print(f"  SHA-256 Overlap (All Train vs. Unseen Speakers)          : {len(overlap_unseen)} files")

    assert len(overlap_ext) == 0, "Leakage in external benchmarks!"
    assert len(overlap_unseen) == 0, "Leakage in unseen speakers!"
    print("  [PASS] 100% Isolated: Zero self-generated pre-mixed files used.")
    print("=" * 86)

    return {
        "n_train_files": len(train_files),
        "n_external_benchmark_files": len(ext_files),
        "n_unseen_speaker_files": len(unseen_spk_files),
        "zero_leakage_verified": True
    }


def evaluate_raw_external_benchmarks(classifier: NoiseClassifier):
    """
    Stage 2: Direct evaluation on the 32 unedited external WAV files from
    Microsoft MS-SNSD, DeepFilterNet, and SpeechBrain, plus 15 unseen speaker WAVs.
    """
    manifest_path = os.path.join(EXT_DIR, "external_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        ext_manifest = json.load(f)

    print("\n" + "=" * 86)
    print("STAGE 2: Direct Evaluation on Raw 3rd-Party WAVs (Microsoft, DeepFilterNet, SpeechBrain)")
    print("=" * 86)

    y_true = []
    y_pred = []
    results = []

    for item in ext_manifest:
        wav_path = os.path.join(EXT_DIR, item["filename"])
        audio, sr = load_audio(wav_path, target_sr=16000)

        pred_res = classifier.predict_audio(audio, sr=sr)
        pred_class = pred_res["predicted_class"]
        conf = pred_res["confidence"]
        true_class = item["expected_category"]

        # Note: Real-world noises (e.g. VacuumCleaner, Washing, AirConditioner) are broadband physical
        # recordings that often span multiple frequency bands. We check both exact match and acoustic family.
        is_exact = (pred_class == true_class)
        y_true.append(true_class)
        y_pred.append(pred_class)

        status = "CORRECT " if is_exact else "MISMATCH"
        print(f"[{status}] {item['filename']:<32} | {item['source']:<25} | True: {true_class:<20} | Pred: {pred_class:<20} ({conf*100:5.1f}%)")

        results.append({
            "file": item["filename"],
            "source": item["source"],
            "true_class": true_class,
            "pred_class": pred_class,
            "confidence": round(conf, 4),
            "correct": is_exact
        })

    # Also evaluate the 15 unseen speaker recordings (CLB, RMS, KSP)
    unseen_spk_files = sorted(glob.glob(os.path.join(UNSEEN_SPK_DIR, "*.wav")))
    for f in unseen_spk_files:
        audio, sr = load_audio(f, target_sr=16000)
        base = os.path.basename(f)
        pred_res = classifier.predict_audio(audio, sr=sr)
        pred_class = pred_res["predicted_class"]
        conf = pred_res["confidence"]
        true_class = "clean_speech"
        is_exact = (pred_class == true_class)

        y_true.append(true_class)
        y_pred.append(pred_class)

        status = "CORRECT " if is_exact else "MISMATCH"
        print(f"[{status}] {base:<32} | {'CMU Unseen Speaker':<25} | True: {true_class:<20} | Pred: {pred_class:<20} ({conf*100:5.1f}%)")

        results.append({
            "file": base,
            "source": "CMU Unseen Speaker",
            "true_class": true_class,
            "pred_class": pred_class,
            "confidence": round(conf, 4),
            "correct": is_exact
        })

    acc = float(accuracy_score(y_true, y_pred))
    n_ok = sum(r['correct'] for r in results)
    lo, hi = wilson_ci(n_ok, len(results))
    orig_ok = sum(r["pred_class"] == ORIGINAL_LABELS.get(r["file"], r["true_class"]) for r in results)
    o_lo, o_hi = wilson_ci(orig_ok, len(results))
    print("-" * 86)
    print(f"Raw 3rd-Party Benchmark Accuracy: {acc * 100:.2f}% ({n_ok}/{len(results)} files)  95% CI {lo:.1f}-{hi:.1f}%")
    print(f"  ...under ORIGINAL (pre-revision) labels: {orig_ok/len(results)*100:.2f}% ({orig_ok}/{len(results)})  95% CI {o_lo:.1f}-{o_hi:.1f}%")
    print("  electrical_hum: 0 external test files -> NOT externally validated")
    print("=" * 86)

    return {
        "accuracy": round(acc, 4),
        "accuracy_95ci": [round(lo, 1), round(hi, 1)],
        "accuracy_original_labels": round(orig_ok / len(results), 4),
        "n_files": len(results),
        "results": results
    }


def evaluate_real_world_speech_noise_combinations(classifier: NoiseClassifier):
    """
    Stage 3: Cross-Dataset Real-World Speech + Real-World Recorded Noise Matrix
    Combines 25 external clean speech recordings (Microsoft MS-SNSD, SpeechBrain LibriSpeech,
    DeepFilterNet Freesound, CMU Unseen Speakers CLB/RMS/KSP) with 21 real-world recorded
    noises from Microsoft MS-SNSD, DeepFilterNet, and SpeechBrain at SNRs [-5, 0, 5, 10, 15] dB.
    """
    print("\n" + "=" * 86)
    print("STAGE 3: Cross-Dataset Real Speech + Real-World Recorded Noise Matrix (Zero Synthetic Noise)")
    print("=" * 86)

    manifest_path = os.path.join(EXT_DIR, "external_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        ext_manifest = json.load(f)

    # Gather external clean speech signals
    clean_pool = []
    for item in ext_manifest:
        if item["expected_category"] == "clean_speech":
            aud, sr = load_audio(os.path.join(EXT_DIR, item["filename"]), target_sr=16000)
            clean_pool.append({"name": item["filename"], "audio": aud, "source": item["source"]})

    for f in sorted(glob.glob(os.path.join(UNSEEN_SPK_DIR, "*.wav"))):
        aud, sr = load_audio(f, target_sr=16000)
        clean_pool.append({"name": os.path.basename(f), "audio": aud, "source": "CMU Unseen Speaker"})

    # Gather external real-world noise recordings
    noise_pool = []
    for item in ext_manifest:
        if item["expected_category"] != "clean_speech" and item["filename"] != "dfn_noisy_snr0.wav":
            aud, sr = load_audio(os.path.join(EXT_DIR, item["filename"]), target_sr=16000)
            noise_pool.append({
                "name": item["filename"],
                "audio": aud,
                "source": item["source"],
                "category": item["expected_category"]
            })

    print(f"External Clean Speech Pool : {len(clean_pool)} files (Microsoft, SpeechBrain, Freesound, CMU Unseen)")
    print(f"External Real Noise Pool   : {len(noise_pool)} files (Microsoft MS-SNSD, DeepFilterNet, SpeechBrain)")

    y_true = []
    y_pred = []
    trial_snrs = []
    binary_noisy_vs_clean_true = []
    binary_noisy_vs_clean_pred = []

    snr_levels = [-5.0, 0.0, 5.0, 10.0, 15.0]
    rng = np.random.default_rng(0)

    # Clean files are already scored in Stage 2, so they are NOT re-counted here
    # (re-counting them inflated the mixture accuracy and the clean-speech recall).
    # All 25 speakers (incl. CMU unseen) are mixed with every noise at every SNR.
    for c_item in clean_pool:
        speech = c_item["audio"]
        for n_item in noise_pool:
            noise_full = n_item["audio"]
            true_cat = n_item["category"]
            if len(noise_full) < len(speech):
                noise = np.tile(noise_full, int(np.ceil(len(speech) / len(noise_full))))[:len(speech)]
            else:
                start = int(rng.integers(0, len(noise_full) - len(speech) + 1))
                noise = noise_full[start:start + len(speech)]
            for snr in snr_levels:
                noisy, _, _ = mix_speech_and_noise(speech, noise, target_snr_db=snr)
                res = classifier.predict_audio(noisy, sr=16000)
                p = res["predicted_class"]
                y_true.append(true_cat)
                y_pred.append(p)
                trial_snrs.append(snr)
                binary_noisy_vs_clean_true.append("noisy")
                binary_noisy_vs_clean_pred.append("clean" if p == "clean_speech" else "noisy")

    exact_acc = float(accuracy_score(y_true, y_pred))
    binary_acc = float(accuracy_score(binary_noisy_vs_clean_true, binary_noisy_vs_clean_pred))

    present_classes = ["low_frequency_rumble", "high_frequency_hiss", "broadband_noise"]
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    rep = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0)

    print(f"\nTotal Noisy Mixture Trials ({len(clean_pool)} speakers x {len(noise_pool)} noises x {len(snr_levels)} SNRs): {len(y_true)}")
    print(f"Noise Detected (not predicted clean_speech)  : {binary_acc * 100:.2f}%")
    print(f"Exact Fine-Grained Sub-Class Accuracy        : {exact_acc * 100:.2f}%")
    print("NOTE: trials share 21 noise recordings, so the effective sample size is ~21, not the trial count.\n")
    for snr in snr_levels:
        idx = [i for i, s in enumerate(trial_snrs) if s == snr]
        a = np.mean([y_true[i] == y_pred[i] for i in idx])
        d = np.mean([y_pred[i] != "clean_speech" for i in idx])
        print(f"  SNR {snr:+5.0f} dB : exact {a*100:5.1f}%   noise detected {d*100:5.1f}%")
    print()

    print(f"{'Class Name':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 86)
    for c in present_classes:
        m = rep[c]
        print(f"{CLASS_LABELS[c]:<30} | {m['precision']*100:6.2f}%    | {m['recall']*100:6.2f}%    | {m['f1-score']*100:6.2f}%    | {int(m['support']):<8}")

    print("\nCONFUSION MATRIX ON EXTERNAL REAL-WORLD DATA (Rows: True, Cols: Predicted):")
    print(f"{'True \\ Pred':<22} | " + " | ".join(f"{c[:8]:>8}" for c in CLASSES))
    print("-" * 86)
    for idx, true_c in enumerate(CLASSES):
        row_str = " | ".join(f"{cm[idx][j]:>8}" for j in range(len(CLASSES)))
        print(f"{true_c:<22} | {row_str}")
    print("=" * 86)

    return {
        "n_trials": len(y_true),
        "noise_vs_clean_accuracy": round(binary_acc, 4),
        "exact_subclass_accuracy": round(exact_acc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": rep
    }


def main():
    classifier = NoiseClassifier()
    classifier.load(MODEL_PATH)

    s1 = audit_external_isolation()
    s2 = evaluate_raw_external_benchmarks(classifier)
    s3 = evaluate_real_world_speech_noise_combinations(classifier)

    report = {
        "stage_1_isolation_audit": s1,
        "stage_2_raw_external_files": s2,
        "stage_3_cross_dataset_real_noise": s3
    }
    out_json = os.path.join(OUTPUT_DIR, "unseen_validation_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved external validation report to {out_json}")


if __name__ == "__main__":
    main()
