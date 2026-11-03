"""
Unit tests for Machine Learning Noise Classifier and Adaptive Filter Dispatcher.
"""

import os
import numpy as np
import pytest

from src.audio_io import generate_clean_speech_synthetic
from src.noise_generation import (
    generate_electrical_hum,
    generate_low_frequency_rumble,
    generate_high_frequency_hiss,
    generate_broadband_noise,
    mix_speech_and_noise
)
from src.ml_model import NoiseClassifier, CLASSES


def test_classifier_untrained_error():
    clf = NoiseClassifier()
    with pytest.raises(RuntimeError):
        clf.predict_audio(np.zeros(16000))


def test_classifier_load_and_predict_known_classes():
    model_path = "models/noise_classifier.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Pre-trained model artifact not found, run scripts/train_model.py first.")

    clf = NoiseClassifier()
    clf.load(model_path)
    assert clf.is_trained is True

    sr = 16000
    speech = generate_clean_speech_synthetic(duration=2.5, sr=sr, seed=10)

    # 1. Test Electrical Hum
    hum = generate_electrical_hum(len(speech), sr=sr, base_freq=50.0, seed=10)
    noisy_hum, _, _ = mix_speech_and_noise(speech, hum, target_snr_db=0.0)
    pred_hum = clf.predict_audio(noisy_hum, sr=sr)
    assert pred_hum["predicted_class"] == "electrical_hum"
    assert pred_hum["confidence"] > 0.4

    # 2. Test Low-Frequency Rumble
    rumble = generate_low_frequency_rumble(len(speech), sr=sr, cutoff_hz=100.0, seed=10)
    noisy_rumble, _, _ = mix_speech_and_noise(speech, rumble, target_snr_db=0.0)
    pred_rumble = clf.predict_audio(noisy_rumble, sr=sr)
    assert pred_rumble["predicted_class"] == "low_frequency_rumble"

    # 3. Test High-Frequency Hiss
    hiss = generate_high_frequency_hiss(len(speech), sr=sr, cutoff_hz=4000.0, seed=10)
    noisy_hiss, _, _ = mix_speech_and_noise(speech, hiss, target_snr_db=0.0)
    pred_hiss = clf.predict_audio(noisy_hiss, sr=sr)
    assert pred_hiss["predicted_class"] == "high_frequency_hiss"

    # 4. Test Clean Speech
    pred_clean = clf.predict_audio(speech, sr=sr)
    assert pred_clean["predicted_class"] == "clean_speech"


def test_adaptive_filtering_execution():
    model_path = "models/noise_classifier.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Pre-trained model artifact not found.")

    clf = NoiseClassifier()
    clf.load(model_path)

    sr = 16000
    speech = generate_clean_speech_synthetic(duration=2.0, sr=sr, seed=42)
    hum = generate_electrical_hum(len(speech), sr=sr, base_freq=50.0)
    noisy, _, _ = mix_speech_and_noise(speech, hum, target_snr_db=5.0)

    filtered, pred_info, filter_info = clf.apply_adaptive_filter(noisy, sr=sr)

    assert len(filtered) == len(noisy)
    assert np.all(np.isfinite(filtered))
    assert pred_info["predicted_class"] == "electrical_hum"
    assert "strategy" in filter_info
    assert "dsp_execution" in filter_info
