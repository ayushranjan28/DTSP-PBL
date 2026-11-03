"""
Unit tests for physical and spectral audio feature extraction.
"""

import numpy as np
import pytest

from src.features import extract_features_from_audio, FEATURE_NAMES
from src.noise_generation import generate_electrical_hum, generate_high_frequency_hiss


def test_extract_features_shape_and_finite():
    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float32)

    feat_vec, feat_dict = extract_features_from_audio(audio, sr=sr)

    assert len(feat_vec) == len(FEATURE_NAMES)
    assert np.all(np.isfinite(feat_vec))
    assert len(feat_dict) == len(FEATURE_NAMES)

    for name in FEATURE_NAMES:
        assert name in feat_dict
        assert np.isfinite(feat_dict[name])


def test_hum_features_discriminate_50hz():
    sr = 16000
    n_samples = sr * 2

    hum_50 = generate_electrical_hum(n_samples, sr=sr, base_freq=50.0)
    hiss = generate_high_frequency_hiss(n_samples, sr=sr, cutoff_hz=4000.0)

    _, feat_hum = extract_features_from_audio(hum_50, sr=sr)
    _, feat_hiss = extract_features_from_audio(hiss, sr=sr)

    # 50 Hz ratio must be substantially higher for hum than hiss
    assert feat_hum["hum_50hz_ratio"] > feat_hiss["hum_50hz_ratio"]
    # Spectral centroid must be substantially higher for hiss than hum
    assert feat_hiss["spectral_centroid_mean"] > feat_hum["spectral_centroid_mean"]
