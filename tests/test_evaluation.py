"""
Unit tests for quantitative evaluation metrics (reference-based and unpaired descriptive).
"""

import numpy as np
import pytest

from src.audio_io import generate_clean_speech_synthetic
from src.noise_generation import generate_broadband_noise, mix_speech_and_noise
from src.evaluation import compute_reference_based_metrics, compute_unpaired_metrics


def test_reference_based_metrics_on_known_signals():
    sr = 16000
    speech = generate_clean_speech_synthetic(duration=2.0, sr=sr, seed=42)
    noise = generate_broadband_noise(len(speech), sr=sr, noise_type="white", seed=42)

    target_snr = 10.0
    noisy, _, _ = mix_speech_and_noise(speech, noise, target_snr_db=target_snr)

    # Test 1: Baseline noisy signal compared with itself
    metrics_noisy = compute_reference_based_metrics(
        clean_ref=speech,
        noisy_input=noisy,
        processed=noisy,
        sr=sr,
        processing_time_sec=0.05
    )

    assert metrics_noisy["has_reference"] is True
    assert metrics_noisy["input_snr_db"] == pytest.approx(target_snr, abs=0.2)
    assert metrics_noisy["output_snr_db"] == pytest.approx(target_snr, abs=0.2)
    assert metrics_noisy["snr_improvement_db"] == pytest.approx(0.0, abs=0.05)
    assert metrics_noisy["audio_duration_sec"] == pytest.approx(2.0, abs=0.05)
    assert metrics_noisy["real_time_factor"] == pytest.approx(0.05 / 2.0, abs=1e-3)
    assert metrics_noisy["is_real_time_capable"] is True

    # Test 2: Perfect denoising where processed == clean_ref
    metrics_perfect = compute_reference_based_metrics(
        clean_ref=speech,
        noisy_input=noisy,
        processed=speech,
        sr=sr,
        processing_time_sec=0.02
    )

    assert metrics_perfect["output_snr_db"] >= 90.0
    assert metrics_perfect["snr_improvement_db"] > 70.0
    assert metrics_perfect["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics_perfect["pearson_correlation"] == pytest.approx(1.0, abs=1e-4)


def test_unpaired_metrics():
    sr = 16000
    sig_in = np.random.randn(sr * 2).astype(np.float32)
    sig_out = sig_in * 0.5  # 6 dB attenuation

    metrics = compute_unpaired_metrics(sig_in, sig_out, sr=sr, processing_time_sec=0.1)

    assert metrics["has_reference"] is False
    assert metrics["total_attenuation_db"] == pytest.approx(6.0, abs=0.5)
    assert "input_rms_dbfs" in metrics
    assert "output_rms_dbfs" in metrics
    assert "rumble_band_attenuation_db" in metrics
    assert "hum_50hz_attenuation_db" in metrics
    assert "hiss_band_attenuation_db" in metrics
    assert "interpretability_note" in metrics
