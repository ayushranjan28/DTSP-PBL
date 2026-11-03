"""
Unit tests for STFT Spectral Gating and Decision-Directed Audio Wiener Filtering.
"""

import numpy as np
import pytest

from src.audio_io import generate_clean_speech_synthetic
from src.noise_generation import generate_broadband_noise, mix_speech_and_noise
from src.spectral_denoising import spectral_gating, wiener_filter_audio, estimate_noise_profile


def test_spectral_gating_runs_and_preserves_shape():
    sr = 16000
    speech = generate_clean_speech_synthetic(duration=1.5, sr=sr, seed=42)
    noise = generate_broadband_noise(len(speech), sr=sr, noise_type="white", seed=42)
    noisy, _, _ = mix_speech_and_noise(speech, noise, target_snr_db=5.0)

    denoised, info = spectral_gating(
        noisy, sr=sr, suppression_db=-18.0, over_subtraction=1.5,
        time_smoothing=2, freq_smoothing=2
    )

    assert len(denoised) == len(noisy)
    assert denoised.dtype == np.float32
    assert np.all(np.isfinite(denoised))
    assert info["method"] == "Spectral Gating"
    assert "gain_floor" in info


def test_wiener_filter_audio_runs_and_preserves_shape():
    sr = 16000
    speech = generate_clean_speech_synthetic(duration=1.5, sr=sr, seed=42)
    noise = generate_broadband_noise(len(speech), sr=sr, noise_type="pink", seed=42)
    noisy, _, _ = mix_speech_and_noise(speech, noise, target_snr_db=5.0)

    denoised, info = wiener_filter_audio(
        noisy, sr=sr, alpha=0.95, overestimation=1.2, gain_floor_db=-20.0
    )

    assert len(denoised) == len(noisy)
    assert denoised.dtype == np.float32
    assert np.all(np.isfinite(denoised))
    assert info["method"] == "Audio Wiener Filter (Decision-Directed)"
    assert info["alpha_smoothing"] == 0.95


def test_noise_profile_estimation():
    sr = 16000
    sig = np.random.randn(sr * 2).astype(np.float32)

    # Automatic quiet frames
    psd_auto, mag_auto = estimate_noise_profile(sig, sr=sr, n_fft=512)
    assert len(psd_auto) == 512 // 2 + 1
    assert np.all(psd_auto > 0)

    # Explicit user segment
    psd_seg, mag_seg = estimate_noise_profile(sig, sr=sr, n_fft=512, noise_segment=(0.0, 0.5))
    assert len(psd_seg) == 512 // 2 + 1
    assert np.all(psd_seg > 0)
