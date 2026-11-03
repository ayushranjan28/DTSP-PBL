"""
Unit tests for noise generation and calibrated SNR audio mixing.
"""

import numpy as np
import pytest

from src.audio_io import generate_clean_speech_synthetic
from src.noise_generation import (
    generate_electrical_hum,
    generate_low_frequency_rumble,
    generate_high_frequency_hiss,
    generate_broadband_noise,
    generate_mixed_noise,
    mix_speech_and_noise
)


def test_noise_generators_output_shape_and_range():
    sr = 16000
    n_samples = sr * 2

    hum = generate_electrical_hum(n_samples, sr=sr, base_freq=50.0)
    assert len(hum) == n_samples
    assert np.all(np.isfinite(hum))
    assert np.isclose(np.sqrt(np.mean(hum ** 2)), 1.0, atol=1e-2)

    rumble = generate_low_frequency_rumble(n_samples, sr=sr, cutoff_hz=120.0)
    assert len(rumble) == n_samples
    assert np.all(np.isfinite(rumble))
    assert np.isclose(np.sqrt(np.mean(rumble ** 2)), 1.0, atol=1e-2)

    hiss = generate_high_frequency_hiss(n_samples, sr=sr, cutoff_hz=3500.0)
    assert len(hiss) == n_samples
    assert np.all(np.isfinite(hiss))
    assert np.isclose(np.sqrt(np.mean(hiss ** 2)), 1.0, atol=1e-2)

    white = generate_broadband_noise(n_samples, sr=sr, noise_type="white")
    assert len(white) == n_samples
    assert np.all(np.isfinite(white))
    assert np.isclose(np.sqrt(np.mean(white ** 2)), 1.0, atol=1e-2)

    pink = generate_broadband_noise(n_samples, sr=sr, noise_type="pink")
    assert len(pink) == n_samples
    assert np.all(np.isfinite(pink))
    assert np.isclose(np.sqrt(np.mean(pink ** 2)), 1.0, atol=1e-2)

    mixed = generate_mixed_noise(n_samples, sr=sr)
    assert len(mixed) == n_samples
    assert np.all(np.isfinite(mixed))
    assert np.isclose(np.sqrt(np.mean(mixed ** 2)), 1.0, atol=1e-2)


@pytest.mark.parametrize("target_snr", [-5.0, 0.0, 5.0, 10.0, 20.0])
def test_mix_speech_and_noise_exact_snr(target_snr):
    sr = 16000
    speech = generate_clean_speech_synthetic(duration=2.0, sr=sr, seed=42)
    noise = generate_broadband_noise(len(speech), sr=sr, noise_type="white", seed=42)

    mixture, scaled_noise, meta = mix_speech_and_noise(speech, noise, target_snr_db=target_snr)

    assert len(mixture) == len(speech)
    assert np.all(np.isfinite(mixture))
    assert meta["target_snr_db"] == target_snr
    # Tolerance of 0.2 dB on unclipped mixtures
    if not meta["peak_normalized_for_clipping"]:
        assert abs(meta["measured_snr_db"] - target_snr) < 0.2
