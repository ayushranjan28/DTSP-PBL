"""
Unit tests for FIR, IIR Butterworth, and Notch filters.
"""

import numpy as np
import pytest

from src.filters import (
    apply_fir_filter,
    apply_butterworth_filter,
    apply_notch_filter,
    compute_fir_response,
    compute_iir_sos_response,
    compute_notch_response
)


def test_fir_lowpass_attenuation():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Low frequency tone (300 Hz) + High frequency tone (5000 Hz)
    sig_low = np.sin(2 * np.pi * 300 * t)
    sig_high = np.sin(2 * np.pi * 5000 * t)
    combined = sig_low + sig_high

    # Filter with low-pass cutoff at 1000 Hz
    filtered, taps, info = apply_fir_filter(
        combined, cutoff=1000.0, num_taps=101, btype="lowpass", sr=sr, zero_phase=True
    )

    assert len(filtered) == len(combined)
    assert np.all(np.isfinite(filtered))

    # Verify high frequency is heavily attenuated
    fft_in = np.abs(np.fft.rfft(combined))
    fft_out = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(combined), 1.0 / sr)

    idx_low = np.argmin(np.abs(freqs - 300))
    idx_high = np.argmin(np.abs(freqs - 5000))

    # Low frequency should be preserved near 0 dB
    assert fft_out[idx_low] / fft_in[idx_low] > 0.85
    # High frequency should be attenuated by at least 20 dB (factor of 0.1)
    assert fft_out[idx_high] / fft_in[idx_high] < 0.05


def test_butterworth_highpass_attenuation():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig_rumble = np.sin(2 * np.pi * 60 * t)
    sig_voice = np.sin(2 * np.pi * 1000 * t)
    combined = sig_rumble + sig_voice

    filtered, sos, info = apply_butterworth_filter(
        combined, cutoff=150.0, order=4, btype="highpass", sr=sr, zero_phase=True
    )

    assert len(filtered) == len(combined)
    assert np.all(np.isfinite(filtered))

    fft_in = np.abs(np.fft.rfft(combined))
    fft_out = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(combined), 1.0 / sr)

    idx_rumble = np.argmin(np.abs(freqs - 60))
    idx_voice = np.argmin(np.abs(freqs - 1000))

    # Voice preserved
    assert fft_out[idx_voice] / fft_in[idx_voice] > 0.90
    # Rumble attenuated
    assert fft_out[idx_rumble] / fft_in[idx_rumble] < 0.05


def test_notch_filter_attenuates_50hz():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    hum = np.sin(2 * np.pi * 50 * t)
    speech = np.sin(2 * np.pi * 400 * t)
    signal_in = hum + speech

    filtered, coeffs, info = apply_notch_filter(
        signal_in, notch_freq=50.0, q=30.0, include_harmonics=False, sr=sr, zero_phase=True
    )

    assert len(filtered) == len(signal_in)
    assert np.all(np.isfinite(filtered))

    fft_in = np.abs(np.fft.rfft(signal_in))
    fft_out = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(signal_in), 1.0 / sr)

    idx_50 = np.argmin(np.abs(freqs - 50))
    idx_400 = np.argmin(np.abs(freqs - 400))

    # 50 Hz attenuated by at least 20 dB (factor < 0.10)
    assert fft_out[idx_50] / fft_in[idx_50] < 0.10
    # 400 Hz preserved
    assert fft_out[idx_400] / fft_in[idx_400] > 0.95


def test_filter_parameter_validation():
    # Cutoff above Nyquist must raise ValueError
    with pytest.raises(ValueError):
        apply_fir_filter(np.zeros(100), cutoff=9000.0, sr=16000)

    with pytest.raises(ValueError):
        apply_butterworth_filter(np.zeros(100), cutoff=9000.0, sr=16000)


def test_frequency_response_functions():
    sr = 16000
    _, taps, _ = apply_fir_filter(np.zeros(100), cutoff=1000.0, sr=sr)
    w_fir, mag_fir, phase_fir = compute_fir_response(taps, sr=sr)
    assert len(w_fir) > 0
    assert np.all(np.isfinite(mag_fir))

    _, sos, _ = apply_butterworth_filter(np.zeros(100), cutoff=1000.0, sr=sr)
    w_iir, mag_iir, phase_iir = compute_iir_sos_response(sos, sr=sr)
    assert len(w_iir) > 0
    assert np.all(np.isfinite(mag_iir))

    _, coeffs, _ = apply_notch_filter(np.zeros(100), notch_freq=50.0, sr=sr)
    w_notch, mag_notch = compute_notch_response(coeffs, sr=sr)
    assert len(w_notch) > 0
    assert np.all(np.isfinite(mag_notch))
