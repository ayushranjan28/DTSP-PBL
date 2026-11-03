"""
Audio Feature Extraction Module
Extracts physical, spectral, and noise-floor isolated features for machine learning noise classification.
Implements:
  1. Peak-invariant amplitude normalization.
  2. Noise-floor frame isolation (lowest RMS energy frames) so active speech formants
     do not mask background noise signatures.
  3. High-resolution FFT comb-harmonic local prominence for 50/60 Hz hum vs. broadband rumble.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import scipy.signal as signal
import librosa

FEATURE_NAMES = [
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "energy_ratio_rumble",      # 0 - 150 Hz
    "energy_ratio_low",         # 150 - 500 Hz
    "energy_ratio_mid",         # 500 - 3000 Hz
    "energy_ratio_high",        # 3000 - 6000 Hz
    "energy_ratio_air",         # 6000 - 8000 Hz
    "dominant_peak_freq",       # Frequency of maximum spectral peak
    "dominant_peak_ratio",      # Ratio of peak power to mean power
    "hum_50hz_ratio",           # Narrowband comb-prominence weighted 50 Hz hum ratio
    "hum_60hz_ratio",           # Narrowband comb-prominence weighted 60 Hz hum ratio
    "spectral_bandwidth_mean"
]


def extract_features_from_audio(
    audio: np.ndarray,
    sr: int = 16000,
    frame_length: int = 1024,
    hop_length: int = 256
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract comprehensive acoustic and noise-floor isolated feature vector from audio signal.

    Parameters:
        audio: 1D audio array.
        sr: Sample rate in Hz.
        frame_length: Analysis frame length.
        hop_length: Hop length between frames.

    Returns:
        feature_vector: 1D numpy array of ordered features.
        feature_dict: Dictionary mapping feature names to scalar values.
    """
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))

    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0)

    # Peak-normalize for level-invariant feature extraction across external datasets
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-7:
        y = audio / peak
    else:
        y = audio.copy()

    # 1. Frame-level RMS Energy & Continuous Noise-Floor Weighting (No Hardcoded Thresholds)
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # Exclude pure digital zero-padding frames (e.g. zero-padded lead-in/lead-out in benchmark files)
    non_zero_indices = np.where(rms_frames > 2e-4)[0]
    active_rms = rms_frames[non_zero_indices] if len(non_zero_indices) >= 5 else rms_frames

    p20_rms = float(np.percentile(active_rms, 20))
    p90_rms = float(np.percentile(active_rms, 90)) + 1e-8
    floor_to_peak_ratio = float(np.clip(p20_rms / p90_rms, 0.0, 1.0))

    rms_mean = float(floor_to_peak_ratio)
    rms_std = float(np.std(active_rms) / (np.mean(active_rms) + 1e-8))

    # Extract lowest 25% active RMS frames for background estimation using smooth nonlinear tanh weighting
    thresh_25 = np.percentile(active_rms, 25)
    if len(non_zero_indices) >= 5:
        quiet_indices = non_zero_indices[rms_frames[non_zero_indices] <= thresh_25]
    else:
        quiet_indices = np.where(rms_frames <= thresh_25)[0]
    w_floor = float(np.tanh(floor_to_peak_ratio * 5.0) * 0.80)

    # 2. Zero-Crossing Rate (continuously weighted by w_floor)
    zcr_all = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_quiet = zcr_all[quiet_indices] if len(quiet_indices) > 0 else zcr_all
    zcr_mean = float(w_floor * np.mean(zcr_quiet) + (1.0 - w_floor) * np.mean(zcr_all))
    zcr_std = float(np.std(zcr_all))

    # 3. Spectral Centroid, Flatness, Roll-off, Bandwidth (continuously weighted by w_floor)
    centroid_all = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    flatness_all = librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop_length)[0]
    rolloff_all = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, roll_percent=0.85)[0]
    bandwidth_all = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

    if len(quiet_indices) > 0:
        spectral_centroid_mean = float(w_floor * np.mean(centroid_all[quiet_indices]) + (1.0 - w_floor) * np.mean(centroid_all))
        spectral_flatness_mean = float(w_floor * np.mean(flatness_all[quiet_indices]) + (1.0 - w_floor) * np.mean(flatness_all))
        spectral_rolloff_mean = float(w_floor * np.mean(rolloff_all[quiet_indices]) + (1.0 - w_floor) * np.mean(rolloff_all))
        spectral_bandwidth_mean = float(w_floor * np.mean(bandwidth_all[quiet_indices]) + (1.0 - w_floor) * np.mean(bandwidth_all))
    else:
        spectral_centroid_mean = float(np.mean(centroid_all))
        spectral_flatness_mean = float(np.mean(flatness_all))
        spectral_rolloff_mean = float(np.mean(rolloff_all))
        spectral_bandwidth_mean = float(np.mean(bandwidth_all))

    spectral_centroid_std = float(np.std(centroid_all))
    spectral_flatness_std = float(np.std(flatness_all))
    spectral_rolloff_std = float(np.std(rolloff_all))

    # 4. Continuous STFT Noise-Floor PSD + High-Resolution Welch PSD (nfft=4096)
    nfft = 4096
    nperseg_global = min(len(y), 4096)
    freqs, psd_global = signal.welch(y, fs=sr, nperseg=nperseg_global, nfft=nfft)
    psd_g_norm = psd_global / (np.sum(psd_global) + 1e-12)

    S_power = np.abs(librosa.stft(y, n_fft=nfft, win_length=frame_length, hop_length=hop_length)) ** 2
    valid_q = quiet_indices[quiet_indices < S_power.shape[1]]
    if len(valid_q) > 0:
        psd_floor = np.mean(S_power[:, valid_q], axis=1)
        psd_f_norm = psd_floor / (np.sum(psd_floor) + 1e-12)
        psd = w_floor * psd_f_norm + (1.0 - w_floor) * psd_g_norm
    else:
        psd = psd_g_norm

    total_power = float(np.sum(psd)) + 1e-12

    # Band ratios
    mask_rumble = (freqs >= 0) & (freqs < 150)
    energy_ratio_rumble = float(np.sum(psd[mask_rumble]) / total_power)

    mask_low = (freqs >= 150) & (freqs < 500)
    energy_ratio_low = float(np.sum(psd[mask_low]) / total_power)

    mask_mid = (freqs >= 500) & (freqs < 3000)
    energy_ratio_mid = float(np.sum(psd[mask_mid]) / total_power)

    mask_high = (freqs >= 3000) & (freqs < 6000)
    energy_ratio_high = float(np.sum(psd[mask_high]) / total_power)

    mask_air = (freqs >= 6000) & (freqs <= sr / 2.0)
    energy_ratio_air = float(np.sum(psd[mask_air]) / total_power)

    # 5. Dominant Spectral Peak
    peak_idx = int(np.argmax(psd))
    dominant_peak_freq = float(freqs[peak_idx])
    mean_psd = float(np.mean(psd)) + 1e-12
    dominant_peak_ratio = float(psd[peak_idx] / mean_psd)

    # 6. Narrowband Comb-Harmonic Prominence for 50 Hz and 60 Hz Hum
    # Use global high-resolution PSD so coherent sinusoidal hum maintains maximal frequency resolution
    psd_gl_norm = psd_global / (np.sum(psd_global) + 1e-12)

    # 50 Hz fundamental + harmonics (50, 100, 150, 200 Hz) vs off-harmonic guard bands
    mask_50_h = (
        ((freqs >= 46.0) & (freqs <= 54.0)) |
        ((freqs >= 96.0) & (freqs <= 104.0)) |
        ((freqs >= 146.0) & (freqs <= 154.0)) |
        ((freqs >= 196.0) & (freqs <= 204.0))
    )
    mask_50_guard = (
        ((freqs >= 28.0) & (freqs <= 42.0)) |
        ((freqs >= 66.0) & (freqs <= 88.0)) |
        ((freqs >= 112.0) & (freqs <= 138.0))
    )
    p_50_peak = float(np.max(psd_gl_norm[(freqs >= 46.0) & (freqs <= 54.0)]))
    p_50_guard = float(np.median(psd_gl_norm[mask_50_guard])) + 1e-12
    prominence_50 = p_50_peak / p_50_guard
    # Gate by local prominence: smooth broadband rumble has prominence ~ 1-3, true hum has > 8
    gate_50 = float(np.tanh(max(0.0, prominence_50 - 3.0) / 5.0))
    hum_50hz_ratio = float(np.sum(psd_gl_norm[mask_50_h]) * gate_50)

    # 60 Hz fundamental + harmonics (60, 120, 180, 240 Hz) vs off-harmonic guard bands
    mask_60_h = (
        ((freqs >= 56.0) & (freqs <= 64.0)) |
        ((freqs >= 116.0) & (freqs <= 124.0)) |
        ((freqs >= 176.0) & (freqs <= 184.0)) |
        ((freqs >= 236.0) & (freqs <= 244.0))
    )
    mask_60_guard = (
        ((freqs >= 36.0) & (freqs <= 50.0)) |
        ((freqs >= 72.0) & (freqs <= 106.0)) |
        ((freqs >= 132.0) & (freqs <= 166.0))
    )
    p_60_peak = float(np.max(psd_gl_norm[(freqs >= 56.0) & (freqs <= 64.0)]))
    p_60_guard = float(np.median(psd_gl_norm[mask_60_guard])) + 1e-12
    prominence_60 = p_60_peak / p_60_guard
    gate_60 = float(np.tanh(max(0.0, prominence_60 - 3.0) / 5.0))
    hum_60hz_ratio = float(np.sum(psd_gl_norm[mask_60_h]) * gate_60)

    feature_dict = {
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_centroid_std": spectral_centroid_std,
        "spectral_flatness_mean": spectral_flatness_mean,
        "spectral_flatness_std": spectral_flatness_std,
        "spectral_rolloff_mean": spectral_rolloff_mean,
        "spectral_rolloff_std": spectral_rolloff_std,
        "energy_ratio_rumble": energy_ratio_rumble,
        "energy_ratio_low": energy_ratio_low,
        "energy_ratio_mid": energy_ratio_mid,
        "energy_ratio_high": energy_ratio_high,
        "energy_ratio_air": energy_ratio_air,
        "dominant_peak_freq": dominant_peak_freq,
        "dominant_peak_ratio": dominant_peak_ratio,
        "hum_50hz_ratio": hum_50hz_ratio,
        "hum_60hz_ratio": hum_60hz_ratio,
        "spectral_bandwidth_mean": spectral_bandwidth_mean
    }

    feature_vector = np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)

    return feature_vector, feature_dict
