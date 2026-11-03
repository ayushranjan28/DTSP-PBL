"""
Quantitative Evaluation Module
Provides reference-based metrics (SNR, SNR improvement, MSE, RMSE, Pearson r, RTF)
and unpaired descriptive spectral indicators when no reference is available.
"""

from typing import Dict, Any, Optional
import time
import numpy as np
import scipy.signal as signal
from scipy.stats import pearsonr


def compute_reference_based_metrics(
    clean_ref: np.ndarray,
    noisy_input: np.ndarray,
    processed: np.ndarray,
    sr: int = 16000,
    processing_time_sec: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate objective reference-based signal metrics against an aligned clean reference.

    Conventions:
        - clean_ref: True speech reference signal s[n].
        - noisy_input: Unprocessed input y[n] = s[n] + v[n].
        - processed: Filtered output signal s_hat[n].
        - All signals aligned and of equal length. No arbitrary independent re-normalization
          is applied to avoid masking attenuation or linear scaling distortion.

    Equations:
        Input Error: e_in[n] = y[n] - s[n]
        Output Error: e_out[n] = s_hat[n] - s[n]
        SNR_in = 10 * log10( sum(s^2) / sum(e_in^2) )
        SNR_out = 10 * log10( sum(s^2) / sum(e_out^2) )
        Delta_SNR = SNR_out - SNR_in
        MSE = mean( (s_hat - s)^2 )
        RMSE = sqrt( MSE )
        Pearson_r = cov(s_hat, s) / (std(s_hat) * std(s))
        RTF = processing_time / duration
    """
    # Ensure equal lengths
    min_len = min(len(clean_ref), len(noisy_input), len(processed))
    s = clean_ref[:min_len].astype(np.float64)
    y = noisy_input[:min_len].astype(np.float64)
    s_hat = processed[:min_len].astype(np.float64)

    duration = float(min_len) / float(sr) if sr > 0 else 0.0
    eps = 1e-12

    # Clean speech power
    p_speech = np.sum(s ** 2)

    # Input SNR
    e_in = y - s
    p_noise_in = np.sum(e_in ** 2)
    input_snr_db = float(10.0 * np.log10(p_speech / (p_noise_in + eps))) if p_noise_in > eps else 99.0

    # Output SNR
    e_out = s_hat - s
    p_noise_out = np.sum(e_out ** 2)
    output_snr_db = float(10.0 * np.log10(p_speech / (p_noise_out + eps))) if p_noise_out > eps else 99.0

    # SNR Improvement
    snr_improvement_db = float(output_snr_db - input_snr_db)

    # MSE & RMSE
    mse = float(np.mean(e_out ** 2))
    rmse = float(np.sqrt(mse))

    # Pearson correlation coefficient
    std_s = np.std(s)
    std_shat = np.std(s_hat)
    if std_s > 1e-9 and std_shat > 1e-9:
        r_val, _ = pearsonr(s, s_hat)
        pearson_corr = float(r_val)
    else:
        pearson_corr = 0.0

    # Real-Time Factor
    rtf = float(processing_time_sec / duration) if duration > 0 else 0.0

    return {
        "evaluation_mode": "Reference-Based (Clean Reference Available)",
        "has_reference": True,
        "input_snr_db": round(input_snr_db, 2),
        "output_snr_db": round(output_snr_db, 2),
        "snr_improvement_db": round(snr_improvement_db, 2),
        "mse": float(f"{mse:.6e}"),
        "rmse": float(f"{rmse:.6e}"),
        "pearson_correlation": round(pearson_corr, 4),
        "processing_time_sec": round(processing_time_sec, 4),
        "audio_duration_sec": round(duration, 3),
        "real_time_factor": round(rtf, 4),
        "is_real_time_capable": rtf < 1.0
    }


def compute_unpaired_metrics(
    noisy_input: np.ndarray,
    processed: np.ndarray,
    sr: int = 16000,
    processing_time_sec: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate descriptive acoustic indicators when NO clean reference exists.
    Strictly avoids fabricating subjective speech quality scores or false SNR values.

    Parameters:
        noisy_input: Input recording.
        processed: Filtered audio.
        sr: Sample rate in Hz.
        processing_time_sec: Execution runtime.

    Returns:
        Dict of descriptive spectral, dynamic, and computational statistics.
    """
    min_len = min(len(noisy_input), len(processed))
    y = noisy_input[:min_len]
    s_hat = processed[:min_len]
    duration = float(min_len) / float(sr) if sr > 0 else 0.0

    # RMS levels (dBFS)
    rms_in = float(np.sqrt(np.mean(y ** 2)))
    rms_out = float(np.sqrt(np.mean(s_hat ** 2)))
    rms_in_dbfs = float(20 * np.log10(rms_in + 1e-9))
    rms_out_dbfs = float(20 * np.log10(rms_out + 1e-9))
    overall_attenuation_db = float(rms_in_dbfs - rms_out_dbfs)

    # Power Spectral Densities via Welch's method
    nperseg = min(len(s_hat), 1024)
    freqs, psd_in = signal.welch(y, fs=sr, nperseg=nperseg)
    _, psd_out = signal.welch(s_hat, fs=sr, nperseg=nperseg)

    eps = 1e-12
    # Sub-band attenuation measurements
    # Rumble attenuation (0 - 150 Hz)
    mask_rumble = (freqs >= 0) & (freqs < 150)
    p_rumble_in = np.sum(psd_in[mask_rumble])
    p_rumble_out = np.sum(psd_out[mask_rumble])
    rumble_attenuation_db = float(10 * np.log10((p_rumble_in + eps) / (p_rumble_out + eps)))

    # 50 Hz Hum attenuation (47 - 53 Hz)
    mask_50 = (freqs >= 47) & (freqs <= 53)
    p_50_in = np.sum(psd_in[mask_50])
    p_50_out = np.sum(psd_out[mask_50])
    hum_attenuation_db = float(10 * np.log10((p_50_in + eps) / (p_50_out + eps)))

    # Hiss attenuation (> 3500 Hz)
    mask_hiss = (freqs >= 3500) & (freqs <= sr / 2.0)
    p_hiss_in = np.sum(psd_in[mask_hiss])
    p_hiss_out = np.sum(psd_out[mask_hiss])
    hiss_attenuation_db = float(10 * np.log10((p_hiss_in + eps) / (p_hiss_out + eps)))

    # RTF
    rtf = float(processing_time_sec / duration) if duration > 0 else 0.0

    return {
        "evaluation_mode": "Unpaired Descriptive (No Clean Reference)",
        "has_reference": False,
        "input_rms_dbfs": round(rms_in_dbfs, 2),
        "output_rms_dbfs": round(rms_out_dbfs, 2),
        "total_attenuation_db": round(overall_attenuation_db, 2),
        "rumble_band_attenuation_db": round(rumble_attenuation_db, 2),
        "hum_50hz_attenuation_db": round(hum_attenuation_db, 2),
        "hiss_band_attenuation_db": round(hiss_attenuation_db, 2),
        "processing_time_sec": round(processing_time_sec, 4),
        "audio_duration_sec": round(duration, 3),
        "real_time_factor": round(rtf, 4),
        "is_real_time_capable": rtf < 1.0,
        "interpretability_note": (
            "No clean reference provided. Metrics report physical energy attenuation across frequency bands, "
            "not objective speech quality improvements."
        )
    }
