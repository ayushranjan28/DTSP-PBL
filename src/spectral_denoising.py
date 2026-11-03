"""
Spectral Denoising Module
Implements STFT-based Spectral Gating (with time-frequency smoothing and noise profiling)
and true audio Wiener filtering via Ephraim-Malah Decision-Directed a priori SNR estimation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d, gaussian_filter


def estimate_noise_profile(
    y: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    noise_segment: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the noise power spectral density (PSD) and mean magnitude profile
    from a specified noise-only time window or automatically from quiet frames.

    Parameters:
        y: 1D audio array.
        sr: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop size for STFT.
        noise_segment: Optional (start_sec, end_sec) tuple.

    Returns:
        noise_psd: 1D array of noise power per frequency bin (n_fft // 2 + 1,).
        noise_mag: 1D array of mean noise magnitude per frequency bin.
    """
    # Compute STFT
    window = signal.windows.hann(n_fft, sym=False)
    _, _, zxx = signal.stft(y, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)
    mag = np.abs(zxx)
    power = mag ** 2

    n_frames = mag.shape[1]

    if noise_segment is not None:
        start_sec, end_sec = noise_segment
        start_frame = max(0, int(start_sec * sr / hop_length))
        end_frame = min(n_frames, int(end_sec * sr / hop_length))
        if end_frame > start_frame:
            selected_power = power[:, start_frame:end_frame]
            selected_mag = mag[:, start_frame:end_frame]
        else:
            # Fallback if range invalid
            selected_power = power[:, :min(n_frames, 10)]
            selected_mag = mag[:, :min(n_frames, 10)]
    else:
        # Automatic: exclude dead digital zero-padding / muted lead-in frames first,
        # then select the lowest 35th percentile of active recording frames
        frame_energies = np.sum(power, axis=0)
        p90_energy = np.percentile(frame_energies, 90)
        active_mask = frame_energies > (0.002 * p90_energy)
        active_indices = np.where(active_mask)[0]
        if len(active_indices) >= 5:
            threshold = np.percentile(frame_energies[active_indices], 35)
            quiet_indices = active_indices[frame_energies[active_indices] <= threshold]
        else:
            threshold = np.percentile(frame_energies, 30)
            quiet_indices = np.where(frame_energies <= threshold)[0]

        if len(quiet_indices) < 3:
            quiet_indices = np.argsort(frame_energies)[:max(3, n_frames // 10)]

        selected_power = power[:, quiet_indices]
        selected_mag = mag[:, quiet_indices]

    # Use 80th percentile across quiet frames so fluctuating background noise peaks
    # in speech pauses are captured rather than underestimated by the mean
    noise_psd = np.percentile(selected_power, 80, axis=1).astype(np.float32)
    noise_mag = np.percentile(selected_mag, 80, axis=1).astype(np.float32)

    # Floor to avoid division by zero
    noise_psd = np.maximum(noise_psd, 1e-12)
    noise_mag = np.maximum(noise_mag, 1e-6)

    return noise_psd, noise_mag


def spectral_gating(
    audio: np.ndarray,
    sr: int = 16000,
    noise_segment: Optional[Tuple[float, float]] = None,
    n_fft: int = 512,
    hop_length: int = 128,
    suppression_db: float = -18.0,
    over_subtraction: float = 1.5,
    time_smoothing: int = 2,
    freq_smoothing: int = 2
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform STFT Spectral Gating with frequency-dependent noise thresholding,
    over-subtraction factor, and 2D mask smoothing.

    Parameters:
        audio: 1D audio array.
        sr: Sample rate in Hz.
        noise_segment: Optional (start_sec, end_sec) noise-only window.
        n_fft: FFT window length.
        hop_length: Hop length between frames.
        suppression_db: Minimum gain floor in decibels (e.g. -12 dB to -30 dB).
        over_subtraction: Multiplier for estimated noise threshold (>1.0 reduces residue).
        time_smoothing: Kernel size for smoothing mask along time axis.
        freq_smoothing: Kernel size for smoothing mask along frequency axis.

    Returns:
        denoised_audio: 1D float32 numpy array.
        info: Dict with processing metadata and DSP commentary.
    """
    window = signal.windows.hann(n_fft, sym=False)
    f, t, zxx = signal.stft(audio, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)

    mag = np.abs(zxx)
    phase = np.angle(zxx)

    _, noise_mag = estimate_noise_profile(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, noise_segment=noise_segment)

    # Threshold per frequency bin: over_subtraction * noise_mag
    threshold = (over_subtraction * noise_mag)[:, np.newaxis]

    # Gain floor: linear multiplier corresponding to suppression_db
    gain_floor = 10.0 ** (suppression_db / 20.0)

    # Spectral gate mask calculation
    # Smooth sigmoid or power-subtraction gating
    snr_sub = np.maximum(0.0, (mag - threshold) / (mag + 1e-9))
    mask = np.maximum(gain_floor, snr_sub)

    # Apply 2D smoothing to mitigate "musical noise" (isolated random speckles)
    if time_smoothing > 0 or freq_smoothing > 0:
        sigma = (max(0.5, float(freq_smoothing) / 2.0), max(0.5, float(time_smoothing) / 2.0))
        mask = gaussian_filter(mask, sigma=sigma)
        mask = np.clip(mask, gain_floor, 1.0)

    # Reconstruct modified STFT
    filtered_mag = mag * mask
    filtered_zxx = filtered_mag * np.exp(1j * phase)

    _, denoised = signal.istft(filtered_zxx, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Align length with input
    if len(denoised) > len(audio):
        denoised = denoised[:len(audio)]
    elif len(denoised) < len(audio):
        denoised = np.pad(denoised, (0, len(audio) - len(denoised)))

    info = {
        "method": "Spectral Gating",
        "n_fft": n_fft,
        "hop_length": hop_length,
        "suppression_db": suppression_db,
        "over_subtraction": over_subtraction,
        "time_smoothing": time_smoothing,
        "freq_smoothing": freq_smoothing,
        "gain_floor": round(float(gain_floor), 4),
        "noise_profile_source": "user_segment" if noise_segment else "auto_quiet_percentile",
        "dsp_note": (
            f"STFT Spectral Gate with {suppression_db} dB gain floor and over-subtraction {over_subtraction}. "
            "2D Gaussian time-frequency smoothing reduces musical noise artefacts. "
            "Caution: Extreme attenuation can attenuate quiet fricatives and unvoiced consonants."
        )
    }

    return denoised.astype(np.float32), info


def wiener_filter_audio(
    audio: np.ndarray,
    sr: int = 16000,
    noise_segment: Optional[Tuple[float, float]] = None,
    n_fft: int = 512,
    hop_length: int = 128,
    alpha: float = 0.95,
    overestimation: float = 1.2,
    gain_floor_db: float = -20.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    True audio-domain Wiener Filter using the Ephraim-Malah Decision-Directed (DD)
    approach for tracking non-stationary speech and suppressing stationary noise.

    Formulation:
        Y(t, f) = STFT(noisy_audio)
        Noise PSD P_N(f) estimated from quiet frames or noise segment.
        Posterior SNR: gamma(t, f) = |Y(t, f)|^2 / (beta * P_N(f))
        Prior SNR (Decision-Directed):
            xi(t, f) = alpha * (|S_prev(t-1, f)|^2 / P_N(f)) + (1 - alpha) * max(gamma(t, f) - 1, 0)
        Wiener Gain Transfer Function:
            H(t, f) = max( xi(t, f) / (xi(t, f) + 1), G_min )
        Clean Estimate: S_hat(t, f) = H(t, f) * Y(t, f)

    Parameters:
        audio: 1D audio array.
        sr: Sample rate in Hz.
        noise_segment: Optional (start_sec, end_sec) tuple.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        alpha: Decision-Directed smoothing parameter (typically 0.92 to 0.98).
        overestimation: Noise overestimation factor beta (1.0 to 2.5).
        gain_floor_db: Minimum allowable filter gain in dB to avoid musical noise.

    Returns:
        denoised_audio: 1D float32 numpy array.
        info: Dict with processing metadata and DSP commentary.
    """
    window = signal.windows.hann(n_fft, sym=False)
    f, t, zxx = signal.stft(audio, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)

    mag = np.abs(zxx)
    phase = np.angle(zxx)
    power = mag ** 2

    n_bins, n_frames = mag.shape
    noise_psd, _ = estimate_noise_profile(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, noise_segment=noise_segment)

    # Frequency-dependent psychoacoustic weighting: apply stronger noise suppression
    # below fundamental voice pitch (<130 Hz) and above core formants (>4200 Hz)
    freq_weight = np.ones(n_bins, dtype=np.float32)
    freq_weight[f < 130.0] = 1.8
    freq_weight[f > 4200.0] = 1.6

    # Scaled noise PSD
    scaled_noise_psd = overestimation * noise_psd * freq_weight
    gain_min = 10.0 ** (gain_floor_db / 20.0)

    # Initialize matrices
    gain = np.zeros((n_bins, n_frames), dtype=np.float32)
    s_hat_mag_prev = np.zeros(n_bins, dtype=np.float32)

    # Use responsive alpha (capped at 0.86) to prevent reverberant noise tails after speech syllables
    eff_alpha = min(float(alpha), 0.86)

    # Decision-Directed Recursive Loop across time frames
    for i in range(n_frames):
        p_frame = power[:, i]

        # A posteriori SNR
        gamma = np.maximum(p_frame / (scaled_noise_psd + 1e-12), 1e-4)

        if i == 0:
            # Initial frame a priori SNR estimate
            xi = np.maximum(gamma - 1.0, 0.0)
        else:
            # Decision-directed combination of previous frame output and current measurement
            prev_snr = (s_hat_mag_prev ** 2) / (scaled_noise_psd + 1e-12)
            xi = eff_alpha * prev_snr + (1.0 - eff_alpha) * np.maximum(gamma - 1.0, 0.0)

        # Parametric Wiener transfer function H = (xi / (xi + 1))^1.25 for sharper background suppression
        h = (xi / (xi + 1.0 + 1e-9)) ** 1.25
        h = np.clip(h, gain_min, 1.0)

        gain[:, i] = h
        s_hat_mag_prev = h * mag[:, i]

    # Apply 2D Gaussian time-frequency smoothing to eliminate musical noise speckles
    gain = gaussian_filter(gain, sigma=(1.0, 1.2))
    gain = np.clip(gain, gain_min, 1.0)

    # Apply gain to complex STFT
    clean_zxx = gain * zxx

    # ISTFT reconstruction
    _, denoised = signal.istft(clean_zxx, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Align length
    if len(denoised) > len(audio):
        denoised = denoised[:len(audio)]
    elif len(denoised) < len(audio):
        denoised = np.pad(denoised, (0, len(audio) - len(denoised)))

    info = {
        "method": "Audio Wiener Filter (Decision-Directed)",
        "alpha_smoothing": alpha,
        "overestimation_beta": overestimation,
        "gain_floor_db": gain_floor_db,
        "gain_floor_linear": round(float(gain_min), 4),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "noise_profile_source": "user_segment" if noise_segment else "auto_quiet_percentile",
        "dsp_note": (
            f"Frequency-domain audio Wiener filter with Ephraim-Malah Decision-Directed recursion (alpha={alpha}). "
            f"A priori SNR tracking minimizes musical noise while maintaining speech intelligibility. "
            f"Noise overestimation factor = {overestimation}, gain floor = {gain_floor_db} dB."
        )
    }

    return denoised.astype(np.float32), info
