"""
Classical Digital Signal Processing Filters Module
Implements FIR (window-based linear phase), IIR Butterworth (second-order sections),
and single/harmonic IIR Notch filters with causal and zero-phase options.
"""

from typing import Tuple, Union, List, Dict, Any, Optional
import numpy as np
import scipy.signal as signal


# ============================================================================
# A. FIR FILTERING (scipy.signal.firwin)
# ============================================================================

def design_fir_filter(
    cutoff: Union[float, Tuple[float, float], List[float]],
    num_taps: int = 101,
    btype: str = "lowpass",
    window: str = "hamming",
    sr: int = 16000
) -> np.ndarray:
    """
    Design a linear-phase Finite Impulse Response (FIR) filter using the window method.

    Parameters:
        cutoff: Cutoff frequency in Hz.
                - Float for 'lowpass' or 'highpass'.
                - (low, high) tuple/list for 'bandpass'.
        num_taps: Number of filter coefficients (taps). Must be positive.
                  For highpass/bandpass filters with attenuation at Nyquist,
                  num_taps is automatically made odd (Type I linear phase).
        btype: 'lowpass', 'highpass', or 'bandpass'.
        window: Window function ('hamming', 'hann', 'blackman', 'boxcar').
        sr: Sample rate in Hz.

    Returns:
        1D numpy array of FIR filter impulse response coefficients (b).
    """
    nyquist = sr / 2.0
    btype = btype.lower()

    # Ensure num_taps is odd for Type I linear phase (required for highpass at Nyquist)
    if btype in ["highpass", "bandpass", "bandstop"] and num_taps % 2 == 0:
        num_taps += 1

    if btype == "lowpass":
        if isinstance(cutoff, (list, tuple)):
            cutoff_val = float(cutoff[0])
        else:
            cutoff_val = float(cutoff)
        if not (0 < cutoff_val < nyquist):
            raise ValueError(f"Cutoff frequency {cutoff_val} Hz must be within (0, Nyquist={nyquist} Hz).")
        b = signal.firwin(num_taps, cutoff_val, pass_zero="lowpass", window=window, fs=sr)

    elif btype == "highpass":
        if isinstance(cutoff, (list, tuple)):
            cutoff_val = float(cutoff[0])
        else:
            cutoff_val = float(cutoff)
        if not (0 < cutoff_val < nyquist):
            raise ValueError(f"Cutoff frequency {cutoff_val} Hz must be within (0, Nyquist={nyquist} Hz).")
        b = signal.firwin(num_taps, cutoff_val, pass_zero="highpass", window=window, fs=sr)

    elif btype == "bandpass":
        if not (isinstance(cutoff, (list, tuple)) and len(cutoff) == 2):
            raise ValueError("Bandpass filter requires a tuple/list of two cutoff frequencies (low, high).")
        low, high = float(cutoff[0]), float(cutoff[1])
        if not (0 < low < high < nyquist):
            raise ValueError(f"Bandpass cutoffs ({low}, {high}) must satisfy 0 < low < high < Nyquist={nyquist} Hz.")
        b = signal.firwin(num_taps, [low, high], pass_zero="bandpass", window=window, fs=sr)

    else:
        raise ValueError(f"Unsupported FIR filter type '{btype}'. Use 'lowpass', 'highpass', or 'bandpass'.")

    return b.astype(np.float64)


def apply_fir_filter(
    audio: np.ndarray,
    cutoff: Union[float, Tuple[float, float], List[float]],
    num_taps: int = 101,
    btype: str = "lowpass",
    window: str = "hamming",
    sr: int = 16000,
    zero_phase: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply FIR filter to an audio signal.

    Parameters:
        audio: 1D numpy array.
        cutoff: Cutoff frequency/frequencies in Hz.
        num_taps: Filter length.
        btype: 'lowpass', 'highpass', or 'bandpass'.
        window: Window type.
        sr: Sample rate in Hz.
        zero_phase: If True, uses filtfilt (zero phase, offline only).
                    If False, uses causal lfilter (introduces (num_taps-1)/2 group delay).

    Returns:
        filtered_audio: 1D numpy array.
        taps: 1D array of filter coefficients.
        info: Dict with DSP explanations and parameters.
    """
    b = design_fir_filter(cutoff, num_taps=num_taps, btype=btype, window=window, sr=sr)

    if zero_phase:
        filtered = signal.filtfilt(b, [1.0], audio)
    else:
        filtered = signal.lfilter(b, [1.0], audio)

    delay_samples = (len(b) - 1) / 2.0
    delay_ms = (delay_samples / sr) * 1000.0

    info = {
        "method": "FIR Filter",
        "btype": btype,
        "cutoff_hz": cutoff,
        "num_taps": len(b),
        "window": window,
        "sample_rate": sr,
        "zero_phase": zero_phase,
        "group_delay_samples": delay_samples if not zero_phase else 0.0,
        "group_delay_ms": round(delay_ms, 2) if not zero_phase else 0.0,
        "dsp_note": (
            f"Type I linear-phase FIR filter designed via window method ({window}). "
            f"Transition bandwidth is approximately {round(3.3 * sr / len(b), 1)} Hz. "
            + ("Zero-phase filtering applied via bidirectional filtering (offline only)."
               if zero_phase else
               f"Causal linear-phase filtering introduces constant group delay of {round(delay_ms, 2)} ms.")
        )
    }

    return filtered.astype(np.float32), b, info


# ============================================================================
# B. IIR BUTTERWORTH FILTERING (Second-Order Sections)
# ============================================================================

def design_butterworth_filter(
    cutoff: Union[float, Tuple[float, float], List[float]],
    order: int = 4,
    btype: str = "lowpass",
    sr: int = 16000
) -> np.ndarray:
    """
    Design an IIR Butterworth filter represented in Second-Order Sections (SOS)
    for optimal numerical stability.

    Parameters:
        cutoff: Cutoff frequency in Hz (float or (low, high)).
        order: Filter order N (e.g. 2, 4, 6).
        btype: 'lowpass', 'highpass', or 'bandpass'.
        sr: Sample rate in Hz.

    Returns:
        sos: Array of shape (n_sections, 6) containing biquad coefficients.
    """
    nyquist = sr / 2.0
    btype = btype.lower()

    if order < 1 or order > 12:
        raise ValueError(f"IIR filter order {order} should be between 1 and 12 for stability.")

    if btype in ["lowpass", "highpass"]:
        cutoff_val = float(cutoff[0]) if isinstance(cutoff, (list, tuple)) else float(cutoff)
        if not (0 < cutoff_val < nyquist):
            raise ValueError(f"Cutoff frequency {cutoff_val} Hz must be within (0, Nyquist={nyquist} Hz).")
        sos = signal.butter(order, cutoff_val, btype=btype, fs=sr, output="sos")

    elif btype == "bandpass":
        if not (isinstance(cutoff, (list, tuple)) and len(cutoff) == 2):
            raise ValueError("Bandpass filter requires a tuple/list of two cutoff frequencies (low, high).")
        low, high = float(cutoff[0]), float(cutoff[1])
        if not (0 < low < high < nyquist):
            raise ValueError(f"Bandpass cutoffs ({low}, {high}) must satisfy 0 < low < high < Nyquist={nyquist} Hz.")
        sos = signal.butter(order, [low, high], btype=btype, fs=sr, output="sos")

    else:
        raise ValueError(f"Unsupported IIR filter type '{btype}'. Use 'lowpass', 'highpass', or 'bandpass'.")

    return sos


def apply_butterworth_filter(
    audio: np.ndarray,
    cutoff: Union[float, Tuple[float, float], List[float]],
    order: int = 4,
    btype: str = "lowpass",
    sr: int = 16000,
    zero_phase: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply IIR Butterworth filter using Second-Order Sections (SOS).

    Parameters:
        audio: 1D numpy array.
        cutoff: Cutoff frequency/frequencies.
        order: Butterworth filter order.
        btype: 'lowpass', 'highpass', or 'bandpass'.
        sr: Sample rate in Hz.
        zero_phase: If True, uses sosfiltfilt (offline zero-phase, effective order doubled).
                    If False, uses sosfilt (causal, non-linear phase, real-time capable).

    Returns:
        filtered_audio: 1D numpy array.
        sos: SOS coefficient matrix.
        info: Dict with DSP explanations and parameters.
    """
    sos = design_butterworth_filter(cutoff, order=order, btype=btype, sr=sr)

    if zero_phase:
        filtered = signal.sosfiltfilt(sos, audio)
    else:
        filtered = signal.sosfilt(sos, audio)

    info = {
        "method": "IIR Butterworth Filter (SOS)",
        "btype": btype,
        "cutoff_hz": cutoff,
        "order": order,
        "effective_order": order * 2 if zero_phase else order,
        "sample_rate": sr,
        "zero_phase": zero_phase,
        "dsp_note": (
            f"Maximally flat passband Butterworth response implemented as {sos.shape[0]} Second-Order Sections (SOS). "
            + ("Zero-phase filtering applied via forward-backward filtering (sosfiltfilt): "
               "eliminates phase distortion, doubles roll-off slope to -12N dB/oct, offline only."
               if zero_phase else
               "Causal filtering (sosfilt): real-time capable with non-linear phase response and frequency-dependent group delay.")
        )
    }

    return filtered.astype(np.float32), sos, info


# ============================================================================
# C. NOTCH FILTERING (Narrowband 50/60 Hz and Harmonics)
# ============================================================================

def design_notch_filter(
    freq_hz: float = 50.0,
    q: float = 30.0,
    sr: int = 16000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a second-order IIR notch filter at freq_hz with quality factor Q.

    Transfer function:
        H(z) = (1 - 2*cos(w0)*z^-1 + z^-2) / (1 - 2*r*cos(w0)*z^-1 + r^2*z^-2)
        where w0 = 2*pi*f0/sr, bw = f0/Q, r approx 1 - pi*bw/sr.

    Parameters:
        freq_hz: Notch center frequency in Hz.
        q: Quality factor (higher Q means narrower notch band).
        sr: Sample rate in Hz.

    Returns:
        b, a: Numerator and denominator polynomial coefficients.
    """
    nyquist = sr / 2.0
    if not (0 < freq_hz < nyquist):
        raise ValueError(f"Notch frequency {freq_hz} Hz must be between 0 and Nyquist ({nyquist} Hz).")
    if q <= 0:
        raise ValueError("Quality factor Q must be positive.")

    b, a = signal.iirnotch(freq_hz, q, fs=sr)
    return b, a


def apply_notch_filter(
    audio: np.ndarray,
    notch_freq: float = 50.0,
    q: float = 30.0,
    include_harmonics: bool = True,
    max_harmonics: int = 4,
    sr: int = 16000,
    zero_phase: bool = False
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """
    Apply single or harmonic IIR notch filter to remove narrowband interference
    such as electrical hum (50 Hz / 60 Hz).

    Parameters:
        audio: 1D numpy array.
        notch_freq: Fundamental interference frequency (50 or 60 Hz).
        q: Quality factor.
        include_harmonics: If True, also filters integer harmonics (e.g. 100, 150, 200 Hz).
        max_harmonics: Number of harmonics to filter up to Nyquist.
        sr: Sample rate in Hz.
        zero_phase: Whether to use zero-phase filtfilt.

    Returns:
        filtered_audio: 1D numpy array.
        filter_coeffs: List of (b, a) tuples for each notch stage.
        info: Dict with DSP explanations and parameters.
    """
    nyquist = sr / 2.0
    filtered = audio.copy()
    filter_coeffs = []
    applied_freqs = []

    harmonics_count = max_harmonics if include_harmonics else 1
    for h in range(1, harmonics_count + 1):
        f = notch_freq * h
        if f >= nyquist - 10.0:
            break
        # Slight decrease in Q for higher harmonics if desired, or constant Q
        b, a = design_notch_filter(f, q=q, sr=sr)
        filter_coeffs.append((b, a))
        applied_freqs.append(round(f, 1))

        if zero_phase:
            filtered = signal.filtfilt(b, a, filtered)
        else:
            filtered = signal.lfilter(b, a, filtered)

    info = {
        "method": "IIR Notch Filter",
        "fundamental_freq_hz": notch_freq,
        "q_factor": q,
        "bandwidth_hz": round(notch_freq / q, 2),
        "applied_frequencies_hz": applied_freqs,
        "harmonics_filtered": len(applied_freqs),
        "sample_rate": sr,
        "zero_phase": zero_phase,
        "dsp_note": (
            f"Narrowband notch attenuation targeting {applied_freqs} Hz with Q={q}. "
            f"-3 dB bandwidth at fundamental is {round(notch_freq / q, 2)} Hz. "
            + ("Applied symmetrically via zero-phase filtering." if zero_phase else "Applied as causal recursive IIR stages.")
        )
    }

    return filtered.astype(np.float32), filter_coeffs, info


# ============================================================================
# D. FREQUENCY RESPONSE CALCULATIONS FOR VISUALIZATION
# ============================================================================

def compute_fir_response(b: np.ndarray, sr: int = 16000, worN: int = 1024) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute frequency response of an FIR filter.

    Returns:
        frequencies: Frequency axis in Hz.
        magnitude_db: Magnitude response in dB.
        phase_rad: Phase response in radians.
    """
    w, h = signal.freqz(b, [1.0], worN=worN, fs=sr)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-6))
    phase = np.unwrap(np.angle(h))
    return w, mag_db, phase


def compute_iir_sos_response(sos: np.ndarray, sr: int = 16000, worN: int = 1024, zero_phase: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute frequency response of an IIR SOS filter.

    Returns:
        frequencies: Frequency axis in Hz.
        magnitude_db: Magnitude response in dB.
        phase_rad: Phase response in radians (0 if zero_phase).
    """
    w, h = signal.sosfreqz(sos, worN=worN, fs=sr)
    if zero_phase:
        # sosfiltfilt squares the magnitude and zeroes the phase
        mag_db = 40 * np.log10(np.maximum(np.abs(h), 1e-6))
        phase = np.zeros_like(w)
    else:
        mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-6))
        phase = np.unwrap(np.angle(h))
    return w, mag_db, phase


def compute_notch_response(filter_coeffs: List[Tuple[np.ndarray, np.ndarray]], sr: int = 16000, worN: int = 1024, zero_phase: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute combined frequency response of cascaded notch filters.

    Returns:
        frequencies: Frequency axis in Hz.
        magnitude_db: Magnitude response in dB.
    """
    w = np.linspace(0, sr / 2.0, worN)
    total_h = np.ones(worN, dtype=np.complex128)

    for b, a in filter_coeffs:
        _, h = signal.freqz(b, a, worN=w, fs=sr)
        total_h *= h

    if zero_phase:
        mag_db = 40 * np.log10(np.maximum(np.abs(total_h), 1e-6))
    else:
        mag_db = 20 * np.log10(np.maximum(np.abs(total_h), 1e-6))

    return w, mag_db
