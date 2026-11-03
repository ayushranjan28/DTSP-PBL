"""
Noise Generation and Controlled Simulation Module
Implements physically motivated acoustic noise generators and calibrated SNR mixing.
"""

from typing import Tuple, Dict, Any
import numpy as np
import scipy.signal as signal


def generate_electrical_hum(
    n_samples: int,
    sr: int = 16000,
    base_freq: float = 50.0,
    n_harmonics: int = 4,
    seed: int = 42
) -> np.ndarray:
    """
    Generate electrical power-line hum interference (50 Hz or 60 Hz fundamental
    plus decaying integer harmonics).

    Parameters:
        n_samples: Number of output samples.
        sr: Sample rate in Hz.
        base_freq: Fundamental frequency in Hz (typically 50.0 or 60.0).
        n_harmonics: Number of harmonics to generate.
        seed: Random seed for phase initialization.

    Returns:
        1D float32 numpy array.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
    hum = np.zeros(n_samples, dtype=np.float32)

    for h in range(1, n_harmonics + 1):
        f = base_freq * h
        if f >= sr / 2.0:
            break
        amp = 1.0 / (h ** 0.8)
        # Random initial phase for realistic unaligned phase relationship
        phase = rng.uniform(0, 2 * np.pi)
        hum += amp * np.sin(2 * np.pi * f * t + phase)

    # Normalize RMS
    rms = np.sqrt(np.mean(hum ** 2))
    if rms > 0:
        hum = hum / rms
    return hum.astype(np.float32)


def generate_low_frequency_rumble(
    n_samples: int,
    sr: int = 16000,
    cutoff_hz: float = 120.0,
    order: int = 4,
    seed: int = 42
) -> np.ndarray:
    """
    Generate low-frequency acoustic rumble (HVAC, wind, mic handling)
    via steep low-pass filtering of Gaussian pink/white noise.

    Parameters:
        n_samples: Number of samples.
        sr: Sample rate in Hz.
        cutoff_hz: Upper frequency cutoff for rumble.
        order: Filter order.
        seed: Random seed.

    Returns:
        1D float32 numpy array.
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, n_samples)

    # Butterworth low-pass filter
    sos = signal.butter(order, cutoff_hz, btype='lowpass', fs=sr, output='sos')
    rumble = signal.sosfilt(sos, white)

    # Normalize RMS
    rms = np.sqrt(np.mean(rumble ** 2))
    if rms > 0:
        rumble = rumble / rms
    return rumble.astype(np.float32)


def generate_high_frequency_hiss(
    n_samples: int,
    sr: int = 16000,
    cutoff_hz: float = 3500.0,
    order: int = 4,
    seed: int = 42
) -> np.ndarray:
    """
    Generate high-frequency tape hiss / preamp thermal noise
    via high-pass filtering of white noise.

    Parameters:
        n_samples: Number of samples.
        sr: Sample rate in Hz.
        cutoff_hz: Lower frequency cutoff for hiss.
        order: Filter order.
        seed: Random seed.

    Returns:
        1D float32 numpy array.
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, n_samples)

    # High-pass filter above cutoff_hz
    sos = signal.butter(order, cutoff_hz, btype='highpass', fs=sr, output='sos')
    hiss = signal.sosfilt(sos, white)

    # Normalize RMS
    rms = np.sqrt(np.mean(hiss ** 2))
    if rms > 0:
        hiss = hiss / rms
    return hiss.astype(np.float32)


def generate_broadband_noise(
    n_samples: int,
    sr: int = 16000,
    noise_type: str = "white",
    seed: int = 42
) -> np.ndarray:
    """
    Generate broadband acoustic noise (white or pink).

    Parameters:
        n_samples: Number of samples.
        sr: Sample rate in Hz.
        noise_type: 'white' or 'pink' (1/f).
        seed: Random seed.

    Returns:
        1D float32 numpy array.
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, n_samples)

    if noise_type.lower() == "pink":
        # 1/f pink noise filter using Paul Kellet's refined filter method
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        out = signal.lfilter(b, a, white)
    else:
        out = white

    rms = np.sqrt(np.mean(out ** 2))
    if rms > 0:
        out = out / rms
    return out.astype(np.float32)


def generate_mixed_noise(
    n_samples: int,
    sr: int = 16000,
    seed: int = 42
) -> np.ndarray:
    """
    Generate composite real-world interference mixing electrical hum,
    low-frequency rumble, and ambient broadband noise.

    Parameters:
        n_samples: Number of samples.
        sr: Sample rate in Hz.
        seed: Random seed.

    Returns:
        1D float32 numpy array.
    """
    hum = generate_electrical_hum(n_samples, sr=sr, base_freq=50.0, seed=seed)
    rumble = generate_low_frequency_rumble(n_samples, sr=sr, cutoff_hz=140.0, seed=seed + 1)
    broadband = generate_broadband_noise(n_samples, sr=sr, noise_type="pink", seed=seed + 2)

    # Weighted blend: 40% hum, 35% rumble, 25% broadband
    mixed = 0.40 * hum + 0.35 * rumble + 0.25 * broadband
    rms = np.sqrt(np.mean(mixed ** 2))
    if rms > 0:
        mixed = mixed / rms
    return mixed.astype(np.float32)


def get_noise_by_class(
    noise_class: str,
    n_samples: int,
    sr: int = 16000,
    seed: int = 42
) -> np.ndarray:
    """
    Dispatcher to generate normalized noise matching target category.

    Supported classes:
    - 'electrical_hum'
    - 'low_frequency_rumble'
    - 'high_frequency_hiss'
    - 'broadband_noise'
    - 'mixed_noise'
    """
    c = noise_class.lower()
    if "hum" in c:
        return generate_electrical_hum(n_samples, sr=sr, seed=seed)
    elif "rumble" in c:
        return generate_low_frequency_rumble(n_samples, sr=sr, seed=seed)
    elif "hiss" in c:
        return generate_high_frequency_hiss(n_samples, sr=sr, seed=seed)
    elif "broadband" in c or "white" in c or "pink" in c:
        return generate_broadband_noise(n_samples, sr=sr, seed=seed)
    elif "mixed" in c:
        return generate_mixed_noise(n_samples, sr=sr, seed=seed)
    else:
        return generate_broadband_noise(n_samples, sr=sr, seed=seed)


def mix_speech_and_noise(
    speech: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Mix clean speech and additive noise at an exact target Signal-to-Noise Ratio (SNR).

    Equation:
        SNR(dB) = 10 * log10( P_signal / P_noise )
        P_signal = mean( speech^2 )
        P_noise_target = P_signal / 10^(target_snr_db / 10)
        scale_factor = sqrt( P_noise_target / P_noise_actual )

    Parameters:
        speech: 1D array of clean speech.
        noise: 1D array of noise (will be repeated or sliced to match speech length).
        target_snr_db: Target SNR in decibels (e.g. -5, 0, 5, 10, 20).

    Returns:
        noisy_audio: 1D numpy array of speech + scaled_noise.
        scaled_noise: 1D numpy array of scaled noise.
        metadata: Dict with measured SNR, power levels, clipping status.
    """
    # Align lengths
    n_samples = len(speech)
    if len(noise) < n_samples:
        repeats = int(np.ceil(n_samples / len(noise)))
        noise_aligned = np.tile(noise, repeats)[:n_samples]
    else:
        noise_aligned = noise[:n_samples]

    p_signal = float(np.mean(speech ** 2))
    p_noise = float(np.mean(noise_aligned ** 2))

    if p_signal < 1e-12:
        raise ValueError("Speech signal power is near zero (silent audio). Cannot mix to target SNR.")
    if p_noise < 1e-12:
        raise ValueError("Noise signal power is near zero. Cannot mix to target SNR.")

    # Target noise power
    p_noise_target = p_signal / (10.0 ** (target_snr_db / 10.0))
    scale_factor = np.sqrt(p_noise_target / p_noise)

    scaled_noise = noise_aligned * scale_factor
    mixture = speech + scaled_noise

    # Actual measured SNR
    p_scaled_noise = float(np.mean(scaled_noise ** 2))
    measured_snr_db = float(10.0 * np.log10(p_signal / (p_scaled_noise + 1e-12)))

    # Check for clipping
    peak = float(np.max(np.abs(mixture)))
    clipped_samples = int(np.sum(np.abs(mixture) > 1.0))
    clipping_ratio = float(clipped_samples / n_samples)

    # Optional soft scaling if clipping occurs, but preserve relative relationship
    was_scaled = False
    if peak > 1.0:
        # Avoid wrap-around distortion; document the scaling
        mixture = mixture / (peak + 1e-4)
        scaled_noise = scaled_noise / (peak + 1e-4)
        was_scaled = True

    metadata = {
        "target_snr_db": round(target_snr_db, 2),
        "measured_snr_db": round(measured_snr_db, 2),
        "p_signal": p_signal,
        "p_noise_target": p_noise_target,
        "scale_factor": float(scale_factor),
        "peak_before_clip": round(peak, 4),
        "clipped_samples": clipped_samples,
        "clipping_ratio": round(clipping_ratio, 4),
        "peak_normalized_for_clipping": was_scaled,
        "is_simulated": True
    }

    return mixture.astype(np.float32), scaled_noise.astype(np.float32), metadata
