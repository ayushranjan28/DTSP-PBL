"""
Audio Input/Output and Metadata Handling Module
Provides robust loading, resampling, format validation, WAV encoding, and metadata extraction.
"""

import io
from typing import Tuple, Dict, Any, Union
import numpy as np
import soundfile as sf
import librosa


def _read_with_pyav(source: Union[str, io.BytesIO, bytes]) -> Tuple[np.ndarray, int]:
    """
    Fallback audio decoder using PyAV (FFmpeg bindings).
    Supports compressed/forensic speech formats including AMR (.amr, .awb),
    MP3, AAC, M4A, OGG, Opus, FLAC, and uncompressed WAV.
    """
    import av

    if isinstance(source, bytes):
        source = io.BytesIO(source)

    container = av.open(source)
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        raise ValueError("No audio stream found in source.")

    frames = []
    orig_sr = None
    for frame in container.decode(audio_stream):
        if orig_sr is None:
            orig_sr = frame.sample_rate
        frames.append(frame.to_ndarray())

    if not frames:
        raise ValueError("Audio source contains no decodable frames.")

    combined = np.concatenate(frames, axis=-1)

    # Normalize integer PCM to float32 [-1.0, 1.0]
    if np.issubdtype(combined.dtype, np.integer):
        max_val = float(np.iinfo(combined.dtype).max)
        audio = (combined.astype(np.float32) / max_val)
    else:
        audio = combined.astype(np.float32)

    # PyAV returns shape (channels, samples). Transpose to (samples, channels) like soundfile.
    if audio.ndim == 2:
        audio = audio.T

    return audio, int(orig_sr)


def load_audio(
    source: Union[str, io.BytesIO, bytes],
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Load an audio signal from a filepath, BytesIO buffer, or raw audio bytes.
    Safely resamples to target_sr and converts to mono if requested.
    Supports WAV, AMR (Adaptive Multi-Rate), MP3, M4A, FLAC, OGG, and Opus.

    Parameters:
        source: Filepath string, io.BytesIO buffer, or raw audio bytes.
        target_sr: Target sample rate in Hz (default 16000 Hz).
        mono: Whether to downmix multi-channel audio to mono (default True).
        normalize: Whether to peak-normalize audio to [-0.99, 0.99] (default False).

    Returns:
        audio: 1D numpy array of float32 samples in range [-1.0, 1.0].
        sr: Actual sample rate (target_sr).

    Raises:
        ValueError: If audio is empty or non-finite.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    try:
        audio, orig_sr = sf.read(source, dtype="float32")
    except Exception as sf_err:
        # Fall back to PyAV for AMR, MP3, AAC, M4A, OGG, etc.
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            audio, orig_sr = _read_with_pyav(source)
        except Exception as av_err:
            raise ValueError(f"Could not read audio source: soundfile error ({sf_err}), av error ({av_err})")

    if audio is None or len(audio) == 0:
        raise ValueError("Audio source contains no samples (empty signal).")

    # Handle multi-channel audio
    if audio.ndim > 1:
        if mono:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio.T  # Transpose to (channels, samples)

    # Check for NaN or Inf
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

    # Resample if necessary using librosa's high-quality resampler
    if target_sr is not None and orig_sr != target_sr:
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        else:
            audio = np.array([
                librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr)
                for ch in audio
            ])
        sr = target_sr
    else:
        sr = orig_sr

    if normalize and np.max(np.abs(audio)) > 1e-6:
        audio = 0.95 * (audio / np.max(np.abs(audio)))

    return audio.astype(np.float32), sr


def save_audio(filepath: str, audio: np.ndarray, sr: int) -> None:
    """
    Save audio to a 16-bit PCM WAV file with clipping prevention.

    Parameters:
        filepath: Destination path string.
        audio: 1D or 2D numpy array.
        sr: Sampling rate in Hz.
    """
    # Prevent digital wrap-around clipping
    clipped = np.clip(audio, -1.0, 1.0)
    sf.write(filepath, clipped, sr, subtype="PCM_16")


def audio_to_bytes(audio: np.ndarray, sr: int) -> bytes:
    """
    Encode an audio array into in-memory 16-bit PCM WAV bytes for web playback and streaming.

    Parameters:
        audio: 1D or 2D numpy array.
        sr: Sampling rate in Hz.

    Returns:
        WAV format bytes.
    """
    buffer = io.BytesIO()
    clipped = np.clip(audio, -1.0, 1.0)
    sf.write(buffer, clipped, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def get_audio_metadata(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Extract technical audio parameters and quality metrics.

    Parameters:
        audio: 1D numpy array.
        sr: Sampling rate in Hz.

    Returns:
        Dictionary of audio metadata.
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1 if audio.shape[0] > audio.shape[1] else 0)

    n_samples = len(audio)
    duration = float(n_samples) / float(sr) if sr > 0 else 0.0

    peak = float(np.max(np.abs(audio))) if n_samples > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if n_samples > 0 else 0.0
    rms_dbfs = float(20 * np.log10(rms + 1e-9))
    peak_dbfs = float(20 * np.log10(peak + 1e-9))
    crest_factor = float(peak / (rms + 1e-9))

    # Clipping detection (|x| >= 0.999)
    clipping_count = int(np.sum(np.abs(audio) >= 0.999))
    clipping_ratio = float(clipping_count / n_samples) if n_samples > 0 else 0.0

    return {
        "sample_rate": sr,
        "n_samples": n_samples,
        "duration_seconds": round(duration, 3),
        "peak_amplitude": round(peak, 4),
        "peak_dbfs": round(peak_dbfs, 2),
        "rms_amplitude": round(rms, 4),
        "rms_dbfs": round(rms_dbfs, 2),
        "crest_factor": round(crest_factor, 2),
        "clipping_samples": clipping_count,
        "clipping_percentage": round(clipping_ratio * 100, 3),
        "is_silent": rms < 1e-5
    }


def generate_clean_speech_synthetic(
    duration: float = 3.5,
    sr: int = 16000,
    seed: int = 42
) -> np.ndarray:
    """
    Synthesize an academic-quality speech-like harmonic signal with natural formant
    resonances and syllables for reproducible testing when no recording is uploaded.

    Models a glottal pulse train modulated through vocal tract formants:
    - F0 ~ 130 Hz (fundamental frequency with natural intonation contour)
    - F1 ~ 600 Hz (first vowel formant)
    - F2 ~ 1400 Hz (second vowel formant)
    - F3 ~ 2500 Hz (third vowel formant)

    Parameters:
        duration: Length in seconds (default 3.5 s).
        sr: Sample rate in Hz (default 16000 Hz).
        seed: Random seed for reproducibility.

    Returns:
        1D float32 numpy array normalized to [-0.85, 0.85].
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)

    # Pitch contour with natural vibrato and sentence declination
    f0 = 135.0 - 15.0 * (t / duration) + 3.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # Glottal source: rich harmonic pulse train
    glottal = np.zeros_like(t)
    for harmonic in range(1, 25):
        amp = 1.0 / (harmonic ** 0.8)
        glottal += amp * np.sin(harmonic * phase)

    # Resonances (vocal tract formants F1, F2, F3)
    formant_1 = np.sin(2 * np.pi * 650.0 * t) * np.exp(-t * 2.0 % 0.3)
    formant_2 = np.sin(2 * np.pi * 1550.0 * t) * 0.5 * np.exp(-t * 2.5 % 0.3)
    formant_3 = np.sin(2 * np.pi * 2600.0 * t) * 0.25 * np.exp(-t * 3.0 % 0.3)

    combined_formants = glottal * (0.6 + 0.3 * np.sin(2 * np.pi * 3.5 * t)) + 0.3 * formant_1 + 0.2 * formant_2 + 0.1 * formant_3

    # Syllabic envelope modulation (approx 3.5 syllables per second)
    syllable_rate = 3.5
    envelope = np.clip(np.sin(2 * np.pi * syllable_rate * t)**2, 0.05, 1.0)
    # Pause in middle for realistic speech sentence structure
    pause_mask = (t > 1.4) & (t < 1.7)
    envelope[pause_mask] *= 0.05

    # Gentle fade in/out
    ramp_len = int(0.04 * sr)
    fade = np.ones_like(t)
    fade[:ramp_len] = np.linspace(0, 1, ramp_len)
    fade[-ramp_len:] = np.linspace(1, 0, ramp_len)

    speech = combined_formants * envelope * fade

    # Gentle bandpass [100 Hz, 5000 Hz]
    speech = librosa.util.normalize(speech) * 0.85
    return speech.astype(np.float32)
