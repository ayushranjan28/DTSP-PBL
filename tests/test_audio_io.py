"""
Unit tests for Audio IO, resampling, validation, and metadata extraction.
"""

import os
import io
import tempfile
import numpy as np
import pytest
import soundfile as sf

from src.audio_io import (
    load_audio,
    save_audio,
    audio_to_bytes,
    get_audio_metadata,
    generate_clean_speech_synthetic
)


def test_generate_clean_speech_synthetic():
    sr = 16000
    duration = 2.0
    speech = generate_clean_speech_synthetic(duration=duration, sr=sr, seed=123)

    assert isinstance(speech, np.ndarray)
    assert speech.dtype == np.float32
    assert len(speech) == int(duration * sr)
    assert np.all(np.isfinite(speech))
    assert np.max(np.abs(speech)) <= 1.0
    assert np.max(np.abs(speech)) > 0.1


def test_save_and_load_audio_roundtrip():
    sr = 16000
    original = generate_clean_speech_synthetic(duration=1.0, sr=sr, seed=42)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_audio(tmp_path, original, sr)
        assert os.path.exists(tmp_path)

        loaded, loaded_sr = load_audio(tmp_path, target_sr=16000)
        assert loaded_sr == 16000
        assert len(loaded) == len(original)
        assert np.all(np.isfinite(loaded))

        # Check numerical roundtrip (16-bit PCM quantization introduces small error < 1e-3)
        diff = np.abs(loaded - original)
        assert np.max(diff) < 2e-3
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_audio_to_bytes():
    sr = 16000
    audio = generate_clean_speech_synthetic(duration=1.0, sr=sr)
    raw_bytes = audio_to_bytes(audio, sr)

    assert isinstance(raw_bytes, bytes)
    assert len(raw_bytes) > 0

    # Read back bytes
    buf = io.BytesIO(raw_bytes)
    data, r_sr = sf.read(buf, dtype="float32")
    assert r_sr == sr
    assert len(data) == len(audio)


def test_get_audio_metadata():
    sr = 16000
    duration = 1.5
    audio = generate_clean_speech_synthetic(duration=duration, sr=sr)
    meta = get_audio_metadata(audio, sr)

    assert meta["sample_rate"] == 16000
    assert meta["duration_seconds"] == pytest.approx(1.5, abs=0.05)
    assert meta["n_samples"] == int(duration * sr)
    assert meta["peak_amplitude"] > 0
    assert meta["rms_dbfs"] < 0
    assert meta["is_silent"] is False


def test_load_audio_invalid():
    with pytest.raises(ValueError):
        load_audio(io.BytesIO(b"not an audio file"))
