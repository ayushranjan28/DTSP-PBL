"""
FastAPI Backend Server for SonicPure DSP Workstation
Provides high-performance REST APIs for audio signal processing, ML noise diagnosis,
calibrated noise mixing, multi-method benchmarking, and audio streaming.
"""

import os
import sys
import time
import io
import json
from typing import Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.audio_io import (
    load_audio,
    save_audio,
    audio_to_bytes,
    get_audio_metadata,
    generate_clean_speech_synthetic
)
from src.noise_generation import (
    generate_electrical_hum,
    generate_low_frequency_rumble,
    generate_high_frequency_hiss,
    generate_broadband_noise,
    generate_mixed_noise,
    mix_speech_and_noise
)
import src.filters as dsp_filters
import src.spectral_denoising as spec_denoising
from src.features import extract_features_from_audio, FEATURE_NAMES
from src.ml_model import NoiseClassifier, CLASSES, CLASS_LABELS
from src.evaluation import compute_reference_based_metrics, compute_unpaired_metrics
from src.reporting import create_experiment_summary, results_to_dataframe, export_markdown_report

app = FastAPI(title="SonicPure DSP Audio Workstation", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_CATALOG = [
    {
        "id": "clean_female_sentence_01.wav",
        "name": "🎙️ Female: \"For the twentieth time that evening the two men shook hands.\"",
        "category": "Female Speech",
        "gender": "Female",
        "f0_hz": 210,
        "description": "Studio recording of a conversational English sentence with natural pauses; ideal for testing Wiener and spectral gating."
    },
    {
        "id": "clean_female_sentence_02.wav",
        "name": "🎙️ Female: \"He turned sharply, and faced Gregson across the table.\"",
        "category": "Female Speech",
        "gender": "Female",
        "f0_hz": 215,
        "description": "Rich in sharp unvoiced sibilants ('sh', 's') and stop consonants ('t', 'p', 'k') for testing lowpass and tape hiss suppression."
    },
    {
        "id": "clean_female_sentence_03.wav",
        "name": "🎙️ Female: \"If I ever needed a fighter in my life I need one now.\"",
        "category": "Female Speech",
        "gender": "Female",
        "f0_hz": 225,
        "description": "Expressive dynamic intonation contour and vocal emotion, tests noise floor tracking across varying energy levels."
    },
    {
        "id": "clean_female_sentence_04.wav",
        "name": "🎙️ Female: \"The girl faced him, her eyes shining with sudden fear.\"",
        "category": "Female Speech",
        "gender": "Female",
        "f0_hz": 205,
        "description": "Clear studio narrative cadence with subtle breath pauses, tests transient speech preservation."
    },
    {
        "id": "clean_male_sentence_01.wav",
        "name": "🎙️ Male: \"For the twentieth time that evening the two men shook hands.\"",
        "category": "Male Speech",
        "gender": "Male",
        "f0_hz": 115,
        "description": "Deep resonant male chest voice (~115 Hz pitch); benchmark for 50/60 Hz hum notch filters and HVAC rumble."
    },
    {
        "id": "clean_male_sentence_02.wav",
        "name": "🎙️ Male: \"He turned sharply, and faced Gregson across the table.\"",
        "category": "Male Speech",
        "gender": "Male",
        "f0_hz": 118,
        "description": "Crisp male articulation with formants across 100-4000 Hz, tests intelligibility and consonant preservation."
    },
    {
        "id": "clean_male_sentence_03.wav",
        "name": "🎙️ Male: \"Gregson shoved back his chair and rose to his feet.\"",
        "category": "Male Speech",
        "gender": "Male",
        "f0_hz": 112,
        "description": "Baritone narrative voice with sentence-ending pitch declination, ideal for Wiener filter SNR metrics."
    },
    {
        "id": "clean_male_sentence_04.wav",
        "name": "🎙️ Male: \"Now you're coming down to business, Phil, he exclaimed.\"",
        "category": "Male Speech",
        "gender": "Male",
        "f0_hz": 122,
        "description": "Lively conversational cadence with expressive inflection, tests non-stationary speech vs stationary noise suppression."
    },
    {
        "id": "clean_acoustic_instrument.wav",
        "name": "🎸 Acoustic Guitar (Plucked Chord Arpeggio)",
        "category": "Instrumental",
        "gender": "Instrument",
        "f0_hz": 165,
        "description": "Sharp attack transients and natural acoustic resonance decay, tests phase linearity."
    },
    {
        "id": "clean_male_speech_01.wav",
        "name": "🔬 Synthetic Resonator (Glottal Pulse Baseline)",
        "category": "Synthetic Speech",
        "gender": "Synthetic",
        "f0_hz": 110,
        "description": "Glottal pulse train with mathematical formants for controlled acoustic benchmarks."
    },
    {
        "id": "clean_sample.wav",
        "name": "📁 Classic Reference Audio",
        "category": "Reference",
        "gender": "General",
        "f0_hz": 120,
        "description": "Baseline reference audio sample."
    }
]


# Global State Container
class StudioState:
    def __init__(self):
        self.sr = 16000
        self.active_sample_id = "clean_female_sentence_01.wav"
        self.active_noise_preset = "hum"

        # Initialize clean reference from bundled clean sample catalog
        sample_path = os.path.join(os.path.dirname(__file__), "data", "samples", self.active_sample_id)
        if os.path.exists(sample_path):
            self.clean_reference, _ = load_audio(sample_path, target_sr=self.sr)
        else:
            self.clean_reference = generate_clean_speech_synthetic(duration=4.5, sr=self.sr)

        hum = generate_electrical_hum(len(self.clean_reference), sr=self.sr, base_freq=50.0)
        noisy, _, _ = mix_speech_and_noise(self.clean_reference, hum, target_snr_db=5.0)
        self.current_audio = noisy
        self.processed_audio = None
        self.last_method = None
        self.last_exec_info = {}
        self.audio_source_name = "🎙️ Female Sentence 1 + 50 Hz Mains Hum (5 dB SNR)"
        self.has_clean_ref = True
        self.is_simulated = True
        self.ml_prediction = None
        self.benchmark_results = []
        self.method_audio_cache = {}

        # Load ML Classifier
        self.classifier = NoiseClassifier(random_state=42)
        model_path = os.path.join(os.path.dirname(__file__), "models", "noise_classifier.joblib")
        if os.path.exists(model_path):
            self.classifier.load(model_path)


state = StudioState()


def get_active_sample_name() -> str:
    for item in SAMPLE_CATALOG:
        if item["id"] == state.active_sample_id:
            return item["name"]
    return state.active_sample_id


def apply_preset_internal(preset: str):
    sr = state.sr
    if state.clean_reference is None:
        state.clean_reference = generate_clean_speech_synthetic(duration=4.5, sr=sr)
    clean = state.clean_reference
    n = len(clean)
    state.active_noise_preset = preset

    sample_label = get_active_sample_name().split("(")[0].strip()

    if preset == "hum":
        noise = generate_electrical_hum(n, sr=sr, base_freq=50.0)
        noisy, _, _ = mix_speech_and_noise(clean, noise, target_snr_db=5.0)
        state.current_audio = noisy
        state.audio_source_name = f"{sample_label} + 50 Hz Mains Hum (5 dB SNR)"
        state.has_clean_ref = True
        state.is_simulated = True

    elif preset == "rumble":
        noise = generate_low_frequency_rumble(n, sr=sr, cutoff_hz=130.0)
        noisy, _, _ = mix_speech_and_noise(clean, noise, target_snr_db=3.0)
        state.current_audio = noisy
        state.audio_source_name = f"{sample_label} + HVAC Rumble <130 Hz (3 dB SNR)"
        state.has_clean_ref = True
        state.is_simulated = True

    elif preset == "hiss":
        noise = generate_high_frequency_hiss(n, sr=sr, cutoff_hz=3600.0)
        noisy, _, _ = mix_speech_and_noise(clean, noise, target_snr_db=6.0)
        state.current_audio = noisy
        state.audio_source_name = f"{sample_label} + Preamp Tape Hiss >3.6 kHz (6 dB SNR)"
        state.has_clean_ref = True
        state.is_simulated = True

    elif preset == "broadband":
        noise = generate_broadband_noise(n, sr=sr, noise_type="pink")
        noisy, _, _ = mix_speech_and_noise(clean, noise, target_snr_db=5.0)
        state.current_audio = noisy
        state.audio_source_name = f"{sample_label} + Ambient Pink Noise (5 dB SNR)"
        state.has_clean_ref = True
        state.is_simulated = True

    elif preset == "clean":
        state.current_audio = clean.copy()
        state.audio_source_name = f"{sample_label} (Clean Reference Baseline)"
        state.has_clean_ref = True
        state.is_simulated = False

    state.processed_audio = None
    state.last_method = None
    state.ml_prediction = None
    state.method_audio_cache = {}

    return get_status()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/status")
def get_status():
    """Return system readiness, active audio metadata, and model information."""
    aud_meta = get_audio_metadata(state.current_audio, state.sr)
    model_ready = state.classifier.is_trained
    test_acc = state.classifier.metadata.get("test_accuracy", 0.98) if model_ready else 0.0

    return {
        "status": "ready",
        "sample_rate": state.sr,
        "audio_source_name": state.audio_source_name,
        "active_sample_id": state.active_sample_id,
        "active_sample_name": get_active_sample_name(),
        "active_noise_preset": state.active_noise_preset,
        "has_clean_ref": state.has_clean_ref,
        "is_simulated": state.is_simulated,
        "audio_metadata": aud_meta,
        "has_processed": state.processed_audio is not None,
        "last_method": state.last_method,
        "model_ready": model_ready,
        "model_test_accuracy": round(test_acc * 100, 2),
        "supported_classes": CLASSES,
        "class_labels": CLASS_LABELS
    }


@app.get("/api/samples")
def list_samples():
    """Return catalog of all available clean sample audio files."""
    return {
        "samples": SAMPLE_CATALOG,
        "active_sample_id": state.active_sample_id
    }


@app.post("/api/select-sample")
def select_sample(sample_id: str = Form(...), preset: Optional[str] = Form(None)):
    """Switch active clean sample reference and re-mix active noise scenario."""
    sample_path = os.path.join(os.path.dirname(__file__), "data", "samples", sample_id)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample file not found: {sample_id}")

    clean, _ = load_audio(sample_path, target_sr=state.sr)
    state.clean_reference = clean
    state.active_sample_id = sample_id

    pr = preset or state.active_noise_preset or "hum"
    return apply_preset_internal(pr)


@app.post("/api/quick-preset")
def set_quick_preset(preset: str = Form(...)):
    """1-Click Scenario Preset Loader."""
    return apply_preset_internal(preset)


@app.post("/api/mix-noise")
def mix_custom_noise(noise_type: str = Form(...), snr_db: float = Form(...)):
    """Mix calibrated noise into clean speech reference at exact SNR."""
    sr = state.sr
    if state.clean_reference is None:
        state.clean_reference = generate_clean_speech_synthetic(duration=3.5, sr=sr)
    clean = state.clean_reference
    n = len(clean)

    if "50" in noise_type:
        noise = generate_electrical_hum(n, sr=sr, base_freq=50.0)
    elif "60" in noise_type:
        noise = generate_electrical_hum(n, sr=sr, base_freq=60.0)
    elif "rumble" in noise_type.lower():
        noise = generate_low_frequency_rumble(n, sr=sr, cutoff_hz=130.0)
    elif "hiss" in noise_type.lower():
        noise = generate_high_frequency_hiss(n, sr=sr, cutoff_hz=3600.0)
    elif "pink" in noise_type.lower():
        noise = generate_broadband_noise(n, sr=sr, noise_type="pink")
    elif "white" in noise_type.lower():
        noise = generate_broadband_noise(n, sr=sr, noise_type="white")
    else:
        noise = generate_mixed_noise(n, sr=sr)

    noisy, _, meta = mix_speech_and_noise(clean, noise, target_snr_db=snr_db)
    state.current_audio = noisy
    state.audio_source_name = f"Custom Mixture: Speech + {noise_type} @ {snr_db} dB"
    state.has_clean_ref = True
    state.is_simulated = True
    state.processed_audio = None
    state.last_method = None

    return {
        "status": "success",
        "measured_snr_db": meta["measured_snr_db"],
        "metadata": get_audio_metadata(state.current_audio, state.sr)
    }


@app.post("/api/upload")
async def upload_audio_file(file: UploadFile = File(...), ref_file: Optional[UploadFile] = File(None)):
    """Upload custom WAV recording with optional clean reference."""
    contents = await file.read()
    try:
        aud, sr = load_audio(io.BytesIO(contents), target_sr=16000)
        state.current_audio = aud
        state.sr = sr
        state.audio_source_name = f"Uploaded: {file.filename}"
        state.is_simulated = False

        if ref_file is not None:
            ref_contents = await ref_file.read()
            ref_aud, _ = load_audio(io.BytesIO(ref_contents), target_sr=16000)
            min_l = min(len(aud), len(ref_aud))
            state.current_audio = aud[:min_l]
            state.clean_reference = ref_aud[:min_l]
            state.has_clean_ref = True
        else:
            state.clean_reference = None
            state.has_clean_ref = False

        state.processed_audio = None
        state.last_method = None
        state.ml_prediction = None
        return get_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/apply-filter")
def apply_dsp_filter(
    filter_type: str = Form(...),
    cutoff_hz: float = Form(1000.0),
    cutoff_high_hz: float = Form(3400.0),
    btype: str = Form("lowpass"),
    fir_taps: int = Form(101),
    iir_order: int = Form(4),
    notch_freq: float = Form(50.0),
    notch_q: float = Form(30.0),
    notch_harmonics: bool = Form(True),
    suppression_db: float = Form(-18.0),
    over_subtraction: float = Form(1.5),
    wiener_alpha: float = Form(0.95),
    wiener_beta: float = Form(1.25),
    wiener_floor: float = Form(-18.0),
    zero_phase: bool = Form(True)
):
    """Execute chosen DSP filter on active audio."""
    audio_in = state.current_audio
    sr = state.sr
    t0 = time.perf_counter()

    if filter_type == "fir":
        cut = (cutoff_hz, cutoff_high_hz) if btype == "bandpass" else cutoff_hz
        filtered, taps, info = dsp_filters.apply_fir_filter(
            audio_in, cutoff=cut, num_taps=fir_taps, btype=btype,
            window="hamming", sr=sr, zero_phase=zero_phase
        )
        method_name = f"FIR {btype.capitalize()}"

    elif filter_type == "iir":
        cut = (cutoff_hz, cutoff_high_hz) if btype == "bandpass" else cutoff_hz
        filtered, sos, info = dsp_filters.apply_butterworth_filter(
            audio_in, cutoff=cut, order=iir_order, btype=btype,
            sr=sr, zero_phase=zero_phase
        )
        method_name = f"Butterworth {btype.capitalize()} (SOS)"

    elif filter_type == "notch":
        filtered, coeffs, info = dsp_filters.apply_notch_filter(
            audio_in, notch_freq=notch_freq, q=notch_q,
            include_harmonics=notch_harmonics, max_harmonics=4,
            sr=sr, zero_phase=zero_phase
        )
        method_name = f"Harmonic Notch ({notch_freq:.0f} Hz)"

    elif filter_type == "spectral_gating":
        filtered, info = spec_denoising.spectral_gating(
            audio_in, sr=sr, suppression_db=suppression_db,
            over_subtraction=over_subtraction, time_smoothing=2, freq_smoothing=2
        )
        method_name = "STFT Spectral Gating"

    elif filter_type == "wiener":
        filtered, info = spec_denoising.wiener_filter_audio(
            audio_in, sr=sr, alpha=wiener_alpha, overestimation=wiener_beta,
            gain_floor_db=wiener_floor
        )
        method_name = "Audio Wiener Filter (DD)"

    else:
        filtered = audio_in.copy()
        info = {}
        method_name = "Bypass"

    t_sec = time.perf_counter() - t0
    state.processed_audio = filtered
    state.last_method = method_name
    state.last_exec_info = info
    state.last_exec_info["processing_time_sec"] = t_sec

    # Compute Metrics
    if state.has_clean_ref and state.clean_reference is not None:
        metrics = compute_reference_based_metrics(
            clean_ref=state.clean_reference,
            noisy_input=audio_in,
            processed=filtered,
            sr=sr,
            processing_time_sec=t_sec
        )
    else:
        metrics = compute_unpaired_metrics(
            noisy_input=audio_in,
            processed=filtered,
            sr=sr,
            processing_time_sec=t_sec
        )

    return {
        "status": "success",
        "method": method_name,
        "processing_time_ms": round(t_sec * 1000, 1),
        "metrics": metrics,
        "info": info
    }


@app.post("/api/adaptive-ml")
def run_adaptive_ml():
    """Run full ML diagnosis and autonomous filter execution."""
    if not state.classifier.is_trained:
        raise HTTPException(status_code=500, detail="ML Classifier not trained.")

    audio_in = state.current_audio
    sr = state.sr
    t0 = time.perf_counter()

    filtered, pred_info, strat_info = state.classifier.apply_adaptive_filter(audio_in, sr=sr)
    t_sec = time.perf_counter() - t0

    state.processed_audio = filtered
    state.last_method = f"Adaptive ML ({pred_info['predicted_class']})"
    state.ml_prediction = pred_info

    # Metrics
    if state.has_clean_ref and state.clean_reference is not None:
        metrics = compute_reference_based_metrics(
            clean_ref=state.clean_reference,
            noisy_input=audio_in,
            processed=filtered,
            sr=sr,
            processing_time_sec=t_sec
        )
    else:
        metrics = compute_unpaired_metrics(
            noisy_input=audio_in,
            processed=filtered,
            sr=sr,
            processing_time_sec=t_sec
        )

    return {
        "status": "success",
        "prediction": pred_info,
        "strategy": strat_info,
        "processing_time_ms": round(t_sec * 1000, 1),
        "metrics": metrics
    }


@app.post("/api/benchmark")
def run_benchmark():
    """Execute complete benchmark matrix across all 6 filtering strategies + baseline."""
    audio_in = state.current_audio
    sr = state.sr

    methods_run = [
        {"id": "baseline", "name": "No Filtering (Baseline)", "type": "baseline"},
        {"id": "fir", "name": "FIR Bandpass (300-3400 Hz)", "type": "fir"},
        {"id": "butterworth", "name": "Butterworth Highpass (150 Hz)", "type": "iir"},
        {"id": "notch", "name": "Harmonic Notch (50 Hz)", "type": "notch"},
        {"id": "spec_gate", "name": "STFT Spectral Gating", "type": "spec_gate"},
        {"id": "wiener", "name": "Audio Wiener Filter (DD)", "type": "wiener"},
        {"id": "adaptive", "name": "ML-Assisted Adaptive Filtering", "type": "adaptive"}
    ]

    results = []
    state.method_audio_cache = {}

    for m in methods_run:
        m_id = m["id"]
        t0 = time.perf_counter()

        if m_id == "baseline":
            out_sig = audio_in
        elif m_id == "fir":
            out_sig, _, _ = dsp_filters.apply_fir_filter(audio_in, cutoff=(300.0, 3400.0), num_taps=101, btype="bandpass", sr=sr, zero_phase=True)
        elif m_id == "butterworth":
            out_sig, _, _ = dsp_filters.apply_butterworth_filter(audio_in, cutoff=150.0, order=4, btype="highpass", sr=sr, zero_phase=True)
        elif m_id == "notch":
            out_sig, _, _ = dsp_filters.apply_notch_filter(audio_in, notch_freq=50.0, q=30.0, include_harmonics=True, max_harmonics=4, sr=sr, zero_phase=True)
        elif m_id == "spec_gate":
            out_sig, _ = spec_denoising.spectral_gating(audio_in, sr=sr, suppression_db=-18.0, over_subtraction=1.5, time_smoothing=2, freq_smoothing=2)
        elif m_id == "wiener":
            out_sig, _ = spec_denoising.wiener_filter_audio(audio_in, sr=sr, alpha=0.95, overestimation=1.25, gain_floor_db=-18.0)
        elif m_id == "adaptive":
            out_sig, _, _ = state.classifier.apply_adaptive_filter(audio_in, sr=sr)

        t_sec = time.perf_counter() - t0
        state.method_audio_cache[m_id] = out_sig

        if state.has_clean_ref and state.clean_reference is not None:
            met = compute_reference_based_metrics(clean_ref=state.clean_reference, noisy_input=audio_in, processed=out_sig, sr=sr, processing_time_sec=t_sec)
        else:
            met = compute_unpaired_metrics(noisy_input=audio_in, processed=out_sig, sr=sr, processing_time_sec=t_sec)

        met["method_id"] = m_id
        met["method_name"] = m["name"]
        results.append(met)

    state.benchmark_results = results
    return {"status": "success", "results": results}


@app.post("/api/export-all-unnoised")
def export_all_unnoised_files():
    """Execute all filter topologies and export unnoised WAV files to outputs/denoised_showcase/."""
    sr = state.sr
    audio_in = state.current_audio
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "denoised_showcase")
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    # 0. Clean Reference (if available)
    if state.has_clean_ref and state.clean_reference is not None:
        save_audio(os.path.join(output_dir, "00_clean_reference.wav"), state.clean_reference, sr)
        saved_files.append({
            "name": "Clean Reference (Ground Truth)",
            "filename": "00_clean_reference.wav",
            "audio_id": "clean",
            "tag": "clean"
        })

    # 1. Noisy Input
    save_audio(os.path.join(output_dir, "01_noisy_input.wav"), audio_in, sr)
    saved_files.append({
        "name": "Noisy Audio (Unfiltered)",
        "filename": "01_noisy_input.wav",
        "audio_id": "noisy",
        "tag": "noisy"
    })

    # 2. FIR Bandpass
    fir_sig, _, _ = dsp_filters.apply_fir_filter(audio_in, cutoff=(300.0, 3400.0), num_taps=101, btype="bandpass", sr=sr, zero_phase=True)
    save_audio(os.path.join(output_dir, "02_unnoised_fir_bandpass.wav"), fir_sig, sr)
    state.method_audio_cache["fir"] = fir_sig
    saved_files.append({
        "name": "FIR Bandpass (300-3400 Hz)",
        "filename": "02_unnoised_fir_bandpass.wav",
        "audio_id": "fir",
        "tag": "dsp"
    })

    # 3. Butterworth Highpass
    iir_sig, _, _ = dsp_filters.apply_butterworth_filter(audio_in, cutoff=150.0, order=4, btype="highpass", sr=sr, zero_phase=True)
    save_audio(os.path.join(output_dir, "03_unnoised_butterworth_highpass.wav"), iir_sig, sr)
    state.method_audio_cache["butterworth"] = iir_sig
    saved_files.append({
        "name": "Butterworth Highpass (150 Hz SOS)",
        "filename": "03_unnoised_butterworth_highpass.wav",
        "audio_id": "butterworth",
        "tag": "dsp"
    })

    # 4. Harmonic Notch
    notch_sig, _, _ = dsp_filters.apply_notch_filter(audio_in, notch_freq=50.0, q=35.0, include_harmonics=True, max_harmonics=4, sr=sr, zero_phase=True)
    save_audio(os.path.join(output_dir, "04_unnoised_harmonic_notch.wav"), notch_sig, sr)
    state.method_audio_cache["notch"] = notch_sig
    saved_files.append({
        "name": "Harmonic Notch Filter (50 Hz)",
        "filename": "04_unnoised_harmonic_notch.wav",
        "audio_id": "notch",
        "tag": "dsp"
    })

    # 5. STFT Spectral Gating
    sg_sig, _ = spec_denoising.spectral_gating(audio_in, sr=sr, suppression_db=-18.0, over_subtraction=1.5, time_smoothing=2, freq_smoothing=2)
    save_audio(os.path.join(output_dir, "05_unnoised_spectral_gating.wav"), sg_sig, sr)
    state.method_audio_cache["spec_gate"] = sg_sig
    saved_files.append({
        "name": "STFT Spectral Gating (2D Smoothed)",
        "filename": "05_unnoised_spectral_gating.wav",
        "audio_id": "spec_gate",
        "tag": "spectral"
    })

    # 6. Audio Wiener Filter
    w_sig, _ = spec_denoising.wiener_filter_audio(audio_in, sr=sr, alpha=0.95, overestimation=1.25, gain_floor_db=-18.0)
    save_audio(os.path.join(output_dir, "06_unnoised_wiener_filter.wav"), w_sig, sr)
    state.method_audio_cache["wiener"] = w_sig
    saved_files.append({
        "name": "Audio Wiener Filter (Ephraim-Malah DD)",
        "filename": "06_unnoised_wiener_filter.wav",
        "audio_id": "wiener",
        "tag": "spectral"
    })

    # 7. ML Adaptive Filter
    ad_sig, pred, _ = state.classifier.apply_adaptive_filter(audio_in, sr=sr)
    save_audio(os.path.join(output_dir, "07_unnoised_ml_adaptive.wav"), ad_sig, sr)
    state.method_audio_cache["adaptive"] = ad_sig
    saved_files.append({
        "name": f"ML-Adaptive ({pred.get('predicted_class', 'Optimal')})",
        "filename": "07_unnoised_ml_adaptive.wav",
        "audio_id": "adaptive",
        "tag": "ml"
    })

    return {
        "status": "success",
        "output_directory": os.path.abspath(output_dir),
        "files_count": len(saved_files),
        "saved_files": saved_files
    }


@app.get("/api/audio/{track_type}")
def get_audio_stream(track_type: str):
    """Serve in-memory 16-bit PCM WAV stream for playback."""
    sr = state.sr

    if track_type == "noisy" or track_type == "input":
        sig = state.current_audio
    elif track_type == "clean" or track_type == "reference":
        sig = state.clean_reference if state.clean_reference is not None else state.current_audio
    elif track_type == "processed":
        sig = state.processed_audio if state.processed_audio is not None else state.current_audio
    elif track_type in state.method_audio_cache:
        sig = state.method_audio_cache[track_type]
    else:
        sig = state.current_audio

    wav_bytes = audio_to_bytes(sig, sr)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/api/waveform-data")
def get_waveform_points(track: str = "both", samples: int = 400):
    """Downsample audio signals for high-performance canvas waveform rendering."""
    def downsample(sig):
        step = max(1, len(sig) // samples)
        return [round(float(v), 3) for v in sig[::step][:samples]]

    data = {
        "noisy": downsample(state.current_audio),
        "duration": round(len(state.current_audio) / state.sr, 2)
    }
    if state.processed_audio is not None:
        data["processed"] = downsample(state.processed_audio)
    if state.has_clean_ref and state.clean_reference is not None:
        data["clean"] = downsample(state.clean_reference)

    return data


@app.get("/api/filter-response")
def get_filter_response_data(filter_type: str = "fir", cutoff: float = 1000.0, cutoff_high: float = 3400.0, btype: str = "lowpass", order: int = 4, q: float = 30.0):
    """Return theoretical frequency response for interactive canvas plotting."""
    sr = state.sr
    if filter_type == "fir":
        cut = (cutoff, cutoff_high) if btype == "bandpass" else cutoff
        b = dsp_filters.design_fir_filter(cut, num_taps=101, btype=btype, sr=sr)
        w, mag, _ = dsp_filters.compute_fir_response(b, sr=sr, worN=512)
    elif filter_type == "iir":
        cut = (cutoff, cutoff_high) if btype == "bandpass" else cutoff
        sos = dsp_filters.design_butterworth_filter(cut, order=order, btype=btype, sr=sr)
        w, mag, _ = dsp_filters.compute_iir_sos_response(sos, sr=sr, worN=512, zero_phase=True)
    elif filter_type == "notch":
        b, a = dsp_filters.design_notch_filter(freq_hz=cutoff, q=q, sr=sr)
        w, mag = dsp_filters.compute_notch_response([(b, a)], sr=sr, worN=512, zero_phase=True)
    else:
        w = np.linspace(0, sr / 2, 512)
        mag = np.zeros(512)

    return {
        "frequencies": [round(float(f), 1) for f in w],
        "magnitude_db": [round(float(m), 2) for m in mag]
    }


@app.get("/api/export/csv")
def export_benchmark_csv():
    """Download benchmark results as CSV."""
    if not state.benchmark_results:
        raise HTTPException(status_code=400, detail="No benchmark results available yet.")
    df = results_to_dataframe(state.benchmark_results)
    csv_str = df.to_csv(index=False)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sonicpure_benchmark.csv"}
    )


@app.get("/api/export/json")
def export_experiment_json():
    """Download experiment metadata as JSON."""
    meta = get_audio_metadata(state.current_audio, state.sr)
    clean_bms = [{k: v for k, v in b.items() if k != "audio_bytes"} for b in state.benchmark_results]
    summary = create_experiment_summary(meta, clean_bms, state.ml_prediction)
    return JSONResponse(content=summary, headers={"Content-Disposition": "attachment; filename=sonicpure_experiment.json"})


# Mount Frontend Static Directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "web")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=False)
