"""
Machine Learning Noise Classifier and Adaptive Filter Dispatcher
Implements training, evaluation, persistence, and dynamic filter strategy selection.
"""

from typing import Dict, Any, Tuple, Optional, List
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.features import extract_features_from_audio, FEATURE_NAMES
import src.filters as dsp_filters
import src.spectral_denoising as spec_denoising

CLASSES = [
    "electrical_hum",
    "low_frequency_rumble",
    "high_frequency_hiss",
    "broadband_noise",
    "clean_speech"
]

CLASS_LABELS = {
    "electrical_hum": "Electrical Hum (50/60 Hz)",
    "low_frequency_rumble": "Low-Frequency Rumble (<150 Hz)",
    "high_frequency_hiss": "High-Frequency Hiss (>3 kHz)",
    "broadband_noise": "Broadband Noise",
    "clean_speech": "Clean Speech (Nominal Noise)"
}


class NoiseClassifier:
    """
    Random Forest Classifier for audio noise classification and adaptive filter dispatch.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.classes: List[str] = CLASSES
        self.metadata: Dict[str, Any] = {}
        self.is_trained: bool = False

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        provenance: str = "Synthetic controlled noise augmented on speech"
    ) -> Dict[str, Any]:
        """
        Train the model pipeline and evaluate on held-out test data.
        """
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        y_train_pred = self.pipeline.predict(X_train)
        train_acc = float(accuracy_score(y_train, y_train_pred))

        metrics: Dict[str, Any] = {
            "train_accuracy": round(train_acc, 4),
            "n_train_samples": len(y_train),
            "provenance": provenance
        }

        if X_test is not None and y_test is not None:
            y_test_pred = self.pipeline.predict(X_test)
            test_acc = float(accuracy_score(y_test, y_test_pred))
            report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_test_pred, labels=self.classes).tolist()

            metrics.update({
                "test_accuracy": round(test_acc, 4),
                "n_test_samples": len(y_test),
                "classification_report": report,
                "confusion_matrix": cm
            })

        # Feature importances
        rf = self.pipeline.named_steps["rf"]
        importances = rf.feature_importances_
        feature_importance_dict = {
            name: round(float(imp), 4)
            for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
        }
        metrics["feature_importances"] = feature_importance_dict
        self.metadata = metrics

        return metrics

    def predict_audio(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Extract features from audio and predict noise classification with confidence.
        """
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model is not trained. Please train or load a saved model first.")

        feat_vec, feat_dict = extract_features_from_audio(audio, sr=sr)
        feat_vec_2d = feat_vec.reshape(1, -1)

        pred_class = str(self.pipeline.predict(feat_vec_2d)[0])
        prob_array = self.pipeline.predict_proba(feat_vec_2d)[0]

        class_probabilities = {
            cls_name: round(float(prob_array[idx]), 4)
            for idx, cls_name in enumerate(self.pipeline.classes_)
        }

        # If a majority (>50%) of trees agree the audio is noisy (P(clean_speech) < 0.50),
        # ensure clean_speech does not win by plurality when noise votes split across sub-bands
        clean_prob = class_probabilities.get("clean_speech", 0.0)
        if pred_class == "clean_speech" and clean_prob < 0.50:
            noise_candidates = {k: v for k, v in class_probabilities.items() if k != "clean_speech"}
            pred_class = max(noise_candidates, key=noise_candidates.get)

        confidence = float(class_probabilities[pred_class])

        return {
            "predicted_class": pred_class,
            "class_label": CLASS_LABELS.get(pred_class, pred_class),
            "confidence": round(confidence, 4),
            "class_probabilities": class_probabilities,
            "feature_dict": feat_dict
        }

    def get_adaptive_strategy(
        self,
        predicted_class: str,
        feature_dict: Dict[str, float],
        sr: int = 16000
    ) -> Dict[str, Any]:
        """
        Determine optimal DSP filtering strategy and parameters based on predicted noise class.
        """
        nyquist = sr / 2.0

        if predicted_class == "electrical_hum":
            # Decide fundamental hum frequency between 50 Hz and 60 Hz based on extracted energy ratios
            f50 = feature_dict.get("hum_50hz_ratio", 0.0)
            f60 = feature_dict.get("hum_60hz_ratio", 0.0)
            hum_freq = 60.0 if f60 > f50 else 50.0

            return {
                "strategy": "Harmonic Notch Filter",
                "method_key": "notch",
                "reason": (
                    f"Classified as narrowband electrical hum. Prominence detected around {hum_freq} Hz. "
                    "Harmonic notch filtering eliminates the powerline fundamental and harmonics with zero loss to midband speech."
                ),
                "params": {
                    "notch_freq": hum_freq,
                    "q": 35.0,
                    "include_harmonics": True,
                    "max_harmonics": 4,
                    "zero_phase": True
                }
            }

        elif predicted_class == "low_frequency_rumble":
            # Highpass Butterworth filter
            return {
                "strategy": "High-Pass Butterworth Filter (SOS)",
                "method_key": "iir",
                "reason": (
                    "Classified as low-frequency rumble (HVAC/wind/handling). "
                    "4th-order high-pass Butterworth filter at 150 Hz removes acoustic rumble below human fundamental pitch."
                ),
                "params": {
                    "cutoff": 150.0,
                    "order": 4,
                    "btype": "highpass",
                    "zero_phase": True
                }
            }

        elif predicted_class == "high_frequency_hiss":
            # Spectral gating or steep low-pass
            cutoff = min(4000.0, nyquist - 200.0)
            return {
                "strategy": "Low-Pass FIR Filter (Linear Phase)",
                "method_key": "fir",
                "reason": (
                    "Classified as high-frequency hiss / thermal noise (>3 kHz). "
                    f"FIR low-pass filter (order 101, cutoff {cutoff} Hz) cleanly rolls off tape hiss while preserving telephone/narrowband speech."
                ),
                "params": {
                    "cutoff": cutoff,
                    "num_taps": 101,
                    "btype": "lowpass",
                    "window": "hamming",
                    "zero_phase": True
                }
            }

        elif predicted_class == "broadband_noise":
            # Decision-directed audio Wiener filter
            return {
                "strategy": "Audio Wiener Filter (Decision-Directed)",
                "method_key": "wiener",
                "reason": (
                    "Classified as broadband noise (white/pink/environmental background). "
                    "Applies 100 Hz sub-bass rumble removal followed by a 2D-smoothed Decision-Directed Wiener filter "
                    "to suppress distributed noise across the voice spectrum."
                ),
                "params": {
                    "alpha": 0.85,
                    "overestimation": 2.2,
                    "gain_floor_db": -25.0
                }
            }

        elif predicted_class == "clean_speech":
            # Clean: Gentle Wiener or direct pass-through
            return {
                "strategy": "Subtle Denoising / Pass-Through",
                "method_key": "clean_pass",
                "reason": (
                    "Audio is classified as clean speech with nominal background noise. "
                    "Applying minimal processing to prevent unnecessary speech coloration or phase distortion."
                ),
                "params": {
                    "alpha": 0.85,
                    "overestimation": 1.5,
                    "gain_floor_db": -14.0
                }
            }

        else:
            # Mixed noise: Multi-stage cascade
            return {
                "strategy": "Composite Multi-Stage Cascade",
                "method_key": "cascade",
                "reason": (
                    "Complex mixed noise detected. Running a multi-stage filtering cascade: "
                    "1) 50 Hz Notch filter, 2) 120 Hz High-pass rumble filter, and 3) Decision-Directed Wiener filter."
                ),
                "params": {
                    "notch_freq": 50.0,
                    "hp_cutoff": 120.0,
                    "wiener_floor_db": -16.0
                }
            }

    def apply_adaptive_filter(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """
        Execute full adaptive pipeline: feature extraction -> ML prediction -> strategy selection -> filter execution.
        """
        pred_result = self.predict_audio(audio, sr=sr)
        pred_class = pred_result["predicted_class"]
        strategy = self.get_adaptive_strategy(pred_class, pred_result["feature_dict"], sr=sr)

        method_key = strategy["method_key"]
        params = strategy["params"]

        if method_key == "notch":
            filtered, _, filter_info = dsp_filters.apply_notch_filter(
                audio,
                notch_freq=params["notch_freq"],
                q=params["q"],
                include_harmonics=params["include_harmonics"],
                max_harmonics=params["max_harmonics"],
                sr=sr,
                zero_phase=params["zero_phase"]
            )
        elif method_key == "iir":
            hp_audio, _, filter_info = dsp_filters.apply_butterworth_filter(
                audio,
                cutoff=params["cutoff"],
                order=params["order"],
                btype=params["btype"],
                sr=sr,
                zero_phase=params["zero_phase"]
            )
            # Follow high-pass rumble removal with Decision-Directed Wiener pass to clean residual mid-band noise
            filtered, _ = spec_denoising.wiener_filter_audio(
                hp_audio,
                sr=sr,
                alpha=0.85,
                overestimation=1.6,
                gain_floor_db=-20.0
            )
        elif method_key == "fir":
            lp_audio, _, filter_info = dsp_filters.apply_fir_filter(
                audio,
                cutoff=params["cutoff"],
                num_taps=params["num_taps"],
                btype=params["btype"],
                window=params["window"],
                sr=sr,
                zero_phase=params["zero_phase"]
            )
            # Follow low-pass hiss removal with Decision-Directed Wiener pass to clean residual mid-band hiss
            filtered, _ = spec_denoising.wiener_filter_audio(
                lp_audio,
                sr=sr,
                alpha=0.85,
                overestimation=1.6,
                gain_floor_db=-20.0
            )
        elif method_key == "wiener":
            # Strip <100 Hz sub-bass rumble before Decision-Directed Wiener filtering
            hp_audio, _, _ = dsp_filters.apply_butterworth_filter(
                audio,
                cutoff=100.0,
                order=4,
                btype="highpass",
                sr=sr,
                zero_phase=True
            )
            filtered, filter_info = spec_denoising.wiener_filter_audio(
                hp_audio,
                sr=sr,
                alpha=params["alpha"],
                overestimation=params["overestimation"],
                gain_floor_db=params["gain_floor_db"]
            )
        elif method_key == "clean_pass":
            filtered, filter_info = spec_denoising.wiener_filter_audio(
                audio,
                sr=sr,
                alpha=params["alpha"],
                overestimation=params["overestimation"],
                gain_floor_db=params["gain_floor_db"]
            )
        elif method_key == "cascade":
            # Multi-stage cascade
            stage1, _, _ = dsp_filters.apply_notch_filter(audio, notch_freq=50.0, q=30.0, sr=sr, zero_phase=True)
            stage2, _, _ = dsp_filters.apply_butterworth_filter(stage1, cutoff=120.0, order=4, btype="highpass", sr=sr, zero_phase=True)
            filtered, filter_info = spec_denoising.wiener_filter_audio(stage2, sr=sr, alpha=0.94, overestimation=1.2, gain_floor_db=-16.0)
            filter_info["method"] = "Cascade (Notch + HP + Wiener)"
        else:
            filtered = audio.copy()
            filter_info = {"method": "Bypass"}

        combined_info = {
            **strategy,
            "dsp_execution": filter_info
        }

        return filtered.astype(np.float32), pred_result, combined_info

    def save(self, filepath: str) -> None:
        """Serialize model and metadata to disk."""
        payload = {
            "pipeline": self.pipeline,
            "classes": self.classes,
            "metadata": self.metadata,
            "is_trained": self.is_trained,
            "random_state": self.random_state
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(payload, filepath)

    def load(self, filepath: str) -> None:
        """Load serialized model and metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        payload = joblib.load(filepath)
        self.pipeline = payload["pipeline"]
        self.classes = payload.get("classes", CLASSES)
        self.metadata = payload.get("metadata", {})
        self.is_trained = payload.get("is_trained", True)
        self.random_state = payload.get("random_state", 42)
