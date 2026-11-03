"""
Reporting and Export Module
Generates structured JSON experiment summaries, CSV comparison logs, and formatted markdown reports.
"""

from typing import Dict, Any, List
import json
import pandas as pd
from datetime import datetime


def create_experiment_summary(
    input_info: Dict[str, Any],
    method_results: List[Dict[str, Any]],
    ml_prediction: Dict[str, Any] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Construct standardized JSON-serializable experiment metadata record.
    """
    return {
        "project": "Adaptive ML-Assisted Audio Noise Reduction Using FIR/IIR Filters",
        "timestamp": datetime.now().isoformat(),
        "input_metadata": input_info,
        "ml_classification": ml_prediction,
        "benchmark_results": method_results,
        "user_notes": notes,
        "disclaimer": (
            "Results are objectively measured using digital signal processing metrics. "
            "Unpaired evaluations reflect energy attenuation rather than subjective perceptual speech intelligibility."
        )
    }


def results_to_dataframe(method_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of method benchmark dictionaries into a clean Pandas DataFrame.
    """
    records = []
    for r in method_results:
        rec = {
            "Method": r.get("method", "Unknown"),
            "Runtime (ms)": round(r.get("processing_time_sec", 0.0) * 1000, 2),
            "RTF": r.get("real_time_factor", None),
        }
        if "snr_improvement_db" in r:
            rec["Input SNR (dB)"] = r.get("input_snr_db", None)
            rec["Output SNR (dB)"] = r.get("output_snr_db", None)
            rec["SNR Imp. (dB)"] = r.get("snr_improvement_db", None)
            rec["RMSE"] = r.get("rmse", None)
            rec["Pearson r"] = r.get("pearson_correlation", None)
        else:
            rec["Output RMS (dBFS)"] = r.get("output_rms_dbfs", None)
            rec["Rumble Atten. (dB)"] = r.get("rumble_band_attenuation_db", None)
            rec["50Hz Atten. (dB)"] = r.get("hum_50hz_attenuation_db", None)
            rec["Hiss Atten. (dB)"] = r.get("hiss_band_attenuation_db", None)

        records.append(rec)

    return pd.DataFrame(records)


def export_markdown_report(summary: Dict[str, Any]) -> str:
    """
    Format complete experiment report as readable Markdown.
    """
    ts = summary.get("timestamp", "N/A")
    input_meta = summary.get("input_metadata", {})
    benchmarks = summary.get("benchmark_results", [])
    ml_pred = summary.get("ml_classification", {})

    lines = [
        "# Audio Denoising Experiment Report",
        f"**Date:** {ts}",
        "",
        "## 1. Input Audio Summary",
        f"- **Duration:** {input_meta.get('duration_seconds', 'N/A')} s",
        f"- **Sample Rate:** {input_meta.get('sample_rate', 'N/A')} Hz",
        f"- **RMS Level:** {input_meta.get('rms_dbfs', 'N/A')} dBFS",
        f"- **Simulated Mixture:** {input_meta.get('is_simulated', False)}",
        ""
    ]

    if ml_pred:
        lines.extend([
            "## 2. Machine Learning Noise Diagnosis",
            f"- **Predicted Class:** `{ml_pred.get('predicted_class', 'N/A')}`",
            f"- **Label:** {ml_pred.get('class_label', 'N/A')}",
            f"- **Confidence:** {ml_pred.get('confidence', 0.0) * 100:.1f}%",
            ""
        ])

    lines.extend([
        "## 3. Method Benchmark Results",
        "| Method | Runtime (ms) | RTF | SNR Imp. (dB) | Output SNR (dB) | RMSE |",
        "|---|---|---|---|---|---|"
    ])

    for b in benchmarks:
        name = b.get("method", "N/A")
        ms = round(b.get("processing_time_sec", 0.0) * 1000, 1)
        rtf = b.get("real_time_factor", "N/A")
        snr_imp = b.get("snr_improvement_db", "N/A")
        out_snr = b.get("output_snr_db", "N/A")
        rmse = b.get("rmse", "N/A")
        lines.append(f"| {name} | {ms} | {rtf} | {snr_imp} | {out_snr} | {rmse} |")

    lines.extend([
        "",
        "## 4. Academic Evaluation Notice",
        summary.get("disclaimer", "")
    ])

    return "\n".join(lines)
