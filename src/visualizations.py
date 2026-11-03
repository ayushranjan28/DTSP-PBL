"""
Visualizations Module
Generates responsive Plotly interactive charts with a soothing, eye-friendly design system:
warm slate backgrounds, soft curves, balanced pastel-tinted accents, and clean typography.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Soothing Color Palette
COLOR_NOISY = "#f43f5e"     # Warm Rose/Coral for noisy input
COLOR_PROCESSED = "#10b981" # Mint/Emerald for cleaned output
COLOR_CLEAN = "#0ea5e9"     # Calm Sky Blue for ground-truth reference
COLOR_ACCENT = "#6366f1"    # Soft Indigo for primary metrics
COLOR_BG = "#f8fafc"        # Very light soothing slate
COLOR_GRID = "rgba(226, 232, 240, 0.7)"
COLOR_TEXT = "#1e293b"      # Deep slate for high readability without harsh contrast

SOOTHING_LAYOUT = dict(
    font=dict(family="'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif", color=COLOR_TEXT, size=12),
    paper_bgcolor="rgba(248, 250, 252, 0.8)",
    plot_bgcolor="#ffffff",
    hoverlabel=dict(
        bgcolor="#1e293b",
        font=dict(color="#f8fafc", family="'Plus Jakarta Sans', 'Inter', sans-serif", size=12),
        bordercolor="rgba(0,0,0,0)"
    )
)


def plot_waveforms_plotly(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int = 16000,
    clean_ref: Optional[np.ndarray] = None,
    max_display_samples: int = 24000
) -> go.Figure:
    """
    Generate interactive Plotly dual/triple waveform comparison with soothing accents.
    """
    n_samples = min(len(original), len(processed))
    step = max(1, n_samples // max_display_samples)

    t = np.arange(0, n_samples, step) / sr
    orig_sub = original[:n_samples:step]
    proc_sub = processed[:n_samples:step]

    rows = 3 if clean_ref is not None else 2
    titles = ("🔴 Noisy / Input Signal", "🟢 Denoised / Enhanced Signal")
    if clean_ref is not None:
        titles = ("🔴 Noisy / Input Signal", "🟢 Denoised / Enhanced Signal", "🔵 Clean Reference Speech")

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.10, subplot_titles=titles)

    fig.add_trace(
        go.Scatter(
            x=t, y=orig_sub, mode="lines", name="Noisy Input",
            line=dict(color=COLOR_NOISY, width=1.4),
            fill="tozeroy", fillcolor="rgba(244, 63, 94, 0.06)"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=proc_sub, mode="lines", name="Processed Output",
            line=dict(color=COLOR_PROCESSED, width=1.4),
            fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.06)"
        ),
        row=2, col=1
    )

    if clean_ref is not None:
        ref_sub = clean_ref[:n_samples:step]
        fig.add_trace(
            go.Scatter(
                x=t, y=ref_sub, mode="lines", name="Clean Reference",
                line=dict(color=COLOR_CLEAN, width=1.4),
                fill="tozeroy", fillcolor="rgba(14, 165, 233, 0.06)"
            ),
            row=3, col=1
        )

    fig.update_layout(
        **SOOTHING_LAYOUT,
        height=180 * rows + 60,
        margin=dict(l=45, r=25, t=40, b=35),
        hovermode="x unified",
        showlegend=False
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLOR_GRID, zeroline=False, title_text="Time (seconds)")
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID, range=[-1.05, 1.05], zeroline=True, zerolinecolor=COLOR_GRID)
    return fig


def plot_spectrograms_plotly(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128
) -> go.Figure:
    """
    Generate synchronized dual spectrograms in dB with shared soothing color range.
    """
    window = signal.windows.hann(n_fft, sym=False)
    f, t_orig, z_orig = signal.stft(original, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, t_proc, z_proc = signal.stft(processed, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)

    db_orig = 20 * np.log10(np.maximum(np.abs(z_orig), 1e-5))
    db_proc = 20 * np.log10(np.maximum(np.abs(z_proc), 1e-5))

    vmax = max(np.max(db_orig), np.max(db_proc))
    vmin = vmax - 60.0  # 60 dB dynamic range display

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Noisy Input Spectrogram (dB)", "Enhanced Output Spectrogram (dB)"),
        shared_yaxes=True,
        horizontal_spacing=0.08
    )

    fig.add_trace(
        go.Heatmap(
            x=t_orig, y=f, z=db_orig,
            colorscale="Cividis", zmin=vmin, zmax=vmax,
            showscale=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            x=t_proc, y=f, z=db_proc,
            colorscale="Cividis", zmin=vmin, zmax=vmax,
            colorbar=dict(title="dB", len=0.85, thickness=14)
        ),
        row=1, col=2
    )

    fig.update_layout(
        **SOOTHING_LAYOUT,
        height=360,
        margin=dict(l=55, r=25, t=45, b=40)
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1, gridcolor=COLOR_GRID)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2, gridcolor=COLOR_GRID)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1, gridcolor=COLOR_GRID)
    return fig


def plot_filter_frequency_response_plotly(
    frequencies: np.ndarray,
    magnitude_db: np.ndarray,
    filter_name: str = "Filter Frequency Response",
    cutoff_lines: Optional[List[float]] = None
) -> go.Figure:
    """
    Plot filter magnitude response in decibels with cutoff indicators.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frequencies, y=magnitude_db,
            mode="lines",
            name="Magnitude (dB)",
            line=dict(color="#4f46e5", width=2.4),
            fill="tozeroy",
            fillcolor="rgba(79, 70, 229, 0.05)"
        )
    )

    # 0 dB reference line and -3 dB cutoff line
    fig.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8", annotation_text="0 dB (Unity)", annotation_font_size=11)
    fig.add_hline(y=-3.0, line_dash="dot", line_color="#f43f5e", annotation_text="-3 dB Cutoff", annotation_font_size=11)

    if cutoff_lines:
        for c in cutoff_lines:
            fig.add_vline(x=c, line_dash="dash", line_color="#f59e0b", annotation_text=f"{c:.0f} Hz", annotation_font_size=10)

    fig.update_layout(
        **SOOTHING_LAYOUT,
        title=dict(text=filter_name, font=dict(size=14, color=COLOR_TEXT)),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        yaxis=dict(range=[max(-80.0, float(np.min(magnitude_db)) - 5), 5.0], gridcolor=COLOR_GRID),
        xaxis=dict(gridcolor=COLOR_GRID),
        height=340,
        margin=dict(l=55, r=30, t=50, b=40)
    )
    return fig


def plot_ml_classification_bars_plotly(
    class_probabilities: Dict[str, float],
    predicted_class: str
) -> go.Figure:
    """
    Bar chart showing classification probabilities across candidate noise classes.
    """
    labels = list(class_probabilities.keys())
    probs = [class_probabilities[k] * 100 for k in labels]
    colors = [
        "#10b981" if k == predicted_class else "#cbd5e1"
        for k in labels
    ]

    clean_labels = [
        k.replace("_", " ").title() for k in labels
    ]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=clean_labels,
            orientation="h",
            marker=dict(color=colors, cornerradius=6),
            text=[f"  {p:.1f}%" for p in probs],
            textposition="auto"
        )
    )

    fig.update_layout(
        **SOOTHING_LAYOUT,
        title=dict(text="ML Noise Classification Confidence", font=dict(size=14, color=COLOR_TEXT)),
        xaxis_title="Probability (%)",
        xaxis=dict(range=[0, 105], gridcolor=COLOR_GRID),
        yaxis=dict(gridcolor="transparent"),
        height=300,
        margin=dict(l=150, r=30, t=50, b=40)
    )
    return fig


def plot_comparison_metrics_plotly(comparison_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Bar chart comparing methods across processing time and SNR improvement (if available).
    """
    methods = [d["method"] for d in comparison_data]
    runtimes = [d.get("processing_time_sec", 0.0) * 1000 for d in comparison_data]  # ms

    has_snr = any("snr_improvement_db" in d for d in comparison_data)

    if has_snr:
        snr_improvements = [d.get("snr_improvement_db", 0.0) for d in comparison_data]
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("SNR Improvement (dB, higher = cleaner)", "Processing Runtime (ms, lower = faster)")
        )

        colors = ["#10b981" if s >= 0 else "#f43f5e" for s in snr_improvements]
        fig.add_trace(
            go.Bar(x=methods, y=snr_improvements, marker_color=colors, marker_cornerradius=6, name="SNR Imp. (dB)"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=methods, y=runtimes, marker_color="#6366f1", marker_cornerradius=6, name="Runtime (ms)"),
            row=1, col=2
        )
        fig.update_xaxes(tickangle=-25)
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=methods, y=runtimes, marker_color="#6366f1", marker_cornerradius=6, name="Runtime (ms)"))
        fig.update_layout(title="Processing Runtime Comparison (ms)")

    fig.update_layout(
        **SOOTHING_LAYOUT,
        height=360,
        margin=dict(l=45, r=25, t=55, b=60),
        showlegend=False
    )
    fig.update_xaxes(gridcolor=COLOR_GRID)
    fig.update_yaxes(gridcolor=COLOR_GRID)
    return fig
