from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

# ==============================
# Global render params (600x300)
# ============================
_IMG_W_PX = 600
_IMG_H_PX = 300
_DPI = 100
_FIGSIZE = (_IMG_W_PX / _DPI, _IMG_H_PX / _DPI)  # (6, 3)

# Margins (in data-x units and in y fraction)
_X_MARGIN_PCT = 0.02   # 2% of series length on each side
_Y_MARGIN_PCT = 0.05   # 5% of y-range

_FIG_ADJUST = dict(left=0.09, right=0.995, bottom=0.22, top=0.86)

# ---------- small helper for consistent ticks + grid ----------
def _apply_time_ticks_and_grid(ax, xmax: int, step: int = 25) -> None:
    """Add evenly-spaced x-ticks and a light grid."""
    ax.set_xticks(np.arange(0, max(0, int(xmax)) + 1, step))
    ax.grid(True, which="both", color="lightgray", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
# --------------------------------------------------------------


def _apply_margins_xy(ax, *, x_len: int, y_values: Optional[np.ndarray] = None) -> None:
    """Add a small margin on each side of the plots."""
    if x_len > 1:
        dx = max(1.0, x_len * _X_MARGIN_PCT)
        ax.set_xlim(-dx, (x_len - 1) + dx)

    if y_values is not None:
        y = np.asarray(y_values, dtype=float)
        if y.size:
            lo = np.nanmin(y)
            hi = np.nanmax(y)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                dy = (hi - lo) * _Y_MARGIN_PCT
                ax.set_ylim(lo - dy, hi + dy)


# -----------------------------
# Token y-axis helpers (00..99)
# -----------------------------
def _to_unit01(y: np.ndarray) -> np.ndarray:
    """
    Convert to [0,1] the same way as generate_csv: (x/2)+0.5,
    but only if it looks like the signal is in [-1,1].
    Otherwise assume it's already in [0,1].
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    lo, hi = np.nanmin(y), np.nanmax(y)
    # heuristic: looks like [-1,1]
    if np.isfinite(lo) and np.isfinite(hi) and (lo < -0.05) and (hi <= 1.05):
        return (y / 2.0) + 0.5
    return y


def _looks_like_m11(y: np.ndarray) -> bool:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return False
    lo, hi = np.nanmin(y), np.nanmax(y)
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo < -0.05) and (hi <= 1.05))


def _token_labels_for_ticks(ticks_tok: np.ndarray) -> list[str]:
    """Convert integer tokens to 2-digit strings."""
    ticks_tok = np.asarray(ticks_tok, dtype=int)
    ticks_tok = np.clip(ticks_tok, 0, 99)
    return [f"{v:02d}" for v in ticks_tok]


def _set_token_yaxis(ax, y_values: np.ndarray, *, pad_tokens: int = 1, max_ticks: int = 6) -> None:
    """
    Display y-axis labels as tokens 00..99, with a DYNAMIC range based on the plotted series:
      [min_token - pad_tokens, max_token + pad_tokens] clipped to [0,99].

    Important: tick positions are in DATA units, but labels are token strings.
    """
    y = np.asarray(y_values, dtype=float)
    if y.size == 0 or not np.any(np.isfinite(y)):
        ax.tick_params(axis="y", left=False, labelleft=False)
        return

    # Reference in [0,1] token space
    y_unit = _to_unit01(y)
    y_unit = np.clip(y_unit, 0.0, 1.0)

    # tokens in [0..99] (float, before rounding)
    tok_f = y_unit * 99.0
    tok_min = int(np.floor(np.nanmin(tok_f)))
    tok_max = int(np.ceil(np.nanmax(tok_f)))

    tok_min = max(0, tok_min - int(pad_tokens))
    tok_max = min(99, tok_max + int(pad_tokens))
    if tok_max < tok_min:
        tok_min, tok_max = 0, 99

    span = tok_max - tok_min

    # choose up to max_ticks ticks within [tok_min, tok_max]
    if span == 0:
        tick_tok = np.array([tok_min], dtype=int)
    else:
        n = min(int(max_ticks), span + 1)
        tick_tok = np.unique(np.rint(np.linspace(tok_min, tok_max, num=n)).astype(int))

    tick_unit = tick_tok.astype(float) / 100.0  # [0,1]

    # map token-unit ticks back into DATA space
    if _looks_like_m11(y):
        tick_data = 2.0 * (tick_unit - 0.5)  # inverse of (x/2)+0.5
        ylim_lo = 2.0 * ((tok_min / 100.0) - 0.5)
        ylim_hi = 2.0 * ((tok_max / 100.0) - 0.5)
    else:
        tick_data = tick_unit
        ylim_lo = tok_min / 100.0
        ylim_hi = tok_max / 100.0

    ax.set_yticks(tick_data)
    ax.set_yticklabels(_token_labels_for_ticks(tick_tok))
    # ax.set_ylabel("Value (tokens)")
    ax.tick_params(axis="y", left=True, labelleft=True)

    # Force scale to [min_token-1, max_token+1] (in DATA units)
    if np.isfinite(ylim_lo) and np.isfinite(ylim_hi) and (ylim_hi > ylim_lo):
        ax.set_ylim(ylim_lo, ylim_hi)


def _style_axis(
    ax,
    xmax: int,
    title: str,
    step: int = 25,
    *,
    y_tokens: bool = False,
    y_values: Optional[np.ndarray] = None,
) -> None:
    ax.set_title(title)
    ax.set_xlabel("Time")

    if y_tokens and (y_values is not None):
        _set_token_yaxis(ax, y_values=y_values, pad_tokens=1, max_ticks=6)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

    _apply_time_ticks_and_grid(ax, xmax=xmax, step=step)


def compute_rolling_stats_with_centroid(
    series: np.ndarray,
    window: int = 30,
    fs: float = 1.0,
) -> pd.DataFrame:
    """Return DataFrame with rolling mean, std and STFT-centroid (0-99 tokens)."""
    series = np.asarray(series, dtype=float)
    df = pd.DataFrame({"value": series})

    mean_col = f"rolling_mean_{window}"
    std_col  = f"rolling_std_{window}"
    df[mean_col] = df["value"].rolling(window, min_periods=1).mean()
    df[std_col]  = df["value"].rolling(window, min_periods=1).std(ddof=0)

    nperseg  = window * 2
    noverlap = window
    freqs, times, Sxx = spectrogram(
        series, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
    )
    P = np.abs(Sxx)
    if P.size == 0 or P.shape[1] == 0:
        cent_full = np.zeros(len(series), dtype=float)
    else:
        cent = np.sum(freqs[:, None] * P, axis=0) / (np.sum(P, axis=0) + 1e-10)

        # causal (end-of-window) frame positions
        hop = (nperseg - noverlap)
        frame_positions = (np.arange(len(cent), dtype=float) * hop) + (nperseg - 1)

        cent_full = np.interp(
            np.arange(len(series), dtype=float),
            frame_positions,
            cent,
            left=cent[0],
            right=cent[-1],
        )

    lo, hi = float(np.min(cent_full)), float(np.max(cent_full))
    cent_scaled = (cent_full - lo) / (hi - lo) * 99 if hi > lo else np.zeros_like(cent_full)

    centroid_col = f"stft_centroid_{window}"
    df[centroid_col] = np.round(cent_scaled).astype(int).clip(0, 99)
    return df


def plot_series_and_predictions(
    series: np.ndarray,
    gt_anomaly_intervals: list[list[tuple[int, int]]] | None,
    anomalies: Optional[dict] = None,
    single_series_figsize: tuple[int, int] = (20, 3),
):
    """
    600x300, title + x ticks + y token labels, with margins.
    """
    series = np.asarray(series)
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)

    if series.ndim == 1:
        y = series
        ax.plot(np.arange(len(series)), series, color="steelblue", label="sensor_1")
        xmax = len(series) - 1
        _apply_margins_xy(ax, x_len=len(series), y_values=y)
        _style_axis(ax, xmax=xmax, title="Raw values", step=25, y_tokens=True, y_values=y)
    else:
        # if multiple sensors, still label tokens based on all values pooled
        for i in range(series.shape[1]):
            ax.plot(np.arange(series.shape[0]), series[:, i], color="steelblue", label=f"sensor_{i+1}")
        xmax = series.shape[0] - 1
        y_all = series.reshape(-1)
        _apply_margins_xy(ax, x_len=series.shape[0], y_values=y_all)
        _style_axis(ax, xmax=xmax, title="Raw values", step=25, y_tokens=True, y_values=y_all)

    # ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.subplots_adjust(**_FIG_ADJUST)
    return fig


def plot_and_save_rolling_plots(
    df: pd.DataFrame,
    window: int,
    std_plot_path: str,
    mean_plot_path: str,
    *,
    series: Optional[np.ndarray] = None,
    fs: float = 1.0,
    stft_plot_path: Optional[str] = None,
) -> None:
    """Save 600×300 PNGs for rolling std, mean and (optionally) spectrogram with token y-axis."""
    Path(std_plot_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mean_plot_path).parent.mkdir(parents=True, exist_ok=True)
    if stft_plot_path:
        Path(stft_plot_path).parent.mkdir(parents=True, exist_ok=True)

    mean_col = f"rolling_mean_{window}"
    std_col  = f"rolling_std_{window}"
    filtered = df
    x = filtered.index.to_numpy()
    xmax = int(x.max()) if x.size else 0

    # 1) Rolling std plot
    fig_std, ax_std = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    y_std = filtered[std_col].to_numpy(dtype=float)
    ax_std.plot(x, y_std, color="tab:orange", label=std_col)
    _apply_margins_xy(ax_std, x_len=len(filtered), y_values=y_std)
    _style_axis(
        ax_std,
        xmax=xmax,
        title=f"Moving standard deviation (window size = {window})",
        step=25,
        y_tokens=True,
        y_values=y_std,
    )

    for ax in fig_std.axes:
        ax.set_ylim(-0.05, 1.05)  # -5..105 in token space
    # ax_std.legend(loc="upper right", fontsize=8, frameon=True)
    fig_std.subplots_adjust(**_FIG_ADJUST)
    fig_std.savefig(std_plot_path, dpi=_DPI)  # NO bbox_inches!
    plt.close(fig_std)

    # 2) Rolling mean plot
    fig_mean, ax_mean = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    y_mean = filtered[mean_col].to_numpy(dtype=float)
    ax_mean.plot(x, y_mean, color="tab:green", label=mean_col)
    _apply_margins_xy(ax_mean, x_len=len(filtered), y_values=y_mean)
    _style_axis(
        ax_mean,
        xmax=xmax,
        title=f"Moving average (window size = {window})",
        step=25,
        y_tokens=True,
        y_values=y_mean,
    )

    for ax in fig_mean.axes:
        ax.set_ylim(-0.05, 1.05)  # -5..105 in token space
    # ax_mean.legend(loc="upper right", fontsize=8, frameon=True)
    fig_mean.subplots_adjust(**_FIG_ADJUST)
    fig_mean.savefig(mean_plot_path, dpi=_DPI)  # NO bbox_inches!
    plt.close(fig_mean)

    # 3) Spectrogram plot (backward-only alignment: show energy at END of each window)
    if stft_plot_path is not None and series is not None and len(series) > 0:
        sig = np.asarray(series, dtype=float)

        stft_window = max(2, window // 2)  # half of rolling window size
        nperseg = stft_window * 2
        noverlap = stft_window             # 50% overlap

        freqs, times, Sxx = spectrogram(
            sig, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, mode="psd"
        )
        if Sxx.size == 0 or Sxx.shape[1] == 0:
            return

        P_db = 10 * np.log10(Sxx + 1e-12)
        n_frames = Sxx.shape[1]

        # ---- CAUSAL X-EDGES (end-of-window), stable, no missing first bin ----
        hop = (nperseg - noverlap)  # in samples
        ends = (np.arange(n_frames, dtype=float) * hop) + (nperseg - 1)  # end index per frame

        if n_frames == 1:
            x_edges = np.array([0.0, float(len(sig))], dtype=float)
        else:
            mids = 0.5 * (ends[:-1] + ends[1:])
            x_edges = np.concatenate(([0.0], mids, [float(len(sig))]))

        x_edges = np.clip(x_edges, 0.0, float(len(sig)))
        # enforce monotonic edges (pcolormesh requires it)
        x_edges = np.maximum.accumulate(x_edges)
        x_edges[-1] = float(len(sig))
        # -----------------------------------------------------

        # y edges
        dfreq = (freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        y_edges = np.concatenate(([max(0.0, freqs[0] - dfreq / 2)], freqs + dfreq / 2))

        fig_sp, ax_sp = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        ax_sp.pcolormesh(x_edges, y_edges, P_db, shading="auto", cmap="magma")

        _apply_margins_xy(ax_sp, x_len=len(sig), y_values=freqs)
        # Keep Hz labels for spectrogram (tokens here would be misleading)
        _style_axis(
            ax_sp,
            xmax=len(sig) - 1,
            title=f"Spectrogram (window size = {stft_window})",
            step=25,
            y_tokens=False,
        )

        fig_sp.subplots_adjust(**_FIG_ADJUST)
        fig_sp.savefig(stft_plot_path, dpi=_DPI)  # NO bbox_inches!
        plt.close(fig_sp)
