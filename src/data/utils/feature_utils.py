# feature_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import spectrogram


@dataclass(frozen=True)
class FeatureConfig:
    """Compact config for rolling & spectral features."""
    window: int = 30
    fs: float = 1.0
    nperseg_factor: int = 2   # nperseg = window * 2
    noverlap_factor: int = 1  # noverlap = window * 1
    rolling_min_periods: Optional[int] = None
    rolling_ddof: int = 0


def rescale_minus1_1_to_0_1(x: np.ndarray) -> np.ndarray:
    """Map [-1,1] values to [0,1]."""
    return (x / 2.0) + 0.5


def rolling_mean_std_1d(
    x: np.ndarray,
    window: int,
    *,
    min_periods: int,
    ddof: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean & std along a 1-D array using NumPy only."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    mean = np.full(n, np.nan, dtype=float)
    std = np.full(n, np.nan, dtype=float)

    # cumul for mean
    csum = np.cumsum(np.insert(x, 0, 0.0))
    # cumul des carrés pour std
    csum2 = np.cumsum(np.insert(x * x, 0, 0.0))

    for i in range(n):
        start = max(0, i - window + 1)
        length = i - start + 1
        if length < min_periods:
            continue
        s1 = csum[i + 1] - csum[start]
        s2 = csum2[i + 1] - csum2[start]
        m = s1 / length
        v = (s2 / length) - (m * m)
        # stabilité numérique
        v = max(v, 0.0)
        mean[i] = m
        std[i] = np.sqrt(v) if ddof == 0 else np.sqrt(
            v * length / max(length - ddof, 1))

    return mean, std


def stft_centroid_1d(
    x: np.ndarray,
    *,
    fs: float,
    window: int,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    """Compute STFT spectral centroid interpolated to full series length."""
    x = np.asarray(x, dtype=float)
    freqs, times, Sxx = spectrogram(
        x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
    )
    P = np.abs(Sxx)
    cent = np.sum(freqs[:, None] * P, axis=0) / (np.sum(P, axis=0) + 1e-10)

    # Interpolation frames -> full length
    frame_pos = np.linspace(0, len(x) - 1, num=len(cent))
    cent_full = np.interp(np.arange(len(x)), frame_pos, cent)
    return cent_full


def compute_features_for_series(
    series: np.ndarray,
    cfg: FeatureConfig,
    *,
    rescale_to_0_1: bool = True,
    dtype: np.dtype = np.float32,
) -> Dict[str, np.ndarray]:
    """Extract rolling mean/std and STFT centroid for each channel."""
    arr = np.asarray(series)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"series doit être 1D ou 2D, reçu shape={arr.shape}")

    values = arr.astype(np.float32, copy=False)
    if rescale_to_0_1:
        values = rescale_minus1_1_to_0_1(values)

    T, C = values.shape
    min_periods = cfg.rolling_min_periods if cfg.rolling_min_periods is not None else cfg.window

    rolling_mean = np.full((T, C), np.nan, dtype=np.float32)
    rolling_std = np.full((T, C), np.nan, dtype=np.float32)
    centroid = np.full((T, C), np.nan, dtype=np.float32)

    nperseg = cfg.window * cfg.nperseg_factor
    noverlap = cfg.window * cfg.noverlap_factor

    for c in range(C):
        x = values[:, c].astype(float, copy=False)

        m, s = rolling_mean_std_1d(
            x, cfg.window, min_periods=min_periods, ddof=cfg.rolling_ddof
        )
        rolling_mean[:, c] = m.astype(np.float32, copy=False)
        rolling_std[:, c] = s.astype(np.float32, copy=False)

        cent_full = stft_centroid_1d(
            x, fs=cfg.fs, window=cfg.window, nperseg=nperseg, noverlap=noverlap
        )
        centroid[:, c] = cent_full.astype(np.float32, copy=False)

    return {
        "values": values.astype(dtype, copy=False),
        "rolling_mean": rolling_mean.astype(dtype, copy=False),
        "rolling_std": rolling_std.astype(dtype, copy=False),
        "stft_centroid": centroid.astype(dtype, copy=False),
    }

##################################################
# Preprocessing for tokenization
##################################################

def scale_round_and_pad(arr: np.ndarray) -> np.ndarray:
    """Scale [0,1] to 0-100, round, and format as 2-digit strings (NaN→empty)."""
    scaled = np.where(np.isnan(arr), np.nan, np.rint(arr * 100))

    def _fmt(v) -> str:
        if np.isnan(v):
            return ""
        v_int = int(v)
        if v_int < 0:
            return f"-{abs(v_int):02d}"
        else:
            return f"{v_int:02d}"

    vfmt = np.vectorize(_fmt, otypes=[object])
    return vfmt(scaled)


def series_to_float_dataframe(series: np.ndarray, window: int, fs: float = 1.0) -> pd.DataFrame:
    """Convert raw series to DataFrame with rolling stats and STFT centroid."""
    series = np.asarray(series)
    if series.ndim == 1:
        series = series[:, None]

    # 1) Rescale raw into [0,1]
    series = (series / 2) + 0.5

    # 2) Raw DF
    cols = [f"sensor_{i+1}" for i in range(series.shape[1])]
    df = pd.DataFrame(series, columns=cols)

    # 3) Rolling mean & std (pandas std ddof=1 par défaut)
    for col in cols:
        df[f"{col}_mean"] = df[col].rolling(window, min_periods=window).mean()
        df[f"{col}_std"] = df[col].rolling(window, min_periods=window).std()

    # 4) Spectral centroid * 10
    for col in cols:
        x = df[col].to_numpy(dtype=float)
        freqs, times, Sxx = spectrogram(
            x, fs=fs, window="hann", nperseg=window*2, noverlap=window)
        P = np.abs(Sxx)
        cent = np.sum(freqs[:, None] * P, axis=0) / (np.sum(P, axis=0) + 1e-10)
        frame_pos = np.linspace(0, len(x) - 1, num=len(cent))
        cent_full = np.interp(np.arange(len(x)), frame_pos, cent)
        df[f"{col}_centroid"] = cent_full * 10

    df.index.name = "time_step"
    return df


def float_df_to_tokens(df_float: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Convert DataFrame floats to 2-digit string tokens."""
    arr = df_float.to_numpy(dtype=float)
    tok = scale_round_and_pad(arr)
    return tok, list(df_float.columns)
