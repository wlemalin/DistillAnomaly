from typing import Any

import numpy as np


def _scale_to_0_99_float(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly rescale array to [0, 99] and return float32."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - vmin) / (vmax - vmin) * 99.0
    return np.clip(y, 0.0, 99.0).astype(np.float32)


def _scale_to_0_99_int(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly rescale array to [0, 99] and return int32."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(x, dtype=np.int32)
    y = (x - vmin) / (vmax - vmin) * 99.0
    y = np.rint(y).astype(np.int32)
    y = np.clip(y, 0, 99)
    return y


def _int_to_2digit_tokens(x_int: np.ndarray) -> np.ndarray:
    """Convert integers 0-99 to 2-digit string tokens."""
    flat = x_int.reshape(-1)
    tok = np.array([f"{int(v):02d}" for v in flat], dtype=object)
    return tok.reshape(x_int.shape)


def series_to_scaled_float_and_tokens_00_99(series_cut: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale each column to [0, 99] and return float, int, tokens, vmin, vmax."""
    series_cut = np.asarray(series_cut, dtype=np.float32)
    T, S = series_cut.shape
    scaled_float = np.zeros((T, S), dtype=np.float32)
    vmins = np.zeros((S,), dtype=np.float32)
    vmaxs = np.zeros((S,), dtype=np.float32)
    for j in range(S):
        col = series_cut[:, j]
        vmin = float(np.nanmin(col))
        vmax = float(np.nanmax(col))
        vmins[j] = vmin
        vmaxs[j] = vmax
        scaled_float[:, j] = _scale_to_0_99_float(col, vmin, vmax)
    scaled_int = np.rint(scaled_float).astype(np.int32)
    scaled_int = np.clip(scaled_int, 0, 99)
    tokens = _int_to_2digit_tokens(scaled_int)
    return scaled_float, scaled_int.astype(np.float32), tokens, vmins, vmaxs


def float_df_to_scaled_float_and_tokens_fixed_00_99(df_float) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Scale DataFrame to [0, 99] with centroid columns sharing global range."""
    columns = list(df_float.columns)
    X = df_float.to_numpy(dtype=np.float32)
    centroid_cols = [i for i, c in enumerate(
        columns) if "centroid" in c.lower()]
    if centroid_cols:
        centroid_vals = X[:, centroid_cols]
        cmin = float(np.nanmin(centroid_vals))
        cmax = float(np.nanmax(centroid_vals))
    else:
        cmin = cmax = 0.0
    T, D = X.shape
    scaled_float = np.zeros((T, D), dtype=np.float32)
    vmins = np.zeros((D,), dtype=np.float32)
    vmaxs = np.zeros((D,), dtype=np.float32)
    for j in range(D):
        col = X[:, j]
        if j in centroid_cols:
            vmin, vmax = cmin, cmax
        else:
            vmin = float(np.nanmin(col))
            vmax = float(np.nanmax(col))
        vmins[j] = vmin
        vmaxs[j] = vmax
        scaled_float[:, j] = _scale_to_0_99_float(col, vmin, vmax)
    scaled_int = np.rint(scaled_float).astype(np.int32)
    scaled_int = np.clip(scaled_int, 0, 99)
    tokens = _int_to_2digit_tokens(scaled_int)
    scale_info = {
        "vmins": vmins,
        "vmaxs": vmaxs,
        "centroid_global_min": float(cmin),
        "centroid_global_max": float(cmax),
    }
    return scaled_float, tokens, columns, scale_info
