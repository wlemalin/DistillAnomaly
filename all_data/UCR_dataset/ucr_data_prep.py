#!/usr/bin/env python3
"""
Crop & scale .txt time-series + create **exact** plots used in generate_dataset.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scipy.signal import spectrogram
import argparse
import random
import re
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# REAL helpers (same as generate_dataset.py) -----------------------
from src.data.utils.plot_utils import (
    plot_series_and_predictions,
    plot_and_save_rolling_plots,
    compute_rolling_stats_with_centroid,
)
from src.data.utils.scale_utils import (
    series_to_scaled_float_and_tokens_00_99,
    float_df_to_scaled_float_and_tokens_fixed_00_99,
)
# ------------------------------------------------------------------


# ---------- small helper for consistent ticks + grid ----------
def _apply_time_ticks_and_grid(ax, xmax: int, step: int = 25) -> None:
    ax.set_xticks(np.arange(0, max(0, int(xmax)) + 1, step))
    ax.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
# --------------------------------------------------------------


def extract_crop_indices(filename: str, length: int) -> tuple[int, int, int, int]:
    """
    Extract anomaly [start, end] from filename.
    Then randomly sample a valid 300-sample window that contains the full anomaly.
    Returns: (crop_start, crop_end, new_start, new_end)
    """
    match = re.search(r"_(\d+)_(\d+)\.txt$", filename)
    if not match:
        raise ValueError(
            f"Filename '{filename}' doesn't match *_start_end.txt pattern.")
    start, end = int(match.group(1)), int(match.group(2))

    lower_bound = max(0, end - 280)
    upper_bound = min(start - 30, length - 300)

    if lower_bound > upper_bound:
        raise ValueError(
            f"File {filename} has anomaly range too wide or series too short.")

    crop_start = random.randint(lower_bound, upper_bound)
    crop_end = crop_start + 300

    new_start = start - crop_start
    new_end = end - crop_start

    if not (0 <= new_start < new_end <= 300):
        raise ValueError(
            f"Cropping logic failed to contain anomaly in {filename}")

    return crop_start, crop_end, new_start, new_end


def load_and_crop(txt_path: Path) -> tuple[np.ndarray, int, int]:
    """Load single-column float data, crop, and return (data, new_start, new_end)."""
    try:
        data = np.loadtxt(txt_path)
        if data.ndim == 1:
            data = data[:, None]
    except Exception as e:
        raise ValueError(f"Could not read {txt_path}: {e}")

    crop_start, crop_end, new_start, new_end = extract_crop_indices(
        txt_path.name, len(data))
    return data[crop_start:crop_end], new_start, new_end


def txt_to_csv(txt_path: Path, window: int):
    """Process one .txt file and save to CSV + **exact** plots."""
    root_dir = txt_path.parent
    csv_dir = root_dir / "../csv_data"
    figs_dir = root_dir / "../figs_data"
    csv_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    # ── Crop first ──
    series, new_start, new_end = load_and_crop(txt_path)

    # ── 1) Scale to token space (0-99) exactly like generate_dataset.py ----------
    scaled_float, _, _, _, _ = series_to_scaled_float_and_tokens_00_99(series)

    # build feature dataframe & scale it
    df_float = compute_rolling_stats_with_centroid(series[:, 0], window=window, fs=1.0)
    features_scaled_float, _, columns, _ = float_df_to_scaled_float_and_tokens_fixed_00_99(df_float)

    # ── File stem and base names ──
    orig_stem = txt_path.stem
    base = f"{orig_stem}_{new_start}_{new_end}"

    # ── 2) Save rolling diagnostics (mean, std, spectrogram) --------------------
    # prepare DataFrame in token-unit space (0-1) for plotting
    df_stats = pd.DataFrame({
        f"rolling_mean_{window}": features_scaled_float[:, columns.index(f"rolling_mean_{window}")] / 100.0,
        f"rolling_std_{window}" : features_scaled_float[:, columns.index(f"rolling_std_{window}")]  / 100.0,
    })
    plot_and_save_rolling_plots(
        df_stats,
        window=window,
        std_plot_path=str(figs_dir / f"{base}_std.png"),
        mean_plot_path=str(figs_dir / f"{base}_mean.png"),
        series=scaled_float[:, 0] / 100.0,               # token-unit 0-1
        fs=1.0,
        stft_plot_path=str(figs_dir / f"{base}_stft.png"),
    )

    # ── 3) Save RAW time-series plot (token-unit y-axis, 600×300) --------------
    raw_path = figs_dir / f"{base}_raw.png"
    fig = plot_series_and_predictions(
        series=scaled_float / 100.0,                     # (L, 1) in token-unit space
        gt_anomaly_intervals=[[(new_start, new_end)]],
        single_series_figsize=(20, 3),
    )
    fig.savefig(raw_path, dpi=100)   # 600×300 px
    plt.close(fig)

    # ── 4) Build & save CSV (rounding/scaling) ---------------------------------
    # re-use already-computed feature tokens
    df_out = pd.DataFrame(
        np.concatenate([scaled_float[:, :1], features_scaled_float], axis=1),
        columns=["sensor_1"] + columns
    ).astype(int).astype(str)          # 00-99 tokens as strings
    df_out.index.name = "time_step"

    csv_name = f"{base}.csv"
    csv_path = csv_dir / csv_name
    df_out.to_csv(csv_path, index=True)


def walk_and_convert_txt(start_dir: Path, window: int):
    """Find all matching .txt files and process them."""
    txt_files = sorted(start_dir.rglob("*_[0-9]*_[0-9]*.txt"), key=lambda p: str(p))
    for txt_path in tqdm(txt_files, desc="Processing .txt files"):
        try:
            txt_to_csv(txt_path, window)
        except Exception as e:
            print(f"Skipping {txt_path.name}: {e}")


# ─────────────────────────────── Entry Point ──────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop & scale .txt time-series + create **exact** plots used in generate_dataset.py.",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Directory containing .txt files with anomaly timestamps in names.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Rolling window length for mean/std (default: 30).",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    start_dir = Path(args.path).expanduser().resolve()
    if start_dir.is_file() and start_dir.suffix == ".txt":
        txt_to_csv(start_dir, args.window)
    else:
        walk_and_convert_txt(start_dir, args.window)
