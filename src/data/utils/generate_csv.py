#!/usr/bin/env python3
# generate_csv_tokens.py
"""
Convert pickled token matrices into per-series CSV files
containing only the 4 requested sensor columns.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

REQ_COLS = ["sensor_1", "sensor_1_mean", "sensor_1_std", "sensor_1_centroid"]


def load_pkl(pkl_path: Path) -> dict:
    """Load a pickle file and ensure it contains a dictionary."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{pkl_path} contains no dict.")
    return data


def extract_tokens_and_columns(payload: dict) -> tuple[list[np.ndarray], list[str]]:
    """Extract token matrices and column names from the payload dict."""
    toks = payload.get("features_tokens", None) or payload.get("tokens", None)
    cols = payload.get("feature_columns", None) or payload.get("columns", None)

    if toks is None:
        raise KeyError("Key 'features_tokens' not found in pickle.")
    if cols is None:
        raise KeyError("Key 'feature_columns' not found in pickle.")
    if not isinstance(toks, list):
        raise TypeError(
            "'features_tokens' must be a list (one matrix per series).")
    if not isinstance(cols, list):
        raise TypeError("'feature_columns' must be a list of column names.")

    return toks, cols


def pkl_to_csv(pkl_path: Path, sensor: int = 1):
    """Convert a single data.pkl into CSV files, one per series."""
    root_dir = pkl_path.parent
    csv_dir = root_dir / "csv_data"
    csv_dir.mkdir(exist_ok=True)

    payload = load_pkl(pkl_path)
    toks_list, cols = extract_tokens_and_columns(payload)

    # requested columns (sensor 1 by default)
    if sensor != 1:
        req = [f"sensor_{sensor}", f"sensor_{sensor}_mean",
               f"sensor_{sensor}_std", f"sensor_{sensor}_centroid"]
    else:
        req = REQ_COLS

    idxs = []
    for c in req:
        if c not in cols:
            raise KeyError(
                f"Required column '{c}' missing. Available (sample): {cols[:10]}")
        idxs.append(cols.index(c))

    bar_desc = f"→ {pkl_path.relative_to(Path.cwd())}"
    for i, tok_mat in enumerate(tqdm(toks_list, desc=bar_desc)):
        mat = np.asarray(tok_mat, dtype=object)
        if mat.ndim != 2:
            raise ValueError(
                f"features_tokens[{i}] is not 2D: shape={mat.shape}")

        sub = mat[:, idxs]
        df = pd.DataFrame(sub, columns=req, dtype="string")
        df.index.name = "time_step"

        out = csv_dir / f"series_{i+1:05d}.csv"
        df.to_csv(out, index=True)


def walk_and_convert(start: Path, sensor: int = 1):
    """Recursively find every data.pkl under start and convert it."""
    for pkl_path in start.rglob("data.pkl"):
        pkl_to_csv(pkl_path, sensor=sensor)


def main():
    ap = argparse.ArgumentParser(
        description="Convert data.pkl -> per-series CSV using only the 4 token columns."
    )
    ap.add_argument("path", type=str, help="Root folder or path to data.pkl")
    ap.add_argument("--sensor", type=int, default=1,
                    help="Sensor to export (1-based). default=1")
    args = ap.parse_args()

    p = Path(args.path).expanduser().resolve()
    if p.is_file() and p.name == "data.pkl":
        pkl_to_csv(p, sensor=args.sensor)
    else:
        walk_and_convert(p, sensor=args.sensor)


if __name__ == "__main__":
    main()
