#!/usr/bin/env python3
"""
generate_dataset.py – create a synthetic, token-ready time-series dataset for anomaly detection.

File layout
-----------
<data_dir>/
├── data.pkl                 # complete payload (see SyntheticDataset.save)
└── figs/
    ├── {id}.png             # raw series (token units)
    ├── {id}_mean.png        # rolling mean (token units)
    └── {id}_std.png         # rolling std  (token units)
    

PKL payload schema (stable)
---------------------------
series                  -> list[np.ndarray]        # 0-99 integers, shape (final_length, sensors)
series_raw              -> list[np.ndarray]        # original float,  shape (final_length, sensors)
series_tokens           -> list[np.ndarray]        # str, shape (final_length, sensors)
anom                    -> list[list[list[tuple[int,int]]]]  # per-sensor intervals (rebased)
features_float          -> list[np.ndarray]        # float engineered features, (final_length, feats)
features_tokens         -> list[np.ndarray]        # str tokens,               (final_length, feats)
feature_columns         -> list[str]               # column names in fixed order
series_scaled_float     -> list[np.ndarray]        # float in 0-99 before rounding, (final_length, sensors)
features_scaled_float   -> list[np.ndarray]        # float in 0-99 before rounding, (final_length, feats)
scale_info              -> list[dict]              # per-sample scaling metadata
factual_descriptions    -> list[Optional[str]]     # legacy: event_1 description
factual_description_map -> dict[tuple[pattern,split,id], str]  # legacy lookup
factual_description{1,2,3,4} -> list[Optional[str]] # four granularity levels
factual_description_map_4 -> dict[tuple[pattern,split,id,level], str]  # new lookup
anom_info               -> list[Optional[dict]]    # generator event matched to interval
meta                    -> dict                    # global metadata (generator, lengths, params, ...)

Quick start
-----------
# Train split, range anomalies, 400 series, 30-step rolling window
python synthesize_data.py \
        --generate \
        --pattern-type range \
        --split train \
        --num_series 400 \
        --window 30 \
        --seed 42
"""

from __future__ import annotations
import argparse
import os
import pickle
from typing import Any, Callable, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.anomaly_utils import (_choose_trim_start, _flatten_intervals,
                                 _match_event)
from utils.data_generators import (synthetic_flat_trend_anomalies,
                                   synthetic_freq_anomalies,
                                   synthetic_point_anomalies,
                                   synthetic_range_anomalies,
                                   synthetic_trend_anomalies)
from utils.explanation_utils import (build_factual_description_oracle,
                                     build_factual_descriptions_oracle)
from utils.feature_utils import series_to_float_dataframe
from utils.plot_utils import (plot_and_save_rolling_plots,
                              plot_series_and_predictions)
from utils.scale_utils import (float_df_to_scaled_float_and_tokens_fixed_00_99,
                               series_to_scaled_float_and_tokens_00_99)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOISE_STD: float = 0.08  # Gaussian noise std used when --add_noise
TOKEN_MIN: float = 0.0
TOKEN_MAX: float = 99.0
TOKEN_UNIT_DIV: float = 100.0  # Convert 0..99 token space to 0.00..0.99 for plotting


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _first_existing(columns: list[str], candidates: list[str]) -> Optional[str]:
    """
    Return the first candidate name that exists in 'columns', or None if none match.
    """
    colset = set(columns)
    return next((c for c in candidates if c in colset), None)


def _available_synthetic_funcs(namespace: dict[str, Any]) -> list[str]:
    """
    List available synthetic generators exposed in the given namespace.
    """
    return sorted([k for k, v in namespace.items() if k.startswith("synthetic_") and callable(v)])


def _resolve_synthetic_func(name: str, namespace: dict[str, Any]) -> Callable[..., Any]:
    try:
        fn = namespace[name]
    except KeyError as e:
        available = _available_synthetic_funcs(namespace)
        raise KeyError(f"'{name}' not found. Available: {available}") from e

    if not callable(fn):
        raise KeyError(f"'{name}' exists but is not callable.")
    return fn


def _extract_first_channel_raw(data_2d: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the first sensor/channel as a 1D float array when possible.

    Some downstream description builders benefit from raw (unscaled) values.
    If the array shape doesn't match expectations, return None (unchanged behavior).
    """
    if data_2d.ndim == 2 and data_2d.shape[1] >= 1:
        return data_2d[:, 0].astype(np.float32)
    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Torch Dataset wrapping synthetic series + derived features + anomaly intervals.

    Key characteristics:
      - Each accepted sample has exactly ONE flattened anomaly interval.
      - Series are generated longer (warmup+final_length), features computed on the
        long series, then the cut is applied so that anomaly is fully inside the cut.
      - Scaling/tokenization is done in 00..99
      - The saved 'factual_description_map' uses a (pattern_type, split, id) key with
        a 1-based id consistent with plot filenames.
    """

    def __init__(
        self,
        data_dir: str = "data/synthetic/range/",
        synthetic_func_name: str = "synthetic_range_anomalies",
    ):
        self.data_dir = data_dir
        self.figs_dir = os.path.join(data_dir, "figs")

        # Time series values
        self.series: list[np.ndarray] = []
        self.series_raw: list[np.ndarray] = []
        self.series_scaled_float: list[np.ndarray] = []
        self.series_tokens: list[np.ndarray] = []

        # Additional feature values
        self.features_float: list[np.ndarray] = []
        self.features_scaled_float: list[np.ndarray] = []
        self.features_tokens: list[np.ndarray] = []
        self.feature_columns: list[str] | None = None

        # Anomaly intervals (per-sensor).
        self.anom: list[list[list[tuple[int, int]]]] = []

        # Natural language description fields
        self.factual_descriptions: list[Optional[str]] = []
        self.factual_description_map: dict[tuple[str, str, int], str] = {}

        self.factual_description1: list[Optional[str]] = []
        self.factual_description2: list[Optional[str]] = []
        self.factual_description3: list[Optional[str]] = []
        self.factual_description4: list[Optional[str]] = []

        # Optional but very useful for downstream lookup
        self.factual_description_map_4: dict[tuple[str, str, int, str], str] = {}

        # Scaling metadata and optional generator/event metadata
        self.scale_info: list[dict[str, Any]] = []
        self.anom_info: list[Optional[dict[str, Any]]] = []

        # Dataset-level metadata
        self.meta: dict[str, Any] = {}

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)

        self.synthetic_func = _resolve_synthetic_func(
            synthetic_func_name, globals())
        print(f"Using generation function: {synthetic_func_name}")

    def generate(
        self,
        num_series: int = 400,
        seed: Optional[int] = 42,
        add_noise: bool = False,
        window: int = 30,
        final_length: int = 300,
        # warmup: int = 0,
        fs: float = 1.0,
        max_tries: int = 200_000,
        split: str = "train",
        pattern_type: str = "freq",
    ) -> None:
        """
        Generate a synthetic dataset, save it to PKL, and create plots.

        Notes on accepted samples :
          - Exactly one anomaly interval after flattening.
          - Cut start must satisfy internal constraints (trim_start >= window etc.).
          - Anomaly must be fully inside the final cut.
          - Feature columns must be consistent across all accepted samples.
        """
        np.random.seed(seed)
        print(
            f"Using {'fixed' if seed is not None else 'non-fixed'} random seed"
            + (f": {seed}" if seed is not None else ".")
        )

        # # If warmup is not provided, default to 2 * window to ensure stable rolling features.
        # warmup = warmup if warmup > 0 else (2 * window)
        # orig_length = final_length + warmup

        orig_length = final_length + 2 * window

        # Metadata for downstream reproducibility.
        self.meta = {
            "generator": self.synthetic_func.__name__,
            "pattern_type": pattern_type,
            "split": split,
            "num_series": int(num_series),
            "window": int(window),
            "fs": float(fs),
            "final_length": int(final_length),
            # "warmup": int(warmup),
            "orig_length": int(orig_length),
            "constraints": {
                "total_intervals_eq_1": True,
                "features_computed_on_long_series_then_cut": True,
                "trim_start_ge_window": True,
                "anomaly_fully_inside_cut": True,
            },
            "tokens": "local scale->round->clip (00..99); centroid uses global min/max over centroid columns",
            "plots": "plots use scaled-unrounded in token-unit space (divide by 100.0) for y-axis stability",
        }

        print(
            f"Generating {num_series} series using {self.synthetic_func.__name__}... "
            f"(orig_len={orig_length} -> cut={final_length}) | pattern_type={pattern_type} | split={split}"
        )

        accepted = 0
        tries = 0
        bar = tqdm(total=num_series, desc="Generating Series & Plots")

        while accepted < num_series:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(
                    f"Too many tries ({max_tries}). Adjust anomaly params or warmup/final_length."
                )

            # Generators differ slightly in expected argument names (test_size vs n_samples).
            # We keep the original branching logic to avoid any behavioral shift.
            gen_args: dict[str, Any] = {
                "number_of_sensors": 1,
                "ratio_of_anomalous_sensors": 1.0,
                # dataset-level seeding already handled via np.random.seed(seed)
                "seed": None,
            }
            if self.synthetic_func.__name__ == "synthetic_range_anomalies":
                gen_args["test_size"] = orig_length
            else:
                gen_args["n_samples"] = orig_length

            # Synthetic generation can fail (e.g., constraints not met inside generator),
            # so we retry silently just like the original code.
            try:
                ret = self.synthetic_func(**gen_args)
            except Exception:
                continue

            # Some generators optionally return (data, intervals, info).
            gen_info: Optional[dict[str, Any]] = None
            if isinstance(ret, tuple) and len(ret) == 3:
                data_long, anomaly_locations_long, gen_info = ret
            else:
                data_long, anomaly_locations_long = ret

            data_long = np.asarray(data_long, dtype=np.float32)

            if add_noise:
                # Noise is applied on the long series to match original behavior.
                data_long += np.random.normal(0, NOISE_STD,
                                              data_long.shape).astype(np.float32)

            # Enforce exactly one anomaly interval after flattening.
            flat = _flatten_intervals(anomaly_locations_long)
            if len(flat) != 1:
                continue
            anom_sensor, anom_start, anom_end = flat[0]

            # Choose a cut start so that:
            #   - the cut is final_length long
            #   - the anomaly remains fully inside the cut
            #   - trim_start respects window-related constraints
            trim_start = _choose_trim_start(
                orig_len=orig_length,
                final_len=final_length,
                window=window,
                anom_start=anom_start,
                anom_end=anom_end,
            )
            if trim_start is None:
                continue
            trim_end = trim_start + final_length

            # Compute engineered features on the long series first,
            # then slice to align features with the cut.
            try:
                df_float_long = series_to_float_dataframe(
                    data_long, window=window, fs=fs)
            except Exception:
                continue

            data_raw = data_long[trim_start:trim_end, :]
            df_float_cut = df_float_long.iloc[trim_start:trim_end]

            # Sanity check: cut lengths must match final_length exactly.
            if data_raw.shape[0] != final_length or len(df_float_cut) != final_length:
                continue

            # Rebase anomaly interval into the cut coordinates.
            new_start = anom_start - trim_start
            new_end = anom_end - trim_start
            if not (0 <= new_start < new_end <= final_length):
                continue

            # Tokenize/scale the raw series in 00..99 space.
            (
                series_scaled_float,
                series_scaled_int,
                data_tokens,
                s_vmins,
                s_vmaxs,
            ) = series_to_scaled_float_and_tokens_00_99(data_raw)

            # Tokenize/scale engineered features in a fixed 00..99 scheme.
            # Any failure -> reject sample, consistent with original behavior.
            try:
                features_scaled_float, tok_mat, columns, feat_scale = (
                    float_df_to_scaled_float_and_tokens_fixed_00_99(
                        df_float_cut)
                )
            except Exception:
                continue

            # Enforce consistent feature column ordering across the dataset.
            if self.feature_columns is None:
                self.feature_columns = list(columns)
            elif list(columns) != self.feature_columns:
                continue

            # Keep the original anomaly container structure:
            # list_of_sensors -> list_of_intervals (per sensor).
            new_anom = [[] for _ in range(1)]
            new_anom[anom_sensor] = [(int(new_start), int(new_end))]

            # Persist series + features + tokens.
            self.series.append(series_scaled_int.astype(np.float32))
            self.series_raw.append(data_raw.astype(np.float32))
            self.series_tokens.append(data_tokens)
            self.anom.append(new_anom)

            self.series_scaled_float.append(
                series_scaled_float.astype(np.float32))
            self.features_float.append(df_float_cut.to_numpy(dtype=np.float32))
            self.features_tokens.append(tok_mat)
            self.features_scaled_float.append(
                features_scaled_float.astype(np.float32))

            # Store scaling metadata so downstream can reverse/interpret scaling if needed.
            self.scale_info.append(
                {
                    "series_vmins": s_vmins,
                    "series_vmaxs": s_vmaxs,
                    "features": feat_scale,
                    "window": int(window),
                    "fs": float(fs),
                }
            )

            # 1-based index is used for:
            #   - plot filenames
            #   - factual_description_map key (pattern_type, split, id)
            series_idx = accepted + 1

            # Try to match the generated anomaly to an event description (if generator provides info).
            event_long = _match_event(
                gen_info, anom_sensor, anom_start, anom_end)
            event_cut: Optional[dict[str, Any]] = None
            if event_long is not None:
                event_cut = dict(event_long)
                event_cut["start"] = int(event_long["start"] - trim_start)
                event_cut["end"] = int(event_long["end"] - trim_start)



            # Build a factual description: oracle first (richer), fallback otherwise.
            gt_intervals_cut = [(int(new_start), int(new_end))]
            times_cut = list(range(final_length))
            values_cut = series_scaled_int[:, 0].astype(np.int32).tolist()
            raw_cut_1d = _extract_first_channel_raw(data_raw)

            descs = build_factual_descriptions_oracle(
                pattern_type=pattern_type,
                start=int(new_start),
                end=int(new_end),
                values_0_99=values_cut,
                rolling_avg_0_99=values_cut,
                raw_values=raw_cut_1d,
                event=event_cut,
                add_noise=add_noise,
            )

            d1 = descs.get("event_1")
            d2 = descs.get("event_2")
            d3 = descs.get("event_3")
            d4 = descs.get("event_4")

            self.factual_description1.append(d1)
            self.factual_description2.append(d2)
            self.factual_description3.append(d3)
            self.factual_description4.append(d4)

            # Keep the old field if you want it (use event_1 as the legacy one)
            self.factual_descriptions.append(d1)

            # Optional maps for easy lookup
            sid = int(series_idx)
            for lvl_name, txt in [("event_1", d1), ("event_2", d2), ("event_3", d3), ("event_4", d4)]:
                if txt is not None:
                    self.factual_description_map_4[(pattern_type, split, sid, lvl_name)] = str(txt)

            # Keep the original map too (legacy): point it to event_1
            if d1 is not None:
                self.factual_description_map[(pattern_type, split, sid)] = str(d1)

            self.anom_info.append(event_cut)

            # -------------------------------------------------------------------
            # Plot 1: basic series visualization (scaled_float in token-unit space)
            # -------------------------------------------------------------------
            fig = None
            try:
                fig = plot_series_and_predictions(
                    series=(series_scaled_float / TOKEN_UNIT_DIV),
                    single_series_figsize=(10, 1.5),
                    gt_anomaly_intervals=None,
                    anomalies=None,
                )

                # Keep y-axis stable across plots (-5..105 mapped to -0.05..1.05 in unit space).
                for ax in fig.axes:
                    ax.set_ylim(-0.05, 1.05)

                fig_path = os.path.join(self.figs_dir, f"{series_idx:d}.png")
                fig.savefig(fig_path)
            except Exception:
                pass
            finally:
                if fig is not None:
                    plt.close(fig)
                else:
                    plt.close()

            # -------------------------------------------------------------------
            # Plot 2: rolling mean/std + STFT-style diagnostics
            # -------------------------------------------------------------------
            try:
                std_p = os.path.join(self.figs_dir, f"{series_idx:d}_std.png")
                mean_p = os.path.join(
                    self.figs_dir, f"{series_idx:d}_mean.png")
                stft_p = os.path.join(
                    self.figs_dir, f"{series_idx:d}_stft.png")

                cols = self.feature_columns or []
                df_feat = pd.DataFrame(features_scaled_float, columns=cols)

                # Prefer feature columns if present; otherwise compute rolling stats as fallback.
                mean_src = _first_existing(
                    cols,
                    [
                        "sensor_1_mean",
                        f"rolling_mean_{window}",
                        f"rolling_mean_{int(window)}",
                    ],
                )
                std_src = _first_existing(
                    cols,
                    [
                        "sensor_1_std",
                        f"rolling_std_{window}",
                        f"rolling_std_{int(window)}",
                    ],
                )

                # Plotting uses token-unit space, not raw 0-99 integers.
                y_unit = (series_scaled_float[:, 0] /
                          TOKEN_UNIT_DIV).astype(np.float32)

                if mean_src is not None:
                    mean_unit = (df_feat[mean_src].to_numpy(
                        dtype=np.float32) / TOKEN_UNIT_DIV)
                else:
                    mean_unit = (
                        pd.Series(y_unit)
                        .rolling(window, min_periods=1)
                        .mean()
                        .to_numpy(dtype=np.float32)
                    )

                if std_src is not None:
                    std_unit = (df_feat[std_src].to_numpy(
                        dtype=np.float32) / TOKEN_UNIT_DIV)
                else:
                    std_unit = (
                        pd.Series(y_unit)
                        .rolling(window, min_periods=1)
                        .std(ddof=0)
                        .fillna(0.0)
                        .to_numpy(dtype=np.float32)
                    )

                df_stats = pd.DataFrame(
                    {
                        f"rolling_mean_{window}": mean_unit,
                        f"rolling_std_{window}": std_unit,
                    }
                )
                df_stats.index = np.arange(len(df_stats))

                plot_and_save_rolling_plots(
                    df_stats,
                    window,
                    std_p,
                    mean_p,
                    series=(data_raw[:, 0] if (data_raw.ndim ==
                            2 and data_raw.shape[1] >= 1) else None),
                    fs=fs,
                    stft_plot_path=stft_p,
                )
            except Exception:
                # Diagnostics are best-effort; failing here should not invalidate a sample.
                pass

            accepted += 1
            bar.update(1)

        bar.close()
        self.save()
        print(f"Finished generating {num_series} series and plots.")

    def save(self) -> None:
        """
        Serialize the dataset to a single PKL file.

        The payload schema is intentionally explicit and stable so downstream scripts
        can load older datasets safely using .get() fallbacks.
        """
        save_path = os.path.join(self.data_dir, "data.pkl")
        payload = {
            "series": self.series,
            "series_raw": self.series_raw,
            "series_tokens": self.series_tokens,
            "anom": self.anom,
            "features_float": self.features_float,
            "features_tokens": self.features_tokens,
            "feature_columns": self.feature_columns,
            "meta": self.meta,
            "factual_descriptions": self.factual_descriptions,
            "factual_description_map": self.factual_description_map,
            "factual_description1": self.factual_description1,
            "factual_description2": self.factual_description2,
            "factual_description3": self.factual_description3,
            "factual_description4": self.factual_description4,
            "factual_description_map_4": self.factual_description_map_4,
            "anom_info": self.anom_info,
            "series_scaled_float": self.series_scaled_float,
            "features_scaled_float": self.features_scaled_float,
            "scale_info": self.scale_info,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Data saved successfully to {save_path}")

    def load(self) -> None:
        """
        Load the dataset from PKL.

        Uses conservative .get() fallbacks to tolerate older payload schemas.
        """
        load_path = os.path.join(self.data_dir, "data.pkl")
        print(f"Attempting to load data from: {load_path}")
        if not os.path.exists(load_path):
            print(f"File not found: {load_path}")
            self.series, self.anom = [], []
            return

        with open(load_path, "rb") as f:
            payload = pickle.load(f)

        self.series = payload.get("series", payload.get("series_raw", []))
        self.series_raw = payload.get("series_raw", [])
        self.series_tokens = payload.get("series_tokens", [])
        self.anom = payload.get("anom", [])

        self.features_float = payload.get("features_float", [])
        self.features_tokens = payload.get("features_tokens", [])
        self.feature_columns = payload.get(
            "feature_columns", payload.get("columns", None))

        self.series_scaled_float = payload.get("series_scaled_float", [])
        self.features_scaled_float = payload.get("features_scaled_float", [])
        self.scale_info = payload.get("scale_info", [])

        self.meta = payload.get("meta", {})
        self.factual_descriptions = payload.get("factual_descriptions", [])
        self.factual_description_map = payload.get(
            "factual_description_map", {})

        self.factual_description1 = payload.get("factual_description1", [])
        self.factual_description2 = payload.get("factual_description2", [])
        self.factual_description3 = payload.get("factual_description3", [])
        self.factual_description4 = payload.get("factual_description4", [])
        self.factual_description_map_4 = payload.get("factual_description_map_4", {})

        self.anom_info = payload.get("anom_info", [])

        print(
            f"Loaded dataset '{os.path.basename(os.path.normpath(self.data_dir))}' with {len(self.series)} series."
        )

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, idx: int):
        """
        Return (anom_intervals_tensor, series_tensor).

        Only the first sensor is used for the returned interval tensor, matching
        the original structure where number_of_sensors=1.
        """
        if idx >= len(self.series):
            raise IndexError(f"Index {idx} out of range ({len(self.series)})")

        series_data = self.series[idx]
        series_tensor = torch.tensor(series_data, dtype=torch.float32)

        intervals = self.anom[idx][0] if idx < len(
            self.anom) and self.anom[idx] else []
        anom_tensor = (
            torch.tensor(intervals, dtype=torch.float32) if intervals else torch.empty(
                (0, 2), dtype=torch.float32)
        )
        return anom_tensor, series_tensor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    dataset = SyntheticDataset(args.data_dir, args.synthetic_func)

    if args.generate:
        dataset.generate(
            num_series=args.num_series,
            seed=args.seed,
            add_noise=args.add_noise,
            window=args.window,
            final_length=args.final_length,
            # warmup=args.warmup,
            fs=args.fs,
            split=args.split,
            pattern_type=args.pattern_type,
        )
        return

    dataset.load()
    if not dataset.series:
        print(f"No data loaded from {args.data_dir}. Use --generate.")
        return

    print(
        f"Dataset loaded successfully. Contains {len(dataset.series)} series.")
    num_with_anom_info = sum(
        1 for lst in dataset.anom if lst and lst[0]) if dataset.anom else 0
    print(
        f"Anomaly interval information loaded for {num_with_anom_info} series (first sensor).")

    if dataset.factual_descriptions:
        n_desc = sum(1 for d in dataset.factual_descriptions if d)
        print(
            f"Factual descriptions loaded: {n_desc}/{len(dataset.factual_descriptions)} (non-null)")

    if dataset.factual_description_map:
        print(
            f"Factual description map loaded: {len(dataset.factual_description_map)} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate/load synthetic dataset & plots + PKL enriched "
            "(features_float + features_tokens + factual_description)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    available_funcs = _available_synthetic_funcs(globals())
    default_func = "synthetic_range_anomalies"
    if not available_funcs:
        available_funcs = [default_func]
    elif default_func not in available_funcs:
        default_func = available_funcs[0]
    
    parser.add_argument("--num_series", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str,
                        default="all_data/synthetic/range/")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--add_noise", action="store_true")


    parser.add_argument(
        "--synthetic_func",
        type=str,
        default=default_func,
        choices=sorted(list(set(available_funcs))),
    )

    parser.add_argument(
        "--pattern-type",
        type=str,
        required=True,
        choices=["range", "point", "freq", "trend",
                 "noisy-point", "noisy-freq", "noisy-trend"],
        help="Pattern label stored in PKL and used downstream as key (pattern_type, split, id).",
    )
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--fs", type=float, default=1.0)
    parser.add_argument("--final_length", type=int, default=300)
    # parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "eval", "test"])

    args = parser.parse_args()

    if args.synthetic_func not in globals() or not callable(globals()[args.synthetic_func]):
        print(
            f"FATAL ERROR: Selected function '{args.synthetic_func}' not found/callable.")
        raise SystemExit(1)

    try:
        main(args)
        print("Script finished successfully.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()
        print("Script finished with errors.")
