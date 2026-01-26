#!/usr/bin/env python3
from statistics import median
import json
import re
import sys
from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional, Tuple


TEMPDIFF_TOL = 0.25   # tolerance for speed ratio
ALT_TOL = 0.10        # tie-breaker on alternation density
EPS = 1e-9

# Compatibility stub so other code importing this symbol won't crash.
FIXED_DESCRIPTIONS: Dict[str, str] = {}

# ──────────────────────────────────────────────────────────────────────
# Parsing from your 'input' prompt (first numeric value after colon)
# ──────────────────────────────────────────────────────────────────────
def parse_series_with_timestamps(input_text: str) -> Tuple[List[int], List[float]]:
    """
    Parse lines like '29: 16, 53, 38' and return (timestamps, values).
    Uses the FIRST numeric value after the colon (the 'value' column).
    """
    series: List[float] = []
    timestamps: List[int] = []
    regex = re.compile(r'^\s*(\d+):\s*([-+]?\d*\.?\d+)\s*,')
    for line in input_text.splitlines():
        m = regex.match(line)
        if m:
            ts = int(m.group(1))
            val = float(m.group(2))
            timestamps.append(ts)
            series.append(val)
    return timestamps, series

# ──────────────────────────────────────────────────────────────────────
# Helpers (window mapping, baseline selection, smoothing, diffs)
# ──────────────────────────────────────────────────────────────────────
def _slice_by_ts(timestamps: List[int], start_ts: int, end_ts: int) -> Tuple[int, int]:
    """Map timestamp window to index window (inclusive), clamped and ordered."""
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    if not timestamps:
        return 0, 0
    first_ts = timestamps[0]
    s_idx = ts_to_idx.get(start_ts, start_ts - first_ts)
    e_idx = ts_to_idx.get(end_ts, end_ts - first_ts)
    n = len(timestamps)
    s_idx = max(0, min(s_idx, n - 1))
    e_idx = max(0, min(e_idx, n - 1))
    if s_idx > e_idx:
        s_idx, e_idx = e_idx, s_idx
    return s_idx, e_idx

def _choose_baseline(n: int, s_idx: int, e_idx: int) -> Optional[Tuple[int, int]]:
    """Same-length baseline: prefer BEFORE, else AFTER, else best available."""
    win_len = e_idx - s_idx + 1
    if win_len < 2:
        return None
    pre_start, pre_end = s_idx - win_len, s_idx - 1
    post_start, post_end = e_idx + 1, e_idx + win_len
    if pre_start >= 0:
        return pre_start, pre_end
    if post_end <= n - 1:
        return post_start, post_end
    left_len = max(0, s_idx)
    right_len = max(0, n - 1 - e_idx)
    if left_len >= right_len and left_len >= 2:
        return 0, s_idx - 1
    if right_len >= 2:
        return e_idx + 1, n - 1
    return None

def _smooth(vals: List[float], k: int = 3) -> List[float]:
    """Tiny moving average smoothing (odd k). Keeps edges unchanged."""
    if k <= 1 or len(vals) < k:
        return vals[:]
    w = k // 2
    out: List[float] = []
    run = sum(vals[:k])
    out.extend([vals[i] for i in range(w)])
    out.append(run / k)
    for i in range(k, len(vals)):
        run += vals[i] - vals[i - k]
        out.append(run / k)
    out.extend([vals[i] for i in range(len(vals) - w, len(vals))])
    return out

def _first_diff_per_time(vals: List[float], ts: List[int]) -> List[float]:
    """(x[i+1]-x[i]) / max(1, dt)."""
    diffs: List[float] = []
    for i in range(len(vals) - 1):
        dt = ts[i + 1] - ts[i]
        if dt <= 0:
            dt = 1
        diffs.append((vals[i + 1] - vals[i]) / dt)
    return diffs

def _mad(vals: List[float]) -> float:
    """Median absolute deviation."""
    if not vals:
        return 0.0
    m = median(vals)
    return median([abs(v - m) for v in vals])

def _alt_density(diffs: List[float]) -> float:
    """Density of sign alternations in derivative."""
    if len(diffs) < 2:
        return 0.0
    def _sign(x: float) -> int:
        if abs(x) < EPS: return 0
        return 1 if x > 0 else -1
    prev = None
    cnt = 0
    for d in diffs:
        s = _sign(d)
        if s == 0: 
            continue
        if prev is None:
            prev = s
        elif s != prev:
            cnt += 1
            prev = s
    return cnt / max(1, len(diffs))

# ──────────────────────────────────────────────────────────────────────
# Baseline sentence helper
# ──────────────────────────────────────────────────────────────────────
def _baseline_sentence(pattern_type: str) -> str:
    if pattern_type == "noisy-point":
        return "Normally, the series is a noisy periodic sine wave centered on a stable level."
    if pattern_type == "point":
        return "Normally, the series is a periodic sine wave centered on a stable level."
    if pattern_type in ("freq", "noisy-freq"):
        return "Normally, the series is a (noisy) sine wave with a stable oscillation frequency."
    if pattern_type in ("trend", "noisy-trend"):
        return "Normally, the series is a sine wave whose baseline drifts slowly over time."
    if pattern_type == "range":
        return "Normally, the series stays within a narrow band with mild random noise."
    return "Normally, the series oscillates around a stable baseline."

def _with_indices(start_ts: int, end_ts: int) -> str:
    return f"From index {start_ts} to {end_ts}, "

# ──────────────────────────────────────────────────────────────────────
# Programmatic describers (baseline first, then “From A to B …”)
# ──────────────────────────────────────────────────────────────────────
def describe_range(series: List[float], timestamps: List[int], start_ts: int, end_ts: int) -> str:
    if not series or not timestamps:
        return "ERROR: empty series"
    n = len(series)
    s_idx, e_idx = _slice_by_ts(timestamps, start_ts, end_ts)
    base_span = _choose_baseline(n, s_idx, e_idx)

    normal = _baseline_sentence("range")
    if base_span is None:
        return f"{normal} {_with_indices(start_ts, end_ts)}the series exhibits a sustained deviation from this band."

    b_start, b_end = base_span
    bs_vals = series[b_start:b_end + 1]
    an_vals = series[s_idx:e_idx + 1]
    if not bs_vals or not an_vals:
        return f"{normal} {_with_indices(start_ts, end_ts)}the series exhibits a sustained deviation from this band."
    mu_b = sum(bs_vals) / len(bs_vals)
    mu_a = sum(an_vals) / len(an_vals)
    direction = "a sustained elevation above the baseline" if mu_a > mu_b else "a sustained depression below the baseline"
    return f"{normal} {_with_indices(start_ts, end_ts)}the series exhibits {direction}."

def describe_frequency_change_temporal(series: List[float], timestamps: List[int], start_ts: int, end_ts: int, noisy: bool=False) -> Optional[str]:
    """
    Decide higher/lower oscillation frequency vs baseline using:
    - primary: median(|Δx/Δt|)/MAD ratio,
    - tie-breaker: alternation density ratio.
    Returns a clean baseline-first description with indices.
    """
    if not series or not timestamps:
        return None

    n = len(series)
    s_idx, e_idx = _slice_by_ts(timestamps, start_ts, end_ts)
    base_span = _choose_baseline(n, s_idx, e_idx)
    if base_span is None:
        return _baseline_sentence("noisy-freq" if noisy else "freq") + " " + _with_indices(start_ts, end_ts) + "the series exhibits a different oscillation rhythm than usual."

    b_start, b_end = base_span
    an_vals = series[s_idx:e_idx + 1]
    bs_vals = series[b_start:b_end + 1]
    an_ts   = timestamps[s_idx:e_idx + 1]
    bs_ts   = timestamps[b_start:b_end + 1]

    if noisy:
        an_vals = _smooth(an_vals, k=3)
        bs_vals = _smooth(bs_vals, k=3)

    an_d = _first_diff_per_time(an_vals, an_ts)
    bs_d = _first_diff_per_time(bs_vals, bs_ts)
    if not an_d or not bs_d:
        return _baseline_sentence("noisy-freq" if noisy else "freq") + " " + _with_indices(start_ts, end_ts) + "the series exhibits a different oscillation rhythm than usual."

    scale_an = _mad(an_vals) + EPS
    scale_bs = _mad(bs_vals) + EPS
    an_speed = median([abs(x) for x in an_d]) / scale_an
    bs_speed = median([abs(x) for x in bs_d]) / scale_bs
    speed_ratio = an_speed / (bs_speed + EPS)
    alt_ratio = _alt_density(an_d) / (_alt_density(bs_d) + EPS)

    higher = (speed_ratio > 1 + TEMPDIFF_TOL) or (abs(speed_ratio - 1) <= TEMPDIFF_TOL and alt_ratio > 1 + ALT_TOL)
    lower  = (speed_ratio < 1 - TEMPDIFF_TOL) or (abs(speed_ratio - 1) <= TEMPDIFF_TOL and alt_ratio < 1 - ALT_TOL)

    normal = _baseline_sentence("noisy-freq" if noisy else "freq")
    if higher:
        qual = "a higher oscillation frequency than the baseline"
    elif lower:
        qual = "a lower oscillation frequency than the baseline"
    else:
        qual = "a slightly different oscillation frequency than the baseline"
    return f"{normal} {_with_indices(start_ts, end_ts)}the series exhibits {qual}."

def describe_trend(series: List[float], timestamps: List[int], start_ts: int, end_ts: int) -> str:
    """
    Trend anomalies in your data are always increases.
    Produce a baseline-first sentence, then explicitly say the trend increases over [start,end].
    """
    if not series or not timestamps:
        return "ERROR: empty series"

    n = len(series)
    s_idx, e_idx = _slice_by_ts(timestamps, start_ts, end_ts)
    base_span = _choose_baseline(n, s_idx, e_idx)
    normal = _baseline_sentence("trend")

    # If we can't compute a clean baseline, still state increase
    if base_span is None or (e_idx - s_idx + 1) < 2:
        return f"{normal} {_with_indices(start_ts, end_ts)}the series exhibits an increased upward trend over this interval."

    b_start, b_end = base_span
    bs_vals = series[b_start:b_end+1]
    an_vals = series[s_idx:e_idx+1]

    # Optional slope calc (for internal check / no numbers in output)
    def slope(vals: List[float]) -> Optional[float]:
        if len(vals) < 2:
            return None
        return (vals[-1] - vals[0]) / (len(vals) - 1)

    # We won’t change the wording based on the sign — per your rule it’s always an increase
    # but we still compute to avoid crashing on edge cases
    _ = slope(bs_vals)
    _ = slope(an_vals)

    return f"{normal} {_with_indices(start_ts, end_ts)}the series exhibits an increased upward trend over this interval compared to the baseline."

def describe_point_like(series: List[float], timestamps: List[int], start_ts: int, end_ts: int, noisy: bool=False) -> str:
    """Loss of periodic structure due to noise."""
    normal = _baseline_sentence("noisy-point" if noisy else "point")
    return f"{normal} {_with_indices(start_ts, end_ts)}the series exhibits extreme noise that disrupts the periodic pattern."

# ──────────────────────────────────────────────────────────────────────
# Public entry points (kept names for your other script)
# ──────────────────────────────────────────────────────────────────────
def classify(series: List[float], timestamps: List[int], start_ts: int, end_ts: int, baseline: float=50) -> str:
    """For 'range' patterns: baseline-first description with explicit indices."""
    return describe_range(series, timestamps, start_ts, end_ts)

# ──────────────────────────────────────────────────────────────────────
# Batch processing (unchanged interface)
# ──────────────────────────────────────────────────────────────────────
def process_jsonl(in_path: str, out_path: str):
    """
    Reads JSONL, adds 'ground_truth_description' per item, and writes a JSON file.
    """
    data: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print("⚠️ Skipping invalid JSON line")
                continue

            pt = item.get("pattern_type")
            gt = item.get("ground_truth")
            if gt and isinstance(gt, list) and len(gt) >= 1 and isinstance(gt[0], list):
                try:
                    start_ts, end_ts = gt[0]
                except Exception:
                    print("⚠️ Invalid ground_truth format, skipping description")
                    data.append(item)
                    continue

                timestamps, series = parse_series_with_timestamps(item.get("input", ""))
                if not series or not timestamps:
                    print("⚠️ Empty series or timestamps")
                    data.append(item)
                    continue

                desc: Optional[str] = None
                if pt == "range":
                    desc = describe_range(series, timestamps, start_ts, end_ts)
                elif pt in ("freq", "noisy-freq"):
                    desc = describe_frequency_change_temporal(series, timestamps, start_ts, end_ts, noisy=(pt == "noisy-freq"))
                elif pt in ("trend", "noisy-trend"):
                    desc = describe_trend(series, timestamps, start_ts, end_ts)
                elif pt in ("point", "noisy-point"):
                    desc = describe_point_like(series, timestamps, start_ts, end_ts, noisy=(pt == "noisy-point"))

                item["ground_truth_description"] = desc or "ERROR: could not compute description"

            data.append(item)

    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)

    print(f"Done. Processed {len(data)} items → {out_path}")

def process_directory(in_dir: Path, out_dir: Path) -> int:
    """
    Process every *.jsonl in in_dir, write {name}_gt_desc.json to out_dir.
    Returns the number of files processed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for in_file in sorted(in_dir.glob("*.jsonl")):
        out_file = out_dir / f"{in_file.stem}_gt_desc.json"
        try:
            print(f"\n=== Processing: {in_file.name} ===")
            process_jsonl(str(in_file), str(out_file))
            count += 1
        except Exception as e:
            print(f"❌ Error processing {in_file.name}: {e}")
    print(f"\nAll done. Processed {count} file(s). Output dir: {out_dir}")
    return count

def main():
    parser = argparse.ArgumentParser(
        description="Add baseline-first ground truth descriptions with explicit anomaly indices to JSONL evaluation files."
    )
    parser.add_argument("--dir", type=str, help="Directory containing *.jsonl files to process.")
    parser.add_argument("--out-dir", type=str, required=False, help="Directory to write outputs as {input_name}_gt_desc.json (defaults to --dir).")
    # Back-compat single-file mode (optional)
    parser.add_argument("--in-file", type=str, help="(Optional) Single input JSONL file path (fallback to DEFAULT_INPUT).")
    parser.add_argument("--out-file", type=str, help="(Optional) Single output JSON file path (fallback to DEFAULT_OUTPUT).")

    args = parser.parse_args()

    if args.dir:
        in_dir = Path(args.dir)
        if not in_dir.exists() or not in_dir.is_dir():
            print(f"❌ --dir not found or not a directory: {in_dir}")
            sys.exit(1)
        out_dir = Path(args.out_dir) if args.out_dir else in_dir
        process_directory(in_dir, out_dir)
    else:
        in_file = Path(args.in_file) if args.in_file else Path(DEFAULT_INPUT)
        out_file = Path(args.out_file) if args.out_file else Path(DEFAULT_OUTPUT)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"(single-file mode) {in_file} -> {out_file}")
        process_jsonl(str(in_file), str(out_file))

if __name__ == "__main__":
    main()
