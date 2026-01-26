#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-JSONL Affiliation Metrics Summary

Processes multiple JSONL files using the affiliation metrics backend logic
and produces a consolidated summary table. Accepts JSONL files as command-line
arguments for precise control over order and filtering.

Usage:
    python multi_affiliation_summary.py file1.jsonl file2.jsonl file3.jsonl
    python multi_affiliation_summary.py --synth --out summary.json results/*.jsonl
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# ========== COPY YOUR ENTIRE BACKEND LOGIC HERE ==========
# (All the imports, dataclasses, parsing functions, metric computation, etc.)


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _try_json_loads_maybe_double_encoded(s: str) -> Any:
    obj = json.loads(s)
    if isinstance(obj, str):
        obj2 = json.loads(obj)
        return obj2
    return obj


def _extract_json_object_text(raw: str) -> Optional[str]:
    s = raw.strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i: j + 1].strip()
    return None


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    def length(self) -> int:
        return max(0, self.end - self.start)

    def distance_to_point(self, t: int) -> int:
        if t < self.start:
            return self.start - t
        if t >= self.end:
            return t - self.end
        return 0


def clip_to_domain(iv: Interval, T0: int, T1: int) -> Interval:
    s = max(iv.start, T0)
    e = min(iv.end, T1)
    if e < s:
        e = s
    return Interval(s, e)


def survival_precision(d: int, len_gt: int, A: int, B: int, a: int, b: int) -> float:
    if d == 0:
        return 1.0
    len_I = B - A
    if len_I <= 0:
        return 0.0
    left_cap = a - A
    right_cap = B - b
    neigh = len_gt + min(d, left_cap) + min(d, right_cap)
    val = 1.0 - (neigh / float(len_I))
    return 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)


def survival_recall(d: int, A: int, B: int, y: int) -> float:
    if d == 0:
        return 1.0
    len_I = B - A
    if len_I <= 0:
        return 0.0
    neigh = min(d, y - A) + min(d, B - y)
    val = 1.0 - (neigh / float(len_I))
    return 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)


def parse_gt_interval(row: Dict[str, Any]) -> Optional[Interval]:
    gts = row.get("ground_truth", [])
    if not gts:
        return None
    s, e = int(gts[0][0]), int(gts[0][1])
    return Interval(s, e + 1)


def parse_pred_interval(row: Dict[str, Any]) -> Optional[Interval]:
    try:
        if "generated_output" in row:
            raw = row["generated_output"]
        elif "output" in row:
            raw = row["output"]
        else:
            raw = "{}"

        if isinstance(raw, dict):
            go = raw
        else:
            if not isinstance(raw, str):
                raw = str(raw)
            s = raw.strip()
            try:
                go = _try_json_loads_maybe_double_encoded(s)
            except Exception:
                extracted = _extract_json_object_text(s)
                if not extracted:
                    return None
                try:
                    go = _try_json_loads_maybe_double_encoded(extracted)
                except Exception:
                    ex = extracted.strip()
                    if (len(ex) >= 2) and ((ex[0] == ex[-1]) and ex[0] in ("'", '"')):
                        ex = ex[1:-1].strip()
                        go = _try_json_loads_maybe_double_encoded(ex)
                    else:
                        return None

        if not isinstance(go, dict):
            return None

        anomalies = go.get("anomalies", [])
        if not anomalies:
            return None

        first = anomalies[0]
        if isinstance(first, dict) and "start" in first and "end" in first:
            s_idx, e_idx = int(first["start"]), int(first["end"])
        elif isinstance(first, (list, tuple)) and len(first) >= 2:
            s_idx, e_idx = int(first[0]), int(first[1])
        else:
            return None

        if e_idx < s_idx:
            s_idx, e_idx = e_idx, s_idx

        return Interval(s_idx, e_idx + 1)
    except Exception:
        return None


def compute_affiliation_single(
    T0: int, T1: int, gt: Optional[Interval], pred: Optional[Interval]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if gt is None:
        return (None, None, None, None)

    A, B = T0, T1
    gt_c = clip_to_domain(gt, A, B)
    if gt_c.length() == 0:
        return (None, None, None, None)

    if pred is not None:
        pred_c = clip_to_domain(pred, A, B)
        has_pred = pred_c.length() > 0
    else:
        pred_c = None
        has_pred = False

    precision = None
    avg_d_pred_to_gt = None
    if has_pred:
        len_gt = gt_c.length()
        a, b = gt_c.start, gt_c.end
        surv_vals = []
        d_vals = []
        for t in range(pred_c.start, pred_c.end):
            d = gt_c.distance_to_point(t)
            d_vals.append(d)
            s = survival_precision(d=d, len_gt=len_gt, A=A, B=B, a=a, b=b)
            surv_vals.append(s)
        precision = (sum(surv_vals) / len(surv_vals)) if surv_vals else None
        avg_d_pred_to_gt = (sum(d_vals) / len(d_vals)) if d_vals else None

    recall_vals = []
    d_gt_to_pred_vals = []
    for y in range(gt_c.start, gt_c.end):
        if has_pred:
            d = pred_c.distance_to_point(y)
            d_gt_to_pred_vals.append(d)
        else:
            d = 10**9
        s = survival_recall(d=d, A=A, B=B, y=y)
        recall_vals.append(s)
    recall = (sum(recall_vals) / len(recall_vals)) if recall_vals else None
    avg_d_gt_to_pred = (sum(d_gt_to_pred_vals) /
                        len(d_gt_to_pred_vals)) if d_gt_to_pred_vals else None

    return (precision, recall, avg_d_pred_to_gt, avg_d_gt_to_pred)


def f1_from_pr(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None:
        return None
    s = p + r
    if s <= 0:
        return None
    return 2.0 * p * r / s


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

# ========== END BACKEND LOGIC ==========


def process_single_jsonl_synth(jsonl_path: Path, T0: int, T1: int) -> Dict[str, Any]:
    """Process a single JSONL file in synth mode - group by pattern_type and return F1 scores."""
    rows = load_jsonl(jsonl_path)

    # Group metrics by pattern_type
    by_ptype_f1s: Dict[str, List[float]] = defaultdict(list)
    all_f1s: List[float] = []  # For overall file-level F1

    for row in rows:
        pt = str(row.get("pattern_type", "unknown"))
        gt = parse_gt_interval(row)
        pred = parse_pred_interval(row)

        prec, rec, dP2G, dG2P = compute_affiliation_single(T0, T1, gt, pred)
        f1 = f1_from_pr(prec, rec)

        if f1 is not None:
            by_ptype_f1s[pt].append(f1)
            all_f1s.append(f1)

    # Calculate average F1 for each pattern type and overall
    result = {
        "file": str(jsonl_path.name),
        "overall_f1": sum(all_f1s) / len(all_f1s) if all_f1s else None,
        "by_pattern_type": {}
    }

    all_pattern_types: Set[str] = set(by_ptype_f1s.keys())

    for pt in sorted(all_pattern_types):
        f1_list = by_ptype_f1s.get(pt, [])
        f1_avg = sum(f1_list) / len(f1_list) if f1_list else None
        result["by_pattern_type"][pt] = {
            "f1": f1_avg
        }

    return result


def process_single_jsonl_normal(jsonl_path: Path, T0: int, T1: int) -> Dict[str, Any]:
    """Process a single JSONL file in normal mode - return overall metrics."""
    rows = load_jsonl(jsonl_path)

    overall_precisions: List[float] = []
    overall_recalls: List[float] = []
    overall_f1s: List[float] = []

    for row in rows:
        pt = str(row.get("pattern_type", "unknown"))
        gt = parse_gt_interval(row)
        pred = parse_pred_interval(row)

        prec, rec, dP2G, dG2P = compute_affiliation_single(T0, T1, gt, pred)
        f1 = f1_from_pr(prec, rec)

        if rec is not None:
            overall_recalls.append(rec)
        if prec is not None:
            overall_precisions.append(prec)
        if f1 is not None:
            overall_f1s.append(f1)

    return {
        "file": str(jsonl_path.name),
        "precision": sum(overall_precisions)/len(overall_precisions) if overall_precisions else None,
        "recall": sum(overall_recalls)/len(overall_recalls) if overall_recalls else None,
        "f1": sum(overall_f1s)/len(overall_f1s) if overall_f1s else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Multi-JSONL Affiliation Metrics Summary")
    ap.add_argument("jsonl_files", nargs="+", type=str,
                    help="JSONL files to process")
    ap.add_argument("--T0", type=int, default=0,
                    help="Domain start (inclusive). Default 0")
    ap.add_argument("--T1", type=int, default=300,
                    help="Domain end (exclusive). Default 300")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to write JSON/CSV report")
    ap.add_argument(
        "--format", choices=["table", "csv", "json"], default="table", help="Output format")
    ap.add_argument("--synth", action="store_true",
                    help="Enable synthetic data specific processing (groups by pattern_type, shows F1 only)")
    args = ap.parse_args()

    if not args.jsonl_files:
        print("Error: No JSONL files provided", file=sys.stderr)
        sys.exit(1)

    results = []
    for jsonl_file in args.jsonl_files:
        jsonl_path = Path(jsonl_file)
        if not jsonl_path.exists():
            print(f"Warning: File not found: {jsonl_file}", file=sys.stderr)
            continue

        try:
            if args.synth:
                result = process_single_jsonl_synth(
                    jsonl_path, args.T0, args.T1)
            else:
                result = process_single_jsonl_normal(
                    jsonl_path, args.T0, args.T1)
            results.append(result)
        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}", file=sys.stderr)
            continue

    if not results:
        print("Error: No files were successfully processed", file=sys.stderr)
        sys.exit(1)

    # Print summary table
    print(f"\n{'='*100}")
    print(f"AFFILIATION METRICS SUMMARY ({len(results)} files)")
    print(f"Domain: [{args.T0}, {args.T1})")
    if args.synth:
        print("Synthetic mode: ENABLED (grouped by pattern_type, F1 only)")
    print(f"{'='*100}")

    if args.synth:
        # Synth mode: show F1 scores by pattern_type with Overall column
        all_pattern_types: Set[str] = set()
        for result in results:
            all_pattern_types.update(result["by_pattern_type"].keys())

        pattern_types = sorted(all_pattern_types)

        # Create header with Overall + pattern types
        header = f"{'File':<30} {'Overall':<10}"
        for pt in pattern_types:
            header += f" {pt:<10}"
        print(header)
        print("-" * (40 + len(pattern_types) * 11))

        # Print rows with Overall + pattern type F1 scores
        for result in results:
            overall_f1 = result.get("overall_f1")
            row = f"{result['file']:<30}"
            if overall_f1 is not None:
                row += f" {overall_f1:<10.4f}"
            else:
                row += f" {'NA':<10}"

            for pt in pattern_types:
                pt_data = result["by_pattern_type"].get(pt, {})
                f1_val = pt_data.get("f1")
                if f1_val is not None:
                    row += f" {f1_val:<10.4f}"
                else:
                    row += f" {'NA':<10}"
            print(row)

    else:
        # Normal mode: show overall metrics
        print(
            f"{'File':<30} {'Series':<8} {'Eval':<6} {'Precision':<10} {'Recall':<8} {'F1':<8}")
        print("-" * 70)

        for result in results:
            file_name = result["file"]
            precision = f"{result['precision']:.4f}" if result['precision'] is not None else "NA"
            recall = f"{result['recall']:.4f}" if result['recall'] is not None else "NA"
            f1 = f"{result['f1']:.4f}" if result['f1'] is not None else "NA"

            print(f"{file_name:<30} {result['total_series']:<8} {result.get('evaluable_series', 0):<6} "
                  f"{precision:<10} {recall:<8} {f1:<8}")

    # Save results if requested
    if args.out:
        if args.format == "json" or args.out.endswith('.json'):
            with open(args.out, 'w') as f:
                json.dump({
                    "domain": [args.T0, args.T1],
                    "synth_mode": args.synth,
                    "files_processed": len(results),
                    "results": results
                }, f, indent=2)
            print(f"\nSaved JSON report to {args.out}")
        else:
            if args.synth:
                # For synth mode, create a flattened CSV with pattern types as columns
                csv_data = []
                for result in results:
                    row = {
                        "file": result["file"],
                        "overall_f1": result.get("overall_f1")
                    }
                    for pt, pt_data in result["by_pattern_type"].items():
                        row[f"f1_{pt}"] = pt_data["f1"]
                    csv_data.append(row)
                df = pd.DataFrame(csv_data)
            else:
                df = pd.DataFrame([
                    {
                        "file": r["file"],
                        "total_series": r["total_series"],
                        "evaluable_series": r.get("evaluable_series", 0),
                        "precision": r["precision"],
                        "recall": r["recall"],
                        "f1": r["f1"]
                    }
                    for r in results
                ])
            df.to_csv(args.out, index=False)
            print(f"\nSaved CSV report to {args.out}")


if __name__ == "__main__":
    main()
