#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import defaultdict

import re

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _try_json_loads_maybe_double_encoded(s: str) -> Any:
    """Try json.loads(s). If it returns a string (double-encoded), retry."""
    obj = json.loads(s)
    if isinstance(obj, str):
        obj2 = json.loads(obj)
        return obj2
    return obj


def _extract_json_object_text(raw: str) -> Optional[str]:
    """Extract JSON {...} from markdown fences or raw text."""
    s = raw.strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i : j + 1].strip()
    return None


# ---------------------------
# Interval utilities (half-open)
# ---------------------------

@dataclass(frozen=True)
class Interval:
    start: int  # inclusive
    end: int    # exclusive

    def length(self) -> int:
        return max(0, self.end - self.start)

    def distance_to_point(self, t: int) -> int:
        """Distance from integer t to this interval on the real line."""
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


# ---------------------------
# Affiliation survival functions
# ---------------------------

def survival_precision(d: int, len_gt: int, A: int, B: int, a: int, b: int) -> float:
    """S_precision(d) = 1 - ( |gt| + min(d, a-A) + min(d, B-b) ) / |I|"""
    if d == 0:
        return 1.0
    len_I = B - A
    if len_I <= 0:
        return 0.0
    left_cap = a - A
    right_cap = B - b
    neigh = len_gt + min(d, left_cap) + min(d, right_cap)
    val = 1.0 - (neigh / float(len_I))
    return max(0.0, min(1.0, val))


def survival_recall(d: int, A: int, B: int, y: int) -> float:
    """S_recall(d) = 1 - ( min(d, y-A) + min(d, B-y) ) / |I|"""
    if d == 0:
        return 1.0
    len_I = B - A
    if len_I <= 0:
        return 0.0
    neigh = min(d, y - A) + min(d, B - y)
    val = 1.0 - (neigh / float(len_I))
    return max(0.0, min(1.0, val))


# ---------------------------
# Parsing helpers
# ---------------------------

def parse_gt_interval(row: Dict[str, Any]) -> Optional[Interval]:
    """Parse ground_truth: [[s, e]] (inclusive). Returns half-open Interval or None."""
    gts = row.get("ground_truth", [])
    if not gts:
        return None
    s, e = int(gts[0][0]), int(gts[0][1])
    return Interval(s, e + 1)


def parse_pred_interval(row: Dict[str, Any]) -> Optional[Interval]:
    """
    Parse generated_output: JSON with "anomalies": [{"start": s, "end": e}].
    Handles markdown fences, double-encoding, and inclusive->half-open conversion.
    Returns None if missing, malformed, or empty.
    """
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


# ---------------------------
# Core affiliation computation
# ---------------------------

def compute_affiliation_single(
    T0: int,
    T1: int,
    gt: Optional[Interval],
    pred: Optional[Interval],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], bool]:
    """
    Returns (precision, recall, avg_d_pred_to_gt, avg_d_gt_to_pred, has_prediction).

    - Precision is None if no prediction.
    - Recall is None if no prediction.
    - has_prediction indicates if pred was valid after clipping.
    """
    has_pred = pred is not None

    if gt is None:
        return (None, None, None, None, has_pred)

    A, B = T0, T1
    gt_c = clip_to_domain(gt, A, B)

    if gt_c.length() == 0:
        return (None, None, None, None, has_pred)

    if has_pred:
        pred_c = clip_to_domain(pred, A, B)
        has_pred = pred_c.length() > 0

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
            s = survival_precision(
                d=d,
                len_gt=len_gt,
                A=A,
                B=B,
                a=a,
                b=b,
            )
            surv_vals.append(s)

        precision = sum(surv_vals) / len(surv_vals) if surv_vals else None
        avg_d_pred_to_gt = sum(d_vals) / len(d_vals) if d_vals else None

    recall = None
    avg_d_gt_to_pred = None

    if has_pred:
        recall_vals = []
        d_gt_to_pred_vals = []

        for y in range(gt_c.start, gt_c.end):
            d = pred_c.distance_to_point(y)
            d_gt_to_pred_vals.append(d)
            s = survival_recall(d=d, A=A, B=B, y=y)
            recall_vals.append(s)

        recall = sum(recall_vals) / len(recall_vals) if recall_vals else None
        avg_d_gt_to_pred = sum(d_gt_to_pred_vals) / len(d_gt_to_pred_vals) if d_gt_to_pred_vals else None

    return (precision, recall, avg_d_pred_to_gt, avg_d_gt_to_pred, has_pred)


def f1_from_pr(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None:
        return None
    s = p + r
    if s <= 0:
        return None
    return 2.0 * p * r / s


# ---------------------------
# I/O
# ---------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------
# Aggregation helpers
# ---------------------------

def avg(lst: List[float]) -> Optional[float]:
    return sum(lst) / len(lst) if lst else None


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Affiliation Precision/Recall/F1 with invalid detection rate"
    )
    ap.add_argument("jsonl", type=str, help="Path to dataset JSONL")
    ap.add_argument("--T0", type=int, default=0, help="Domain start (inclusive). Default 0")
    ap.add_argument("--T1", type=int, default=300, help="Domain end (exclusive). Default 300")
    ap.add_argument("--out", type=str, default=None, help="Optional path to write a JSON report")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.jsonl))
    T0, T1 = args.T0, args.T1

    per_series = []

    overall = {
        "p": [],
        "r": [],
        "f1": [],
        "dp2g": [],
        "dg2p": [],
    }

    by_ptype: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {
            "p": [],
            "r": [],
            "f1": [],
            "dp2g": [],
            "dg2p": [],
        }
    )

    total_with_gt = 0
    valid_pred_count = 0

    for idx, row in enumerate(rows, start=1):
        pt = str(row.get("pattern_type", "unknown"))
        gt = parse_gt_interval(row)
        pred = parse_pred_interval(row)

        prec, rec, dP2G, dG2P, has_pred = compute_affiliation_single(T0, T1, gt, pred)
        f1 = f1_from_pr(prec, rec)

        per_series.append({
            "index": idx,
            "pattern_type": pt,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "avg_d_pred_to_gt": dP2G,
            "avg_d_gt_to_pred": dG2P,
            "gt": None if gt is None else [gt.start, gt.end],
            "pred": None if pred is None else [pred.start, pred.end],
            "has_prediction": has_pred,
        })

        if gt is None:
            continue

        total_with_gt += 1

        if has_pred:
            valid_pred_count += 1

        p_value = prec if prec is not None else 0.0
        r_value = rec if rec is not None else 0.0
        f1_value = f1 if f1 is not None else 0.0

        overall["p"].append(p_value)
        overall["r"].append(r_value)
        overall["f1"].append(f1_value)

        by_ptype[pt]["p"].append(p_value)
        by_ptype[pt]["r"].append(r_value)
        by_ptype[pt]["f1"].append(f1_value)

        if dP2G is not None:
            overall["dp2g"].append(dP2G)
            by_ptype[pt]["dp2g"].append(dP2G)

        if dG2P is not None:
            overall["dg2p"].append(dG2P)
            by_ptype[pt]["dg2p"].append(dG2P)

    invalid_count = total_with_gt - valid_pred_count
    invalid_rate = (invalid_count / total_with_gt * 100) if total_with_gt > 0 else 0.0

    print("\nPer-series (single GT & single Pred)")
    print("Idx | pattern_type         | Precision | Recall |   F1   | d(P→GT) | d(GT→P) | HasPred")
    print("--------------------------------------------------------------------------------------")

    for r in per_series:
        p = "NA" if r["precision"] is None else f"{r['precision']:.4f}"
        q = "NA" if r["recall"] is None else f"{r['recall']:.4f}"
        f = "NA" if r["f1"] is None else f"{r['f1']:.4f}"
        dp = "NA" if r["avg_d_pred_to_gt"] is None else f"{r['avg_d_pred_to_gt']:.2f}"
        dg = "NA" if r["avg_d_gt_to_pred"] is None else f"{r['avg_d_gt_to_pred']:.2f}"
        hp = "Y" if r["has_prediction"] else "N"

        print(
            f"{r['index']:>3} | {r['pattern_type']:<21} | "
            f"{p:>9} | {q:>6} | {f:>6} | {dp:>7} | {dg:>7} | {hp:>7}"
        )

    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    print(f"Total series with GT:      {total_with_gt}")
    print(f"Valid predictions:         {valid_pred_count}")
    print(f"Invalid/missing preds:     {invalid_count}")
    print(f"Invalid detection rate:    {invalid_rate:.2f}%")
    print()

    def fmt(m: Optional[float], dec: int = 4) -> str:
        return "NA" if m is None else f"{m:.{dec}f}"

    print("STRICT MODE (penalizes missing/invalid predictions)")
    print("-" * 60)
    print(f"Precision: {fmt(avg(overall['p']))}")
    print(f"Recall:    {fmt(avg(overall['r']))}")
    print(f"F1:        {fmt(avg(overall['f1']))}")
    print(f"d(P→GT):   {fmt(avg(overall['dp2g']), 2)}")
    print(f"d(GT→P):   {fmt(avg(overall['dg2p']), 2)}")

    print()
    print("PER-PATTERN-TYPE SUMMARY")
    print("-" * 70)
    print("pattern_type               | Precision |   Recall |       F1 | n_total")
    print("-" * 70)

    ptype_summary = {}

    for pt in sorted(by_ptype.keys()):
        p = avg(by_ptype[pt]["p"])
        r = avg(by_ptype[pt]["r"])
        f = avg(by_ptype[pt]["f1"])
        n_total = len(by_ptype[pt]["p"])

        ptype_summary[pt] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "avg_d_pred_to_gt": avg(by_ptype[pt]["dp2g"]),
            "avg_d_gt_to_pred": avg(by_ptype[pt]["dg2p"]),
            "n_total": n_total,
        }

        print(
            f"{pt:<25} | {fmt(p):>9} | {fmt(r):>8} | "
            f"{fmt(f):>8} | {n_total:>7}"
        )

    if args.out:
        report = {
            "domain": [T0, T1],
            "summary": {
                "total_series_with_gt": total_with_gt,
                "valid_predictions": valid_pred_count,
                "invalid_predictions": invalid_count,
                "invalid_detection_rate_percent": invalid_rate,
                "precision": avg(overall["p"]),
                "recall": avg(overall["r"]),
                "f1": avg(overall["f1"]),
                "avg_d_pred_to_gt": avg(overall["dp2g"]),
                "avg_d_gt_to_pred": avg(overall["dg2p"]),
            },
            "by_pattern_type": ptype_summary,
            "per_series": per_series,
            "notes": {
                "missing_or_invalid_predictions": "Missing/invalid predictions count as 0.",
                "invalid_rate": "Percentage of series with GT where prediction was missing or invalid.",
                "intervals_are_half_open": True,
            },
        }

        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
