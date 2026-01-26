#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import defaultdict


import json
import re
from typing import Any, Dict, Optional

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)

def _try_json_loads_maybe_double_encoded(s: str) -> Any:
    """
    Tente json.loads(s). Si ça donne une string (JSON double-encodé),
    retente json.loads sur ce résultat.
    """
    obj = json.loads(s)
    if isinstance(obj, str):
        # parfois on a une string qui contient encore du JSON
        obj2 = json.loads(obj)
        return obj2
    return obj

def _extract_json_object_text(raw: str) -> Optional[str]:
    """
    Extrait un objet JSON {...} depuis:
      - un bloc ```json ... ```
      - ou un texte contenant {...} avec du bruit avant/après
    Retourne le substring JSON si trouvé, sinon None.
    """
    s = raw.strip()

    # 1) Cas ```json ... ```
    m = _JSON_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()

    # 2) Cas bruit autour: chercher le premier '{' et le dernier '}'
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
    e = min(iv.end,   T1)
    if e < s:
        e = s
    return Interval(s, e)

# ---------------------------
# Affiliation survival functions (discrete form)
# ---------------------------

def survival_precision(d: int, len_gt: int, A: int, B: int, a: int, b: int) -> float:
    """
    S_precision(d) = 1 - ( |gt| + min(d, a-A) + min(d, B-b) ) / |I|
    with I = [A,B). Special-case d==0 -> 1.0 (perfect alignment).
    """
    if d == 0:
        return 1.0
    len_I = B - A
    if len_I <= 0:
        return 0.0
    left_cap  = a - A
    right_cap = B - b
    neigh = len_gt + min(d, left_cap) + min(d, right_cap)
    val = 1.0 - (neigh / float(len_I))
    return 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)

def survival_recall(d: int, A: int, B: int, y: int) -> float:
    """
    S_recall(d) = 1 - ( min(d, y-A) + min(d, B-y) ) / |I|
    Special-case d==0 -> 1.0 (perfect alignment).
    """
    if d == 0:
        return 1.0
    len_I = B - A
    if len_I <= 0:
        return 0.0
    neigh = min(d, y - A) + min(d, B - y)
    val = 1.0 - (neigh / float(len_I))
    return 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)

# ---------------------------
# Parsing helpers (single GT + single PRED)
# ---------------------------

def parse_gt_interval(row: Dict[str, Any]) -> Optional[Interval]:
    """
    ground_truth: [[s, e]] (inclusive). If empty -> None.
    """
    gts = row.get("ground_truth", [])
    if not gts:
        return None
    s, e = int(gts[0][0]), int(gts[0][1])
    return Interval(s, e + 1)  # inclusive -> half-open

def parse_pred_interval(row: Dict[str, Any]) -> Optional[Interval]:
    """
    generated_output: JSON string with "anomalies": [] or [{"start": s, "end": e, ...}]
    (inclusive bounds). If empty/malformed -> None.

    Fallback: accepte aussi des sorties enveloppées en Markdown fences:
      ```json
      {...}
      ```
    et/ou du texte autour, et/ou du JSON double-encodé.
    """
    try:
        if "generated_output" in row:
            raw = row["generated_output"]
        elif "output" in row:
            raw = row["output"]
        else:
            raw = "{}"

        # Déjà dict -> OK
        if isinstance(raw, dict):
            go = raw
        else:
            # Si ce n'est pas une string, on tente de le stringifier proprement
            if not isinstance(raw, str):
                raw = str(raw)

            s = raw.strip()

            # 1) Tentative normale (ton comportement actuel)
            try:
                go = _try_json_loads_maybe_double_encoded(s)
            except Exception:
                # 2) Fallback: extraire un objet JSON depuis ```json ... ``` ou depuis {...}
                extracted = _extract_json_object_text(s)
                if not extracted:
                    return None
                try:
                    go = _try_json_loads_maybe_double_encoded(extracted)
                except Exception:
                    # 3) Ultra-fallback: parfois tout est entouré de guillemets
                    # (ex: "\"{...}\""). On retire une couche de quotes si plausible.
                    ex = extracted.strip()
                    if (len(ex) >= 2) and ((ex[0] == ex[-1]) and ex[0] in ("'", '"')):
                        ex = ex[1:-1].strip()
                        go = _try_json_loads_maybe_double_encoded(ex)
                    else:
                        return None

        # À ce stade, go doit être un dict
        if not isinstance(go, dict):
            return None

        anomalies = go.get("anomalies", [])
        if not anomalies:
            return None

        first = anomalies[0]

        # Cas standard: dict {"start":..., "end":...}
        if isinstance(first, dict) and "start" in first and "end" in first:
            s_idx, e_idx = int(first["start"]), int(first["end"])

        # Mini-fallback: liste/tuple [start, end]
        elif isinstance(first, (list, tuple)) and len(first) >= 2:
            s_idx, e_idx = int(first[0]), int(first[1])

        else:
            return None

        # Sécuriser si inversé
        if e_idx < s_idx:
            s_idx, e_idx = e_idx, s_idx

        return Interval(s_idx, e_idx + 1)  # inclusive -> half-open

    except Exception:
        return None


# ---------------------------
# Core: single GT, single PRED, single zone = whole domain
# ---------------------------

def compute_affiliation_single(
    T0: int, T1: int, gt: Optional[Interval], pred: Optional[Interval]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returns (precision, recall, avg_d_pred_to_gt, avg_d_gt_to_pred).
    Precision is None if no pred. If no GT, returns all None (nothing to evaluate).
    """
    if gt is None:
        return (None, None, None, None)

    # zone I is the whole domain
    A, B = T0, T1

    # clip intervals to domain to be safe
    gt_c = clip_to_domain(gt, A, B)
    if gt_c.length() == 0:
        # a degenerate GT would make the metric ill-defined; treat as no event
        return (None, None, None, None)

    if pred is not None:
        pred_c = clip_to_domain(pred, A, B)
        has_pred = pred_c.length() > 0
    else:
        pred_c = None
        has_pred = False

    # --- Precision (average over t in PRED of S_precision(dist(t, GT)))
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

    # --- Recall (average over y in GT of S_recall(dist(y, PRED)))
    # if no pred, distances will be huge -> survival ~0; for raw distance, report None
    recall_vals = []
    d_gt_to_pred_vals = []
    for y in range(gt_c.start, gt_c.end):
        if has_pred:
            d = pred_c.distance_to_point(y)
            d_gt_to_pred_vals.append(d)
        else:
            d = 10**9  # no prediction anywhere (for metric); keep raw distance as None
        s = survival_recall(d=d, A=A, B=B, y=y)
        recall_vals.append(s)
    recall = (sum(recall_vals) / len(recall_vals)) if recall_vals else None
    avg_d_gt_to_pred = (sum(d_gt_to_pred_vals) / len(d_gt_to_pred_vals)) if d_gt_to_pred_vals else None

    return (precision, recall, avg_d_pred_to_gt, avg_d_gt_to_pred)

# ---------------------------
# Utils: F1
# ---------------------------

def f1_from_pr(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None:
        return None
    s = p + r
    if s <= 0:
        return None
    return 2.0 * p * r / s

# ---------------------------
# I/O + aggregation
# ---------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser(description="Affiliation Precision/Recall/F1 (single GT + single Pred per series) + raw distances")
    ap.add_argument("jsonl", type=str, help="Path to dataset JSONL")
    ap.add_argument("--T0", type=int, default=29, help="Domain start (inclusive). Default 29")
    ap.add_argument("--T1", type=int, default=300, help="Domain end (exclusive). Default 300")
    ap.add_argument("--out", type=str, default=None, help="Optional path to write a JSON report")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.jsonl))
    T0, T1 = args.T0, args.T1

    per_series = []
    # event-level collections
    overall_precisions: List[float] = []
    overall_recalls: List[float] = []
    overall_f1s: List[float] = []
    overall_d_pred_to_gt: List[float] = []
    overall_d_gt_to_pred: List[float] = []

    by_ptype_precisions: Dict[str, List[float]] = defaultdict(list)
    by_ptype_recalls: Dict[str, List[float]] = defaultdict(list)
    by_ptype_f1s: Dict[str, List[float]] = defaultdict(list)
    by_ptype_d_pred_to_gt: Dict[str, List[float]] = defaultdict(list)
    by_ptype_d_gt_to_pred: Dict[str, List[float]] = defaultdict(list)

    for idx, row in enumerate(rows, start=1):
        pt = str(row.get("pattern_type", "unknown"))
        gt  = parse_gt_interval(row)
        pred = parse_pred_interval(row)

        prec, rec, dP2G, dG2P = compute_affiliation_single(T0, T1, gt, pred)
        f1 = f1_from_pr(prec, rec)

        per_series.append({
            "index": idx,
            "pattern_type": pt,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "avg_d_pred_to_gt": dP2G,
            "avg_d_gt_to_pred": dG2P,
            "gt": None if gt is None else [gt.start, gt.end],       # half-open in report
            "pred": None if pred is None else [pred.start, pred.end]
        })

        # aggregate per pattern_type first
        if rec is not None:
            by_ptype_recalls[pt].append(rec)
            overall_recalls.append(rec)
        if prec is not None:
            by_ptype_precisions[pt].append(prec)
            overall_precisions.append(prec)
        if f1 is not None:
            by_ptype_f1s[pt].append(f1)
            overall_f1s.append(f1)
        if dP2G is not None:
            by_ptype_d_pred_to_gt[pt].append(dP2G)
            overall_d_pred_to_gt.append(dP2G)
        if dG2P is not None:
            by_ptype_d_gt_to_pred[pt].append(dG2P)
            overall_d_gt_to_pred.append(dG2P)

    # ---- Print per-series
    print("\nPer-series (single GT & single Pred)")
    print("Idx | pattern_type         | Precision | Recall |   F1   | d(P→GT) | d(GT→P)")
    print("----------------------------------------------------------------------------")
    for r in per_series:
        p  = "NA" if r["precision"] is None else f"{r['precision']:.4f}"
        q  = "NA" if r["recall"]    is None else f"{r['recall']:.4f}"
        f  = "NA" if r["f1"]        is None else f"{r['f1']:.4f}"
        dp = "NA" if r["avg_d_pred_to_gt"] is None else f"{r['avg_d_pred_to_gt']:.2f}"
        dg = "NA" if r["avg_d_gt_to_pred"] is None else f"{r['avg_d_gt_to_pred']:.2f}"
        print(f"{r['index']:>3} | {r['pattern_type']:<21} | {p:>9} | {q:>6} | {f:>6} | {dp:>7} | {dg:>7}")

    # ---- Per-pattern_type (event-level) BEFORE overall
    print("\nPer-pattern_type (event-level)")
    print("pattern_type                | Precision | Recall |   F1   | d(P→GT) | d(GT→P) ")
    print("-------------------------------------------------------------------------------")
    ptype_summary = {}
    for pt in sorted(set(list(by_ptype_recalls.keys()) + list(by_ptype_precisions.keys())
                         + list(by_ptype_f1s.keys())
                         + list(by_ptype_d_pred_to_gt.keys()) + list(by_ptype_d_gt_to_pred.keys()))):
        p_list  = by_ptype_precisions.get(pt, [])
        r_list  = by_ptype_recalls.get(pt, [])
        f_list  = by_ptype_f1s.get(pt, [])
        dp_list = by_ptype_d_pred_to_gt.get(pt, [])
        dg_list = by_ptype_d_gt_to_pred.get(pt, [])
        p  = None if not p_list  else sum(p_list)  / len(p_list)
        r  = None if not r_list  else sum(r_list)  / len(r_list)
        f  = None if not f_list  else sum(f_list)  / len(f_list)
        dp = None if not dp_list else sum(dp_list) / len(dp_list)
        dg = None if not dg_list else sum(dg_list) / len(dg_list)
        ptype_summary[pt] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "avg_d_pred_to_gt": dp,
            "avg_d_gt_to_pred": dg,
        }
        ps  = "NA" if p  is None else f"{p:.4f}"
        rs  = "NA" if r  is None else f"{r:.4f}"
        fs  = "NA" if f  is None else f"{f:.4f}"
        dps = "NA" if dp is None else f"{dp:.2f}"
        dgs = "NA" if dg is None else f"{dg:.2f}"
        print(f"{pt:<27} | {ps:>9} | {rs:>6} | {fs:>6} | {dps:>7} | {dgs:>7} |")

    # ---- Overall (event-level)
    print("\nOverall (event-level)")
    if overall_precisions:
        print(f"Precision: {sum(overall_precisions)/len(overall_precisions):.4f}")
    else:
        print("Precision: NA")
    if overall_recalls:
        print(f"Recall:    {sum(overall_recalls)/len(overall_recalls):.4f}")
    else:
        print("Recall:    NA")
    if overall_f1s:
        print(f"F1:        {sum(overall_f1s)/len(overall_f1s):.4f}")
    else:
        print("F1:        NA")
    if overall_d_pred_to_gt:
        print(f"d(P→GT):   {sum(overall_d_pred_to_gt)/len(overall_d_pred_to_gt):.2f}")
    else:
        print("d(P→GT):   NA")
    if overall_d_gt_to_pred:
        print(f"d(GT→P):   {sum(overall_d_gt_to_pred)/len(overall_d_gt_to_pred):.2f}")
    else:
        print("d(GT→P):   NA")

    # ---- Optional JSON report
    if args.out:
        report = {
            "domain": [T0, T1],
            "overall": {
                "precision": None if not overall_precisions else sum(overall_precisions)/len(overall_precisions),
                "recall": None if not overall_recalls else sum(overall_recalls)/len(overall_recalls),
                "f1": None if not overall_f1s else sum(overall_f1s)/len(overall_f1s),
                "avg_d_pred_to_gt": None if not overall_d_pred_to_gt else sum(overall_d_pred_to_gt)/len(overall_d_pred_to_gt),
                "avg_d_gt_to_pred": None if not overall_d_gt_to_pred else sum(overall_d_gt_to_pred)/len(overall_d_gt_to_pred),
                "n_event_precisions": len(overall_precisions),
                "n_event_recalls": len(overall_recalls),
                "n_event_f1": len(overall_f1s),
                "n_event_d_pred_to_gt": len(overall_d_pred_to_gt),
                "n_event_d_gt_to_pred": len(overall_d_gt_to_pred),
            },
            "by_pattern_type": ptype_summary,
            "per_series": per_series,
            "notes": {
                "intervals_are_half_open": True,
                "gt_pred_from_json_are_inclusive_and_converted_to_half_open": True,
                "survival_d_eq_0_is_1_pointwise": True,
                "f1_is_harmonic_mean_of_affiliation_precision_and_recall": True
            }
        }
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {args.out}")
        
if __name__ == "__main__":
    main()

