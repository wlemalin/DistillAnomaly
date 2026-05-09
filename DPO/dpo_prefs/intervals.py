import json
import re
from typing import Any, List, Optional, Tuple

import torch


def _normalize_intervals(x) -> List[Tuple[int, int]]:
    if x is None:
        return []
    if isinstance(x, (tuple, list)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
        s, e = int(x[0]), int(x[1])
        return [(min(s, e), max(s, e))]
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        out = []
        for it in x:
            if isinstance(it, list) and len(it) == 2 and all(isinstance(v, (int, float)) for v in it):
                s, e = int(it[0]), int(it[1])
                out.append((min(s, e), max(s, e)))
        return out
    return []


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda t: (t[0], t[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _interval_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    a_s, a_e = a
    b_s, b_e = b
    if a_e < b_s:
        return float(b_s - (a_e + 1))
    if b_e < a_s:
        return float(a_s - (b_e + 1))
    return 0.0


def _mean_min_distance(
    src: List[Tuple[int, int]],
    dst: List[Tuple[int, int]],
    normalize_by: Optional[float] = None,
) -> float:
    if not src and not dst:
        return 0.0
    if not dst:
        if not src:
            return 0.0
        return float(normalize_by if normalize_by is not None else 1e6)

    dists = []
    for a in src:
        md = min(_interval_distance(a, b) for b in dst)
        dists.append(md)

    m = float(sum(dists) / max(1, len(dists)))
    if normalize_by is not None and normalize_by > 0:
        m = m / normalize_by
    return m


def affiliation_like_score_1d(
    gt: List[Tuple[int, int]],
    pr: List[Tuple[int, int]],
    *,
    end_inclusive: bool = True,
    normalize: str = "span",
    eps: float = 1e-12,
) -> float:
    gt_m = _merge_intervals(gt)
    pr_m = _merge_intervals(pr)

    if not gt_m and not pr_m:
        return 1.0
    if not gt_m or not pr_m:
        return 0.0

    if normalize == "span":
        all_iv = gt_m + pr_m
        lo = min(s for s, _ in all_iv)
        hi = max(e for _, e in all_iv)
        span = float((hi - lo + 1) if end_inclusive else (hi - lo))
        span = max(span, 1.0)
    else:
        span = None

    d_g2p = _mean_min_distance(gt_m, pr_m, normalize_by=span)
    d_p2g = _mean_min_distance(pr_m, gt_m, normalize_by=span)

    s_g2p = 1.0 / (1.0 + d_g2p)
    s_p2g = 1.0 / (1.0 + d_p2g)

    denom = (s_g2p + s_p2g)
    if denom < eps:
        return 0.0
    return float(2.0 * s_g2p * s_p2g / denom)


_INTERVAL_JSON_BLOCK = re.compile(
    r"\[\s*\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]\s*(?:,\s*\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]\s*)*\]",
    re.DOTALL,
)
_INTERVAL_PAIR = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
_START_END = re.compile(r"start\s*[:=]\s*(-?\d+).*?end\s*[:=]\s*(-?\d+)", re.IGNORECASE | re.DOTALL)


def extract_predicted_intervals(text: str) -> List[Tuple[int, int]]:
    if not text:
        return []

    m = _INTERVAL_JSON_BLOCK.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return _merge_intervals(_normalize_intervals(obj))
        except Exception:
            pass

    m2 = _INTERVAL_PAIR.search(text)
    if m2:
        s = int(float(m2.group(1)))
        e = int(float(m2.group(2)))
        return [(min(s, e), max(s, e))]

    m3 = _START_END.search(text)
    if m3:
        s = int(m3.group(1))
        e = int(m3.group(2))
        return [(min(s, e), max(s, e))]

    nums = re.findall(r"-?\d+", text)
    if len(nums) >= 2:
        s = int(nums[-2])
        e = int(nums[-1])
        return [(min(s, e), max(s, e))]

    return []


@torch.no_grad()
def score_candidates_affiliation_like(
    ground_truth_obj: Any,
    candidates: List[str],
    *,
    end_inclusive: bool = True,
    normalize: str = "span",
) -> List[float]:
    gt = _merge_intervals(_normalize_intervals(ground_truth_obj))
    scores = []
    for txt in candidates:
        pr = _merge_intervals(extract_predicted_intervals(txt))
        scores.append(
            affiliation_like_score_1d(
                gt,
                pr,
                end_inclusive=end_inclusive,
                normalize=normalize,
            )
        )
    return scores
