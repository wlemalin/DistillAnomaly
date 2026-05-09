#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
START_RE = re.compile(r'("start"\s*:\s*)(-?\d+)')
END_RE = re.compile(r'("end"\s*:\s*)(-?\d+)')

__all__ = [
    "extract_first_gt_interval",
    "ensure_ordered_outputs",
    "build_gt_fixed_candidate",
    "maybe_prepend_gt_fixed_candidate",
]


# =============================================================================
# Ground-truth helpers
# =============================================================================

def extract_first_gt_interval(ground_truth: Any) -> Optional[Tuple[int, int]]:
    """
    Expected format:
        ground_truth = [[start, end], ...]
    Returns the first interval if present.
    """
    if not isinstance(ground_truth, list) or not ground_truth:
        return None

    first = ground_truth[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        if isinstance(first[0], int) and isinstance(first[1], int):
            return int(first[0]), int(first[1])

    return None


def _extract_first_gt_interval_from_record(
    record: Optional[Mapping[str, Any]],
    key: str = "ground_truth",
) -> Optional[Tuple[int, int]]:
    if record is None:
        return None
    return extract_first_gt_interval(record.get(key))


# =============================================================================
# JSON-in-fence helpers
# =============================================================================

def _split_fenced(text: str) -> Tuple[str, str, str]:
    """
    Returns:
        prefix, inner_json_like, suffix

    prefix and suffix keep the fences around inner content, so that
    prefix + dumped_json + suffix reconstructs the original wrapper.
    """
    m = FENCE_RE.search(text or "")
    if not m:
        return "", text or "", ""

    inner = m.group(1)
    prefix = text[: m.start(1)]
    suffix = text[m.end(1) :]
    return prefix, inner, suffix


def _json_load_maybe(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _get_first_anomaly_obj(parsed: Any) -> Optional[Dict[str, Any]]:
    """
    Accepts several common formats:
      1) {"anomalies": [{"start": ..., "end": ..., ...}, ...]}
      2) {"start": ..., "end": ..., ...}
      3) [{"start": ..., "end": ..., ...}, ...]
    """
    if isinstance(parsed, dict):
        if "anomalies" in parsed and isinstance(parsed["anomalies"], list) and parsed["anomalies"]:
            first = parsed["anomalies"][0]
            if isinstance(first, dict):
                return first

        if "start" in parsed and "end" in parsed:
            return parsed

    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]

    return None


# =============================================================================
# Text patch helpers
# =============================================================================

def _replace_interval_pair_once_outside_parentheses(
    desc: str,
    old_start: int,
    old_end: int,
    new_start: int,
    new_end: int,
) -> str:
    """
    Replace ONLY ONE occurrence of the pair (old_start, old_end), outside
    parentheses, by (new_start, new_end).

    This preserves later mentions often used in citations like "(at index 286)".
    """
    if not isinstance(desc, str) or not desc:
        return desc

    if old_start == new_start and old_end == new_end:
        return desc

    pat_s = re.compile(rf"(?<!\d){re.escape(str(old_start))}(?!\d)")
    pat_e = re.compile(rf"(?<!\d){re.escape(str(old_end))}(?!\d)")

    parts: List[str] = []
    depth = 0
    start_idx = 0
    replaced = False

    def _process_outside_segment(seg: str) -> str:
        nonlocal replaced
        if replaced:
            return seg

        for ms in pat_s.finditer(seg):
            me = pat_e.search(seg, ms.end())
            if me:
                seg2 = (
                    seg[:ms.start()]
                    + str(new_start)
                    + seg[ms.end():me.start()]
                    + str(new_end)
                    + seg[me.end():]
                )
                replaced = True
                return seg2

        return seg

    for i, ch in enumerate(desc):
        if ch == "(":
            if depth == 0:
                outside = desc[start_idx:i]
                parts.append(_process_outside_segment(outside))
                start_idx = i
            depth += 1

        elif ch == ")":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    parts.append(desc[start_idx:i + 1])
                    start_idx = i + 1

    tail = desc[start_idx:]
    if depth == 0:
        parts.append(_process_outside_segment(tail))
    else:
        parts.append(tail)

    return "".join(parts)


def _extract_pred_intervals_cheap(text: str) -> List[List[int]]:
    """
    Cheap fallback used when we need the rejected interval after prepending
    the GT-fixed candidate.
    """
    if not isinstance(text, str):
        return []

    m1 = START_RE.search(text)
    m2 = END_RE.search(text)
    if not (m1 and m2):
        return []

    try:
        return [[int(m1.group(2)), int(m2.group(2))]]
    except Exception:
        return []


# =============================================================================
# Candidate construction
# =============================================================================

def build_gt_fixed_candidate(text: str, gt_start: int, gt_end: int) -> Optional[str]:
    """
    Build a new completion by taking the best current candidate and replacing:
      - anomaly.start
      - anomaly.end
      - first matching (start, end) pair in description, outside parentheses
    """
    prefix, inner, suffix = _split_fenced(text)
    parsed = _json_load_maybe(inner.strip())
    if parsed is None:
        return None

    anomaly = _get_first_anomaly_obj(parsed)
    if anomaly is None:
        return None

    old_start = anomaly.get("start")
    old_end = anomaly.get("end")
    if not isinstance(old_start, int) or not isinstance(old_end, int):
        return None

    anomaly["start"] = int(gt_start)
    anomaly["end"] = int(gt_end)

    desc = anomaly.get("description")
    if isinstance(desc, str) and desc:
        anomaly["description"] = _replace_interval_pair_once_outside_parentheses(
            desc=desc,
            old_start=int(old_start),
            old_end=int(old_end),
            new_start=int(gt_start),
            new_end=int(gt_end),
        )

    dumped = json.dumps(parsed, ensure_ascii=False, indent=2)
    return prefix + dumped + suffix


# =============================================================================
# Preference-record helpers
# =============================================================================

def ensure_ordered_outputs(pref_record: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize a preference record so downstream code can always rely on
    'ordered_outputs'.

    If 'ordered_outputs' is missing but 'chosen'/'rejected' exist,
    create:
        ordered_outputs = [chosen, rejected]
    """
    out = dict(pref_record)

    ys = out.get("ordered_outputs")
    if isinstance(ys, list) and ys and isinstance(ys[0], str):
        return out

    if isinstance(ys, tuple) and ys and isinstance(ys[0], str):
        out["ordered_outputs"] = list(ys)
        return out

    chosen = out.get("chosen")
    rejected = out.get("rejected")
    if isinstance(chosen, str) and isinstance(rejected, str):
        out["ordered_outputs"] = [chosen, rejected]

    return out


def _prepend_list_field(out: Dict[str, Any], field: str, value: Any) -> None:
    vals = out.get(field)
    if isinstance(vals, list):
        out[field] = [value] + list(vals)


def _prepend_like_first_or_value(
    out: Dict[str, Any],
    field: str,
    explicit_value: Any = None,
) -> None:
    vals = out.get(field)
    if not isinstance(vals, list):
        return

    if explicit_value is None:
        if not vals:
            return
        prefix = vals[0]
    else:
        prefix = explicit_value

    out[field] = [prefix] + list(vals)


def maybe_prepend_gt_fixed_candidate(
    pref_record: Mapping[str, Any],
    *,
    ground_truth: Any = None,
    source_record: Optional[Mapping[str, Any]] = None,
    force: bool = False,
    injected_mode: str = "gt_interval_fix_once",
    ordered_scores_prefix_value: Any = None,
    ordered_labels_prefix_value: Any = None,
) -> Tuple[Dict[str, Any], bool]:
    """
    Main entry point.

    Takes an already-built preference record and, if possible, prepends a new
    GT-fixed version of the current best candidate to ordered_outputs.

    Why prepend instead of replacing?
    ---------------------------------
    This preserves the original best candidate as the new rejected item, which
    matches the behavior of your post-hoc script.

    Compatibility notes
    -------------------
    - Keeps 'ordered_outputs', 'chosen', 'rejected' aligned
    - Updates chosen/rejected pred intervals
    - Prefixes ordered_metas / ordered_false_counts / ordered_affil_scores
    - Also prefixes ordered_scores / ordered_labels to avoid length mismatches
      later with LiPO / LambdaLoss training
    """
    out = ensure_ordered_outputs(pref_record)

    interval = extract_first_gt_interval(ground_truth)
    if interval is None:
        interval = _extract_first_gt_interval_from_record(source_record)

    if interval is None:
        return out, False

    ys = out.get("ordered_outputs")
    if not isinstance(ys, list) or not ys or not isinstance(ys[0], str):
        return out, False

    gt_start, gt_end = interval
    old_chosen_pred = out.get("chosen_pred_intervals")
    old_chosen_score = out.get("chosen_score")
    old_chosen_false = out.get("chosen_citation_false")

    new_best = build_gt_fixed_candidate(ys[0], gt_start, gt_end)
    if new_best is None:
        return out, False

    if (not force) and new_best == ys[0]:
        return out, False

    out = dict(out)
    out["ordered_outputs"] = [new_best] + list(ys)

    # Pairwise-compatible fields
    out["chosen"] = out["ordered_outputs"][0]
    if len(out["ordered_outputs"]) >= 2:
        out["rejected"] = out["ordered_outputs"][1]

    gt_list = ground_truth if isinstance(ground_truth, list) and ground_truth else [[gt_start, gt_end]]
    out["chosen_pred_intervals"] = gt_list

    if isinstance(old_chosen_pred, list) and old_chosen_pred:
        out["rejected_pred_intervals"] = old_chosen_pred
    else:
        out["rejected_pred_intervals"] = _extract_pred_intervals_cheap(out.get("rejected", ""))

    # Debug / traceability
    out["gt_interval_fix_applied"] = True

    # Optional list fields used downstream
    if isinstance(out.get("ordered_metas"), list):
        metas = out["ordered_metas"]
        base0 = metas[0] if metas and isinstance(metas[0], dict) else {}
        meta = dict(base0)
        meta["mode"] = injected_mode
        meta["gt_interval"] = [gt_start, gt_end]
        out["ordered_metas"] = [meta] + list(metas)

    _prepend_list_field(out, "ordered_false_counts", 0)
    _prepend_list_field(out, "ordered_affil_scores", 1.0)
    _prepend_like_first_or_value(out, "ordered_scores", ordered_scores_prefix_value)
    _prepend_like_first_or_value(out, "ordered_labels", ordered_labels_prefix_value)

    # Keep common BT logging fields aligned if they exist
    if isinstance(out.get("ordered_scores"), list) and len(out["ordered_scores"]) >= 2:
        out["chosen_score"] = out["ordered_scores"][0]
        out["rejected_score"] = out["ordered_scores"][1]
    elif old_chosen_score is not None:
        out["chosen_score"] = old_chosen_score if ordered_scores_prefix_value is None else ordered_scores_prefix_value
        out["rejected_score"] = old_chosen_score

    if isinstance(out.get("ordered_false_counts"), list) and len(out["ordered_false_counts"]) >= 2:
        out["chosen_citation_false"] = out["ordered_false_counts"][0]
        out["rejected_citation_false"] = out["ordered_false_counts"][1]
    elif old_chosen_false is not None:
        out["chosen_citation_false"] = 0
        out["rejected_citation_false"] = old_chosen_false

    return out, True
