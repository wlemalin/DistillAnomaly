from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _safe_min(x: list[int]) -> Optional[int]:
    return min(x) if x else None


def _safe_max(x: list[int]) -> Optional[int]:
    return max(x) if x else None


def _safe_median(x: list[int]) -> Optional[int]:
    if not x:
        return None
    return int(np.median(np.asarray(x, dtype=np.float32)))


def _safe_mean(x: list[int]) -> Optional[float]:
    if not x:
        return None
    return float(np.mean(np.asarray(x, dtype=np.float32)))


def _safe_std(x: list[float]) -> Optional[float]:
    if not x:
        return None
    return float(np.std(np.asarray(x, dtype=np.float32)))


def _validate_bounds(start: int, end: int, n: int) -> bool:
    return not (start >= end or start < 0 or end > n)


def _dir_txt(direction: Any) -> str:
    d = str(direction).strip().lower()
    if d == "up":
        return "upward"
    if d == "down":
        return "downward"
    return "noticeable"


def _std_clause(pre_std: Optional[float], seg_std: Optional[float]) -> str:
    if pre_std is None or seg_std is None or pre_std <= 1e-9:
        return ""
    ratio = seg_std / pre_std
    if ratio > 1.10:
        return f"standard deviation increases from ~{pre_std:.2f} to ~{seg_std:.2f} (≈×{ratio:.2f})"
    if ratio < 0.90:
        return f"standard deviation decreases from ~{pre_std:.2f} to ~{seg_std:.2f} (≈×{ratio:.2f})"
    return f"standard deviation stays similar (~{pre_std:.2f} → ~{seg_std:.2f})"


def build_event_1_description(
    *,
    pattern_type: str,
    start: int,
    end: int,
    values_0_99: list[int],
    raw_values: Optional[np.ndarray],
    event: Optional[dict[str, Any]],
    add_noise: bool,
) -> Optional[str]:
    """
    Event 1: original oracle description (full narrative).
    """
    if not _validate_bounds(start, end, len(values_0_99)):
        return None

    dur = int(end - start)
    pre0 = max(0, start - max(30, dur))
    pre = values_0_99[pre0:start]
    seg = values_0_99[start:end]

    pre_med = _safe_median(pre)
    seg_med = _safe_median(seg)
    pre_std = _safe_std([float(v) for v in pre])
    seg_std = _safe_std([float(v) for v in seg])

    if event is None:
        return None

    kind = str(event.get("kind", "")).strip()
    detail = event.get("detail", {}) or {}

    if kind == "shift":
        dir_txt = _dir_txt(detail.get("direction", None))

        pieces: list[str] = []

        return (
            "Normally, the series stays within a relatively narrow band. "
            f"From index {start} to {end}, the series exhibits a sustained {dir_txt} level shift."
        )

    if kind == "noise":
        astd = detail.get("anomaly_std", None)

        if raw_values is not None:
            raw_pre = raw_values[pre0:start]
            raw_seg = raw_values[start:end]
            pre_std_raw = float(np.std(raw_pre.astype(
                np.float32))) if raw_pre.size else None
            seg_std_raw = float(np.std(raw_seg.astype(
                np.float32))) if raw_seg.size else None
        else:
            pre_std_raw = None
            seg_std_raw = None

        var_clause = ""

        base = (
            "Normally, the series is a periodic sine wave centered on a stable level."
            if pattern_type == "point"
            else "Normally, the series is a noisy periodic sine wave centered on a stable level."
        )

        return (
            f"{base} From index {start} to {end}, the time series shows extreme noise interference "
            f"which {'briefly ' if dur <= 10 else ''}disrupts the periodic pattern."
        )

    if kind == "freq_mult":
        mult = detail.get("multiplier", None)
        if isinstance(mult, (float, int)) and float(mult) > 0:
            mult = float(mult)
            if mult > 1.0:
                mult_txt = f"frequency is {'briefly ' if dur <= 20 else ''}multiplied by approximately {mult:.1f} during the anomaly interval"
            elif mult < 1.0:
                mult_txt = f"frequency is {'briefly ' if dur <= 20 else ''}divided by approximately {1.0 / mult:.1f} during the anomaly interval"
            else:
                mult_txt = "frequency is unchanged (multiplier ~1.00)"
        else:
            mult_txt = "the frequency changed during the anomaly interval"

        base = (
            "Normally, the series has a stable oscillation frequency."
            if pattern_type == "freq"
            else "Normally, the series is a noisy periodic sine wave centered on a stable level."
        )

        return (
            f"{base} From index {start} to {end}, there is an anomaly characterized by a change in the "
            f"signal's frequency: {mult_txt}."
        )

    if kind == "slope_change":
        normal = detail.get("normal_slope", None)
        abnormal = detail.get("abnormal_slope", None)

        slope_txt = "a baseline slope change was injected"
        try:
            normal_f = float(normal)
            abnormal_f = float(abnormal)
            if abs(normal_f) > 1e-12:
                slope_mult = abnormal_f / normal_f
                slope_txt = f"baseline slope is multipied by approximately {slope_mult:.1f} during the anomaly"
        except Exception:
            pass

        base = (
            "Normally, the series is a sine wave whose baseline drifts slowly."
            if pattern_type == "trend"
            else "Normally, the series is a noisy sine wave whose baseline drifts slowly."
        )

        return (
            f"{base} From index {start} to {end}, there is an anomaly characterized by a trend-shift: {slope_txt}."
        )

    return None


def build_event_2_description(
    *,
    pattern_type: str,
    start: int,
    end: int,
    values_0_99: list[int],
    raw_values: Optional[np.ndarray],
    event: Optional[dict[str, Any]],
    add_noise: bool,
    rolling_avg_0_99: Optional[list[int]] = None,
) -> Optional[str]:
    """
    Event 2: central tendency view using extremes inside anomaly vs median during normal times,
             computed on the precomputed tokenized rolling average (0..99).
    """
    if not _validate_bounds(start, end, len(values_0_99)):
        return None

    ra = rolling_avg_0_99 if (rolling_avg_0_99 and len(
        rolling_avg_0_99) == len(values_0_99)) else None
    if ra is None:
        return "central tendency view unavailable (missing precomputed rolling average)."

    dur = int(end - start)
    pre0 = max(0, start - max(30, dur))

    pre_ra = ra[pre0:start]
    seg_ra = ra[start:end]

    baseline = _safe_median(pre_ra)
    seg_min = _safe_min(seg_ra)
    seg_max = _safe_max(seg_ra)

    if baseline is None or seg_min is None or seg_max is None:
        center = "baseline/extremes could not be estimated"
    else:
        # Decide which extreme to use.
        kind = str(event.get("kind", "")).strip(
        ) if isinstance(event, dict) else ""
        detail = (event.get("detail", {}) or {}
                  ) if isinstance(event, dict) else {}
        dir_hint = str(detail.get("direction", "")).strip(
        ).lower() if kind == "shift" else ""

        if dir_hint == "down":
            extreme = seg_min
            label = "minimum"
        elif dir_hint == "up":
            extreme = seg_max
            label = "maximum"
        else:
            # pick the most deviant extreme
            dmin = abs(seg_min - baseline)
            dmax = abs(seg_max - baseline)
            if dmax >= dmin:
                extreme = seg_max
                label = "maximum"
            else:
                extreme = seg_min
                label = "minimum"

        delta = extreme - baseline
        # Small dead-zone to avoid noisy wording when tokens barely move
        if abs(delta) <= 1:
            # center = f"While the baseline average is around ~{baseline}; anomaly {label} ~{extreme} (nearly unchanged)"
            center = ""
        elif delta > 0:
            center = f" While the usual moving average is around {baseline}, during the anomaly it rises up to a {label} of {extreme} (+{delta})"
        else:
            center = f" While the usual moving average is around {baseline}, during the anomaly it drops down to a {label} of {extreme} ({delta})"

    # Keep your per-kind phrasing (only the 'center' computation changed)
    kind = str(event.get("kind", "")).strip(
    ) if isinstance(event, dict) else ""
    detail = (event.get("detail", {}) or {}) if isinstance(event, dict) else {}

    if kind == "slope_change":
        return f"{center}."

    return f"{center}."


def build_event_3_description(
    *,
    pattern_type: str,
    start: int,
    end: int,
    values_0_99: list[int],
    raw_values: Optional[np.ndarray],
    event: Optional[dict[str, Any]],
    add_noise: bool,
) -> Optional[str]:
    """
    Event 3: standard deviation / dispersion only.
    """
    if not _validate_bounds(start, end, len(values_0_99)):
        return None

    dur = int(end - start)
    pre0 = max(0, start - max(30, dur))
    pre = values_0_99[pre0:start]
    seg = values_0_99[start:end]

    pre_std_tok = _safe_std([float(v) for v in pre])
    seg_std_tok = _safe_std([float(v) for v in seg])
    tok_clause = _std_clause(
        pre_std_tok, seg_std_tok) or "std could not be estimated reliably (token space)"

    if raw_values is not None:
        raw_pre = raw_values[pre0:start]
        raw_seg = raw_values[start:end]
        pre_std_raw = float(np.std(raw_pre.astype(np.float32))
                            ) if raw_pre.size else None
        seg_std_raw = float(np.std(raw_seg.astype(np.float32))
                            ) if raw_seg.size else None
    else:
        pre_std_raw = None
        seg_std_raw = None
    raw_clause = _std_clause(pre_std_raw, seg_std_raw)

    kind = str(event.get("kind", "")).strip(
    ) if isinstance(event, dict) else ""
    detail = (event.get("detail", {}) or {}) if isinstance(event, dict) else {}

    if kind == "shift":
        dir_txt = _dir_txt(detail.get("direction", None))
        return f"variability changes during a {dir_txt} shift: {tok_clause}."

    if kind == "noise":
        astd = detail.get("anomaly_std", None)
        oracle_clause = ""
        if isinstance(astd, (float, int)):
            oracle_clause = f"generated noise std (raw units) was {float(astd):.3f}"

        parts = [p for p in [raw_clause, oracle_clause] if p] or [tok_clause]
        extra = "; ".join(parts)

        base = (
            "Normally, the series is a periodic sine wave."
            if pattern_type == "point"
            else "Normally, the series is a noisy periodic sine wave."
        )
        return f"{base} From index {start} to {end}, the anomaly is dominated by dispersion/variance: {extra}."

    if kind == "freq_mult":
        return f"From index {start} to {end}, frequency changes; dispersion view: {tok_clause}."

    if kind == "slope_change":
        return f"From index {start} to {end}, a trend-shift occurs; dispersion view: {tok_clause}."

    return f"TODO(event_3): implement dispersion view for kind='{kind or 'unknown'}' on [{start}, {end}) (currently: {tok_clause})."


def build_event_4_description(
    *,
    pattern_type: str,
    start: int,
    end: int,
    values_0_99: list[int],
    raw_values: Optional[np.ndarray],
    event: Optional[dict[str, Any]],
    add_noise: bool,
) -> Optional[str]:
    """
    Event 4: placeholder for a full forensic explanation.
    """
    if not _validate_bounds(start, end, len(values_0_99)):
        return None
    kind = str(event.get("kind", "")).strip(
    ) if isinstance(event, dict) else "unknown"
    return f" "


def build_factual_descriptions_oracle(
    *,
    pattern_type: str,
    start: int,
    end: int,
    values_0_99: list[int],
    rolling_avg_0_99: list[int],
    raw_values: Optional[np.ndarray],
    event: Optional[dict[str, Any]],
    add_noise: bool,
) -> dict[str, Optional[str]]:
    """
    Returns 4 fixed keys: event_1..event_4.
    """
    if not _validate_bounds(start, end, len(values_0_99)):
        return {"event_1": None, "event_2": None, "event_3": None, "event_4": None}

    return {
        "event_1": build_event_1_description(
            pattern_type=pattern_type,
            start=start,
            end=end,
            values_0_99=values_0_99,
            raw_values=raw_values,
            event=event,
            add_noise=add_noise,
        ),
        "event_2": build_event_2_description(
            pattern_type=pattern_type,
            start=start,
            end=end,
            values_0_99=values_0_99,
            raw_values=raw_values,
            event=event,
            add_noise=add_noise,
            rolling_avg_0_99=rolling_avg_0_99,
        ),
        "event_3": build_event_3_description(
            pattern_type=pattern_type,
            start=start,
            end=end,
            values_0_99=values_0_99,
            raw_values=raw_values,
            event=event,
            add_noise=add_noise,
        ),
        "event_4": build_event_4_description(
            pattern_type=pattern_type,
            start=start,
            end=end,
            values_0_99=values_0_99,
            raw_values=raw_values,
            event=event,
            add_noise=add_noise,
        ),
    }


def build_factual_description_oracle(
    *,
    pattern_type: str,
    start: int,
    end: int,
    values_0_99: list[int],
    raw_values: Optional[np.ndarray],
    event: Optional[dict[str, Any]],
    add_noise: bool,
) -> Optional[str]:
    """
    Backwards-compatible API: returns event_1 only.
    """
    return build_event_1_description(
        pattern_type=pattern_type,
        start=start,
        end=end,
        values_0_99=values_0_99,
        raw_values=raw_values,
        event=event,
        add_noise=add_noise,
    )
