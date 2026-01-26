from typing import Any, Optional

import numpy as np
from scipy import stats


def add_anomalies_to_univariate_series(
    x: np.ndarray,
    normal_duration_rate: float,
    anomaly_duration_rate: float,
    anomaly_size_range: tuple[float, float],
    minimum_anomaly_duration: int,
    minimum_normal_duration: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[dict[str, Any]]]:
    """Inject random mean-shift anomalies into a 1-D series."""
    is_dummy_range = (anomaly_size_range[0] == 0 and anomaly_size_range[1] == 1)
    if not is_dummy_range and anomaly_size_range[0] >= anomaly_size_range[1]:
        raise ValueError(
            f"The anomaly size range {anomaly_size_range} should be strictly increasing.")

    x_copy = x.copy()
    N = len(x_copy)
    distr_duration_normal = stats.expon(scale=normal_duration_rate)
    distr_duration_anomalous = stats.expon(scale=anomaly_duration_rate)

    max_number_of_intervals = 8
    location = 0
    anomaly_intervals: list[tuple[int, int]] = []
    events: list[dict[str, Any]] = []

    for _ in range(max_number_of_intervals):
        random_states = np.random.randint(0, np.iinfo(np.int32).max, size=2)

        norm_dur = distr_duration_normal.rvs(random_state=random_states[0])
        norm_dur = max(norm_dur, minimum_normal_duration)
        anom_start = location + int(norm_dur)
        if anom_start >= N:
            break

        anom_dur = distr_duration_anomalous.rvs(random_state=random_states[1])
        anom_dur = max(anom_dur, minimum_anomaly_duration)
        anom_end = min(N, anom_start + int(anom_dur))
        if anom_start >= anom_end:
            location = anom_end
            continue

        detail: dict[str, Any] = {}
        if not is_dummy_range:
            shift_sign = 1 if np.random.randint(low=0, high=2) == 1 else -1
            shift_magnitude = np.random.uniform(
                anomaly_size_range[0], anomaly_size_range[1], size=anom_end - anom_start)
            x_copy[anom_start:anom_end] += shift_sign * shift_magnitude

            detail = {
                "direction": "up" if shift_sign > 0 else "down",
                "magnitude_mean": float(np.mean(shift_magnitude)) if shift_magnitude.size else 0.0,
                "magnitude_min": float(np.min(shift_magnitude)) if shift_magnitude.size else 0.0,
                "magnitude_max": float(np.max(shift_magnitude)) if shift_magnitude.size else 0.0,
                "size_range": (float(anomaly_size_range[0]), float(anomaly_size_range[1])),
            }

        location = anom_end
        anomaly_intervals.append((anom_start, anom_end))
        events.append(
            {
                "start": int(anom_start),
                "end": int(anom_end),
                "kind": "shift" if not is_dummy_range else "interval",
                "detail": detail,
            }
        )
        if location >= N:
            break

    return x_copy, anomaly_intervals, events


def _flatten_intervals(intervals_per_sensor: list[list[tuple[int, int]]]) -> list[tuple[int, int, int]]:
    """Flatten per-sensor interval lists into (sensor, start, end) tuples."""
    flat: list[tuple[int, int, int]] = []
    for si, lst in enumerate(intervals_per_sensor):
        for s, e in lst:
            flat.append((si, int(s), int(e)))
    return flat


def _choose_trim_start(*, orig_len: int, final_len: int, window: int, anom_start: int, anom_end: int) -> Optional[int]:
    """Pick a random trim start that keeps the anomaly inside the window."""
    if final_len > orig_len:
        return None
    lower = max(0, anom_end - final_len, window)
    upper = min(anom_start, orig_len - final_len)
    if lower > upper:
        return None
    return int(np.random.randint(lower, upper + 1))


def _match_event(gen_info: Optional[dict[str, Any]], sensor: int, start: int, end: int) -> Optional[dict[str, Any]]:
    """Retrieve the event record matching the given sensor and interval."""
    if not gen_info or "events" not in gen_info:
        return None
    events = gen_info.get("events", [])
    for ev in events:
        try:
            if int(ev.get("sensor", -1)) == int(sensor) and int(ev.get("start")) == int(start) and int(ev.get("end")) == int(end):
                return ev
        except Exception:
            continue
    return None
