from typing import Any, Optional

import numpy as np
from scipy import stats

from utils.anomaly_utils import add_anomalies_to_univariate_series


def synthetic_point_anomalies(
    n_samples: int = 300,
    number_of_sensors: int = 1,
    frequency: float = 0.03,
    normal_duration_rate: float = 240.0,
    anomaly_duration_rate: float = 25.0,
    minimum_anomaly_duration: int = 5,
    minimum_normal_duration: int = 100,
    anomaly_std: float = 0.5,
    ratio_of_anomalous_sensors: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], dict[str, Any]]:
    """Generate sinusoidal multi-sensor data with Gaussian noise anomalies."""
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_samples)
    x = np.array([np.sin(2 * np.pi * (frequency + 0.01 * i) * t)
                 for i in range(number_of_sensors)]).T

    if number_of_sensors > 0:
        num_anom_sensors = max(
            1 if ratio_of_anomalous_sensors > 0 else 0,
            int(round(number_of_sensors * ratio_of_anomalous_sensors)),
        )
        num_anom_sensors = min(number_of_sensors, num_anom_sensors)
        sensors_with_anomalies = (
            np.random.choice(number_of_sensors, num_anom_sensors,
                             replace=False) if num_anom_sensors > 0 else []
        )
    else:
        sensors_with_anomalies = []

    anomaly_intervals = [[] for _ in range(number_of_sensors)]
    events: list[dict[str, Any]] = []

    for sensor in sensors_with_anomalies:
        _, intervals, _ = add_anomalies_to_univariate_series(
            np.zeros(n_samples, dtype=np.float32),
            normal_duration_rate,
            anomaly_duration_rate,
            (0.0, 1.0),
            minimum_anomaly_duration,
            minimum_normal_duration,
        )

        original_sensor_data = np.sin(
            2 * np.pi * (frequency + 0.01 * sensor) * t)
        modified_sensor_data = original_sensor_data.copy()
        sensor_anomalies: list[tuple[int, int]] = []

        for start, end in intervals:
            if start < end:
                anomaly = np.random.normal(0, anomaly_std, end - start)
                modified_sensor_data[start:end] = anomaly
                sensor_anomalies.append((start, end))
                events.append(
                    {
                        "sensor": int(sensor),
                        "start": int(start),
                        "end": int(end),
                        "kind": "noise",
                        "detail": {"anomaly_std": float(anomaly_std)},
                    }
                )

        x[:, sensor] = modified_sensor_data
        anomaly_intervals[sensor] = sensor_anomalies

    gen_info = {
        "pattern_type": "point",
        "params": {
            "frequency": float(frequency),
            "normal_duration_rate": float(normal_duration_rate),
            "anomaly_duration_rate": float(anomaly_duration_rate),
            "minimum_anomaly_duration": int(minimum_anomaly_duration),
            "minimum_normal_duration": int(minimum_normal_duration),
            "anomaly_std": float(anomaly_std),
            "ratio_of_anomalous_sensors": float(ratio_of_anomalous_sensors),
        },
        "events": events,
    }
    return x, anomaly_intervals, gen_info


def synthetic_freq_anomalies(
    n_samples: int = 300,
    number_of_sensors: int = 1,
    frequency: float = 0.03,
    normal_duration_rate: float = 135.0,
    anomaly_duration_rate: float = 15.0,
    minimum_anomaly_duration: int = 7,
    minimum_normal_duration: int = 100,
    frequency_multiplier: float = 3.0,
    ratio_of_anomalous_sensors: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], dict[str, Any]]:
    """Generate sinusoidal data with intermittent frequency-multiplier anomalies."""
    if seed is not None:
        np.random.seed(seed)

    x = np.zeros((n_samples, number_of_sensors), dtype=np.float32)

    if number_of_sensors > 0:
        num_anom_sensors = max(
            1 if ratio_of_anomalous_sensors > 0 else 0,
            int(round(number_of_sensors * ratio_of_anomalous_sensors)),
        )
        num_anom_sensors = min(number_of_sensors, num_anom_sensors)
        sensors_with_anomalies = (
            np.random.choice(number_of_sensors, num_anom_sensors,
                             replace=False) if num_anom_sensors > 0 else []
        )
    else:
        sensors_with_anomalies = []

    anomaly_intervals = [[] for _ in range(number_of_sensors)]
    events: list[dict[str, Any]] = []

    for sensor in range(number_of_sensors):
        base_freq = frequency + 0.01 * sensor
        freq_function = np.full(n_samples, base_freq, dtype=np.float32)

        if sensor in sensors_with_anomalies:
            current_time = 0
            while current_time < n_samples:
                normal_duration = max(minimum_normal_duration, int(
                    stats.expon(scale=normal_duration_rate).rvs()))
                current_time += normal_duration
                if current_time >= n_samples:
                    break

                anomaly_duration = max(minimum_anomaly_duration, int(
                    stats.expon(scale=anomaly_duration_rate).rvs()))
                anomaly_end = min(n_samples, current_time + anomaly_duration)
                if current_time >= anomaly_end:
                    current_time = anomaly_end
                    continue

                multiplier = (
                    frequency_multiplier
                    if np.random.random() < 0.5
                    else (1.0 / frequency_multiplier if frequency_multiplier != 0 else 1.0)
                )
                freq_function[current_time:anomaly_end] *= multiplier
                freq_function[current_time:anomaly_end] = np.maximum(
                    1e-9, freq_function[current_time:anomaly_end])

                anomaly_intervals[sensor].append((current_time, anomaly_end))
                events.append(
                    {
                        "sensor": int(sensor),
                        "start": int(current_time),
                        "end": int(anomaly_end),
                        "kind": "freq_mult",
                        "detail": {"multiplier": float(multiplier), "base_freq": float(base_freq)},
                    }
                )
                current_time = anomaly_end

        phase = np.cumsum(2 * np.pi * freq_function).astype(np.float32)
        x[:, sensor] = np.sin(phase).astype(np.float32)

    gen_info = {
        "pattern_type": "freq",
        "params": {
            "frequency": float(frequency),
            "normal_duration_rate": float(normal_duration_rate),
            "anomaly_duration_rate": float(anomaly_duration_rate),
            "minimum_anomaly_duration": int(minimum_anomaly_duration),
            "minimum_normal_duration": int(minimum_normal_duration),
            "frequency_multiplier": float(frequency_multiplier),
            "ratio_of_anomalous_sensors": float(ratio_of_anomalous_sensors),
        },
        "events": events,
    }
    return x, anomaly_intervals, gen_info


def generate_abnormal_slope(normal_slope: float, abnormal_slope_range: tuple[float, float], inverse_ratio: float) -> float:
    """Draw a slope value outside the normal range, optionally inverted."""
    min_slope, max_slope = abnormal_slope_range
    if np.isinf(max_slope):
        max_slope = max(abs(normal_slope) * 10, min_slope *
                        2) if normal_slope != 0 else min_slope * 2
    if max_slope <= min_slope:
        max_slope = min_slope + 1.0

    if np.random.random() > inverse_ratio:
        lower = max(abs(normal_slope), min_slope)
        upper = max_slope
        if lower >= upper:
            return np.sign(normal_slope) * lower if normal_slope != 0 else lower
        magnitude = np.random.uniform(lower, upper)
        return np.sign(normal_slope) * magnitude if normal_slope != 0 else magnitude
    else:
        lower = -max_slope
        upper = -min_slope
        if lower >= upper:
            return np.sign(normal_slope) * lower if normal_slope != 0 else lower
        return np.random.uniform(lower, upper)


def synthetic_trend_anomalies(
    n_samples: int = 300,
    number_of_sensors: int = 1,
    frequency: float = 0.02,
    normal_duration_rate: float = 510.0,
    anomaly_duration_rate: float = 100.0,
    minimum_anomaly_duration: int = 50,
    minimum_normal_duration: int = 150,
    ratio_of_anomalous_sensors: float = 1.0,
    normal_slope: float = 3.0,
    abnormal_slope_range: tuple[float, float] = (6.0, 20.0),
    inverse_ratio: float = 0.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], dict[str, Any]]:
    """Generate sinusoidal data superimposed with piece-wise linear trend anomalies."""
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_samples)
    x = np.zeros((n_samples, number_of_sensors), dtype=np.float32)

    if number_of_sensors > 0:
        num_anom_sensors = max(
            1 if ratio_of_anomalous_sensors > 0 else 0,
            int(round(number_of_sensors * ratio_of_anomalous_sensors)),
        )
        num_anom_sensors = min(number_of_sensors, num_anom_sensors)
        sensors_with_anomalies = (
            np.random.choice(number_of_sensors, num_anom_sensors,
                             replace=False) if num_anom_sensors > 0 else []
        )
    else:
        sensors_with_anomalies = []

    anomaly_intervals = [[] for _ in range(number_of_sensors)]
    events: list[dict[str, Any]] = []

    for sensor in range(number_of_sensors):
        base_freq = frequency + 0.01 * sensor
        trend = np.zeros(n_samples, dtype=np.float32)
        current_value = 0.0
        current_time = 0

        if sensor in sensors_with_anomalies:
            _, intervals, _ = add_anomalies_to_univariate_series(
                np.zeros(n_samples, dtype=np.float32),
                normal_duration_rate,
                anomaly_duration_rate,
                (0.0, 1.0),
                minimum_anomaly_duration,
                minimum_normal_duration,
            )
            last_interval_end = 0
            for start, end in intervals:
                start = max(last_interval_end, start)
                end = max(start, end)
                if start >= n_samples:
                    break
                end = min(n_samples, end)
                if start >= end:
                    continue

                if start > current_time:
                    time_segment = t[current_time:start] - t[current_time]
                    trend[current_time:start] = current_value + \
                        normal_slope * time_segment / n_samples
                    current_value = float(
                        trend[start - 1]) if start > current_time else current_value

                abnormal_slope = generate_abnormal_slope(
                    normal_slope, abnormal_slope_range, inverse_ratio)
                time_segment = t[start:end] - t[start]
                trend[start:end] = current_value + \
                    abnormal_slope * time_segment / n_samples
                current_value = float(
                    trend[end - 1]) if end > start else current_value

                current_time = end
                anomaly_intervals[sensor].append((start, end))
                last_interval_end = end

                events.append(
                    {
                        "sensor": int(sensor),
                        "start": int(start),
                        "end": int(end),
                        "kind": "slope_change",
                        "detail": {"normal_slope": float(normal_slope), "abnormal_slope": float(abnormal_slope)},
                    }
                )

        if current_time < n_samples:
            time_segment = t[current_time:] - t[current_time]
            trend[current_time:] = current_value + \
                normal_slope * time_segment / n_samples

        x[:, sensor] = (np.sin(2 * np.pi * base_freq * t) +
                        trend).astype(np.float32)

        min_val, max_val = float(
            np.min(x[:, sensor])), float(np.max(x[:, sensor]))
        if max_val > min_val:
            x[:, sensor] = (2 * (x[:, sensor] - min_val) /
                            (max_val - min_val) - 1).astype(np.float32)
        elif np.any(x[:, sensor]):
            x[:, sensor] = 0.0

    gen_info = {
        "pattern_type": "trend",
        "params": {
            "frequency": float(frequency),
            "normal_duration_rate": float(normal_duration_rate),
            "anomaly_duration_rate": float(anomaly_duration_rate),
            "minimum_anomaly_duration": int(minimum_anomaly_duration),
            "minimum_normal_duration": int(minimum_normal_duration),
            "ratio_of_anomalous_sensors": float(ratio_of_anomalous_sensors),
            "normal_slope": float(normal_slope),
            "abnormal_slope_range": (float(abnormal_slope_range[0]), float(abnormal_slope_range[1])),
            "inverse_ratio": float(inverse_ratio),
        },
        "events": events,
    }
    return x, anomaly_intervals, gen_info


def synthetic_flat_trend_anomalies(**args):
    """Wrapper for synthetic_trend_anomalies with gentle slope settings."""
    flat_args = {"normal_slope": 3.0, "abnormal_slope_range": (
        0.1, 1.5), "inverse_ratio": 0.0}
    flat_args.update(args)
    return synthetic_trend_anomalies(**flat_args)


def synthetic_range_anomalies(
    number_of_sensors: int = 1,
    train_size: int = 5_000,
    test_size: int = 300,
    nominal_data_mean: float = 0.0,
    nominal_data_std: float = 0.1,
    normal_duration_rate: float = 240.0,
    anomaly_duration_rate: float = 20.0,
    anomaly_size_range: tuple = (0.5, 0.8),
    minimum_anomaly_duration: int = 5,
    minimum_normal_duration: int = 100,
    ratio_of_anomalous_sensors: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], dict[str, Any]]:
    """Generate Gaussian noise with sporadic mean-shift anomalies on test data."""
    if seed is not None:
        np.random.seed(seed)

    test_data = np.random.normal(nominal_data_mean, nominal_data_std, size=(
        test_size, number_of_sensors)).astype(np.float32)

    if number_of_sensors > 0:
        num_anom_sensors = max(
            1 if ratio_of_anomalous_sensors > 0 else 0,
            int(round(number_of_sensors * ratio_of_anomalous_sensors)),
        )
        num_anom_sensors = min(number_of_sensors, num_anom_sensors)
        sensors_with_anomalies = (
            np.random.choice(number_of_sensors, num_anom_sensors,
                             replace=False) if num_anom_sensors > 0 else []
        )
    else:
        sensors_with_anomalies = []

    all_anomaly_intervals = [[] for _ in range(number_of_sensors)]
    events: list[dict[str, Any]] = []

    for sensor in sensors_with_anomalies:
        modified_series, anomaly_locations, evs = add_anomalies_to_univariate_series(
            test_data[:, sensor],
            normal_duration_rate,
            anomaly_duration_rate,
            (float(anomaly_size_range[0]), float(anomaly_size_range[1])),
            minimum_anomaly_duration,
            minimum_normal_duration,
        )
        test_data[:, sensor] = modified_series
        all_anomaly_intervals[sensor] = anomaly_locations

        for ev in evs:
            if ev.get("kind") == "shift":
                events.append(
                    {
                        "sensor": int(sensor),
                        "start": int(ev["start"]),
                        "end": int(ev["end"]),
                        "kind": "shift",
                        "detail": dict(ev.get("detail", {})),
                    }
                )

    gen_info = {
        "pattern_type": "range",
        "params": {
            "nominal_data_mean": float(nominal_data_mean),
            "nominal_data_std": float(nominal_data_std),
            "normal_duration_rate": float(normal_duration_rate),
            "anomaly_duration_rate": float(anomaly_duration_rate),
            "anomaly_size_range": (float(anomaly_size_range[0]), float(anomaly_size_range[1])),
            "minimum_anomaly_duration": int(minimum_anomaly_duration),
            "minimum_normal_duration": int(minimum_normal_duration),
            "ratio_of_anomalous_sensors": float(ratio_of_anomalous_sensors),
        },
        "events": events,
    }
    return test_data, all_anomaly_intervals, gen_info
