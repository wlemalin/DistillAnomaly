#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate affiliation metrics over several experiment directories.

Expected file naming convention:

    experiment-name-1.jsonl
    experiment-name-2.jsonl
    experiment-name-3.jsonl

The final "-N" suffix is treated as the run id.
All runs belonging to the same experiment are averaged together.

This script imports metric/parsing logic from affiliation_metrics.py.
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from affil_overview import (
    compute_affiliation_single,
    f1_from_pr,
    load_jsonl,
    parse_gt_interval,
    parse_pred_interval,
)


RUN_SUFFIX_RE = re.compile(r"^(?P<experiment>.+)-(?P<run>\d+)$")


NUMERIC_METRICS = [
    "precision",
    "recall",
    "f1",
    "avg_d_pred_to_gt",
    "avg_d_gt_to_pred",
    "invalid_detection_rate_percent",
]

COUNT_METRICS = [
    "total_rows",
    "total_with_gt",
    "valid_predictions",
    "invalid_predictions",
]


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def safe_std(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    if len(xs) <= 1:
        return 0.0 if len(xs) == 1 else None
    return statistics.stdev(xs)


def fmt_float(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{decimals}f}"


def fmt_count(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def relpath(path: Path) -> str:
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:
        return str(path)


def split_experiment_and_run(
    jsonl_path: Path,
    include_unmatched: bool = False,
) -> Optional[Tuple[str, Optional[int]]]:
    """
    Convert:

        eval-qwen-v3-outsample-dpop-bt-ep6-1.jsonl

    into:

        ("eval-qwen-v3-outsample-dpop-bt-ep6", 1)

    By default, files without a final "-N" suffix are ignored.
    """
    stem = jsonl_path.stem
    match = RUN_SUFFIX_RE.match(stem)

    if match:
        return match.group("experiment"), int(match.group("run"))

    if include_unmatched:
        return stem, None

    return None


def discover_jsonl_files(
    input_paths: List[str],
    pattern: str,
    recursive: bool,
) -> List[Path]:
    files: List[Path] = []

    for raw_path in input_paths:
        path = Path(raw_path)

        if path.is_file():
            if path.suffix == ".jsonl":
                files.append(path)
            else:
                print(f"Warning: ignored non-JSONL file: {path}", file=sys.stderr)
            continue

        if path.is_dir():
            glob_pattern = f"**/{pattern}" if recursive else pattern
            files.extend(sorted(path.glob(glob_pattern)))
            continue

        print(f"Warning: input path does not exist: {path}", file=sys.stderr)

    unique_files = sorted({p.resolve() for p in files if p.is_file() and p.suffix == ".jsonl"})
    return unique_files


def process_single_jsonl(jsonl_path: Path, T0: int, T1: int) -> Dict[str, Any]:
    """
    Compute strict affiliation metrics for one JSONL file.

    Strict convention:
    - missing/invalid prediction => precision = 0, recall = 0, f1 = 0
    - distance metrics are averaged only when defined
    """
    rows = load_jsonl(jsonl_path)

    overall = {
        "p": [],
        "r": [],
        "f1": [],
        "dp2g": [],
        "dg2p": [],
    }

    by_ptype: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "p": [],
            "r": [],
            "f1": [],
            "dp2g": [],
            "dg2p": [],
            "total_with_gt": 0,
            "valid_predictions": 0,
        }
    )

    total_with_gt = 0
    valid_pred_count = 0

    for row in rows:
        pt = str(row.get("pattern_type", "unknown"))
        gt = parse_gt_interval(row)
        pred = parse_pred_interval(row)

        if gt is None:
            continue

        prec, rec, dP2G, dG2P, has_pred = compute_affiliation_single(T0, T1, gt, pred)
        f1 = f1_from_pr(prec, rec)

        total_with_gt += 1
        by_ptype[pt]["total_with_gt"] += 1

        if has_pred:
            valid_pred_count += 1
            by_ptype[pt]["valid_predictions"] += 1

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
    invalid_rate = (invalid_count / total_with_gt * 100.0) if total_with_gt else 0.0

    by_pattern_type: Dict[str, Dict[str, Any]] = {}

    for pt, bucket in sorted(by_ptype.items()):
        pt_total = bucket["total_with_gt"]
        pt_valid = bucket["valid_predictions"]
        pt_invalid = pt_total - pt_valid
        pt_invalid_rate = (pt_invalid / pt_total * 100.0) if pt_total else 0.0

        by_pattern_type[pt] = {
            "total_with_gt": pt_total,
            "valid_predictions": pt_valid,
            "invalid_predictions": pt_invalid,
            "invalid_detection_rate_percent": pt_invalid_rate,
            "precision": safe_mean(bucket["p"]),
            "recall": safe_mean(bucket["r"]),
            "f1": safe_mean(bucket["f1"]),
            "avg_d_pred_to_gt": safe_mean(bucket["dp2g"]),
            "avg_d_gt_to_pred": safe_mean(bucket["dg2p"]),
        }

    return {
        "file": str(jsonl_path),
        "total_rows": len(rows),
        "total_with_gt": total_with_gt,
        "valid_predictions": valid_pred_count,
        "invalid_predictions": invalid_count,
        "invalid_detection_rate_percent": invalid_rate,
        "precision": safe_mean(overall["p"]),
        "recall": safe_mean(overall["r"]),
        "f1": safe_mean(overall["f1"]),
        "avg_d_pred_to_gt": safe_mean(overall["dp2g"]),
        "avg_d_gt_to_pred": safe_mean(overall["dg2p"]),
        "by_pattern_type": by_pattern_type,
    }


def aggregate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for key in NUMERIC_METRICS:
        values = [record.get(key) for record in records]
        out[f"{key}_mean"] = safe_mean(values)
        out[f"{key}_std"] = safe_std(values)

    for key in COUNT_METRICS:
        values = [record.get(key) for record in records]
        out[f"{key}_mean"] = safe_mean(values)
        out[f"{key}_sum"] = sum(float(v) for v in values if v is not None)

    return out


def aggregate_pattern_types(file_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_ptypes = sorted(
        {
            pt
            for metrics in file_metrics
            for pt in metrics.get("by_pattern_type", {}).keys()
        }
    )

    out: Dict[str, Any] = {}

    for pt in all_ptypes:
        records = [
            metrics["by_pattern_type"][pt]
            for metrics in file_metrics
            if pt in metrics.get("by_pattern_type", {})
        ]

        out[pt] = aggregate_records(records)
        out[pt]["n_runs_with_pattern_type"] = len(records)

    return out


def group_files_by_experiment(
    files: List[Path],
    include_unmatched: bool,
) -> Tuple[Dict[Tuple[Path, str], List[Dict[str, Any]]], List[str]]:
    groups: Dict[Tuple[Path, str], List[Dict[str, Any]]] = defaultdict(list)
    warnings: List[str] = []

    for jsonl_path in files:
        parsed = split_experiment_and_run(jsonl_path, include_unmatched=include_unmatched)

        if parsed is None:
            warnings.append(
                f"Ignored file without final run suffix '-N': {relpath(jsonl_path)}"
            )
            continue

        experiment, run_id = parsed
        directory = jsonl_path.parent.resolve()

        groups[(directory, experiment)].append(
            {
                "path": jsonl_path,
                "run_id": run_id,
            }
        )

    return groups, warnings


def aggregate_experiment_group(
    directory: Path,
    experiment: str,
    run_items: List[Dict[str, Any]],
    T0: int,
    T1: int,
    expected_runs: Optional[int],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []

    sorted_items = sorted(
        run_items,
        key=lambda item: (
            item["run_id"] is None,
            item["run_id"] if item["run_id"] is not None else 10**9,
            item["path"].name,
        ),
    )

    run_results: List[Dict[str, Any]] = []

    seen_run_ids: Dict[int, int] = defaultdict(int)

    for item in sorted_items:
        run_id = item["run_id"]
        path = item["path"]

        if run_id is not None:
            seen_run_ids[run_id] += 1

        try:
            metrics = process_single_jsonl(path, T0=T0, T1=T1)
        except Exception as exc:
            warnings.append(f"Failed to process {relpath(path)}: {exc}")
            continue

        run_results.append(
            {
                "run_id": run_id,
                "file": relpath(path),
                "metrics": metrics,
            }
        )

    if not run_results:
        return None, warnings

    duplicate_run_ids = sorted(
        run_id for run_id, count in seen_run_ids.items() if count > 1
    )

    if duplicate_run_ids:
        warnings.append(
            f"Duplicate run ids for {relpath(directory)}/{experiment}: {duplicate_run_ids}"
        )

    observed_run_ids = sorted(
        run_result["run_id"]
        for run_result in run_results
        if run_result["run_id"] is not None
    )

    missing_run_ids: List[int] = []

    if expected_runs is not None and expected_runs > 0:
        expected = set(range(1, expected_runs + 1))
        observed = set(observed_run_ids)
        missing_run_ids = sorted(expected - observed)

        if missing_run_ids:
            warnings.append(
                f"Missing run ids for {relpath(directory)}/{experiment}: {missing_run_ids}"
            )

    file_metrics = [run_result["metrics"] for run_result in run_results]
    aggregated = aggregate_records(file_metrics)

    result = {
        "directory": relpath(directory),
        "experiment": experiment,
        "label": f"{relpath(directory)}/{experiment}",
        "n_runs": len(run_results),
        "run_ids": observed_run_ids,
        "missing_run_ids": missing_run_ids,
        "runs": run_results,
        "by_pattern_type": aggregate_pattern_types(file_metrics),
        **aggregated,
    }

    return result, warnings


def build_summary(args: argparse.Namespace) -> Dict[str, Any]:
    files = discover_jsonl_files(
        input_paths=args.paths,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    groups, warnings = group_files_by_experiment(
        files,
        include_unmatched=args.include_unmatched,
    )

    experiments: List[Dict[str, Any]] = []

    for (directory, experiment), run_items in sorted(
        groups.items(),
        key=lambda kv: (str(kv[0][0]), kv[0][1]),
    ):
        result, group_warnings = aggregate_experiment_group(
            directory=directory,
            experiment=experiment,
            run_items=run_items,
            T0=args.T0,
            T1=args.T1,
            expected_runs=args.expected_runs,
        )

        warnings.extend(group_warnings)

        if result is not None:
            experiments.append(result)

    return {
        "domain": [args.T0, args.T1],
        "inputs": args.paths,
        "pattern": args.pattern,
        "recursive": args.recursive,
        "expected_runs": args.expected_runs,
        "n_files_discovered": len(files),
        "n_experiments": len(experiments),
        "experiments": experiments,
        "warnings": warnings,
    }


def experiment_to_csv_row(exp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "directory": exp["directory"],
        "experiment": exp["experiment"],
        "label": exp["label"],
        "n_runs": exp["n_runs"],
        "run_ids": ",".join(str(x) for x in exp["run_ids"]),
        "missing_run_ids": ",".join(str(x) for x in exp["missing_run_ids"]),
        "total_rows_mean": exp.get("total_rows_mean"),
        "total_with_gt_mean": exp.get("total_with_gt_mean"),
        "valid_predictions_mean": exp.get("valid_predictions_mean"),
        "invalid_predictions_mean": exp.get("invalid_predictions_mean"),
        "invalid_detection_rate_percent_mean": exp.get("invalid_detection_rate_percent_mean"),
        "invalid_detection_rate_percent_std": exp.get("invalid_detection_rate_percent_std"),
        "precision_mean": exp.get("precision_mean"),
        "precision_std": exp.get("precision_std"),
        "recall_mean": exp.get("recall_mean"),
        "recall_std": exp.get("recall_std"),
        "f1_mean": exp.get("f1_mean"),
        "f1_std": exp.get("f1_std"),
        "avg_d_pred_to_gt_mean": exp.get("avg_d_pred_to_gt_mean"),
        "avg_d_pred_to_gt_std": exp.get("avg_d_pred_to_gt_std"),
        "avg_d_gt_to_pred_mean": exp.get("avg_d_gt_to_pred_mean"),
        "avg_d_gt_to_pred_std": exp.get("avg_d_gt_to_pred_std"),
    }


def write_csv(summary: Dict[str, Any], out_path: Optional[Path] = None) -> None:
    rows = [experiment_to_csv_row(exp) for exp in summary["experiments"]]

    fieldnames = [
        "directory",
        "experiment",
        "label",
        "n_runs",
        "run_ids",
        "missing_run_ids",
        "total_rows_mean",
        "total_with_gt_mean",
        "valid_predictions_mean",
        "invalid_predictions_mean",
        "invalid_detection_rate_percent_mean",
        "invalid_detection_rate_percent_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "avg_d_pred_to_gt_mean",
        "avg_d_pred_to_gt_std",
        "avg_d_gt_to_pred_mean",
        "avg_d_gt_to_pred_std",
    ]

    if out_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(summary: Dict[str, Any]) -> None:
    experiments = summary["experiments"]

    print()
    print("=" * 140)
    print("AFFILIATION METRICS SUMMARY")
    print("=" * 140)
    print(f"Domain: [{summary['domain'][0]}, {summary['domain'][1]})")
    print(f"Files discovered: {summary['n_files_discovered']}")
    print(f"Experiments:      {summary['n_experiments']}")
    print(f"Expected runs:    {summary['expected_runs']}")
    print("=" * 140)

    if not experiments:
        print("No experiments found.")
        return

    header = (
        f"{'Experiment':<68} "
        f"{'Runs':>4} "
        f"{'Missing':>8} "
        f"{'GT/run':>8} "
        f"{'Valid/run':>10} "
        f"{'Invalid%':>9} "
        f"{'Prec':>8} "
        f"{'Recall':>8} "
        f"{'F1':>8} "
        f"{'F1 std':>8} "
        f"{'dP→GT':>8} "
        f"{'dGT→P':>8}"
    )

    print(header)
    print("-" * len(header))

    for exp in experiments:
        missing = (
            "-"
            if not exp["missing_run_ids"]
            else ",".join(str(x) for x in exp["missing_run_ids"])
        )

        row = (
            f"{exp['label']:<68} "
            f"{exp['n_runs']:>4} "
            f"{missing:>8} "
            f"{fmt_count(exp.get('total_with_gt_mean')):>8} "
            f"{fmt_count(exp.get('valid_predictions_mean')):>10} "
            f"{fmt_float(exp.get('invalid_detection_rate_percent_mean'), 2):>9} "
            f"{fmt_float(exp.get('precision_mean')):>8} "
            f"{fmt_float(exp.get('recall_mean')):>8} "
            f"{fmt_float(exp.get('f1_mean')):>8} "
            f"{fmt_float(exp.get('f1_std')):>8} "
            f"{fmt_float(exp.get('avg_d_pred_to_gt_mean'), 2):>8} "
            f"{fmt_float(exp.get('avg_d_gt_to_pred_mean'), 2):>8}"
        )

        print(row)

    if summary["warnings"]:
        print()
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")


def save_output(summary: Dict[str, Any], out_path: Path) -> None:
    suffix = out_path.suffix.lower()

    if suffix == ".json":
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {out_path}")
        return

    write_csv(summary, out_path=out_path)
    print(f"\nSaved CSV report to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate strict affiliation metrics across JSONL experiment runs."
    )

    ap.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="One or several directories/files to aggregate.",
    )

    ap.add_argument(
        "--T0",
        type=int,
        default=0,
        help="Domain start, inclusive. Default: 0.",
    )

    ap.add_argument(
        "--T1",
        type=int,
        default=300,
        help="Domain end, exclusive. Default: 300.",
    )

    ap.add_argument(
        "--expected-runs",
        type=int,
        default=3,
        help="Expected number of runs per experiment. Default: 3.",
    )

    ap.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern used inside directories. Default: *.jsonl.",
    )

    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Search JSONL files recursively inside input directories.",
    )

    ap.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Include JSONL files without final '-N' run suffix as single-run experiments.",
    )

    ap.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Stdout format. Default: table.",
    )

    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path. Use .json for full nested report, otherwise CSV.",
    )

    args = ap.parse_args()

    summary = build_summary(args)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    elif args.format == "csv":
        write_csv(summary, out_path=None)
    else:
        print_table(summary)

    if args.out:
        save_output(summary, Path(args.out))


if __name__ == "__main__":
    main()
