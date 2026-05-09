#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate citation verification metrics over several experiment directories.

Expected file naming convention:

    experiment-name-1.jsonl
    experiment-name-2.jsonl
    experiment-name-3.jsonl

The final "-N" suffix is treated as the run id.
All runs belonging to the same experiment are averaged together.

This script imports the citation verification backend from citation_verification.py.
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

from eval_explanations import parse_context_data, verify_citations


RUN_SUFFIX_RE = re.compile(r"^(?P<experiment>.+)-(?P<run>\d+)$")


SUMMARY_METRICS = [
    "avg_true_per_record",
    "avg_false_per_record",
    "correct_citations_percent",
    "fully_truthful_explanations_percent",
    "feature_confusion_frequency_percent",
    "time_confusion_frequency_percent",
    "other_error_frequency_percent",
]

COUNT_METRICS = [
    "processed_records",
    "total_citations",
    "true_citations",
    "false_citations",
    "perfect_records",
    "feature_confusion_errors",
    "time_imprecision_errors",
    "other_errors",
]


def relpath(path: Path) -> str:
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:
        return str(path)


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def safe_std(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    return statistics.stdev(xs)


def fmt_float(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value:.{decimals}f}"


def fmt_count(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def init_stats() -> Dict[str, int]:
    return {
        "count": 0,
        "true": 0,
        "false": 0,
        "perfect": 0,
        "feature_confusion_errors": 0,
        "time_imprecision_errors": 0,
        "other_errors": 0,
    }


def load_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    invalid_lines = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                invalid_lines += 1

    return records, invalid_lines


def default_empty_eval_result(reason: str) -> Dict[str, Any]:
    return {
        "has_citations": False,
        "true_count": 0,
        "false_count": 0,
        "total_citations": 0,
        "score": 0.0,
        "logs": [reason],
        "failed_examples": [],
        "feature_confusion_errors": 0,
        "feature_confusion_frequency": 0.0,
        "time_imprecision_errors": 0,
        "time_imprecision_frequency": 0.0,
        "other_errors": 0,
        "other_error_frequency": 0.0,
    }


def update_stats(stats: Dict[str, int], eval_result: Dict[str, Any]) -> None:
    true_count = int(eval_result.get("true_count", 0))
    false_count = int(eval_result.get("false_count", 0))

    stats["count"] += 1
    stats["true"] += true_count
    stats["false"] += false_count

    stats["feature_confusion_errors"] += int(eval_result.get("feature_confusion_errors", 0))
    stats["time_imprecision_errors"] += int(eval_result.get("time_imprecision_errors", 0))
    stats["other_errors"] += int(eval_result.get("other_errors", 0))

    if false_count == 0:
        stats["perfect"] += 1


def summarize_stats(stats: Dict[str, int]) -> Dict[str, Any]:
    processed_records = stats["count"]
    true_citations = stats["true"]
    false_citations = stats["false"]
    total_citations = true_citations + false_citations

    feature_confusion_errors = stats["feature_confusion_errors"]
    time_imprecision_errors = stats["time_imprecision_errors"]
    other_errors = stats["other_errors"]

    return {
        "processed_records": processed_records,
        "total_citations": total_citations,
        "true_citations": true_citations,
        "false_citations": false_citations,
        "perfect_records": stats["perfect"],

        "avg_true_per_record": (
            true_citations / processed_records if processed_records > 0 else None
        ),
        "avg_false_per_record": (
            false_citations / processed_records if processed_records > 0 else None
        ),
        "correct_citations_percent": (
            100.0 * true_citations / total_citations if total_citations > 0 else None
        ),
        "fully_truthful_explanations_percent": (
            100.0 * stats["perfect"] / processed_records if processed_records > 0 else None
        ),

        "feature_confusion_errors": feature_confusion_errors,
        "time_imprecision_errors": time_imprecision_errors,
        "other_errors": other_errors,

        "feature_confusion_frequency_percent": (
            100.0 * feature_confusion_errors / total_citations
            if total_citations > 0
            else None
        ),
        "time_confusion_frequency_percent": (
            100.0 * time_imprecision_errors / total_citations
            if total_citations > 0
            else None
        ),
        "other_error_frequency_percent": (
            100.0 * other_errors / total_citations
            if total_citations > 0
            else None
        ),
    }


def process_single_jsonl(
    jsonl_path: Path,
    prompt_key: str,
    answer_key: str,
    plain_text: bool,
    context_window: int,
    max_failed_examples: int,
) -> Dict[str, Any]:
    records, invalid_json_lines = load_jsonl(jsonl_path)

    overall_stats = init_stats()
    by_pattern_stats: Dict[str, Dict[str, int]] = defaultdict(init_stats)

    missing_key_records = 0
    empty_context_records = 0

    for record in records:
        if prompt_key not in record or answer_key not in record:
            missing_key_records += 1
            continue

        pattern_type = str(record.get("pattern_type", "unknown"))

        prompt_text = record[prompt_key]
        eval_text = record[answer_key]

        data_map = parse_context_data(prompt_text, plain_text=plain_text)

        if not data_map:
            empty_context_records += 1
            eval_result = default_empty_eval_result("FAIL: Could not parse context data")
        else:
            eval_result = verify_citations(
                eval_text,
                data_map,
                context_window=context_window,
                max_failed_examples=max_failed_examples,
            )

        update_stats(overall_stats, eval_result)
        update_stats(by_pattern_stats[pattern_type], eval_result)

    by_pattern_type = {
        pattern_type: summarize_stats(stats)
        for pattern_type, stats in sorted(by_pattern_stats.items())
    }

    return {
        "file": relpath(jsonl_path),
        "input_records": len(records),
        "invalid_json_lines": invalid_json_lines,
        "missing_key_records": missing_key_records,
        "empty_context_records": empty_context_records,
        **summarize_stats(overall_stats),
        "by_pattern_type": by_pattern_type,
    }


def split_experiment_and_run(
    jsonl_path: Path,
    include_unmatched: bool = False,
) -> Optional[Tuple[str, Optional[int]]]:
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

    return sorted({p.resolve() for p in files if p.is_file() and p.suffix == ".jsonl"})


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


def aggregate_metric_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for metric in SUMMARY_METRICS:
        values = [record.get(metric) for record in records]
        out[f"{metric}_mean"] = safe_mean(values)
        out[f"{metric}_std"] = safe_std(values)

    for metric in COUNT_METRICS:
        values = [record.get(metric) for record in records]
        out[f"{metric}_mean"] = safe_mean(values)
        out[f"{metric}_sum"] = sum(float(v) for v in values if v is not None)

    return out


def aggregate_pattern_types(run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_pattern_types = sorted(
        {
            pattern_type
            for metrics in run_metrics
            for pattern_type in metrics.get("by_pattern_type", {}).keys()
        }
    )

    out: Dict[str, Any] = {}

    for pattern_type in all_pattern_types:
        records = [
            metrics["by_pattern_type"][pattern_type]
            for metrics in run_metrics
            if pattern_type in metrics.get("by_pattern_type", {})
        ]

        out[pattern_type] = {
            **aggregate_metric_records(records),
            "n_runs_with_pattern_type": len(records),
        }

    return out


def aggregate_experiment_group(
    directory: Path,
    experiment: str,
    run_items: List[Dict[str, Any]],
    args: argparse.Namespace,
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

    seen_run_ids: Dict[int, int] = defaultdict(int)
    runs: List[Dict[str, Any]] = []

    for item in sorted_items:
        path = item["path"]
        run_id = item["run_id"]

        if run_id is not None:
            seen_run_ids[run_id] += 1

        try:
            metrics = process_single_jsonl(
                jsonl_path=path,
                prompt_key=args.prompt_key,
                answer_key=args.answer_key,
                plain_text=args.plain_text,
                context_window=args.context_window,
                max_failed_examples=args.max_failed_examples,
            )
        except Exception as exc:
            warnings.append(f"Failed to process {relpath(path)}: {exc}")
            continue

        runs.append(
            {
                "run_id": run_id,
                "file": relpath(path),
                "metrics": metrics,
            }
        )

    if not runs:
        return None, warnings

    duplicate_run_ids = sorted(
        run_id for run_id, count in seen_run_ids.items() if count > 1
    )

    if duplicate_run_ids:
        warnings.append(
            f"Duplicate run ids for {relpath(directory)}/{experiment}: {duplicate_run_ids}"
        )

    observed_run_ids = sorted(
        run["run_id"]
        for run in runs
        if run["run_id"] is not None
    )

    missing_run_ids: List[int] = []

    if args.expected_runs is not None and args.expected_runs > 0:
        expected = set(range(1, args.expected_runs + 1))
        observed = set(observed_run_ids)
        missing_run_ids = sorted(expected - observed)

        if missing_run_ids:
            warnings.append(
                f"Missing run ids for {relpath(directory)}/{experiment}: {missing_run_ids}"
            )

    run_metrics = [run["metrics"] for run in runs]

    return {
        "directory": relpath(directory),
        "experiment": experiment,
        "label": f"{relpath(directory)}/{experiment}",
        "n_runs": len(runs),
        "run_ids": observed_run_ids,
        "missing_run_ids": missing_run_ids,
        "runs": runs,
        **aggregate_metric_records(run_metrics),
        "by_pattern_type": aggregate_pattern_types(run_metrics),
    }, warnings


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
            args=args,
        )

        warnings.extend(group_warnings)

        if result is not None:
            experiments.append(result)

    return {
        "inputs": args.paths,
        "pattern": args.pattern,
        "recursive": args.recursive,
        "expected_runs": args.expected_runs,
        "prompt_key": args.prompt_key,
        "answer_key": args.answer_key,
        "plain_text": args.plain_text,
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

        "processed_records_mean": exp.get("processed_records_mean"),
        "total_citations_mean": exp.get("total_citations_mean"),
        "true_citations_mean": exp.get("true_citations_mean"),
        "false_citations_mean": exp.get("false_citations_mean"),
        "perfect_records_mean": exp.get("perfect_records_mean"),

        "avg_true_per_record_mean": exp.get("avg_true_per_record_mean"),
        "avg_false_per_record_mean": exp.get("avg_false_per_record_mean"),

        "correct_citations_percent_mean": exp.get("correct_citations_percent_mean"),
        "correct_citations_percent_std": exp.get("correct_citations_percent_std"),

        "fully_truthful_explanations_percent_mean": exp.get(
            "fully_truthful_explanations_percent_mean"
        ),
        "fully_truthful_explanations_percent_std": exp.get(
            "fully_truthful_explanations_percent_std"
        ),

        "feature_confusion_frequency_percent_mean": exp.get(
            "feature_confusion_frequency_percent_mean"
        ),
        "feature_confusion_frequency_percent_std": exp.get(
            "feature_confusion_frequency_percent_std"
        ),

        "time_confusion_frequency_percent_mean": exp.get(
            "time_confusion_frequency_percent_mean"
        ),
        "time_confusion_frequency_percent_std": exp.get(
            "time_confusion_frequency_percent_std"
        ),

        "other_error_frequency_percent_mean": exp.get(
            "other_error_frequency_percent_mean"
        ),
        "other_error_frequency_percent_std": exp.get(
            "other_error_frequency_percent_std"
        ),

        "feature_confusion_errors_mean": exp.get("feature_confusion_errors_mean"),
        "time_imprecision_errors_mean": exp.get("time_imprecision_errors_mean"),
        "other_errors_mean": exp.get("other_errors_mean"),
    }


def write_csv(summary: Dict[str, Any], out_path: Optional[Path] = None) -> None:
    rows = [experiment_to_csv_row(exp) for exp in summary["experiments"]]

    fieldnames = list(experiment_to_csv_row({
        "directory": "",
        "experiment": "",
        "label": "",
        "n_runs": 0,
        "run_ids": [],
        "missing_run_ids": [],
    }).keys())

    if out_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pattern_csv(summary: Dict[str, Any], out_path: Path) -> None:
    fieldnames = [
        "directory",
        "experiment",
        "label",
        "pattern_type",
        "n_runs_with_pattern_type",
        "processed_records_mean",
        "total_citations_mean",
        "correct_citations_percent_mean",
        "fully_truthful_explanations_percent_mean",
        "feature_confusion_frequency_percent_mean",
        "time_confusion_frequency_percent_mean",
        "other_error_frequency_percent_mean",
        "avg_true_per_record_mean",
        "avg_false_per_record_mean",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp in summary["experiments"]:
            for pattern_type, metrics in sorted(exp["by_pattern_type"].items()):
                writer.writerow({
                    "directory": exp["directory"],
                    "experiment": exp["experiment"],
                    "label": exp["label"],
                    "pattern_type": pattern_type,
                    "n_runs_with_pattern_type": metrics.get("n_runs_with_pattern_type"),
                    "processed_records_mean": metrics.get("processed_records_mean"),
                    "total_citations_mean": metrics.get("total_citations_mean"),
                    "correct_citations_percent_mean": metrics.get(
                        "correct_citations_percent_mean"
                    ),
                    "fully_truthful_explanations_percent_mean": metrics.get(
                        "fully_truthful_explanations_percent_mean"
                    ),
                    "feature_confusion_frequency_percent_mean": metrics.get(
                        "feature_confusion_frequency_percent_mean"
                    ),
                    "time_confusion_frequency_percent_mean": metrics.get(
                        "time_confusion_frequency_percent_mean"
                    ),
                    "other_error_frequency_percent_mean": metrics.get(
                        "other_error_frequency_percent_mean"
                    ),
                    "avg_true_per_record_mean": metrics.get("avg_true_per_record_mean"),
                    "avg_false_per_record_mean": metrics.get("avg_false_per_record_mean"),
                })


def print_table(summary: Dict[str, Any]) -> None:
    print()
    print("=" * 145)
    print("CITATION VERIFICATION METRICS SUMMARY")
    print("=" * 145)
    print(f"Files discovered: {summary['n_files_discovered']}")
    print(f"Experiments:      {summary['n_experiments']}")
    print(f"Expected runs:    {summary['expected_runs']}")
    print(f"prompt_key:       {summary['prompt_key']}")
    print(f"answer_key:       {summary['answer_key']}")
    print("=" * 145)

    if not summary["experiments"]:
        print("No experiments found.")
        return

    header = (
        f"{'Experiment':<68} "
        f"{'Runs':>4} "
        f"{'Missing':>8} "
        f"{'Records':>8} "
        f"{'Cites':>8} "
        f"{'% Correct':>10} "
        f"{'% Clean':>9} "
        f"{'% Feat':>8} "
        f"{'% Time':>8} "
        f"{'% Other':>8} "
        f"{'Avg T':>7} "
        f"{'Avg F':>7}"
    )

    print(header)
    print("-" * len(header))

    for exp in summary["experiments"]:
        missing = (
            "-"
            if not exp["missing_run_ids"]
            else ",".join(str(x) for x in exp["missing_run_ids"])
        )

        print(
            f"{exp['label']:<68} "
            f"{exp['n_runs']:>4} "
            f"{missing:>8} "
            f"{fmt_count(exp.get('processed_records_mean')):>8} "
            f"{fmt_count(exp.get('total_citations_mean')):>8} "
            f"{fmt_float(exp.get('correct_citations_percent_mean')):>10} "
            f"{fmt_float(exp.get('fully_truthful_explanations_percent_mean')):>9} "
            f"{fmt_float(exp.get('feature_confusion_frequency_percent_mean')):>8} "
            f"{fmt_float(exp.get('time_confusion_frequency_percent_mean')):>8} "
            f"{fmt_float(exp.get('other_error_frequency_percent_mean')):>8} "
            f"{fmt_float(exp.get('avg_true_per_record_mean')):>7} "
            f"{fmt_float(exp.get('avg_false_per_record_mean')):>7}"
        )

    if summary["warnings"]:
        print()
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")


def save_output(summary: Dict[str, Any], out_path: Path) -> None:
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {out_path}")
        return

    write_csv(summary, out_path=out_path)
    print(f"\nSaved CSV report to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate citation verification metrics across JSONL experiment runs."
    )

    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="One or several directories/files to aggregate.",
    )

    parser.add_argument(
        "--prompt-key",
        default="input",
        help="JSON key containing the data context. Default: input.",
    )

    parser.add_argument(
        "--answer-key",
        default="output",
        help="JSON key containing the text to evaluate. Default: output.",
    )

    parser.add_argument(
        "--plain-text",
        action="store_true",
        help="Parse context as 'timestamp: X, feature: Y, ...'.",
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=4,
        help="Context window passed to verify_citations. Default: 4.",
    )

    parser.add_argument(
        "--max-failed-examples",
        type=int,
        default=5,
        help="Max failed examples kept per record. Default: 5.",
    )

    parser.add_argument(
        "--expected-runs",
        type=int,
        default=3,
        help="Expected number of runs per experiment. Default: 3.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern used inside directories. Default: *.jsonl.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search JSONL files recursively inside input directories.",
    )

    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Include JSONL files without final '-N' suffix as single-run experiments.",
    )

    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Stdout format. Default: table.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path. Use .json for full nested report, otherwise CSV.",
    )

    parser.add_argument(
        "--out-patterns",
        type=str,
        default=None,
        help="Optional long CSV with per-pattern-type aggregated metrics.",
    )

    args = parser.parse_args()

    summary = build_summary(args)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    elif args.format == "csv":
        write_csv(summary, out_path=None)
    else:
        print_table(summary)

    if args.out:
        save_output(summary, Path(args.out))

    if args.out_patterns:
        write_pattern_csv(summary, Path(args.out_patterns))
        print(f"\nSaved per-pattern CSV report to {args.out_patterns}")


if __name__ == "__main__":
    main()
