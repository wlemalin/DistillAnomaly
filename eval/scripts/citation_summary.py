#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Citation Verification Metrics Summary

Processes multiple JSONL files containing citation verification results
and produces a consolidated summary table using the exact same computation
logic as the original verification script.

Usage:
    python citation_verification_summary.py results/*.jsonl
    python citation_verification_summary.py --prompt-key context --answer-key output --plain-text results/*.jsonl
    python citation_verification_summary.py --synth --out-citations citations.csv --out-explanations explanations.csv results/*.jsonl
    python citation_verification_summary.py --out-metrics metrics.csv results/*.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
import re
import csv

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records."""
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out

def parse_context_data(prompt_text: str, plain_text: bool = False) -> Dict[int, Dict[str, int]]:
    """
    Reconstructs time series data from the prompt text - exactly like original script.
    
    Default mode:
      - Lines like: 12 : 23
      - Or multi-feature lines: 12 : 23, 1, 0, 5
      - Optional header: "# Columns per line: timestamp : value, moving_average, ..."

    --plain-text mode:
      - Lines like: timestamp: 12, value: 23, moving_average: 1, moving_std: 0
      - Supports 1..N features (whatever appears on each line)
    """
    if not isinstance(prompt_text, str):
        return {}

    def _to_int(raw: str) -> Optional[int]:
        raw = raw.strip()
        try:
            if "." in raw:
                f = float(raw)
                if f.is_integer():
                    return int(f)
                return None
            return int(raw)
        except ValueError:
            return None

    def _norm_key(k: str) -> str:
        k = k.strip().lower()
        aliases = {
            "ts": "timestamp",
            "time": "timestamp",
            "val": "value",
        }
        return aliases.get(k, k)

    if plain_text:
        data_map: Dict[int, Dict[str, int]] = {}
        lines = prompt_text.splitlines()

        pair_re = re.compile(r"([A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*([\-]?\d+(?:\.\d+)?)")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            pairs = pair_re.findall(line)
            if not pairs:
                continue

            row_feats: Dict[str, int] = {}
            ts_val: Optional[int] = None

            for k_raw, v_raw in pairs:
                k = _norm_key(k_raw)
                v = _to_int(v_raw)
                if v is None:
                    continue

                if k == "timestamp":
                    ts_val = v
                else:
                    row_feats[k] = v

            if ts_val is None or not row_feats:
                continue

            data_map[ts_val] = row_feats

        return data_map

    # ---- Default (current) format parsing ---------------------------------
    data_map: Dict[int, Dict[str, int]] = {}

    header_pattern = re.search(r"#\s*Columns per line\s*:\s*timestamp\s*:\s*(.+)", prompt_text)
    if header_pattern:
        col_str = header_pattern.group(1).strip()
        feature_names = [x.strip() for x in col_str.split(",") if x.strip()]
    else:
        feature_names = ["value", "moving_average", "moving_std", "centroid"]

    lines = prompt_text.splitlines()
    row_pattern = re.compile(r"^(\d+)\s*:\s*(.+)$")

    for line in lines:
        line = line.strip()
        m = row_pattern.match(line)
        if not m:
            continue

        ts = int(m.group(1))
        vals_str = m.group(2).strip()

        parts = [p.strip() for p in vals_str.split(",") if p.strip()]
        vals: List[int] = []
        ok = True

        for p in parts:
            v = _to_int(p)
            if v is None:
                ok = False
                break
            vals.append(v)

        if not ok or not vals:
            continue

        row_data: Dict[str, int] = {}
        limit = min(len(feature_names), len(vals))
        for i in range(limit):
            row_data[feature_names[i]] = vals[i]

        data_map[ts] = row_data

    return data_map

def verify_citations(
    text: str,
    data_map: Dict[int, Dict[str, int]],
    context_window: int = 3,
    max_failed_examples: int = 5
) -> Dict[str, Any]:
    """
    Parses text for citations (feature@timestamp:value) and verifies them
    against the provided data_map - exactly like original script.
    """
    if not isinstance(text, str):
        return {
            "has_citations": False,
            "true_count": 0,
            "false_count": 0,
            "score": 0.0,
            "logs": ["FAIL: Input text is not a string."],
            "failed_examples": [],
            "feature_confusion_errors": 0,
            "feature_confusion_share": 0.0,
            "time_imprecision_errors": 0,
            "time_imprecision_share": 0.0,
            "other_errors": 0,
            "other_error_share": 0.0,
        }

    pattern = re.compile(r'([\w\-_]+)@(\d+)\s*:\s*([\-]?\d+)')
    matches = pattern.findall(text)

    FEATURE_ALIASES = {
        "index": "value",
        "values": "value",
        "average": "moving_average",
    }

    def normalize_feature_name(raw: str) -> str:
        k = raw.strip()
        k_lower = k.lower()
        return FEATURE_ALIASES.get(k_lower, k)

    logs: List[str] = []
    true_facts = 0
    false_facts = 0
    failed_examples: List[Dict[str, Any]] = []

    feature_confusion_errors = 0
    time_imprecision_errors = 0
    other_errors = 0

    if not matches:
        return {
            "has_citations": False,
            "true_count": 0,
            "false_count": 0,
            "score": 0.0,
            "logs": ["FAIL: No citations found in format (feature@time:value)."],
            "failed_examples": [],
            "feature_confusion_errors": 0,
            "feature_confusion_share": 0.0,
            "time_imprecision_errors": 0,
            "time_imprecision_share": 0.0,
            "other_errors": 0,
            "other_error_share": 0.0,
        }

    def add_fail_example(example: Dict[str, Any]):
        if len(failed_examples) < max_failed_examples:
            failed_examples.append(example)

    def _feature_confusion_same_ts(ts_: int, cited_feature_: str, cited_val_: int) -> Optional[str]:
        row = data_map.get(ts_)
        if not row:
            return None
        for f_name, f_val in row.items():
            if f_name == cited_feature_:
                continue
            if f_val == cited_val_:
                return f_name
        return None

    def _time_imprecision_pm2(ts_: int, cited_feature_: str, cited_val_: int) -> Optional[int]:
        for delta in (-4, -1, 1, 4):
            t2 = ts_ + delta
            row = data_map.get(t2)
            if not row:
                continue
            if cited_feature_ in row and row[cited_feature_] == cited_val_:
                return t2
        return None

    def _attribute_error_exclusive(ts_: int, cited_feature_: str, cited_val_: int) -> Dict[str, Any]:
        if ts_ in data_map:
            matched_feat = _feature_confusion_same_ts(ts_, cited_feature_, cited_val_)
            if matched_feat is not None:
                return {
                    "category": "feature_confusion_same_ts",
                    "matched_feature": matched_feat,
                }

        matched_ts = _time_imprecision_pm2(ts_, cited_feature_, cited_val_)
        if matched_ts is not None:
            return {
                "category": "time_imprecision_pm2",
                "matched_timestamp": matched_ts,
            }

        return {"category": "other"}

    for feature, ts_str, val_str in matches:
        ts = int(ts_str)
        raw_feature = feature
        feature = normalize_feature_name(feature)

        try:
            val_cited = int(val_str)
        except ValueError:
            false_facts += 1
            other_errors += 1
            msg = f"FAIL: Malformed value '{val_str}' for {feature}@{ts}"
            logs.append(msg)
            add_fail_example({
                "feature": feature,
                "raw_feature": raw_feature,
                "timestamp": ts,
                "cited_value_raw": val_str,
                "reason": "malformed_value",
                "error_attribution": {"category": "other"},
            })
            continue

        # Check Timestamp exists
        if ts not in data_map:
            false_facts += 1
            attribution = _attribute_error_exclusive(ts, feature, val_cited)

            if attribution["category"] == "feature_confusion_same_ts":
                feature_confusion_errors += 1
            elif attribution["category"] == "time_imprecision_pm2":
                time_imprecision_errors += 1
            else:
                other_errors += 1

            logs.append(f"FAIL: {feature}@{ts} - Timestamp out of bounds")
            continue

        # Check Feature Name exists
        if feature not in data_map[ts]:
            false_facts += 1
            attribution = _attribute_error_exclusive(ts, feature, val_cited)

            if attribution["category"] == "feature_confusion_same_ts":
                feature_confusion_errors += 1
            elif attribution["category"] == "time_imprecision_pm2":
                time_imprecision_errors += 1
            else:
                other_errors += 1

            available = list(data_map[ts].keys())
            logs.append(f"FAIL: {feature}@{ts} - Feature not found (Available: {available})")
            continue

        # Check Value Accuracy
        actual_val = data_map[ts][feature]
        if actual_val != val_cited:
            false_facts += 1
            attribution = _attribute_error_exclusive(ts, feature, val_cited)

            if attribution["category"] == "feature_confusion_same_ts":
                feature_confusion_errors += 1
            elif attribution["category"] == "time_imprecision_pm2":
                time_imprecision_errors += 1
            else:
                other_errors += 1

            logs.append(f"FAIL: {feature}@{ts} - Cited {val_cited} vs Actual {actual_val}")
        else:
            true_facts += 1
            logs.append(f"PASS: {feature}@{ts} - Cited {val_cited} matches Actual {actual_val}")

    total_claims = true_facts + false_facts
    score = (true_facts / total_claims) if total_claims > 0 else 0.0

    fc_share = (feature_confusion_errors / false_facts) if false_facts > 0 else 0.0
    ti_share = (time_imprecision_errors / false_facts) if false_facts > 0 else 0.0
    other_share = (other_errors / false_facts) if false_facts > 0 else 0.0

    return {
        "has_citations": True,
        "true_count": true_facts,
        "false_count": false_facts,
        "score": round(score, 2),
        "logs": logs,
        "feature_confusion_errors": feature_confusion_errors,
        "feature_confusion_share": round(fc_share, 3),
        "time_imprecision_errors": time_imprecision_errors,
        "time_imprecision_share": round(ti_share, 3),
        "other_errors": other_errors,
        "other_error_share": round(other_share, 3),
    }

def process_single_jsonl_file(jsonl_path: Path, prompt_key: str, answer_key: str, plain_text: bool) -> Dict[str, Any]:
    """Process a single JSONL file and return verification results."""
    rows = load_jsonl(jsonl_path)
    
    results = []
    for record in rows:
        # Skip records with missing keys (same logic as original script)
        if prompt_key not in record or answer_key not in record:
            continue
            
        prompt_text = record[prompt_key]
        eval_text = record[answer_key]
        p_type = record.get("pattern_type", "unknown")
        
        # Parse data context
        data_map = parse_context_data(prompt_text, plain_text=plain_text)
        
        if not data_map:
            results.append({
                "pattern_type": p_type,
                "has_citations": False,
                "true_count": 0,
                "false_count": 0,
                "score": 0.0,
                "logs": ["FAIL: Could not parse context data"]
            })
            continue
        
        # Verify citations
        eval_result = verify_citations(eval_text, data_map)
        eval_result["pattern_type"] = p_type
        results.append(eval_result)
    
    return {"file": str(jsonl_path.name), "results": results}

def aggregate_stats_by_file(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate stats separately for each file to create matrix format."""
    
    # Stats by file and pattern type
    file_stats = {}
    all_pattern_types = set()
    
    for file_result in all_results:
        filename = file_result["file"]
        file_stats[filename] = defaultdict(lambda: {
            "count": 0, "true": 0, "false": 0, "perfect": 0,
            "feature_confusion_errors": 0, "time_imprecision_errors": 0, "other_errors": 0
        })
        
        for result in file_result["results"]:
            p_type = result.get("pattern_type", "unknown")
            all_pattern_types.add(p_type)
            
            if result.get("has_citations", False):
                file_stats[filename][p_type]["count"] += 1
                if "true_count" in result:
                    file_stats[filename][p_type]["true"] += result["true_count"]
                    file_stats[filename][p_type]["false"] += result["false_count"]
                    file_stats[filename][p_type]["feature_confusion_errors"] += result.get("feature_confusion_errors", 0)
                    file_stats[filename][p_type]["time_imprecision_errors"] += result.get("time_imprecision_errors", 0)
                    file_stats[filename][p_type]["other_errors"] += result.get("other_errors", 0)
                    
                    if result["false_count"] == 0:
                        file_stats[filename][p_type]["perfect"] += 1
    
    # Convert to regular dict and sort
    sorted_pattern_types = sorted(all_pattern_types)
    clean_file_stats = {}
    for filename, stats in file_stats.items():
        clean_file_stats[filename] = {pt: dict(stats[pt]) for pt in sorted_pattern_types}
    
    return {
        "file_stats": clean_file_stats,
        "pattern_types": sorted_pattern_types,
        "total_files": len(all_results)
    }

def save_citations_matrix_csv(file_stats: Dict[str, Dict[str, Dict[str, int]]], pattern_types: List[str], output_path: str):
    """Save % Correct Citations matrix as CSV (Pattern Type vs Files)."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header with file names
        header = ['Pattern Type'] + list(file_stats.keys())
        writer.writerow(header)
        
        # Data rows for each pattern type
        for p_type in pattern_types:
            row = [p_type]
            for filename in file_stats.keys():
                stats = file_stats[filename].get(p_type, {"count": 0, "true": 0, "false": 0})
                count = stats["count"]
                if count > 0:
                    total_facts = stats["true"] + stats["false"]
                    accuracy = (stats["true"] / total_facts * 100) if total_facts > 0 else 0.0
                    row.append(f"{accuracy:.1f}")
                else:
                    row.append("NA")
            writer.writerow(row)
        
        # Overall row
        overall_row = ['OVERALL']
        for filename in file_stats.keys():
            file_total_true = sum(stats["true"] for stats in file_stats[filename].values())
            file_total_false = sum(stats["false"] for stats in file_stats[filename].values())
            file_total_all = file_total_true + file_total_false
            file_overall_accuracy = (file_total_true / file_total_all * 100) if file_total_all > 0 else 0.0
            overall_row.append(f"{file_overall_accuracy:.1f}")
        writer.writerow(overall_row)

def save_explanations_matrix_csv(file_stats: Dict[str, Dict[str, Dict[str, int]]], pattern_types: List[str], output_path: str):
    """Save % Truthful Explanations matrix as CSV (Pattern Type vs Files)."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header with file names
        header = ['Pattern Type'] + list(file_stats.keys())
        writer.writerow(header)
        
        # Data rows for each pattern type
        for p_type in pattern_types:
            row = [p_type]
            for filename in file_stats.keys():
                stats = file_stats[filename].get(p_type, {"count": 0, "perfect": 0})
                count = stats["count"]
                if count > 0:
                    pct_truthful = (stats["perfect"] / count * 100)
                    row.append(f"{pct_truthful:.1f}")
                else:
                    row.append("NA")
            writer.writerow(row)
        
        # Overall row
        overall_row = ['OVERALL']
        for filename in file_stats.keys():
            file_total_perfect = sum(stats["perfect"] for stats in file_stats[filename].values())
            file_total_records = sum(stats["count"] for stats in file_stats[filename].values())
            file_overall_truthful = (file_total_perfect / file_total_records * 100) if file_total_records > 0 else 0.0
            overall_row.append(f"{file_overall_truthful:.1f}")
        writer.writerow(overall_row)

def save_non_synth_metrics_csv(file_stats: Dict[str, Dict[str, Dict[str, int]]], pattern_types: List[str], output_path: str):
    """Save non-synth mode metrics matrix as CSV (Metric vs Files) using actual file names."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Calculate metrics for each file
        file_metrics = {}
        for filename in file_stats.keys():
            total_true = sum(stats["true"] for stats in file_stats[filename].values())
            total_false = sum(stats["false"] for stats in file_stats[filename].values())
            total_perfect = sum(stats["perfect"] for stats in file_stats[filename].values())
            total_records = sum(stats["count"] for stats in file_stats[filename].values())
            total_all = total_true + total_false
            
            total_feature_confusion = sum(stats["feature_confusion_errors"] for stats in file_stats[filename].values())
            total_time_imprecision = sum(stats["time_imprecision_errors"] for stats in file_stats[filename].values())
            total_other_errors = sum(stats["other_errors"] for stats in file_stats[filename].values())
            total_errors = total_feature_confusion + total_time_imprecision + total_other_errors
            
            file_metrics[filename] = {
                "correct_citations": (total_true / total_all * 100) if total_all > 0 else 0.0,
                "truthful_explanations": (total_perfect / total_records * 100) if total_records > 0 else 0.0,
                "feature_confusion": (total_feature_confusion / total_errors * 100) if total_errors > 0 else 0.0,
                "temporal_confusion": (total_time_imprecision / total_errors * 100) if total_errors > 0 else 0.0
            }
        
        # Header with actual file names (not TS1, TS2, etc.)
        header = ['Metric'] + list(file_stats.keys())
        writer.writerow(header)
        
        # Metric rows
        metrics = [
            ('% Correct citations', 'correct_citations'),
            ('% Fully truthful explanations', 'truthful_explanations'),
            ('% Errors - feature confusion', 'feature_confusion'),
            ('% Errors - temporal confusion', 'temporal_confusion')
        ]
        
        for metric_name, metric_key in metrics:
            row = [metric_name]
            for filename in file_stats.keys():
                row.append(f"{file_metrics[filename][metric_key]:.1f}")
            writer.writerow(row)

def print_summary_table(stats: Dict[str, Dict[str, int]], total_records: int):
    """Print summary table exactly like original script."""
    width = 85
    print("\n" + "="*width)
    print(f" EVALUATION SUMMARY (Total Records: {total_records})")
    print("="*width)

    print(f"{'PATTERN TYPE':<20} | {'AVG TRUE':<8} | {'AVG FALSE':<9} | {'ACCURACY':<8} | {'% CLEAN':<7}")
    print("-" * width)

    grand_true = 0
    grand_false = 0
    grand_perfect = 0

    for p_type in sorted(stats.keys()):
        s = stats[p_type]
        count = s["count"]
        if count == 0:
            continue

        avg_true = s["true"] / count
        avg_false = s["false"] / count

        total_facts = s["true"] + s["false"]
        accuracy = (s["true"] / total_facts * 100) if total_facts > 0 else 0.0

        pct_perfect = (s["perfect"] / count * 100)

        grand_true += s["true"]
        grand_false += s["false"]
        grand_perfect += s["perfect"]

        print(f"{p_type:<20} | {avg_true:<8.2f} | {avg_false:<9.2f} | {accuracy:>6.1f}%  | {pct_perfect:>6.1f}%")

    print("-" * width)

    if total_records > 0:
        g_avg_true = grand_true / total_records
        g_avg_false = grand_false / total_records
        g_total = grand_true + grand_false
        g_acc = (grand_true / g_total * 100) if g_total > 0 else 0.0
        g_pct_perf = (grand_perfect / total_records * 100)

        print(f"{'OVERALL':<20} | {g_avg_true:<8.2f} | {g_avg_false:<9.2f} | {g_acc:>6.1f}%  | {g_pct_perf:>6.1f}%")
    print("="*width + "\n")

def print_error_attribution_summary(stats: Dict[str, Dict[str, int]], total_records: int):
    """Print error attribution summary exactly like original script."""
    width = 85
    print("\n" + "=" * width)
    print(f" ERROR ATTRIBUTION (among FALSE citations) (Total Records: {total_records})")
    print("=" * width)

    print(f"{'PATTERN TYPE':<20} | {'FALSE':<6} | {'% FEAT CONF':<11} | {'% TIME ±2':<10} | {'% OTHER':<7}")
    print("-" * width)

    g_false = 0
    g_fc = 0
    g_ti = 0
    g_other = 0

    for p_type in sorted(stats.keys()):
        s = stats[p_type]
        f = s.get("false", 0)

        fc = s.get("feature_confusion_errors", 0)
        ti = s.get("time_imprecision_errors", 0)
        oth = s.get("other_errors", 0)

        if f > 0:
            p_fc = 100 * fc / f
            p_ti = 100 * ti / f
            p_oth = 100 * oth / f
        else:
            p_fc = p_ti = p_oth = 0.0

        g_false += f
        g_fc += fc
        g_ti += ti
        g_other += oth

        print(f"{p_type:<20} | {f:<6d} | {p_fc:>9.1f}%  | {p_ti:>8.1f}%  | {p_oth:>6.1f}%")

    print("-" * width)

    if g_false > 0:
        g_p_fc = 100 * g_fc / g_false
        g_p_ti = 100 * g_ti / g_false
        g_p_oth = 100 * g_other / g_false
    else:
        g_p_fc = g_p_ti = g_p_oth = 0.0

    print(f"{'OVERALL':<20} | {g_false:<6d} | {g_p_fc:>9.1f}%  | {g_p_ti:>8.1f}%  | {g_p_oth:>6.1f}%")
    print("=" * width + "\n")

def print_synth_summary(stats: Dict[str, Dict[str, int]], total_records: int):
    """Print synth mode summary with pattern type breakdown."""
    print(f"\n{'='*100}")
    print(f"CITATION VERIFICATION SUMMARY - SYNTH MODE (Total Records: {total_records})")
    print(f"{'='*100}")
    
    # Get all pattern types
    pattern_types = sorted(stats.keys())
    
    # Print header
    header = f"{'Pattern Type':<20} | {'Records':<8} | {'Avg True':<8} | {'Avg False':<9} | {'Accuracy':<8} | {'% Clean':<7}"
    print(header)
    print("-" * 85)
    
    # Print each pattern type
    for p_type in pattern_types:
        s = stats[p_type]
        count = s["count"]
        if count == 0:
            continue

        avg_true = s["true"] / count
        avg_false = s["false"] / count
        total_facts = s["true"] + s["false"]
        accuracy = (s["true"] / total_facts * 100) if total_facts > 0 else 0.0
        pct_perfect = (s["perfect"] / count * 100)

        print(f"{p_type:<20} | {count:<8d} | {avg_true:<8.2f} | {avg_false:<9.2f} | {accuracy:>6.1f}%  | {pct_perfect:>6.1f}%")
    
    print("-" * 85)
    
    # Overall totals
    if total_records > 0:
        grand_true = sum(s["true"] for s in stats.values())
        grand_false = sum(s["false"] for s in stats.values())
        grand_perfect = sum(s["perfect"] for s in stats.values())
        
        g_avg_true = grand_true / total_records
        g_avg_false = grand_false / total_records
        g_total = grand_true + grand_false
        g_acc = (grand_true / g_total * 100) if g_total > 0 else 0.0
        g_pct_perf = (grand_perfect / total_records * 100)

        print(f"{'OVERALL':<20} | {total_records:<8d} | {g_avg_true:<8.2f} | {g_avg_false:<9.2f} | {g_acc:>6.1f}%  | {g_pct_perf:>6.1f}%")
    
    print("="*100)

def main():
    parser = argparse.ArgumentParser(
        description="Citation Verification Metrics Summary - Processes multiple JSONL files and summarizes citation verification metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python citation_verification_summary.py results/*.jsonl
  python citation_verification_summary.py --prompt-key context --answer-key output --plain-text results/*.jsonl
  python citation_verification_summary.py --synth --out-citations citations.csv --out-explanations explanations.csv results/*.jsonl
  python citation_verification_summary.py --out-metrics metrics.csv results/*.jsonl
        """
    )
    
    parser.add_argument("jsonl_files", nargs="+", type=str, 
                       help="JSONL files to process (can use wildcards)")
    parser.add_argument("--prompt-key", default="input",
                       help="JSON key containing the data context (default: 'input')")
    parser.add_argument("--answer-key", default="output", 
                       help="JSON key containing the text to evaluate (default: 'input')")
    parser.add_argument("--plain-text", action="store_true",
                       help="Parse context as 'timestamp: X, feature: Y, ...' instead of 'ts : v1, v2, ...'")
    parser.add_argument("--synth", action="store_true",
                       help="Enable synthetic data mode (show breakdown by pattern type)")
    parser.add_argument("--out-citations", type=str, default=None,
                       help="Save Correct Citations matrix as CSV to specified file (synth mode only)")
    parser.add_argument("--out-explanations", type=str, default=None,
                       help="Save Truthful Explanations matrix as CSV to specified file (synth mode only)")
    parser.add_argument("--out-metrics", type=str, default=None,
                       help="Save metrics matrix as CSV to specified file (non-synth mode only)")
    
    args = parser.parse_args()
    
    if not args.jsonl_files:
        print("Error: No JSONL files provided", file=sys.stderr)
        sys.exit(1)
    
    # Validate CSV arguments
    if args.synth and (args.out_citations or args.out_explanations):
        if not args.out_citations or not args.out_explanations:
            print("Error: When using --synth mode with CSV export, both --out-citations and --out-explanations must be specified", file=sys.stderr)
            sys.exit(1)
    
    if not args.synth and args.out_metrics:
        # Non-synth mode with metrics export is valid
        pass
    elif not args.synth and (args.out_citations or args.out_explanations):
        print("Error: --out-citations and --out-explanations are only available in synth mode", file=sys.stderr)
        sys.exit(1)
    
    # Convert file paths to Path objects and filter existing files
    jsonl_paths = []
    for file_pattern in args.jsonl_files:
        path = Path(file_pattern)
        if path.exists():
            jsonl_paths.append(path)
        else:
            # Try glob pattern
            import glob
            matches = glob.glob(file_pattern)
            for match in matches:
                match_path = Path(match)
                if match_path.exists():
                    jsonl_paths.append(match_path)
    
    if not jsonl_paths:
        print("Error: No valid JSONL files found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing {len(jsonl_paths)} JSONL files...")
    print(f"Using prompt_key='{args.prompt_key}', answer_key='{args.answer_key}'")
    if args.plain_text:
        print("Using plain-text parsing mode")
    
    # Process each file and collect results
    all_file_results = []
    for jsonl_path in jsonl_paths:
        print(f"  Processing: {jsonl_path.name}")
        file_result = process_single_jsonl_file(jsonl_path, args.prompt_key, args.answer_key, args.plain_text)
        all_file_results.append(file_result)
    
    # Aggregate stats by file for matrix format
    aggregated = aggregate_stats_by_file(all_file_results)
    file_stats = aggregated["file_stats"]
    pattern_types = aggregated["pattern_types"]
    
    if not file_stats or not pattern_types:
        print("Error: No records with valid citations found", file=sys.stderr)
        sys.exit(1)
    
    # Print summaries
    if args.synth:
        # For display, aggregate all files together
        total_records = sum(sum(stats["count"] for stats in file_stats[filename].values()) for filename in file_stats.keys())
        aggregated_stats = {}
        for p_type in pattern_types:
            aggregated_stats[p_type] = {
                "count": sum(file_stats[filename].get(p_type, {}).get("count", 0) for filename in file_stats.keys()),
                "true": sum(file_stats[filename].get(p_type, {}).get("true", 0) for filename in file_stats.keys()),
                "false": sum(file_stats[filename].get(p_type, {}).get("false", 0) for filename in file_stats.keys()),
                "perfect": sum(file_stats[filename].get(p_type, {}).get("perfect", 0) for filename in file_stats.keys()),
                "feature_confusion_errors": sum(file_stats[filename].get(p_type, {}).get("feature_confusion_errors", 0) for filename in file_stats.keys()),
                "time_imprecision_errors": sum(file_stats[filename].get(p_type, {}).get("time_imprecision_errors", 0) for filename in file_stats.keys()),
                "other_errors": sum(file_stats[filename].get(p_type, {}).get("other_errors", 0) for filename in file_stats.keys()),
            }
        print_synth_summary(aggregated_stats, total_records)
    else:
        # For non-synth, show simple aggregated summary
        total_records = sum(sum(stats["count"] for stats in file_stats[filename].values()) for filename in file_stats.keys())
        aggregated_stats = {}
        for p_type in pattern_types:
            aggregated_stats[p_type] = {
                "count": sum(file_stats[filename].get(p_type, {}).get("count", 0) for filename in file_stats.keys()),
                "true": sum(file_stats[filename].get(p_type, {}).get("true", 0) for filename in file_stats.keys()),
                "false": sum(file_stats[filename].get(p_type, {}).get("false", 0) for filename in file_stats.keys()),
                "perfect": sum(file_stats[filename].get(p_type, {}).get("perfect", 0) for filename in file_stats.keys()),
                "feature_confusion_errors": sum(file_stats[filename].get(p_type, {}).get("feature_confusion_errors", 0) for filename in file_stats.keys()),
                "time_imprecision_errors": sum(file_stats[filename].get(p_type, {}).get("time_imprecision_errors", 0) for filename in file_stats.keys()),
                "other_errors": sum(file_stats[filename].get(p_type, {}).get("other_errors", 0) for filename in file_stats.keys()),
            }
        print_summary_table(aggregated_stats, total_records)
        print_error_attribution_summary(aggregated_stats, total_records)
    
    # Save CSV files if requested
    if args.synth and args.out_citations and args.out_explanations:
        save_citations_matrix_csv(file_stats, pattern_types, args.out_citations)
        save_explanations_matrix_csv(file_stats, pattern_types, args.out_explanations)
        print(f"\nSaved % Correct Citations matrix to: {args.out_citations}")
        print(f"Saved % Truthful Explanations matrix to: {args.out_explanations}")
    elif not args.synth and args.out_metrics:
        save_non_synth_metrics_csv(file_stats, pattern_types, args.out_metrics)
        print(f"\nSaved metrics matrix to: {args.out_metrics}")

if __name__ == "__main__":
    main()
