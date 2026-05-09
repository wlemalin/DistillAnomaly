#!/usr/bin/env python3
import json
import re
import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Any, Optional


# ──────────────────────────────────────────────────────────────────────
# 1. CORE VERIFICATION LOGIC
# ──────────────────────────────────────────────────────────────────────

def print_error_attribution_summary(stats: Dict[str, Dict[str, int]], total_records: int):
    """
    Prints citation-level global frequencies of failure modes.

    Interpretation:
      - % FEAT CONF = feature_confusion_errors / total_citations
      - % TIME CONF = time_imprecision_errors / total_citations
      - % OTHER     = other_errors / total_citations

    where:

      total_citations = true citations + false citations

    So these are NOT record-level frequencies and NOT shares among false citations.
    """
    width = 105
    print("\n" + "=" * width)
    print(f" ERROR ATTRIBUTION FREQUENCY (citation-level) (Total Records: {total_records})")
    print("=" * width)

    print(
        f"{'PATTERN TYPE':<20} | "
        f"{'CITES':<7} | "
        f"{'TRUE':<6} | "
        f"{'FALSE':<6} | "
        f"{'% FEAT CONF':<12} | "
        f"{'% TIME CONF':<12} | "
        f"{'% OTHER':<9} | "
        f"{'FEAT ERR':<8} | "
        f"{'TIME ERR':<8} | "
        f"{'OTHER ERR':<9}"
    )
    print("-" * width)

    g_true = 0
    g_false = 0
    g_fc_errors = 0
    g_ti_errors = 0
    g_other_errors = 0

    for p_type in sorted(stats.keys()):
        s = stats[p_type]

        true_count = s.get("true", 0)
        false_count = s.get("false", 0)
        total_citations = true_count + false_count

        fc_errors = s.get("feature_confusion_errors", 0)
        ti_errors = s.get("time_imprecision_errors", 0)
        other_errors = s.get("other_errors", 0)

        if total_citations > 0:
            p_fc = 100 * fc_errors / total_citations
            p_ti = 100 * ti_errors / total_citations
            p_other = 100 * other_errors / total_citations
        else:
            p_fc = p_ti = p_other = 0.0

        g_true += true_count
        g_false += false_count
        g_fc_errors += fc_errors
        g_ti_errors += ti_errors
        g_other_errors += other_errors

        print(
            f"{p_type:<20} | "
            f"{total_citations:<7d} | "
            f"{true_count:<6d} | "
            f"{false_count:<6d} | "
            f"{p_fc:>10.1f}%  | "
            f"{p_ti:>10.1f}%  | "
            f"{p_other:>7.1f}%  | "
            f"{fc_errors:<8d} | "
            f"{ti_errors:<8d} | "
            f"{other_errors:<9d}"
        )

    print("-" * width)

    g_total_citations = g_true + g_false

    if g_total_citations > 0:
        g_p_fc = 100 * g_fc_errors / g_total_citations
        g_p_ti = 100 * g_ti_errors / g_total_citations
        g_p_other = 100 * g_other_errors / g_total_citations
    else:
        g_p_fc = g_p_ti = g_p_other = 0.0

    print(
        f"{'OVERALL':<20} | "
        f"{g_total_citations:<7d} | "
        f"{g_true:<6d} | "
        f"{g_false:<6d} | "
        f"{g_p_fc:>10.1f}%  | "
        f"{g_p_ti:>10.1f}%  | "
        f"{g_p_other:>7.1f}%  | "
        f"{g_fc_errors:<8d} | "
        f"{g_ti_errors:<8d} | "
        f"{g_other_errors:<9d}"
    )

    print("=" * width + "\n")


def build_context_window(
    data_map: Dict[int, Dict[str, int]],
    ts: int,
    window: int = 3,
) -> Dict[str, Any]:
    """
    Retourne un extrait de la série autour de ts.

    Si ts est hors bornes, on clamp vers le timestamp existant le plus proche.
    """
    if not data_map:
        return {"requested_ts": ts, "center_ts": None, "rows": []}

    available_ts = sorted(data_map.keys())
    min_ts, max_ts = available_ts[0], available_ts[-1]

    if ts <= min_ts:
        center = min_ts
    elif ts >= max_ts:
        center = max_ts
    else:
        center = ts

    rows = []
    for t in range(center - window, center + window + 1):
        if t in data_map:
            row = {"timestamp": t, **data_map[t]}
        else:
            row = {"timestamp": t, "_missing": True}
        rows.append(row)

    return {
        "requested_ts": ts,
        "center_ts": center,
        "bounds": [min_ts, max_ts],
        "rows": rows,
    }


def format_context_for_console(ctx: Dict[str, Any]) -> str:
    rows = ctx.get("rows", [])
    if not rows:
        return "  (aucune donnée de contexte)\n"

    feats = set()
    for r in rows:
        for k in r.keys():
            if k not in ("timestamp", "_missing"):
                feats.add(k)

    feat_list = sorted(feats)

    out = []
    out.append(
        f"  Contexte (requested_ts={ctx.get('requested_ts')}, "
        f"center_ts={ctx.get('center_ts')}, bounds={ctx.get('bounds')}):"
    )
    out.append("  " + "-" * 78)

    header = ["timestamp"] + feat_list + ["_missing"]
    out.append("  " + " | ".join(f"{h:>14}" for h in header))
    out.append("  " + "-" * 78)

    for r in rows:
        line_parts = [f"{r.get('timestamp'):>14}"]

        for f in feat_list:
            v = r.get(f, "")
            if isinstance(v, int):
                line_parts.append(f"{v:>14d}")
            else:
                line_parts.append(f"{str(v):>14}")

        line_parts.append(f"{str(r.get('_missing', False)):>14}")
        out.append("  " + " | ".join(line_parts))

    out.append("  " + "-" * 78)
    return "\n".join(out) + "\n"


def verify_citations(
    text: str,
    data_map: Dict[int, Dict[str, int]],
    context_window: int = 3,
    max_failed_examples: int = 5,
) -> Dict[str, Any]:
    """
    Parses text for citations of the form:

        feature@timestamp:value

    and verifies them against data_map.

    Error attribution is exclusive:
      1. feature_confusion_same_ts
      2. time_imprecision_pm2
      3. other

    Per-record error frequencies are citation-level:

      feature_confusion_frequency = feature_confusion_errors / total_citations
      time_imprecision_frequency  = time_imprecision_errors / total_citations
      other_error_frequency       = other_errors / total_citations
    """
    if not isinstance(text, str):
        return {
            "has_citations": False,
            "true_count": 0,
            "false_count": 0,
            "total_citations": 0,
            "score": 0.0,
            "logs": ["FAIL: Input text is not a string."],
            "failed_examples": [],
            "feature_confusion_errors": 0,
            "feature_confusion_frequency": 0.0,
            "time_imprecision_errors": 0,
            "time_imprecision_frequency": 0.0,
            "other_errors": 0,
            "other_error_frequency": 0.0,
        }

    pattern = re.compile(r"([\w\-_]+)@(\d+)\s*:\s*([\-]?\d+)")
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
            "total_citations": 0,
            "score": 0.0,
            "logs": ["FAIL: No citations found in format (feature@time:value)."],
            "failed_examples": [],
            "feature_confusion_errors": 0,
            "feature_confusion_frequency": 0.0,
            "time_imprecision_errors": 0,
            "time_imprecision_frequency": 0.0,
            "other_errors": 0,
            "other_error_frequency": 0.0,
        }

    def add_fail_example(example: Dict[str, Any]):
        if len(failed_examples) < max_failed_examples:
            failed_examples.append(example)

    def _feature_confusion_same_ts(
        ts_: int,
        cited_feature_: str,
        cited_val_: int,
    ) -> Optional[str]:
        row = data_map.get(ts_)
        if not row:
            return None

        for f_name, f_val in row.items():
            if f_name == cited_feature_:
                continue
            if f_val == cited_val_:
                return f_name

        return None

    def _time_imprecision_pm2(
        ts_: int,
        cited_feature_: str,
        cited_val_: int,
    ) -> Optional[int]:
        """
        Kept as in your original script.

        Note: despite the name pm2, the checked deltas are:
            -4, -1, +1, +4
        """
        for delta in (-4, -1, 1, 4):
            t2 = ts_ + delta
            row = data_map.get(t2)

            if not row:
                continue

            if cited_feature_ in row and row[cited_feature_] == cited_val_:
                return t2

        return None

    def _attribute_error_exclusive(
        ts_: int,
        cited_feature_: str,
        cited_val_: int,
    ) -> Dict[str, Any]:
        if ts_ in data_map:
            matched_feat = _feature_confusion_same_ts(
                ts_,
                cited_feature_,
                cited_val_,
            )

            if matched_feat is not None:
                return {
                    "category": "feature_confusion_same_ts",
                    "matched_feature": matched_feat,
                }

        matched_ts = _time_imprecision_pm2(
            ts_,
            cited_feature_,
            cited_val_,
        )

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

            logs.append(f"FAIL: Malformed value '{val_str}' for {feature}@{ts}")

            add_fail_example({
                "feature": feature,
                "raw_feature": raw_feature,
                "timestamp": ts,
                "cited_value_raw": val_str,
                "reason": "malformed_value",
                "error_attribution": {"category": "other"},
                "context": build_context_window(
                    data_map,
                    ts,
                    window=context_window,
                ),
            })

            continue

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

            add_fail_example({
                "feature": feature,
                "raw_feature": raw_feature,
                "timestamp": ts,
                "cited_value": val_cited,
                "reason": "timestamp_missing",
                "error_attribution": attribution,
                "context": build_context_window(
                    data_map,
                    ts,
                    window=context_window,
                ),
            })

            continue

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

            logs.append(
                f"FAIL: {feature}@{ts} - Feature not found "
                f"(Available: {available})"
            )

            add_fail_example({
                "feature": feature,
                "raw_feature": raw_feature,
                "timestamp": ts,
                "cited_value": val_cited,
                "reason": "feature_missing",
                "available_features_at_ts": available,
                "error_attribution": attribution,
                "context": build_context_window(
                    data_map,
                    ts,
                    window=context_window,
                ),
            })

            continue

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

            logs.append(
                f"FAIL: {feature}@{ts} - Cited {val_cited} vs Actual {actual_val}"
            )

            add_fail_example({
                "feature": feature,
                "timestamp": ts,
                "cited_value": val_cited,
                "actual_value": actual_val,
                "reason": "value_mismatch",
                "error_attribution": attribution,
                "context": build_context_window(
                    data_map,
                    ts,
                    window=context_window,
                ),
            })

        else:
            true_facts += 1
            logs.append(
                f"PASS: {feature}@{ts} - Cited {val_cited} matches Actual {actual_val}"
            )

    total_claims = true_facts + false_facts
    score = true_facts / total_claims if total_claims > 0 else 0.0

    feature_confusion_frequency = (
        feature_confusion_errors / total_claims if total_claims > 0 else 0.0
    )
    time_imprecision_frequency = (
        time_imprecision_errors / total_claims if total_claims > 0 else 0.0
    )
    other_error_frequency = (
        other_errors / total_claims if total_claims > 0 else 0.0
    )

    return {
        "has_citations": True,
        "true_count": true_facts,
        "false_count": false_facts,
        "total_citations": total_claims,
        "score": round(score, 2),
        "logs": logs,
        "failed_examples": failed_examples,

        "feature_confusion_errors": feature_confusion_errors,
        "feature_confusion_frequency": round(feature_confusion_frequency, 3),

        "time_imprecision_errors": time_imprecision_errors,
        "time_imprecision_frequency": round(time_imprecision_frequency, 3),

        "other_errors": other_errors,
        "other_error_frequency": round(other_error_frequency, 3),
    }


# ──────────────────────────────────────────────────────────────────────
# 2. DATA PARSING
# ──────────────────────────────────────────────────────────────────────

def parse_context_data(prompt_text: str, plain_text: bool = False) -> Dict[int, Dict[str, int]]:
    """
    Reconstructs time series data from the prompt text.

    Default mode:
      - Lines like: 12 : 23
      - Or multi-feature lines: 12 : 23, 1, 0, 5
      - Optional header: "# Columns per line: timestamp : value, moving_average, ..."

    --plain-text mode:
      - Lines like: timestamp: 12, value: 23, moving_average: 1, moving_std: 0
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

        pair_re = re.compile(
            r"([A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*([\-]?\d+(?:\.\d+)?)"
        )

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

    data_map: Dict[int, Dict[str, int]] = {}

    header_pattern = re.search(
        r"#\s*Columns per line\s*:\s*timestamp\s*:\s*(.+)",
        prompt_text,
    )

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


# ──────────────────────────────────────────────────────────────────────
# 3. REPORTING HELPERS
# ──────────────────────────────────────────────────────────────────────

def print_summary(stats: Dict[str, Dict[str, int]], total_records: int):
    width = 85
    print("\n" + "=" * width)
    print(f" EVALUATION SUMMARY (Total Records: {total_records})")
    print("=" * width)

    print(
        f"{'PATTERN TYPE':<20} | "
        f"{'AVG TRUE':<8} | "
        f"{'AVG FALSE':<9} | "
        f"{'ACCURACY':<8} | "
        f"{'% CLEAN':<7}"
    )
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

        pct_perfect = s["perfect"] / count * 100

        grand_true += s["true"]
        grand_false += s["false"]
        grand_perfect += s["perfect"]

        print(
            f"{p_type:<20} | "
            f"{avg_true:<8.2f} | "
            f"{avg_false:<9.2f} | "
            f"{accuracy:>6.1f}%  | "
            f"{pct_perfect:>6.1f}%"
        )

    print("-" * width)

    if total_records > 0:
        g_avg_true = grand_true / total_records
        g_avg_false = grand_false / total_records
        g_total = grand_true + grand_false
        g_acc = (grand_true / g_total * 100) if g_total > 0 else 0.0
        g_pct_perf = grand_perfect / total_records * 100

        print(
            f"{'OVERALL':<20} | "
            f"{g_avg_true:<8.2f} | "
            f"{g_avg_false:<9.2f} | "
            f"{g_acc:>6.1f}%  | "
            f"{g_pct_perf:>6.1f}%"
        )

    print("=" * width + "\n")


# ──────────────────────────────────────────────────────────────────────
# 4. MAIN RUNNER
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM citations in JSONL files."
    )

    parser.add_argument(
        "input_file",
        help="Path to input .jsonl file",
    )

    parser.add_argument(
        "--prompt-key",
        default="input",
        help="JSON key containing the data context. Default: input.",
    )

    parser.add_argument(
        "--answer-key",
        default="input",
        help="JSON key containing the text to evaluate. Default: input.",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="evaluated_output.jsonl",
        help="Output file path. Default: evaluated_output.jsonl.",
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=4,
        help="Nombre de timestamps avant/après à afficher autour d'une citation.",
    )

    parser.add_argument(
        "--show-fails",
        type=int,
        default=0,
        help="Affiche jusqu'à N citations fausses par record. 0 = désactivé.",
    )

    parser.add_argument(
        "--max-records-with-fails",
        type=int,
        default=30,
        help="Limite le nombre de records pour lesquels on imprime des erreurs.",
    )

    parser.add_argument(
        "--plain-text",
        action="store_true",
        help="Parse context as 'timestamp: X, feature: Y, ...' instead of 'ts : v1, v2, ...'.",
    )

    args = parser.parse_args()

    stats = defaultdict(lambda: {
        "count": 0,
        "true": 0,
        "false": 0,
        "perfect": 0,
        "feature_confusion_errors": 0,
        "time_imprecision_errors": 0,
        "other_errors": 0,
    })

    processed_count = 0
    printed_fail_records = 0

    print(f"Reading from: {args.input_file}")
    print(f"Writing to:   {args.output}")
    print(f"evaluating:   record['{args.answer_key}']")
    print(f"against:      record['{args.prompt_key}']")

    try:
        with open(args.input_file, "r", encoding="utf-8") as fin, \
             open(args.output, "w", encoding="utf-8") as fout:

            for line in fin:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if args.prompt_key not in record or args.answer_key not in record:
                    record["eval_error"] = "Keys missing"
                    fout.write(json.dumps(record) + "\n")
                    continue

                prompt_text = record[args.prompt_key]
                eval_text = record[args.answer_key]
                p_type = record.get("pattern_type", "unknown")

                data_map = parse_context_data(
                    prompt_text,
                    plain_text=args.plain_text,
                )

                if not data_map:
                    eval_result = {
                        "has_citations": False,
                        "true_count": 0,
                        "false_count": 0,
                        "total_citations": 0,
                        "score": 0.0,
                        "logs": ["FAIL: Could not parse context data"],
                        "failed_examples": [],
                        "feature_confusion_errors": 0,
                        "feature_confusion_frequency": 0.0,
                        "time_imprecision_errors": 0,
                        "time_imprecision_frequency": 0.0,
                        "other_errors": 0,
                        "other_error_frequency": 0.0,
                    }
                else:
                    eval_result = verify_citations(
                        eval_text,
                        data_map,
                        context_window=args.context_window,
                        max_failed_examples=max(args.show_fails, 5),
                    )

                if args.show_fails > 0 and eval_result.get("false_count", 0) > 0:
                    if printed_fail_records < args.max_records_with_fails:
                        printed_fail_records += 1
                        rec_id = record.get("id", record.get("uid", "NA"))

                        print("\n" + "!" * 90)
                        print(f"RECORD FAIL (pattern_type={p_type}, id={rec_id})")
                        print(
                            f"false_count={eval_result.get('false_count')} | "
                            f"true_count={eval_result.get('true_count')} | "
                            f"score={eval_result.get('score')}"
                        )
                        print("Quelques citations fausses :")

                        failed = eval_result.get("failed_examples", [])[:args.show_fails]

                        for i, ex in enumerate(failed, start=1):
                            attrib = ex.get("error_attribution", {})
                            cat = attrib.get("category")

                            print(
                                f"\n  [{i}] "
                                f"reason={ex.get('reason')}  "
                                f"feature={ex.get('feature')}  "
                                f"ts={ex.get('timestamp')}  "
                                f"attrib={cat}"
                            )

                            if "cited_value" in ex:
                                print(f"      cited={ex.get('cited_value')}")

                            if "actual_value" in ex:
                                print(f"      actual={ex.get('actual_value')}")

                            if ex.get("reason") == "feature_missing":
                                print(
                                    f"      available_features_at_ts="
                                    f"{ex.get('available_features_at_ts')}"
                                )

                            if cat == "feature_confusion_same_ts":
                                print(
                                    f"      matched_feature="
                                    f"{attrib.get('matched_feature')}"
                                )

                            if cat == "time_imprecision_pm2":
                                print(
                                    f"      matched_timestamp="
                                    f"{attrib.get('matched_timestamp')}"
                                )

                            print(format_context_for_console(ex.get("context", {})))

                        print("!" * 90 + "\n")

                if "true_count" in eval_result:
                    stats[p_type]["count"] += 1
                    stats[p_type]["true"] += eval_result.get("true_count", 0)
                    stats[p_type]["false"] += eval_result.get("false_count", 0)

                    stats[p_type]["feature_confusion_errors"] += eval_result.get(
                        "feature_confusion_errors",
                        0,
                    )
                    stats[p_type]["time_imprecision_errors"] += eval_result.get(
                        "time_imprecision_errors",
                        0,
                    )
                    stats[p_type]["other_errors"] += eval_result.get(
                        "other_errors",
                        0,
                    )

                    if eval_result.get("false_count", 0) == 0:
                        stats[p_type]["perfect"] += 1

                record["evaluation"] = eval_result
                fout.write(json.dumps(record) + "\n")
                processed_count += 1

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    print_summary(stats, processed_count)
    print_error_attribution_summary(stats, processed_count)


if __name__ == "__main__":
    main()
