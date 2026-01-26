#!/usr/bin/env python3
"""
Unified Synthetic-Anomaly Annotation Pipeline
Phase 1: CSV → rephrase + verify  →  Phase 2: polish with images →  <base>_<image_mode>.jsonl
Optional Phase 3: Filter false_facts==0 and add structured output key
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import typing as ty
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from gt_detection import get_gt_intervals
from gt_describer import (FIXED_DESCRIPTIONS, classify,
                          describe_frequency_change_temporal)

# ---------- PATH RESOLUTION FIXES ----------

# Add current script directory to Python path for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# Add repo src directories to path for utils
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "data" / "utils"))

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# Fixed data root path
DATA_ROOT = REPO_ROOT / "all_data" / "synthetic"
print(f"[INFO] Data root: {DATA_ROOT}")
print(f"[INFO] Script dir: {SCRIPT_DIR}")
print(f"[INFO] Repo root: {REPO_ROOT}")

PATTERNS = ["range", "trend", "freq", "point",
            "noisy-trend", "noisy-point", "noisy-freq"]
SPLITS = ["train"]
# ID_MAX = 150
ID_MAX = 1

# ---------- JSONL I/O ----------


def iter_jsonl(path: str) -> ty.Iterable[dict]:
    """Stream JSON objects from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str, obj: dict) -> None:
    """Append one JSON object to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------- phase-1 ----------


def gt_intervals(pattern: str, sid: int, split: str) -> list[tuple[int, int]]:
    """Return ground-truth anomaly intervals for a series."""
    return get_gt_intervals(pattern, sid, split)


def classify_gt(pattern: str, vals: list[float], times: list[int], start: int, end: int) -> str | None:
    """Generate factual description using domain helpers."""
    try:
        if pattern == "range":
            return classify(vals, times, start, end)
        if pattern in ("freq", "noisy-freq"):
            return describe_frequency_change_temporal(vals, times, start, end, noisy="noisy" in pattern)
        if pattern in FIXED_DESCRIPTIONS:
            return FIXED_DESCRIPTIONS[pattern]
    except Exception as e:
        print(f"[WARN] factual fail {pattern}/{start}-{end}: {e}")
    return None


def build_factual(pattern: str, gt: list[tuple[int, int]], times: list[int], vals: list[float]) -> str | None:
    """Build fallback-safe factual description."""
    if not gt or len(gt) != 1:
        return None
    desc = classify_gt(pattern, vals, times, gt[0][0], gt[0][1])
    if not desc or desc.startswith("ERROR"):
        return f"From index {gt[0][0]} to {gt[0][1]} the series deviates from its normal behaviour."
    return desc


def format_series(series: list[tuple[int, ...]], feats: list[str]) -> str:
    out: list[str] = []
    k = len(feats)

    for row in series:
        if len(row) < 1 + k:
            raise ValueError(
                f"Row has {len(row) - 1} values but {k} features were requested"
            )

        t = int(row[0])
        vals = row[1 : 1 + k]

        parts = [f"timestamp: {t}"]
        for name, v in zip(feats, vals):
            parts.append(f"{name}: {int(v)}")

        out.append(", ".join(parts))

    return "\n".join(out) + "\n"


def verify(citations: str, data: list[tuple[int, ...]], feats: list[str]) -> tuple[int, int, list[str]]:
    """Check citation strings against actual data."""
    pat = re.compile(r'([\w\-_]+)@(\d+)\s*:\s*([\-]?\d+(?:\.\d+)?)')
    matches = pat.findall(citations)
    logs: list[str] = []
    if not matches:
        return 0, 1, ["FAIL: no citations"]
    t_map: dict[int, dict[str, float]] = {}
    for row in data:
        t = row[0]
        t_map[t] = {feats[i]: float(row[1 + i])
                    for i in range(len(feats)) if 1 + i < len(row)}
    ok = 0
    bad = 0
    for feat, ts, val in matches:
        ts_int = int(ts)
        if ts_int not in t_map or feat not in t_map[ts_int]:
            bad += 1
            logs.append(f"FAIL: {feat}@{ts} missing")
            continue
        if abs(float(val) - t_map[ts_int][feat]) > 1.5:
            bad += 1
            logs.append(f"FAIL: {feat}@{ts} value mismatch")
        else:
            ok += 1
            logs.append(f"PASS: {feat}@{ts}")
    return ok, bad, logs


# Exact prompt strings from original script #1
PROMPT_TEMPLATE = """
Given the time series below{images_note}, determine whether there is an anomalous interval.
If the series is entirely normal, return the empty JSON template. Otherwise, detect and describe the nature of the anomaly.

Series characteristics: {description}

Return **only** a JSON object formatted exactly as follows:
{{
  "anomalies": [
    {{"start": , "end": , "description": ""}}
  ]
}}
(Use empty list if no anomaly).

Time series (len: {length}):
# Columns per line: timestamp : {col_names}
{series}
"""

IMAGE_MODE_DESCRIPTIONS: dict[str, str] = {
    "text": "",
    "ts1":   " and the time-series plot image",
    "ts2":  " and the following plot images: raw values and moving average",
    "ts3":  " and the following plot images: raw values, moving average, and moving standard deviation",
    "ts4":  " and the following plot images: raw values, moving average, moving standard deviation, and the STFT spectrogram",
}

PATTERN_DESC: dict[str, str] = {
    "range":        "The series stays within a narrow numeric band while exhibiting some random noise.",
    "trend":        "The series is a sine wave whose baseline drifts at a constant pace over time.",
    "freq":         "The series is a clean sine wave centred on a fixed mean level.",
    "point":        "The series is a clean sine wave centred on a fixed mean level.",
    "noisy-trend":  "The series is a noisy sine wave whose baseline drifts at a constant pace over time.",
    "noisy-freq":   "The series is a noisy sine wave centred on a fixed mean level.",
    "noisy-point":  "The series is a noisy sine wave centred on a fixed mean level.",
}

REPHRASE_PROMPT_WITH_DATA = """
Rewrite the following anomaly description to be clearer, unambiguous, and easy to understand.

**CRITICAL REQUIREMENTS:**
1. You must justify the description by citing specific data points from the provided series.
2. You must not include more than 5 citations in total.

Citation format:
- Format citations exactly as: (feature_name@timestamp:value)
- Example: "The mean shifts upwards (moving_average@50:88) while variance spikes (std@51:12)."
- Example: "The frequency doubles, evident as values oscillate rapidly (value@120:99, value@122:10)."
- Do NOT invent values. Only use values strictly present in the 'Series Data' below.
- Choose at most 5 of the most representative data points for your explanation.

Original Description:
{desc}

Series Data (columns: {col_names}):
{series_str}
"""


def build_prompt(series_str: str, L: int, pattern: str, img_mode: str, feats: list[str]) -> str:
    """Compose prompt text for LLM."""
    desc = PATTERN_DESC.get(pattern, "Time series data.")
    images_note = IMAGE_MODE_DESCRIPTIONS.get(img_mode, "")
    col_names_str = ", ".join(feats)
    return PROMPT_TEMPLATE.format(
        description=desc,
        length=L,
        series=series_str,
        images_note=images_note,
        col_names=col_names_str
    )


def chat_completion(prompt: str) -> str | None:
    """Call OpenAI chat endpoint and return text or None on failure."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    payload = {"model": "gpt-4o", "messages": [
        {"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 250}
    for attempt in range(3):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            return re.sub(r'(?i)^```(?:json|text)?\s*|\s*```$', '', txt).strip()
        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"   [Err] {e}, retry in {wait}s…")
            time.sleep(wait)
    return None


def iter_series(root: Path, done: set[tuple[str, str, str]]):
    """Stream series records that are not yet processed."""
    print(f"[DEBUG] Looking for CSV files in: {root}")
    print(f"[DEBUG] Pattern: */*/csv_data/series_*.csv")
    
    csv_files = list(root.glob("*/*/csv_data/series_*.csv"))
    print(f"[DEBUG] Found {len(csv_files)} CSV files")
    
    for csv in csv_files:
        parts = csv.parts
        print(f"[DEBUG] Processing: {csv}")
        print(f"[DEBUG] Parts: {parts[-4:]}")
        
        try:
            patt, split = parts[-4], parts[-3]
            sid = int(csv.stem.split("_")[-1])
        except (ValueError, IndexError) as e:
            print(f"[DEBUG] Skipping due to parsing error: {e}")
            continue
            
        if PATTERNS and patt not in PATTERNS:
            print(f"[DEBUG] Skipping {patt} - not in PATTERNS")
            continue
        if SPLITS and split not in SPLITS:
            print(f"[DEBUG] Skipping {split} - not in SPLITS")
            continue
        if ID_MAX and sid > ID_MAX:
            print(f"[DEBUG] Skipping {sid} - > ID_MAX")
            continue
        if (patt, split, str(sid)) in done:
            print(f"[DEBUG] Skipping {patt}/{split}/{sid} - already done")
            continue
            
        try:
            df = pd.read_csv(csv, index_col=0, na_values=["", "NaN"])
            n_feat = df.shape[1]
            cols = [pd.to_numeric(df.iloc[:, i], errors="coerce")
                    for i in range(n_feat)]
            mask = np.logical_and.reduce([c.notna() for c in cols])
            clipped = [np.clip(c[mask].round().astype(int), 0, 99) for c in cols]
            times = df.index[mask].astype(int).tolist()
            series_rows = list(zip(*clipped))
            series = [(int(t), *vals) for t, vals in zip(times, series_rows)]
            first_vals = [int(row[0]) for row in series_rows]
            
            print(f"[DEBUG] Yielding record: {patt}/{split}/{sid}")
            yield {"path": str(csv), "pattern_type": patt, "split": split, "id": sid, "series": series, "times": times, "values": first_vals}
        except Exception as e:
            print(f"[ERROR] Failed to process {csv}: {e}")
            continue


def phase1_yield(args: argparse.Namespace, done: set[tuple[str, str, str]]):
    """Generator that yields phase-1 records."""
    feat_limit = {"ts1": 1, "ts2": 2, "ts3": 3,
                  "ts4": 4}.get(args.image_mode, None)
    
    record_count = 0
    for rec in iter_series(DATA_ROOT, done):
        patt, split, sid = rec["pattern_type"], rec["split"], rec["id"]
        print(f"[DEBUG] Phase1 processing: {patt}/{split}/{sid}")
        
        gt = gt_intervals(patt, sid, split)
        print(f"[DEBUG] Ground truth: {gt}")
        
        if not gt or len(gt) != 1:
            print(f"[DEBUG] Skipping - no single ground truth interval")
            continue
            
        avail = len(rec["series"][0]) - 1
        n = min(avail, feat_limit) if feat_limit else avail
        feats = ["value", "moving_average", "moving_std", "centroid"][:n]
        
        # Now we pass the feature names list for verbose labeling
        series_str = format_series(rec["series"], feats)
        prompt = build_prompt(series_str, len(
            rec["series"]), patt, args.image_mode, feats)
        factual = build_factual(patt, gt, rec["times"], rec["values"])
        
        print(f"[DEBUG] Factual description: {factual}")
        
        rephrased, true, false, logs = None, 0, 0, []
        if factual and not factual.startswith("ERROR"):
            print(f"Processing {patt}/{sid}...", end="", flush=True)
            time.sleep(args.sleep)
            rephrased = chat_completion(REPHRASE_PROMPT_WITH_DATA.format(
                desc=factual, series_str=series_str, col_names=", ".join(feats)))
            if rephrased:
                true, false, logs = verify(rephrased, rec["series"], feats)
                print(f" T:{true} F:{false}")
            else:
                print(" NO RESP")
                
        base_dir = Path(rec["path"]).parent.parent / "figs"
        img_paths = images_for_mode(sid, base_dir, args.image_mode)
                
        record_count += 1
        print(f"[DEBUG] Yielding phase1 record #{record_count}")
        
        yield {
            "pattern_type": patt,
            "split": split,
            "id": sid,
            "ground_truth": gt,
            "input": prompt,
            "image_paths": img_paths,
            "factual_description": factual,
            "rephrased_description": rephrased,
            "true_facts": true,
            "false_facts": false,
            "verification_logs": logs,
        }
    
    print(f"[DEBUG] Phase1 complete - processed {record_count} records")

# ---------- phase-2 ----------


def to_data_url(path: str) -> str:
    """Convert image file to data URL."""
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        ext = Path(path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (
            ".jpg", ".jpeg") else "application/octet-stream"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

def to_relpath_str(p: Path, base: Path) -> str:
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def resolve_maybe_relative(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()

def images_for_mode(sid: int, figs_dir: Path, image_mode: str) -> list[str]:
    """
    Synthetic dataset naming:
      {sid}.png, {sid}_mean.png, {sid}_std.png, {sid}_stft.png
    Retourne des chemins RELATIFS à REPO_ROOT, dans un ordre stable.
    """
    if image_mode == "text":
        return []

    suffixes_by_mode: dict[str, list[str]] = {
        "ts1": [".png"],
        "ts2": [".png", "_mean.png"],
        "ts3": [".png", "_mean.png", "_std.png"],
        "ts4": [".png", "_mean.png", "_std.png", "_stft.png"],
    }
    suffixes = suffixes_by_mode.get(image_mode, [])

    out: list[str] = []
    for sfx in suffixes:
        p = figs_dir / f"{sid}{sfx}"
        if p.exists():
            out.append(to_relpath_str(p, REPO_ROOT))
        else:
            print(f"[WARN] Missing image for mode={image_mode}: {p}")

    return out

def build_content(text: str, img_paths: list[str], max_img: int) -> list[dict]:
    """Build request content with text and ≤max_img images."""
    content: list[dict] = [{"type": "input_text", "text": text}]

    selected = img_paths[:max_img] if max_img >= 0 else img_paths
    for p in selected:
        abs_p = resolve_maybe_relative(p, REPO_ROOT)
        if not abs_p.exists():
            print(f"[WARN] missing img {p}")
            continue
        content.append({"type": "input_image", "image_url": to_data_url(str(abs_p))})

    return [{"role": "user", "content": content}]


def extract_text(resp: dict) -> str:
    """Extract text from Responses endpoint reply."""
    if isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
        return resp["output_text"].strip()
    chunks: list[str] = []
    for item in resp.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") in ("output_text", "text") and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "\n".join(chunks).strip()


# Exact prompt string from original script #2
PROMPT_PREFIX = (
    "Improve the below description in such a way that it will help a human "
    "identify the time series anomaly better: (keep syntax simple, no markdown)"
    "Keep the value citations formatted as '(feature_name@timestamp:value)'"
)


def call_responses(
    prompt: str,
    img_paths: list[str],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    sleep_base: float,
) -> str:
    """Call OpenAI Responses endpoint with images."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    payload = {"model": model, "input": build_content(
        prompt, img_paths, max_img=8), "temperature": temperature, "max_output_tokens": max_tokens}
    last_err: str | None = None
    for attempt in range(retries):
        try:
            r = requests.post("https://api.openai.com/v1/responses",
                              headers=headers, json=payload, timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:4000]}")
            return extract_text(r.json())
        except Exception as e:
            last_err = str(e)
            wait = sleep_base * (2 ** attempt)
            print(f"[Err] {last_err} — retry in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} tries: {last_err}")

# ---------- phase-3 (in-stream filtering) ----------
#  (kept for compatibility – not used when --filter is on)
def process_and_filter_jsonl(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Filter JSONL: keep only false_facts==0 and add structured output key.
    Returns (kept, total) counts."""
    kept = total = 0
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1

            # Filter condition
            if obj.get("false_facts") != 0:
                continue

            # Build structured output
            try:
                start = obj["ground_truth"][0][0]
                end = obj["ground_truth"][0][1]
            except (KeyError, IndexError, TypeError):
                continue

            description = obj.get("_gpt_output_text", "")
            anomalies_obj = {
                "anomalies": [
                    {
                        "start": start,
                        "end": end,
                        "description": description,
                    }
                ]
            }
            obj["output"] = json.dumps(anomalies_obj, separators=(",", ":"))

            # Write filtered record
            fout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
    return kept, total


# ---------- main (re-written) ----------
def main() -> None:
    """Run full pipeline: phase-1 → phase-2 → optional phase-3 filtering."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image-mode", choices=["text", "ts1", "ts2", "ts3", "ts4"], default="text")
    ap.add_argument("--base-name", default="annotations",
                    help="base name for final JSONL")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="sleep between LLM calls phase1")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=250)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep-base", type=float, default=2.0)
    ap.add_argument("--sleep-between", type=float, default=2.0,
                    help="sleep between calls phase2")
    ap.add_argument("--filter", action="store_true",
                    help="apply false_facts==0 filter and add output key (phase-3)")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "src" / "annotations" / "jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide which file we are going to write
    if args.filter:
        final_name = str(out_dir / f"{args.base_name}_{args.image_mode}_filtered.jsonl")
    else:
        final_name = str(out_dir / f"{args.base_name}_{args.image_mode}.jsonl")

    # Build “done” set from the *same* file we are about to append to
    done = set()
    if os.path.exists(final_name):
        done = {(r["pattern_type"], r["split"], str(r["id"]))
                for r in iter_jsonl(final_name)}
    print(f"[INFO] Starting pipeline with {len(done)} already processed records")
    print(f"[INFO] Output file: {final_name}")

    processed = skipped = 0
    fout = open(final_name, "a", encoding="utf-8")   # keep handle open

    try:
        for rec in phase1_yield(args, done):
            key = f"{rec['pattern_type']}|{rec['split']}|{rec['id']}"
            text = rec.get("rephrased_description")
            if not isinstance(text, str) or not text.strip():
                if not args.filter:                 # only write if we keep everything
                    append_jsonl(final_name, {**rec, "_status": "skipped_no_text"})
                skipped += 1
                continue

            img_paths = rec.get("image_paths") or []
            try:
                polished = call_responses(
                    PROMPT_PREFIX + text.strip(),
                    img_paths,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_output_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep_base=args.sleep_base,
                )
            except Exception as e:
                if not args.filter:                 # only write if we keep everything
                    append_jsonl(final_name, {**rec, "_status": "error", "_error": str(e)})
                print(f"[FAIL] {key}: {e}")
                continue

            # ---- decide whether we keep this record ----
            rec_out = {**rec, "_gpt_output_text": polished,
                       "_prompt_sent": PROMPT_PREFIX + text.strip()}

            if args.filter:
                # upstream filter: false_facts must be 0
                if rec_out.get("false_facts") != 0:
                    skipped += 1
                    continue
                # add structured output key
                try:
                    start, end = rec_out["ground_truth"][0]
                except (KeyError, IndexError, TypeError):
                    skipped += 1
                    continue
                anomalies_obj = {
                    "anomalies": [{"start": start,
                                   "end": end,
                                   "description": polished}]
                }
                rec_out["output"] = json.dumps(anomalies_obj, separators=(",", ":"))

            # ---- write the (possibly filtered) record ----
            fout.write(json.dumps(rec_out, ensure_ascii=False, separators=(",", ":")) + "\n")
            fout.flush()
            processed += 1
            print(f"[OK] {key} | imgs={len(img_paths)} | out_len={len(polished)}")

            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        fout.close()

    print(f"Pipeline complete. processed={processed} skipped={skipped} -> {final_name}")

    # If --filter was requested we already wrote the filtered file;
    # nothing more to do.

if __name__ == "__main__":
    main()
