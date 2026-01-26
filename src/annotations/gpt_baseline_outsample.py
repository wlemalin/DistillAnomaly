#!/usr/bin/env python3
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

try:
    from gt_detection import get_gt_intervals
except Exception:
    get_gt_intervals = None  # type: ignore


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start.parent.parent


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

REPO_ROOT = find_repo_root(SCRIPT_DIR)
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "data" / "utils"))

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

DATASET_ROOT = REPO_ROOT / "all_data" / "UCR_dataset"
CSV_ROOT = DATASET_ROOT / "csv_data"
FIGS_ROOT = DATASET_ROOT / "figs_data"

if not CSV_ROOT.exists():
    raise FileNotFoundError(f"CSV_ROOT not found: {CSV_ROOT}")
if not FIGS_ROOT.exists():
    raise FileNotFoundError(f"FIGS_ROOT not found: {FIGS_ROOT}")

print("[DBG] FIGS_ROOT exists:", FIGS_ROOT.exists())
print("[DBG] CSV_ROOT exists :", CSV_ROOT.exists())
print("[DBG] FIGS sample:", [p.name for p in FIGS_ROOT.glob("*.png")][:5])
print("[DBG] CSV  sample:", [p.name for p in CSV_ROOT.glob("*.csv")][:5])

print(f"[INFO] Dataset root: {DATASET_ROOT}")
print(f"[INFO] CSV root: {CSV_ROOT}")
print(f"[INFO] FIGS root: {FIGS_ROOT}")
print(f"[INFO] Script dir: {SCRIPT_DIR}")
print(f"[INFO] Repo root: {REPO_ROOT}")


FEATURES_BY_IMAGE_MODE: dict[str, list[str]] = {
    "text": ["value"],
    "ts1": ["value"],
    "ts2": ["value", "moving_average"],
    "ts3": ["value", "moving_average", "moving_std"],
    "ts4": ["value", "moving_average", "moving_std"],
}


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


def iter_jsonl(path: str) -> ty.Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


PROMPT_TEMPLATE = """
Given the time series below{images_note}, determine whether there is an anomalous interval.
Then, detect and describe the nature of the anomaly.
The description you give should contain citations formatted as '(feature@timestamp:value)' for the values that justify your detection.

Return **only** a JSON object formatted exactly as follows:
{{
  "anomalies": [
    {{"start": , "end": , "description": ""}}
  ]
}}

Time series (len: {length}):
# Columns per line: timestamp : {col_names}
{series}
"""

IMAGE_MODE_DESCRIPTIONS: dict[str, str] = {
    "text": "",
    "ts1": " and the time-series plot image",
    "ts2": " and the following plot images: raw values and moving average",
    "ts3": " and the following plot images: raw values, moving average, and moving standard deviation",
    "ts4": " and the following plot images: raw values, moving average, moving standard deviation, and the STFT spectrogram",
}

PATTERN_DESC: dict[str, str] = {
    "range": "The series stays within a narrow numeric band while exhibiting some random noise.",
    "trend": "The series is a sine wave whose baseline drifts at a constant pace over time.",
    "freq": "The series is a clean sine wave centred on a fixed mean level.",
    "point": "The series is a clean sine wave centred on a fixed mean level.",
    "noisy-trend": "The series is a noisy sine wave whose baseline drifts at a constant pace over time.",
    "noisy-freq": "The series is a noisy sine wave centred on a fixed mean level.",
    "noisy-point": "The series is a noisy sine wave centred on a fixed mean level.",
}


def format_series(series: list[tuple[int, ...]], feats: list[str]) -> str:
    out: list[str] = []
    k = len(feats)
    for row in series:
        if len(row) < 1 + k:
            raise ValueError(f"Row has {len(row)-1} values but {k} features were requested")
        t = int(row[0])
        vals = row[1 : 1 + k]
        parts = [f"timestamp: {t}"]
        for name, v in zip(feats, vals):
            parts.append(f"{name}: {int(v)}")
        out.append(", ".join(parts))
    return "\n".join(out) + "\n"


def build_prompt(series_str: str, L: int, pattern: str, img_mode: str, feats: list[str]) -> str:
    _ = PATTERN_DESC.get(pattern, "Time series data.")
    images_note = IMAGE_MODE_DESCRIPTIONS.get(img_mode, "")
    col_names_str = ", ".join(feats)
    return PROMPT_TEMPLATE.format(
        length=L,
        series=series_str,
        images_note=images_note,
        col_names=col_names_str,
    )


_GT_RE = re.compile(r"_(\d+)_(\d+)$")


def parse_gt_from_stem(stem: str) -> list[tuple[int, int]]:
    m = _GT_RE.search(stem)
    if not m:
        return []
    start = int(m.group(1))
    end = int(m.group(2))
    if end < start:
        start, end = end, start
    return [(start, end)]


def gt_intervals(pattern: str, sid: int | str, split: str) -> list[tuple[int, int]]:
    if get_gt_intervals is None:
        return []
    try:
        return get_gt_intervals(pattern, sid, split)  # type: ignore[misc]
    except Exception:
        return []


def iter_series(done: set[tuple[str, str, str]]):
    csv_files = sorted(CSV_ROOT.glob("*.csv"))
    for csv in csv_files:
        patt = "ucr"
        split = "train"
        sid = csv.stem

        if (patt, split, str(sid)) in done:
            continue

        try:
            df = pd.read_csv(csv, index_col=0, na_values=["", "NaN"])
            n_feat = df.shape[1]
            cols = [pd.to_numeric(df.iloc[:, i], errors="coerce") for i in range(n_feat)]
            mask = np.logical_and.reduce([c.notna() for c in cols])
            clipped = [np.clip(c[mask].round().astype(int), 0, 99) for c in cols]
            times = df.index[mask].astype(int).tolist()

            series_rows = list(zip(*clipped))
            series = [(int(t), *vals) for t, vals in zip(times, series_rows)]
            first_vals = [int(row[0]) for row in series_rows] if series_rows else []

            yield {
                "path": str(csv),
                "pattern_type": patt,
                "split": split,
                "id": sid,
                "series": series,
                "times": times,
                "values": first_vals,
            }
        except Exception as e:
            print(f"[ERROR] Failed to process {csv}: {e}")
            continue


def to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        ext = Path(path).suffix.lower()
        if ext == ".png":
            mime = "image/png"
        elif ext in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        else:
            mime = "application/octet-stream"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def build_content(text: str, img_paths: list[str], max_img: int = 8) -> list[dict]:
    content: list[dict] = [{"type": "input_text", "text": text}]
    attached = 0
    for p in img_paths[:max_img]:
        abs_p = resolve_maybe_relative(p, REPO_ROOT)
        if not abs_p.exists():
            print(f"[WARN] Missing image at attach time: {p}")
            continue
        content.append({"type": "input_image", "image_url": to_data_url(str(abs_p))})
        attached += 1
    print(f"[DBG] images: requested={len(img_paths)} attached={attached}")
    return [{"role": "user", "content": content}]


def extract_text(resp: dict) -> str:
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


_CODE_FENCE_RE = re.compile(r"(?is)^\s*```(?:json|text)?\s*|\s*```\s*$")


def strip_code_fences(txt: str) -> str:
    return _CODE_FENCE_RE.sub("", txt).strip()


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
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "input": build_content(prompt, img_paths, max_img=8),
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    last_err: str | None = None
    for attempt in range(retries):
        try:
            r = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:4000]}")
            return extract_text(r.json())
        except Exception as e:
            last_err = str(e)
            wait = sleep_base * (2**attempt)
            print(f"[ERR] {last_err} -- retry in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} tries: {last_err}")


def _pick_existing_png(figs_dir: Path, filename_no_ext: str) -> Path | None:
    p1 = figs_dir / (filename_no_ext + ".png")
    if p1.exists():
        return p1
    p2 = figs_dir / (filename_no_ext + ".PNG")
    if p2.exists():
        return p2
    return None


def images_for_mode(stem: str, figs_dir: Path, image_mode: str) -> list[str]:
    if image_mode == "text":
        return []

    suffixes_by_mode: dict[str, list[str]] = {
        "ts1": ["raw"],
        "ts2": ["raw", "mean"],
        "ts3": ["raw", "mean", "std"],
        "ts4": ["raw", "mean", "std", "stft"],
    }
    suffixes = suffixes_by_mode.get(image_mode, [])

    found_abs: list[Path] = []
    missing: list[str] = []

    for suf in suffixes:
        base = f"{stem}_{suf}"
        p = _pick_existing_png(figs_dir, base)
        if p is None:
            missing.append(base + ".png")
            continue
        found_abs.append(p)

    if missing:
        print(f"[WARN] Missing images for stem={stem} mode={image_mode}: {missing}")

    return [to_relpath_str(p, REPO_ROOT) for p in found_abs]


def try_parse_json(text: str) -> tuple[str | None, str | None]:
    cleaned = strip_code_fences(text)
    try:
        obj = json.loads(cleaned)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")), None
    except Exception:
        pass

    m = re.search(r"(?s)\{.*\}", cleaned)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":")), None
        except Exception as e:
            return None, f"JSON parse failed (extracted block): {e}"

    return None, "JSON parse failed (no JSON object found)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-mode", choices=["text", "ts1", "ts2", "ts3", "ts4"], default="ts1")
    ap.add_argument("--base-name", default="annotations", help="base name for final JSONL")
    ap.add_argument("--sleep", type=float, default=1.0, help="sleep before each model call")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=250)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep-base", type=float, default=2.0)
    ap.add_argument("--sleep-between", type=float, default=2.0, help="sleep between records")
    ap.add_argument(
        "--filter",
        action="store_true",
        help="keep only records whose model output is valid JSON; also add 'output' key",
    )
    args = ap.parse_args()

    out_dir = REPO_ROOT / "src" / "annotations" / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_name = str(out_dir / f"{args.base_name}_{args.image_mode}.jsonl")

    done: set[tuple[str, str, str]] = set()
    if os.path.exists(final_name):
        for r in iter_jsonl(final_name):
            try:
                done.add((r["pattern_type"], r["split"], str(r["id"])))
            except Exception:
                continue

    print(f"[INFO] Starting with {len(done)} already processed records")
    print(f"[INFO] Output file: {final_name}")

    processed = 0
    skipped = 0

    with open(final_name, "a", encoding="utf-8") as fout:
        try:
            for rec in iter_series(done):
                patt, split, sid = rec["pattern_type"], rec["split"], rec["id"]
                key = f"{patt}|{split}|{sid}"

                gt = parse_gt_from_stem(str(sid))
                if not gt:
                    print(f"[WARN] No GT parsed from stem for {sid}")

                if gt and rec["series"]:
                    start, end = gt[0]
                    L = len(rec["series"])
                    if start < 0 or end < 0 or start >= L or end >= L:
                        print(f"[WARN] GT {gt[0]} out of bounds for series len={L} ({sid})")

                if not rec["series"]:
                    print(f"[WARN] Empty series for {sid}, skipping")
                    continue

                desired_feats = FEATURES_BY_IMAGE_MODE[args.image_mode]
                avail = len(rec["series"][0]) - 1
                use_n = min(avail, len(desired_feats))
                feats = desired_feats[:use_n]

                if use_n < len(desired_feats):
                    print(f"[WARN] Not enough columns for mode={args.image_mode}: need={len(desired_feats)} have={avail}")

                series_for_prompt = [
                    (int(row[0]), *[int(x) for x in row[1 : 1 + use_n]])
                    for row in rec["series"]
                ]

                series_str = format_series(series_for_prompt, feats)
                prompt = build_prompt(series_str, len(rec["series"]), patt, args.image_mode, feats)

                img_paths = images_for_mode(str(sid), FIGS_ROOT, args.image_mode)

                if args.sleep > 0:
                    time.sleep(args.sleep)

                try:
                    out_txt = call_responses(
                        prompt=prompt,
                        img_paths=img_paths,
                        model=args.model,
                        temperature=args.temperature,
                        max_tokens=args.max_output_tokens,
                        timeout=args.timeout,
                        retries=args.retries,
                        sleep_base=args.sleep_base,
                    )

                    out_clean = strip_code_fences(out_txt)
                    norm_json, json_err = try_parse_json(out_clean)

                    if args.filter and norm_json is None:
                        skipped += 1
                        continue

                    rec_out: dict[str, ty.Any] = {
                        "pattern_type": patt,
                        "split": split,
                        "id": sid,
                        "ground_truth": gt,
                        "input": prompt,
                        "image_paths": img_paths,
                        "_gpt_output_text": out_clean,
                        "_json_error": json_err,
                        "_meta": {
                            "model": args.model,
                            "temperature": args.temperature,
                            "max_output_tokens": args.max_output_tokens,
                            "image_mode": args.image_mode,
                            "images_used": img_paths,
                        },
                        "_status": "ok" if norm_json is not None else "ok_non_json",
                    }

                    if norm_json is not None:
                        rec_out["output"] = norm_json

                    fout.write(json.dumps(rec_out, ensure_ascii=False, separators=(",", ":")) + "\n")
                    fout.flush()
                    processed += 1
                    print(
                        f"[OK] {key} | imgs={len(img_paths)} | gt={gt[0] if gt else None} | json={'yes' if norm_json else 'no'}"
                    )

                except Exception as e:
                    if not args.filter:
                        rec_out = {
                            "pattern_type": patt,
                            "split": split,
                            "id": sid,
                            "ground_truth": gt,
                            "input": prompt,
                            "image_paths": img_paths,
                            "_status": "error",
                            "_error": str(e),
                        }
                        fout.write(json.dumps(rec_out, ensure_ascii=False, separators=(",", ":")) + "\n")
                        fout.flush()
                        processed += 1
                    print(f"[FAIL] {key}: {e}")

                if args.sleep_between > 0:
                    time.sleep(args.sleep_between)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")

    print(f"Done. processed={processed} skipped={skipped} -> {final_name}")


if __name__ == "__main__":
    main()
