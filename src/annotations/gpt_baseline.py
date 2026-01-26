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
from gt_detection import get_gt_intervals

# ---------- PATH RESOLUTION FIXES ----------
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "data" / "utils"))

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# Fixed data root path (same as before)
DATA_ROOT = REPO_ROOT / "all_data" / "synthetic"
print(f"[INFO] Data root: {DATA_ROOT}")
print(f"[INFO] Script dir: {SCRIPT_DIR}")
print(f"[INFO] Repo root: {REPO_ROOT}")

PATTERNS = ["range", "trend", "freq", "point", "noisy-trend", "noisy-point", "noisy-freq"]
SPLITS = ["train"]
ID_MAX = 15

# ---------- JSONL I/O ----------

def iter_jsonl(path: str) -> ty.Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ---------- same prompt logic as before ----------

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

def select_features(image_mode: str, avail_cols: int) -> list[str]:
    feature_names = ["value", "moving_average", "moving_std", "centroid"]
    mode_limit = {"ts1": 1, "ts2": 2, "ts3": 3, "ts4": 4}.get(image_mode, None)

    if mode_limit is None:  # mode "text" => on peut imprimer tout ce qu'on sait nommer
        n = min(avail_cols, len(feature_names))
    else:
        n = min(avail_cols, mode_limit, len(feature_names))

    return feature_names[:n]

def to_relpath_str(p: Path, base: Path) -> str:
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def make_paths_relative(paths: list[str], base: Path) -> list[str]:
    out: list[str] = []
    for s in paths:
        out.append(to_relpath_str(Path(s), base))
    return out


def build_prompt(series_str: str, L: int, pattern: str, img_mode: str, feats: list[str]) -> str:
    images_note = IMAGE_MODE_DESCRIPTIONS.get(img_mode, "")
    col_names_str = ", ".join(feats)

    return PROMPT_TEMPLATE.format(
        length=L,
        series=series_str,
        images_note=images_note,
        col_names=col_names_str,
    )

# ---------- series iteration (same spirit as before) ----------

def gt_intervals(pattern: str, sid: int, split: str) -> list[tuple[int, int]]:
    return get_gt_intervals(pattern, sid, split)

def iter_series(root: Path, done: set[tuple[str, str, str]]):
    csv_files = list(root.glob("*/*/csv_data/series_*.csv"))
    for csv in csv_files:
        parts = csv.parts
        try:
            patt, split = parts[-4], parts[-3]
            sid = int(csv.stem.split("_")[-1])
        except (ValueError, IndexError):
            continue

        if PATTERNS and patt not in PATTERNS:
            continue
        if SPLITS and split not in SPLITS:
            continue
        if ID_MAX and sid > ID_MAX:
            continue
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
            first_vals = [int(row[0]) for row in series_rows]
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

# ---------- images + Responses endpoint ----------

def to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        ext = Path(path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "application/octet-stream"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

def resolve_maybe_relative(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def build_content(text: str, img_paths: list[str], max_img: int = 8) -> list[dict]:
    content: list[dict] = [{"type": "input_text", "text": text}]
    for p in img_paths[:max_img]:
        abs_p = resolve_maybe_relative(p, REPO_ROOT)
        if not abs_p.exists():
            continue
        content.append({"type": "input_image", "image_url": to_data_url(str(abs_p))})
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
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict[str, ty.Any] = {
        "model": model,
        "input": build_content(prompt, img_paths, max_img=8),
        "max_output_tokens": max_tokens,
    }

    if "gpt-5" not in model:
        payload["temperature"] = temperature

    last_err: str | None = None
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            return extract_text(r.json())
        except Exception as e:
            last_err = str(e)
            wait = sleep_base * (2 ** attempt)
            print(f"[Err] {last_err} — retry in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} tries: {last_err}")

def images_for_mode(sid: int, figs_dir: Path, image_mode: str) -> list[str]:
    if image_mode == "text":
        return []

    # expected names: {id}.png, {id}_mean.png, {id}_std.png, {id}_stft.png
    base = figs_dir / f"{sid}.png"
    mean = figs_dir / f"{sid}_mean.png"
    std  = figs_dir / f"{sid}_std.png"
    stft = figs_dir / f"{sid}_stft.png"

    out: list[Path] = []
    if image_mode in ("ts1", "ts2", "ts3", "ts4") and base.exists():
        out.append(base)
    if image_mode in ("ts2", "ts3", "ts4") and mean.exists():
        out.append(mean)
    if image_mode in ("ts3", "ts4") and std.exists():
        out.append(std)
    if image_mode == "ts4" and stft.exists():
        out.append(stft)

    # fallback: if nothing matched, include whatever exists in that figs dir for this id
    if not out:
        candidates = [base, mean, std, stft]
        out = [p for p in candidates if p.exists()]

    return [str(p) for p in out]

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

# ---------- main (same argparse as before) ----------

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
    ap.add_argument("--filter", action="store_true",
                    help="compat mode: keep only records whose model output is valid JSON; also add 'output' key")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "src" / "annotations" / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_name = str(out_dir / f"{args.base_name}_{args.image_mode}.jsonl")

    # resume set (same approach as before)
    done: set[tuple[str, str, str]] = set()
    if os.path.exists(final_name):
        for r in iter_jsonl(final_name):
            try:
                done.add((r["pattern_type"], r["split"], str(r["id"])))
            except Exception:
                continue

    print(f"[INFO] Starting with {len(done)} already processed records")
    print(f"[INFO] Output file: {final_name}")

    # same feature limit logic as before (controls how many columns are printed)
    feat_limit = {"ts1": 1, "ts2": 2, "ts3": 3, "ts4": 4}.get(args.image_mode, None)

    processed = skipped = 0
    fout = open(final_name, "a", encoding="utf-8")
    try:
        for rec in iter_series(DATA_ROOT, done):
            patt, split, sid = rec["pattern_type"], rec["split"], rec["id"]
            key = f"{patt}|{split}|{sid}"

            gt = gt_intervals(patt, sid, split)
            # on garde le même garde-fou que phase1 (un seul intervalle GT)
            if not gt or len(gt) != 1:
                skipped += 1
                continue

            avail = len(rec["series"][0]) - 1
            feats = select_features(args.image_mode, avail)
            n = len(feats)

            series_for_prompt = [
                (int(row[0]), *[int(x) for x in row[1 : 1 + n]])
                for row in rec["series"]
            ]

            series_str = format_series(series_for_prompt, feats)
            prompt = build_prompt(series_str, len(rec["series"]), patt, args.image_mode, feats)

            # images dir = .../csv_data/.. -> parent.parent/figs (même logique)
            figs_dir = Path(rec["path"]).parent.parent / "figs"
            img_paths_abs = images_for_mode(sid, figs_dir, args.image_mode)
            img_paths = make_paths_relative(img_paths_abs, REPO_ROOT) 

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

                rec_out = {
                    "pattern_type": patt,
                    "split": split,
                    "id": sid,
                    "ground_truth": gt,
                    "input": prompt,               # <- la clé que tu veux conserver
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

                # compat: si parseable, on ajoute "output" (string JSON compact)
                if norm_json is not None:
                    rec_out["output"] = norm_json

                fout.write(json.dumps(rec_out, ensure_ascii=False, separators=(",", ":")) + "\n")
                fout.flush()
                processed += 1
                print(f"[OK] {key} | imgs={len(img_paths)} | json={'yes' if norm_json else 'no'}")

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
    finally:
        fout.close()

    print(f"Done. processed={processed} skipped={skipped} -> {final_name}")

if __name__ == "__main__":
    main()

