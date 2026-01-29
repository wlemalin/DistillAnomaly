#!/usr/bin/env python3

"""
Variant of gpt_baseline.py that:
- applies STL-style deseasonalization (via ACF-based period detection)
- builds the 0shot-text-vision prompt from the paper (Figure 13)
- constructs index-aware series input using (index, value) format (integers only)
- passes the deseasonalized series to the LLM along with a plot image
- parses the LLM output to extract (start, end) and stores it in a JSON-string field "output"
- preserves output structure for downstream evaluation compatibility

UPDATED:
- loads series from all_data/UCR_dataset/csv_data/*.csv (flat layout)
- pattern_type="ucr", split="train", id = csv stem (string)
- ground_truth parsed from filename suffix _<start>_<end>
- generated plots stored under all_data/UCR_dataset/figs_anomllm/
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

# Optional (kept for compat, but UCR GT is parsed from filename)
try:
    from gt_detection import get_gt_intervals  # type: ignore
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

# --- UCR dataset layout (like your second script) ---
DATASET_ROOT = REPO_ROOT / "all_data" / "UCR_dataset"
CSV_ROOT = DATASET_ROOT / "csv_data"

if not CSV_ROOT.exists():
    raise FileNotFoundError(f"CSV_ROOT not found: {CSV_ROOT}")

print(f"[INFO] Dataset root: {DATASET_ROOT}")
print(f"[INFO] CSV root: {CSV_ROOT}")
print(f"[INFO] Script dir: {SCRIPT_DIR}")
print(f"[INFO] Repo root: {REPO_ROOT}")

PROMPT_TEMPLATE = """
You are given a univariate time series measured at hourly intervals. Assume there are up to 1 anomalies.
Output a list of all detected anomalies as (start, end) index pairs. If there is no anomaly, return an empty list.

Time series:
{series}
""".strip()

PAIR_PAREN_RE = re.compile(r"\(\s*(-?\d+)\s*[,;]\s*(-?\d+)\s*\)")
PAIR_BRACK_RE = re.compile(r"\[\s*(-?\d+)\s*[,;]\s*(-?\d+)\s*\]")

_GT_RE = re.compile(r"_(\d+)_(\d+)$")


def parse_gt_from_stem(stem: str) -> list[tuple[int, int]]:
    """
    UCR style: ground-truth encoded at end of filename stem as _<start>_<end>
    Example: something_120_180.csv  -> [(120, 180)]
    """
    m = _GT_RE.search(stem)
    if not m:
        return []
    start = int(m.group(1))
    end = int(m.group(2))
    if end < start:
        start, end = end, start
    return [(start, end)]


def parse_predicted_intervals(text: str) -> list[list[int]]:
    """
    Extrait des intervalles d'anomalies depuis la sortie LLM.
    Retour: liste de [start, end] (JSON-friendly).
    """
    if not text or not text.strip():
        return []

    low = text.lower()

    if re.search(r"\[\s*\]", text) or ("no anomaly" in low) or ("no anomalies" in low) or ("empty list" in low):
        return []

    pairs: list[tuple[int, int]] = []

    for a, b in PAIR_PAREN_RE.findall(text):
        pairs.append((int(a), int(b)))

    if not pairs:
        for a, b in PAIR_BRACK_RE.findall(text):
            pairs.append((int(a), int(b)))

    if not pairs:
        m = re.search(r"(-?\d+)\s*(?:to|à|-|–|—)\s*(-?\d+)", text, flags=re.IGNORECASE)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))

    out: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for a, b in pairs:
        start, end = (a, b) if a <= b else (b, a)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append([start, end])

    return out


def estimate_period(series: np.ndarray, max_lag: int = 100) -> int:
    # garde-fous pour séries courtes
    n = int(series.shape[0])
    if n < 4:
        return 1
    nlags = min(max_lag, n - 2)
    if nlags < 1:
        return 1
    acorr = acf(series, nlags=nlags, fft=True)
    # si acf renvoie quelque chose de bizarre, fallback
    if acorr is None or len(acorr) < 2:
        return 1
    return int(np.argmax(acorr[1:]) + 1)


def deseasonalize(series: np.ndarray, period: int) -> np.ndarray:
    if period is None or period < 2 or period >= len(series):
        return series
    padded = pd.Series(series)
    try:
        result = seasonal_decompose(padded, model="additive", period=period, extrapolate_trend="freq")
        seasonal = np.asarray(result.seasonal)
        if seasonal.shape[0] != series.shape[0]:
            return series
        return series - seasonal
    except Exception:
        return series


def plot_series(index: np.ndarray, values: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 2))
    plt.plot(index, values, color="black", linewidth=1)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def build_indexed_text(index: list[int], values: list[float]) -> str:
    return "\n".join([f"({int(t)},{int(round(v))})" for t, v in zip(index, values)])


def to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


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


def call_llm(prompt: str, image_path: Path, model: str, temperature: float, max_tokens: int, timeout: int) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": to_data_url(str(image_path))},
    ]
    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return extract_text(r.json())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--base-name", default="anomllm_0shot_text_vision")
    # réinterprété pour dataset plat: limite sur le nombre de fichiers traités
    ap.add_argument("--id-max", type=int, default=999, help="process at most this many CSV files (compat limiter)")
    args = ap.parse_args()

    # (optionnel) aligné avec les autres scripts qui écrivent dans src/annotations/clean
    out_dir = REPO_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.base_name}.jsonl"

    seen: set[tuple[str, str, str]] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add((r["pattern_type"], r["split"], str(r["id"])))
                except Exception:
                    continue

    # UCR dataset: flat CSV list
    csv_files = sorted(CSV_ROOT.glob("*.csv"))
    if args.id_max and args.id_max > 0:
        csv_files = csv_files[: args.id_max]

    # on génère nos figures ici (dataset-level folder)
    img_dir = DATASET_ROOT / "figs_anomllm"
    img_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as fout:
        for csv_file in csv_files:
            pattern = "ucr"
            split = "train"
            sid = csv_file.stem  # string id

            key = (pattern, split, str(sid))
            if key in seen:
                continue

            try:
                df = pd.read_csv(csv_file, index_col=0, na_values=["", "NaN"])
                if df.shape[1] < 1:
                    raise ValueError("CSV has no value column")

                s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
                mask = s.notna()
                if not mask.any():
                    raise ValueError("Empty / all-NaN series")

                series_raw = s[mask].astype(float).values
                t_index = df.index[mask].astype(int).to_numpy()

                period = estimate_period(series_raw)
                x_deseason = deseasonalize(series_raw, period)

                img_path = img_dir / f"{sid}_deseason.png"
                plot_series(t_index, x_deseason, img_path)

                text_series = build_indexed_text(t_index.tolist(), x_deseason.tolist())
                prompt = PROMPT_TEMPLATE.format(series=text_series)

                # UCR ground truth from filename suffix
                gt = parse_gt_from_stem(str(sid))
                if not gt:
                    # fallback possible si tu as une logique externe (optionnel)
                    if get_gt_intervals is not None:
                        try:
                            gt = get_gt_intervals(pattern, sid, split)  # type: ignore[misc]
                        except Exception:
                            gt = []

                out_txt = call_llm(prompt, img_path, args.model, args.temperature, args.max_output_tokens, args.timeout)

                pred_intervals = parse_predicted_intervals(out_txt)[:1]  # up to 1 anomaly
                anomalies = [
                    {"start": int(start), "end": int(end), "description": out_txt.strip()}
                    for start, end in pred_intervals
                ]
                output_str = json.dumps({"anomalies": anomalies}, ensure_ascii=False, separators=(",", ":"))

                rec = {
                    "pattern_type": pattern,
                    "split": split,
                    "id": sid,
                    "ground_truth": gt,
                    "input": prompt,
                    "image_paths": [str(img_path.relative_to(REPO_ROOT))],
                    "output": output_str,
                    "_gpt_output_text": out_txt,
                    "_json_error": "not-json",
                    "_meta": {
                        "model": args.model,
                        "temperature": args.temperature,
                        "max_output_tokens": args.max_output_tokens,
                        "image_mode": "anomllm-0shot-text-vision",
                        "images_used": [str(img_path.relative_to(REPO_ROOT))],
                        "dataset_root": str(DATASET_ROOT.relative_to(REPO_ROOT)) if DATASET_ROOT.is_relative_to(REPO_ROOT) else str(DATASET_ROOT),
                        "csv_path": str(csv_file.relative_to(REPO_ROOT)) if csv_file.is_relative_to(REPO_ROOT) else str(csv_file),
                        "period_estimate": int(period),
                    },
                    "_status": "ok_non_json",
                }

                fout.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                fout.flush()
                print(f"[OK] {key}")

            except Exception as e:
                print(f"[FAIL] {key}: {e}")

    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
