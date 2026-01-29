#!/usr/bin/env python3

"""
Variant of gpt_baseline.py that:
- applies STL-style deseasonalization (via ACF-based period detection)
- builds the 0shot-text-vision prompt from the paper (Figure 13)
- constructs index-aware series input using (index, value) format (integers only)
- passes the deseasonalized series to the LLM along with a plot image
- parses the LLM output to extract (start, end) and stores it in a JSON-string field "output"
- preserves output structure for downstream evaluation compatibility
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from gt_detection import get_gt_intervals

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

DATA_ROOT = REPO_ROOT / "all_data" / "synthetic"

PROMPT_TEMPLATE = """
You are given a univariate time series measured at hourly intervals. Assume there are up to 1 anomalies.
Output a list of all detected anomalies as (start, end) index pairs. If there is no anomaly, return an empty list.

Time series:
{series}
""".strip()

# ---- NEW: ID_MAX filtering (same spirit as the other script)
ID_MAX = 15

PAIR_PAREN_RE = re.compile(r"\(\s*(-?\d+)\s*[,;]\s*(-?\d+)\s*\)")
PAIR_BRACK_RE = re.compile(r"\[\s*(-?\d+)\s*[,;]\s*(-?\d+)\s*\]")


def parse_predicted_intervals(text: str) -> list[list[int]]:
    """
    Extrait des intervalles d'anomalies depuis la sortie LLM.
    Retour: liste de [start, end] (JSON-friendly).
    """
    if not text or not text.strip():
        return []

    low = text.lower()

    # Cas "pas d'anomalie"
    if re.search(r"\[\s*\]", text) or ("no anomaly" in low) or ("no anomalies" in low) or ("empty list" in low):
        return []

    pairs: list[tuple[int, int]] = []

    # Priorité: (A,B)
    for a, b in PAIR_PAREN_RE.findall(text):
        pairs.append((int(a), int(b)))

    # Fallback: [A,B] / [[A,B]]
    if not pairs:
        for a, b in PAIR_BRACK_RE.findall(text):
            pairs.append((int(a), int(b)))

    # Fallback: "A to B" / "A - B" / "A à B"
    if not pairs:
        m = re.search(r"(-?\d+)\s*(?:to|à|-|–|—)\s*(-?\d+)", text, flags=re.IGNORECASE)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))

    # Normalisation: start <= end, dédoublonnage en gardant l'ordre
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
    acorr = acf(series, nlags=max_lag, fft=True)
    return int(np.argmax(acorr[1:]) + 1)  # skip lag-0


def deseasonalize(series: np.ndarray, period: int) -> np.ndarray:
    padded = pd.Series(series)
    try:
        result = seasonal_decompose(padded, model="additive", period=period, extrapolate_trend="freq")
        return series - result.seasonal
    except Exception:
        return series  # fallback: no decomposition


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
    return "\n".join([f"({t},{int(round(v))})" for t, v in zip(index, values)])


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
    # optionnel: permettre de surcharger sans changer le comportement par défaut
    ap.add_argument("--id-max", type=int, default=ID_MAX, help="skip series with id > id-max (compat filter)")
    args = ap.parse_args()

    out_dir = REPO_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.base_name}.jsonl"

    seen = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add((r["pattern_type"], r["split"], str(r["id"])))
                except Exception:
                    continue

    fout = out_path.open("a", encoding="utf-8")

    for pattern_dir in DATA_ROOT.glob("*/train"):
        pattern = pattern_dir.parent.name
        for csv_file in pattern_dir.glob("csv_data/series_*.csv"):
            sid = int(csv_file.stem.split("_")[-1])

            # ---- NEW: ID_MAX filter (same as the other script)
            if args.id_max and sid > args.id_max:
                continue

            key = (pattern, "train", str(sid))
            if key in seen:
                continue

            try:
                df = pd.read_csv(csv_file, index_col=0)
                series_raw = df.iloc[:, 0].dropna().astype(float).values
                t_index = df.index[: len(series_raw)]

                period = estimate_period(series_raw)
                x_deseason = deseasonalize(series_raw, period)

                img_dir = csv_file.parent.parent / "figs_anomllm"
                img_path = img_dir / f"{sid}_deseason.png"
                plot_series(t_index, x_deseason, img_path)

                text_series = build_indexed_text(t_index.tolist(), x_deseason.tolist())
                prompt = PROMPT_TEMPLATE.format(series=text_series)
                gt = get_gt_intervals(pattern, sid, "train")

                out_txt = call_llm(prompt, img_path, args.model, args.temperature, args.max_output_tokens, args.timeout)

                # output must be a JSON string like:
                # "{\"anomalies\":[{\"start\":111,\"end\":161,\"description\":\"...\"}]}"
                pred_intervals = parse_predicted_intervals(out_txt)[:1]  # "up to 1 anomalies"
                anomalies = [
                    {"start": int(start), "end": int(end), "description": out_txt.strip()}
                    for start, end in pred_intervals
                ]
                output_str = json.dumps({"anomalies": anomalies}, ensure_ascii=False, separators=(",", ":"))

                rec = {
                    "pattern_type": pattern,
                    "split": "train",
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
                    },
                    "_status": "ok_non_json",
                }

                fout.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                fout.flush()
                print(f"[OK] {key}")

            except Exception as e:
                print(f"[FAIL] {key}: {e}")

    fout.close()
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
