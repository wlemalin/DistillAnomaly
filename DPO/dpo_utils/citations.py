import argparse, json, os, sys, math, logging, warnings, random, re
from pathlib import Path
from typing import Dict, Sequence, Any, List, Tuple, Optional


# =============================================================================
# Citation truthfulness (reprise logique de ton eval)
# =============================================================================

_CIT_RE = re.compile(r'([\w\-_]+)@(\d+)\s*:\s*([\-]?\d+)')
_FEATURE_ALIASES = {
    "index": "value",
    "values": "value",
    "average": "moving_average",
}

def _normalize_feature_name(raw: str) -> str:
    k = raw.strip()
    return _FEATURE_ALIASES.get(k.lower(), k)

def _auto_plain_text(prompt_text: str) -> bool:
    # Auto-détection simple du format "timestamp: X, value: Y, ..."
    return bool(re.search(r"\btimestamp\s*:", prompt_text))

def parse_context_data(prompt_text: str, plain_text: bool = False) -> Dict[int, Dict[str, int]]:
    """
    Reconstruit la série à partir du prompt (valeurs int strictes).
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
        aliases = {"ts": "timestamp", "time": "timestamp", "val": "value"}
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

def verify_citations_truthfulness(text: str, data_map: Dict[int, Dict[str, int]]) -> Dict[str, Any]:
    """
    Vérifie les citations dans le format: feature@timestamp:value
    Règle: égalité entière stricte.
    """
    if not isinstance(text, str):
        return {"has_citations": False, "true_count": 0, "false_count": 0, "score": 0.0}

    matches = _CIT_RE.findall(text)
    if not matches:
        return {"has_citations": False, "true_count": 0, "false_count": 0, "score": 0.0}

    true_f = 0
    false_f = 0

    for feat_raw, ts_s, val_s in matches:
        feat = _normalize_feature_name(feat_raw)
        ts = int(ts_s)
        try:
            cited = int(val_s)
        except ValueError:
            false_f += 1
            continue

        row = data_map.get(ts)
        if row is None:
            false_f += 1
            continue
        if feat not in row:
            false_f += 1
            continue
        if row[feat] != cited:
            false_f += 1
            continue

        true_f += 1

    tot = true_f + false_f
    score = (true_f / tot) if tot > 0 else 0.0
    return {"has_citations": True, "true_count": true_f, "false_count": false_f, "score": round(score, 2)}


