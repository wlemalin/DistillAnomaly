import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from dpo_io.mm_inputs import _prep_mm_prompt, _record_teacher_text
from .citations import _auto_plain_text, parse_context_data, verify_citations_truthfulness
from .intervals import extract_predicted_intervals, score_candidates_affiliation_like
from .gt_interval_fix import maybe_prepend_gt_fixed_candidate


# ---------------------------------------------------------------------
# Citation parsing for controlled corruption (single edit)
# ---------------------------------------------------------------------

# Same "shape" as citations.py, but we also want spans + keep ':' spacing
_CIT_FIND_RE = re.compile(r"([\w\-_]+)@(\d+)\s*:\s*([\-]?\d+)")
_CIT_FULL_RE = re.compile(r"^([\w\-_]+)@(\d+)(\s*:\s*)([\-]?\d+)$")

_FEATURE_ALIASES = {
    "index": "value",
    "values": "value",
    "average": "moving_average",
}


def _norm_feature_name(raw: str) -> str:
    k = raw.strip()
    return _FEATURE_ALIASES.get(k.lower(), k)


def _extract_citations_with_spans(text: str) -> List[Dict[str, Any]]:
    """
    Retourne une liste de citations trouvées dans `text` avec:
      - feat_raw, feat_norm
      - ts (int), val (int)
      - mid (str) => l'espacement original autour de ':'
      - span (start,end) => pour remplacer exactement UNE occurrence
      - full (str) => texte matché
    """
    if not isinstance(text, str) or not text:
        return []

    out: List[Dict[str, Any]] = []
    for m in _CIT_FIND_RE.finditer(text):
        feat_raw, ts_s, val_s = m.group(1), m.group(2), m.group(3)
        full = m.group(0)

        # conserver l'espacement exact autour du ':'
        mid = ":"
        m2 = _CIT_FULL_RE.match(full)
        if m2:
            mid = m2.group(3)

        try:
            ts = int(ts_s)
            val = int(val_s)
        except Exception:
            continue

        out.append(
            {
                "feat_raw": feat_raw,
                "feat_norm": _norm_feature_name(feat_raw),
                "ts": ts,
                "val": val,
                "mid": mid,
                "span": (m.start(), m.end()),
                "full": full,
            }
        )
    return out


def _replace_one_span(text: str, span: Tuple[int, int], replacement: str) -> str:
    a, b = span
    return text[:a] + replacement + text[b:]


def _try_temporal_confusion(
    text: str,
    cite: Dict[str, Any],
    data_map: Dict[int, Dict[str, int]],
    *,
    max_offset: int,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Temporal confusion:
      - on garde feature + value
      - on décale timestamp vers un ts proche EXISTANT
      - et on impose que (feat, new_ts, val) soit FAUX (sinon on refuse)
    """
    ts = cite["ts"]
    val = cite["val"]
    feat_raw = cite["feat_raw"]
    feat_norm = cite["feat_norm"]
    mid = cite["mid"]

    deltas = [d for d in range(-max_offset, max_offset + 1) if d != 0]
    random.shuffle(deltas)

    for d in deltas:
        new_ts = ts + d
        row = data_map.get(new_ts)
        if row is None:
            continue
        if feat_norm not in row:
            continue

        # éviter le cas où le triplet resterait vrai
        if row[feat_norm] == val:
            continue

        repl = f"{feat_raw}@{new_ts}{mid}{val}"
        new_text = _replace_one_span(text, cite["span"], repl)
        meta = {
            "mode": "temporal_confusion",
            "delta": int(d),
            "original": cite["full"],
            "modified": repl,
        }
        return new_text, meta

    return None


def _try_feature_confusion(
    text: str,
    cite: Dict[str, Any],
    data_map: Dict[int, Dict[str, int]],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Feature confusion:
      - on garde timestamp + value
      - on change la feature vers une autre feature existante au même ts
      - et on impose que (new_feat, ts, val) soit FAUX (sinon on refuse)
    """
    ts = cite["ts"]
    val = cite["val"]
    feat_norm = cite["feat_norm"]
    mid = cite["mid"]

    row = data_map.get(ts)
    if row is None:
        return None

    alts = [k for k, v in row.items() if k != feat_norm and v != val]
    if not alts:
        return None

    random.shuffle(alts)
    new_feat = alts[0]

    repl = f"{new_feat}@{ts}{mid}{val}"
    new_text = _replace_one_span(text, cite["span"], repl)
    meta = {
        "mode": "feature_confusion",
        "original": cite["full"],
        "modified": repl,
        "swapped_to": new_feat,
    }
    return new_text, meta


def make_synthetic_rejected(
    chosen_text: str,
    data_map: Dict[int, Dict[str, int]],
    *,
    max_offset: int = 3,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Fabrique un rejected synthétique à partir de chosen_text en appliquant
    EXACTEMENT une corruption (sur une seule citation).

    Stratégie:
      - on tire un mode (temporal vs feature), mais fallback sur l'autre si ça échoue
      - on essaye plusieurs citations si besoin
      - on vérifie que le résultat a au moins 1 fausse citation (false_count > 0)
    """
    cites = _extract_citations_with_spans(chosen_text)
    if not cites:
        return None

    random.shuffle(cites)

    modes = ["temporal", "feature"]
    random.shuffle(modes)

    for mode in modes:
        for cite in cites:
            if mode == "temporal":
                res = _try_temporal_confusion(
                    chosen_text, cite, data_map, max_offset=max_offset
                )
            else:
                res = _try_feature_confusion(
                    chosen_text, cite, data_map
                )

            if res is None:
                continue

            rejected_text, meta = res
            ev = verify_citations_truthfulness(rejected_text, data_map)

            if ev.get("has_citations", False) and ev.get("false_count", 0) > 0:
                meta["verification"] = {
                    "true_count": int(ev.get("true_count", 0)),
                    "false_count": int(ev.get("false_count", 0)),
                    "score": float(ev.get("score", 0.0)),
                }
                return rejected_text, meta

    return None


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

@torch.no_grad()
def generate_candidates(
    model,
    tokenizer,
    mm_prompt: Dict[str, Any],
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    gen_kwargs = dict(
        input_ids=mm_prompt["input_ids"],
        attention_mask=mm_prompt["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_beams=1,
        num_return_sequences=num_candidates,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if mm_prompt.get("has_images", False):
        gen_kwargs["pixel_values"] = mm_prompt["pixel_values"]
        gen_kwargs["image_grid_thw"] = mm_prompt["image_grid_thw"]

    gen_ids = model.generate(**gen_kwargs)

    out: List[str] = []
    prompt_len = mm_prompt["prompt_len"]
    for j in range(gen_ids.shape[0]):
        cont = gen_ids[j, prompt_len:]
        txt = tokenizer.decode(cont, skip_special_tokens=True)
        out.append(txt)
    return out


# ---------------------------------------------------------------------
# Preferences builder (winner unchanged, loser synthetic)
# ---------------------------------------------------------------------

def build_preferences(
    model,
    tokenizer,
    image_processor,
    records: Sequence[Dict[str, Any]],
    img_tok_id: int,
    merge: int,
    start_id: int | None,
    end_id: int | None,
    st_model_path: str,   # conservé pour compat
    device: str,
    out_prefs_jsonl: str,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pref_strategy: str,   # conservé pour compat
    min_margin: float,    # conservé pour compat
    seed: int,
) -> int:
    os.makedirs(Path(out_prefs_jsonl).parent, exist_ok=True)

    random.seed(seed)
    model.eval()

    kept = 0
    fixed_count = 0
    tmp = out_prefs_jsonl + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            _ = _record_teacher_text(rec)

            mm_prompt = _prep_mm_prompt(
                rec=rec,
                image_processor=image_processor,
                tokenizer=tokenizer,
                img_tok_id=img_tok_id,
                merge=merge,
                start_id=start_id,
                end_id=end_id,
                device=device,
            )

            cands = generate_candidates(
                model=model,
                tokenizer=tokenizer,
                mm_prompt=mm_prompt,
                num_candidates=num_candidates,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            gt = rec.get("ground_truth", None)
            if gt is None:
                continue

            prompt_text = rec.get("input", "") or ""
            data_map = parse_context_data(prompt_text, plain_text=_auto_plain_text(prompt_text))
            if not data_map:
                continue

            cit_evals = [verify_citations_truthfulness(t, data_map) for t in cands]

            def is_perfect(ev: Dict[str, Any]) -> bool:
                return (
                    ev.get("has_citations", False)
                    and ev.get("false_count", 0) == 0
                    and ev.get("true_count", 0) > 0
                )

            perfect_idx = [j for j, ev in enumerate(cit_evals) if is_perfect(ev)]
            if not perfect_idx:
                continue

            affil_scores = score_candidates_affiliation_like(
                ground_truth_obj=gt,
                candidates=cands,
                end_inclusive=True,
                normalize="span",
            )

            if len(perfect_idx) == 1:
                chosen_idx = perfect_idx[0]
            else:
                chosen_idx = int(
                    max(
                        perfect_idx,
                        key=lambda j: (affil_scores[j], cit_evals[j].get("true_count", 0)),
                    )
                )

            chosen_text = cands[chosen_idx]
            chosen_cit = cit_evals[chosen_idx]
            chosen_score = float(affil_scores[chosen_idx])

            synth = make_synthetic_rejected(chosen_text, data_map, max_offset=3)
            if synth is None:
                continue

            rejected_text, rejected_meta = synth
            rejected_cit = verify_citations_truthfulness(rejected_text, data_map)

            if not rejected_cit.get("has_citations", False) or rejected_cit.get("false_count", 0) <= 0:
                continue

            rejected_score = float(
                score_candidates_affiliation_like(
                    ground_truth_obj=gt,
                    candidates=[rejected_text],
                    end_inclusive=True,
                    normalize="span",
                )[0]
            )

            obj = {
                "input": rec.get("input", ""),
                "image_paths": rec.get("image_paths", []),
                "ground_truth": gt,

                "candidates": cands,
                "candidate_affil_scores": affil_scores,
                "candidate_citation_eval": cit_evals,

                "chosen": chosen_text,
                "rejected": rejected_text,

                "chosen_score": chosen_score,
                "rejected_score": rejected_score,

                "chosen_citation_true": int(chosen_cit.get("true_count", 0)),
                "chosen_citation_false": int(chosen_cit.get("false_count", 0)),
                "rejected_citation_true": int(rejected_cit.get("true_count", 0)),
                "rejected_citation_false": int(rejected_cit.get("false_count", 0)),

                "chosen_pred_intervals": extract_predicted_intervals(chosen_text),
                "rejected_pred_intervals": extract_predicted_intervals(rejected_text),

                "rejected_is_synthetic": True,
                "rejected_corruption": rejected_meta,
                "rejected_source": {
                    "chosen_candidate_idx": int(chosen_idx),
                },
            }

            if "pattern_type" in rec:
                obj["pattern_type"] = rec["pattern_type"]

            # -----------------------------------------------------------------
            # Upstream GT-interval fix:
            # prepend a corrected version of the best candidate if possible.
            # In BT mode, we set chosen_score ~= 1.0 for the injected top item.
            # -----------------------------------------------------------------
            obj, did_fix = maybe_prepend_gt_fixed_candidate(
                obj,
                source_record=rec,
                ordered_scores_prefix_value=1.0,
            )
            fixed_count += int(did_fix)

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

            if (i + 1) % 20 == 0:
                logging.info(
                    "Préférences: %d/%d traités • kept=%d • gt_fixed=%d",
                    i + 1,
                    len(records),
                    kept,
                    fixed_count,
                )

    os.replace(tmp, out_prefs_jsonl)
    logging.info(
        "Saved preferences to %s • kept=%d • gt_fixed=%d",
        out_prefs_jsonl,
        kept,
        fixed_count,
    )
    model.train()
    return kept


# ---------------------------------------------------------------------
# DPO-PL helpers
# ---------------------------------------------------------------------

def _citation_truth_counts_local(
    text: str,
    data_map: Dict[int, Dict[str, int]],
) -> Dict[str, Any]:
    """
    Comptage 'local' basé sur _extract_citations_with_spans + data_map.
    Compte chaque occurrence (donc si la même citation apparaît 2 fois, elle compte 2).
    """
    cites = _extract_citations_with_spans(text)
    true_count = 0
    false_count = 0

    for c in cites:
        ts = c["ts"]
        feat = c["feat_norm"]
        val = c["val"]
        row = data_map.get(ts)
        if row is not None and row.get(feat, None) == val:
            true_count += 1
        else:
            false_count += 1

    return {
        "has_citations": bool(cites),
        "true_count": int(true_count),
        "false_count": int(false_count),
        "num_citations": int(len(cites)),
    }


def _corrupt_one_true_citation_incremental(
    text: str,
    data_map: Dict[int, Dict[str, int]],
    *,
    max_offset: int,
    before_false: int,
    rng: random.Random,
    max_tries: int = 200,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Applique UNE corruption sur UNE citation actuellement vraie,
    et exige after_false == before_false + 1 (donc progression stricte +1 erreur).
    """
    for _ in range(max_tries):
        cites = _extract_citations_with_spans(text)
        if not cites:
            return None

        true_cites = []
        for c in cites:
            row = data_map.get(c["ts"])
            if row is not None and row.get(c["feat_norm"], None) == c["val"]:
                true_cites.append(c)

        if not true_cites:
            return None

        rng.shuffle(true_cites)
        modes = ["temporal", "feature"]
        rng.shuffle(modes)

        for mode in modes:
            for cite in true_cites:
                if mode == "temporal":
                    res = _try_temporal_confusion(
                        text, cite, data_map, max_offset=max_offset
                    )
                else:
                    res = _try_feature_confusion(text, cite, data_map)

                if res is None:
                    continue

                cand_text, meta = res
                after = _citation_truth_counts_local(cand_text, data_map)

                if after["has_citations"] and after["false_count"] == before_false + 1:
                    meta = dict(meta)
                    meta["counts_after"] = {
                        "true_count": after["true_count"],
                        "false_count": after["false_count"],
                        "num_citations": after["num_citations"],
                    }
                    return cand_text, meta

    return None


def make_ordered_outputs_by_error_count(
    y0_text: str,
    data_map: Dict[int, Dict[str, int]],
    *,
    max_level: int,
    max_offset: int = 3,
    seed: int = 0,
) -> Optional[Tuple[List[str], List[Dict[str, Any]]]]:
    """
    Construit (y0, y1, y2, ...) où l'indice k == nombre de citations fausses.
    - y0 doit être parfait (0 faux, >=1 vrai).
    - chaque étape ajoute exactement +1 erreur.
    """
    rng = random.Random(seed)

    base = _citation_truth_counts_local(y0_text, data_map)
    if (not base["has_citations"]) or base["true_count"] <= 0 or base["false_count"] != 0:
        return None

    ys: List[str] = [y0_text]
    metas: List[Dict[str, Any]] = [{
        "mode": "none",
        "level": 0,
        "counts_after": {
            "true_count": base["true_count"],
            "false_count": base["false_count"],
            "num_citations": base["num_citations"],
        },
    }]

    current = y0_text
    before_false = 0

    for level in range(1, max_level + 1):
        step = _corrupt_one_true_citation_incremental(
            current,
            data_map,
            max_offset=max_offset,
            before_false=before_false,
            rng=rng,
        )
        if step is None:
            break

        new_text, meta = step
        meta["level"] = level

        ys.append(new_text)
        metas.append(meta)

        current = new_text
        before_false += 1

    if len(ys) < 2:
        return None

    return ys, metas


def build_preferences_ordered_list(
    model,
    tokenizer,
    image_processor,
    records: Sequence[Dict[str, Any]],
    img_tok_id: int,
    merge: int,
    start_id: int | None,
    end_id: int | None,
    st_model_path: str,   # conservé pour compat
    device: str,
    out_prefs_jsonl: str,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    *,
    max_error_level: int = 5,
    max_offset: int = 3,
) -> int:
    os.makedirs(Path(out_prefs_jsonl).parent, exist_ok=True)

    random.seed(seed)
    model.eval()

    kept = 0
    fixed_count = 0
    tmp = out_prefs_jsonl + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            _ = _record_teacher_text(rec)

            mm_prompt = _prep_mm_prompt(
                rec=rec,
                image_processor=image_processor,
                tokenizer=tokenizer,
                img_tok_id=img_tok_id,
                merge=merge,
                start_id=start_id,
                end_id=end_id,
                device=device,
            )

            cands = generate_candidates(
                model=model,
                tokenizer=tokenizer,
                mm_prompt=mm_prompt,
                num_candidates=num_candidates,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            gt = rec.get("ground_truth", None)
            if gt is None:
                continue

            prompt_text = rec.get("input", "") or ""
            data_map = parse_context_data(prompt_text, plain_text=_auto_plain_text(prompt_text))
            if not data_map:
                continue

            cit_evals = [verify_citations_truthfulness(t, data_map) for t in cands]

            def is_perfect(ev: Dict[str, Any]) -> bool:
                return (
                    ev.get("has_citations", False)
                    and ev.get("false_count", 0) == 0
                    and ev.get("true_count", 0) > 0
                )

            perfect_idx = [j for j, ev in enumerate(cit_evals) if is_perfect(ev)]
            if not perfect_idx:
                continue

            affil_scores = score_candidates_affiliation_like(
                ground_truth_obj=gt,
                candidates=cands,
                end_inclusive=True,
                normalize="span",
            )

            if len(perfect_idx) == 1:
                chosen_idx = perfect_idx[0]
            else:
                chosen_idx = int(
                    max(
                        perfect_idx,
                        key=lambda j: (affil_scores[j], cit_evals[j].get("true_count", 0)),
                    )
                )

            chosen_text = cands[chosen_idx]

            ordered = make_ordered_outputs_by_error_count(
                y0_text=chosen_text,
                data_map=data_map,
                max_level=max_error_level,
                max_offset=max_offset,
                seed=seed + i,
            )
            if ordered is None:
                continue

            ys, y_metas = ordered

            y_affil = score_candidates_affiliation_like(
                ground_truth_obj=gt,
                candidates=ys,
                end_inclusive=True,
                normalize="span",
            )

            obj = {
                "input": rec.get("input", ""),
                "image_paths": rec.get("image_paths", []),
                "ground_truth": gt,

                "candidates": cands,
                "candidate_affil_scores": affil_scores,
                "candidate_citation_eval": cit_evals,

                # compat pairwise: chosen/rejected = (y0,y1)
                "chosen": ys[0],
                "rejected": ys[1],

                # ordered list
                "ordered_outputs": ys,
                "ordered_metas": y_metas,
                "ordered_affil_scores": [float(x) for x in y_affil],
                "ordered_false_counts": list(range(len(ys))),

                "chosen_pred_intervals": extract_predicted_intervals(ys[0]),
                "rejected_pred_intervals": extract_predicted_intervals(ys[1]),
            }

            if "pattern_type" in rec:
                obj["pattern_type"] = rec["pattern_type"]

            # -----------------------------------------------------------------
            # Upstream GT-interval fix:
            # prepend a corrected version of y0 if possible.
            # This keeps list-wise compatibility and updates chosen/rejected.
            # -----------------------------------------------------------------
            obj, did_fix = maybe_prepend_gt_fixed_candidate(
                obj,
                source_record=rec,
            )
            fixed_count += int(did_fix)

            # logs / compat pairwise supplémentaires
            if isinstance(obj.get("ordered_affil_scores"), list) and len(obj["ordered_affil_scores"]) >= 2:
                obj["chosen_score"] = float(obj["ordered_affil_scores"][0])
                obj["rejected_score"] = float(obj["ordered_affil_scores"][1])

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

            if (i + 1) % 20 == 0:
                logging.info(
                    "Préférences (ordered): %d/%d traités • kept=%d • gt_fixed=%d",
                    i + 1,
                    len(records),
                    kept,
                    fixed_count,
                )

    os.replace(tmp, out_prefs_jsonl)
    logging.info(
        "Saved ordered preferences to %s • kept=%d • gt_fixed=%d",
        out_prefs_jsonl,
        kept,
        fixed_count,
    )
    model.train()
    return kept
