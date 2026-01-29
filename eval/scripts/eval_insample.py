#!/usr/bin/env python
"""
Standalone 'APRES' evaluation script for a Qwen2.5-VL LoRA checkpoint.

Usage:
python vl_lora_eval_apres.py \
  --base_model_dir ../hf_models/Qwen2.5-VL-3B-Instruct \
  --adapter_dir ./qwen2.5-vl-3b-ft-lora-eval \
  --data_json /path/to/all_samples.jsonl \
  --val_ratio 0.10 \
  --seed 42 \
  --st_model /home2020/home/beta/wlaemlin/hf_models/all_MiniLM-L6-v2 \
  --out_json ./evaluation_apres.jsonl

Notes:
- Loads the base Qwen2.5-VL model and applies LoRA adapters from --adapter_dir.
- Re-splits the provided JSONL deterministically (same seed/ratio as training).
- Generates outputs on the eval split, computes cosine similarity vs teacher,
  and saves results to a JSONL (one record per line) with fields:
  input, output, teacher_output, image_paths, optional ground_truth,
  optional pattern_type, and score (cosine).
"""

import argparse, json, os, sys, math, logging, warnings
from pathlib import Path
from typing import Dict, Sequence, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from peft import PeftModel
from sentence_transformers import SentenceTransformer
from datasets import Dataset as HFDataset, DatasetDict

# ----------------------------- CLI ----------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument
    g("--base_model_dir", required=True, help="Path to the base Qwen2.5-VL model directory")
    g("--adapter_dir",    required=True, help="Path to the LoRA checkpoint directory (with adapter_model.safetensors)")
    g("--data_json",      required=True, help="Single JSONL with all samples; will be split train/eval")
    g("--val_ratio",      type=float, default=0.10, help="Fraction for eval/test split")
    g("--seed",           type=int,   default=42, help="Split seed to reproduce training split")
    g("--st_model", default="/home2020/home/beta/wlaemlin/hf_models/all_MiniLM-L6-v2",
      help="SentenceTransformer path or model ID (offline path recommended)")
    g("--out_json",       default="./evaluation_apres.jsonl", help="Where to write the JSONL results")
    g("--debug",          action="store_true")
    return p

# --------------------------- Logging --------------------------------------

def setup_logging(debug: bool):
    lvl = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------------------------- Data ----------------------------------------

def load_jsonl_dataset(path: str) -> HFDataset:
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    return HFDataset.from_list(lines)

# --------------------------- Helpers --------------------------------------

def _even_grid_from_npatch(n: int) -> tuple[int, int]:
    root = int(math.sqrt(n))
    for w in range(root, 1, -1):
        if n % w == 0:
            h = n // w
            if (h % 2 == 0) and (w % 2 == 0):
                return h, w
    for w in (2, 4, 6, 8):
        if n % w == 0 and (n // w) % 2 == 0:
            return n // w, w
    raise ValueError(f"Cannot find even factors for {n} patches")

def _resolve_image_paths(names: list[str]) -> list[str]:
    """
    Match the path resolution used during training:
    if relative, resolve against repo root (two levels above this file).
    """
    fixed = []
    base = (Path(__file__).resolve().parent / "../../").resolve()
    for n in names:
        p = Path(n)
        fixed.append(str((base / n).resolve()) if not p.is_absolute() else str(p))
    return fixed

@torch.no_grad()
def _prep_mm_inputs_for_generation(
    rec: Dict[str, Any],
    image_processor,
    tokenizer,
    img_tok_id: int,
    merge: int,
    start_id: int | None,
    end_id: int | None,
    device: str,
):
    """
    Build a single-sample multimodal input for generation:
    [<vision tokens>] + [prompt tokens], using chat template if available.
    """
    # --- images ---
    #names = rec["image_paths"] if isinstance(rec["image_paths"], list) else [rec["image_paths"]]
    names = rec["image_paths"] if isinstance(rec["image_paths"], list) else []
    names = _resolve_image_paths(names)

    pix_list, grid_list, seg_list = [], [], []

    def _load_one(fname: str):
        img = Image.open(fname).convert("RGB")
        pix = image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)  # (n_patch, hid)
        n_patch, hid = pix.shape
        h_grid, w_grid = _even_grid_from_npatch(n_patch)
        target = h_grid * w_grid
        if target > n_patch:
            pad = torch.zeros(target - n_patch, hid, dtype=pix.dtype)
            pix = torch.cat([pix, pad], 0)
        n_feat = (h_grid // merge) * (w_grid // merge)
        seg = torch.full((1, n_feat), img_tok_id, dtype=torch.long)
        if start_id is not None and end_id is not None:
            seg = torch.cat([torch.tensor([[start_id]]), seg, torch.tensor([[end_id]])], 1)
        return pix, torch.tensor([1, h_grid, w_grid]), seg

    for n in names:
        p, g, s = _load_one(n)
        pix_list.append(p); grid_list.append(g); seg_list.append(s)

    pixel_values   = torch.cat(pix_list, 0).unsqueeze(0).to(device)  # (1, Σpatch, hid)
    image_grid_thw = torch.stack(grid_list).to(device)               # (n_img, 3)  <-- 2D (expected)
    img_tok_tensor = torch.cat(seg_list, 1).to(device)               # (1, Σfeat)

    # --- text (chat template preferred) ---
    prompt_text = rec.get("input") or ""
    if hasattr(tokenizer, "apply_chat_template"):
        chat = [{"role": "user", "content": prompt_text}]
        prompt_ids = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(device)
        prompt_msk = torch.ones_like(prompt_ids)
    else:
        q = tokenizer(prompt_text + "\n", return_tensors="pt")
        prompt_ids  = q["input_ids"].to(device)
        prompt_msk  = q["attention_mask"].to(device)

    input_ids = torch.cat([img_tok_tensor, prompt_ids], 1)           # (1, feats + prompt_len)
    attn_mask = torch.cat([torch.ones_like(img_tok_tensor), prompt_msk], 1)

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "prompt_len": input_ids.shape[1],
    }

def _record_teacher_text(rec: Dict[str, Any]) -> str:
    if rec.get("output") is not None:
        return str(rec["output"])
    oj = rec.get("output_json")
    if oj is not None:
        return json.dumps(oj, ensure_ascii=False)
    raise KeyError("Record missing 'output' and 'output_json'")

@torch.no_grad()
def evaluate_split_to_jsonl(
    model,
    tokenizer,
    image_processor,
    records: Sequence[Dict[str, Any]],
    img_tok_id: int,
    merge: int,
    start_id: int | None,
    end_id: int | None,
    out_path: str,
    st_model_path: str,
    device: str,
    max_new_tokens: int = 512,
) -> float:
    """
    Generate per-sample, save JSONL with cosine scores, and return the average score.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    st_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    st = SentenceTransformer(st_model_path, device=st_device)

    gen_texts: list[str] = []
    ref_texts: list[str] = []

    model.eval()

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            mm = _prep_mm_inputs_for_generation(
                rec, image_processor, tokenizer,
                img_tok_id, merge, start_id, end_id, device
            )

            gen_ids = model.generate(
                input_ids=mm["input_ids"],
                attention_mask=mm["attention_mask"],
                pixel_values=mm["pixel_values"],
                image_grid_thw=mm["image_grid_thw"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            cont_ids = gen_ids[0, mm["prompt_len"]:]
            gen_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
            ref_text = _record_teacher_text(rec)

            gen_texts.append(gen_text)
            ref_texts.append(ref_text)

            out_obj = {
                "input": rec.get("input", ""),
                "output": gen_text,
                "teacher_output": ref_text,
                "image_paths": rec["image_paths"],
            }
            if "ground_truth" in rec: out_obj["ground_truth"] = rec["ground_truth"]
            if "pattern_type" in rec: out_obj["pattern_type"] = rec["pattern_type"]

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            if (i + 1) % 20 == 0:
                logging.info("Evaluated %d/%d samples", i + 1, len(records))

    emb_gen  = st.encode(gen_texts, convert_to_tensor=True, normalize_embeddings=True, device=st_device)
    emb_ref  = st.encode(ref_texts, convert_to_tensor=True, normalize_embeddings=True, device=st_device)
    cs = F.cosine_similarity(emb_gen, emb_ref).tolist()
    avg_score = float(sum(cs) / max(1, len(cs)))

    final_path = out_path
    with open(tmp_path, "r", encoding="utf-8") as fin, open(final_path, "w", encoding="utf-8") as fout:
        for s, line in zip(cs, fin):
            obj = json.loads(line)
            obj["score"] = float(s)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.remove(tmp_path)

    logging.info("Saved evaluation to %s (avg cosine=%.4f)", final_path, avg_score)
    model.train()
    return avg_score

# ---------------------------- Main ----------------------------------------

if __name__ == "__main__":
    args = build_parser().parse_args()
    setup_logging(args.debug)

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pinned memory.*")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Running on %s • Torch %s", device, torch.__version__)

    # Load base model and apply LoRA adapters
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        device_map="auto" if device == "cuda" else None,
        dtype="auto",  #torch_dtype="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        is_trainable=False,
    )

    # Prefer tokenizer / processor from adapter_dir (has chat_template.json), fallback to base
    tok_src = args.adapter_dir if (Path(args.adapter_dir) / "tokenizer_config.json").exists() else args.base_model_dir
    proc_src = args.adapter_dir if (Path(args.adapter_dir) / "preprocessor_config.json").exists() else args.base_model_dir

    tokenizer = AutoTokenizer.from_pretrained(tok_src, padding_side="left", local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained(proc_src, local_files_only=True)

    # Model config bits
    IMAGE_PATCH_ID = model.base_model.config.image_token_id
    MERGE = getattr(model.base_model.config.vision_config, "spatial_merge_size", 2)
    start_id  = getattr(model.base_model.config, "vision_start_token_id", None)
    end_id    = getattr(model.base_model.config, "vision_end_token_id", None)

    # Load & split dataset (deterministic)
    ds_full: HFDataset = load_jsonl_dataset(args.data_json)
    split: DatasetDict = ds_full.train_test_split(test_size=args.val_ratio, seed=args.seed)
    eval_records = split["test"]  # HF Dataset; indexable sequence of dicts
    logging.info("Eval samples: %d", len(eval_records))

    out_json = args.out_json
    os.makedirs(Path(out_json).parent, exist_ok=True)

    score = evaluate_split_to_jsonl(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        records=eval_records,
        img_tok_id=IMAGE_PATCH_ID,
        merge=MERGE,
        start_id=start_id,
        end_id=end_id,
        out_path=out_json,
        st_model_path=args.st_model,
        device=device,
        max_new_tokens=512,
    )
    logging.info("APRES evaluation cosine average: %.4f", score)
