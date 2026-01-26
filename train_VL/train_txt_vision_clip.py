#!/usr/bin/env python
"""
LoRA fine-tuning on Qwen2.5-VL with optional InfoNCE loss.

  Vision tokens <-> Answer tokens  (CLIP-like in-batch InfoNCE)

Notes:
  - --alpha_itc is kept for backward compatibility: if provided and both
    --alpha_itc_vis and --alpha_itc_ts are 0, then alpha_itc_vis = alpha_itc.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (AutoImageProcessor, AutoTokenizer,
                          Qwen2_5_VLForConditionalGeneration, Trainer,
                          TrainingArguments)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# CLI
# -----------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument
    # paths
    g("--model_dir", required=True)
    g("--data_json", required=True,
      help="Single JSONL with all samples; will be split into train/eval")
    g("--out_dir",   required=True,
      help="Directory to save checkpoints and LoRA adapters")
    # split params
    g("--val_ratio", type=float, default=0.10, help="Fraction for eval/test split")
    g("--seed",      type=int,   default=42)
    # training
    g("--epochs",     type=int,   default=3)
    g("--lr",         type=float, default=1e-5)
    g("--batch_size", type=int,   default=1)
    g("--grad_accum", type=int,   default=4)
    g("--fp16",       action="store_true", help="Enable fp16")
    g("--bf16",       action="store_true", help="Enable bf16")
    g("--debug",      action="store_true")

    # Backward-compat flag (maps to alpha_itc_vis if new alphas are not set)
    g("--alpha_itc", type=float, default=0.0,
      help="(compat) Weight for vision-text contrastive loss (0=off)")

    # New: separate weights
    g("--alpha_itc_vis", type=float, default=0.0,
      help="Weight for vision<->answer InfoNCE (0=off)")
    g("--alpha_itc_ts",  type=float, default=0.0,
      help="Weight for time-series<->answer InfoNCE (0=off)")
    g("--itc_temp",      type=float, default=0.1, help="InfoNCE temperature")

    # kept for compatibility (optional)
    g("--use_proj_heads", action="store_true",
      help="Use small projection heads (optional)")

    # resume
    g("--resume", action="store_true",
      help="Resume training from the second most recent *valid* checkpoint in out_dir "
           "(falls back to most recent if only one).")
    return p


def setup_logging(debug: bool):
    lvl = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_jsonl_dataset(path: str) -> HFDataset:
    with open(path, "r") as f:
        lines = [json.loads(l) for l in f]
    return HFDataset.from_list(lines)


def _list_valid_checkpoints(out_dir: str) -> List[Path]:
    """Return folders that can resume training."""
    base = Path(out_dir)
    cks = [p for p in base.glob("checkpoint-*") if p.is_dir()]
    cks.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    def is_valid(p: Path) -> bool:
        has_state = (p / "trainer_state.json").exists()
        has_weights = (
            p / "adapter_model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        has_opt = (p / "optimizer.pt").exists() or (p /
                                                    "optimizer.bin").exists()
        return has_state and (has_weights or has_opt)

    return [p for p in cks if is_valid(p)]


def _pick_resume_checkpoint(out_dir: str) -> str | None:
    cks = _list_valid_checkpoints(out_dir)
    if not cks:
        return None
    if len(cks) >= 2:
        return str(cks[1])
    return str(cks[0])

# -----------------------------
# Helpers
# -----------------------------


def _even_grid_from_npatch(n: int) -> Tuple[int, int]:
    """Even (h,w) such that h*w=n."""
    root = int(math.isqrt(n))
    for w in range(root, 1, -1):
        if n % w == 0:
            h = n // w
            if (h % 2 == 0) and (w % 2 == 0):
                return h, w
    for w in (2, 4, 6, 8):
        if n % w == 0 and (n // w) % 2 == 0:
            return n // w, w
    raise ValueError(f"Cannot find even factors for {n} patches")


def _concat_with_truncation_keep_images_and_answer_and_mask(
    img_tok_tensor: torch.Tensor,     # (1, I)
    prompt_ids: torch.Tensor,         # (1, P)
    answer_ids: torch.Tensor,         # (1, A)
    prompt_focus_mask: torch.Tensor,  # (1, P) 0/1
    pad_id: int,
    max_len: int | None,
    debug: bool = False,
):
    """Concat [IMG][PROMPT][ANSWER] and return (input_ids,attn,labels,itc_mask)."""
    input_ids = torch.cat(
        [img_tok_tensor, prompt_ids, answer_ids], dim=1)  # (1, L)
    attn_mask = torch.ones_like(input_ids)

    labels = torch.cat([
        torch.full_like(img_tok_tensor, -100),
        torch.full_like(prompt_ids, -100),
        answer_ids
    ], dim=1)

    itc_src_mask = torch.cat([
        torch.zeros_like(img_tok_tensor, dtype=torch.long),
        prompt_focus_mask.long(),
        torch.zeros_like(answer_ids, dtype=torch.long),
    ], dim=1)

    if max_len is None or input_ids.size(1) <= max_len:
        return (
            input_ids.squeeze(0),
            attn_mask.squeeze(0),
            labels.squeeze(0),
            itc_src_mask.squeeze(0),
        )

    L = input_ids.size(1)
    overflow = L - max_len

    prompt_len = prompt_ids.size(1)
    ans_len = answer_ids.size(1)

    trim_from_prompt = min(overflow, prompt_len)
    remaining = overflow - trim_from_prompt

    trim_from_answer = 0
    if remaining > 0:
        trim_from_answer = min(remaining, max(0, ans_len - 1))
        if debug:
            logging.warning(
                "[TRUNC] Image+Answer exceed max. Trimmed %d from answer start.", trim_from_answer)

    new_prompt_ids = prompt_ids[:,
                                trim_from_prompt:] if trim_from_prompt > 0 else prompt_ids
    new_prompt_mask = prompt_focus_mask[:,
                                        trim_from_prompt:] if trim_from_prompt > 0 else prompt_focus_mask
    new_answer_ids = answer_ids[:,
                                trim_from_answer:] if trim_from_answer > 0 else answer_ids

    input_ids = torch.cat(
        [img_tok_tensor, new_prompt_ids, new_answer_ids], dim=1)
    attn_mask = torch.ones_like(input_ids)

    labels = torch.cat([
        torch.full_like(img_tok_tensor, -100),
        torch.full_like(new_prompt_ids, -100),
        new_answer_ids
    ], dim=1)

    itc_src_mask = torch.cat([
        torch.zeros_like(img_tok_tensor, dtype=torch.long),
        new_prompt_mask.long(),
        torch.zeros_like(new_answer_ids, dtype=torch.long),
    ], dim=1)

    if debug:
        kept = input_ids.size(1)
        logging.debug("[TRUNC] kept=%d of max=%s (overflow=%d)",
                      kept, str(max_len), L - kept)

    return (
        input_ids.squeeze(0),
        attn_mask.squeeze(0),
        labels.squeeze(0),
        itc_src_mask.squeeze(0),
    )


def mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp_min(1e-6)


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric in-batch InfoNCE."""
    logits = z_b @ z_a.T / max(tau, 1e-6)  # (B,B)
    labels = torch.arange(z_b.size(0), device=z_b.device)
    return 0.5 * F.cross_entropy(logits, labels) + 0.5 * F.cross_entropy(logits.T, labels)


def _extract_focus_substring(prompt_text: str, marker: str = "Columns per line") -> str | None:
    """Text after marker (for TS contrastive)."""
    if marker not in prompt_text:
        return None
    after = prompt_text.split(marker, 1)[1]
    nl = after.find("\n")
    focus = after[nl + 1:] if nl >= 0 else ""
    focus = focus.strip("\n")
    return focus if focus.strip() else None


def _find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> int:
    """First index of needle in haystack."""
    K = needle.numel()
    P = haystack.numel()
    if K == 0 or P < K:
        return -1
    for s in range(0, P - K + 1):
        if torch.equal(haystack[s:s+K], needle):
            return s
    return -1

# -----------------------------
# Dataset (builds vision+text sequence)
# -----------------------------


class QwenVLDataset(Dataset):
    """
    Produces:
      pixel_values:   (Σpatch, hid)
      image_grid_thw: (n_img, 3)
      input_ids, attention_mask, labels: [IMG][PROMPT][ANSWER]
      itc_src_mask: 0/1 mask selecting time-series tokens inside prompt portion
    """

    def __init__(
        self,
        image_processor,
        tokenizer,
        img_tok_id: int,
        merge: int,
        start_id: int | None,
        end_id: int | None,
        max_len: int | None,
        debug: bool,
        records: Sequence[Dict[str, Any]] | None = None,
        jsonl_path: str | None = None,
    ):
        if records is not None:
            self.recs = list(records)
        elif jsonl_path is not None:
            self.recs = [json.loads(l) for l in open(jsonl_path)]
        else:
            raise ValueError("Provide either records or jsonl_path")

        self.proc = image_processor
        self.tok = tokenizer

        self.img_tok_id = img_tok_id
        self.merge = merge
        self.start_id = start_id
        self.end_id = end_id
        self.max_len = max_len
        self.debug = debug

        # FIX: Dynamically set the data root based on the JSONL file location
        if jsonl_path is not None:
            self.data_root = Path(jsonl_path).resolve().parent
        else:
            # Fallback to current working directory if records are passed directly
            self.data_root = Path.cwd()

    def __len__(self):
        return len(self.recs)

    def _load_one_image(self, fname: str):
        path = Path(fname)
        
        # FIX: Use the dynamic data_root instead of a hardcoded string
        if not path.is_absolute():
            path = (self.data_root / path).resolve()
        
        if self.debug:
            logging.debug(f"Attempting to load image from: {path}")

        if not path.exists():
            raise FileNotFoundError(f"Image not found at resolved path: {path}")

        img = Image.open(path).convert("RGB")

        pix = self.proc(img, return_tensors="pt")[
            "pixel_values"].squeeze(0)  # (n_patch, hid)
        n_patch, hid = pix.shape

        h_grid, w_grid = _even_grid_from_npatch(n_patch)
        target = h_grid * w_grid
        if target > n_patch:
            pad = torch.zeros(target - n_patch, hid, dtype=pix.dtype)
            pix = torch.cat([pix, pad], 0)

        n_feat = (h_grid // self.merge) * (w_grid // self.merge)

        seg = torch.full((1, n_feat), self.img_tok_id, dtype=torch.long)
        if (self.start_id is not None) and (self.end_id is not None):
            seg = torch.cat([torch.tensor([[self.start_id]]),
                            seg, torch.tensor([[self.end_id]])], dim=1)

        return pix, torch.tensor([1, h_grid, w_grid]), seg

    def __getitem__(self, idx):
        rec = self.recs[idx]
        names = rec["image_paths"] if isinstance(rec["image_paths"], list) else [
            rec["image_paths"]]

        pix_list: List[torch.Tensor] = []
        grid_list: List[torch.Tensor] = []
        seg_list:  List[torch.Tensor] = []
        for n in names:
            p, g, s = self._load_one_image(n)
            pix_list.append(p)
            grid_list.append(g)
            seg_list.append(s)

        pixel_values = torch.cat(pix_list, 0)
        image_grid_thw = torch.stack(grid_list, 0)
        img_tok_tensor = torch.cat(seg_list, 1)

        prompt_text = rec.get("input") or ""
        answer_text = rec.get("output")
        if answer_text is None:
            oj = rec.get("output_json")
            if oj is not None:
                answer_text = json.dumps(oj, ensure_ascii=False)
            else:
                raise KeyError(
                    f"No 'output' or 'output_json' for sample idx {idx}")

        user_msgs = [{"role": "user", "content": prompt_text}]
        prompt_ids = self.tok.apply_chat_template(
            user_msgs, tokenize=True, add_generation_prompt=False, return_tensors="pt",
        )  # (1,P)

        assistant_msgs = [{"role": "assistant", "content": answer_text}]
        answer_ids = self.tok.apply_chat_template(
            assistant_msgs, tokenize=True, add_generation_prompt=False, return_tensors="pt",
        )  # (1,A)

        # Build focus mask over prompt_ids: select only tokens after "Columns per line"
        prompt_focus_mask = torch.zeros_like(
            prompt_ids, dtype=torch.long)  # (1,P)

        focus_text = _extract_focus_substring(
            prompt_text, marker="Columns per line")
        if focus_text is not None:
            focus_ids = self.tok(
                focus_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]  # (K,)

            start = _find_subsequence(prompt_ids[0], focus_ids)
            if start >= 0:
                prompt_focus_mask[0, start:start + focus_ids.numel()] = 1
            else:
                if self.debug:
                    logging.warning(
                        "[ITC-TS] Could not locate focus substring tokens inside prompt_ids (idx=%d). "
                        "TS src mask will be empty (fallback may be used).", idx
                    )

        input_ids, attn_mask, labels, itc_src_mask = _concat_with_truncation_keep_images_and_answer_and_mask(
            img_tok_tensor=img_tok_tensor,
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            prompt_focus_mask=prompt_focus_mask,
            pad_id=self.tok.pad_token_id,
            max_len=self.max_len,
            debug=self.debug,
        )

        return {
            "pixel_values":    pixel_values,
            "image_grid_thw":  image_grid_thw,
            "input_ids":       input_ids,
            "attention_mask":  attn_mask,
            "labels":          labels,
            "itc_src_mask":    itc_src_mask,
        }

# -----------------------------
# BatchSampler: batches with diverse pattern_type
# -----------------------------


class DiversePatternTypeBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: QwenVLDataset, batch_size: int, drop_last: bool = False):
        """Each batch contains different pattern_types (needed for contrastive)."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.by_type: Dict[Any, List[int]] = {}
        for idx, rec in enumerate(self.dataset.recs):
            pt = rec.get("pattern_type", None)
            if pt is None:
                raise KeyError(f"Sample idx {idx} has no 'pattern_type' key")
            self.by_type.setdefault(pt, []).append(idx)

        self.types: List[Any] = list(self.by_type.keys())

    def __iter__(self):
        pools = {t: idxs.copy() for t, idxs in self.by_type.items()}
        for idxs in pools.values():
            random.shuffle(idxs)

        active_types = [t for t in self.types if len(pools[t]) > 0]

        while len(active_types) > 0:
            random.shuffle(active_types)
            chosen_types = active_types[: self.batch_size]

            batch: List[int] = []
            to_remove: List[Any] = []

            for t in chosen_types:
                if len(pools[t]) == 0:
                    to_remove.append(t)
                    continue
                batch.append(pools[t].pop())
                if len(pools[t]) == 0:
                    to_remove.append(t)

            for t in to_remove:
                if t in active_types:
                    active_types.remove(t)

            if len(batch) == 0:
                break
            if len(batch) < self.batch_size and self.drop_last:
                continue

            yield batch

    def __len__(self) -> int:
        total = sum(len(v) for v in self.by_type.values())
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)

# -----------------------------
# Collator
# -----------------------------


def make_collate_fn(tokenizer):
    """Right-pad sequences and concat images."""
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch: List[Dict[str, torch.Tensor]]):
        ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
        msk = torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
        lbls = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=-100)
        itc_src = torch.nn.utils.rnn.pad_sequence(
            [b["itc_src_mask"] for b in batch], batch_first=True, padding_value=0)

        pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
        image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

        return {
            "input_ids": ids,
            "attention_mask": msk,
            "labels": lbls,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "itc_src_mask": itc_src,
        }
    return collate_fn

# -----------------------------
# Trainer with optional ITC losses + custom sampler
# -----------------------------


class VLTrainer(Trainer):
    """Adds vision-ts contrastive losses and diverse sampler."""

    def get_train_dataloader(self):
        # enable diverse sampler if ANY ITC is enabled
        a_vis = getattr(self.args, "alpha_itc_vis", 0.0)
        a_ts = getattr(self.args, "alpha_itc_ts", 0.0)
        if (a_vis <= 0.0) and (a_ts <= 0.0):
            return super().get_train_dataloader()

        train_dataset = self.train_dataset
        if train_dataset is None:
            raise ValueError("train_dataset is None")

        batch_size = self.args.per_device_train_batch_size
        batch_sampler = DiversePatternTypeBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
            drop_last=self.args.dataloader_drop_last,
        )

        return DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            output_hidden_states=True,
            return_dict=True,
        )
        lm_loss = out.loss

        a_vis = getattr(self.args, "alpha_itc_vis", 0.0)
        a_ts = getattr(self.args, "alpha_itc_ts", 0.0)
        if (a_vis <= 0.0) and (a_ts <= 0.0):
            return (lm_loss, out) if return_outputs else lm_loss

        h = out.hidden_states[-1]           # (B, L, D)
        ids = inputs["input_ids"]             # (B, L)
        att = inputs["attention_mask"].bool()  # (B, L)

        # Answer tokens mask (target for both)
        ans_mask = (inputs["labels"] != -100) & att

        # Optional projection heads
        has_heads = hasattr(model, "txt_proj") and hasattr(model, "img_proj")
        txt_proj = model.txt_proj.to(
            h.device) if has_heads else torch.nn.Identity()
        img_proj = model.img_proj.to(
            h.device) if has_heads else torch.nn.Identity()

        # Always compute answer embedding once
        ans_emb = mean_pool(h, ans_mask)  # (B,D)
        z_ans = F.normalize(txt_proj(ans_emb), dim=-1)

        itc_losses = []

        # --- ITC (vision <-> answer) ---
        if a_vis > 0.0:
            IMG_ID = model.config.image_token_id
            START_ID = getattr(model.config, "vision_start_token_id", None)
            END_ID = getattr(model.config, "vision_end_token_id", None)

            img_mask = (ids == IMG_ID)
            if START_ID is not None:
                img_mask |= (ids == START_ID)
            if END_ID is not None:
                img_mask |= (ids == END_ID)
            img_mask = img_mask & att

            img_emb = mean_pool(h, img_mask)  # (B,D)
            z_img = F.normalize(img_proj(img_emb), dim=-1)

            itc_vis = info_nce(z_a=z_img, z_b=z_ans,
                               tau=getattr(self.args, "itc_temp", 0.07))
            itc_losses.append((a_vis, itc_vis))

        # --- ITC (time-series text span <-> answer) ---
        if a_ts > 0.0:
            src_mask = inputs["itc_src_mask"].bool() & att

            # Safety fallback if src is empty: non-answer, non-image tokens
            if not src_mask.any():
                IMG_ID = model.config.image_token_id
                START_ID = getattr(model.config, "vision_start_token_id", None)
                END_ID = getattr(model.config, "vision_end_token_id", None)

                img_mask = (ids == IMG_ID)
                if START_ID is not None:
                    img_mask |= (ids == START_ID)
                if END_ID is not None:
                    img_mask |= (ids == END_ID)
                img_mask = img_mask & att

                src_mask = att & (~ans_mask) & (~img_mask)

            ts_emb = mean_pool(h, src_mask)  # (B,D)
            # For text<->text, it’s generally fine to use txt_proj for both
            z_ts = F.normalize(txt_proj(ts_emb), dim=-1)

            itc_ts = info_nce(z_a=z_ts, z_b=z_ans, tau=getattr(
                self.args, "itc_temp", 0.07))
            itc_losses.append((a_ts, itc_ts))

        # Combine losses with a stable scaling
        denom = 1.0 + sum(w for w, _ in itc_losses)
        total = lm_loss
        for w, l in itc_losses:
            total = total + w * l
        loss = total / denom

        return (loss, out) if return_outputs else loss


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = build_parser().parse_args()
    setup_logging(args.debug)

    # Backward compatibility: if user uses --alpha_itc only, map it to vision ITC
    if (args.alpha_itc > 0.0) and (args.alpha_itc_vis == 0.0) and (args.alpha_itc_ts == 0.0):
        args.alpha_itc_vis = args.alpha_itc

    os.makedirs(args.out_dir, exist_ok=True)
    logging.info("Saving to: %s", args.out_dir)

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=".*pinned memory.*")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Device: %s • Torch %s", device, torch.__version__)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        device_map="auto" if device == "cuda" else None,
        torch_dtype="auto",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_dir, local_files_only=True)

    tokenizer.padding_side = "right"

    # Optional small projection heads
    if args.use_proj_heads:
        txt_hidden = getattr(model.config, "hidden_size", None) or getattr(
            model.config, "text_config", None).hidden_size
        model.txt_proj = torch.nn.Linear(txt_hidden, txt_hidden)
        model.img_proj = torch.nn.Linear(txt_hidden, txt_hidden)

    # LoRA (text attention by default)
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info("Trainable parameters   : {:,} / {:,} ({:.2f}%)".format(
        trainable, total, 100.0 * trainable / total
    ))
    logging.info("Vision hidden size     : %s", getattr(
        model.config, "vision_config", None).hidden_size)

    IMAGE_PATCH_ID = model.config.image_token_id
    MERGE = getattr(model.config.vision_config, "spatial_merge_size", 2)
    vision_start_id = getattr(model.config, "vision_start_token_id", None)
    vision_end_id = getattr(model.config, "vision_end_token_id", None)
    max_ctx = getattr(model.config, "max_position_embeddings", None)

    ds_full: HFDataset = load_jsonl_dataset(args.data_json)
    split: DatasetDict = ds_full.train_test_split(
        test_size=args.val_ratio, seed=args.seed)
    logging.info("Dataset split — train: %d, eval: %d",
                 len(split["train"]), len(split["test"]))

    train_ds = QwenVLDataset(
        image_processor=image_processor,
        tokenizer=tokenizer,
        img_tok_id=IMAGE_PATCH_ID,
        merge=MERGE,
        start_id=vision_start_id, 
        end_id=vision_end_id,
        max_len=max_ctx,
        debug=args.debug,
        records=split["train"],
        jsonl_path=args.data_json,
    )
    val_ds = QwenVLDataset(
        image_processor=image_processor,
        tokenizer=tokenizer,
        img_tok_id=IMAGE_PATCH_ID,
        merge=MERGE,
        start_id=vision_start_id, 
        end_id=vision_end_id,
        max_len=max_ctx,
        debug=args.debug,
        records=split["test"],
        jsonl_path=args.data_json,
    )

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=1,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        logging_steps=10,
        log_level="debug" if args.debug else "info",
        report_to="none",
    )
    # attach custom fields so Trainer can read them
    targs.alpha_itc_vis = args.alpha_itc_vis
    targs.alpha_itc_ts = args.alpha_itc_ts
    targs.itc_temp = args.itc_temp

    trainer = VLTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collate_fn(tokenizer),
    )

    resume_path = None
    if args.resume:
        resume_path = _pick_resume_checkpoint(args.out_dir)
        if resume_path:
            logging.info("Resuming from checkpoint: %s", resume_path)
        else:
            logging.warning(
                "No valid checkpoint found in %s; starting from scratch.", args.out_dir)

    if resume_path:
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()
