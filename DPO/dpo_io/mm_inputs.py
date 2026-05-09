import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image


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
    fixed = []
    base = (Path(__file__).resolve().parent / "../../").resolve()
    for n in names:
        p = Path(n)
        fixed.append(str((base / n).resolve()) if not p.is_absolute() else str(p))
    return fixed


def _record_teacher_text(rec: Dict[str, Any]) -> str:
    if rec.get("output") is not None:
        return str(rec["output"])
    oj = rec.get("output_json")
    if oj is not None:
        return json.dumps(oj, ensure_ascii=False)
    raise KeyError("Record missing 'output' and 'output_json'")


@torch.no_grad()
def _prep_mm_prompt(
    rec: Dict[str, Any],
    image_processor,
    tokenizer,
    img_tok_id: int,
    merge: int,
    start_id: int | None,
    end_id: int | None,
    device: str,
) -> Dict[str, Any]:
    # --- images ---
    names_raw = rec.get("image_paths", [])
    if isinstance(names_raw, str):
        names = [names_raw]
    elif isinstance(names_raw, list):
        names = names_raw
    else:
        names = []

    names = _resolve_image_paths(names) if names else []

    has_images = len(names) > 0
    pixel_values = None
    image_grid_thw = None
    img_tok_tensor = None

    if has_images:
        pix_list, grid_list, seg_list = [], [], []

        def _load_one(fname: str):
            img = Image.open(fname).convert("RGB")
            pix = image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
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
            pix_list.append(p)
            grid_list.append(g)
            seg_list.append(s)

        pixel_values = torch.cat(pix_list, 0).to(device)
        image_grid_thw = torch.stack(grid_list).to(device)
        img_tok_tensor = torch.cat(seg_list, 1).to(device)

    # --- texte ---
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
        prompt_ids = q["input_ids"].to(device)
        prompt_msk = q["attention_mask"].to(device)

    if has_images:
        input_ids = torch.cat([img_tok_tensor, prompt_ids], 1)
        attn_mask = torch.cat([torch.ones_like(img_tok_tensor), prompt_msk], 1)
        prompt_len = input_ids.shape[1]
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "prompt_len": prompt_len,
            "has_images": True,
        }

    return {
        "input_ids": prompt_ids,
        "attention_mask": prompt_msk,
        "prompt_len": prompt_ids.shape[1],
        "has_images": False,
    }


def _append_completion(
    tokenizer,
    mm_prompt: Dict[str, Any],
    completion_text: str,
    device: str,
) -> Dict[str, Any]:
    comp = tokenizer(
        completion_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    eos = tokenizer.eos_token_id
    if eos is not None:
        comp = torch.cat([comp, torch.tensor([[eos]], device=device, dtype=comp.dtype)], dim=1)

    input_ids = torch.cat([mm_prompt["input_ids"], comp], dim=1)
    attention_mask = torch.cat([mm_prompt["attention_mask"], torch.ones_like(comp)], dim=1)

    completion_start = mm_prompt["prompt_len"]
    completion_end = input_ids.shape[1]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_start": completion_start,
        "completion_end": completion_end,
        "completion_len": completion_end - completion_start,
        "has_images": mm_prompt.get("has_images", False),
    }
    if out["has_images"]:
        out["pixel_values"] = mm_prompt["pixel_values"]
        out["image_grid_thw"] = mm_prompt["image_grid_thw"]
    return out
