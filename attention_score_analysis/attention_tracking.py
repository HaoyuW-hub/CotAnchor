"""
Attention Score Tracking Experiment.

For each generated CoT token, measures the mean attention score (averaged
over heads) that the current token pays to the condition token (the number n
in the prompt), across all layers.

Produces an (n_generated_tokens, n_layers) attention matrix per sample,
then aggregates across samples to yield avg_attn_matrix.
"""

import sys
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from config import MODELS_DIR

MAX_NEW_TOKENS = 512
TRUNCATE_COT_AT = 200
NUM_LAYERS = 28


def find_condition_token_pos(tokenizer, prompt: str, number: int) -> int:
    """Return 0-indexed position of the last token of str(number) in the prompt."""
    num_str = str(number)
    idx = prompt.find(num_str)
    if idx == -1:
        raise ValueError(f"Number '{num_str}' not found in prompt.")
    prefix = prompt[:idx + len(num_str)]
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
    return prefix_ids.shape[1] - 1


def generate_and_extract_attention(
    model_wrapper,
    prompt: str,
    cond_pos: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[np.ndarray, str]:
    """
    Generate CoT and extract per-step, per-layer attention scores paid to the
    condition token at position cond_pos.

    outputs.attentions: tuple[n_new_tokens] of tuple[n_layers] of
        (batch=1, heads, seq_q, seq_k)

    For each step t and layer l, we take:
        attn_score[t, l] = mean over heads of attn[0, :, -1, cond_pos]
    (the last query token attending to the condition key position).

    Returns:
        attn_array: float32 (n_generated, NUM_LAYERS)
        generated_text: decoded new tokens
    """
    inputs = model_wrapper.tokenizer(prompt, return_tensors="pt").to(model_wrapper.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model_wrapper.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=model_wrapper.tokenizer.pad_token_id,
        )

    n_generated = len(outputs.attentions)
    if n_generated == 0:
        return np.empty((0, NUM_LAYERS), dtype=np.float32), ""

    attn_array = np.empty((n_generated, NUM_LAYERS), dtype=np.float32)
    attn_array[:] = np.nan

    for t in range(n_generated):
        # At step t, the sequence length of the key is input_length + t + 1.
        # cond_pos must be within this range.
        for l in range(NUM_LAYERS):
            if l >= len(outputs.attentions[t]):
                continue
            # shape: (1, n_heads, seq_q, seq_k)
            attn_tensor = outputs.attentions[t][l]
            seq_k = attn_tensor.shape[-1]
            if cond_pos >= seq_k:
                continue
            # mean over heads, last query token, condition key position
            score = attn_tensor[0, :, -1, cond_pos].mean().item()
            attn_array[t, l] = score
            # free tensor immediately
            del attn_tensor

    generated_ids = outputs.sequences[0, input_length:]
    generated_text = model_wrapper.tokenizer.decode(generated_ids, skip_special_tokens=True)

    del outputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return attn_array, generated_text


def run_experiment(
    model_wrapper,
    dataset,
    max_new_tokens: int = MAX_NEW_TOKENS,
    truncate_at: int = TRUNCATE_COT_AT,
) -> dict:
    """
    Run the full attention tracking experiment.

    For each sample: find condition token position, generate CoT,
    extract attention scores, accumulate into sum matrices.

    Returns:
        dict with avg_attn_matrix, avg_attn_per_step, valid_counts, metadata
    """
    attn_sum = np.zeros((truncate_at, NUM_LAYERS), dtype=np.float64)
    valid_counts = np.zeros(truncate_at, dtype=np.int64)

    per_sample_results = []
    n_skipped = 0

    for idx, sample in enumerate(tqdm(dataset, desc="Attention tracking")):
        number = sample["number"]
        prompt = sample["prompt"]

        try:
            cond_pos = find_condition_token_pos(model_wrapper.tokenizer, prompt, number)
        except ValueError as e:
            print(f"  Warning: sample {idx}: {e}, skipping.")
            n_skipped += 1
            continue

        attn_array, generated_text = generate_and_extract_attention(
            model_wrapper, prompt, cond_pos, max_new_tokens
        )

        n_gen = attn_array.shape[0]
        if n_gen == 0:
            print(f"  Warning: sample {idx} generated 0 tokens, skipping.")
            n_skipped += 1
            continue

        effective = min(n_gen, truncate_at)
        # nan-safe accumulation: only add where values are finite
        finite_mask = np.isfinite(attn_array[:effective])
        attn_sum[:effective] += np.where(finite_mask, attn_array[:effective], 0.0)
        valid_counts[:effective] += finite_mask.any(axis=1).astype(np.int64)

        per_sample_results.append({
            "index": idx,
            "number": int(number),
            "n_generated": n_gen,
            "cond_pos": cond_pos,
            "mean_attn_step0": float(np.nanmean(attn_array[0])) if n_gen > 0 else None,
        })

        del attn_array

    print(f"\nCompleted {len(dataset) - n_skipped}/{len(dataset)} samples ({n_skipped} skipped)")

    safe_counts = np.maximum(valid_counts[:, None], 1)
    avg_attn = np.where(
        valid_counts[:, None] > 0,
        attn_sum / safe_counts,
        np.nan,
    ).astype(np.float32)

    # Layer-averaged A_t = (1/L) * sum_l attn[t, l]
    avg_attn_per_step = np.nanmean(avg_attn, axis=1).astype(np.float32)

    return {
        "avg_attn_matrix": avg_attn,           # (truncate_at, NUM_LAYERS)
        "avg_attn_per_step": avg_attn_per_step, # (truncate_at,)
        "valid_counts": valid_counts,
        "per_sample_results": per_sample_results,
        "n_samples": len(dataset),
        "n_skipped": n_skipped,
        "max_new_tokens": max_new_tokens,
        "truncate_at": truncate_at,
        "num_layers": NUM_LAYERS,
    }


def save_results_json(results: dict, filename: str = "attention_tracking_results.json"):
    """Save experiment results to JSON under MODELS_DIR."""
    out = {
        "avg_attn_matrix": results["avg_attn_matrix"].tolist(),
        "avg_attn_per_step": results["avg_attn_per_step"].tolist(),
        "valid_counts": results["valid_counts"].tolist(),
        "per_sample_results": results["per_sample_results"],
        "metadata": {
            "n_samples": results["n_samples"],
            "n_skipped": results["n_skipped"],
            "max_new_tokens": results["max_new_tokens"],
            "truncate_at": results["truncate_at"],
            "num_layers": results["num_layers"],
        },
    }
    path = MODELS_DIR / filename
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved: {path}")
