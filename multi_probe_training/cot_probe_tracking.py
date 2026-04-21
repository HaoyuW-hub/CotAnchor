"""
CoT Probe Tracking Experiment.

Takes probes trained at the last input token position and tracks their
predictive power across all layers as the model generates CoT tokens.
Produces (n_generated_tokens, n_layers) MSE and accuracy matrices.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from config import MODELS_DIR, NUM_POSITIVE_SAMPLES, NUM_NEGATIVE_SAMPLES

# Re-import MultiPositionProbe from sibling module
from probe_training_multilayer import MultiPositionProbe  # noqa: E402


MAX_NEW_TOKENS = 512
TRUNCATE_COT_AT = 200
NUM_LAYERS = 28
ACC_THRESHOLD = 0.01  # |pred - true| <= threshold * true


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probes(probe_file: str = "multi_position_probes.pkl") -> Tuple[MultiPositionProbe, int]:
    """
    Load trained probes and derive the last input token position.

    Returns:
        (probe_system, last_input_token_pos)
    """
    probe_system = MultiPositionProbe(num_layers=NUM_LAYERS)
    probe_system.load(probe_file)
    last_pos = probe_system.num_tokens - 1
    print(f"Loaded probes: {probe_system.num_layers} layers, {probe_system.num_tokens} token positions")
    print(f"Using last input token position: {last_pos}")
    return probe_system, last_pos


# ---------------------------------------------------------------------------
# Generation + hidden state extraction
# ---------------------------------------------------------------------------

def generate_and_extract(
    model_wrapper,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[np.ndarray, str]:
    """
    Generate CoT and extract hidden states for each generated token step.

    Uses outputs.hidden_states from model.generate() directly — no extra
    forward passes. Layer indexing matches the training convention:
      outputs.hidden_states[t][layer]  layer = 0..NUM_LAYERS-1

    Args:
        model_wrapper: ModelWrapper instance (model already loaded)
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate

    Returns:
        hidden_states_array: float32 array of shape (n_generated, NUM_LAYERS, hidden_dim)
        generated_text: decoded generated text (new tokens only)
    """
    inputs = model_wrapper.tokenizer(prompt, return_tensors="pt").to(model_wrapper.device)

    with torch.no_grad():
        outputs = model_wrapper.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=model_wrapper.tokenizer.pad_token_id,
        )

    # outputs.hidden_states: tuple[n_new_tokens] of tuple[n_layers+1] of (1, seq, hidden)
    # We index layers 0..NUM_LAYERS-1, matching the training convention.
    n_generated = len(outputs.hidden_states)
    if n_generated == 0:
        return np.empty((0, NUM_LAYERS, 0), dtype=np.float32), ""

    hidden_dim = outputs.hidden_states[0][0].shape[-1]
    hidden_states_array = np.empty((n_generated, NUM_LAYERS, hidden_dim), dtype=np.float32)

    for t in range(n_generated):
        for l in range(NUM_LAYERS):
            # [0, -1, :] = batch 0, last (newly generated) token
            hidden_states_array[t, l, :] = (
                outputs.hidden_states[t][l][0, -1, :].cpu().float().numpy()
            )

    input_length = inputs.input_ids.shape[1]
    generated_ids = outputs.sequences[0, input_length:]
    generated_text = model_wrapper.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return hidden_states_array, generated_text


# ---------------------------------------------------------------------------
# Apply probes to extracted hidden states
# ---------------------------------------------------------------------------

def apply_probes(
    hidden_states_array: np.ndarray,
    probe_system: MultiPositionProbe,
    last_pos: int,
    true_n: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply probes[layer][last_pos] to each (step, layer) hidden state.

    Args:
        hidden_states_array: (n_generated, NUM_LAYERS, hidden_dim)
        probe_system: trained MultiPositionProbe
        last_pos: last input token position key used to select probes
        true_n: ground-truth integer n for this sample

    Returns:
        mse_matrix: (n_generated, NUM_LAYERS) pointwise squared error in log space
        acc_matrix: (n_generated, NUM_LAYERS) 1.0 if within ACC_THRESHOLD else 0.0
    """
    n_generated, n_layers, _ = hidden_states_array.shape
    mse_matrix = np.full((n_generated, n_layers), np.nan, dtype=np.float32)
    acc_matrix = np.full((n_generated, n_layers), np.nan, dtype=np.float32)

    log_true = np.log1p(true_n)

    for l in range(n_layers):
        if l not in probe_system.probes or last_pos not in probe_system.probes[l]:
            continue
        probe = probe_system.probes[l][last_pos]
        # Batch predict over all time steps for this layer
        X = hidden_states_array[:, l, :]  # (n_generated, hidden_dim)
        pred_log = probe.predict(X)        # (n_generated,)
        pred_orig = np.expm1(pred_log)

        mse_matrix[:, l] = (pred_log - log_true) ** 2
        acc_matrix[:, l] = (np.abs(pred_orig - true_n) <= ACC_THRESHOLD * true_n).astype(np.float32)

    return mse_matrix, acc_matrix


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    model_wrapper,
    dataset,
    probe_system: MultiPositionProbe,
    last_pos: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    truncate_at: int = TRUNCATE_COT_AT,
) -> dict:
    """
    Run the full tracking experiment over all samples.

    For each sample: generate CoT, extract hidden states, apply probes,
    accumulate per-step MSE and accuracy using valid_counts to handle
    varying CoT lengths without zero-padding bias.

    Returns:
        dict with avg_mse_matrix, avg_acc_matrix, valid_counts, metadata
    """
    mse_sum = np.zeros((truncate_at, NUM_LAYERS), dtype=np.float64)
    acc_sum = np.zeros((truncate_at, NUM_LAYERS), dtype=np.float64)
    valid_counts = np.zeros(truncate_at, dtype=np.int64)

    per_sample_results = []
    n_skipped = 0

    for idx, sample in enumerate(tqdm(dataset, desc="CoT tracking")):
        true_n = float(sample['number'])
        prompt = sample['prompt']

        hidden_states_array, generated_text = generate_and_extract(
            model_wrapper, prompt, max_new_tokens
        )

        n_gen = hidden_states_array.shape[0]
        if n_gen == 0:
            print(f"  Warning: sample {idx} generated 0 tokens, skipping.")
            n_skipped += 1
            continue

        mse_mat, acc_mat = apply_probes(hidden_states_array, probe_system, last_pos, true_n)

        # Truncate to truncate_at steps
        effective = min(n_gen, truncate_at)
        mse_sum[:effective] += mse_mat[:effective].astype(np.float64)
        acc_sum[:effective] += acc_mat[:effective].astype(np.float64)
        valid_counts[:effective] += 1

        # Lightweight per-sample record (avoid storing full matrices)
        per_sample_results.append({
            "index": idx,
            "number": int(true_n),
            "n_generated": n_gen,
            "mean_mse_step0": float(np.nanmean(mse_mat[0])) if n_gen > 0 else None,
            "mean_acc_step0": float(np.nanmean(acc_mat[0])) if n_gen > 0 else None,
        })

        # Free GPU-sized array ASAP
        del hidden_states_array

    print(f"\nCompleted {len(dataset) - n_skipped}/{len(dataset)} samples ({n_skipped} skipped)")

    # Safe division: positions with no samples → NaN
    safe_counts = np.maximum(valid_counts[:, None], 1)
    avg_mse = np.where(valid_counts[:, None] > 0, mse_sum / safe_counts, np.nan).astype(np.float32)
    avg_acc = np.where(valid_counts[:, None] > 0, acc_sum / safe_counts, np.nan).astype(np.float32)

    return {
        "avg_mse_matrix": avg_mse,           # (truncate_at, NUM_LAYERS)
        "avg_acc_matrix": avg_acc,           # (truncate_at, NUM_LAYERS)
        "valid_counts": valid_counts,        # (truncate_at,)
        "per_sample_results": per_sample_results,
        "n_samples": len(dataset),
        "n_skipped": n_skipped,
        "max_new_tokens": max_new_tokens,
        "truncate_at": truncate_at,
        "last_input_token_pos": last_pos,
        "num_layers": NUM_LAYERS,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_results_json(results: dict, filename: str = "cot_tracking_results.json"):
    """Save experiment results to JSON under MODELS_DIR."""
    out = {
        "avg_mse_matrix": results["avg_mse_matrix"].tolist(),
        "avg_acc_matrix": results["avg_acc_matrix"].tolist(),
        "valid_counts": results["valid_counts"].tolist(),
        "per_sample_results": results["per_sample_results"],
        "metadata": {
            "n_samples": results["n_samples"],
            "n_skipped": results["n_skipped"],
            "max_new_tokens": results["max_new_tokens"],
            "truncate_at": results["truncate_at"],
            "last_input_token_pos": results["last_input_token_pos"],
            "num_layers": results["num_layers"],
            "acc_threshold": ACC_THRESHOLD,
        },
    }
    path = MODELS_DIR / filename
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved: {path}")
