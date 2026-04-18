"""
Evaluate trained multi-position probes on a dataset.
Loads saved probes from disk and computes MSE and accuracy without retraining.

Usage:
    python evaluate_probes.py                          # evaluate on default dataset
    python evaluate_probes.py --probes my_probes.pkl   # custom probe file
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from config import MODELS_DIR, FIGURES_DIR
from model_utils import ModelWrapper
from data_preparation import load_dataset

from probe_training_multilayer import MultiPositionProbe, extract_all_positions_data
from probe_visualization import (
    visualize_mse_heatmap,
    visualize_acc_heatmap,
    visualize_mse_statistics,
    visualize_acc_statistics,
)


def evaluate(
    probe_system: MultiPositionProbe,
    X_all: dict,
    y: np.ndarray,
) -> dict:
    """
    Run all probes over X_all and compute per-(layer, token_pos) MSE and accuracy.

    Args:
        probe_system: Loaded MultiPositionProbe
        X_all: {layer: (n_samples, n_tokens, hidden_dim)}
        y: (n_samples,) true values in original scale

    Returns:
        dict with mse_matrix, acc_matrix, and per-sample predictions
    """
    num_layers = probe_system.num_layers
    num_tokens = probe_system.num_tokens
    n_samples = len(y)
    y_log = np.log1p(y)

    mse_matrix = np.zeros((num_layers, num_tokens))
    acc_matrix = np.zeros((num_layers, num_tokens))

    print(f"\nEvaluating {num_layers * num_tokens} probes on {n_samples} samples...")

    for layer in tqdm(range(num_layers), desc="Layers"):
        if layer not in X_all or layer not in probe_system.probes:
            continue
        X_layer = X_all[layer]  # (n_samples, n_tokens, hidden_dim)

        for token_pos in range(num_tokens):
            if token_pos not in probe_system.probes[layer]:
                continue

            X_pos = X_layer[:, token_pos, :]  # (n_samples, hidden_dim)
            y_pred_log = probe_system.predict(X_pos, layer, token_pos)
            y_pred_orig = np.expm1(y_pred_log)

            mse_matrix[layer, token_pos] = mean_squared_error(y_log, y_pred_log)
            acc_matrix[layer, token_pos] = np.mean(
                np.abs(y_pred_orig - y) <= 0.01 * np.abs(y)
            )

    return {'mse_matrix': mse_matrix, 'acc_matrix': acc_matrix}


def print_summary(mse_matrix: np.ndarray, acc_matrix: np.ndarray):
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")

    best_layer, best_tok = np.unravel_index(acc_matrix.argmax(), acc_matrix.shape)
    worst_layer, worst_tok = np.unravel_index(acc_matrix.argmin(), acc_matrix.shape)

    print(f"MSE (log space) — avg: {mse_matrix.mean():.4f}  min: {mse_matrix.min():.4f}  max: {mse_matrix.max():.4f}")
    print(f"Accuracy (1%)   — avg: {acc_matrix.mean():.4f}  max: {acc_matrix.max():.4f}  min: {acc_matrix.min():.4f}")
    print(f"Best  probe: layer {best_layer}, token {best_tok}  → acc {acc_matrix[best_layer, best_tok]:.4f}")
    print(f"Worst probe: layer {worst_layer}, token {worst_tok}  → acc {acc_matrix[worst_layer, worst_tok]:.4f}")

    print("\nTop-10 (layer, token) by accuracy:")
    flat_idx = np.argsort(acc_matrix.ravel())[::-1][:10]
    for rank, idx in enumerate(flat_idx, 1):
        l, t = np.unravel_index(idx, acc_matrix.shape)
        print(f"  {rank:2d}. layer {l:2d}, token {t:2d}  acc={acc_matrix[l, t]:.4f}  mse={mse_matrix[l, t]:.4f}")


def save_eval_json(mse_matrix: np.ndarray, acc_matrix: np.ndarray, filename: str = "eval_results.json"):
    filepath = MODELS_DIR / filename
    data = {
        'mse_matrix': mse_matrix.tolist(),
        'acc_matrix': acc_matrix.tolist(),
        'avg_mse': float(mse_matrix.mean()),
        'min_mse': float(mse_matrix.min()),
        'max_mse': float(mse_matrix.max()),
        'avg_acc': float(acc_matrix.mean()),
        'max_acc': float(acc_matrix.max()),
        'num_layers': int(mse_matrix.shape[0]),
        'num_tokens': int(mse_matrix.shape[1]),
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Evaluation results saved to {filepath}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained multi-position probes")
    parser.add_argument('--probes', default='multi_position_probes.pkl',
                        help="Probe pkl filename under MODELS_DIR (default: multi_position_probes.pkl)")
    parser.add_argument('--no-vis', action='store_true',
                        help="Skip visualization")
    parser.add_argument('--output', default='eval_results.json',
                        help="Output JSON filename under MODELS_DIR (default: eval_results.json)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load probes
    probe_system = MultiPositionProbe(num_layers=0)  # num_layers overwritten by load()
    probe_system.load(args.probes)
    print(f"Loaded probes: {probe_system.num_layers} layers × {probe_system.num_tokens} token positions")

    # Load dataset and model
    dataset = load_dataset()
    model_wrapper = ModelWrapper()
    model_wrapper.load_model()

    # Extract hidden states
    X_all, y = extract_all_positions_data(model_wrapper, dataset, probe_system.num_layers)
    model_wrapper.cleanup()

    # Evaluate
    eval_results = evaluate(probe_system, X_all, y)
    mse_matrix = eval_results['mse_matrix']
    acc_matrix = eval_results['acc_matrix']

    print_summary(mse_matrix, acc_matrix)
    save_eval_json(mse_matrix, acc_matrix, args.output)

    if not args.no_vis:
        visualize_mse_heatmap(
            mse_matrix,
            save_path=FIGURES_DIR / "eval_mse_heatmap.png",
            title="Eval MSE (log space): Layer vs Token Position",
            percentile_clip=95.0
        )
        visualize_acc_heatmap(
            acc_matrix,
            save_path=FIGURES_DIR / "eval_acc_heatmap.png",
            title="Eval Accuracy (1% threshold): Layer vs Token Position"
        )
        visualize_mse_statistics(
            mse_matrix,
            save_path=FIGURES_DIR / "eval_mse_statistics.png",
            percentile_clip=95.0
        )
        visualize_acc_statistics(
            acc_matrix,
            save_path=FIGURES_DIR / "eval_acc_statistics.png"
        )
