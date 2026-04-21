"""
Entry point for the CoT probe tracking experiment.

Usage:
    python run_cot_tracking.py [--probe-file multi_position_probes.pkl]
                               [--max-new-tokens 512]
                               [--truncate-at 200]
                               [--num-samples N]
                               [--output-prefix cot_tracking]
"""

import argparse
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from config import FIGURES_DIR
from model_utils import ModelWrapper
from data_preparation import load_dataset

from cot_probe_tracking import (
    load_probes,
    run_experiment,
    save_results_json,
)
from cot_tracking_visualization import (
    visualize_cot_mse_heatmap,
    visualize_cot_acc_heatmap,
    visualize_cot_layer_avg,
    visualize_cot_step_avg,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CoT probe tracking experiment")
    parser.add_argument("--probe-file", default="multi_position_probes.pkl",
                        help="Filename of saved probes under models/")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate per sample")
    parser.add_argument("--truncate-at", type=int, default=200,
                        help="Max generated steps to keep in result matrices")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Subset of dataset (None = full dataset)")
    parser.add_argument("--output-prefix", default="cot_tracking",
                        help="Prefix for output filenames")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 80)
    print("COT PROBE TRACKING EXPERIMENT")
    print("=" * 80)
    print(f"  probe file     : {args.probe_file}")
    print(f"  max new tokens : {args.max_new_tokens}")
    print(f"  truncate at    : {args.truncate_at}")
    print(f"  num samples    : {args.num_samples or 'all'}")
    print(f"  output prefix  : {args.output_prefix}")
    print()

    # --- Data ---
    dataset = load_dataset()
    if args.num_samples is not None:
        dataset = dataset[:args.num_samples]
    print(f"Dataset size: {len(dataset)} samples")

    # --- Model ---
    model_wrapper = ModelWrapper()
    model_wrapper.load_model()

    # --- Probes ---
    probe_system, last_pos = load_probes(args.probe_file)

    # --- Experiment ---
    results = run_experiment(
        model_wrapper=model_wrapper,
        dataset=dataset,
        probe_system=probe_system,
        last_pos=last_pos,
        max_new_tokens=args.max_new_tokens,
        truncate_at=args.truncate_at,
    )

    # --- Save ---
    save_results_json(results, filename=f"{args.output_prefix}_results.json")

    # --- Cleanup GPU before heavy plotting ---
    model_wrapper.cleanup()

    # --- Visualize ---
    prefix = args.output_prefix
    avg_mse = results["avg_mse_matrix"]
    avg_acc = results["avg_acc_matrix"]
    valid_counts = results["valid_counts"]

    visualize_cot_mse_heatmap(
        avg_mse, valid_counts,
        save_path=FIGURES_DIR / f"{prefix}_mse_heatmap.png",
        title="CoT Probe MSE (log space): Generated Token Step vs Layer",
    )
    visualize_cot_acc_heatmap(
        avg_acc, valid_counts,
        save_path=FIGURES_DIR / f"{prefix}_acc_heatmap.png",
        title="CoT Probe Accuracy (1%): Generated Token Step vs Layer",
    )
    visualize_cot_layer_avg(
        avg_mse, avg_acc,
        save_path=FIGURES_DIR / f"{prefix}_layer_avg.png",
    )
    visualize_cot_step_avg(
        avg_mse, avg_acc,
        save_path=FIGURES_DIR / f"{prefix}_step_avg.png",
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
