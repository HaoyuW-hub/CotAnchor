"""
Entry point for the attention score tracking experiment.

Usage:
    python run_attention_tracking.py [--max-new-tokens 512]
                                     [--truncate-at 200]
                                     [--num-samples N]
                                     [--output-prefix attn_tracking]
"""

import argparse
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from config import FIGURES_DIR
from model_utils import ModelWrapper
from data_preparation import load_dataset

from attention_tracking import run_experiment, save_results_json
from attention_visualization import (
    visualize_attention_heatmap,
    visualize_attention_layer_avg,
    visualize_attention_per_layer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Attention score tracking experiment")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--truncate-at", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-prefix", default="attn_tracking")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 80)
    print("ATTENTION SCORE TRACKING EXPERIMENT")
    print("=" * 80)
    print(f"  max new tokens : {args.max_new_tokens}")
    print(f"  truncate at    : {args.truncate_at}")
    print(f"  num samples    : {args.num_samples or 'all'}")
    print(f"  output prefix  : {args.output_prefix}")
    print()

    dataset = load_dataset()
    if args.num_samples is not None:
        dataset = dataset[:args.num_samples]
    print(f"Dataset size: {len(dataset)} samples")

    model_wrapper = ModelWrapper()
    model_wrapper.load_model()

    results = run_experiment(
        model_wrapper=model_wrapper,
        dataset=dataset,
        max_new_tokens=args.max_new_tokens,
        truncate_at=args.truncate_at,
    )

    save_results_json(results, filename=f"{args.output_prefix}_results.json")

    model_wrapper.cleanup()

    prefix = args.output_prefix
    avg_attn = results["avg_attn_matrix"]
    avg_attn_per_step = results["avg_attn_per_step"]
    valid_counts = results["valid_counts"]

    visualize_attention_heatmap(
        avg_attn, valid_counts,
        save_path=FIGURES_DIR / f"{prefix}_heatmap.png",
    )
    visualize_attention_layer_avg(
        avg_attn_per_step, valid_counts,
        save_path=FIGURES_DIR / f"{prefix}_layer_avg.png",
    )
    visualize_attention_per_layer(
        avg_attn, valid_counts,
        save_path=FIGURES_DIR / f"{prefix}_per_layer.png",
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
