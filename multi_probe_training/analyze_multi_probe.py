"""
Analysis script for trained multi-position probes
Provides detailed analysis and additional visualizations
"""

import sys
from pathlib import Path
# Add parent directory to path to import from CotAnchor root
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from probe_training_multilayer import MultiPositionProbe
from config import MODELS_DIR, FIGURES_DIR


def analyze_layer_trends(mse_matrix: np.ndarray, save_path: Path = None):
    """
    Analyze how MSE changes across layers.
    Identifies which layers have the best number representation.
    """
    n_layers, n_tokens = mse_matrix.shape

    # Calculate statistics per layer
    mse_mean = mse_matrix.mean(axis=1)
    mse_std = mse_matrix.std(axis=1)
    mse_min = mse_matrix.min(axis=1)
    mse_max = mse_matrix.max(axis=1)

    # Find best layer
    best_layer = np.argmin(mse_mean)
    print(f"\nBest layer (lowest avg MSE): Layer {best_layer}")
    print(f"  Average MSE: {mse_mean[best_layer]:.4f}")
    print(f"  Std MSE: {mse_std[best_layer]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(range(n_layers), mse_mean, marker='o', linewidth=2, label='Mean MSE')
    ax.fill_between(
        range(n_layers),
        mse_mean - mse_std,
        mse_mean + mse_std,
        alpha=0.3,
        label='±1 std'
    )
    ax.plot(range(n_layers), mse_min, '--', alpha=0.5, label='Min MSE')
    ax.plot(range(n_layers), mse_max, '--', alpha=0.5, label='Max MSE')

    ax.axvline(best_layer, color='red', linestyle='--', alpha=0.5, label=f'Best layer ({best_layer})')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE Trends Across Layers', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layer trends plot saved to {save_path}")

    plt.show()

    return best_layer


def analyze_token_positions(mse_matrix: np.ndarray, save_path: Path = None):
    """
    Analyze which token positions have the best number representation.
    """
    n_layers, n_tokens = mse_matrix.shape

    # Calculate statistics per token position
    mse_mean = mse_matrix.mean(axis=0)
    mse_std = mse_matrix.std(axis=0)

    # Find best positions
    best_positions = np.argsort(mse_mean)[:5]
    worst_positions = np.argsort(mse_mean)[-5:]

    print(f"\nTop 5 best token positions (lowest avg MSE):")
    for i, pos in enumerate(best_positions):
        print(f"  {i+1}. Position {pos}: MSE = {mse_mean[pos]:.4f}")

    print(f"\nTop 5 worst token positions (highest avg MSE):")
    for i, pos in enumerate(worst_positions):
        print(f"  {i+1}. Position {pos}: MSE = {mse_mean[pos]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(range(n_tokens), mse_mean, alpha=0.7)

    # Highlight best and worst positions
    for pos in best_positions:
        bars[pos].set_color('green')
        bars[pos].set_alpha(0.8)

    for pos in worst_positions:
        bars[pos].set_color('red')
        bars[pos].set_alpha(0.8)

    ax.errorbar(range(n_tokens), mse_mean, yerr=mse_std, fmt='none', ecolor='black', alpha=0.3)

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Average MSE', fontsize=12)
    ax.set_title('MSE by Token Position (Green=Best, Red=Worst)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Token position analysis saved to {save_path}")

    plt.show()

    return best_positions, worst_positions


def find_optimal_probe(mse_matrix: np.ndarray):
    """
    Find the single best (layer, token_position) combination.
    """
    best_idx = np.unravel_index(np.argmin(mse_matrix), mse_matrix.shape)
    best_layer, best_token = best_idx

    print(f"\nOptimal probe location:")
    print(f"  Layer: {best_layer}")
    print(f"  Token Position: {best_token}")
    print(f"  MSE: {mse_matrix[best_layer, best_token]:.4f}")

    return best_layer, best_token


def analyze_layer_token_interaction(mse_matrix: np.ndarray, save_path: Path = None):
    """
    Analyze how different layers perform at different token positions.
    """
    n_layers, n_tokens = mse_matrix.shape

    # Select a few representative layers
    layer_indices = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    layer_labels = ['Early', 'Early-Mid', 'Middle', 'Mid-Late', 'Late']

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, label in zip(layer_indices, layer_labels):
        if idx < n_layers:
            ax.plot(range(n_tokens), mse_matrix[idx, :], marker='o', label=f'{label} (L{idx})', alpha=0.7)

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Layer-Token Interaction: MSE Across Positions for Different Layers', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layer-token interaction plot saved to {save_path}")

    plt.show()


def generate_summary_report(mse_matrix: np.ndarray, output_path: Path = None):
    """
    Generate a comprehensive summary report.
    """
    n_layers, n_tokens = mse_matrix.shape

    report = {
        'overall_statistics': {
            'mean_mse': float(mse_matrix.mean()),
            'std_mse': float(mse_matrix.std()),
            'min_mse': float(mse_matrix.min()),
            'max_mse': float(mse_matrix.max()),
            'median_mse': float(np.median(mse_matrix))
        },
        'layer_statistics': {
            'best_layer': int(np.argmin(mse_matrix.mean(axis=1))),
            'worst_layer': int(np.argmax(mse_matrix.mean(axis=1))),
            'layer_mse_range': float(mse_matrix.mean(axis=1).max() - mse_matrix.mean(axis=1).min())
        },
        'token_statistics': {
            'best_token_position': int(np.argmin(mse_matrix.mean(axis=0))),
            'worst_token_position': int(np.argmax(mse_matrix.mean(axis=0))),
            'token_mse_range': float(mse_matrix.mean(axis=0).max() - mse_matrix.mean(axis=0).min())
        },
        'optimal_probe': {
            'layer': int(np.unravel_index(np.argmin(mse_matrix), mse_matrix.shape)[0]),
            'token_position': int(np.unravel_index(np.argmin(mse_matrix), mse_matrix.shape)[1]),
            'mse': float(mse_matrix.min())
        }
    }

    # Print report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    print("\nOverall Statistics:")
    for key, value in report['overall_statistics'].items():
        print(f"  {key}: {value:.4f}")

    print("\nLayer Statistics:")
    for key, value in report['layer_statistics'].items():
        print(f"  {key}: {value}")

    print("\nToken Statistics:")
    for key, value in report['token_statistics'].items():
        print(f"  {key}: {value}")

    print("\nOptimal Probe:")
    for key, value in report['optimal_probe'].items():
        print(f"  {key}: {value}")

    # Save to JSON
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {output_path}")

    return report


def main():
    """Main analysis pipeline"""

    print("="*80)
    print("MULTI-POSITION PROBE ANALYSIS")
    print("="*80)

    # Load trained probes
    print("\nLoading trained probes...")
    probe_system = MultiPositionProbe(num_layers=28)  # Adjust if needed
    try:
        probe_system.load("multi_position_probes.pkl")
    except FileNotFoundError:
        print("Error: Trained probes not found!")
        print("Please run probe_training_multilayer.py first.")
        return

    mse_matrix = probe_system.mse_matrix
    print(f"Loaded MSE matrix: {mse_matrix.shape}")

    # Run analyses
    print("\n" + "="*80)
    print("LAYER ANALYSIS")
    print("="*80)
    best_layer = analyze_layer_trends(
        mse_matrix,
        save_path=FIGURES_DIR / "analysis_layer_trends.png"
    )

    print("\n" + "="*80)
    print("TOKEN POSITION ANALYSIS")
    print("="*80)
    best_positions, worst_positions = analyze_token_positions(
        mse_matrix,
        save_path=FIGURES_DIR / "analysis_token_positions.png"
    )

    print("\n" + "="*80)
    print("OPTIMAL PROBE")
    print("="*80)
    best_layer, best_token = find_optimal_probe(mse_matrix)

    print("\n" + "="*80)
    print("LAYER-TOKEN INTERACTION")
    print("="*80)
    analyze_layer_token_interaction(
        mse_matrix,
        save_path=FIGURES_DIR / "analysis_layer_token_interaction.png"
    )

    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    report = generate_summary_report(
        mse_matrix,
        output_path=MODELS_DIR / "analysis_summary_report.json"
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
