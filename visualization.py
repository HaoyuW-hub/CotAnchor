"""
Visualization Module
Create plots and figures for the pilot experiment
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import json
from pathlib import Path

from config import FIGURES_DIR, RESULTS_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_drift_curve_single(
    result: Dict,
    save_path: Path = None,
    show: bool = True
):
    """
    Plot drift curve for a single sample

    Args:
        result: Drift tracking result for one sample
        save_path: Path to save figure
        show: Whether to display the plot
    """
    drift_metrics = result['drift_metrics']
    positions = [m['position'] for m in drift_metrics]
    probe_scores = [m['probe_score'] for m in drift_metrics]
    cosine_sims = [m['cosine_similarity'] for m in drift_metrics]
    anchors = result['anchors']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot probe scores
    ax1.plot(positions, probe_scores, 'b-', linewidth=2, label='Probe Score')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    # Mark anchor positions
    for anchor in anchors:
        ax1.axvline(x=anchor['position'], color='red', alpha=0.3, linestyle='--')
        ax1.text(anchor['position'], ax1.get_ylim()[1] * 0.95,
                anchor['token'], rotation=90, verticalalignment='top',
                fontsize=8, color='red')

    ax1.set_ylabel('Probe Score (P(is prime))')
    ax1.set_title(f"Representation Drift: {result['sample_id']} (n={result['number']}, "
                 f"{'prime' if result['is_prime'] else 'composite'})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot cosine similarity
    ax2.plot(positions, cosine_sims, 'g-', linewidth=2, label='Cosine Similarity')

    # Mark anchor positions
    for anchor in anchors:
        ax2.axvline(x=anchor['position'], color='red', alpha=0.3, linestyle='--')

    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Cosine Similarity with Initial State')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_drift_curves_comparison(
    results: List[Dict],
    save_path: Path = None,
    show: bool = True
):
    """
    Plot drift curves for multiple samples in comparison

    Args:
        results: List of drift tracking results
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Separate prime and composite samples
    prime_results = [r for r in results if r['is_prime']]
    composite_results = [r for r in results if not r['is_prime']]

    # Plot probe scores
    for result in prime_results:
        positions = [m['position'] for m in result['drift_metrics']]
        scores = [m['probe_score'] for m in result['drift_metrics']]
        ax1.plot(positions, scores, 'b-', alpha=0.5, linewidth=1)

    for result in composite_results:
        positions = [m['position'] for m in result['drift_metrics']]
        scores = [m['probe_score'] for m in result['drift_metrics']]
        ax1.plot(positions, scores, 'r-', alpha=0.5, linewidth=1)

    # Add legend
    ax1.plot([], [], 'b-', label=f'Prime numbers (n={len(prime_results)})', linewidth=2)
    ax1.plot([], [], 'r-', label=f'Composite numbers (n={len(composite_results)})', linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Probe Score')
    ax1.set_title('Probe Score Drift Across Samples')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot cosine similarities
    for result in prime_results:
        positions = [m['position'] for m in result['drift_metrics']]
        cosines = [m['cosine_similarity'] for m in result['drift_metrics']]
        ax2.plot(positions, cosines, 'b-', alpha=0.5, linewidth=1)

    for result in composite_results:
        positions = [m['position'] for m in result['drift_metrics']]
        cosines = [m['cosine_similarity'] for m in result['drift_metrics']]
        ax2.plot(positions, cosines, 'r-', alpha=0.5, linewidth=1)

    ax2.plot([], [], 'b-', label=f'Prime numbers', linewidth=2)
    ax2.plot([], [], 'r-', label=f'Composite numbers', linewidth=2)
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Cosine Similarity Drift Across Samples')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_anchor_effects(
    anchor_analysis: Dict,
    save_path: Path = None,
    show: bool = True
):
    """
    Visualize anchor token effects

    Args:
        anchor_analysis: Results from anchor analysis
        save_path: Path to save figure
        show: Whether to display the plot
    """
    aggregate_stats = anchor_analysis['aggregate_statistics']

    if aggregate_stats['total_anchors'] == 0:
        print("No anchor tokens found to visualize")
        return

    # Collect all anchor effects
    all_effects = []
    for sample in anchor_analysis['sample_analyses']:
        all_effects.extend(sample.get('anchor_effects', []))

    probe_changes = [e['probe_score_change'] for e in all_effects]
    cosine_changes = [e['cosine_change'] for e in all_effects]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram of probe score changes
    axes[0, 0].hist(probe_changes, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Probe Score Change')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Probe Score Changes at Anchors')
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram of cosine changes
    axes[0, 1].hist(cosine_changes, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Cosine Similarity Change')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Cosine Changes at Anchors')
    axes[0, 1].grid(True, alpha=0.3)

    # Scatter plot
    axes[1, 0].scatter(probe_changes, cosine_changes, alpha=0.6, s=50)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Probe Score Change')
    axes[1, 0].set_ylabel('Cosine Similarity Change')
    axes[1, 0].set_title('Correlation: Probe Change vs Cosine Change')
    axes[1, 0].grid(True, alpha=0.3)

    # Token statistics bar plot
    token_stats = anchor_analysis['token_statistics']
    if token_stats:
        sorted_tokens = sorted(token_stats.items(),
                              key=lambda x: x[1]['count'],
                              reverse=True)[:10]
        tokens = [t[0] for t in sorted_tokens]
        counts = [t[1]['count'] for t in sorted_tokens]

        axes[1, 1].barh(tokens, counts, color='coral')
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_title('Top 10 Most Frequent Anchor Tokens')
        axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_summary_statistics(
    results: List[Dict],
    drift_analysis: Dict,
    save_path: Path = None,
    show: bool = True
):
    """
    Create summary visualization of all results

    Args:
        results: Drift tracking results
        drift_analysis: Results from analyze_drift_patterns
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Probe score changes by sample
    sample_ids = [r['sample_id'] for r in results]
    probe_changes = [r['statistics']['probe_score_change'] for r in results]
    colors = ['blue' if r['is_prime'] else 'red' for r in results]

    axes[0, 0].bar(range(len(sample_ids)), probe_changes, color=colors, alpha=0.7)
    axes[0, 0].set_xticks(range(len(sample_ids)))
    axes[0, 0].set_xticklabels(sample_ids, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Probe Score Change')
    axes[0, 0].set_title('Probe Score Drift by Sample')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Prime'),
                      Patch(facecolor='red', alpha=0.7, label='Composite')]
    axes[0, 0].legend(handles=legend_elements)

    # 2. Cosine similarity changes by sample
    cosine_changes = [r['statistics']['cosine_change'] for r in results]

    axes[0, 1].bar(range(len(sample_ids)), cosine_changes, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(len(sample_ids)))
    axes[0, 1].set_xticklabels(sample_ids, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Cosine Similarity Change')
    axes[0, 1].set_title('Cosine Similarity Drift by Sample')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Distribution comparison
    prime_probe_changes = [r['statistics']['probe_score_change']
                          for r in results if r['is_prime']]
    composite_probe_changes = [r['statistics']['probe_score_change']
                              for r in results if not r['is_prime']]

    axes[1, 0].hist([prime_probe_changes, composite_probe_changes],
                   label=['Prime', 'Composite'],
                   color=['blue', 'red'],
                   alpha=0.6,
                   bins=10)
    axes[1, 0].set_xlabel('Probe Score Change')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution: Prime vs Composite')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Summary statistics table
    axes[1, 1].axis('off')
    summary_text = f"""
    PILOT EXPERIMENT SUMMARY

    Total Samples: {len(results)}
    Prime Samples: {sum(1 for r in results if r['is_prime'])}
    Composite Samples: {sum(1 for r in results if not r['is_prime'])}

    Probe Score Change:
      Mean: {drift_analysis['probe_score_change']['mean']:.4f}
      Std: {drift_analysis['probe_score_change']['std']:.4f}
      Range: [{drift_analysis['probe_score_change']['min']:.4f},
              {drift_analysis['probe_score_change']['max']:.4f}]

    Cosine Similarity Change:
      Mean: {drift_analysis['cosine_similarity_change']['mean']:.4f}
      Std: {drift_analysis['cosine_similarity_change']['std']:.4f}

    Total Anchors Found: {drift_analysis['anchor_count']['total']}
    Avg Anchors per Sample: {drift_analysis['anchor_count']['mean']:.2f}
    """

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def generate_all_visualizations(results_dir: Path = RESULTS_DIR, figures_dir: Path = FIGURES_DIR):
    """
    Generate all visualizations from saved results

    Args:
        results_dir: Directory containing results JSON files
        figures_dir: Directory to save figures
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Load drift tracking results
    drift_results_path = results_dir / "drift_tracking_results.json"
    if not drift_results_path.exists():
        print(f"Error: {drift_results_path} not found")
        return

    with open(drift_results_path, 'r') as f:
        drift_results = json.load(f)

    # Load anchor analysis if available
    anchor_results_path = results_dir / "anchor_analysis.json"
    if anchor_results_path.exists():
        with open(anchor_results_path, 'r') as f:
            anchor_analysis = json.load(f)
    else:
        anchor_analysis = None
        print("Warning: Anchor analysis results not found")

    # Plot individual drift curves for first few samples
    for i, result in enumerate(drift_results[:3]):
        save_path = figures_dir / f"drift_curve_{result['sample_id']}.png"
        plot_drift_curve_single(result, save_path=save_path, show=False)

    # Plot comparison of all samples
    save_path = figures_dir / "drift_curves_comparison.png"
    plot_drift_curves_comparison(drift_results, save_path=save_path, show=False)

    # Plot anchor effects if available
    if anchor_analysis:
        save_path = figures_dir / "anchor_effects.png"
        plot_anchor_effects(anchor_analysis, save_path=save_path, show=False)

    # Import and run drift pattern analysis
    from drift_tracking import analyze_drift_patterns
    drift_analysis = analyze_drift_patterns(drift_results)

    # Plot summary statistics
    save_path = figures_dir / "summary_statistics.png"
    plot_summary_statistics(drift_results, drift_analysis, save_path=save_path, show=False)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS GENERATED")
    print("="*80)


if __name__ == "__main__":
    generate_all_visualizations()
