"""
Visualization utilities for the CoT probe tracking experiment.

All plots follow the same conventions as probe_visualization.py:
  - viridis_r for MSE (darker = lower = better)
  - viridis   for accuracy (darker = higher = better)
  - 300 dpi PNG output
  - NaN cells masked (shown as grey)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def visualize_cot_mse_heatmap(
    avg_mse_matrix: np.ndarray,
    valid_counts: np.ndarray,
    save_path: Path = None,
    title: str = "CoT Probe MSE: Generated Token Step vs Layer",
    percentile_clip: float = 95.0,
):
    """
    Heatmap of avg MSE across (generated_token_step, layer).

    Args:
        avg_mse_matrix: (truncate_at, NUM_LAYERS), NaN where no samples reached
        valid_counts:   (truncate_at,) number of samples contributing per step
    """
    finite = avg_mse_matrix[np.isfinite(avg_mse_matrix)]
    if finite.size == 0:
        print("No finite values in MSE matrix, skipping heatmap.")
        return
    vmin = finite.min()
    vmax = np.percentile(finite, percentile_clip)
    print(f"CoT MSE range: [{finite.min():.4f}, {finite.max():.4f}] — clipping at {percentile_clip}th pct: {vmax:.4f}")

    masked = np.ma.masked_invalid(avg_mse_matrix)
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color='lightgrey')

    n_steps, n_layers = avg_mse_matrix.shape
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [8, 1.5], "hspace": 0.08}
    )

    im = axes[0].imshow(
        masked,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax,
        origin='upper',
    )
    axes[0].set_ylabel('Generated Token Step', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    _set_layer_xticks(axes[0], n_layers, show_labels=False)
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label(f'Avg MSE in log space (clipped at {percentile_clip}th pct)', rotation=270, labelpad=20, fontsize=11)

    # Valid-counts bar chart below
    axes[1].bar(np.arange(n_layers), np.full(n_layers, 1), alpha=0)  # invisible spacer
    axes[1].step(np.arange(n_steps), valid_counts, where='mid', color='steelblue', linewidth=1.5, label='# samples')
    axes[1].set_xlabel('Generated Token Step', fontsize=12)
    axes[1].set_ylabel('# samples', fontsize=10)
    axes[1].set_xlim(-0.5, n_steps - 0.5)
    axes[1].grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def visualize_cot_acc_heatmap(
    avg_acc_matrix: np.ndarray,
    valid_counts: np.ndarray,
    save_path: Path = None,
    title: str = "CoT Probe Accuracy (1%): Generated Token Step vs Layer",
):
    """
    Heatmap of avg accuracy across (generated_token_step, layer).
    """
    masked = np.ma.masked_invalid(avg_acc_matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgrey')

    n_steps, n_layers = avg_acc_matrix.shape
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [8, 1.5], "hspace": 0.08}
    )

    im = axes[0].imshow(
        masked,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        vmin=0.0,
        vmax=1.0,
        origin='upper',
    )
    axes[0].set_ylabel('Generated Token Step', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    _set_layer_xticks(axes[0], n_layers, show_labels=False)
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('Accuracy  |pred − true| ≤ 1% × true', rotation=270, labelpad=20, fontsize=11)

    axes[1].step(np.arange(n_steps), valid_counts, where='mid', color='steelblue', linewidth=1.5)
    axes[1].set_xlabel('Generated Token Step', fontsize=12)
    axes[1].set_ylabel('# samples', fontsize=10)
    axes[1].set_xlim(-0.5, n_steps - 0.5)
    axes[1].grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Line plots: averaged over one dimension
# ---------------------------------------------------------------------------

def visualize_cot_layer_avg(
    avg_mse_matrix: np.ndarray,
    avg_acc_matrix: np.ndarray,
    save_path: Path = None,
):
    """
    Two panels: MSE and accuracy averaged over layers, plotted vs. generated step.
    Shows how the average probe quality changes as CoT grows.
    """
    # nanmean to ignore NaN-filled tail
    mse_by_step = np.nanmean(avg_mse_matrix, axis=1)  # (truncate_at,)
    acc_by_step = np.nanmean(avg_acc_matrix, axis=1)

    n_steps = avg_mse_matrix.shape[0]
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, mse_by_step, linewidth=2, color='tab:blue')
    axes[0].set_xlabel('Generated Token Step', fontsize=11)
    axes[0].set_ylabel('Avg MSE (log space, averaged over layers)', fontsize=11)
    axes[0].set_title('Probe MSE vs. CoT Length', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, acc_by_step, linewidth=2, color='tab:green')
    axes[1].set_xlabel('Generated Token Step', fontsize=11)
    axes[1].set_ylabel('Avg Accuracy (averaged over layers)', fontsize=11)
    axes[1].set_title('Probe Accuracy vs. CoT Length', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def visualize_cot_step_avg(
    avg_mse_matrix: np.ndarray,
    avg_acc_matrix: np.ndarray,
    save_path: Path = None,
):
    """
    Two panels: MSE and accuracy averaged over generated steps, plotted vs. layer.
    Mirrors probe_visualization.visualize_mse_statistics for easy comparison.
    """
    mse_by_layer = np.nanmean(avg_mse_matrix, axis=0)  # (NUM_LAYERS,)
    acc_by_layer = np.nanmean(avg_acc_matrix, axis=0)

    n_layers = avg_mse_matrix.shape[1]
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(layers, mse_by_layer, marker='o', linewidth=2, color='tab:orange')
    axes[0].set_xlabel('Layer', fontsize=11)
    axes[0].set_ylabel('Avg MSE (averaged over CoT steps)', fontsize=11)
    axes[0].set_title('Probe MSE by Layer (CoT context)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(layers)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layers, acc_by_layer, marker='o', linewidth=2, color='tab:purple')
    axes[1].set_xlabel('Layer', fontsize=11)
    axes[1].set_ylabel('Avg Accuracy (averaged over CoT steps)', fontsize=11)
    axes[1].set_title('Probe Accuracy by Layer (CoT context)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(layers)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _set_layer_xticks(ax, n_layers: int, show_labels: bool = True):
    tick_step = max(1, n_layers // 14)
    ticks = np.arange(0, n_layers, tick_step)
    ax.set_xticks(ticks)
    if show_labels:
        ax.set_xticklabels(ticks)
        ax.set_xlabel('Layer', fontsize=12)
