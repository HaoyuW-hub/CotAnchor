"""
Visualization utilities for the attention score tracking experiment.

Three plots:
  1. Heatmap: avg attention score per (step, layer)
  2. Line plot: A_t (layer-averaged) vs generated step t
  3. Multi-line: per-layer decay curves for selected layers
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


HIGHLIGHT_LAYERS = [0, 6, 13, 20, 27]  # representative layers to show in multi-line plot


def visualize_attention_heatmap(
    avg_attn_matrix: np.ndarray,
    valid_counts: np.ndarray,
    save_path: Path = None,
    title: str = "Condition Token Attention Score: Generated Step vs Layer",
):
    """
    Heatmap of mean attention score across (generated_step, layer).

    Args:
        avg_attn_matrix: (truncate_at, NUM_LAYERS), NaN where no data
        valid_counts: (truncate_at,) number of samples per step
    """
    masked = np.ma.masked_invalid(avg_attn_matrix)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="lightgrey")

    n_steps, n_layers = avg_attn_matrix.shape
    finite = avg_attn_matrix[np.isfinite(avg_attn_matrix)]
    vmin = 0.0
    vmax = float(np.percentile(finite, 99)) if finite.size > 0 else 1.0

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [8, 1.5], "hspace": 0.08},
    )

    im = axes[0].imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )
    axes[0].set_ylabel("Generated Token Step", fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight="bold")
    _set_layer_xticks(axes[0], n_layers, show_labels=False)
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label("Mean attention score (avg over heads & samples)", rotation=270, labelpad=20, fontsize=11)

    axes[1].step(np.arange(n_steps), valid_counts, where="mid", color="steelblue", linewidth=1.5)
    axes[1].set_xlabel("Generated Token Step", fontsize=12)
    axes[1].set_ylabel("# samples", fontsize=10)
    axes[1].set_xlim(-0.5, n_steps - 0.5)
    axes[1].grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def visualize_attention_layer_avg(
    avg_attn_per_step: np.ndarray,
    valid_counts: np.ndarray,
    save_path: Path = None,
    title: str = "A_t: Layer-Averaged Condition Token Attention vs CoT Length",
):
    """
    Line plot of A_t (attention averaged over all layers) vs generated step t.
    Visualises the dilution curve described in intro.md.
    """
    n_steps = len(avg_attn_per_step)
    steps = np.arange(n_steps)
    valid_mask = valid_counts > 0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps[valid_mask], avg_attn_per_step[valid_mask], linewidth=2, color="tab:blue", label="A_t (layer mean)")
    ax.set_xlabel("Generated Token Step (CoT length)", fontsize=12)
    ax.set_ylabel("Mean Attention Score to Condition Token", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def visualize_attention_per_layer(
    avg_attn_matrix: np.ndarray,
    valid_counts: np.ndarray,
    highlight_layers: list = None,
    save_path: Path = None,
    title: str = "Per-Layer Condition Token Attention vs CoT Length",
):
    """
    Multi-line plot: one curve per selected layer showing attention decay.

    Args:
        highlight_layers: layer indices to plot; defaults to HIGHLIGHT_LAYERS
    """
    if highlight_layers is None:
        highlight_layers = HIGHLIGHT_LAYERS

    n_steps, n_layers = avg_attn_matrix.shape
    steps = np.arange(n_steps)
    valid_mask = valid_counts > 0

    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, layer in enumerate(highlight_layers):
        if layer >= n_layers:
            continue
        curve = avg_attn_matrix[:, layer]
        ax.plot(
            steps[valid_mask],
            curve[valid_mask],
            linewidth=1.8,
            color=cmap(i % 10),
            label=f"Layer {layer}",
        )

    ax.set_xlabel("Generated Token Step (CoT length)", fontsize=12)
    ax.set_ylabel("Mean Attention Score to Condition Token", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def _set_layer_xticks(ax, n_layers: int, show_labels: bool = True):
    tick_step = max(1, n_layers // 14)
    ticks = np.arange(0, n_layers, tick_step)
    ax.set_xticks(ticks)
    if show_labels:
        ax.set_xticklabels(ticks)
        ax.set_xlabel("Layer", fontsize=12)
