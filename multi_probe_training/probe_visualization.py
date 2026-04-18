"""
Visualization utilities for multi-position probe training results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_mse_heatmap(
    mse_matrix: np.ndarray,
    save_path: Path = None,
    title: str = "Probe MSE Heatmap",
    percentile_clip: float = 95.0
):
    """
    Heatmap of per-(layer, token_position) MSE with outlier clipping.

    Args:
        mse_matrix: (n_layers, n_tokens) MSE values in log space
        percentile_clip: Values above this percentile are clipped for display
    """
    n_layers, n_tokens = mse_matrix.shape
    vmax = np.percentile(mse_matrix, percentile_clip)
    mse_clipped = np.clip(mse_matrix, None, vmax)
    print(f"MSE range: [{mse_matrix.min():.4f}, {mse_matrix.max():.4f}] — clipping at {percentile_clip}th pct: {vmax:.4f}")

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        mse_clipped.T,
        aspect='auto',
        cmap='viridis_r',  # darker = lower MSE = better
        interpolation='nearest',
        vmin=mse_matrix.min(),
        vmax=vmax
    )

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Token Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels(np.arange(n_layers))
    ax.set_yticks(np.arange(n_tokens))
    ax.set_yticklabels(np.arange(n_tokens))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'MSE in log space (clipped at {percentile_clip}th pct)', rotation=270, labelpad=20, fontsize=11)

    ax.set_xticks(np.arange(n_layers) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_tokens) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def visualize_acc_heatmap(
    acc_matrix: np.ndarray,
    save_path: Path = None,
    title: str = "Probe Accuracy Heatmap (1% threshold)"
):
    """
    Heatmap of per-(layer, token_position) accuracy.

    Args:
        acc_matrix: (n_layers, n_tokens) accuracy values in [0, 1]
    """
    n_layers, n_tokens = acc_matrix.shape

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        acc_matrix.T,
        aspect='auto',
        cmap='viridis',  # darker = higher accuracy = better
        interpolation='nearest',
        vmin=0.0,
        vmax=1.0
    )

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Token Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels(np.arange(n_layers))
    ax.set_yticks(np.arange(n_tokens))
    ax.set_yticklabels(np.arange(n_tokens))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy  |pred − true| ≤ 1% × true', rotation=270, labelpad=20, fontsize=11)

    ax.set_xticks(np.arange(n_layers) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_tokens) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def visualize_mse_statistics(
    mse_matrix: np.ndarray,
    save_path: Path = None,
    percentile_clip: float = 95.0
):
    """
    Line plots of MSE averaged over layers and over token positions.

    Args:
        mse_matrix: (n_layers, n_tokens) MSE values
        percentile_clip: Values above this percentile are clipped before averaging
    """
    n_layers, n_tokens = mse_matrix.shape
    vmax = np.percentile(mse_matrix, percentile_clip)
    mse_clipped = np.clip(mse_matrix, None, vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(n_layers), mse_clipped.mean(axis=1), marker='o', linewidth=2)
    axes[0].set_xlabel('Layer', fontsize=11)
    axes[0].set_ylabel('Average MSE (clipped)', fontsize=11)
    axes[0].set_title(f'MSE by Layer (clipped at {percentile_clip}th pct)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(n_tokens), mse_clipped.mean(axis=0), marker='o', linewidth=2, color='orange')
    axes[1].set_xlabel('Token Position', fontsize=11)
    axes[1].set_ylabel('Average MSE (clipped)', fontsize=11)
    axes[1].set_title(f'MSE by Token Position (clipped at {percentile_clip}th pct)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def visualize_acc_statistics(
    acc_matrix: np.ndarray,
    save_path: Path = None
):
    """
    Line plots of accuracy averaged over layers and over token positions.

    Args:
        acc_matrix: (n_layers, n_tokens) accuracy values in [0, 1]
    """
    n_layers, n_tokens = acc_matrix.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(n_layers), acc_matrix.mean(axis=1), marker='o', linewidth=2, color='green')
    axes[0].set_xlabel('Layer', fontsize=11)
    axes[0].set_ylabel('Average Accuracy', fontsize=11)
    axes[0].set_title('Accuracy by Layer (averaged over tokens)', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(n_tokens), acc_matrix.mean(axis=0), marker='o', linewidth=2, color='purple')
    axes[1].set_xlabel('Token Position', fontsize=11)
    axes[1].set_ylabel('Average Accuracy', fontsize=11)
    axes[1].set_title('Accuracy by Token Position (averaged over layers)', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
