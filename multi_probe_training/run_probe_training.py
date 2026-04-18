"""
Entry point for multi-layer multi-position probe training.
Orchestrates data extraction, training, saving, and visualization.
"""

import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from config import FIGURES_DIR
from model_utils import ModelWrapper
from data_preparation import load_dataset

from probe_training_multilayer import (
    MultiPositionProbe,
    extract_all_positions_data,
    save_results_json,
)
from probe_visualization import (
    visualize_mse_heatmap,
    visualize_acc_heatmap,
    visualize_mse_statistics,
    visualize_acc_statistics,
)

NUM_LAYERS = 28
ALPHA = 100.0

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-LAYER MULTI-POSITION PROBE TRAINING")
    print("=" * 80)

    dataset = load_dataset()

    model_wrapper = ModelWrapper()
    model_wrapper.load_model()

    X_all, y = extract_all_positions_data(model_wrapper, dataset, NUM_LAYERS)

    probe_system = MultiPositionProbe(num_layers=NUM_LAYERS, alpha=ALPHA)
    results = probe_system.train(X_all, y, test_size=0.2)

    probe_system.save()
    save_results_json(results, probe_system.mse_matrix, probe_system.acc_matrix)

    model_wrapper.cleanup()

    visualize_mse_heatmap(
        probe_system.mse_matrix,
        save_path=FIGURES_DIR / "probe_mse_heatmap.png",
        title="Probe MSE (log space): Layer vs Token Position",
        percentile_clip=95.0
    )
    visualize_acc_heatmap(
        probe_system.acc_matrix,
        save_path=FIGURES_DIR / "probe_acc_heatmap.png",
        title="Probe Accuracy (1% threshold): Layer vs Token Position"
    )
    visualize_mse_statistics(
        probe_system.mse_matrix,
        save_path=FIGURES_DIR / "probe_mse_statistics.png",
        percentile_clip=95.0
    )
    visualize_acc_statistics(
        probe_system.acc_matrix,
        save_path=FIGURES_DIR / "probe_acc_statistics.png"
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
