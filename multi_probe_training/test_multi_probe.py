"""
Test script for multi-position probe training
Tests with a small subset of data to verify correctness
"""

import numpy as np
import torch
from probe_training_multilayer import (
    MultiPositionProbe,
    extract_all_positions_data,
    visualize_mse_heatmap,
    visualize_mse_statistics
)
from model_utils import ModelWrapper
from data_preparation import load_dataset
from config import FIGURES_DIR

def test_multi_probe():
    """Test the multi-position probe training with small dataset"""

    print("="*80)
    print("TESTING MULTI-POSITION PROBE TRAINING")
    print("="*80)

    # Configuration for testing
    NUM_LAYERS = 5  # Test with fewer layers
    ALPHA = 1.0
    TEST_SAMPLES = 10  # Use only 10 samples for quick test

    # Load small subset of dataset
    print("\nLoading dataset...")
    full_dataset = load_dataset()
    dataset = full_dataset[:TEST_SAMPLES]
    print(f"Using {len(dataset)} samples for testing")

    # Load model
    print("\nLoading model...")
    model_wrapper = ModelWrapper()
    model_wrapper.load_model()

    # Extract data
    print("\nExtracting hidden states...")
    X_all, y = extract_all_positions_data(model_wrapper, dataset, NUM_LAYERS)

    # Verify data shapes
    print("\nVerifying data shapes...")
    for layer in range(NUM_LAYERS):
        print(f"Layer {layer}: {X_all[layer].shape}")
    print(f"Targets: {y.shape}")

    # Train probes
    print("\nTraining probes...")
    probe_system = MultiPositionProbe(num_layers=NUM_LAYERS, alpha=ALPHA)
    results = probe_system.train(X_all, y, test_size=0.2)

    # Check results
    print("\nChecking results...")
    print(f"MSE matrix shape: {probe_system.mse_matrix.shape}")
    print(f"Expected shape: ({NUM_LAYERS}, {probe_system.num_tokens})")

    # Test prediction
    print("\nTesting prediction...")
    test_layer = 2
    test_pos = 5
    test_sample = X_all[test_layer][0, test_pos, :]  # First sample, position 5
    prediction = probe_system.predict(test_sample, test_layer, test_pos)
    print(f"Prediction for layer {test_layer}, position {test_pos}: {prediction[0]:.2f}")
    print(f"Actual value: {y[0]}")

    # Visualize
    print("\nGenerating visualizations...")
    heatmap_path = FIGURES_DIR / "test_probe_mse_heatmap.png"
    visualize_mse_heatmap(
        probe_system.mse_matrix,
        save_path=heatmap_path,
        title="Test: Probe MSE Heatmap"
    )

    stats_path = FIGURES_DIR / "test_probe_mse_statistics.png"
    visualize_mse_statistics(probe_system.mse_matrix, save_path=stats_path)

    # Save test probes
    print("\nSaving test probes...")
    probe_system.save("test_multi_position_probes.pkl")

    # Test loading
    print("\nTesting probe loading...")
    probe_system_loaded = MultiPositionProbe(num_layers=NUM_LAYERS, alpha=ALPHA)
    probe_system_loaded.load("test_multi_position_probes.pkl")

    # Verify loaded probe works
    prediction_loaded = probe_system_loaded.predict(test_sample, test_layer, test_pos)
    print(f"Prediction from loaded probe: {prediction_loaded[0]:.2f}")
    assert np.allclose(prediction, prediction_loaded), "Loaded probe predictions don't match!"

    # Cleanup
    model_wrapper.cleanup()

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nYou can now run the full training with:")
    print("python probe_training_multilayer.py")


if __name__ == "__main__":
    test_multi_probe()
