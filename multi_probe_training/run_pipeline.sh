#!/bin/bash
# Quick Start Script for Multi-Position Probe Training
# This script runs the complete pipeline

echo "=========================================="
echo "Multi-Position Probe Training Pipeline"
echo "=========================================="
echo ""

# Step 1: Test with small dataset (optional but recommended)
echo "Step 1: Running quick test..."
echo "This will verify the code works correctly with a small dataset."
read -p "Run test? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python test_multi_probe.py
    if [ $? -ne 0 ]; then
        echo "Test failed! Please check the error messages above."
        exit 1
    fi
    echo "Test passed!"
    echo ""
fi

# Step 2: Full training
echo "Step 2: Running full training..."
echo "This will train probes for all token positions and layers."
echo "WARNING: This may take a while depending on your dataset size and model."
read -p "Proceed with full training? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python probe_training_multilayer.py
    if [ $? -ne 0 ]; then
        echo "Training failed! Please check the error messages above."
        exit 1
    fi
    echo "Training completed!"
    echo ""
fi

# Step 3: Analysis
echo "Step 3: Running analysis..."
echo "This will generate detailed analysis and additional visualizations."
read -p "Run analysis? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python analyze_multi_probe.py
    if [ $? -ne 0 ]; then
        echo "Analysis failed! Please check the error messages above."
        exit 1
    fi
    echo "Analysis completed!"
    echo ""
fi

# Summary
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  models/multi_position_probes.pkl"
echo "  models/multi_probe_results.json"
echo "  models/analysis_summary_report.json"
echo ""
echo "Generated figures:"
echo "  figures/probe_mse_heatmap.png"
echo "  figures/probe_mse_statistics.png"
echo "  figures/analysis_layer_trends.png"
echo "  figures/analysis_token_positions.png"
echo "  figures/analysis_layer_token_interaction.png"
echo ""
echo "Check these files for your results!"
