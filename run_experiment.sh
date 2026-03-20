#!/bin/bash

# Quick start script for the pilot experiment

echo "=========================================="
echo "Pilot Experiment Quick Start"
echo "=========================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with your HuggingFace token:"
    echo '  HF_TOKEN="your_token_here"'
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "Anchor" ]; then
    echo "Creating virtual environment..."
    python -m venv Anchor
fi

# Activate virtual environment
echo "Activating virtual environment..."
source Anchor/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Ask user what to run
echo ""
echo "What would you like to run?"
echo "1) Full pipeline (all 20 samples)"
echo "2) Quick test (3 samples)"
echo "3) Data preparation only"
echo "4) Visualization only (from existing results)"
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "Running full pipeline..."
        python main.py --mode full
        ;;
    2)
        echo "Running quick test (3 samples)..."
        python main.py --mode full --max-samples 3
        ;;
    3)
        echo "Running data preparation..."
        python main.py --mode data
        ;;
    4)
        echo "Generating visualizations..."
        python main.py --mode viz
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Check results/ for JSON output"
echo "Check figures/ for visualizations"
echo "=========================================="
