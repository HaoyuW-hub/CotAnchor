"""
Main Experiment Runner
Run the complete pilot experiment pipeline
"""

import argparse
from pathlib import Path
import os
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'

# Import modules
from config import RESULTS_DIR, FIGURES_DIR
from data_preparation import create_dataset, save_dataset, load_dataset
from model_utils import ModelWrapper
from probe_training import PrimeProbe, extract_training_data, visualize_probe_results
from drift_tracking import track_drift_batch, analyze_drift_patterns
from anchor_analysis import analyze_anchors_batch, identify_successful_anchors
from visualization import generate_all_visualizations


def run_full_pipeline(
    max_samples: int = None,
    skip_probe_training: bool = False,
    skip_drift_tracking: bool = False
):
    """
    Run the complete pilot experiment pipeline

    Args:
        max_samples: Maximum number of samples to process (None for all)
        skip_probe_training: Skip probe training if probe already exists
        skip_drift_tracking: Skip drift tracking if results already exist
    """
    print("\n" + "="*80)
    print("PILOT EXPERIMENT: REPRESENTATION DRIFT IN LONG COT REASONING")
    print("="*80)

    # Step 1: Data Preparation
    print("\n" + "-"*80)
    print("STEP 1: DATA PREPARATION")
    print("-"*80)
    dataset = load_dataset()

    # Step 2: Model Loading
    print("\n" + "-"*80)
    print("STEP 2: MODEL LOADING")
    print("-"*80)
    model_wrapper = ModelWrapper()
    model_wrapper.load_model()

    # Step 3: Probe Training
    print("\n" + "-"*80)
    print("STEP 3: PROBE TRAINING")
    print("-"*80)

    probe = PrimeProbe()
    probe_path = Path("models/prime_probe.pkl")

    if skip_probe_training and probe_path.exists():
        print("Loading existing probe...")
        probe.load()
    else:
        print("Training new probe...")
        X, y = extract_training_data(model_wrapper, dataset)
        results = probe.train(X, y, test_size=0.2)
        probe.save()

        # Visualize probe training results
        save_path = FIGURES_DIR / "probe_training_results.png"
        visualize_probe_results(results, save_path)

    # Step 4: Drift Tracking
    print("\n" + "-"*80)
    print("STEP 4: DRIFT TRACKING")
    print("-"*80)

    drift_results_path = RESULTS_DIR / "drift_tracking_results.json"

    if skip_drift_tracking and drift_results_path.exists():
        print("Loading existing drift tracking results...")
        import json
        with open(drift_results_path, 'r') as f:
            drift_results = json.load(f)
    else:
        print("Running drift tracking...")
        drift_results = track_drift_batch(
            dataset=dataset,
            model_wrapper=model_wrapper,
            probe=probe,
            max_samples=max_samples,
            save_results=True
        )

    # Analyze drift patterns
    drift_analysis = analyze_drift_patterns(drift_results)

    # Step 5: Anchor Analysis
    print("\n" + "-"*80)
    print("STEP 5: ANCHOR ANALYSIS")
    print("-"*80)

    anchor_analysis = analyze_anchors_batch(
        results=drift_results,
        window_size=50,
        save_results=True
    )

    # Identify successful anchors
    successful_anchors = identify_successful_anchors(
        anchor_analysis,
        threshold=0.05
    )

    # Step 6: Visualization
    print("\n" + "-"*80)
    print("STEP 6: VISUALIZATION")
    print("-"*80)

    generate_all_visualizations()

    # Step 7: Summary Report
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print(f"\nDataset:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Processed samples: {len(drift_results)}")

    print(f"\nProbe Performance:")
    print(f"  Layer: {probe.layer}")
    print(f"  Training completed: {probe.is_trained}")

    print(f"\nDrift Analysis:")
    print(f"  Mean probe score change: {drift_analysis['probe_score_change']['mean']:.4f}")
    print(f"  Mean cosine change: {drift_analysis['cosine_similarity_change']['mean']:.4f}")

    print(f"\nAnchor Analysis:")
    print(f"  Total anchors found: {anchor_analysis['aggregate_statistics']['total_anchors']}")
    print(f"  Successful anchors: {len(successful_anchors)}")

    print(f"\nOutput Files:")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Figures directory: {FIGURES_DIR}")

    # Cleanup
    model_wrapper.cleanup()

    print("\n" + "="*80)
    print("PILOT EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)


def run_data_preparation_only():
    """Run only data preparation step"""
    print("Generating dataset...")
    dataset = create_dataset()
    save_dataset(dataset)
    print("Data preparation completed!")


def run_visualization_only():
    """Run only visualization step"""
    print("Generating visualizations...")
    generate_all_visualizations()
    print("Visualization completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pilot Experiment: Representation Drift in Long CoT Reasoning"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "data", "viz"],
        help="Experiment mode: full pipeline, data preparation only, or visualization only"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )

    parser.add_argument(
        "--skip-probe-training",
        action="store_true",
        help="Skip probe training if probe already exists"
    )

    parser.add_argument(
        "--skip-drift-tracking",
        action="store_true",
        help="Skip drift tracking if results already exist"
    )

    args = parser.parse_args()

    if args.mode == "full":
        run_full_pipeline(
            max_samples=args.max_samples,
            skip_probe_training=args.skip_probe_training,
            skip_drift_tracking=args.skip_drift_tracking
        )
    elif args.mode == "data":
        run_data_preparation_only()
    elif args.mode == "viz":
        run_visualization_only()
