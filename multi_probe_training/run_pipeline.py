"""
Quick Start Guide - Python Version
Run the complete multi-position probe training pipeline
"""

import sys
import subprocess
from pathlib import Path

def run_command(description, command):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(description)
    print(f"{'='*80}\n")

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        return False

    print(f"\n✓ {description} completed successfully!")
    return True


def main():
    print("="*80)
    print("MULTI-POSITION PROBE TRAINING PIPELINE")
    print("="*80)

    # Check if we're in the right directory
    if not Path("probe_training_multilayer.py").exists():
        print("\n❌ Error: probe_training_multilayer.py not found!")
        print("Please run this script from the CotAnchor directory.")
        sys.exit(1)

    # Step 1: Test (optional)
    print("\nStep 1: Quick Test (Optional)")
    print("This verifies the code works with a small dataset.")
    response = input("Run test? (y/n): ").strip().lower()

    if response == 'y':
        if not run_command(
            "Running quick test",
            "python test_multi_probe.py"
        ):
            sys.exit(1)

    # Step 2: Full training
    print("\nStep 2: Full Training")
    print("This trains probes for all token positions and layers.")
    print("⚠️  WARNING: This may take a while depending on dataset size.")
    response = input("Proceed with full training? (y/n): ").strip().lower()

    if response == 'y':
        if not run_command(
            "Running full training",
            "python probe_training_multilayer.py"
        ):
            sys.exit(1)
    else:
        print("\nTraining skipped. Exiting.")
        sys.exit(0)

    # Step 3: Analysis
    print("\nStep 3: Analysis")
    print("This generates detailed analysis and visualizations.")
    response = input("Run analysis? (y/n): ").strip().lower()

    if response == 'y':
        if not run_command(
            "Running analysis",
            "python analyze_multi_probe.py"
        ):
            sys.exit(1)

    # Summary
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)

    print("\n📁 Generated files:")
    print("  models/")
    print("    ├── multi_position_probes.pkl")
    print("    ├── multi_probe_results.json")
    print("    └── analysis_summary_report.json")

    print("\n📊 Generated figures:")
    print("  figures/")
    print("    ├── probe_mse_heatmap.png")
    print("    ├── probe_mse_statistics.png")
    print("    ├── analysis_layer_trends.png")
    print("    ├── analysis_token_positions.png")
    print("    └── analysis_layer_token_interaction.png")

    print("\n✓ Check these files for your results!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
