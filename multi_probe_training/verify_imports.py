#!/usr/bin/env python
"""
Verify that all imports work correctly in multi_probe_training folder
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

print("="*80)
print("IMPORT VERIFICATION TEST")
print("="*80)
print(f"\nParent directory: {parent_dir}")
print(f"Python path[0]: {sys.path[0]}")

# Test 1: config
print("\n[1/5] Testing config import...")
try:
    from config import MODELS_DIR, FIGURES_DIR, DATA_DIR
    print(f"  ✓ config imported successfully")
    print(f"    MODELS_DIR: {MODELS_DIR}")
    print(f"    FIGURES_DIR: {FIGURES_DIR}")
    print(f"    DATA_DIR: {DATA_DIR}")
except Exception as e:
    print(f"  ✗ config import failed: {e}")
    sys.exit(1)

# Test 2: model_utils
print("\n[2/5] Testing model_utils import...")
try:
    from model_utils import ModelWrapper
    print(f"  ✓ model_utils imported successfully")
    print(f"    ModelWrapper class available")
except Exception as e:
    print(f"  ✗ model_utils import failed: {e}")
    sys.exit(1)

# Test 3: data_preparation
print("\n[3/5] Testing data_preparation import...")
try:
    from data_preparation import load_dataset
    print(f"  ✓ data_preparation imported successfully")
    print(f"    load_dataset function available")
except Exception as e:
    print(f"  ✗ data_preparation import failed: {e}")
    sys.exit(1)

# Test 4: probe_training_multilayer
print("\n[4/5] Testing probe_training_multilayer import...")
try:
    from probe_training_multilayer import MultiPositionProbe
    print(f"  ✓ probe_training_multilayer imported successfully")
    print(f"    MultiPositionProbe class available")
except Exception as e:
    print(f"  ✗ probe_training_multilayer import failed: {e}")
    sys.exit(1)

# Test 5: Required packages
print("\n[5/5] Testing required packages...")
required_packages = [
    ('numpy', 'numpy'),
    ('sklearn', 'sklearn'),
    ('torch', 'torch'),
    ('matplotlib', 'matplotlib'),
    ('tqdm', 'tqdm'),
]

all_packages_ok = True
for package_name, import_name in required_packages:
    try:
        __import__(import_name)
        print(f"  ✓ {package_name} available")
    except ImportError:
        print(f"  ✗ {package_name} NOT available")
        all_packages_ok = False

if not all_packages_ok:
    print("\n⚠️  Some packages are missing. Install with:")
    print("  pip install numpy scikit-learn torch matplotlib tqdm seaborn")
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL IMPORTS VERIFIED SUCCESSFULLY!")
print("="*80)
print("\nYou can now run:")
print("  python test_multi_probe.py")
print("  python probe_training_multilayer.py")
print("  python analyze_multi_probe.py")
print("  python run_pipeline.py")
