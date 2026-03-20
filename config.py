"""
Configuration file for the Pilot Experiment
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEVICE = "cuda"  # or "cpu" if no GPU available
MAX_LENGTH = 1000  # Maximum tokens to generate
TEMPERATURE = 0.7
TOP_P = 0.95

# API Keys (loaded from .env)
HF_TOKEN = os.getenv("HF_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Experiment Parameters
TARGET_LAYER = 16  # Layer to extract hidden states from
ALTERNATIVE_LAYERS = [12, 16, 20, 24]  # Layers to try if primary fails
SAMPLE_INTERVAL = 20  # Extract hidden state every N tokens
NUM_POSITIVE_SAMPLES = 10  # Number of prime examples
NUM_NEGATIVE_SAMPLES = 10  # Number of composite examples

# Data Configuration
PRIME_NUMBERS = [797, 997, 883, 991, 787, 929, 953, 967, 983, 977]
COMPOSITE_NUMBERS = [801, 999, 885, 993, 789, 931, 955, 969, 985, 979]

# Prompt Template
PROMPT_TEMPLATE = """Let n = {number}. First, explain why n is {prime_or_composite}. Then, calculate (n+1)/2 and discuss its properties through a long chain of thought. Finally, verify if n remains the same throughout your reasoning."""

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
    dir_path.mkdir(exist_ok=True)

# Anchor tokens to track
ANCHOR_TOKENS = ["Wait", "wait", "However", "however", "Actually", "actually",
                 "Let me", "So", "Therefore", "Hmm", "Oops"]

# Visualization settings
FIGURE_DPI = 300
FIGURE_SIZE = (12, 6)
