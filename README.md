# Pilot Experiment: Representation Drift in Long CoT Reasoning

This repository contains the implementation of a pilot experiment to investigate **representation drift** and **anchor mechanisms** in long chain-of-thought (CoT) reasoning using large language models.

## 📋 Overview

The experiment validates whether:
1. **Representation drift exists**: Specific mathematical constraints (e.g., "n is prime") show measurable drift in hidden layer representations during long CoT reasoning
2. **Anchors can stabilize representations**: Special tokens like "Wait" or "However" can cause representations to regress toward initial states

## 🎯 Experiment Design

### Task: Prime Number Detection
- **Positive examples**: Large prime numbers (797, 997, 883, etc.)
- **Negative examples**: Composite numbers (801, 999, 885, etc.)
- **Sample size**: 20 samples (10 positive, 10 negative)

### Model
- **Primary model**: DeepSeek-R1-Distill-Qwen-7B
- **Target layer**: Layer 16 (encodes semantic and logical information)

### Methodology
1. **Train linear probe** to detect "is prime" concept in hidden states
2. **Track drift** by monitoring probe scores during CoT generation (every 20 tokens)
3. **Analyze anchors** by identifying special tokens and their effects on representation recovery

## 📁 Project Structure

```
experiment/
├── config.py                 # Configuration and constants
├── data_preparation.py       # Dataset generation
├── model_utils.py           # Model loading and hidden state extraction
├── probe_training.py        # Linear probe training
├── drift_tracking.py        # Representation drift tracking
├── anchor_analysis.py       # Anchor token effect analysis
├── visualization.py         # Plotting and visualization
├── main.py                  # Main experiment runner
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not tracked in git)
├── data/                    # Generated datasets
├── results/                 # Experiment results (JSON)
├── models/                  # Trained probes
└── figures/                 # Generated plots
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda
conda create -n pilot_exp python=3.10
conda activate pilot_exp
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your API keys:

```bash
HF_TOKEN="your_huggingface_token"
DEEPSEEK_API_KEY="your_deepseek_api_key"  # Optional
AIHUBMIX_KEY="your_aihubmix_key"  # Optional
```

### 3. Run Experiment

**Full pipeline** (all steps):
```bash
python main.py --mode full
```

**Quick test** (3 samples only):
```bash
python main.py --mode full --max-samples 3
```

**Data preparation only**:
```bash
python main.py --mode data
```

**Visualization only** (from existing results):
```bash
python main.py --mode viz
```

### 4. Individual Modules

You can also run individual modules:

```bash
# Generate dataset
python data_preparation.py

# Train probe
python probe_training.py

# Track drift (requires trained probe)
python drift_tracking.py

# Analyze anchors (requires drift results)
python anchor_analysis.py

# Generate visualizations (requires results)
python visualization.py
```

## 📊 Expected Outputs

### Results (JSON files in `results/`)
- `pilot_dataset.json`: Generated dataset with prompts
- `drift_tracking_results.json`: Drift metrics for each sample
- `anchor_analysis.json`: Anchor token effects

### Figures (PNG files in `figures/`)
- `probe_training_results.png`: Probe performance metrics
- `drift_curve_{sample_id}.png`: Individual drift curves (first 3 samples)
- `drift_curves_comparison.png`: Comparison across all samples
- `anchor_effects.png`: Anchor token effect distributions
- `summary_statistics.png`: Overall experiment summary

## 📈 Success Criteria

The pilot experiment is considered successful if:

1. **Probe accuracy > 90%**: Linear probe can accurately detect "is prime" concept
2. **Significant drift observed**: Probe score decreases by >10% after 500 tokens
3. **Anchor effects detected**: At least 2-3 samples show representation recovery after "Wait" tokens

## 🔬 Key Metrics

### Drift Metrics
- **Probe score**: Probability that model's representation indicates "n is prime"
- **Cosine similarity**: Alignment between current and initial hidden states
- **Probe score change**: Difference between final and initial probe scores
- **Cosine change**: Difference between final and initial cosine similarity

### Anchor Metrics
- **Anchor Effectiveness Score (AES)**: Representation recovery after anchor token
- **Positive/negative changes**: Count of anchors that improve/degrade representations

## 🛠️ Troubleshooting

### Memory Issues
If you encounter OOM errors:
- Reduce `max_length` in `config.py`
- Increase `sample_interval` to extract fewer hidden states
- Use a smaller model or CPU instead of GPU

### Model Loading Issues
- Ensure your HuggingFace token has access to the model
- Check internet connection for model download
- Verify sufficient disk space (~15GB for model)

### No Anchors Found
- This is normal if the model doesn't generate "Wait" tokens
- Try increasing `max_length` for longer CoT
- Check `ANCHOR_TOKENS` list in `config.py`

## 📝 Configuration Options

Edit `config.py` to customize:

```python
# Model settings
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TARGET_LAYER = 16  # Which layer to extract from
MAX_LENGTH = 1000  # Maximum tokens to generate

# Sampling settings
SAMPLE_INTERVAL = 20  # Extract every N tokens
NUM_POSITIVE_SAMPLES = 10
NUM_NEGATIVE_SAMPLES = 10

# Anchor tokens to track
ANCHOR_TOKENS = ["Wait", "However", "Actually", "Let me", "So"]
```

## 🔍 Next Steps

After completing the pilot experiment:

1. **Analyze results** to determine if drift is observable
2. **Scale up** to full experiment with more samples and tasks
3. **Test interventions** by actively inserting anchor tokens
4. **Explore other tasks** (logic puzzles, multi-hop QA)
5. **Try different layers** to find optimal representation layer

## 📚 References

Based on the research plan described in `Research_plan.md`:
- Task 1: Representation drift observation
- Task 2: Anchor token stability effects

## 📄 License

This is research code for academic purposes.

## 🤝 Contributing

This is a pilot experiment for thesis research. For questions or suggestions, please contact the researcher.

## 📧 Contact

[Add your contact information]

---

**Note**: This pilot experiment is designed to validate the experimental pipeline before conducting the full-scale study. Results from this pilot will inform the design of the larger experiment.
