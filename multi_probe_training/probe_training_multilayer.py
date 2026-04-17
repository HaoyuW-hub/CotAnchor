"""
Multi-layer Multi-position Probe Training Module
Train probes for each token position and each layer to predict the number n
"""

import numpy as np
import pickle
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json

import sys
from pathlib import Path
# Add parent directory to path to import from CotAnchor root
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from config import MODELS_DIR, FIGURES_DIR
from model_utils import ModelWrapper
from data_preparation import load_dataset


class MultiPositionProbe:
    """
    Multi-position probe system that trains separate probes for each
    (token_position, layer) combination to predict the number n.
    """

    def __init__(self, num_layers: int, alpha: float = 1.0):
        """
        Args:
            num_layers: Number of layers in the model
            alpha: Regularization strength for Ridge regression
        """
        self.num_layers = num_layers
        self.alpha = alpha
        # probes[layer][token_pos] = trained Ridge model
        self.probes = {}
        self.mse_matrix = None  # Will store MSE for visualization
        self.num_tokens = None

    def train(
        self,
        X_all: Dict[int, np.ndarray],  # {layer: (n_samples, n_tokens, hidden_dim)}
        y: np.ndarray,  # (n_samples,) - target numbers
        test_size: float = 0.2
    ) -> Dict:
        """
        Train probes for all (layer, token_position) combinations.

        Args:
            X_all: Dictionary mapping layer index to hidden states
                   Shape: (n_samples, n_tokens, hidden_dim)
            y: Target numbers (n_samples,)
            test_size: Fraction for test split

        Returns:
            Dictionary with training results and MSE matrix
        """
        print(f"\n{'='*80}")
        print("MULTI-POSITION PROBE TRAINING")
        print(f"{'='*80}")

        # Get dimensions
        first_layer = list(X_all.keys())[0]
        n_samples, n_tokens, hidden_dim = X_all[first_layer].shape
        self.num_tokens = n_tokens

        print(f"Number of samples: {n_samples}")
        print(f"Number of tokens: {n_tokens}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Target range: [{y.min()}, {y.max()}]")

        # Split indices
        if test_size > 0 and n_samples > 4:
            train_idx, test_idx = train_test_split(
                np.arange(n_samples), test_size=test_size, random_state=42
            )
        else:
            train_idx = test_idx = np.arange(n_samples)

        y_train, y_test = y[train_idx], y[test_idx]

        # Initialize MSE matrix: (n_layers, n_tokens)
        mse_train_matrix = np.zeros((self.num_layers, n_tokens))
        mse_test_matrix = np.zeros((self.num_layers, n_tokens))

        # Train probes for each layer and token position
        print(f"\nTraining {self.num_layers * n_tokens} probes...")

        for layer in tqdm(range(self.num_layers), desc="Layers"):
            if layer not in X_all:
                print(f"Warning: Layer {layer} not found in data, skipping...")
                continue

            self.probes[layer] = {}
            X_layer = X_all[layer]  # (n_samples, n_tokens, hidden_dim)

            for token_pos in range(n_tokens):
                # Extract hidden states for this token position
                X_pos = X_layer[:, token_pos, :]  # (n_samples, hidden_dim)
                X_train = X_pos[train_idx]
                X_test = X_pos[test_idx]

                # Train Ridge probe
                probe = Ridge(alpha=self.alpha)
                probe.fit(X_train, y_train)

                # Evaluate
                y_pred_train = probe.predict(X_train)
                y_pred_test = probe.predict(X_test)

                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)

                # Store results
                self.probes[layer][token_pos] = probe
                mse_train_matrix[layer, token_pos] = mse_train
                mse_test_matrix[layer, token_pos] = mse_test

        self.mse_matrix = mse_test_matrix

        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Average Test MSE: {mse_test_matrix.mean():.4f}")
        print(f"Min Test MSE: {mse_test_matrix.min():.4f}")
        print(f"Max Test MSE: {mse_test_matrix.max():.4f}")

        return {
            'mse_train_matrix': mse_train_matrix,
            'mse_test_matrix': mse_test_matrix,
            'y_train': y_train,
            'y_test': y_test,
            'train_idx': train_idx,
            'test_idx': test_idx
        }

    def predict(self, X: np.ndarray, layer: int, token_pos: int) -> np.ndarray:
        """
        Predict using a specific probe.

        Args:
            X: Hidden states (n_samples, hidden_dim) or (hidden_dim,)
            layer: Layer index
            token_pos: Token position index

        Returns:
            Predicted values
        """
        if layer not in self.probes or token_pos not in self.probes[layer]:
            raise ValueError(f"No probe trained for layer {layer}, position {token_pos}")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self.probes[layer][token_pos].predict(X)

    def save(self, filename: str = "multi_position_probes.pkl"):
        """Save all trained probes and MSE matrix"""
        filepath = MODELS_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump({
                'probes': self.probes,
                'mse_matrix': self.mse_matrix,
                'num_layers': self.num_layers,
                'num_tokens': self.num_tokens,
                'alpha': self.alpha
            }, f)
        print(f"Probes saved to {filepath}")

    def load(self, filename: str = "multi_position_probes.pkl"):
        """Load trained probes"""
        filepath = MODELS_DIR / filename
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.probes = data['probes']
        self.mse_matrix = data['mse_matrix']
        self.num_layers = data['num_layers']
        self.num_tokens = data['num_tokens']
        self.alpha = data['alpha']
        print(f"Probes loaded from {filepath}")


def extract_all_positions_data(
    model_wrapper: ModelWrapper,
    dataset: List[Dict],
    num_layers: int
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Extract hidden states from all token positions and all layers.

    Args:
        model_wrapper: Loaded model wrapper
        dataset: Dataset with prompts and numbers
        num_layers: Number of layers to extract

    Returns:
        X_all: Dictionary {layer: (n_samples, n_tokens, hidden_dim)}
        y: Target numbers (n_samples,)
    """
    print(f"\n{'='*80}")
    print("EXTRACTING HIDDEN STATES")
    print(f"{'='*80}")

    # First pass: determine number of tokens (use first sample)
    sample_prompt = dataset[0]['prompt']
    inputs = model_wrapper.tokenizer(sample_prompt, return_tensors="pt")
    n_tokens = inputs['input_ids'].shape[1]
    print(f"Number of tokens per prompt: {n_tokens}")

    # Initialize storage
    n_samples = len(dataset)
    # We'll determine hidden_dim from first extraction
    X_all = {}  # Will be {layer: list of arrays}
    y_list = []

    print(f"Processing {n_samples} samples...")

    for i, sample in enumerate(tqdm(dataset, desc="Samples")):
        prompt = sample['prompt']

        # Tokenize
        inputs = model_wrapper.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(model_wrapper.device)

        # Extract hidden states from all layers
        with torch.no_grad():
            outputs = model_wrapper.model(
                input_ids=input_ids,
                output_hidden_states=True
            )

            # outputs.hidden_states: tuple of (num_layers+1) tensors
            # Each tensor: (batch_size, seq_len, hidden_dim)
            for layer in range(num_layers):
                hidden_state = outputs.hidden_states[layer]  # (1, n_tokens, hidden_dim)
                hidden_state_np = hidden_state[0].cpu().numpy()  # (n_tokens, hidden_dim)

                if layer not in X_all:
                    X_all[layer] = []
                X_all[layer].append(hidden_state_np)

        y_list.append(sample['number'])

    # Convert lists to arrays
    print("\nConverting to arrays...")
    for layer in range(num_layers):
        X_all[layer] = np.array(X_all[layer])  # (n_samples, n_tokens, hidden_dim)
        print(f"Layer {layer}: {X_all[layer].shape}")

    y = np.array(y_list)
    print(f"Targets: {y.shape}")
    print(f"Target range: [{y.min()}, {y.max()}]")

    return X_all, y


def visualize_mse_heatmap(
    mse_matrix: np.ndarray,
    save_path: Path = None,
    title: str = "Probe MSE Heatmap"
):
    """
    Visualize MSE as a heatmap with layers on x-axis and token positions on y-axis.

    Args:
        mse_matrix: (n_layers, n_tokens) array of MSE values
        save_path: Path to save figure
        title: Plot title
    """
    n_layers, n_tokens = mse_matrix.shape

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create heatmap
    im = ax.imshow(
        mse_matrix.T,  # Transpose so tokens are on y-axis
        aspect='auto',
        cmap='viridis_r',  # Reverse so darker = lower MSE (better)
        interpolation='nearest'
    )

    # Set labels
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Token Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set ticks
    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels(np.arange(n_layers))
    ax.set_yticks(np.arange(n_tokens))
    ax.set_yticklabels(np.arange(n_tokens))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MSE (lower is better)', rotation=270, labelpad=20, fontsize=11)

    # Add grid
    ax.set_xticks(np.arange(n_layers) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_tokens) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")

    plt.show()


def visualize_mse_statistics(
    mse_matrix: np.ndarray,
    save_path: Path = None
):
    """
    Create additional visualizations: MSE by layer and MSE by token position.
    """
    n_layers, n_tokens = mse_matrix.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MSE by layer (averaged over token positions)
    mse_by_layer = mse_matrix.mean(axis=1)
    axes[0].plot(range(n_layers), mse_by_layer, marker='o', linewidth=2)
    axes[0].set_xlabel('Layer', fontsize=11)
    axes[0].set_ylabel('Average MSE', fontsize=11)
    axes[0].set_title('MSE by Layer (averaged over tokens)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # MSE by token position (averaged over layers)
    mse_by_token = mse_matrix.mean(axis=0)
    axes[1].plot(range(n_tokens), mse_by_token, marker='o', linewidth=2, color='orange')
    axes[1].set_xlabel('Token Position', fontsize=11)
    axes[1].set_ylabel('Average MSE', fontsize=11)
    axes[1].set_title('MSE by Token Position (averaged over layers)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved to {save_path}")

    plt.show()


def save_results_json(
    results: Dict,
    mse_matrix: np.ndarray,
    filename: str = "multi_probe_results.json"
):
    """Save training results to JSON file"""
    filepath = MODELS_DIR / filename

    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {
        'mse_train_matrix': results['mse_train_matrix'].tolist(),
        'mse_test_matrix': results['mse_test_matrix'].tolist(),
        'avg_test_mse': float(mse_matrix.mean()),
        'min_test_mse': float(mse_matrix.min()),
        'max_test_mse': float(mse_matrix.max()),
        'num_layers': mse_matrix.shape[0],
        'num_tokens': mse_matrix.shape[1]
    }

    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    import torch

    print("="*80)
    print("MULTI-LAYER MULTI-POSITION PROBE TRAINING")
    print("="*80)

    # Configuration
    NUM_LAYERS = 28  # Adjust based on your model
    ALPHA = 1.0  # Ridge regularization

    # Load dataset
    dataset = load_dataset()

    # Load model
    model_wrapper = ModelWrapper()
    model_wrapper.load_model()

    # Extract training data from all positions and layers
    X_all, y = extract_all_positions_data(model_wrapper, dataset, NUM_LAYERS)

    # Train probes
    probe_system = MultiPositionProbe(num_layers=NUM_LAYERS, alpha=ALPHA)
    results = probe_system.train(X_all, y, test_size=0.2)

    # Save probes
    probe_system.save()

    # Save results to JSON
    save_results_json(results, probe_system.mse_matrix)

    # Visualize MSE heatmap
    heatmap_path = FIGURES_DIR / "probe_mse_heatmap.png"
    visualize_mse_heatmap(
        probe_system.mse_matrix,
        save_path=heatmap_path,
        title="Probe MSE: Layer vs Token Position"
    )

    # Visualize statistics
    stats_path = FIGURES_DIR / "probe_mse_statistics.png"
    visualize_mse_statistics(probe_system.mse_matrix, save_path=stats_path)

    # Cleanup
    model_wrapper.cleanup()

    print("\n" + "="*80)
    print("MULTI-POSITION PROBE TRAINING COMPLETED")
    print("="*80)
