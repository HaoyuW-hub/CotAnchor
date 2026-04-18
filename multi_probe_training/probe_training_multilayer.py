"""
Multi-layer Multi-position Probe Training Module
Train probes for each token position and each layer to predict the number n
"""

import numpy as np
import pickle
import torch
import json
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from config import MODELS_DIR
from model_utils import ModelWrapper
from data_preparation import load_dataset


class MultiPositionProbe:
    """
    Trains separate Ridge probes for each (layer, token_position) combination
    to predict the number n from hidden states.
    """

    def __init__(self, num_layers: int, alpha: float = 100.0):
        self.num_layers = num_layers
        self.alpha = alpha
        # probes[layer][token_pos] = trained Ridge model
        self.probes = {}
        self.mse_matrix = None   # (n_layers, n_tokens) test MSE in log space
        self.acc_matrix = None   # (n_layers, n_tokens) test accuracy at 1% threshold
        self.num_tokens = None

    def train(
        self,
        X_all: Dict[int, np.ndarray],  # {layer: (n_samples, n_tokens, hidden_dim)}
        y: np.ndarray,                 # (n_samples,) target numbers
        test_size: float = 0.2
    ) -> Dict:
        """
        Train probes for all (layer, token_position) combinations.

        Returns dict with mse/acc matrices for train and test splits.
        """
        print(f"\n{'='*80}")
        print("MULTI-POSITION PROBE TRAINING")
        print(f"{'='*80}")

        first_layer = list(X_all.keys())[0]
        n_samples, n_tokens, hidden_dim = X_all[first_layer].shape
        self.num_tokens = n_tokens

        print(f"Samples: {n_samples} | Tokens: {n_tokens} | Layers: {self.num_layers} | Hidden dim: {hidden_dim}")
        print(f"Target range: [{y.min()}, {y.max()}]")

        if test_size > 0 and n_samples > 4:
            train_idx, test_idx = train_test_split(
                np.arange(n_samples), test_size=test_size, random_state=42
            )
        else:
            train_idx = test_idx = np.arange(n_samples)

        y_train, y_test = y[train_idx], y[test_idx]
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)

        mse_train_matrix = np.zeros((self.num_layers, n_tokens))
        mse_test_matrix = np.zeros((self.num_layers, n_tokens))
        acc_train_matrix = np.zeros((self.num_layers, n_tokens))
        acc_test_matrix = np.zeros((self.num_layers, n_tokens))

        print(f"\nTraining {self.num_layers * n_tokens} probes...")

        for layer in tqdm(range(self.num_layers), desc="Layers"):
            if layer not in X_all:
                print(f"Warning: Layer {layer} not found in data, skipping...")
                continue

            self.probes[layer] = {}
            X_layer = X_all[layer]  # (n_samples, n_tokens, hidden_dim)

            for token_pos in range(n_tokens):
                X_pos = X_layer[:, token_pos, :]  # (n_samples, hidden_dim)
                X_train = X_pos[train_idx]
                X_test = X_pos[test_idx]

                probe = Ridge(alpha=self.alpha)
                probe.fit(X_train, y_train_log)

                y_pred_train_log = probe.predict(X_train)
                y_pred_test_log = probe.predict(X_test)

                mse_train_matrix[layer, token_pos] = mean_squared_error(y_train_log, y_pred_train_log)
                mse_test_matrix[layer, token_pos] = mean_squared_error(y_test_log, y_pred_test_log)

                # Accuracy: |pred_orig - true| <= 1% * true
                y_pred_train_orig = np.expm1(y_pred_train_log)
                y_pred_test_orig = np.expm1(y_pred_test_log)
                acc_train_matrix[layer, token_pos] = np.mean(
                    np.abs(y_pred_train_orig - y_train) <= 0.01 * np.abs(y_train)
                )
                acc_test_matrix[layer, token_pos] = np.mean(
                    np.abs(y_pred_test_orig - y_test) <= 0.01 * np.abs(y_test)
                )

                self.probes[layer][token_pos] = probe

        self.mse_matrix = mse_test_matrix
        self.acc_matrix = acc_test_matrix

        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Test MSE  — avg: {mse_test_matrix.mean():.4f}  min: {mse_test_matrix.min():.4f}  max: {mse_test_matrix.max():.4f}")
        print(f"Test Acc (1%) — avg: {acc_test_matrix.mean():.4f}  max: {acc_test_matrix.max():.4f}")

        return {
            'mse_train_matrix': mse_train_matrix,
            'mse_test_matrix': mse_test_matrix,
            'acc_train_matrix': acc_train_matrix,
            'acc_test_matrix': acc_test_matrix,
            'y_train': y_train,
            'y_test': y_test,
            'train_idx': train_idx,
            'test_idx': test_idx
        }

    def predict(self, X: np.ndarray, layer: int, token_pos: int) -> np.ndarray:
        """Return log-space predictions for a specific (layer, token_pos) probe."""
        if layer not in self.probes or token_pos not in self.probes[layer]:
            raise ValueError(f"No probe trained for layer {layer}, position {token_pos}")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.probes[layer][token_pos].predict(X)

    def predict_original(self, X: np.ndarray, layer: int, token_pos: int) -> np.ndarray:
        """Predict and inverse-transform from log space to original scale."""
        return np.expm1(self.predict(X, layer, token_pos))

    def save(self, filename: str = "multi_position_probes.pkl"):
        filepath = MODELS_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump({
                'probes': self.probes,
                'mse_matrix': self.mse_matrix,
                'acc_matrix': self.acc_matrix,
                'num_layers': self.num_layers,
                'num_tokens': self.num_tokens,
                'alpha': self.alpha
            }, f)
        print(f"Probes saved to {filepath}")

    def load(self, filename: str = "multi_position_probes.pkl"):
        filepath = MODELS_DIR / filename
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.probes = data['probes']
        self.mse_matrix = data['mse_matrix']
        self.acc_matrix = data.get('acc_matrix', None)
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

    Returns:
        X_all: {layer: (n_samples, max_tokens, hidden_dim)} — padded with zeros
        y: (n_samples,) target numbers
    """
    print(f"\n{'='*80}")
    print("EXTRACTING HIDDEN STATES")
    print(f"{'='*80}")

    n_samples = len(dataset)
    X_all: Dict[int, List] = {}
    y_list = []

    print(f"Processing {n_samples} samples...")

    for sample in tqdm(dataset, desc="Samples"):
        inputs = model_wrapper.tokenizer(sample['prompt'], return_tensors="pt")
        input_ids = inputs['input_ids'].to(model_wrapper.device)

        with torch.no_grad():
            outputs = model_wrapper.model(input_ids=input_ids, output_hidden_states=True)
            for layer in range(num_layers):
                hidden_state_np = outputs.hidden_states[layer][0].cpu().numpy()  # (seq_len, hidden_dim)
                X_all.setdefault(layer, []).append(hidden_state_np)

        y_list.append(sample['number'])

    # Pad to max token length across all samples
    print("\nConverting to arrays...")
    max_tokens = max(arr.shape[0] for arr in X_all[0])
    hidden_dim = X_all[0][0].shape[1]
    print(f"Max token length: {max_tokens}, hidden dim: {hidden_dim}")

    for layer in range(num_layers):
        padded = np.zeros((n_samples, max_tokens, hidden_dim), dtype=np.float32)
        for i, arr in enumerate(X_all[layer]):
            padded[i, :arr.shape[0], :] = arr
        X_all[layer] = padded

    y = np.array(y_list)
    print(f"Targets: {y.shape}, range: [{y.min()}, {y.max()}]")

    return X_all, y


def save_results_json(
    results: Dict,
    mse_matrix: np.ndarray,
    acc_matrix: np.ndarray,
    filename: str = "multi_probe_results.json"
):
    """Save training results (matrices + summary stats) to JSON."""
    filepath = MODELS_DIR / filename
    data = {
        'mse_train_matrix': results['mse_train_matrix'].tolist(),
        'mse_test_matrix': results['mse_test_matrix'].tolist(),
        'acc_train_matrix': results['acc_train_matrix'].tolist(),
        'acc_test_matrix': results['acc_test_matrix'].tolist(),
        'avg_test_mse': float(mse_matrix.mean()),
        'min_test_mse': float(mse_matrix.min()),
        'max_test_mse': float(mse_matrix.max()),
        'avg_test_acc': float(acc_matrix.mean()),
        'max_test_acc': float(acc_matrix.max()),
        'num_layers': int(mse_matrix.shape[0]),
        'num_tokens': int(mse_matrix.shape[1])
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filepath}")
