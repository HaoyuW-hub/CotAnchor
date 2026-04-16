"""
Probe Training Module
Train linear regressor to extract the number from hidden states
"""

import numpy as np
import pickle
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import MODELS_DIR, FIGURES_DIR, TARGET_LAYER
from model_utils import ModelWrapper
from data_preparation import load_dataset


class NumberProbe:
    """
    Linear probe for extracting the number from hidden states.
    Uses BayesianRidge regression to produce both a predicted number value
    and a predictive standard deviation, from which a confidence score is derived.
    """

    def __init__(self, layer: int = TARGET_LAYER):
        self.layer = layer
        # BayesianRidge provides native uncertainty estimation via
        # predictive std, while also regularising the high-dim hidden states
        self.probe = BayesianRidge()
        self.is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train the probe

        Args:
            X: Hidden states (n_samples, hidden_dim)
            y: Target numbers (n_samples,) - the actual numbers to predict
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary with training metrics
        """
        print(f"\nTraining probe on layer {self.layer}...")
        print(f"Training samples: {X.shape[0]}")
        print(f"Hidden dimension: {X.shape[1]}")
        print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")

        # Split data if we have enough samples
        if test_size > 0 and X.shape[0] > 4:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            X_train, X_test = X, X
            y_train, y_test = y, y

        # Train probe
        self.probe.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred_train = self.probe.predict(X_train)
        y_pred_test = self.probe.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        print(f"\nTraining MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Training RMSE: {np.sqrt(train_mse):.4f}")
        print(f"Test RMSE: {np.sqrt(test_mse):.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")

        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the number value

        Args:
            X: Hidden states (n_samples, hidden_dim) or (hidden_dim,)

        Returns:
            Predicted number values
        """
        if not self.is_trained:
            raise ValueError("Probe must be trained before prediction")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self.probe.predict(X)

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict number value and confidence simultaneously.

        Confidence is derived from BayesianRidge's predictive std:
            confidence = 1 / (1 + std)
        Maps to [0, 1]: std → 0 gives confidence → 1 (certain);
        larger std gives confidence → 0 (uncertain).

        Args:
            X: Hidden states (n_samples, hidden_dim) or (hidden_dim,)

        Returns:
            predicted:   Predicted number values  (n_samples,)
            confidence:  Confidence scores in [0, 1] (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Probe must be trained before prediction")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predicted, std = self.probe.predict(X, return_std=True)
        confidence = 1.0 / (1.0 + std)
        return predicted, confidence

    def save(self, filename: str = "number_probe.pkl"):
        """Save trained probe"""
        filepath = MODELS_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump({
                'probe': self.probe,
                'layer': self.layer,
                'is_trained': self.is_trained
            }, f)
        print(f"Probe saved to {filepath}")

    def load(self, filename: str = "number_probe.pkl"):
        """Load trained probe"""
        filepath = MODELS_DIR / filename
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.probe = data['probe']
        self.layer = data['layer']
        self.is_trained = data['is_trained']
        print(f"Probe loaded from {filepath}")


def extract_training_data(
    model_wrapper: ModelWrapper,
    dataset: List[Dict],
    layer: int = TARGET_LAYER
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract hidden states from initial prompts for training.

    Uses the last token of the full prompt (position=-1) as the probe input,
    consistent with what drift tracking sees at each generation step.

    Rationale: during drift tracking the probe is applied to the last token's
    hidden state at every sampling point. Training on the same position
    (prompt-final last token) avoids train/inference distribution mismatch.
    Whether n is linearly decodable from this position is itself a key
    empirical question: high R² validates the approach; low R² suggests
    the number is not linearly accessible from the prompt's last token.

    Args:
        model_wrapper: Loaded model wrapper
        dataset: Dataset with prompts and numbers
        layer: Which layer to extract from

    Returns:
        X: Hidden states (n_samples, hidden_dim) — last token of each prompt
        y: Target numbers (n_samples,)
    """
    print(f"\nExtracting training data from layer {layer}...")
    print("Token position: last token of prompt (position=-1)")

    X_list = []
    y_list = []

    for i, sample in enumerate(dataset):
        print(f"Processing sample {i+1}/{len(dataset)}: {sample['id']}")

        # Use last token of the full prompt — same position drift tracking
        # will query during generation, ensuring distribution consistency
        hidden_state = model_wrapper.get_initial_representation(
            sample['prompt'],
            layer=layer
            # target_text not set → defaults to position=-1 (last token)
        )

        X_list.append(hidden_state.squeeze())
        y_list.append(sample['number'])

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\nExtracted {X.shape[0]} samples with dimension {X.shape[1]}")
    print(f"Number range: [{y.min()}, {y.max()}]")
    print(f"Number mean: {y.mean():.2f}, std: {y.std():.2f}")

    return X, y


def visualize_probe_results(results: Dict, save_path: Path = None):
    """Visualize probe training results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scatter plot: Predicted vs Actual (Test Set)
    axes[0, 0].scatter(results['y_test'], results['y_pred_test'], alpha=0.6, edgecolors='k')
    axes[0, 0].plot([results['y_test'].min(), results['y_test'].max()],
                     [results['y_test'].min(), results['y_test'].max()],
                     'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual Number')
    axes[0, 0].set_ylabel('Predicted Number')
    axes[0, 0].set_title('Predicted vs Actual (Test Set)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Residual plot (Test Set)
    residuals = results['y_test'] - results['y_pred_test']
    axes[0, 1].scatter(results['y_pred_test'], residuals, alpha=0.6, edgecolors='k')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Number')
    axes[0, 1].set_ylabel('Residual (Actual - Predicted)')
    axes[0, 1].set_title('Residual Plot (Test Set)')
    axes[0, 1].grid(True, alpha=0.3)

    # MAE comparison
    mae_values = [results['train_mae'], results['test_mae']]
    axes[1, 0].bar(['Train', 'Test'], mae_values, color=['skyblue', 'lightcoral'])
    axes[1, 0].set_ylabel('Mean Absolute Error')
    axes[1, 0].set_title('MAE Comparison')
    for i, v in enumerate(mae_values):
        axes[1, 0].text(i, v + max(mae_values)*0.02, f'{v:.2f}', ha='center', va='bottom')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # R² comparison
    r2_values = [results['train_r2'], results['test_r2']]
    axes[1, 1].bar(['Train', 'Test'], r2_values, color=['skyblue', 'lightcoral'])
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('R² Score Comparison')
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].axhline(y=0.9, color='g', linestyle='--', lw=1, label='0.9 threshold')
    for i, v in enumerate(r2_values):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probe results visualization saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("="*80)
    print("NUMBER PROBE TRAINING")
    print("="*80)

    # Load dataset
    dataset = load_dataset()

    # Load model
    model_wrapper = ModelWrapper()
    model_wrapper.load_model()

    # Extract training data
    X, y = extract_training_data(model_wrapper, dataset, layer=TARGET_LAYER)

    # Train probe
    probe = NumberProbe(layer=TARGET_LAYER)
    results = probe.train(X, y, test_size=0.2)

    # Save probe
    probe.save()

    # Visualize results
    save_path = FIGURES_DIR / "number_probe_training_results.png"
    visualize_probe_results(results, save_path)

    # Cleanup
    model_wrapper.cleanup()

    print("\n" + "="*80)
    print("NUMBER PROBE TRAINING COMPLETED")
    print("="*80)
