"""
Probe Training Module
Train linear classifier to detect "is prime" concept in hidden states
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import MODELS_DIR, FIGURES_DIR, TARGET_LAYER
from model_utils import ModelWrapper
from data_preparation import load_dataset


class PrimeProbe:
    """Linear probe for detecting 'is prime' concept"""

    def __init__(self, layer: int = TARGET_LAYER):
        self.layer = layer
        self.probe = LogisticRegression(max_iter=1000, random_state=42)
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
            y: Labels (n_samples,) - 1 for prime, 0 for composite
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary with training metrics
        """
        print(f"\nTraining probe on layer {self.layer}...")
        print(f"Training samples: {X.shape[0]}")
        print(f"Hidden dimension: {X.shape[1]}")

        # Split data if we have enough samples
        if test_size > 0 and X.shape[0] > 4:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
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

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        print(f"\nTraining accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

        if test_size > 0:
            print("\nClassification Report (Test Set):")
            print(classification_report(y_test, y_pred_test,
                                       target_names=['Composite', 'Prime']))

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of being prime

        Args:
            X: Hidden states (n_samples, hidden_dim) or (hidden_dim,)

        Returns:
            Probability of class 1 (prime)
        """
        if not self.is_trained:
            raise ValueError("Probe must be trained before prediction")

        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Return probability of class 1 (prime)
        return self.probe.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_trained:
            raise ValueError("Probe must be trained before prediction")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self.probe.predict(X)

    def save(self, filename: str = "prime_probe.pkl"):
        """Save trained probe"""
        filepath = MODELS_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump({
                'probe': self.probe,
                'layer': self.layer,
                'is_trained': self.is_trained
            }, f)
        print(f"Probe saved to {filepath}")

    def load(self, filename: str = "prime_probe.pkl"):
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
    Extract hidden states from initial prompts for training

    Args:
        model_wrapper: Loaded model wrapper
        dataset: Dataset with prompts and labels
        layer: Which layer to extract from

    Returns:
        X: Hidden states (n_samples, hidden_dim)
        y: Labels (n_samples,)
    """
    print(f"\nExtracting training data from layer {layer}...")

    X_list = []
    y_list = []

    for i, sample in enumerate(dataset):
        print(f"Processing sample {i+1}/{len(dataset)}: {sample['id']}")

        # Get initial representation (after prompt)
        hidden_state = model_wrapper.get_initial_representation(
            sample['prompt'],
            layer=layer
        )

        X_list.append(hidden_state.squeeze())
        y_list.append(sample['label'])

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\nExtracted {X.shape[0]} samples with dimension {X.shape[1]}")
    print(f"Label distribution: {np.bincount(y)}")

    return X, y


def visualize_probe_results(results: Dict, save_path: Path = None):
    """Visualize probe training results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Confusion matrix for test set
    cm = confusion_matrix(results['y_test'], results['y_pred_test'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Composite', 'Prime'],
                yticklabels=['Composite', 'Prime'])
    axes[0].set_title('Confusion Matrix (Test Set)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Accuracy comparison
    accuracies = [results['train_accuracy'], results['test_accuracy']]
    axes[1].bar(['Train', 'Test'], accuracies, color=['skyblue', 'lightcoral'])
    axes[1].set_ylim([0, 1.1])
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Probe Performance')
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    axes[1].legend()

    # Add accuracy values on bars
    for i, v in enumerate(accuracies):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probe results visualization saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("="*80)
    print("PROBE TRAINING")
    print("="*80)

    # Load dataset
    dataset = load_dataset()

    # Load model
    model_wrapper = ModelWrapper()
    model_wrapper.load_model()

    # Extract training data
    X, y = extract_training_data(model_wrapper, dataset, layer=TARGET_LAYER)

    # Train probe
    probe = PrimeProbe(layer=TARGET_LAYER)
    results = probe.train(X, y, test_size=0.2)

    # Save probe
    probe.save()

    # Visualize results
    save_path = FIGURES_DIR / "probe_training_results.png"
    visualize_probe_results(results, save_path)

    # Cleanup
    model_wrapper.cleanup()

    print("\n" + "="*80)
    print("PROBE TRAINING COMPLETED")
    print("="*80)
