"""
Drift Tracking Module
Track representation drift during long chain-of-thought generation
"""

import numpy as np
import torch
from typing import Dict, List
from scipy.spatial.distance import cosine
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from config import (
    SAMPLE_INTERVAL, TARGET_LAYER, RESULTS_DIR, FIGURES_DIR,
    MAX_LENGTH, TEMPERATURE, TOP_P
)
from model_utils import ModelWrapper, find_anchor_tokens
from probe_training import NumberProbe
from config import ANCHOR_TOKENS


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    # Handle the case where vectors might be 2D with shape (1, dim)
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()

    # Convert to float64 for higher precision to avoid overflow
    vec1 = vec1.astype(np.float64)
    vec2 = vec2.astype(np.float64)

    # Normalize vectors to prevent overflow in dot products
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # If norms are zero, return 0 (vectors are undefined)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Normalize and compute cosine similarity directly
    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2

    return np.dot(vec1_norm, vec2_norm)


def track_drift_single_sample(
    model_wrapper: ModelWrapper,
    probe: NumberProbe,
    prompt: str,
    sample_id: str,
    max_length: int = MAX_LENGTH,
    sample_interval: int = SAMPLE_INTERVAL,
    layer: int = TARGET_LAYER
) -> Dict:
    """
    Track representation drift for a single sample

    Args:
        model_wrapper: Loaded model wrapper
        probe: Trained probe for extracting the number concept
        prompt: Input prompt
        sample_id: Identifier for this sample
        max_length: Maximum tokens to generate
        sample_interval: Extract hidden state every N tokens
        layer: Which layer to track

    Returns:
        Dictionary containing drift metrics and analysis
    """
    print(f"\nTracking drift for sample: {sample_id}")
    print(f"Prompt: {prompt[:100]}...")

    # Generate with hidden state collection
    generation_result = model_wrapper.generate_with_hidden_states(
        prompt=prompt,
        max_length=max_length,
        sample_interval=sample_interval,
        layer=layer,
        temperature=TEMPERATURE,
        top_p=TOP_P
    )

    # Get initial hidden state (last token of prompt, position=-1)
    h_0 = generation_result['initial_hidden_state']

    # Calculate drift metrics at each sample point
    drift_metrics = []
    predicted_numbers = []
    probe_confidences = []
    cosine_similarities = []

    for i, (hidden_state, position) in enumerate(zip(
        generation_result['hidden_states'],
        generation_result['sample_positions']
    )):
        # Calculate cosine similarity with initial state
        cos_sim = calculate_cosine_similarity(hidden_state, h_0)
        cosine_similarities.append(cos_sim)

        # Predicted number and confidence from probe
        predicted, confidence = probe.predict_with_confidence(hidden_state)
        predicted_number = predicted[0]
        probe_confidence = confidence[0]
        predicted_numbers.append(predicted_number)
        probe_confidences.append(probe_confidence)

        drift_metrics.append({
            'position': position,
            'step': i,
            'cosine_similarity': float(cos_sim),
            'predicted_number': float(predicted_number),
            'probe_confidence': float(probe_confidence)
        })

    # Find anchor tokens
    anchors = find_anchor_tokens(
        generation_result['tokens'],
        ANCHOR_TOKENS
    )

    # Calculate drift statistics
    if len(cosine_similarities) > 0:
        initial_pred    = predicted_numbers[0]  if predicted_numbers  else None
        final_pred      = predicted_numbers[-1] if predicted_numbers  else None
        pred_change     = final_pred - initial_pred if (initial_pred is not None and final_pred is not None) else None

        initial_conf    = probe_confidences[0]  if probe_confidences  else None
        final_conf      = probe_confidences[-1] if probe_confidences  else None
        conf_change     = final_conf - initial_conf if (initial_conf is not None and final_conf is not None) else None

        initial_cos = cosine_similarities[0]
        final_cos   = cosine_similarities[-1]
        cos_change  = final_cos - initial_cos
    else:
        initial_pred = final_pred = pred_change = None
        initial_conf = final_conf = conf_change = None
        initial_cos  = final_cos  = cos_change  = None

    result = {
        'sample_id': sample_id,
        'prompt': prompt,
        'generated_text': generation_result['generated_text'],
        'input_length': generation_result['input_length'],
        'total_length': generation_result['total_length'],
        'drift_metrics': drift_metrics,
        'anchors': [{'position': pos, 'token': tok} for pos, tok in anchors],
        'statistics': {
            'initial_predicted_number': float(initial_pred) if initial_pred is not None else None,
            'final_predicted_number':   float(final_pred)   if final_pred   is not None else None,
            'predicted_number_change':  float(pred_change)  if pred_change  is not None else None,
            'initial_confidence':       float(initial_conf) if initial_conf is not None else None,
            'final_confidence':         float(final_conf)   if final_conf   is not None else None,
            'confidence_change':        float(conf_change)  if conf_change  is not None else None,
            'initial_cosine':           float(initial_cos)  if initial_cos  is not None else None,
            'final_cosine':             float(final_cos)    if final_cos    is not None else None,
            'cosine_change':            float(cos_change)   if cos_change   is not None else None,
            'num_anchors': len(anchors)
        }
    }

    return result


def track_drift_batch(
    dataset: List[Dict],
    model_wrapper: ModelWrapper,
    probe: NumberProbe,
    max_samples: int = 10,
    save_results: bool = True
) -> List[Dict]:
    """
    Track drift for multiple samples

    Args:
        dataset: List of samples with prompts
        model_wrapper: Loaded model wrapper
        probe: Trained probe
        max_samples: Maximum number of samples to process (None for all)
        save_results: Whether to save results to file

    Returns:
        List of drift tracking results
    """
    print("\n" + "="*80)
    print("BATCH DRIFT TRACKING")
    print("="*80)

    if max_samples:
        dataset = dataset[:max_samples]

    results = []

    for i, sample in enumerate(dataset):
        print(f"\nProcessing {i+1}/{len(dataset)}")

        try:
            result = track_drift_single_sample(
                model_wrapper=model_wrapper,
                probe=probe,
                prompt=sample['prompt'],
                sample_id=sample['id']
            )

            # Add ground truth label
            result['ground_truth_label'] = sample['label']
            result['is_prime'] = sample['is_prime']
            result['number'] = sample['number']

            results.append(result)

            # Print summary
            stats = result['statistics']
            print(f"\nSummary for {sample['id']}:")
            print(f"  Number: {sample['number']} ({'prime' if sample['is_prime'] else 'composite'})")
            print(f"  Generated length: {result['total_length']} tokens")
            print(f"  Predicted number change: {stats['predicted_number_change']:.2f}")
            print(f"  Confidence change:        {stats['confidence_change']:.4f}")
            print(f"  Cosine change:            {stats['cosine_change']:.4f}")
            print(f"  Anchors found:            {stats['num_anchors']}")

        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")
            continue

    # Save results
    if save_results:
        save_path = RESULTS_DIR / "drift_tracking_results.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {save_path}")

    return results


def visualize_drift(result: Dict, save_dir: Path = None):
    """
    Visualize probe confidence and cosine similarity over generation tokens
    for a single sample, with anchor token positions marked.

    Args:
        result:   Output dict from track_drift_single_sample
        save_dir: Directory to save the figure; None = show only
    """
    metrics      = result['drift_metrics']
    anchors      = result['anchors']
    sample_id    = result['sample_id']
    true_number  = result.get('number', '?')
    input_length = result['input_length']

    positions    = [m['position'] for m in metrics]
    confidences  = [m['probe_confidence']  for m in metrics]
    cos_sims     = [m['cosine_similarity'] for m in metrics]
    pred_numbers = [m['predicted_number']  for m in metrics]

    # Relative positions (tokens generated so far)
    rel_pos = [p - input_length for p in positions]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Representation Drift — Sample {sample_id}  (n = {true_number})",
        fontsize=13, fontweight='bold'
    )

    anchor_positions = [a['position'] - input_length for a in anchors]
    anchor_tokens    = [a['token'] for a in anchors]

    def mark_anchors(ax):
        for apos, atok in zip(anchor_positions, anchor_tokens):
            ax.axvline(x=apos, color='orange', linestyle='--', alpha=0.6, lw=1)
            ax.text(apos, ax.get_ylim()[1], atok,
                    fontsize=7, color='orange', rotation=45,
                    ha='left', va='bottom')

    # ── Panel 1: Probe confidence ──────────────────────────────────────────
    axes[0].plot(rel_pos, confidences, color='steelblue', lw=1.8, marker='o', ms=3)
    axes[0].set_ylabel('Probe Confidence', fontsize=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=confidences[0], color='steelblue', linestyle=':', lw=1,
                    label=f'initial = {confidences[0]:.3f}')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    mark_anchors(axes[0])

    # ── Panel 2: Cosine similarity with h_0 ───────────────────────────────
    axes[1].plot(rel_pos, cos_sims, color='tomato', lw=1.8, marker='o', ms=3)
    axes[1].set_ylabel('Cosine Similarity (vs h₀)', fontsize=10)
    axes[1].set_ylim(max(0, min(cos_sims) - 0.05), 1.05)
    axes[1].axhline(y=1.0, color='grey', linestyle=':', lw=1)
    axes[1].grid(True, alpha=0.3)
    mark_anchors(axes[1])

    # ── Panel 3: Predicted number ──────────────────────────────────────────
    axes[2].plot(rel_pos, pred_numbers, color='seagreen', lw=1.8, marker='o', ms=3)
    axes[2].axhline(y=true_number, color='seagreen', linestyle=':', lw=1,
                    label=f'true n = {true_number}')
    axes[2].set_ylabel('Predicted n', fontsize=10)
    axes[2].set_xlabel('Generated tokens', fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    mark_anchors(axes[2])

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        path = save_dir / f"drift_{sample_id}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved drift plot → {path}")

    plt.show()
    plt.close()



    """
    Analyze overall drift patterns across all samples

    Args:
        results: List of drift tracking results

    Returns:
        Summary statistics
    """
    print("\n" + "="*80)
    print("DRIFT PATTERN ANALYSIS")
    print("="*80)

    # Aggregate statistics
    pred_changes = []
    conf_changes = []
    cosine_changes = []
    num_anchors_list = []

    prime_pred_changes = []
    composite_pred_changes = []

    for result in results:
        stats = result['statistics']

        if stats['predicted_number_change'] is not None:
            pred_changes.append(stats['predicted_number_change'])
            if result['is_prime']:
                prime_pred_changes.append(stats['predicted_number_change'])
            else:
                composite_pred_changes.append(stats['predicted_number_change'])

        if stats['confidence_change'] is not None:
            conf_changes.append(stats['confidence_change'])

        if stats['cosine_change'] is not None:
            cosine_changes.append(stats['cosine_change'])

        num_anchors_list.append(stats['num_anchors'])

    analysis = {
        'total_samples': len(results),
        'predicted_number_change': {
            'mean': float(np.mean(pred_changes)) if pred_changes else None,
            'std':  float(np.std(pred_changes))  if pred_changes else None,
            'min':  float(np.min(pred_changes))  if pred_changes else None,
            'max':  float(np.max(pred_changes))  if pred_changes else None
        },
        'confidence_change': {
            'mean': float(np.mean(conf_changes)) if conf_changes else None,
            'std':  float(np.std(conf_changes))  if conf_changes else None,
            'min':  float(np.min(conf_changes))  if conf_changes else None,
            'max':  float(np.max(conf_changes))  if conf_changes else None
        },
        'cosine_similarity_change': {
            'mean': float(np.mean(cosine_changes)) if cosine_changes else None,
            'std':  float(np.std(cosine_changes))  if cosine_changes else None,
            'min':  float(np.min(cosine_changes))  if cosine_changes else None,
            'max':  float(np.max(cosine_changes))  if cosine_changes else None
        },
        'anchor_count': {
            'mean':  float(np.mean(num_anchors_list)),
            'std':   float(np.std(num_anchors_list)),
            'total': int(np.sum(num_anchors_list))
        },
        'by_label': {
            'prime_pred_change_mean':     float(np.mean(prime_pred_changes))     if prime_pred_changes     else None,
            'composite_pred_change_mean': float(np.mean(composite_pred_changes)) if composite_pred_changes else None
        }
    }

    # Print summary
    print(f"\nTotal samples analyzed: {analysis['total_samples']}")
    print(f"\nPredicted number change:")
    print(f"  Mean: {analysis['predicted_number_change']['mean']:.4f}")
    print(f"  Std:  {analysis['predicted_number_change']['std']:.4f}")
    print(f"  Range: [{analysis['predicted_number_change']['min']:.4f}, {analysis['predicted_number_change']['max']:.4f}]")

    print(f"\nProbe confidence change:")
    print(f"  Mean: {analysis['confidence_change']['mean']:.4f}")
    print(f"  Std:  {analysis['confidence_change']['std']:.4f}")

    print(f"\nCosine similarity change:")
    print(f"  Mean: {analysis['cosine_similarity_change']['mean']:.4f}")
    print(f"  Std:  {analysis['cosine_similarity_change']['std']:.4f}")

    print(f"\nAnchor tokens:")
    print(f"  Mean per sample: {analysis['anchor_count']['mean']:.2f}")
    print(f"  Total found:     {analysis['anchor_count']['total']}")

    print(f"\nBy label:")
    print(f"  Prime     - mean pred change: {analysis['by_label']['prime_pred_change_mean']:.4f}")
    print(f"  Composite - mean pred change: {analysis['by_label']['composite_pred_change_mean']:.4f}")

    return analysis


if __name__ == "__main__":
    from data_preparation import load_dataset

    print("="*80)
    print("DRIFT TRACKING TEST")
    print("="*80)

    # Load dataset
    dataset = load_dataset()

    # Load model
    model_wrapper = ModelWrapper()
    model_wrapper.load_model()

    # Load probe
    probe = NumberProbe()
    probe.load()

    # Track drift for first 3 samples
    results = track_drift_batch(
        dataset=dataset,
        model_wrapper=model_wrapper,
        probe=probe,
        max_samples=3,
        save_results=True
    )

    # Analyze patterns
    analysis = analyze_drift_patterns(results)

    # Visualize drift for each sample
    fig_dir = FIGURES_DIR / "drift"
    for result in results:
        visualize_drift(result, save_dir=fig_dir)

    # Cleanup
    model_wrapper.cleanup()

    print("\n" + "="*80)
    print("DRIFT TRACKING TEST COMPLETED")
    print("="*80)
