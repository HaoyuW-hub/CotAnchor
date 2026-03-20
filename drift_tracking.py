"""
Drift Tracking Module
Track representation drift during long chain-of-thought generation
"""

import numpy as np
import torch
from typing import Dict, List
from scipy.spatial.distance import cosine
import json
from pathlib import Path

from config import (
    SAMPLE_INTERVAL, TARGET_LAYER, RESULTS_DIR,
    MAX_LENGTH, TEMPERATURE, TOP_P
)
from model_utils import ModelWrapper, find_anchor_tokens
from probe_training import PrimeProbe
from config import ANCHOR_TOKENS


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    # Handle the case where vectors might be 2D with shape (1, dim)
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()

    # Cosine similarity = 1 - cosine distance
    return 1 - cosine(vec1, vec2)


def track_drift_single_sample(
    model_wrapper: ModelWrapper,
    probe: PrimeProbe,
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
        probe: Trained probe for detecting prime concept
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

    # Get initial hidden state
    h_0 = generation_result['initial_hidden_state']

    # Calculate drift metrics at each sample point
    drift_metrics = []
    probe_scores = []
    cosine_similarities = []

    for i, (hidden_state, position) in enumerate(zip(
        generation_result['hidden_states'],
        generation_result['sample_positions']
    )):
        # Calculate cosine similarity with initial state
        cos_sim = calculate_cosine_similarity(hidden_state, h_0)
        cosine_similarities.append(cos_sim)

        # Calculate probe score (confidence that n is prime)
        probe_score = probe.predict_proba(hidden_state)[0]
        probe_scores.append(probe_score)

        drift_metrics.append({
            'position': position,
            'step': i,
            'cosine_similarity': float(cos_sim),
            'probe_score': float(probe_score)
        })

    # Find anchor tokens
    anchors = find_anchor_tokens(
        generation_result['tokens'],
        ANCHOR_TOKENS
    )

    # Calculate drift statistics
    if len(cosine_similarities) > 0:
        initial_score = probe_scores[0] if len(probe_scores) > 0 else None
        final_score = probe_scores[-1] if len(probe_scores) > 0 else None
        score_change = final_score - initial_score if initial_score and final_score else None

        initial_cos = cosine_similarities[0]
        final_cos = cosine_similarities[-1]
        cos_change = final_cos - initial_cos
    else:
        initial_score = final_score = score_change = None
        initial_cos = final_cos = cos_change = None

    result = {
        'sample_id': sample_id,
        'prompt': prompt,
        'generated_text': generation_result['generated_text'],
        'input_length': generation_result['input_length'],
        'total_length': generation_result['total_length'],
        'drift_metrics': drift_metrics,
        'anchors': [{'position': pos, 'token': tok} for pos, tok in anchors],
        'statistics': {
            'initial_probe_score': float(initial_score) if initial_score else None,
            'final_probe_score': float(final_score) if final_score else None,
            'probe_score_change': float(score_change) if score_change else None,
            'initial_cosine': float(initial_cos) if initial_cos else None,
            'final_cosine': float(final_cos) if final_cos else None,
            'cosine_change': float(cos_change) if cos_change else None,
            'num_anchors': len(anchors)
        }
    }

    return result


def track_drift_batch(
    dataset: List[Dict],
    model_wrapper: ModelWrapper,
    probe: PrimeProbe,
    max_samples: int = None,
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
            print(f"  Probe score change: {stats['probe_score_change']:.4f}")
            print(f"  Cosine change: {stats['cosine_change']:.4f}")
            print(f"  Anchors found: {stats['num_anchors']}")

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


def analyze_drift_patterns(results: List[Dict]) -> Dict:
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
    probe_changes = []
    cosine_changes = []
    num_anchors_list = []

    prime_probe_changes = []
    composite_probe_changes = []

    for result in results:
        stats = result['statistics']

        if stats['probe_score_change'] is not None:
            probe_changes.append(stats['probe_score_change'])

            if result['is_prime']:
                prime_probe_changes.append(stats['probe_score_change'])
            else:
                composite_probe_changes.append(stats['probe_score_change'])

        if stats['cosine_change'] is not None:
            cosine_changes.append(stats['cosine_change'])

        num_anchors_list.append(stats['num_anchors'])

    analysis = {
        'total_samples': len(results),
        'probe_score_change': {
            'mean': float(np.mean(probe_changes)) if probe_changes else None,
            'std': float(np.std(probe_changes)) if probe_changes else None,
            'min': float(np.min(probe_changes)) if probe_changes else None,
            'max': float(np.max(probe_changes)) if probe_changes else None
        },
        'cosine_similarity_change': {
            'mean': float(np.mean(cosine_changes)) if cosine_changes else None,
            'std': float(np.std(cosine_changes)) if cosine_changes else None,
            'min': float(np.min(cosine_changes)) if cosine_changes else None,
            'max': float(np.max(cosine_changes)) if cosine_changes else None
        },
        'anchor_count': {
            'mean': float(np.mean(num_anchors_list)),
            'std': float(np.std(num_anchors_list)),
            'total': int(np.sum(num_anchors_list))
        },
        'by_label': {
            'prime_probe_change_mean': float(np.mean(prime_probe_changes)) if prime_probe_changes else None,
            'composite_probe_change_mean': float(np.mean(composite_probe_changes)) if composite_probe_changes else None
        }
    }

    # Print summary
    print(f"\nTotal samples analyzed: {analysis['total_samples']}")
    print(f"\nProbe score change:")
    print(f"  Mean: {analysis['probe_score_change']['mean']:.4f}")
    print(f"  Std: {analysis['probe_score_change']['std']:.4f}")
    print(f"  Range: [{analysis['probe_score_change']['min']:.4f}, {analysis['probe_score_change']['max']:.4f}]")

    print(f"\nCosine similarity change:")
    print(f"  Mean: {analysis['cosine_similarity_change']['mean']:.4f}")
    print(f"  Std: {analysis['cosine_similarity_change']['std']:.4f}")

    print(f"\nAnchor tokens:")
    print(f"  Mean per sample: {analysis['anchor_count']['mean']:.2f}")
    print(f"  Total found: {analysis['anchor_count']['total']}")

    print(f"\nBy label:")
    print(f"  Prime numbers - mean probe change: {analysis['by_label']['prime_probe_change_mean']:.4f}")
    print(f"  Composite numbers - mean probe change: {analysis['by_label']['composite_probe_change_mean']:.4f}")

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
    probe = PrimeProbe()
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

    # Cleanup
    model_wrapper.cleanup()

    print("\n" + "="*80)
    print("DRIFT TRACKING TEST COMPLETED")
    print("="*80)
