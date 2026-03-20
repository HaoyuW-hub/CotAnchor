"""
Anchor Analysis Module
Analyze the effect of anchor tokens on representation drift
"""

import numpy as np
from typing import List, Dict
import json
from pathlib import Path

from config import RESULTS_DIR, SAMPLE_INTERVAL


def calculate_anchor_effect(
    drift_metrics: List[Dict],
    anchor_position: int,
    window_size: int = 50
) -> Dict:
    """
    Calculate the effect of an anchor token on representation drift

    Args:
        drift_metrics: List of drift metric dictionaries
        anchor_position: Position of the anchor token
        window_size: Size of window before/after anchor to analyze

    Returns:
        Dictionary with anchor effect statistics
    """
    # Find metrics before and after anchor
    before_metrics = [m for m in drift_metrics
                     if m['position'] < anchor_position
                     and m['position'] >= anchor_position - window_size]

    after_metrics = [m for m in drift_metrics
                    if m['position'] > anchor_position
                    and m['position'] <= anchor_position + window_size]

    if not before_metrics or not after_metrics:
        return None

    # Get last metric before and first metric after
    last_before = max(before_metrics, key=lambda x: x['position'])
    first_after = min(after_metrics, key=lambda x: x['position'])

    # Calculate changes
    probe_score_change = first_after['probe_score'] - last_before['probe_score']
    cosine_change = first_after['cosine_similarity'] - last_before['cosine_similarity']

    return {
        'probe_score_before': last_before['probe_score'],
        'probe_score_after': first_after['probe_score'],
        'probe_score_change': probe_score_change,
        'cosine_before': last_before['cosine_similarity'],
        'cosine_after': first_after['cosine_similarity'],
        'cosine_change': cosine_change,
        'position_before': last_before['position'],
        'position_after': first_after['position']
    }


def analyze_anchors_single_sample(result: Dict, window_size: int = 50) -> Dict:
    """
    Analyze anchor effects for a single sample

    Args:
        result: Drift tracking result for one sample
        window_size: Window size for before/after comparison

    Returns:
        Analysis of anchor effects
    """
    drift_metrics = result['drift_metrics']
    anchors = result['anchors']

    anchor_effects = []

    for anchor in anchors:
        effect = calculate_anchor_effect(
            drift_metrics,
            anchor['position'],
            window_size
        )

        if effect:
            anchor_effects.append({
                'anchor_token': anchor['token'],
                'anchor_position': anchor['position'],
                **effect
            })

    # Calculate average effects
    if anchor_effects:
        avg_probe_change = np.mean([e['probe_score_change'] for e in anchor_effects])
        avg_cosine_change = np.mean([e['cosine_change'] for e in anchor_effects])
    else:
        avg_probe_change = None
        avg_cosine_change = None

    return {
        'sample_id': result['sample_id'],
        'num_anchors': len(anchors),
        'anchor_effects': anchor_effects,
        'average_probe_change': float(avg_probe_change) if avg_probe_change else None,
        'average_cosine_change': float(avg_cosine_change) if avg_cosine_change else None
    }


def analyze_anchors_batch(
    results: List[Dict],
    window_size: int = 50,
    save_results: bool = True
) -> Dict:
    """
    Analyze anchor effects across all samples

    Args:
        results: List of drift tracking results
        window_size: Window size for analysis
        save_results: Whether to save results to file

    Returns:
        Comprehensive anchor analysis
    """
    print("\n" + "="*80)
    print("ANCHOR EFFECT ANALYSIS")
    print("="*80)

    sample_analyses = []
    all_effects = []

    for result in results:
        analysis = analyze_anchors_single_sample(result, window_size)
        sample_analyses.append(analysis)

        if analysis['anchor_effects']:
            all_effects.extend(analysis['anchor_effects'])

    # Aggregate statistics
    if all_effects:
        probe_changes = [e['probe_score_change'] for e in all_effects]
        cosine_changes = [e['cosine_change'] for e in all_effects]

        # Count positive vs negative changes
        positive_probe_changes = sum(1 for c in probe_changes if c > 0)
        negative_probe_changes = sum(1 for c in probe_changes if c < 0)

        positive_cosine_changes = sum(1 for c in cosine_changes if c > 0)
        negative_cosine_changes = sum(1 for c in cosine_changes if c < 0)

        aggregate_stats = {
            'total_anchors': len(all_effects),
            'probe_score_change': {
                'mean': float(np.mean(probe_changes)),
                'std': float(np.std(probe_changes)),
                'min': float(np.min(probe_changes)),
                'max': float(np.max(probe_changes)),
                'positive_count': positive_probe_changes,
                'negative_count': negative_probe_changes
            },
            'cosine_change': {
                'mean': float(np.mean(cosine_changes)),
                'std': float(np.std(cosine_changes)),
                'min': float(np.min(cosine_changes)),
                'max': float(np.max(cosine_changes)),
                'positive_count': positive_cosine_changes,
                'negative_count': negative_cosine_changes
            }
        }
    else:
        aggregate_stats = {
            'total_anchors': 0,
            'probe_score_change': None,
            'cosine_change': None
        }

    # Token-specific analysis
    token_effects = {}
    for effect in all_effects:
        token = effect['anchor_token']
        if token not in token_effects:
            token_effects[token] = {
                'count': 0,
                'probe_changes': [],
                'cosine_changes': []
            }
        token_effects[token]['count'] += 1
        token_effects[token]['probe_changes'].append(effect['probe_score_change'])
        token_effects[token]['cosine_changes'].append(effect['cosine_change'])

    # Compute averages per token
    token_statistics = {}
    for token, data in token_effects.items():
        token_statistics[token] = {
            'count': data['count'],
            'mean_probe_change': float(np.mean(data['probe_changes'])),
            'mean_cosine_change': float(np.mean(data['cosine_changes']))
        }

    final_analysis = {
        'sample_analyses': sample_analyses,
        'aggregate_statistics': aggregate_stats,
        'token_statistics': token_statistics
    }

    # Print summary
    print(f"\nTotal anchor tokens found: {aggregate_stats['total_anchors']}")

    if aggregate_stats['total_anchors'] > 0:
        print(f"\nProbe score changes:")
        print(f"  Mean: {aggregate_stats['probe_score_change']['mean']:.4f}")
        print(f"  Std: {aggregate_stats['probe_score_change']['std']:.4f}")
        print(f"  Positive changes: {aggregate_stats['probe_score_change']['positive_count']}")
        print(f"  Negative changes: {aggregate_stats['probe_score_change']['negative_count']}")

        print(f"\nCosine similarity changes:")
        print(f"  Mean: {aggregate_stats['cosine_change']['mean']:.4f}")
        print(f"  Std: {aggregate_stats['cosine_change']['std']:.4f}")
        print(f"  Positive changes: {aggregate_stats['cosine_change']['positive_count']}")
        print(f"  Negative changes: {aggregate_stats['cosine_change']['negative_count']}")

        print(f"\nMost frequent anchor tokens:")
        sorted_tokens = sorted(token_statistics.items(),
                             key=lambda x: x[1]['count'],
                             reverse=True)
        for token, stats in sorted_tokens[:5]:
            print(f"  '{token}': {stats['count']} occurrences, "
                  f"avg probe change: {stats['mean_probe_change']:.4f}, "
                  f"avg cosine change: {stats['mean_cosine_change']:.4f}")

    # Save results
    if save_results:
        save_path = RESULTS_DIR / "anchor_analysis.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(final_analysis, f, indent=2, ensure_ascii=False)
        print(f"\nAnchor analysis saved to {save_path}")

    return final_analysis


def identify_successful_anchors(
    anchor_analysis: Dict,
    threshold: float = 0.05
) -> List[Dict]:
    """
    Identify anchor tokens that successfully cause representation recovery

    Args:
        anchor_analysis: Results from analyze_anchors_batch
        threshold: Minimum positive change to consider successful

    Returns:
        List of successful anchor instances
    """
    successful_anchors = []

    for sample in anchor_analysis['sample_analyses']:
        for effect in sample.get('anchor_effects', []):
            # An anchor is "successful" if it causes positive changes in both metrics
            if (effect['probe_score_change'] > threshold and
                effect['cosine_change'] > threshold):
                successful_anchors.append({
                    'sample_id': sample['sample_id'],
                    **effect
                })

    print(f"\nFound {len(successful_anchors)} successful anchor instances")
    print(f"Success rate: {len(successful_anchors) / anchor_analysis['aggregate_statistics']['total_anchors'] * 100:.2f}%"
          if anchor_analysis['aggregate_statistics']['total_anchors'] > 0 else "N/A")

    return successful_anchors


if __name__ == "__main__":
    print("="*80)
    print("ANCHOR ANALYSIS TEST")
    print("="*80)

    # Load drift tracking results
    results_path = RESULTS_DIR / "drift_tracking_results.json"

    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Please run drift_tracking.py first.")
    else:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # Analyze anchors
        anchor_analysis = analyze_anchors_batch(results, window_size=50)

        # Identify successful anchors
        successful = identify_successful_anchors(anchor_analysis, threshold=0.05)

        print("\n" + "="*80)
        print("ANCHOR ANALYSIS COMPLETED")
        print("="*80)
