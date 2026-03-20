"""
Data Preparation Module
Generates prompts for prime number detection task
"""

import json
from pathlib import Path
from typing import List, Dict
from config import (
    PROMPT_TEMPLATE, PRIME_NUMBERS, COMPOSITE_NUMBERS,
    NUM_POSITIVE_SAMPLES, NUM_NEGATIVE_SAMPLES, DATA_DIR
)


def is_prime(n: int) -> bool:
    """Check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def generate_prompt(number: int, is_prime_num: bool) -> str:
    """Generate a prompt for a given number"""
    prime_or_composite = "a prime number" if is_prime_num else "a composite number"
    return PROMPT_TEMPLATE.format(
        number=number,
        prime_or_composite=prime_or_composite
    )


def create_dataset() -> List[Dict]:
    """
    Create dataset with positive (prime) and negative (composite) examples

    Returns:
        List of dictionaries containing:
        - number: the number
        - is_prime: boolean
        - prompt: the full prompt text
    """
    dataset = []

    # Add prime numbers (positive examples)
    for i, prime in enumerate(PRIME_NUMBERS[:NUM_POSITIVE_SAMPLES]):
        dataset.append({
            "id": f"pos_{i}",
            "number": prime,
            "is_prime": True,
            "label": 1,
            "prompt": generate_prompt(prime, True)
        })

    # Add composite numbers (negative examples)
    for i, composite in enumerate(COMPOSITE_NUMBERS[:NUM_NEGATIVE_SAMPLES]):
        dataset.append({
            "id": f"neg_{i}",
            "number": composite,
            "is_prime": False,
            "label": 0,
            "prompt": generate_prompt(composite, False)
        })

    return dataset


def save_dataset(dataset: List[Dict], filename: str = "pilot_dataset.json"):
    """Save dataset to JSON file"""
    filepath = DATA_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Dataset saved to {filepath}")
    print(f"Total samples: {len(dataset)}")
    print(f"Positive (prime) samples: {sum(1 for d in dataset if d['is_prime'])}")
    print(f"Negative (composite) samples: {sum(1 for d in dataset if not d['is_prime'])}")


def load_dataset(filename: str = "pilot_dataset.json") -> List[Dict]:
    """Load dataset from JSON file"""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"Dataset not found at {filepath}. Creating new dataset...")
        dataset = create_dataset()
        save_dataset(dataset, filename)
        return dataset

    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples from {filepath}")
    return dataset


if __name__ == "__main__":
    # Generate and save dataset
    print("Generating pilot experiment dataset...")
    print(f"Prime numbers: {PRIME_NUMBERS[:NUM_POSITIVE_SAMPLES]}")
    print(f"Composite numbers: {COMPOSITE_NUMBERS[:NUM_NEGATIVE_SAMPLES]}")

    dataset = create_dataset()
    save_dataset(dataset)

    # Display sample
    print("\n" + "="*80)
    print("Sample prompt (Prime number):")
    print("="*80)
    print(dataset[0]["prompt"])

    print("\n" + "="*80)
    print("Sample prompt (Composite number):")
    print("="*80)
    print(dataset[NUM_POSITIVE_SAMPLES]["prompt"])
