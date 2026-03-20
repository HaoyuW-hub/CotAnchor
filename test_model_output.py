"""
Test file to check model output for dataset prompts
"""

import os
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
import json
from pathlib import Path
from config import DATA_DIR, MODEL_NAME, DEVICE, MAX_LENGTH, TEMPERATURE, TOP_P
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    """Load DeepSeek model"""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE if DEVICE == "cuda" else None
    )
    return tokenizer, model

def generate_response(tokenizer, model, prompt):
    """Generate response for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if DEVICE == "cuda":
        inputs = inputs.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def load_dataset():
    """Load dataset from JSON file"""
    filepath = DATA_DIR / "pilot_dataset.json"
    if not filepath.exists():
        print(f"Dataset not found at {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples from {filepath}")
    return dataset

def test_model_on_prompts():
    """Test model on a few prompts from dataset"""
    print("Testing model on dataset prompts")

    # Load dataset
    dataset = load_dataset()
    if len(dataset) == 0:
        return

    # Load model
    tokenizer, model = load_model()

    # Test on 5 samples
    test_samples = dataset[:5]

    for i, sample in enumerate(test_samples):
        print(f"\n=== Sample {i+1} ===")
        print(f"Number: {sample['number']}")
        print(f"Is prime: {sample['is_prime']}")
        print(f"Label: {sample['label']}")
        print(f"\nPrompt:")
        print(sample['prompt'])
        print(f"\nModel Response:")

        response = generate_response(tokenizer, model, sample['prompt'])
        print(response)
        print("---")

    # Cleanup
    del model
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

if __name__ == "__main__":
    test_model_on_prompts()