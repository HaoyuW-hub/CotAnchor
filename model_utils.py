"""
Model Utilities
Load model and extract hidden states
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional
import gc
from config import MODEL_NAME, DEVICE, HF_TOKEN, TARGET_LAYER


class ModelWrapper:
    """Wrapper for loading model and extracting hidden states"""

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.hidden_states_cache = []

    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=HF_TOKEN,
            trust_remote_code=True
        )

        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            output_hidden_states=True
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        print("Model loaded successfully!")

    def extract_hidden_state(
        self,
        input_ids: torch.Tensor,
        layer: int = TARGET_LAYER,
        position: int = -1
    ) -> torch.Tensor:
        """
        Extract hidden state from a specific layer and position

        Args:
            input_ids: Input token IDs
            layer: Which layer to extract from
            position: Which token position (-1 for last token)

        Returns:
            Hidden state vector
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True
            )
            # outputs.hidden_states is a tuple of (num_layers + 1) tensors
            # Shape: [batch_size, seq_len, hidden_size]
            hidden_state = outputs.hidden_states[layer]

            # Extract specific position
            if position == -1:
                return hidden_state[:, -1, :].cpu()
            else:
                return hidden_state[:, position, :].cpu()

    def generate_with_hidden_states(
        self,
        prompt: str,
        max_length: int = 1000,
        sample_interval: int = 20,
        layer: int = TARGET_LAYER,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> Dict:
        """
        Generate text and collect hidden states at regular intervals

        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            sample_interval: Extract hidden state every N tokens
            layer: Which layer to extract from
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            Dictionary containing:
            - generated_text: The generated text
            - tokens: List of generated tokens
            - hidden_states: List of hidden states at sample points
            - sample_positions: Token positions where states were sampled
        """
        print(f"\nGenerating with prompt: {prompt[:100]}...")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Storage for results
        hidden_states_list = []
        sample_positions = []

        # Generate with sampling
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Extract generated tokens
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)

        # Extract hidden states at intervals
        # Note: For efficiency, we'll need to re-run forward passes
        # or use a different generation approach with hooks
        for pos in range(input_length, len(generated_ids), sample_interval):
            if pos >= len(generated_ids):
                break

            # Run forward pass up to this position
            curr_ids = generated_ids[:pos+1].unsqueeze(0)
            hidden_state = self.extract_hidden_state(curr_ids, layer, position=-1)

            hidden_states_list.append(hidden_state.numpy())
            sample_positions.append(pos)

        # Also get initial hidden state (after prompt)
        initial_ids = generated_ids[:input_length].unsqueeze(0)
        initial_hidden = self.extract_hidden_state(initial_ids, layer, position=-1)

        return {
            "generated_text": generated_text,
            "tokens": tokens,
            "hidden_states": hidden_states_list,
            "sample_positions": sample_positions,
            "initial_hidden_state": initial_hidden.numpy(),
            "input_length": input_length,
            "total_length": len(generated_ids)
        }

    def get_initial_representation(
        self,
        prompt: str,
        layer: int = TARGET_LAYER
    ) -> np.ndarray:
        """
        Get initial hidden state representation after processing the prompt

        Args:
            prompt: Input prompt
            layer: Which layer to extract from

        Returns:
            Hidden state vector as numpy array
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        hidden_state = self.extract_hidden_state(inputs.input_ids, layer, position=-1)
        return hidden_state.numpy()

    def cleanup(self):
        """Clean up model from memory"""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def find_anchor_tokens(tokens: List[str], anchor_keywords: List[str]) -> List[Tuple[int, str]]:
    """
    Find positions of anchor tokens in the generated sequence

    Args:
        tokens: List of token strings
        anchor_keywords: List of anchor keywords to search for

    Returns:
        List of (position, token) tuples
    """
    anchors = []
    for i, token in enumerate(tokens):
        for keyword in anchor_keywords:
            if keyword.lower() in token.lower():
                anchors.append((i, token))
                break
    return anchors


if __name__ == "__main__":
    # Test model loading
    print("Testing model loading...")
    wrapper = ModelWrapper()
    wrapper.load_model()

    # Test prompt
    test_prompt = "Let n = 797. Explain why n is a prime number."

    # Test hidden state extraction
    print("\nTesting hidden state extraction...")
    hidden_state = wrapper.get_initial_representation(test_prompt)
    print(f"Hidden state shape: {hidden_state.shape}")
    print(f"Hidden state norm: {np.linalg.norm(hidden_state):.4f}")

    print("\nModel utilities test completed!")
