"""
Extract and process model activations for SAE training.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm


class ActivationExtractor:
    """Extract internal activations from transformer models."""
    
    def __init__(self, model_name: str = "EleutherAI/pythia-70m", device: str = None):
        """
        Initialize the activation extractor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extract_batch_activations(
        self, 
        texts: List[str], 
        layer_idx: int = -1,
        max_length: int = 128
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations for a batch of texts.
        
        Args:
            texts: List of input texts
            layer_idx: Which layer to extract from (-1 for last layer)
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with activations and metadata
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Forward pass with output_hidden_states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract hidden states from specified layer
        hidden_states = outputs.hidden_states[layer_idx]
        
        # Get attention mask for valid tokens
        attention_mask = inputs.attention_mask
        
        return {
            'activations': hidden_states.cpu(),
            'attention_mask': attention_mask.cpu(),
            'input_ids': inputs.input_ids.cpu()
        }
    
    def extract_token_activations(
        self,
        text: str,
        target_tokens: List[str] = ["if", "then"],
        layer_idx: int = -1
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations specifically for target tokens.
        
        Args:
            text: Input text
            target_tokens: Tokens to extract activations for
            layer_idx: Which layer to extract from
            
        Returns:
            Dictionary mapping tokens to their activations
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[layer_idx][0]  # Remove batch dim
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Find target token positions
        token_activations = {}
        for target in target_tokens:
            positions = []
            for i, token in enumerate(tokens):
                if target.lower() in token.lower():
                    positions.append(i)
            
            if positions:
                # Average activations across all occurrences
                target_acts = torch.stack([hidden_states[p] for p in positions])
                token_activations[target] = target_acts.mean(dim=0).cpu()
        
        return token_activations
    
    def extract_dataset_activations(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        layer_idx: int = -1,
        batch_size: int = 32,
        max_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract activations for an entire dataset.
        
        Args:
            df: DataFrame with texts
            text_column: Column name containing texts
            layer_idx: Which layer to extract from
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process
            
        Returns:
            Dictionary with stacked activations and metadata
        """
        texts = df[text_column].tolist()
        if max_samples:
            texts = texts[:max_samples]
        
        all_activations = []
        all_masks = []
        all_metadata = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
            batch_texts = texts[i:i+batch_size]
            batch_data = self.extract_batch_activations(batch_texts, layer_idx)
            
            all_activations.append(batch_data['activations'])
            all_masks.append(batch_data['attention_mask'])
            
            # Store metadata
            for j, text in enumerate(batch_texts):
                all_metadata.append({
                    'text': text,
                    'index': i + j,
                    'has_if': 'if' in text.lower(),
                    'has_then': 'then' in text.lower()
                })
        
        # Concatenate all batches
        activations = torch.cat(all_activations, dim=0)
        masks = torch.cat(all_masks, dim=0)
        
        return {
            'activations': activations.numpy(),
            'attention_masks': masks.numpy(),
            'metadata': all_metadata,
            'shape': activations.shape,
            'model_name': self.model.config._name_or_path,
            'layer_idx': layer_idx
        }
    

def prepare_activations_for_sae(
    activations_dict: Dict[str, np.ndarray],
    pool_method: str = 'mean'
) -> np.ndarray:
    """
    Prepare activations for SAE training by pooling across sequence dimension.
    
    Args:
        activations_dict: Output from extract_dataset_activations
        pool_method: How to pool ('mean', 'max', 'first', 'last')
        
    Returns:
        Pooled activations array (n_samples, hidden_dim)
    """
    activations = activations_dict['activations']
    masks = activations_dict['attention_masks']
    
    # activations shape: (batch, seq_len, hidden_dim)
    if pool_method == 'mean':
        # Mean pool over valid tokens only
        masks_expanded = np.expand_dims(masks, -1)
        masked_acts = activations * masks_expanded
        pooled = masked_acts.sum(axis=1) / masks.sum(axis=1, keepdims=True)
    elif pool_method == 'max':
        # Max pool
        pooled = activations.max(axis=1)
    elif pool_method == 'first':
        # Take first token
        pooled = activations[:, 0, :]
    elif pool_method == 'last':
        # Take last valid token
        batch_size = activations.shape[0]
        last_indices = masks.sum(axis=1).astype(int) - 1
        pooled = np.array([activations[i, last_indices[i], :] for i in range(batch_size)])
    else:
        raise ValueError(f"Unknown pooling method: {pool_method}")
    
    return pooled


def save_activations(activations_dict: Dict, filepath: str):
    """Save activations to file."""
    torch.save(activations_dict, filepath)
    print(f"Saved activations to {filepath}")


def load_activations(filepath: str) -> Dict:
    """Load activations from file."""
    return torch.load(filepath, map_location='cpu')


if __name__ == "__main__":
    # Test extraction with a few examples
    test_texts = [
        "If it rains, then the ground is wet.",
        "The cat sleeps on the mat.",
        "If A is true, then B follows."
    ]
    
    extractor = ActivationExtractor("EleutherAI/pythia-70m", device="cpu")
    
    # Test batch extraction
    batch_acts = extractor.extract_batch_activations(test_texts, layer_idx=-1)
    print(f"Batch activations shape: {batch_acts['activations'].shape}")
    
    # Test token extraction
    token_acts = extractor.extract_token_activations(
        test_texts[0], 
        target_tokens=["if", "then"]
    )
    print(f"Token activations: {list(token_acts.keys())}")