#!/usr/bin/env python
"""
Stable SAE with simple architecture.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Force CPU for stability (MPS seems unstable)
device = torch.device("cpu")
print(f"Using device: {device}")

print("=" * 70)
print("STABLE SPARSE AUTOENCODER ANALYSIS")
print("=" * 70)

# Load dataset
texts = []
types = []
with open('data/processed/quantifier_conditional_balanced.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row['text'])
        types.append(row['type'])

types_array = np.array(types[:500])  # Limit for speed
texts = texts[:500]
print(f"\nDataset: {len(texts)} sentences")

# Extract activations
print("\nExtracting activations...")
model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

activations = []
batch_size = 16
for i in range(0, len(texts), batch_size):
    if i % 100 == 0:
        print(f"  {i}/{len(texts)}")
    batch = texts[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, 
                      truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs.attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        activations.append(pooled.cpu())

activations = torch.cat(activations, dim=0)

# Normalize to [-1, 1] range
acts_min = activations.min()
acts_max = activations.max()
activations = 2 * (activations - acts_min) / (acts_max - acts_min) - 1
print(f"  Normalized to [{activations.min():.2f}, {activations.max():.2f}]")

# Simple SAE without tied weights
class SimpleSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super().__init__()
        
        # Simple encoder-decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Stabilize
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Keep output bounded
        )
        
        # Small initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        code = self.encoder(x)
        
        # Simple top-k sparsity
        k = 50  # Allow more features for better reconstruction
        if k < code.shape[1]:
            topk_vals, topk_idx = torch.topk(code, k, dim=1)
            mask = torch.zeros_like(code)
            mask.scatter_(1, topk_idx, 1)
            code = code * mask
        
        recon = self.decoder(code)
        return recon, code

print("\nTraining Stable SAE...")
input_dim = activations.shape[1]
hidden_dim = 1024

sae = SimpleSAE(input_dim, hidden_dim).to(device)
print(f"  Architecture: {input_dim} → {hidden_dim} (sparse) → {input_dim}")

# Very conservative optimization
optimizer = optim.Adam(sae.parameters(), lr=1e-4)
acts_tensor = activations.to(device)

losses = []
print("\n  Training...")
for epoch in range(200):
    recon, code = sae(acts_tensor)
    
    # Simple MSE loss
    loss = nn.functional.mse_loss(recon, acts_tensor)
    
    # Add very light L1 regularization
    if epoch > 50:
        loss = loss + 0.0001 * code.abs().mean()
    
    optimizer.zero_grad()
    loss.backward()
    
    # Small gradient clipping
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 50 == 0:
        active = (code > 0).float().sum(1).mean().item()
        print(f"    Epoch {epoch}: Loss={loss:.4f}, Active={active:.1f}/{hidden_dim}")

# Check if training worked
if losses[-1] < losses[0]:
    print(f"\n  ✓ Training successful! Loss decreased {losses[0]:.4f} → {losses[-1]:.4f}")
else:
    print(f"\n  ✗ Training may have issues. Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

# Analyze
sae.eval()
with torch.no_grad():
    _, codes = sae(acts_tensor)
    codes = codes.numpy()

print(f"\n  Final statistics:")
print(f"    • Active features per sample: {(codes > 0).sum(1).mean():.1f}")
print(f"    • Max activation: {codes.max():.2f}")
print(f"    • Reconstruction loss: {losses[-1]:.4f}")

# Feature analysis
print("\nAnalyzing features...")

# Separate by type
cond_mask = types_array == 'conditional'
quant_mask = np.isin(types_array, ['pure_universal', 'restricted_universal',
                                     'negative_universal', 'generic_universal', 
                                     'any_universal'])
ctrl_mask = types_array == 'control'

if cond_mask.sum() > 0 and quant_mask.sum() > 0 and ctrl_mask.sum() > 0:
    cond_codes = codes[cond_mask]
    quant_codes = codes[quant_mask]
    ctrl_codes = codes[ctrl_mask]
    
    # Find differential features
    cond_active = (cond_codes > 0.1).mean(0)  # Frequency of activation
    quant_active = (quant_codes > 0.1).mean(0)
    ctrl_active = (ctrl_codes > 0.1).mean(0)
    
    # Features more active in conditionals/quantifiers than controls
    cond_diff = cond_active - ctrl_active
    quant_diff = quant_active - ctrl_active
    
    # Lower threshold for more sensitivity
    threshold = 0.05  # Was 0.1
    
    # Shared features: active in both cond and quant, but not control
    shared_mask = (cond_diff > threshold) & (quant_diff > threshold)
    cond_only_mask = (cond_diff > threshold) & (quant_diff <= threshold)
    quant_only_mask = (quant_diff > threshold) & (cond_diff <= threshold)
    
    # Also find top differential features by magnitude
    cond_top10 = np.argsort(cond_diff)[-10:][::-1]
    quant_top10 = np.argsort(quant_diff)[-10:][::-1]
    
    print(f"\n  Top differential features (vs control):")
    print(f"    Conditional top: {cond_top10[:5].tolist()}")
    print(f"    Quantifier top: {quant_top10[:5].tolist()}")
    print(f"    Overlap in top 10: {len(set(cond_top10) & set(quant_top10))}")
    
    print(f"\n  Feature overlap:")
    print(f"    • Conditional-specific: {cond_only_mask.sum()}")
    print(f"    • Quantifier-specific: {quant_only_mask.sum()}")
    print(f"    • Shared logical features: {shared_mask.sum()}")
    
    if shared_mask.sum() > 0:
        shared_features = np.where(shared_mask)[0]
        print(f"\n  Shared features: {shared_features[:10].tolist()}")
        
        # Show activation rates
        for feat in shared_features[:3]:
            print(f"    Feature {feat}: Cond={cond_active[feat]:.2%}, "
                  f"Quant={quant_active[feat]:.2%}, Ctrl={ctrl_active[feat]:.2%}")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)