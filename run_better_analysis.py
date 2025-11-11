#!/usr/bin/env python
"""
Better SAE training with lower loss while maintaining sparsity.
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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("=" * 70)
print("IMPROVED SPARSE AUTOENCODER ANALYSIS")
print("=" * 70)

# Load dataset
texts = []
types = []
with open('data/processed/quantifier_conditional_balanced.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row['text'])
        types.append(row['type'])

types_array = np.array(types)
type_counts = Counter(types)
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
batch_size = 32
for i in range(0, len(texts), batch_size):
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

# Normalize
activations = (activations - activations.mean(0)) / (activations.std(0) + 1e-8)
print(f"  Shape: {activations.shape}")

# Better SAE with gradual sparsity
class ImprovedSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super().__init__()
        
        # Encoder with bias
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder (no bias, tied weights)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.decoder.weight = nn.Parameter(self.encoder.weight.t().clone())
        
        # Initialize carefully
        nn.init.xavier_normal_(self.encoder.weight, gain=1.0)
        nn.init.zeros_(self.encoder.bias)
    
    def forward(self, x, k=None):
        # Encode with ReLU
        h = torch.relu(self.encoder(x))
        
        # Optional top-k sparsity
        if k is not None and k < h.shape[1]:
            topk_vals, topk_idx = torch.topk(h, k, dim=1)
            mask = torch.zeros_like(h)
            mask.scatter_(1, topk_idx, 1)
            h = h * mask
        
        # Decode
        recon = self.decoder(h)
        return recon, h

print("\nTraining Improved SAE...")
input_dim = activations.shape[1]
hidden_dim = 2048

sae = ImprovedSAE(input_dim, hidden_dim).to(device)
print(f"  Architecture: {input_dim} → {hidden_dim} → {input_dim}")

# Start without sparsity constraint
optimizer = optim.AdamW(sae.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                          steps_per_epoch=1, epochs=300)

acts_tensor = activations.to(device)

print("\n  Phase 1: Training without sparsity constraint...")
for epoch in range(100):
    recon, code = sae(acts_tensor, k=None)  # No top-k initially
    
    # Just reconstruction loss
    loss = nn.functional.mse_loss(recon, acts_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    if epoch % 25 == 0:
        active = (code > 0).float().sum(1).mean().item()
        print(f"    Epoch {epoch}: Loss={loss:.4f}, Active={active:.1f}")

print("\n  Phase 2: Gradually introducing sparsity...")
# Now gradually increase sparsity
k_schedule = [100, 80, 60, 50, 40, 30]
for k in k_schedule:
    for epoch in range(50):
        recon, code = sae(acts_tensor, k=k)
        
        # Reconstruction + mild L1
        mse_loss = nn.functional.mse_loss(recon, acts_tensor)
        l1_loss = 0.001 * code.abs().mean()
        loss = mse_loss + l1_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    print(f"    k={k}: Loss={loss:.4f}, MSE={mse_loss:.4f}")

# Final evaluation
sae.eval()
print("\n  Final evaluation with k=30...")
with torch.no_grad():
    recon, codes = sae(acts_tensor, k=30)
    final_mse = nn.functional.mse_loss(recon, acts_tensor).item()
    codes = codes.cpu().numpy()

print(f"    Final MSE: {final_mse:.4f}")
print(f"    Active features: {(codes > 0).sum(1).mean():.1f}")
print(f"    Max activation: {codes.max():.2f}")
print(f"    Mean activation (when active): {codes[codes>0].mean():.2f}")

# Analyze overlap
print("\nAnalyzing feature overlap...")

# Separate by type
cond_codes = codes[types_array == 'conditional']
quant_codes = codes[np.isin(types_array, ['pure_universal', 'restricted_universal',
                                           'negative_universal', 'generic_universal', 
                                           'any_universal'])]
ctrl_codes = codes[types_array == 'control']

# Calculate which features are most active for each type
def get_top_features(codes, top_n=20):
    """Get features that activate most frequently."""
    activation_freq = (codes > 0).mean(axis=0)
    return np.argsort(activation_freq)[-top_n:][::-1]

cond_top = get_top_features(cond_codes)
quant_top = get_top_features(quant_codes)
ctrl_top = get_top_features(ctrl_codes)

# Find overlaps
cond_set = set(cond_top)
quant_set = set(quant_top)
ctrl_set = set(ctrl_top)

shared_cond_quant = cond_set & quant_set
unique_cond = cond_set - quant_set - ctrl_set
unique_quant = quant_set - cond_set - ctrl_set

print(f"\nTop 20 most active features per type:")
print(f"  • Conditional-specific: {len(unique_cond)}")
print(f"  • Quantifier-specific: {len(unique_quant)}")
print(f"  • Shared (Cond & Quant): {len(shared_cond_quant)}")

if len(shared_cond_quant) > 0:
    print(f"\n  Shared features: {sorted(list(shared_cond_quant))[:10]}")
    
    # Show example sentences for a shared feature
    shared_feat = list(shared_cond_quant)[0]
    print(f"\n  Example activations for shared feature {shared_feat}:")
    
    # Get top activating sentences
    cond_acts = cond_codes[:, shared_feat]
    quant_acts = quant_codes[:, shared_feat]
    
    cond_top_idx = np.argsort(cond_acts)[-3:][::-1]
    quant_top_idx = np.argsort(quant_acts)[-3:][::-1]
    
    cond_texts = [t for t, ty in zip(texts, types_array) if ty == 'conditional']
    quant_texts = [t for t, ty in zip(texts, types_array) if 'universal' in ty]
    
    print("    Conditionals:")
    for idx in cond_top_idx:
        if idx < len(cond_texts):
            print(f"      [{cond_acts[idx]:.2f}] {cond_texts[idx][:60]}...")
    
    print("    Quantifiers:")
    for idx in quant_top_idx:
        if idx < len(quant_texts):
            print(f"      [{quant_acts[idx]:.2f}] {quant_texts[idx][:60]}...")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)