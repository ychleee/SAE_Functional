#!/usr/bin/env python
"""
Fixed analysis with properly working sparse autoencoder.
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

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("=" * 70)
print("FIXED ANALYSIS: CONDITIONALS VS QUANTIFIERS")
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
print(f"  • Conditionals: {type_counts.get('conditional', 0)}")
print(f"  • Quantifiers: {sum(v for k, v in type_counts.items() if 'universal' in k)}")
print(f"  • Controls: {type_counts.get('control', 0)}")

# Load model and extract activations
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
print(f"  Activations: {activations.shape}")

# Normalize activations BEFORE training
activations = (activations - activations.mean(0)) / (activations.std(0) + 1e-8)
print(f"  Normalized: mean={activations.mean():.3f}, std={activations.std():.3f}")

# Fixed Sparse Autoencoder
class FixedSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, k=20):
        super().__init__()
        self.k = k
        
        # Smaller network for stability
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Better initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        
        # Top-k sparsity
        if self.k is not None:
            topk_vals, topk_idx = torch.topk(h, self.k, dim=1)
            mask = torch.zeros_like(h)
            mask.scatter_(1, topk_idx, 1)
            h = h * mask
        
        # Decode
        recon = self.decoder(h)
        return recon, h

# Train with lower learning rate and L2 regularization
print("\nTraining Fixed SAE...")
input_dim = activations.shape[1]
hidden_dim = 1024  # Smaller for stability
k = 20  # Fewer active features

sae = FixedSAE(input_dim, hidden_dim, k).to(device)
print(f"  Architecture: {input_dim} → {hidden_dim} (top-{k}) → {input_dim}")

# Much lower learning rate and weight decay
optimizer = optim.AdamW(sae.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

acts_tensor = activations.to(device)
best_loss = float('inf')

for epoch in range(100):
    recon, code = sae(acts_tensor)
    
    # MSE loss + L1 sparsity
    mse_loss = nn.functional.mse_loss(recon, acts_tensor)
    l1_loss = 0.01 * code.abs().mean()  # Lighter sparsity
    loss = mse_loss + l1_loss
    
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 0.5)
    
    optimizer.step()
    scheduler.step()
    
    if epoch % 25 == 0:
        active = (code > 0).float().sum(1).mean().item()
        max_act = code.max().item()
        print(f"  Epoch {epoch}: Loss={loss:.4f}, Active={active:.1f}, Max={max_act:.2f}")
    
    # Track best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_state = sae.state_dict()

# Load best model
sae.load_state_dict(best_state)
sae.eval()

# Analyze
print("\nAnalyzing features...")
with torch.no_grad():
    _, codes = sae(acts_tensor)
    codes = codes.cpu().numpy()

# Separate by type
cond_codes = codes[types_array == 'conditional']
quant_codes = codes[np.isin(types_array, ['pure_universal', 'restricted_universal',
                                           'negative_universal', 'generic_universal', 
                                           'any_universal'])]
ctrl_codes = codes[types_array == 'control']

# Statistics
print(f"\nFeature Statistics:")
print(f"  Conditionals: {(cond_codes > 0).sum(1).mean():.1f} active features")
print(f"  Quantifiers: {(quant_codes > 0).sum(1).mean():.1f} active features")
print(f"  Controls: {(ctrl_codes > 0).sum(1).mean():.1f} active features")

# Find differential features
cond_mean = (cond_codes > 0).mean(0)  # Activation frequency
quant_mean = (quant_codes > 0).mean(0)
ctrl_mean = (ctrl_codes > 0).mean(0)

# Features that activate more for conditionals/quantifiers than controls
threshold = 0.1
cond_specific = np.where((cond_mean - ctrl_mean > threshold) & 
                         (cond_mean > quant_mean))[0]
quant_specific = np.where((quant_mean - ctrl_mean > threshold) & 
                          (quant_mean > cond_mean))[0]
shared = np.where((cond_mean - ctrl_mean > threshold) & 
                  (quant_mean - ctrl_mean > threshold))[0]

print(f"\nFeature Discovery:")
print(f"  • Conditional-specific: {len(cond_specific)}")
print(f"  • Quantifier-specific: {len(quant_specific)}")
print(f"  • Shared logical: {len(shared)}")

if len(shared) > 0:
    print(f"\nTop shared features (by activation frequency):")
    for i, feat in enumerate(shared[:5]):
        print(f"    Feature {feat}: Cond={cond_mean[feat]:.2f}, "
              f"Quant={quant_mean[feat]:.2f}, Ctrl={ctrl_mean[feat]:.2f}")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)