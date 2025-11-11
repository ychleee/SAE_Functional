#!/usr/bin/env python
"""
Run the quantifier-conditional overlap analysis locally.
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

# Use MPS if available (Apple Silicon), otherwise CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("=" * 70)
print("RUNNING LOCAL ANALYSIS: CONDITIONALS VS QUANTIFIERS")
print("=" * 70)

# 1. Load Dataset
print("\n1. Loading dataset...")
texts = []
types = []

dataset_file = 'data/processed/quantifier_conditional_balanced.csv'
with open(dataset_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row['text'])
        types.append(row['type'])

types_array = np.array(types)
type_counts = Counter(types)

conditionals = type_counts.get('conditional', 0)
controls = type_counts.get('control', 0)
quantifiers = sum(v for k, v in type_counts.items() if 'universal' in k)

print(f"  • Conditionals: {conditionals} sentences")
print(f"  • Quantifiers: {quantifiers} sentences")
print(f"  • Controls: {controls} sentences")
print(f"  • Total: {len(texts)} sentences")

# 2. Load Language Model and Extract Activations
print("\n2. Loading language model...")
model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"  ✓ Loaded {model_name}")

# 3. Extract Activations
print("\n3. Extracting activations...")
batch_size = 32
activations = []

for i in range(0, len(texts), batch_size):
    if i % 100 == 0:
        print(f"  Processing {i}/{len(texts)}...")
    
    batch = texts[i:i+batch_size]
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        
        # Mean pooling
        mask = inputs.attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        activations.append(pooled.cpu())

activations = torch.cat(activations, dim=0)
print(f"  ✓ Activations shape: {activations.shape}")

# 4. Train Sparse Autoencoder
print("\n4. Training Sparse Autoencoder...")

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, sparsity_coeff=0.1, top_k=30):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.sparsity_coeff = sparsity_coeff
        self.top_k = top_k
        
        # Tie weights
        self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        
        # Initialize for sparsity
        nn.init.xavier_uniform_(self.encoder.weight, gain=0.1)
        nn.init.zeros_(self.encoder.bias)
    
    def encode(self, x):
        h = torch.relu(self.encoder(x))
        
        # Always apply top-k sparsity (not just during training)
        if self.top_k is not None:
            topk_vals, topk_idx = torch.topk(h, self.top_k, dim=1)
            mask = torch.zeros_like(h)
            mask.scatter_(1, topk_idx, 1)
            h = h * mask
        
        return h
    
    def forward(self, x):
        code = self.encode(x)
        recon = self.decoder(code)
        return recon, code

# Create and train SAE
input_dim = activations.shape[1]
hidden_dim = 2048
sae = SparseAutoencoder(input_dim, hidden_dim, sparsity_coeff=0.1, top_k=30).to(device)

print(f"  SAE: {input_dim} → {hidden_dim} → {input_dim}")
print(f"  Sparsity: L1=0.1, Top-k=30")

optimizer = optim.Adam(sae.parameters(), lr=0.0005)
acts_tensor = activations.to(device)

# Training loop
print("\n  Training...")
for epoch in range(200):
    beta_warm = min(1.0, epoch / 50)
    
    recon, code = sae(acts_tensor)
    recon_loss = nn.functional.mse_loss(recon, acts_tensor)
    sparse_loss = beta_warm * sae.sparsity_coeff * code.abs().mean()
    total_loss = recon_loss + sparse_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 50 == 0:
        active_per_sample = (code > 0).float().sum(dim=1).mean().item()
        print(f"    Epoch {epoch}: Loss={total_loss:.4f}, Active features={active_per_sample:.1f}")

# Final check
sae.eval()
with torch.no_grad():
    _, final_codes = sae(acts_tensor)
    active_per_sample = (final_codes > 0).float().sum(dim=1).mean().item()
    max_activation = final_codes.max().item()
    mean_activation = final_codes[final_codes > 0].mean().item() if (final_codes > 0).any() else 0

print(f"\n  ✓ Training complete!")
print(f"  ✓ Active features per sample: {active_per_sample:.1f}")
print(f"  ✓ Max activation: {max_activation:.2f}")
print(f"  ✓ Mean activation when active: {mean_activation:.2f}")

# 5. Analyze Feature Overlap
print("\n5. Analyzing feature overlap...")

# Encode by type
with torch.no_grad():
    sparse_codes = sae.encode(acts_tensor).cpu().numpy()

# Separate by type
cond_mask = types_array == 'conditional'
quant_mask = np.isin(types_array, ['pure_universal', 'restricted_universal', 
                                     'negative_universal', 'generic_universal', 'any_universal'])
ctrl_mask = types_array == 'control'

cond_codes = sparse_codes[cond_mask]
quant_codes = sparse_codes[quant_mask]
ctrl_codes = sparse_codes[ctrl_mask]

# Calculate mean activations
mean_cond = cond_codes.mean(axis=0)
mean_quant = quant_codes.mean(axis=0)
mean_ctrl = ctrl_codes.mean(axis=0)

# Find characteristic features (threshold = 0.01)
threshold = 0.01
cond_diff = mean_cond - mean_ctrl
quant_diff = mean_quant - mean_ctrl

cond_specific = np.where((cond_diff > threshold) & (cond_diff > quant_diff))[0]
quant_specific = np.where((quant_diff > threshold) & (quant_diff > cond_diff))[0]
shared = np.where((cond_diff > threshold) & (quant_diff > threshold) & 
                  (np.abs(cond_diff - quant_diff) < threshold))[0]

print(f"  • Conditional-specific features: {len(cond_specific)}")
print(f"  • Quantifier-specific features: {len(quant_specific)}")
print(f"  • Shared features: {len(shared)}")

total_relevant = len(cond_specific) + len(quant_specific) + len(shared)
if total_relevant > 0:
    overlap_pct = len(shared) / total_relevant * 100
    print(f"  • Overlap: {overlap_pct:.1f}%")

# Show top shared features
if len(shared) > 0:
    print("\n  Top shared features:")
    for i, feat_idx in enumerate(shared[:5]):
        print(f"    {i+1}. Feature {feat_idx}: Cond={mean_cond[feat_idx]:.3f}, "
              f"Quant={mean_quant[feat_idx]:.3f}, Ctrl={mean_ctrl[feat_idx]:.3f}")

# Compute cosine similarity manually (avoid sklearn dependency)
def cosine_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

similarity = np.array([
    [1.0, cosine_sim(mean_cond, mean_quant), cosine_sim(mean_cond, mean_ctrl)],
    [cosine_sim(mean_quant, mean_cond), 1.0, cosine_sim(mean_quant, mean_ctrl)],
    [cosine_sim(mean_ctrl, mean_cond), cosine_sim(mean_ctrl, mean_quant), 1.0]
])

print(f"\n  Cosine similarities:")
print(f"    • Conditional-Quantifier: {similarity[0, 1]:.3f}")
print(f"    • Conditional-Control: {similarity[0, 2]:.3f}")
print(f"    • Quantifier-Control: {similarity[1, 2]:.3f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)