#!/usr/bin/env python
"""
Test the minimal SAE training pipeline locally
"""

import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=" * 60)
print("Testing Minimal SAE Pipeline")
print("=" * 60)

# 1. Load data without pandas
print("\n1. Loading data...")
texts = []
labels = []

try:
    with open('data/processed/conditionals_dataset.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            labels.append(row['has_conditional'] == 'True')
    
    print(f"✓ Loaded {len(texts)} sentences")
    print(f"  Conditionals: {sum(labels)}")
    print(f"  Controls: {len(labels) - sum(labels)}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# 2. Create fake activations (since we can't run the model locally)
print("\n2. Creating mock activations...")
n_samples = min(500, len(texts))
input_dim = 512  # Typical for small models
activations = torch.randn(n_samples, input_dim)
print(f"✓ Mock activations shape: {activations.shape}")

# 3. Define simple SAE
print("\n3. Creating SAE model...")
class SimpleSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, sparsity=0.01):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.sparsity = sparsity
        
        # Tie weights
        self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        
        # Initialize
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
    
    def forward(self, x):
        code = torch.relu(self.encoder(x))
        recon = self.decoder(code)
        return recon, code

try:
    sae = SimpleSAE(input_dim, hidden_dim=1024)
    print(f"✓ SAE created: {input_dim} -> 1024 features")
except Exception as e:
    print(f"✗ Error creating SAE: {e}")
    exit(1)

# 4. Test training step
print("\n4. Testing training...")
optimizer = optim.Adam(sae.parameters(), lr=0.001)

try:
    for epoch in range(3):  # Just 3 epochs for testing
        # Forward
        recon, code = sae(activations)
        
        # Loss
        recon_loss = nn.functional.mse_loss(recon, activations)
        sparse_loss = sae.sparsity * code.abs().mean()
        loss = recon_loss + sparse_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
    
    print("✓ Training works!")
except Exception as e:
    print(f"✗ Error during training: {e}")
    exit(1)

# 5. Test analysis
print("\n5. Testing feature analysis...")
try:
    with torch.no_grad():
        _, codes = sae(activations)
        codes = codes.numpy()
    
    # Split by label
    labels_array = np.array(labels[:n_samples])
    cond_codes = codes[labels_array]
    ctrl_codes = codes[~labels_array]
    
    # Find differential features
    cond_mean = cond_codes.mean(0)
    ctrl_mean = ctrl_codes.mean(0)
    diff = cond_mean - ctrl_mean
    
    # Top features
    top_idx = np.argsort(diff)[-5:][::-1]
    
    print("✓ Top 5 differential features:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. Feature {idx}: diff={diff[idx]:.3f}")
    
    # Stats
    print(f"\n✓ Active features per sample:")
    print(f"  Conditionals: {(cond_codes > 0).sum(1).mean():.1f}")
    print(f"  Controls: {(ctrl_codes > 0).sum(1).mean():.1f}")
    
except Exception as e:
    print(f"✗ Error during analysis: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! The pipeline works locally.")
print("This code should work in Colab too.")
print("=" * 60)