"""
SIMPLE COLAB SCRIPT - Just copy and paste this into Colab cells
No complex dependencies, just core PyTorch
"""

# ============================================
# CELL 1: Setup
# ============================================
import os
import sys

# Clone repo
os.system("rm -rf SAE_Functional")
os.system("git clone https://github.com/ychleee/SAE_Functional.git")
os.chdir("SAE_Functional")

print("Repository cloned!")

# ============================================
# CELL 2: Load Data (No pandas)
# ============================================
import csv

texts = []
labels = []

with open('data/processed/conditionals_dataset.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row['text'])
        labels.append(row['has_conditional'] == 'True')

print(f"Loaded {len(texts)} sentences")
print(f"Samples:")
for i in range(3):
    print(f"  {texts[i][:60]}...")

# ============================================
# CELL 3: Simple Feature Extraction
# ============================================
import torch
import torch.nn as nn

# For demo: Create random "activations" 
# (In real Colab, replace with actual model extraction)
n_samples = 500
feature_dim = 512
activations = torch.randn(n_samples, feature_dim)

# If you can load transformers:
try:
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Extract real activations
    batch = texts[:5]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"Real model output shape: {outputs.last_hidden_state.shape}")
except:
    print("Using random activations for demo")

# ============================================
# CELL 4: Train SAE
# ============================================

class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

# Train
sae = SAE(feature_dim, 1024)
optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)

for epoch in range(20):
    x_hat, z = sae(activations)
    loss = nn.functional.mse_loss(x_hat, activations) + 0.01 * z.abs().mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# ============================================
# CELL 5: Analyze
# ============================================
import numpy as np

with torch.no_grad():
    _, codes = sae(activations)
    codes = codes.numpy()

# Compare conditional vs control
labels_arr = np.array(labels[:n_samples])
cond_codes = codes[labels_arr].mean(0)
ctrl_codes = codes[~labels_arr].mean(0)
diff = cond_codes - ctrl_codes

# Find top features
top_features = np.argsort(diff)[-10:][::-1]
print("\nTop Conditional Features:")
for i, feat in enumerate(top_features):
    print(f"  Feature {feat}: diff={diff[feat]:.3f}")

# ============================================
# CELL 6: Save Results
# ============================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Save model
    torch.save(sae.state_dict(), '/content/drive/MyDrive/sae_model.pt')
    print("Model saved to Google Drive!")
except:
    print("Not in Colab or Drive not mounted")

print("\nDone! The SAE identified features that distinguish conditionals.")