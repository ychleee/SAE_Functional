#!/usr/bin/env python
"""
Cell content for loading dataset in Colab.
Copy this code into the notebook cell.
"""

code = """
import csv
import numpy as np
import pandas as pd
from collections import Counter

# Load dataset
texts = []
types = []
has_quantifier = []

# Try datasets in order of preference
import os
dataset_options = [
    'data/processed/quantifier_conditional_balanced.csv',
    'data/processed/quantifier_conditional_complete.csv',
    'data/processed/quantifier_conditional_dataset.csv'
]

dataset_file = None
for option in dataset_options:
    if os.path.exists(option):
        dataset_file = option
        break

if dataset_file is None:
    raise FileNotFoundError("No dataset file found!")
    
print(f"Loading dataset from: {dataset_file}")

with open(dataset_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row['text'])
        types.append(row['type'])
        has_quantifier.append(row.get('has_quantifier', 'False') == 'True')

# Statistics
type_counts = Counter(types)
print("="*60)
print("DATASET STATISTICS")
print("="*60)
print(f"Total sentences: {len(texts)}")

# Group for cleaner display
conditionals = type_counts.get('conditional', 0)
controls = type_counts.get('control', 0)
quantifiers = sum(v for k, v in type_counts.items() if 'universal' in k)

print(f"\\n📊 Main Categories:")
print(f"  • Conditionals: {conditionals:3d} sentences")
print(f"  • Quantifiers:  {quantifiers:3d} sentences")
print(f"  • Controls:     {controls:3d} sentences")

print(f"\\n📈 Balance Check:")
print(f"  • Ratio: {conditionals}:{quantifiers}:{controls}")

# Show samples
print("\\n" + "="*60)
print("SAMPLE SENTENCES")
print("="*60)

# Show one of each type
shown_types = set()
for i, (text, t) in enumerate(zip(texts, types)):
    if t not in shown_types:
        print(f"\\n[{t}]")
        print(f"  {text[:80]}...")
        shown_types.add(t)
        if len(shown_types) >= min(7, len(type_counts)):
            break
"""

print("Copy the code below into your Colab cell:")
print("=" * 60)
print(code)
print("=" * 60)