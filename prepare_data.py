#!/usr/bin/env python
"""
Prepare datasets for SAE training on conditional features.
Run this script locally before training on Colab.
"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append('src')

from data_utils import (
    ConditionalDatasetGenerator,
    create_minimal_test_set,
    save_dataset
)

def main():
    print("=" * 60)
    print("Data Preparation for SAE Conditional Analysis")
    print("=" * 60)
    
    # Load configuration
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data directory
    data_dir = Path(config['paths']['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    print("\n1. Generating synthetic conditional dataset...")
    generator = ConditionalDatasetGenerator(seed=42)
    
    # Generate main dataset
    n_samples = config['data']['n_samples']
    df = generator.generate_dataset(n_samples=n_samples)
    
    print(f"   Generated {len(df)} sentences")
    print(f"   - Conditionals: {df['has_conditional'].sum()}")
    print(f"   - Controls: {(~df['has_conditional']).sum()}")
    print(f"   - Types: {', '.join(df['type'].unique())}")
    
    # Save main dataset
    main_path = data_dir / 'conditionals_dataset.csv'
    save_dataset(df, str(main_path))
    print(f"   ✓ Saved to: {main_path}")
    
    # Create test set
    print("\n2. Creating inference test set...")
    test_df = create_minimal_test_set()
    test_path = data_dir / 'inference_test_set.csv'
    save_dataset(test_df, str(test_path))
    print(f"   ✓ Saved {len(test_df)} test cases to: {test_path}")
    
    # Create balanced dataset
    print("\n3. Creating balanced dataset...")
    n_per_type = df['has_conditional'].value_counts().min()
    balanced_df = df.groupby('has_conditional').sample(n=n_per_type, random_state=42)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    balanced_path = data_dir / 'balanced_dataset.csv'
    save_dataset(balanced_df, str(balanced_path))
    print(f"   ✓ Saved balanced dataset ({len(balanced_df)} samples) to: {balanced_path}")
    
    # Display sample sentences
    print("\n4. Sample sentences:")
    print("-" * 40)
    for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
        print(f"{i}. [{row['type']}]")
        print(f"   {row['text']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print(f"   Total files created: 3")
    print(f"   Total samples: {len(df) + len(test_df) + len(balanced_df)}")
    print("\nNext steps:")
    print("1. Review the generated data in data/processed/")
    print("2. Run: git add data/processed/*.csv")
    print("3. Run: git commit -m 'Add prepared datasets'")
    print("4. Run: git push")
    print("5. Upload notebooks/02_sae_training.ipynb to Google Colab")
    print("=" * 60)

if __name__ == "__main__":
    main()