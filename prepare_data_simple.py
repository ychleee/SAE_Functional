#!/usr/bin/env python
"""
Simple data preparation script without pandas dependency.
"""

import csv
import random
from pathlib import Path
import yaml

def generate_conditionals(n_samples=1000, seed=42):
    """Generate synthetic conditional sentences."""
    random.seed(seed)
    
    # Template components
    subjects = ["John", "Mary", "The cat", "The student", "The team", "She", "He", "They"]
    verbs_present = ["runs", "reads", "works", "studies", "plays", "sings", "writes", "thinks"]
    verbs_future = ["will run", "will read", "will work", "will study", "will play", "will sing"]
    objects = ["the book", "quickly", "hard", "the game", "a song", "carefully", "the report", "deeply"]
    weather = ["rains", "snows", "is sunny", "is cloudy", "is windy"]
    consequences = ["the picnic is canceled", "we stay inside", "we go to the beach", 
                   "the game continues", "school closes", "traffic increases"]
    
    data = []
    
    # Generate simple conditionals (n_samples // 5)
    for _ in range(n_samples // 5):
        if_clause = random.choice(["it " + w for w in weather])
        then_clause = random.choice(consequences)
        text = f"If {if_clause}, then {then_clause}."
        data.append({
            'text': text,
            'type': 'simple_conditional',
            'has_conditional': True
        })
    
    # Generate logical conditionals (n_samples // 5)
    for _ in range(n_samples // 5):
        A = random.choice(["A", "P", "X", "the statement"])
        B = random.choice(["B", "Q", "Y", "the conclusion"])
        templates = [
            f"If {A} is true, then {B} is true.",
            f"If {A} and not {B}, then contradiction.",
            f"If {A} or {B}, then at least one holds.",
        ]
        text = random.choice(templates)
        data.append({
            'text': text,
            'type': 'logical_conditional',
            'has_conditional': True
        })
    
    # Generate complex conditionals (n_samples // 5)
    for _ in range(n_samples // 5):
        subj = random.choice(subjects)
        verb1 = random.choice(verbs_present)
        verb2 = random.choice(verbs_future)
        text = f"If {subj} {verb1} and the weather is good, then {subj} {verb2}."
        data.append({
            'text': text,
            'type': 'complex_conditional',
            'has_conditional': True
        })
    
    # Generate control sentences (2 * n_samples // 5)
    for _ in range(2 * n_samples // 5):
        subj = random.choice(subjects)
        verb = random.choice(verbs_present)
        obj = random.choice(objects)
        templates = [
            f"{subj} {verb} {obj}.",
            f"{subj} {verb} and enjoys it.",
            f"Because of the weather, {subj} {verb}.",
            f"{subj} always {verb} on weekdays.",
        ]
        text = random.choice(templates)
        data.append({
            'text': text,
            'type': 'control',
            'has_conditional': False
        })
    
    # Shuffle data
    random.shuffle(data)
    return data

def save_csv(data, filepath):
    """Save data to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'type', 'has_conditional'])
        writer.writeheader()
        writer.writerows(data)

def main():
    print("=" * 60)
    print("Data Preparation for SAE Conditional Analysis")
    print("=" * 60)
    
    # Load config
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data directory
    data_dir = Path(config['paths']['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate main dataset
    print("\n1. Generating synthetic dataset...")
    n_samples = config['data']['n_samples']
    data = generate_conditionals(n_samples)
    
    # Count types
    conditionals = sum(1 for d in data if d['has_conditional'])
    controls = sum(1 for d in data if not d['has_conditional'])
    
    print(f"   Generated {len(data)} sentences")
    print(f"   - Conditionals: {conditionals}")
    print(f"   - Controls: {controls}")
    
    # Save main dataset
    main_path = data_dir / 'conditionals_dataset.csv'
    save_csv(data, main_path)
    print(f"   ✓ Saved to: {main_path}")
    
    # Create test set
    print("\n2. Creating inference test set...")
    test_data = [
        {'premise': 'If it rains, then the ground is wet.',
         'fact': 'It rains.',
         'conclusion': 'The ground is wet.',
         'valid': True,
         'type': 'modus_ponens'},
        {'premise': 'If the light is on, then someone is home.',
         'fact': 'The light is on.',
         'conclusion': 'Someone is home.',
         'valid': True,
         'type': 'modus_ponens'},
        {'premise': 'If it rains, then the ground is wet.',
         'fact': 'The ground is wet.',
         'conclusion': 'It rains.',
         'valid': False,
         'type': 'affirming_consequent'},
        {'premise': 'If it rains, then the ground is wet.',
         'fact': 'The ground is not wet.',
         'conclusion': 'It does not rain.',
         'valid': True,
         'type': 'modus_tollens'},
    ]
    
    test_path = data_dir / 'inference_test_set.csv'
    with open(test_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['premise', 'fact', 'conclusion', 'valid', 'type'])
        writer.writeheader()
        writer.writerows(test_data)
    print(f"   ✓ Saved {len(test_data)} test cases to: {test_path}")
    
    # Display samples
    print("\n3. Sample sentences:")
    print("-" * 40)
    for i in range(min(5, len(data))):
        print(f"{i+1}. [{data[i]['type']}]")
        print(f"   {data[i]['text']}")
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("\nNext steps:")
    print("1. Review data in data/processed/")
    print("2. git add data/processed/*.csv")
    print("3. git commit -m 'Add prepared datasets'")
    print("4. git push")
    print("5. Upload notebooks/02_sae_training.ipynb to Colab")
    print("=" * 60)

if __name__ == "__main__":
    main()