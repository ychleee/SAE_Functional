#!/usr/bin/env python
"""Clean up the improved dataset to remove numbering and extra text."""

import csv
import re

def clean_sentence(text):
    """Remove numbering and clean up sentence."""
    # Remove leading numbers like "10. " or "1. "
    text = re.sub(r'^\d+\.\s+', '', text)
    
    # Skip meta-text lines
    if text.startswith("Here are") or text.startswith("Generate"):
        return None
    
    # Clean up extra spaces
    text = ' '.join(text.split())
    
    return text.strip()

# Read the dataset
input_file = 'data/processed/conditionals_dataset_improved.csv'
output_file = 'data/processed/conditionals_dataset_clean.csv'

cleaned_data = []

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cleaned_text = clean_sentence(row['text'])
        if cleaned_text and len(cleaned_text) > 10:  # Skip very short or None
            cleaned_data.append({
                'text': cleaned_text,
                'type': row['type'],
                'has_conditional': row['has_conditional']
            })

# Write cleaned data
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['text', 'type', 'has_conditional'])
    writer.writeheader()
    writer.writerows(cleaned_data)

print(f"Cleaned dataset saved to {output_file}")
print(f"Total sentences: {len(cleaned_data)}")

# Show samples
print("\nSample cleaned sentences:")
print("-" * 40)
for i in range(min(5, len(cleaned_data))):
    print(f"{i+1}. [{cleaned_data[i]['type']}] {cleaned_data[i]['text'][:80]}...")