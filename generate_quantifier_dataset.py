#!/usr/bin/env python
"""
Generate dataset comparing universal quantifiers with conditionals.
Tests the hypothesis that universal quantifiers encode implicit conditionals.
"""

import os
import sys
import csv
import time
import random
from pathlib import Path
from typing import List, Dict

# Add path to read env file
sys.path.append('/Users/leeyoungchan/development/0chanly')

def load_api_key():
    """Load Anthropic API key from 0chanly project."""
    env_path = '/Users/leeyoungchan/development/0chanly/.env.local'
    
    with open(env_path, 'r') as f:
        for line in f:
            if 'ANTHROPIC_API_KEY' in line:
                return line.split('=')[1].strip()
    
    raise ValueError("Could not find ANTHROPIC_API_KEY")

def generate_universals_batch(client, category: str, n: int = 20) -> List[Dict]:
    """Generate universal quantifier sentences."""
    
    prompts = {
        'pure_universal': """Generate {n} sentences with universal quantifiers (all, every, each).
Format: Use "all/every/each" to make universal claims.
Examples:
- All mammals breathe air
- Every prime number greater than 2 is odd
- Each student received a textbook

Generate {n} different universal statements about various topics:""",
        
        'restricted_universal': """Generate {n} sentences with restricted universal quantifiers.
Format: Universal claims about specific groups or conditions.
Examples:
- All students in the advanced class scored above 90%
- Every car in the parking lot has a permit
- Each book on the shelf is categorized by genre

Generate {n} different restricted universal statements:""",
        
        'negative_universal': """Generate {n} sentences with negative universal quantifiers (no, none, nothing).
Format: Universal negative claims.
Examples:
- No reptiles have fur
- None of the answers were correct
- Nothing travels faster than light

Generate {n} different negative universal statements:""",
        
        'generic_universal': """Generate {n} generic statements that imply universality without explicit quantifiers.
Format: General statements about categories that imply "all".
Examples:
- Birds have wings
- Water freezes at 0 degrees Celsius
- Triangles have three sides

Generate {n} different generic universal statements:""",
        
        'any_universal': """Generate {n} sentences using "any" as a universal quantifier.
Format: Statements using "any" to express universality.
Examples:
- Any number multiplied by zero equals zero
- Any student can access the library
- Any metal conducts electricity

Generate {n} different statements using 'any':"""
    }
    
    prompt = prompts[category].format(n=n)
    
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1500,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse response
    lines = message.content[0].text.strip().split('\n')
    sentences = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Generate') or line.startswith('Here'):
            continue
        
        # Remove numbering
        if line[0].isdigit() and '. ' in line[:4]:
            line = line.split('. ', 1)[1] if '. ' in line else line
        elif line.startswith('- '):
            line = line[2:]
        
        # Skip if it's not a proper sentence
        if len(line) > 10 and not line.startswith('Format'):
            sentences.append({
                'text': line.strip(),
                'type': category,
                'has_quantifier': True,
                'quantifier_type': category.replace('_', ' ')
            })
    
    return sentences[:n]

def generate_conditionals_batch(client, n: int = 20) -> List[Dict]:
    """Generate conditional sentences for comparison."""
    
    prompt = f"""Generate {n} conditional sentences using if-then structure.
Mix different types: causal, temporal, logical, predictive.
Examples:
- If water reaches 100°C, then it boils
- If you practice daily, then you improve
- If a number is even, then it's divisible by 2

Generate {n} varied conditional sentences:"""
    
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1500,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    lines = message.content[0].text.strip().split('\n')
    sentences = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Generate'):
            continue
        
        # Remove numbering
        if line[0].isdigit() and '. ' in line[:4]:
            line = line.split('. ', 1)[1]
        elif line.startswith('- '):
            line = line[2:]
        
        if 'if' in line.lower() and len(line) > 10:
            sentences.append({
                'text': line.strip(),
                'type': 'conditional',
                'has_quantifier': False,
                'quantifier_type': 'none'
            })
    
    return sentences[:n]

def generate_control_batch(client, n: int = 20) -> List[Dict]:
    """Generate control sentences without conditionals or universals."""
    
    prompt = f"""Generate {n} simple declarative sentences.
Do NOT use conditionals (if-then) or universal quantifiers (all, every, no, any).
Include variety: descriptions, actions, facts, observations.
Examples:
- The museum opens at nine o'clock
- Sarah completed her assignment yesterday
- The coffee tastes bitter this morning
- Rain is expected this afternoon

Generate {n} different regular sentences:"""
    
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1500,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    lines = message.content[0].text.strip().split('\n')
    sentences = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Generate'):
            continue
        
        # Remove numbering
        if line[0].isdigit() and '. ' in line[:4]:
            line = line.split('. ', 1)[1]
        elif line.startswith('- '):
            line = line[2:]
        
        # Filter out any accidental conditionals or universals
        lower = line.lower()
        if ('if' not in lower and 'then' not in lower and 
            'all' not in lower and 'every' not in lower and
            'each' not in lower and 'any' not in lower and
            'no ' not in lower and 'none' not in lower and
            len(line) > 10):
            
            sentences.append({
                'text': line.strip(),
                'type': 'control',
                'has_quantifier': False,
                'quantifier_type': 'none'
            })
    
    return sentences[:n]

def main():
    """Generate complete dataset for quantifier-conditional comparison."""
    print("=" * 70)
    print("GENERATING QUANTIFIER-CONDITIONAL COMPARISON DATASET")
    print("=" * 70)
    
    # Load API key
    try:
        api_key = load_api_key()
        print("✓ API key loaded")
    except Exception as e:
        print(f"Error loading API key: {e}")
        return
    
    # Initialize Anthropic client
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    print("✓ Anthropic client initialized")
    
    all_sentences = []
    
    # Generate universal quantifier sentences
    print("\n" + "=" * 50)
    print("GENERATING UNIVERSAL QUANTIFIER SENTENCES")
    print("=" * 50)
    
    universal_categories = [
        'pure_universal',
        'restricted_universal', 
        'negative_universal',
        'generic_universal',
        'any_universal'
    ]
    
    for category in universal_categories:
        print(f"\n→ Generating {category.replace('_', ' ')} sentences...")
        try:
            sentences = generate_universals_batch(client, category, n=40)
            all_sentences.extend(sentences)
            print(f"  ✓ Generated {len(sentences)} {category} sentences")
            
            # Show samples
            if len(sentences) >= 2:
                print(f"  Sample: \"{sentences[0]['text'][:60]}...\"")
                print(f"  Sample: \"{sentences[1]['text'][:60]}...\"")
            
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Generate conditional sentences
    print("\n" + "=" * 50)
    print("GENERATING CONDITIONAL SENTENCES")
    print("=" * 50)
    
    print("\n→ Generating conditional sentences...")
    try:
        for i in range(10):  # 10 batches of 40 = 400 total
            conditionals = generate_conditionals_batch(client, n=40)
            all_sentences.extend(conditionals)
            print(f"  ✓ Batch {i+1}: Generated {len(conditionals)} conditionals")
            if i == 0 and len(conditionals) >= 2:
                print(f"  Sample: \"{conditionals[0]['text'][:60]}...\"")
            time.sleep(1)
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Generate control sentences
    print("\n" + "=" * 50)
    print("GENERATING CONTROL SENTENCES")
    print("=" * 50)
    
    print("\n→ Generating control sentences...")
    try:
        for i in range(10):  # 10 batches of 40 = 400 total
            controls = generate_control_batch(client, n=40)
            all_sentences.extend(controls)
            print(f"  ✓ Batch {i+1}: Generated {len(controls)} controls")
            if i == 0 and len(controls) >= 2:
                print(f"  Sample: \"{controls[0]['text'][:60]}...\"")
            time.sleep(1)
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Shuffle dataset
    random.shuffle(all_sentences)
    
    # Save to CSV
    output_path = Path('data/processed/quantifier_conditional_dataset.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['text', 'type', 'has_quantifier', 'quantifier_type']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_sentences)
    
    # Calculate statistics
    stats = {}
    for sentence in all_sentences:
        t = sentence['type']
        stats[t] = stats.get(t, 0) + 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n✅ Total sentences: {len(all_sentences)}")
    print("\nBreakdown by type:")
    for type_name, count in sorted(stats.items()):
        print(f"  • {type_name}: {count}")
    print(f"\n📁 Saved to: {output_path}")
    print("=" * 70)
    
    # Show final samples
    print("\nSample sentences from dataset:")
    print("-" * 50)
    
    # Show one of each type
    shown_types = set()
    for sentence in all_sentences[:50]:
        if sentence['type'] not in shown_types:
            print(f"[{sentence['type']}]: {sentence['text']}")
            shown_types.add(sentence['type'])
            if len(shown_types) >= len(stats):
                break

if __name__ == "__main__":
    main()