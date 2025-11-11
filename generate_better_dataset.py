#!/usr/bin/env python
"""
Generate high-quality conditional dataset using Claude API.
Creates natural, grammatically correct sentences for SAE training.
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import List, Dict
import random

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

def generate_conditionals_batch(client, category: str, n: int = 10) -> List[str]:
    """Generate a batch of conditional sentences for a specific category."""
    
    prompts = {
        'causal': """Generate {n} realistic conditional sentences about cause and effect.
Format: Each sentence should use "If [cause], then [effect]" structure.
Examples:
- If you water plants regularly, then they grow healthy.
- If the battery dies, then the phone stops working.

Generate {n} different sentences with varied subjects and natural language:""",
        
        'temporal': """Generate {n} realistic conditional sentences about time and events.
Format: Each sentence should use "If [time/event], then [action/consequence]" structure.
Examples:
- If the meeting ends early, then we can grab lunch.
- If it's past midnight, then the stores are closed.

Generate {n} different sentences with varied scenarios:""",
        
        'social': """Generate {n} realistic conditional sentences about social situations.
Format: Each sentence should use "If [social situation], then [response/behavior]" structure.
Examples:
- If someone helps you, then you should thank them.
- If you're running late, then notify the host.

Generate {n} different sentences about social norms and interactions:""",
        
        'predictive': """Generate {n} realistic conditional sentences about predictions and likelihood.
Format: Each sentence should use "If [condition], then [likely outcome]" structure.
Examples:
- If interest rates rise, then housing prices typically fall.
- If you study hard, then you're likely to pass the exam.

Generate {n} different predictive sentences:""",
        
        'logical': """Generate {n} conditional sentences about logical relationships.
Format: Each sentence should use "If [premise], then [logical conclusion]" structure.
Examples:
- If all mammals are warm-blooded and whales are mammals, then whales are warm-blooded.
- If the light is on, then electricity is flowing.

Generate {n} different logical conditionals:"""
    }
    
    prompt = prompts[category].format(n=n)
    
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse the response
    lines = message.content[0].text.strip().split('\n')
    sentences = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    # Clean up numbering if present
    cleaned = []
    for s in sentences:
        # Remove numbering like "1. " or "- "
        if s[0].isdigit() and s[1:3] == '. ':
            s = s[3:]
        elif s.startswith('- '):
            s = s[2:]
        cleaned.append(s.strip())
    
    return cleaned[:n]

def generate_control_sentences(client, n: int = 20) -> List[str]:
    """Generate control sentences without conditionals."""
    
    prompt = f"""Generate {n} simple declarative sentences about everyday activities and observations.
Do NOT use conditional structures (no if-then).
Include variety: actions, descriptions, habitual activities, and observations.

Examples:
- The coffee shop opens at seven in the morning.
- Sarah enjoys reading mystery novels.
- The train arrives every fifteen minutes.
- Most birds migrate during winter months.

Generate {n} different sentences:"""
    
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    lines = message.content[0].text.strip().split('\n')
    sentences = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Remove numbering
        if line[0].isdigit() and line[1:3] == '. ':
            line = line[3:]
        elif line.startswith('- '):
            line = line[2:]
        
        # Make sure it's not a conditional
        if 'if' not in line.lower():
            sentences.append(line.strip())
    
    return sentences[:n]

def main():
    """Generate complete dataset using Claude API."""
    print("=" * 60)
    print("Generating High-Quality Conditional Dataset")
    print("=" * 60)
    
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
    
    # Generate sentences
    all_sentences = []
    
    print("\nGenerating conditional sentences...")
    categories = ['causal', 'temporal', 'social', 'predictive', 'logical']
    
    for category in categories:
        print(f"  Generating {category} conditionals...")
        try:
            sentences = generate_conditionals_batch(client, category, n=20)
            for s in sentences:
                all_sentences.append({
                    'text': s,
                    'type': f'{category}_conditional',
                    'has_conditional': True
                })
            print(f"    ✓ Generated {len(sentences)} {category} conditionals")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print("\nGenerating control sentences...")
    try:
        # Generate multiple batches of controls
        for i in range(5):
            controls = generate_control_sentences(client, n=20)
            for s in controls:
                all_sentences.append({
                    'text': s,
                    'type': 'control',
                    'has_conditional': False
                })
            print(f"  ✓ Generated batch {i+1} ({len(controls)} sentences)")
            time.sleep(1)
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Shuffle the dataset
    random.shuffle(all_sentences)
    
    # Save to CSV
    output_path = Path('data/processed/conditionals_dataset_improved.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'type', 'has_conditional'])
        writer.writeheader()
        writer.writerows(all_sentences)
    
    print(f"\n✓ Saved {len(all_sentences)} sentences to {output_path}")
    
    # Display samples
    print("\nSample sentences from new dataset:")
    print("-" * 40)
    
    # Show some conditionals
    conditionals = [s for s in all_sentences if s['has_conditional']][:3]
    for s in conditionals:
        print(f"[{s['type']}] {s['text']}")
    
    print()
    
    # Show some controls
    controls = [s for s in all_sentences if not s['has_conditional']][:3]
    for s in controls:
        print(f"[{s['type']}] {s['text']}")
    
    # Statistics
    n_cond = sum(1 for s in all_sentences if s['has_conditional'])
    n_ctrl = len(all_sentences) - n_cond
    
    print("\n" + "=" * 60)
    print(f"✅ Dataset generation complete!")
    print(f"  Total sentences: {len(all_sentences)}")
    print(f"  Conditionals: {n_cond}")
    print(f"  Controls: {n_ctrl}")
    print(f"  File: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()