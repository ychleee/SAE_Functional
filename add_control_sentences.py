#!/usr/bin/env python
"""
Add control sentences to the existing dataset.
Quick fix for missing control sentences due to API rate limiting.
"""

import csv
import random
from pathlib import Path

# Control sentences (no conditionals, no universals)
control_sentences = [
    "The museum opens at nine o'clock every morning.",
    "Sarah completed her assignment yesterday afternoon.",
    "The coffee shop serves excellent pastries.",
    "Rain is expected this afternoon.",
    "The train arrives at platform three.",
    "John walks to work most days.",
    "The library closes at eight tonight.",
    "Fresh vegetables are available at the market.",
    "The meeting lasted two hours.",
    "She enjoys reading mystery novels.",
    "The park is crowded on weekends.",
    "Tom plays guitar in a local band.",
    "The restaurant received excellent reviews.",
    "Students study in the library.",
    "The cat sleeps on the windowsill.",
    "Traffic is heavy during rush hour.",
    "The movie starts at seven thirty.",
    "Birds sing in the morning.",
    "The store offers a discount today.",
    "Children play in the playground.",
    "The office building has ten floors.",
    "She teaches mathematics at the university.",
    "The bus runs every fifteen minutes.",
    "Flowers bloom in the garden.",
    "The doctor sees patients until five.",
    "Music plays softly in the background.",
    "The bakery sells fresh bread daily.",
    "He jogs three miles each morning.",
    "The concert sold out quickly.",
    "Snow covered the mountain peaks.",
    "The phone rang several times.",
    "She speaks three languages fluently.",
    "The team won yesterday's game.",
    "Leaves fall from trees in autumn.",
    "The river flows through the valley.",
    "Stars shine brightly at night.",
    "The chef prepares meals expertly.",
    "Dogs bark at strangers sometimes.",
    "The clock strikes twelve at noon.",
    "Paint dries slowly in humidity.",
    "The sun sets behind the mountains.",
    "Fish swim in the aquarium.",
    "The artist displays her paintings downtown.",
    "Wind blows through the trees.",
    "The baby sleeps peacefully.",
    "Customers wait patiently in line.",
    "The television broadcasts news hourly.",
    "Waves crash against the shore.",
    "The teacher explains concepts clearly.",
    "Clouds drift across the sky.",
    "The mailman delivers letters daily.",
    "Birds migrate south for winter.",
    "The computer processes data quickly.",
    "Rain falls gently on the roof.",
    "The singer performs on stage tonight.",
    "Trees provide shade in summer.",
    "The mechanic repairs cars efficiently.",
    "Sunlight streams through the window.",
    "The farmer harvests crops in fall.",
    "Music echoes through the hall.",
    "The plane lands at the airport.",
    "Grass grows quickly after rain.",
    "The nurse checks patients regularly.",
    "Thunder rumbles in the distance.",
    "The writer works on her novel.",
    "Ice melts in warm weather.",
    "The student takes notes carefully.",
    "Fog covers the city morning.",
    "The electrician fixes the wiring.",
    "Butterflies flutter in the garden.",
    "The lawyer reviews the contract.",
    "Steam rises from hot coffee.",
    "The athlete trains every day.",
    "Moonlight illuminates the path.",
    "The cashier counts the money.",
    "Bees buzz around the flowers.",
    "The architect designs modern buildings.",
    "Dust settles on unused shelves.",
    "The pilot flies international routes.",
    "Shadows lengthen at sunset.",
    "The dentist examines teeth carefully."
]

def main():
    # Read existing dataset
    input_file = Path('data/processed/quantifier_conditional_dataset.csv')
    output_file = Path('data/processed/quantifier_conditional_complete.csv')
    
    existing_data = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_data.append(row)
    
    # Add control sentences
    for sentence in control_sentences:
        existing_data.append({
            'text': sentence,
            'type': 'control',
            'has_quantifier': 'False',
            'quantifier_type': 'none'
        })
    
    # Shuffle the complete dataset
    random.shuffle(existing_data)
    
    # Write complete dataset
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['text', 'type', 'has_quantifier', 'quantifier_type']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)
    
    # Statistics
    type_counts = {}
    for row in existing_data:
        t = row['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("=" * 60)
    print("COMPLETE DATASET STATISTICS")
    print("=" * 60)
    print(f"Total sentences: {len(existing_data)}")
    print("\nBreakdown by type:")
    for t, count in sorted(type_counts.items()):
        print(f"  • {t}: {count}")
    print(f"\nSaved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()