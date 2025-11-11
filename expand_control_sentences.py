#!/usr/bin/env python
"""
Expand control sentences to better balance the dataset.
Target: ~200 control sentences to match other categories.
"""

import csv
import random
from pathlib import Path

# More control sentences (no conditionals, no universals)
additional_control_sentences = [
    # Daily activities
    "She drinks coffee in the morning.",
    "The students walk to class together.",
    "He reads the newspaper at breakfast.",
    "The family watches television after dinner.",
    "Workers arrive at the office early.",
    "Children eat lunch in the cafeteria.",
    "The teacher writes on the whiteboard.",
    "Passengers board the airplane quietly.",
    "The chef tastes the soup carefully.",
    "Shoppers browse through the aisles.",
    
    # Descriptions
    "The mountain towers above the valley.",
    "Fresh snow covers the ground.",
    "The ocean stretches to the horizon.",
    "Ancient trees line the pathway.",
    "The building stands fifty stories tall.",
    "Colorful paintings decorate the walls.",
    "The bridge spans the wide river.",
    "Dark clouds gather overhead.",
    "The fountain sparkles in sunlight.",
    "Old books fill the shelves.",
    
    # Actions and events
    "The company announced record profits.",
    "Scientists discovered a new species.",
    "The artist unveiled her masterpiece.",
    "Athletes competed in the tournament.",
    "The orchestra performed brilliantly.",
    "Volunteers cleaned the beach yesterday.",
    "The committee approved the proposal.",
    "Engineers designed the new bridge.",
    "The jury reached a verdict.",
    "Protesters marched through downtown.",
    
    # Nature and environment
    "Lightning flashes across the sky.",
    "Waves roll onto the beach.",
    "The volcano erupted last month.",
    "Autumn leaves cover the ground.",
    "The glacier moves slowly downhill.",
    "Desert sand shifts with wind.",
    "The forest grows dense and dark.",
    "Morning dew covers the grass.",
    "The canyon echoes with sound.",
    "Wildflowers bloom along the trail.",
    
    # Technology and modern life
    "The smartphone battery drains quickly.",
    "Emails arrive throughout the day.",
    "The website loads in seconds.",
    "Social media connects people globally.",
    "The printer jams occasionally.",
    "Software updates install automatically.",
    "The laptop overheats during use.",
    "Video calls replaced many meetings.",
    "The router needs resetting sometimes.",
    "Digital cameras capture sharp images.",
    
    # Business and work
    "The meeting starts in ten minutes.",
    "Sales increased last quarter.",
    "The deadline approaches quickly.",
    "Employees receive annual bonuses.",
    "The presentation went smoothly.",
    "Customers complain about delays.",
    "The budget exceeds projections.",
    "Management restructured the department.",
    "The project finished ahead of schedule.",
    "Stock prices fluctuate daily.",
    
    # Education and learning
    "Students submit assignments online.",
    "The professor lectures twice weekly.",
    "Exams begin next Monday.",
    "The library opens at eight.",
    "Textbooks cost hundreds of dollars.",
    "Graduation happens in May.",
    "Research continues in the lab.",
    "The seminar attracted many attendees.",
    "Homework takes several hours.",
    "The course covers advanced topics.",
    
    # Entertainment and leisure
    "The movie received mixed reviews.",
    "Fans cheered for their team.",
    "The band played three encores.",
    "The museum features ancient artifacts.",
    "Tourists photograph the monument.",
    "The play runs through Sunday.",
    "The festival attracts thousands yearly.",
    "The game ended in overtime.",
    "The exhibit closes next week.",
    "The concert hall seats thousands.",
    
    # Food and dining
    "The restaurant serves Italian cuisine.",
    "Fresh bread smells wonderful.",
    "The waiter brings the check.",
    "Dinner reservations filled quickly.",
    "The menu changes seasonally.",
    "Coffee costs five dollars.",
    "The kitchen closes at nine.",
    "Dessert arrives after the meal.",
    "The bar offers craft cocktails.",
    "Brunch includes unlimited mimosas.",
    
    # Transportation
    "The subway runs underground.",
    "Traffic moves slowly downtown.",
    "The ferry crosses the harbor.",
    "Taxis wait outside the airport.",
    "The highway extends for miles.",
    "Buses stop at marked locations.",
    "The train departs on schedule.",
    "Bicycles share the road.",
    "The parking lot fills quickly.",
    "Rush hour begins at five.",
    
    # Health and wellness
    "The gym opens before dawn.",
    "Doctors recommend regular exercise.",
    "The pharmacy stocks medications.",
    "Nurses work long shifts.",
    "The hospital treats emergencies.",
    "Vitamins support good health.",
    "The clinic offers vaccinations.",
    "Meditation reduces stress levels.",
    "The therapist listens carefully.",
    "Surgery requires careful preparation.",
    
    # Weather and seasons
    "Temperature drops at night.",
    "Humidity makes summers uncomfortable.",
    "The forecast predicts sunshine.",
    "Winter brings shorter days.",
    "Spring flowers bloom early.",
    "The drought continues for months.",
    "Storms develop over the ocean.",
    "Frost forms on windows.",
    "The heatwave broke records.",
    "Fog reduces visibility significantly."
]

def main():
    # Read existing complete dataset
    input_file = Path('data/processed/quantifier_conditional_complete.csv')
    output_file = Path('data/processed/quantifier_conditional_balanced.csv')
    
    existing_data = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_data.append(row)
    
    # Add additional control sentences
    for sentence in additional_control_sentences:
        existing_data.append({
            'text': sentence,
            'type': 'control',
            'has_quantifier': 'False',
            'quantifier_type': 'none'
        })
    
    # Shuffle the complete dataset
    random.seed(42)  # For reproducibility
    random.shuffle(existing_data)
    
    # Write balanced dataset
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
    
    print("=" * 70)
    print("BALANCED DATASET STATISTICS")
    print("=" * 70)
    print(f"Total sentences: {len(existing_data)}")
    print("\nBreakdown by type:")
    
    # Group counts for better display
    conditionals = type_counts.get('conditional', 0)
    controls = type_counts.get('control', 0)
    quantifiers = sum(v for k, v in type_counts.items() if 'universal' in k or k == 'any_universal')
    
    print(f"\n  MAIN CATEGORIES:")
    print(f"  • Conditionals:  {conditionals:3d} sentences")
    print(f"  • Quantifiers:   {quantifiers:3d} sentences (all types)")
    print(f"  • Controls:      {controls:3d} sentences")
    
    print(f"\n  QUANTIFIER SUBTYPES:")
    for t, count in sorted(type_counts.items()):
        if 'universal' in t:
            print(f"    - {t}: {count}")
    
    print(f"\n📁 Saved to: {output_file}")
    print("\n✅ Dataset is now well-balanced for comparison!")
    print("=" * 70)

if __name__ == "__main__":
    main()