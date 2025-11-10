"""
Data utilities for creating and processing conditional datasets.
"""

import random
from typing import List, Dict, Tuple
import pandas as pd
import json


class ConditionalDatasetGenerator:
    """Generate synthetic conditional sentences for SAE training."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # Template components
        self.subjects = ["John", "Mary", "The cat", "The student", "The team", "She", "He", "They"]
        self.verbs_present = ["runs", "reads", "works", "studies", "plays", "sings", "writes", "thinks"]
        self.verbs_future = ["will run", "will read", "will work", "will study", "will play", "will sing"]
        self.objects = ["the book", "quickly", "hard", "the game", "a song", "carefully", "the report", "deeply"]
        self.weather_conditions = ["rains", "snows", "is sunny", "is cloudy", "is windy"]
        self.consequences = ["the picnic is canceled", "we stay inside", "we go to the beach", 
                           "the game continues", "school closes", "traffic increases"]
        
    def generate_simple_conditional(self) -> str:
        """Generate a simple if-then conditional."""
        templates = [
            lambda: f"If it {random.choice(self.weather_conditions)}, then {random.choice(self.consequences)}.",
            lambda: f"If {random.choice(self.subjects)} {random.choice(self.verbs_present)} {random.choice(self.objects)}, then everyone is happy.",
            lambda: f"If the test is hard, then {random.choice(self.subjects)} {random.choice(self.verbs_future)} more.",
        ]
        return random.choice(templates)()
    
    def generate_logical_conditional(self) -> str:
        """Generate conditionals with logical connectives."""
        A = random.choice(["A", "P", "X", "the statement"])
        B = random.choice(["B", "Q", "Y", "the conclusion"]) 
        
        templates = [
            f"If {A} is true, then {B} is true.",
            f"If {A} and not {B}, then contradiction.",
            f"If {A} or {B}, then at least one holds.",
            f"If not {A}, then {B} must be false.",
            f"If {A} implies {B}, and {A} is true, then {B}.",
        ]
        return random.choice(templates)
    
    def generate_complex_conditional(self) -> str:
        """Generate more complex nested or compound conditionals."""
        subj = random.choice(self.subjects)
        verb1 = random.choice(self.verbs_present)
        verb2 = random.choice(self.verbs_future)
        
        templates = [
            f"If {subj} {verb1} and the weather is good, then {subj} {verb2}.",
            f"If either {subj} {verb1} or it {random.choice(self.weather_conditions)}, then plans change.",
            f"If {subj} {verb1} because of the deadline, then the project succeeds.",
            f"Although {subj} {verb1}, if conditions change, then {subj} {verb2}.",
        ]
        return random.choice(templates)
    
    def generate_non_conditional(self) -> str:
        """Generate control sentences without conditionals."""
        subj = random.choice(self.subjects)
        verb = random.choice(self.verbs_present)
        obj = random.choice(self.objects)
        
        templates = [
            f"{subj} {verb} {obj}.",
            f"{subj} {verb} and enjoys it.",
            f"Because of the weather, {subj} {verb}.",
            f"{subj} always {verb} on weekdays.",
            f"Everyone knows that {subj} {verb}.",
        ]
        return random.choice(templates)
    
    def generate_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate a balanced dataset of conditionals and controls."""
        data = []
        
        # Generate different types of sentences
        n_per_type = n_samples // 5
        
        for _ in range(n_per_type):
            data.append({
                'text': self.generate_simple_conditional(),
                'type': 'simple_conditional',
                'has_conditional': True
            })
        
        for _ in range(n_per_type):
            data.append({
                'text': self.generate_logical_conditional(),
                'type': 'logical_conditional', 
                'has_conditional': True
            })
            
        for _ in range(n_per_type):
            data.append({
                'text': self.generate_complex_conditional(),
                'type': 'complex_conditional',
                'has_conditional': True
            })
            
        for _ in range(n_per_type * 2):  # More control sentences
            data.append({
                'text': self.generate_non_conditional(),
                'type': 'control',
                'has_conditional': False
            })
        
        # Shuffle the data
        random.shuffle(data)
        
        return pd.DataFrame(data)
    

def create_minimal_test_set() -> pd.DataFrame:
    """Create a minimal test set for modus ponens and basic inference."""
    test_cases = [
        # Modus ponens tests
        {"premise": "If it rains, then the ground is wet.", 
         "fact": "It rains.",
         "conclusion": "The ground is wet.",
         "valid": True,
         "type": "modus_ponens"},
        
        {"premise": "If the light is on, then someone is home.",
         "fact": "The light is on.", 
         "conclusion": "Someone is home.",
         "valid": True,
         "type": "modus_ponens"},
        
        # Invalid inferences
        {"premise": "If it rains, then the ground is wet.",
         "fact": "The ground is wet.",
         "conclusion": "It rains.", 
         "valid": False,
         "type": "affirming_consequent"},
        
        # Modus tollens
        {"premise": "If it rains, then the ground is wet.",
         "fact": "The ground is not wet.",
         "conclusion": "It does not rain.",
         "valid": True,
         "type": "modus_tollens"},
    ]
    
    return pd.DataFrame(test_cases)


def save_dataset(df: pd.DataFrame, filepath: str):
    """Save dataset to CSV or JSON."""
    if filepath.endswith('.csv'):
        df.to_csv(filepath, index=False)
    elif filepath.endswith('.json'):
        df.to_json(filepath, orient='records', indent=2)
    else:
        raise ValueError("Filepath must end with .csv or .json")


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV or JSON."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath)
    else:
        raise ValueError("Filepath must end with .csv or .json")


if __name__ == "__main__":
    # Test the generator
    generator = ConditionalDatasetGenerator()
    
    # Generate a small sample
    df = generator.generate_dataset(n_samples=100)
    print(f"Generated {len(df)} sentences")
    print(f"Conditionals: {df['has_conditional'].sum()}")
    print(f"Controls: {(~df['has_conditional']).sum()}")
    print("\nSample sentences:")
    for _, row in df.head(5).iterrows():
        print(f"[{row['type']}] {row['text']}")