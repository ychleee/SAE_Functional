"""
Feature analysis and interpretation utilities.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


class FeatureInterpreter:
    """Interpret and analyze SAE features."""
    
    def __init__(self, sae_model, activations_dict: Dict, device: str = 'cpu'):
        """
        Initialize interpreter.
        
        Args:
            sae_model: Trained SparseAutoencoder
            activations_dict: Dictionary with activations and metadata
            device: Device for computation
        """
        self.model = sae_model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.activations = activations_dict['activations']
        self.metadata = activations_dict['metadata']
        
        # Encode all activations once
        self._encode_all()
    
    def _encode_all(self):
        """Encode all activations to sparse codes."""
        with torch.no_grad():
            acts_tensor = torch.FloatTensor(self.activations).to(self.device)
            self.sparse_codes = self.model.encode(acts_tensor).cpu().numpy()
    
    def find_feature_examples(
        self,
        feature_idx: int,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Find texts that most strongly activate a specific feature.
        
        Args:
            feature_idx: Index of feature to analyze
            top_k: Number of top examples to return
            threshold: Minimum activation threshold
            
        Returns:
            DataFrame with top activating examples
        """
        # Get activations for this feature
        feature_acts = self.sparse_codes[:, feature_idx]
        
        # Find top activating examples
        above_threshold = feature_acts > threshold
        valid_indices = np.where(above_threshold)[0]
        
        if len(valid_indices) == 0:
            return pd.DataFrame()
        
        # Sort by activation strength
        sorted_indices = valid_indices[np.argsort(feature_acts[valid_indices])[::-1]]
        top_indices = sorted_indices[:top_k]
        
        # Create results dataframe
        results = []
        for idx in top_indices:
            results.append({
                'text': self.metadata[idx]['text'],
                'activation': feature_acts[idx],
                'has_if': self.metadata[idx].get('has_if', False),
                'has_then': self.metadata[idx].get('has_then', False),
                'index': idx
            })
        
        return pd.DataFrame(results)
    
    def analyze_conditional_features(
        self,
        top_k: int = 20
    ) -> Dict:
        """
        Find features that correlate with conditional sentences.
        
        Args:
            top_k: Number of top features to identify
            
        Returns:
            Analysis results dictionary
        """
        # Create masks for different text types
        has_if = np.array([m.get('has_if', False) for m in self.metadata])
        has_then = np.array([m.get('has_then', False) for m in self.metadata])
        has_both = has_if & has_then
        
        # Calculate average activations for each group
        if_activations = self.sparse_codes[has_if].mean(axis=0)
        then_activations = self.sparse_codes[has_then].mean(axis=0)
        both_activations = self.sparse_codes[has_both].mean(axis=0)
        baseline_activations = self.sparse_codes[~has_if].mean(axis=0)
        
        # Calculate differential scores
        if_diff = if_activations - baseline_activations
        then_diff = then_activations - baseline_activations
        both_diff = both_activations - baseline_activations
        
        # Find top features
        top_if_features = np.argsort(if_diff)[-top_k:][::-1]
        top_then_features = np.argsort(then_diff)[-top_k:][::-1]
        top_both_features = np.argsort(both_diff)[-top_k:][::-1]
        
        return {
            'if_features': {
                'indices': top_if_features.tolist(),
                'scores': if_diff[top_if_features].tolist()
            },
            'then_features': {
                'indices': top_then_features.tolist(),
                'scores': then_diff[top_then_features].tolist()
            },
            'conditional_features': {
                'indices': top_both_features.tolist(),
                'scores': both_diff[top_both_features].tolist()
            },
            'statistics': {
                'n_if_texts': has_if.sum(),
                'n_then_texts': has_then.sum(),
                'n_conditional_texts': has_both.sum(),
                'n_baseline_texts': (~has_if).sum()
            }
        }
    
    def feature_correlation_matrix(
        self,
        feature_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Compute correlation matrix between features.
        
        Args:
            feature_indices: Specific features to analyze (None for all)
            
        Returns:
            Correlation matrix
        """
        if feature_indices is None:
            codes = self.sparse_codes
        else:
            codes = self.sparse_codes[:, feature_indices]
        
        # Compute correlation
        correlation = np.corrcoef(codes.T)
        return correlation
    
    def cluster_features(
        self,
        n_clusters: int = 10,
        method: str = 'kmeans'
    ) -> Dict:
        """
        Cluster features based on activation patterns.
        
        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans' or 'hierarchical')
            
        Returns:
            Clustering results
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering
        
        # Use feature activation patterns as features
        feature_patterns = self.sparse_codes.T
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        cluster_labels = clusterer.fit_predict(feature_patterns)
        
        # Analyze clusters
        clusters = {}
        for i in range(n_clusters):
            cluster_features = np.where(cluster_labels == i)[0]
            clusters[f'cluster_{i}'] = {
                'features': cluster_features.tolist(),
                'size': len(cluster_features)
            }
        
        return {
            'labels': cluster_labels,
            'clusters': clusters,
            'n_clusters': n_clusters
        }
    
    def visualize_feature_activation(
        self,
        feature_idx: int,
        n_examples: int = 100
    ) -> plt.Figure:
        """
        Visualize activation pattern of a specific feature.
        
        Args:
            feature_idx: Feature to visualize
            n_examples: Number of examples to show
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Get feature activations
        feature_acts = self.sparse_codes[:, feature_idx]
        
        # 1. Histogram of activations
        axes[0, 0].hist(feature_acts[feature_acts > 0], bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('Activation Strength')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'Feature {feature_idx} Activation Distribution')
        
        # 2. Activation by text type
        has_if = np.array([m.get('has_if', False) for m in self.metadata])
        if_acts = feature_acts[has_if]
        other_acts = feature_acts[~has_if]
        
        axes[0, 1].boxplot([if_acts[if_acts > 0] if len(if_acts[if_acts > 0]) > 0 else [0],
                           other_acts[other_acts > 0] if len(other_acts[other_acts > 0]) > 0 else [0]],
                          labels=['Conditional', 'Non-conditional'])
        axes[0, 1].set_ylabel('Activation Strength')
        axes[0, 1].set_title('Activation by Text Type')
        
        # 3. Time series of activations
        axes[1, 0].plot(feature_acts[:n_examples], alpha=0.7)
        axes[1, 0].set_xlabel('Example Index')
        axes[1, 0].set_ylabel('Activation')
        axes[1, 0].set_title(f'First {n_examples} Examples')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # 4. Co-activation with other features
        active_mask = feature_acts > 0
        if active_mask.sum() > 0:
            co_activations = self.sparse_codes[active_mask].mean(axis=0)
            top_co = np.argsort(co_activations)[-10:][::-1]
            top_co = [f for f in top_co if f != feature_idx][:5]
            
            axes[1, 1].bar(range(len(top_co)), co_activations[top_co])
            axes[1, 1].set_xticks(range(len(top_co)))
            axes[1, 1].set_xticklabels([f'F{i}' for i in top_co])
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Avg Co-activation')
            axes[1, 1].set_title('Top Co-activating Features')
        else:
            axes[1, 1].text(0.5, 0.5, 'No active examples', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.suptitle(f'Feature {feature_idx} Analysis')
        plt.tight_layout()
        
        return fig


class CausalInterventions:
    """Perform causal interventions on SAE features."""
    
    def __init__(self, sae_model, base_model, tokenizer, device: str = 'cpu'):
        """
        Initialize intervention handler.
        
        Args:
            sae_model: Trained SAE
            base_model: Original language model
            tokenizer: Model tokenizer
            device: Device for computation
        """
        self.sae = sae_model
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        
        self.sae.to(self.device)
        self.base_model.to(self.device)
        self.base_model.eval()
    
    def ablate_feature(
        self,
        text: str,
        feature_idx: int,
        layer_idx: int = -1
    ) -> Dict:
        """
        Ablate (zero out) a specific feature and observe changes.
        
        Args:
            text: Input text
            feature_idx: Feature to ablate
            layer_idx: Layer to intervene at
            
        Returns:
            Results dictionary
        """
        # Get original activations
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            # Original forward pass
            outputs_orig = self.base_model(**inputs, output_hidden_states=True)
            hidden_orig = outputs_orig.hidden_states[layer_idx]
            
            # Encode to sparse code
            sparse_code = self.sae.encode(hidden_orig)
            
            # Ablate feature
            sparse_code_ablated = sparse_code.clone()
            sparse_code_ablated[:, :, feature_idx] = 0
            
            # Decode back
            hidden_modified = self.sae.decode(sparse_code_ablated)
            
            # TODO: Run model with modified activations
            # This requires model surgery - inserting modified activations
            
        return {
            'original_activation': sparse_code[0, :, feature_idx].cpu().numpy(),
            'text': text,
            'feature_idx': feature_idx
        }
    
    def amplify_feature(
        self,
        text: str,
        feature_idx: int,
        scale_factor: float = 2.0,
        layer_idx: int = -1
    ) -> Dict:
        """
        Amplify a specific feature and observe changes.
        
        Args:
            text: Input text
            feature_idx: Feature to amplify
            scale_factor: How much to scale the feature
            layer_idx: Layer to intervene at
            
        Returns:
            Results dictionary
        """
        # Similar to ablate but multiply instead of zero
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx]
            
            sparse_code = self.sae.encode(hidden)
            sparse_code_amplified = sparse_code.clone()
            sparse_code_amplified[:, :, feature_idx] *= scale_factor
            
            hidden_modified = self.sae.decode(sparse_code_amplified)
        
        return {
            'original_activation': sparse_code[0, :, feature_idx].cpu().numpy(),
            'amplified_activation': sparse_code_amplified[0, :, feature_idx].cpu().numpy(),
            'text': text,
            'feature_idx': feature_idx,
            'scale_factor': scale_factor
        }


def create_feature_report(
    interpreter: FeatureInterpreter,
    feature_idx: int,
    n_examples: int = 5
) -> str:
    """
    Create a text report for a specific feature.
    
    Args:
        interpreter: FeatureInterpreter instance
        feature_idx: Feature to analyze
        n_examples: Number of examples to include
        
    Returns:
        Formatted report string
    """
    examples_df = interpreter.find_feature_examples(feature_idx, top_k=n_examples)
    
    report = f"=" * 60 + "\n"
    report += f"Feature {feature_idx} Analysis Report\n"
    report += f"=" * 60 + "\n\n"
    
    if len(examples_df) > 0:
        report += f"Top {n_examples} Activating Examples:\n"
        report += "-" * 40 + "\n"
        
        for i, row in examples_df.iterrows():
            report += f"\n{i+1}. Activation: {row['activation']:.3f}\n"
            report += f"   Text: {row['text']}\n"
            report += f"   Has 'if': {row['has_if']}, Has 'then': {row['has_then']}\n"
    else:
        report += "No examples found with activation above threshold.\n"
    
    # Get statistics
    feature_acts = interpreter.sparse_codes[:, feature_idx]
    active_count = (feature_acts > 0).sum()
    
    report += f"\n" + "=" * 60 + "\n"
    report += f"Statistics:\n"
    report += f"- Active in {active_count}/{len(feature_acts)} examples ({100*active_count/len(feature_acts):.1f}%)\n"
    report += f"- Mean activation (when active): {feature_acts[feature_acts > 0].mean():.3f}\n"
    report += f"- Max activation: {feature_acts.max():.3f}\n"
    
    return report


if __name__ == "__main__":
    print("Feature analysis module loaded successfully")