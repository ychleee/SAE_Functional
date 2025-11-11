#!/usr/bin/env python
"""
Analyze feature overlap between conditional sentences and universal quantifiers.
Tests hypothesis that they share underlying logical features.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class FeatureOverlapAnalyzer:
    """Analyze overlap between conditional and quantifier features."""
    
    def __init__(self, sae_model, device='cpu'):
        """
        Initialize analyzer.
        
        Args:
            sae_model: Trained Sparse Autoencoder
            device: Device for computation
        """
        self.sae = sae_model
        self.device = device
        self.sae.to(device)
        self.sae.eval()
        
        self.features_conditional = None
        self.features_quantifier = None
        self.features_control = None
    
    def encode_sentences(self, activations, labels):
        """
        Encode activations to sparse features.
        
        Args:
            activations: Neural network activations
            labels: Sentence type labels
            
        Returns:
            Dictionary of sparse codes by type
        """
        with torch.no_grad():
            acts_tensor = torch.FloatTensor(activations).to(self.device)
            sparse_codes = self.sae.encode(acts_tensor).cpu().numpy()
        
        # Replace any NaN values with 0
        sparse_codes = np.nan_to_num(sparse_codes, nan=0.0)
        
        # Separate by type
        cond_mask = labels == 'conditional'
        quant_mask = np.isin(labels, [
            'pure_universal', 'restricted_universal', 
            'negative_universal', 'generic_universal', 'any_universal'
        ])
        ctrl_mask = labels == 'control'
        
        codes_by_type = {
            'conditional': sparse_codes[cond_mask] if np.any(cond_mask) else np.array([]),
            'quantifier': sparse_codes[quant_mask] if np.any(quant_mask) else np.array([]),
            'control': sparse_codes[ctrl_mask] if np.any(ctrl_mask) else np.array([])
        }
        
        # Ensure we have at least empty arrays with correct shape
        for key in codes_by_type:
            if codes_by_type[key].size == 0:
                codes_by_type[key] = np.zeros((0, sparse_codes.shape[1]))
        
        return codes_by_type, sparse_codes
    
    def identify_characteristic_features(self, codes_by_type, threshold=0.1):
        """
        Identify features characteristic of each sentence type.
        
        Args:
            codes_by_type: Dictionary of sparse codes by type
            threshold: Minimum activation difference to consider significant
            
        Returns:
            Dictionary of characteristic features
        """
        # Calculate mean activations for each type, handling empty arrays
        if len(codes_by_type['conditional']) > 0:
            mean_conditional = np.nan_to_num(codes_by_type['conditional'].mean(axis=0), nan=0.0)
        else:
            mean_conditional = np.zeros(codes_by_type['conditional'].shape[1] if codes_by_type['conditional'].size > 0 else 0)
            
        if len(codes_by_type['quantifier']) > 0:
            mean_quantifier = np.nan_to_num(codes_by_type['quantifier'].mean(axis=0), nan=0.0)
        else:
            mean_quantifier = np.zeros(codes_by_type['quantifier'].shape[1] if codes_by_type['quantifier'].size > 0 else 0)
            
        if len(codes_by_type['control']) > 0:
            mean_control = np.nan_to_num(codes_by_type['control'].mean(axis=0), nan=0.0)
        else:
            mean_control = np.zeros(codes_by_type['control'].shape[1] if codes_by_type['control'].size > 0 else 0)
        
        # Identify characteristic features (higher than control)
        cond_diff = mean_conditional - mean_control
        quant_diff = mean_quantifier - mean_control
        
        # Find top features for each type
        features = {
            'conditional_specific': np.where(
                (cond_diff > threshold) & (cond_diff > quant_diff)
            )[0],
            'quantifier_specific': np.where(
                (quant_diff > threshold) & (quant_diff > cond_diff)
            )[0],
            'shared': np.where(
                (cond_diff > threshold) & (quant_diff > threshold) &
                (np.abs(cond_diff - quant_diff) < threshold)
            )[0],
            'control': np.where(
                (mean_control > mean_conditional) & 
                (mean_control > mean_quantifier)
            )[0]
        }
        
        # Calculate overlap statistics
        features['overlap_stats'] = {
            'jaccard_index': len(features['shared']) / (
                len(features['conditional_specific']) + 
                len(features['quantifier_specific']) + 
                len(features['shared'])
            ) if (len(features['conditional_specific']) + 
                  len(features['quantifier_specific']) + 
                  len(features['shared'])) > 0 else 0,
            'conditional_mean': mean_conditional,
            'quantifier_mean': mean_quantifier,
            'control_mean': mean_control
        }
        
        return features
    
    def compute_similarity_matrix(self, codes_by_type):
        """
        Compute similarity between sentence types.
        
        Args:
            codes_by_type: Dictionary of sparse codes
            
        Returns:
            Similarity matrix and statistics
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Determine feature dimension
        feature_dim = None
        for key in ['conditional', 'quantifier', 'control']:
            if key in codes_by_type and codes_by_type[key].size > 0:
                feature_dim = codes_by_type[key].shape[1]
                break
        
        if feature_dim is None:
            # If all arrays are empty, return identity matrix
            return np.eye(3), {'Conditional': np.zeros(1), 'Quantifier': np.zeros(1), 'Control': np.zeros(1)}
        
        # Get mean activation patterns
        patterns = {}
        for key, label in [('conditional', 'Conditional'), 
                          ('quantifier', 'Quantifier'), 
                          ('control', 'Control')]:
            if key in codes_by_type and len(codes_by_type[key]) > 0:
                pattern = codes_by_type[key].mean(axis=0)
                # Replace NaN and inf with 0
                pattern = np.nan_to_num(pattern, nan=0.0, posinf=0.0, neginf=0.0)
                patterns[label] = pattern
            else:
                # If no samples, use zeros
                patterns[label] = np.zeros(feature_dim)
        
        # Compute pairwise similarities
        pattern_matrix = np.array(list(patterns.values()))
        
        # Final NaN check
        pattern_matrix = np.nan_to_num(pattern_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Handle edge case where patterns might be all zeros
        if np.all(pattern_matrix == 0):
            similarity = np.eye(len(patterns))
        else:
            # Normalize to avoid numerical issues
            norms = np.linalg.norm(pattern_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            pattern_matrix_normalized = pattern_matrix / norms
            similarity = cosine_similarity(pattern_matrix_normalized)
        
        return similarity, patterns
    
    def statistical_tests(self, codes_by_type):
        """
        Perform statistical tests on feature differences.
        
        Args:
            codes_by_type: Dictionary of sparse codes
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # T-test for each feature between conditionals and quantifiers
        n_features = codes_by_type['conditional'].shape[1]
        p_values = []
        effect_sizes = []
        
        for feat_idx in range(n_features):
            cond_acts = codes_by_type['conditional'][:, feat_idx]
            quant_acts = codes_by_type['quantifier'][:, feat_idx]
            
            # T-test
            t_stat, p_val = stats.ttest_ind(cond_acts, quant_acts)
            p_values.append(p_val)
            
            # Cohen's d effect size
            pooled_std = np.sqrt(
                (np.var(cond_acts) + np.var(quant_acts)) / 2
            )
            if pooled_std > 0:
                cohens_d = (cond_acts.mean() - quant_acts.mean()) / pooled_std
            else:
                cohens_d = 0
            effect_sizes.append(cohens_d)
        
        # Multiple comparison correction
        from statsmodels.stats.multitest import multipletests
        corrected = multipletests(p_values, method='fdr_bh')
        
        results['significant_features'] = np.where(corrected[0])[0]
        results['p_values'] = np.array(p_values)
        results['corrected_p_values'] = corrected[1]
        results['effect_sizes'] = np.array(effect_sizes)
        
        # Overall distribution test
        cond_flat = codes_by_type['conditional'].flatten()
        quant_flat = codes_by_type['quantifier'].flatten()
        ks_stat, ks_p = stats.ks_2samp(cond_flat, quant_flat)
        
        results['ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_p
        }
        
        return results
    
    def create_venn_diagram(self, features):
        """
        Create Venn diagram of feature overlap.
        
        Args:
            features: Dictionary of characteristic features
            
        Returns:
            Matplotlib figure
        """
        from matplotlib_venn import venn2
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create sets
        cond_set = set(features['conditional_specific']) | set(features['shared'])
        quant_set = set(features['quantifier_specific']) | set(features['shared'])
        
        # Create Venn diagram
        venn = venn2([cond_set, quant_set], 
                     ('Conditional Features', 'Quantifier Features'))
        
        # Customize colors
        if venn.get_patch_by_id('10'):
            venn.get_patch_by_id('10').set_color('lightblue')
            venn.get_patch_by_id('10').set_alpha(0.7)
        if venn.get_patch_by_id('01'):
            venn.get_patch_by_id('01').set_color('lightgreen')
            venn.get_patch_by_id('01').set_alpha(0.7)
        if venn.get_patch_by_id('11'):
            venn.get_patch_by_id('11').set_color('yellow')
            venn.get_patch_by_id('11').set_alpha(0.7)
        
        # Add labels
        plt.title('Feature Overlap: Conditionals vs Universal Quantifiers', 
                  fontsize=14, fontweight='bold')
        
        # Add statistics
        overlap_pct = len(features['shared']) / (
            len(cond_set | quant_set)
        ) * 100 if len(cond_set | quant_set) > 0 else 0
        
        plt.text(0.5, -0.6, 
                f"Shared Features: {len(features['shared'])} ({overlap_pct:.1f}%)",
                ha='center', transform=ax.transAxes, fontsize=12)
        
        return fig
    
    def create_tsne_visualization(self, sparse_codes, labels):
        """
        Create t-SNE visualization of sentence embeddings.
        
        Args:
            sparse_codes: All sparse codes
            labels: Sentence type labels
            
        Returns:
            Matplotlib figure
        """
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = tsne.fit_transform(sparse_codes)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors
        color_map = {
            'conditional': 'blue',
            'pure_universal': 'green',
            'restricted_universal': 'lightgreen',
            'negative_universal': 'darkgreen',
            'generic_universal': 'lime',
            'any_universal': 'olive',
            'control': 'gray'
        }
        
        # Plot each type
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=color_map.get(label, 'black'),
                      label=label.replace('_', ' ').title(),
                      alpha=0.6, s=20)
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title('t-SNE Visualization of Sentence Features', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def create_heatmap(self, features):
        """
        Create heatmap of feature activations.
        
        Args:
            features: Dictionary with feature statistics
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Select top features for visualization
        top_shared = features['shared'][:20] if len(features['shared']) >= 20 else features['shared']
        top_cond = features['conditional_specific'][:10]
        top_quant = features['quantifier_specific'][:10]
        
        selected_features = np.concatenate([top_shared, top_cond, top_quant])
        
        if len(selected_features) == 0:
            return fig
        
        # Create activation matrix
        data_matrix = np.array([
            features['overlap_stats']['conditional_mean'][selected_features],
            features['overlap_stats']['quantifier_mean'][selected_features],
            features['overlap_stats']['control_mean'][selected_features]
        ])
        
        # Main heatmap
        sns.heatmap(data_matrix, 
                   xticklabels=[f'F{i}' for i in selected_features],
                   yticklabels=['Conditional', 'Quantifier', 'Control'],
                   cmap='YlOrRd', annot=False, fmt='.2f',
                   ax=axes[0], cbar_kws={'label': 'Activation'})
        axes[0].set_title('Feature Activation Patterns')
        axes[0].set_xlabel('Feature Index')
        
        # Difference heatmap (Conditional vs Quantifier)
        diff_matrix = (features['overlap_stats']['conditional_mean'][selected_features] - 
                      features['overlap_stats']['quantifier_mean'][selected_features]).reshape(1, -1)
        
        sns.heatmap(diff_matrix,
                   xticklabels=[f'F{i}' for i in selected_features],
                   yticklabels=['Cond - Quant'],
                   cmap='coolwarm', center=0, annot=False,
                   ax=axes[1], cbar_kws={'label': 'Difference'})
        axes[1].set_title('Feature Activation Difference')
        axes[1].set_xlabel('Feature Index')
        
        # Correlation matrix
        corr_data = np.corrcoef(data_matrix)
        sns.heatmap(corr_data,
                   xticklabels=['Cond', 'Quant', 'Ctrl'],
                   yticklabels=['Cond', 'Quant', 'Ctrl'],
                   cmap='coolwarm', center=0, annot=True, fmt='.2f',
                   ax=axes[2], vmin=-1, vmax=1)
        axes[2].set_title('Type Correlation Matrix')
        
        plt.suptitle('Feature Analysis: Conditionals vs Universal Quantifiers',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, codes_by_type, features, stats_results):
        """
        Generate text report of findings.
        
        Args:
            codes_by_type: Sparse codes by type
            features: Characteristic features
            stats_results: Statistical test results
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 70)
        report.append("FEATURE OVERLAP ANALYSIS: CONDITIONALS VS UNIVERSAL QUANTIFIERS")
        report.append("=" * 70)
        report.append("")
        
        # Sample sizes
        report.append("DATASET STATISTICS:")
        report.append(f"• Conditional sentences: {len(codes_by_type['conditional'])}")
        report.append(f"• Quantifier sentences: {len(codes_by_type['quantifier'])}")
        report.append(f"• Control sentences: {len(codes_by_type['control'])}")
        report.append("")
        
        # Feature statistics
        report.append("FEATURE DISCOVERY:")
        report.append(f"• Features specific to conditionals: {len(features['conditional_specific'])}")
        report.append(f"• Features specific to quantifiers: {len(features['quantifier_specific'])}")
        report.append(f"• Shared logical features: {len(features['shared'])}")
        report.append(f"• Jaccard similarity index: {features['overlap_stats']['jaccard_index']:.3f}")
        report.append("")
        
        # Top shared features
        if len(features['shared']) > 0:
            report.append("TOP SHARED FEATURES (encoding logical relationships):")
            for i, feat_idx in enumerate(features['shared'][:10]):
                cond_act = features['overlap_stats']['conditional_mean'][feat_idx]
                quant_act = features['overlap_stats']['quantifier_mean'][feat_idx]
                ctrl_act = features['overlap_stats']['control_mean'][feat_idx]
                report.append(f"  {i+1}. Feature {feat_idx}:")
                report.append(f"     Conditional: {cond_act:.3f}, Quantifier: {quant_act:.3f}, Control: {ctrl_act:.3f}")
            report.append("")
        
        # Statistical significance
        report.append("STATISTICAL TESTS:")
        report.append(f"• Significantly different features (FDR corrected): {len(stats_results['significant_features'])}")
        report.append(f"• KS test statistic: {stats_results['ks_test']['statistic']:.4f}")
        report.append(f"• KS test p-value: {stats_results['ks_test']['p_value']:.4e}")
        
        # Effect sizes
        large_effects = np.where(np.abs(stats_results['effect_sizes']) > 0.8)[0]
        report.append(f"• Features with large effect size (|d| > 0.8): {len(large_effects)}")
        report.append("")
        
        # Key findings
        report.append("KEY FINDINGS:")
        overlap_pct = len(features['shared']) / (
            len(features['conditional_specific']) + 
            len(features['quantifier_specific']) + 
            len(features['shared'])
        ) * 100 if (len(features['conditional_specific']) + 
                    len(features['quantifier_specific']) + 
                    len(features['shared'])) > 0 else 0
        
        if overlap_pct > 30:
            report.append("✓ Substantial overlap found between conditionals and quantifiers")
            report.append(f"  → {overlap_pct:.1f}% of relevant features are shared")
            report.append("  → Suggests universal quantifiers encode implicit conditional logic")
        elif overlap_pct > 10:
            report.append("✓ Moderate overlap found between conditionals and quantifiers")
            report.append(f"  → {overlap_pct:.1f}% of relevant features are shared")
            report.append("  → Some shared logical processing, but distinct representations")
        else:
            report.append("✗ Limited overlap found between conditionals and quantifiers")
            report.append(f"  → Only {overlap_pct:.1f}% of relevant features are shared")
            report.append("  → Suggests distinct neural representations")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)

def main():
    """Run overlap analysis on existing SAE model."""
    print("Feature Overlap Analyzer")
    print("This script analyzes trained SAE models.")
    print("Run generate_quantifier_dataset.py first to create the dataset.")
    print("Then run the Colab notebook to train SAE and use this for analysis.")

if __name__ == "__main__":
    main()