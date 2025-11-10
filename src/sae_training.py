"""
Sparse Autoencoder implementation and training utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder following Anthropic's architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_coeff: float = 0.01,
        use_bias: bool = True
    ):
        """
        Initialize SAE.
        
        Args:
            input_dim: Dimension of input activations
            hidden_dim: Dimension of sparse code (number of features)
            sparsity_coeff: L1 regularization coefficient
            use_bias: Whether to use bias terms
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff
        
        # Encoder: x -> ReLU(Wx + b)
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.relu = nn.ReLU()
        
        # Decoder: c -> W^T c (tied weights)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        # Tie decoder weights to encoder (transpose)
        self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation."""
        return self.relu(self.encoder(x))
    
    def decode(self, c: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation back to input space."""
        return self.decoder(c)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Returns:
            (reconstruction, sparse_code)
        """
        c = self.encode(x)
        x_hat = self.decode(c)
        return x_hat, c
    
    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with reconstruction and sparsity terms.
        
        Returns:
            (total_loss, metrics_dict)
        """
        x_hat, c = self.forward(x)
        
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_hat, x)
        
        # Sparsity loss (L1)
        sparsity_loss = self.sparsity_coeff * c.abs().mean()
        
        # Total loss
        total_loss = recon_loss + sparsity_loss
        
        # Compute metrics
        with torch.no_grad():
            # Sparsity: fraction of zero activations
            sparsity = (c == 0).float().mean().item()
            # Average number of active features per sample
            active_features = (c > 0).float().sum(dim=1).mean().item()
        
        metrics = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'sparsity': sparsity,
            'active_features': active_features
        }
        
        return total_loss, metrics


class SAETrainer:
    """Trainer for Sparse Autoencoders."""
    
    def __init__(
        self,
        model: SparseAutoencoder,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: SAE model to train
            device: Device to train on
        """
        self.model = model
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.history = {'train': [], 'val': []}
    
    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the SAE.
        
        Args:
            train_data: Training activations (n_samples, input_dim)
            val_data: Validation activations
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_data).to(self.device)
        if val_data is not None:
            val_tensor = torch.FloatTensor(val_data).to(self.device)
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_metrics = []
            
            for batch in train_loader:
                x = batch[0]
                
                # Forward pass
                loss, metrics = self.model.loss(x)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_metrics.append(metrics)
            
            # Average metrics
            avg_train_metrics = self._average_metrics(train_metrics)
            self.history['train'].append(avg_train_metrics)
            
            # Validation
            if val_data is not None:
                val_metrics = self._evaluate(val_tensor, batch_size)
                self.history['val'].append(val_metrics)
                
                # Early stopping
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            
            # Print progress
            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch:3d}: "
                msg += f"Train Loss: {avg_train_metrics['total_loss']:.4f} "
                msg += f"(R: {avg_train_metrics['recon_loss']:.4f}, "
                msg += f"S: {avg_train_metrics['sparsity_loss']:.4f}) "
                msg += f"Active: {avg_train_metrics['active_features']:.1f}"
                
                if val_data is not None:
                    msg += f" | Val Loss: {val_metrics['total_loss']:.4f}"
                
                print(msg)
        
        # Load best model if using validation
        if val_data is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def _evaluate(self, data: torch.Tensor, batch_size: int) -> Dict[str, float]:
        """Evaluate model on data."""
        self.model.eval()
        metrics = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                _, batch_metrics = self.model.loss(batch)
                metrics.append(batch_metrics)
        
        return self._average_metrics(metrics)
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Average metrics across batches."""
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
        return avg_metrics
    
    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Total loss
        axes[0].plot(self.history['train'], label='Train')
        if self.history['val']:
            axes[0].plot([m['total_loss'] for m in self.history['val']], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].legend()
        axes[0].set_title('Total Loss')
        
        # Reconstruction vs Sparsity
        axes[1].plot([m['recon_loss'] for m in self.history['train']], label='Reconstruction')
        axes[1].plot([m['sparsity_loss'] for m in self.history['train']], label='Sparsity')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].set_title('Loss Components')
        
        # Active features
        axes[2].plot([m['active_features'] for m in self.history['train']])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Active Features')
        axes[2].set_title('Average Active Features')
        
        plt.tight_layout()
        return fig


def analyze_features(
    model: SparseAutoencoder,
    activations: np.ndarray,
    metadata: List[Dict],
    top_k: int = 10
) -> Dict:
    """
    Analyze which features activate for different text types.
    
    Args:
        model: Trained SAE
        activations: Input activations
        metadata: Metadata for each activation
        top_k: Number of top features to analyze
        
    Returns:
        Analysis dictionary
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Encode all activations
    with torch.no_grad():
        acts_tensor = torch.FloatTensor(activations).to(device)
        sparse_codes = model.encode(acts_tensor).cpu().numpy()
    
    # Find features most active for conditionals
    conditional_mask = np.array([m['has_if'] for m in metadata])
    conditional_codes = sparse_codes[conditional_mask]
    non_conditional_codes = sparse_codes[~conditional_mask]
    
    # Average activation per feature
    conditional_avg = conditional_codes.mean(axis=0)
    non_conditional_avg = non_conditional_codes.mean(axis=0)
    
    # Difference scores
    diff_scores = conditional_avg - non_conditional_avg
    
    # Top differential features
    top_conditional_features = np.argsort(diff_scores)[-top_k:][::-1]
    top_non_conditional_features = np.argsort(-diff_scores)[-top_k:][::-1]
    
    return {
        'conditional_features': top_conditional_features.tolist(),
        'non_conditional_features': top_non_conditional_features.tolist(),
        'conditional_scores': diff_scores[top_conditional_features].tolist(),
        'feature_stats': {
            'total_features': model.hidden_dim,
            'avg_active_conditional': (conditional_codes > 0).sum(axis=1).mean(),
            'avg_active_non_conditional': (non_conditional_codes > 0).sum(axis=1).mean(),
        }
    }


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    
    # Create fake activations
    n_samples = 1000
    input_dim = 256
    hidden_dim = 512
    
    train_data = np.random.randn(n_samples, input_dim).astype(np.float32)
    val_data = np.random.randn(200, input_dim).astype(np.float32)
    
    # Create and train model
    model = SparseAutoencoder(input_dim, hidden_dim, sparsity_coeff=0.01)
    trainer = SAETrainer(model, device='cpu')
    
    print("Training SAE on random data...")
    history = trainer.train(
        train_data, val_data,
        epochs=50,
        batch_size=128,
        verbose=True
    )
    
    print("\nFinal metrics:")
    print(f"Train loss: {history['train'][-1]['total_loss']:.4f}")
    print(f"Active features: {history['train'][-1]['active_features']:.1f}")