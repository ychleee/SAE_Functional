# Sparse Autoencoder for Conditional Features

Discovering interpretable features for functional words and conditionals in language models using Sparse Autoencoders (SAEs).

## Project Overview

This project investigates whether language models contain latent features corresponding to logical and conditional reasoning operations. We use SAEs to extract interpretable features from model activations when processing conditional sentences ("if...then" constructs) and functional words (and, or, not, etc.).

### Research Questions
- Do functional words correspond to distinct, monosemantic features in SAE representations?
- Can we find feature-level evidence for logical operations or compositional reasoning?
- How do models internally represent different types of conditionals?

## Project Structure

```
sparse-autoencoder-conditionals/
├── src/                        # Core Python modules
│   ├── data_utils.py          # Dataset generation
│   ├── activation_extraction.py # Model activation extraction
│   ├── sae_training.py        # SAE architecture & training
│   └── feature_analysis.py    # Feature interpretation
├── notebooks/
│   ├── 01_data_preparation.ipynb  # Local: prepare datasets
│   ├── 02_sae_training.ipynb      # Colab: train SAE with GPU
│   └── 03_analysis.ipynb          # Local: analyze results
├── configs/
│   └── training_config.yaml   # Hyperparameters
├── data/                      # Datasets (gitignored large files)
├── models/                    # Trained models (gitignored)
└── results/                   # Analysis outputs
```

## Workflow

This project uses a **hybrid local + Colab workflow**:

1. **Local Development**: Write code, prepare data, analyze results
2. **Colab GPU Training**: Train SAEs on free GPUs
3. **Result Transfer**: Via Google Drive or direct download

### Quick Start

#### Step 1: Local Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/sparse-autoencoder-conditionals
cd sparse-autoencoder-conditionals

# Install local dependencies
pip install -r requirements.txt

# Prepare dataset
jupyter notebook notebooks/01_data_preparation.ipynb
```

#### Step 2: Colab Training
1. Upload `notebooks/02_sae_training.ipynb` to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells to:
   - Clone your GitHub repo
   - Extract activations from language model
   - Train SAE
   - Save results to Google Drive

#### Step 3: Local Analysis
```bash
# Download results from Google Drive to models/checkpoints/
# Then run analysis notebook
jupyter notebook notebooks/03_analysis.ipynb
```

## Key Components

### Data Generation (`src/data_utils.py`)
- Synthetic conditional sentences with controlled complexity
- Balanced datasets with control sentences
- Test sets for logical inference evaluation

### Activation Extraction (`src/activation_extraction.py`)
- Extract hidden states from transformer models (Pythia-70M, GPT-2)
- Target specific layers and tokens
- Pool activations for SAE training

### SAE Training (`src/sae_training.py`)
- Implements Anthropic-style SAE architecture
- L1 sparsity regularization
- Training with early stopping

### Feature Analysis (`src/feature_analysis.py`)
- Identify features correlated with conditionals
- Clustering and visualization
- Causal intervention experiments

## Configuration

Edit `configs/training_config.yaml` to adjust:
- Model selection (Pythia-70M, GPT-2-small)
- SAE architecture (hidden dimensions, sparsity)
- Training hyperparameters
- Analysis settings

## Results

The SAE discovers features that:
- Activate specifically for conditional constructs
- Differentiate between logical connectives
- Show compositional structure

Example findings are saved in `results/` including:
- Feature activation patterns
- Clustering visualizations
- Intervention effect measurements

## Requirements

### Local Environment
- Python 3.8+
- PyTorch 2.0+
- Jupyter
- See `requirements.txt` for full list

### Colab Environment
- GPU runtime (T4 or better)
- See `requirements_colab.txt` for dependencies

## Citation

This project builds on:
- Anthropic's SAE work (2024)
- TransformerLens/SAELens libraries
- Theoretical work on conditional semantics

## Next Steps

1. **Scale up**: Try larger models (Pythia-410M, GPT-2-medium)
2. **Natural data**: Use real conditional datasets
3. **Semantic theories**: Test specific hypotheses (probabilistic vs dynamic semantics)
4. **Causal interventions**: Implement full activation patching

## License

MIT

## Contact

For questions or collaborations, please open an issue or contact [your email].