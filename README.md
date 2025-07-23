# DeepAGLA: Deep Learning Enhanced Accelerated Griffin-Lim Algorithm

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-1.8+-orange.svg)](https://pytorch-lightning.readthedocs.io/)

*A deep learning approach to accelerate audio phase reconstruction using learnable Griffin-Lim iterations*

</div>

## 📖 Overview

DeepAGLA implements a neural network-based enhancement of the Accelerated Griffin-Lim Algorithm (presented [here](https://arxiv.org/pdf/2306.12504)) for high-quality audio phase reconstruction from magnitude spectrograms. The model uses learnable parameters (α, β, γ) in place of fixed coefficients, enabling data-driven optimization for superior audio quality.

### Key Features

- 🧠 **Learnable AGLA Parameters**: Neural network learns optimal α, β, γ coefficients
- 🔄 **Progressive Layer Training**: Adaptive layer activation during training
- 💾 **Gradient Checkpointing**: Memory-efficient training for deep models
- 📊 **Comprehensive Metrics**: SSNR, SI-SDR, SNR, LSD, and spectral convergence
- 🎵 **Audio Logging**: Real-time WandB audio reconstruction tracking
- 🔍 **Hyperparameter Optimization**: Built-in Ax-based optimization

## 🛠️ Installation

### Requirements

```bash
# Install Python dependencies
pip install torch torchvision torchaudio
pip install pytorch-lightning wandb
pip install librosa soundfile scipy
pip install ax-platform  # For hyperparameter optimization
pip install tqdm numpy matplotlib
```

### Clone Repository

```bash
git clone https://github.com/GuyPerets106/DeepAGLA.git
cd DeepAGLA
```

## 📊 Data Generation

### Step 1: Generate Training Data

The project uses the SaSpeeech dataset for training. You can either download it automatically or use your own audio files.

#### Option A: Auto-download SaSpeeech Dataset

```bash
# Download and process both GOLD and AUTO datasets (recommended)
python generate_data.py --download both --data_dir ./data

# Or download specific subsets:
python generate_data.py --download gold --data_dir ./data  # High-quality subset
python generate_data.py --download auto --data_dir ./data  # Automatic subset
```

#### Option B: Use Custom Audio Files

```bash
# Process your own WAV files
python generate_data.py --wav_dir /path/to/your/wavs --data_dir ./data
```

### Data Output

The script will create:
- `data/audio_gold_auto.npy`: Combined dataset (recommended)
- `data/audio_gold.npy`: High-quality subset only
- `data/audio_auto.npy`: Automatic subset only

**Expected format**: `(N, T)` where N is number of samples, T is time samples (44,100 for 2-second clips at 22kHz)

## ⚙️ Configuration

### Step 2: Modify Training Parameters (Optional)

Edit `definitions.py` to customize training settings:

```python
# Core Settings
PROJECT_NAME = "deep-agla-experiment"      # WandB project name
DATA_PATH = "./data/audio_gold_auto.npy"   # Path to your dataset
N_LAYERS = 64                              # Number of AGLA layers (16-256)
BATCH_SIZE = 8                             # Batch size (adjust for GPU memory)
NUM_EPOCHS = 200                           # Training epochs

# Audio Processing
SAMPLE_RATE = 22050                        # Audio sample rate
N_FFT = 512                                # FFT size
HOP = N_FFT // 4                          # Hop length

# Optimization
LEARNING_RATE = 0.008                      # Base learning rate
WEIGHT_DECAY = 1e-4                        # L2 regularization

# Data Splits
VAL_SPLIT = 0.05                           # Validation split (5%)
TEST_SPLIT = 0.05                          # Test split (5%)
```

### Advanced Configuration

For fine-tuned control, modify these parameters in `train.py`:

```python
# Loss weights (in train.py main function)
loss_weights = {
    "time_l1": 0.9,        # Time-domain L1 loss weight
    "time_mse": 0.9,       # Time-domain MSE loss weight  
    "spec_l1": 0.1,        # Spectral L1 loss weight
    "log_spec_l1": 0.1,    # Log-spectral L1 loss weight
}

# Model architecture
model = DeepAGLA(
    n_layers=N_LAYERS,              # Number of learnable AGLA layers
    use_checkpointing=True,         # Enable gradient checkpointing
    layer_lr_decay=0.97,           # Layer-wise learning rate decay
    audio_log_interval=5,          # Log audio every N epochs
)
```

## 🚀 Training

### Step 3A: Single Training Run

Run a single training session with default or custom parameters:

```bash
# Basic training with default parameters
python train.py

# Training will automatically:
# - Load data from DATA_PATH in definitions.py
# - Create WandB logs with audio reconstruction samples
# - Save best model checkpoints
# - Display training progress and metrics
```

### Step 3B: Hyperparameter Optimization

Use Ax optimization to find the best hyperparameters:

```bash
# Run hyperparameter search (20 trials)
python ax_search.py --trials 20

# Custom number of trials
python ax_search.py --trials 50 --run_name "optimization_experiment"
```

**Optimization Search Space:**
- Learning rate: 1e-4 to 1e-2 (log scale)
- Batch size: [64, 128, 256]
- Weight decay: 1e-5 to 1e-2 (log scale)
- Loss weights: 0.1 to 0.9 for each component

### Training Outputs

```
📁 Project Structure After Training:
├── wandb/                          # WandB logs and metrics
├── ax_trials/                      # Hyperparameter optimization results
│   └── trial_lr0.001_bs128_*/     # Individual trial results
├── checkpoints/                    # Model checkpoints
│   ├── best_snr_model.ckpt        # Best performing model
│   └── last.ckpt                  # Latest checkpoint
└── logs/                          # Training logs
```

## 📈 Monitoring Training

### WandB Integration

The project automatically logs to Weights & Biases:

- **Real-time Metrics**: SSNR, SNR, SI-SDR, loss curves
- **Audio Samples**: Reconstructed audio every 5 epochs
- **Gradient Analysis**: Layer-wise gradient norms
- **Model Artifacts**: Best checkpoints with metadata

## 🎯 Model Architecture

### Progressive Layer Training

DeepAGLA uses adaptive progressive training:

- **Small models** (≤16 layers): Train all layers from start
- **Medium models** (17-64 layers): Start with 25% layers, increment every 3 epochs
- **Large models** (65-256 layers): Start with 12.5% layers, increment every 4 epochs
- **Very large models** (>256 layers): Start with 6.25% layers, increment every 5 epochs

### Layer Structure

Each AGLA layer learns three parameters:
```python
t = (1 - γ) * d_prev + γ * proj_pc1(proj_pc2(c, s))
c = t + α * (t - t_prev)  
d = t + β * (t - t_prev)
```

Where:
- `α, β, γ`: Learnable parameters (initialized based on layer depth)
- `proj_pc1`: Time-domain consistency projection
- `proj_pc2`: Magnitude consistency projection
