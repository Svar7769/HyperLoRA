# HyperLoRA Training

HyperLoRA implementation for multi-agent reinforcement learning with context-aware LoRA adapters using a Transformer-based hypernetwork.

## Overview

This implementation combines:
- **PyTorch VMAS** for vectorized multi-agent simulation
- **JAX/Flax** for efficient neural network computation
- **Transformer-based Hypernetwork** for generating context-aware LoRA adapters
- **Shared Policy** with dynamic LoRA adaptation

## Files

- `env_setup.py`: VMAS environment initialization
- `lora_policy.py`: Policy network with LoRA adapter support
- `hypernetwork.py`: Transformer-based hypernetwork for generating LoRA adapters
- `train.py`: Main training script
- `config.yaml`: Configuration file with all hyperparameters

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# CRITICAL: Install NumPy 1.x FIRST (before anything else)
pip install "numpy>=1.23.0,<2.0.0"

# Install PyTorch (will use existing NumPy 1.x)
pip install "torch>=2.0.0,<2.2.0" "torchvision>=0.15.0,<0.17.0"

# Install JAX/Flax
pip install jax jaxlib flax optax

# Install other dependencies
pip install pyyaml gymnasium

# Install VMAS
pip install vmas
# OR if installing from local directory:
# pip install -e ../VectorizedMultiAgentSimulator
```

### Option 3: Clean Reinstall (if you have NumPy 2.x issues)

```bash
# Deactivate and remove virtual environment
deactivate
rm -rf .venv

# Create fresh virtual environment
python -m venv .venv
source .venv/bin/activate

# Follow Option 1 or Option 2 above
```

### Troubleshooting

**Problem:** `NumPy 2.x` errors or `_ARRAY_API not found`

**Solution:**
```bash
# Uninstall NumPy 2.x
pip uninstall -y numpy

# Install NumPy 1.x
pip install "numpy>=1.23.0,<2.0.0"

# Reinstall PyTorch (it will use the correct NumPy)
pip install --force-reinstall --no-cache-dir "torch>=2.0.0,<2.2.0"
```

**Problem:** `Gym` deprecation warnings

**Solution:** This is expected. VMAS internally uses Gym, but we've included Gymnasium for compatibility. The warnings can be ignored.

**Problem:** VMAS not found

**Solution:** Install from the local VectorizedMultiAgentSimulator directory:
```bash
pip install -e ../VectorizedMultiAgentSimulator
```

## Usage

### Basic Training

Run training with default configuration:

```bash
python train.py
```

### Custom Configuration File

Specify a custom config file:

```bash
python train.py --config my_config.yaml
```

### Command Line Overrides

Override specific parameters via command line:

```bash
python train.py --scenario balance --num-envs 128 --num-episodes 2000 --learning-rate 1e-3
```

### Available Command Line Arguments

**Environment:**
- `--scenario`: VMAS scenario name (default: from config)
- `--num-envs`: Number of parallel environments (default: from config)
- `--num-agents`: Number of agents per environment (default: from config)

**Training:**
- `--num-episodes`: Number of training episodes (default: from config)
- `--learning-rate`: Learning rate (default: from config)
- `--seed`: Random seed (default: from config)

**Model:**
- `--lora-rank`: Rank of LoRA adapters (default: from config)

**Device:**
- `--cuda`: Enable CUDA/GPU acceleration (default: from config)
- `--cuda-device`: CUDA device ID for multi-GPU setups (default: from config)

**Logging:**
- `--log-dir`: Directory for logs (default: from config)
- `--checkpoint-dir`: Directory for checkpoints (default: from config)
- `--no-logging`: Disable logging and checkpointing

**Weights & Biases:**
- `--wandb`: Enable Weights & Biases logging
- `--wandb-project`: W&B project name (default: from config)
- `--wandb-entity`: W&B entity/username (default: from config)
- `--wandb-name`: W&B run name (default: auto-generated)

**Visualization:**
- `--render-final-policy`: Generate GIF of final trained policy
- `--gif-steps N`: Number of steps to render in GIF (default: 100)

### GPU/CUDA Support

HyperLoRA supports GPU acceleration for both JAX (neural networks) and PyTorch (VMAS environment).

**Enable CUDA in config.yaml:**
```yaml
device:
  use_cuda: true  # Enable GPU acceleration
  cuda_device: 0  # GPU device ID (for multi-GPU systems)
```

**Enable CUDA via command line:**
```bash
python train.py --cuda true
```

**Multi-GPU training:**
```bash
# Use GPU 1 instead of GPU 0
python train.py --cuda true --cuda-device 1
```

**Requirements:**
- NVIDIA GPU with CUDA support
- JAX with CUDA support: `pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- PyTorch with CUDA support: Already included in standard PyTorch installation

**Note:** If CUDA is requested but not available, the system will automatically fall back to CPU with a warning.

### Example Commands

Train on balance scenario with 256 parallel environments:
```bash
python train.py --scenario balance --num-envs 256
```

Train with custom learning rate and LoRA rank:
```bash
python train.py --learning-rate 5e-4 --lora-rank 8
```

Quick test run without logging:
```bash
python train.py --num-episodes 10 --no-logging
```

**Train and generate GIF visualization:**
```bash
python train.py --render-final-policy --gif-steps 100
```

See [GIF_VISUALIZATION_GUIDE.md](GIF_VISUALIZATION_GUIDE.md) for detailed visualization options.

### Using Weights & Biases

Enable W&B logging with command line:
```bash
# First time: login to wandb
wandb login

# Enable wandb logging
python train.py --wandb --wandb-project my-hyperlora-project

# With custom entity and run name
python train.py --wandb --wandb-entity myusername --wandb-name experiment-1
```

Or set in `config.yaml`:
```yaml
logging:
  wandb_project: "hyperlora-vmas"
  wandb_entity: "your-username"
```

Then just run:
```bash
python train.py  # Will auto-enable wandb if project is set in config
```

**Logged metrics:**
- `reward/mean`: Average reward across all agents and environments
- `reward/min`: Minimum reward
- `reward/max`: Maximum reward  
- `reward/std`: Reward standard deviation
- `loss/action`: Action prediction loss
- `loss/advantage`: Advantage-weighted loss

## Configuration

The `config.yaml` file contains all hyperparameters organized into sections:

````

## Configuration

The `config.yaml` file contains all hyperparameters organized into sections:

### Environment Settings
```yaml
env:
  scenario_name: "balance"
  num_agents: 4
  num_envs: 64
  device: "cuda"
```

### Training Settings
```yaml
training:
  num_episodes: 1000
  rollout_steps: 100
  learning_rate: 3.0e-4
  log_interval: 10
  seed: 42
```

### Model Architecture
```yaml
model:
  obs_dim: 18
  hidden_dim: 64
  action_dim: 2
  lora_rank: 4
  context_dim: 10
  task_embed_dim: 32
  transformer_dim: 256
  transformer_heads: 4
  transformer_layers: 2
```

### Optimizer Settings
```yaml
optimizer:
  type: "adam"
  learning_rate: 3.0e-4
  beta1: 0.9
  beta2: 0.999
  eps: 1.0e-8
```

## Architecture

### Hypernetwork
- Uses a Transformer encoder to process task and capability vectors
- Generates context-aware LoRA adapters (A1, B1, A2, B2)
- Learnable context token aggregates information from multiple sources

### LoRA Policy
- Shared policy network with two dense layers
- Dynamic LoRA adaptation at each layer
- Supports batched inference across all agents

### Training Loop
1. Generate static LoRA adapters for all agents (once per episode)
2. Rollout with PyTorch environment, JAX policy
3. Collect trajectory data
4. Update both policy and hypernetwork via JAX gradient descent

## Logging

Training logs are saved to:
- `logs/<experiment_name>_<timestamp>/training_log.txt`: Episode-wise metrics
- `logs/<experiment_name>_<timestamp>/config.yaml`: Configuration used

Checkpoints are saved to:
- `checkpoints/<experiment_name>_<timestamp>/checkpoint_<episode>.npz`: Periodic checkpoints
- `checkpoints/<experiment_name>_<timestamp>/final_checkpoint.npz`: Final model

## Loading Checkpoints

```python
import numpy as np

# Load checkpoint
checkpoint = np.load('checkpoints/final_checkpoint.npz', allow_pickle=True)
policy_params = checkpoint['policy_params']
hn_params = checkpoint['hn_params']
episode = checkpoint['episode']
```

## Performance Tips

1. **Batch Size**: Increase `num_envs` for better GPU utilization
2. **LoRA Rank**: Lower rank (2-4) for faster training, higher rank (8-16) for more expressiveness
3. **Transformer Size**: Adjust `transformer_dim` and `transformer_layers` based on task complexity
4. **Learning Rate**: Start with 3e-4, reduce if training is unstable

## Citation

Based on the HyperVLA paper's Transformer-based hypernetwork architecture for generating context-aware LoRA adapters.

## License

[Specify your license here]
