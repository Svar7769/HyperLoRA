# DiCo Football

Running the DiCo (Diversity-Inducing Cooperative) baseline on the VMAS 3v3 football environment.

## Environment Setup (venv)

The code is pure-Python, mixing JAX (policy/critic/training) with PyTorch (VMAS simulator). Tested on Python 3.10–3.12.

```bash
cd /home/es2121/Downloads/dico_football/dico_football

# 1. Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2. Install runtime dependencies
pip install \
    "jax[cpu]" \
    flax \
    optax \
    distrax \
    chex \
    numpy \
    pyyaml \
    matplotlib \
    imageio \
    tqdm \
    torch \
    vmas \
    wandb
```

For GPU JAX, replace `jax[cpu]` with the appropriate CUDA build (see the JAX install matrix). `config_dico_football.yaml` has `device.use_cuda: false` by default, so CPU JAX is enough to smoke-test.

Quick sanity check after install:

```bash
python -c "import jax, flax, optax, distrax, vmas, torch; print('ok', jax.default_backend())"
```

### Known missing file

`env_setup.py:366` imports `from football_wrapper import FootballWrapper`, but `football_wrapper.py` is **not present** in this folder. The `football` scenario branch in `make_vmas_env()` will raise `ModuleNotFoundError` until that wrapper is restored. All other entry points (non-football scenarios, checkpoint loading, SND computation, DiCo policy forward pass) work without it.

### Logging (local only)

W&B has been replaced by a local logger (`local_logger.py`). Every run writes:

- `logs/<exp_name>_<timestamp>/metrics.jsonl` — one JSON object per `wandb.log()` call
- `logs/<exp_name>_<timestamp>/figs/*.png` — any matplotlib figures that used to go to `wandb.Image`
- `logs/<exp_name>_<timestamp>/config.yaml` — resolved run config
- `checkpoints/<exp_name>_<timestamp>/` — model checkpoints

The `--wandb*` CLI flags are accepted but ignored (kept for compatibility with `run_dico_football_seeds.py`). Use `--no-logging` to disable logging entirely.

## File Overview

### Entry Points

| File | Purpose |
|------|---------|
| `train.py` | Main training script. Run with `--config config_dico_football.yaml --use-dico true` |
| `run_dico_football_seeds.py` | Multi-seed runner. Iterates over seeds (and optionally target SND values) using `train.py` |
| `evaluate.py` | Loads a checkpoint and runs quantitative evaluation + optional GIF rendering |

### Configuration

| File | Purpose |
|------|---------|
| `config_dico_football.yaml` | DiCo football config: env settings (3v3, AI red team, dense reward), PPO hyperparams, diversity control (target SND 0.5, tau 0.01), 32 minibatches, model architecture ([128,128] MLP), optimizer, logging |

### Core Modules

| File | Purpose |
|------|---------|
| `dico_policy.py` | `DiCoPolicy` (with diversity control) and `DiCoHomogeneousPolicy` (fixed scaling). Architecture: `π_i(a|o) = φ_homo(o) + λ * φ_hetero_i(o)` |
| `critic.py` | `CentralizedCritic` / `CentralizedCriticRNN` — centralized value function for MAPPO |
| `snd.py` | System Neural Diversity: pairwise Wasserstein-2 distances between agent policy outputs. Key function: `calculate_snd_dico()` |
| `env_setup.py` | `make_vmas_env()` — creates and configures the VMAS environment |
| `football.py` | VMAS football scenario definition (field, agents, ball, goals, rewards). Based on ProrokLab's `BaseScenario` |
| `football_wrapper.py` | Wraps the VMAS football env for the training pipeline: observation formatting, capability context, AI red team handling |
| `render_gif.py` | GIF rendering utilities for visualizing trained policies |
| `lora_policy.py` | `LoRAPolicy` / `GRULoRAPolicy` — used by HyperLoRA mode only, but imported in `train.py` |
| `hypernetwork.py` | `Hypernetwork` — used by HyperLoRA mode only, but imported in `train.py` |

## Where DiCo is Trained in `train.py`

### 1. Imports and CLI Parsing

| Lines | What happens |
|-------|-------------|
| L34-39 | Import SND functions from `snd.py` (`calculate_snd`, `calculate_snd_dico`, etc.) |
| L48-49 | Import `DiCoPolicy`, `DiCoHomogeneousPolicy` from `dico_policy.py` |
| L173-183 | CLI arg overrides: `--use-dico`, `--diversity-control`, `--target-snd` applied to config |
| L314-365 | argparse definitions for `--use-dico`, `--target-snd`, `--eval-snds` |

### 2. Diversity Control Initialization

| Lines | What happens |
|-------|-------------|
| L2668-2675 | Read `use_diversity_control`, `target_snd`, `snd_moving_average_coef` from config. Initialize `current_snd_ma = target_snd`, `diversity_scaling = 1.0` |
| L2677-2694 | Optional SND observation buffer and adapter SND buffer settings |

### 3. Model Initialization (DiCo-specific branch)

| Lines | What happens |
|-------|-------------|
| L3327-3328 | Read `use_dico` flag from config |
| L3336-3346 | If `use_dico`: disable hypernetwork, print DiCo architecture info |
| L3401-3426 | **Create DiCo policy**: if `use_diversity_control` → `DiCoPolicy(num_agents, hidden_dims, action_dim, ...)`, else → `DiCoHomogeneousPolicy(...)`. Set `hypernetwork = None` |
| L3586-3593 | Initialize DiCo policy params with `shared_policy.init(policy_rng, dummy_obs, dummy_agent_ids)` |

### 4. Action Selection During Rollout

| Lines | What happens |
|-------|-------------|
| L586-615 | `get_actions_and_log_probs()`: if `agent_ids is not None` (DiCo), calls `shared_policy.apply(params, obs, agent_ids, diversity_scaling, rng_key)` to sample actions |
| L729-746 | `_get_actions_dico()`: deterministic action selection for evaluation — returns clipped mean |

### 5. SND Calculation and Diversity Scaling Update (per-episode)

| Lines | What happens |
|-------|-------------|
| L6108-6117 | **Calculate unscaled SND**: `calculate_snd_dico(policy_params, obs, policy, num_agents, sample_size, diversity_scaling=1.0)` |
| L6119-6135 | **Update moving average**: `current_snd_ma = τ * current_snd_ma + (1-τ) * snd_unscaled` (with NaN protection) |
| L6137-6160 | **Compute diversity scaling λ**: `λ = target_snd / max(current_snd_ma, ε)`, clipped to `[0.001, max_diversity_scaling]` |
| L6172-6179 | Calculate scaled SND (with `diversity_scaling=λ`) for monitoring |

### 6. PPO Training Step (DiCo-specific)

| Lines | What happens |
|-------|-------------|
| L1887-2063 | `_train_step_dico()`: MAPPO loss function for DiCo. Calls `shared_policy.apply(params, obs, agent_ids, diversity_scaling)` to get mean/log_std, computes PPO clipped surrogate loss + value loss + entropy bonus. Returns updated `policy_state`, `critic_state`, `loss_info` |
| L2163-2218 | `create_minibatches_dico()`: splits data into minibatches maintaining per-agent ↔ global-state correspondence |
| L9125-9172 | **Minibatch training loop**: iterates over PPO epochs × minibatches, calling `_train_step_dico()` on each. Accumulates and averages loss info |
| L9174-9176 | **Full-batch fallback**: calls `_train_step_dico()` directly if minibatches disabled |

### 7. Checkpointing

| Lines | What happens |
|-------|-------------|
| L10124-10127 | Save `current_snd_ma` in checkpoint for proper diversity scaling during evaluation |

## Quick Start

### Single run

```bash
python train.py --config config_dico_football.yaml --use-dico true
```

### Multi-seed sweep

```bash
# Default seeds [0, 42, 342, 3421, 34210]
python run_dico_football_seeds.py

# Custom seeds with wandb
python run_dico_football_seeds.py --seeds 0 42 342 --wandb

# Sweep over multiple target SND values × seeds
python run_dico_football_seeds.py --target-snd 0.1 0.2 0.5 --seeds 0 42 342
```

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/<checkpoint_dir> --num-agents 3
```

## Key Config Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model.use_dico` | `true` | Enables DiCo architecture |
| `training.use_diversity_control` | `true` | Enables SND-based diversity scaling (λ) |
| `training.target_snd` | `0.5` | Desired System Neural Diversity level |
| `training.snd_moving_average_coef` | `0.01` | Exponential moving average τ for SND tracking |
| `training.max_diversity_scaling` | `1000.0` | Upper bound for λ |
| `env.ai_red_agents` | `true` | Red team controlled by heuristic AI |
| `env.num_agents` | `3` | 3v3 football |
| `training.num_episodes` | `3600` | Total training episodes |
| `training.rollout_steps` | `300` | Steps per rollout |
| `training.num_minibatches` | `32` | Minibatch count (3,600 samples each) |
