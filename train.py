import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import torch
import optax
import distrax
from flax.training import train_state
import argparse
import yaml
from pathlib import Path
import os
import sys
from datetime import datetime
import matplotlib
import time

# ============================================================================
# JAX Performance Configuration
# ============================================================================
# Print backend device info at startup
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
# ============================================================================

matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt

from env_setup import make_vmas_env
from lora_policy import LoRAPolicy, GRULoRAPolicy
from cash_policy import CASHPolicy
from hypernetwork import Hypernetwork
from critic import CentralizedCritic, CentralizedCriticRNN, CentralizedCriticDeepSets
from render_gif import generate_policy_gif
from snd import (
    calculate_snd,
    calculate_snd_statistics,
    calculate_adapter_snd,
    calculate_snd_dico,
    compute_snd_from_action_means,
    plot_adapter_impact_distribution,
)
from evaluate import run_quantitative_evaluation, print_training_eval_results
from flax import linen as nn

# Import for JAX environment wrapper
import warnings

# Import DICO policy for baseline comparison
from dico_policy import DiCoPolicy, DiCoHomogeneousPolicy, DiCoDeepSetsPolicy

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


# ============================================================================
# Utility Functions for Dynamic Agent Masking
# ============================================================================


def create_attention_mask(current_num_agents, max_agents):
    """
    Creates a standard attention mask for dynamic agent counts.

    Args:
        current_num_agents: Number of active agents in this batch
        max_agents: Maximum number of agents (fixed capacity)

    Returns:
        Attention mask of shape (1, 1, max_agents, max_agents) for broadcasting.
        Format: True for "keep", False for "mask out" (Flax convention).
        Active agents can attend to each other; padded positions are masked.
    """
    # For current_num_agents == max_agents, all agents are active
    # No masking needed - return None to skip masking
    if current_num_agents == max_agents:
        return None

    # Create a boolean mask of shape (1, 1, max_agents, max_agents)
    # True where both query and key agents are active (< current_num_agents)
    # False where either is a padded position (>= current_num_agents)
    mask = jnp.ones((max_agents, max_agents), dtype=bool)
    # Set padded positions to False
    mask = mask.at[current_num_agents:, :].set(False)  # Padded queries can't attend
    mask = mask.at[:, current_num_agents:].set(False)  # Can't attend to padded keys

    # Add batch and head dimensions: (1, 1, max_agents, max_agents)
    mask = mask[None, None, :, :]
    return mask


def generate_positional_encoding(num_agents, encoding_dim, device="cpu"):
    """
    Generate sinusoidal positional encodings for agent IDs (like in Transformers).

    This encoding scales better with variable agent counts compared to one-hot encoding,
    as it doesn't require the dimensionality to match max_agents.

    Args:
        num_agents: Number of agents to encode
        encoding_dim: Dimension of the positional encoding (typically 16-32)
        device: PyTorch device for tensor creation

    Returns:
        Tensor of shape (num_agents, encoding_dim) with positional encodings
    """
    # Create position indices: [0, 1, 2, ..., num_agents-1]
    positions = torch.arange(num_agents, dtype=torch.float32, device=device).unsqueeze(
        1
    )

    # Create dimension indices and compute frequencies
    dim_indices = torch.arange(encoding_dim, dtype=torch.float32, device=device)
    div_term = torch.exp(dim_indices * -(np.log(10000.0) / encoding_dim))

    # Compute sinusoidal encodings
    pos_encoding = torch.zeros(num_agents, encoding_dim, device=device)
    pos_encoding[:, 0::2] = torch.sin(positions * div_term[0::2])  # Even dimensions
    pos_encoding[:, 1::2] = torch.cos(positions * div_term[1::2])  # Odd dimensions

    return pos_encoding


# ============================================================================
# Configuration Loading
# ============================================================================


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def override_config(config, args):
    """Override config values with command line arguments."""
    if args.scenario is not None:
        config["env"]["scenario_name"] = args.scenario
    if args.num_envs is not None:
        config["env"]["num_envs"] = args.num_envs
    if args.num_agents is not None:
        config["env"]["num_agents"] = args.num_agents
    if args.max_agents is not None:
        config["env"]["max_agents"] = args.max_agents
    if args.num_episodes is not None:
        config["training"]["num_episodes"] = args.num_episodes
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
        config["optimizer"]["learning_rate"] = args.learning_rate
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.lora_rank is not None:
        config["model"]["lora_rank"] = args.lora_rank
    if args.lora_mode is not None:
        config["model"]["lora_mode"] = args.lora_mode
    if args.log_dir is not None:
        config["logging"]["log_dir"] = args.log_dir
    if args.checkpoint_dir is not None:
        config["logging"]["checkpoint_dir"] = args.checkpoint_dir
    if args.cuda is not None:
        config["device"]["use_cuda"] = args.cuda
    if args.cuda_device is not None:
        config["device"]["cuda_device"] = args.cuda_device
    if hasattr(args, "use_hypernetwork") and args.use_hypernetwork is not None:
        config["model"]["use_hypernetwork"] = args.use_hypernetwork

    # DICO architecture override
    if hasattr(args, "use_dico") and args.use_dico is not None:
        config["model"]["use_dico"] = args.use_dico

    # Diversity control overrides
    if hasattr(args, "diversity_control") and args.diversity_control is not None:
        config["training"]["use_diversity_control"] = args.diversity_control
    if hasattr(args, "target_snd") and args.target_snd is not None:
        config["training"]["target_snd"] = args.target_snd
    if hasattr(args, "train_snds") and args.train_snds:
        config["training"]["target_snd_list"] = args.train_snds
    if (
        hasattr(args, "train_snd_interval")
        and args.train_snd_interval is not None
        and args.train_snd_interval > 0
    ):
        config["training"]["target_snd_change_interval"] = args.train_snd_interval
    if hasattr(args, "train_snd_mode") and args.train_snd_mode is not None:
        config["training"]["target_snd_mode"] = args.train_snd_mode
    if hasattr(args, "per_env_diversity_scaling"):
        config["training"]["per_env_diversity_scaling"] = args.per_env_diversity_scaling

    # Global state clipping overrides
    if hasattr(args, "clip_global_state") and args.clip_global_state is not None:
        config["training"]["clip_global_state"] = args.clip_global_state
    if (
        hasattr(args, "clip_global_state_min")
        and args.clip_global_state_min is not None
    ):
        config["training"]["clip_global_state_min"] = args.clip_global_state_min
    if (
        hasattr(args, "clip_global_state_max")
        and args.clip_global_state_max is not None
    ):
        config["training"]["clip_global_state_max"] = args.clip_global_state_max

    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HyperLoRA on VMAS environments")

    # Config file
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )

    # Environment arguments
    parser.add_argument("--scenario", type=str, default=None, help="VMAS scenario name")
    parser.add_argument(
        "--num-envs", type=int, default=None, help="Number of parallel environments"
    )
    parser.add_argument(
        "--num-agents", type=int, default=None, help="Number of agents per environment"
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=None,
        help="Maximum number of agents for deployment (default: same as num-agents)",
    )

    # Training arguments
    parser.add_argument(
        "--num-episodes", type=int, default=None, help="Number of training episodes"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Global state clipping arguments
    parser.add_argument(
        "--clip-global-state",
        dest="clip_global_state",
        action="store_true",
        help="Enable clipping of global state before critic (value loss stabilization)",
    )
    parser.add_argument(
        "--no-clip-global-state",
        dest="clip_global_state",
        action="store_false",
        help="Disable clipping of global state before critic",
    )
    parser.set_defaults(clip_global_state=None)
    parser.add_argument(
        "--clip-global-state-min",
        type=float,
        default=None,
        help="Minimum value for global state clipping",
    )
    parser.add_argument(
        "--clip-global-state-max",
        type=float,
        default=None,
        help="Maximum value for global state clipping",
    )

    # Model arguments
    parser.add_argument(
        "--lora-rank", type=int, default=None, help="Rank of LoRA adapters"
    )
    parser.add_argument(
        "--lora-mode",
        type=str,
        default=None,
        choices=["final_only", "last_hidden", "all"],
        help="LoRA complexity mode: 'final_only' (output only), 'last_hidden' (last hidden + output), 'all' (all layers)",
    )
    parser.add_argument(
        "--use-hypernetwork",
        dest="use_hypernetwork",
        action="store_true",
        help="Enable hypernetwork (HyperLoRA)",
    )
    parser.add_argument(
        "--no-hypernetwork",
        dest="use_hypernetwork",
        action="store_false",
        help="Disable hypernetwork (use standard MAPPO)",
    )
    parser.set_defaults(use_hypernetwork=None)

    # Device arguments
    parser.add_argument(
        "--cuda", type=bool, default=None, help="Enable CUDA/GPU acceleration"
    )
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device ID")

    # Logging arguments
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for logs")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Directory for checkpoints"
    )
    parser.add_argument("--no-logging", action="store_true", help="Disable logging")

    # Weights & Biases arguments
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="W&B project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity/username"
    )
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")

    # Debug
    parser.add_argument(
        "--debug-nan",
        action="store_true",
        help="Enable per-step NaN sanity checks (forces host-device syncs; slow)",
    )

    # Architecture selection
    parser.add_argument(
        "--use-dico",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Use DICO per-agent architecture instead of HyperLoRA (true/false)",
    )

    # Visualization arguments
    parser.add_argument(
        "--render-final-policy",
        action="store_true",
        help="Generate a GIF visualization of the final trained policy",
    )
    parser.add_argument(
        "--gif-steps",
        type=int,
        default=None,
        help="Number of steps to render in the GIF (default: use max_eval_steps from config)",
    )

    # Diversity control arguments
    parser.add_argument(
        "--diversity-control",
        dest="diversity_control",
        action="store_true",
        help="Enable diversity control (SND-based adapter scaling)",
    )
    parser.add_argument(
        "--no-diversity-control",
        dest="diversity_control",
        action="store_false",
        help="Disable diversity control",
    )
    parser.set_defaults(diversity_control=None)
    parser.add_argument(
        "--target-snd",
        type=float,
        default=None,
        help="Target SND (System Neural Diversity) level for diversity control",
    )
    parser.add_argument(
        "--per-env-diversity-scaling",
        action="store_true",
        help="Use per-environment diversity scaling (each env scaled by its own target). Default: uniform scaling based on mean target.",
    )
    parser.add_argument(
        "--eval-snds",
        type=float,
        nargs="+",
        default=None,
        help="Target SND values for post-training evaluation sweep (e.g., --eval-snds 0.5 0.7 1.0 1.2 1.5 1.8)",
    )
    parser.add_argument(
        "--train-snds",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of target SND values to cycle/sample DURING rollout training (e.g., --train-snds 0.3 0.5 1.0)",
    )
    parser.add_argument(
        "--train-snd-interval",
        type=int,
        default=None,
        help="Rollout step interval for switching target SND during training (0/None disables switching)",
    )
    parser.add_argument(
        "--train-snd-mode",
        type=str,
        choices=["cycle", "random"],
        default=None,
        help="How to choose the next target SND during training switches: cycle or random",
    )

    # Performance profiling arguments
    parser.add_argument(
        "--jax-profile",
        action="store_true",
        help="Enable JAX profiling (creates trace for TensorBoard). Warning: slows down training significantly.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./jax_profiles",
        help="Directory to save JAX profiler traces",
    )
    parser.add_argument(
        "--disable-jit",
        action="store_true",
        help="Disable JIT compilation for debugging (WARNING: very slow, only for debugging)",
    )

    return parser.parse_args()


# ============================================================================
# JAX Helper Functions (JIT-compiled)
# ============================================================================


def detect_food_in_range(obs_list, num_envs, num_agents):
    """
    Detect which agents have food within lidar range by checking the in_range_flag.

    Args:
        obs_list: List of observations from VMAS [agent0_obs, agent1_obs, ...]
                 Each has shape: (num_envs, obs_dim)
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment

    Returns:
        food_in_range: Boolean tensor of shape (num_envs, num_agents)
                      True if food is within lidar range for that agent
    """
    # Observation structure: pos(2) + vel(2) + food(4) + lidar(...)
    # Food observation: [rel_x, rel_y, eaten_status, in_range_flag]
    # in_range_flag is at index 7 (after pos[2], vel[2], food_pos[2], eaten_status[1])

    food_in_range_list = []
    for agent_obs in obs_list:
        # Extract in_range_flag (index 7)
        in_range_flag = agent_obs[:, 7]  # (num_envs,)
        food_in_range_list.append(in_range_flag > 0.5)  # Threshold to boolean

    # Stack to (num_envs, num_agents)
    food_in_range = torch.stack(food_in_range_list, dim=1)
    return food_in_range


def detect_landmark_reached(rewards, num_envs, num_agents, num_adversaries):
    """
    Detect if any good agent reached a landmark in grassland environment.
    Landmarks give +20 reward when reached.

    Args:
        rewards: List of rewards [agent0_reward, agent1_reward, ...]
                 Each has shape: (num_envs,)
        num_envs: Number of parallel environments
        num_agents: Total number of agents (good + adversaries)
        num_adversaries: Number of adversary agents

    Returns:
        torch.Tensor of shape (num_envs, num_agents) with True if landmark reached
    """
    # Stack rewards: (num_agents, num_envs)
    rewards_stacked = torch.stack(rewards, dim=0)
    # Transpose to: (num_envs, num_agents)
    rewards_transposed = rewards_stacked.transpose(0, 1)

    # Create mask for good agents only (adversaries are first, good agents follow)
    good_agent_mask = torch.zeros_like(rewards_transposed, dtype=torch.bool)
    good_agent_mask[:, num_adversaries:] = True

    # Detect landmark reached: reward >= 19.0 (accounting for slight numerical errors)
    # Only for good agents
    landmark_reached = (rewards_transposed >= 19.0) & good_agent_mask

    return landmark_reached


def extract_food_positions(obs_list, num_agents, scenario_name):
    """
    Extract relative positions of ALL food items for each agent from observations.

    For dispersion_vmas with global observability:
    - Observation structure: pos(2) + vel(2) + [food_0(3) + food_1(3) + ... + food_N(3)]
    - Each food entry: [rel_x, rel_y, eaten_status]
    - We concatenate [rel_x, rel_y] for all N food items per agent

    Args:
        obs_list: List of agent observations [agent0_obs, agent1_obs, ...]
                  Each has shape (num_envs, obs_dim)
        num_agents: Number of agents (= number of food items)
        scenario_name: Name of the scenario

    Returns:
        Tensor of shape (num_envs, num_agents, num_agents*2) containing relative
        positions to ALL food items for each agent, or None if not applicable
    """
    # Only extract for dispersion_vmas (not dispersion)
    if scenario_name != "dispersion_vmas":
        return None

    all_agent_food_positions = []

    for agent_obs in obs_list:
        # For this agent, extract relative positions to ALL food items
        # Food k starts at index 4 + k*3 and has 3 values: [rel_x, rel_y, eaten]
        # We concatenate [rel_x, rel_y] for all k in range(num_agents)
        food_rel_positions = []
        for k in range(num_agents):
            food_start_idx = 4 + k * 3
            rel_pos_k = agent_obs[
                :, food_start_idx : food_start_idx + 2
            ]  # (num_envs, 2)
            food_rel_positions.append(rel_pos_k)
        # Concatenate all food positions: (num_envs, num_agents*2)
        all_food = torch.cat(food_rel_positions, dim=-1)
        all_agent_food_positions.append(all_food)

    # Stack to (num_agents, num_envs, num_agents*2) then transpose to (num_envs, num_agents, num_agents*2)
    food_positions_tensor = torch.stack(all_agent_food_positions, dim=0).transpose(0, 1)

    return food_positions_tensor


def extract_agent_positions(obs_list, num_agents, scenario_name):
    """
    Extract absolute agent positions from observations.

    For dispersion_vmas with global observability:
    - Observation structure: pos(2) + vel(2) + [food_0(3) + food_1(3) + ... + food_N(3)]
    - Agent position is at indices [0:2]

    For wind_flocking_position:
    - Observation structure: pos(2) + vel(2) + rel_pos_to_others(2*(n-1)) + wind(2)
    - Agent position is at indices [0:2]

    Args:
        obs_list: List of agent observations [agent0_obs, agent1_obs, ...]
                  Each has shape (num_envs, obs_dim)
        num_agents: Number of agents
        scenario_name: Name of the scenario

    Returns:
        Tensor of shape (num_envs, num_agents, 2) containing RELATIVE positions
        of each agent (relative to center of mass), or None if not applicable
    """
    # Only extract for specific scenarios
    if scenario_name not in ["dispersion_vmas", "wind_flocking_position"]:
        return None

    num_envs = obs_list[0].shape[0]
    agent_positions = []

    for i, agent_obs in enumerate(obs_list):
        # Agent position is the first 2 dimensions of observation
        agent_pos = agent_obs[:, :2]  # (num_envs, 2)
        agent_positions.append(agent_pos)

    # Stack to (num_agents, num_envs, 2) then transpose to (num_envs, num_agents, 2)
    agent_positions_tensor = torch.stack(agent_positions, dim=0).transpose(0, 1)

    # Compute center of mass for each environment
    # Shape: (num_envs, 2)
    center_of_mass = agent_positions_tensor.mean(dim=1)  # Average over agents

    # Compute relative positions (relative to center of mass)
    # This ensures each agent gets different inputs to hypernetwork
    # Shape: (num_envs, num_agents, 2)
    relative_positions = agent_positions_tensor - center_of_mass.unsqueeze(1)

    return relative_positions


@jax.jit
def _get_static_adapters(
    hn_params,
    task_batch,
    context_batch,
    lidar_batch=None,
    food_positions_batch=None,
    agent_positions_batch=None,
    target_snd_batch=None,
    env_context_batch=None,
    mask=None,
    diversity_scaling=1.0,
):
    """
    Generate LoRA adapters for all agents using cross-agent attention.

    Args:
        hn_params: Hypernetwork parameters
        task_batch: Task embeddings (num_envs, num_agents, task_dim)
        context_batch: Context/capability vectors (num_envs, num_agents, context_dim)
        lidar_batch: Optional initial lidar readings (num_envs, num_agents, lidar_dim)
        food_positions_batch: Optional food positions (num_envs, num_agents, food_dim)
        agent_positions_batch: Optional absolute agent positions (num_envs, num_agents, 2)
        target_snd_batch: Optional target SND values (num_envs, num_agents, target_snd_dim)
        env_context_batch: Optional environment context (e.g., package properties) (num_envs, num_agents, env_context_dim)
        mask: Optional attention mask (num_envs, 1, num_agents, num_agents)
        diversity_scaling: Additional scaling for final layer only (default: 1.0)

    Returns:
        Dictionary of adapter matrices for all agents (flattened to batch_size)
    """
    return hypernetwork.apply(
        {"params": hn_params},
        task_batch,
        context_batch,
        lidar_batch,
        food_positions_batch,
        agent_positions_batch,
        target_snd_batch,
        env_context_batch,
        mask,
        diversity_scaling,
    )


@jax.jit
def _build_cash_context_from_capabilities(capability_batch):
    """
    Build CASH capability context from per-agent capabilities.

    Args:
        capability_batch: (num_envs, num_agents, capability_dim)

    Returns:
        cash_capability_context: (num_envs, num_agents, capability_dim + num_agents*capability_dim)
            [ego_capabilities, team_capabilities_concatenated]
    """
    num_envs, num_agents, capability_dim = capability_batch.shape
    team_caps = capability_batch.reshape(num_envs, num_agents * capability_dim)
    team_caps = jnp.repeat(team_caps[:, None, :], num_agents, axis=1)
    return jnp.concatenate([capability_batch, team_caps], axis=-1)


@jax.jit
def _get_actions_and_log_probs(
    policy_params,
    obs_batch,
    adapters_dict,
    rng_key,
    agent_ids=None,
    diversity_scaling=1.0,
    hidden_state=None,
    dones_batch=None,
    cash_capability_context_batch=None,
):
    """
    Get actions and log probabilities from the policy using LoRA adapters or DiCo.

    Args:
        policy_params: Policy parameters
        obs_batch: Observations (batch_size, obs_dim)
        adapters_dict: Dictionary of LoRA adapter matrices (for HyperLoRA) or None (for DiCo)
        rng_key: JAX random key for sampling
        agent_ids: Agent IDs for DiCo routing (batch_size,) or None
        diversity_scaling: Diversity scaling factor for DiCo
        hidden_state: Hidden state for GRU policy (batch_size, hidden_dim) or None
        dones_batch: Done flags for GRU reset (batch_size,) or None

    Returns:
        actions: Sampled actions (batch_size, action_dim)
        log_probs: Log probabilities (batch_size,)
        new_hidden: Updated hidden state (for GRU) or None (for MLP)
    """
    # For DiCo: pass agent_ids and diversity_scaling
    if agent_ids is not None:
        action, log_prob, mean, std = shared_policy.apply(
            {"params": policy_params},
            obs_batch,
            agent_ids,
            diversity_scaling,
            rng_key,
            method=shared_policy.get_action_and_log_prob,
        )
        return action, log_prob, None
    else:
        # CASH path: hyper-adapter runs inside policy at each step.
        if cash_capability_context_batch is not None:
            batch_size = obs_batch.shape[0]
            obs_seq = obs_batch[None, ...]
            if dones_batch is None:
                dones_seq = jnp.zeros((1, batch_size), dtype=bool)
            else:
                dones_seq = dones_batch[None, ...]
            cash_context_seq = cash_capability_context_batch[None, ...]
            policy_x = (obs_seq, dones_seq, cash_context_seq)
            action, log_prob, new_hidden, mean, std = shared_policy.apply(
                {"params": policy_params},
                hidden_state,
                policy_x,
                rng_key,
                method=shared_policy.get_action_and_log_prob,
            )
            action = jnp.squeeze(action, axis=0)
            log_prob = jnp.squeeze(log_prob, axis=0)
            mean = jnp.squeeze(mean, axis=0)
            std = jnp.squeeze(std, axis=0)
            return action, log_prob, new_hidden, mean, std

        # For HyperLoRA: pass adapters_dict
        # Check if using GRU policy
        if hidden_state is not None:
            # GRU policy: need to format input properly
            batch_size = obs_batch.shape[0]
            obs_seq = obs_batch[None, ...]  # Add time dimension: (1, batch, obs_dim)
            if dones_batch is None:
                dones_seq = jnp.zeros((1, batch_size), dtype=bool)
            else:
                dones_seq = dones_batch[None, ...]  # (1, batch)
            avail_seq = None  # Continuous actions, no masking
            policy_x = (obs_seq, dones_seq, avail_seq)

            # GRU returns 5 values: (action, log_prob, new_hidden, mean, std)
            action, log_prob, new_hidden, mean, std = shared_policy.apply(
                {"params": policy_params},
                hidden_state,
                policy_x,
                adapters_dict,
                rng_key,
                method=shared_policy.get_action_and_log_prob,
            )
            # Squeeze time dimension: (1, batch, ...) -> (batch, ...)
            action = jnp.squeeze(action, axis=0)
            log_prob = jnp.squeeze(log_prob, axis=0)
            mean = jnp.squeeze(mean, axis=0)
            std = jnp.squeeze(std, axis=0)
            return action, log_prob, new_hidden, mean, std
        else:
            # MLP policy: direct call (returns 4 values)
            action, log_prob, mean, std = shared_policy.apply(
                {"params": policy_params},
                obs_batch,
                adapters_dict,
                rng_key,
                method=shared_policy.get_action_and_log_prob,
            )
            return action, log_prob, None, mean, std


@jax.jit
def _get_actions(policy_params, obs_batch, adapters_dict):
    """
    Get deterministic actions from the policy (evaluation mode).

    Args:
        policy_params: Policy parameters
        obs_batch: Observations (batch_size, obs_dim)
        adapters_dict: Dictionary of LoRA adapter matrices

    Returns:
        Actions (batch_size, action_dim)
    """
    mean, log_std = shared_policy.apply(
        {"params": policy_params}, obs_batch, adapters_dict
    )
    # Clip actions to valid range [-1, 1] for VMAS
    return jnp.clip(mean, -1.0, 1.0)


@jax.jit
def _get_policy_mean_without_adapters(
    policy_params, obs_batch, hidden_state=None, dones_batch=None
):
    """
    Get policy mean WITHOUT LoRA adapters (baseline backbone only).

    This is used to measure how much the adapters modify the base policy.

    Args:
        policy_params: Policy parameters
        obs_batch: Observations (batch_size, obs_dim)
        hidden_state: Hidden state for GRU policy (batch_size, hidden_dim) or None
        dones_batch: Done flags for GRU reset (batch_size,) or None

    Returns:
        mean: Action distribution mean from backbone only (batch_size, action_dim)
    """
    # Create empty adapters dictionary (no LoRA modifications)
    empty_adapters = {}

    # Check if using GRU policy
    if hidden_state is not None:
        # GRU policy: need to format input properly
        batch_size = obs_batch.shape[0]
        obs_seq = obs_batch[None, ...]  # Add time dimension: (1, batch, obs_dim)
        if dones_batch is None:
            dones_seq = jnp.zeros((1, batch_size), dtype=bool)
        else:
            dones_seq = dones_batch[None, ...]  # (1, batch)
        avail_seq = None  # Continuous actions, no masking
        policy_x = (obs_seq, dones_seq, avail_seq)

        # Apply policy without adapters
        _, output = shared_policy.apply(
            {"params": policy_params}, hidden_state, policy_x, empty_adapters
        )
        mean_seq, _ = output
        mean = mean_seq[0]  # Remove time dimension: (batch, action_dim)
    else:
        # MLP policy: direct call
        mean, _ = shared_policy.apply(
            {"params": policy_params}, obs_batch, empty_adapters
        )

    return mean


@jax.jit
def _get_actions_dico(policy_params, obs_batch, agent_ids, diversity_scaling=1.0):
    """
    Get deterministic actions from DiCo policy (evaluation mode).

    Args:
        policy_params: Policy parameters
        obs_batch: Observations (batch_size, obs_dim)
        agent_ids: Agent IDs for routing (batch_size,)
        diversity_scaling: Diversity scaling factor

    Returns:
        Actions (batch_size, action_dim)
    """
    mean, log_std = shared_policy.apply(
        {"params": policy_params}, obs_batch, agent_ids, diversity_scaling
    )
    # Clip actions to valid range [-1, 1] for VMAS
    return jnp.clip(mean, -1.0, 1.0)


def _get_actions_gru(policy_params, obs_batch, adapters_dict, hidden_state):
    """
    Get deterministic actions from GRU policy (evaluation mode).

    Args:
        policy_params: Policy parameters
        obs_batch: Observations (batch_size, obs_dim)
        adapters_dict: Dictionary of LoRA adapter matrices
        hidden_state: GRU hidden state (batch_size, hidden_dim)

    Returns:
        actions: Actions (batch_size, action_dim)
        new_hidden_state: Updated hidden state (batch_size, hidden_dim)
    """
    batch_size = obs_batch.shape[0]
    obs_seq = obs_batch[None, ...]  # Add time dimension: (1, batch, obs_dim)
    dones_seq = jnp.zeros((1, batch_size), dtype=bool)
    avail_seq = None  # Continuous actions, no masking
    policy_x = (obs_seq, dones_seq, avail_seq)

    new_hidden_state, output = shared_policy.apply(
        {"params": policy_params}, hidden_state, policy_x, adapters_dict
    )
    # For continuous actions, output is (mean, log_std)
    mean_seq, log_std_seq = output
    mean = mean_seq[0]  # Remove time dimension
    # Clip actions to valid range [-1, 1] for VMAS
    actions = jnp.clip(mean, -1.0, 1.0)
    return actions, new_hidden_state


@jax.jit
def _get_value(critic_params, global_state, critic_hidden=None, dones=None):
    """
    Get value estimate from centralized critic.

    Args:
        critic_params: Critic parameters
        global_state: Global state (all agents' observations concatenated)
        critic_hidden: Hidden state for RNN critic (optional)
        dones: Episode termination flags for RNN critic (optional)

    Returns:
        value: Per-agent state value estimates (batch, num_agents) [and new hidden state if RNN critic]
    """
    if critic_hidden is not None and dones is not None:
        # RNN critic: add time dimension and return (hidden, value)
        global_state_seq = jnp.expand_dims(global_state, 0)  # (1, batch, state_dim)
        dones_seq = jnp.expand_dims(dones, 0)  # (1, batch)
        new_hidden, value = critic.apply(
            {"params": critic_params}, critic_hidden, (global_state_seq, dones_seq)
        )
        # Remove time dimension from value: (1, batch, num_agents) -> (batch, num_agents)
        return new_hidden, jnp.squeeze(value, axis=0)
    else:
        # MLP critic: simple forward pass -> (batch, num_agents)
        return critic.apply({"params": critic_params}, global_state)


@partial(
    jax.jit,
    static_argnums=(21, 22, 23),
    static_argnames=("is_discrete", "value_clip_range"),
)  # num_agents/current_num_agents/max_agents static by position; is_discrete and value_clip_range static by name
def _train_step_with_hn(
    policy_state,
    hn_state,
    critic_state,
    obs_batch,
    global_state_batch,
    actions_batch,
    old_log_probs_batch,
    advantages_batch,
    returns_batch,
    task_batch,
    context_batch,
    lidar_batch,
    food_position_batch,
    agent_position_batch,
    target_snd_batch,
    env_context_batch,
    mask,
    action_masks_batch,  # Added for SMAX action masking
    clip_epsilon,
    entropy_coef,
    value_loss_coef,
    num_agents,
    current_num_agents,
    max_agents,
    lora_scaling_factor=1.0,
    diversity_scaling=1.0,
    is_discrete=False,
    old_values_batch=None,  # (num_steps * num_envs,) old critic predictions for value clipping
    value_clip_range=None,  # float or None; static so JAX can branch at compile time
):
    """
    Perform one MAPPO training step for policy, hypernetwork, and centralized critic.
    Used when use_hypernetwork=True.

    Args:
        policy_state: Policy training state
        hn_state: Hypernetwork training state
        critic_state: Centralized critic training state
        obs_batch: Batch of local observations
        global_state_batch: Batch of global states (all agents' obs concatenated)
        actions_batch: Batch of actions taken
        old_log_probs_batch: Batch of old log probabilities
        old_values_batch: Batch of old value predictions (for value clipping)
        advantages_batch: Batch of advantages (GAE)
        returns_batch: Batch of returns
        task_batch: Batch of task embeddings
        context_batch: Batch of context vectors
        lidar_batch: Batch of initial lidar readings
        food_position_batch: Batch of relative food positions
        agent_position_batch: Batch of agent positions
        target_snd_batch: Batch of target SND values
        mask: Attention mask for cross-agent attention
        clip_epsilon: PPO clipping parameter
        entropy_coef: Entropy coefficient
        value_loss_coef: Value loss coefficient
        num_agents: Number of agents per environment
        current_num_agents: Current number of active agents (for loss masking)
        max_agents: Maximum number of agents (fixed capacity)

    Returns:
        Updated policy_state, hn_state, critic_state, and loss info
    """

    def loss_fn(policy_params, hn_params, critic_params):
        # Generate adapters with cross-agent attention
        # diversity_scaling is captured from the outer scope (passed via closure)
        adapters_dict = hypernetwork.apply(
            {"params": hn_params},
            task_batch,
            context_batch,
            lidar_batch,
            None,  # food_position_batch - not used
            agent_position_batch,
            target_snd_batch,
            env_context_batch,
            mask,
            diversity_scaling,
        )

        # Check if using discrete actions (SMAX)
        # is_discrete is captured from the outer scope (passed via closure)

        if is_discrete:
            # SMAX: Policy outputs logits for categorical distribution
            # Check if using GRU policy (needs special handling)
            if hasattr(shared_policy, "gru_hidden_dim"):
                # GRU policy requires (hidden, x, adapters) signature
                # For training, we use zero-initialized hidden states
                batch_size = obs_batch.shape[0]
                dummy_hidden = jnp.zeros((batch_size, shared_policy.gru_hidden_dim))

                # Add time dimension: (1, batch_size, ...)
                obs_seq = obs_batch[None, ...]
                dones_seq = jnp.zeros((1, batch_size), dtype=bool)
                # Determine action dim from action_masks_batch or use a reasonable default
                if action_masks_batch is not None:
                    action_dim = action_masks_batch.shape[-1]
                    avail_seq = action_masks_batch[None, ...]
                else:
                    # Fallback: assume actions_batch gives us the action dim
                    action_dim = (
                        actions_batch.max() + 1
                        if len(actions_batch.shape) == 1
                        else actions_batch.shape[-1]
                    )
                    avail_seq = jnp.ones((1, batch_size, action_dim))

                policy_x = (obs_seq, dones_seq, avail_seq)
                _, logits_seq = shared_policy.apply(
                    {"params": policy_params}, dummy_hidden, policy_x, adapters_dict
                )
                logits = logits_seq[0]  # Remove time dimension
            else:
                # Standard policy (non-GRU)
                logits, _ = shared_policy.apply(
                    {"params": policy_params}, obs_batch, adapters_dict
                )

            # CRITICAL: Apply action masks if provided
            if action_masks_batch is not None:
                # Mask invalid actions by setting their logits to -inf
                masked_logits = jnp.where(
                    action_masks_batch.astype(bool),
                    logits,
                    jnp.full_like(logits, -1e10),
                )
            else:
                masked_logits = logits

            # Create categorical distribution with masked logits
            dist = distrax.Categorical(logits=masked_logits)
            # Compute log probabilities for discrete actions
            new_log_probs = dist.log_prob(actions_batch)
            # Entropy for discrete distribution
            entropy_per_agent = dist.entropy()
        else:
            # Continuous actions: Get mean and log_std for current policy (PRE-TANH)
            mean, log_std = shared_policy.apply(
                {"params": policy_params}, obs_batch, adapters_dict
            )
            std = jnp.exp(log_std)

            # CRITICAL: Ensure std never collapses to 0 or becomes NaN
            # This MUST match the logic in lora_policy.py exactly
            std = jnp.maximum(
                std, 0.3
            )  # Higher minimum std to prevent entropy collapse

            # Create the SAME tanh-transformed distribution as in policy sampling
            base_dist = distrax.Normal(mean, std)
            tanh_bijector = distrax.Tanh()
            dist = distrax.Transformed(base_dist, tanh_bijector)

            # Compute log probabilities without clipping (CRITICAL FIX)
            # Action clipping should only happen during environment interaction, not in loss
            new_log_probs = dist.log_prob(actions_batch).sum(axis=-1)

            # IMPORTANT: Entropy should be from the BASE distribution (pre-tanh)
            # We want to encourage exploration in the Gaussian space, not the tanh-squashed space
            entropy_per_agent = base_dist.entropy().sum(axis=-1)  # (batch_size,)

        # NaN protection for log probabilities
        new_log_probs = jnp.nan_to_num(
            new_log_probs, nan=-100.0, posinf=-100.0, neginf=-100.0
        )
        new_log_probs = jnp.clip(new_log_probs, -100.0, 100.0)

        # CRITICAL: Clip log prob difference before computing ratio
        # This prevents ratio explosion when policies diverge
        log_prob_diff = new_log_probs - old_log_probs_batch
        log_prob_diff = jnp.clip(log_prob_diff, -3.0, 3.0)  # Prevents exp overflow

        # Compute ratio from clipped log prob difference
        ratio = jnp.exp(log_prob_diff)

        # PPO clipped objective - use clipped log prob difference (ratio already computed above)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

        # Compute both terms per agent
        surr1 = ratio * advantages_batch
        surr2 = clipped_ratio * advantages_batch
        policy_loss_per_agent = -jnp.minimum(surr1, surr2)  # (batch_size,)

        # CRITICAL: Create valid agent mask for loss masking
        # Shape: (batch_size,) - True for real agents, False for padded agents
        # obs_batch has shape (num_steps * num_envs * num_agents, obs_dim)
        batch_size_total = obs_batch.shape[0]
        # num_agents here is actually env_num_agents (passed from caller)
        # batch_size_total = num_steps * num_envs * num_agents

        # Create mask: [True, True, ..., False, False] for each environment
        # Shape: (num_agents,) where num_agents is env_num_agents
        # Use comparison with indices to avoid jnp.arange in JIT
        agent_indices = jnp.arange(num_agents, dtype=jnp.int32)
        agent_mask_template = agent_indices < current_num_agents
        # Tile for all timesteps and environments: (num_steps * num_envs, num_agents)
        # Then flatten: (num_steps * num_envs * num_agents,)
        valid_agent_mask = jnp.tile(agent_mask_template, batch_size_total // num_agents)

        # Apply mask to policy loss (zero out padded agents)
        policy_loss_masked = policy_loss_per_agent * valid_agent_mask
        policy_loss = policy_loss_masked.sum() / jnp.maximum(
            valid_agent_mask.sum(), 1.0
        )

        # Apply mask to entropy
        entropy_masked = entropy_per_agent * valid_agent_mask
        entropy = entropy_masked.sum() / jnp.maximum(valid_agent_mask.sum(), 1.0)

        # Add small regularization to prevent complete collapse (continuous actions only)
        # This ensures there's always some gradient signal
        if not is_discrete:
            policy_regularization = 0.001 * jnp.mean(jnp.square(mean))
            policy_loss = policy_loss + policy_regularization

        # Centralized value function loss
        # Critic always outputs (batch, num_agents) regardless of shared/per-agent mode
        values = critic.apply({"params": critic_params}, global_state_batch)

        # Flatten values to match returns_batch shape
        # values: (num_steps * num_envs, num_agents) -> (num_steps * num_envs * num_agents,)
        # returns_batch: (num_steps * num_envs * num_agents,)
        values_flat = values.reshape(-1)  # (num_steps * num_envs * num_agents,)

        # Compute value loss per agent — with optional PPO value clipping.
        # value_clip_range is a compile-time static, so the if/else is resolved at trace time.
        if value_clip_range is not None and old_values_batch is not None:
            # Standard PPO value clip: use max(unclipped_loss, clipped_loss) so the
            # critic cannot move more than ±value_clip_range from the old prediction.
            old_values_flat = old_values_batch.reshape(-1)
            values_clipped = old_values_flat + jnp.clip(
                values_flat - old_values_flat,
                -value_clip_range,
                value_clip_range,
            )
            vf_loss_unclipped = (values_flat - returns_batch) ** 2
            vf_loss_clipped = (values_clipped - returns_batch) ** 2
            value_loss_per_agent = jnp.maximum(vf_loss_unclipped, vf_loss_clipped)
        else:
            value_loss_per_agent = (values_flat - returns_batch) ** 2

        # Apply mask to value loss (critical for mixed training)
        value_loss_masked = value_loss_per_agent * valid_agent_mask
        value_loss = value_loss_masked.sum() / jnp.maximum(valid_agent_mask.sum(), 1.0)

        # CRITICAL: Add adapter regularization to ensure hypernetwork learns
        # Only regularize adapters that exist in adapters_dict (robust to lora_mode)
        adapter_norm = 0.0
        adapter_count = 0
        for v in adapters_dict.values():
            adapter_norm += jnp.mean(jnp.square(v))
            adapter_count += 1
        adapter_mean_norm = (
            jnp.sqrt(adapter_norm / max(adapter_count, 1)) if adapter_count > 0 else 0.0
        )

        # L2 regularization on adapters to prevent unbounded growth
        # This penalizes large adapter magnitudes, encouraging the hypernetwork to produce
        # small, efficient adapters that rely on the base policy
        adapter_regularization_coef = 0.1
        adapter_regularization_loss = adapter_regularization_coef * adapter_norm

        # Total loss with entropy bonus, value loss, and adapter regularization
        total_loss = (
            policy_loss
            + value_loss_coef * value_loss
            - entropy_coef * entropy
            + adapter_regularization_loss
        )

        # Build info dict - only include std metrics for continuous actions
        info_dict = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((new_log_probs - old_log_probs_batch) ** 2),
            "mean_ratio": jnp.mean(ratio),  # Track average probability ratio
            "advantages_mean": jnp.mean(advantages_batch),
            "advantages_std": jnp.std(advantages_batch),
            "advantages_max": jnp.max(jnp.abs(advantages_batch)),
            "adapter_norm": adapter_mean_norm,  # Track adapter growth
        }

        # Only add std metrics for continuous actions (not discrete)
        if not is_discrete:
            info_dict["mean_std"] = jnp.mean(std)
            info_dict["min_std"] = jnp.min(std)
            info_dict["max_std"] = jnp.max(std)

        return total_loss, info_dict

    # Compute gradients for all three networks
    (loss, info), grads = jax.value_and_grad(
        lambda p, h, c: loss_fn(p, h, c), argnums=(0, 1, 2), has_aux=True
    )(policy_state.params, hn_state.params, critic_state.params)

    policy_grads, hn_grads, critic_grads = grads

    # Compute gradient norms for debugging
    policy_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(policy_grads))
    )
    hn_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(hn_grads))
    )
    critic_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(critic_grads))
    )

    # Add to info
    info["policy_grad_norm"] = policy_grad_norm
    info["hn_grad_norm"] = hn_grad_norm
    info["critic_grad_norm"] = critic_grad_norm

    # NaN protection: replace NaN gradients with zeros
    policy_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), policy_grads
    )
    hn_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), hn_grads
    )
    critic_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), critic_grads
    )

    # Update all three networks
    policy_state = policy_state.apply_gradients(grads=policy_grads)
    hn_state = hn_state.apply_gradients(grads=hn_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)

    return policy_state, hn_state, critic_state, info


@partial(
    jax.jit, static_argnums=(23, 24, 25, 28)
)  # num_agents, current_num_agents, max_agents, is_discrete
def _train_step_with_hn_gru(
    policy_state,
    hn_state,
    critic_state,
    obs_sequence,  # (num_steps, num_actors, obs_dim)
    global_state_sequence,  # (num_steps, num_envs, global_state_dim)
    actions_sequence,  # (num_steps, num_actors)
    old_log_probs_sequence,  # (num_steps, num_actors)
    advantages_sequence,  # (num_steps, num_actors)
    returns_sequence,  # (num_steps, num_actors)
    task_sequence,  # (num_steps, num_actors, task_dim)
    context_sequence,  # (num_steps, num_actors, context_dim)
    lidar_sequence,  # (num_steps, num_actors, lidar_dim)
    food_position_sequence,  # (num_steps, num_actors, 2)
    agent_position_sequence,  # (num_steps, num_actors, 2)
    target_snd_sequence,  # (num_steps, num_actors, target_snd_dim)
    mask,  # Attention mask
    action_masks_sequence,  # (num_steps, num_actors, action_dim)
    init_hidden_state,  # Initial hidden state for actor RNN
    init_critic_hidden,  # Initial hidden state for critic RNN
    dones_sequence,  # (num_steps, num_envs) - Episode termination flags
    clip_epsilon,
    entropy_coef,
    value_loss_coef,
    num_agents,
    current_num_agents,
    max_agents,
    lora_scaling_factor=1.0,
    diversity_scaling=1.0,
    is_discrete=False,
):
    """
    GRU-specific training step that properly handles sequential data.
    Follows HyperMARL's approach of processing entire rollout sequences.

    Key differences from standard training:
    - Input data has shape (num_steps, num_actors, ...)
    - Hidden states are propagated through the sequence
    - Losses are computed per-timestep then averaged

    Args:
        All sequence arguments have shape (num_steps, ...) to preserve temporal structure
        init_hidden_state: Initial RNN hidden state for the sequence
    """

    def loss_fn(policy_params, hn_params, critic_params):
        num_steps = obs_sequence.shape[0]
        num_actors = obs_sequence.shape[1]

        # Compute inferred_num_envs (needed for both hypernetwork and no-hypernetwork paths)
        inferred_num_envs = num_actors // num_agents

        # Generate adapters (or use zeros if no hypernetwork)
        if hn_params is not None:
            # Use hypernetwork to generate adapters
            # Use first timestep's context to generate adapters
            task_first = task_sequence[0]  # (num_actors, task_dim)
            context_first = context_sequence[0]  # (num_actors, context_dim)
            lidar_first = lidar_sequence[0] if lidar_sequence is not None else None
            food_position_first = (
                food_position_sequence[0]
                if food_position_sequence is not None
                else None
            )
            agent_position_first = (
                agent_position_sequence[0]
                if agent_position_sequence is not None
                else None
            )
            # IMPORTANT: Also extract first timestep mask for GRU case
            # mask has shape (num_steps, num_envs, 1, num_agents, num_agents) in GRU
            # but hypernetwork expects (num_envs, 1, num_agents, num_agents)
            mask_first = mask[0] if mask is not None else None

            # Reshape for hypernetwork (inferred_num_envs already computed above)
            if task_first.shape[-1] > 0:
                task_3d = task_first.reshape(
                    inferred_num_envs, num_agents, task_first.shape[-1]
                )
            else:
                task_3d = task_first.reshape(inferred_num_envs, num_agents, 0)

            if context_first.shape[-1] > 0:
                context_3d = context_first.reshape(
                    inferred_num_envs, num_agents, context_first.shape[-1]
                )
            else:
                context_3d = context_first.reshape(inferred_num_envs, num_agents, 0)

            if lidar_first is not None and lidar_first.shape[-1] > 0:
                lidar_3d = lidar_first.reshape(
                    inferred_num_envs, num_agents, lidar_first.shape[-1]
                )
            else:
                lidar_3d = None

            if food_position_first is not None and food_position_first.shape[-1] > 0:
                food_position_3d = food_position_first.reshape(
                    inferred_num_envs, num_agents, food_position_first.shape[-1]
                )
            else:
                food_position_3d = None

            if agent_position_first is not None and agent_position_first.shape[-1] > 0:
                agent_position_3d = agent_position_first.reshape(
                    inferred_num_envs, num_agents, agent_position_first.shape[-1]
                )
            else:
                agent_position_3d = None

            # Extract target_snd from sequence (also static, use first timestep)
            if target_snd_sequence is not None:
                target_snd_first = target_snd_sequence[
                    0
                ]  # (num_actors, target_snd_dim)
                if target_snd_first.shape[-1] > 0:
                    target_snd_3d = target_snd_first.reshape(
                        inferred_num_envs, num_agents, target_snd_first.shape[-1]
                    )
                else:
                    target_snd_3d = None
            else:
                target_snd_3d = None

            # Generate adapters once
            adapters_dict = hypernetwork.apply(
                {"params": hn_params},
                task_3d,
                context_3d,
                lidar_3d,
                None,  # food_position_3d - not used
                agent_position_3d,
                target_snd_3d,
                None,  # env_context - not used in GRU rollout currently
                mask_first,  # Use first timestep's mask, not full sequence
                diversity_scaling,
            )

            # Compute adapter norm for monitoring
            adapter_norm = jnp.sqrt(
                sum(
                    jnp.sum(jnp.square(v))
                    for v in jax.tree_util.tree_leaves(adapters_dict)
                )
            )
        else:
            # No hypernetwork: use zero-rank adapters
            # For GRU policy, only need final layer adapters
            adapters_dict = {
                "A1": jnp.zeros((num_actors, 0, shared_policy.gru_hidden_dim)),
                "B1": jnp.zeros((num_actors, shared_policy.action_dim, 0)),
            }
            adapter_norm = 0.0

        # Debug: Check adapter shapes and values
        # jax.debug.print("Adapter A1 shape: {}, sample: {}", adapters_dict["A1"].shape, adapters_dict["A1"][0, :2, :5])
        # jax.debug.print("Adapter B1 shape: {}, sample: {}", adapters_dict["B1"].shape, adapters_dict["B1"][0, :5, :])

        # Process ENTIRE sequence through GRU at once (not step-by-step)
        # obs_sequence: (num_steps, num_actors, obs_dim)
        # CRITICAL: Use actual dones to reset GRU hidden state during training,
        # matching the rollout behavior where hidden states are reset on episode boundaries.
        #
        # TIMING: stored dones[t] = result of env.step() at step t (dones AFTER action t)
        # During rollout, at step t we pass prev_dones (= dones from step t-1) to reset BEFORE processing obs[t]
        # So for training: dones_for_actor[0] = False (no reset at start), dones_for_actor[t] = stored_dones[t-1]
        # This is a shift-by-1: prepend False, drop the last dones
        #
        # dones_sequence is (num_steps, num_envs) - expand to (num_steps, num_actors)
        # CRITICAL FIX: Verify shape is correct before expanding
        assert dones_sequence.shape == (
            num_steps,
            inferred_num_envs,
        ), f"dones_sequence shape {dones_sequence.shape} != expected ({num_steps}, {inferred_num_envs})"

        dones_shifted = jnp.concatenate(
            [
                jnp.zeros(
                    (1, dones_sequence.shape[1]), dtype=bool
                ),  # No reset at step 0
                dones_sequence[:-1],  # dones from step 0..N-2 used at step 1..N-1
            ],
            axis=0,
        )  # (num_steps, num_envs)
        dones_for_actor = jnp.repeat(
            dones_shifted, num_agents, axis=1
        )  # (num_steps, num_actors)
        assert dones_for_actor.shape == (
            num_steps,
            num_actors,
        ), f"dones_for_actor shape {dones_for_actor.shape} != expected ({num_steps}, {num_actors})"
        # For continuous actions, pass None; for discrete, pass action masks
        avail_sequence = (
            action_masks_sequence if action_masks_sequence is not None else None
        )

        policy_x = (obs_sequence, dones_for_actor, avail_sequence)

        final_hidden_state, policy_output = shared_policy.apply(
            {"params": policy_params}, init_hidden_state, policy_x, adapters_dict
        )

        # Process outputs for loss computation (now all timesteps at once)
        if is_discrete:
            # Discrete actions: policy_output is logits (num_steps, num_actors, action_dim)
            logits_all = policy_output

            if action_masks_sequence is not None:
                masked_logits = jnp.where(
                    action_masks_sequence.astype(bool),
                    logits_all,
                    jnp.full_like(logits_all, -1e10),
                )
            else:
                masked_logits = logits_all

            # Reshape for distribution: flatten first two dims
            masked_logits_flat = masked_logits.reshape(-1, masked_logits.shape[-1])
            # Handle different action shapes for discrete actions
            if actions_sequence.ndim == 3 and actions_sequence.shape[-1] == 1:
                # Actions stored as (num_steps, num_actors, 1) - squeeze last dim
                actions_flat = actions_sequence.reshape(-1, 1).squeeze(-1)
            elif actions_sequence.ndim == 2:
                # Actions stored as (num_steps, num_actors)
                actions_flat = actions_sequence.reshape(-1)
            else:
                # Fallback
                actions_flat = actions_sequence.reshape(-1)

            dist = distrax.Categorical(logits=masked_logits_flat)
            new_log_probs_all = dist.log_prob(actions_flat).reshape(
                num_steps, num_actors
            )
            entropy_all = dist.entropy().reshape(num_steps, num_actors)
        else:
            # Continuous actions: policy_output is (mean, log_std)
            mean_all, log_std_all = (
                policy_output  # Both: (num_steps, num_actors, action_dim)
            )
            action_std_all = jnp.exp(log_std_all)

            # Flatten for distribution
            mean_flat = mean_all.reshape(-1, mean_all.shape[-1])
            std_flat = action_std_all.reshape(-1, action_std_all.shape[-1])
            actions_flat = (
                actions_sequence.reshape(-1, actions_sequence.shape[-1])
                if actions_sequence.ndim > 2
                else actions_sequence.reshape(-1, 1)
            )

            # CRITICAL: Use Tanh-transformed distribution to match rollout behavior
            base_dist = distrax.Normal(loc=mean_flat, scale=std_flat)
            tanh_bijector = distrax.Tanh()
            dist = distrax.Transformed(base_dist, tanh_bijector)

            # Log prob uses transformed distribution (includes Tanh jacobian correction)
            new_log_probs_all = (
                dist.log_prob(actions_flat).sum(axis=-1).reshape(num_steps, num_actors)
            )
            # Entropy uses base distribution (Tanh transformation doesn't have constant jacobian)
            entropy_all = (
                base_dist.entropy().sum(axis=-1).reshape(num_steps, num_actors)
            )

        # NaN protection for log probabilities
        new_log_probs_all = jnp.nan_to_num(
            new_log_probs_all, nan=-100.0, posinf=-100.0, neginf=-100.0
        )
        new_log_probs_all = jnp.clip(new_log_probs_all, -100.0, 100.0)

        # Compute losses for all timesteps
        # old_log_probs_sequence, advantages_sequence, returns_sequence: (num_steps, num_actors)
        log_prob_diff_all = new_log_probs_all - old_log_probs_sequence
        log_prob_diff_all = jnp.clip(log_prob_diff_all, -3.0, 3.0)
        ratio_all = jnp.exp(log_prob_diff_all)
        clipped_ratio_all = jnp.clip(ratio_all, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surr1_all = ratio_all * advantages_sequence
        surr2_all = clipped_ratio_all * advantages_sequence
        policy_loss_all = -jnp.minimum(surr1_all, surr2_all)

        # Value loss - process global state sequence through RNN critic
        # global_state_sequence: (num_steps, num_envs, global_state_dim)
        # dones_sequence: (num_steps, num_envs)
        # Use RNN critic to get values with temporal context
        _, values_all = critic.apply(
            {"params": critic_params},
            init_critic_hidden,
            (global_state_sequence, dones_sequence),
        )
        # values_all: (num_steps, num_envs, num_agents) or (num_steps, num_envs, 1) for shared critic
        num_envs = global_state_sequence.shape[1]
        agents_per_env = num_actors // num_envs

        # Flatten: (num_steps, num_envs, num_agents) -> (num_steps, num_actors)
        values_flat = values_all.reshape(
            values_all.shape[0], -1
        )  # (num_steps, num_actors)
        value_loss_all = jnp.square(values_flat - returns_sequence)

        # Apply agent mask
        agent_indices = jnp.arange(num_agents, dtype=jnp.int32)
        agent_mask = agent_indices < current_num_agents
        agent_mask_full = jnp.tile(
            agent_mask, num_actors // num_agents
        )  # (num_actors,)
        agent_mask_broadcast = agent_mask_full[
            None, :
        ]  # (1, num_actors) for broadcasting

        policy_loss_masked = policy_loss_all * agent_mask_broadcast
        entropy_masked = entropy_all * agent_mask_broadcast
        value_loss_masked = value_loss_all * agent_mask_broadcast

        # Compute approx KL divergence for monitoring
        log_ratio_all = new_log_probs_all - old_log_probs_sequence
        approx_kl_all = (log_ratio_all**2) * agent_mask_broadcast

        # Sum over time and agents
        policy_loss = policy_loss_masked.sum() / jnp.maximum(
            agent_mask_broadcast.sum() * num_steps, 1.0
        )
        entropy = entropy_masked.sum() / jnp.maximum(
            agent_mask_broadcast.sum() * num_steps, 1.0
        )
        value_loss = value_loss_masked.sum() / jnp.maximum(
            agent_mask_broadcast.sum() * num_steps, 1.0
        )
        approx_kl = approx_kl_all.sum() / jnp.maximum(
            agent_mask_broadcast.sum() * num_steps, 1.0
        )

        # OLD STEP-BY-STEP CODE REMOVED
        # (This was the bug: treating each timestep as an independent sequence of length 1)
        def step_fn(carry, step_inputs):
            # THIS FUNCTION IS NO LONGER USED - KEEPING FOR REFERENCE ONLY
            hidden_state = carry
            (
                obs_step,
                actions_step,
                old_log_probs_step,
                advantages_step,
                returns_step,
                global_state_step,
                action_masks_step,
            ) = step_inputs

            # OLD IMPLEMENTATION - REMOVED (was the bug)
            # This was treating each timestep as an independent sequence of length 1
            # Keeping function stub for code structure
            pass

        # Note: step_fn is no longer used - sequence is processed all at once above
        # The old scan-based approach was incorrectly treating each timestep independently

        # L2 regularization on adapters to prevent unbounded growth
        adapter_squared_norm = sum(
            jnp.sum(jnp.square(v)) for v in jax.tree_util.tree_leaves(adapters_dict)
        )
        adapter_regularization_coef = 0.001  # Small coefficient to not dominate
        adapter_regularization_loss = adapter_regularization_coef * adapter_squared_norm

        # Total loss with adapter regularization
        total_loss = (
            policy_loss
            - entropy_coef * entropy
            + value_loss_coef * value_loss
            + adapter_regularization_loss
        )

        # Compute adapter norm for monitoring
        adapter_mean_norm = adapter_norm

        info_dict = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "mean_ratio": ratio_all.mean(),
            "adapter_norm": adapter_mean_norm,
            "advantages_mean": advantages_sequence.mean(),
            "advantages_std": advantages_sequence.std(),
            "advantages_max": jnp.max(jnp.abs(advantages_sequence)),
            "approx_kl": approx_kl,
        }

        return total_loss, info_dict

    # Compute gradients
    if hn_state is not None:
        # With hypernetwork: compute gradients for policy, hypernetwork, and critic
        (loss, info), grads = jax.value_and_grad(
            lambda p, h, c: loss_fn(p, h, c), argnums=(0, 1, 2), has_aux=True
        )(policy_state.params, hn_state.params, critic_state.params)

        policy_grads, hn_grads, critic_grads = grads

        hn_grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(hn_grads))
        )

        # NaN/Inf protection for hypernetwork grads
        hn_grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), hn_grads
        )
    else:
        # Without hypernetwork: only compute gradients for policy and critic
        (loss, info), grads = jax.value_and_grad(
            lambda p, c: loss_fn(p, None, c), argnums=(0, 1), has_aux=True
        )(policy_state.params, critic_state.params)

        policy_grads, critic_grads = grads
        hn_grads = None
        hn_grad_norm = 0.0

    # Compute gradient norms
    policy_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(policy_grads))
    )
    critic_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(critic_grads))
    )

    info["policy_grad_norm"] = policy_grad_norm
    info["hn_grad_norm"] = hn_grad_norm
    info["critic_grad_norm"] = critic_grad_norm

    # NaN/Inf protection
    policy_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), policy_grads
    )
    critic_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), critic_grads
    )

    # Update networks
    policy_state = policy_state.apply_gradients(grads=policy_grads)
    if hn_state is not None:
        hn_state = hn_state.apply_gradients(grads=hn_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)

    return policy_state, hn_state, critic_state, info


@partial(
    jax.jit, static_argnums=(15, 16, 17)
)  # num_agents, current_num_agents, max_agents
def _train_step_cash_gru(
    policy_state,
    critic_state,
    obs_sequence,  # (num_steps, num_actors, obs_dim)
    cash_capability_context_sequence,  # (num_steps, num_actors, cash_cap_dim)
    global_state_sequence,  # (num_steps, num_envs, global_state_dim)
    actions_sequence,  # (num_steps, num_actors, action_dim)
    old_log_probs_sequence,  # (num_steps, num_actors)
    advantages_sequence,  # (num_steps, num_actors)
    returns_sequence,  # (num_steps, num_actors)
    init_hidden_state,
    init_critic_hidden,
    dones_sequence,  # (num_steps, num_envs)
    clip_epsilon,
    entropy_coef,
    value_loss_coef,
    num_agents,
    current_num_agents,
    max_agents,
):
    """GRU PPO step for CASH with per-timestep hyper-decoder generation."""

    def loss_fn(policy_params, critic_params):
        num_steps = obs_sequence.shape[0]
        num_actors = obs_sequence.shape[1]
        inferred_num_envs = num_actors // num_agents

        dones_shifted = jnp.concatenate(
            [jnp.zeros((1, dones_sequence.shape[1]), dtype=bool), dones_sequence[:-1]],
            axis=0,
        )
        # Expand per-env done flags to per-actor without dynamic jnp.repeat.
        # dones_shifted: (T, E) -> (T, E, A) -> (T, E*A)
        dones_for_actor = jnp.broadcast_to(
            dones_shifted[:, :, None],
            (num_steps, inferred_num_envs, num_agents),
        ).reshape(num_steps, num_actors)

        policy_x = (obs_sequence, dones_for_actor, cash_capability_context_sequence)
        _, policy_output = shared_policy.apply(
            {"params": policy_params}, init_hidden_state, policy_x
        )
        mean_all, log_std_all = policy_output
        std_all = jnp.exp(log_std_all)

        mean_flat = mean_all.reshape(-1, mean_all.shape[-1])
        std_flat = std_all.reshape(-1, std_all.shape[-1])
        actions_flat = actions_sequence.reshape(-1, actions_sequence.shape[-1])

        base_dist = distrax.Normal(loc=mean_flat, scale=std_flat)
        tanh_bijector = distrax.Tanh()
        dist = distrax.Transformed(base_dist, tanh_bijector)
        new_log_probs_all = (
            dist.log_prob(actions_flat).sum(axis=-1).reshape(num_steps, num_actors)
        )
        entropy_all = base_dist.entropy().sum(axis=-1).reshape(num_steps, num_actors)

        new_log_probs_all = jnp.nan_to_num(
            new_log_probs_all, nan=-100.0, posinf=-100.0, neginf=-100.0
        )
        new_log_probs_all = jnp.clip(new_log_probs_all, -100.0, 100.0)

        log_prob_diff_all = new_log_probs_all - old_log_probs_sequence
        log_prob_diff_all = jnp.clip(log_prob_diff_all, -3.0, 3.0)
        ratio_all = jnp.exp(log_prob_diff_all)
        clipped_ratio_all = jnp.clip(ratio_all, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surr1_all = ratio_all * advantages_sequence
        surr2_all = clipped_ratio_all * advantages_sequence
        policy_loss_all = -jnp.minimum(surr1_all, surr2_all)

        _, values_all = critic.apply(
            {"params": critic_params},
            init_critic_hidden,
            (global_state_sequence, dones_sequence),
        )
        values_flat = values_all.reshape(values_all.shape[0], -1)
        value_loss_all = jnp.square(values_flat - returns_sequence)

        agent_indices = jnp.arange(num_agents, dtype=jnp.int32)
        agent_mask_template = agent_indices < current_num_agents
        valid_agent_mask = jnp.broadcast_to(
            agent_mask_template[None, None, :],
            (num_steps, inferred_num_envs, num_agents),
        ).reshape(num_steps, num_actors)

        policy_loss = (policy_loss_all * valid_agent_mask).sum() / jnp.maximum(
            valid_agent_mask.sum(), 1.0
        )
        value_loss = (value_loss_all * valid_agent_mask).sum() / jnp.maximum(
            valid_agent_mask.sum(), 1.0
        )
        entropy = (entropy_all * valid_agent_mask).sum() / jnp.maximum(
            valid_agent_mask.sum(), 1.0
        )

        total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
        info_dict = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((new_log_probs_all - old_log_probs_sequence) ** 2),
            "mean_ratio": jnp.mean(ratio_all),
            "advantages_mean": jnp.mean(advantages_sequence),
            "advantages_std": jnp.std(advantages_sequence),
            "advantages_max": jnp.max(jnp.abs(advantages_sequence)),
            "mean_std": jnp.mean(std_all),
            "min_std": jnp.min(std_all),
            "max_std": jnp.max(std_all),
        }
        return total_loss, info_dict

    (loss, info), grads = jax.value_and_grad(
        lambda p, c: loss_fn(p, c), argnums=(0, 1), has_aux=True
    )(policy_state.params, critic_state.params)
    policy_grads, critic_grads = grads

    policy_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(policy_grads))
    )
    critic_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(critic_grads))
    )
    info["policy_grad_norm"] = policy_grad_norm
    info["hn_grad_norm"] = jnp.asarray(0.0)
    info["critic_grad_norm"] = critic_grad_norm

    policy_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), policy_grads
    )
    critic_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), critic_grads
    )

    policy_state = policy_state.apply_gradients(grads=policy_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)
    return policy_state, critic_state, info


@partial(
    jax.jit, static_argnums=(13, 14, 15, 16)
)  # num_agents, current_num_agents, max_agents, is_discrete
def _train_step_no_hn(
    policy_state,
    critic_state,
    obs_batch,
    global_state_batch,
    actions_batch,
    old_log_probs_batch,
    advantages_batch,
    returns_batch,
    adapters_dict,
    action_masks_batch,
    clip_epsilon,
    entropy_coef,
    value_loss_coef,
    num_agents,
    current_num_agents,
    max_agents,
    is_discrete=False,
):
    """
    Perform one MAPPO training step for policy and centralized critic (no hypernetwork).
    Used when use_hypernetwork=False.

    Args:
        policy_state: Policy training state
        critic_state: Centralized critic training state
        obs_batch: Batch of local observations
        global_state_batch: Batch of global states (all agents' obs concatenated)
        actions_batch: Batch of actions taken
        old_log_probs_batch: Batch of old log probabilities
        old_values_batch: Batch of old value predictions (for value clipping)
        advantages_batch: Batch of advantages (GAE)
        returns_batch: Batch of returns
        adapters_dict: Dictionary of zero adapters (no LoRA)
        clip_epsilon: PPO clipping parameter
        entropy_coef: Entropy coefficient
        value_loss_coef: Value loss coefficient
        num_agents: Number of agents per environment
        current_num_agents: Current number of active agents (for loss masking)
        max_agents: Maximum number of agents (fixed capacity)

    Returns:
        Updated policy_state, critic_state, and loss info
    """

    def loss_fn(policy_params, critic_params):
        # No hypernetwork - use zero adapters

        # Check if policy is GRU-based
        is_gru_policy = hasattr(shared_policy, "gru_hidden_dim")

        # Check if using discrete actions (SMAX)
        if is_discrete:
            # SMAX: Policy outputs logits for categorical distribution
            if is_gru_policy:
                # For GRU policy: need hidden state and proper input format
                batch_size = obs_batch.shape[0]
                dummy_hidden = jnp.zeros((batch_size, shared_policy.gru_hidden_dim))
                # For single-step (not sequences), obs_batch is (batch, obs_dim)
                # Need to add time dimension for GRU: (1, batch, obs_dim)
                obs_seq = obs_batch[None, ...]  # Add time dimension
                dones_seq = jnp.zeros((1, batch_size), dtype=bool)
                avail_seq = (
                    action_masks_batch[None, ...]
                    if action_masks_batch is not None
                    else None
                )
                policy_x = (obs_seq, dones_seq, avail_seq)
                _, policy_out = shared_policy.apply(
                    {"params": policy_params}, dummy_hidden, policy_x, adapters_dict
                )
                # Remove time dimension: (1, batch, action_dim) -> (batch, action_dim)
                logits = policy_out[0]
            else:
                logits, _ = shared_policy.apply(
                    {"params": policy_params}, obs_batch, adapters_dict
                )

            # CRITICAL: Apply action masks if provided
            if action_masks_batch is not None:
                # Mask invalid actions by setting their logits to -inf
                masked_logits = jnp.where(
                    action_masks_batch.astype(bool),
                    logits,
                    jnp.full_like(logits, -1e10),
                )
            else:
                masked_logits = logits

            # Create categorical distribution with masked logits
            dist = distrax.Categorical(logits=masked_logits)
            # Compute log probabilities for discrete actions
            new_log_probs = dist.log_prob(actions_batch)
            # Entropy for discrete distribution
            entropy_per_agent = dist.entropy()
        else:
            # Continuous actions: Get mean and log_std for current policy (PRE-TANH)
            if is_gru_policy:
                # For GRU policy: need hidden state and proper input format
                batch_size = obs_batch.shape[0]
                dummy_hidden = jnp.zeros((batch_size, shared_policy.gru_hidden_dim))
                # For single-step (not sequences), obs_batch is (batch, obs_dim)
                # Need to add time dimension for GRU: (1, batch, obs_dim)
                obs_seq = obs_batch[None, ...]  # Add time dimension
                dones_seq = jnp.zeros((1, batch_size), dtype=bool)
                avail_seq = jnp.ones(
                    (1, batch_size, shared_policy.action_dim)
                )  # Dummy for continuous
                policy_x = (obs_seq, dones_seq, avail_seq)
                _, policy_out = shared_policy.apply(
                    {"params": policy_params}, dummy_hidden, policy_x, adapters_dict
                )
                # Remove time dimension: (1, batch, action_dim) -> (batch, action_dim)
                mean, log_std = policy_out
                mean = mean[0]
                log_std = log_std[0]
            else:
                mean, log_std = shared_policy.apply(
                    {"params": policy_params}, obs_batch, adapters_dict
                )
            std = jnp.exp(log_std)

            # Save unclamped std for monitoring/reporting
            std_unclamped = std

            # NOTE: Unlike MLP policy, GRU policy doesn't clamp std to minimum 0.3
            # This matches the hypernetwork GRU training path behavior
            # The policy's own log_std clipping (log_std_min=-2.0) provides: min std = exp(-2) ≈ 0.135

            # Create the SAME tanh-transformed distribution as in policy sampling
            base_dist = distrax.Normal(mean, std)
            tanh_bijector = distrax.Tanh()
            dist = distrax.Transformed(base_dist, tanh_bijector)

            # Compute log probabilities without clipping (CRITICAL FIX)
            # Action clipping should only happen during environment interaction, not in loss
            new_log_probs = dist.log_prob(actions_batch).sum(axis=-1)

            # IMPORTANT: Entropy should be from the BASE distribution (pre-tanh)
            # We want to encourage exploration in the Gaussian space, not the tanh-squashed space
            entropy_per_agent = base_dist.entropy().sum(axis=-1)  # (batch_size,)

        # NaN protection for log probabilities
        new_log_probs = jnp.nan_to_num(
            new_log_probs, nan=-100.0, posinf=-100.0, neginf=-100.0
        )
        new_log_probs = jnp.clip(new_log_probs, -100.0, 100.0)

        # CRITICAL: Clip log prob difference before computing ratio
        log_prob_diff = new_log_probs - old_log_probs_batch
        log_prob_diff = jnp.clip(log_prob_diff, -3.0, 3.0)

        # Compute probability ratio
        ratio = jnp.exp(log_prob_diff)

        # PPO clipped objective per agent
        surr1 = ratio * advantages_batch
        surr2 = (
            jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_batch
        )
        policy_loss_per_agent = -jnp.minimum(surr1, surr2)  # (batch_size,)

        # CRITICAL: Create valid agent mask for loss masking
        # Shape: (batch_size,) - True for real agents, False for padded agents
        batch_size_total = obs_batch.shape[0]

        # Create mask: [True, True, ..., False, False] for each environment
        # num_agents here is actually env_num_agents (passed from caller)
        agent_mask_template = jnp.arange(num_agents) < current_num_agents
        # Tile for all timesteps and environments
        valid_agent_mask = jnp.tile(agent_mask_template, batch_size_total // num_agents)

        # Apply mask to policy loss
        policy_loss_masked = policy_loss_per_agent * valid_agent_mask
        policy_loss = policy_loss_masked.sum() / jnp.maximum(
            valid_agent_mask.sum(), 1.0
        )

        # Entropy bonus for exploration
        # Apply mask to entropy
        entropy_masked = entropy_per_agent * valid_agent_mask
        entropy = entropy_masked.sum() / jnp.maximum(valid_agent_mask.sum(), 1.0)

        # Add small regularization to prevent complete collapse (continuous actions only)
        if not is_discrete:
            policy_regularization = 0.001 * jnp.mean(jnp.square(mean))
            policy_loss = policy_loss + policy_regularization

        # Critic loss (MSE between value predictions and returns)
        # global_state_batch is (batch, global_state_dim) where batch = num_envs * num_agents
        # We need to reshape to (num_envs, global_state_dim) since critic operates on env level
        batch_size = global_state_batch.shape[0]
        num_envs = batch_size // num_agents
        # Take every num_agents-th entry (they all have the same global state per env)
        global_state_envs = global_state_batch[
            ::num_agents
        ]  # (num_envs, global_state_dim)

        if is_gru_policy:
            # For RNN critic: need hidden state and (global_state, dones) tuple
            dummy_critic_hidden = jnp.zeros((num_envs, critic.gru_hidden_dim))
            # Add time dimension for RNN: (1, num_envs, global_state_dim)
            global_state_seq = global_state_envs[None, ...]
            dones_seq = jnp.zeros((1, num_envs), dtype=bool)
            critic_x = (global_state_seq, dones_seq)
            _, values_seq = critic.apply(
                {"params": critic_params}, dummy_critic_hidden, critic_x
            )
            # Remove time dimension: (1, num_envs, num_agents) -> (num_envs, num_agents)
            values = jnp.squeeze(values_seq, axis=0)
        else:
            # MLP critic always outputs (num_envs, num_agents) regardless of shared mode
            values = critic.apply({"params": critic_params}, global_state_envs)

        # Flatten values to match returns_batch shape
        # values: (num_envs, num_agents) -> (num_envs * num_agents,)
        values_flat = values.reshape(-1)

        # Compute value loss per agent
        value_loss_per_agent = (values_flat - returns_batch) ** 2

        # Apply mask to value loss
        value_loss_masked = value_loss_per_agent * valid_agent_mask
        value_loss = value_loss_masked.sum() / jnp.maximum(valid_agent_mask.sum(), 1.0)

        # Total loss
        total_loss = policy_loss - entropy_coef * entropy + value_loss_coef * value_loss

        # Build return info dict (handle discrete vs continuous)
        info_dict = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((new_log_probs - old_log_probs_batch) ** 2),
            "mean_ratio": jnp.mean(ratio),
            "advantages_mean": jnp.mean(advantages_batch),
            "advantages_std": jnp.std(advantages_batch),
            "advantages_max": jnp.max(jnp.abs(advantages_batch)),
            "adapter_norm": 0.0,  # No adapters
        }

        # Add std stats only for continuous actions
        if not is_discrete:
            info_dict["mean_std"] = jnp.mean(std_unclamped)
            info_dict["min_std"] = jnp.min(std_unclamped)
            info_dict["max_std"] = jnp.max(std_unclamped)
        else:
            # For discrete actions, add dummy values to maintain consistency
            info_dict["mean_std"] = 0.0
            info_dict["min_std"] = 0.0
            info_dict["max_std"] = 0.0

        return total_loss, info_dict

    # Compute gradients for policy and critic
    (loss, info), grads = jax.value_and_grad(
        lambda p, c: loss_fn(p, c), argnums=(0, 1), has_aux=True
    )(policy_state.params, critic_state.params)

    policy_grads, critic_grads = grads

    # Compute gradient norms for debugging
    policy_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(policy_grads))
    )
    critic_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(critic_grads))
    )

    # Add to info
    info["policy_grad_norm"] = policy_grad_norm
    info["hn_grad_norm"] = 0.0  # No hypernetwork
    info["critic_grad_norm"] = critic_grad_norm

    # NaN protection: replace NaN gradients with zeros
    policy_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), policy_grads
    )
    critic_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), critic_grads
    )

    # Update both networks
    policy_state = policy_state.apply_gradients(grads=policy_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)

    return policy_state, critic_state, info


def _train_step_dico(
    policy_state,
    critic_state,
    obs_batch,
    global_state_batch,
    actions_batch,
    old_log_probs_batch,
    advantages_batch,
    returns_batch,
    agent_ids_batch,
    diversity_scaling,
    action_masks_batch,  # Added for SMAX action masking
    clip_epsilon,
    entropy_coef,
    value_loss_coef,
    num_agents,
    current_num_agents,
    max_agents,
    is_discrete=False,  # Added for SMAX discrete actions
):
    """
    Perform one MAPPO training step for DiCo policy and centralized critic.

    Args:
        policy_state: Policy training state
        critic_state: Centralized critic training state
        obs_batch: Batch of local observations
        global_state_batch: Batch of global states (all agents' obs concatenated)
        actions_batch: Batch of actions taken
        old_log_probs_batch: Batch of old log probabilities
        advantages_batch: Batch of advantages (GAE)
        returns_batch: Batch of returns
        agent_ids_batch: Agent IDs for routing (batch_size,)
        diversity_scaling: Diversity scaling factor
        clip_epsilon: PPO clipping parameter
        entropy_coef: Entropy coefficient
        value_loss_coef: Value loss coefficient
        num_agents: Number of agents per environment
        current_num_agents: Current number of active agents (for loss masking)
        max_agents: Maximum number of agents (fixed capacity)

    Returns:
        Updated policy_state, critic_state, and loss info
    """

    def loss_fn(policy_params, critic_params):
        if is_discrete:
            # SMAX: Policy outputs logits for categorical distribution
            logits, _ = shared_policy.apply(
                {"params": policy_params}, obs_batch, agent_ids_batch, diversity_scaling
            )

            # Apply action masks if provided
            if action_masks_batch is not None:
                masked_logits = jnp.where(
                    action_masks_batch.astype(bool),
                    logits,
                    jnp.full_like(logits, -1e10),
                )
            else:
                masked_logits = logits

            # Create categorical distribution
            dist = distrax.Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(actions_batch)
            entropy_per_agent = dist.entropy()
        else:
            # Get mean and log_std from DiCo policy
            mean, log_std = shared_policy.apply(
                {"params": policy_params}, obs_batch, agent_ids_batch, diversity_scaling
            )

            # std already computed via biased_softplus_1.0 in policy
            std = jnp.exp(log_std)

            # Create tanh-transformed distribution
            base_dist = distrax.Normal(mean, std)
            tanh_bijector = distrax.Tanh()
            dist = distrax.Transformed(base_dist, tanh_bijector)

            # Clip actions to valid range before computing log prob
            actions_clipped = jnp.clip(actions_batch, -0.9999, 0.9999)

            # Compute log probabilities
            new_log_probs = dist.log_prob(actions_clipped).sum(axis=-1)

            # NaN protection for log probs
            new_log_probs = jnp.nan_to_num(
                new_log_probs, nan=-10.0, posinf=-10.0, neginf=-10.0
            )

            entropy_per_agent = base_dist.entropy().sum(axis=-1)

        # Clip log prob difference
        log_prob_diff = new_log_probs - old_log_probs_batch
        log_prob_diff = jnp.clip(log_prob_diff, -3.0, 3.0)

        # Compute probability ratio
        ratio = jnp.exp(log_prob_diff)
        # NaN protection for ratio
        ratio = jnp.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

        # PPO clipped objective per agent
        surr1 = ratio * advantages_batch
        surr2 = (
            jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_batch
        )
        policy_loss_per_agent = -jnp.minimum(surr1, surr2)

        # NaN protection for policy loss
        policy_loss_per_agent = jnp.nan_to_num(
            policy_loss_per_agent, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Create valid agent mask
        batch_size_total = obs_batch.shape[0]
        agent_mask_template = jnp.arange(num_agents) < current_num_agents
        valid_agent_mask = jnp.tile(agent_mask_template, batch_size_total // num_agents)

        # Apply mask to policy loss
        policy_loss_masked = policy_loss_per_agent * valid_agent_mask
        policy_loss = policy_loss_masked.sum() / jnp.maximum(
            valid_agent_mask.sum(), 1.0
        )

        # Entropy bonus (already computed above)
        entropy_masked = entropy_per_agent * valid_agent_mask
        entropy = entropy_masked.sum() / jnp.maximum(valid_agent_mask.sum(), 1.0)

        # Critic loss with NaN protection
        # Critic always outputs (num_envs_batch, num_agents) regardless of shared/per-agent mode
        values = critic.apply({"params": critic_params}, global_state_batch)
        # Clip values to prevent extreme predictions
        values = jnp.clip(values, -100.0, 100.0)
        # Replace any NaN/inf in values
        values = jnp.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)

        # Flatten values to match returns_batch shape
        # values: (num_envs_batch, num_agents) -> (num_envs_batch * num_agents,)
        values_flat = values.reshape(-1)

        # Clip returns to prevent extreme targets
        returns_clipped = jnp.clip(returns_batch, -100.0, 100.0)
        returns_clipped = jnp.nan_to_num(
            returns_clipped, nan=0.0, posinf=100.0, neginf=-100.0
        )

        value_loss_per_agent = (values_flat - returns_clipped) ** 2
        value_loss_masked = value_loss_per_agent * valid_agent_mask
        value_loss = value_loss_masked.sum() / jnp.maximum(valid_agent_mask.sum(), 1.0)

        # Total loss
        total_loss = policy_loss - entropy_coef * entropy + value_loss_coef * value_loss

        return total_loss, {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((new_log_probs - old_log_probs_batch) ** 2),
            "mean_std": jnp.mean(std),
            "min_std": jnp.min(std),
            "max_std": jnp.max(std),
            "mean_ratio": jnp.mean(ratio),
            "advantages_mean": jnp.mean(advantages_batch),
            "advantages_std": jnp.std(advantages_batch),
            "advantages_max": jnp.max(jnp.abs(advantages_batch)),
            "adapter_norm": jnp.asarray(0.0),  # N/A for DiCo
        }

    # Compute gradients
    (loss, info), grads = jax.value_and_grad(
        lambda p, c: loss_fn(p, c), argnums=(0, 1), has_aux=True
    )(policy_state.params, critic_state.params)

    policy_grads, critic_grads = grads

    # Compute gradient norms
    policy_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(policy_grads))
    )
    critic_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(critic_grads))
    )

    # Check for NaN in gradients (before filtering). tree_reduce stays inside
    # JAX so this function is safe to call under lax.scan / jax.jit.
    _false = jnp.array(False)
    policy_has_nan = jax.tree_util.tree_reduce(
        lambda acc, g: acc | jnp.isnan(g).any(), policy_grads, _false
    )
    critic_has_nan = jax.tree_util.tree_reduce(
        lambda acc, g: acc | jnp.isnan(g).any(), critic_grads, _false
    )

    info["policy_grad_norm"] = policy_grad_norm
    info["hn_grad_norm"] = jnp.asarray(0.0)  # No hypernetwork
    info["critic_grad_norm"] = critic_grad_norm
    info["policy_grad_has_nan"] = policy_has_nan
    info["critic_grad_has_nan"] = critic_has_nan

    # NaN protection: replace NaN gradients with zeros
    policy_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), policy_grads
    )
    critic_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g), critic_grads
    )

    # Update networks
    policy_state = policy_state.apply_gradients(grads=policy_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)

    return policy_state, critic_state, info


# ============================================================================
# Scanned PPO update (DiCo path)
#
# The Python double-loop `for ppo_epoch × for mb_idx` performed
# ppo_epochs * num_minibatches (= 320 for football config) separate XLA
# launches per episode. Each launch pays host→device dispatch overhead that
# dominates the PPO phase on fast GPUs. This helper runs the whole PPO update
# inside a single jit'd lax.scan, keeping state on-device.
#
# Assumptions (matched to the DiCo-football path):
#   - DiCo (per-agent policy heads, not a hypernetwork)
#   - Continuous actions (is_discrete=False), so action_masks is not used
#   - No minibatch shuffling (create_minibatches_dico doesn't support it
#     because per-agent indices are coupled to global-state indices)
#   - batch_size % num_minibatches == 0 and
#     global_state_batch_size % num_minibatches == 0
# ============================================================================


def _ppo_update_dico_scan(
    policy_state,
    critic_state,
    obs_batch,
    global_state_batch,
    actions_batch,
    old_log_probs_batch,
    advantages_batch,
    returns_batch,
    agent_ids_batch,
    diversity_scaling,
    clip_epsilon,
    entropy_coef,
    value_loss_coef,
    num_minibatches,
    ppo_epochs,
    num_agents,
    current_num_agents,
    max_agents,
    is_discrete,
):
    batch_size = obs_batch.shape[0]
    mb_size = batch_size // num_minibatches
    gs_batch_size = global_state_batch.shape[0]
    gs_mb_size = gs_batch_size // num_minibatches

    def _split(x, n_chunks, chunk):
        return x.reshape((n_chunks, chunk) + x.shape[1:])

    obs_m = _split(obs_batch, num_minibatches, mb_size)
    acts_m = _split(actions_batch, num_minibatches, mb_size)
    old_lp_m = _split(old_log_probs_batch, num_minibatches, mb_size)
    adv_m = _split(advantages_batch, num_minibatches, mb_size)
    ret_m = _split(returns_batch, num_minibatches, mb_size)
    aids_m = _split(agent_ids_batch, num_minibatches, mb_size)
    gs_m = _split(global_state_batch, num_minibatches, gs_mb_size)

    def mb_step(carry, batch):
        ps, cs = carry
        obs_i, gs_i, acts_i, old_lp_i, adv_i, ret_i, aids_i = batch
        ps, cs, info = _train_step_dico(
            ps,
            cs,
            obs_i,
            gs_i,
            acts_i,
            old_lp_i,
            adv_i,
            ret_i,
            aids_i,
            diversity_scaling,
            None,  # action_masks (continuous path)
            clip_epsilon,
            entropy_coef,
            value_loss_coef,
            num_agents,
            current_num_agents,
            max_agents,
            is_discrete=is_discrete,
        )
        return (ps, cs), info

    def epoch_step(carry, _):
        ps, cs = carry
        batches = (obs_m, gs_m, acts_m, old_lp_m, adv_m, ret_m, aids_m)
        (ps, cs), infos = jax.lax.scan(mb_step, (ps, cs), batches)
        # Mean across the num_minibatches axis so each epoch exposes one info dict.
        infos_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), infos)
        return (ps, cs), infos_mean

    (policy_state, critic_state), epoch_infos = jax.lax.scan(
        epoch_step, (policy_state, critic_state), xs=None, length=ppo_epochs
    )
    # Report the last epoch's averaged info (matches the old loop's end-state).
    loss_info = jax.tree_util.tree_map(lambda x: x[-1], epoch_infos)
    return policy_state, critic_state, loss_info


_ppo_update_dico_scan = jax.jit(
    _ppo_update_dico_scan,
    static_argnames=(
        "num_minibatches",
        "ppo_epochs",
        "num_agents",
        "current_num_agents",
        "max_agents",
        "is_discrete",
    ),
)


# ============================================================================
# Mini-batch Creation
# ============================================================================


def create_minibatches(data_dict, num_minibatches, shuffle=True, rng_key=None):
    """
    Split training data into mini-batches.

    Args:
        data_dict: Dictionary of training data arrays (obs, actions, etc.)
        num_minibatches: Number of mini-batches to create
        shuffle: Whether to shuffle data before splitting
        rng_key: JAX random key for shuffling

    Returns:
        List of data_dict mini-batches
    """
    # Get batch size from any data array
    batch_size = next(iter(data_dict.values())).shape[0]
    minibatch_size = batch_size // num_minibatches

    if batch_size % num_minibatches != 0:
        raise ValueError(
            f"Batch size {batch_size} must be evenly divisible by num_minibatches {num_minibatches}"
        )

    # Create indices
    indices = jnp.arange(batch_size)

    # Shuffle if requested
    if shuffle and rng_key is not None:
        indices = jax.random.permutation(rng_key, indices)

    # Split into mini-batches
    minibatches = []
    for i in range(num_minibatches):
        start_idx = i * minibatch_size
        end_idx = start_idx + minibatch_size
        mb_indices = indices[start_idx:end_idx]

        # Create mini-batch by indexing each array in data_dict
        minibatch = {}
        for key, value in data_dict.items():
            if value is not None:
                minibatch[key] = value[mb_indices]
            else:
                minibatch[key] = None
        minibatches.append(minibatch)

    return minibatches


def create_minibatches_dico(
    data_dict, num_minibatches, num_agents, shuffle=True, rng_key=None
):
    """
    Split DICO training data into mini-batches with special handling for global state.

    For DICO, most arrays have shape (batch_size, ...) where batch_size = num_steps * num_envs * num_agents,
    but global_state has shape (num_envs_steps, ...) where num_envs_steps = num_steps * num_envs.

    We maintain the correspondence: per-agent index i maps to global-state index i // num_agents.

    Args:
        data_dict: Dictionary of training data arrays
        num_minibatches: Number of mini-batches to create
        num_agents: Number of agents per environment
        shuffle: Whether to shuffle data before splitting (currently not supported to maintain correspondence)
        rng_key: JAX random key for shuffling

    Returns:
        List of data_dict mini-batches
    """
    # Get batch size from obs (which has per-agent entries)
    batch_size = data_dict["obs"].shape[0]
    minibatch_size = batch_size // num_minibatches

    # Global state batch size
    global_state_batch_size = data_dict["global_state"].shape[0]
    global_state_minibatch_size = global_state_batch_size // num_minibatches

    if batch_size % num_minibatches != 0:
        raise ValueError(
            f"Batch size {batch_size} must be evenly divisible by num_minibatches {num_minibatches}"
        )

    if global_state_batch_size % num_minibatches != 0:
        raise ValueError(
            f"Global state batch size {global_state_batch_size} must be evenly divisible by num_minibatches {num_minibatches}"
        )

    # Verify the expected relationship: batch_size = global_state_batch_size * num_agents
    if batch_size != global_state_batch_size * num_agents:
        raise ValueError(
            f"Batch size mismatch: expected {global_state_batch_size * num_agents}, got {batch_size}"
        )

    # For DICO, we typically don't shuffle to maintain stability
    # If shuffle is requested, we'd need to shuffle at the environment level
    # to maintain the per-agent / global-state correspondence
    if shuffle and rng_key is not None:
        print(
            "Warning: Shuffling in DICO minibatches not yet implemented. Skipping shuffle."
        )

    # Split into mini-batches by taking contiguous chunks
    # This maintains the correspondence: per-agent indices [i:j] correspond to
    # global-state indices [i//num_agents : j//num_agents]
    minibatches = []
    for i in range(num_minibatches):
        # Per-agent data slicing
        start_idx = i * minibatch_size
        end_idx = start_idx + minibatch_size

        # Global state slicing (divide by num_agents to get environment indices)
        gs_start_idx = start_idx // num_agents
        gs_end_idx = end_idx // num_agents

        # Create mini-batch
        minibatch = {}
        for key, value in data_dict.items():
            if value is None:
                minibatch[key] = None
            elif key == "global_state":
                # Use global state indices
                minibatch[key] = value[gs_start_idx:gs_end_idx]
            else:
                # Use per-agent indices
                minibatch[key] = value[start_idx:end_idx]

        minibatches.append(minibatch)

    return minibatches


def create_minibatches_sequential(
    obs_batch,
    global_state_batch,
    actions_batch,
    old_log_probs_batch,
    advantages_batch,
    returns_batch,
    task_batch,
    context_batch,
    lidar_batch,
    food_position_batch,
    agent_position_batch,
    target_snd_batch,
    mask_batch,
    action_masks_batch,
    dones_batch,
    init_hidden_batch,
    init_critic_hidden_batch,
    num_minibatches,
    num_steps,
    num_envs,
    num_agents,
    shuffle=True,
    rng_key=None,
):
    """
    Split sequential (GRU) training data into mini-batches while preserving temporal structure.

    For GRU policies, we split across the environment dimension, not the time dimension.
    Each mini-batch contains all timesteps but only a subset of environments.

    Args:
        All *_batch args: Sequential data with shape (num_steps, batch_size, ...)
        init_hidden_batch: Initial policy hidden states (num_envs * num_agents, hidden_dim)
        init_critic_hidden_batch: Initial critic hidden states (num_envs, hidden_dim)
        num_minibatches: Number of mini-batches
        num_steps: Number of timesteps in sequence
        num_envs: Number of environments
        num_agents: Number of agents per environment
        shuffle: Whether to shuffle environments before splitting
        rng_key: JAX random key for shuffling

    Returns:
        List of tuples containing mini-batch data including sliced hidden states
    """
    # For GRU, batch_size = num_envs * num_agents
    # We want to split across environments while keeping sequences intact
    if num_envs % num_minibatches != 0:
        raise ValueError(
            f"Number of environments {num_envs} must be evenly divisible by num_minibatches {num_minibatches}"
        )

    envs_per_minibatch = num_envs // num_minibatches
    agents_per_minibatch = envs_per_minibatch * num_agents

    # Create environment indices
    env_indices = jnp.arange(num_envs)

    # Shuffle environments if requested
    if shuffle and rng_key is not None:
        env_indices = jax.random.permutation(rng_key, env_indices)

    minibatches = []
    for i in range(num_minibatches):
        start_env = i * envs_per_minibatch
        end_env = start_env + envs_per_minibatch
        mb_env_indices = env_indices[start_env:end_env]

        # For agent-level data (obs, actions, etc.), we need to select agents from selected envs
        # Shape: (num_steps, num_envs * num_agents, ...) -> (num_steps, envs_per_minibatch * num_agents, ...)
        # OPTIMIZED: Vectorized agent index generation
        agent_offsets = jnp.arange(num_agents)
        agent_indices = (
            mb_env_indices[:, None] * num_agents + agent_offsets[None, :]
        ).flatten()

        # Index each array
        mb_obs = obs_batch[:, agent_indices] if obs_batch is not None else None
        mb_actions = (
            actions_batch[:, agent_indices] if actions_batch is not None else None
        )
        mb_old_log_probs = (
            old_log_probs_batch[:, agent_indices]
            if old_log_probs_batch is not None
            else None
        )
        mb_advantages = (
            advantages_batch[:, agent_indices] if advantages_batch is not None else None
        )
        mb_returns = (
            returns_batch[:, agent_indices] if returns_batch is not None else None
        )
        mb_action_masks = (
            action_masks_batch[:, agent_indices]
            if action_masks_batch is not None
            else None
        )

        # For global state and dones, index by environment
        mb_global_state = (
            global_state_batch[:, mb_env_indices]
            if global_state_batch is not None
            else None
        )
        mb_dones = dones_batch[:, mb_env_indices] if dones_batch is not None else None

        # For task/context/lidar, these have shape (num_steps, num_envs * num_agents, feature_dim)
        mb_task = task_batch[:, agent_indices] if task_batch is not None else None
        mb_context = (
            context_batch[:, agent_indices] if context_batch is not None else None
        )
        mb_lidar = lidar_batch[:, agent_indices] if lidar_batch is not None else None
        mb_food_positions = (
            food_position_batch[:, agent_indices]
            if food_position_batch is not None
            else None
        )
        mb_agent_positions = (
            agent_position_batch[:, agent_indices]
            if agent_position_batch is not None
            else None
        )
        mb_target_snd = (
            target_snd_batch[:, agent_indices] if target_snd_batch is not None else None
        )

        # For mask, shape is (num_steps, num_envs, 1, num_agents, num_agents)
        # Index by environment
        mb_mask = mask_batch[:, mb_env_indices] if mask_batch is not None else None

        # For hidden states, slice by agent indices and environment indices
        mb_init_hidden = (
            init_hidden_batch[agent_indices] if init_hidden_batch is not None else None
        )
        mb_init_critic_hidden = (
            init_critic_hidden_batch[mb_env_indices]
            if init_critic_hidden_batch is not None
            else None
        )

        minibatches.append(
            (
                mb_obs,
                mb_global_state,
                mb_actions,
                mb_old_log_probs,
                mb_advantages,
                mb_returns,
                mb_task,
                mb_context,
                mb_lidar,
                mb_food_positions,
                mb_agent_positions,
                mb_target_snd,
                mb_mask,
                mb_action_masks,
                mb_dones,
                mb_init_hidden,
                mb_init_critic_hidden,
            )
        )

    return minibatches


# ============================================================================
# Training Configuration
# ============================================================================


def create_train_state(
    model, params, config, max_grad_norm=1.0, lr_key="learning_rate"
):
    """Create a training state with optimizer and gradient clipping.

    Args:
        model: The model to create state for
        params: Model parameters
        config: Configuration dictionary
        max_grad_norm: Maximum gradient norm for clipping
        lr_key: Key in config["optimizer"] for learning rate (allows separate LRs)
    """
    lr = config["optimizer"][lr_key]
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),  # Clip gradients
        optax.adam(
            learning_rate=lr,
            b1=config["optimizer"]["beta1"],
            b2=config["optimizer"]["beta2"],
            eps=config["optimizer"]["eps"],
        ),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# ============================================================================
# Main Training Script
# ============================================================================


def main():
    # ========================================================================
    # Parse Arguments and Load Config
    # ========================================================================
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        config_path = Path(__file__).parent / "config.yaml"

    config = load_config(config_path)
    config = override_config(config, args)

    # ========================================================================
    # Configure JAX based on debugging/profiling flags
    # ========================================================================
    if args.disable_jit:
        print("WARNING: JIT compilation disabled - training will be VERY slow!")
        jax.config.update("jax_disable_jit", True)

    # Profiling setup
    if args.jax_profile:
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        print(f"JAX profiling enabled - traces will be saved to: {profile_dir}")
        print(
            "IMPORTANT: Profiling adds significant overhead. First episode will be slowest."
        )
        print(f"To view traces: tensorboard --logdir={profile_dir}")

    # Debug: Show which config file was loaded
    print(f"\nLoaded config from: {config_path.absolute()}")
    if config["env"].get("scenario_name") == "reverse_transport":
        print(
            f"Dynamic env changes: {config['env'].get('use_dynamic_env_changes', False)}"
        )
        print(f"Env change interval: {config['env'].get('env_change_interval', 0)}")
        print(f"Env change type: {config['env'].get('env_change_type', 'N/A')}\n")

    # Print configuration
    print("=" * 80)
    print("HyperLoRA Training Configuration")
    print("=" * 80)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 80)

    # ========================================================================
    # Setup Logging and Checkpointing
    # ========================================================================
    if not args.no_logging:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{config['experiment']['name']}_{timestamp}"
        log_dir = Path(config["logging"]["log_dir"]) / exp_name
        checkpoint_dir = Path(config["logging"]["checkpoint_dir"]) / exp_name

        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save config to log directory
        with open(log_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Also save config to checkpoint directory for deployment/evaluation
        with open(checkpoint_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Logging to: {log_dir}")
        print(f"Checkpoints to: {checkpoint_dir}")

    # ========================================================================
    # Setup local logger (JSONL in <log_dir>/metrics.jsonl + figs under figs/)
    # wandb has been disabled; flag is kept for compatibility with run scripts.
    # ========================================================================
    use_wandb = not args.no_logging
    if use_wandb:
        wandb_name = args.wandb_name or exp_name
        wandb.init(
            project=config["logging"].get("wandb_project"),
            entity=config["logging"].get("wandb_entity"),
            name=wandb_name,
            config=config,
            tags=config["experiment"].get("tags", []),
            dir=str(log_dir),
        )
        print(f"Local logger initialized: {wandb.run.url}")

        if not args.no_logging:
            print(f"Logging to: {log_dir}")
            print(f"Checkpoints to: {checkpoint_dir}")

    # ========================================================================
    # Setup Device (CUDA/CPU)
    # ========================================================================
    use_cuda = config["device"].get("use_cuda", False)
    cuda_device_id = config["device"].get("cuda_device", 0)

    # Configure JAX device
    if use_cuda:
        try:
            gpu_devices = jax.devices("gpu")
            if not gpu_devices:
                print(
                    "WARNING: CUDA requested but no GPU available for JAX. Falling back to CPU."
                )
                use_cuda = False
                jax_device = jax.devices("cpu")[0]
            else:
                jax_device = gpu_devices[cuda_device_id]
                print(f"JAX using GPU: {jax_device}")
        except RuntimeError:
            print(
                "WARNING: CUDA requested but GPU backend not available for JAX. Falling back to CPU."
            )
            use_cuda = False
            jax_device = jax.devices("cpu")[0]
    else:
        jax_device = jax.devices("cpu")[0]
        print(f"JAX using CPU: {jax_device}")

    # Configure PyTorch device for VMAS environment
    # VMAS is optimized for CPU - always use CPU regardless of JAX device
    torch_device = "cpu"
    print(f"PyTorch/VMAS using device: {torch_device} (optimized for CPU)")
    if use_cuda:
        print(f"  (JAX will still use GPU: {jax_device})")

    # ========================================================================
    # Set Random Seeds
    # ========================================================================
    seed = config["training"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ========================================================================
    # Extract Hyperparameters from Config
    # ========================================================================
    # Environment
    scenario_name = config["env"]["scenario_name"]
    num_agents = config["env"]["num_agents"]
    max_agents = config["env"].get(
        "max_agents", num_agents
    )  # Default to num_agents if not specified
    num_envs = config["env"]["num_envs"]
    continuous_actions = config["env"]["continuous_actions"]
    penalise_by_time = config["env"].get("penalise_by_time", False)
    share_reward = config["env"].get("share_reward", False)
    distance_shaping_coef = config["env"].get("distance_shaping_coef", 0.1)

    # Mixed agent training settings
    mixed_agent_training = config["env"].get("mixed_agent_training", False)
    min_agents = config["env"].get("min_agents", num_agents)
    per_env_agent_variation = config["env"].get("per_env_agent_variation", False)

    # Per-env variation requires mixed agent training
    if per_env_agent_variation and not mixed_agent_training:
        print(
            "Warning: per_env_agent_variation requires mixed_agent_training=true. Disabling per_env_agent_variation."
        )
        per_env_agent_variation = False

    if mixed_agent_training:
        print(f"\n{'='*80}")
        print(f"MIXED AGENT TRAINING ENABLED")
        print(f"{'='*80}")
        print(f"  Range: {min_agents} to {max_agents} agents")
        if per_env_agent_variation:
            print(
                f"  Mode: PER-ENVIRONMENT variation (each env can have different agent count)"
            )
        else:
            print(
                f"  Mode: PER-EPISODE variation (all envs have same agent count per episode)"
            )
        print(
            f"  All agents in each episode have SAME capabilities (homogeneous teams)"
        )
        print(f"  This trains the self-attention to handle variable team sizes")
        print(f"{'='*80}\n")
    else:
        print(f"\nFixed agent training: {num_agents} agents per episode")

    # Fixed capability and food position settings
    use_fixed_capabilities = config["env"].get("use_fixed_capabilities", False)
    fixed_food_positions = config["env"].get("fixed_food_positions", False)
    fixed_n_food = config["env"].get(
        "fixed_n_food", None
    )  # Fixed number of food items (None = default)

    # Training
    num_episodes = config["training"]["num_episodes"]
    rollout_steps = config["training"]["rollout_steps"]
    log_interval = config["training"]["log_interval"]

    # Curriculum Learning Parameters (used by football and SMAX)
    curriculum_stage1_episodes = config["training"].get(
        "curriculum_stage1_episodes", 500
    )
    curriculum_stage2_episodes = config["training"].get(
        "curriculum_stage2_episodes", 500
    )
    football_use_curriculum = scenario_name == "football" and not config["model"].get(
        "use_cash", False
    )

    # Diversity Control settings
    # Note: SND is only applicable for continuous action spaces
    use_diversity_control = config["training"].get("use_diversity_control", False)
    # Disable diversity control for discrete action environments (SMAX)
    if scenario_name == "smax":
        use_diversity_control = False
    target_snd = float(config["training"].get("target_snd", 2.5))  # Ensure float type
    snd_ma_coef = config["training"].get("snd_moving_average_coef", 0.9)
    current_snd_ma = target_snd  # Initialize with target (we track scaled SND values)
    diversity_scaling = 1.0  # Initialize to 1.0 (no scaling on first episode)

    # Optional scheduled target-SND switches during rollout.
    # Supports both training.* keys and legacy env.* keys for backward compatibility.
    train_target_snd_list = config["training"].get("target_snd_list", None)
    if train_target_snd_list is None:
        train_target_snd_list = config["env"].get("target_snd_list", None)
    if train_target_snd_list is not None:
        train_target_snd_list = [float(v) for v in train_target_snd_list]

    train_target_snd_interval = int(
        config["training"].get(
            "target_snd_change_interval", config["env"].get("snd_change_interval", 0)
        )
        or 0
    )
    train_target_snd_mode = str(
        config["training"].get(
            "target_snd_mode", config["env"].get("target_snd_mode", "cycle")
        )
    ).lower()
    if train_target_snd_mode not in ["cycle", "random"]:
        train_target_snd_mode = "cycle"

    enable_rollout_target_snd_switch = (
        train_target_snd_list is not None
        and len(train_target_snd_list) > 0
        and train_target_snd_interval > 0
    )

    # SND Observation Buffer settings
    use_snd_obs_buffer = config["training"].get("use_snd_obs_buffer", False)
    snd_obs_buffer_size = config["training"].get("snd_obs_buffer_size", 1000)
    snd_obs_filter_food = config["training"].get("snd_obs_filter_food", False)
    snd_obs_buffer_sample_size = config["training"].get(
        "snd_obs_buffer_sample_size", 64
    )

    # Adapter-based SND settings (using hypernetwork query observations)
    use_adapter_snd = config["training"].get("use_adapter_snd", False)
    # Buffer allows up to 2 queries per agent per episode:
    # 1. Initial query at episode start (1 per agent)
    # 2. Event-triggered requery during rollout (up to 1 per agent)
    # Maximum buffer size per episode: 2 * num_agents * num_envs
    # In mixed training, use max_agents to accommodate worst-case scenario
    buffer_size_agents = max_agents if mixed_agent_training else num_agents
    adapter_snd_buffer_size = 2 * buffer_size_agents * num_envs
    adapter_snd_sample_size = config["training"].get("adapter_snd_sample_size", 128)

    # Global state clipping
    clip_global_state = config["training"].get("clip_global_state", False)
    clip_global_state_min = config["training"].get("clip_global_state_min", -10.0)
    clip_global_state_max = config["training"].get("clip_global_state_max", 10.0)

    # Model dimensions (obs_dim and action_dim will be detected from environment)
    policy_hidden_dims = config["model"].get("policy_hidden_dims", [64, 64])
    lora_rank = config["model"].get("lora_rank", 0)  # Default to 0 for DiCo
    task_embed_dim = config["model"].get("task_embed_dim", 0)  # Default to 0 for DiCo
    target_snd_dim = config["model"].get("target_snd_dim", 0)  # Default to 0 (disabled)

    # Context dimension: capability features (speed, lidar_range) or (speed, size, shoot_power) for football
    # CTDE approach: use actual agent capabilities instead of one-hot IDs
    use_capability_context = config["model"].get("use_capability_context", True)
    use_onehot_context = config["model"].get("use_onehot_context", True)
    use_positional_context = config["model"].get(
        "use_positional_context", False
    )  # New: positional encoding
    positional_encoding_dim = config["model"].get(
        "positional_encoding_dim", 16
    )  # Default 16D encoding

    # Determine context_dim based on scenario (will be updated for simple_tag/grassland later)
    if scenario_name == "football":
        context_dim = (
            5 if use_capability_context else 0
        )  # [speed, size, shoot_power, pos_x, pos_y]
    elif scenario_name == "simple_tag":
        context_dim = -1  # Placeholder, will be set to obs_dim after it's determined
    elif scenario_name == "sampling":
        context_dim = 0  # No capability context for sampling, uses lidar instead
    elif scenario_name == "grassland":
        context_dim = -1  # Placeholder, will be set to obs_dim after it's determined
    elif scenario_name == "smax":
        context_dim = (
            7 if use_capability_context else 0
        )  # Unit capability features (7 dimensions)
    elif scenario_name == "dispersion_vmas":
        # For dispersion_vmas: use positional encoding (scalable) OR one-hot encoding (legacy)
        # Positional encoding is preferred as it doesn't scale with max_agents
        # When use_capability_context=True, prepend [max_speed] (1D capability).
        if use_positional_context:
            # Positional encoding: fixed dimension regardless of agent count
            context_dim = (1 if use_capability_context else 0) + positional_encoding_dim
        else:
            # Legacy one-hot encoding: dimension = max_agents (doesn't scale well)
            context_dim = (1 if use_capability_context else 0) + (
                max_agents if use_onehot_context else 0
            )
    elif scenario_name == "reverse_transport":
        # Use [max_speed, force_multiplier] as 2-D capability context
        context_dim = 2 if use_capability_context else 0
    elif scenario_name == "pressure_plate":
        # Capabilities removed from pressure_plate; no capability context
        context_dim = 0
    else:
        context_dim = 2 if use_capability_context else 0  # [speed, lidar_range]

    # Lidar context configuration
    use_lidar_context = config["model"].get("use_lidar_context", False)
    # dispersion_vmas and pressure_plate use global observability (no lidar)
    if scenario_name in ["dispersion_vmas", "pressure_plate"]:
        lidar_dim = 0
    else:
        lidar_dim = (
            24 if use_lidar_context else 0
        )  # Last 24 dims of obs are lidar readings (12 for agents + 12 for food)

    # Hypernetwork Transformer (only needed for HyperLoRA)
    transformer_dim = config["model"].get("transformer_dim", 128)
    transformer_heads = config["model"].get("transformer_heads", 4)
    transformer_layers = config["model"].get("transformer_layers", 2)
    use_cross_agent_attention = config["model"].get("use_cross_agent_attention", True)

    # ========================================================================
    # Environment Initialization (PyTorch/VMAS)
    # ========================================================================
    print("\nInitializing VMAS environment...")

    # Initialize RNG for capability randomization
    np_rng = np.random.default_rng(seed)

    # Get capability ranges from config (for randomization)
    speed_range = config["env"].get("speed_range", [0.5, 1.5])
    force_multiplier_range = config["env"].get("force_multiplier_range", [0.5, 1.5])
    lidar_range_range = config["env"].get("lidar_range_range", [0.3, 0.7])
    max_speed_range = config["env"].get(
        "max_speed_range", [0.5, 1.5]
    )  # For dispersion_vmas

    # For mixed training, we need to set up capabilities for max_agents
    env_num_agents = max_agents if mixed_agent_training else num_agents

    # Setup agent capabilities (fixed or default for randomization)
    if use_fixed_capabilities:
        # Use fixed capabilities from config
        fixed_caps = config["env"].get("fixed_capabilities", {})
        fixed_speeds = fixed_caps.get("speed", [])
        fixed_max_speeds = fixed_caps.get("max_speed", [])  # For dispersion_vmas

        # For reverse_transport: only use speed and force_multiplier (no lidar_range)
        if scenario_name == "reverse_transport":
            fixed_force_multipliers = fixed_caps.get("force_multiplier", [])
            fixed_lidar_ranges = []  # Not used for reverse_transport
        elif scenario_name == "dispersion_vmas":
            # For dispersion_vmas: only use max_speed (no lidar_range or force_multiplier)
            fixed_lidar_ranges = []  # Not used for dispersion_vmas
            fixed_force_multipliers = []  # Not used for dispersion_vmas
        else:
            fixed_lidar_ranges = fixed_caps.get("lidar_range", [])
            fixed_force_multipliers = []  # Not used for other scenarios

        # Auto-generate if not enough values provided or if lengths don't match
        if scenario_name == "reverse_transport":
            # For reverse_transport: check speed and force_multiplier lengths
            if (
                len(fixed_speeds) != env_num_agents
                or len(fixed_force_multipliers) != env_num_agents
            ):
                print(
                    f"Warning: Fixed capabilities not fully specified for {env_num_agents} agents."
                )
                print(
                    f"  Provided lengths - speed: {len(fixed_speeds)}, force_multiplier: {len(fixed_force_multipliers)}"
                )
                print("  Using default randomization ranges instead.")
                use_fixed_capabilities = False
        elif scenario_name == "dispersion_vmas":
            # For dispersion_vmas: check max_speed length
            if len(fixed_max_speeds) != env_num_agents:
                print(
                    f"Warning: Fixed max_speed list length doesn't match env_num_agents={env_num_agents}"
                )
                print(f"Auto-generating fixed max_speeds with distinct values...")
                # Generate distinct max_speed values
                fixed_max_speeds = []
                max_speed_values = [0.5, 0.75, 1.0, 1.25, 1.5]  # 5 distinct speeds
                for i in range(env_num_agents):
                    fixed_max_speeds.append(max_speed_values[i % len(max_speed_values)])
        else:
            # For other scenarios: check speed and lidar_range lengths
            if (
                len(fixed_speeds) != env_num_agents
                or len(fixed_lidar_ranges) != env_num_agents
            ):
                print(
                    f"WARNING: Fixed capabilities list length doesn't match env_num_agents={env_num_agents}"
                )
                print(f"Auto-generating fixed capabilities with distinct values...")
                # Generate distinct capability combinations
                fixed_speeds = []
                fixed_lidar_ranges = []

                # For other scenarios: generate speed and lidar_range
                speed_values = [0.5, 1.5]  # Low and high speeds
                lidar_values = [0.5, 0.5]  # Low and high ranges

                for i in range(env_num_agents):
                    fixed_speeds.append(speed_values[i % 2])
                    fixed_lidar_ranges.append(lidar_values[(i // 2) % 2])

        default_speeds = fixed_speeds
        default_lidar_ranges = fixed_lidar_ranges
        default_force_multipliers = fixed_force_multipliers
        default_max_speeds = fixed_max_speeds  # For dispersion_vmas

        print(f"\n{'='*80}")
        print("USING FIXED CAPABILITIES (no randomization)")
        print(f"{'='*80}")
        for i in range(env_num_agents):
            if scenario_name == "reverse_transport":
                print(
                    f"  Agent {i}: speed={default_speeds[i]:.3f}, force_multiplier={default_force_multipliers[i]:.3f}"
                )
            elif scenario_name == "dispersion_vmas":
                print(f"  Agent {i}: max_speed={default_max_speeds[i]:.3f}")
            else:
                print(
                    f"  Agent {i}: speed={default_speeds[i]:.1f}, lidar_range={default_lidar_ranges[i]:.1f}"
                )
        print(f"{'='*80}\n")
    else:
        # Use default capabilities for initial setup (will be randomized per episode)
        default_speeds = [1.0] * env_num_agents
        default_lidar_ranges = [0.5] * env_num_agents
        default_force_multipliers = [0.5] * env_num_agents
        default_max_speeds = [1.0] * env_num_agents  # For dispersion_vmas
        use_homogeneous_caps = config["env"].get("homogeneous_capabilities", True)
        if use_homogeneous_caps:
            print(f"\nUsing RANDOMIZED capabilities (will vary each episode)")
            print(f"  Mode: HOMOGENEOUS (all agents in team get same random values)")
        else:
            print(f"\nUsing RANDOMIZED capabilities (will vary each episode)")
            print(f"  Mode: HETEROGENEOUS (each agent gets different random values)")

    if fixed_food_positions:
        print(f"Using FIXED food positions in corners\n")
    else:
        print(f"Using RANDOM food positions\n")

    # Create agent_capabilities dict based on scenario
    if scenario_name == "reverse_transport":
        agent_capabilities = {
            "speed": default_speeds,
            "force_multiplier": default_force_multipliers,
        }
    elif scenario_name == "dispersion_vmas":
        agent_capabilities = {
            "max_speed": default_max_speeds,
        }
    else:
        agent_capabilities = {
            "speed": default_speeds,
            "lidar_range": default_lidar_ranges,
        }

    # For mixed agent training, create environment with max_agents capacity
    env_num_agents = max_agents if mixed_agent_training else num_agents

    # Prepare football-specific parameters if using football scenario
    football_kwargs = {}
    simple_tag_kwargs = {}
    sampling_kwargs = {}
    grassland_kwargs = {}
    smax_kwargs = {}
    reverse_transport_kwargs = {}
    pressure_plate_kwargs = {}
    if scenario_name == "smax":
        # Setup SMAX specific parameters
        smax_kwargs = {
            "map_name": config["env"].get("map_name", "3m"),
            "num_allies": config["env"].get("num_allies", 3),
            "num_enemies": config["env"].get("num_enemies", 3),
            "map_width": config["env"].get("map_width", 32),
            "map_height": config["env"].get("map_height", 32),
            "world_steps_per_env_step": config["env"].get(
                "world_steps_per_env_step", 8
            ),
            "time_per_step": config["env"].get("time_per_step", 1.0 / 16),
            "observation_type": config["env"].get("observation_type", "unit_list"),
            "action_type": config["env"].get("action_type", "discrete"),
            "use_self_play_reward": config["env"].get("use_self_play_reward", False),
            "see_enemy_actions": config["env"].get("see_enemy_actions", True),
            "won_battle_bonus": config["env"].get("won_battle_bonus", 1.0),
            "walls_cause_death": config["env"].get("walls_cause_death", True),
            "max_steps": config["env"].get("max_steps", 100),
            "smacv2_position_generation": config["env"].get(
                "smacv2_position_generation", False
            ),
            "smacv2_unit_type_generation": config["env"].get(
                "smacv2_unit_type_generation", False
            ),
            "damage_reward_multiplier": config["env"].get(
                "damage_reward_multiplier", 1.0
            ),
            "distance_reward_scale": config["env"].get("distance_reward_scale", 0.0),
        }
        # For SMAX with HeuristicEnemySMAX, only allies are trainable
        env_num_agents = smax_kwargs["num_allies"]
    elif scenario_name == "football":
        football_kwargs = {
            "num_red_agents": config["env"].get("num_red_agents", num_agents),
            "ai_red_agents": config["env"].get("ai_red_agents", True),
            "ai_blue_agents": config["env"].get("ai_blue_agents", False),
            "physically_different": config["env"].get("physically_different", False),
            "enable_shooting": config["env"].get("enable_shooting", False),
            "dense_reward": config["env"].get("dense_reward", True),
            "observe_teammates": config["env"].get("observe_teammates", True),
            "observe_adversaries": config["env"].get("observe_adversaries", True),
            "disable_ai_red": config["env"].get("disable_ai_red", False),
            "pos_shaping_factor_ball_goal": config["env"].get(
                "pos_shaping_factor_ball_goal", 10.0
            ),
            "pos_shaping_factor_agent_ball": config["env"].get(
                "pos_shaping_factor_agent_ball", 0.1
            ),
        }
        if football_use_curriculum:
            football_kwargs.update(
                {
                    "dense_reward": True,
                    "disable_ai_red": True,
                    "pos_shaping_factor_ball_goal": 100.0,
                    "pos_shaping_factor_agent_ball": 10.0,
                }
            )
        print(f"\nFootball Configuration:")
        print(f"  Blue agents (trainable): {num_agents}")
        print(f"  Red agents (adversaries): {football_kwargs['num_red_agents']}")
        if football_use_curriculum:
            print(f"  Curriculum Learning: ENABLED")
            print(
                f"    - Stage 1 (episodes 0-{curriculum_stage1_episodes-1}): No adversaries"
            )
            print(
                f"    - Stage 2 (episodes {curriculum_stage1_episodes}-{curriculum_stage1_episodes + curriculum_stage2_episodes-1}): Half-strength adversaries"
            )
            print(
                f"    - Stage 3 (episodes {curriculum_stage1_episodes + curriculum_stage2_episodes}+): Full-strength adversaries"
            )
        else:
            print(f"  Curriculum Learning: DISABLED")
    elif scenario_name == "simple_tag":
        # Setup simple_tag specific parameters
        num_adversaries = config["env"].get("num_adversaries", 3)
        num_good_agents = config["env"].get("num_agents", 1)

        # Handle agent capabilities for simple_tag (separate for adversaries and good agents)
        if use_fixed_capabilities:
            fixed_caps = config["env"].get("fixed_capabilities", {})
            adversary_speeds = fixed_caps.get(
                "adversary_speeds", [1.0] * num_adversaries
            )
            agent_speeds = fixed_caps.get("agent_speeds", [1.3] * num_good_agents)
            adversary_lidar_ranges = fixed_caps.get(
                "adversary_lidar_ranges", [0.5] * num_adversaries
            )
            agent_lidar_ranges = fixed_caps.get(
                "agent_lidar_ranges", [0.6] * num_good_agents
            )
        else:
            adversary_speeds = [1.0] * num_adversaries
            agent_speeds = [1.3] * num_good_agents
            adversary_lidar_ranges = [0.5] * num_adversaries
            agent_lidar_ranges = [0.6] * num_good_agents

        agent_capabilities = {
            "adversary_speeds": adversary_speeds,
            "agent_speeds": agent_speeds,
            "adversary_lidar_ranges": adversary_lidar_ranges,
            "agent_lidar_ranges": agent_lidar_ranges,
        }

        simple_tag_kwargs = {
            "num_good_agents": num_good_agents,
            "num_adversaries": num_adversaries,
            "num_landmarks": config["env"].get("num_landmarks", 2),
            "shape_agent_rew": config["env"].get("shape_agent_rew", False),
            "shape_adversary_rew": config["env"].get("shape_adversary_rew", True),
            "agents_share_rew": config["env"].get("agents_share_rew", False),
            "adversaries_share_rew": config["env"].get("adversaries_share_rew", True),
            "observe_same_team": config["env"].get("observe_same_team", True),
            "observe_pos": config["env"].get("observe_pos", True),
            "observe_vel": config["env"].get("observe_vel", True),
            "bound": config["env"].get("bound", 1.0),
            "respawn_at_catch": config["env"].get("respawn_at_catch", False),
        }

        # For simple_tag, num_agents is total (adversaries + good agents)
        env_num_agents = num_adversaries + num_good_agents
    elif scenario_name == "sampling":
        # Setup sampling specific parameters
        sampling_kwargs = {
            "shared_rew": config["env"].get("shared_rew", True),
            "comms_range": config["env"].get("comms_range", 0.0),
            "lidar_range": config["env"].get("lidar_range", 0.2),
            "agent_radius": config["env"].get("agent_radius", 0.025),
            "xdim": config["env"].get("xdim", 1.0),
            "ydim": config["env"].get("ydim", 1.0),
            "grid_spacing": config["env"].get("grid_spacing", 0.05),
            "n_gaussians": config["env"].get("n_gaussians", 3),
            "cov": config["env"].get("cov", 0.05),
            "collisions": config["env"].get("collisions", True),
            "spawn_same_pos": config["env"].get("spawn_same_pos", False),
            "norm": config["env"].get("norm", True),
        }
        # For sampling, use the specified num_agents directly
        env_num_agents = max_agents if mixed_agent_training else num_agents
    elif scenario_name == "grassland":
        # Setup grassland specific parameters
        num_adversaries = config["env"].get("num_adversaries", 6)
        num_good_agents = config["env"].get("num_agents", 6)

        grassland_kwargs = {
            "n_agents_good": num_good_agents,
            "n_agents_adversaries": num_adversaries,
            "obs_agents": config["env"].get("obs_agents", True),
            "ratio": config["env"].get("ratio", 5),
        }

        # For grassland, num_agents is total (adversaries + good agents)
        env_num_agents = num_adversaries + num_good_agents
    elif scenario_name == "reverse_transport":
        # Setup reverse_transport specific parameters
        base_package_mass = config["env"].get("package_mass", 50)
        scale_package_mass_with_agents = config["env"].get(
            "scale_package_mass_with_agents", False
        )
        # Compute mass-per-agent ratio from config (base mass / config num_agents)
        mass_per_agent = (
            base_package_mass / num_agents if scale_package_mass_with_agents else None
        )
        reverse_transport_kwargs = {
            "package_width": config["env"].get("package_width", 0.6),
            "package_length": config["env"].get("package_length", 0.6),
            "package_mass": base_package_mass,
            "package_mass_range": config["env"].get("package_mass_range", [1, 100]),
        }
        print(f"\nReverse Transport Configuration:")
        print(f"  Package Mass: {reverse_transport_kwargs['package_mass']}")
        if scale_package_mass_with_agents:
            print(f"  Mass per agent: {mass_per_agent:.2f} (scales with team size)")
        print(f"  Package Mass Range: {reverse_transport_kwargs['package_mass_range']}")
        print(
            f"  Package Dimensions: {reverse_transport_kwargs['package_width']} x {reverse_transport_kwargs['package_length']}\n"
        )
    elif scenario_name == "pressure_plate":
        # Setup pressure_plate specific parameters
        pressure_plate_kwargs = {
            "n_ground_robots": env_num_agents,
            "x_semidim": config["env"].get("x_semidim", 2.0),
            "y_semidim": config["env"].get("y_semidim", 2.0),
            "plate_radius": config["env"].get("plate_radius", 0.15),
            "plate_margin": config["env"].get("plate_margin", 0.8),
            "door_size": config["env"].get("door_size", 0.6),
            "goal_radius": config["env"].get("goal_radius", 0.3),
            "with_drone": config["env"].get("with_drone", False),
            "use_global_obs": config["env"].get("use_global_obs", True),
            "plate_reward": config["env"].get("plate_reward", 0.1),
            "goal_reward": config["env"].get("goal_reward", 10.0),
            "time_penalty": config["env"].get("time_penalty", -0.01),
            "reward_type": config["env"].get("reward_type", "sparse"),
            "training_spawn_side": config["env"].get("training_spawn_side", "both"),
        }
        print(f"\nPressure Plate Configuration:")
        print(f"  Global Observability: {pressure_plate_kwargs['use_global_obs']}")
        print(f"  Number of Ground Robots: {pressure_plate_kwargs['n_ground_robots']}")
        print(f"  Config num_agents/max_agents: {num_agents}/{max_agents}")
        print(f"  With Drone: {pressure_plate_kwargs['with_drone']}\n")

    env = make_vmas_env(
        scenario_name,
        env_num_agents,
        num_envs,
        device=torch_device,
        continuous_actions=continuous_actions,
        penalise_by_time=penalise_by_time,
        share_reward=share_reward,
        distance_shaping_coef=distance_shaping_coef,
        agent_capabilities=agent_capabilities,
        fixed_food_positions=fixed_food_positions,
        fixed_n_food=fixed_n_food,  # Pass fixed food count
        **football_kwargs,
        **simple_tag_kwargs,
        **sampling_kwargs,
        **grassland_kwargs,
        **smax_kwargs,
        **reverse_transport_kwargs,
        **pressure_plate_kwargs,
    )

    # Cache agent list for SMAX to avoid repeated dictionary access in rollout loop
    if scenario_name == "smax":
        agents_list = env.agents  # Cache this to avoid repeated lookups

    # Get actual observation and action dimensions from the environment
    print("Detecting environment dimensions...")
    if scenario_name == "smax":
        # SMAX is JAX-based, needs special handling
        rng_key = jax.random.PRNGKey(seed)
        temp_obs_dict, temp_state = env.reset(rng_key)
        # Get obs dim from first agent's observation
        first_agent = env.agents[0]
        actual_obs_dim = temp_obs_dict[first_agent].shape[-1]
        # Action dim is different for allies vs enemies in asymmetric scenarios
        actual_action_dim = env.action_spaces[first_agent].n  # Discrete action space
        print(
            f"SMAX environment: {len(env.agents)} agents, obs_dim={actual_obs_dim}, action_dim={actual_action_dim}"
        )
    else:
        temp_obs = env.reset()
        actual_obs_dim = temp_obs[0].shape[-1]  # Get obs dim from first agent

        # Handle action dimension based on environment type
        if scenario_name == "football":
            actual_action_dim = env.action_dim
        elif scenario_name == "smax":
            actual_action_dim = env.num_ally_actions  # Discrete actions for allies
        else:
            actual_action_dim = env.get_agent_action_size(env.agents[0])

    # Debug: Check if agents have sensors (skip for football and SMAX)
    if scenario_name not in ["football", "smax"]:
        print(f"\nAgent sensor debug:")
        for i, agent in enumerate(env.agents):
            print(
                f"  Agent {i}: has {len(agent.sensors) if hasattr(agent, 'sensors') and agent.sensors else 0} sensors"
            )
            if hasattr(agent, "sensors") and agent.sensors:
                for j, sensor in enumerate(agent.sensors):
                    print(f"    Sensor {j}: {type(sensor).__name__}")

    print(f"\nDetected obs_dim: {actual_obs_dim}, action_dim: {actual_action_dim}")
    if scenario_name == "smax":
        # SMAX observations are already per-agent in the dict
        print(f"First agent obs shape: {temp_obs_dict[env.agents[0]].shape}")
    else:
        print(f"First agent obs shape: {temp_obs[0].shape}")
        print(f"Full observation (first env, first agent): {temp_obs[0][0]}")

    # Override config with actual dimensions
    obs_dim = actual_obs_dim
    action_dim = actual_action_dim

    # For SMAX with enemy number curriculum, we need to use max obs_dim and max_action_dim
    # because the GRU policy will be initialized with fixed input/output dimensions
    max_obs_dim = obs_dim  # Default to current obs_dim
    max_action_dim = action_dim  # Default to current action_dim
    if scenario_name == "smax":
        use_curriculum = config["training"].get("use_curriculum", False)
        curriculum_type = config["training"].get("curriculum_type", "damage_scale")
        if use_curriculum and curriculum_type == "enemy_number":
            # Get max num enemies to determine max obs_dim
            max_num_enemies = config["training"].get("curriculum_stage3_num_enemies", 3)
            # Create temp environment with max enemies to get max obs dim
            print(f"\nDetermining max_obs_dim for {max_num_enemies} enemies...")

            # Update smax_kwargs with max enemies
            temp_smax_kwargs = smax_kwargs.copy()
            temp_smax_kwargs["num_enemies"] = max_num_enemies

            temp_env_max = make_vmas_env(
                scenario_name,
                env_num_agents,
                num_envs,
                device=torch_device,
                continuous_actions=continuous_actions,
                penalise_by_time=penalise_by_time,
                share_reward=share_reward,
                distance_shaping_coef=distance_shaping_coef,
                agent_capabilities=agent_capabilities,
                fixed_food_positions=fixed_food_positions,
                **temp_smax_kwargs,
            )
            rng_key_max = jax.random.PRNGKey(seed + 999)
            temp_obs_dict_max, _ = temp_env_max.reset(rng_key_max)
            max_obs_dim = temp_obs_dict_max[temp_env_max.agents[0]].shape[-1]
            max_action_dim = temp_env_max.action_spaces[temp_env_max.agents[0]].n
            print(f"max_obs_dim with {max_num_enemies} enemies: {max_obs_dim}")
            print(f"max_action_dim with {max_num_enemies} enemies: {max_action_dim}")
            print(
                f"current obs_dim with {len(env.agents) - env_num_agents} enemies: {obs_dim}"
            )
            print(f"current action_dim: {action_dim}")

    # Update context_dim for simple_tag and grassland now that obs_dim is known
    if (
        scenario_name == "simple_tag" or scenario_name == "grassland"
    ) and context_dim == -1:
        context_dim = obs_dim if use_capability_context else 0

    # For SMAX with capability context, use 7-dim capability features
    if scenario_name == "smax" and use_capability_context:
        context_dim = 7  # 7 features: health, attack, attack_range, velocity, sight_range, radius, attack_speed

    # Calculate actual lidar dimension from observation
    # Observation structure: pos(2) + vel(2) + food(4) + lidar_agents(12) + lidar_food(12)
    # So lidar_dim = obs_dim - 8 (subtract pos, vel, and food observation)
    if use_lidar_context and scenario_name == "sampling":
        # Sampling observation: pos(2) + vel(2) + neighbor_samples(8) = 12
        # Use neighbor samples (last 8 dims) as context instead of lidar
        lidar_dim = 8  # Use the 8 neighbor samples as context
        print(
            f"Sampling environment: using neighbor_samples as context, lidar_dim={lidar_dim}"
        )
    elif use_lidar_context and scenario_name not in [
        "football",
        "smax",
        "dispersion_vmas",
    ]:
        lidar_dim = obs_dim - 8  # Dynamically calculate based on actual obs_dim
        print(f"Calculated lidar_dim from observation: {lidar_dim}")
    elif scenario_name in [
        "football",
        "smax",
        "dispersion_vmas",
        "reverse_transport",
        "pressure_plate",
    ]:
        lidar_dim = 0  # Football, SMAX, dispersion_vmas, reverse_transport, and pressure_plate don't use lidar
        print(f"{scenario_name.upper()} environment: no lidar context")

    # Optional position-based hypernetwork inputs (scenario/config dependent).
    # Keep food positions disabled by default unless explicitly requested.
    food_position_dim = 0
    use_agent_position_context = config["model"].get(
        "use_agent_position_context", False
    )
    agent_position_dim = (
        config["model"].get("agent_position_dim", 0)
        if use_agent_position_context
        else 0
    )

    # wind_flocking_position relies on (x, y) position conditioning for adapter SND.
    if scenario_name == "wind_flocking_position" and use_agent_position_context:
        if agent_position_dim <= 0:
            agent_position_dim = 2
            print(
                "wind_flocking_position: overriding agent_position_dim to 2 "
                "(x, y position context enabled)"
            )

    # Environment context dimension (for dynamic environment properties)
    # For reverse_transport: [package_mass, package_width, package_length]
    # For pressure_plate: [left_plate_pos, right_plate_pos, door_open, goal_pos]
    if scenario_name == "reverse_transport":
        env_context_dim = 3  # mass, width, length
        print(
            f"Using environment context for hypernetwork, env_context_dim={env_context_dim}"
        )
    elif scenario_name == "pressure_plate":
        # Calculate env_context_dim based on config flags
        use_env_context = config["model"].get("use_env_context", False)
        if use_env_context:
            env_context_plate_positions = config["model"].get(
                "env_context_plate_positions", True
            )
            env_context_door_state = config["model"].get("env_context_door_state", True)
            env_context_goal_position = config["model"].get(
                "env_context_goal_position", True
            )
            use_agent_id_context = config["model"].get("use_agent_id_context", False)

            env_context_dim = 0
            if env_context_plate_positions:
                env_context_dim += 4  # left_plate (2D) + right_plate (2D)
            if env_context_door_state:
                env_context_dim += 1  # door_open flag
            if env_context_goal_position:
                env_context_dim += 2  # goal position (2D)
            if use_agent_id_context:
                env_context_dim += (
                    max_agents  # one-hot agent ID for role differentiation
                )

            print(
                f"Using environment context for hypernetwork, env_context_dim={env_context_dim}"
            )
            print(f"  - Plate positions: {env_context_plate_positions}")
            print(f"  - Door state: {env_context_door_state}")
            print(f"  - Goal position: {env_context_goal_position}")
            print(f"  - Agent IDs: {use_agent_id_context}")
        else:
            env_context_dim = 0
    else:
        env_context_dim = 0

    print(
        f"Using obs_dim={obs_dim}, action_dim={action_dim}, context_dim={context_dim} (capability-based)"
    )
    if scenario_name == "dispersion_vmas":
        context_parts = []
        if use_capability_context:
            context_parts.append("capabilities(2)")
        if use_positional_context:
            context_parts.append(f"positional(dim={positional_encoding_dim})")
        elif use_onehot_context:
            context_parts.append(f"one-hot(dim={max_agents})")
        context_str = " + ".join(context_parts) if context_parts else "none"
        print(f"Context encoding: {context_str}, food_position_dim={food_position_dim}")
    elif scenario_name in ["simple_tag", "grassland"]:
        print(f"Context encoding: Full observation (obs_dim={obs_dim})")
    elif scenario_name == "smax":
        print(f"Context encoding: Unit capabilities (7 features)")
    else:
        print(
            f"Context encoding: Each agent gets a capability vector [speed, lidar_range]"
        )
    print(
        f"Training with {num_agents} agents (max capacity: {max_agents} for deployment)"
    )

    # ========================================================================
    # Model Initialization (JAX/Flax)
    # ========================================================================
    print("Initializing models...")

    # Check if using DICO architecture (per-agent networks)
    use_dico = config["model"].get("use_dico", False)
    use_cash = config["model"].get("use_cash", False)

    use_hypernetwork = config["model"].get("use_hypernetwork", True)
    adaptive_hypernetwork = config["model"].get("adaptive_hypernetwork", False)
    lora_mode = config["model"].get("lora_mode", "final_only")
    lora_scaling_factor = config["model"].get("lora_scaling_factor", 1.0)
    use_gru_policy = config["model"].get("use_gru_policy", False)

    # DICO disables hypernetwork (uses per-agent networks instead)
    if use_dico:
        use_hypernetwork = False
        use_cash = False
        print("\n" + "=" * 80)
        print("USING DICO ARCHITECTURE (Bettini et al. 2024)")
        print("=" * 80)
        print("  - Homogeneous: 1 shared MLP (φ_homo)")
        print(f"  - Heterogeneous: {env_num_agents} per-agent MLPs (φ_hetero_i)")
        print("  - Output: π_i(a|o) = φ_homo(o) + λ * φ_hetero_i(o)")
        print("  - No hypernetwork (direct per-agent networks)")
        print("=" * 80 + "\n")

    if use_cash:
        use_hypernetwork = False
        use_dico = False
        use_gru_policy = True
        print("\n" + "=" * 80)
        print("USING CASH BASELINE (capability-aware shared hypernetworks)")
        print("  - Per-step hyper-decoder generation")
        print("  - Shared GRU backbone")
        print("  - Hyper-adapter includes LayerNorm")
        print("=" * 80 + "\n")

    # Update wandb run name with architecture details (if not custom named)
    if use_wandb and not args.wandb_name:
        backbone = "gru" if use_gru_policy else "mlp"
        hn_status = "hn" if use_hypernetwork else "nohn"

        # Add capability configuration summary to distinguish runs
        cap_suffix = ""
        if use_fixed_capabilities:
            fixed_caps = config["env"].get("fixed_capabilities", {})
            if scenario_name == "reverse_transport":
                speeds = fixed_caps.get("speed", [])
                forces = fixed_caps.get("force_multiplier", [])
                if speeds and forces:
                    # Check if heterogeneous (more than one unique value)
                    unique_speeds = len(set(speeds))
                    unique_forces = len(set(forces))
                    if unique_speeds > 1 or unique_forces > 1:
                        cap_suffix = f"_hetcaps_s{unique_speeds}f{unique_forces}"
                    else:
                        cap_suffix = "_homcaps"
            else:
                speeds = fixed_caps.get("speed", [])
                lidar = fixed_caps.get("lidar_range", [])
                if speeds or lidar:
                    unique_speeds = len(set(speeds)) if speeds else 0
                    unique_lidar = len(set(lidar)) if lidar else 0
                    if unique_speeds > 1 or unique_lidar > 1:
                        cap_suffix = f"_hetcaps_s{unique_speeds}l{unique_lidar}"
                    else:
                        cap_suffix = "_homcaps"

        rank_suffix = f"_rank{lora_rank}" if lora_rank > 0 else ""
        wandb_name = f"divinterp_{scenario_name}_{backbone}_{hn_status}{cap_suffix}{rank_suffix}_seed{seed}"
        wandb.run.name = wandb_name

    # Extract GRU parameters (needed regardless of hypernetwork usage)
    if use_gru_policy:
        gru_hidden_dim = config["model"].get("gru_hidden_dim", 64)
        fc_dim_size = config["model"].get("fc_dim_size", 128)
        gru_num_layers = config["model"].get("gru_num_layers", 1)

    # Policy dimensions
    # For SMAX with curriculum, use max dimensions to ensure policy can handle all stages
    policy_input_dim = max_obs_dim if scenario_name == "smax" else obs_dim
    policy_action_dim = max_action_dim if scenario_name == "smax" else action_dim
    policy_dims = {
        "obs_dim": policy_input_dim,
        "hidden_dims": policy_hidden_dims,
        "action_dim": policy_action_dim,
        "lora_rank": lora_rank,
    }

    global shared_policy, hypernetwork

    # Debug: Print policy selection info
    print(f"\n{'='*80}")
    print(f"POLICY INITIALIZATION")
    print(f"{'='*80}")
    print(f"  scenario_name: {scenario_name}")
    print(f"  use_dico: {use_dico}")
    print(f"  use_cash: {use_cash}")
    print(f"  use_hypernetwork: {use_hypernetwork}")
    print(f"  use_diversity_control: {use_diversity_control}")
    print(f"{'='*80}\n")

    if use_cash:
        shared_policy = CASHPolicy(
            num_agents=env_num_agents,
            capability_dim=context_dim,
            gru_hidden_dim=gru_hidden_dim,
            fc_dim_size=fc_dim_size,
            action_dim=policy_action_dim,
            hyper_hidden_dim=config["model"].get("cash_hyper_hidden_dim", 128),
            hyper_num_layers=config["model"].get("cash_hyper_layers", 2),
            decoder_hidden_dim=config["model"].get("cash_decoder_hidden_dim", 64),
            use_two_layer_decoder=config["model"].get(
                "cash_use_two_layer_decoder", True
            ),
            log_std_max=config["model"].get("log_std_max", 0.0),
            min_std=config["model"].get("min_std", 0.3),
        )
        hypernetwork = None
        print(
            f"Using CASH policy (gru_hidden_dim={gru_hidden_dim}, capability_dim={context_dim}, action_dim={policy_action_dim})"
        )
    elif use_dico:
        # DICO architecture: homogeneous + per-agent heterogeneous networks
        # Use Deep Sets (GATv2) for football scenario as in Bettini et al. 2024
        use_deep_sets = scenario_name == "football"
        print(f"DICO mode: use_deep_sets = {use_deep_sets}")

        if use_diversity_control:
            if use_deep_sets:
                shared_policy = DiCoDeepSetsPolicy(
                    num_agents=env_num_agents,
                    hidden_dims=tuple(policy_hidden_dims),
                    action_dim=action_dim,
                    gat_out_features=32,  # Bettini: intermediate_sizes=[32]
                    edge_radius=10.0,  # Bettini: edge_radius=10
                )
                print(
                    f"Using DICO policy with Deep Sets (GATv2) for football "
                    f"({env_num_agents} agents, edge_radius=10.0)"
                )
            else:
                shared_policy = DiCoPolicy(
                    num_agents=env_num_agents,
                    hidden_dims=tuple(policy_hidden_dims),
                    action_dim=action_dim,
                )
                print(
                    f"Using DICO policy with {env_num_agents} per-agent heterogeneous networks"
                )
        else:
            # DICO without diversity control (fixed scaling)
            # Note: Deep Sets version not implemented for fixed scaling
            # (Bettini et al. always use diversity control)
            shared_policy = DiCoHomogeneousPolicy(
                num_agents=env_num_agents,
                hidden_dims=tuple(policy_hidden_dims),
                action_dim=action_dim,
            )
            print(
                "Using DICO without diversity control (homo + hetero with fixed scaling)"
            )
        hypernetwork = None
    elif use_hypernetwork:
        if use_gru_policy:
            # Use GRU-based policy (common in MARL baselines)
            shared_policy = GRULoRAPolicy(
                gru_hidden_dim=gru_hidden_dim,
                fc_dim_size=fc_dim_size,
                action_dim=policy_action_dim,
                log_std_max=config["model"].get("log_std_max", 0.0),
                min_std=config["model"].get("min_std", 0.3),
                discrete_actions=(scenario_name == "smax"),
            )

            # Add gru_hidden_dim to policy_dims for hypernetwork
            policy_dims["gru_hidden_dim"] = gru_hidden_dim

            print(
                f"Using GRU-based HyperLoRA (hidden_dim={gru_hidden_dim}, fc_dim={fc_dim_size}, action_dim={policy_action_dim})"
            )
            print(
                f"  Policy object: gru_hidden_dim={shared_policy.gru_hidden_dim}, action_dim={shared_policy.action_dim}"
            )
            # For GRU policy, hidden_dims in policy_dims should be empty since
            # GRU replaces the MLP hidden layers
            policy_dims["hidden_dims"] = []
            policy_dims["gru_hidden_dim"] = gru_hidden_dim
        else:
            # Use MLP-based policy
            shared_policy = LoRAPolicy(
                hidden_dims=tuple(policy_hidden_dims),
                action_dim=policy_action_dim,
                lora_mode=lora_mode,
                log_std_max=config["model"].get("log_std_max", 0.0),
                min_std=config["model"].get("min_std", 0.3),
                discrete_actions=(scenario_name == "smax"),
            )
            print(
                f"Using MLP-based HyperLoRA with lora_mode='{lora_mode}', scaling_factor={lora_scaling_factor} (static)"
            )

        hypernetwork = Hypernetwork(
            policy_dims=policy_dims,
            context_dim=context_dim,
            task_embed_dim=task_embed_dim,
            lidar_dim=lidar_dim,
            food_position_dim=food_position_dim,
            agent_position_dim=agent_position_dim,
            target_snd_dim=target_snd_dim,
            env_context_dim=env_context_dim,
            max_agents=max_agents,
            transformer_dim=transformer_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            lora_mode=lora_mode,
            scaling_factor=lora_scaling_factor,
            use_cross_agent_attention=use_cross_agent_attention,
        )
        if use_lidar_context:
            print(
                f"  - Using initial lidar readings as additional context (lidar_dim={lidar_dim})"
            )
        if not use_cross_agent_attention:
            print(
                "  - Cross-agent attention DISABLED: Each agent processes only its own features (better for zero-shot generalization)"
            )
        else:
            print(
                "  - Cross-agent attention ENABLED: Agents attend to all other agents' features"
            )
    else:
        # Standard MAPPO (no hypernetwork)
        if use_gru_policy:
            # Use GRU-based policy without LoRA/hypernetwork
            shared_policy = GRULoRAPolicy(
                gru_hidden_dim=gru_hidden_dim,
                fc_dim_size=fc_dim_size,
                action_dim=policy_action_dim,
                log_std_max=config["model"].get("log_std_max", 0.0),
                min_std=config["model"].get("min_std", 0.3),
                discrete_actions=(scenario_name == "smax"),
            )
            print(
                f"Using standard MAPPO with GRU policy (hidden_dim={gru_hidden_dim}, no hypernetwork)"
            )
        else:
            # Standard policy: LoRAPolicy with lora_rank=0 disables adapters
            shared_policy = LoRAPolicy(
                hidden_dims=tuple(policy_hidden_dims),
                action_dim=action_dim,
                log_std_max=config["model"].get("log_std_max", 0.0),
                min_std=config["model"].get("min_std", 0.3),
                discrete_actions=(scenario_name == "smax"),
            )
            print("Using standard MAPPO (no hypernetwork)")
        hypernetwork = None

    # Initialize centralized critic for MAPPO
    global critic
    critic_hidden_dim = config["model"]["critic_hidden_dim"]
    critic_num_layers = config["model"]["critic_num_layers"]
    use_shared_critic = config["model"].get(
        "use_shared_critic", False
    )  # Default: per-agent values
    # For mixed training, critic must handle max_agents; otherwise use num_agents
    # For SMAX curriculum, critic must handle max total units (allies + max enemies)
    if scenario_name == "smax":
        # Use max total units from config (allies + max enemies)
        max_total_units = smax_kwargs["num_allies"] + config["env"].get(
            "num_enemies", 3
        )
        global_state_dim = obs_dim * max_total_units
    else:
        global_state_dim = (
            obs_dim * env_num_agents
        )  # Concatenated observations from all agents

    # Use RNN critic when GRU policy is enabled for temporal consistency
    # Use Deep Sets critic for football with DICO (as in Bettini et al. 2024)
    print(f"\n{'='*80}")
    print(f"CRITIC INITIALIZATION")
    print(f"{'='*80}")
    print(f"  scenario_name: {scenario_name}")
    print(f"  use_dico: {use_dico}")
    print(f"  use_gru_policy: {use_gru_policy}")
    print(f"  use_shared_critic: {use_shared_critic}")
    print(f"{'='*80}\n")

    if use_gru_policy:
        critic = CentralizedCriticRNN(
            gru_hidden_dim=gru_hidden_dim,
            fc_dim_size=gru_hidden_dim,  # Match actor embedding size
            num_agents=env_num_agents,  # Per-agent value predictions
            use_shared_value=use_shared_critic,
        )
        print(
            f"Using RNN critic (gru_hidden_dim={gru_hidden_dim}, shared_value={use_shared_critic})"
        )
    elif use_dico and scenario_name == "football":
        # Use Deep Sets critic for football (Bettini et al. 2024)
        critic = CentralizedCriticDeepSets(
            hidden_dim=critic_hidden_dim,
            num_layers=critic_num_layers,
            num_agents=env_num_agents,
            gat_out_features=32,  # Bettini: intermediate_sizes=[32]
            use_shared_value=use_shared_critic,
        )
        print(
            f"Using Deep Sets (GATv2) critic for football "
            f"(hidden_dim={critic_hidden_dim}, num_layers={critic_num_layers}, shared_value={use_shared_critic})"
        )
    else:
        critic = CentralizedCritic(
            hidden_dim=critic_hidden_dim,
            num_layers=critic_num_layers,
            num_agents=env_num_agents,  # Per-agent value predictions
            # use_shared_value=use_shared_critic,
        )
        print(
            f"Using MLP critic (hidden_dim={critic_hidden_dim}, num_layers={critic_num_layers}, shared_value={use_shared_critic})"
        )

    # Initialize parameters with dummy inputs
    rng = jax.random.PRNGKey(0)
    rng, policy_rng, hn_rng, critic_rng = jax.random.split(rng, 4)

    # For the shared-policy baseline (no HN) on scenarios that expose dynamic env
    # context (reverse_transport: package mass/size; pressure_plate: door/plate/goal),
    # append the context directly to the policy observation so the baseline receives
    # the same information that the HyperLoRA hypernetwork receives via env_context_batch.
    _env_context_scenarios = ("reverse_transport", "pressure_plate")
    if (
        not use_hypernetwork
        and not use_dico
        and scenario_name in _env_context_scenarios
        and env_context_dim > 0
    ):
        policy_obs_dim = obs_dim + env_context_dim
        print(
            f"[Shared baseline] Appending env_context ({env_context_dim}D) to observations: "
            f"policy_obs_dim = {obs_dim} + {env_context_dim} = {policy_obs_dim}"
        )
    elif use_dico and scenario_name == "dispersion_vmas" and use_capability_context:
        # For DiCo with dispersion_vmas, append max_speed capability (1D) to observations
        policy_obs_dim = obs_dim + 1
        print(
            f"[DiCo] Appending max_speed capability (1D) to observations: "
            f"policy_obs_dim = {obs_dim} + 1 = {policy_obs_dim}"
        )
    else:
        policy_obs_dim = obs_dim

    # For SMAX with curriculum, use max_obs_dim to initialize policy
    dummy_obs_dim = max_obs_dim if scenario_name == "smax" else policy_obs_dim

    # For Deep Sets (football DICO), need batch_size = multiple of num_agents
    # to properly initialize GAT encoder
    if use_dico and scenario_name == "football":
        dummy_batch_size = env_num_agents  # Need at least num_agents for GAT
        dummy_obs = jnp.ones((dummy_batch_size, dummy_obs_dim))
        dummy_agent_ids = jnp.arange(env_num_agents)  # One ID per agent
    else:
        dummy_obs = jnp.ones((1, dummy_obs_dim))
        dummy_agent_ids = jnp.array([0])  # For DICO: agent ID

    dummy_global_state = jnp.ones((1, global_state_dim))

    if use_cash:
        dummy_hidden = jnp.zeros((env_num_agents, gru_hidden_dim))
        dummy_obs_seq = jnp.ones((1, env_num_agents, dummy_obs_dim))
        dummy_dones = jnp.zeros((1, env_num_agents), dtype=bool)
        # CASH expects raw ego capabilities and builds team capabilities internally.
        dummy_capability_context = jnp.ones((1, env_num_agents, context_dim))
        dummy_x = (dummy_obs_seq, dummy_dones, dummy_capability_context)
        policy_params = shared_policy.init(policy_rng, dummy_hidden, dummy_x)["params"]
        hn_params = None
    elif use_dico:
        # DICO: Initialize with agent_ids (both full DICO and without diversity control need agent_ids)
        policy_params = shared_policy.init(policy_rng, dummy_obs, dummy_agent_ids)[
            "params"
        ]
        hn_params = None
    elif use_hypernetwork:
        # Create dummy adapters for all layers
        dummy_adapters = {}

        if use_gru_policy:
            # For GRU policy: only final adapter from gru_hidden_dim -> action_dim
            final_idx = 1  # GRU only has one adapter layer
            dummy_adapters[f"A{final_idx}"] = jnp.ones((1, lora_rank, gru_hidden_dim))
            dummy_adapters[f"B{final_idx}"] = jnp.ones(
                (1, policy_action_dim, lora_rank)
            )
        else:
            # For MLP policy: adapters for each layer
            # Use policy_input_dim (which is max_obs_dim for SMAX) for first layer
            input_dim = policy_input_dim
            for i, output_dim in enumerate(policy_hidden_dims):
                layer_idx = i + 1
                dummy_adapters[f"A{layer_idx}"] = jnp.ones((1, lora_rank, input_dim))
                dummy_adapters[f"B{layer_idx}"] = jnp.ones((1, output_dim, lora_rank))
                input_dim = output_dim
            # Final adapter for output layer
            final_idx = len(policy_hidden_dims) + 1
            dummy_adapters[f"A{final_idx}"] = jnp.ones(
                (1, lora_rank, policy_hidden_dims[-1])
            )
            dummy_adapters[f"B{final_idx}"] = jnp.ones(
                (1, policy_action_dim, lora_rank)
            )

        # Dummy inputs for hypernetwork with cross-agent attention structure
        # Shape: (num_envs=1, num_agents=max_agents, feature_dim)
        dummy_task = jnp.ones((1, max_agents, task_embed_dim))
        dummy_context = jnp.ones((1, max_agents, context_dim))
        dummy_lidar = jnp.ones((1, max_agents, lidar_dim)) if lidar_dim > 0 else None
        dummy_food_positions = (
            jnp.ones((1, max_agents, food_position_dim))
            if food_position_dim > 0
            else None
        )
        dummy_agent_positions = (
            jnp.ones((1, max_agents, agent_position_dim))
            if agent_position_dim > 0
            else None
        )
        dummy_target_snd = (
            jnp.ones((1, max_agents, target_snd_dim)) if target_snd_dim > 0 else None
        )
        dummy_env_context = (
            jnp.ones((1, max_agents, env_context_dim)) if env_context_dim > 0 else None
        )
        dummy_mask = create_attention_mask(
            max_agents, max_agents
        )  # Returns None when equal
        # Tile mask if it exists, otherwise keep as None
        if dummy_mask is not None:
            dummy_mask = jnp.tile(
                dummy_mask, (1, 1, 1, 1)
            )  # (1, 1, max_agents, max_agents)

        # For GRU policy, initialize with dummy hidden state matching training shape
        if use_gru_policy:
            # Use same shape as will be used during training: (batch_size, gru_hidden_dim)
            # batch_size during training will be num_envs * env_num_agents
            dummy_hidden = jnp.zeros((1, gru_hidden_dim))
            # New call signature: (hidden, x, adapters) where x = (obs, dones, avail_actions)
            # IMPORTANT: ScannedRNN uses nn.scan with in_axes=0, so inputs need a time dimension
            # Shape should be (time_steps, batch, ...) - use (1, 1, ...) for initialization
            dummy_obs_seq = dummy_obs[None, ...]  # Add time dimension: (1, 1, obs_dim)
            dummy_dones = jnp.zeros((1, 1), dtype=bool)  # (time_steps, batch)
            dummy_avail_actions = jnp.ones(
                (1, 1, policy_action_dim)
            )  # (time_steps, batch, action_dim)
            dummy_x = (dummy_obs_seq, dummy_dones, dummy_avail_actions)
            policy_params = shared_policy.init(
                policy_rng, dummy_hidden, dummy_x, dummy_adapters
            )["params"]
        else:
            policy_params = shared_policy.init(policy_rng, dummy_obs, dummy_adapters)[
                "params"
            ]

        print(f"Policy parameters initialized:")
        print(f"  dummy_obs shape: {dummy_obs.shape}")
        print(f"  shared_policy type: {type(shared_policy).__name__}")
        if hasattr(shared_policy, "gru_hidden_dim"):
            print(f"  Policy gru_hidden_dim: {shared_policy.gru_hidden_dim}")
            # Check actual GRU parameters
            if "gru" in policy_params:
                gru_params = policy_params["gru"]
                if "ir" in gru_params and "kernel" in gru_params["ir"]:
                    kernel_shape = gru_params["ir"]["kernel"].shape
                    print(f"  GRU 'ir' kernel shape: {kernel_shape}")
                    # For GRUCell, kernel shape is (input_dim, 2*features)
                    expected_features = kernel_shape[1] // 2
                    print(f"  This means GRU has features={expected_features}")
                    if expected_features != shared_policy.gru_hidden_dim:
                        print(
                            f"  ERROR: Mismatch! Policy expects gru_hidden_dim={shared_policy.gru_hidden_dim}"
                        )
                if "ir" in gru_params and "kernel" in gru_params["ir"]:
                    print(
                        f"  GRU 'ir' kernel shape: {gru_params['ir']['kernel'].shape}"
                    )

        hn_params = hypernetwork.init(
            hn_rng,
            dummy_task,
            dummy_context,
            dummy_lidar,
            dummy_food_positions,
            dummy_agent_positions,
            dummy_target_snd,
            dummy_env_context,
            dummy_mask,
        )["params"]
    else:
        # No adapters: pass zeros for adapter dict
        dummy_adapters = {}

        if use_gru_policy:
            # For GRU policy: only final adapter layer with rank 0
            final_idx = 1
            dummy_adapters[f"A{final_idx}"] = jnp.zeros((1, 0, gru_hidden_dim))
            dummy_adapters[f"B{final_idx}"] = jnp.zeros((1, action_dim, 0))

            # GRU policy needs hidden state and (obs, dones, avail_actions) tuple
            # IMPORTANT: ScannedRNN uses nn.scan with in_axes=0, so inputs need a time dimension
            # Shape should be (time_steps, batch, ...) - use (1, 1, ...) for initialization
            dummy_hidden = jnp.zeros((1, gru_hidden_dim))
            dummy_obs_seq = dummy_obs[None, ...]  # Add time dimension: (1, 1, obs_dim)
            dummy_dones = jnp.zeros((1, 1), dtype=bool)  # (time_steps, batch)
            dummy_avail_actions = jnp.ones(
                (1, 1, policy_action_dim)
            )  # (time_steps, batch, action_dim)
            dummy_x = (dummy_obs_seq, dummy_dones, dummy_avail_actions)

            policy_params = shared_policy.init(
                policy_rng, dummy_hidden, dummy_x, dummy_adapters
            )["params"]
        else:
            # For MLP policy: zero-rank adapters for each layer
            # Use policy_obs_dim so the first adapter matches the (possibly augmented) input
            input_dim = policy_obs_dim
            for i, output_dim in enumerate(policy_hidden_dims):
                layer_idx = i + 1
                dummy_adapters[f"A{layer_idx}"] = jnp.zeros((1, 0, input_dim))
                dummy_adapters[f"B{layer_idx}"] = jnp.zeros((1, output_dim, 0))
                input_dim = output_dim
            # Final adapter for output layer
            final_idx = len(policy_hidden_dims) + 1
            dummy_adapters[f"A{final_idx}"] = jnp.zeros((1, 0, policy_hidden_dims[-1]))
            dummy_adapters[f"B{final_idx}"] = jnp.zeros((1, action_dim, 0))

            policy_params = shared_policy.init(policy_rng, dummy_obs, dummy_adapters)[
                "params"
            ]

        hn_params = None

    # Initialize critic parameters (RNN critic needs hidden state and dones)
    if use_gru_policy:
        dummy_hidden = jnp.zeros((1, gru_hidden_dim))  # Critic hidden state
        dummy_dones = jnp.zeros(
            (1, 1), dtype=bool
        )  # Dummy dones for sequence processing
        # RNN critic expects (hidden, (global_state, dones)) signature
        dummy_global_state_seq = jnp.expand_dims(
            dummy_global_state, 0
        )  # Add time dimension
        critic_params = critic.init(
            critic_rng, dummy_hidden, (dummy_global_state_seq, dummy_dones)
        )["params"]
    else:
        critic_params = critic.init(critic_rng, dummy_global_state)["params"]

    # Get max grad norm from config
    max_grad_norm = config["training"].get("max_grad_norm", 0.5)
    hn_grad_norm = config["training"].get(
        "hypernetwork_grad_norm", 0.1
    )  # CRITICAL: Very low for HN stability
    critic_grad_norm = config["training"].get(
        "critic_grad_norm", 10.0
    )  # Higher for critic

    policy_state = create_train_state(
        shared_policy, policy_params, config, max_grad_norm, lr_key="learning_rate"
    )
    if use_hypernetwork:
        hn_state = create_train_state(
            hypernetwork, hn_params, config, hn_grad_norm, lr_key="hn_learning_rate"
        )
    else:
        hn_state = None
    critic_state = create_train_state(
        critic, critic_params, config, critic_grad_norm, lr_key="critic_learning_rate"
    )

    if use_cuda:
        with jax.default_device(jax_device):
            policy_state = jax.device_put(policy_state, jax_device)
            if use_hypernetwork:
                hn_state = jax.device_put(hn_state, jax_device)
            critic_state = jax.device_put(critic_state, jax_device)

    print("Models initialized successfully!")

    # ========================================================================
    # Main Training Loop
    # ========================================================================
    print("\nStarting training...")

    # ========================================================================
    # Collect Initial Observations for SND Calculation (if enabled)
    # ========================================================================
    snd_obs_buffer = None
    if use_diversity_control and use_snd_obs_buffer:
        print(
            f"\nCollecting {snd_obs_buffer_size} initial observations for SND calculation..."
        )
        if snd_obs_filter_food:
            print("  Filtering: Only collecting observations where food is detected")

        snd_obs_buffer = []
        collected_count = 0

        while collected_count < snd_obs_buffer_size:
            # Reset environment to get initial observations
            if scenario_name == "smax":
                rng_key = jax.random.PRNGKey(seed + collected_count)
                obs_dict, state = env.reset(rng_key)
                # Convert dict to list for consistency
                obs = [obs_dict[agent] for agent in env.agents]
            else:
                obs = env.reset()

            # For each environment's agent 0 observation (to maintain consistency with SND calculation)
            if scenario_name == "smax":
                # SMAX: observations are already JAX arrays
                for env_idx in range(num_envs):
                    obs_sample = obs[0][env_idx]  # Agent 0 from env env_idx

                    # Filter for food detection if enabled
                    if snd_obs_filter_food:
                        # For SMAX, check if any enemies are visible (assuming last N dims are enemy observations)
                        # This is scenario-specific, so we'll keep all for SMAX
                        snd_obs_buffer.append(obs_sample)
                        collected_count += 1
                    else:
                        snd_obs_buffer.append(obs_sample)
                        collected_count += 1

                    if collected_count >= snd_obs_buffer_size:
                        break
            else:
                # VMAS: observations are torch tensors
                # Get agent 0's max lidar range for food detection check
                agent_0_lidar_range = (
                    default_lidar_ranges[0] if default_lidar_ranges else 0.5
                )

                for env_idx in range(num_envs):
                    obs_sample = (
                        obs[0][env_idx].cpu().numpy()
                    )  # Agent 0 from env env_idx

                    # Debug: Print first 5 observations
                    if collected_count < 5:
                        print(f"\n  DEBUG - Observation {collected_count}:")
                        print(f"    Agent 0 lidar_range: {agent_0_lidar_range}")
                        print(f"    Full obs (shape {obs_sample.shape}): {obs_sample}")
                        if obs_dim > 8:
                            print(f"    pos(2) + vel(2) + food(4): {obs_sample[:8]}")
                            lidar_readings = obs_sample[8:]
                            print(
                                f"    lidar readings ({len(lidar_readings)} dims): {lidar_readings}"
                            )
                            # Lidar returns actual distances: max_range when nothing detected
                            # Values < max_range indicate detection
                            food_detected = np.any(
                                lidar_readings < agent_0_lidar_range - 1e-6
                            )
                            print(
                                f"    Food detected (any < {agent_0_lidar_range}): {food_detected}"
                            )

                    # Filter for food detection if enabled
                    if snd_obs_filter_food:
                        # Check if lidar readings indicate food is detected
                        # Observation structure: pos(2) + vel(2) + food(4) + lidar(varies)
                        # For dispersion/sampling scenarios, lidar_food is typically the last 12 dims
                        if obs_dim > 8:
                            lidar_readings = obs_sample[
                                8:
                            ]  # Everything after pos, vel, food
                            # Lidar returns actual distance values
                            # When nothing detected: returns max_range (e.g., 0.5 or 0.3)
                            # When food detected: returns distance < max_range
                            if np.any(lidar_readings < agent_0_lidar_range - 1e-6):
                                snd_obs_buffer.append(obs_sample)
                                collected_count += 1
                        else:
                            # If no lidar, just add the observation
                            snd_obs_buffer.append(obs_sample)
                            collected_count += 1
                    else:
                        snd_obs_buffer.append(obs_sample)
                        collected_count += 1

                    if collected_count >= snd_obs_buffer_size:
                        break

        # Convert to JAX array for efficient sampling later
        snd_obs_buffer = jnp.array(snd_obs_buffer)
        print(
            f"  Collected {len(snd_obs_buffer)} observations (shape: {snd_obs_buffer.shape})"
        )

        if snd_obs_filter_food and len(snd_obs_buffer) < snd_obs_buffer_size:
            print(
                f"  Warning: Only collected {len(snd_obs_buffer)} observations with food detection"
            )
            print(
                f"           (requested {snd_obs_buffer_size}). Consider disabling snd_obs_filter_food."
            )

    # ========================================================================
    # Initialize Adapter-based SND Buffer (for storing hypernetwork query observations)
    # ========================================================================
    adapter_snd_buffer = None
    previous_trajectory_obs = (
        None  # Store trajectory obs from previous rollout for SND calculation
    )
    if use_adapter_snd and use_hypernetwork:
        print(
            f"\nInitializing adapter SND buffer (max size: {adapter_snd_buffer_size})..."
        )
        # Buffer stores: observations, task vectors, context vectors, lidar (if used)
        # Each entry is a dict with keys: 'obs', 'task', 'context', 'lidar'
        adapter_snd_buffer = []

    # ========================================================================
    # Initialize Adapter Effect Tracking (for measuring adapter impact on policy)
    # ========================================================================
    # Track difference between policy outputs WITH and WITHOUT adapters
    # This helps understand how much adapters modify the base policy behavior
    adapter_effect_buffer = (
        []
    )  # Stores raw adapter impacts per step (batch_size, action_dim)
    adapter_effect_norms_buffer = []  # Stores L2 norms for backward compatibility

    # Buffers for action distribution tracking
    backbone_actions_buffer = []  # Stores backbone-only action means
    combined_actions_buffer = []  # Stores combined (backbone + adapters) action means

    # Buffers for action distribution tracking
    backbone_actions_buffer = []  # Stores backbone-only action means
    combined_actions_buffer = []  # Stores combined (backbone + adapters) action means

    # Evaluation parameters from config
    eval_interval = config["evaluation"].get("eval_interval", 100)
    eval_start_episode = config["evaluation"].get("eval_start_episode", 0)
    num_eval_episodes = config["evaluation"].get("num_eval_episodes", 10)
    max_eval_steps = config["evaluation"].get("max_eval_steps", 200)

    # Curriculum learning parameters (for SMAX)
    use_curriculum = config["training"].get("use_curriculum", False)
    curriculum_type = config["training"].get(
        "curriculum_type", "damage_scale"
    )  # "enemy_number" or "damage_scale"

    # Enemy number curriculum parameters (stage1 and stage2 episodes already loaded above)
    curriculum_stage1_num_enemies = config["training"].get(
        "curriculum_stage1_num_enemies", 1
    )
    curriculum_stage2_num_enemies = config["training"].get(
        "curriculum_stage2_num_enemies", 2
    )
    curriculum_stage3_num_enemies = config["training"].get(
        "curriculum_stage3_num_enemies", 3
    )

    # Damage scale curriculum parameters (old approach)
    curriculum_stage2_damage_scale = config["training"].get(
        "curriculum_stage2_damage_scale", 0.5
    )
    curriculum_stage3_episodes = config["training"].get("curriculum_stage3_episodes", 0)
    curriculum_stage3_damage_scale = config["training"].get(
        "curriculum_stage3_damage_scale", 0.75
    )
    curriculum_stage4_episodes = config["training"].get("curriculum_stage4_episodes", 0)
    curriculum_stage4_damage_scale = config["training"].get(
        "curriculum_stage4_damage_scale", 0.85
    )
    curriculum_stage5_episodes = config["training"].get("curriculum_stage5_episodes", 0)
    curriculum_stage5_damage_scale = config["training"].get(
        "curriculum_stage5_damage_scale", 0.35
    )
    curriculum_stage6_episodes = config["training"].get("curriculum_stage6_episodes", 0)
    curriculum_stage6_damage_scale = config["training"].get(
        "curriculum_stage6_damage_scale", 0.50
    )
    curriculum_stage7_episodes = config["training"].get("curriculum_stage7_episodes", 0)
    curriculum_stage7_damage_scale = config["training"].get(
        "curriculum_stage7_damage_scale", 0.75
    )

    def get_curriculum_params(episode_num):
        """Determine curriculum parameters based on episode number."""
        if not use_curriculum or scenario_name != "smax":
            return None, None, None  # No curriculum for non-SMAX environments

        if curriculum_type == "enemy_number":
            # Enemy number curriculum: gradually increase number of enemies
            stage1_end = curriculum_stage1_episodes
            stage2_end = stage1_end + curriculum_stage2_episodes

            if episode_num < stage1_end:
                # Stage 1: Fight 1 enemy (3v1)
                return True, 1.0, curriculum_stage1_num_enemies
            elif episode_num < stage2_end:
                # Stage 2: Fight 2 enemies (3v2)
                return True, 1.0, curriculum_stage2_num_enemies
            else:
                # Stage 3: Fight 3 enemies (3v3) - full game
                return True, 1.0, curriculum_stage3_num_enemies
        else:
            # Damage scale curriculum (old approach)
            stage1_end = curriculum_stage1_episodes
            stage2_end = stage1_end + curriculum_stage2_episodes
            stage3_end = stage2_end + curriculum_stage3_episodes
            stage4_end = stage3_end + curriculum_stage4_episodes
            stage5_end = stage4_end + curriculum_stage5_episodes
            stage6_end = stage5_end + curriculum_stage6_episodes
            stage7_end = stage6_end + curriculum_stage7_episodes

            if episode_num < stage1_end:
                # Stage 1: Enemies don't shoot
                return False, 1.0, None
            elif episode_num < stage2_end:
                # Stage 2: Enemies shoot with reduced damage
                return True, curriculum_stage2_damage_scale, None
            elif curriculum_stage3_episodes > 0 and episode_num < stage3_end:
                # Stage 3: Enemies shoot with moderate damage (optional stage)
                return True, curriculum_stage3_damage_scale, None
            elif curriculum_stage4_episodes > 0 and episode_num < stage4_end:
                # Stage 4: Enemies shoot with high damage (optional stage)
                return True, curriculum_stage4_damage_scale, None
            elif curriculum_stage5_episodes > 0 and episode_num < stage5_end:
                # Stage 5
                return True, curriculum_stage5_damage_scale, None
            elif curriculum_stage6_episodes > 0 and episode_num < stage6_end:
                # Stage 6
                return True, curriculum_stage6_damage_scale, None
            elif curriculum_stage7_episodes > 0 and episode_num < stage7_end:
                # Stage 7
                return True, curriculum_stage7_damage_scale, None
            else:
                # Final stage: Full training
                return True, 1.0, None

    # Track current curriculum stage to detect transitions
    current_curriculum_stage = None

    # Initialize adapter storage for visualization (set during adapter generation each episode)
    rollout_adapters_unscaled = None
    rollout_adapters_scaled = None

    # ETA tracking: cumulative post-warmup time so we can project total run length.
    # Episode 0 (and 1) dominate compile time; skip them from the rolling average.
    _eta_warmup_episodes = 2
    _eta_post_warmup_total = 0.0
    _eta_post_warmup_count = 0
    _train_loop_wall_start = time.time()

    for episode in range(num_episodes):
        # ====================================================================
        # JAX Profiling: Capture episodes 5-7 for representative performance
        # Skipping first few episodes to avoid JIT compilation overhead
        # ====================================================================
        if args.jax_profile and episode == 5:
            print(f"\n{'='*80}")
            print("Starting JAX profiler (episodes 5-7)...")
            print(f"{'='*80}\n")
            jax.profiler.start_trace(args.profile_dir)

        if args.jax_profile and episode == 7:
            jax.profiler.stop_trace()
            print(f"\n{'='*80}")
            print(f"JAX profiler trace saved to: {args.profile_dir}")
            print("View with: tensorboard --logdir={args.profile_dir}")
            print(f"{'='*80}\n")

        # ====================================================================
        # Reset Adapter SND Buffer at start of each episode
        # Buffer will initially have num_envs entries (one per environment)
        # Can grow up to 2 * num_agents * num_envs as agents are requeried
        # ====================================================================
        if use_adapter_snd and adapter_snd_buffer is not None:
            adapter_snd_buffer = []  # Clear and refill each episode

        # ====================================================================
        # Reset Adapter Effect Buffer at start of each episode
        # Track difference between policy outputs WITH and WITHOUT adapters
        # ====================================================================
        adapter_effect_buffer = []  # Clear for new episode
        adapter_effect_norms_buffer = []  # Clear norms buffer too
        backbone_actions_buffer = []  # Clear backbone actions buffer
        combined_actions_buffer = []  # Clear combined actions buffer
        backbone_actions_buffer = []  # Clear backbone actions buffer
        combined_actions_buffer = []  # Clear combined actions buffer

        # ====================================================================
        # Curriculum Learning: Update environment if stage changes (SMAX only)
        # ====================================================================
        if use_curriculum and scenario_name == "smax":
            enemy_shoots, damage_scale, num_enemies_curriculum = get_curriculum_params(
                episode
            )

            # Determine current stage based on curriculum type
            if curriculum_type == "enemy_number":
                stage1_end = curriculum_stage1_episodes
                stage2_end = stage1_end + curriculum_stage2_episodes

                if episode < stage1_end:
                    new_stage = 1
                elif episode < stage2_end:
                    new_stage = 2
                else:
                    new_stage = 3  # Final stage
            else:
                # Damage scale curriculum
                stage1_end = curriculum_stage1_episodes
                stage2_end = stage1_end + curriculum_stage2_episodes
                stage3_end = stage2_end + curriculum_stage3_episodes
                stage4_end = stage3_end + curriculum_stage4_episodes
                stage5_end = stage4_end + curriculum_stage5_episodes
                stage6_end = stage5_end + curriculum_stage6_episodes
                stage7_end = stage6_end + curriculum_stage7_episodes

                if episode < stage1_end:
                    new_stage = 1
                elif episode < stage2_end:
                    new_stage = 2
                elif curriculum_stage3_episodes > 0 and episode < stage3_end:
                    new_stage = 3
                elif curriculum_stage4_episodes > 0 and episode < stage4_end:
                    new_stage = 4
                elif curriculum_stage5_episodes > 0 and episode < stage5_end:
                    new_stage = 5
                elif curriculum_stage6_episodes > 0 and episode < stage6_end:
                    new_stage = 6
                elif curriculum_stage7_episodes > 0 and episode < stage7_end:
                    new_stage = 7
                else:
                    new_stage = 8  # Final stage

            # Recreate environment if stage has changed
            if current_curriculum_stage != new_stage:
                print(f"\n{'='*80}")
                print(
                    f"CURRICULUM TRANSITION: Entering Stage {new_stage} at Episode {episode}"
                )

                if curriculum_type == "enemy_number":
                    # Enemy number curriculum
                    if new_stage == 1:
                        print(
                            f"  Stage 1: 3 allies vs {num_enemies_curriculum} enemy (learning basic combat)"
                        )
                    elif new_stage == 2:
                        print(
                            f"  Stage 2: 3 allies vs {num_enemies_curriculum} enemies (learning coordination)"
                        )
                    else:
                        print(
                            f"  Stage 3: 3 allies vs {num_enemies_curriculum} enemies - FULL GAME (final difficulty)"
                        )
                else:
                    # Damage scale curriculum
                    if new_stage == 1:
                        print(
                            f"  Stage 1: Enemies DO NOT shoot (learning basic movement & targeting)"
                        )
                    elif new_stage == 2:
                        print(
                            f"  Stage 2: Enemies shoot with {damage_scale*100:.0f}% damage (very gentle introduction)"
                        )
                    elif new_stage == 3:
                        print(
                            f"  Stage 3: Enemies shoot with {damage_scale*100:.0f}% damage (moderate difficulty)"
                        )
                    elif new_stage == 4:
                        print(
                            f"  Stage 4: Enemies shoot with {damage_scale*100:.0f}% damage (high difficulty)"
                        )
                    elif new_stage == 5:
                        print(
                            f"  Stage 5: Enemies shoot with {damage_scale*100:.0f}% damage"
                        )
                    elif new_stage == 6:
                        print(
                            f"  Stage 6: Enemies shoot with {damage_scale*100:.0f}% damage"
                        )
                    elif new_stage == 7:
                        print(
                            f"  Stage 7: Enemies shoot with {damage_scale*100:.0f}% damage"
                        )
                    else:
                        print(
                            f"  Stage 8: Full training with 100% enemy damage (FINAL STAGE)"
                        )
                print(f"{'='*80}\n")

                # CRITICAL: Reset entropy to force exploration of new actions when action space changes
                if curriculum_type == "enemy_number":
                    # Action space changes when enemy count changes (new attack actions become available)
                    # Reset entropy to initial value to encourage exploration of new attack actions
                    entropy_coef = config["training"]["entropy_coef"]
                    print(
                        f"  ENTROPY RESET: Resetting entropy coefficient to {entropy_coef:.3f} to explore new actions"
                    )
                    print(
                        f"  (New actions for attacking enemy {num_enemies_curriculum-1} are now available)\n"
                    )

                # Update SMAX kwargs with new curriculum parameters
                smax_kwargs["enemy_shoots"] = enemy_shoots
                smax_kwargs["enemy_damage_scale"] = damage_scale
                if num_enemies_curriculum is not None:
                    smax_kwargs["num_enemies"] = num_enemies_curriculum

                # Recreate environment with new parameters
                env = make_vmas_env(
                    scenario_name,
                    env_num_agents,
                    num_envs,
                    device=torch_device,
                    continuous_actions=continuous_actions,
                    penalise_by_time=penalise_by_time,
                    share_reward=share_reward,
                    distance_shaping_coef=distance_shaping_coef,
                    agent_capabilities=agent_capabilities,
                    fixed_food_positions=fixed_food_positions,
                    **football_kwargs,
                    **simple_tag_kwargs,
                    **sampling_kwargs,
                    **grassland_kwargs,
                    **smax_kwargs,
                    **reverse_transport_kwargs,
                )

                current_curriculum_stage = new_stage

        # ====================================================================
        # Curriculum Learning: Update environment if stage changes (Football)
        # ====================================================================
        if scenario_name == "football" and football_use_curriculum:
            # Determine current stage based on episode number
            stage1_end = curriculum_stage1_episodes
            stage2_end = stage1_end + curriculum_stage2_episodes

            if episode < stage1_end:
                new_stage = 1
            elif episode < stage2_end:
                new_stage = 2
            else:
                new_stage = 3  # Final stage

            # Recreate environment if stage has changed
            if current_curriculum_stage != new_stage:
                print(f"\n{'='*80}")
                print(
                    f"FOOTBALL CURRICULUM TRANSITION: Entering Stage {new_stage} at Episode {episode}"
                )

                if new_stage == 1:
                    # Stage 1: No adversaries
                    print(
                        f"  Stage 1: Training WITHOUT adversaries (learning basic ball control)"
                    )
                    print(f"    - Dense reward: ENABLED")
                    print(f"    - Ball-to-goal shaping: 100.0")
                    print(f"    - Agent-to-ball shaping: 10.0")
                    football_kwargs["ai_red_agents"] = True
                    football_kwargs["disable_ai_red"] = True
                    football_kwargs["dense_reward"] = True
                    football_kwargs["pos_shaping_factor_ball_goal"] = 100.0
                    football_kwargs["pos_shaping_factor_agent_ball"] = 10.0
                elif new_stage == 2:
                    # Stage 2: Half-strength adversaries
                    print(f"  Stage 2: Training with HALF-STRENGTH adversaries")
                    print(f"    - ai_speed_strength: 0.5")
                    print(f"    - ai_decision_strength: 0.5")
                    print(f"    - ai_precision_strength: 0.5")
                    print(f"    - Dense reward: ENABLED (reduced shaping)")
                    print(f"    - Ball-to-goal shaping: 50.0 (50% reduced)")
                    print(f"    - Agent-to-ball shaping: 5.0 (50% reduced)")
                    football_kwargs["ai_red_agents"] = True
                    football_kwargs["disable_ai_red"] = False
                    football_kwargs["ai_strength"] = (
                        1.0,
                        0.5,
                    )  # (blue, red) - controls speed
                    football_kwargs["ai_decision_strength"] = (1.0, 0.5)
                    football_kwargs["ai_precision_strength"] = (1.0, 0.5)
                    football_kwargs["dense_reward"] = True  # Keep dense reward enabled
                    football_kwargs["pos_shaping_factor_ball_goal"] = (
                        50.0  # Reduced from Stage 1
                    )
                    football_kwargs["pos_shaping_factor_agent_ball"] = (
                        5.0  # Reduced from Stage 1
                    )
                else:
                    # Stage 3: Full-strength adversaries
                    print(
                        f"  Stage 3: Training with FULL-STRENGTH adversaries (FINAL STAGE)"
                    )
                    print(f"    - Dense reward: ENABLED (further reduced shaping)")
                    print(
                        f"    - Ball-to-goal shaping: 25.0 (75% reduced from Stage 1)"
                    )
                    print(
                        f"    - Agent-to-ball shaping: 2.5 (75% reduced from Stage 1)"
                    )
                    football_kwargs["ai_red_agents"] = True
                    football_kwargs["disable_ai_red"] = False
                    football_kwargs["ai_strength"] = (1.0, 1.0)
                    football_kwargs["ai_decision_strength"] = (1.0, 1.0)
                    football_kwargs["ai_precision_strength"] = (1.0, 1.0)
                    football_kwargs["dense_reward"] = True  # Keep dense reward enabled
                    football_kwargs["pos_shaping_factor_ball_goal"] = (
                        25.0  # Further reduced
                    )
                    football_kwargs["pos_shaping_factor_agent_ball"] = (
                        2.5  # Further reduced
                    )

                print(f"{'='*80}\n")

                # Recreate environment with new parameters
                env = make_vmas_env(
                    scenario_name,
                    env_num_agents,
                    num_envs,
                    device=torch_device,
                    continuous_actions=continuous_actions,
                    penalise_by_time=penalise_by_time,
                    share_reward=share_reward,
                    distance_shaping_coef=distance_shaping_coef,
                    agent_capabilities=agent_capabilities,
                    fixed_food_positions=fixed_food_positions,
                    **football_kwargs,
                    **simple_tag_kwargs,
                    **sampling_kwargs,
                    **grassland_kwargs,
                    **smax_kwargs,
                    **reverse_transport_kwargs,
                    **pressure_plate_kwargs,
                )

                current_curriculum_stage = new_stage

        # ====================================================================
        # Determine number of agents for this episode (mixed training)
        # ====================================================================
        if mixed_agent_training:
            if per_env_agent_variation:
                # Per-environment variation: sample different agent count for each environment
                # Shape: [num_envs]
                current_num_agents_per_env = np_rng.integers(
                    min_agents, max_agents + 1, size=num_envs
                )
                # For compatibility with existing code, use max as the "current_num_agents"
                current_num_agents = int(np.max(current_num_agents_per_env))
            else:
                # Per-episode variation: all environments have same agent count
                current_num_agents = np_rng.integers(min_agents, max_agents + 1)
                # Create a uniform vector for consistency
                current_num_agents_per_env = np.full(
                    num_envs, current_num_agents, dtype=np.int32
                )
        else:
            # Use fixed number of agents
            current_num_agents = num_agents
            current_num_agents_per_env = np.full(
                num_envs, current_num_agents, dtype=np.int32
            )

        # Log agent count changes occasionally
        if mixed_agent_training and (episode % log_interval == 0 or episode < 5):
            if per_env_agent_variation:
                print(
                    f"\nEpisode {episode}: Training with {current_num_agents_per_env.min()}-{current_num_agents_per_env.max()} agents "
                    f"across {num_envs} envs (max capacity: {max_agents})"
                )
            else:
                print(
                    f"\nEpisode {episode}: Training with {current_num_agents} agents (max capacity: {max_agents})"
                )
        # Setup agent capabilities for this episode
        # For mixed agent training, determine if capabilities should be homogeneous or heterogeneous
        # Check if we want homogeneous (same for all agents) or heterogeneous (different per agent) capabilities
        # For capability interpolation experiment, we want heterogeneous capabilities
        use_homogeneous_caps = config["env"].get("homogeneous_capabilities", True)

        if mixed_agent_training:
            # For reverse_transport: use speed and force_multiplier
            if scenario_name == "reverse_transport":
                if use_fixed_capabilities:
                    # Use fixed capabilities from config-derived defaults
                    # (already validated/expanded to env_num_agents earlier).
                    agent_speeds = default_speeds[:current_num_agents]
                    agent_force_multipliers = default_force_multipliers[
                        :current_num_agents
                    ]
                elif use_homogeneous_caps:
                    # All agents get the same random values (homogeneous team)
                    base_speed = np_rng.uniform(speed_range[0], speed_range[1])
                    base_force = np_rng.uniform(
                        force_multiplier_range[0], force_multiplier_range[1]
                    )
                    agent_speeds = [base_speed] * current_num_agents
                    agent_force_multipliers = [base_force] * current_num_agents
                else:
                    # Each agent gets different random values (heterogeneous team)
                    agent_speeds = np_rng.uniform(
                        speed_range[0], speed_range[1], size=current_num_agents
                    ).tolist()
                    agent_force_multipliers = np_rng.uniform(
                        force_multiplier_range[0],
                        force_multiplier_range[1],
                        size=current_num_agents,
                    ).tolist()

                # Pad to max_agents with dummy values
                agent_speeds += [1.0] * (max_agents - current_num_agents)
                agent_force_multipliers += [1.0] * (max_agents - current_num_agents)
            elif scenario_name == "dispersion_vmas":
                # For dispersion_vmas: use only max_speed
                if use_fixed_capabilities:
                    base_max_speed = 1.0
                    agent_max_speeds = [base_max_speed] * current_num_agents
                elif use_homogeneous_caps:
                    # All agents get the same random max_speed (homogeneous team)
                    base_max_speed = np_rng.uniform(
                        max_speed_range[0], max_speed_range[1]
                    )
                    agent_max_speeds = [base_max_speed] * current_num_agents
                else:
                    # Each agent gets different random max_speed (heterogeneous team)
                    agent_max_speeds = np_rng.uniform(
                        max_speed_range[0], max_speed_range[1], size=current_num_agents
                    ).tolist()

                    # Debug: Print heterogeneous capabilities on first episode
                    if episode == 1:
                        print(
                            f"\n[DEBUG] Generated HETEROGENEOUS max_speeds (episode {episode}, {current_num_agents} active agents):"
                        )
                        for i in range(min(4, current_num_agents)):
                            print(f"  Agent {i}: max_speed={agent_max_speeds[i]:.3f}")

                # Pad to max_agents with dummy values
                agent_max_speeds += [1.0] * (max_agents - current_num_agents)
            else:
                # For other scenarios: use speed and lidar_range
                if use_fixed_capabilities:
                    base_speed = 0.5
                    base_lidar = 0.5
                    agent_speeds = [base_speed] * current_num_agents
                    agent_lidar_ranges = [base_lidar] * current_num_agents
                elif use_homogeneous_caps:
                    # All agents get the same random values (homogeneous team)
                    base_speed = np_rng.uniform(speed_range[0], speed_range[1])
                    base_lidar = np_rng.uniform(
                        lidar_range_range[0], lidar_range_range[1]
                    )
                    agent_speeds = [base_speed] * current_num_agents
                    agent_lidar_ranges = [base_lidar] * current_num_agents
                else:
                    # Each agent gets different random values (heterogeneous team)
                    agent_speeds = np_rng.uniform(
                        speed_range[0], speed_range[1], size=current_num_agents
                    ).tolist()
                    agent_lidar_ranges = np_rng.uniform(
                        lidar_range_range[0],
                        lidar_range_range[1],
                        size=current_num_agents,
                    ).tolist()

                    # Debug: Print heterogeneous capabilities on first episode
                    if episode == 1:
                        print(
                            f"\n[DEBUG] Generated HETEROGENEOUS capabilities (episode {episode}, {current_num_agents} active agents):"
                        )
                        for i in range(min(4, current_num_agents)):
                            print(
                                f"  Agent {i}: speed={agent_speeds[i]:.3f}, lidar_range={agent_lidar_ranges[i]:.3f}"
                            )

                # Pad to max_agents with dummy values
                agent_speeds += [1.0] * (max_agents - current_num_agents)
                agent_lidar_ranges += [0.5] * (max_agents - current_num_agents)
        else:
            # Original behavior: heterogeneous capabilities or fixed
            if use_fixed_capabilities:
                # Use fixed capabilities (no randomization)
                if scenario_name == "reverse_transport":
                    agent_speeds = default_speeds
                    agent_force_multipliers = default_force_multipliers
                elif scenario_name == "dispersion_vmas":
                    agent_max_speeds = default_max_speeds
                else:
                    agent_speeds = default_speeds
                    agent_lidar_ranges = default_lidar_ranges
            else:
                # Randomize agent capabilities for this episode (CTDE approach)
                # This enables the policy to generalize to different capability combinations
                if scenario_name == "reverse_transport":
                    agent_speeds = np_rng.uniform(
                        speed_range[0], speed_range[1], size=num_agents
                    ).tolist()
                    agent_force_multipliers = np_rng.uniform(
                        force_multiplier_range[0],
                        force_multiplier_range[1],
                        size=num_agents,
                    ).tolist()
                elif scenario_name == "dispersion_vmas":
                    # Randomize max_speed for dispersion_vmas
                    if use_homogeneous_caps:
                        # All agents get the same random max_speed
                        base_max_speed = np_rng.uniform(
                            max_speed_range[0], max_speed_range[1]
                        )
                        agent_max_speeds = [base_max_speed] * num_agents
                    else:
                        # Each agent gets different random max_speed
                        agent_max_speeds = np_rng.uniform(
                            max_speed_range[0], max_speed_range[1], size=num_agents
                        ).tolist()
                else:
                    agent_speeds = np_rng.uniform(
                        speed_range[0], speed_range[1], size=num_agents
                    ).tolist()
                    agent_lidar_ranges = np_rng.uniform(
                        lidar_range_range[0], lidar_range_range[1], size=num_agents
                    ).tolist()

        # Update environment capabilities without recreating the environment
        # Skip for football as capabilities are fixed in the wrapper
        if scenario_name == "simple_tag":
            # For simple_tag, we need separate capabilities for adversaries and good agents
            if use_fixed_capabilities:
                # Use fixed capabilities from config
                fixed_caps = config["env"].get("fixed_capabilities", {})
                adversary_speeds = fixed_caps.get(
                    "adversary_speeds", [1.0] * config["env"].get("num_adversaries", 3)
                )
                agent_speeds = fixed_caps.get(
                    "agent_speeds", [1.3] * config["env"].get("num_agents", 1)
                )
                adversary_lidar_ranges = fixed_caps.get(
                    "adversary_lidar_ranges",
                    [0.5] * config["env"].get("num_adversaries", 3),
                )
                agent_lidar_ranges = fixed_caps.get(
                    "agent_lidar_ranges", [0.6] * config["env"].get("num_agents", 1)
                )
            else:
                # Randomize capabilities for simple_tag
                num_adversaries = config["env"].get("num_adversaries", 3)
                num_good_agents = config["env"].get("num_agents", 1)
                adversary_speeds = np_rng.uniform(
                    0.8, 1.2, size=num_adversaries
                ).tolist()
                agent_speeds = np_rng.uniform(1.1, 1.5, size=num_good_agents).tolist()
                adversary_lidar_ranges = np_rng.uniform(
                    0.4, 0.6, size=num_adversaries
                ).tolist()
                agent_lidar_ranges = np_rng.uniform(
                    0.5, 0.7, size=num_good_agents
                ).tolist()

            agent_capabilities = {
                "adversary_speeds": adversary_speeds,
                "agent_speeds": agent_speeds,
                "adversary_lidar_ranges": adversary_lidar_ranges,
                "agent_lidar_ranges": agent_lidar_ranges,
            }
            env.scenario.update_agent_capabilities(agent_capabilities)

            # Create merged lists for context creation (adversaries first, then good agents)
            agent_speeds = adversary_speeds + agent_speeds
            agent_lidar_ranges = adversary_lidar_ranges + agent_lidar_ranges
        elif scenario_name == "grassland":
            # For grassland, no capability updates needed (agents have fixed speeds in the scenario)
            pass
        elif scenario_name == "smax":
            # For SMAX, no capability updates needed (unit types are part of the environment state)
            pass
        elif scenario_name == "reverse_transport":
            # Update both speed and force_multiplier each episode
            agent_capabilities = {
                "speed": agent_speeds,
                "force_multiplier": agent_force_multipliers,
            }
            if hasattr(env, "scenario") and hasattr(
                env.scenario, "update_agent_capabilities"
            ):
                env.scenario.update_agent_capabilities(agent_capabilities)

            # Scale package mass proportionally with current team size
            if scale_package_mass_with_agents and mass_per_agent is not None:
                scaled_mass = mass_per_agent * current_num_agents
                env.scenario.update_package_properties(package_mass=scaled_mass)
                if episode % log_interval == 0 or episode < 5:
                    print(
                        f"  Package mass scaled to {scaled_mass:.1f} for {current_num_agents} agents"
                    )
        elif scenario_name == "pressure_plate":
            # Update only speed for pressure_plate (no lidar or force_multiplier)
            agent_capabilities = {
                "speed": agent_speeds,
            }
            if hasattr(env, "scenario") and hasattr(
                env.scenario, "update_agent_capabilities"
            ):
                env.scenario.update_agent_capabilities(agent_capabilities)
        elif scenario_name == "dispersion_vmas":
            # Update only max_speed for dispersion_vmas
            agent_capabilities = {
                "max_speed": agent_max_speeds,
            }
            if hasattr(env, "scenario") and hasattr(
                env.scenario, "update_agent_capabilities"
            ):
                env.scenario.update_agent_capabilities(agent_capabilities)
        elif scenario_name != "football":
            agent_capabilities = {
                "speed": agent_speeds,
                "lidar_range": agent_lidar_ranges,
            }
            env.scenario.update_agent_capabilities(agent_capabilities)

            # Debug: Verify capabilities were applied to environment
            if episode == 1:
                print(
                    f"\n[DEBUG] Updated environment capabilities (episode {episode}):"
                )
                for i in range(min(4, len(agent_speeds))):
                    actual_speed = getattr(env.agents[i], "_max_speed", "N/A")
                    actual_lidar = getattr(env.agents[i], "_obs_range", "N/A")
                    print(
                        f"  Agent {i}: requested speed={agent_speeds[i]:.3f}, lidar={agent_lidar_ranges[i]:.3f}"
                    )
                    print(
                        f"            actual    speed={actual_speed}, lidar={actual_lidar}"
                    )

        # Reset environment (PyTorch for VMAS, JAX for SMAX)
        if scenario_name == "smax":
            # SMAX is JAX-based, use JAX RNG - vectorize across num_envs
            episode_key = jax.random.PRNGKey(seed + episode)
            # Create separate keys for each environment
            env_keys = jax.random.split(episode_key, num_envs)

            # Vectorize the reset function across environments
            def reset_single_env(key):
                return env.reset(key)

            # Use vmap to reset all environments in parallel
            vmapped_reset = jax.vmap(reset_single_env)
            obs_dicts, env_states = vmapped_reset(env_keys)

            # Convert dict of arrays to list of observations per agent
            # obs_dicts has shape: {agent_name: (num_envs, obs_dim)}
            obs = [obs_dicts[agent] for agent in env.agents]
            # Convert JAX arrays to PyTorch tensors
            obs = [torch.from_numpy(np.array(o)).float().to(torch_device) for o in obs]
            # obs is now list of (num_envs, obs_dim) tensors

            # Store vectorized environment state
            env_state = env_states
        else:
            # PyTorch/VMAS environment
            obs = (
                env.reset()
            )  # Returns observations for env_num_agents (which is max_agents for mixed training)

        # No need to pad - environment always has max_agents when mixed training is enabled
        # We'll just use current_num_agents and ignore the rest

        # Create capability-based context vectors for each agent
        # For dispersion: Each agent gets a capability vector [speed, lidar_range]
        # For football: Each agent gets a capability vector [speed, size, shoot_power]
        # For simple_tag/grassland: Full observation vector
        # For SMAX: Unit capability features (7 dimensions: health, attack, attack_range, velocity, sight_range, radius, attack_speed)
        # Shape: (num_envs, env_num_agents, context_dim) - matches environment agent count

        if use_capability_context:
            if scenario_name == "smax":
                # For SMAX, use unit capability features as context
                from smax_capabilities import get_unit_capabilities

                # Get unit types from environment state (HeuristicEnemySMAX wraps the state)
                # After vectorization, env_state has shape with leading num_envs dimension
                smax_state = (
                    env_state.state if hasattr(env_state, "state") else env_state
                )
                # unit_types now has shape (num_envs, total_units) - take first env for static context
                unit_types = smax_state.unit_types[
                    0, : env.num_allies
                ]  # JAX array of unit type indices (allies only) from env 0
                # Extract normalized capability features (health, attack, range, speed, etc.)
                capability_features = get_unit_capabilities(
                    env, unit_types
                )  # (num_allies, 7)
                # Convert to PyTorch and add batch dimension
                capability_vectors = (
                    torch.from_numpy(np.array(capability_features))
                    .float()
                    .to(torch_device)
                )
                # Expand to batch size: (num_allies, 7) -> (num_envs, num_allies, 7)
                static_context = capability_vectors.unsqueeze(0).expand(
                    num_envs, -1, -1
                )
            elif scenario_name == "football":
                # Get capability vectors: (env_num_agents, 3) -- same across envs
                capability_batch = (
                    torch.from_numpy(env.get_capability_vectors(normalize=True))
                    .to(torch_device)
                    .unsqueeze(0)
                    .expand(num_envs, -1, -1)
                )  # (num_envs, env_num_agents, 3)
                # Get initial positions after reset: (num_envs, env_num_agents, 2)
                initial_positions = env.get_initial_positions(normalize=True).to(
                    torch_device
                )
                # Full context: (num_envs, env_num_agents, 5)
                static_context = torch.cat(
                    [capability_batch, initial_positions], dim=-1
                )
            elif scenario_name == "simple_tag" or scenario_name == "grassland":
                # Use full initial observations as context
                # obs: [agent0_obs, agent1_obs, ...] each (num_envs, obs_dim)
                # Stack all agent observations from first environment
                capability_list = []
                for i in range(env_num_agents):
                    agent_obs = obs[i][0, :]  # (obs_dim,) - full observation
                    capability_list.append(agent_obs)
                capability_vectors = torch.stack(
                    capability_list, dim=0
                )  # (env_num_agents, obs_dim)
            elif scenario_name == "reverse_transport":
                # Use [max_speed, force_multiplier] as 2-D capability vector.
                # getattr returns None (not the default) when the attribute exists but
                # was set to None, so explicitly fall back to the config values.
                capability_list = []
                for i in range(env_num_agents):
                    raw_speed = getattr(env.agents[i], "_max_speed", None)
                    speed = (
                        float(raw_speed)
                        if raw_speed is not None
                        else float(
                            agent_speeds[i] if agent_speeds[i] is not None else 1.0
                        )
                    )
                    raw_force = getattr(env.agents[i], "_force_multiplier", None)
                    force = (
                        float(raw_force)
                        if raw_force is not None
                        else float(
                            agent_force_multipliers[i]
                            if agent_force_multipliers[i] is not None
                            else 1.0
                        )
                    )
                    capability_list.append(
                        torch.tensor(
                            [speed, force], device=torch_device, dtype=torch.float32
                        )
                    )
                capability_vectors = torch.stack(
                    capability_list, dim=0
                )  # (env_num_agents, 2)
            elif scenario_name == "dispersion_vmas":
                # Dispersion_vmas: use [max_speed] as 1D capability
                capability_list = []
                for i in range(env_num_agents):
                    # Get agent max_speed
                    try:
                        # Try to get from agent object attributes (with underscores for VMAS)
                        max_speed_attr = getattr(
                            env.agents[i], "_max_speed", agent_max_speeds[i]
                        )

                        # Handle None values by falling back to defaults
                        max_speed = (
                            float(max_speed_attr)
                            if max_speed_attr is not None
                            else float(
                                agent_max_speeds[i]
                                if agent_max_speeds[i] is not None
                                else 1.0
                            )
                        )
                    except (AttributeError, IndexError, TypeError):
                        # Fall back to our stored values (with None handling)
                        max_speed = (
                            float(agent_max_speeds[i])
                            if agent_max_speeds[i] is not None
                            else 1.0
                        )

                    capability_list.append(
                        torch.tensor(
                            [max_speed],
                            device=torch_device,
                            dtype=torch.float32,
                        )
                    )
                capability_vectors = torch.stack(
                    capability_list, dim=0
                )  # (env_num_agents, 1)
            else:
                # Dispersion and other scenarios: use [speed, lidar_range] as capabilities
                # Read actual agent capabilities from the environment
                capability_list = []
                for i in range(env_num_agents):
                    # Get agent speed and lidar_range
                    try:
                        # Try to get from agent object attributes (with underscores for VMAS)
                        speed_attr = getattr(
                            env.agents[i], "_max_speed", agent_speeds[i]
                        )
                        lidar_attr = getattr(
                            env.agents[i], "_obs_range", agent_lidar_ranges[i]
                        )

                        # Handle None values by falling back to defaults
                        speed = (
                            float(speed_attr)
                            if speed_attr is not None
                            else float(
                                agent_speeds[i] if agent_speeds[i] is not None else 1.0
                            )
                        )
                        lidar_range = (
                            float(lidar_attr)
                            if lidar_attr is not None
                            else float(
                                agent_lidar_ranges[i]
                                if agent_lidar_ranges[i] is not None
                                else 0.5
                            )
                        )
                    except (AttributeError, IndexError, TypeError):
                        # Fall back to our stored values (with None handling)
                        speed = (
                            float(agent_speeds[i])
                            if agent_speeds[i] is not None
                            else 1.0
                        )
                        lidar_range = (
                            float(agent_lidar_ranges[i])
                            if agent_lidar_ranges[i] is not None
                            else 0.5
                        )

                    capability_list.append(
                        torch.tensor(
                            [speed, lidar_range],
                            device=torch_device,
                            dtype=torch.float32,
                        )
                    )
                capability_vectors = torch.stack(
                    capability_list, dim=0
                )  # (env_num_agents, 2)

            # Debug: Print capability values on first episode only
            if episode == 1 and use_capability_context:
                print(f"\n[DEBUG] Capability context values (episode {episode}):")
                if scenario_name == "football":
                    print(
                        f"  Shape: {static_context.shape} - [speed, size, shoot_power, pos_x, pos_y]"
                    )
                    for i in range(min(4, env_num_agents)):
                        print(f"    Agent {i}: {static_context[0, i].cpu().numpy()}")
                elif scenario_name == "reverse_transport":
                    print(
                        f"  Shape: {capability_vectors.shape} - [speed, force_multiplier]"
                    )
                elif scenario_name == "dispersion_vmas":
                    print(f"  Shape: {capability_vectors.shape} - [max_speed]")
                elif scenario_name == "pressure_plate":
                    print(
                        f"  Shape: {capability_vectors.shape} - (no capability context)"
                    )
                elif scenario_name in ["smax", "simple_tag", "grassland"]:
                    print(
                        f"  Shape: {capability_vectors.shape} - scenario-specific features"
                    )
                else:
                    print(f"  Shape: {capability_vectors.shape} - [speed, lidar_range]")
                if scenario_name != "football":
                    for i in range(min(4, env_num_agents)):  # Show first 4 agents
                        print(f"    Agent {i}: {capability_vectors[i].cpu().numpy()}")

            # Expand to include batch dimension (football already built static_context above)
            if scenario_name != "football":
                static_context = capability_vectors.unsqueeze(0).expand(
                    num_envs, -1, -1
                )  # (num_envs, env_num_agents, context_dim)
        else:
            # Empty context if disabled
            static_context = torch.zeros(
                num_envs,
                env_num_agents,
                0,
                device=torch_device,
                dtype=torch.float32,
            )

        # For dispersion_vmas: Use positional encoding (if enabled) or one-hot encoding for role differentiation
        # When use_capability_context=True, prepend max_speed to the positional/one-hot encoding
        if scenario_name == "dispersion_vmas":
            if use_positional_context:
                # Generate positional encodings: scalable to any number of agents
                # Shape: (env_num_agents, positional_encoding_dim)
                agent_pos_encoding = generate_positional_encoding(
                    env_num_agents, positional_encoding_dim, device=torch_device
                )
                # Expand to batch: (num_envs, env_num_agents, positional_encoding_dim)
                agent_pos_encoding_expanded = agent_pos_encoding.unsqueeze(0).expand(
                    num_envs, -1, -1
                )

                # Combine with capability context if enabled
                if use_capability_context:
                    # static_context already contains [max_speed] for each agent
                    # Concatenate: [max_speed] + positional_encoding
                    static_context = torch.cat(
                        [static_context, agent_pos_encoding_expanded], dim=-1
                    )  # (num_envs, env_num_agents, 1 + positional_encoding_dim)
                else:
                    static_context = agent_pos_encoding_expanded  # (num_envs, env_num_agents, positional_encoding_dim)
            elif use_onehot_context:
                # Create one-hot encodings for agent indices: [0, 1, 2, ..., env_num_agents-1]
                # Use max_agents as the dimension to handle variable team sizes
                agent_one_hot = torch.nn.functional.one_hot(
                    torch.arange(env_num_agents, device=torch_device),
                    num_classes=max_agents,
                ).float()  # (env_num_agents, max_agents)

                # Expand to batch: (num_envs, env_num_agents, max_agents)
                agent_one_hot_expanded = agent_one_hot.unsqueeze(0).expand(
                    num_envs, -1, -1
                )

                # Combine with capability context if enabled
                if use_capability_context:
                    # static_context already contains [max_speed] for each agent
                    # Concatenate: [max_speed] + one_hot_encoding
                    static_context = torch.cat(
                        [static_context, agent_one_hot_expanded], dim=-1
                    )  # (num_envs, env_num_agents, 1 + max_agents)
                else:
                    # REPLACE static_context entirely with just one-hot IDs (no capability features)
                    static_context = (
                        agent_one_hot_expanded  # (num_envs, env_num_agents, max_agents)
                    )
            else:
                # No positional or one-hot encoding: use only capability context or empty
                if not use_capability_context:
                    # No positional or one-hot encoding and no capability context: use empty context tensor (context_dim=0)
                    static_context = torch.zeros(
                        num_envs,
                        env_num_agents,
                        0,
                        device=torch_device,
                        dtype=torch.float32,
                    )

        # Extract initial lidar readings if enabled (skip for football, SMAX, and dispersion_vmas)
        # Observation structure for dispersion (lidar version): pos(2) + vel(2) + food(4) + lidar_agents(12) + lidar_food(12) = 32
        # dispersion_vmas uses global observability (no lidar), observation: pos(2) + vel(2) + [food_i: rel_pos(2) + eaten(1)]*N_foods
        # Lidar readings are the last 24 dimensions (12 for agents + 12 for food)
        if use_lidar_context and scenario_name not in [
            "football",
            "smax",
            "dispersion_vmas",
        ]:
            # obs is a list of tensors: [agent0_obs, agent1_obs, ...] - has env_num_agents elements
            # Each agent_obs has shape: (num_envs, obs_dim)
            # Extract last lidar_dim dimensions from each agent's observation
            lidar_list = [agent_obs[:, -lidar_dim:] for agent_obs in obs]
            # Stack to (env_num_agents, num_envs, lidar_dim) then transpose to (num_envs, env_num_agents, lidar_dim)
            initial_lidar = torch.stack(lidar_list, dim=0).transpose(0, 1)
        else:
            initial_lidar = None

        # Extract relative food positions for dispersion_vmas (global observability)
        # This provides each agent with their matching food's relative position
        # Only extract if food_position_dim > 0 (i.e., food is used as hypernetwork context)
        if food_position_dim > 0:
            initial_food_positions = extract_food_positions(
                obs, env_num_agents, scenario_name
            )
            # Validate shape - must be (num_envs, num_agents, food_position_dim) or None
            if initial_food_positions is not None:
                expected_shape = (num_envs, env_num_agents, food_position_dim)
                if initial_food_positions.shape != expected_shape:
                    print(
                        f"Warning: Food positions have unexpected shape {initial_food_positions.shape}, expected {expected_shape}. Disabling food position context."
                    )
                    initial_food_positions = None
        else:
            initial_food_positions = None

        # Extract agent positions for dispersion_vmas (global observability)
        # This provides each agent's absolute position in the environment
        initial_agent_positions = extract_agent_positions(
            obs, env_num_agents, scenario_name
        )
        # Validate shape - must be (num_envs, num_agents, 2) or None
        if initial_agent_positions is not None:
            expected_shape = (num_envs, env_num_agents, 2)
            if initial_agent_positions.shape != expected_shape:
                print(
                    f"Warning: Agent positions have unexpected shape {initial_agent_positions.shape}, expected {expected_shape}. Disabling agent position context."
                )
                initial_agent_positions = None

        # Extract environment context for reverse_transport (package properties)
        # This allows the hypernetwork to adapt policies when package properties change during an episode
        if scenario_name == "reverse_transport" and env_context_dim > 0:
            # Get current package properties from environment
            package_props = env.scenario.get_package_properties()
            # Create environment context: [mass, width, length]
            env_context_vec = torch.tensor(
                [
                    package_props["mass"],
                    package_props["width"],
                    package_props["length"],
                ],
                device=torch_device,
                dtype=torch.float32,
            )
            # Broadcast to all envs and agents: (num_envs, num_agents, env_context_dim)
            initial_env_context = (
                env_context_vec.unsqueeze(0)
                .unsqueeze(0)
                .expand(num_envs, env_num_agents, env_context_dim)
            )
            # Track current environment properties for change detection
            current_package_props = package_props.copy()

            if episode == 0:
                print(f"\n[DEBUG] Environment context initialized:")
                print(
                    f"  Package properties: mass={package_props['mass']:.2f}, width={package_props['width']:.2f}, length={package_props['length']:.2f}"
                )
                print(f"  initial_env_context shape: {initial_env_context.shape}")
        elif scenario_name == "pressure_plate" and env_context_dim > 0:
            # Get current environment state from pressure_plate scenario.
            # Build per-agent environment context with RELATIVE positions so each
            # agent receives plate/goal positions relative to itself (consistent
            # with the policy's observation convention).

            # Get configuration flags
            env_context_plate_positions = config["model"].get(
                "env_context_plate_positions", True
            )
            env_context_door_state = config["model"].get("env_context_door_state", True)
            env_context_goal_position = config["model"].get(
                "env_context_goal_position", True
            )
            use_agent_id_context = config["model"].get("use_agent_id_context", False)

            # Agent positions: (num_envs, env_num_agents, 2) for computing relative context
            _ground_robots_ctx = sorted(
                [a for a in env.agents if "ground_robot" in a.name],
                key=lambda a: a.name,
            )
            _agent_pos_ctx = torch.stack(
                [a.state.pos[:, :2] for a in _ground_robots_ctx], dim=1
            )  # (num_envs, env_num_agents, 2)

            # Build per-agent context parts: each (num_envs, env_num_agents, d)
            env_context_parts = []

            if env_context_plate_positions:
                left_plate_pos = env.scenario.plate_left.state.pos[
                    :, :2
                ]  # (num_envs, 2)
                right_plate_pos = env.scenario.plate_right.state.pos[
                    :, :2
                ]  # (num_envs, 2)
                # Relative to each agent: (num_envs, env_num_agents, 2)
                left_rel = left_plate_pos.unsqueeze(1) - _agent_pos_ctx
                right_rel = right_plate_pos.unsqueeze(1) - _agent_pos_ctx
                env_context_parts.extend([left_rel, right_rel])

            if env_context_door_state:
                door_open = env.scenario.door_open.float()  # (num_envs,)
                # Expand to (num_envs, env_num_agents, 1)
                door_open_exp = door_open[:, None, None].expand(
                    num_envs, env_num_agents, 1
                )
                env_context_parts.append(door_open_exp)

            if env_context_goal_position:
                goal_pos = env.scenario.goal.state.pos[:, :2]  # (num_envs, 2)
                # Relative to each agent: (num_envs, env_num_agents, 2)
                goal_rel = goal_pos.unsqueeze(1) - _agent_pos_ctx
                env_context_parts.append(goal_rel)

            if use_agent_id_context:
                # Add one-hot agent IDs for role differentiation
                # Shape: (num_envs, env_num_agents, max_agents)
                agent_ids_onehot = torch.zeros(
                    num_envs, env_num_agents, max_agents, device=torch_device
                )
                for i in range(env_num_agents):
                    agent_ids_onehot[:, i, i] = 1.0
                env_context_parts.append(agent_ids_onehot)

            # (num_envs, env_num_agents, env_context_dim)
            initial_env_context = torch.cat(env_context_parts, dim=-1)

            if episode == 0:
                print(
                    f"\n[DEBUG] Environment context initialized for pressure_plate (per-agent relative):"
                )
                if env_context_plate_positions:
                    print(
                        f"  Left plate pos  (env0): {left_plate_pos[0].cpu().numpy()}"
                    )
                    print(
                        f"  Right plate pos (env0): {right_plate_pos[0].cpu().numpy()}"
                    )
                    print(f"  Agent 0 pos (env0): {_agent_pos_ctx[0, 0].cpu().numpy()}")
                    print(
                        f"  Left plate rel (env0, agent0): {initial_env_context[0, 0, :2].cpu().numpy()}"
                    )
                if env_context_door_state:
                    print(f"  Door open (env0): {door_open[0].item()}")
                if env_context_goal_position:
                    print(f"  Goal pos (env0): {goal_pos[0].cpu().numpy()}")
                print(f"  initial_env_context shape: {initial_env_context.shape}")
                print(
                    f"  initial_env_context[0, 0]: {initial_env_context[0, 0].cpu().numpy()}"
                )
        else:
            initial_env_context = None
            current_package_props = None

        # Track pressure plate states for hypernetwork requerying
        if scenario_name == "pressure_plate" and use_hypernetwork:
            # Track door state and plate activations for requerying
            prev_door_open = torch.zeros(
                num_envs, dtype=torch.bool, device=torch_device
            )
            door_opened_count = torch.zeros(
                num_envs, dtype=torch.long, device=torch_device
            )
            plate_activation_count = torch.zeros(
                num_envs, dtype=torch.long, device=torch_device
            )

            if episode == 0:
                print(
                    f"\n[DEBUG] Pressure plate state tracking initialized for hypernetwork requerying"
                )
                print(f"  Will requery when: door opens, or second plate activated")
        else:
            prev_door_open = None
            door_opened_count = None
            plate_activation_count = None

        # Print context info for first episode
        if episode == 0:
            cap_mode = "FIXED" if use_fixed_capabilities else "randomized per episode"
            if use_capability_context and static_context.shape[-1] >= 2:
                print(f"\nCapability vectors for all agents ({cap_mode}):")
                # For SMAX, only iterate over allies (static_context only has allies)
                num_agents_to_print = (
                    static_context.shape[1] if scenario_name == "smax" else num_agents
                )
                for agent_idx in range(num_agents_to_print):
                    if scenario_name == "smax" and static_context.shape[-1] == 6:
                        # Print unit type
                        unit_type_idx = torch.argmax(
                            static_context[0, agent_idx]
                        ).item()
                        unit_types = [
                            "marine",
                            "marauder",
                            "stalker",
                            "zealot",
                            "zergling",
                            "hydralisk",
                        ]
                        print(
                            f"  Ally {agent_idx}: unit_type={unit_types[unit_type_idx]}"
                        )
                    elif scenario_name == "football" and static_context.shape[-1] >= 3:
                        msg = (
                            f"  Agent {agent_idx}: speed={static_context[0, agent_idx, 0].item():.3f}, "
                            f"size={static_context[0, agent_idx, 1].item():.3f}, "
                            f"shoot_power={static_context[0, agent_idx, 2].item():.3f}"
                        )
                        if static_context.shape[-1] >= 5:
                            msg += (
                                f", pos_x={static_context[0, agent_idx, 3].item():.3f}, "
                                f"pos_y={static_context[0, agent_idx, 4].item():.3f}"
                            )
                        print(msg)
                    elif (
                        scenario_name == "reverse_transport"
                        and static_context.shape[-1] == 2
                    ):
                        print(
                            f"  Agent {agent_idx}: speed={static_context[0, agent_idx, 0].item():.3f}, "
                            f"force_multiplier={static_context[0, agent_idx, 1].item():.3f}"
                        )
                    else:
                        print(
                            f"  Agent {agent_idx}: speed={static_context[0, agent_idx, 0].item():.3f}, "
                            f"lidar_range={static_context[0, agent_idx, 1].item():.3f}"
                        )
                if scenario_name == "smax":
                    print(f"  (Enemies use heuristic policy - not trained)")
            else:
                print(f"\nCapability vectors disabled (using empty context)")

            print(f"Context shape: {static_context.shape}")
            if use_lidar_context and scenario_name not in ["football", "smax"]:
                print(f"Using initial lidar readings as additional context")
                if initial_lidar is not None:
                    print(f"Lidar context shape: {initial_lidar.shape}")
                else:
                    print(f"Lidar context shape: None (lidar_dim={lidar_dim})")

        # Create task embedding (skip for dispersion_vmas - provides no useful information)
        if scenario_name == "dispersion_vmas":
            static_task = torch.zeros(
                num_envs, env_num_agents, 0, device=torch_device
            )  # Empty task
        else:
            static_task = torch.ones(
                num_envs, env_num_agents, task_embed_dim, device=torch_device
            )

        # Create target SND tensor with MIXED values per environment for better training diversity
        # Shape: (num_envs, env_num_agents, target_snd_dim)
        if target_snd_dim > 0:
            # Sample different target_snd values for each environment
            # Check if we have a sampler available for per-env sampling
            sampler_check = None
            if "__main__" in sys.modules:
                main_module = sys.modules["__main__"]
                if hasattr(main_module, "get_sampler"):
                    try:
                        sampler_check = main_module.get_sampler()
                    except:
                        pass
            if sampler_check is None and "train_random_diversity" in sys.modules:
                try:
                    import train_random_diversity

                    sampler_check = train_random_diversity.get_sampler()
                except:
                    pass

            # Sample per-environment target_snd values for diversity
            if sampler_check is not None:
                # Sample different target for each environment
                target_snd_per_env = np.array(
                    [sampler_check.sample(np_rng) for _ in range(num_envs)]
                )
                # Store mean for logging purposes
                target_snd_mean = float(target_snd_per_env.mean())
                use_per_env_scaling = config["training"].get(
                    "per_env_diversity_scaling", False
                )
                if episode % 10 == 0 or episode < 5:
                    print(
                        f"  Mixed target_snd per env: min={target_snd_per_env.min():.4f}, max={target_snd_per_env.max():.4f}, mean={target_snd_mean:.4f}"
                    )
                    if use_per_env_scaling:
                        print(
                            f"  Using per-env diversity scaling (each env scaled by its own target)"
                        )
                    else:
                        print(
                            f"  Using uniform diversity scaling (all envs scaled by mean target={target_snd_mean:.4f})"
                        )
            else:
                # Fallback: use the single target_snd value for all envs
                target_snd_per_env = np.full(num_envs, target_snd)
                target_snd_mean = target_snd

            # Create tensor with per-env target values
            # Shape: (num_envs, 1, target_snd_dim) -> broadcast to (num_envs, env_num_agents, target_snd_dim)
            target_snd_per_env_tensor = (
                torch.from_numpy(target_snd_per_env).float().to(torch_device)
            )
            static_target_snd = target_snd_per_env_tensor[:, None, None].expand(
                num_envs, env_num_agents, target_snd_dim
            )
        else:
            static_target_snd = torch.zeros(
                (num_envs, env_num_agents, 0),
                device=torch_device,
                dtype=torch.float32,
            )
            target_snd_per_env = np.full(num_envs, target_snd)
            target_snd_mean = target_snd

        # Optional: override the episode target with a rollout-switch schedule seed value.
        # This keeps the very first query aligned with the scheduled target progression.
        if enable_rollout_target_snd_switch:
            if train_target_snd_mode == "random":
                scheduled_target_snd = float(np_rng.choice(train_target_snd_list))
            else:
                global_rollout_step_start = episode * rollout_steps
                scheduled_index = (
                    global_rollout_step_start // train_target_snd_interval
                ) % len(train_target_snd_list)
                scheduled_target_snd = float(train_target_snd_list[scheduled_index])

            target_snd_per_env = np.full(num_envs, scheduled_target_snd)
            target_snd_mean = scheduled_target_snd

            if target_snd_dim > 0:
                static_target_snd = torch.full(
                    (num_envs, env_num_agents, target_snd_dim),
                    scheduled_target_snd,
                    device=torch_device,
                    dtype=torch.float32,
                )

            if episode % log_interval == 0 or episode < 5:
                print(
                    f"  Rollout target_snd switching enabled: initial target_snd={scheduled_target_snd:.4f}, interval={train_target_snd_interval}, mode={train_target_snd_mode}"
                )

        # ====================================================================
        # Prepare inputs for cross-agent attention (keep per-environment structure)
        # ====================================================================
        batch_size = num_envs * env_num_agents  # Use env_num_agents for batch size

        # Keep agent dimension: (num_envs, env_num_agents, feature_dim)
        # No flattening yet - hypernetwork needs to see agent structure

        # ====================================================================
        # WRAPPER: PyTorch -> JAX (for context and task)
        # ====================================================================
        # Convert to numpy first, handling device properly
        context_np = (
            static_context.detach().cpu().numpy()
            if static_context.requires_grad
            else static_context.cpu().numpy()
        )
        task_np = (
            static_task.detach().cpu().numpy()
            if static_task.requires_grad
            else static_task.cpu().numpy()
        )

        # Shape: (num_envs, num_agents, context_dim) — None when dim is 0
        # For dispersion_vmas: context is just one-hot IDs (never None), task is empty (None)
        jax_context = jnp.asarray(context_np) if context_np.shape[-1] > 0 else None
        jax_task = jnp.asarray(task_np) if task_np.shape[-1] > 0 else None

        # Convert target_snd to JAX (None for dispersion_vmas)
        target_snd_np = (
            static_target_snd.detach().cpu().numpy()
            if static_target_snd.requires_grad
            else static_target_snd.cpu().numpy()
        )
        jax_target_snd = (
            jnp.asarray(target_snd_np) if target_snd_np.shape[-1] > 0 else None
        )

        if use_lidar_context and initial_lidar is not None:
            lidar_np = (
                initial_lidar.detach().cpu().numpy()
                if initial_lidar.requires_grad
                else initial_lidar.cpu().numpy()
            )
            # 1. Handle Infinity (Simulators often return inf for "no hit")
            lidar_np = np.nan_to_num(lidar_np, posinf=1.0, neginf=0.0)
            # 2. Hard Clip (Ensure values stay in -1 to 1 range for the Transformer)
            lidar_np = np.clip(lidar_np, -1.0, 1.0)
            jax_lidar = jnp.asarray(lidar_np)  # (num_envs, num_agents, lidar_dim)
        else:
            jax_lidar = None

        # Convert food positions to JAX if available (for dispersion_vmas)
        if initial_food_positions is not None:
            food_pos_np = (
                initial_food_positions.detach().cpu().numpy()
                if initial_food_positions.requires_grad
                else initial_food_positions.cpu().numpy()
            )
            jax_food_positions = jnp.asarray(food_pos_np)  # (num_envs, num_agents, 2)
        else:
            jax_food_positions = None

        # Convert agent positions to JAX if available (for dispersion_vmas)
        if initial_agent_positions is not None:
            agent_pos_np = (
                initial_agent_positions.detach().cpu().numpy()
                if initial_agent_positions.requires_grad
                else initial_agent_positions.cpu().numpy()
            )
            jax_agent_positions = jnp.asarray(agent_pos_np)  # (num_envs, num_agents, 2)
        else:
            jax_agent_positions = None

        # Convert environment context to JAX if available (for reverse_transport)
        if initial_env_context is not None:
            env_context_np = (
                initial_env_context.detach().cpu().numpy()
                if initial_env_context.requires_grad
                else initial_env_context.cpu().numpy()
            )
            jax_env_context = jnp.asarray(
                env_context_np
            )  # (num_envs, num_agents, env_context_dim)
            if episode == 0:
                print(f"\n[DEBUG] Converted initial_env_context to JAX:")
                print(f"  jax_env_context shape: {jax_env_context.shape}")
                print(
                    f"  jax_env_context sample (env0, agent0): {jax_env_context[0, 0]}"
                )
        else:
            jax_env_context = None
            if episode == 0 and env_context_dim > 0:
                print(
                    f"\n[DEBUG WARNING] initial_env_context is None but env_context_dim={env_context_dim}"
                )

        # Create attention mask for dynamic agent counts
        # CRITICAL: Use env_num_agents (not max_agents) to match tensor dimensions
        if current_num_agents < env_num_agents:
            jax_mask = create_attention_mask(current_num_agents, env_num_agents)
        else:
            jax_mask = None  # No masking when all agents are active

        # Move to JAX device if using CUDA
        if use_cuda:
            if jax_context is not None:
                jax_context = jax.device_put(jax_context, jax_device)
            if jax_task is not None:
                jax_task = jax.device_put(jax_task, jax_device)
            if jax_target_snd is not None:
                jax_target_snd = jax.device_put(jax_target_snd, jax_device)
            if jax_mask is not None:
                jax_mask = jax.device_put(jax_mask, jax_device)
            if use_lidar_context:
                jax_lidar = jax.device_put(jax_lidar, jax_device)
            if jax_env_context is not None:
                jax_env_context = jax.device_put(jax_env_context, jax_device)

        # Verify shapes are consistent (for debugging)
        if episode == 0:
            print(f"\nShape verification (episode 0):")
            print(
                f"  env_num_agents={env_num_agents}, current_num_agents={current_num_agents}"
            )
            print(f"  jax_task: {jax_task.shape if jax_task is not None else None}")
            print(
                f"  jax_context: {jax_context.shape if jax_context is not None else None}"
            )
            if jax_target_snd is not None:
                print(f"  jax_target_snd shape: {jax_target_snd.shape}")
            if jax_lidar is not None:
                print(f"  jax_lidar shape: {jax_lidar.shape}")
            if jax_env_context is not None:
                print(f"  jax_env_context shape: {jax_env_context.shape}")
                print(
                    f"  jax_env_context sample (env0, agent0): {jax_env_context[0, 0]}"
                )
            else:
                print(
                    f"  jax_env_context: None (WARNING: env_context_dim={env_context_dim})"
                )
            if jax_mask is not None:
                print(f"  jax_mask shape: {jax_mask.shape}")
            print(
                f"  batch_size: {batch_size} (should be num_envs * env_num_agents = {num_envs} * {env_num_agents})"
            )

        # ====================================================================
        # Run Hypernetwork (JAX) - Generate adapters for all agents (if enabled)
        # ====================================================================
        if use_cash:
            # CASH generates decoder parameters per-step from observations + capabilities.
            # No static adapters are created at episode start.
            adapters_dict = None
            diversity_stats = None
            diversity_scaling = 1.0
        elif use_hypernetwork:
            # Diversity Monitoring: Calculate legacy SND ONLY if NOT using adapter_snd
            # When use_adapter_snd=True, skip the legacy calculation to save compute
            # (adapter SND will be calculated later and will override these values anyway)
            calculate_snd_metrics = (
                use_hypernetwork
                and not use_adapter_snd  # Skip legacy SND if using adapter_snd
            )
            if calculate_snd_metrics:
                # Determine sample size and get observations
                if use_snd_obs_buffer and snd_obs_buffer is not None:
                    # Use observations from the pre-collected buffer
                    # Sample from buffer if it's larger than needed
                    actual_buffer_size = len(snd_obs_buffer)
                    if actual_buffer_size >= snd_obs_buffer_sample_size:
                        # Random sample from buffer
                        sample_indices = np_rng.choice(
                            actual_buffer_size,
                            size=snd_obs_buffer_sample_size,
                            replace=False,
                        )
                        common_obs_jax = snd_obs_buffer[
                            sample_indices
                        ]  # (sample_size, obs_dim)
                        sample_size = snd_obs_buffer_sample_size
                    else:
                        # Use all available observations if buffer is smaller
                        common_obs_jax = snd_obs_buffer
                        sample_size = actual_buffer_size
                else:
                    # Original behavior: use current observations from rollout
                    sample_size = num_envs  # Use all environments for accurate SND

                    # CRITICAL: For proper SND calculation, all agents must be evaluated on the SAME observations
                    # to isolate policy diversity from observation diversity
                    # Use agent 0's observations as the common observation set
                    # obs is a list of [agent0_obs, agent1_obs, ...] where each has shape (num_envs, obs_dim)
                    common_obs_torch = obs[0][
                        :sample_size
                    ]  # (sample_size, obs_dim) - agent 0's observations

                    # Convert to JAX
                    common_obs_jax = jnp.asarray(
                        common_obs_torch.cpu().numpy()
                    )  # (sample_size, obs_dim)

                # When using observation buffer, we need to generate adapters for sample_size, not num_envs
                # Create task/context/lidar tensors that match sample_size
                # IMPORTANT: Compute adapter_sample_size BEFORE tiling observations
                if (
                    use_snd_obs_buffer
                    and snd_obs_buffer is not None
                    and sample_size != num_envs
                ):
                    # Create task/context/lidar for the sample_size
                    # Note: if sample_size > num_envs, slicing will only give us num_envs entries
                    jax_task_sample = jax_task[
                        :sample_size
                    ]  # (min(sample_size, num_envs), env_num_agents, task_dim)
                    jax_context_sample = jax_context[
                        :sample_size
                    ]  # (min(sample_size, num_envs), env_num_agents, context_dim)
                    jax_lidar_sample = (
                        jax_lidar[:sample_size] if jax_lidar is not None else None
                    )
                    jax_food_positions_sample = (
                        jax_food_positions[:sample_size]
                        if jax_food_positions is not None
                        else None
                    )
                    jax_agent_positions_sample = (
                        jax_agent_positions[:sample_size]
                        if jax_agent_positions is not None
                        else None
                    )
                    jax_target_snd_sample = (
                        jax_target_snd[:sample_size]
                        if jax_target_snd is not None
                        else None
                    )
                    jax_env_context_sample = (
                        jax_env_context[:sample_size]
                        if jax_env_context is not None
                        else None
                    )
                    jax_mask_sample = (
                        jax_mask if jax_mask is None else jax_mask
                    )  # Mask doesn't depend on num_envs
                    # Compute actual env count after slicing (handles sample_size > num_envs case)
                    adapter_sample_size = jax_task_sample.shape[0]
                else:
                    # Use the full tensors (original behavior)
                    jax_task_sample = jax_task
                    jax_context_sample = jax_context
                    jax_lidar_sample = jax_lidar
                    jax_food_positions_sample = jax_food_positions
                    jax_agent_positions_sample = jax_agent_positions
                    jax_target_snd_sample = jax_target_snd
                    jax_env_context_sample = jax_env_context
                    jax_mask_sample = jax_mask
                    adapter_sample_size = num_envs

                # Build observation batch for SND evaluation.
                # For dispersion_vmas (not using buffer): use each agent's OWN observations
                # so that adapter a is evaluated on agent a's actual observation — matching
                # the snapshot approach used in snd.py's dispersion_vmas path.
                # For all other cases: tile agent-0 observations across all agent slots.
                if scenario_name == "dispersion_vmas" and not (
                    use_snd_obs_buffer and snd_obs_buffer is not None
                ):
                    # Build agent-major obs: [ag0_envs..., ag1_envs..., ...]
                    # obs is a list [agent0_obs, agent1_obs, ...] of PyTorch tensors (num_envs, obs_dim)
                    obs_per_agent_jax = []
                    for _a in range(env_num_agents):
                        _obs_a_np = obs[_a][:adapter_sample_size].detach().cpu().numpy()
                        obs_per_agent_jax.append(jnp.asarray(_obs_a_np))
                    sample_obs_flat = jnp.concatenate(
                        obs_per_agent_jax, axis=0
                    )  # (env_num_agents * adapter_sample_size, obs_dim)
                else:
                    # Shape: (adapter_sample_size * env_num_agents, obs_dim)
                    sample_obs_flat = jnp.tile(
                        common_obs_jax[:adapter_sample_size], (env_num_agents, 1)
                    )  # Repeat for each agent

                # Compute UNSCALED adapters to measure intrinsic diversity
                adapters_dict_unscaled = _get_static_adapters(
                    hn_state.params,
                    jax_task_sample,
                    jax_context_sample,
                    jax_lidar_sample,
                    None,  # food_positions_batch not used here
                    jax_agent_positions_sample,
                    jax_target_snd_sample,
                    jax_env_context_sample,
                    jax_mask_sample,
                    diversity_scaling=1.0,  # No scaling
                )

                sample_adapters_unscaled = {}
                for key, adapter in adapters_dict_unscaled.items():
                    adapter_shape = adapter.shape
                    reshaped = adapter.reshape(
                        adapter_sample_size, env_num_agents, *adapter_shape[1:]
                    )
                    # Transpose to agent-major to match observation ordering
                    # From: [env0_agent0, env0_agent1, ..., env1_agent0, env1_agent1, ...]
                    # To: [agent0_env0, agent0_env1, ..., agent1_env0, agent1_env1, ...]
                    sampled_transposed = reshaped.transpose(
                        1, 0, *range(2, len(reshaped.shape))
                    )
                    sample_adapters_unscaled[key] = sampled_transposed.reshape(
                        adapter_sample_size * env_num_agents, *adapter_shape[1:]
                    )

                # Forward pass through policy to get action distributions
                if use_gru_policy:
                    # GRU policy: need hidden state and proper input format
                    batch_size_snd = adapter_sample_size * env_num_agents
                    init_hidden = shared_policy.initialize_carry(
                        batch_size_snd, shared_policy.gru_hidden_dim
                    )
                    # Format: (obs, dones, avail_actions) - add time dimension
                    sample_obs_seq = sample_obs_flat[None, ...]  # (1, batch, obs_dim)
                    dones_seq = jnp.zeros((1, batch_size_snd), dtype=bool)
                    avail_seq = None  # Continuous actions, no masking
                    policy_x = (sample_obs_seq, dones_seq, avail_seq)

                    _, output = shared_policy.apply(
                        {"params": policy_state.params},
                        init_hidden,
                        policy_x,
                        sample_adapters_unscaled,
                    )
                    # For continuous actions, output is (mean_seq, log_std_seq)
                    mean_seq, _ = output
                    mean_sample_unscaled = mean_seq[
                        0
                    ]  # Remove time dim: (batch, action_dim)
                else:
                    # MLP policy: direct call
                    mean_sample_unscaled, _ = shared_policy.apply(
                        {"params": policy_state.params},
                        sample_obs_flat,
                        sample_adapters_unscaled,
                    )

                action_dim = mean_sample_unscaled.shape[-1]
                # Reshape: output is structured as [agent0 outputs, agent1 outputs, ...]
                # Shape: (env_num_agents * adapter_sample_size, action_dim) -> (env_num_agents, adapter_sample_size, action_dim) -> (adapter_sample_size, env_num_agents, action_dim)
                mean_reshaped_unscaled = mean_sample_unscaled.reshape(
                    env_num_agents, adapter_sample_size, action_dim
                ).transpose(1, 0, 2)

                # Compute pairwise distances using ONLY mean (L2 norm, matching Bettini)
                pair_distances_unscaled = []
                for i in range(env_num_agents):
                    for j in range(i + 1, env_num_agents):
                        dist = jnp.linalg.norm(
                            mean_reshaped_unscaled[:, i, :]
                            - mean_reshaped_unscaled[:, j, :],
                            ord=2,
                            axis=-1,
                        )
                        pair_distances_unscaled.append(dist)
                pairwise_distances_unscaled = jnp.stack(
                    pair_distances_unscaled, axis=-1
                )
                snd_unscaled = float(jnp.mean(pairwise_distances_unscaled))

                # Calculate old SND-based diversity control (will be overridden if adapter_snd is used)
                # NaN protection: check SND before updating moving average
                if not jnp.isnan(snd_unscaled) and not jnp.isinf(snd_unscaled):
                    # Update moving average with the UNSCALED SND (may be overridden by adapter_snd)
                    current_snd_ma_temp = (
                        1 - snd_ma_coef
                    ) * current_snd_ma + snd_ma_coef * snd_unscaled
                else:
                    # Keep previous moving average if current SND is invalid
                    snd_unscaled = current_snd_ma
                    current_snd_ma_temp = current_snd_ma

                # Compute diversity scaling based on config setting
                max_scaling = config["training"].get("max_diversity_scaling", 100.0)
                min_snd_floor = config["training"].get("min_snd_floor", 1e-6)
                use_per_env_scaling = config["training"].get(
                    "per_env_diversity_scaling", False
                )

                if use_per_env_scaling:
                    # PER-ENVIRONMENT scaling: each env scaled by its own target
                    # Shape: (num_envs,) -> will be broadcast to (num_envs * num_agents, 1, 1)
                    target_snd_per_env_jax = jnp.array(
                        target_snd_per_env
                    )  # (num_envs,)
                    diversity_scaling_per_env = jnp.sqrt(
                        target_snd_per_env_jax
                        / jnp.maximum(current_snd_ma_temp, min_snd_floor)
                    )
                    diversity_scaling_per_env = jnp.clip(
                        diversity_scaling_per_env, 0.001, jnp.sqrt(max_scaling)
                    )
                    diversity_scaling_temp = float(jnp.mean(diversity_scaling_per_env))
                else:
                    # UNIFORM scaling: all envs use mean target (more stable)
                    diversity_scaling_temp = float(
                        jnp.sqrt(
                            target_snd_mean
                            / jnp.maximum(current_snd_ma_temp, min_snd_floor)
                        )
                    )
                    diversity_scaling_temp = float(
                        jnp.clip(diversity_scaling_temp, 0.001, jnp.sqrt(max_scaling))
                    )
                    diversity_scaling_per_env = jnp.full(
                        num_envs, diversity_scaling_temp
                    )

                # For logging, use mean of per-env scaling
                diversity_scaling_temp = float(jnp.mean(diversity_scaling_per_env))

                # Now generate SCALED adapters FOR SND LOGGING (using sampled inputs)
                # Use mean scaling for SND computation (since SND is global metric)
                adapters_dict_snd = _get_static_adapters(
                    hn_state.params,
                    jax_task_sample,
                    jax_context_sample,
                    jax_lidar_sample,
                    jax_food_positions_sample,
                    jax_agent_positions_sample,
                    jax_target_snd_sample,
                    jax_env_context_sample,
                    jax_mask_sample,
                    diversity_scaling_temp,  # Use mean for SND logging
                )

                # Also compute scaled SND for logging comparison
                sample_adapters_scaled = {}
                for key, adapter in adapters_dict_snd.items():
                    adapter_shape = adapter.shape
                    reshaped = adapter.reshape(
                        adapter_sample_size, env_num_agents, *adapter_shape[1:]
                    )
                    # Transpose to agent-major to match observation ordering
                    # From: [env0_agent0, env0_agent1, ..., env1_agent0, env1_agent1, ...]
                    # To: [agent0_env0, agent0_env1, ..., agent1_env0, agent1_env1, ...]
                    sampled_transposed = reshaped.transpose(
                        1, 0, *range(2, len(reshaped.shape))
                    )
                    sample_adapters_scaled[key] = sampled_transposed.reshape(
                        adapter_sample_size * env_num_agents, *adapter_shape[1:]
                    )

                # Forward pass with scaled adapters
                if use_gru_policy:
                    # GRU policy: reuse hidden state from unscaled pass
                    batch_size_snd = adapter_sample_size * env_num_agents
                    init_hidden = shared_policy.initialize_carry(
                        batch_size_snd, shared_policy.gru_hidden_dim
                    )
                    sample_obs_seq = sample_obs_flat[None, ...]
                    dones_seq = jnp.zeros((1, batch_size_snd), dtype=bool)
                    avail_seq = None
                    policy_x = (sample_obs_seq, dones_seq, avail_seq)

                    _, output = shared_policy.apply(
                        {"params": policy_state.params},
                        init_hidden,
                        policy_x,
                        sample_adapters_scaled,
                    )
                    # For continuous actions, output is (mean_seq, log_std_seq)
                    mean_seq, _ = output
                    mean_sample_scaled = mean_seq[
                        0
                    ]  # Remove time dim: (batch, action_dim)
                else:
                    # MLP policy
                    mean_sample_scaled, _ = shared_policy.apply(
                        {"params": policy_state.params},
                        sample_obs_flat,
                        sample_adapters_scaled,
                    )

                # Reshape: output is structured as [agent0 outputs, agent1 outputs, ...]
                # Shape: (env_num_agents * adapter_sample_size, action_dim) -> (env_num_agents, adapter_sample_size, action_dim) -> (adapter_sample_size, env_num_agents, action_dim)
                mean_reshaped_scaled = mean_sample_scaled.reshape(
                    env_num_agents, adapter_sample_size, action_dim
                ).transpose(1, 0, 2)

                pair_distances_scaled = []
                for i in range(env_num_agents):
                    for j in range(i + 1, env_num_agents):
                        dist = jnp.linalg.norm(
                            mean_reshaped_scaled[:, i, :]
                            - mean_reshaped_scaled[:, j, :],
                            ord=2,
                            axis=-1,
                        )
                        pair_distances_scaled.append(dist)
                pairwise_distances_scaled = jnp.stack(pair_distances_scaled, axis=-1)
                snd_scaled = float(jnp.mean(pairwise_distances_scaled))

                # Finalize diversity_scaling and current_snd_ma
                # If adapter_snd was calculated, these will be overridden in the next section
                # Otherwise, use the action-based SND values
                if use_diversity_control:
                    # Apply diversity control: use per-env scaling
                    # diversity_scaling_per_env is already computed, keep it
                    # For scalar operations, use mean
                    diversity_scaling = diversity_scaling_temp
                    current_snd_ma = current_snd_ma_temp
                else:
                    # Monitor only: don't apply scaling but still track SND
                    diversity_scaling_per_env = jnp.ones(num_envs)  # No scaling
                    diversity_scaling = 1.0
                    current_snd_ma = (
                        current_snd_ma_temp  # Still update MA for monitoring
                    )

                # Generate SCALED adapters for ACTUAL ROLLOUT (using full num_envs)
                # Apply per-environment scaling for actual rollout
                # Expand per-env scaling to per-agent: (num_envs,) -> (num_envs * env_num_agents, 1, 1)
                # Each environment's scaling is repeated for all its agents
                diversity_scaling_broadcast = jnp.repeat(
                    diversity_scaling_per_env, env_num_agents
                )[
                    :, None, None
                ]  # Reshape for broadcasting with adapter tensor
                adapters_dict = _get_static_adapters(
                    hn_state.params,
                    jax_task,
                    jax_context,
                    jax_lidar,
                    jax_food_positions,
                    jax_agent_positions,
                    jax_target_snd,
                    jax_env_context,
                    jax_mask,
                    diversity_scaling_broadcast,  # Per-env scaling repeated per agent
                )

                # Store adapters for visualization (may be updated after adapter_snd calculation)
                rollout_adapters_unscaled = _get_static_adapters(
                    hn_state.params,
                    jax_task,
                    jax_context,
                    jax_lidar,
                    jax_food_positions,
                    jax_agent_positions,
                    jax_target_snd,
                    jax_env_context,
                    jax_mask,
                    1.0,  # Unscaled
                )
                rollout_adapters_scaled = adapters_dict

                # Store for logging (will be updated if adapter_snd is calculated)
                diversity_stats = {
                    "current_snd": snd_unscaled,  # May be overridden by adapter_snd
                    "snd_unscaled": snd_unscaled,
                    "snd_scaled": snd_scaled,
                    "current_snd_ma": current_snd_ma,  # May be overridden by adapter_snd
                    "diversity_scaling": diversity_scaling,  # Mean scaling for logging
                    "diversity_scaling_used": diversity_scaling,  # May be overridden by adapter_snd
                    "target_snd": target_snd_mean,  # Mean target for logging
                    "target_snd_min": float(target_snd_per_env.min()),  # Log range
                    "target_snd_max": float(target_snd_per_env.max()),
                    "diversity_control_active": use_diversity_control,  # Flag to indicate if scaling is applied
                }
            else:
                # No legacy SND calculation - this branch is used when:
                # 1. Not using diversity control at all, OR
                # 2. Using adapter_snd (skip legacy SND to save compute)
                #
                # When using adapter_snd: Generate initial unscaled adapters here,
                # then adapter SND calculation will compute diversity_scaling and
                # regenerate scaled adapters later (around line 5665)
                diversity_scaling = 1.0
                # Generate adapters for rollout (unscaled)
                adapters_dict = _get_static_adapters(
                    hn_state.params,
                    jax_task,
                    jax_context,
                    jax_lidar,
                    None,  # food_positions_batch not used here
                    jax_agent_positions,
                    jax_target_snd,
                    jax_env_context,
                    jax_mask,
                    diversity_scaling,
                )
                # For visualization: both are the same since no scaling
                rollout_adapters_unscaled = adapters_dict
                rollout_adapters_scaled = adapters_dict
                diversity_stats = None

            # ================================================================
            # Store initial query observations to adapter SND buffer
            # ================================================================
            if use_adapter_snd and adapter_snd_buffer is not None:
                # At episode start, HN is queried for ALL active agents in ALL environments
                # In mixed training: only add entries for current_num_agents (not all env_num_agents)
                # Add one entry PER ACTIVE AGENT (buffer size = num_envs * current_num_agents)
                # For per-env variation: iterate and check each env's agent count

                # For GRU: save the initial hidden state (zeros at episode start)
                if use_gru_policy:
                    # Hidden state is initialized to zeros at episode start (line ~4619)
                    # Shape: (batch_size, hidden_dim) where batch_size = num_envs * num_agents
                    initial_hidden = jnp.zeros(
                        (batch_size, shared_policy.gru_hidden_dim)
                    )
                else:
                    initial_hidden = None

                for env_idx in range(num_envs):
                    # Determine number of active agents for this environment
                    if mixed_agent_training and per_env_agent_variation:
                        num_agents_in_env = int(current_num_agents_per_env[env_idx])
                    elif mixed_agent_training:
                        num_agents_in_env = current_num_agents
                    else:
                        num_agents_in_env = env_num_agents

                    for agent_idx in range(num_agents_in_env):
                        # For GRU: extract this specific agent's hidden state
                        if use_gru_policy:
                            # Compute flat index for this agent
                            flat_idx = env_idx * env_num_agents + agent_idx
                            agent_hidden = initial_hidden[flat_idx]  # (hidden_dim,)
                        else:
                            agent_hidden = None

                        entry = {
                            "task": (
                                jax_task[env_idx][agent_idx]
                                if jax_task is not None
                                else None
                            ),  # (task_dim,)
                            "context": (
                                jax_context[env_idx][agent_idx]
                                if jax_context is not None
                                else None
                            ),  # (context_dim,)
                            "lidar": (
                                jax_lidar[env_idx][agent_idx]
                                if jax_lidar is not None
                                else None
                            ),  # (lidar_dim,) or None
                            "target_snd": (
                                jax_target_snd[env_idx][agent_idx]
                                if jax_target_snd is not None
                                else None
                            ),
                            "query_type": "initial",  # Mark as initial query
                            "hidden_state": agent_hidden,  # GRU hidden state at query time
                            "agent_idx": agent_idx,  # Store agent index for food position extraction
                        }
                        adapter_snd_buffer.append(entry)
                # Buffer now has num_envs * current_num_agents entries (e.g., 128 * 2 = 256)
                # Each entry is one agent's context

                # Verification: Check initial hidden states are zeros (only on first episode)
                if use_gru_policy and episode == 0:
                    sample_hidden = adapter_snd_buffer[0]["hidden_state"]
                    h_norm = float(jnp.linalg.norm(sample_hidden))
                    print(
                        f"  Initial query hidden state norm (should be ~0): {h_norm:.8f}"
                    )

            # ================================================================
            # Calculate Adapter-based SND (if enabled) and update diversity control
            # ================================================================
            adapter_snd_value = None
            adapter_snd_scaled = None
            if (
                use_adapter_snd
                and adapter_snd_buffer is not None
                and len(adapter_snd_buffer) > 0
            ):
                # Generate a new RNG key for adapter SND calculation
                rng_key_adapter_snd = jax.random.PRNGKey(episode * 1000 + 42)

                # Calculate adapter-based SND using observations from trajectory replay buffer
                # Uses task/context/lidar from adapter_snd_buffer and observations from previous rollout
                adapter_snd_value = calculate_adapter_snd(
                    adapter_snd_buffer=adapter_snd_buffer,
                    sample_size=adapter_snd_sample_size,
                    hn_params=hn_state.params,
                    policy_params=policy_state.params,
                    hypernetwork=hypernetwork,
                    policy_model=shared_policy,
                    num_agents=env_num_agents,
                    rng_key=rng_key_adapter_snd,
                    use_gru_policy=use_gru_policy,
                    trajectory_obs=previous_trajectory_obs,  # Use observations from previous rollout
                    scenario_name=scenario_name,  # Pass scenario name for food position extraction
                )

                # Use Adapter SND for diversity control calculations
                if (
                    adapter_snd_value is not None
                    and not jnp.isnan(adapter_snd_value)
                    and not jnp.isinf(adapter_snd_value)
                    and use_diversity_control
                ):
                    # Update moving average with Adapter SND
                    current_snd_ma = (
                        1 - snd_ma_coef
                    ) * current_snd_ma + snd_ma_coef * adapter_snd_value

                    # Compute diversity scaling based on config setting
                    max_scaling = config["training"].get("max_diversity_scaling", 100.0)
                    min_snd_floor = config["training"].get("min_snd_floor", 1e-6)
                    use_per_env_scaling = config["training"].get(
                        "per_env_diversity_scaling", False
                    )

                    if use_per_env_scaling:
                        # PER-ENVIRONMENT scaling: each env scaled by its own target
                        target_snd_per_env_jax = jnp.array(
                            target_snd_per_env
                        )  # (num_envs,)
                        diversity_scaling_per_env = jnp.sqrt(
                            target_snd_per_env_jax
                            / jnp.maximum(current_snd_ma, min_snd_floor)
                        )
                        diversity_scaling_per_env = jnp.clip(
                            diversity_scaling_per_env, 0.001, jnp.sqrt(max_scaling)
                        )
                        diversity_scaling = float(jnp.mean(diversity_scaling_per_env))
                    else:
                        # UNIFORM scaling: all envs use mean target (more stable)
                        diversity_scaling = float(
                            jnp.sqrt(
                                target_snd_mean
                                / jnp.maximum(current_snd_ma, min_snd_floor)
                            )
                        )
                        diversity_scaling = float(
                            jnp.clip(diversity_scaling, 0.001, jnp.sqrt(max_scaling))
                        )
                        diversity_scaling_per_env = jnp.full(
                            num_envs, diversity_scaling
                        )

                    # For scalar operations and logging, use mean
                    diversity_scaling = float(jnp.mean(diversity_scaling_per_env))

                    # Calculate scaled adapter SND using the same inputs but with mean diversity_scaling
                    adapter_snd_scaled = calculate_adapter_snd(
                        adapter_snd_buffer=adapter_snd_buffer,
                        sample_size=adapter_snd_sample_size,
                        hn_params=hn_state.params,
                        policy_params=policy_state.params,
                        hypernetwork=hypernetwork,
                        policy_model=shared_policy,
                        num_agents=env_num_agents,
                        rng_key=rng_key_adapter_snd,
                        use_gru_policy=use_gru_policy,
                        trajectory_obs=previous_trajectory_obs,
                        diversity_scaling=diversity_scaling,  # Use computed scaling factor
                        scenario_name=scenario_name,  # Pass scenario name for food position extraction
                    )

                # Update diversity_stats with Adapter SND as primary metric
                if diversity_stats is not None:
                    diversity_stats["adapter_snd"] = adapter_snd_value
                    if adapter_snd_value is not None:
                        diversity_stats["current_snd"] = (
                            adapter_snd_value  # Use adapter SND as primary
                        )
                        diversity_stats["current_snd_ma"] = current_snd_ma
                        diversity_stats["diversity_scaling"] = diversity_scaling
                        diversity_stats["diversity_scaling_used"] = diversity_scaling
                        # Add scaled adapter SND if computed successfully
                        if (
                            "adapter_snd_scaled" in locals()
                            and adapter_snd_scaled is not None
                        ):
                            diversity_stats["adapter_snd_scaled"] = adapter_snd_scaled
                elif diversity_stats is None and adapter_snd_value is not None:
                    # Create diversity_stats
                    diversity_stats = {
                        "adapter_snd": adapter_snd_value,
                        "current_snd": adapter_snd_value,  # Use adapter SND as primary
                        "snd_unscaled": 0.0,  # Not calculated when adapter_snd takes over
                        "snd_scaled": 0.0,  # Not calculated when adapter_snd takes over
                        "current_snd_ma": current_snd_ma,
                        "diversity_scaling": diversity_scaling,
                        "diversity_scaling_used": diversity_scaling,
                        "target_snd": target_snd_mean,  # Mean target for logging
                        "target_snd_min": float(target_snd_per_env.min()),  # Log range
                        "target_snd_max": float(target_snd_per_env.max()),
                    }
                    # Add scaled adapter SND if computed successfully
                    if (
                        "adapter_snd_scaled" in locals()
                        and adapter_snd_scaled is not None
                    ):
                        diversity_stats["adapter_snd_scaled"] = adapter_snd_scaled

                # Regenerate adapters with updated diversity_scaling based on Adapter SND (if calculated)
                if (
                    use_diversity_control
                    and adapter_snd_value is not None
                    and not jnp.isnan(adapter_snd_value)
                    and not jnp.isinf(adapter_snd_value)
                ):
                    # Generate UNSCALED adapters (diversity_scaling=1.0) for visualization comparison
                    rollout_adapters_unscaled = _get_static_adapters(
                        hn_state.params,
                        jax_task,
                        jax_context,
                        jax_lidar,
                        jax_food_positions,
                        jax_agent_positions,
                        jax_target_snd,
                        jax_env_context,
                        jax_mask,
                        1.0,  # Unscaled
                    )

                    # Generate SCALED adapters (with diversity control) for actual rollout
                    # Apply per-environment scaling
                    # Expand per-env scaling to per-agent: (num_envs,) -> (num_envs * env_num_agents, 1, 1)
                    diversity_scaling_broadcast = jnp.repeat(
                        diversity_scaling_per_env, env_num_agents
                    )[
                        :, None, None
                    ]  # Reshape for broadcasting with adapter tensor
                    adapters_dict = _get_static_adapters(
                        hn_state.params,
                        jax_task,
                        jax_context,
                        jax_lidar,
                        jax_food_positions,
                        jax_agent_positions,
                        jax_target_snd,
                        jax_env_context,
                        jax_mask,
                        diversity_scaling_broadcast,  # Per-env scaling repeated per agent
                    )
                    rollout_adapters_scaled = adapters_dict  # Store for visualization

        elif use_dico:
            # DiCo: Compute SND over per-agent policies using snd.py
            if use_diversity_control:
                # Determine sample size and get observations
                if use_snd_obs_buffer and snd_obs_buffer is not None:
                    # Use observations from the pre-collected buffer
                    # Sample from buffer if it's larger than needed
                    actual_buffer_size = len(snd_obs_buffer)
                    if actual_buffer_size >= snd_obs_buffer_sample_size:
                        # Random sample from buffer
                        sample_indices = np_rng.choice(
                            actual_buffer_size,
                            size=snd_obs_buffer_sample_size,
                            replace=False,
                        )
                        common_obs_jax = snd_obs_buffer[
                            sample_indices
                        ]  # (sample_size, obs_dim)
                        sample_size = snd_obs_buffer_sample_size
                    else:
                        # Use all available observations if buffer is smaller
                        common_obs_jax = snd_obs_buffer
                        sample_size = actual_buffer_size
                else:
                    # Use current environment observations (from reset)
                    # For DICO, use num_snd_samples from config (defaults to 32 for efficiency)
                    dico_sample_size = config["training"].get("num_snd_samples", 32)
                    sample_size = min(num_envs, dico_sample_size)
                    # Use agent 0's observations as common observations for all agents
                    obs_agent0_np = obs[0][:sample_size].detach().cpu().numpy()
                    common_obs_jax = jnp.asarray(
                        obs_agent0_np
                    )  # (sample_size, obs_dim)

                # Get actual sample size (in case sample_size > what's available)
                actual_sample_size = common_obs_jax.shape[0]

                # Debug: Verify observations are identical for all agents (for linear scaling)
                if episode % log_interval == 0 and episode == 0:
                    print(f"  Debug - Observation setup for SND:")
                    print(f"    Using {actual_sample_size} observations")
                    print(
                        f"    Same obs for all agents: {True}"
                    )  # We tile the same obs

                # Tile for all agents to match agent_ids pattern
                # agent_ids pattern: [0, 1, 2, ..., N-1, 0, 1, 2, ..., N-1, ...]  (repeats each env)
                # obs pattern: [env0_obs, env0_obs, ..., env1_obs, env1_obs, ...]  (repeat each obs num_agents times)
                # This ensures all agents evaluate the SAME observations (critical for linear SND scaling)
                sample_obs_flat = jnp.repeat(
                    common_obs_jax, env_num_agents, axis=0
                )  # (env_num_agents * actual_sample_size, obs_dim)

                # For DiCo with dispersion_vmas, append max_speed capability to observations
                if (
                    scenario_name == "dispersion_vmas"
                    and use_capability_context
                    and "agent_max_speeds" in locals()
                    and agent_max_speeds is not None
                ):
                    # Tile agent_max_speeds across all sample observations
                    # agent_max_speeds: list of max_speed values for each agent
                    # Need to tile it: [speed0, speed1, ..., speedN-1] repeated actual_sample_size times
                    max_speed_flat = np.tile(
                        np.array(agent_max_speeds, dtype=np.float32), actual_sample_size
                    )  # (env_num_agents * actual_sample_size,)
                    # Reshape to (batch_size, 1) to append to observations
                    max_speed_obs = jnp.asarray(max_speed_flat[:, np.newaxis])
                    sample_obs_flat = jnp.concatenate(
                        [sample_obs_flat, max_speed_obs], axis=-1
                    )

                # Use calculate_snd_dico function for proper SND calculation
                # Calculate UNSCALED SND first (for moving average update)
                snd_unscaled = calculate_snd_dico(
                    policy_state.params,
                    sample_obs_flat,
                    shared_policy,
                    env_num_agents,
                    actual_sample_size,
                    diversity_scaling=1.0,
                )

                # NaN protection: check SND before updating moving average
                if not jnp.isnan(snd_unscaled) and not jnp.isinf(snd_unscaled):
                    # Update moving average
                    current_snd_ma = (
                        1 - snd_ma_coef
                    ) * current_snd_ma + snd_ma_coef * snd_unscaled
                    # Protect against NaN propagation in moving average
                    if jnp.isnan(current_snd_ma) or jnp.isinf(current_snd_ma):
                        # Use mean target if available (mixed targets), otherwise use config target
                        current_snd_ma = (
                            target_snd_mean
                            if "target_snd_mean" in locals()
                            else target_snd
                        )
                else:
                    # Keep previous moving average if current SND is invalid
                    snd_unscaled = current_snd_ma

                # Compute diversity scaling based on UNSCALED SND
                # For DICO use linear ratio: lambda = target_snd / current_snd_ma
                # This controls how much the heterogeneous component contributes: π = homo + λ*hetero
                # Mathematically: SND should scale linearly with λ since distances scale linearly
                # Use mean target if available (for mixed targets), otherwise use config target
                # DiCo applies scaling per-agent, so we use a single scalar value
                max_scaling = config["training"].get("max_diversity_scaling", 100.0)
                min_snd_floor = config["training"].get(
                    "min_snd_floor", 1e-6
                )  # Prevent division by zero
                target_for_scaling = (
                    target_snd_mean if "target_snd_mean" in locals() else target_snd
                )
                diversity_scaling = target_for_scaling / jnp.maximum(
                    current_snd_ma, min_snd_floor
                )
                diversity_scaling = jnp.clip(diversity_scaling, 0.0001, max_scaling)

                # NaN/Inf protection for diversity_scaling (using JAX operations)
                diversity_scaling = jnp.where(
                    jnp.isnan(diversity_scaling) | jnp.isinf(diversity_scaling),
                    1.0,
                    diversity_scaling,
                )

                # NaN/Inf protection already applied above via jnp.where. Skip the
                # float() pull on every episode: it forces a device->host sync just
                # to print a warning that can't fire (jnp.where replaced any
                # NaN/Inf with 1.0). On log episodes downstream code already
                # converts to float when it needs to print.

                # Calculate SCALED SND (with diversity_scaling applied) only on
                # log episodes. calculate_snd_dico is roughly as expensive as a
                # full policy forward pass; computing it every episode just to
                # print it doubles SND cost. On non-log episodes we fall back to
                # the linear prediction (snd_unscaled * lambda), which is what
                # the diagnostic below claims should hold anyway.
                if episode % log_interval == 0:
                    snd_scaled = calculate_snd_dico(
                        policy_state.params,
                        sample_obs_flat,
                        shared_policy,
                        env_num_agents,
                        actual_sample_size,
                        diversity_scaling=diversity_scaling,
                    )
                    expected_scaling_ratio = float(diversity_scaling)
                    actual_scaling_ratio = float(
                        snd_scaled / jnp.maximum(snd_unscaled, 1e-8)
                    )
                    scaling_efficiency = actual_scaling_ratio / max(
                        expected_scaling_ratio, 1e-8
                    )
                    print(f"  Debug - DICO Scaling Analysis:")
                    print(
                        f"    Expected scaling ratio (λ): {expected_scaling_ratio:.4f}"
                    )
                    print(f"    Actual SND scaling ratio: {actual_scaling_ratio:.4f}")
                    print(
                        f"    Scaling efficiency: {scaling_efficiency:.4f} (should be ~1.0 for linear)"
                    )
                    print(
                        f"    SND: {snd_unscaled:.4f} → {snd_scaled:.4f} (expected: {snd_unscaled * expected_scaling_ratio:.4f})"
                    )
                else:
                    # Cheap linear prediction; no second forward pass.
                    snd_scaled = snd_unscaled * diversity_scaling

                diversity_stats = {
                    "current_snd": snd_unscaled,
                    "snd_unscaled": snd_unscaled,
                    "snd_scaled": snd_scaled,
                    "current_snd_ma": current_snd_ma,
                    "diversity_scaling": diversity_scaling,
                    "diversity_scaling_used": diversity_scaling,
                    "target_snd": (
                        target_snd_mean if "target_snd_mean" in locals() else target_snd
                    ),  # Mean target for logging
                    "diversity_control_active": True,
                }
            else:
                diversity_scaling = 1.0
                diversity_stats = None

            # DiCo doesn't use adapters
            adapters_dict = None
        else:
            # Standard MAPPO (no hypernetwork, no DiCo)
            # All agents share the same policy parameters - SND should be ~0

            # Calculate SND for baseline (should be zero)
            # CRITICAL: For proper SND calculation, use the same observations for all agents
            sample_size = num_envs
            common_obs_torch = obs[0][
                :sample_size
            ]  # (sample_size, obs_dim) - agent 0's observations
            common_obs_jax = jnp.asarray(common_obs_torch.cpu().numpy())

            # If env_context is appended to policy observations during the rollout
            # (reverse_transport / pressure_plate shared baseline), we must append
            # the current env_context here too so the observation dim matches the
            # policy's input dim (policy_obs_dim = obs_dim + env_context_dim).
            if (
                scenario_name in ("reverse_transport", "pressure_plate")
                and jax_env_context is not None
                and env_context_dim > 0
            ):
                # jax_env_context: (num_envs, env_num_agents, env_context_dim)
                # Take the first agent's context slice: (num_envs, env_context_dim)
                env_ctx_for_snd = jax_env_context[:sample_size, 0, :]
                common_obs_jax = jnp.concatenate(
                    [common_obs_jax, env_ctx_for_snd], axis=-1
                )

            # Since all agents share the SAME policy (no adapters) and we give them the SAME observations,
            # they should produce IDENTICAL outputs, giving SND = 0

            # Evaluate the shared policy once (all agents will produce identical outputs)
            if use_gru_policy:
                init_hidden = shared_policy.initialize_carry(
                    sample_size, shared_policy.gru_hidden_dim
                )
                obs_seq = common_obs_jax[None, ...]  # (1, sample_size, policy_obs_dim)
                dones_seq = jnp.zeros((1, sample_size), dtype=bool)
                avail_seq = None
                policy_x = (obs_seq, dones_seq, avail_seq)

                # Create empty adapters — input_dim must match policy_obs_dim
                empty_adapters = {}
                input_dim = policy_obs_dim
                for i, output_dim in enumerate(policy_hidden_dims):
                    layer_idx = i + 1
                    empty_adapters[f"A{layer_idx}"] = jnp.zeros(
                        (sample_size, 0, input_dim)
                    )
                    empty_adapters[f"B{layer_idx}"] = jnp.zeros(
                        (sample_size, output_dim, 0)
                    )
                    input_dim = output_dim
                final_idx = len(policy_hidden_dims) + 1
                empty_adapters[f"A{final_idx}"] = jnp.zeros(
                    (sample_size, 0, policy_hidden_dims[-1])
                )
                empty_adapters[f"B{final_idx}"] = jnp.zeros(
                    (sample_size, action_dim, 0)
                )

                _, output = shared_policy.apply(
                    {"params": policy_state.params},
                    init_hidden,
                    policy_x,
                    empty_adapters,
                )
                mean_seq, _ = output
                agent_mean = mean_seq[0]  # (sample_size, action_dim)
            else:
                agent_mean, _ = shared_policy.apply(
                    {"params": policy_state.params},
                    common_obs_jax,
                    {},  # Empty adapter dict
                )  # (sample_size, action_dim)

            # All agents produce identical outputs with the same policy and same observations
            # Replicate for all agents to compute SND (should be 0)
            all_agent_means_baseline = jnp.stack(
                [agent_mean for _ in range(env_num_agents)], axis=0
            )  # (env_num_agents, sample_size, action_dim)

            # Compute pairwise Wasserstein-2 distances (should be ~0)
            pair_distances = []
            for i in range(env_num_agents):
                for j in range(i + 1, env_num_agents):
                    obs_wise_dist_sq = jnp.sum(
                        (all_agent_means_baseline[i] - all_agent_means_baseline[j])
                        ** 2,
                        axis=-1,
                    )
                    w2_dist = jnp.sqrt(jnp.mean(obs_wise_dist_sq))
                    pair_distances.append(w2_dist)

            if len(pair_distances) > 0:
                snd_baseline = float(jnp.mean(jnp.array(pair_distances)))
            else:
                snd_baseline = 0.0

            diversity_stats = {
                "current_snd": snd_baseline,
                "snd_unscaled": snd_baseline,
                "snd_scaled": snd_baseline,
                "current_snd_ma": snd_baseline,
                "diversity_scaling": 1.0,
                "diversity_scaling_used": 1.0,
                "target_snd": (
                    target_snd_mean if "target_snd_mean" in locals() else target_snd
                ),
            }
            diversity_scaling = 1.0

            # No adapters: pass zeros for adapter dict
            adapters_dict = {}
            input_dim = obs_dim
            for i, output_dim in enumerate(policy_hidden_dims):
                layer_idx = i + 1
                adapters_dict[f"A{layer_idx}"] = jnp.zeros((batch_size, 0, input_dim))
                adapters_dict[f"B{layer_idx}"] = jnp.zeros((batch_size, output_dim, 0))
                input_dim = output_dim
            # Final adapter for output layer
            final_idx = len(policy_hidden_dims) + 1
            adapters_dict[f"A{final_idx}"] = jnp.zeros(
                (batch_size, 0, policy_hidden_dims[-1])
            )
            adapters_dict[f"B{final_idx}"] = jnp.zeros((batch_size, action_dim, 0))

        # ====================================================================
        # Rollout Loop
        # ====================================================================
        episode_rewards = []
        trajectory_data = {
            "obs": [],
            "global_states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "action_masks": [],  # For SMAX discrete action masking
            "dones": [],  # Track episode boundaries for GAE
            "init_hidden": None,  # Store initial hidden states for actor GRU training
            "init_critic_hidden": None,  # Store initial hidden states for critic RNN training
            "context": [],  # Store per-timestep context for hypernetwork
            "cash_context": [],  # Store per-step CASH capability context
            "lidar": [],  # Store per-timestep lidar for observation-based context
            "food_positions": [],  # Store per-timestep food positions for dispersion_vmas
            "agent_positions": [],  # Store per-timestep agent positions for dispersion_vmas
            "env_context": [],  # Store per-timestep environment context (e.g., package properties)
        }

        # Note: rollout_adapters_unscaled and rollout_adapters_scaled are set BEFORE this
        # during adapter generation (lines ~3997-4005), so we don't reinitialize them here

        # CRITICAL: Initialize hidden states to ZERO at start of each rollout
        # For episodic tasks, hidden states should reset at episode boundaries (tracked by dones)
        # The hidden states will properly reset during rollout when done=True via ScannedRNN
        if use_gru_policy:
            # Initialize to zeros - states will evolve during rollout and reset on dones
            init_hidden = jnp.zeros((batch_size, shared_policy.gru_hidden_dim))
            init_critic_hidden = jnp.zeros((num_envs, critic.gru_hidden_dim))
            trajectory_data["init_hidden"] = init_hidden
            trajectory_data["init_critic_hidden"] = init_critic_hidden
            # Also reset the persistent states for rollout
            gru_hidden_states = init_hidden
            critic_hidden_states = init_critic_hidden

        # Track episode resets for debugging
        num_episode_resets = 0
        total_steps_before_resets = []

        # Create JAX RNG key for sampling (use different name to avoid shadowing NumPy rng)
        jax_rng = jax.random.PRNGKey(episode)

        # Track which agents have already detected food to ensure we only requery once
        # Use env_num_agents for consistent shape with environment
        food_detected_already = torch.zeros(
            num_envs, env_num_agents, dtype=torch.bool, device=torch_device
        )

        # Check initial observation: mark agents that can already see food at episode start
        # This prevents requerying the hypernetwork for food that was visible from initialization
        # NOTE: dispersion_vmas has global observability, so skip this check
        if use_hypernetwork and adaptive_hypernetwork and scenario_name == "dispersion":
            initial_food_in_range = detect_food_in_range(obs, num_envs, max_agents)
            food_detected_already = initial_food_in_range.clone()

            # Log initial detections if any (only occasionally to avoid spam)
            if (
                episode % log_interval == 0 or episode == 0
            ) and initial_food_in_range.any():
                num_initial = initial_food_in_range.sum().item()
                print(
                    f"  [Episode {episode}] {num_initial} agent(s) can see food at initialization (won't requery)"
                )

        # ====================================================================
        # TIMING: Start rollout phase
        # ====================================================================
        rollout_start_time = time.time()

        for step in range(rollout_steps):  # Rollout steps per episode

            # ================================================================
            # Dynamic environment changes (reverse_transport)
            # ================================================================
            if scenario_name == "reverse_transport":
                use_dynamic_env = config["env"].get("use_dynamic_env_changes", False)
                env_change_interval = config["env"].get("env_change_interval", 0)

                # Debug: Print interval on first check
                if step == 1 and episode == 0:
                    print(
                        f"\n[Dynamic Env Config] use_dynamic_env_changes={use_dynamic_env}, env_change_interval={env_change_interval}"
                    )
                    print(
                        f"[Dynamic Env Config] env_change_type={config['env'].get('env_change_type', 'N/A')}\n"
                    )

                if (
                    use_dynamic_env
                    and env_change_interval > 0
                    and step > 0
                    and step % env_change_interval == 0
                ):
                    # Debug: Print when condition triggers
                    if episode == 0:
                        print(
                            f"  [DEBUG] Step {step}: Condition triggered (step % {env_change_interval} == {step % env_change_interval})"
                        )

                    env_change_type = config["env"].get("env_change_type", "random")

                    # Get property ranges from config
                    mass_range = config["env"].get("package_mass_range", [1, 100])
                    width_range = config["env"].get("package_width_range", [0.4, 0.8])
                    length_range = config["env"].get("package_length_range", [0.4, 0.8])

                    # Generate new properties based on change type
                    if env_change_type == "random":
                        new_mass = np_rng.uniform(mass_range[0], mass_range[1])
                        new_width = np_rng.uniform(width_range[0], width_range[1])
                        new_length = np_rng.uniform(length_range[0], length_range[1])
                    elif env_change_type == "schedule":
                        # Cycle through predefined values
                        cycle_idx = (step // env_change_interval) % 3
                        if cycle_idx == 0:
                            new_mass, new_width, new_length = (
                                mass_range[0],
                                width_range[0],
                                length_range[0],
                            )
                        elif cycle_idx == 1:
                            new_mass = (mass_range[0] + mass_range[1]) / 2
                            new_width = (width_range[0] + width_range[1]) / 2
                            new_length = (length_range[0] + length_range[1]) / 2
                        else:
                            new_mass, new_width, new_length = (
                                mass_range[1],
                                width_range[1],
                                length_range[1],
                            )
                    elif env_change_type == "gradual":
                        # Gradually increase properties from min to max
                        progress = min(step / rollout_steps, 1.0)
                        new_mass = (
                            mass_range[0] + (mass_range[1] - mass_range[0]) * progress
                        )
                        new_width = (
                            width_range[0]
                            + (width_range[1] - width_range[0]) * progress
                        )
                        new_length = (
                            length_range[0]
                            + (length_range[1] - length_range[0]) * progress
                        )
                    elif env_change_type == "decreasing":
                        # Gradually decrease properties from max to min
                        progress = min(step / rollout_steps, 1.0)
                        new_mass = (
                            mass_range[1] - (mass_range[1] - mass_range[0]) * progress
                        )
                        new_width = (
                            width_range[1]
                            - (width_range[1] - width_range[0]) * progress
                        )
                        new_length = (
                            length_range[1]
                            - (length_range[1] - length_range[0]) * progress
                        )
                    else:
                        new_mass, new_width, new_length = None, None, None

                    # Update properties in environment
                    if new_mass is not None:
                        changed = env.scenario.update_package_properties(
                            package_mass=new_mass,
                            package_width=new_width,
                            package_length=new_length,
                        )
                        if changed:
                            # Update jax_env_context to reflect the new properties
                            if jax_env_context is not None:
                                env_context_vec = torch.tensor(
                                    [new_mass, new_width, new_length],
                                    device=torch_device,
                                    dtype=torch.float32,
                                )
                                updated_env_context = (
                                    env_context_vec.unsqueeze(0)
                                    .unsqueeze(0)
                                    .expand(num_envs, env_num_agents, env_context_dim)
                                )
                                env_context_np = (
                                    updated_env_context.detach().cpu().numpy()
                                    if updated_env_context.requires_grad
                                    else updated_env_context.cpu().numpy()
                                )
                                jax_env_context = jnp.asarray(env_context_np)
                                if use_cuda:
                                    jax_env_context = jax.device_put(
                                        jax_env_context, jax_device
                                    )

                            if (episode * rollout_steps + step) % 100 == 0:
                                # Calculate expected brightness for verification
                                min_mass_color = float(mass_range[0])
                                max_mass_color = float(mass_range[1])
                                if max_mass_color - min_mass_color > 0:
                                    norm_mass = max(
                                        0.0,
                                        min(
                                            1.0,
                                            (new_mass - min_mass_color)
                                            / (max_mass_color - min_mass_color),
                                        ),
                                    )
                                else:
                                    norm_mass = 0.0
                                expected_brightness = 1.0 - 0.9 * norm_mass

                                print(
                                    f"  [Episode {episode}, Step {step}] Changed package: "
                                    f"mass={new_mass:.2f}, width={new_width:.2f}, length={new_length:.2f}"
                                )
                                print(
                                    f"  [Episode {episode}, Step {step}] Expected color: norm={norm_mass:.3f}, brightness={expected_brightness:.3f}"
                                )

            # ================================================================
            # Scheduled target-SND switch and synchronized HN requery
            # ================================================================
            if (
                enable_rollout_target_snd_switch
                and step > 0
                and step % train_target_snd_interval == 0
            ):
                if train_target_snd_mode == "random":
                    new_target_snd = float(np_rng.choice(train_target_snd_list))
                else:
                    global_rollout_step = episode * rollout_steps + step
                    scheduled_index = (
                        global_rollout_step // train_target_snd_interval
                    ) % len(train_target_snd_list)
                    new_target_snd = float(train_target_snd_list[scheduled_index])

                # Update target for both diversity control and HN target context.
                target_snd_per_env = np.full(num_envs, new_target_snd)
                target_snd_mean = new_target_snd

                if target_snd_dim > 0:
                    jax_target_snd = jnp.full(
                        (num_envs, env_num_agents, target_snd_dim),
                        new_target_snd,
                        dtype=jnp.float32,
                    )
                    if use_cuda:
                        jax_target_snd = jax.device_put(jax_target_snd, jax_device)

                # Refresh agent-position context from current observations when available.
                if use_agent_position_context:
                    updated_agent_positions = extract_agent_positions(
                        obs, env_num_agents, scenario_name
                    )
                    if updated_agent_positions is not None:
                        updated_agent_positions_np = (
                            updated_agent_positions.detach().cpu().numpy()
                        )
                        jax_agent_positions = jnp.asarray(updated_agent_positions_np)
                        if use_cuda:
                            jax_agent_positions = jax.device_put(
                                jax_agent_positions, jax_device
                            )

                if use_diversity_control:
                    max_scaling = config["training"].get("max_diversity_scaling", 100.0)
                    min_snd_floor = config["training"].get("min_snd_floor", 1e-6)
                    use_per_env_scaling = config["training"].get(
                        "per_env_diversity_scaling", False
                    )

                    if use_per_env_scaling:
                        target_snd_per_env_jax = jnp.array(target_snd_per_env)
                        diversity_scaling_per_env = jnp.sqrt(
                            target_snd_per_env_jax
                            / jnp.maximum(current_snd_ma, min_snd_floor)
                        )
                        diversity_scaling_per_env = jnp.clip(
                            diversity_scaling_per_env, 0.001, jnp.sqrt(max_scaling)
                        )
                    else:
                        diversity_scaling = float(
                            jnp.sqrt(
                                target_snd_mean
                                / jnp.maximum(current_snd_ma, min_snd_floor)
                            )
                        )
                        diversity_scaling = float(
                            jnp.clip(diversity_scaling, 0.001, jnp.sqrt(max_scaling))
                        )
                        diversity_scaling_per_env = jnp.full(
                            num_envs, diversity_scaling
                        )

                    diversity_scaling = float(jnp.mean(diversity_scaling_per_env))
                    diversity_scaling_broadcast = jnp.repeat(
                        diversity_scaling_per_env, env_num_agents
                    )[:, None, None]
                    query_scaling = diversity_scaling_broadcast
                else:
                    query_scaling = (
                        diversity_scaling_broadcast
                        if "diversity_scaling_broadcast" in locals()
                        else diversity_scaling
                    )

                # Requery adapters so policy rollout immediately uses the new target.
                if use_hypernetwork:
                    adapters_dict = _get_static_adapters(
                        hn_state.params,
                        jax_task,
                        jax_context,
                        jax_lidar,
                        jax_food_positions,
                        jax_agent_positions,
                        jax_target_snd,
                        jax_env_context,
                        jax_mask,
                        query_scaling,
                    )
                    rollout_adapters_scaled = adapters_dict

                if diversity_stats is not None:
                    diversity_stats["target_snd"] = target_snd_mean
                    diversity_stats["target_snd_min"] = float(target_snd_per_env.min())
                    diversity_stats["target_snd_max"] = float(target_snd_per_env.max())
                    diversity_stats["diversity_scaling"] = diversity_scaling
                    diversity_stats["diversity_scaling_used"] = diversity_scaling

                if episode % log_interval == 0 or episode < 5:
                    print(
                        f"  [Episode {episode}, Step {step}] Switched target_snd to {new_target_snd:.4f} and requeried hypernetwork"
                    )

            # ================================================================
            # Track previous dones for GRU reset during rollout
            # At step 0: dones are True (treat start of episode as a reset)
            # At step N: dones come from step N-1's env.step()
            # ================================================================
            if use_gru_policy:
                if step == 0:
                    # First step of rollout: the hidden state was already saved
                    # Use zeros because the carrier already has the correct state
                    prev_dones_flat = jnp.zeros(batch_size, dtype=bool)
                else:
                    # Use dones from the PREVIOUS env.step() result
                    # This tells the GRU which environments just reset
                    prev_dones_flat = jnp.array(
                        trajectory_data["dones"][-1]
                    )  # (batch_size,)

            # ================================================================
            # WRAPPER: PyTorch -> JAX (for observations)
            # ================================================================
            # VMAS returns list of observations: [agent0_obs, agent1_obs, ...] - has env_num_agents elements
            # Each agent_obs has shape: (num_envs, obs_dim)
            # Stack them: (env_num_agents, num_envs, obs_dim)
            obs_stacked = torch.stack(obs, dim=0)  # (env_num_agents, num_envs, obs_dim)
            # Transpose to: (num_envs, env_num_agents, obs_dim)
            obs_transposed = obs_stacked.transpose(
                0, 1
            )  # (num_envs, env_num_agents, obs_dim)

            # Flatten to: (num_envs * env_num_agents, obs_dim)
            obs_flat = obs_transposed.reshape(batch_size, -1)  # (batch_size, obs_dim)

            # For SMAX with enemy curriculum, pad observations to max_obs_dim
            if scenario_name == "smax" and obs_flat.shape[1] < max_obs_dim:
                padding_size = max_obs_dim - obs_flat.shape[1]
                obs_flat = torch.cat(
                    [
                        obs_flat,
                        torch.zeros(batch_size, padding_size, device=torch_device),
                    ],
                    dim=1,
                )

            obs_np = (
                obs_flat.detach().cpu().numpy()
                if obs_flat.requires_grad
                else obs_flat.cpu().numpy()
            )

            # Check for NaN in observations (gated; op is numpy-cheap but still a full scan each step)
            if args.debug_nan and np.isnan(obs_np).any():
                print(
                    f"WARNING: NaN detected in observations at step {step}, episode {episode}"
                )
                print(f"  NaN count: {np.isnan(obs_np).sum()} / {obs_np.size}")
                print(f"  Replacing NaN with zeros")
                obs_np = np.nan_to_num(obs_np, nan=0.0, posinf=0.0, neginf=0.0)

            jax_obs = jnp.asarray(obs_np)

            # For the shared-policy baseline (no HN) on scenarios with dynamic env
            # context (reverse_transport, pressure_plate), append the context directly
            # to the policy observations so the baseline has access to the same
            # information that the HyperLoRA hypernetwork receives via env_context_batch.
            if (
                not use_hypernetwork
                and not use_dico
                and scenario_name in ("reverse_transport", "pressure_plate")
                and jax_env_context is not None
            ):
                # jax_env_context: (num_envs, env_num_agents, env_context_dim)
                # Flatten to (batch_size, env_context_dim) matching jax_obs layout
                env_context_flat = jax_env_context.reshape(batch_size, env_context_dim)
                jax_obs = jnp.concatenate([jax_obs, env_context_flat], axis=-1)

            # For DiCo with dispersion_vmas, append max_speed capability to observations
            if (
                use_dico
                and scenario_name == "dispersion_vmas"
                and use_capability_context
                and "agent_max_speeds" in locals()
                and agent_max_speeds is not None
            ):
                # agent_max_speeds: list of max_speed values for each agent
                # Need to tile it across all environments: (num_envs * env_num_agents,)
                max_speed_flat = np.tile(
                    np.array(agent_max_speeds, dtype=np.float32), num_envs
                )  # (batch_size,)
                # Reshape to (batch_size, 1) to append to observations
                max_speed_obs = jnp.asarray(max_speed_flat[:, np.newaxis])
                jax_obs = jnp.concatenate([jax_obs, max_speed_obs], axis=-1)

            # Create agent IDs for DICO (which agent each observation belongs to)
            # Shape: (batch_size,) with values 0, 1, 2, ..., env_num_agents-1 repeated
            if use_dico and use_diversity_control:
                # Create repeating pattern: [0, 1, 2, ..., n-1, 0, 1, 2, ..., n-1, ...]
                # Weight gathering approach is agnostic to batch organization
                agent_ids_np = np.tile(np.arange(env_num_agents), num_envs)
                jax_agent_ids = jnp.asarray(agent_ids_np)  # (batch_size,)
            else:
                jax_agent_ids = None

            if use_cash:
                if jax_context is not None:
                    cash_context_3d = jax_context
                else:
                    cash_context_3d = jnp.zeros(
                        (
                            num_envs,
                            env_num_agents,
                            context_dim,
                        ),
                        dtype=jnp.float32,
                    )
                jax_cash_context_flat = cash_context_3d.reshape(batch_size, -1)
            else:
                jax_cash_context_flat = None

            # Move to JAX device if using CUDA
            if use_cuda:
                jax_obs = jax.device_put(jax_obs, jax_device)

            # Create global state for centralized critic (reshape to num_envs, max_agents * obs_dim)
            # For mixed training: only include active agents in global state
            # For SMAX curriculum: pad to max total units (allies + max enemies) for consistent critic input
            if scenario_name == "smax":
                # SMAX: obs_transposed has shape (num_envs, num_allies, actual_obs_dim)
                # actual_obs_dim varies with number of enemies (includes enemy features in observation)
                # Get actual observation dimension from current observations
                current_obs_dim = obs_transposed.shape[2]

                # Get total number of units in current environment
                current_total_units = (
                    smax_kwargs["num_allies"] + smax_kwargs["num_enemies"]
                )
                max_total_units = smax_kwargs["num_allies"] + config["env"].get(
                    "num_enemies", 3
                )  # Max from config

                # For SMAX, need to pad to max total units to keep critic input consistent
                # But we can't change obs_dim per unit, so instead we pad the flattened state
                # Flatten current observations
                current_global_state = obs_transposed.reshape(
                    num_envs, -1
                )  # (num_envs, current_total_units * current_obs_dim)

                # Calculate target size based on max configuration
                # We need to pad to match the size expected by the critic
                current_size = current_global_state.shape[1]
                target_size = (
                    obs_dim * max_total_units
                )  # obs_dim from initialization (with max enemies)

                if current_size < target_size:
                    padding = torch.zeros(
                        num_envs,
                        target_size - current_size,
                        device=torch_device,
                        dtype=torch.float32,
                    )
                    global_state_flat = torch.cat(
                        [current_global_state, padding], dim=1
                    )
                else:
                    global_state_flat = current_global_state

            elif mixed_agent_training:
                if per_env_agent_variation:
                    # Per-environment variation: each env has different number of active agents
                    # Strategy: pad each environment to max_agents, masking out inactive ones
                    # obs_transposed: (num_envs, env_num_agents, obs_dim)

                    # For each environment, zero out observations of inactive agents
                    agent_indices = np.arange(env_num_agents)  # (env_num_agents,)
                    # Create mask: (num_envs, env_num_agents)
                    active_mask = (
                        agent_indices[np.newaxis, :]
                        < current_num_agents_per_env[:, np.newaxis]
                    )
                    # Convert to torch and expand for obs_dim: (num_envs, env_num_agents, 1)
                    active_mask_torch = (
                        torch.from_numpy(active_mask).to(torch_device).unsqueeze(-1)
                    )

                    # Zero out inactive agents' observations
                    masked_obs = obs_transposed * active_mask_torch

                    # Flatten to create global state: (num_envs, env_num_agents * obs_dim)
                    global_state_flat = masked_obs.reshape(num_envs, -1)
                else:
                    # Per-episode variation: all environments have same number of active agents
                    # Extract only active agents: (num_envs, current_num_agents, obs_dim)
                    active_obs = obs_transposed[:, :current_num_agents, :]
                    # Pad to max_agents to maintain consistent critic input size
                    if current_num_agents < max_agents:
                        padding = torch.zeros(
                            num_envs,
                            max_agents - current_num_agents,
                            obs_dim,
                            device=torch_device,
                            dtype=torch.float32,
                        )
                        active_obs = torch.cat([active_obs, padding], dim=1)
                    global_state_flat = active_obs.reshape(
                        num_envs, -1
                    )  # (num_envs, max_agents * obs_dim)
            else:
                global_state_flat = obs_transposed.reshape(
                    num_envs, -1
                )  # (num_envs, max_agents * obs_dim)

            global_state_np = (
                global_state_flat.detach().cpu().numpy()
                if global_state_flat.requires_grad
                else global_state_flat.cpu().numpy()
            )
            jax_global_state = jnp.asarray(global_state_np)
            if use_cuda:
                jax_global_state = jax.device_put(jax_global_state, jax_device)

            # Optionally clip global state before critic
            if clip_global_state:
                jax_global_state = jnp.clip(
                    jax_global_state, clip_global_state_min, clip_global_state_max
                )

            # Get value estimates from centralized critic
            if use_gru_policy and critic_hidden_states is not None:
                # RNN critic: pass hidden states and dones
                dones_np = torch.zeros(num_envs, dtype=torch.bool, device=torch_device)
                # Check if any environment has terminated
                # Note: VMAS returns list of done flags per agent
                if step > 0:  # Use previous step's dones
                    for agent_idx in range(len(dones)):
                        dones_np = dones_np | dones[agent_idx]
                jax_dones = jnp.array(dones_np.cpu().numpy())
                critic_hidden_states, values = _get_value(
                    critic_state.params,
                    jax_global_state,
                    critic_hidden_states,
                    jax_dones,
                )
            else:
                # MLP critic: simple forward pass
                values = _get_value(critic_state.params, jax_global_state)

            # ================================================================
            # Check for environment changes and requery hypernetwork (reverse_transport)
            # ================================================================
            if (
                use_hypernetwork
                and scenario_name == "reverse_transport"
                and current_package_props is not None
            ):
                # Check if package properties have changed
                new_package_props = env.scenario.get_package_properties()
                if new_package_props != current_package_props:
                    # Package properties have changed - requery hypernetwork
                    # Create new environment context
                    env_context_vec = torch.tensor(
                        [
                            new_package_props["mass"],
                            new_package_props["width"],
                            new_package_props["length"],
                        ],
                        device=torch_device,
                        dtype=torch.float32,
                    )
                    # Broadcast to all envs and agents
                    updated_env_context = (
                        env_context_vec.unsqueeze(0)
                        .unsqueeze(0)
                        .expand(num_envs, env_num_agents, env_context_dim)
                    )
                    # Convert to JAX
                    env_context_np = (
                        updated_env_context.detach().cpu().numpy()
                        if updated_env_context.requires_grad
                        else updated_env_context.cpu().numpy()
                    )
                    jax_env_context = jnp.asarray(env_context_np)
                    if use_cuda:
                        jax_env_context = jax.device_put(jax_env_context, jax_device)

                    # Requery hypernetwork with new environment context
                    adapters_dict = _get_static_adapters(
                        hn_state.params,
                        jax_task,
                        jax_context,
                        jax_lidar,
                        jax_food_positions,
                        jax_agent_positions,
                        jax_target_snd,
                        jax_env_context,
                        jax_mask,
                        (
                            diversity_scaling_broadcast
                            if "diversity_scaling_broadcast" in locals()
                            else diversity_scaling
                        ),
                    )

                    # Update current properties
                    current_package_props = new_package_props

                    # Log the change (only once per episode to avoid spam)
                    if step == 0 or (episode * rollout_steps + step) % 100 == 0:
                        print(
                            f"  [Episode {episode}, Step {step}] Package properties changed: "
                            f"mass={new_package_props['mass']:.2f}, "
                            f"width={new_package_props['width']:.2f}, "
                            f"length={new_package_props['length']:.2f} - Requeried hypernetwork"
                        )

            # ================================================================
            # Check for pressure plate events and requery hypernetwork (pressure_plate)
            # ================================================================
            if (
                use_hypernetwork
                and adaptive_hypernetwork
                and scenario_name == "pressure_plate"
                and prev_door_open is not None
            ):
                # Get current door state from environment
                current_door_open = env.scenario.door_open  # (num_envs,)

                # Check which environments have new door openings
                newly_opened = current_door_open & ~prev_door_open

                # Count total plate activations (both left and right plates)
                # A plate is active if any robot is on it
                ground_robots = [a for a in env.agents if "ground_robot" in a.name]
                left_plate_active = torch.zeros(
                    num_envs, dtype=torch.bool, device=torch_device
                )
                right_plate_active = torch.zeros(
                    num_envs, dtype=torch.bool, device=torch_device
                )

                for robot in ground_robots:
                    # Check left plate
                    dist_left = torch.linalg.norm(
                        robot.state.pos - env.scenario.plate_left.state.pos, dim=-1
                    )
                    left_plate_active |= dist_left < (
                        env.scenario.plate_radius + robot.shape.radius + 0.05
                    )

                    # Check right plate
                    dist_right = torch.linalg.norm(
                        robot.state.pos - env.scenario.plate_right.state.pos, dim=-1
                    )
                    right_plate_active |= dist_right < (
                        env.scenario.plate_radius + robot.shape.radius + 0.05
                    )

                # Count active plates
                current_active_plates = (
                    left_plate_active.long() + right_plate_active.long()
                )  # 0, 1, or 2

                # Detect when we go from 1 to 2 plates (second plate just activated)
                second_plate_activated = (current_active_plates == 2) & (
                    plate_activation_count == 1
                )

                # Update plate activation count (keep max seen)
                plate_activation_count = torch.maximum(
                    plate_activation_count, current_active_plates
                )

                # Determine which environments need requerying
                should_requery = newly_opened | second_plate_activated

                if should_requery.any():
                    # Update environment context with current door state (if using env context)
                    if env_context_dim > 0 and jax_env_context is not None:
                        # Get configuration flags
                        env_context_door_state = config["model"].get(
                            "env_context_door_state", True
                        )

                        if env_context_door_state:
                            # Rebuild environment context with updated door state.
                            # Use per-environment positions since they differ across
                            # randomised environments.
                            env_context_plate_positions = config["model"].get(
                                "env_context_plate_positions", True
                            )
                            env_context_goal_position = config["model"].get(
                                "env_context_goal_position", True
                            )
                            use_agent_id_context = config["model"].get(
                                "use_agent_id_context", False
                            )

                            # Build per-agent relative context (positions relative to
                            # each agent's current position).
                            _ground_robots_rq = sorted(
                                [a for a in env.agents if "ground_robot" in a.name],
                                key=lambda a: a.name,
                            )
                            _agent_pos_rq = torch.stack(
                                [a.state.pos[:, :2] for a in _ground_robots_rq], dim=1
                            )  # (num_envs, env_num_agents, 2)

                            # Build per-agent context parts: each (num_envs, env_num_agents, d)
                            env_context_parts = []

                            if env_context_plate_positions:
                                left_plate_pos = env.scenario.plate_left.state.pos[
                                    :, :2
                                ]  # (num_envs, 2)
                                right_plate_pos = env.scenario.plate_right.state.pos[
                                    :, :2
                                ]  # (num_envs, 2)
                                left_rel = left_plate_pos.unsqueeze(1) - _agent_pos_rq
                                right_rel = right_plate_pos.unsqueeze(1) - _agent_pos_rq
                                env_context_parts.extend([left_rel, right_rel])

                            if env_context_door_state:
                                # Use CURRENT per-env door state (this is what changed!)
                                door_open_exp = current_door_open.float()[
                                    :, None, None
                                ].expand(
                                    num_envs, env_num_agents, 1
                                )  # (num_envs, env_num_agents, 1)
                                env_context_parts.append(door_open_exp)

                            if env_context_goal_position:
                                goal_pos = env.scenario.goal.state.pos[
                                    :, :2
                                ]  # (num_envs, 2)
                                goal_rel = goal_pos.unsqueeze(1) - _agent_pos_rq
                                env_context_parts.append(goal_rel)

                            if use_agent_id_context:
                                # Add one-hot agent IDs for role differentiation
                                # Shape: (num_envs, env_num_agents, max_agents)
                                agent_ids_onehot = torch.zeros(
                                    num_envs,
                                    env_num_agents,
                                    max_agents,
                                    device=torch_device,
                                )
                                for i in range(env_num_agents):
                                    agent_ids_onehot[:, i, i] = 1.0
                                env_context_parts.append(agent_ids_onehot)

                            # (num_envs, env_num_agents, env_context_dim)
                            updated_env_context = torch.cat(env_context_parts, dim=-1)

                            # Convert to JAX
                            env_context_np = (
                                updated_env_context.detach().cpu().numpy()
                                if updated_env_context.requires_grad
                                else updated_env_context.cpu().numpy()
                            )
                            jax_env_context = jnp.asarray(env_context_np)
                            if use_cuda:
                                jax_env_context = jax.device_put(
                                    jax_env_context, jax_device
                                )

                    # Requery hypernetwork ONLY for agents in affected environments
                    # Get indices of agents that need requerying
                    # should_requery is (num_envs,), we need to expand to all agents in those envs
                    requery_env_mask = should_requery.cpu().numpy()  # (num_envs,)

                    # Create flat mask for all agents: (num_envs * env_num_agents,)
                    requery_flat_mask = np.repeat(requery_env_mask, env_num_agents)
                    requery_indices = np.where(requery_flat_mask)[0]
                    num_agents_to_requery = len(requery_indices)

                    if num_agents_to_requery > 0:
                        # Extract subset of agents that need requerying
                        # Reshape from (num_envs, env_num_agents, ...) to (batch_size, ...)

                        # Handle task and context - may be None for scenarios without capability context
                        jax_task_subset = None
                        jax_context_subset = None
                        if jax_task is not None:
                            jax_task_flat = jax_task.reshape(batch_size, -1)
                            jax_task_subset = jax_task_flat[requery_indices]
                            jax_task_subset = jax_task_subset[:, None, :]

                        if jax_context is not None:
                            jax_context_flat = jax_context.reshape(batch_size, -1)
                            jax_context_subset = jax_context_flat[requery_indices]
                            jax_context_subset = jax_context_subset[:, None, :]

                        # Extract environment context subset if available
                        jax_env_context_subset = None
                        if env_context_dim > 0 and jax_env_context is not None:
                            jax_env_context_flat = jax_env_context.reshape(
                                batch_size, -1
                            )
                            jax_env_context_subset = jax_env_context_flat[
                                requery_indices
                            ]
                            jax_env_context_subset = jax_env_context_subset[:, None, :]

                        # Create mask for single-agent requery (no cross-agent attention needed)
                        jax_mask_subset = None

                        # Requery hypernetwork ONLY for affected agents
                        new_adapters_subset = _get_static_adapters(
                            hn_state.params,
                            jax_task_subset,
                            jax_context_subset,
                            None,  # jax_lidar not used in pressure_plate
                            None,  # jax_food_positions not used
                            None,  # jax_agent_positions not used
                            None,  # jax_target_snd not used with single-agent requery
                            jax_env_context_subset,
                            jax_mask_subset,
                            1.0,  # diversity_scaling: use neutral scaling for requery
                        )

                        # Update adapters ONLY for agents in affected environments
                        for key in adapters_dict.keys():
                            adapters_dict[key] = (
                                adapters_dict[key]
                                .at[requery_indices]
                                .set(new_adapters_subset[key])
                            )

                    # Update door opened count for logging
                    door_opened_count = door_opened_count + newly_opened.long()

                    # Log the requery event
                    num_door_opens = newly_opened.sum().item()
                    num_second_plates = second_plate_activated.sum().item()
                    num_affected_envs = should_requery.sum().item()

                    if num_door_opens > 0 or num_second_plates > 0:
                        events = []
                        if num_door_opens > 0:
                            events.append(f"door opened (x{num_door_opens} envs)")
                        if num_second_plates > 0:
                            events.append(
                                f"2nd plate activated (x{num_second_plates} envs)"
                            )

                        if (
                            episode % 10 == 0 or episode < 5
                        ):  # Log occasionally to avoid spam
                            print(
                                f"  [Episode {episode}, Step {step}] Pressure plate event: {', '.join(events)} - Requeried {num_agents_to_requery}/{batch_size} agents ({100*num_agents_to_requery/batch_size:.1f}%) in {num_affected_envs} envs"
                            )

                # Update previous door state for next step
                prev_door_open = current_door_open.clone()

            # ================================================================
            # Run Policy (JAX) - Get actions and log probs using adapters or DiCo
            # ================================================================
            jax_rng, action_rng = jax.random.split(jax_rng)

            # For SMAX with discrete actions, we need to handle differently
            if scenario_name == "smax":
                # SMAX uses discrete actions - policy outputs logits
                # We'll sample discrete actions and compute log probs

                # CRITICAL: Get action masks from environment - vectorized across all envs
                def get_masks_single_env(state):
                    masks_dict = env.get_avail_actions(state)
                    # Stack masks for all agents: (num_agents, num_actions)
                    return jnp.stack(
                        [masks_dict[agent] for agent in env.agents], axis=0
                    )

                # Vmap across num_envs
                vmapped_get_masks = jax.vmap(get_masks_single_env)
                action_masks_jax = vmapped_get_masks(
                    env_state
                )  # (num_envs, num_agents, num_actions)

                action_masks_flat = action_masks_jax.reshape(
                    -1, action_masks_jax.shape[-1]
                )  # (batch_size, num_actions)

                # For SMAX with curriculum, pad action masks to match policy output
                if action_masks_flat.shape[1] < max_action_dim:
                    padding_size = max_action_dim - action_masks_flat.shape[1]
                    # Pad with False (invalid actions)
                    action_masks_flat = jnp.concatenate(
                        [
                            action_masks_flat,
                            jnp.zeros(
                                (action_masks_flat.shape[0], padding_size),
                                dtype=action_masks_flat.dtype,
                            ),
                        ],
                        axis=1,
                    )

                if use_dico:
                    agent_ids_np = np.tile(np.arange(env_num_agents), num_envs)
                    agent_ids_jax = jnp.array(agent_ids_np)
                    logits_flat, _ = shared_policy.apply(
                        {"params": policy_state.params},
                        jax_obs,
                        agent_ids_jax,
                        diversity_scaling=diversity_scaling,
                    )
                    # DiCo doesn't use GRU, so no hidden state updates
                    new_hidden_states = None
                else:
                    # HyperLoRA path - may use GRU with proper hidden state management
                    if use_gru_policy:
                        # New GRULoRAPolicy signature: (hidden, x, adapters)
                        # where x = (obs, dones, avail_actions)
                        # ScannedRNN expects inputs with time dimension: (time, batch, ...)
                        # During rollout, we process one timestep at a time, so time=1
                        obs_seq = jax_obs[None, ...]  # (1, batch_size, obs_dim)
                        # CRITICAL: Pass actual previous dones so GRU resets on episode boundaries
                        # This must match what training does (using actual dones_sequence)
                        dones_seq = prev_dones_flat[None, ...]  # (1, batch_size)
                        avail_actions_seq = action_masks_flat[
                            None, ...
                        ]  # (1, batch_size, action_dim)

                        # Pack inputs as tuple
                        policy_x = (obs_seq, dones_seq, avail_actions_seq)
                        new_hidden_states, logits_seq = shared_policy.apply(
                            {"params": policy_state.params},
                            gru_hidden_states,
                            policy_x,
                            adapters_dict,
                        )
                        # Remove time dimension from output: (1, batch_size, action_dim) -> (batch_size, action_dim)
                        logits_flat = logits_seq[0]
                    else:
                        logits_flat, _ = shared_policy.apply(
                            {"params": policy_state.params},
                            jax_obs,
                            adapters_dict,
                        )
                        new_hidden_states = None

                # Update hidden states for next timestep
                if use_gru_policy and new_hidden_states is not None:
                    gru_hidden_states = new_hidden_states

                # CRITICAL: Apply action masks by setting invalid actions to -inf
                # This prevents sampling of invalid actions
                masked_logits = jnp.where(
                    action_masks_flat.astype(bool),
                    logits_flat,
                    jnp.full_like(
                        logits_flat, -1e10
                    ),  # Large negative number for invalid actions
                )

                # Sample discrete actions from masked categorical distribution
                dist = distrax.Categorical(logits=masked_logits)
                jax_actions_flat = dist.sample(seed=action_rng)
                log_probs_flat = dist.log_prob(jax_actions_flat)

                # Debug: Print action distribution for first few steps
                if step < 3 and episode % 100 == 0:
                    print(f"  [Step {step}] Action sampling debug:")
                    print(
                        f"    Action masks shape: {action_masks_flat.shape}, sample: {action_masks_flat[0]}"
                    )
                    print(
                        f"    Logits shape: {logits_flat.shape}, sample: {logits_flat[0]}"
                    )
                    print(
                        f"    Actions sampled: {jax_actions_flat[:6]} (first 6 agents)"
                    )
                    print(
                        f"    Action value counts: {jnp.bincount(jax_actions_flat, length=8)}"
                    )

                # Debug: Check when attack actions become available
                if episode % 100 == 0 and step % 10 == 0:
                    # Count how many agents have any attack action available (actions 5+)
                    attack_masks = action_masks_flat[
                        :, 5:
                    ]  # Attack actions start at index 5
                    agents_can_attack = jnp.any(attack_masks, axis=1).sum()
                    if agents_can_attack > 0:
                        print(
                            f"  [Step {step}] {agents_can_attack}/{action_masks_flat.shape[0]} agents can attack"
                        )

            # For DiCo: create agent_ids array for routing
            elif use_dico:
                agent_ids_np = np.tile(np.arange(env_num_agents), num_envs)
                agent_ids_jax = jnp.array(agent_ids_np)

                # NaN checks force a device->host sync per step (~300/episode).
                # Gated behind --debug-nan.
                if args.debug_nan and jnp.isnan(jax_obs).any():
                    print(
                        f"WARNING: NaN in observations before policy at step {step}, episode {episode}"
                    )
                    print(f"  NaN count: {jnp.isnan(jax_obs).sum()} / {jax_obs.size}")

                jax_actions_flat, log_probs_flat, _ = _get_actions_and_log_probs(
                    policy_state.params,
                    jax_obs,
                    None,
                    action_rng,
                    agent_ids=agent_ids_jax,
                    diversity_scaling=diversity_scaling,
                )

                # Diagnostic: check policy outputs (gated)
                if args.debug_nan and (
                    jnp.isnan(jax_actions_flat).any() or jnp.isnan(log_probs_flat).any()
                ):
                    print(
                        f"WARNING: NaN in policy output at step {step}, episode {episode}"
                    )
                    print(
                        f"  Actions NaN: {jnp.isnan(jax_actions_flat).sum()} / {jax_actions_flat.size}"
                    )
                    print(
                        f"  Log probs NaN: {jnp.isnan(log_probs_flat).sum()} / {log_probs_flat.size}"
                    )
                    print(f"  Diversity scaling: {diversity_scaling}")

                    # Check policy parameter norms
                    param_norms = {
                        k: float(jnp.sqrt(jnp.sum(v**2)))
                        for k, v in jax.tree_util.tree_leaves_with_path(
                            policy_state.params
                        )[
                            :5
                        ]  # First 5 params
                    }
                    print(f"  Sample param norms: {param_norms}")
            elif use_cash:
                jax_actions_flat, log_probs_flat, new_hidden, mean_flat, std_flat = (
                    _get_actions_and_log_probs(
                        policy_state.params,
                        jax_obs,
                        None,
                        action_rng,
                        hidden_state=gru_hidden_states,
                        dones_batch=prev_dones_flat,
                        cash_capability_context_batch=jax_cash_context_flat,
                    )
                )
                if new_hidden is not None:
                    gru_hidden_states = new_hidden
            else:
                # HyperLoRA path
                jax_actions_flat, log_probs_flat, new_hidden, mean_flat, std_flat = (
                    _get_actions_and_log_probs(
                        policy_state.params,
                        jax_obs,
                        adapters_dict,
                        action_rng,
                        hidden_state=gru_hidden_states if use_gru_policy else None,
                        dones_batch=(
                            prev_dones_flat if use_gru_policy else None
                        ),  # CRITICAL FIX: Pass actual dones for GRU reset!
                    )
                )

                # Update hidden states for GRU
                if use_gru_policy and new_hidden is not None:
                    gru_hidden_states = new_hidden

                # ============================================================
                # Track Adapter Effect: Compare policy WITH vs WITHOUT adapters
                # ============================================================
                if use_hypernetwork:
                    # Get policy mean WITHOUT adapters (baseline backbone only)
                    mean_no_adapters = _get_policy_mean_without_adapters(
                        policy_state.params,
                        jax_obs,
                        hidden_state=gru_hidden_states if use_gru_policy else None,
                        dones_batch=(prev_dones_flat if use_gru_policy else None),
                    )

                    # Compute raw adapter impacts (preserving sign and magnitude per action dim)
                    # Shape: (batch_size, action_dim)
                    adapter_impact_raw = mean_flat - mean_no_adapters

                    # Also compute L2 norm for backward compatibility
                    # Shape: (batch_size, action_dim) -> (batch_size,)
                    adapter_effect_per_agent = jnp.linalg.norm(
                        adapter_impact_raw, axis=-1
                    )

                    # Store both raw impacts and norms for logging
                    adapter_effect_buffer.append(
                        adapter_impact_raw
                    )  # Raw values with sign
                    adapter_effect_norms_buffer.append(
                        adapter_effect_per_agent
                    )  # Norms

                    # Store action distributions for plotting
                    backbone_actions_buffer.append(mean_no_adapters)  # Backbone only
                    combined_actions_buffer.append(mean_flat)  # Backbone + adapters

            # ================================================================
            # WRAPPER: JAX -> PyTorch (for actions)
            # ================================================================
            actions_flat = np.asarray(jax_actions_flat)

            # Check for NaN actions and replace with zeros
            if np.isnan(actions_flat).any():
                print(
                    f"WARNING: NaN detected in actions at step {step}, episode {episode}"
                )
                print(
                    f"  NaN count: {np.isnan(actions_flat).sum()} / {actions_flat.size}"
                )
                print(f"  Replacing NaN with zeros")
                actions_flat = np.nan_to_num(
                    actions_flat, nan=0.0, posinf=0.0, neginf=0.0
                )

            # Reshape to (num_envs, env_num_agents, action_dim)
            actions_reshaped = actions_flat.reshape(num_envs, env_num_agents, -1)

            # CRITICAL: For mixed training, zero out actions for inactive agents
            # This prevents the environment from processing meaningless actions
            if mixed_agent_training:
                if per_env_agent_variation:
                    # Per-environment variation: different agent count per environment
                    # Create mask: (num_envs, env_num_agents)
                    # Each environment has its own active agent count
                    agent_indices = np.arange(env_num_agents)  # (env_num_agents,)
                    # Broadcast comparison: (num_envs, 1) vs (env_num_agents,) -> (num_envs, env_num_agents)
                    agent_mask = (
                        agent_indices[np.newaxis, :]
                        < current_num_agents_per_env[:, np.newaxis]
                    )
                    # Expand dims for action_dim: (num_envs, env_num_agents, 1)
                    agent_mask_expanded = agent_mask[:, :, np.newaxis]
                else:
                    # Per-episode variation: all environments have same agent count
                    # Create agent mask: (env_num_agents,)
                    agent_mask = np.arange(env_num_agents) < current_num_agents
                    # Expand dims to broadcast: (1, env_num_agents, 1)
                    agent_mask_expanded = agent_mask[np.newaxis, :, np.newaxis]
                # Zero out inactive agents' actions
                actions_reshaped = actions_reshaped * agent_mask_expanded

            # ================================================================
            # Environment Step (PyTorch for VMAS, JAX for SMAX)
            # ================================================================
            if scenario_name == "smax":
                # SMAX uses JAX and discrete actions
                # Convert discrete action indices to dict format
                # For SMAX, actions are discrete integers (not continuous vectors)
                step_key = jax.random.fold_in(episode_key, step)

                # Actions should be integers, not floats
                if actions_reshaped.shape[-1] == 1:
                    # Already single dimension, just squeeze
                    actions_int = actions_reshaped.squeeze(-1).astype(np.int32)
                else:
                    # Take first dimension (shouldn't happen for discrete)
                    actions_int = actions_reshaped[:, :, 0].astype(np.int32)

                # SMAX doesn't vectorize environments - we need to step each env separately
                # Vectorize using JAX vmap for parallel execution
                step_keys = jax.random.split(
                    jax.random.PRNGKey(seed + episode * 1000 + step), num_envs
                )

                # actions_int shape: (num_envs, env_num_agents)
                # Convert to list of action dicts for each environment
                def step_single_env(key, state, actions):
                    # Don't use int() on JAX arrays - keep as JAX integers for vmap compatibility
                    actions_dict = {
                        agent: actions[i] for i, agent in enumerate(env.agents)
                    }
                    return env.step_env(key, state, actions_dict)

                # Vectorize across environments
                vmapped_step = jax.vmap(step_single_env)
                next_obs_dicts, env_states, rewards_dicts, dones_dicts, _ = (
                    vmapped_step(step_keys, env_state, actions_int)
                )

                # Update environment state
                env_state = env_states

                # Convert observations: dict of {agent: (num_envs, obs_dim)} -> list of (num_envs, obs_dim)
                # OPTIMIZED: Stack first, then convert once instead of per-agent conversion
                next_obs_array = jnp.stack(
                    [next_obs_dicts[agent] for agent in agents_list], axis=1
                )  # (num_envs, num_agents, obs_dim)
                next_obs_np = np.asarray(next_obs_array)  # Single JAX->NumPy conversion
                next_obs_torch = torch.from_numpy(next_obs_np).float().to(torch_device)
                next_obs = [
                    next_obs_torch[:, i, :] for i in range(env_num_agents)
                ]  # Split to list (no conversion, just views)

                # Convert rewards and dones (JAX arrays to PyTorch)
                # rewards_dicts: dict of {agent: (num_envs,)}
                # dones_dicts: dict of {agent: (num_envs,)} + {"__all__": (num_envs,)}

                # Debug: Print rewards for first few steps
                if step < 5 and episode % 100 == 0:
                    rewards_sample = {
                        k: np.array(v)[0] for k, v in rewards_dicts.items()
                    }
                    print(
                        f"  [Step {step}] SMAX rewards sample (env 0): {rewards_sample}"
                    )

                # Debug: Print summary stats periodically to catch any non-zero rewards
                if episode % 100 == 0 and step % 20 == 0:
                    total_reward = sum(
                        float(np.array(rewards_dicts[agent]).mean())
                        for agent in env.agents
                    )
                    if total_reward != 0.0:
                        print(
                            f"  [Step {step}] NON-ZERO REWARD! Mean total: {total_reward}"
                        )
                        # Also print action mask and actual actions taken
                        print(f"    Action mask sample (env 0): {action_masks_flat[0]}")
                        print(
                            f"    Actions taken: {jax_actions_flat[:12]} (first 12 agents)"
                        )
                        print(
                            f"    Action distribution: {jnp.bincount(jax_actions_flat, length=8)}"
                        )

                # Convert rewards: dict of {agent: (num_envs,)} -> list of (num_envs,) tensors
                # OPTIMIZED: Stack first, then convert once
                rewards_array = jnp.stack(
                    [rewards_dicts[agent] for agent in agents_list], axis=1
                )  # (num_envs, num_agents)
                rewards_np = np.asarray(rewards_array)  # Single conversion
                rewards_torch = torch.from_numpy(rewards_np).float().to(torch_device)
                rewards = [
                    rewards_torch[:, i] for i in range(env_num_agents)
                ]  # Views, not copies

                # CRITICAL: For SMAX, use the episode-level "__all__" done flag
                # dones_dicts["__all__"] has shape (num_envs,)
                episode_dones = np.array(
                    dones_dicts.get("__all__", jnp.zeros(num_envs))
                )
                dones = [
                    torch.from_numpy(episode_dones).bool().to(torch_device)
                    for agent in env.agents
                ]
                info = {}

                # Reset individual environments that are done
                if episode_dones.any():
                    num_done = int(episode_dones.sum())
                    if episode % 100 == 0:  # Log occasionally
                        print(
                            f"  [Episode {episode}, Step {step}] Resetting {num_done}/{num_envs} environments that finished"
                        )

                    # Generate reset keys for all environments
                    reset_keys = jax.random.split(
                        jax.random.PRNGKey(seed + episode * 10000 + step * 100),
                        num_envs,
                    )

                    # Create reset function that conditionally resets a single environment
                    def reset_if_done(reset_key, state, obs_dict, done):
                        def do_reset():
                            new_obs_dict, new_state = env.reset(reset_key)
                            return new_obs_dict, new_state

                        def no_reset():
                            return obs_dict, state

                        return jax.lax.cond(done, do_reset, no_reset)

                    # Vmap over environments - vmap automatically slices arguments along axis 0
                    vmapped_reset_if_done = jax.vmap(reset_if_done)
                    next_obs_dicts, env_state = vmapped_reset_if_done(
                        reset_keys, env_state, next_obs_dicts, dones_dicts["__all__"]
                    )

                    # Update next_obs with potentially reset observations
                    # OPTIMIZED: Stack first, then convert once
                    next_obs_array = jnp.stack(
                        [next_obs_dicts[agent] for agent in agents_list], axis=1
                    )
                    next_obs_np = np.asarray(next_obs_array)
                    next_obs_torch = (
                        torch.from_numpy(next_obs_np).float().to(torch_device)
                    )
                    next_obs = [next_obs_torch[:, i, :] for i in range(env_num_agents)]

            else:
                # Convert to list of tensors for VMAS
                # VMAS expects: [agent0_actions, agent1_actions, ...]
                # Each with shape: (num_envs, action_dim)
                # Send actions for ALL agents in environment (env_num_agents)
                # For mixed training: only first current_num_agents contribute to learning
                torch_actions = []
                for agent_idx in range(env_num_agents):
                    agent_actions = torch.tensor(
                        actions_reshaped[:, agent_idx, :],
                        device=torch_device,
                        dtype=torch.float32,
                    )
                    torch_actions.append(agent_actions)

                # Environment Step (PyTorch/VMAS)
                next_obs, rewards, dones, info = env.step(torch_actions)

            # ================================================================
            # CRITICAL: Store trajectory data IMMEDIATELY after action is taken
            # Store the obs, context, lidar that were used to generate THIS action
            # Do this BEFORE the requery check (which updates context/lidar for NEXT step)
            # ================================================================
            trajectory_data["obs"].append(jax_obs)
            trajectory_data["global_states"].append(jax_global_state)
            trajectory_data["actions"].append(jax_actions_flat)
            # Store CURRENT context and lidar that were used for this action
            trajectory_data["context"].append(
                jax_context.reshape(batch_size, -1) if jax_context is not None else None
            )
            trajectory_data["cash_context"].append(jax_cash_context_flat)
            if jax_lidar is not None:
                trajectory_data["lidar"].append(jax_lidar.reshape(batch_size, -1))
                # Debug: Print lidar sample at specific steps
                if episode % log_interval == 0 and step in [0, 50, 100, 150]:
                    lidar_sample = jax_lidar.reshape(batch_size, -1)[0]  # First agent
                    print(
                        f"  [DEBUG Step {step}] Storing lidar for agent 0 (BEFORE requery): {lidar_sample[:5]}..."
                    )
            else:
                trajectory_data["lidar"].append(None)

            # Store food positions if available
            if jax_food_positions is not None:
                trajectory_data["food_positions"].append(
                    jax_food_positions.reshape(batch_size, -1)
                )
            else:
                trajectory_data["food_positions"].append(None)

            # Store agent positions if available (dispersion_vmas only)
            if jax_agent_positions is not None:
                trajectory_data["agent_positions"].append(
                    jax_agent_positions.reshape(batch_size, -1)
                )
            else:
                trajectory_data["agent_positions"].append(None)

            # Store environment context if available (reverse_transport only)
            if jax_env_context is not None:
                trajectory_data["env_context"].append(
                    jax_env_context.reshape(batch_size, -1)
                )
            else:
                trajectory_data["env_context"].append(None)

            # Reset GRU hidden states for environments that are done
            # OPTIMIZED: Vectorized reset using JAX masking instead of Python loop
            if use_gru_policy:
                if isinstance(dones, torch.Tensor):
                    dones_array = dones.cpu().numpy()
                else:
                    dones_array = np.array(dones)

                # Flatten if multi-dimensional (some envs return per-agent dones)
                if len(dones_array.shape) > 1:
                    # Use any() along agent dimension to get per-env done status
                    dones_per_env = dones_array.any(axis=-1)
                else:
                    dones_per_env = dones_array

                if dones_per_env.any():
                    # VECTORIZED: Create mask for all agents in done environments
                    # Shape: (num_envs,) -> (num_envs, env_num_agents) -> (batch_size,)
                    actual_num_envs = min(num_envs, len(dones_per_env))
                    dones_mask = jnp.array(
                        dones_per_env[:actual_num_envs]
                    )  # (actual_num_envs,)
                    # Repeat for each agent in environment: (actual_num_envs, env_num_agents)
                    dones_mask_agents = jnp.repeat(dones_mask, env_num_agents)
                    # Pad if needed
                    if actual_num_envs * env_num_agents < gru_hidden_states.shape[0]:
                        padding = jnp.zeros(
                            gru_hidden_states.shape[0] - len(dones_mask_agents),
                            dtype=bool,
                        )
                        dones_mask_agents = jnp.concatenate(
                            [dones_mask_agents, padding]
                        )
                    # Reset: keep old values where not done, set to 0 where done
                    gru_hidden_states = jnp.where(
                        dones_mask_agents[:, None],  # Broadcast over hidden_dim
                        0.0,
                        gru_hidden_states,
                    )

            # CRITICAL: For mixed training, zero out padded agents' rewards\n            # Padded agents (beyond current_num_agents) shouldn't contribute to learning
            if mixed_agent_training:
                for agent_idx in range(current_num_agents, env_num_agents):
                    rewards[agent_idx] = torch.zeros_like(rewards[agent_idx])

            # No padding needed - environment always returns max_agents observations

            # ================================================================
            # Check if Food Entered Lidar Range or Landmark Reached and Requery Hypernetwork
            # ================================================================
            # NOTE: dispersion_vmas has global observability, so no adaptive requerying needed
            if (
                use_hypernetwork
                and adaptive_hypernetwork
                and scenario_name not in ["smax", "football", "dispersion_vmas"]
            ):
                # Different detection logic based on scenario
                if scenario_name in ["sampling", "dispersion"]:
                    # Detect which agents have food in lidar range now
                    current_food_in_range = detect_food_in_range(
                        next_obs, num_envs, max_agents
                    )

                    # Find agents that just detected food for the FIRST time
                    # (food is in range now AND wasn't detected before)
                    newly_detected = current_food_in_range & ~food_detected_already
                elif scenario_name == "grassland":
                    # Detect which good agents reached a landmark this step
                    # This happens when they receive +20 reward
                    newly_detected = detect_landmark_reached(
                        rewards, num_envs, env_num_agents, num_adversaries
                    )
                else:
                    # No adaptive requerying for other scenarios
                    newly_detected = torch.zeros(
                        num_envs, env_num_agents, dtype=torch.bool, device=torch_device
                    )

                # If any agent newly detected food/landmark, requery hypernetwork for ONLY those agents
                if newly_detected.any():
                    # Flatten newly_detected to (batch_size,) for masking
                    newly_detected_flat = newly_detected.reshape(-1).cpu().numpy()

                    # Get indices of agents that need requerying
                    requery_indices = np.where(newly_detected_flat)[0]
                    num_requeried = len(requery_indices)

                    # For requerying, we need to extract the subset but maintain proper shape
                    # Original shapes: (num_envs, num_agents, feature_dim)
                    # When flattened: (num_envs * num_agents, feature_dim)
                    # We need to reshape subset to (num_requeried, 1, feature_dim) for hypernetwork

                    # Flatten and extract (handle None for task and context)
                    if jax_task is not None:
                        jax_task_flat = jax_task.reshape(batch_size, -1)
                        jax_task_subset = jax_task_flat[requery_indices]
                        jax_task_subset = jax_task_subset[:, None, :]
                    else:
                        jax_task_subset = None

                    if jax_context is not None:
                        jax_context_flat = jax_context.reshape(batch_size, -1)
                        jax_context_subset = jax_context_flat[requery_indices]
                        jax_context_subset = jax_context_subset[:, None, :]
                    else:
                        jax_context_subset = None

                    # Extract target_snd subset if present
                    if jax_target_snd is not None:
                        jax_target_snd_flat = jax_target_snd.reshape(batch_size, -1)
                        jax_target_snd_subset = jax_target_snd_flat[requery_indices]
                        jax_target_snd_subset = jax_target_snd_subset[:, None, :]
                    else:
                        jax_target_snd_subset = None

                    # For grassland, update context with new observations after landmark reached
                    if scenario_name == "grassland":
                        # Extract updated observations for agents that reached landmarks
                        next_obs_stacked = torch.stack(next_obs, dim=0)
                        next_obs_transposed = next_obs_stacked.transpose(0, 1)
                        next_obs_flat = next_obs_transposed.reshape(batch_size, -1)

                        # Convert to numpy and JAX
                        next_obs_np = (
                            next_obs_flat.detach().cpu().numpy()
                            if next_obs_flat.requires_grad
                            else next_obs_flat.cpu().numpy()
                        )
                        next_obs_np = np.nan_to_num(
                            next_obs_np, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        jax_next_obs = jnp.asarray(next_obs_np)

                        if use_cuda:
                            jax_next_obs = jax.device_put(jax_next_obs, jax_device)

                        # Extract subset and reshape
                        jax_context_subset = jax_next_obs[requery_indices]
                        jax_context_subset = jax_context_subset[:, None, :]

                    if use_lidar_context:
                        # Extract updated lidar readings from next_obs for detected agents
                        updated_lidar_list = [
                            agent_obs[:, -lidar_dim:] for agent_obs in next_obs
                        ]
                        updated_lidar = torch.stack(
                            updated_lidar_list, dim=0
                        ).transpose(0, 1)
                        updated_lidar_flat = updated_lidar.reshape(batch_size, -1)

                        # Convert to numpy and JAX
                        updated_lidar_np = updated_lidar_flat.cpu().numpy()
                        updated_lidar_np = np.nan_to_num(
                            updated_lidar_np, posinf=1.0, neginf=0.0
                        )
                        updated_lidar_np = np.clip(updated_lidar_np, -1.0, 1.0)
                        jax_lidar_updated = jnp.asarray(updated_lidar_np)

                        if use_cuda:
                            jax_lidar_updated = jax.device_put(
                                jax_lidar_updated, jax_device
                            )

                        # Extract ONLY subset for requerying
                        jax_lidar_subset = jax_lidar_updated[requery_indices]
                        # Reshape to (num_requeried, 1, lidar_dim)
                        jax_lidar_subset = jax_lidar_subset[:, None, :]
                    else:
                        jax_lidar_subset = None

                    # Extract food positions for requerying if applicable
                    if jax_food_positions is not None:
                        # Extract updated food positions from next_obs for detected agents (dispersion_vmas only)
                        updated_food_positions = extract_food_positions(
                            next_obs, env_num_agents, scenario_name
                        )
                        # Validate shape
                        if (
                            updated_food_positions is not None
                            and updated_food_positions.shape
                            != (num_envs, env_num_agents, food_position_dim)
                        ):
                            updated_food_positions = None

                        # Extract updated agent positions from next_obs (dispersion_vmas only)
                        updated_agent_positions = extract_agent_positions(
                            next_obs, env_num_agents, scenario_name
                        )
                        # Validate shape
                        if (
                            updated_agent_positions is not None
                            and updated_agent_positions.shape
                            != (num_envs, env_num_agents, 2)
                        ):
                            updated_agent_positions = None

                        if updated_food_positions is not None:
                            # Convert to JAX
                            updated_food_pos_np = (
                                updated_food_positions.detach().cpu().numpy()
                                if updated_food_positions.requires_grad
                                else updated_food_positions.cpu().numpy()
                            )
                            jax_food_pos_updated = jnp.asarray(updated_food_pos_np)

                            if use_cuda:
                                jax_food_pos_updated = jax.device_put(
                                    jax_food_pos_updated, jax_device
                                )

                            # Flatten and extract subset for requerying
                            jax_food_pos_flat = jax_food_pos_updated.reshape(
                                batch_size, -1
                            )
                            jax_food_positions_subset = jax_food_pos_flat[
                                requery_indices
                            ]
                            # Reshape to (num_requeried, 1, 2)
                            jax_food_positions_subset = jax_food_positions_subset[
                                :, None, :
                            ]
                        else:
                            jax_food_positions_subset = None

                        if updated_agent_positions is not None:
                            # Convert to JAX
                            updated_agent_pos_np = (
                                updated_agent_positions.detach().cpu().numpy()
                                if updated_agent_positions.requires_grad
                                else updated_agent_positions.cpu().numpy()
                            )
                            jax_agent_pos_updated = jnp.asarray(updated_agent_pos_np)

                            if use_cuda:
                                jax_agent_pos_updated = jax.device_put(
                                    jax_agent_pos_updated, jax_device
                                )

                            # Flatten and extract subset for requerying
                            jax_agent_pos_flat = jax_agent_pos_updated.reshape(
                                batch_size, -1
                            )
                            jax_agent_positions_subset = jax_agent_pos_flat[
                                requery_indices
                            ]
                            # Reshape to (num_requeried, 1, 2)
                            jax_agent_positions_subset = jax_agent_positions_subset[
                                :, None, :
                            ]
                        else:
                            jax_agent_positions_subset = None
                    else:
                        jax_food_positions_subset = None
                        jax_agent_positions_subset = None

                    # Create mask for single-agent requery (no cross-agent attention needed)
                    jax_mask_subset = None  # Single agent per batch, no masking needed

                    # Extract env_context subset if present
                    if jax_env_context is not None:
                        jax_env_context_flat = jax_env_context.reshape(batch_size, -1)
                        jax_env_context_subset = jax_env_context_flat[requery_indices]
                        jax_env_context_subset = jax_env_context_subset[:, None, :]
                    else:
                        jax_env_context_subset = None

                    # Requery hypernetwork ONLY for agents that detected food
                    new_adapters_subset = _get_static_adapters(
                        hn_state.params,
                        jax_task_subset,
                        jax_context_subset,
                        jax_lidar_subset,
                        jax_food_positions_subset,
                        jax_agent_positions_subset,
                        jax_target_snd_subset,
                        jax_env_context_subset,
                        jax_mask_subset,
                        diversity_scaling,
                    )

                    # Update adapters ONLY for agents that newly detected food
                    for key in adapters_dict.keys():
                        # Use JAX's .at[] indexing to update only specific rows
                        adapters_dict[key] = (
                            adapters_dict[key]
                            .at[requery_indices]
                            .set(new_adapters_subset[key])
                        )

                    # CRITICAL: Update context/lidar to reflect what was actually used
                    # This ensures future timesteps in this rollout use updated context
                    # Flatten contexts to (batch_size, feature_dim) (handle None case)
                    if jax_context is not None and jax_context_subset is not None:
                        jax_context_flat = jax_context.reshape(batch_size, -1)
                        jax_context_flat = jax_context_flat.at[requery_indices].set(
                            jax_context_subset.reshape(num_requeried, -1)
                        )
                        jax_context = jax_context_flat.reshape(
                            num_envs, env_num_agents, -1
                        )

                    if use_lidar_context and jax_lidar_subset is not None:
                        jax_lidar_flat = jax_lidar.reshape(batch_size, -1)
                        jax_lidar_flat = jax_lidar_flat.at[requery_indices].set(
                            jax_lidar_subset.reshape(num_requeried, -1)
                        )
                        jax_lidar = jax_lidar_flat.reshape(num_envs, env_num_agents, -1)

                    # ================================================================
                    # Add requeried agents as NEW entries to buffer
                    # ================================================================
                    if use_adapter_snd and adapter_snd_buffer is not None:
                        # Add one entry per requeried agent
                        # CRITICAL: requery_indices are flat indices from batch_size = num_envs * env_num_agents
                        # We must use env_num_agents for conversion, NOT current_num_agents
                        for idx in requery_indices:
                            # Convert flat index to (env_idx, agent_idx)
                            # Since idx comes from a batch structured as (num_envs, env_num_agents)
                            env_idx = idx // env_num_agents
                            agent_idx = idx % env_num_agents

                            # For GRU: extract this agent's current hidden state
                            # flat_idx = env_idx * env_num_agents + agent_idx = idx (by construction)
                            if use_gru_policy:
                                # Access hidden state directly using original flat index
                                agent_hidden = gru_hidden_states[idx]  # (hidden_dim,)
                            else:
                                agent_hidden = None

                            entry = {
                                "task": (
                                    jax_task[env_idx][agent_idx]
                                    if jax_task is not None
                                    else None
                                ),  # (task_dim,)
                                "context": (
                                    jax_context[env_idx][agent_idx]
                                    if jax_context is not None
                                    else None
                                ),  # (context_dim,) - updated context
                                "lidar": (
                                    jax_lidar[env_idx][agent_idx]
                                    if jax_lidar is not None
                                    else None
                                ),  # (lidar_dim,) - updated
                                "target_snd": (
                                    jax_target_snd[env_idx][agent_idx]
                                    if jax_target_snd is not None
                                    else None
                                ),
                                "query_type": "requery",  # Mark as event-triggered requery
                                "hidden_state": agent_hidden,  # GRU hidden state at requery time
                                "agent_idx": agent_idx,  # Store agent index for food position extraction
                            }
                            adapter_snd_buffer.append(entry)

                        # Keep buffer under maximum size
                        if len(adapter_snd_buffer) > adapter_snd_buffer_size:
                            # This shouldn't happen normally, but safeguard against overflow
                            adapter_snd_buffer = adapter_snd_buffer[
                                -adapter_snd_buffer_size:
                            ]

                    # Log requery event (only occasionally to avoid spam)
                    if episode % log_interval == 0 or episode == 0:
                        # Get which specific agents were updated
                        requeried_envs = newly_detected.nonzero(as_tuple=True)
                        event_name = (
                            "reached landmark"
                            if scenario_name == "grassland"
                            else "detected food"
                        )
                        if num_requeried <= 5:  # Only show details for small numbers
                            print(
                                f"  [Episode {episode}, Step {step}] Requeried HN for {num_requeried} agent(s) ({event_name}): env={requeried_envs[0].tolist()}, agent={requeried_envs[1].tolist()}"
                            )
                        else:
                            print(
                                f"  [Episode {episode}, Step {step}] Requeried HN for {num_requeried}/{batch_size} agent(s) ({100*num_requeried/batch_size:.1f}%) ({event_name})"
                            )

                # Update tracking: mark agents that have detected food (sampling and dispersion)
                if scenario_name in ["sampling", "dispersion"]:
                    food_detected_already = (
                        food_detected_already | current_food_in_range
                    )

            # Store action masks for SMAX (None for continuous action envs)
            if scenario_name == "smax":
                trajectory_data["action_masks"].append(action_masks_flat)
            else:
                trajectory_data["action_masks"].append(None)
            # NaN protection for log probs before storing
            log_probs_safe = jnp.nan_to_num(
                log_probs_flat, nan=-100.0, posinf=-100.0, neginf=-100.0
            )
            log_probs_safe = jnp.clip(log_probs_safe, -100.0, 100.0)
            trajectory_data["log_probs"].append(log_probs_safe)
            # NaN protection for values before storing (critical for GAE)
            values_safe = jnp.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)
            values_safe = jnp.clip(values_safe, -100.0, 100.0)
            trajectory_data["values"].append(values_safe)

            # Convert rewards properly
            # VMAS returns list of rewards for all agents in environment
            # Each has shape: (num_envs,)
            # No padding needed - environment always has max_agents when mixed training enabled

            rewards_stacked = torch.stack(rewards, dim=0)  # (max_agents, num_envs)
            rewards_transposed = rewards_stacked.transpose(
                0, 1
            )  # (num_envs, max_agents)
            rewards_np = (
                rewards_transposed.detach().cpu().numpy()
                if rewards_transposed.requires_grad
                else rewards_transposed.cpu().numpy()
            )
            # Critical: Replace NaN/inf in rewards before storing
            rewards_np = np.nan_to_num(rewards_np, nan=0.0, posinf=0.0, neginf=0.0)

            # CRITICAL: For mixed training, mask out inactive agents BEFORE flattening
            if mixed_agent_training:
                if per_env_agent_variation:
                    # Per-environment variation: different agent count per environment
                    # Create mask: (num_envs, env_num_agents)
                    agent_indices = np.arange(env_num_agents)  # (env_num_agents,)
                    # Broadcast: (num_envs, 1) vs (env_num_agents,) -> (num_envs, env_num_agents)
                    agent_mask = (
                        agent_indices[np.newaxis, :]
                        < current_num_agents_per_env[:, np.newaxis]
                    )
                else:
                    # Per-episode variation: all environments have same agent count
                    # Zero out rewards for inactive agents (>= current_num_agents)
                    # Shape: (num_envs, env_num_agents)
                    agent_mask = np.arange(env_num_agents) < current_num_agents
                rewards_np = (
                    rewards_np * agent_mask[np.newaxis, :]
                    if not per_env_agent_variation
                    else rewards_np * agent_mask
                )

            trajectory_data["rewards"].append(np.asarray(rewards_np).flatten())

            # Calculate mean reward for logging
            reward_mean = float(rewards_np.mean())
            # Protect against NaN/inf rewards
            if np.isnan(reward_mean) or np.isinf(reward_mean):
                reward_mean = 0.0
            episode_rewards.append(reward_mean)

            # Check if episode is done (handle both list and tensor formats)
            if isinstance(dones, list):
                # SMAX: list of tensors, check if all agents/envs are done
                all_done = all(d.all().item() for d in dones)
            else:
                # VMAS: single tensor
                all_done = dones.all().item()

            # Store done flags for each environment-agent pair (needed for RNN critic)
            # Shape should match batch_size (num_envs * env_num_agents)
            if scenario_name == "smax":
                # dones is a list of done flags per agent: [env0_agent0, env0_agent1, ...]
                # Stack and flatten to get (num_envs * env_num_agents,)
                dones_stacked = torch.stack(dones, dim=0)  # (env_num_agents, num_envs)
                dones_transposed = dones_stacked.transpose(
                    0, 1
                )  # (num_envs, env_num_agents)
                dones_flat = dones_transposed.reshape(
                    -1
                )  # (num_envs * env_num_agents,)
                dones_np = dones_flat.cpu().numpy()
            else:
                # VMAS: dones is a single tensor of shape (num_envs,)
                # Need to repeat it for all agents: (num_envs,) -> (num_envs * env_num_agents,)
                dones_np = dones.cpu().numpy()  # (num_envs,)
                # CORRECT: repeat each env's done flag for all its agents
                # Result: [E0, E0, E0, E1, E1, E1, ...] matching [Ag0_E0, Ag1_E0, Ag2_E0, Ag0_E1, ...]
                dones_np = np.repeat(
                    dones_np, env_num_agents
                )  # (num_envs * env_num_agents,)

            trajectory_data["dones"].append(jnp.array(dones_np, dtype=bool))

            # CRITICAL: Reset environment if episode terminates during rollout
            # This prevents collecting meaningless data from terminated states
            if all_done:
                num_episode_resets += 1
                total_steps_before_resets.append(step + 1)

                # Debug: Show why episode ended (SMAX only)
                if scenario_name == "smax" and (
                    episode % log_interval == 0 or episode < 5
                ):
                    # Check termination reason from env_state
                    # For HeuristicEnemySMAX, state is wrapped in env_state.state
                    smax_state = (
                        env_state.state if hasattr(env_state, "state") else env_state
                    )
                    allies_alive = np.array(
                        smax_state.unit_alive[: env.num_allies]
                    ).sum()
                    enemies_alive = np.array(
                        smax_state.unit_alive[env.num_allies :]
                    ).sum()
                    time_step = int(np.array(smax_state.time))

                    reason = []
                    if allies_alive == 0:
                        reason.append("all allies dead")
                    if enemies_alive == 0:
                        reason.append("all enemies dead (WON!)")
                    if time_step >= env.max_steps:
                        reason.append(f"time limit ({time_step}/{env.max_steps})")

                    print(
                        f"  [Episode {episode}, Step {step + 1}] Reset: {' | '.join(reason)} (Allies: {allies_alive}, Enemies: {enemies_alive})"
                    )

                if scenario_name == "smax":
                    # Reset SMAX with a new episode key
                    reset_key = jax.random.fold_in(episode_key, step + 1000)
                    obs_dict, env_state = env.reset(reset_key)
                    next_obs = [obs_dict[agent] for agent in env.agents]
                    next_obs = [
                        torch.from_numpy(np.array(o)).float().to(torch_device)
                        for o in next_obs
                    ]
                    next_obs = [o.unsqueeze(0).expand(num_envs, -1) for o in next_obs]
                else:
                    # Reset VMAS environment
                    next_obs = env.reset()

            # Update observation for next iteration
            obs = next_obs

        # ====================================================================
        # Rollout Summary (Debug)
        # ====================================================================
        if episode % log_interval == 0 or episode < 5:
            total_episodes_in_rollout = num_episode_resets + 1  # +1 for initial episode
            print(
                f"  [Episode {episode}] Rollout completed: {total_episodes_in_rollout} episode(s) in {rollout_steps} steps"
            )
            if num_episode_resets > 0:
                # Calculate actual episode lengths from reset points
                episode_lengths = []
                prev_step = 0
                for reset_step in total_steps_before_resets:
                    episode_lengths.append(reset_step - prev_step)
                    prev_step = reset_step
                # Add final episode length (from last reset to end of rollout)
                episode_lengths.append(rollout_steps - prev_step)

                avg_episode_length = np.mean(episode_lengths)
                print(
                    f"    Episode lengths: {episode_lengths} (avg: {avg_episode_length:.1f} steps)"
                )

        # ====================================================================
        # Save trajectory observations for next episode's adapter SND calculation
        # ====================================================================
        if use_adapter_snd:
            previous_trajectory_obs = trajectory_data[
                "obs"
            ]  # List of arrays (num_steps, batch_size, obs_dim)

        # ====================================================================
        # Compute Advantages using GAE (Generalized Advantage Estimation)
        # ====================================================================
        gamma = config["training"]["gamma"]
        gae_lambda = config["training"]["gae_lambda"]
        reward_scale = config["training"].get("reward_scale", 1.0)

        rewards_array = np.array(trajectory_data["rewards"])  # (num_steps, batch_size)

        # Scale rewards to improve gradient magnitude (important for sparse rewards)
        rewards_array = rewards_array * reward_scale
        values_array = np.array(
            [np.asarray(v) for v in trajectory_data["values"]]
        )  # (num_steps, num_envs, env_num_agents) - critic always outputs per-agent shape
        num_steps = len(rewards_array)

        # Reshape rewards to (num_steps, num_envs, env_num_agents)
        rewards_reshaped = rewards_array.reshape(num_steps, num_envs, env_num_agents)

        # IMPORTANT: With per-agent critic, we now have per-agent values
        # With shared critic, we broadcast the shared value to all agents
        # values_array shape: (num_steps, num_envs, env_num_agents)
        # rewards_reshaped shape: (num_steps, num_envs, env_num_agents)
        # GAE is computed per agent since each agent has unique returns

        # NaN protection for values_array before GAE computation
        values_array = np.nan_to_num(values_array, nan=0.0, posinf=100.0, neginf=-100.0)
        values_array = np.clip(values_array, -100.0, 100.0)

        # NaN protection for rewards
        rewards_reshaped = np.nan_to_num(
            rewards_reshaped, nan=0.0, posinf=10.0, neginf=-10.0
        )
        rewards_reshaped = np.clip(rewards_reshaped, -10.0, 10.0)

        # Compute per-agent GAE
        advantages = np.zeros_like(
            rewards_reshaped
        )  # (num_steps, num_envs, env_num_agents)
        returns = np.zeros_like(
            rewards_reshaped
        )  # (num_steps, num_envs, env_num_agents)
        gae = np.zeros((num_envs, env_num_agents))  # GAE per agent

        # Bootstrap value for last step (assume 0 for terminal states)
        next_value = np.zeros((num_envs, env_num_agents))

        # Get done flags array - reshape from per-agent to per-environment
        # trajectory_data["dones"] has shape (num_steps, batch_size) where batch_size = num_envs * env_num_agents
        dones_array_flat = np.array(trajectory_data["dones"])  # (num_steps, batch_size)
        # Reshape to (num_steps, num_envs, env_num_agents)
        dones_array_reshaped = dones_array_flat.reshape(
            num_steps, num_envs, env_num_agents
        )
        # Reduce to per-environment dones (any agent done = env done)
        dones_array = np.any(dones_array_reshaped, axis=-1).astype(
            float
        )  # (num_steps, num_envs)

        # Compute per-agent GAE backwards through time
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # Last step: use zero next value (bootstrap)
                next_non_terminal = (1.0 - dones_array[t])[
                    :, np.newaxis
                ]  # (num_envs, 1)
                next_val = next_value
            else:
                # Use done flag from NEXT timestep for proper bootstrap
                next_non_terminal = (1.0 - dones_array[t])[
                    :, np.newaxis
                ]  # (num_envs, 1)
                next_val = values_array[t + 1]

            # TD error with proper terminal handling - now per agent
            delta = (
                rewards_reshaped[t]
                + gamma * next_non_terminal * next_val
                - values_array[t]
            )
            # Clip delta to prevent extreme values
            delta = np.clip(delta, -100.0, 100.0)

            # GAE accumulation (resets at episode boundaries due to next_non_terminal=0)
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            # Clip GAE to prevent accumulation of extreme values
            gae = np.clip(gae, -100.0, 100.0)

            advantages[t] = gae
            returns[t] = gae + values_array[t]

        # advantages and returns are already per-agent: (num_steps, num_envs, env_num_agents)
        # Just rename for clarity
        advantages_expanded = advantages
        returns_expanded = returns

        # CRITICAL: For mixed training, zero out padded agents BEFORE normalization
        # This ensures normalization statistics are computed only over active agents
        if mixed_agent_training:
            if per_env_agent_variation:
                # Per-environment variation: different agent count per environment
                # Create mask: (1, num_envs, env_num_agents)
                agent_indices = np.arange(env_num_agents)  # (env_num_agents,)
                agent_mask = (
                    agent_indices[np.newaxis, np.newaxis, :]
                    < current_num_agents_per_env[np.newaxis, :, np.newaxis]
                )
                # Zero out inactive agents
                advantages_expanded = advantages_expanded * agent_mask
                # Note: returns don't need masking as they're used with masked value loss
            else:
                # Per-episode variation: all envs have same number of active agents
                # Create mask for active agents: (env_num_agents,)
                agent_mask = np.arange(env_num_agents) < current_num_agents
                # Expand to match advantages shape: (1, 1, env_num_agents)
                agent_mask_expanded = agent_mask[np.newaxis, np.newaxis, :]
                # Zero out padded agents
                advantages_expanded = advantages_expanded * agent_mask_expanded
                # Note: returns don't need masking as they're used with masked value loss

        # Flatten
        advantages_flat = advantages_expanded.reshape(-1)
        returns_flat = returns_expanded.reshape(-1)

        # Store raw advantages for debugging
        raw_advantages_mean = advantages_flat.mean()
        raw_advantages_std = advantages_flat.std()

        # CRITICAL: Normalize advantages to stabilize policy updates
        # For mixed training, compute stats only over active agents
        if mixed_agent_training:
            # Create mask for active agents in flattened array
            # Reshape to (num_steps * num_envs, env_num_agents)
            advantages_reshaped = advantages_flat.reshape(-1, env_num_agents)
            # Mask: True for active agents (< current_num_agents)
            agent_mask = np.arange(env_num_agents) < current_num_agents
            # Apply mask to get only active advantages
            active_advantages = advantages_reshaped[:, :current_num_agents].flatten()
            advantages_mean = active_advantages.mean()
            advantages_std = active_advantages.std()
        else:
            advantages_mean = advantages_flat.mean()
            advantages_std = advantages_flat.std()
        advantages_flat = (advantages_flat - advantages_mean) / (advantages_std + 1e-8)

        # CRITICAL: Clip normalized advantages to prevent extreme outliers
        # This is a common practice in PPO implementations to stabilize training
        advantage_clip_range = config["training"].get("advantage_clip_range", 10.0)
        advantages_flat = np.clip(
            advantages_flat, -advantage_clip_range, advantage_clip_range
        )

        # Debug output for first episode and periodically
        if episode == 0 or episode == 1 or episode % 100 == 0:
            print(f"\n[DEBUG Episode {episode}]")
            if mixed_agent_training:
                print(f"  Current agents: {current_num_agents}/{env_num_agents}")
                # Check if padded rewards are actually zero
                if current_num_agents < env_num_agents:
                    padded_rewards = rewards_reshaped[:, :, current_num_agents:].mean()
                    print(
                        f"  Padded agents' rewards (should be 0): {padded_rewards:.6f}"
                    )
            print(
                f"  Rewards (scaled) mean: {rewards_array.mean():.6f}, std: {rewards_array.std():.6f}"
            )
            print(
                f"  Values mean: {values_array.mean():.6f}, std: {values_array.std():.6f}"
            )
            print(
                f"  Raw Advantages mean: {raw_advantages_mean:.6f}, std: {raw_advantages_std:.6f}"
            )
            print(
                f"  Normalized Advantages mean: {advantages_flat.mean():.6f}, std: {advantages_flat.std():.6f}, min: {advantages_flat.min():.6f}, max: {advantages_flat.max():.6f}"
            )

        advantages_batch = jnp.array(advantages_flat)
        returns_batch = jnp.array(returns_flat)

        # Old value predictions for PPO value clipping.
        # Shape: (num_steps * num_envs * num_agents,) — per-agent values matching returns_batch.
        old_values_flat = values_array.reshape(
            -1
        )  # (num_steps * num_envs * env_num_agents,)
        old_values_batch_np = old_values_flat  # kept as numpy; converted to jnp below

        # Normalize returns to prevent value function explosion
        normalize_returns = config["training"].get("normalize_returns", False)
        if normalize_returns:
            returns_mean = returns_batch.mean()
            returns_std = returns_batch.std()
            returns_batch = (returns_batch - returns_mean) / (returns_std + 1e-8)
            # Also normalize old values to the same scale so value clipping is meaningful
            old_values_batch_np = (old_values_batch_np - float(returns_mean)) / (
                float(returns_std) + 1e-8
            )
            if episode % log_interval == 0:
                print(
                    f"  Returns - mean: {returns_mean:.2f}, std: {returns_std:.2f} (normalized)"
                )
        else:
            if episode % log_interval == 0:
                print(
                    f"  Returns - mean: {returns_batch.mean():.2f}, std: {returns_batch.std():.2f}, max: {returns_batch.max():.2f}"
                )

        # ====================================================================
        # Training Step (JAX)
        # ====================================================================
        # Convert old value predictions to JAX array now that optional normalisation is done
        old_values_batch = jnp.array(
            old_values_batch_np
        )  # (num_steps * num_envs * num_agents,)

        # Determine value clip range (None disables clipping; using same eps as policy clip)
        _clip_value_loss = config["training"].get("clip_value_loss", False)
        value_clip_range = (
            float(config["training"]["ppo_clip_epsilon"]) if _clip_value_loss else None
        )

        # Stack trajectory data
        obs_batch = jnp.concatenate(
            trajectory_data["obs"], axis=0
        )  # (num_steps * num_envs * env_num_agents, obs_dim)
        global_state_batch = jnp.concatenate(
            trajectory_data["global_states"], axis=0
        )  # (num_steps * num_envs, global_state_dim)
        # Optionally clip global state batch before critic in training
        if clip_global_state:
            global_state_batch = jnp.clip(
                global_state_batch, clip_global_state_min, clip_global_state_max
            )
        actions_batch = jnp.concatenate(trajectory_data["actions"], axis=0)
        old_log_probs_batch = jnp.concatenate(trajectory_data["log_probs"], axis=0)
        dones_batch = jnp.concatenate(trajectory_data["dones"], axis=0)

        # Get action masks batch for SMAX (None for continuous actions)
        if scenario_name == "smax" and trajectory_data["action_masks"][0] is not None:
            action_masks_batch = jnp.concatenate(
                trajectory_data["action_masks"], axis=0
            )
        else:
            action_masks_batch = None

        # Additional NaN protection for old_log_probs_batch (defensive)
        old_log_probs_batch = jnp.nan_to_num(
            old_log_probs_batch, nan=-100.0, posinf=-100.0, neginf=-100.0
        )
        old_log_probs_batch = jnp.clip(old_log_probs_batch, -100.0, 100.0)

        # Use stored per-timestep context and lidar from trajectory
        # This ensures we use the ACTUAL context that was active when adapters were generated
        num_trajectory_steps = obs_batch.shape[0]
        num_batches = num_trajectory_steps // batch_size

        # Stack context from trajectory (already flattened during storage)
        if trajectory_data["context"][0] is not None:
            jax_context_batch_flat = jnp.concatenate(
                trajectory_data["context"], axis=0
            )  # (num_trajectory_steps, context_dim)

            # Reshape to (num_batches * num_envs, env_num_agents, context_dim) for hypernetwork
            context_dim = jax_context_batch_flat.shape[-1]
            jax_context_batch = jax_context_batch_flat.reshape(
                num_batches * num_envs, env_num_agents, context_dim
            )
        else:
            jax_context_batch = None

        if use_cash and trajectory_data["cash_context"][0] is not None:
            jax_cash_context_batch = jnp.concatenate(
                trajectory_data["cash_context"], axis=0
            )  # (num_steps * batch_size, cash_context_dim)
        else:
            jax_cash_context_batch = None

        # Task is static, so we can still tile it
        if jax_task is not None:
            jax_task_batch = jnp.tile(
                jax_task[None, :, :, :], (num_batches, 1, 1, 1)
            )  # (num_batches, num_envs, env_num_agents, task_dim)
            jax_task_batch = jax_task_batch.reshape(
                num_batches * num_envs, env_num_agents, -1
            )
        else:
            jax_task_batch = None

        # Target SND is also static, so we can tile it
        if jax_target_snd is not None:
            jax_target_snd_batch = jnp.tile(
                jax_target_snd[None, :, :, :], (num_batches, 1, 1, 1)
            )  # (num_batches, num_envs, env_num_agents, target_snd_dim)
            jax_target_snd_batch = jax_target_snd_batch.reshape(
                num_batches * num_envs, env_num_agents, -1
            )
        else:
            jax_target_snd_batch = None

        # Handle mask tiling (may be None)
        if jax_mask is not None:
            # jax_mask has shape (1, 1, env_num_agents, env_num_agents)
            # Need to tile to (num_batches, num_envs, 1, env_num_agents, env_num_agents)
            # First expand to (num_envs, 1, env_num_agents, env_num_agents) by tiling
            jax_mask_expanded = jnp.tile(jax_mask, (num_envs, 1, 1, 1))
            # Then add batch dimension and tile: (num_batches, num_envs, 1, env_num_agents, env_num_agents)
            jax_mask_batch = jnp.tile(
                jax_mask_expanded[None, :, :, :, :], (num_batches, 1, 1, 1, 1)
            )
            # Reshape to (num_batches * num_envs, 1, env_num_agents, env_num_agents)
            jax_mask_batch = jax_mask_batch.reshape(
                num_batches * num_envs, 1, env_num_agents, env_num_agents
            )
        else:
            jax_mask_batch = None

        # Use stored lidar from trajectory (may have changed during rollout)
        if use_lidar_context and trajectory_data["lidar"][0] is not None:
            jax_lidar_batch_flat = jnp.concatenate(
                trajectory_data["lidar"], axis=0
            )  # (num_trajectory_steps, lidar_dim)

            # Debug: Check if lidar varies across timesteps (should if requeried)
            if episode % log_interval == 0 and len(trajectory_data["lidar"]) > 1:
                lidar_first_step = trajectory_data["lidar"][
                    0
                ]  # (batch_size, lidar_dim)
                lidar_mid_step = trajectory_data["lidar"][
                    len(trajectory_data["lidar"]) // 2
                ]
                lidar_last_step = trajectory_data["lidar"][-1]

                # Only compute if lidar is not empty
                if lidar_first_step.size > 0:
                    # Compute average change across ALL agents
                    lidar_diff_mid = jnp.abs(lidar_first_step - lidar_mid_step).mean()
                    lidar_diff_last = jnp.abs(lidar_first_step - lidar_last_step).mean()
                    lidar_max_diff = jnp.abs(lidar_first_step - lidar_last_step).max()

                    print(f"[DEBUG Training] Lidar variation (avg over all agents):")
                    print(
                        f"  First->Mid: {lidar_diff_mid:.6f}, First->Last: {lidar_diff_last:.6f}, Max diff: {lidar_max_diff:.6f}"
                    )

                    # Show samples from first and an agent with maximum change
                    lidar_first_agent0 = lidar_first_step[0]
                    lidar_last_agent0 = lidar_last_step[0]
                    print(
                        f"  Agent 0 - First: {lidar_first_agent0[:5]}..., Last: {lidar_last_agent0[:5]}..."
                    )

                    # Find agent with biggest change
                    per_agent_diff = jnp.abs(lidar_first_step - lidar_last_step).sum(
                        axis=1
                    )
                    max_change_agent_idx = jnp.argmax(per_agent_diff)
                    if max_change_agent_idx > 0:
                        print(
                            f"  Agent {max_change_agent_idx} (max change) - First: {lidar_first_step[max_change_agent_idx][:5]}..., Last: {lidar_last_step[max_change_agent_idx][:5]}..."
                        )
                else:
                    print(
                        f"[DEBUG Training] Lidar is empty (dimension 0) - skipping variation check"
                    )

            lidar_dim = jax_lidar_batch_flat.shape[-1]
            jax_lidar_batch = jax_lidar_batch_flat.reshape(
                num_batches * num_envs, env_num_agents, lidar_dim
            )
        else:
            jax_lidar_batch = None

        # Use stored food positions from trajectory (for dispersion_vmas)
        if trajectory_data["food_positions"][0] is not None:
            jax_food_positions_batch_flat = jnp.concatenate(
                trajectory_data["food_positions"], axis=0
            )  # (num_trajectory_steps, food_position_dim)
            food_position_dim = jax_food_positions_batch_flat.shape[-1]
            jax_food_positions_batch = jax_food_positions_batch_flat.reshape(
                num_batches * num_envs, env_num_agents, food_position_dim
            )
        else:
            jax_food_positions_batch = None

        # Use stored agent positions from trajectory (for dispersion_vmas)
        if trajectory_data["agent_positions"][0] is not None:
            jax_agent_positions_batch_flat = jnp.concatenate(
                trajectory_data["agent_positions"], axis=0
            )  # (num_trajectory_steps, agent_position_dim)
            agent_position_dim_traj = jax_agent_positions_batch_flat.shape[-1]
            jax_agent_positions_batch = jax_agent_positions_batch_flat.reshape(
                num_batches * num_envs, env_num_agents, agent_position_dim_traj
            )
        else:
            jax_agent_positions_batch = None

        # Use stored environment context from trajectory (for reverse_transport)
        if trajectory_data["env_context"][0] is not None:
            jax_env_context_batch_flat = jnp.concatenate(
                trajectory_data["env_context"], axis=0
            )  # (num_trajectory_steps, env_context_dim)
            env_context_dim_traj = jax_env_context_batch_flat.shape[-1]
            jax_env_context_batch = jax_env_context_batch_flat.reshape(
                num_batches * num_envs, env_num_agents, env_context_dim_traj
            )
            if episode == 0:
                print(f"\n[DEBUG] Created jax_env_context_batch from trajectory:")
                print(f"  jax_env_context_batch shape: {jax_env_context_batch.shape}")
                print(f"  Sample (batch0, agent0): {jax_env_context_batch[0, 0]}")
        else:
            jax_env_context_batch = None
            if episode == 0 and env_context_dim > 0:
                print(
                    f"\n[DEBUG WARNING] jax_env_context_batch is None but env_context_dim={env_context_dim}"
                )
                print(
                    f"  trajectory_data['env_context'] length: {len(trajectory_data['env_context'])}"
                )
                print(
                    f"  trajectory_data['env_context'][0]: {trajectory_data['env_context'][0]}"
                )

        # Get PPO hyperparameters from config
        clip_epsilon = config["training"]["ppo_clip_epsilon"]
        entropy_coef = config["training"]["entropy_coef"]
        entropy_decay = config["training"].get("entropy_decay", 1.0)
        min_entropy_coef = config["training"].get("min_entropy_coef", 0.0)
        value_loss_coef = config["training"]["value_loss_coef"]
        ppo_epochs = config["training"].get("ppo_epochs", 4)

        # Mini-batch settings
        use_minibatches = config["training"].get("use_minibatches", False)
        num_minibatches = config["training"].get("num_minibatches", 4)
        shuffle_minibatches = config["training"].get("shuffle_minibatches", True)

        # Log mini-batch configuration on first episode
        if episode == 0 and use_minibatches:
            total_samples = rollout_steps * num_envs * env_num_agents
            minibatch_size = total_samples // num_minibatches
            print(f"\n[Mini-batch Training Enabled]")
            print(f"  Total samples per rollout: {total_samples}")
            print(f"  Number of mini-batches: {num_minibatches}")
            print(f"  Samples per mini-batch: {minibatch_size}")
            print(f"  Memory reduction: {100/num_minibatches:.1f}% of full batch")
            print(f"  Shuffle mini-batches: {shuffle_minibatches}")

        # Apply entropy decay based on episode number
        current_entropy_coef = max(
            min_entropy_coef, entropy_coef * (entropy_decay**episode)
        )

        # Track parameter changes for debugging
        if episode % log_interval == 0 or episode == 1:
            # Get initial policy parameters for comparison
            initial_policy_params = jax.tree_util.tree_map(
                lambda x: x.copy(), policy_state.params
            )

        # ====================================================================
        # TIMING: End rollout, start PPO updates
        # ====================================================================
        rollout_time = time.time() - rollout_start_time
        ppo_start_time = time.time()

        # --------------------------------------------------------------------
        # Fast path: DiCo + minibatches + MLP (no GRU).
        # Runs the whole PPO update (ppo_epochs × num_minibatches gradient
        # steps) inside a single jit'd lax.scan so XLA can dispatch the entire
        # update as one launch instead of ppo_epochs * num_minibatches Python
        # round-trips. Falls back to the legacy Python loop for every other
        # configuration (HyperLoRA, GRU, no-minibatches, SMAX, etc.).
        # --------------------------------------------------------------------
        _use_dico_scan = (
            use_dico
            and use_minibatches
            and not use_gru_policy
            and scenario_name != "smax"  # discrete/action-masks path not covered
        )
        if _use_dico_scan:
            batch_size = obs_batch.shape[0]
            num_timesteps = batch_size // (num_envs * env_num_agents)
            agent_ids_batch = jnp.tile(
                jnp.arange(env_num_agents), num_timesteps * num_envs
            )
            policy_state, critic_state, loss_info = _ppo_update_dico_scan(
                policy_state,
                critic_state,
                obs_batch,
                global_state_batch,
                actions_batch,
                old_log_probs_batch,
                advantages_batch,
                returns_batch,
                agent_ids_batch,
                diversity_scaling,
                clip_epsilon,
                current_entropy_coef,
                value_loss_coef,
                num_minibatches=num_minibatches,
                ppo_epochs=ppo_epochs,
                num_agents=env_num_agents,
                current_num_agents=current_num_agents,
                max_agents=env_num_agents,
                is_discrete=False,
            )
            # Skip the legacy python double-loop below.
            ppo_epochs_to_run = 0
        else:
            ppo_epochs_to_run = ppo_epochs

        # Perform multiple PPO training epochs over the same data
        for ppo_epoch in range(ppo_epochs_to_run):
            # ================================================================
            # TIMING: Start PPO epoch
            # ================================================================
            epoch_start_time = time.time()

            # Generate shuffle key for this epoch if using mini-batches
            if use_minibatches and shuffle_minibatches:
                rng, shuffle_rng = jax.random.split(rng)
            else:
                shuffle_rng = None
            if use_gru_policy:
                # GRU-specific training path: need to reshape data to preserve temporal structure
                # This applies whether using hypernetwork or not - GRU requires sequence processing
                # obs_batch shape: (num_steps * num_envs * env_num_agents, obs_dim)
                # Need to reshape to: (num_steps, num_envs * env_num_agents, obs_dim) for GRU
                num_steps = rollout_steps
                total_agents = num_envs * env_num_agents

                # Reshape all data to have sequence structure: (num_steps, batch_size, ...)
                obs_seq = obs_batch.reshape(num_steps, total_agents, -1)

                # Global state is per-environment, not per-agent
                global_state_seq = global_state_batch.reshape(num_steps, num_envs, -1)
                actions_seq = (
                    actions_batch.reshape(num_steps, total_agents, -1)
                    if actions_batch.ndim > 1
                    else actions_batch.reshape(num_steps, total_agents)
                )
                old_log_probs_seq = old_log_probs_batch.reshape(num_steps, total_agents)
                advantages_seq = advantages_batch.reshape(num_steps, total_agents)
                returns_seq = returns_batch.reshape(num_steps, total_agents)

                # Reshape action masks if present
                if action_masks_batch is not None:
                    action_masks_seq = action_masks_batch.reshape(
                        num_steps, total_agents, -1
                    )
                else:
                    action_masks_seq = None

                # For task/context batches, we need to reshape from stored trajectory data
                # Task is static, so tile for all timesteps
                if jax_task is not None:
                    task_dim = jax_task.shape[-1]  # Get from original jax_task
                    jax_task_seq = jnp.tile(
                        jax_task[None, :, :, :], (num_steps, 1, 1, 1)
                    )  # (num_steps, num_envs, env_num_agents, task_dim)
                    jax_task_seq = jax_task_seq.reshape(
                        num_steps, num_envs * env_num_agents, task_dim
                    )
                else:
                    jax_task_seq = None

                # Target SND is also static, so tile for all timesteps
                if jax_target_snd is not None:
                    jax_target_snd_seq = jnp.tile(
                        jax_target_snd[None, :, :, :], (num_steps, 1, 1, 1)
                    )  # (num_steps, num_envs, env_num_agents, target_snd_dim)
                    jax_target_snd_seq = jax_target_snd_seq.reshape(
                        num_steps, num_envs * env_num_agents, -1
                    )
                else:
                    jax_target_snd_seq = None

                # Use stored per-timestep context from trajectory (may have changed during rollout)
                # jax_context_batch is already prepared: (num_batches * num_envs, env_num_agents, context_dim)
                if jax_context_batch is not None:
                    context_dim = jax_context_batch.shape[-1]
                    # Reshape to sequence format: (num_steps, num_envs * env_num_agents, context_dim)
                    jax_context_seq = jax_context_batch.reshape(
                        num_steps, num_envs * env_num_agents, context_dim
                    )
                else:
                    jax_context_seq = None

                # Use stored per-timestep lidar from trajectory (may have changed during rollout)
                if jax_lidar_batch is not None:
                    lidar_dim = jax_lidar_batch.shape[-1]
                    jax_lidar_seq = jax_lidar_batch.reshape(
                        num_steps, num_envs * env_num_agents, lidar_dim
                    )
                else:
                    jax_lidar_seq = None

                # Use stored per-timestep food positions from trajectory
                if jax_food_positions_batch is not None:
                    food_position_dim = jax_food_positions_batch.shape[-1]
                    jax_food_positions_seq = jax_food_positions_batch.reshape(
                        num_steps, num_envs * env_num_agents, food_position_dim
                    )
                else:
                    jax_food_positions_seq = None

                # Use stored per-timestep agent positions from trajectory (dispersion_vmas)
                if jax_agent_positions_batch is not None:
                    agent_position_dim_seq = jax_agent_positions_batch.shape[-1]
                    jax_agent_positions_seq = jax_agent_positions_batch.reshape(
                        num_steps, num_envs * env_num_agents, agent_position_dim_seq
                    )
                else:
                    jax_agent_positions_seq = None

                if jax_mask is not None:
                    # jax_mask is (1, 1, env_num_agents, env_num_agents), tile for all steps
                    jax_mask_seq = jnp.tile(
                        jax_mask[None, :, :, :, :], (num_steps, num_envs, 1, 1, 1)
                    )
                    jax_mask_seq = jax_mask_seq.reshape(
                        num_steps, num_envs, 1, env_num_agents, env_num_agents
                    )
                else:
                    jax_mask_seq = None

                # Get initial hidden states (should be stored from rollout start, but for now use zeros)
                if trajectory_data["init_hidden"] is not None:
                    init_hidden = trajectory_data["init_hidden"]
                else:
                    init_hidden = jnp.zeros((total_agents, gru_hidden_dim))

                # Get initial critic hidden states
                if trajectory_data["init_critic_hidden"] is not None:
                    init_critic_hidden = trajectory_data["init_critic_hidden"]
                else:
                    init_critic_hidden = jnp.zeros((num_envs, gru_hidden_dim))

                # Reshape dones to sequence format: (num_steps, num_envs)
                dones_seq = dones_batch.reshape(num_steps, num_envs, env_num_agents)
                # Reduce to environment-level dones (any agent done = env done)
                dones_seq = jnp.any(dones_seq, axis=-1)  # (num_steps, num_envs)

                if use_cash:
                    cash_context_seq = jax_cash_context_batch.reshape(
                        num_steps, total_agents, -1
                    )

                    if use_minibatches:
                        shuffle_for_gru = False
                        minibatches = create_minibatches_sequential(
                            obs_seq,
                            global_state_seq,
                            actions_seq,
                            old_log_probs_seq,
                            advantages_seq,
                            returns_seq,
                            None,
                            cash_context_seq,
                            None,
                            None,
                            None,
                            None,
                            None,
                            action_masks_seq,
                            dones_seq,
                            init_hidden,
                            init_critic_hidden,
                            num_minibatches,
                            num_steps,
                            num_envs,
                            env_num_agents,
                            shuffle=shuffle_for_gru,
                            rng_key=shuffle_rng,
                        )

                        accumulated_loss_info = None
                        for mb_idx, (
                            mb_obs,
                            mb_global_state,
                            mb_actions,
                            mb_old_log_probs,
                            mb_advantages,
                            mb_returns,
                            _mb_task_unused,
                            mb_cash_context,
                            _mb_lidar_unused,
                            _mb_food_positions_unused,
                            _mb_agent_positions_unused,
                            _mb_target_snd_unused,
                            _mb_mask_unused,
                            _mb_action_masks_unused,
                            mb_dones,
                            mb_init_hidden,
                            mb_init_critic_hidden,
                        ) in enumerate(minibatches):
                            policy_state, critic_state, loss_info = (
                                _train_step_cash_gru(
                                    policy_state,
                                    critic_state,
                                    mb_obs,
                                    mb_cash_context,
                                    mb_global_state,
                                    mb_actions,
                                    mb_old_log_probs,
                                    mb_advantages,
                                    mb_returns,
                                    mb_init_hidden,
                                    mb_init_critic_hidden,
                                    mb_dones,
                                    clip_epsilon,
                                    current_entropy_coef,
                                    value_loss_coef,
                                    env_num_agents,
                                    current_num_agents,
                                    env_num_agents,
                                )
                            )
                            if accumulated_loss_info is None:
                                accumulated_loss_info = {
                                    k: v for k, v in loss_info.items()
                                }
                            else:
                                for k, v in loss_info.items():
                                    accumulated_loss_info[k] += v

                        loss_info = {
                            k: v / num_minibatches
                            for k, v in accumulated_loss_info.items()
                        }
                    else:
                        policy_state, critic_state, loss_info = _train_step_cash_gru(
                            policy_state,
                            critic_state,
                            obs_seq,
                            cash_context_seq,
                            global_state_seq,
                            actions_seq,
                            old_log_probs_seq,
                            advantages_seq,
                            returns_seq,
                            init_hidden,
                            init_critic_hidden,
                            dones_seq,
                            clip_epsilon,
                            current_entropy_coef,
                            value_loss_coef,
                            env_num_agents,
                            current_num_agents,
                            env_num_agents,
                        )
                    continue

                # Split into mini-batches if requested
                if use_minibatches:
                    # CRITICAL: Do NOT shuffle for GRU policies!
                    # Shuffling breaks the correspondence between rollout and training observations
                    # This causes Mean Ratio != 1.0 even with learning_rate=0.0
                    # For temporal models, we must maintain the exact sequence order
                    shuffle_for_gru = False  # Never shuffle GRU minibatches

                    # Create mini-batches by splitting across environments
                    minibatches = create_minibatches_sequential(
                        obs_seq,
                        global_state_seq,
                        actions_seq,
                        old_log_probs_seq,
                        advantages_seq,
                        returns_seq,
                        jax_task_seq,
                        jax_context_seq,
                        jax_lidar_seq,
                        jax_food_positions_seq,
                        jax_agent_positions_seq,
                        jax_target_snd_seq,
                        jax_mask_seq,
                        action_masks_seq,
                        dones_seq,
                        init_hidden,
                        init_critic_hidden,
                        num_minibatches,
                        num_steps,
                        num_envs,
                        env_num_agents,
                        shuffle=shuffle_for_gru,  # Use explicit False for GRU
                        rng_key=shuffle_rng,
                    )

                    # Accumulate losses across mini-batches
                    accumulated_loss_info = None

                    for mb_idx, (
                        mb_obs,
                        mb_global_state,
                        mb_actions,
                        mb_old_log_probs,
                        mb_advantages,
                        mb_returns,
                        mb_task,
                        mb_context,
                        mb_lidar,
                        mb_food_positions,
                        mb_agent_positions,
                        mb_target_snd,
                        mb_mask,
                        mb_action_masks,
                        mb_dones,
                        mb_init_hidden,
                        mb_init_critic_hidden,
                    ) in enumerate(minibatches):
                        # Mini-batch hidden states are already sliced by create_minibatches_sequential

                        # Call GRU-specific training step
                        policy_state, hn_state, critic_state, loss_info = (
                            _train_step_with_hn_gru(
                                policy_state,
                                hn_state,
                                critic_state,
                                mb_obs,
                                mb_global_state,
                                mb_actions,
                                mb_old_log_probs,
                                mb_advantages,
                                mb_returns,
                                mb_task,
                                mb_context,
                                mb_lidar,
                                mb_food_positions,
                                mb_agent_positions,
                                mb_target_snd,
                                mb_mask,
                                mb_action_masks,
                                mb_init_hidden,
                                mb_init_critic_hidden,
                                mb_dones,
                                clip_epsilon,
                                current_entropy_coef,
                                value_loss_coef,
                                env_num_agents,
                                current_num_agents,
                                env_num_agents,
                                lora_scaling_factor,
                                diversity_scaling,
                                is_discrete=(scenario_name == "smax"),
                            )
                        )

                        # Accumulate loss info
                        if accumulated_loss_info is None:
                            accumulated_loss_info = {k: v for k, v in loss_info.items()}
                        else:
                            for k, v in loss_info.items():
                                accumulated_loss_info[k] += v

                    # Average accumulated losses
                    loss_info = {
                        k: v / num_minibatches for k, v in accumulated_loss_info.items()
                    }

                else:
                    # Call GRU-specific training step (full batch)
                    policy_state, hn_state, critic_state, loss_info = (
                        _train_step_with_hn_gru(
                            policy_state,
                            hn_state,
                            critic_state,
                            obs_seq,
                            global_state_seq,
                            actions_seq,
                            old_log_probs_seq,
                            advantages_seq,
                            returns_seq,
                            jax_task_seq,
                            jax_context_seq,
                            jax_lidar_seq,
                            jax_food_positions_seq,
                            jax_agent_positions_seq,
                            jax_target_snd_seq,
                            jax_mask_seq,
                            action_masks_seq,
                            init_hidden,
                            init_critic_hidden,
                            dones_seq,
                            clip_epsilon,
                            current_entropy_coef,
                            value_loss_coef,
                            env_num_agents,
                            current_num_agents,
                            env_num_agents,
                            lora_scaling_factor,
                            diversity_scaling,
                            is_discrete=(scenario_name == "smax"),
                        )
                    )
            elif use_hypernetwork:
                # Split into mini-batches if requested
                if use_minibatches:
                    # For HyperLoRA, task/context/lidar have special structure (batch_envs, num_agents, feature)
                    # We need to split them properly to maintain this structure

                    # Calculate dimensions
                    total_batch_size = obs_batch.shape[
                        0
                    ]  # rollout_steps * num_envs * env_num_agents
                    minibatch_size = total_batch_size // num_minibatches

                    # For task/context/lidar/mask, the batch dimension is over (rollout_steps * num_envs)
                    # not over individual agents
                    total_env_batches = (
                        num_batches * num_envs
                    )  # rollout_steps * num_envs
                    minibatch_env_size = total_env_batches // num_minibatches

                    # CRITICAL: Shuffle at the environment level to maintain correspondence
                    # between agent observations and their environment's context/task/lidar
                    # We shuffle environments, then expand to get agent-level indices
                    if shuffle_minibatches and shuffle_rng is not None:
                        # Shuffle environment indices (each env has env_num_agents agents)
                        env_indices = jax.random.permutation(
                            shuffle_rng, jnp.arange(total_env_batches)
                        )
                        # OPTIMIZED: Vectorized expansion of env indices to agent indices
                        # Instead of Python loop, use broadcasting:
                        # env_indices shape: (total_env_batches,)
                        # Create offset array: [0, 1, 2, ..., env_num_agents-1]
                        agent_offsets = jnp.arange(env_num_agents)
                        # Broadcast: (total_env_batches, 1) * env_num_agents + (1, env_num_agents)
                        obs_indices = (
                            env_indices[:, None] * env_num_agents
                            + agent_offsets[None, :]
                        ).flatten()
                    else:
                        env_indices = jnp.arange(total_env_batches)
                        obs_indices = jnp.arange(total_batch_size)

                    # Accumulate losses across mini-batches
                    accumulated_loss_info = None

                    for mb_idx in range(num_minibatches):
                        # Slice observation-level data (flat batch)
                        # Use consecutive chunks from the shuffled indices
                        start_idx = mb_idx * minibatch_size
                        end_idx = start_idx + minibatch_size
                        mb_obs_indices = obs_indices[start_idx:end_idx]

                        mb_obs = obs_batch[mb_obs_indices]
                        mb_actions = actions_batch[mb_obs_indices]
                        mb_old_log_probs = old_log_probs_batch[mb_obs_indices]
                        mb_advantages = advantages_batch[mb_obs_indices]
                        mb_returns = returns_batch[mb_obs_indices]
                        mb_action_masks = (
                            action_masks_batch[mb_obs_indices]
                            if action_masks_batch is not None
                            else None
                        )

                        # For environment-level data, use consecutive chunks from shuffled env_indices
                        start_env = mb_idx * minibatch_env_size
                        end_env = start_env + minibatch_env_size
                        mb_env_indices = env_indices[start_env:end_env]

                        mb_global_state = global_state_batch[mb_env_indices]
                        mb_old_values = old_values_batch[
                            mb_obs_indices
                        ]  # Per-agent values, not per-env

                        # Slice environment-level structured data (task/context/lidar/mask)
                        mb_task = (
                            jax_task_batch[mb_env_indices]
                            if jax_task_batch is not None
                            else None
                        )
                        mb_context = (
                            jax_context_batch[mb_env_indices]
                            if jax_context_batch is not None
                            else None
                        )
                        mb_lidar = (
                            jax_lidar_batch[mb_env_indices]
                            if jax_lidar_batch is not None
                            else None
                        )
                        mb_food_positions = (
                            jax_food_positions_batch[mb_env_indices]
                            if jax_food_positions_batch is not None
                            else None
                        )
                        mb_agent_positions = (
                            jax_agent_positions_batch[mb_env_indices]
                            if jax_agent_positions_batch is not None
                            else None
                        )
                        mb_target_snd = (
                            jax_target_snd_batch[mb_env_indices]
                            if jax_target_snd_batch is not None
                            else None
                        )
                        mb_env_context = (
                            jax_env_context_batch[mb_env_indices]
                            if jax_env_context_batch is not None
                            else None
                        )
                        mb_mask = (
                            jax_mask_batch[mb_env_indices]
                            if jax_mask_batch is not None
                            else None
                        )

                        # Call training step with mini-batch
                        policy_state, hn_state, critic_state, loss_info = (
                            _train_step_with_hn(
                                policy_state,
                                hn_state,
                                critic_state,
                                mb_obs,
                                mb_global_state,
                                mb_actions,
                                mb_old_log_probs,
                                mb_advantages,
                                mb_returns,
                                mb_task,
                                mb_context,
                                mb_lidar,
                                mb_food_positions,
                                mb_agent_positions,
                                mb_target_snd,
                                mb_env_context,
                                mb_mask,
                                mb_action_masks,
                                clip_epsilon,
                                current_entropy_coef,
                                value_loss_coef,
                                env_num_agents,
                                current_num_agents,
                                env_num_agents,
                                lora_scaling_factor,
                                diversity_scaling,
                                is_discrete=(scenario_name == "smax"),
                                old_values_batch=mb_old_values,
                                value_clip_range=value_clip_range,
                            )
                        )

                        # Accumulate loss info
                        if accumulated_loss_info is None:
                            accumulated_loss_info = {k: v for k, v in loss_info.items()}
                        else:
                            for k, v in loss_info.items():
                                accumulated_loss_info[k] += v

                    # Average accumulated losses
                    loss_info = {
                        k: v / num_minibatches for k, v in accumulated_loss_info.items()
                    }

                else:
                    # Call training step (full batch)
                    policy_state, hn_state, critic_state, loss_info = (
                        _train_step_with_hn(
                            policy_state,
                            hn_state,
                            critic_state,
                            obs_batch,
                            global_state_batch,
                            actions_batch,
                            old_log_probs_batch,
                            advantages_batch,
                            returns_batch,
                            jax_task_batch,
                            jax_context_batch,
                            jax_lidar_batch,
                            jax_food_positions_batch,
                            jax_agent_positions_batch,
                            jax_target_snd_batch,
                            jax_env_context_batch,
                            jax_mask_batch,
                            action_masks_batch,
                            clip_epsilon,
                            current_entropy_coef,
                            value_loss_coef,
                            env_num_agents,
                            current_num_agents,
                            env_num_agents,
                            lora_scaling_factor,
                            diversity_scaling,
                            is_discrete=(scenario_name == "smax"),
                            old_values_batch=old_values_batch,
                            value_clip_range=value_clip_range,
                        )
                    )
            elif use_dico:
                # Create agent_ids_batch for DiCo training
                # Shape: (batch_size,) where batch_size = num_steps * num_envs * env_num_agents
                batch_size = obs_batch.shape[0]
                num_timesteps = batch_size // (num_envs * env_num_agents)
                agent_ids_batch = jnp.tile(
                    jnp.arange(env_num_agents), num_timesteps * num_envs
                )

                # MEMORY FIX: Do NOT replicate global_state per agent
                # global_state_batch has shape (num_steps * num_envs, global_state_dim)
                # Keep it as is - the critic will use it directly
                # This saves memory by a factor of num_agents!

                # Split into mini-batches if requested
                if use_minibatches:
                    # Prepare data dictionary for mini-batching
                    # NOTE: obs, actions, etc. have shape (num_steps * num_envs * num_agents, ...)
                    # while global_state has shape (num_steps * num_envs, ...)
                    # We handle this in the minibatch creation
                    data_dict = {
                        "obs": obs_batch,
                        "global_state": global_state_batch,
                        "actions": actions_batch,
                        "old_log_probs": old_log_probs_batch,
                        "advantages": advantages_batch,
                        "returns": returns_batch,
                        "agent_ids": agent_ids_batch,
                        "action_masks": action_masks_batch,
                    }

                    # Create mini-batches for DICO
                    # Special handling: global_state needs to be sliced differently
                    minibatches = create_minibatches_dico(
                        data_dict,
                        num_minibatches,
                        env_num_agents,
                        shuffle=shuffle_minibatches,
                        rng_key=shuffle_rng,
                    )

                    # Accumulate losses across mini-batches
                    accumulated_loss_info = None

                    # ========================================================
                    # TIMING: Start minibatch loop
                    # ========================================================
                    minibatch_start_time = time.time()

                    for mb_idx, minibatch in enumerate(minibatches):
                        # Call training step with mini-batch. No block_until_ready here:
                        # the async dispatch lets XLA overlap host scheduling with device
                        # execution across the 32 minibatches × 10 PPO epochs.
                        policy_state, critic_state, loss_info = _train_step_dico(
                            policy_state,
                            critic_state,
                            minibatch["obs"],
                            minibatch["global_state"],
                            minibatch["actions"],
                            minibatch["old_log_probs"],
                            minibatch["advantages"],
                            minibatch["returns"],
                            minibatch["agent_ids"],
                            diversity_scaling,
                            minibatch["action_masks"],
                            clip_epsilon,
                            current_entropy_coef,
                            value_loss_coef,
                            env_num_agents,
                            current_num_agents,
                            env_num_agents,
                            is_discrete=(scenario_name == "smax"),
                        )

                        # Accumulate loss info
                        if accumulated_loss_info is None:
                            accumulated_loss_info = {k: v for k, v in loss_info.items()}
                        else:
                            for k, v in loss_info.items():
                                accumulated_loss_info[k] += v

                    # Average accumulated losses
                    loss_info = {
                        k: v / num_minibatches for k, v in accumulated_loss_info.items()
                    }

                else:
                    # Call training step (full batch)
                    policy_state, critic_state, loss_info = _train_step_dico(
                        policy_state,
                        critic_state,
                        obs_batch,
                        global_state_batch,  # NOT replicated
                        actions_batch,
                        old_log_probs_batch,
                        advantages_batch,
                        returns_batch,
                        agent_ids_batch,
                        diversity_scaling,
                        action_masks_batch,
                        clip_epsilon,
                        current_entropy_coef,
                        value_loss_coef,
                        env_num_agents,
                        current_num_agents,
                        env_num_agents,
                        is_discrete=(scenario_name == "smax"),
                    )
            else:
                # Split into mini-batches if requested
                if use_minibatches:
                    # Prepare data dictionary for mini-batching
                    data_dict = {
                        "obs": obs_batch,
                        "global_state": global_state_batch,
                        "actions": actions_batch,
                        "old_log_probs": old_log_probs_batch,
                        "advantages": advantages_batch,
                        "returns": returns_batch,
                        "action_masks": action_masks_batch,
                    }

                    # Create mini-batches
                    minibatches = create_minibatches(
                        data_dict,
                        num_minibatches,
                        shuffle=shuffle_minibatches,
                        rng_key=shuffle_rng,
                    )

                    # Accumulate losses across mini-batches
                    accumulated_loss_info = None

                    for mb_idx, minibatch in enumerate(minibatches):
                        # Call training step with mini-batch
                        policy_state, critic_state, loss_info = _train_step_no_hn(
                            policy_state,
                            critic_state,
                            minibatch["obs"],
                            minibatch["global_state"],
                            minibatch["actions"],
                            minibatch["old_log_probs"],
                            minibatch["advantages"],
                            minibatch["returns"],
                            adapters_dict,
                            minibatch["action_masks"],
                            clip_epsilon,
                            current_entropy_coef,
                            value_loss_coef,
                            env_num_agents,
                            current_num_agents,
                            env_num_agents,
                            is_discrete=(scenario_name == "smax"),
                        )

                        # Accumulate loss info
                        if accumulated_loss_info is None:
                            accumulated_loss_info = {k: v for k, v in loss_info.items()}
                        else:
                            for k, v in loss_info.items():
                                accumulated_loss_info[k] += v

                    # Average accumulated losses
                    loss_info = {
                        k: v / num_minibatches for k, v in accumulated_loss_info.items()
                    }

                else:
                    # Call training step (full batch)
                    policy_state, critic_state, loss_info = _train_step_no_hn(
                        policy_state,
                        critic_state,
                        obs_batch,
                        global_state_batch,
                        actions_batch,
                        old_log_probs_batch,
                        advantages_batch,
                        returns_batch,
                        adapters_dict,
                        action_masks_batch,
                        clip_epsilon,
                        current_entropy_coef,
                        value_loss_coef,
                        env_num_agents,
                        current_num_agents,
                        env_num_agents,
                        is_discrete=(scenario_name == "smax"),
                    )

        # Check for NaN in parameters after training step.
        # This walks every param leaf and forces one host sync per leaf, so we
        # only run it every log_interval (or when --debug-nan is set).
        nan_param_check = args.debug_nan or (episode % log_interval == 0)
        policy_params_flat = (
            jax.tree_util.tree_leaves(policy_state.params) if nan_param_check else []
        )
        if nan_param_check and any(jnp.isnan(p).any() for p in policy_params_flat):
            print(
                f"\nERROR: NaN detected in policy parameters after training at episode {episode}"
            )
            print(f"  Diversity scaling used: {diversity_scaling}")
            print(f"  Current SND MA: {current_snd_ma}")
            # Check if gradients had NaN (from loss_info if available)
            if "policy_grad_norm" in loss_info:
                grad_norm = loss_info["policy_grad_norm"]
                print(f"  Policy gradient norm: {grad_norm}")
                if jnp.isnan(grad_norm) or jnp.isinf(grad_norm):
                    print(
                        f"  WARNING: Gradient norm was NaN/Inf - gradients were replaced with zeros"
                    )
            # Get first few param norms to diagnose
            param_info = {}
            for k, v in list(jax.tree_util.tree_leaves_with_path(policy_state.params))[
                :5
            ]:
                param_info[k] = (
                    float(jnp.sqrt(jnp.sum(v**2))) if not jnp.isnan(v).any() else "NaN"
                )
            print(f"  Sample params: {param_info}")
            raise ValueError("Training failed due to NaN in parameters")

        # Track how much parameters changed
        if episode % log_interval == 0 or episode == 1:
            param_changes = jax.tree_util.tree_map(
                lambda x, y: jnp.abs(x - y).mean(),
                policy_state.params,
                initial_policy_params,
            )
            avg_param_change = jnp.mean(
                jnp.array(jax.tree_util.tree_leaves(param_changes))
            )
            max_param_change = jnp.max(
                jnp.array(jax.tree_util.tree_leaves(param_changes))
            )
            loss_info["avg_param_change"] = avg_param_change
            loss_info["max_param_change"] = max_param_change

        # ====================================================================
        # TIMING: End PPO updates, compute timing breakdown
        # ====================================================================
        ppo_time = time.time() - ppo_start_time
        total_episode_time = rollout_time + ppo_time

        # Store timing info for logging
        timing_info = {
            "rollout_time": rollout_time,
            "ppo_time": ppo_time,
            "total_time": total_episode_time,
            "rollout_pct": (rollout_time / total_episode_time) * 100,
            "ppo_pct": (ppo_time / total_episode_time) * 100,
        }

        # ====================================================================
        # Always-on ETA line. First 2 episodes include JIT compile and are
        # excluded from the rolling average so the projection reflects
        # steady-state cost.
        # ====================================================================
        if episode >= _eta_warmup_episodes:
            _eta_post_warmup_total += total_episode_time
            _eta_post_warmup_count += 1
            avg_ep = _eta_post_warmup_total / _eta_post_warmup_count
            remaining = num_episodes - episode - 1
            eta_sec = avg_ep * remaining
            eta_h = eta_sec / 3600.0
            elapsed_h = (time.time() - _train_loop_wall_start) / 3600.0
            print(
                f"[ETA] ep {episode}/{num_episodes} "
                f"| this_ep={total_episode_time:.2f}s "
                f"(rollout {rollout_time:.2f}s, ppo {ppo_time:.2f}s) "
                f"| avg_ep={avg_ep:.2f}s "
                f"| elapsed={elapsed_h:.2f}h "
                f"| ETA={eta_h:.2f}h ({eta_sec/60.0:.1f}min)"
            )
        else:
            print(
                f"[ETA] ep {episode}/{num_episodes} "
                f"| this_ep={total_episode_time:.2f}s "
                f"(rollout {rollout_time:.2f}s, ppo {ppo_time:.2f}s) "
                f"| warmup (excluded from ETA)"
            )

        # ====================================================================
        # Sample target SND for NEXT iteration (AFTER PPO update)
        # This creates forward-looking diversity control:
        # - Current iteration uses adapters from PREVIOUS iteration's target_snd
        # - SND is computed using NEXT iteration's target_snd
        # - Diversity control adjusts based on where we want to go, not where we are
        # ====================================================================
        sampler = None
        if "__main__" in sys.modules:
            main_module = sys.modules["__main__"]
            if hasattr(main_module, "get_sampler"):
                try:
                    sampler = main_module.get_sampler()
                except:
                    pass

        # Fallback: check if train_random_diversity is explicitly imported
        if sampler is None and "train_random_diversity" in sys.modules:
            try:
                import train_random_diversity

                sampler = train_random_diversity.get_sampler()
            except:
                pass

        # If we have a sampler, use it to randomize target_snd for NEXT iteration
        # Sample a single representative value that will be used for diversity control computation
        # (The actual per-env sampling happens at the start of the NEXT episode)
        if sampler is not None:
            target_snd = sampler.sample(np_rng)
            if episode % 10 == 0 or episode < 5:  # Print first few and every 10th
                print(
                    f"Episode {episode}: Sampled target_snd for NEXT iteration = {target_snd:.6f}"
                )
            # Log to wandb if enabled
            if use_wandb:
                wandb.log({"episode_target_snd_next": target_snd}, step=episode)

        # ====================================================================
        # Logging
        # ====================================================================
        if episode % log_interval == 0 or episode == 1:
            # Check for NaN gradients and log warning
            if loss_info.get("policy_grad_has_nan", False) or loss_info.get(
                "critic_grad_has_nan", False
            ):
                print(f"\nWARNING: NaN gradients detected at episode {episode}")
                print(
                    f"  Policy grads NaN: {loss_info.get('policy_grad_has_nan', False)}"
                )
                print(
                    f"  Critic grads NaN: {loss_info.get('critic_grad_has_nan', False)}"
                )
                print(f"  Policy loss: {loss_info.get('policy_loss', 'N/A')}")
                print(f"  Value loss: {loss_info.get('value_loss', 'N/A')}")
                print(f"  Diversity scaling: {diversity_scaling}")
                print(
                    f"  Gradients were replaced with zeros to prevent NaN propagation"
                )

            avg_reward = np.mean(episode_rewards)
            min_reward = np.min(episode_rewards)
            max_reward = np.max(episode_rewards)
            std_reward = np.std(episode_rewards)

            # Protect against NaN/inf in aggregated stats
            avg_reward = (
                0.0 if (np.isnan(avg_reward) or np.isinf(avg_reward)) else avg_reward
            )
            min_reward = (
                0.0 if (np.isnan(min_reward) or np.isinf(min_reward)) else min_reward
            )
            max_reward = (
                0.0 if (np.isnan(max_reward) or np.isinf(max_reward)) else max_reward
            )
            std_reward = (
                0.0 if (np.isnan(std_reward) or np.isinf(std_reward)) else std_reward
            )

            # Build log message - include std metrics only for continuous actions
            log_msg = (
                f"Episode {episode}/{num_episodes}: Avg Reward = {avg_reward:.3f}, "
                f"Policy Loss = {loss_info['policy_loss']:.4f}, "
                f"Value Loss = {loss_info['value_loss']:.4f}, "
                f"Entropy = {loss_info['entropy']:.4f}, "
            )
            if "mean_std" in loss_info:
                log_msg += (
                    f"Mean Std = {loss_info['mean_std']:.4f} "
                    f"[Min: {loss_info['min_std']:.4f}, Max: {loss_info['max_std']:.4f}], "
                )
            log_msg += f"Mean Ratio = {loss_info['mean_ratio']:.4f}"
            print(log_msg)

            # Log timing breakdown
            print(
                f"  [Timing] Rollout: {timing_info['rollout_time']:.2f}s ({timing_info['rollout_pct']:.1f}%), "
                f"PPO: {timing_info['ppo_time']:.2f}s ({timing_info['ppo_pct']:.1f}%), "
                f"Total: {timing_info['total_time']:.2f}s"
            )

            # Log diversity metrics if available (calculated even when diversity control is off)
            if diversity_stats is not None:
                diversity_control_active = diversity_stats.get(
                    "diversity_control_active", True
                )
                control_status = (
                    "ACTIVE" if diversity_control_active else "MONITORING ONLY"
                )
                print(f"  Diversity Metrics ({control_status}):")
                print(f"    SND Unscaled: {diversity_stats['snd_unscaled']:.8f}")
                print(f"    SND Scaled: {diversity_stats['snd_scaled']:.8f}")
                print(f"    SND Moving Avg: {diversity_stats['current_snd_ma']:.8f}")
                print(f"    Target SND: {diversity_stats['target_snd']:.8f}")
                if diversity_control_active:
                    # For DiCo: scaling applied once (to hetero component)
                    # For HyperLoRA: scaling applied twice (to A and B matrices)
                    scaling_factor = diversity_stats["diversity_scaling_used"]
                    if use_dico:
                        print(f"    Scaling Factor Applied: {scaling_factor:.8f}")
                    else:
                        print(f"    Scaling Factor Applied: {scaling_factor**2:.8f}")
                else:
                    print(f"    Scaling Factor: 1.0 (diversity control disabled)")
                # Log adapter-based SND if available
                if "adapter_snd" in diversity_stats:
                    print(f"    Adapter SND: {diversity_stats['adapter_snd']:.8f}")
                    if "adapter_snd_scaled" in diversity_stats:
                        print(
                            f"    Adapter SND Scaled: {diversity_stats['adapter_snd_scaled']:.8f}"
                        )
                    print(
                        f"    Adapter SND Buffer Size: {len(adapter_snd_buffer) if adapter_snd_buffer is not None else 0}"
                    )
                    # In mixed training, show expected size for current episode
                    if mixed_agent_training:
                        expected_size = (
                            2 * current_num_agents * num_envs
                        )  # max: initial + requery per agent
                        print(
                            f"    Expected Buffer Size (for {current_num_agents} agents): ~{current_num_agents * num_envs}-{expected_size}"
                        )

            # Log adapter effect metrics (HyperLoRA only)
            if use_hypernetwork and len(adapter_effect_norms_buffer) > 0:
                # Concatenate all timesteps: list of (batch_size,) -> (total_samples,)
                adapter_effects_all = jnp.concatenate(adapter_effect_norms_buffer)

                # Compute statistics over entire episode
                adapter_effect_mean = float(jnp.mean(adapter_effects_all))
                adapter_effect_std = float(jnp.std(adapter_effects_all))
                adapter_effect_min = float(jnp.min(adapter_effects_all))
                adapter_effect_max = float(jnp.max(adapter_effects_all))

                print(f"  Adapter Effect (||μ_with - μ_without||_2):")
                print(f"    Mean: {adapter_effect_mean:.6f}")
                print(f"    Std:  {adapter_effect_std:.6f}")
                print(f"    Min:  {adapter_effect_min:.6f}")
                print(f"    Max:  {adapter_effect_max:.6f}")

            # Visualize action outputs every 10 episodes using ACTUAL ROLLOUT ADAPTERS
            if use_diversity_control and diversity_stats is not None:
                if "adapter_snd" in diversity_stats:
                    # Visualize action outputs every 10 episodes using ACTUAL ROLLOUT ADAPTERS
                    if (
                        False
                    ):  # Disabled - was: episode > 0 and episode % 40 == 0 and rollout_adapters_unscaled is not None and rollout_adapters_scaled is not None and len(trajectory_data["obs"]) > 0
                        print(
                            f"    Generating action output visualization for episode {episode}"
                        )
                        print(
                            f"    Using {batch_size} actual rollout adapters (256 = 128 envs × 2 agents)"
                        )

                        # Create plots directory
                        plots_dir = os.path.join(log_dir, "adapter_snd_plots")
                        os.makedirs(plots_dir, exist_ok=True)

                        # Sample observations from trajectory for evaluation
                        vis_rng_key = jax.random.PRNGKey(episode)
                        trajectory_obs_array = jnp.array(
                            trajectory_data["obs"]
                        )  # (timesteps, batch_size, obs_dim)
                        num_timesteps = trajectory_obs_array.shape[0]

                        # Sample timesteps for evaluation (use up to 256 observations)
                        num_eval_obs = min(256, num_timesteps * batch_size)
                        if num_timesteps * batch_size > 256:
                            # Sample random timesteps
                            sample_key, vis_rng_key = jax.random.split(vis_rng_key)
                            sample_indices = jax.random.choice(
                                sample_key,
                                num_timesteps * batch_size,
                                shape=(num_eval_obs,),
                                replace=False,
                            )
                            # Flatten trajectory and sample
                            obs_flat = trajectory_obs_array.reshape(
                                -1, trajectory_obs_array.shape[-1]
                            )
                            eval_obs = obs_flat[sample_indices]
                        else:
                            # Use all observations
                            eval_obs = trajectory_obs_array.reshape(
                                -1, trajectory_obs_array.shape[-1]
                            )
                            num_eval_obs = eval_obs.shape[0]

                        print(
                            f"    Evaluating {batch_size} adapters on {num_eval_obs} observations"
                        )

                        # Evaluate UNSCALED adapters on observations
                        num_adapters = batch_size
                        obs_tiled = jnp.tile(
                            eval_obs, (num_adapters, 1)
                        )  # (num_adapters * num_eval_obs, obs_dim)

                        # Tile unscaled adapters to match observation batch
                        adapters_unscaled_tiled = {}
                        for key, value in rollout_adapters_unscaled.items():
                            value_repeated = jnp.repeat(value, num_eval_obs, axis=0)
                            adapters_unscaled_tiled[key] = value_repeated

                        # Get action means for unscaled adapters
                        if use_gru_policy:
                            batch_size_eval = num_adapters * num_eval_obs
                            init_hidden = shared_policy.initialize_carry(
                                batch_size_eval, shared_policy.gru_hidden_dim
                            )
                            obs_seq = obs_tiled[None, ...]
                            dones_seq = jnp.zeros((1, batch_size_eval), dtype=bool)
                            avail_seq = None
                            policy_x = (obs_seq, dones_seq, avail_seq)
                            _, output = shared_policy.apply(
                                {"params": policy_state.params},
                                init_hidden,
                                policy_x,
                                adapters_unscaled_tiled,
                            )
                            mean_seq, _ = output
                            action_means_unscaled = mean_seq[0]
                        else:
                            action_means_unscaled, _ = shared_policy.apply(
                                {"params": policy_state.params},
                                obs_tiled,
                                adapters_unscaled_tiled,
                            )

                        # Reshape: (num_adapters * num_eval_obs, action_dim) -> (num_adapters, num_eval_obs, action_dim)
                        action_dim = action_means_unscaled.shape[-1]
                        action_means_unscaled = action_means_unscaled.reshape(
                            num_adapters, num_eval_obs, action_dim
                        )

                        # Evaluate SCALED adapters on same observations
                        adapters_scaled_tiled = {}
                        for key, value in rollout_adapters_scaled.items():
                            value_repeated = jnp.repeat(value, num_eval_obs, axis=0)
                            adapters_scaled_tiled[key] = value_repeated

                        # Get action means for scaled adapters
                        if use_gru_policy:
                            _, output = shared_policy.apply(
                                {"params": policy_state.params},
                                init_hidden,
                                policy_x,
                                adapters_scaled_tiled,
                            )
                            mean_seq, _ = output
                            action_means_scaled = mean_seq[0]
                        else:
                            action_means_scaled, _ = shared_policy.apply(
                                {"params": policy_state.params},
                                obs_tiled,
                                adapters_scaled_tiled,
                            )

                        action_means_scaled = action_means_scaled.reshape(
                            num_adapters, num_eval_obs, action_dim
                        )

                        print(
                            f"    Action means shape: {action_means_unscaled.shape} (num_adapters={num_adapters}, eval_obs={num_eval_obs}, action_dim={action_dim})"
                        )
                        print(
                            f"    Current diversity_scaling: {float(diversity_scaling):.4f} (as used in actual rollout)"
                        )

                        # Reuse Adapter SND value from training (already computed)
                        # Note: Training SND uses 128 sampled adapters; visualization uses 256 rollout adapters
                        training_adapter_snd = diversity_stats["adapter_snd"]

                        # Process UNSCALED action means
                        action_means_unscaled_np = np.array(action_means_unscaled)
                        action_means_unscaled_squashed = np.tanh(
                            action_means_unscaled_np
                        )

                        # Process SCALED action means
                        action_means_scaled_np = np.array(action_means_scaled)
                        action_means_scaled_squashed = np.tanh(action_means_scaled_np)

                        print(
                            f"    Unscaled (raw) range: [{action_means_unscaled_np.min():.4f}, {action_means_unscaled_np.max():.4f}]"
                        )
                        print(
                            f"    Scaled (raw) range: [{action_means_scaled_np.min():.4f}, {action_means_scaled_np.max():.4f}]"
                        )

                        # Flatten all adapter actions for plotting
                        all_actions_unscaled_squashed = (
                            action_means_unscaled_squashed.reshape(-1, 2)
                        )
                        all_actions_unscaled_raw = action_means_unscaled_np.reshape(
                            -1, 2
                        )
                        all_actions_scaled_squashed = (
                            action_means_scaled_squashed.reshape(-1, 2)
                        )
                        all_actions_scaled_raw = action_means_scaled_np.reshape(-1, 2)

                        # Create 4 plots: unscaled squashed, unscaled raw, scaled squashed, scaled raw

                        # 1. Unscaled squashed actions plot
                        fig1, ax1 = plt.subplots(figsize=(10, 10))
                        ax1.scatter(
                            all_actions_unscaled_squashed[:, 0],
                            all_actions_unscaled_squashed[:, 1],
                            c="steelblue",
                            alpha=0.5,
                            s=20,
                        )
                        ax1.set_xlabel("Action Dimension 0", fontsize=12)
                        ax1.set_ylabel("Action Dimension 1", fontsize=12)
                        ax1.set_title(
                            f"Unscaled Actions (Squashed) - Episode {episode}\n{num_adapters} Rollout Adapters, {num_eval_obs} Obs\nTraining Adapter SND: {training_adapter_snd:.4f}",
                            fontsize=14,
                        )
                        ax1.grid(True, alpha=0.3)
                        ax1.set_xlim(-1.05, 1.05)
                        ax1.set_ylim(-1.05, 1.05)
                        plot_path_unscaled_squashed = os.path.join(
                            plots_dir, f"actions_unscaled_squashed_ep{episode:06d}.png"
                        )
                        plt.savefig(
                            plot_path_unscaled_squashed, dpi=150, bbox_inches="tight"
                        )
                        plt.close(fig1)

                        # 2. Unscaled raw actions plot
                        fig2, ax2 = plt.subplots(figsize=(10, 10))
                        ax2.scatter(
                            all_actions_unscaled_raw[:, 0],
                            all_actions_unscaled_raw[:, 1],
                            c="coral",
                            alpha=0.5,
                            s=20,
                        )
                        ax2.set_xlabel("Action Dimension 0", fontsize=12)
                        ax2.set_ylabel("Action Dimension 1", fontsize=12)
                        ax2.set_title(
                            f"Unscaled Actions (Raw) - Episode {episode}\n{num_adapters} Rollout Adapters, {num_eval_obs} Obs\nTraining Adapter SND: {training_adapter_snd:.4f}",
                            fontsize=14,
                        )
                        ax2.grid(True, alpha=0.3)
                        x_min, x_max = (
                            all_actions_unscaled_raw[:, 0].min(),
                            all_actions_unscaled_raw[:, 0].max(),
                        )
                        y_min, y_max = (
                            all_actions_unscaled_raw[:, 1].min(),
                            all_actions_unscaled_raw[:, 1].max(),
                        )
                        x_padding = max((x_max - x_min) * 0.1, 0.1)
                        y_padding = max((y_max - y_min) * 0.1, 0.1)
                        ax2.set_xlim(x_min - x_padding, x_max + x_padding)
                        ax2.set_ylim(y_min - y_padding, y_max + y_padding)
                        plot_path_unscaled_raw = os.path.join(
                            plots_dir, f"actions_unscaled_raw_ep{episode:06d}.png"
                        )
                        plt.savefig(
                            plot_path_unscaled_raw, dpi=150, bbox_inches="tight"
                        )
                        plt.close(fig2)

                        # 3. Scaled squashed actions plot
                        fig3, ax3 = plt.subplots(figsize=(10, 10))
                        ax3.scatter(
                            all_actions_scaled_squashed[:, 0],
                            all_actions_scaled_squashed[:, 1],
                            c="mediumseagreen",
                            alpha=0.5,
                            s=20,
                        )
                        ax3.set_xlabel("Action Dimension 0", fontsize=12)
                        ax3.set_ylabel("Action Dimension 1", fontsize=12)
                        ax3.set_title(
                            f"Scaled Actions (Squashed) - Episode {episode}\nScaling={float(diversity_scaling):.2f}, {num_adapters} Rollout Adapters, {num_eval_obs} Obs\nTraining Adapter SND: {training_adapter_snd:.4f}",
                            fontsize=14,
                        )
                        ax3.grid(True, alpha=0.3)
                        ax3.set_xlim(-1.05, 1.05)
                        ax3.set_ylim(-1.05, 1.05)
                        plot_path_scaled_squashed = os.path.join(
                            plots_dir, f"actions_scaled_squashed_ep{episode:06d}.png"
                        )
                        plt.savefig(
                            plot_path_scaled_squashed, dpi=150, bbox_inches="tight"
                        )
                        plt.close(fig3)

                        # 4. Scaled raw actions plot
                        fig4, ax4 = plt.subplots(figsize=(10, 10))
                        ax4.scatter(
                            all_actions_scaled_raw[:, 0],
                            all_actions_scaled_raw[:, 1],
                            c="orchid",
                            alpha=0.5,
                            s=20,
                        )
                        ax4.set_xlabel("Action Dimension 0", fontsize=12)
                        ax4.set_ylabel("Action Dimension 1", fontsize=12)
                        ax4.set_title(
                            f"Scaled Actions (Raw) - Episode {episode}\nScaling={float(diversity_scaling):.2f}, {num_adapters} Rollout Adapters, {num_eval_obs} Obs\nTraining Adapter SND: {training_adapter_snd:.4f}",
                            fontsize=14,
                        )
                        ax4.grid(True, alpha=0.3)
                        x_min, x_max = (
                            all_actions_scaled_raw[:, 0].min(),
                            all_actions_scaled_raw[:, 0].max(),
                        )
                        y_min, y_max = (
                            all_actions_scaled_raw[:, 1].min(),
                            all_actions_scaled_raw[:, 1].max(),
                        )
                        x_padding = max((x_max - x_min) * 0.1, 0.1)
                        y_padding = max((y_max - y_min) * 0.1, 0.1)
                        ax4.set_xlim(x_min - x_padding, x_max + x_padding)
                        ax4.set_ylim(y_min - y_padding, y_max + y_padding)
                        plot_path_scaled_raw = os.path.join(
                            plots_dir, f"actions_scaled_raw_ep{episode:06d}.png"
                        )
                        plt.savefig(plot_path_scaled_raw, dpi=150, bbox_inches="tight")
                        plt.close(fig4)

                        print(
                            f"    Saved unscaled plots: {plot_path_unscaled_squashed}, {plot_path_unscaled_raw}"
                        )
                        print(
                            f"    Saved scaled plots: {plot_path_scaled_squashed}, {plot_path_scaled_raw}"
                        )

            print(
                f"  Grad Norms - Policy: {loss_info['policy_grad_norm']:.6f}, "
                f"HN: {loss_info['hn_grad_norm']:.6f}, "
                f"Critic: {loss_info['critic_grad_norm']:.6f}"
            )
            if "adapter_norm" in loss_info:
                print(f"  Adapter Norm: {loss_info['adapter_norm']:.6f}")
            print(
                f"  Param Changes - Avg: {loss_info.get('avg_param_change', 0):.8f}, "
                f"Max: {loss_info.get('max_param_change', 0):.8f}"
            )
            print(
                f"  Advantages - Mean: {loss_info['advantages_mean']:.6f}, "
                f"Std: {loss_info['advantages_std']:.6f}, "
                f"Max: {loss_info['advantages_max']:.6f}"
            )

            # Save to log file
            if not args.no_logging:
                log_file = log_dir / "training_log.txt"
                with open(log_file, "a") as f:
                    f.write(
                        f"{episode},{avg_reward},{loss_info['policy_loss']:.6f},{loss_info['value_loss']:.6f},{loss_info['entropy']:.6f}\n"
                    )

            # Log to Weights & Biases
            if use_wandb:
                log_dict = {
                    "episode": episode,
                    "reward/mean": avg_reward,
                    "reward/min": min_reward,
                    "reward/max": max_reward,
                    "reward/std": std_reward,
                    "loss/policy": float(loss_info["policy_loss"]),
                    "loss/value": float(loss_info["value_loss"]),
                    "loss/entropy": float(loss_info["entropy"]),
                    "loss/approx_kl": float(loss_info["approx_kl"]),
                    "debug/mean_ratio": float(loss_info["mean_ratio"]),
                }

                # Add std metrics only for continuous actions
                if "mean_std" in loss_info:
                    log_dict["debug/mean_std"] = float(loss_info["mean_std"])

                # Add diversity control metrics (for both HyperLoRA and DiCo)
                # Also log for baseline (no hypernetwork) to verify SND is ~0
                if diversity_stats is not None:
                    log_dict["diversity_control/current_snd"] = diversity_stats[
                        "current_snd"
                    ]
                    log_dict["diversity_control/snd_unscaled"] = diversity_stats[
                        "snd_unscaled"
                    ]
                    log_dict["diversity_control/snd_scaled"] = diversity_stats[
                        "snd_scaled"
                    ]
                    log_dict["diversity_control/snd_moving_avg"] = diversity_stats[
                        "current_snd_ma"
                    ]
                    log_dict["diversity_control/target_snd"] = diversity_stats[
                        "target_snd"
                    ]
                    log_dict["diversity_control/scaling_factor"] = diversity_stats[
                        "diversity_scaling"
                    ]
                    # Log effective scaling factor (squared for HyperLoRA, not squared for DiCo)
                    scaling_factor = diversity_stats["diversity_scaling"]
                    if use_dico:
                        log_dict["diversity_control/scaling_factor_applied"] = (
                            scaling_factor
                        )
                    else:
                        log_dict["diversity_control/scaling_factor_applied"] = (
                            scaling_factor**2
                        )
                    log_dict["diversity_control/active"] = diversity_stats.get(
                        "diversity_control_active", True
                    )
                    # Add adapter-based SND if available
                    if "adapter_snd" in diversity_stats:
                        log_dict["diversity_control/adapter_snd"] = diversity_stats[
                            "adapter_snd"
                        ]
                        if adapter_snd_buffer is not None:
                            log_dict["diversity_control/adapter_snd_buffer_size"] = len(
                                adapter_snd_buffer
                            )
                    # Add scaled adapter-based SND if available
                    if "adapter_snd_scaled" in diversity_stats:
                        log_dict["diversity_control/adapter_snd_scaled"] = (
                            diversity_stats["adapter_snd_scaled"]
                        )

                # Add adapter effect metrics (HyperLoRA only)
                if use_hypernetwork and len(adapter_effect_norms_buffer) > 0:
                    adapter_effects_all = jnp.concatenate(adapter_effect_norms_buffer)
                    log_dict["adapter_effect/mean"] = float(
                        jnp.mean(adapter_effects_all)
                    )
                    log_dict["adapter_effect/std"] = float(jnp.std(adapter_effects_all))
                    log_dict["adapter_effect/min"] = float(jnp.min(adapter_effects_all))
                    log_dict["adapter_effect/max"] = float(jnp.max(adapter_effects_all))

                # Add football win rate metrics
                if scenario_name == "football" and hasattr(env, "get_win_rate"):
                    win_stats = env.get_win_rate()
                    log_dict["football/blue_goals"] = win_stats["blue_goals"]
                    log_dict["football/red_goals"] = win_stats["red_goals"]
                    log_dict["football/win_rate"] = win_stats["win_rate"]
                    log_dict["football/total_episodes"] = win_stats["total_episodes"]
                    # Reset stats for next logging interval
                    env.reset_stats()

                # Plot adapter impact distribution every 50 episodes
                if (
                    use_hypernetwork
                    and episode % 50 == 0
                    and len(adapter_effect_buffer) > 0
                ):
                    from snd import plot_action_distribution

                    plot_adapter_impact_distribution(
                        adapter_effect_buffer, episode, log_dir, use_wandb
                    )

                    # Plot backbone-only action distribution
                    if len(backbone_actions_buffer) > 0:
                        plot_action_distribution(
                            backbone_actions_buffer,
                            episode,
                            log_dir,
                            use_wandb,
                            plot_type="backbone_only",
                        )

                    # Plot combined (backbone + adapters) action distribution
                    if len(combined_actions_buffer) > 0:
                        plot_action_distribution(
                            combined_actions_buffer,
                            episode,
                            log_dir,
                            use_wandb,
                            plot_type="combined",
                        )

                wandb.log(log_dict, step=episode)

        # ====================================================================
        # Evaluation Rollout
        # ====================================================================
        if (
            episode % eval_interval == 0
            and episode > 0
            and episode >= eval_start_episode
        ):
            print(f"\n{'='*80}")
            print(f"Running evaluation at episode {episode}...")
            if use_hypernetwork and use_diversity_control:
                print(
                    f"  Using diversity control - Current SND MA: {current_snd_ma:.6f}"
                )
            print(f"{'='*80}")

            # For pressure_plate: set eval_mode to force left-side spawning during evaluation
            _eval_mode_set = (
                scenario_name == "pressure_plate"
                and hasattr(env, "scenario")
                and hasattr(env.scenario, "eval_mode")
            )
            if _eval_mode_set:
                env.scenario.eval_mode = True

            eval_metrics = run_quantitative_evaluation(
                env=env,
                policy_state=policy_state,
                hn_state=hn_state,
                shared_policy=shared_policy,
                hypernetwork=hypernetwork,
                num_agents=env_num_agents,  # Use env_num_agents (max_agents for mixed training)
                max_agents=max_agents,
                num_envs=num_envs,
                obs_dim=obs_dim,
                action_dim=action_dim,
                max_obs_dim=max_obs_dim,
                max_action_dim=max_action_dim,
                policy_hidden_dims=policy_hidden_dims,
                context_dim=context_dim,
                lidar_dim=lidar_dim,
                task_embed_dim=task_embed_dim,
                use_lidar_context=use_lidar_context,
                torch_device=torch_device,
                jax_device=jax_device,
                use_cuda=use_cuda,
                np_rng=np_rng,
                num_eval_episodes=num_eval_episodes,
                max_eval_steps=max_eval_steps,
                scenario_name=scenario_name,  # Pass scenario name for food position extraction
                randomize_capabilities=False,
                fixed_capabilities=(
                    # For dispersion_vmas: use first eval_max_speed for all agents (homogeneous)
                    # TODO: Iterate through all eval_max_speeds and average results
                    {
                        "max_speed": [config["env"].get("eval_max_speeds", [1.0])[0]]
                        * env_num_agents,
                    }
                    if scenario_name == "dispersion_vmas"
                    else (
                        # CRITICAL: For mixed training, use homogeneous capabilities matching training
                        # All agents get speed=0.5, lidar=0.5 (not the padded agents' values)
                        {
                            "speed": [0.5] * env_num_agents,
                            "lidar_range": [0.5] * env_num_agents,
                        }
                        if mixed_agent_training
                        else (
                            config["env"].get("fixed_capabilities")
                            if use_fixed_capabilities
                            else None
                        )
                    )
                ),
                verbose=True,  # Enable verbose to see food collection tracking
                calculate_snd_metric=use_hypernetwork,
                adaptive_hypernetwork=adaptive_hypernetwork,
                use_gru_policy=use_gru_policy,
                target_snd=target_snd,  # Target SND for adapter generation
                current_snd_ma=current_snd_ma,  # Current moving average for proper diversity scaling
                env_context_dim=env_context_dim,
                config=config,
            )

            # Restore eval_mode after quantitative evaluation
            if _eval_mode_set:
                env.scenario.eval_mode = False

            print_training_eval_results(episode, eval_metrics)

            # Log evaluation metrics to wandb
            if use_wandb:
                log_dict = {
                    "eval/completion_rate": eval_metrics["completion_rate"],
                    "eval/avg_episode_length": eval_metrics["avg_episode_length"],
                    "eval/avg_reward": eval_metrics["avg_reward"],
                }

                # Add SND if available
                if "snd" in eval_metrics and eval_metrics["snd"] is not None:
                    log_dict["eval/snd"] = eval_metrics["snd"]

                # Log the current SND MA and diversity scaling used during evaluation
                if use_hypernetwork and use_diversity_control:
                    log_dict["eval/current_snd_ma"] = current_snd_ma
                    # Log the diversity scaling used (sqrt(target_snd / current_snd_ma))
                    eval_diversity_scaling = float(
                        np.sqrt(target_snd / max(current_snd_ma, 1e-6))
                    )
                    log_dict["eval/diversity_scaling"] = np.clip(
                        eval_diversity_scaling, 0.001, np.sqrt(5.0)
                    )

                # Add reward decomposition if available
                if "reward_decomposition" in eval_metrics:
                    decomp = eval_metrics["reward_decomposition"]
                    if "food_collection" in decomp:
                        log_dict["eval/reward_food_collection"] = decomp[
                            "food_collection"
                        ]
                    if "shaping" in decomp:
                        log_dict["eval/reward_shaping"] = decomp["shaping"]
                    if "time_penalty" in decomp:
                        log_dict["eval/reward_time_penalty"] = decomp["time_penalty"]

                # Add collision metrics if available (for simple_tag)
                if "avg_first_collision_time" in eval_metrics:
                    log_dict["eval/avg_first_collision_time"] = eval_metrics[
                        "avg_first_collision_time"
                    ]
                if "avg_collisions_per_episode" in eval_metrics:
                    log_dict["eval/avg_collisions_per_episode"] = eval_metrics[
                        "avg_collisions_per_episode"
                    ]

                # Add SMAX win rate if available
                if "win_rate" in eval_metrics:
                    log_dict["eval/win_rate"] = eval_metrics["win_rate"]
                    log_dict["eval/wins"] = eval_metrics["wins"]
                    log_dict["eval/battles"] = eval_metrics["battles"]

                # Add food collection percentage if available (dispersion)
                if "food_found_percentage" in eval_metrics:
                    log_dict["eval/food_found_percentage"] = eval_metrics[
                        "food_found_percentage"
                    ]

                # Add agents at goal percentage if available (pressure_plate)
                if "avg_agents_at_goal_percentage" in eval_metrics:
                    log_dict["eval/avg_agents_at_goal_percentage"] = eval_metrics[
                        "avg_agents_at_goal_percentage"
                    ]
                    log_dict["eval/std_agents_at_goal_percentage"] = eval_metrics[
                        "std_agents_at_goal_percentage"
                    ]

                # Add football eval metrics if available
                if "football_win_rate" in eval_metrics:
                    log_dict["eval/football_win_rate"] = eval_metrics[
                        "football_win_rate"
                    ]
                    log_dict["eval/football_blue_goals"] = eval_metrics[
                        "football_blue_goals"
                    ]
                    log_dict["eval/football_red_goals"] = eval_metrics[
                        "football_red_goals"
                    ]
                    log_dict["eval/football_total_goals"] = eval_metrics[
                        "football_total_goals"
                    ]

                wandb.log(log_dict, step=episode)

        # ====================================================================
        # Save Checkpoint
        # ====================================================================
        if (
            not args.no_logging
            and episode % config["logging"]["save_interval"] == 0
            and episode > 0
        ):
            print(f"Saving checkpoint at episode {episode}...")
            checkpoint_path = checkpoint_dir / f"checkpoint_{episode}.npz"
            checkpoint_data = {
                "policy_params": policy_state.params,
                "critic_params": critic_state.params,
                "episode": episode,
            }
            if use_hypernetwork:
                checkpoint_data["hn_params"] = hn_state.params
            # Save current_snd_ma for proper diversity scaling during evaluation
            # for both HyperLoRA and DiCo runs when diversity control is enabled.
            if use_diversity_control and "current_snd_ma" in locals():
                checkpoint_data["current_snd_ma"] = current_snd_ma
            np.savez(checkpoint_path, **checkpoint_data)

    print("\nTraining completed!")

    # Save final checkpoint
    if not args.no_logging:
        final_checkpoint_path = checkpoint_dir / "final_checkpoint.npz"
        checkpoint_data = {
            "policy_params": policy_state.params,
            "critic_params": critic_state.params,
            "episode": num_episodes,
        }
        if use_hypernetwork:
            checkpoint_data["hn_params"] = hn_state.params
        # Save current_snd_ma for proper diversity scaling during evaluation
        # for both HyperLoRA and DiCo runs when diversity control is enabled.
        if use_diversity_control and "current_snd_ma" in locals():
            checkpoint_data["current_snd_ma"] = current_snd_ma
        np.savez(final_checkpoint_path, **checkpoint_data)
        print(f"Final checkpoint saved to: {final_checkpoint_path}")

    # ====================================================================
    # Post-Training Evaluation Sweep over Target SND Values
    # ====================================================================
    eval_snds = getattr(args, "eval_snds", None)
    if eval_snds and use_hypernetwork:
        print(f"\n{'='*80}")
        print(f"POST-TRAINING EVALUATION SWEEP")
        print(f"Evaluating at target SND values: {eval_snds}")
        print(f"{'='*80}")

        # For pressure_plate: set eval_mode during evaluation
        _sweep_eval_mode_set = (
            scenario_name == "pressure_plate"
            and hasattr(env, "scenario")
            and hasattr(env.scenario, "eval_mode")
        )

        sweep_results = []
        for eval_target_snd in eval_snds:
            print(f"\n--- Evaluating with target_snd = {eval_target_snd:.2f} ---")

            if _sweep_eval_mode_set:
                env.scenario.eval_mode = True

            eval_metrics = run_quantitative_evaluation(
                env=env,
                policy_state=policy_state,
                hn_state=hn_state,
                shared_policy=shared_policy,
                hypernetwork=hypernetwork,
                num_agents=env_num_agents,
                max_agents=max_agents,
                num_envs=num_envs,
                obs_dim=obs_dim,
                action_dim=action_dim,
                max_obs_dim=max_obs_dim,
                max_action_dim=max_action_dim,
                policy_hidden_dims=policy_hidden_dims,
                context_dim=context_dim,
                lidar_dim=lidar_dim,
                task_embed_dim=task_embed_dim,
                use_lidar_context=use_lidar_context,
                torch_device=torch_device,
                jax_device=jax_device,
                use_cuda=use_cuda,
                np_rng=np_rng,
                num_eval_episodes=num_eval_episodes,
                max_eval_steps=max_eval_steps,
                scenario_name=scenario_name,
                randomize_capabilities=False,
                fixed_capabilities=(
                    {
                        "max_speed": [config["env"].get("eval_max_speeds", [1.0])[0]]
                        * env_num_agents,
                    }
                    if scenario_name == "dispersion_vmas"
                    else (
                        {
                            "speed": [0.5] * env_num_agents,
                            "lidar_range": [0.5] * env_num_agents,
                        }
                        if mixed_agent_training
                        else (
                            config["env"].get("fixed_capabilities")
                            if use_fixed_capabilities
                            else None
                        )
                    )
                ),
                verbose=False,
                calculate_snd_metric=True,
                adaptive_hypernetwork=adaptive_hypernetwork,
                use_gru_policy=use_gru_policy,
                target_snd=eval_target_snd,
                current_snd_ma=eval_target_snd,  # scaling = sqrt(target/current) = 1.0
                env_context_dim=env_context_dim,
                config=config,
            )

            if _sweep_eval_mode_set:
                env.scenario.eval_mode = False

            sweep_results.append((eval_target_snd, eval_metrics))

            # Log to wandb
            if use_wandb:
                sweep_log = {
                    f"eval_sweep/target_snd_{eval_target_snd:.1f}/completion_rate": eval_metrics[
                        "completion_rate"
                    ],
                    f"eval_sweep/target_snd_{eval_target_snd:.1f}/avg_reward": eval_metrics[
                        "avg_reward"
                    ],
                }
                if "snd" in eval_metrics and eval_metrics["snd"] is not None:
                    sweep_log[f"eval_sweep/target_snd_{eval_target_snd:.1f}/snd"] = (
                        eval_metrics["snd"]
                    )
                if "food_found_percentage" in eval_metrics:
                    sweep_log[
                        f"eval_sweep/target_snd_{eval_target_snd:.1f}/food_found_pct"
                    ] = eval_metrics["food_found_percentage"]
                if "avg_agents_at_goal_percentage" in eval_metrics:
                    sweep_log[
                        f"eval_sweep/target_snd_{eval_target_snd:.1f}/agents_at_goal_pct"
                    ] = eval_metrics["avg_agents_at_goal_percentage"]
                    sweep_log[
                        f"eval_sweep/target_snd_{eval_target_snd:.1f}/agents_at_goal_std"
                    ] = eval_metrics["std_agents_at_goal_percentage"]
                wandb.log(sweep_log)

        # Print summary table
        print(f"\n{'='*80}")
        print(f"POST-TRAINING EVALUATION SUMMARY")
        print(f"{'='*80}")
        header = f"{'Target SND':>12} | {'Avg Reward':>12} | {'Completion':>12}"
        has_food = any("food_found_percentage" in m for _, m in sweep_results)
        has_agents_at_goal = any(
            "avg_agents_at_goal_percentage" in m for _, m in sweep_results
        )
        has_snd = any("snd" in m and m.get("snd") is not None for _, m in sweep_results)
        if has_food:
            header += f" | {'Food Found':>12}"
        if has_agents_at_goal:
            header += f" | {'Agents@Goal':>12}"
        if has_snd:
            header += f" | {'Measured SND':>12}"
        print(header)
        print("-" * len(header))
        for snd_val, metrics in sweep_results:
            row = f"{snd_val:>12.2f} | {metrics['avg_reward']:>12.3f} | {metrics['completion_rate']:>11.2f}%"
            if has_food:
                food_pct = metrics.get("food_found_percentage", 0.0)
                row += f" | {food_pct:>11.2f}%"
            if has_agents_at_goal:
                agents_pct = metrics.get("avg_agents_at_goal_percentage", 0.0)
                row += f" | {agents_pct:>11.2f}%"
            if has_snd:
                measured_snd = metrics.get("snd")
                if measured_snd is not None:
                    row += f" | {measured_snd:>12.6f}"
                else:
                    row += f" | {'N/A':>12}"
            print(row)
        print(f"{'='*80}\n")

    # Generate visualization GIF
    if args.render_final_policy:
        print("\nGenerating visualization GIF of final policy...")

        # CRITICAL: Set capabilities before rendering GIF
        # For mixed training, use homogeneous capabilities (same as training)
        # For fixed capabilities, use config values
        if mixed_agent_training:
            # Homogeneous: all agents get speed=0.5, lidar=0.5
            gif_agent_speeds = [0.5] * env_num_agents
            gif_agent_lidar_ranges = [0.5] * env_num_agents
            gif_agent_force_multipliers = [0.5] * env_num_agents
            print(f"  Using homogeneous capabilities for all {env_num_agents} agents:")
            print(f"    speed=0.5, lidar_range=0.5")
        elif use_fixed_capabilities:
            # Use config's fixed capabilities
            gif_agent_speeds = default_speeds
            gif_agent_lidar_ranges = (
                default_lidar_ranges if default_lidar_ranges else [0.5] * env_num_agents
            )
            gif_agent_force_multipliers = (
                default_force_multipliers
                if default_force_multipliers
                else [0.5] * env_num_agents
            )
            print(f"  Using fixed capabilities from config")
        else:
            # Use default values
            gif_agent_speeds = [1.0] * env_num_agents
            gif_agent_lidar_ranges = [0.5] * env_num_agents
            gif_agent_force_multipliers = [0.5] * env_num_agents
            print(f"  Using default capabilities")

        # Update environment capabilities
        if scenario_name == "simple_tag":
            # For simple_tag, use the config capabilities
            fixed_caps = config["env"].get("fixed_capabilities", {})
            gif_capabilities = {
                "adversary_speeds": fixed_caps.get(
                    "adversary_speeds", [1.0] * config["env"].get("num_adversaries", 3)
                ),
                "agent_speeds": fixed_caps.get(
                    "agent_speeds", [1.3] * config["env"].get("num_agents", 1)
                ),
                "adversary_lidar_ranges": fixed_caps.get(
                    "adversary_lidar_ranges",
                    [0.5] * config["env"].get("num_adversaries", 3),
                ),
                "agent_lidar_ranges": fixed_caps.get(
                    "agent_lidar_ranges", [0.6] * config["env"].get("num_agents", 1)
                ),
            }
            if hasattr(env, "scenario") and hasattr(
                env.scenario, "update_agent_capabilities"
            ):
                env.scenario.update_agent_capabilities(gif_capabilities)
        elif scenario_name == "reverse_transport":
            # Explicit reverse_transport branch: pass "force_multiplier" directly
            # from default_force_multipliers (from config) so it is never silently
            # overridden by the lidar_range fallback in the generic branch below.
            gif_capabilities = {
                "speed": gif_agent_speeds,
                "force_multiplier": gif_agent_force_multipliers,
            }
            if hasattr(env, "scenario") and hasattr(
                env.scenario, "update_agent_capabilities"
            ):
                env.scenario.update_agent_capabilities(gif_capabilities)

            # CRITICAL: Update package mass before generating GIF
            package_mass = config["env"].get("package_mass", 50)
            if hasattr(env, "scenario"):
                # Update scenario's stored package_mass
                env.scenario.package_mass = package_mass
                # Update the actual Landmark's mass attribute
                if hasattr(env.scenario, "package"):
                    env.scenario.package.mass = package_mass

            print(
                f"  reverse_transport capabilities applied: speeds={gif_agent_speeds}, force_multipliers={gif_agent_force_multipliers}"
            )
            print(f"  Package mass: {package_mass} (updated in environment)")
        elif scenario_name not in ["football", "smax"]:
            gif_capabilities = {
                "speed": gif_agent_speeds,
                "lidar_range": gif_agent_lidar_ranges,
            }
            if hasattr(env, "scenario") and hasattr(
                env.scenario, "update_agent_capabilities"
            ):
                env.scenario.update_agent_capabilities(gif_capabilities)

        # For pressure_plate: restrict agent spawning to left side during GIF rendering
        _gif_eval_mode_set = (
            scenario_name == "pressure_plate"
            and hasattr(env, "scenario")
            and hasattr(env.scenario, "eval_mode")
        )
        if _gif_eval_mode_set:
            env.scenario.eval_mode = True

        gif_path = generate_policy_gif(
            env=env,
            policy_state=policy_state,
            hn_state=hn_state if use_hypernetwork else None,
            adapters_dict=adapters_dict if not use_hypernetwork else None,
            config=config,
            checkpoint_dir=checkpoint_dir,
            n_steps=args.gif_steps if args.gif_steps is not None else max_eval_steps,
            use_hypernetwork=use_hypernetwork,
            use_dico=use_dico,
            adaptive_hypernetwork=adaptive_hypernetwork,
            num_agents=env_num_agents,  # Use env_num_agents (max_agents for mixed training)
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_dim=context_dim,
            task_embed_dim=task_embed_dim,
            lidar_dim=lidar_dim,
            env_context_dim=env_context_dim,
            target_snd=target_snd,
            use_cuda=use_cuda,
            jax_device=jax_device,
            torch_device=torch_device,
            get_static_adapters_fn=_get_static_adapters,
            get_actions_fn=(
                _get_actions_dico
                if use_dico
                else (_get_actions_gru if use_gru_policy else _get_actions)
            ),
            use_gru_policy=use_gru_policy,
            gru_hidden_dim=gru_hidden_dim if use_gru_policy else None,
            max_agents=max_agents,
        )

        # Restore eval_mode after GIF rendering
        if _gif_eval_mode_set:
            env.scenario.eval_mode = False

        if gif_path:
            print(f"GIF saved to: {gif_path}")

    # Close wandb run
    if use_wandb:
        wandb.finish()
        print("Weights & Biases run finished")


if __name__ == "__main__":
    main()
