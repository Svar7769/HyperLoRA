"""
Evaluation script for HyperLoRA and Independent Policies.

This script loads a checkpoint and:
1. Auto-detects checkpoint type (HyperLoRA vs Independent Policies)
2. Runs quantitative evaluation (completion rate, avg episode length, avg reward)
3. Optionally generates a GIF visualization of the trained policy

Use this to test generalization to different numbers of agents.

Usage:
    python evaluate.py --num-agents 2
    python evaluate.py --num-agents 4 --num-eval-episodes 50
    python evaluate.py --num-agents 6 --no-gif
    python evaluate.py --checkpoint checkpoints/hyperlora_vmas_20251117_144925
"""

import argparse
import functools
import sys
from pathlib import Path
import numpy as np
import torch
import jax
import jax.numpy as jnp
import yaml

from env_setup import make_vmas_env
from lora_policy import LoRAPolicy
from dico_policy import DiCoPolicy, DiCoHomogeneousPolicy
from cash_policy import CASHPolicy
from hypernetwork import Hypernetwork
from render_gif import generate_policy_gif
from snd import (
    calculate_snd,
    calculate_snd_dico,
    calculate_snd_independent,
    calculate_snd_statistics,
    calculate_adapter_snd,
)
from flax.training.train_state import TrainState
from flax import linen as nn
from typing import Sequence
import optax

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


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


# Actor class for independent policies (no hypernetwork)
class Actor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (64, 64)
    log_std_min: float = -2.0
    log_std_max: float = 1.0

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate HyperLoRA policy")

    parser.add_argument(
        "--num-agents",
        type=int,
        default=None,
        help="Number of agents (default: use checkpoint config)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (default: use latest)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario name (dispersion, simple_tag, etc.). If not specified, inferred from checkpoint or config",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20)",
    )
    parser.add_argument(
        "--max-eval-steps",
        type=int,
        default=200,
        help="Maximum steps per evaluation episode (default: 100)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=32,
        help="Number of parallel environments for evaluation (default: 32)",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF generation (only run quantitative evaluation)",
    )
    parser.add_argument(
        "--gif-steps",
        type=int,
        default=200,
        help="Number of steps for GIF visualization (default: 100)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for agent capabilities (default: 123)",
    )
    parser.add_argument(
        "--randomize-capabilities",
        action="store_true",
        help="Randomize agent capabilities each episode (default: use fixed)",
    )
    parser.add_argument(
        "--fixed-speeds",
        type=str,
        default=None,
        help="Comma-separated list of fixed speeds for each agent (e.g., '0.5,1.5,0.5,1.5')",
    )
    parser.add_argument(
        "--fixed-lidar",
        type=str,
        default=None,
        help="Comma-separated list of fixed lidar ranges for each agent (e.g., '0.3,0.3,0.3,0.3')",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help=(
            "Load a specific intermediate checkpoint instead of the final one. "
            "E.g. --checkpoint-step 200 loads checkpoint_200.npz from the checkpoint dir."
        ),
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="Skip quantitative evaluation and only generate the GIF visualization",
    )
    parser.add_argument(
        "--target-snd",
        type=float,
        default=None,
        help="Target SND value for adapter generation (overrides config). E.g., --target-snd 0.08",
    )
    parser.add_argument(
        "--current-snd-ma",
        type=float,
        default=None,
        help="Current SND moving average from training (for proper diversity scaling). If not specified, attempts to load from checkpoint.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Custom name for the wandb run (e.g., 'eval_dispersion_10agents')",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable wandb logging",
    )

    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the latest checkpoint directory."""
    checkpoints = sorted(checkpoint_dir.glob("hyperlora_vmas_*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def extract_food_positions_from_obs(obs_list, num_agents, scenario_name):
    """
    Extract relative positions of ALL food items for each agent from observations.

    For dispersion_vmas with global observability:
    - Observation structure: pos(2) + vel(2) + [food_0(3) + food_1(3) + ... + food_N(3)]
    - Each food entry: [rel_x, rel_y, eaten_status]
    - We concatenate [rel_x, rel_y] for ALL N food items per agent

    Returns: jnp.ndarray of shape (num_envs, num_agents, num_agents*2) or None
    """
    import jax.numpy as jnp
    import numpy as np

    if scenario_name != "dispersion_vmas":
        return None

    all_agent_food_positions = []
    for agent_obs in obs_list:  # iterate over each agent's observation
        # Extract [rel_x, rel_y] for all food items k in range(num_agents)
        food_rel_positions = []
        for k in range(num_agents):
            food_start_idx = 4 + k * 3
            food_pos_k = agent_obs[
                :, food_start_idx : food_start_idx + 2
            ]  # (num_envs, 2)
            food_pos_k_np = (
                food_pos_k.cpu().numpy()
                if food_pos_k.requires_grad
                else food_pos_k.detach().cpu().numpy()
            )
            food_rel_positions.append(food_pos_k_np)
        # Concatenate all food positions: (num_envs, num_agents*2)
        all_food = np.concatenate(food_rel_positions, axis=-1)
        all_agent_food_positions.append(all_food)

    # Stack to (num_agents, num_envs, num_agents*2) then transpose to (num_envs, num_agents, num_agents*2)
    food_positions_array = jnp.stack(all_agent_food_positions, axis=0)
    food_positions_batch = jnp.transpose(food_positions_array, (1, 0, 2))

    return food_positions_batch


def extract_agent_positions_from_obs(obs_list, num_agents, scenario_name):
    """
    Extract agent positions from observations.

    For dispersion_vmas and wind_flocking_position:
    - Observation structure: pos(2) + vel(2) + [...]
    - Agent position is at indices [0:2]

    Returns: jnp.ndarray of shape (num_envs, num_agents, 2) containing RELATIVE positions
             (relative to center of mass) or None if scenario doesn't use positions
    """
    import jax.numpy as jnp

    if scenario_name not in ["dispersion_vmas", "wind_flocking_position"]:
        return None

    agent_positions = []
    for agent_idx in range(num_agents):
        agent_obs = obs_list[agent_idx]  # (num_envs, obs_dim) - PyTorch tensor
        agent_pos = agent_obs[:, :2]  # (num_envs, 2)

        agent_pos_np = (
            agent_pos.cpu().numpy()
            if agent_pos.requires_grad
            else agent_pos.detach().cpu().numpy()
        )
        agent_positions.append(agent_pos_np)

    # Stack to (num_agents, num_envs, 2) then transpose to (num_envs, num_agents, 2)
    agent_positions_array = jnp.stack(agent_positions, axis=0)
    agent_positions_batch = jnp.transpose(agent_positions_array, (1, 0, 2))

    # Compute center of mass for each environment
    # Shape: (num_envs, 2)
    center_of_mass = jnp.mean(agent_positions_batch, axis=1)

    # Compute relative positions (relative to center of mass)
    # This ensures each agent gets different inputs to hypernetwork
    # Shape: (num_envs, num_agents, 2)
    relative_positions = agent_positions_batch - center_of_mass[:, None, :]

    return relative_positions


def run_quantitative_evaluation(
    env,
    policy_state,
    hn_state,
    shared_policy,
    hypernetwork,
    num_agents,
    num_envs,
    obs_dim,
    action_dim,
    policy_hidden_dims,
    context_dim,
    lidar_dim,
    task_embed_dim,
    use_lidar_context,
    torch_device,
    jax_device,
    use_cuda,
    np_rng,
    num_eval_episodes=20,
    max_eval_steps=200,
    scenario_name=None,
    randomize_capabilities=False,
    fixed_capabilities=None,
    verbose=True,
    calculate_snd_metric=False,
    adaptive_hypernetwork=False,
    max_agents=None,
    max_obs_dim=None,
    max_action_dim=None,
    use_gru_policy=False,
    package_mass=None,
    target_snd=0.01,
    current_snd_ma=None,
    env_context_dim=0,
    config=None,
):
    """
    Run quantitative evaluation and compute metrics.

    Args:
        fixed_capabilities: Optional dict with 'speeds' and 'lidar_ranges' lists.
                           If None, uses default pattern [0.5, 1.5] and [0.3, 0.7].
        verbose: Whether to print progress messages (default: True)
        calculate_snd_metric: Whether to calculate and return SND (default: False)
        adaptive_hypernetwork: Whether to requery hypernetwork when food enters lidar range (default: False)
        package_mass: Package mass for reverse_transport scenario (default: None)
        target_snd: Target SND value for diversity control (default: 0.01)
        current_snd_ma: Current SND moving average from training (default: None, uses target_snd)
        env_context_dim: Dimension of environment context (for pressure_plate: 9 with agent IDs, 7 without, reverse_transport: 3, default: 0)
        config: Configuration dictionary with model settings (required for pressure_plate env context)

    Returns:
        dict: Evaluation metrics including completion_rate, avg_episode_length, avg_reward,
              and optionally 'snd' if calculate_snd_metric=True
    """
    import jax
    import jax.numpy as jnp

    # Check if diversity control is enabled in config
    use_diversity_control = (
        config["training"].get("use_diversity_control", False) if config else False
    )

    # Check if adapter SND is enabled (matches training method)
    use_adapter_snd = (
        config["training"].get("use_adapter_snd", False) if config else False
    )
    adapter_snd_sample_size = (
        config["training"].get("adapter_snd_sample_size", 128) if config else 128
    )

    # Initialize SND tracking variables for proper diversity control
    # During training, adapters are scaled by sqrt(target_snd / current_snd_ma)
    # and the SND MA is updated after each adapter generation
    if current_snd_ma is None:
        current_snd_ma = target_snd  # Initialize with target

    if use_diversity_control:
        # Get SND moving average coefficient from config (default 0.9 like in training)
        snd_ma_coef = (
            config["training"].get("snd_moving_average_coef", 0.9) if config else 0.9
        )
        min_snd_floor = (
            config["training"].get("min_snd_floor", 1e-6) if config else 1e-6
        )
        max_scaling = (
            config["training"].get("max_diversity_scaling", 5.0) if config else 5.0
        )

        # Initial diversity scaling (will be updated after each adapter generation)
        diversity_scaling = float(
            np.sqrt(target_snd / max(current_snd_ma, min_snd_floor))
        )
        diversity_scaling = float(
            np.clip(diversity_scaling, 0.001, np.sqrt(max_scaling))
        )

        if verbose:
            print(f"\n{'='*80}")
            print("Starting quantitative evaluation with dynamic diversity control...")
            print(
                f"  Diversity method: {'Adapter SND' if use_adapter_snd else 'Action SND'}"
            )
            print(f"  Target SND: {target_snd:.6f}")
            print(f"  Initial SND MA: {current_snd_ma:.6f}")
            print(f"  SND MA coefficient: {snd_ma_coef}")
            print(f"  Min SND floor: {min_snd_floor:.6e}")
            print(f"  Max diversity scaling (s²): {max_scaling:.1f}")
            if use_adapter_snd:
                print(f"  Adapter SND sample size: {adapter_snd_sample_size}")
            print(f"  Initial Diversity Scaling: {diversity_scaling:.6f}")
            print(f"{'='*80}\n")
    else:
        # Diversity control disabled - use neutral scaling
        diversity_scaling = 1.0
        snd_ma_coef = 0.9  # Not used but set for consistency
        min_snd_floor = (
            config["training"].get("min_snd_floor", 1e-6) if config else 1e-6
        )
        max_scaling = (
            config["training"].get("max_diversity_scaling", 5.0) if config else 5.0
        )

        if verbose:
            print(f"\n{'='*80}")
            print("Starting quantitative evaluation (diversity control DISABLED)...")
            print(f"  Using neutral diversity scaling: {diversity_scaling:.6f}")
            print(f"{'='*80}\n")

    # Set default values for max dimensions if not provided
    if max_obs_dim is None:
        max_obs_dim = obs_dim
    if max_action_dim is None:
        max_action_dim = action_dim
    if max_agents is None:
        max_agents = num_agents

    # Extract context configuration from config (for dispersion_vmas and other scenarios)
    use_capability_context = (
        config["model"].get("use_capability_context", True) if config else True
    )
    use_onehot_context = (
        config["model"].get("use_onehot_context", True) if config else True
    )
    use_positional_context = (
        config["model"].get("use_positional_context", False) if config else False
    )
    positional_encoding_dim = (
        config["model"].get("positional_encoding_dim", 16) if config else 16
    )

    # Define JIT-compiled helper functions
    use_hypernetwork = hn_state is not None and hypernetwork is not None

    policy_class_name = type(shared_policy).__name__
    use_cash = (
        bool(config["model"].get("use_cash", False))
        if config is not None
        else policy_class_name == "CASHPolicy"
    )

    # Check if using DiCo policy. Do not infer DiCo from generic num_agents,
    # because CASH policy also has a num_agents attribute.
    if config is not None:
        use_dico = bool(config["model"].get("use_dico", False))
    else:
        use_dico = policy_class_name.startswith("DiCo")
    use_dico = use_dico and (not use_cash)

    # Check if using GRU policy
    use_gru_policy = hasattr(shared_policy, "gru_hidden_dim")

    if use_hypernetwork:

        @functools.partial(
            jax.jit, static_argnums=(3, 4)
        )  # batch_num_envs and batch_num_agents are static
        def _get_static_adapters(
            hn_params,
            task_batch,
            context_batch,
            batch_num_envs,
            batch_num_agents,
            lidar_batch=None,
            food_positions_batch=None,
            agent_positions_batch=None,
            env_context_batch=None,
            diversity_scaling=1.0,
            target_snd_value=0.01,
        ):
            """Generate LoRA adapters for all agents in the batch."""
            # Use real food positions if available, otherwise zeros/None
            food_position_dim = hypernetwork.food_position_dim
            if food_position_dim > 0:
                if food_positions_batch is not None:
                    food_batch = food_positions_batch
                else:
                    batch_shape = (batch_num_envs, batch_num_agents)
                    food_batch = jnp.zeros((*batch_shape, food_position_dim))
            else:
                food_batch = None

            # Use real agent positions if available, otherwise zeros/None
            agent_position_dim = hypernetwork.agent_position_dim
            if agent_position_dim > 0:
                if agent_positions_batch is not None:
                    agent_pos_batch = agent_positions_batch
                else:
                    batch_shape = (batch_num_envs, batch_num_agents)
                    agent_pos_batch = jnp.zeros((*batch_shape, agent_position_dim))
            else:
                agent_pos_batch = None

            # Use real environment context if available, otherwise zeros/None
            env_context_dim = hypernetwork.env_context_dim
            if env_context_dim > 0:
                if env_context_batch is not None:
                    env_ctx_batch = env_context_batch
                else:
                    batch_shape = (batch_num_envs, batch_num_agents)
                    env_ctx_batch = jnp.zeros((*batch_shape, env_context_dim))
            else:
                env_ctx_batch = None

            # Create target_snd batch (same value for all agents)
            target_snd_dim = hypernetwork.target_snd_dim
            if target_snd_dim > 0:
                batch_shape = (batch_num_envs, batch_num_agents)
                target_snd_batch = jnp.full(
                    (*batch_shape, target_snd_dim), target_snd_value, dtype=jnp.float32
                )
            else:
                target_snd_batch = None

            return hypernetwork.apply(
                {"params": hn_params},
                task_batch,
                context_batch,
                lidar_batch,
                food_batch,
                agent_pos_batch,
                target_snd_batch,
                env_ctx_batch,
                mask=None,
                diversity_scaling=diversity_scaling,
            )

        if use_gru_policy:

            @jax.jit
            def _get_actions_discrete_gru(
                policy_params, obs_batch, adapters_dict, hidden_states, avail_actions
            ):
                """Get deterministic actions from GRU policy for discrete actions (SMAX)."""
                # Format inputs for GRU: (obs_seq, dones_seq, avail_seq)
                obs_seq = obs_batch[None, ...]  # (1, batch_size, obs_dim)
                dones_seq = jnp.zeros((1, obs_batch.shape[0]), dtype=bool)
                avail_seq = (
                    avail_actions[None, ...] if avail_actions is not None else None
                )  # (1, batch_size, action_dim)

                policy_x = (obs_seq, dones_seq, avail_seq)
                new_hidden, logits_seq = shared_policy.apply(
                    {"params": policy_params}, hidden_states, policy_x, adapters_dict
                )
                logits = logits_seq[0]  # Remove time dimension
                return new_hidden, logits

            @jax.jit
            def _get_actions_continuous_gru(
                policy_params, obs_batch, adapters_dict, hidden_states
            ):
                """Get deterministic actions from GRU policy for continuous actions."""
                obs_seq = obs_batch[None, ...]
                dones_seq = jnp.zeros((1, obs_batch.shape[0]), dtype=bool)
                avail_seq = None

                policy_x = (obs_seq, dones_seq, avail_seq)
                new_hidden, output = shared_policy.apply(
                    {"params": policy_params}, hidden_states, policy_x, adapters_dict
                )
                mean_seq, log_std_seq = output
                mean = mean_seq[0]  # Remove time dimension
                return new_hidden, jnp.clip(mean, -1.0, 1.0)

        else:

            @jax.jit
            def _get_actions(policy_params, obs_batch, adapters_dict):
                """Get deterministic actions from the policy (evaluation mode)."""
                mean, log_std = shared_policy.apply(
                    {"params": policy_params}, obs_batch, adapters_dict
                )
                # Clip actions to valid range [-1, 1] for VMAS (matches evaluate_during_training)
                return jnp.clip(mean, -1.0, 1.0)

    else:
        # No hypernetwork - direct policy evaluation (e.g., DiCo or standard MAPPO)
        if use_cash:

            @jax.jit
            def _get_actions_continuous_cash_gru(
                policy_params, obs_batch, capability_batch, hidden_states
            ):
                """Get deterministic actions from CASH policy (continuous GRU)."""
                obs_seq = obs_batch[None, ...]
                dones_seq = jnp.zeros((1, obs_batch.shape[0]), dtype=bool)
                cap_seq = capability_batch[None, ...]
                policy_x = (obs_seq, dones_seq, cap_seq)

                new_hidden, output = shared_policy.apply(
                    {"params": policy_params}, hidden_states, policy_x
                )
                mean_seq, _ = output
                mean = mean_seq[0]
                return new_hidden, jnp.clip(mean, -1.0, 1.0)

        elif use_dico:

            @jax.jit
            def _get_actions(policy_params, obs_batch, agent_ids):
                """Get deterministic actions from DiCo policy (evaluation mode)."""
                mean, log_std = shared_policy.apply(
                    {"params": policy_params},
                    obs_batch,
                    agent_ids,
                    diversity_scaling=1.0,
                )
                # Clip actions to valid range [-1, 1] for VMAS (matches evaluate_during_training)
                return jnp.clip(mean, -1.0, 1.0)

        else:
            if use_gru_policy:
                # Empty adapters for non-hypernetwork GRU
                @jax.jit
                def _get_actions_discrete_gru(
                    policy_params, obs_batch, adapters_dict, hidden_states
                ):
                    """Get deterministic actions from GRU policy for discrete actions (SMAX)."""
                    obs_seq = obs_batch[None, ...]
                    dones_seq = jnp.zeros((1, obs_batch.shape[0]), dtype=bool)
                    avail_seq = None

                    policy_x = (obs_seq, dones_seq, avail_seq)
                    new_hidden, logits_seq = shared_policy.apply(
                        {"params": policy_params},
                        hidden_states,
                        policy_x,
                        adapters_dict,
                    )
                    logits = logits_seq[0]
                    return new_hidden, logits

                @jax.jit
                def _get_actions_continuous_gru(
                    policy_params, obs_batch, adapters_dict, hidden_states
                ):
                    """Get deterministic actions from GRU policy for continuous actions."""
                    obs_seq = obs_batch[None, ...]
                    dones_seq = jnp.zeros((1, obs_batch.shape[0]), dtype=bool)
                    avail_seq = None

                    policy_x = (obs_seq, dones_seq, avail_seq)
                    new_hidden, output = shared_policy.apply(
                        {"params": policy_params},
                        hidden_states,
                        policy_x,
                        adapters_dict,
                    )
                    mean_seq, log_std_seq = output
                    mean = mean_seq[0]
                    return new_hidden, jnp.clip(mean, -1.0, 1.0)

            else:

                @jax.jit
                def _get_actions(policy_params, obs_batch):
                    """Get deterministic actions from the policy (evaluation mode)."""
                    # Empty adapters for non-hypernetwork MLP
                    empty_adapters = {}
                    mean, log_std = shared_policy.apply(
                        {"params": policy_params}, obs_batch, empty_adapters
                    )
                    # Clip actions to valid range [-1, 1] for VMAS (matches evaluate_during_training)
                    return jnp.clip(mean, -1.0, 1.0)

    total_rewards = []
    episode_lengths = []
    completed_episodes = 0
    agents_at_goal_percentages = []  # Track percentage of agents at goal
    mean_interagent_distances = []  # Track mean interagent distance per episode

    # Track reward decomposition over all episodes
    reward_components = {
        "food_collection": [],
        "shaping": [],
        "time_penalty": [],
    }

    # Track collision metrics for simple_tag
    first_collision_times = []  # Time until first collision per episode
    total_collisions_per_episode = []  # Total collisions per episode

    # Track wins for SMAX
    smax_wins = 0
    smax_total_episodes = 0

    # Track food collection for dispersion
    total_food_available = 0
    total_food_collected = 0

    # Check if this is dispersion environment (check module name)
    is_dispersion_env = (
        hasattr(env, "scenario")
        and hasattr(env.scenario, "__class__")
        and hasattr(env.scenario.__class__, "__module__")
        and "dispersion" in env.scenario.__class__.__module__.lower()
    )

    # Check if this is reverse_transport environment
    is_reverse_transport_env = (
        hasattr(env, "scenario")
        and hasattr(env.scenario, "__class__")
        and hasattr(env.scenario.__class__, "__module__")
        and "reverse_transport" in env.scenario.__class__.__module__.lower()
    )

    # Track goals for football
    is_football_env = hasattr(env, "num_blue_agents") and hasattr(env, "num_red_agents")
    football_blue_goals = 0
    football_red_goals = 0
    football_wins = 0
    football_draws = 0
    football_losses = 0
    # Per-env episode goal accumulators (reset each episode)
    ep_blue_goals = np.zeros(num_envs, dtype=int)
    ep_red_goals = np.zeros(num_envs, dtype=int)

    if verbose:
        print(f"\nRunning {num_eval_episodes} evaluation episodes...")
        print(f"  Parallel envs: {num_envs}")
        print(f"  Max steps per episode: {max_eval_steps}")
        print(f"  Randomize capabilities: {randomize_capabilities}")

        # Print reverse_transport specific info
        if is_reverse_transport_env and fixed_capabilities is not None:
            print(f"  Reverse Transport Capabilities:")
            speeds = fixed_capabilities.get("speed", [])[:num_agents]
            forces = fixed_capabilities.get("force_multiplier", [])[:num_agents]
            for i in range(min(num_agents, len(speeds), len(forces))):
                print(f"    Agent {i}: speed={speeds[i]:.2f}, force={forces[i]:.2f}")
            if hasattr(env.scenario, "package_mass"):
                print(f"  Package mass: {env.scenario.package_mass}")

    # Initialize adapter SND buffer for diversity control (if enabled)
    adapter_snd_buffer_eval = (
        [] if (use_diversity_control and use_adapter_snd and use_hypernetwork) else None
    )
    trajectory_obs_buffer = []  # Buffer to collect observations during rollout

    for eval_ep in range(num_eval_episodes):
        # Clear buffers at start of each episode
        if adapter_snd_buffer_eval is not None:
            adapter_snd_buffer_eval = []
        trajectory_obs_buffer = []

        # Create evaluation key for SMAX (JAX-based random key)
        if hasattr(env, "unit_type_names"):  # SMAX environment
            eval_key = jax.random.fold_in(
                jax.random.PRNGKey(np_rng.integers(0, 2**31)), eval_ep
            )

        # Setup capabilities
        if randomize_capabilities:
            agent_speeds = np_rng.uniform(0.5, 1.5, size=num_agents).tolist()
            agent_lidar_ranges = np_rng.uniform(0.3, 0.7, size=num_agents).tolist()
            # For reverse_transport, also randomize force_multipliers
            if scenario_name == "reverse_transport":
                agent_force_multipliers = np_rng.uniform(
                    0.5, 1.5, size=num_agents
                ).tolist()
            else:
                agent_force_multipliers = None
            # For dispersion_vmas, randomize max_speed
            if scenario_name == "dispersion_vmas":
                agent_max_speeds = np_rng.uniform(0.5, 2.5, size=num_agents).tolist()
            else:
                agent_max_speeds = None
        elif fixed_capabilities is not None:
            # Use provided fixed capabilities from config
            agent_speeds = fixed_capabilities.get("speed", [])
            agent_lidar_ranges = fixed_capabilities.get("lidar_range", [])
            # Also get force_multipliers for reverse_transport
            agent_force_multipliers = fixed_capabilities.get("force_multiplier", None)
            # Get max_speed for dispersion_vmas
            agent_max_speeds = fixed_capabilities.get("max_speed", None)
            # Ensure we have enough values for all agents
            # Different scenarios use different capability keys
            if scenario_name == "reverse_transport":
                needs_fallback = len(agent_speeds) < num_agents
            elif scenario_name == "dispersion_vmas":
                needs_fallback = (
                    agent_max_speeds is None or len(agent_max_speeds) < num_agents
                )
            else:
                needs_fallback = (
                    len(agent_speeds) < num_agents
                    or len(agent_lidar_ranges) < num_agents
                )
            if needs_fallback:
                # Fall back to pattern if not enough values
                speed_values = [0.5, 1.5]
                agent_speeds = [speed_values[i % 2] for i in range(num_agents)]
                # All agents get 0.5 lidar range (matching training config)
                agent_lidar_ranges = [0.5 for _ in range(num_agents)]
            # Also extend force_multipliers if needed
            if (
                agent_force_multipliers is not None
                and len(agent_force_multipliers) < num_agents
            ):
                # Repeat the pattern to cover all agents
                force_values = (
                    agent_force_multipliers if agent_force_multipliers else [1.0, 1.0]
                )
                agent_force_multipliers = [
                    force_values[i % len(force_values)] for i in range(num_agents)
                ]
            # Extend max_speeds if needed (important for dispersion_vmas generalization)
            if agent_max_speeds is not None and len(agent_max_speeds) < num_agents:
                speed_values = agent_max_speeds if agent_max_speeds else [1.0]
                agent_max_speeds = [
                    speed_values[i % len(speed_values)] for i in range(num_agents)
                ]
        else:
            # Generate distinct fixed capabilities for any number of agents
            if scenario_name == "dispersion_vmas":
                # dispersion_vmas uses max_speed, not speed/lidar_range
                agent_speeds = [1.0] * num_agents
                agent_lidar_ranges = [0.5] * num_agents
                agent_force_multipliers = None
                agent_max_speeds = [1.0 + 0.5 * (i % 2) for i in range(num_agents)]
            elif scenario_name == "reverse_transport":
                speed_values = [0.5, 0.5]
                agent_speeds = [
                    speed_values[i % len(speed_values)] for i in range(num_agents)
                ]
                agent_lidar_ranges = [0.5] * num_agents
                agent_force_multipliers = [5.0] * num_agents
                agent_max_speeds = None
            else:
                agent_speeds = []
                agent_lidar_ranges = []
                speed_values = [0.5, 1.5]
                lidar_values = [
                    0.5,
                    0.5,
                ]  # All agents have lidar_range 0.5 (matching training config)
                for i in range(num_agents):
                    agent_speeds.append(speed_values[i % 2])
                    agent_lidar_ranges.append(0.5)  # All agents get 0.5 lidar range
                agent_force_multipliers = None
                agent_max_speeds = None

        agent_capabilities = {"speed": agent_speeds, "lidar_range": agent_lidar_ranges}
        # Add force_multiplier for reverse_transport scenario
        if agent_force_multipliers is not None:
            agent_capabilities["force_multiplier"] = agent_force_multipliers[
                :num_agents
            ]
        # Add max_speed for dispersion_vmas scenario
        if agent_max_speeds is not None:
            agent_capabilities["max_speed"] = agent_max_speeds[:num_agents]
        # Skip capability updates for football (capabilities are fixed in wrapper) and SMAX (JAX-based env)
        if hasattr(env, "scenario") and hasattr(
            env.scenario, "update_agent_capabilities"
        ):
            env.scenario.update_agent_capabilities(agent_capabilities)

        # Update package mass for reverse_transport before reset
        if scenario_name == "reverse_transport" and package_mass is not None:
            if hasattr(env, "scenario"):
                env.scenario.package_mass = package_mass
                if hasattr(env.scenario, "package"):
                    env.scenario.package.mass = package_mass

        # Reset environment (SMAX requires JAX random key)
        if hasattr(env, "unit_type_names"):  # SMAX environment
            rng_key = jax.random.PRNGKey(np_rng.integers(0, 2**31))
            obs_dict, env_state = env.reset(rng_key)
            obs = [obs_dict[agent] for agent in env.agents]
            # Convert JAX arrays to PyTorch tensors and add batch dimension
            obs = [torch.from_numpy(np.array(o)).float().to(torch_device) for o in obs]
            obs = [o.unsqueeze(0).expand(num_envs, -1) for o in obs]
        else:
            obs = env.reset()

        # Create context vectors
        if context_dim > 0:
            # Check if this is SMAX environment
            if hasattr(env, "unit_type_names"):
                # SMAX: use unit capability features as context (HeuristicEnemySMAX wraps the state)
                # Get env_state from reset - will be set below for SMAX
                if "env_state" in locals():
                    from smax_capabilities import get_unit_capabilities

                    smax_state = (
                        env_state.state if hasattr(env_state, "state") else env_state
                    )
                    unit_types = smax_state.unit_types[: env.num_allies]  # Allies only
                    capability_features = get_unit_capabilities(env, unit_types)
                    capability_vectors = (
                        torch.from_numpy(np.array(capability_features))
                        .float()
                        .to(torch_device)
                    )
                else:
                    # Will be set after reset for SMAX
                    capability_vectors = None
            # Check if this is a football wrapper (has get_capability_vectors method)
            elif hasattr(env, "get_capability_vectors"):
                # Football: match context width to checkpoint expectation.
                # Base capability vector is 3D; append positions only if needed.
                capability_batch = (
                    torch.from_numpy(env.get_capability_vectors(normalize=True))
                    .to(torch_device)
                    .unsqueeze(0)
                    .expand(num_envs, -1, -1)
                )  # (num_envs, num_agents, 3)

                if context_dim <= capability_batch.shape[-1]:
                    static_context = capability_batch[:, :, :context_dim]
                else:
                    initial_positions = env.get_initial_positions(normalize=True).to(
                        torch_device
                    )  # (num_envs, num_agents, 2)
                    combined_context = torch.cat(
                        [capability_batch, initial_positions], dim=-1
                    )  # (num_envs, num_agents, 5)
                    if combined_context.shape[-1] >= context_dim:
                        static_context = combined_context[:, :, :context_dim]
                    else:
                        padding = torch.zeros(
                            num_envs,
                            num_agents,
                            context_dim - combined_context.shape[-1],
                            device=torch_device,
                            dtype=combined_context.dtype,
                        )
                        static_context = torch.cat([combined_context, padding], dim=-1)
            elif hasattr(env, "scenario") and hasattr(env.scenario, "__class__"):
                # Check if this is simple_tag or grassland by inspecting the scenario module
                scenario_module = env.scenario.__class__.__module__
                if (
                    "simple_tag" in scenario_module
                    or "grassland" in scenario_module
                    or "custom_simple_tag" in scenario_module
                    or "custom_grassland" in scenario_module
                ):
                    # Simple tag / Grassland: use full observations as context
                    capability_list = []
                    for i in range(num_agents):
                        agent_obs = obs[i][
                            0, :
                        ]  # (obs_dim,) - full observation from first env
                        capability_list.append(agent_obs)
                    capability_vectors = torch.stack(
                        capability_list, dim=0
                    )  # (num_agents, obs_dim)
                elif scenario_name == "dispersion_vmas":
                    # For dispersion_vmas, context_dim = (1 if use_capability_context else 0) + positional_encoding_dim.
                    # Infer use_capability_context from context_dim: if context_dim > 1, capabilities are included.
                    if context_dim > 1 and agent_max_speeds is not None:
                        # Capability context was included: use [max_speed]
                        capability_vectors = torch.tensor(
                            [[agent_max_speeds[i]] for i in range(num_agents)],
                            device=torch_device,
                            dtype=torch.float32,
                        )
                    else:
                        # No capability context: empty base (positional encoding appended below)
                        capability_vectors = torch.zeros(
                            num_agents, 0, device=torch_device, dtype=torch.float32
                        )
                elif scenario_name == "reverse_transport":
                    # Use 2-D [speed, force_multiplier] read from agent attributes.
                    # getattr returns None (not the default) when the attribute
                    # exists but was set to None, so handle that explicitly.
                    def _rt_cap(i, attr, fallback):
                        v = getattr(env.agents[i], attr, None)
                        return float(v) if v is not None else float(fallback)

                    # Use agent_force_multipliers if available, otherwise fall back to agent_lidar_ranges
                    force_fallback = (
                        agent_force_multipliers
                        if agent_force_multipliers is not None
                        else agent_lidar_ranges
                    )
                    capability_vectors = torch.tensor(
                        [
                            [
                                _rt_cap(i, "_max_speed", agent_speeds[i]),
                                _rt_cap(i, "_force_multiplier", force_fallback[i]),
                            ]
                            for i in range(num_agents)
                        ],
                        device=torch_device,
                        dtype=torch.float32,
                    )  # (num_agents, 2)
                elif scenario_name == "pressure_plate":
                    # No capability context for pressure_plate
                    capability_vectors = torch.zeros(
                        num_agents, 0, device=torch_device, dtype=torch.float32
                    )  # (num_agents, 0)
                else:
                    # Other scenarios: use 2D capabilities [speed, lidar_range]
                    capability_vectors = torch.tensor(
                        [
                            [agent_speeds[i], agent_lidar_ranges[i]]
                            for i in range(num_agents)
                        ],
                        device=torch_device,
                        dtype=torch.float32,
                    )
            elif scenario_name == "reverse_transport":
                # Use 2-D [speed, force_multiplier] (fallback path).
                # Handle None explicitly (attribute exists but is None).
                def _rt_cap_fb(i, attr, fallback):
                    v = getattr(env.agents[i], attr, None)
                    return float(v) if v is not None else float(fallback)

                capability_vectors = torch.tensor(
                    [
                        [
                            _rt_cap_fb(i, "_max_speed", agent_speeds[i]),
                            _rt_cap_fb(i, "_force_multiplier", agent_lidar_ranges[i]),
                        ]
                        for i in range(num_agents)
                    ],
                    device=torch_device,
                    dtype=torch.float32,
                )  # (num_agents, 2)
            elif scenario_name == "pressure_plate":
                # No capability context for pressure_plate
                capability_vectors = torch.zeros(
                    num_agents, 0, device=torch_device, dtype=torch.float32
                )  # (num_agents, 0)
            else:
                # Dispersion or other environment: use 2D capabilities [speed, lidar_range]
                capability_vectors = torch.tensor(
                    [
                        [agent_speeds[i], agent_lidar_ranges[i]]
                        for i in range(num_agents)
                    ],
                    device=torch_device,
                    dtype=torch.float32,
                )
            # Football may have built static_context directly above
            if (
                not hasattr(env, "get_capability_vectors")
                and scenario_name != "football"
            ):
                static_context = capability_vectors.unsqueeze(0).expand(
                    num_envs, -1, -1
                )
        else:
            static_context = torch.zeros(
                num_envs, num_agents, 0, device=torch_device, dtype=torch.float32
            )

        # For dispersion_vmas: Use positional encoding (if enabled) or one-hot encoding for role differentiation
        # When use_capability_context=True, prepend max_speed to the positional/one-hot encoding
        if scenario_name == "dispersion_vmas":
            if use_positional_context:
                # Generate positional encodings: scalable to any number of agents
                # Shape: (num_agents, positional_encoding_dim)
                agent_pos_encoding = generate_positional_encoding(
                    num_agents, positional_encoding_dim, device=torch_device
                )
                # Expand to batch: (num_envs, num_agents, positional_encoding_dim)
                agent_pos_encoding_expanded = agent_pos_encoding.unsqueeze(0).expand(
                    num_envs, -1, -1
                )

                # Combine with capability context if enabled
                if context_dim > 1 and static_context.shape[-1] > 0:
                    # static_context already contains [max_speed] for each agent
                    # Concatenate: [max_speed] + positional_encoding
                    static_context = torch.cat(
                        [static_context, agent_pos_encoding_expanded], dim=-1
                    )  # (num_envs, num_agents, 1 + positional_encoding_dim)
                else:
                    static_context = agent_pos_encoding_expanded  # (num_envs, num_agents, positional_encoding_dim)
            elif use_onehot_context:
                # Create one-hot encodings for agent indices: [0, 1, 2, ..., num_agents-1]
                # Use max_agents as the dimension to handle variable team sizes
                agent_one_hot = torch.nn.functional.one_hot(
                    torch.arange(num_agents, device=torch_device),
                    num_classes=max_agents,
                ).float()  # (num_agents, max_agents)

                # Expand to batch: (num_envs, num_agents, max_agents)
                agent_one_hot_expanded = agent_one_hot.unsqueeze(0).expand(
                    num_envs, -1, -1
                )

                # Combine with capability context if enabled
                if context_dim > 1 and static_context.shape[-1] > 0:
                    # static_context already contains [max_speed] for each agent
                    # Concatenate: [max_speed] + one_hot_encoding
                    static_context = torch.cat(
                        [static_context, agent_one_hot_expanded], dim=-1
                    )  # (num_envs, num_agents, 1 + max_agents)
                else:
                    # REPLACE static_context entirely with just one-hot IDs (no capability features)
                    static_context = (
                        agent_one_hot_expanded  # (num_envs, num_agents, max_agents)
                    )
            else:
                # No positional or one-hot encoding: use only capability context or empty
                if context_dim == 0 or static_context.shape[-1] == 0:
                    # No positional or one-hot encoding and no capability context: use empty context tensor (context_dim=0)
                    static_context = torch.zeros(
                        num_envs,
                        num_agents,
                        0,
                        device=torch_device,
                        dtype=torch.float32,
                    )

        # Extract lidar context if enabled
        if use_lidar_context and lidar_dim > 0:
            lidar_list = [agent_obs[:, -lidar_dim:] for agent_obs in obs]
            initial_lidar = torch.stack(lidar_list, dim=0).transpose(0, 1)
        else:
            initial_lidar = None

        # Create task embedding (skip for dispersion_vmas - provides no useful information)
        if scenario_name == "dispersion_vmas":
            static_task = torch.zeros(
                num_envs, num_agents, 0, device=torch_device
            )  # Empty task
        else:
            static_task = torch.ones(
                num_envs, num_agents, task_embed_dim, device=torch_device
            )

        # Extract environment context for scenarios that need it
        if scenario_name == "reverse_transport" and env_context_dim > 0:
            package_props = env.scenario.get_package_properties()
            env_context_vec = torch.tensor(
                [
                    package_props["mass"],
                    package_props["width"],
                    package_props["length"],
                ],
                device=torch_device,
                dtype=torch.float32,
            )
            initial_env_context = (
                env_context_vec.unsqueeze(0)
                .unsqueeze(0)
                .expand(num_envs, num_agents, env_context_dim)
            )
        elif (
            scenario_name == "pressure_plate"
            and env_context_dim > 0
            and config is not None
        ):
            # Get current environment state from pressure_plate scenario.
            # Build per-agent relative environment context (positions relative to each
            # agent, consistent with the policy's observation convention).

            # Get configuration flags
            env_context_plate_positions = config["model"].get(
                "env_context_plate_positions", True
            )
            env_context_door_state = config["model"].get("env_context_door_state", True)
            env_context_goal_position = config["model"].get(
                "env_context_goal_position", True
            )
            use_agent_id_context = config["model"].get("use_agent_id_context", False)

            # Agent positions: (num_envs, num_agents, 2)
            _ground_robots_eval = sorted(
                [a for a in env.agents if "ground_robot" in a.name],
                key=lambda a: a.name,
            )
            _agent_pos_eval = torch.stack(
                [a.state.pos[:, :2] for a in _ground_robots_eval], dim=1
            )  # (num_envs, num_agents, 2)

            # Build per-agent context parts: each (num_envs, num_agents, d)
            _eval_ctx_parts = []

            if env_context_plate_positions:
                left_plate_pos = env.scenario.plate_left.state.pos[
                    :, :2
                ]  # (num_envs, 2)
                right_plate_pos = env.scenario.plate_right.state.pos[
                    :, :2
                ]  # (num_envs, 2)
                left_rel_eval = left_plate_pos.unsqueeze(1) - _agent_pos_eval
                right_rel_eval = right_plate_pos.unsqueeze(1) - _agent_pos_eval
                _eval_ctx_parts.extend([left_rel_eval, right_rel_eval])

            if env_context_door_state:
                door_open = env.scenario.door_open.float()  # (num_envs,)
                door_open_exp = door_open[:, None, None].expand(num_envs, num_agents, 1)
                _eval_ctx_parts.append(door_open_exp)

            if env_context_goal_position:
                goal_pos = env.scenario.goal.state.pos[:, :2]  # (num_envs, 2)
                goal_rel_eval = goal_pos.unsqueeze(1) - _agent_pos_eval
                _eval_ctx_parts.append(goal_rel_eval)

            if use_agent_id_context:
                # Add one-hot agent IDs for role differentiation
                # Shape: (num_envs, num_agents, max_agents)
                torch_device = (
                    left_plate_pos.device
                    if env_context_plate_positions
                    else goal_pos.device
                )
                agent_ids_onehot = torch.zeros(
                    num_envs, num_agents, max_agents, device=torch_device
                )
                for i in range(num_agents):
                    agent_ids_onehot[:, i, i] = 1.0
                _eval_ctx_parts.append(agent_ids_onehot)

            # (num_envs, num_agents, env_context_dim)
            initial_env_context = torch.cat(_eval_ctx_parts, dim=-1)

            if eval_ep == 0:
                print(
                    f"\n[DEBUG] Environment context initialized for evaluation (per-agent relative):"
                )
                if env_context_plate_positions:
                    print(f"  Left plate position: {left_plate_pos[0].cpu().numpy()}")
                    print(f"  Right plate position: {right_plate_pos[0].cpu().numpy()}")
                    print(
                        f"  Agent 0 pos (env0): {_agent_pos_eval[0, 0].cpu().numpy()}"
                    )
                if env_context_door_state:
                    print(f"  Door open: {door_open[0].item()}")
                if env_context_goal_position:
                    print(f"  Goal position: {goal_pos[0].cpu().numpy()}")
                print(f"  initial_env_context shape: {initial_env_context.shape}")
        else:
            initial_env_context = None

        # Convert to JAX (keep shape as num_envs, num_agents, dim) — None when dim is 0
        # For dispersion_vmas: context is just one-hot IDs (never None), task is empty (None)
        context_np = static_context.cpu().numpy()
        task_np = static_task.cpu().numpy()
        jax_context = jnp.asarray(context_np) if context_np.shape[-1] > 0 else None
        jax_task = jnp.asarray(task_np) if task_np.shape[-1] > 0 else None

        if use_lidar_context and initial_lidar is not None:
            lidar_np = initial_lidar.cpu().numpy()
            # Pass raw values to match training/evaluate_during_training.py
            jax_lidar = jnp.asarray(lidar_np)
        else:
            jax_lidar = None

        # Extract food positions from observations if in dispersion scenario
        # Only extract if the hypernetwork actually uses food positions
        if use_hypernetwork and hypernetwork.food_position_dim > 0:
            jax_food_positions = extract_food_positions_from_obs(
                obs, num_agents, scenario_name
            )
        else:
            jax_food_positions = None
        import sys

        print(
            f"[SYNC_CHECK_v2] food_position_dim={hypernetwork.food_position_dim if use_hypernetwork else 'N/A'}, jax_food_positions={jax_food_positions is not None}, jax_context shape={jax_context.shape if jax_context is not None else None}",
            flush=True,
        )
        sys.stderr.write(
            f"[SYNC_CHECK_v2] food_position_dim={hypernetwork.food_position_dim if use_hypernetwork else 'N/A'}, jax_food_positions={jax_food_positions is not None}, jax_context shape={jax_context.shape if jax_context is not None else None}\n"
        )
        sys.stderr.flush()

        # Agent positions are not passed to the hypernetwork (same for all agents,
        # provides no useful differentiation — one-hot IDs are used instead).
        jax_agent_positions = None

        # Convert environment context to JAX if available
        if initial_env_context is not None:
            env_context_np = initial_env_context.cpu().numpy()
            jax_env_context = jnp.asarray(
                env_context_np
            )  # (num_envs, num_agents, env_context_dim)
        else:
            jax_env_context = None

        # Create target_snd batch for hypernetwork if needed
        # Skip for dispersion_vmas - target SND is constant and provides no useful information
        if scenario_name == "dispersion_vmas":
            jax_target_snd = None
        elif use_hypernetwork and hasattr(hypernetwork, "target_snd_dim"):
            target_snd_dim = hypernetwork.target_snd_dim
            if target_snd_dim > 0:
                jax_target_snd = jnp.full(
                    (num_envs, num_agents, target_snd_dim),
                    target_snd,
                    dtype=jnp.float32,
                )
            else:
                jax_target_snd = None
        else:
            jax_target_snd = None

        # Mask is not used during evaluation
        jax_mask = None

        if use_cuda:
            if jax_context is not None:
                jax_context = jax.device_put(jax_context, jax_device)
            if jax_task is not None:
                jax_task = jax.device_put(jax_task, jax_device)
            if jax_lidar is not None:
                jax_lidar = jax.device_put(jax_lidar, jax_device)
            if jax_food_positions is not None:
                jax_food_positions = jax.device_put(jax_food_positions, jax_device)
            if jax_env_context is not None:
                jax_env_context = jax.device_put(jax_env_context, jax_device)
            if jax_target_snd is not None:
                jax_target_snd = jax.device_put(jax_target_snd, jax_device)

        # Generate adapters (only if using hypernetwork)
        if use_hypernetwork:
            # Generate adapters with CURRENT diversity scaling (matching train.py flow)
            # SND calculation happens AFTER rollout
            adapters_dict = _get_static_adapters(
                hn_state.params,
                jax_task,
                jax_context,
                num_envs,
                num_agents,
                lidar_batch=jax_lidar,
                food_positions_batch=jax_food_positions,
                agent_positions_batch=jax_agent_positions,
                env_context_batch=jax_env_context,
                diversity_scaling=diversity_scaling,  # Use current scaling
                target_snd_value=target_snd,
            )

            # Add initial contexts to adapter SND buffer (if using adapter SND)
            if adapter_snd_buffer_eval is not None:
                # Add initial contexts to adapter SND buffer (matching train.py)
                # jax_task, jax_context, jax_lidar have shape (num_envs, num_agents, dim)
                for env_idx in range(num_envs):
                    for agent_idx in range(num_agents):
                        entry = {
                            "task": (
                                jax_task[env_idx, agent_idx]
                                if jax_task is not None
                                else None
                            ),
                            "context": (
                                jax_context[env_idx, agent_idx]
                                if jax_context is not None
                                else None
                            ),
                            "lidar": (
                                jax_lidar[env_idx, agent_idx]
                                if jax_lidar is not None
                                else None
                            ),
                            "query_type": "initial",
                            "hidden_state": None,  # GRU not used typically in eval
                            "agent_idx": agent_idx,
                        }
                        adapter_snd_buffer_eval.append(entry)
        elif use_cash:
            adapters_dict = None
        elif not use_dico:
            # Create zero adapters for non-hypernetwork, non-DiCo case.
            # When env_context is appended to observations (shared baseline on
            # reverse_transport / pressure_plate), the first layer input dim must
            # match the augmented observation size.
            adapters_dict = {}
            batch_size = num_envs * num_agents
            _baseline_appends_ctx = (
                scenario_name in ("reverse_transport", "pressure_plate")
                and env_context_dim > 0
                and jax_env_context is not None
            )
            input_dim = obs_dim + (env_context_dim if _baseline_appends_ctx else 0)
            for i, output_dim in enumerate(policy_hidden_dims):
                layer_idx = i + 1
                adapters_dict[f"A{layer_idx}"] = jnp.zeros((batch_size, 0, input_dim))
                adapters_dict[f"B{layer_idx}"] = jnp.zeros((batch_size, output_dim, 0))
                input_dim = output_dim
            # Final adapter for output layer
            final_idx = len(policy_hidden_dims) + 1
            # For GRU, use gru_hidden_dim; for MLP, use last hidden layer dim
            if use_gru_policy:
                final_input_dim = shared_policy.gru_hidden_dim
            else:
                final_input_dim = (
                    policy_hidden_dims[-1] if policy_hidden_dims else obs_dim
                )
            adapters_dict[f"A{final_idx}"] = jnp.zeros((batch_size, 0, final_input_dim))
            adapters_dict[f"B{final_idx}"] = jnp.zeros((batch_size, action_dim, 0))
        else:
            # DiCo doesn't use adapters
            adapters_dict = None

        # Run evaluation episode
        batch_size = num_envs * num_agents
        batch_lengths = np.zeros(num_envs)
        batch_rewards = np.zeros(num_envs)
        active_mask = np.ones(num_envs, dtype=bool)

        # Initialize GRU hidden states if using GRU policy
        if use_gru_policy:
            gru_hidden_dim = shared_policy.gru_hidden_dim
            gru_hidden_states = jnp.zeros((batch_size, gru_hidden_dim))
            if use_cuda:
                gru_hidden_states = jax.device_put(gru_hidden_states, jax_device)
        else:
            gru_hidden_states = None

        # Track collisions for this episode
        first_collision_time = np.full(num_envs, -1)  # -1 means no collision yet
        collision_count = np.zeros(num_envs)  # Count collisions per env

        # For SMAX: collect state sequence for rendering
        state_sequence = [] if hasattr(env, "unit_type_names") else None

        # Track which agents have already detected food to ensure we only requery once
        food_detected_already = torch.zeros(
            num_envs, num_agents, dtype=torch.bool, device=torch_device
        )

        # Check initial observation: mark agents that can already see food at episode start
        # This prevents requerying the hypernetwork for food that was visible from initialization
        # NOTE: Only apply to dispersion scenario (has food with in_range_flag in observations)
        if adaptive_hypernetwork and scenario_name == "dispersion":
            initial_food_in_range = detect_food_in_range(obs, num_envs, num_agents)
            food_detected_already = initial_food_in_range.clone()

            # Track initial detections (no need to log)
            # if verbose and initial_food_in_range.any():
            #     num_initial = initial_food_in_range.sum().item()

        # Track food collection for dispersion scenario
        # Get actual number of food items from environment (supports fixed_n_food config)
        if is_dispersion_env:
            # For dispersion: get actual number of food items from environment
            num_food_items = len(env.world.landmarks)
            total_food_available += num_envs * num_food_items
            # Track food eaten status for this episode
            episode_food_collected = 0
            # Store initial state of food (to track new collections)
            food_initially_eaten = [food.eaten.clone() for food in env.world.landmarks]

        # Track door state for pressure_plate scenario requerying
        prev_door_open_eval = None
        plate_activation_count_eval = None
        if scenario_name == "pressure_plate":
            prev_door_open_eval = torch.zeros(
                num_envs, dtype=torch.bool, device=torch_device
            )
            plate_activation_count_eval = torch.zeros(
                num_envs, dtype=torch.long, device=torch_device
            )

        # Track package mass for reverse_transport scenario requerying
        current_package_props = None
        if scenario_name == "reverse_transport" and hasattr(env, "scenario"):
            if hasattr(env.scenario, "get_package_properties"):
                current_package_props = env.scenario.get_package_properties()

        for step in range(max_eval_steps):
            # Convert observations
            obs_stacked = torch.stack(obs, dim=0)
            obs_transposed = obs_stacked.transpose(0, 1)
            obs_flat = obs_transposed.reshape(batch_size, -1)

            # Pad observations to max_obs_dim for SMAX with curriculum
            if obs_flat.shape[1] < max_obs_dim:
                padding_size = max_obs_dim - obs_flat.shape[1]
                obs_flat = torch.cat(
                    [
                        obs_flat,
                        torch.zeros(batch_size, padding_size, device=torch_device),
                    ],
                    dim=1,
                )

            obs_np = obs_flat.cpu().numpy()
            # NaN protection for observations
            obs_np = np.nan_to_num(obs_np, nan=0.0, posinf=1.0, neginf=-1.0)
            jax_obs = jnp.asarray(obs_np)

            if use_cuda:
                jax_obs = jax.device_put(jax_obs, jax_device)

            # Collect observations for adapter SND calculation (if enabled)
            if adapter_snd_buffer_eval is not None:
                trajectory_obs_buffer.append(jax_obs)

            # For the shared-policy baseline (no HN, no DiCo) on scenarios that
            # expose env_context, append it to the observations so the policy
            # receives the same information as the HyperLoRA approach.
            if (
                not use_hypernetwork
                and not use_dico
                and scenario_name in ("reverse_transport", "pressure_plate")
                and jax_env_context is not None
            ):
                env_context_flat = jax_env_context.reshape(batch_size, env_context_dim)
                jax_obs = jnp.concatenate([jax_obs, env_context_flat], axis=-1)

            # For DiCo with dispersion_vmas, append max_speed capability to observations
            if (
                use_dico
                and scenario_name == "dispersion_vmas"
                and use_capability_context
                and agent_max_speeds is not None
            ):
                # Tile agent_max_speeds across all environments
                # agent_max_speeds: list of max_speed values for each agent
                max_speed_flat = np.tile(
                    np.array(agent_max_speeds[:num_agents], dtype=np.float32), num_envs
                )  # (batch_size,)
                # Reshape to (batch_size, 1) to append to observations
                max_speed_obs = jnp.asarray(max_speed_flat[:, np.newaxis])
                jax_obs = jnp.concatenate([jax_obs, max_speed_obs], axis=-1)

            # Get deterministic actions (discrete for SMAX, continuous for others)
            if hasattr(env, "unit_type_names"):
                # SMAX: discrete actions
                # Get action masks from environment state
                masks_dict = env.get_avail_actions(env_state)
                # Stack masks for all agents: (num_agents, num_actions)
                action_masks_jax = jnp.stack(
                    [masks_dict[agent] for agent in env.agents], axis=0
                )  # (num_agents, num_actions)
                # Expand to num_envs (SMAX eval uses single env but replicates observations)
                action_masks_jax = jnp.tile(
                    action_masks_jax[None, ...], (num_envs, 1, 1)
                )  # (num_envs, num_agents, num_actions)
                action_masks_flat = action_masks_jax.reshape(
                    -1, action_masks_jax.shape[-1]
                )  # (batch_size, num_actions)

                # Pad action masks to max_action_dim if needed (for curriculum training)
                if action_masks_flat.shape[1] < max_action_dim:
                    padding_size = max_action_dim - action_masks_flat.shape[1]
                    # Pad with zeros (invalid actions)
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

                if use_hypernetwork:
                    if use_gru_policy:
                        # Use JIT-compiled GRU discrete action function
                        gru_hidden_states, logits_flat = _get_actions_discrete_gru(
                            policy_state.params,
                            jax_obs,
                            adapters_dict,
                            gru_hidden_states,
                            action_masks_flat,
                        )
                        # Masking is already handled inside the policy
                    else:
                        logits_flat, _ = shared_policy.apply(
                            {"params": policy_state.params},
                            jax_obs,
                            adapters_dict,
                        )
                elif use_dico:
                    agent_ids_np = (
                        np.tile(np.arange(num_agents), num_envs)
                        % shared_policy.num_agents
                    )
                    agent_ids_jax = jnp.asarray(agent_ids_np)
                    if use_cuda:
                        agent_ids_jax = jax.device_put(agent_ids_jax, jax_device)
                    logits_flat, _ = shared_policy.apply(
                        {"params": policy_state.params},
                        jax_obs,
                        agent_ids_jax,
                        diversity_scaling=1.0,
                    )
                else:
                    # No hypernetwork, no DiCo - use zero adapters
                    if use_gru_policy:
                        # Use JIT-compiled GRU discrete action function with empty adapters
                        gru_hidden_states, logits_flat = _get_actions_discrete_gru(
                            policy_state.params,
                            jax_obs,
                            adapters_dict,
                            gru_hidden_states,
                        )
                        # Apply action masking after getting logits
                        logits_flat = jnp.where(
                            action_masks_flat.astype(bool),
                            logits_flat,
                            jnp.full_like(logits_flat, -1e10),
                        )
                    else:
                        logits_flat, _ = shared_policy.apply(
                            {"params": policy_state.params},
                            jax_obs,
                            adapters_dict,
                        )
                # Take argmax for deterministic evaluation
                jax_actions_flat = jnp.argmax(logits_flat, axis=-1)
            else:
                # Continuous actions
                if use_hypernetwork:
                    if use_gru_policy:
                        # Use JIT-compiled GRU continuous action function
                        gru_hidden_states, jax_actions_flat = (
                            _get_actions_continuous_gru(
                                policy_state.params,
                                jax_obs,
                                adapters_dict,
                                gru_hidden_states,
                            )
                        )
                    else:
                        jax_actions_flat = _get_actions(
                            policy_state.params, jax_obs, adapters_dict
                        )
                elif use_cash:
                    # CASH uses raw capability context and no diversity scaling.
                    if jax_context is not None:
                        cash_context_flat = jax_context.reshape(batch_size, -1)
                    else:
                        cash_context_flat = jnp.zeros(
                            (batch_size, shared_policy.capability_dim),
                            dtype=jnp.float32,
                        )
                    gru_hidden_states, jax_actions_flat = (
                        _get_actions_continuous_cash_gru(
                            policy_state.params,
                            jax_obs,
                            cash_context_flat,
                            gru_hidden_states,
                        )
                    )
                elif use_dico:
                    # Create agent_ids for DiCo policy
                    agent_ids_np = (
                        np.tile(np.arange(num_agents), num_envs)
                        % shared_policy.num_agents
                    )
                    agent_ids_jax = jnp.asarray(agent_ids_np)
                    if use_cuda:
                        agent_ids_jax = jax.device_put(agent_ids_jax, jax_device)
                    jax_actions_flat = _get_actions(
                        policy_state.params, jax_obs, agent_ids_jax
                    )
                else:
                    # No hypernetwork, no DiCo - use zero adapters
                    if use_gru_policy:
                        # Use JIT-compiled GRU continuous action function with empty adapters
                        gru_hidden_states, jax_actions_flat = (
                            _get_actions_continuous_gru(
                                policy_state.params,
                                jax_obs,
                                adapters_dict,
                                gru_hidden_states,
                            )
                        )
                    else:
                        jax_actions_flat = _get_actions(policy_state.params, jax_obs)

            # Convert to PyTorch and step environment
            if hasattr(env, "unit_type_names"):
                # SMAX: discrete actions
                actions_flat = np.asarray(jax_actions_flat, dtype=np.int32)
                actions_reshaped = actions_flat.reshape(num_envs, num_agents)
                # SMAX steps single environment
                step_key = jax.random.fold_in(eval_key, step)
                actions_dict = {
                    agent: int(actions_reshaped[0, i])
                    for i, agent in enumerate(env.agents)
                }
                next_obs_dict, env_state, rewards_dict, dones_dict, _ = env.step_env(
                    step_key, env_state, actions_dict
                )
                # Collect state sequence for rendering
                if state_sequence is not None:
                    state_sequence.append((step_key, env_state, actions_dict))
                # Update context from new state for next iteration
                from smax_capabilities import get_unit_capabilities

                smax_state = (
                    env_state.state if hasattr(env_state, "state") else env_state
                )
                unit_types = smax_state.unit_types[: env.num_allies]
                capability_features = get_unit_capabilities(env, unit_types)
                capability_vectors = (
                    torch.from_numpy(np.array(capability_features))
                    .float()
                    .to(torch_device)
                )
                static_context = capability_vectors.unsqueeze(0).expand(
                    num_envs, -1, -1
                )
                # Convert to list format
                next_obs = [next_obs_dict[agent] for agent in env.agents]
                next_obs = [
                    torch.from_numpy(np.array(o)).float().to(torch_device)
                    for o in next_obs
                ]
                next_obs = [o.unsqueeze(0).expand(num_envs, -1) for o in next_obs]
                rewards = [
                    torch.full(
                        (num_envs,),
                        float(np.array(rewards_dict[agent])),
                        device=torch_device,
                    )
                    for agent in env.agents
                ]
                dones = [
                    torch.full(
                        (num_envs,),
                        bool(np.array(dones_dict[agent])),
                        dtype=torch.bool,
                        device=torch_device,
                    )
                    for agent in env.agents
                ]
                info = {}
            else:
                # VMAS: continuous actions
                actions_flat = np.asarray(jax_actions_flat)
                # NaN protection: replace NaN with 0 and clip to valid range
                actions_flat = np.nan_to_num(
                    actions_flat, nan=0.0, posinf=1.0, neginf=-1.0
                )
                actions_flat = np.clip(actions_flat, -1.0, 1.0)
                actions_reshaped = actions_flat.reshape(num_envs, num_agents, -1)
                torch_actions = [
                    torch.tensor(
                        actions_reshaped[:, i, :],
                        device=torch_device,
                        dtype=torch.float32,
                    )
                    for i in range(num_agents)
                ]
                next_obs, rewards, dones, info = env.step(torch_actions)

            # Check if Food Entered Lidar Range and Requery Hypernetwork
            # Only apply to dispersion scenario (has food with in_range_flag in observations)
            if (
                use_hypernetwork
                and adaptive_hypernetwork
                and scenario_name == "dispersion"
            ):  # Only if using hypernetwork and adaptive mode enabled
                # Detect which agents have food in lidar range now
                current_food_in_range = detect_food_in_range(
                    next_obs, num_envs, num_agents
                )

                # Find agents that just detected food for the FIRST time
                newly_detected = current_food_in_range & ~food_detected_already

                # If any agent newly detected food, requery hypernetwork for ONLY those agents
                if newly_detected.any():
                    num_requeried = newly_detected.sum().item()
                    if verbose:
                        print(
                            f"  [Eval Ep {eval_ep+1}, Step {step}] Food detected! Requerying HN for {num_requeried} agent(s)"
                        )
                    # Extract updated lidar readings from next_obs
                    if use_lidar_context and lidar_dim > 0:
                        updated_lidar_list = [
                            agent_obs[:, -lidar_dim:] for agent_obs in next_obs
                        ]
                        updated_lidar = torch.stack(
                            updated_lidar_list, dim=0
                        ).transpose(0, 1)
                        # Keep 3D shape (num_envs, num_agents, lidar_dim) to match hypernetwork expectations

                        # Convert to numpy and JAX
                        updated_lidar_np = updated_lidar.cpu().numpy()
                        updated_lidar_np = np.nan_to_num(
                            updated_lidar_np, posinf=1.0, neginf=0.0
                        )
                        updated_lidar_np = np.clip(updated_lidar_np, -1.0, 1.0)
                        jax_lidar_updated = jnp.asarray(updated_lidar_np)

                        if use_cuda:
                            jax_lidar_updated = jax.device_put(
                                jax_lidar_updated, jax_device
                            )
                    else:
                        jax_lidar_updated = None

                    # Extract food positions from next_obs if in dispersion scenario
                    jax_food_positions_updated = extract_food_positions_from_obs(
                        next_obs, num_agents, scenario_name
                    )
                    if use_cuda and jax_food_positions_updated is not None:
                        jax_food_positions_updated = jax.device_put(
                            jax_food_positions_updated, jax_device
                        )

                    # Agent positions are not used in the hypernetwork query.
                    jax_agent_positions_updated = None

                    # Requery hypernetwork with updated lidar and food/agent position context
                    food_position_dim = hypernetwork.food_position_dim
                    if food_position_dim > 0:
                        if jax_food_positions_updated is not None:
                            food_batch_updated = jax_food_positions_updated
                        else:
                            batch_shape = (num_envs, num_agents)
                            food_batch_updated = jnp.zeros(
                                (*batch_shape, food_position_dim)
                            )
                    else:
                        food_batch_updated = None

                    agent_position_dim = hypernetwork.agent_position_dim
                    if agent_position_dim > 0:
                        if jax_agent_positions_updated is not None:
                            agent_batch_updated = jax_agent_positions_updated
                        else:
                            batch_shape = (num_envs, num_agents)
                            agent_batch_updated = jnp.zeros(
                                (*batch_shape, agent_position_dim)
                            )
                    else:
                        agent_batch_updated = None

                    # Create target_snd batch for requerying
                    target_snd_dim = hypernetwork.target_snd_dim
                    if target_snd_dim > 0:
                        batch_shape = (num_envs, num_agents)
                        target_snd_batch = jnp.full(
                            (*batch_shape, target_snd_dim),
                            target_snd,
                            dtype=jnp.float32,
                        )
                    else:
                        target_snd_batch = None

                    new_adapters = hypernetwork.apply(
                        {"params": hn_state.params},
                        jax_task,
                        jax_context,
                        jax_lidar_updated,
                        food_batch_updated,
                        agent_batch_updated,
                        target_snd_batch,
                        jax_env_context,  # env_context_batch parameter
                        None,  # mask parameter (no dynamic masking in eval)
                        diversity_scaling,  # Use current diversity scaling (same as episode start)
                    )

                    # Update adapters ONLY for agents that newly detected food
                    newly_detected_flat = newly_detected.reshape(-1).cpu().numpy()

                    # Replace adapters only for agents that newly detected food
                    for key in adapters_dict.keys():
                        mask = newly_detected_flat[:, None, None]
                        adapters_dict[key] = jnp.where(
                            mask, new_adapters[key], adapters_dict[key]
                        )

                    # Only log requery stats if explicitly needed (disabled by default)
                    # if verbose and eval_ep == 0:
                    #     num_requeried = newly_detected.sum().item()
                    #     if num_requeried > 0:
                    #         print(f"    [Step {step}] Requeried HN for {num_requeried} agent(s)")

                # Update tracking
                food_detected_already = food_detected_already | current_food_in_range

            # ================================================================
            # Pressure plate: requery hypernetwork when door opens OR 2nd plate activated
            # (mirrors the logic in train.py)
            # ================================================================
            if (
                use_hypernetwork
                and adaptive_hypernetwork
                and scenario_name == "pressure_plate"
                and prev_door_open_eval is not None
                and env_context_dim > 0
                and jax_env_context is not None
            ):
                # Get current door state from environment
                current_door_open = env.scenario.door_open  # (num_envs,)

                # Check which environments have new door openings
                newly_opened = current_door_open & ~prev_door_open_eval

                # Count total plate activations (both left and right plates)
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
                    plate_activation_count_eval == 1
                )

                # Update plate activation count (keep max seen)
                plate_activation_count_eval = torch.maximum(
                    plate_activation_count_eval, current_active_plates
                )

                # Determine which environments need requerying
                should_requery = newly_opened | second_plate_activated

                if should_requery.any():
                    num_door_opened = newly_opened.sum().item()
                    num_second_plate = second_plate_activated.sum().item()
                    num_affected_envs = should_requery.sum().item()

                    # Update environment context with current door state
                    env_context_door_state = config["model"].get(
                        "env_context_door_state", True
                    )

                    if env_context_door_state:
                        # Rebuild environment context with updated door state
                        env_context_plate_positions = config["model"].get(
                            "env_context_plate_positions", True
                        )
                        env_context_goal_position = config["model"].get(
                            "env_context_goal_position", True
                        )
                        use_agent_id_context = config["model"].get(
                            "use_agent_id_context", False
                        )

                        # Build per-agent relative context
                        _ground_robots_rq = sorted(
                            [a for a in env.agents if "ground_robot" in a.name],
                            key=lambda a: a.name,
                        )
                        _agent_pos_rq = torch.stack(
                            [a.state.pos[:, :2] for a in _ground_robots_rq], dim=1
                        )  # (num_envs, num_agents, 2)

                        # Build per-agent context parts
                        env_context_parts = []

                        if env_context_plate_positions:
                            left_plate_pos = env.scenario.plate_left.state.pos[:, :2]
                            right_plate_pos = env.scenario.plate_right.state.pos[:, :2]
                            left_rel = left_plate_pos.unsqueeze(1) - _agent_pos_rq
                            right_rel = right_plate_pos.unsqueeze(1) - _agent_pos_rq
                            env_context_parts.extend([left_rel, right_rel])

                        if env_context_door_state:
                            # Use CURRENT per-env door state
                            door_open_exp = current_door_open.float()[
                                :, None, None
                            ].expand(num_envs, num_agents, 1)
                            env_context_parts.append(door_open_exp)

                        if env_context_goal_position:
                            goal_pos = env.scenario.goal.state.pos[:, :2]
                            goal_rel = goal_pos.unsqueeze(1) - _agent_pos_rq
                            env_context_parts.append(goal_rel)

                        if use_agent_id_context:
                            # Add one-hot agent IDs
                            agent_ids_onehot = torch.zeros(
                                num_envs,
                                num_agents,
                                max_agents,
                                device=torch_device,
                            )
                            for i in range(num_agents):
                                agent_ids_onehot[:, i, i] = 1.0
                            env_context_parts.append(agent_ids_onehot)

                        # (num_envs, num_agents, env_context_dim)
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
                    requery_env_mask = should_requery.cpu().numpy()  # (num_envs,)

                    # Create flat mask for all agents: (num_envs * num_agents,)
                    requery_flat_mask = np.repeat(requery_env_mask, num_agents)
                    requery_indices = np.where(requery_flat_mask)[0]
                    num_agents_to_requery = len(requery_indices)

                    if num_agents_to_requery > 0:
                        # Extract subset of agents that need requerying
                        # Reshape from (num_envs, num_agents, ...) to (batch_size, ...)
                        batch_size_eval = num_envs * num_agents

                        # Handle task and context - may be None for scenarios without capability context
                        jax_task_subset = None
                        jax_context_subset = None
                        if jax_task is not None:
                            jax_task_flat = jax_task.reshape(batch_size_eval, -1)
                            jax_task_subset = jax_task_flat[requery_indices]
                            jax_task_subset = jax_task_subset[:, None, :]

                        if jax_context is not None:
                            jax_context_flat = jax_context.reshape(batch_size_eval, -1)
                            jax_context_subset = jax_context_flat[requery_indices]
                            jax_context_subset = jax_context_subset[:, None, :]

                        # Extract environment context subset if available
                        jax_env_context_subset = None
                        if env_context_dim > 0 and jax_env_context is not None:
                            jax_env_context_flat = jax_env_context.reshape(
                                batch_size_eval, -1
                            )
                            jax_env_context_subset = jax_env_context_flat[
                                requery_indices
                            ]
                            jax_env_context_subset = jax_env_context_subset[:, None, :]

                        # Create mask for single-agent requery (no cross-agent attention needed)
                        jax_mask_subset = None

                        # Requery hypernetwork ONLY for affected agents
                        new_adapters_subset = hypernetwork.apply(
                            {"params": hn_state.params},
                            jax_task_subset,
                            jax_context_subset,
                            None,  # jax_lidar not used in pressure_plate
                            None,  # jax_food_positions not used
                            None,  # jax_agent_positions not used
                            None,  # jax_target_snd not used with single-agent requery
                            jax_env_context_subset,
                            jax_mask_subset,
                            diversity_scaling,  # Use current diversity scaling (same as episode start)
                        )

                        # Update adapters ONLY for agents in affected environments
                        for key in adapters_dict.keys():
                            adapters_dict[key] = (
                                adapters_dict[key]
                                .at[requery_indices]
                                .set(new_adapters_subset[key])
                            )

                        # Log the requery event
                        if verbose:
                            batch_size_eval = num_envs * num_agents
                            print(
                                f"  [Eval Ep {eval_ep+1}, Step {step}] Pressure plate event: door_opened={num_door_opened}, second_plate={num_second_plate} - Requeried {num_agents_to_requery}/{batch_size_eval} agents ({100*num_agents_to_requery/batch_size_eval:.1f}%) in {num_affected_envs} envs"
                            )

                # Update previous door state for next step
                prev_door_open_eval = current_door_open.clone()

            # ================================================================
            # Reverse transport: dynamic environment changes (scheduled changes)
            # (mirrors the logic in train.py and render_gif.py)
            # ================================================================
            if scenario_name == "reverse_transport" and config is not None:
                use_dynamic_env = config["env"].get("use_dynamic_env_changes", False)
                env_change_interval = config["env"].get("env_change_interval", 0)

                # Debug output on first episode, first step
                if eval_ep == 0 and step == 1:
                    print(
                        f"\n[Eval Dynamic Env Config] use_dynamic_env_changes={use_dynamic_env}, env_change_interval={env_change_interval}"
                    )
                    print(
                        f"[Eval Dynamic Env Config] env_change_type={config['env'].get('env_change_type', 'N/A')}\n"
                    )

                if (
                    use_dynamic_env
                    and env_change_interval > 0
                    and step > 0
                    and step % env_change_interval == 0
                ):
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
                        # Gradually increase/decrease properties
                        progress = min(step / max_eval_steps, 1.0)
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
                    else:
                        new_mass, new_width, new_length = None, None, None

                    # Update properties in environment
                    if new_mass is not None:
                        changed = env.scenario.update_package_properties(
                            package_mass=new_mass,
                            package_width=new_width,
                            package_length=new_length,
                        )
                        if changed and use_hypernetwork and env_context_dim > 0:
                            # Update current properties tracker
                            if current_package_props is not None:
                                current_package_props = {
                                    "mass": new_mass,
                                    "width": new_width,
                                    "length": new_length,
                                }

                            # Create new environment context
                            env_context_vec = torch.tensor(
                                [new_mass, new_width, new_length],
                                device=torch_device,
                                dtype=torch.float32,
                            )
                            # Broadcast to all envs and agents
                            updated_env_context = (
                                env_context_vec.unsqueeze(0)
                                .unsqueeze(0)
                                .expand(num_envs, num_agents, env_context_dim)
                            )
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

                            # Requery hypernetwork with new environment context
                            adapters_dict = _get_static_adapters(
                                hn_state.params,
                                jax_task,
                                jax_context,
                                num_envs,
                                num_agents,
                                jax_lidar,
                                jax_food_positions,
                                jax_agent_positions,
                                jax_env_context,
                                diversity_scaling,  # Use same scaling as initial adapters
                                target_snd,
                            )

                            # Log the change
                            if verbose:
                                print(
                                    f"  [Eval Ep {eval_ep+1}, Step {step}] Scheduled package change: "
                                    f"mass={new_mass:.2f}, width={new_width:.2f}, length={new_length:.2f} - Requeried hypernetwork"
                                )

            # ================================================================
            # Reverse transport: requery hypernetwork when package properties change
            # (happens when continuous_goals enabled and new package spawns - passive detection)
            # ================================================================
            if (
                use_hypernetwork
                and scenario_name == "reverse_transport"
                and current_package_props is not None
                and hasattr(env, "scenario")
            ):
                # Check if package properties have changed (not from scheduled changes above)
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
                        .expand(num_envs, num_agents, env_context_dim)
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
                        num_envs,
                        num_agents,
                        jax_lidar,
                        jax_food_positions,
                        jax_agent_positions,
                        jax_env_context,
                        diversity_scaling,  # Use same scaling as initial adapters
                        target_snd,
                    )

                    # Update current properties
                    current_package_props = new_package_props

                    # Log the change
                    if verbose:
                        print(
                            f"  [Eval Ep {eval_ep+1}, Step {step}] Package properties changed: "
                            f"mass={new_package_props['mass']:.2f}, "
                            f"width={new_package_props['width']:.2f}, "
                            f"length={new_package_props['length']:.2f} - Requeried hypernetwork"
                        )

            # Track food collection for dispersion by checking landmark.eaten status
            if is_dispersion_env:
                # Count how many food items are eaten across all envs
                for food_idx, food in enumerate(env.world.landmarks):
                    # Count newly eaten food (wasn't eaten initially, is eaten now)
                    newly_eaten = food.eaten & ~food_initially_eaten[food_idx]
                    episode_food_collected += newly_eaten.sum().item()
                    # Update tracking
                    food_initially_eaten[food_idx] = food.eaten.clone()

            # Track reward decomposition if available in info
            if info and len(info) > 0:
                first_agent_info = info[0] if isinstance(info, list) else info
                if isinstance(first_agent_info, dict):
                    if "reward_food_collection" in first_agent_info:
                        value = first_agent_info["reward_food_collection"]
                        if hasattr(value, "item"):
                            value = value.item()
                        reward_components["food_collection"].append(value)
                    if "reward_shaping" in first_agent_info:
                        value = first_agent_info["reward_shaping"]
                        if hasattr(value, "item"):
                            value = value.item()
                        reward_components["shaping"].append(value)
                    if "reward_time_penalty" in first_agent_info:
                        value = first_agent_info["reward_time_penalty"]
                        if hasattr(value, "item"):
                            value = value.item()
                        reward_components["time_penalty"].append(value)
                    # Track percentage of agents at goal
                    if "percentage_agents_at_goal" in first_agent_info:
                        value = first_agent_info["percentage_agents_at_goal"]
                        if hasattr(value, "cpu"):
                            value = value.cpu().numpy()
                        agents_at_goal_percentages.extend(value)
                    # Track mean interagent distance
                    if "mean_interagent_distance" in first_agent_info:
                        value = first_agent_info["mean_interagent_distance"]
                        if hasattr(value, "cpu"):
                            value = value.cpu().numpy()
                        mean_interagent_distances.extend(value)

            # Track football goals from sparse rewards
            if is_football_env and info and len(info) > 0:
                first_info = info[0] if isinstance(info, list) else info
                if isinstance(first_info, dict) and "sparse_reward" in first_info:
                    sparse_rewards = first_info["sparse_reward"]
                    if hasattr(sparse_rewards, "cpu"):
                        sparse_rewards = sparse_rewards.cpu().numpy()
                    else:
                        sparse_rewards = np.asarray(sparse_rewards)
                    blue_scored_step = (sparse_rewards > 50).astype(int)
                    red_scored_step = (sparse_rewards < -50).astype(int)
                    football_blue_goals += blue_scored_step.sum()
                    football_red_goals += red_scored_step.sum()
                    ep_blue_goals += blue_scored_step
                    ep_red_goals += red_scored_step

            # Track lengths
            batch_lengths[active_mask] += 1

            # Track rewards
            rewards_stacked = torch.stack(rewards, dim=0)
            rewards_transposed = rewards_stacked.transpose(0, 1)
            rewards_np = rewards_transposed.cpu().numpy()
            step_rewards = rewards_np.mean(axis=1)
            batch_rewards += step_rewards

            # Detect collisions by checking if adversary collided with good agent
            # For simple_tag: collision occurs when adversary reward is high (caught good agent)
            if hasattr(env, "scenario") and hasattr(env.scenario, "adversaries"):
                # Check rewards - adversary gets +3.0 for collision
                for agent_idx, reward in enumerate(rewards):
                    reward_val = reward.cpu().numpy()
                    # If adversary (first agents) has high reward, collision occurred
                    if agent_idx < len(env.scenario.adversaries()):
                        collision_detected = (
                            reward_val > 2.5
                        )  # Threshold for collision reward
                        # Update collision metrics for envs where collision happened
                        collision_count += collision_detected.astype(float)
                        # Record first collision time if not yet recorded
                        for env_idx in range(num_envs):
                            if (
                                collision_detected[env_idx]
                                and first_collision_time[env_idx] == -1
                            ):
                                first_collision_time[env_idx] = step

            # Update active mask
            # For SMAX: dones is a list of tensors (one per agent), convert to single tensor
            if isinstance(dones, list):
                dones = torch.stack(dones, dim=1).any(dim=1)  # [num_envs]
            dones_np = dones.cpu().numpy()

            # Count completions as they happen (dones_np & active_mask avoids
            # double-counting after VMAS auto-resets a finished env)
            newly_done = dones_np & active_mask
            if hasattr(env, "num_blue_agents") and hasattr(
                env, "num_red_agents"
            ):  # Football
                completed_episodes += ((batch_rewards > 0) & newly_done).sum()
            else:
                completed_episodes += newly_done.sum()

            active_mask = active_mask & (~dones_np)

            obs = next_obs

            if not active_mask.any():
                break

        # Update total food collected for dispersion
        if is_dispersion_env:
            total_food_collected += episode_food_collected
            if verbose and (eval_ep + 1) % 5 == 0:
                print(
                    f"    Episode {eval_ep + 1}: {episode_food_collected}/{num_envs * num_food_items} food collected"
                )

        # Completions are accumulated inside the step loop above to capture
        # episodes that finish before the final step.

        # Tally football outcomes at the end of each episode batch.
        # Football episodes end by timeout (dones never fires naturally), so we
        # tally here rather than inside the step loop.
        if is_football_env:
            for env_idx in range(num_envs):
                b = ep_blue_goals[env_idx]
                r = ep_red_goals[env_idx]
                if b > r:
                    football_wins += 1
                elif b == r:
                    football_draws += 1
                else:
                    football_losses += 1
            ep_blue_goals[:] = 0
            ep_red_goals[:] = 0

        total_rewards.extend(batch_rewards)
        episode_lengths.extend(batch_lengths)

        # ================================================================
        # Post-rollout: Calculate adapter SND and update diversity scaling
        # (matching train.py flow - calculate SND after rollout completes)
        # ================================================================
        if (
            adapter_snd_buffer_eval is not None
            and len(adapter_snd_buffer_eval) > 0
            and use_diversity_control
        ):
            try:
                # Calculate adapter SND using collected buffer and trajectory observations
                rng_key_snd = jax.random.PRNGKey(eval_ep * 1000 + 42)
                snd_value = calculate_adapter_snd(
                    adapter_snd_buffer=adapter_snd_buffer_eval,
                    sample_size=adapter_snd_sample_size,
                    hn_params=hn_state.params,
                    policy_params=policy_state.params,
                    hypernetwork=hypernetwork,
                    policy_model=shared_policy,
                    num_agents=num_agents,
                    rng_key=rng_key_snd,
                    use_gru_policy=use_gru_policy,
                    trajectory_obs=(
                        trajectory_obs_buffer
                        if len(trajectory_obs_buffer) > 0
                        else None
                    ),
                    diversity_scaling=1.0,  # Unscaled for SND measurement
                    scenario_name=scenario_name,
                )

                # Update SND moving average
                if not np.isnan(snd_value) and not np.isinf(snd_value):
                    current_snd_ma = (
                        snd_ma_coef * current_snd_ma + (1 - snd_ma_coef) * snd_value
                    )
                else:
                    # Keep previous MA if current SND is invalid
                    snd_value = current_snd_ma

                # Recalculate diversity scaling for NEXT episode
                diversity_scaling = float(
                    np.sqrt(target_snd / max(current_snd_ma, min_snd_floor))
                )
                diversity_scaling = float(
                    np.clip(diversity_scaling, 0.001, np.sqrt(max_scaling))
                )

                if verbose and eval_ep == 0:
                    print(
                        f"  Episode {eval_ep+1}: Adapter SND={snd_value:.6f}, SND_MA={current_snd_ma:.6f}, Next Scaling={diversity_scaling:.6f}"
                    )

            except Exception as e:
                if verbose and eval_ep == 0:
                    print(
                        f"  Warning: Post-rollout adapter SND calculation failed: {e}"
                    )
        # ================================================================

        # For SMAX: check if allies won the battle
        # IMPORTANT: SMAX doesn't vectorize - it's a single environment, not num_envs parallel
        # The num_envs expansion is just for batch processing compatibility
        if hasattr(env, "unit_type_names"):
            smax_total_episodes += 1  # Only 1 actual battle (SMAX doesn't vectorize)
            # Check battle outcome: allies win if all enemies dead AND at least one ally alive
            smax_state = env_state.state if hasattr(env_state, "state") else env_state
            allies_alive = smax_state.unit_alive[: env.num_allies]
            enemies_alive = smax_state.unit_alive[
                env.num_allies : env.num_allies + env.num_enemies
            ]

            # Battle is won if all enemies are dead and at least one ally is alive
            all_enemies_dead = not np.any(np.array(enemies_alive))
            at_least_one_ally_alive = np.any(np.array(allies_alive))
            battle_won = all_enemies_dead and at_least_one_ally_alive

            if battle_won:
                smax_wins += 1  # Count this single battle as a win

        # Store collision metrics for this episode (only for simple_tag)
        # Check if this is simple_tag by verifying collision detection occurred
        if hasattr(env, "scenario") and hasattr(env.scenario, "adversaries"):
            # For first collision time: if no collision occurred, use episode length
            for env_idx in range(num_envs):
                if first_collision_time[env_idx] == -1:
                    # No collision occurred
                    first_collision_times.append(batch_lengths[env_idx])
                else:
                    first_collision_times.append(first_collision_time[env_idx])
            total_collisions_per_episode.extend(collision_count)

        # Render GIF for SMAX environments after each episode
        if state_sequence is not None and len(state_sequence) > 0:
            try:
                from jaxmarl.viz.visualizer import SMAXVisualizer
                import matplotlib

                matplotlib.use("Agg")  # Use non-interactive backend

                # Expand state sequence to include interpolated states
                expanded_state_seq = env.expand_state_seq(state_sequence)

                # Create visualizer and render
                viz = SMAXVisualizer(env._env, expanded_state_seq)

                # Create output directory if needed
                gif_dir = Path(checkpoint_dir) / "gifs"
                gif_dir.mkdir(exist_ok=True)

                # Generate filename
                gif_path = (
                    gif_dir / f"episode_{eval_ep:03d}_reward_{batch_rewards[0]:.2f}.gif"
                )

                # Render and save
                viz.animate(view=False, save_fname=str(gif_path))

                if verbose:
                    print(f"    Saved GIF: {gif_path}")
            except Exception as e:
                if verbose:
                    print(
                        f"    Warning: Failed to render GIF for episode {eval_ep}: {e}"
                    )

        # Progress indicator
        if verbose and (eval_ep + 1) % 5 == 0:
            print(f"  Completed {eval_ep + 1}/{num_eval_episodes} episodes...")

    # Compute metrics
    total_env_episodes = num_eval_episodes * num_envs
    completion_rate = (completed_episodes / total_env_episodes) * 100.0
    avg_episode_length = np.mean(episode_lengths)
    std_episode_length = np.std(episode_lengths)
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    # Compute average reward decomposition
    reward_decomposition = {}
    if reward_components["food_collection"]:
        reward_decomposition["food_collection"] = np.mean(
            reward_components["food_collection"]
        )
    if reward_components["shaping"]:
        reward_decomposition["shaping"] = np.mean(reward_components["shaping"])
    if reward_components["time_penalty"]:
        reward_decomposition["time_penalty"] = np.mean(
            reward_components["time_penalty"]
        )

    metrics = {
        "completion_rate": completion_rate,
        "completed_episodes": completed_episodes,
        "total_episodes": total_env_episodes,
        "avg_episode_length": avg_episode_length,
        "std_episode_length": std_episode_length,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
    }

    # Add reward decomposition if available
    if reward_decomposition:
        metrics["reward_decomposition"] = reward_decomposition

    # Add collision metrics if available (for simple_tag)
    if first_collision_times:
        metrics["avg_first_collision_time"] = np.mean(first_collision_times)
        metrics["std_first_collision_time"] = np.std(first_collision_times)
    if total_collisions_per_episode:
        metrics["avg_collisions_per_episode"] = np.mean(total_collisions_per_episode)
        metrics["std_collisions_per_episode"] = np.std(total_collisions_per_episode)

    # Add SMAX win rate if applicable
    if smax_total_episodes > 0:
        metrics["win_rate"] = (smax_wins / smax_total_episodes) * 100.0
        metrics["wins"] = smax_wins
        metrics["battles"] = smax_total_episodes

    # Add football win rate and goals if applicable
    if is_football_env:
        total_goals = football_blue_goals + football_red_goals
        metrics["football_blue_goals"] = football_blue_goals
        metrics["football_red_goals"] = football_red_goals
        metrics["football_total_goals"] = total_goals
        metrics["football_win_rate"] = (
            football_blue_goals / max(total_goals, 1)
        ) * 100.0
        total_episodes_counted = football_wins + football_draws + football_losses
        metrics["football_wins"] = football_wins
        metrics["football_draws"] = football_draws
        metrics["football_losses"] = football_losses
        metrics["football_episode_win_rate"] = (
            football_wins / max(total_episodes_counted, 1)
        ) * 100.0
        metrics["football_episode_draw_rate"] = (
            football_draws / max(total_episodes_counted, 1)
        ) * 100.0

    # Add food collection percentage for dispersion
    if total_food_available > 0:
        food_found_percentage = (total_food_collected / total_food_available) * 100.0
        metrics["food_found_percentage"] = food_found_percentage

    # Add agents at goal percentage if available
    if agents_at_goal_percentages:
        metrics["avg_agents_at_goal_percentage"] = (
            np.mean(agents_at_goal_percentages) * 100.0
        )
        metrics["std_agents_at_goal_percentage"] = (
            np.std(agents_at_goal_percentages) * 100.0
        )

    # Add mean interagent distance if available
    if mean_interagent_distances:
        metrics["avg_mean_interagent_distance"] = np.mean(mean_interagent_distances)
        metrics["std_mean_interagent_distance"] = np.std(mean_interagent_distances)

    # Calculate SND if requested (only supported with hypernetwork)
    if calculate_snd_metric and use_hypernetwork:
        try:
            # Reset environment to get fresh observations for SND calculation
            if hasattr(env, "unit_type_names"):  # SMAX environment
                import jax
                import jax.numpy as jnp

                rng_key = jax.random.PRNGKey(np_rng.integers(0, 2**31))
                obs_dict, env_state = env.reset(rng_key)
                obs = [obs_dict[agent] for agent in env.agents]
                # Convert JAX arrays to PyTorch tensors
                obs = [
                    torch.from_numpy(np.array(o)).float().to(torch_device) for o in obs
                ]
            else:
                obs = env.reset()

            # Setup capabilities for SND calculation (use fixed pattern)
            agent_speeds = []
            agent_lidar_ranges = []
            speed_values = [0.5, 1.5]
            for i in range(num_agents):
                agent_speeds.append(speed_values[i % 2])
                # All agents get 0.5 lidar range (matching training config)
                agent_lidar_ranges.append(0.5)

            agent_capabilities = {
                "speed": agent_speeds,
                "lidar_range": agent_lidar_ranges,
            }
            # Skip capability updates for football (capabilities are fixed in wrapper) and SMAX (JAX-based env)
            if hasattr(env, "scenario") and hasattr(
                env.scenario, "update_agent_capabilities"
            ):
                env.scenario.update_agent_capabilities(agent_capabilities)

            # Convert observations for SND
            obs_stacked = torch.stack(obs, dim=0)
            obs_transposed = obs_stacked.transpose(0, 1)
            batch_size = num_envs * num_agents
            obs_flat = obs_transposed.reshape(batch_size, -1)
            obs_np = obs_flat.cpu().numpy()
            jax_obs = jnp.asarray(obs_np)

            # Create context vectors
            if context_dim > 0:
                # Check if this is a football wrapper (has get_capability_vectors method)
                if hasattr(env, "get_capability_vectors"):
                    # Football: match context width to checkpoint expectation.
                    capability_batch = (
                        torch.from_numpy(env.get_capability_vectors(normalize=True))
                        .to(torch_device)
                        .unsqueeze(0)
                        .expand(num_envs, -1, -1)
                    )
                    if context_dim <= capability_batch.shape[-1]:
                        static_context = capability_batch[:, :, :context_dim]
                    else:
                        initial_positions = env.get_initial_positions(
                            normalize=True
                        ).to(torch_device)
                        combined_context = torch.cat(
                            [capability_batch, initial_positions], dim=-1
                        )
                        if combined_context.shape[-1] >= context_dim:
                            static_context = combined_context[:, :, :context_dim]
                        else:
                            padding = torch.zeros(
                                num_envs,
                                num_agents,
                                context_dim - combined_context.shape[-1],
                                device=torch_device,
                                dtype=combined_context.dtype,
                            )
                            static_context = torch.cat(
                                [combined_context, padding], dim=-1
                            )
                else:
                    # Dispersion environment: use 2D capabilities [speed, lidar_range]
                    capability_vectors = torch.tensor(
                        [
                            [agent_speeds[i], agent_lidar_ranges[i]]
                            for i in range(num_agents)
                        ],
                        device=torch_device,
                        dtype=torch.float32,
                    )
                    static_context = capability_vectors.unsqueeze(0).expand(
                        num_envs, -1, -1
                    )
            else:
                static_context = torch.zeros(
                    num_envs, num_agents, 0, device=torch_device, dtype=torch.float32
                )

            # Extract lidar context if enabled
            if use_lidar_context and lidar_dim > 0:
                lidar_list = [agent_obs[:, -lidar_dim:] for agent_obs in obs]
                initial_lidar = torch.stack(lidar_list, dim=0).transpose(0, 1)
            else:
                initial_lidar = None

            # Create task embedding
            static_task = torch.ones(
                num_envs, num_agents, task_embed_dim, device=torch_device
            )

            # Convert to JAX (keep shape as num_envs, num_agents, dim) — None when dim is 0
            context_np = static_context.cpu().numpy()
            task_np = static_task.cpu().numpy()
            jax_context = jnp.asarray(context_np) if context_dim > 0 else None
            jax_task = jnp.asarray(task_np) if task_embed_dim > 0 else None

            if use_lidar_context and initial_lidar is not None:
                lidar_np = initial_lidar.cpu().numpy()
                jax_lidar = jnp.asarray(lidar_np)
            else:
                jax_lidar = None

            if use_cuda:
                jax_obs = jax.device_put(jax_obs, jax_device)
                if jax_context is not None:
                    jax_context = jax.device_put(jax_context, jax_device)
                if jax_task is not None:
                    jax_task = jax.device_put(jax_task, jax_device)
                if jax_lidar is not None:
                    jax_lidar = jax.device_put(jax_lidar, jax_device)

            # Calculate SND using same method as training (Expected SND with sampling)
            eval_rng = jax.random.PRNGKey(0)  # Fixed seed for reproducible eval
            snd_stats = calculate_snd_statistics(
                policy_params=policy_state.params,
                hn_params=hn_state.params,
                obs_batch=jax_obs,
                task_batch=jax_task,
                context_batch=jax_context,
                policy_model=shared_policy,
                hypernetwork=hypernetwork,
                num_agents=num_agents,
                num_envs=num_envs,
                num_samples=10,  # Same as training
                rng_key=eval_rng,
            )
            snd_value = snd_stats["snd_total"]
            metrics["snd"] = float(snd_value)

            if verbose:
                print(f"  SND: {snd_value:.6f}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to calculate SND: {e}")
            metrics["snd"] = None

    return metrics


def print_training_eval_results(episode, eval_metrics):
    """Print formatted evaluation results during training."""
    print(f"\n{'='*80}")
    print(f"Evaluation Results (Episode {episode}):")
    print(f"  Completion Rate: {eval_metrics['completion_rate']:.2f}%")
    print(f"  Avg Episode Length: {eval_metrics['avg_episode_length']:.2f} steps")
    print(f"  Avg Reward: {eval_metrics['avg_reward']:.3f}")
    if "snd" in eval_metrics and eval_metrics["snd"] is not None:
        print(f"  SND: {eval_metrics['snd']:.6f}")

    # Print collision metrics if available (for simple_tag)
    if "avg_first_collision_time" in eval_metrics:
        print(
            f"  First Collision: {eval_metrics['avg_first_collision_time']:.2f} steps"
        )
    if "avg_collisions_per_episode" in eval_metrics:
        print(f"  Collisions/Episode: {eval_metrics['avg_collisions_per_episode']:.2f}")

    # Print SMAX win rate if available
    if "win_rate" in eval_metrics:
        print(
            f"  Win Rate: {eval_metrics['win_rate']:.2f}% ({eval_metrics['wins']}/{eval_metrics['battles']} battles)"
        )

    # Print football win rate and goals if available
    if "football_win_rate" in eval_metrics:
        print(
            f"  Football Goal Win Rate: {eval_metrics['football_win_rate']:.1f}% "
            f"(Blue {eval_metrics['football_blue_goals']} - {eval_metrics['football_red_goals']} Red, "
            f"{eval_metrics['football_total_goals']} total goals)"
        )
    if "football_episode_win_rate" in eval_metrics:
        n = (
            eval_metrics["football_wins"]
            + eval_metrics["football_draws"]
            + eval_metrics["football_losses"]
        )
        print(
            f"  Football Episode Win Rate:  {eval_metrics['football_episode_win_rate']:.1f}%  "
            f"Draw Rate: {eval_metrics['football_episode_draw_rate']:.1f}%  "
            f"Loss Rate: {100.0 - eval_metrics['football_episode_win_rate'] - eval_metrics['football_episode_draw_rate']:.1f}%  "
            f"(W {eval_metrics['football_wins']} / D {eval_metrics['football_draws']} / L {eval_metrics['football_losses']}, {n} episodes)"
        )

    # Print food collection percentage if available
    if "food_found_percentage" in eval_metrics:
        print(f"  Food Found: {eval_metrics['food_found_percentage']:.2f}%")

    # Print agents at goal percentage if available
    if "avg_agents_at_goal_percentage" in eval_metrics:
        print(
            f"  Agents At Goal: {eval_metrics['avg_agents_at_goal_percentage']:.2f}% ± {eval_metrics['std_agents_at_goal_percentage']:.2f}%"
        )

    # Print mean interagent distance if available
    if "avg_mean_interagent_distance" in eval_metrics:
        print(
            f"  Mean Interagent Distance: {eval_metrics['avg_mean_interagent_distance']:.4f} ± {eval_metrics['std_mean_interagent_distance']:.4f}"
        )

    print(f"{'='*80}\n")


def print_evaluation_results(eval_metrics, num_agents, checkpoint_trained_agents):
    """Print formatted evaluation results for standalone evaluation."""
    print("\n" + "=" * 80)
    print("QUANTITATIVE EVALUATION RESULTS")
    print("=" * 80)

    if num_agents != checkpoint_trained_agents:
        print(f"\n*** GENERALIZATION TEST ***")
        print(f"  Checkpoint trained with: {checkpoint_trained_agents} agents")
        print(f"  Evaluating with:         {num_agents} agents")
    else:
        print(f"\n  Evaluating with: {num_agents} agents (same as training)")

    print(f"\nMetrics:")
    print(
        f"  Completion Rate:    {eval_metrics['completion_rate']:.2f}% ({eval_metrics['completed_episodes']}/{eval_metrics['total_episodes']} episodes)"
    )
    print(
        f"  Avg Episode Length: {eval_metrics['avg_episode_length']:.2f} ± {eval_metrics['std_episode_length']:.2f} steps"
    )
    print(
        f"  Avg Reward:         {eval_metrics['avg_reward']:.3f} ± {eval_metrics['std_reward']:.3f}"
    )

    # Print collision metrics if available
    if "avg_first_collision_time" in eval_metrics:
        print(
            f"  First Collision:    {eval_metrics['avg_first_collision_time']:.2f} ± {eval_metrics['std_first_collision_time']:.2f} steps"
        )
    if "avg_collisions_per_episode" in eval_metrics:
        print(
            f"  Collisions/Episode: {eval_metrics['avg_collisions_per_episode']:.2f} ± {eval_metrics['std_collisions_per_episode']:.2f}"
        )

    # Print SMAX win rate if available
    if "win_rate" in eval_metrics:
        print(
            f"  Win Rate:           {eval_metrics['win_rate']:.2f}% ({eval_metrics['wins']}/{eval_metrics['battles']} battles)"
        )

    # Print football win rate and goals if available
    if "football_win_rate" in eval_metrics:
        print(
            f"  Football Goal Win Rate:  {eval_metrics['football_win_rate']:.1f}% "
            f"(Blue {eval_metrics['football_blue_goals']} - {eval_metrics['football_red_goals']} Red, "
            f"{eval_metrics['football_total_goals']} total goals)"
        )
    if "football_episode_win_rate" in eval_metrics:
        n = (
            eval_metrics["football_wins"]
            + eval_metrics["football_draws"]
            + eval_metrics["football_losses"]
        )
        print(
            f"  Football Episode Win Rate:  {eval_metrics['football_episode_win_rate']:.1f}%  "
            f"Draw Rate: {eval_metrics['football_episode_draw_rate']:.1f}%  "
            f"Loss Rate: {100.0 - eval_metrics['football_episode_win_rate'] - eval_metrics['football_episode_draw_rate']:.1f}%  "
            f"(W {eval_metrics['football_wins']} / D {eval_metrics['football_draws']} / L {eval_metrics['football_losses']}, {n} episodes)"
        )

    # Print food collection percentage if available
    if "food_found_percentage" in eval_metrics:
        print(f"  Food Found:         {eval_metrics['food_found_percentage']:.2f}%")

    # Print agents at goal percentage if available
    if "avg_agents_at_goal_percentage" in eval_metrics:
        print(
            f"  Agents At Goal:     {eval_metrics['avg_agents_at_goal_percentage']:.2f}% ± {eval_metrics['std_agents_at_goal_percentage']:.2f}%"
        )

    print("=" * 80)


def run_independent_evaluation(
    env,
    policy_state,
    policy_model,
    num_agents,
    num_envs,
    obs_dim,
    action_dim,
    torch_device,
    jax_device,
    use_cuda,
    np_rng,
    num_eval_episodes=20,
    max_eval_steps=200,
    randomize_capabilities=False,
    fixed_capabilities=None,
    verbose=True,
    calculate_snd_metric=True,
    max_agents=None,
    package_mass=None,
    scenario_name=None,
    config=None,
    diversity_scaling=1.0,
):
    """
    Run quantitative evaluation for independent policies (no hypernetwork).

    Args:
        env: VMAS environment
        policy_state: TrainState containing stacked policy parameters for all agents
        policy_model: The Actor model class (nn.Module)
        num_agents: Number of agents
        num_envs: Number of parallel environments
        obs_dim: Observation dimension
        action_dim: Action dimension
        torch_device: PyTorch device
        jax_device: JAX device
        use_cuda: Whether CUDA is enabled
        np_rng: NumPy random generator for capability randomization
        num_eval_episodes: Number of evaluation episodes
        max_eval_steps: Maximum steps per episode
        randomize_capabilities: If False, use fixed capabilities
        fixed_capabilities: Optional dict with 'speeds' and 'lidar_ranges' lists.
                           If None, uses default pattern [0.5, 1.5] and [0.3, 0.7].
        verbose: Whether to print progress messages
        calculate_snd_metric: Whether to calculate and return SND (default: True)
        package_mass: Package mass for reverse_transport scenario (default: None)
        scenario_name: Scenario name for scenario-specific handling
        config: Configuration dictionary with model settings (required for dynamic environment changes)

    Returns:
        dict: Evaluation metrics including completion_rate, avg_episode_length, avg_reward,
              and optionally 'snd' if calculate_snd_metric=True
    """
    import torch

    policy_class_name = type(policy_model).__name__
    use_cash = policy_class_name == "CASHPolicy"
    use_dico = (policy_class_name.startswith("DiCo")) and (not use_cash)
    cash_expected_obs_dim = (
        int(config["model"].get("cash_expected_obs_dim", -1))
        if (config is not None and use_cash)
        else -1
    )
    if use_cash and cash_expected_obs_dim <= 0:
        try:
            _pp = policy_state.params
            if (
                hasattr(_pp, "keys")
                and "hyper_dense_1" in _pp
                and hasattr(_pp["hyper_dense_1"], "keys")
                and "kernel" in _pp["hyper_dense_1"]
            ):
                _hyper_in = int(np.asarray(_pp["hyper_dense_1"]["kernel"]).shape[0])
                _cap_dim = int(getattr(policy_model, "capability_dim", 0))
                _cand = _hyper_in - 2 * _cap_dim
                if _cand > 0:
                    cash_expected_obs_dim = _cand
                    if config is not None:
                        config.setdefault("model", {})[
                            "cash_expected_obs_dim"
                        ] = cash_expected_obs_dim
        except Exception:
            pass
    env_context_dim = int(config["model"].get("env_context_dim", 0)) if config else 0

    cash_param_obs_dim = -1
    cash_param_hyper_in_dim = -1
    cash_param_cap_dim = -1
    if use_cash:
        try:
            _pp = policy_state.params
            if (
                hasattr(_pp, "keys")
                and "Dense_0" in _pp
                and hasattr(_pp["Dense_0"], "keys")
                and "kernel" in _pp["Dense_0"]
            ):
                cash_param_obs_dim = int(np.asarray(_pp["Dense_0"]["kernel"]).shape[0])
            if (
                hasattr(_pp, "keys")
                and "hyper_dense_1" in _pp
                and hasattr(_pp["hyper_dense_1"], "keys")
                and "kernel" in _pp["hyper_dense_1"]
            ):
                cash_param_hyper_in_dim = int(
                    np.asarray(_pp["hyper_dense_1"]["kernel"]).shape[0]
                )
            if cash_param_obs_dim > 0 and cash_param_hyper_in_dim > 0:
                _cand_cap = (cash_param_hyper_in_dim - cash_param_obs_dim) // 2
                if _cand_cap > 0:
                    cash_param_cap_dim = _cand_cap
                    cash_expected_obs_dim = cash_param_obs_dim
                    if config is not None:
                        config.setdefault("model", {})[
                            "cash_expected_obs_dim"
                        ] = cash_expected_obs_dim
        except Exception:
            pass
    append_env_context_to_obs = (
        (not use_dico)
        and (not use_cash)
        and scenario_name in ("reverse_transport", "pressure_plate")
        and env_context_dim > 0
    )

    def _get_env_context_vector():
        if scenario_name == "reverse_transport" and hasattr(env, "scenario"):
            scenario = env.scenario
            package_mass = float(
                getattr(
                    scenario,
                    "package_mass",
                    getattr(getattr(scenario, "package", None), "mass", 0.0),
                )
            )
            package_width = float(getattr(scenario, "package_width", 0.0))
            package_length = float(getattr(scenario, "package_length", 0.0))
            return jnp.asarray([package_mass, package_width, package_length])

        if scenario_name == "pressure_plate":
            return jnp.zeros((env_context_dim,), dtype=jnp.float32)

        return None

    def _append_env_context(obs_batch):
        if not append_env_context_to_obs:
            return obs_batch

        env_context_vec = _get_env_context_vector()
        if env_context_vec is None:
            return obs_batch

        num_envs_inner = obs_batch.shape[0]
        n_agents_inner = obs_batch.shape[1]
        context_batch = jnp.broadcast_to(
            env_context_vec[None, None, :],
            (num_envs_inner, n_agents_inner, env_context_dim),
        )
        return jnp.concatenate([obs_batch, context_batch], axis=-1)

    if append_env_context_to_obs and verbose:
        print(
            f"  Independent baseline eval: appending env_context_dim={env_context_dim} to observations"
        )
    if use_cash and verbose:
        print(f"  CASH eval expected obs dim: {cash_expected_obs_dim}")
        if cash_param_obs_dim > 0 or cash_param_hyper_in_dim > 0:
            print(
                "  CASH checkpoint dims: "
                f"obs_from_Dense_0={cash_param_obs_dim}, "
                f"hyper_in_from_hyper_dense_1={cash_param_hyper_in_dim}, "
                f"cap_from_params={cash_param_cap_dim}"
            )

    def _build_cash_capability_batch(agent_speeds, agent_lidar_ranges):
        """Build per-agent CASH capability vectors with shape (num_envs, num_agents, cap_dim)."""
        cap_dim = (
            int(cash_param_cap_dim)
            if cash_param_cap_dim > 0
            else int(getattr(policy_model, "capability_dim", 0))
        )
        if cap_dim <= 0:
            return jnp.zeros((num_envs, num_agents, 0), dtype=jnp.float32)

        # Default CASH capability features: [speed, lidar_range].
        base = np.stack(
            [
                np.asarray(agent_speeds[:num_agents], dtype=np.float32),
                np.asarray(agent_lidar_ranges[:num_agents], dtype=np.float32),
            ],
            axis=-1,
        )  # (num_agents, 2)

        if base.shape[-1] < cap_dim:
            pad = np.zeros((num_agents, cap_dim - base.shape[-1]), dtype=np.float32)
            base = np.concatenate([base, pad], axis=-1)
        elif base.shape[-1] > cap_dim:
            base = base[:, :cap_dim]

        cap = np.broadcast_to(base[np.newaxis, :, :], (num_envs, num_agents, cap_dim))
        return jnp.asarray(cap)

    if use_cash:

        @jax.jit
        def _get_actions_cash(
            policy_params, obs_batch, capability_batch, hidden_states
        ):
            """Get deterministic CASH actions (GRU + capability-conditioned decoder)."""
            num_envs_inner = obs_batch.shape[0]
            n_agents_inner = obs_batch.shape[1]
            batch_size_inner = num_envs_inner * n_agents_inner

            obs_flat = obs_batch.reshape(batch_size_inner, -1)
            cap_flat = capability_batch.reshape(batch_size_inner, -1)

            obs_seq = obs_flat[None, ...]
            dones_seq = jnp.zeros((1, batch_size_inner), dtype=bool)
            cap_seq = cap_flat[None, ...]
            policy_x = (obs_seq, dones_seq, cap_seq)

            new_hidden, output = policy_model.apply(
                {"params": policy_params}, hidden_states, policy_x
            )
            mean_seq, _ = output
            mean = jnp.clip(mean_seq[0], -1.0, 1.0)
            return new_hidden, mean.reshape(num_envs_inner, n_agents_inner, -1)

    elif use_dico:
        # Debug: check params shape before JIT
        if verbose and hasattr(policy_state, "params"):
            _ps = policy_state.params
            if hasattr(_ps, "keys") and "homo_hidden_0" in _ps:
                _k = _ps["homo_hidden_0"].get("kernel")
                if _k is not None:
                    print(
                        f"[DEBUG] run_independent_evaluation received homo_hidden_0 kernel shape: {np.asarray(_k).shape}"
                    )

        @jax.jit
        def _get_actions(policy_params, obs_batch):
            """Get deterministic actions for DiCo policies using agent IDs."""
            num_envs_inner = obs_batch.shape[0]
            n_agents_inner = obs_batch.shape[1]
            obs_batch_aug = _append_env_context(obs_batch)
            obs_flat = obs_batch_aug.reshape(num_envs_inner * n_agents_inner, -1)
            agent_ids = jnp.tile(jnp.arange(n_agents_inner), num_envs_inner)
            # Wrap agent_ids with modulo so eval agents beyond training count reuse weights
            agent_ids = agent_ids % policy_model.num_agents
            mean, _ = policy_model.apply(
                {"params": policy_params},
                obs_flat,
                agent_ids,
                diversity_scaling,
            )
            mean = jnp.clip(mean, -1.0, 1.0)
            return mean.reshape(num_envs_inner, n_agents_inner, -1)

    else:

        @jax.jit
        def _get_actions(policy_params, obs_batch):
            """Get deterministic actions by applying a shared policy to all agents."""
            num_envs_inner = obs_batch.shape[0]
            n_agents_inner = obs_batch.shape[1]
            obs_batch_aug = _append_env_context(obs_batch)
            obs_flat = obs_batch_aug.reshape(num_envs_inner * n_agents_inner, -1)
            mean, _ = policy_model.apply({"params": policy_params}, obs_flat, {})
            mean = jnp.clip(mean, -1.0, 1.0)
            return mean.reshape(num_envs_inner, n_agents_inner, -1)

    total_rewards = []
    episode_lengths = []
    completed_episodes = 0
    agents_at_goal_percentages = []  # Track percentage of agents at goal

    # Track reward decomposition over all episodes
    reward_components = {
        "food_collection": [],
        "shaping": [],
        "time_penalty": [],
    }

    # Track food collection for dispersion
    total_food_available = 0
    total_food_collected = 0

    # Check if this is dispersion environment
    is_dispersion_env = (
        hasattr(env, "scenario")
        and hasattr(env.scenario, "__class__")
        and hasattr(env.scenario.__class__, "__module__")
        and "dispersion" in env.scenario.__class__.__module__.lower()
    )

    if verbose:
        print(f"\nRunning {num_eval_episodes} evaluation episodes...")
        print(f"  Parallel envs: {num_envs}")
        print(f"  Max steps per episode: {max_eval_steps}")
        print(f"  Randomize capabilities: {randomize_capabilities}")

    for eval_ep in range(num_eval_episodes):
        # Setup capabilities
        if randomize_capabilities:
            agent_speeds = np_rng.uniform(0.5, 1.5, size=num_agents).tolist()
            agent_lidar_ranges = np_rng.uniform(0.3, 0.7, size=num_agents).tolist()
        else:
            # Use fixed capabilities from config or default pattern
            if fixed_capabilities is not None:
                fixed_speeds = fixed_capabilities.get("speed", [])
                fixed_lidar_ranges = fixed_capabilities.get("lidar_range", [])
                if (
                    len(fixed_speeds) >= num_agents
                    and len(fixed_lidar_ranges) >= num_agents
                ):
                    agent_speeds = fixed_speeds[:num_agents]
                    agent_lidar_ranges = fixed_lidar_ranges[:num_agents]
                else:
                    # Fall back to pattern if not enough values
                    agent_speeds = []
                    agent_lidar_ranges = []
                    speed_values = (
                        fixed_speeds if len(fixed_speeds) >= 2 else [0.5, 1.5]
                    )
                    # All agents get 0.5 lidar range (matching training config)
                    for i in range(num_agents):
                        agent_speeds.append(speed_values[i % len(speed_values)])
                        agent_lidar_ranges.append(0.5)
            else:
                # Default pattern
                agent_speeds = []
                agent_lidar_ranges = []
                speed_values = [0.5, 1.5]
                for i in range(num_agents):
                    agent_speeds.append(speed_values[i % 2])
                    # All agents get 0.5 lidar range (matching training config)
                    agent_lidar_ranges.append(0.5)

        agent_capabilities = {"speed": agent_speeds, "lidar_range": agent_lidar_ranges}
        # Skip capability updates for football (capabilities are fixed in wrapper) and SMAX (JAX-based env)
        if hasattr(env, "scenario") and hasattr(
            env.scenario, "update_agent_capabilities"
        ):
            env.scenario.update_agent_capabilities(agent_capabilities)

        # Update package mass for reverse_transport before reset
        if scenario_name == "reverse_transport" and package_mass is not None:
            if hasattr(env, "scenario"):
                env.scenario.package_mass = package_mass
                if hasattr(env.scenario, "package"):
                    env.scenario.package.mass = package_mass

        # Reset environment (SMAX requires JAX random key)
        if hasattr(env, "unit_type_names"):  # SMAX environment
            rng_key = jax.random.PRNGKey(np_rng.integers(0, 2**31))
            obs_dict, env_state = env.reset(rng_key)
            obs = [obs_dict[agent] for agent in env.agents]
            # Convert JAX arrays to PyTorch tensors and add batch dimension
            obs = [torch.from_numpy(np.array(o)).float().to(torch_device) for o in obs]
            obs = [o.unsqueeze(0).expand(num_envs, -1) for o in obs]
        else:
            obs = env.reset()

        # VMAS returns list of observations
        obs_stacked = torch.stack(obs, dim=0)  # (num_agents, num_envs, obs_dim)
        obs_transposed = obs_stacked.transpose(0, 1)  # (num_envs, num_agents, obs_dim)

        obs_np = obs_transposed.cpu().numpy()
        jax_obs = jnp.asarray(obs_np)
        if use_cuda:
            jax_obs = jax.device_put(jax_obs, jax_device)

        if use_cash:
            if cash_expected_obs_dim > 0 and jax_obs.shape[-1] != cash_expected_obs_dim:
                if jax_obs.shape[-1] > cash_expected_obs_dim:
                    jax_obs = jax_obs[:, :, :cash_expected_obs_dim]
                    if eval_ep == 0 and verbose:
                        print(
                            f"[CASH] Cropped obs dim from {obs_transposed.shape[-1]} to expected {cash_expected_obs_dim}"
                        )
                else:
                    pad_dim = cash_expected_obs_dim - jax_obs.shape[-1]
                    jax_obs = jnp.concatenate(
                        [
                            jax_obs,
                            jnp.zeros(
                                (jax_obs.shape[0], jax_obs.shape[1], pad_dim),
                                dtype=jax_obs.dtype,
                            ),
                        ],
                        axis=-1,
                    )
                    if eval_ep == 0 and verbose:
                        print(
                            f"[CASH] Padded obs dim from {obs_transposed.shape[-1]} to expected {cash_expected_obs_dim}"
                        )

            cash_capability_batch = _build_cash_capability_batch(
                agent_speeds, agent_lidar_ranges
            )
            if use_cuda:
                cash_capability_batch = jax.device_put(
                    cash_capability_batch, jax_device
                )
            batch_size = num_envs * num_agents
            gru_hidden_states = jnp.zeros((batch_size, policy_model.gru_hidden_dim))
            if use_cuda:
                gru_hidden_states = jax.device_put(gru_hidden_states, jax_device)
        else:
            cash_capability_batch = None
            gru_hidden_states = None

        # For DiCo with dispersion_vmas, append max_speed capability to observations
        # This must match training where capability context was appended
        use_capability_context = (
            config["model"].get("use_capability_context", True) if config else False
        )
        if (
            use_dico
            and scenario_name == "dispersion_vmas"
            and use_capability_context
            and agent_speeds is not None
        ):
            # agent_speeds: list of max_speed values for each agent
            # jax_obs shape: (num_envs, num_agents, obs_dim)
            # Need to append max_speed for each agent to get (num_envs, num_agents, obs_dim + 1)
            max_speed_array = np.array(agent_speeds[:num_agents], dtype=np.float32)
            # Broadcast to (num_envs, num_agents, 1)
            max_speed_expanded = np.broadcast_to(
                max_speed_array[np.newaxis, :, np.newaxis], (num_envs, num_agents, 1)
            )
            max_speed_jax = jnp.asarray(max_speed_expanded)
            if use_cuda:
                max_speed_jax = jax.device_put(max_speed_jax, jax_device)
            jax_obs = jnp.concatenate([jax_obs, max_speed_jax], axis=-1)
            if eval_ep == 0 and verbose:
                print(
                    f"[DiCo] Appended max_speed to observations, new shape: {jax_obs.shape}"
                )

        episode_reward = np.zeros(num_envs)
        episode_steps = np.zeros(num_envs)
        dones_all = np.zeros(num_envs, dtype=bool)

        # Track food collection for dispersion scenario
        if is_dispersion_env:
            # Get actual number of food items from environment (supports fixed_n_food config)
            num_food_items = len(env.world.landmarks)
            total_food_available += num_envs * num_food_items
            episode_food_collected = 0
            # Store initial state of food (to track new collections)
            food_initially_eaten = [food.eaten.clone() for food in env.world.landmarks]

        for step in range(max_eval_steps):
            # Debug: print obs shape on first step
            if step == 0 and eval_ep == 0 and verbose:
                print(f"[DEBUG] jax_obs shape before _get_actions: {jax_obs.shape}")

            # Get actions from independent policies
            if use_cash:
                gru_hidden_states, jax_actions = _get_actions_cash(
                    policy_state.params,
                    jax_obs,
                    cash_capability_batch,
                    gru_hidden_states,
                )
            else:
                jax_actions = _get_actions(policy_state.params, jax_obs)

            actions_np = np.asarray(jax_actions)
            # NaN protection: replace NaN with 0 and clip to valid range
            actions_np = np.nan_to_num(actions_np, nan=0.0, posinf=1.0, neginf=-1.0)
            actions_np = np.clip(actions_np, -1.0, 1.0)

            torch_actions = []
            for agent_idx in range(num_agents):
                agent_actions = torch.tensor(
                    actions_np[:, agent_idx, :],
                    device=torch_device,
                    dtype=torch.float32,
                )
                torch_actions.append(agent_actions)

            # Step environment
            next_obs, rewards, dones, info = env.step(torch_actions)

            # Track food collection for dispersion by checking landmark.eaten status
            if is_dispersion_env:
                # Count how many food items are eaten across all envs
                for food_idx, food in enumerate(env.world.landmarks):
                    # Count newly eaten food (wasn't eaten initially, is eaten now)
                    newly_eaten = food.eaten & ~food_initially_eaten[food_idx]
                    episode_food_collected += newly_eaten.sum().item()
                    # Update tracking
                    food_initially_eaten[food_idx] = food.eaten.clone()

            # Track reward decomposition if available in info
            if info and len(info) > 0:
                first_agent_info = info[0] if isinstance(info, list) else info
                if isinstance(first_agent_info, dict):
                    if "reward_food_collection" in first_agent_info:
                        value = first_agent_info["reward_food_collection"]
                        if hasattr(value, "item"):
                            value = value.item()
                        reward_components["food_collection"].append(value)
                    if "reward_shaping" in first_agent_info:
                        value = first_agent_info["reward_shaping"]
                        if hasattr(value, "item"):
                            value = value.item()
                        reward_components["shaping"].append(value)
                    if "reward_time_penalty" in first_agent_info:
                        value = first_agent_info["reward_time_penalty"]
                        if hasattr(value, "item"):
                            value = value.item()
                        reward_components["time_penalty"].append(value)
                    # Track percentage of agents at goal
                    if "percentage_agents_at_goal" in first_agent_info:
                        value = first_agent_info["percentage_agents_at_goal"]
                        if hasattr(value, "cpu"):
                            value = value.cpu().numpy()
                        agents_at_goal_percentages.extend(value)

            # ================================================================
            # Reverse transport: dynamic environment changes (for independent policies)
            # Note: Independent policies don't requery adapters (no hypernetwork),
            # but we still apply environment changes to match training conditions
            # ================================================================
            if scenario_name == "reverse_transport" and config is not None:
                use_dynamic_env = config["env"].get("use_dynamic_env_changes", False)
                env_change_interval = config["env"].get("env_change_interval", 0)

                # Debug output on first episode, first step
                if eval_ep == 0 and step == 1:
                    print(
                        f"\n[Independent Eval Dynamic Env] use_dynamic_env_changes={use_dynamic_env}, env_change_interval={env_change_interval}"
                    )

                if (
                    use_dynamic_env
                    and env_change_interval > 0
                    and step > 0
                    and step % env_change_interval == 0
                ):
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
                        # Gradually increase/decrease properties
                        progress = min(step / max_eval_steps, 1.0)
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
                    else:
                        new_mass, new_width, new_length = None, None, None

                    # Update properties in environment
                    if new_mass is not None and hasattr(env, "scenario"):
                        changed = env.scenario.update_package_properties(
                            package_mass=new_mass,
                            package_width=new_width,
                            package_length=new_length,
                        )
                        if changed and verbose:
                            print(
                                f"  [Independent Eval Ep {eval_ep+1}, Step {step}] Scheduled package change: "
                                f"mass={new_mass:.2f}, width={new_width:.2f}, length={new_length:.2f}"
                            )

            # Process rewards
            rewards_stacked = torch.stack(rewards, dim=0)  # (num_agents, num_envs)
            rewards_sum = rewards_stacked.sum(dim=0).cpu().numpy()  # (num_envs,)
            episode_reward += rewards_sum * (1 - dones_all)
            episode_steps += 1 - dones_all

            # Process dones - handle both list of tensors and single tensor
            if isinstance(dones, torch.Tensor):
                dones_env = dones.cpu().numpy()
            else:
                dones_stacked = torch.stack(dones, dim=0)
                dones_env = dones_stacked.all(dim=0).cpu().numpy()
            dones_all = np.logical_or(dones_all, dones_env)

            if dones_all.all():
                break

            # Update obs
            obs_stacked = torch.stack(next_obs, dim=0)
            obs_transposed = obs_stacked.transpose(0, 1)
            obs_np = obs_transposed.cpu().numpy()
            jax_obs = jnp.asarray(obs_np)
            if use_cuda:
                jax_obs = jax.device_put(jax_obs, jax_device)

            if (
                use_cash
                and cash_expected_obs_dim > 0
                and jax_obs.shape[-1] != cash_expected_obs_dim
            ):
                if jax_obs.shape[-1] > cash_expected_obs_dim:
                    jax_obs = jax_obs[:, :, :cash_expected_obs_dim]
                else:
                    pad_dim = cash_expected_obs_dim - jax_obs.shape[-1]
                    jax_obs = jnp.concatenate(
                        [
                            jax_obs,
                            jnp.zeros(
                                (jax_obs.shape[0], jax_obs.shape[1], pad_dim),
                                dtype=jax_obs.dtype,
                            ),
                        ],
                        axis=-1,
                    )

            # Append max_speed for DiCo dispersion_vmas (must match training)
            if (
                use_dico
                and scenario_name == "dispersion_vmas"
                and use_capability_context
                and agent_speeds is not None
            ):
                max_speed_array = np.array(agent_speeds[:num_agents], dtype=np.float32)
                max_speed_expanded = np.broadcast_to(
                    max_speed_array[np.newaxis, :, np.newaxis],
                    (num_envs, num_agents, 1),
                )
                max_speed_jax = jnp.asarray(max_speed_expanded)
                if use_cuda:
                    max_speed_jax = jax.device_put(max_speed_jax, jax_device)
                jax_obs = jnp.concatenate([jax_obs, max_speed_jax], axis=-1)

        # Update total food collected for dispersion
        if is_dispersion_env:
            total_food_collected += episode_food_collected
            if verbose and (eval_ep + 1) % 5 == 0:
                print(
                    f"    Episode {eval_ep + 1}: {episode_food_collected}/{num_envs * num_food_items} food collected"
                )

        total_rewards.append(np.mean(episode_reward))
        episode_lengths.append(np.mean(episode_steps))
        completed_episodes += np.sum(dones_all)

        # Progress indicator
        if verbose and (eval_ep + 1) % 5 == 0:
            print(f"  Completed {eval_ep + 1}/{num_eval_episodes} episodes...")

    # Total number of episodes run = num_eval_episodes * num_envs
    total_episodes = num_eval_episodes * num_envs

    # Compute average reward decomposition
    reward_decomposition = {}
    if reward_components["food_collection"]:
        reward_decomposition["food_collection"] = np.mean(
            reward_components["food_collection"]
        )
    if reward_components["shaping"]:
        reward_decomposition["shaping"] = np.mean(reward_components["shaping"])
    if reward_components["time_penalty"]:
        reward_decomposition["time_penalty"] = np.mean(
            reward_components["time_penalty"]
        )

    # Compute metrics (match format of run_quantitative_evaluation)
    completion_rate = (completed_episodes / total_episodes) * 100.0
    avg_episode_length = np.mean(episode_lengths)
    std_episode_length = np.std(episode_lengths)
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    metrics = {
        "completion_rate": completion_rate,
        "completed_episodes": completed_episodes,
        "total_episodes": total_episodes,
        "avg_episode_length": avg_episode_length,
        "std_episode_length": std_episode_length,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
    }

    # Add reward decomposition if available
    if reward_decomposition:
        metrics["reward_decomposition"] = reward_decomposition

    # Add food collection percentage for dispersion
    if total_food_available > 0:
        food_found_percentage = (total_food_collected / total_food_available) * 100.0
        metrics["food_found_percentage"] = food_found_percentage
        if verbose:
            print(
                f"  Food collection: {total_food_collected}/{total_food_available} ({food_found_percentage:.2f}%)"
            )
    elif is_dispersion_env:
        # Dispersion detected but no food tracked - this is a bug
        print(f"  [WARNING] Dispersion environment but total_food_available = 0")

    # Add agents at goal percentage if available
    if agents_at_goal_percentages:
        metrics["avg_agents_at_goal_percentage"] = (
            np.mean(agents_at_goal_percentages) * 100.0
        )
        metrics["std_agents_at_goal_percentage"] = (
            np.std(agents_at_goal_percentages) * 100.0
        )

    # Calculate SND if requested
    if calculate_snd_metric:
        try:
            if use_cash:
                if verbose:
                    print("  SND: skipped for CASH in run_independent_evaluation")
                metrics["snd"] = None
                return metrics

            # Reset environment to get fresh observations for SND calculation
            if hasattr(env, "unit_type_names"):  # SMAX environment
                rng_key = jax.random.PRNGKey(np_rng.integers(0, 2**31))
                obs_dict, env_state = env.reset(rng_key)
                obs = [obs_dict[agent] for agent in env.agents]
                # Convert JAX arrays to PyTorch tensors and add batch dimension
                obs = [
                    torch.from_numpy(np.array(o)).float().to(torch_device) for o in obs
                ]
                obs = [o.unsqueeze(0).expand(num_envs, -1) for o in obs]
            else:
                obs = env.reset()

            # Setup capabilities for SND calculation (use fixed pattern)
            agent_speeds = []
            agent_lidar_ranges = []
            speed_values = [0.5, 1.5]
            for i in range(num_agents):
                agent_speeds.append(speed_values[i % 2])
                # All agents get 0.5 lidar range (matching training config)
                agent_lidar_ranges.append(0.5)

            agent_capabilities = {
                "speed": agent_speeds,
                "lidar_range": agent_lidar_ranges,
            }
            # Skip capability updates for football (capabilities are fixed in wrapper) and SMAX (JAX-based env)
            if hasattr(env, "scenario") and hasattr(
                env.scenario, "update_agent_capabilities"
            ):
                env.scenario.update_agent_capabilities(agent_capabilities)

            # Convert observations for SND
            obs_stacked = torch.stack(obs, dim=0)
            obs_transposed = obs_stacked.transpose(0, 1)
            obs_transposed_np = obs_transposed.cpu().numpy()
            obs_transposed_jax = jnp.asarray(obs_transposed_np)
            obs_transposed_jax = _append_env_context(obs_transposed_jax)
            batch_size = num_envs * num_agents
            jax_obs = obs_transposed_jax.reshape(batch_size, -1)

            if use_cuda:
                jax_obs = jax.device_put(jax_obs, jax_device)

            # Calculate SND for independent policies
            if use_dico:
                snd_value = calculate_snd_dico(
                    policy_params=policy_state.params,
                    obs_batch=jax_obs,
                    policy_model=policy_model,
                    num_agents=num_agents,
                    num_envs=num_envs,
                    diversity_scaling=diversity_scaling,
                )
            else:
                snd_value = calculate_snd_independent(
                    policy_params=policy_state.params,
                    obs_batch=jax_obs,
                    policy_model=policy_model,
                    num_agents=num_agents,
                    num_envs=num_envs,
                )
            metrics["snd"] = float(snd_value)

            if verbose:
                print(f"  SND: {snd_value:.6f}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to calculate SND: {e}")
            metrics["snd"] = None

    return metrics


def generate_gif_independent(
    env,
    policy_state,
    policy_model,
    num_agents,
    num_envs,
    obs_dim,
    agent_capabilities,
    checkpoint_dir,
    n_steps=200,
    torch_device="cpu",
    jax_device=None,
    use_cuda=False,
):
    """Generate a GIF visualization of independent policies (no hypernetwork)."""
    import imageio
    from datetime import datetime

    env_context_dim = 0
    if hasattr(env, "scenario"):
        if (
            hasattr(env.scenario, "package_mass")
            and hasattr(env.scenario, "package_width")
            and hasattr(env.scenario, "package_length")
        ):
            env_context_dim = 3

    def _append_env_context(obs_batch):
        if env_context_dim != 3 or not hasattr(env, "scenario"):
            return obs_batch

        scenario = env.scenario
        package_mass = float(
            getattr(
                scenario,
                "package_mass",
                getattr(getattr(scenario, "package", None), "mass", 0.0),
            )
        )
        package_width = float(getattr(scenario, "package_width", 0.0))
        package_length = float(getattr(scenario, "package_length", 0.0))

        num_envs_inner = obs_batch.shape[0]
        n_agents_inner = obs_batch.shape[1]
        env_context_vec = jnp.asarray([package_mass, package_width, package_length])
        env_context_batch = jnp.broadcast_to(
            env_context_vec[None, None, :],
            (num_envs_inner, n_agents_inner, env_context_dim),
        )
        return jnp.concatenate([obs_batch, env_context_batch], axis=-1)

    # Define JIT-compiled action getter (shared policy — flatten then reshape)
    @jax.jit
    def _get_actions(policy_params, obs_batch):
        num_envs_inner = obs_batch.shape[0]
        n_agents_inner = obs_batch.shape[1]
        obs_batch_aug = _append_env_context(obs_batch)
        obs_flat = obs_batch_aug.reshape(num_envs_inner * n_agents_inner, -1)
        mean, _ = policy_model.apply({"params": policy_params}, obs_flat, {})
        mean = jnp.clip(mean, -1.0, 1.0)
        return mean.reshape(num_envs_inner, n_agents_inner, -1)

    print(f"Generating GIF with {n_steps} steps...")

    # Update environment capabilities
    if agent_capabilities is not None:
        # Skip capability updates for football (capabilities are fixed in wrapper) and SMAX (JAX-based env)
        if hasattr(env, "scenario") and hasattr(
            env.scenario, "update_agent_capabilities"
        ):
            env.scenario.update_agent_capabilities(agent_capabilities)

    # Reset environment (SMAX requires JAX random key)
    if hasattr(env, "unit_type_names"):  # SMAX environment
        rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        obs_dict, env_state = env.reset(rng_key)
        obs = [obs_dict[agent] for agent in env.agents]
        # Convert JAX arrays to PyTorch tensors (no batch expansion for gif eval - single env)
        obs = [torch.from_numpy(np.array(o)).float().to(torch_device) for o in obs]
    else:
        obs = env.reset()
    frames = []

    for step in range(n_steps):
        # Render frame
        frame = env.render(
            mode="rgb_array",
            agent_index_focus=None,
            visualize_when_rgb=True,
        )
        frames.append(frame)

        # Convert observations to JAX format
        obs_stacked = torch.stack(obs, dim=0)
        obs_transposed = obs_stacked.transpose(0, 1)
        obs_np = obs_transposed.cpu().numpy()
        jax_obs = jnp.asarray(obs_np)

        if use_cuda and jax_device is not None:
            jax_obs = jax.device_put(jax_obs, jax_device)

        # Get actions
        jax_actions = _get_actions(policy_state.params, jax_obs)
        actions_np = np.asarray(jax_actions)
        # NaN protection: replace NaN with 0 and clip to valid range
        actions_np = np.nan_to_num(actions_np, nan=0.0, posinf=1.0, neginf=-1.0)
        actions_np = np.clip(actions_np, -1.0, 1.0)

        # Convert to torch format for environment
        torch_actions = []
        for agent_idx in range(num_agents):
            agent_action = torch.tensor(
                actions_np[:, agent_idx, :],
                device=torch_device,
                dtype=torch.float32,
            )
            torch_actions.append(agent_action)

        # Step environment
        obs, rewards, dones, info = env.step(torch_actions)

        if dones.all():
            print(f"All environments completed at step {step}")
            break

    # Save GIF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_filename = f"policy_eval_independent_{timestamp}.gif"
    gif_path = checkpoint_dir / gif_filename

    imageio.mimsave(gif_path, frames, fps=10)
    print(f"GIF saved to: {gif_path}")

    return gif_path


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("Policy Evaluation (HyperLoRA / Independent)")
    print("=" * 80)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {args.config}")
    print(f"  Scenario: {config['env'].get('scenario_name', 'N/A')}")
    print(
        f"  Dynamic env changes: {config['env'].get('use_dynamic_env_changes', False)}"
    )
    print(f"  Env change interval: {config['env'].get('env_change_interval', 0)}")

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)

    print(f"\nUsing checkpoint: {checkpoint_path}")

    # Load checkpoint
    if args.checkpoint_step is not None:
        checkpoint_filename = f"checkpoint_{args.checkpoint_step}.npz"
    else:
        checkpoint_filename = "final_checkpoint.npz"
    checkpoint_file = checkpoint_path / checkpoint_filename
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_file}")
        sys.exit(1)

    checkpoint_data = np.load(checkpoint_file, allow_pickle=True)
    print("Checkpoint loaded successfully")

    # Auto-detect checkpoint type
    is_hyperlora = "hn_params" in checkpoint_data
    is_independent = "policy_params" in checkpoint_data and not is_hyperlora

    if not is_hyperlora and not is_independent:
        print("\nError: Invalid checkpoint format - missing required parameters.")
        sys.exit(1)

    if is_hyperlora:
        print("\nDetected: HyperLoRA checkpoint (with hypernetwork)")
    else:
        print("\nDetected: Independent Policies checkpoint (no hypernetwork)")

    # Try to load current_snd_ma from checkpoint (for proper diversity scaling)
    checkpoint_current_snd_ma = None
    if "current_snd_ma" in checkpoint_data:
        checkpoint_current_snd_ma = float(checkpoint_data["current_snd_ma"])
        print(
            f"  Loaded current_snd_ma from checkpoint: {checkpoint_current_snd_ma:.6f}"
        )

    # Load checkpoint config if available (check both checkpoint dir and logs dir)
    checkpoint_config = None
    checkpoint_name = checkpoint_path.name  # e.g., "hyperlora_vmas_20251201_174141"

    # Try checkpoint directory first
    checkpoint_config_path = checkpoint_path / "config.yaml"
    if checkpoint_config_path.exists():
        with open(checkpoint_config_path, "r") as f:
            checkpoint_config = yaml.safe_load(f)
        print(f"Loaded checkpoint config from: {checkpoint_config_path}")
    else:
        # Try logs directory with same name
        logs_config_path = Path("logs") / checkpoint_name / "config.yaml"
        if logs_config_path.exists():
            with open(logs_config_path, "r") as f:
                checkpoint_config = yaml.safe_load(f)
            print(f"Loaded checkpoint config from: {logs_config_path}")
        else:
            print(f"\nWARNING: No config.yaml found for checkpoint {checkpoint_name}")
            print(f"         Checked: {checkpoint_config_path}")
            print(f"         Checked: {logs_config_path}")
            print(
                f"         Using current config.yaml settings (may cause shape mismatches!)"
            )

    # If the user did not provide a custom config path, prefer checkpoint config
    # as the evaluation base to avoid mismatches in replicated/generalization runs.
    if checkpoint_config is not None and args.config == "config.yaml":
        config = checkpoint_config
        print("Using checkpoint config as evaluation base")

    if checkpoint_config is not None:
        checkpoint_env = checkpoint_config.get("env", {})
        checkpoint_model = checkpoint_config.get("model", {})
        config_env = config.setdefault("env", {})
        config_model = config.setdefault("model", {})

        # Warn if there are mismatches
        if "num_agents" in checkpoint_env and checkpoint_env[
            "num_agents"
        ] != config_env.get("num_agents"):
            print(
                f"\nWARNING: Checkpoint was trained with {checkpoint_env['num_agents']} agents"
            )
            print(
                f"         but config.yaml specifies {config_env.get('num_agents')} agents"
            )
            print(
                f"         Using {checkpoint_env['num_agents']} agents from checkpoint"
            )
            config_env["num_agents"] = checkpoint_env["num_agents"]

        if "lora_mode" in checkpoint_model and checkpoint_model[
            "lora_mode"
        ] != config_model.get("lora_mode"):
            print(
                f"\nWARNING: Checkpoint was trained with lora_mode='{checkpoint_model['lora_mode']}'"
            )
            print(
                f"         but config.yaml specifies lora_mode='{config_model.get('lora_mode')}'"
            )
            print(f"         Using '{checkpoint_model['lora_mode']}' from checkpoint")
            config_model["lora_mode"] = checkpoint_model["lora_mode"]

        if "lora_rank" in checkpoint_model and checkpoint_model[
            "lora_rank"
        ] != config_model.get("lora_rank"):
            print(
                f"\nWARNING: Checkpoint was trained with lora_rank={checkpoint_model['lora_rank']}"
            )
            print(
                f"         but config.yaml specifies lora_rank={config_model.get('lora_rank')}"
            )
            print(
                f"         Using lora_rank={checkpoint_model['lora_rank']} from checkpoint"
            )
            config_model["lora_rank"] = checkpoint_model["lora_rank"]

        # Override all hypernetwork architecture settings from checkpoint
        hypernetwork_keys = [
            "transformer_dim",
            "transformer_heads",
            "transformer_layers",
            "task_embed_dim",
            "use_capability_context",
            "use_onehot_context",
            "use_positional_context",
            "positional_encoding_dim",
            "use_lidar_context",
            "policy_hidden_dims",
            "lora_scaling_factor",
        ]
        for key in hypernetwork_keys:
            if key in checkpoint_model and checkpoint_model[key] != config_model.get(
                key
            ):
                print(
                    f"\nWARNING: Checkpoint was trained with {key}={checkpoint_model[key]}"
                )
                print(
                    f"         but config.yaml specifies {key}={config_model.get(key)}"
                )
                print(f"         Using {key}={checkpoint_model[key]} from checkpoint")
                config_model[key] = checkpoint_model[key]

        # Sync environment-specific parameters from checkpoint config
        # This ensures evaluation uses the same env settings as training
        env_keys_to_sync = [
            "package_mass",
            "package_width",
            "package_length",
            "continuous_goals",
            "goal_completion_bonus",
            "use_fixed_capabilities",
            "fixed_capabilities",
        ]
        # When a custom config is provided (not the default config.yaml),
        # don't overwrite capability settings that the user explicitly set.
        # This allows evaluate_capability_generalization.py (and similar scripts)
        # to override capabilities via a temp config file.
        custom_config_provided = args.config != "config.yaml"
        capability_keys = {
            "use_fixed_capabilities",
            "fixed_capabilities",
            "package_mass",
        }
        for key in env_keys_to_sync:
            if key in checkpoint_env:
                if (
                    custom_config_provided
                    and key in capability_keys
                    and key in config_env
                ):
                    print(
                        f"\nKeeping eval config override for: {key} (custom config takes priority)"
                    )
                    continue
                if key not in config_env or checkpoint_env[key] != config_env.get(key):
                    print(
                        f"\nSyncing env parameter from checkpoint: {key}={checkpoint_env[key]}"
                    )
                    config_env[key] = checkpoint_env[key]

    # Determine number of agents
    if args.num_agents is not None:
        num_agents = args.num_agents
    else:
        num_agents = config["env"]["num_agents"]

    # Determine max_agents (for dispersion_vmas one-hot encoding dimension)
    if checkpoint_config and "max_agents" in checkpoint_config["env"]:
        max_agents = checkpoint_config["env"]["max_agents"]
    else:
        max_agents = config["env"].get("max_agents", num_agents)

    print(f"\nEvaluating with {num_agents} agents (max_agents={max_agents})")

    # Setup devices
    use_cuda = False
    torch_device = "cpu"
    jax_device = jax.devices("cpu")[0] if not use_cuda else jax.devices("gpu")[0]

    print(f"Using device: {torch_device}")

    # Create heterogeneous agent capabilities
    np_rng = np.random.default_rng(args.seed)

    # Check if fixed capabilities are enabled in config
    use_fixed_capabilities = config["env"].get("use_fixed_capabilities", False)

    # Override with command line flag
    randomize_capabilities = args.randomize_capabilities

    # Parse command line fixed capabilities if provided
    cli_fixed_speeds = None
    cli_fixed_lidar = None
    if args.fixed_speeds:
        cli_fixed_speeds = [float(x) for x in args.fixed_speeds.split(",")]
    if args.fixed_lidar:
        cli_fixed_lidar = [float(x) for x in args.fixed_lidar.split(",")]

    # Determine scenario name (needed before capability loading for reverse_transport)
    if args.scenario:
        scenario_name = args.scenario
    elif checkpoint_config and "scenario" in checkpoint_config["env"]:
        scenario_name = checkpoint_config["env"]["scenario"]
    elif checkpoint_config and "scenario_name" in checkpoint_config["env"]:
        scenario_name = checkpoint_config["env"]["scenario_name"]
    else:
        # Default to dispersion
        scenario_name = config["env"].get("scenario", "dispersion")

    print(f"\nResolved scenario: {scenario_name}")

    # ========================================================================
    # Setup wandb logging (optional)
    # ========================================================================
    use_wandb = WANDB_AVAILABLE and not args.no_logging
    if use_wandb:
        # Create a descriptive run name
        checkpoint_type = "hyperlora" if is_hyperlora else "independent"
        wandb_name = (
            args.wandb_name
            or f"eval_{checkpoint_type}_{scenario_name}_{num_agents}agents"
        )

        # Initialize wandb
        wandb.init(
            project=config.get("logging", {}).get("wandb_project", "hyperlora-eval"),
            entity=config.get("logging", {}).get("wandb_entity"),
            name=wandb_name,
            config={
                **config,
                "eval_checkpoint": str(checkpoint_path),
                "eval_num_agents": num_agents,
                "eval_num_episodes": args.num_eval_episodes,
                "eval_max_steps": args.max_eval_steps,
            },
            tags=["evaluation"] + config.get("experiment", {}).get("tags", []),
        )
        print(f"Wandb initialized: {wandb.run.url}")
    else:
        if not WANDB_AVAILABLE:
            print("Wandb not available - skipping wandb logging")
        else:
            print("Wandb logging disabled (--no-logging flag)")

    # Determine if we should use capability context from checkpoint config
    use_capability_context_eval = config["model"].get("use_capability_context", True)

    # Only modify agent capabilities if the model was trained WITH capability context
    agent_capabilities = None
    if use_capability_context_eval:
        # Priority: CLI args > config fixed capabilities > auto-generate
        if cli_fixed_speeds is not None and cli_fixed_lidar is not None:
            # Use command line specified capabilities
            if (
                len(cli_fixed_speeds) != num_agents
                or len(cli_fixed_lidar) != num_agents
            ):
                print(
                    f"ERROR: --fixed-speeds and --fixed-lidar must have exactly {num_agents} values"
                )
                sys.exit(1)
            fixed_speeds = cli_fixed_speeds
            fixed_lidar_ranges = cli_fixed_lidar
            agent_capabilities = {
                "speed": fixed_speeds,
                "lidar_range": fixed_lidar_ranges,
            }
            print(f"\nUsing FIXED capabilities (from command line):")
        elif use_fixed_capabilities and not randomize_capabilities:
            # Use fixed capabilities from config
            fixed_caps = config["env"].get("fixed_capabilities", {})
            fixed_speeds = fixed_caps.get("speed", [])
            fixed_lidar_ranges = fixed_caps.get("lidar_range", [])
            fixed_force_multipliers = fixed_caps.get("force_multiplier", [])
            fixed_max_speeds = fixed_caps.get("max_speed", [])

            # Different scenarios use different capability keys
            if scenario_name == "reverse_transport":
                has_enough = (
                    len(fixed_speeds) >= num_agents
                    and len(fixed_force_multipliers) >= num_agents
                )
            elif scenario_name == "dispersion_vmas":
                has_enough = len(fixed_max_speeds) >= num_agents
            else:
                has_enough = (
                    len(fixed_speeds) >= num_agents
                    and len(fixed_lidar_ranges) >= num_agents
                )

            if has_enough:
                if scenario_name == "reverse_transport":
                    fixed_speeds = fixed_speeds[:num_agents]
                    fixed_force_multipliers = fixed_force_multipliers[:num_agents]
                elif scenario_name == "dispersion_vmas":
                    fixed_max_speeds = fixed_max_speeds[:num_agents]
                else:
                    fixed_speeds = fixed_speeds[:num_agents]
                    fixed_lidar_ranges = fixed_lidar_ranges[:num_agents]
            else:
                # Auto-generate if not enough values provided
                print(
                    f"WARNING: Fixed capabilities list lengths insufficient "
                    f"for num_agents={num_agents}"
                )
                print(f"Auto-generating fixed capabilities with distinct values...")
                # Generate distinct capability combinations
                fixed_speeds = []
                fixed_lidar_ranges = []
                speed_values = [0.5, 1.5]  # Low and high speeds
                for i in range(num_agents):
                    fixed_speeds.append(speed_values[i % 2])
                    # All agents get 0.5 lidar range (matching training config)
                    fixed_lidar_ranges.append(0.5)

            if scenario_name == "reverse_transport":
                agent_capabilities = {
                    "speed": fixed_speeds,
                    "force_multiplier": fixed_force_multipliers,
                }
            elif scenario_name == "dispersion_vmas":
                agent_capabilities = {
                    "max_speed": fixed_max_speeds,
                }
            else:
                agent_capabilities = {
                    "speed": fixed_speeds,
                    "lidar_range": fixed_lidar_ranges,
                }
            print(f"\nUsing FIXED capabilities (from config):")
        else:
            # Use randomized capabilities (same range as training)
            if scenario_name == "reverse_transport":
                agent_capabilities = {
                    "speed": np_rng.uniform(0.3, 0.7, num_agents).tolist(),
                    "force_multiplier": np_rng.uniform(3.0, 10.0, num_agents).tolist(),
                }
            elif scenario_name == "dispersion_vmas":
                agent_capabilities = {
                    "max_speed": np_rng.uniform(0.5, 2.5, num_agents).tolist(),
                }
            else:
                agent_capabilities = {
                    "speed": np_rng.uniform(0.5, 1.5, num_agents).tolist(),
                    "lidar_range": np_rng.uniform(0.3, 0.7, num_agents).tolist(),
                }
            print(f"\nUsing RANDOMIZED capabilities (same range as training):")

        print(f"\nInitial agent capabilities (for env creation):")
        for i in range(num_agents):
            if scenario_name == "reverse_transport":
                print(
                    f"  Agent {i}: speed={agent_capabilities['speed'][i]:.2f}, "
                    f"force_multiplier={agent_capabilities.get('force_multiplier', [1.0]*num_agents)[i]:.2f}"
                )
            elif scenario_name == "dispersion_vmas":
                print(
                    f"  Agent {i}: max_speed={agent_capabilities['max_speed'][i]:.2f}"
                )
            else:
                print(
                    f"  Agent {i}: speed={agent_capabilities['speed'][i]:.2f}, "
                    f"lidar_range={agent_capabilities['lidar_range'][i]:.2f}"
                )
    else:
        print(
            f"\nModel trained WITHOUT capability context - using DEFAULT agent speeds"
        )
        print(f"Agents will use standard VMAS speeds (not modified)")

    print(f"\nEvaluating scenario: {scenario_name}")

    # Create environment based on scenario
    num_envs = args.num_envs

    if scenario_name == "simple_tag":
        # Import simple_tag scenario
        import simple_tag

        # Get num_adversaries from config
        num_adversaries = config["env"].get("num_adversaries", 3)
        num_good_agents = num_agents - num_adversaries

        print(
            f"Simple Tag: {num_adversaries} adversaries, {num_good_agents} good agents"
        )

        env = make_vmas_env(
            scenario_name="simple_tag",
            num_agents=num_agents,
            num_envs=num_envs,
            device=torch_device,
            continuous_actions=True,
            num_adversaries=num_adversaries,
        )
    elif scenario_name == "grassland":
        # Import grassland scenario
        import grassland_vmas

        # Get num_adversaries from config
        num_adversaries = config["env"].get("num_adversaries", 6)
        num_good_agents = config["env"].get("num_agents", 6)

        print(
            f"Grassland: {num_adversaries} adversaries, {num_good_agents} good agents"
        )

        env = make_vmas_env(
            scenario_name="grassland",
            num_agents=num_agents,
            num_envs=num_envs,
            device=torch_device,
            continuous_actions=True,
            n_agents_good=num_good_agents,
            n_agents_adversaries=num_adversaries,
            obs_agents=config["env"].get("obs_agents", True),
            ratio=config["env"].get("ratio", 5),
        )
    elif scenario_name == "smax":
        # SMAX environment
        print(f"Creating SMAX environment")
        map_name = config["env"].get("map_name", "3m")

        env = make_vmas_env(
            scenario_name="smax",
            num_agents=num_agents,
            num_envs=1,  # SMAX doesn't vectorize
            device=torch_device,
            continuous_actions=False,  # Discrete actions
            map_name=map_name,
            num_allies=config["env"].get("num_allies", 3),
            num_enemies=config["env"].get("num_enemies", 3),
            map_width=config["env"].get("map_width", 32),
            map_height=config["env"].get("map_height", 32),
            observation_type=config["env"].get("observation_type", "unit_list"),
            action_type=config["env"].get("action_type", "discrete"),
        )
    elif scenario_name == "reverse_transport":
        print(f"Creating reverse_transport environment")
        # Build capability dict for reverse_transport: speeds + force_multipliers.
        # evaluate.py stores force_multiplier values under the 'lidar_range' key
        # (matching the config convention where 'lidar_ranges' holds force values).
        # For reverse_transport, always read capabilities consistently from config
        fixed_caps = config["env"].get("fixed_capabilities", {})

        def _expand_capability_list(values, target_len, default_value, capability_name):
            values = list(values) if values is not None else []
            if len(values) == 0:
                values = [default_value]
            if len(values) < target_len:
                print(
                    f"WARNING: fixed_capabilities.{capability_name} has {len(values)} values, "
                    f"but num_agents={target_len}. Repeating pattern to match requested team size."
                )
                return [values[i % len(values)] for i in range(target_len)]
            return values[:target_len]

        speed_values = _expand_capability_list(
            fixed_caps.get("speed", [0.5]), num_agents, 0.5, "speed"
        )
        force_values = _expand_capability_list(
            fixed_caps.get("force_multiplier", [1.0]),
            num_agents,
            1.0,
            "force_multiplier",
        )

        # Build rt_agent_capabilities from config (both speed and force_multiplier)
        rt_agent_capabilities = {
            "speed": speed_values,
            "force_multiplier": force_values,
        }

        # Get env parameters
        pkg_width = config["env"].get("package_width", 0.6)
        pkg_length = config["env"].get("package_length", 0.6)
        pkg_mass = config["env"].get("package_mass", 50)
        pkg_mass_range = config["env"].get("package_mass_range", [1, 100])
        cont_goals = config["env"].get("continuous_goals", False)
        goal_bonus = config["env"].get("goal_completion_bonus", 10.0)

        print(f"\nReverse Transport Environment Settings:")
        print(f"  Package: {pkg_width}x{pkg_length}, mass={pkg_mass}")
        print(f"  Package mass range: {pkg_mass_range}")
        print(f"  Continuous goals: {cont_goals}, bonus: {goal_bonus}")
        print(f"  Agent capabilities:")
        for i in range(num_agents):
            speed = (
                rt_agent_capabilities["speed"][i]
                if i < len(rt_agent_capabilities["speed"])
                else 0.5
            )
            force = (
                rt_agent_capabilities["force_multiplier"][i]
                if i < len(rt_agent_capabilities["force_multiplier"])
                else 1.0
            )
            print(f"    Agent {i}: speed={speed:.2f}, force_multiplier={force:.2f}")

        env = make_vmas_env(
            scenario_name="reverse_transport",
            num_agents=num_agents,
            num_envs=num_envs,
            device=torch_device,
            continuous_actions=True,
            package_width=pkg_width,
            package_length=pkg_length,
            package_mass=pkg_mass,
            package_mass_range=pkg_mass_range,
            continuous_goals=cont_goals,
            goal_completion_bonus=goal_bonus,
            agent_capabilities=rt_agent_capabilities,
        )
    elif scenario_name == "pressure_plate":
        print(f"Creating pressure_plate environment")
        # Build capability dict for pressure_plate: just speeds (no force_multiplier)
        # For pressure_plate, always read capabilities consistently from config
        fixed_caps = config["env"].get("fixed_capabilities", {})

        # Build pp_agent_capabilities from config (only speed)
        pp_agent_capabilities = {
            "speed": fixed_caps.get("speed", [1.0] * num_agents)[:num_agents],
        }

        # Get env parameters
        n_ground_robots = config["env"].get("n_ground_robots", num_agents)
        x_semidim = config["env"].get("x_semidim", 2.0)
        y_semidim = config["env"].get("y_semidim", 2.0)
        plate_radius = config["env"].get("plate_radius", 0.15)
        plate_margin = config["env"].get("plate_margin", 0.8)
        door_size = config["env"].get("door_size", 0.6)
        goal_radius = config["env"].get("goal_radius", 0.3)
        with_drone = config["env"].get("with_drone", False)
        use_global_obs = config["env"].get("use_global_obs", True)
        plate_reward = config["env"].get("plate_reward", 0.1)
        goal_reward = config["env"].get("goal_reward", 10.0)
        time_penalty = config["env"].get("time_penalty", -0.01)

        print(f"\nPressure Plate Environment Settings:")
        print(f"  Ground robots: {n_ground_robots}, With drone: {with_drone}")
        print(f"  Global observability: {use_global_obs}")
        print(f"  Area: {x_semidim}x{y_semidim}, plate_margin: {plate_margin}")
        print(f"  Agent capabilities:")
        for i in range(num_agents):
            speed = (
                pp_agent_capabilities["speed"][i]
                if i < len(pp_agent_capabilities["speed"])
                else 1.0
            )
            print(f"    Agent {i}: speed={speed:.2f}")

        env = make_vmas_env(
            scenario_name="pressure_plate",
            num_agents=num_agents,
            num_envs=num_envs,
            device=torch_device,
            continuous_actions=True,
            n_ground_robots=n_ground_robots,
            x_semidim=x_semidim,
            y_semidim=y_semidim,
            plate_radius=plate_radius,
            plate_margin=plate_margin,
            door_size=door_size,
            goal_radius=goal_radius,
            with_drone=with_drone,
            use_global_obs=use_global_obs,
            plate_reward=plate_reward,
            goal_reward=goal_reward,
            time_penalty=time_penalty,
            agent_capabilities=pp_agent_capabilities,
            eval_mode=True,
        )
    else:
        # Dispersion or other scenarios
        env_kwargs = {
            "scenario_name": scenario_name,
            "num_agents": num_agents,
            "num_envs": num_envs,
            "device": torch_device,
            "continuous_actions": True,
            "penalise_by_time": False,
            "share_reward": False,
            "distance_shaping_coef": 3.0,
            "agent_capabilities": agent_capabilities,
        }
        if scenario_name == "dispersion_vmas":
            env_kwargs["fixed_food_positions"] = config["env"].get(
                "fixed_food_positions", False
            )
            # Pass fixed_n_food so observation dimension matches training
            fixed_n_food = config["env"].get("fixed_n_food", None)
            if fixed_n_food is not None:
                env_kwargs["fixed_n_food"] = fixed_n_food

        env = make_vmas_env(**env_kwargs)

    print(f"Environment created: {num_agents} agents, {num_envs} parallel envs")

    # Get dimensions (handle SMAX differently)
    if scenario_name == "smax":
        rng_key = jax.random.PRNGKey(args.seed)
        temp_obs_dict, temp_state = env.reset(rng_key)
        first_agent = env.agents[0]
        obs_dim = temp_obs_dict[first_agent].shape[-1]
        action_dim = env.action_spaces[first_agent].n  # Discrete
    else:
        temp_obs = env.reset()
        obs_dim = temp_obs[0].shape[-1]
        action_dim = env.get_agent_action_size(env.agents[0])

    print(f"Environment dimensions: obs_dim={obs_dim}, action_dim={action_dim}")

    # Initialize models
    print("\n" + "=" * 80)
    print("Initializing models")
    print("=" * 80)

    if is_hyperlora:
        # ====================================================================
        # HyperLoRA Checkpoint Evaluation
        # ====================================================================
        policy_dims = {
            "obs_dim": obs_dim,
            "hidden_dims": config["model"]["policy_hidden_dims"],
            "action_dim": action_dim,
            "lora_rank": config["model"]["lora_rank"],
        }

        # Use capability context settings from config (checkpoint may have been trained without it)
        use_capability_context = config["model"].get("use_capability_context", True)
        use_onehot_context = config["model"].get("use_onehot_context", True)
        use_positional_context = config["model"].get("use_positional_context", False)
        positional_encoding_dim = config["model"].get("positional_encoding_dim", 16)

        # Set context_dim based on scenario (must match training!)
        if scenario_name == "smax":
            # SMAX uses unit type one-hot (6 dimensions)
            context_dim = 6 if use_capability_context else 0
        elif scenario_name == "simple_tag" or scenario_name == "grassland":
            # Use full observation as context
            context_dim = obs_dim if use_capability_context else 0
        elif scenario_name == "reverse_transport":
            # Use 2-D [speed, force_multiplier] vector matching training
            context_dim = 2
        elif scenario_name == "pressure_plate":
            # Use 1-D [speed] vector matching training
            context_dim = 1 if use_capability_context else 0
        elif scenario_name == "dispersion_vmas":
            # For dispersion_vmas: positional encoding (scalable) OR one-hot encoding (legacy)
            if use_positional_context:
                context_dim = (
                    1 if use_capability_context else 0
                ) + positional_encoding_dim
            else:
                context_dim = (1 if use_capability_context else 0) + (
                    max_agents if use_onehot_context else 0
                )
        else:
            # Other scenarios use capability vectors [speed, lidar_range]
            context_dim = 2 if use_capability_context else 0

        # Lidar context configuration - lidar readings from observations
        use_lidar_context = config["model"].get("use_lidar_context", False)
        # dispersion_vmas and pressure_plate use global observability (no lidar)
        if scenario_name in ["dispersion_vmas", "pressure_plate"]:
            lidar_dim = 0
        else:
            lidar_dim = (
                obs_dim - 8 if use_lidar_context else 0
            )  # Last dims are lidar readings

        # Get scaling factor from config (MUST match training!)
        lora_scaling_factor = config["model"].get("lora_scaling_factor", 1.0)

        print(f"\nModel configuration:")
        print(f"  use_capability_context: {use_capability_context}")
        print(f"  context_dim: {context_dim} (capability features)")
        print(f"  use_lidar_context: {use_lidar_context}")
        print(f"  lidar_dim: {lidar_dim} (from observations)")
        print(f"  lora_scaling_factor: {lora_scaling_factor}")

        # Get target_snd from config (used as context to hypernetwork)
        target_snd = float(config["training"].get("target_snd", 0.01))
        # Override with command-line argument if provided
        if args.target_snd is not None:
            target_snd = float(args.target_snd)
            print(f"  Overriding target_snd with command-line value: {target_snd}")
        target_snd_dim = config["model"].get("target_snd_dim", 0)

        # Determine current_snd_ma for diversity scaling
        # Priority: 1) command-line arg, 2) checkpoint, 3) default to target_snd (no scaling)
        if args.current_snd_ma is not None:
            current_snd_ma_eval = args.current_snd_ma
            print(
                f"  Using current_snd_ma from command line: {current_snd_ma_eval:.6f}"
            )
        elif checkpoint_current_snd_ma is not None:
            current_snd_ma_eval = checkpoint_current_snd_ma
            print(f"  Using current_snd_ma from checkpoint: {current_snd_ma_eval:.6f}")
        else:
            current_snd_ma_eval = (
                None  # Will default to target_snd in evaluation function
            )
            print(
                f"  WARNING: No current_snd_ma available - using target_snd (diversity_scaling = 1.0)"
            )
            print(
                f"    For accurate evaluation matching training, specify --current-snd-ma or save it in checkpoint"
            )

        # Calculate env_context_dim based on scenario
        # For reverse_transport: [package_mass, package_width, package_length]
        # For pressure_plate: [left_plate_pos, right_plate_pos, door_open, goal_pos]
        if scenario_name == "reverse_transport":
            env_context_dim = 3  # mass, width, length
        elif scenario_name == "pressure_plate":
            # Calculate env_context_dim based on config flags
            use_env_context = config["model"].get("use_env_context", False)
            if use_env_context:
                env_context_plate_positions = config["model"].get(
                    "env_context_plate_positions", True
                )
                env_context_door_state = config["model"].get(
                    "env_context_door_state", True
                )
                env_context_goal_position = config["model"].get(
                    "env_context_goal_position", True
                )
                use_agent_id_context = config["model"].get(
                    "use_agent_id_context", False
                )

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
            else:
                env_context_dim = 0
        else:
            # Fallback to config value if provided, otherwise 0
            env_context_dim = config["model"].get("env_context_dim", 0)

        # Auto-detect combined feature dimension from checkpoint's agent_embed kernel.
        # This prevents mismatches when config values don't match what was used during training
        # (e.g., positional_encoding_dim defaulted to 16 during training but config says 8).
        hn_params_raw = checkpoint_data["hn_params"].item()
        if "agent_embed" in hn_params_raw and "kernel" in hn_params_raw["agent_embed"]:
            checkpoint_input_dim = hn_params_raw["agent_embed"]["kernel"].shape[0]
            task_embed_dim_val = config["model"]["task_embed_dim"]
            computed_combined_dim = (
                context_dim
                + lidar_dim
                + target_snd_dim
                + env_context_dim
                + task_embed_dim_val
            )
            if checkpoint_input_dim != computed_combined_dim:
                print(
                    f"\nWARNING: Config-derived combined feature dim ({computed_combined_dim}) "
                    f"does not match checkpoint agent_embed kernel ({checkpoint_input_dim})."
                )
                # The non-context dims (lidar, target_snd, env_context, task_embed) are
                # generally reliable; adjust context_dim to absorb the difference.
                fixed_dims = (
                    lidar_dim + target_snd_dim + env_context_dim + task_embed_dim_val
                )
                corrected_context_dim = checkpoint_input_dim - fixed_dims
                if corrected_context_dim > 0:
                    print(
                        f"  Auto-correcting context_dim: {context_dim} -> {corrected_context_dim} "
                        f"(checkpoint trained with {checkpoint_input_dim}-dim input, "
                        f"fixed dims={fixed_dims})"
                    )
                    # Also fix positional_encoding_dim for dispersion_vmas so context
                    # tensor construction produces the correct number of features.
                    if scenario_name == "dispersion_vmas" and use_positional_context:
                        cap_dims = 1 if use_capability_context else 0
                        corrected_pos_dim = corrected_context_dim - cap_dims
                        if corrected_pos_dim > 0:
                            print(
                                f"  Auto-correcting positional_encoding_dim: "
                                f"{positional_encoding_dim} -> {corrected_pos_dim}"
                            )
                            positional_encoding_dim = corrected_pos_dim
                            # Update config so downstream functions also use the corrected value
                            config["model"][
                                "positional_encoding_dim"
                            ] = corrected_pos_dim
                    context_dim = corrected_context_dim
                else:
                    print(
                        f"  ERROR: Cannot auto-correct - inferred context_dim would be "
                        f"{corrected_context_dim}. Check checkpoint config manually."
                    )

        hypernetwork = Hypernetwork(
            policy_dims=policy_dims,
            context_dim=context_dim,
            task_embed_dim=config["model"]["task_embed_dim"],
            lidar_dim=lidar_dim,
            target_snd_dim=target_snd_dim,
            env_context_dim=env_context_dim,
            transformer_dim=config["model"]["transformer_dim"],
            transformer_heads=config["model"]["transformer_heads"],
            transformer_layers=config["model"]["transformer_layers"],
            lora_mode=config["model"]["lora_mode"],
            scaling_factor=lora_scaling_factor,
            max_agents=max_agents,
            use_cross_agent_attention=config["model"].get(
                "use_cross_agent_attention", True
            ),
        )

        shared_policy = LoRAPolicy(
            hidden_dims=config["model"]["policy_hidden_dims"],
            action_dim=action_dim,
            lora_mode=config["model"]["lora_mode"],
        )

        print("HyperLoRA models initialized")

        # Reconstruct state objects from checkpoint
        policy_params = checkpoint_data["policy_params"].item()
        hn_params = checkpoint_data["hn_params"].item()

        # Create dummy optimizer (not used for inference)
        dummy_optimizer = optax.adam(learning_rate=0.0)

        policy_state = TrainState.create(
            apply_fn=shared_policy.apply,
            params=policy_params,
            tx=dummy_optimizer,
        )

        hn_state = TrainState.create(
            apply_fn=hypernetwork.apply,
            params=hn_params,
            tx=dummy_optimizer,
        )

        print("State objects reconstructed from checkpoint")

        # DEBUG: Dump hn_params structure to understand checkpoint
        def _print_params(d, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    _print_params(v, prefix=f"{prefix}/{k}")
                else:
                    print(f"[PARAMS] {prefix}/{k}: shape={v.shape}, dtype={v.dtype}")

        _print_params(hn_params)
        import sys

        sys.stdout.flush()

        # Define helper functions
        def get_static_adapters(
            hn_params,
            task_batch,
            context_batch,
            lidar_batch=None,
            food_positions_batch=None,
            agent_positions_batch=None,
            target_snd_batch=None,
            env_context_batch=None,
            scaling_factor=1.0,
            diversity_scaling=1.0,
            mask=None,
        ):
            """Generate LoRA adapters."""
            # Use real food positions if available, otherwise zeros/None
            food_position_dim = hypernetwork.food_position_dim
            if food_position_dim > 0:
                if food_positions_batch is not None:
                    food_batch = food_positions_batch
                else:
                    # Fallback: use zeros if no food positions available
                    # Use num_envs and num_agents from outer scope (closure)
                    food_batch = jnp.zeros((num_envs, num_agents, food_position_dim))
            else:
                food_batch = None

            # Use real agent positions if available, otherwise zeros/None
            agent_position_dim = hypernetwork.agent_position_dim
            if agent_position_dim > 0:
                if agent_positions_batch is not None:
                    agent_pos_batch = agent_positions_batch
                else:
                    # Use num_envs and num_agents from outer scope (closure)
                    agent_pos_batch = jnp.zeros(
                        (num_envs, num_agents, agent_position_dim)
                    )
            else:
                agent_pos_batch = None

            # Use provided target_snd_batch or build from scalar target_snd
            _target_snd_dim = hypernetwork.target_snd_dim
            if _target_snd_dim > 0:
                if target_snd_batch is None:
                    # Use num_envs and num_agents from outer scope (closure)
                    _target_snd_batch = jnp.full(
                        (num_envs, num_agents, _target_snd_dim),
                        target_snd,
                        dtype=jnp.float32,
                    )
                else:
                    _target_snd_batch = target_snd_batch
            else:
                _target_snd_batch = None

            # Use provided env_context_batch or zeros/None
            _env_context_dim = hypernetwork.env_context_dim
            if _env_context_dim > 0:
                if env_context_batch is not None:
                    _env_context_batch = env_context_batch
                else:
                    # Use num_envs and num_agents from outer scope (closure)
                    _env_context_batch = jnp.zeros(
                        (num_envs, num_agents, _env_context_dim)
                    )
            else:
                _env_context_batch = None

            adapters = hypernetwork.apply(
                {"params": hn_params},
                task_batch,
                context_batch,
                lidar_batch,
                food_batch,
                agent_pos_batch,
                _target_snd_batch,
                _env_context_batch,
                mask,
                diversity_scaling,  # Use diversity_scaling parameter for hypernetwork generation
            )
            # Apply post-processing scaling if needed (typically 1.0 for GIF generation)
            scaled_adapters = {}
            for key, value in adapters.items():
                scaled_adapters[key] = value * scaling_factor
            return scaled_adapters

        @jax.jit
        def get_actions(policy_params, obs_batch, adapters_dict):
            mean, log_std = shared_policy.apply(
                {"params": policy_params}, obs_batch, adapters_dict
            )
            return jnp.clip(mean, -1.0, 1.0)

        # Get checkpoint's trained number of agents for comparison
        checkpoint_trained_agents = (
            checkpoint_config["env"]["num_agents"]
            if checkpoint_config is not None
            else num_agents
        )

        # Run quantitative evaluation
        if not args.gif_only:
            print("\n" + "=" * 80)
            print("Running Quantitative Evaluation")
            print("=" * 80)

        # Get adaptive hypernetwork flag from config
        adaptive_hypernetwork = config["model"].get("adaptive_hypernetwork", False)

        if not args.gif_only:
            eval_metrics = run_quantitative_evaluation(
                env=env,
                policy_state=policy_state,
                hn_state=hn_state,
                shared_policy=shared_policy,
                hypernetwork=hypernetwork,
                num_agents=num_agents,
                num_envs=num_envs,
                obs_dim=obs_dim,
                action_dim=action_dim,
                policy_hidden_dims=config["model"]["policy_hidden_dims"],
                context_dim=context_dim,
                lidar_dim=lidar_dim,
                task_embed_dim=config["model"]["task_embed_dim"],
                use_lidar_context=use_lidar_context,
                torch_device=torch_device,
                jax_device=jax_device,
                use_cuda=use_cuda,
                np_rng=np_rng,
                num_eval_episodes=args.num_eval_episodes,
                max_eval_steps=args.max_eval_steps,
                scenario_name=scenario_name,
                randomize_capabilities=randomize_capabilities,
                fixed_capabilities=(
                    config["env"].get("fixed_capabilities")
                    if use_fixed_capabilities and not randomize_capabilities
                    else None
                ),
                adaptive_hypernetwork=adaptive_hypernetwork,
                max_agents=max_agents,
                package_mass=(
                    config["env"].get("package_mass")
                    if scenario_name == "reverse_transport"
                    else None
                ),
                target_snd=target_snd,
                current_snd_ma=current_snd_ma_eval,
                env_context_dim=env_context_dim,
                config=config,
            )

            # Print evaluation results
            print_evaluation_results(
                eval_metrics, num_agents, checkpoint_trained_agents
            )

            # Log evaluation metrics to wandb
            if use_wandb:
                wandb_metrics = {
                    f"eval/{k}": v
                    for k, v in eval_metrics.items()
                    if isinstance(v, (int, float, bool))
                }
                wandb.log(wandb_metrics)
                print("Logged metrics to wandb")

        # Generate visualization GIF (optional)
        if not args.no_gif:
            print("\n" + "=" * 80)
            print("Generating Visualization GIF")
            print("=" * 80)

            # Calculate diversity_scaling for GIF generation
            # This mirrors the calculation in run_quantitative_evaluation
            use_diversity_control = config["training"].get(
                "use_diversity_control", False
            )

            if use_diversity_control:
                # Get parameters from config
                snd_ma_coef = config["training"].get("snd_moving_average_coef", 0.9)
                min_snd_floor = config["training"].get("min_snd_floor", 1e-6)
                max_scaling = config["training"].get("max_diversity_scaling", 5.0)

                # Use current_snd_ma if available, otherwise use target_snd
                if current_snd_ma_eval is None:
                    current_snd_for_scaling = target_snd
                else:
                    current_snd_for_scaling = current_snd_ma_eval

                # Calculate diversity scaling
                diversity_scaling = float(
                    np.sqrt(target_snd / max(current_snd_for_scaling, min_snd_floor))
                )
                diversity_scaling = float(
                    np.clip(diversity_scaling, 0.001, np.sqrt(max_scaling))
                )
                print(f"  Diversity scaling for GIF: {diversity_scaling:.6f}")
            else:
                # Diversity control disabled - use neutral scaling
                diversity_scaling = 1.0

            # CRITICAL: Update package mass for reverse_transport before generating GIF
            if scenario_name == "reverse_transport":
                # For decreasing mode, start at max mass; otherwise use config value
                use_dynamic_env = config["env"].get("use_dynamic_env_changes", False)
                env_change_type = config["env"].get("env_change_type", "random")

                if use_dynamic_env and env_change_type == "decreasing":
                    # Start at maximum mass for decreasing mode
                    mass_range = config["env"].get("package_mass_range", [1, 100])
                    package_mass = float(mass_range[1])
                    print(
                        f"  Using decreasing mode: starting at max mass={package_mass}"
                    )
                elif use_dynamic_env and env_change_type == "gradual":
                    # Start at minimum mass for gradual (increasing) mode
                    mass_range = config["env"].get("package_mass_range", [1, 100])
                    package_mass = float(mass_range[0])
                    print(f"  Using gradual mode: starting at min mass={package_mass}")
                else:
                    # Use static mass from config
                    package_mass = config["env"].get("package_mass", 50)

                if hasattr(env, "scenario"):
                    # Update scenario's stored package_mass
                    env.scenario.package_mass = package_mass
                    # Update the actual Landmark's mass attribute
                    if hasattr(env.scenario, "package"):
                        env.scenario.package.mass = package_mass
                print(f"  Package mass: {package_mass} (updated in environment)")

            # Ensure config has the correct scenario_name for render_gif
            gif_config = config.copy()
            gif_config["env"] = config["env"].copy()
            gif_config["env"]["scenario_name"] = scenario_name

            # Debug: Show dynamic env settings being used
            print(
                f"  Dynamic env changes: {gif_config['env'].get('use_dynamic_env_changes', False)}"
            )
            print(
                f"  Env change interval: {gif_config['env'].get('env_change_interval', 0)}"
            )
            print(
                f"  Env change type: {gif_config['env'].get('env_change_type', 'N/A')}"
            )

            gif_path = generate_policy_gif(
                env=env,
                policy_state=policy_state,
                hn_state=hn_state,
                adapters_dict=None,
                config=gif_config,
                checkpoint_dir=checkpoint_path,
                n_steps=args.gif_steps,
                use_hypernetwork=True,
                adaptive_hypernetwork=adaptive_hypernetwork,
                num_agents=num_agents,
                obs_dim=obs_dim,
                context_dim=context_dim,
                task_embed_dim=config["model"]["task_embed_dim"],
                lidar_dim=lidar_dim,
                env_context_dim=env_context_dim,
                max_agents=max_agents,
                diversity_scaling=diversity_scaling,
                use_cuda=use_cuda,
                jax_device=jax_device,
                torch_device=torch_device,
                get_static_adapters_fn=get_static_adapters,
                get_actions_fn=get_actions,
            )

            if gif_path:
                print(f"\nGIF saved to: {gif_path}")
            else:
                print("\nWarning: Failed to generate GIF")
        else:
            print("\nSkipping GIF generation (--no-gif flag set)")

    else:
        # ====================================================================
        # Independent Policies Checkpoint Evaluation
        # ====================================================================
        policy_params = checkpoint_data["policy_params"].item()

        def _is_mapping_like(obj):
            return hasattr(obj, "keys") and hasattr(obj, "__getitem__")

        def _strip_singleton_params_wrappers(tree_obj):
            """Recursively remove nodes of the form {'params': {...}}."""
            if not _is_mapping_like(tree_obj):
                return tree_obj

            if "params" in tree_obj and len(tree_obj.keys()) == 1:
                inner = tree_obj["params"]
                if _is_mapping_like(inner):
                    return _strip_singleton_params_wrappers(inner)

            normalized = {}
            for key in tree_obj.keys():
                normalized[key] = _strip_singleton_params_wrappers(tree_obj[key])
            return normalized

        def _normalize_policy_params(params_obj, use_dico):
            expected_dico_keys = {
                "hetero_network",
                "homo_hidden_0",
                "homo_hidden_1",
                "homo_mean",
                "log_std",
            }
            expected_shared_keys = {"Dense_0", "Dense_1", "Dense_2", "log_std"}

            current = params_obj
            unwrap_depth = 0

            while _is_mapping_like(current) and "params" in current:
                current_keys = set(current.keys())

                if use_dico and current_keys.intersection(expected_dico_keys):
                    break
                if (not use_dico) and current_keys.intersection(expected_shared_keys):
                    break

                next_obj = current["params"]
                if not _is_mapping_like(next_obj):
                    break

                current = next_obj
                unwrap_depth += 1

                if unwrap_depth > 4:
                    break

            return current, unwrap_depth

        def _apply_dico_key_compat(params_obj):
            """Backward-compatibility for older DiCo checkpoints.

            Some checkpoints store hetero submodules as Dense_* while current
            code expects HeteroDense_* scopes.
            """
            if not _is_mapping_like(params_obj):
                return params_obj

            if "hetero_network" not in params_obj or not _is_mapping_like(
                params_obj["hetero_network"]
            ):
                return params_obj

            hetero = params_obj["hetero_network"]
            hetero_keys = list(hetero.keys())

            has_old_dense = any(
                isinstance(k, str) and k.startswith("Dense_") for k in hetero_keys
            )
            has_new_hetero = any(
                isinstance(k, str) and k.startswith("HeteroDense_") for k in hetero_keys
            )

            if has_old_dense and not has_new_hetero:
                remapped = {}
                for key, value in hetero.items():
                    if isinstance(key, str) and key.startswith("Dense_"):
                        new_key = key.replace("Dense_", "HeteroDense_", 1)
                        remapped[new_key] = value
                    else:
                        remapped[key] = value
                params_obj = dict(params_obj)
                params_obj["hetero_network"] = remapped
                print(
                    "Applied DiCo checkpoint key compatibility remap: Dense_* -> HeteroDense_*"
                )

            return params_obj

        def _looks_like_dico_params(params_obj):
            """Infer whether checkpoint params follow DiCo structure."""
            if not _is_mapping_like(params_obj):
                return False

            dico_keys = {
                "hetero_network",
                "homo_hidden_0",
                "homo_hidden_1",
                "homo_mean",
            }
            top_keys = set(str(k) for k in params_obj.keys())
            if top_keys.intersection(dico_keys):
                return True

            # Legacy format: per-agent params keyed by integer-like strings.
            keys = list(params_obj.keys())
            if keys and all(str(k).isdigit() for k in keys):
                first_val = params_obj[keys[0]]
                if _is_mapping_like(first_val):
                    inner_keys = set(str(k) for k in first_val.keys())
                    if inner_keys.intersection({"Dense_0", "Dense_1", "Dense_2"}):
                        return True

            return False

        # Some checkpoints serialize policy params as {"params": {...}} while
        # others store the inner pytree directly. TrainState/apply expects the
        # inner params tree, so unwrap one level when needed.
        if (
            hasattr(policy_params, "keys")
            and "params" in policy_params
            and len(policy_params) == 1
            and hasattr(policy_params["params"], "keys")
        ):
            policy_params = policy_params["params"]
            print("Unwrapped top-level 'params' from independent policy checkpoint")

        use_cash_cfg = bool(config["model"].get("use_cash", False))
        use_dico_cfg = bool(config["model"].get("use_dico", False))
        use_dico_inferred = _looks_like_dico_params(policy_params)
        use_cash_checkpoint = use_cash_cfg
        use_dico_checkpoint = use_dico_cfg and (not use_cash_checkpoint)

        if (not use_cash_checkpoint) and (use_dico_cfg != use_dico_inferred):
            if use_dico_cfg and not use_dico_inferred:
                print(
                    "Config indicates DiCo, but checkpoint params look like shared-policy. "
                    "Using shared-policy evaluation path."
                )
            elif (not use_dico_cfg) and use_dico_inferred:
                print(
                    "Config indicates shared-policy, but checkpoint params look like DiCo. "
                    "Using DiCo evaluation path."
                )
            use_dico_checkpoint = use_dico_inferred

        policy_params, unwrap_depth = _normalize_policy_params(
            policy_params, use_dico_checkpoint
        )
        policy_params = _strip_singleton_params_wrappers(policy_params)
        if use_dico_checkpoint:
            policy_params = _apply_dico_key_compat(policy_params)
        if unwrap_depth > 0:
            print(
                f"Unwrapped nested 'params' in independent policy checkpoint (depth={unwrap_depth})"
            )
        if hasattr(policy_params, "keys"):
            top_keys_preview = list(policy_params.keys())[:10]
            print(f"Independent policy top-level keys: {top_keys_preview}")
            if "hetero_network" in policy_params and hasattr(
                policy_params["hetero_network"], "keys"
            ):
                hetero_keys_preview = list(policy_params["hetero_network"].keys())[:10]
                print(f"DiCo hetero_network keys: {hetero_keys_preview}")
                # Also check HeteroDense_0 kernel shape for consistency
                if "HeteroDense_0" in policy_params["hetero_network"]:
                    hd0 = policy_params["hetero_network"]["HeteroDense_0"]
                    if hasattr(hd0, "keys") and "kernel" in hd0:
                        hd0_kernel = np.asarray(hd0["kernel"])
                        print(
                            f"DiCo HeteroDense_0 kernel shape: {hd0_kernel.shape} (num_agents, in_dim, out_dim)"
                        )

        # When evaluating a DiCo checkpoint, we need to ensure the input dimension
        # matches what the checkpoint was trained with. The key variable is whether
        # capability context (max_speed) was appended during training.
        if use_dico_checkpoint and hasattr(policy_params, "keys"):
            _homo0_key = "homo_hidden_0"
            if (
                _homo0_key in policy_params
                and hasattr(policy_params[_homo0_key], "keys")
                and "kernel" in policy_params[_homo0_key]
            ):
                _stored_kernel = np.asarray(policy_params[_homo0_key]["kernel"])
                _stored_in_dim = int(_stored_kernel.shape[0])
                _use_cap_ctx = config["model"].get("use_capability_context", True)

                print(
                    f"[DiCo] Checkpoint homo_hidden_0 kernel shape: {_stored_kernel.shape}"
                )
                print(f"[DiCo] Environment obs_dim: {obs_dim}")
                print(f"[DiCo] Config use_capability_context: {_use_cap_ctx}")

                # For dispersion_vmas: capability context adds +1 for max_speed
                if scenario_name == "dispersion_vmas":
                    # Determine if checkpoint was trained with or without capability context
                    if _stored_in_dim == obs_dim:
                        # Checkpoint trained WITHOUT capability context
                        if _use_cap_ctx:
                            print(
                                f"[DiCo] Checkpoint input dim matches obs_dim ({_stored_in_dim}). "
                                f"Disabling use_capability_context for evaluation."
                            )
                            config["model"]["use_capability_context"] = False
                    elif _stored_in_dim == obs_dim + 1:
                        # Checkpoint trained WITH capability context - keep it enabled
                        if not _use_cap_ctx:
                            print(
                                f"[DiCo] Checkpoint input dim is obs_dim+1 ({_stored_in_dim}). "
                                f"Enabling use_capability_context for evaluation."
                            )
                            config["model"]["use_capability_context"] = True
                        else:
                            print(
                                f"[DiCo] Checkpoint trained WITH capability context "
                                f"(stored_dim={_stored_in_dim} == obs_dim+1={obs_dim + 1}). Keeping enabled."
                            )
                    else:
                        # Mismatch - obs_dim might be wrong (e.g., fixed_n_food not passed)
                        print(
                            f"[DiCo] WARNING: Checkpoint input dim ({_stored_in_dim}) doesn't match "
                            f"obs_dim ({obs_dim}) or obs_dim+1 ({obs_dim + 1}). "
                        )
                        # Check if fixed_n_food is in config
                        fixed_n_food_cfg = config["env"].get("fixed_n_food", None)
                        print(f"[DiCo] Config fixed_n_food: {fixed_n_food_cfg}")
                        # Try to infer correct setting based on stored dim
                        # If stored_dim suggests no cap ctx, disable it
                        if _stored_in_dim < obs_dim:
                            print(
                                f"[DiCo] stored_dim < obs_dim: checkpoint may have different food count. "
                                f"Disabling capability context and padding kernel."
                            )
                            config["model"]["use_capability_context"] = False
                            _n_new = obs_dim - _stored_in_dim
                            _pad = np.zeros(
                                (_n_new, _stored_kernel.shape[1]),
                                dtype=_stored_kernel.dtype,
                            )
                            policy_params[_homo0_key] = dict(policy_params[_homo0_key])
                            policy_params[_homo0_key]["kernel"] = np.concatenate(
                                [_stored_kernel, _pad], axis=0
                            )
                            print(
                                f"[DiCo] Padded kernel to shape {policy_params[_homo0_key]['kernel'].shape}"
                            )

        diversity_scaling_eval = 1.0
        if use_dico_checkpoint and config["training"].get(
            "use_diversity_control", False
        ):
            target_snd = float(config["training"].get("target_snd", 1.0))
            max_diversity_scaling = float(
                config["training"].get("max_diversity_scaling", 100.0)
            )

            if args.current_snd_ma is not None:
                current_snd_for_eval = max(float(args.current_snd_ma), 1e-6)
            elif checkpoint_current_snd_ma is not None:
                current_snd_for_eval = max(float(checkpoint_current_snd_ma), 1e-6)
            else:
                current_snd_for_eval = target_snd
                print(
                    "Warning: current_snd_ma not found; using target_snd, so diversity_scaling defaults to 1.0. "
                    "Pass --current-snd-ma for exact DiCo scaling."
                )

            diversity_scaling_eval = target_snd / current_snd_for_eval
            diversity_scaling_eval = float(
                np.clip(diversity_scaling_eval, 0.0, max_diversity_scaling)
            )

        # For shared-policy checkpoints the params are a single MLP (not stacked
        # per-agent), so we cannot infer num_agents from param shapes.
        # Just use the requested num_agents as-is.
        num_agents_in_checkpoint = num_agents

        if use_cash_checkpoint:

            def _infer_gru_hidden_dim_from_params(params_obj):
                """Best-effort hidden-size inference from GRU recurrent kernel shapes."""
                if not hasattr(params_obj, "keys"):
                    return None

                def _walk(node):
                    if hasattr(node, "shape") and len(node.shape) == 2:
                        r, c = int(node.shape[0]), int(node.shape[1])
                        if r > 0 and c == 3 * r:
                            return r
                    if hasattr(node, "keys"):
                        for _k in node.keys():
                            val = node[_k]
                            out = _walk(val)
                            if out is not None:
                                return out
                    return None

                return _walk(params_obj)

            def _infer_cash_dims(params_obj, obs_dim_val):
                """Infer CASH dims from checkpoint params; returns dict of optional overrides."""
                inferred = {}
                if not hasattr(params_obj, "keys"):
                    return inferred

                # Hypernetwork hidden width and input size.
                if (
                    "hyper_dense_1" in params_obj
                    and hasattr(params_obj["hyper_dense_1"], "keys")
                    and "kernel" in params_obj["hyper_dense_1"]
                ):
                    hd1_kernel = np.asarray(params_obj["hyper_dense_1"]["kernel"])
                    if hd1_kernel.ndim == 2:
                        hyper_in_dim, hyper_hidden_dim = (
                            int(hd1_kernel.shape[0]),
                            int(hd1_kernel.shape[1]),
                        )
                        inferred["hyper_hidden_dim"] = hyper_hidden_dim
                        # hyper_in = obs + capability + team_capability = obs + 2*capability_dim
                        if (
                            hyper_in_dim >= obs_dim_val
                            and (hyper_in_dim - obs_dim_val) % 2 == 0
                        ):
                            inferred["capability_dim"] = (
                                hyper_in_dim - obs_dim_val
                            ) // 2

                # Number of hyper MLP layers.
                hyper_dense_count = 0
                while f"hyper_dense_{hyper_dense_count + 1}" in params_obj:
                    hyper_dense_count += 1
                if hyper_dense_count > 0:
                    inferred["hyper_num_layers"] = hyper_dense_count

                # FC embedding width from first Dense layer.
                if (
                    "Dense_0" in params_obj
                    and hasattr(params_obj["Dense_0"], "keys")
                    and "kernel" in params_obj["Dense_0"]
                ):
                    dense0_kernel = np.asarray(params_obj["Dense_0"]["kernel"])
                    if dense0_kernel.ndim == 2:
                        inferred["fc_dim_size"] = int(dense0_kernel.shape[1])

                # GRU hidden size from recurrent kernel.
                gru_hidden = _infer_gru_hidden_dim_from_params(params_obj)
                if gru_hidden is not None:
                    inferred["gru_hidden_dim"] = int(gru_hidden)

                return inferred

            cash_num_agents = (
                checkpoint_config["env"]["num_agents"]
                if checkpoint_config is not None
                else num_agents
            )

            inferred_cash = _infer_cash_dims(policy_params, obs_dim)

            cash_capability_dim = inferred_cash.get(
                "capability_dim", config["model"].get("context_dim", 0)
            )
            if cash_capability_dim <= 0:
                # Fallback: infer from scenario conventions used in training code
                if scenario_name == "reverse_transport":
                    cash_capability_dim = 2
                elif scenario_name == "dispersion_vmas":
                    cash_capability_dim = 1
                elif scenario_name == "pressure_plate":
                    cash_capability_dim = 1
                else:
                    cash_capability_dim = 2

            # Infer expected CASH observation width from checkpoint:
            # hyper_in = obs + capability + team_capability = obs + 2 * capability_dim.
            cash_expected_obs_dim = None
            if (
                hasattr(policy_params, "keys")
                and "hyper_dense_1" in policy_params
                and hasattr(policy_params["hyper_dense_1"], "keys")
                and "kernel" in policy_params["hyper_dense_1"]
            ):
                _hd1 = np.asarray(policy_params["hyper_dense_1"]["kernel"])
                if _hd1.ndim == 2:
                    _hyper_in_dim = int(_hd1.shape[0])
                    _cand = _hyper_in_dim - 2 * int(cash_capability_dim)
                    if _cand > 0:
                        cash_expected_obs_dim = _cand
                        config.setdefault("model", {})[
                            "cash_expected_obs_dim"
                        ] = cash_expected_obs_dim

            policy_def = CASHPolicy(
                num_agents=cash_num_agents,
                capability_dim=int(cash_capability_dim),
                gru_hidden_dim=inferred_cash.get(
                    "gru_hidden_dim", config["model"].get("gru_hidden_dim", 64)
                ),
                fc_dim_size=inferred_cash.get(
                    "fc_dim_size", config["model"].get("fc_dim_size", 64)
                ),
                action_dim=action_dim,
                hyper_hidden_dim=inferred_cash.get(
                    "hyper_hidden_dim", config["model"].get("hyper_hidden_dim", 128)
                ),
                hyper_num_layers=inferred_cash.get(
                    "hyper_num_layers", config["model"].get("hyper_num_layers", 2)
                ),
                expected_hyper_input_dim=(
                    int(_hyper_in_dim)
                    if (
                        hasattr(policy_params, "keys")
                        and "hyper_dense_1" in policy_params
                        and hasattr(policy_params["hyper_dense_1"], "keys")
                        and "kernel" in policy_params["hyper_dense_1"]
                    )
                    else 0
                ),
                decoder_hidden_dim=config["model"].get("decoder_hidden_dim", 64),
                use_two_layer_decoder=config["model"].get(
                    "use_two_layer_decoder", True
                ),
                log_std_min=config["model"].get("log_std_min", -2.0),
                log_std_max=config["model"].get("log_std_max", 0.0),
                min_std=config["model"].get("min_std", 0.3),
            )
            print("Independent CASH model initialized")
            print(
                f"Loaded CASH checkpoint (trained with {cash_num_agents} agents), evaluating with {num_agents} agents"
            )
            print(
                "CASH inferred dims: "
                f"capability_dim={int(cash_capability_dim)}, "
                f"gru_hidden_dim={policy_def.gru_hidden_dim}, "
                f"fc_dim_size={policy_def.fc_dim_size}, "
                f"hyper_hidden_dim={policy_def.hyper_hidden_dim}, "
                f"hyper_num_layers={policy_def.hyper_num_layers}"
            )
            if cash_expected_obs_dim is not None:
                print(f"CASH expected obs_dim from checkpoint: {cash_expected_obs_dim}")
        elif use_dico_checkpoint:
            dico_policy_cls = (
                DiCoPolicy
                if config["training"].get("use_diversity_control", False)
                else DiCoHomogeneousPolicy
            )
            # DICO has per-agent weights, so model must be initialized with the
            # training num_agents to match checkpoint parameter shapes.
            # At inference time, agent_ids 0..eval_num_agents-1 select the right weights.
            dico_training_num_agents = (
                checkpoint_config["env"]["num_agents"]
                if checkpoint_config is not None
                else num_agents
            )
            policy_def = dico_policy_cls(
                num_agents=dico_training_num_agents,
                hidden_dims=tuple(config["model"]["policy_hidden_dims"]),
                action_dim=action_dim,
            )
            print("Independent DiCo model initialized")
            print(
                f"Loaded DiCo checkpoint (trained with {dico_training_num_agents} agents), evaluating with {num_agents} agents (diversity_scaling={diversity_scaling_eval:.6f})"
            )
        else:
            policy_def = LoRAPolicy(
                hidden_dims=tuple(config["model"]["policy_hidden_dims"]),
                action_dim=action_dim,
                lora_mode=config["model"].get("lora_mode", "final_only"),
            )
            print("Independent (shared LoRAPolicy) model initialized")
            print(
                f"Loaded shared policy checkpoint, evaluating with {num_agents} agents"
            )

        # Create dummy optimizer (not used for inference)
        dummy_optimizer = optax.adam(learning_rate=0.0)

        policy_state = TrainState.create(
            apply_fn=policy_def.apply,
            params=policy_params,
            tx=dummy_optimizer,
        )

        # Get checkpoint's trained number of agents for comparison
        checkpoint_trained_agents = (
            checkpoint_config["env"]["num_agents"]
            if checkpoint_config is not None
            else num_agents_in_checkpoint
        )

        # Run quantitative evaluation
        if not args.gif_only:
            print("\n" + "=" * 80)
            print("Running Quantitative Evaluation")
            print("=" * 80)

            eval_metrics = run_independent_evaluation(
                env=env,
                policy_state=policy_state,
                policy_model=policy_def,
                num_agents=num_agents,
                num_envs=num_envs,
                obs_dim=obs_dim,
                action_dim=action_dim,
                torch_device=torch_device,
                jax_device=jax_device,
                use_cuda=use_cuda,
                np_rng=np_rng,
                num_eval_episodes=args.num_eval_episodes,
                max_eval_steps=args.max_eval_steps,
                randomize_capabilities=randomize_capabilities,
                fixed_capabilities=(
                    config["env"].get("fixed_capabilities")
                    if use_fixed_capabilities and not randomize_capabilities
                    else None
                ),
                calculate_snd_metric=True,
                package_mass=(
                    config["env"].get("package_mass")
                    if scenario_name == "reverse_transport"
                    else None
                ),
                scenario_name=scenario_name,
                config=config,
                diversity_scaling=diversity_scaling_eval,
            )

            # Print evaluation results
            print_evaluation_results(
                eval_metrics, num_agents, checkpoint_trained_agents
            )

            # Log evaluation metrics to wandb
            if use_wandb:
                wandb_metrics = {
                    f"eval/{k}": v
                    for k, v in eval_metrics.items()
                    if isinstance(v, (int, float, bool))
                }
                wandb.log(wandb_metrics)
                print("Logged metrics to wandb")

        # Generate visualization GIF (optional)
        if not args.no_gif:
            print("\n" + "=" * 80)
            print("Generating Visualization GIF")
            print("=" * 80)

            # Calculate diversity_scaling for GIF generation
            # This mirrors the calculation in run_quantitative_evaluation
            use_diversity_control = config["training"].get(
                "use_diversity_control", False
            )

            if use_diversity_control:
                # Get parameters from config
                snd_ma_coef = config["training"].get("snd_moving_average_coef", 0.9)
                min_snd_floor = config["training"].get("min_snd_floor", 1e-6)
                max_scaling = config["training"].get("max_diversity_scaling", 5.0)

                # Use current_snd_ma if available, otherwise use target_snd
                if current_snd_ma_eval is None:
                    current_snd_for_scaling = target_snd
                else:
                    current_snd_for_scaling = current_snd_ma_eval

                # Calculate diversity scaling
                diversity_scaling = float(
                    np.sqrt(target_snd / max(current_snd_for_scaling, min_snd_floor))
                )
                diversity_scaling = float(
                    np.clip(diversity_scaling, 0.001, np.sqrt(max_scaling))
                )
                print(f"  Diversity scaling for GIF: {diversity_scaling:.6f}")
            else:
                # Diversity control disabled - use neutral scaling
                diversity_scaling = 1.0

            # CRITICAL: Update package mass for reverse_transport before generating GIF
            if scenario_name == "reverse_transport":
                package_mass = config["env"].get("package_mass", 50)
                if hasattr(env, "scenario"):
                    # Update scenario's stored package_mass
                    env.scenario.package_mass = package_mass
                    # Update the actual Landmark's mass attribute
                    if hasattr(env.scenario, "package"):
                        env.scenario.package.mass = package_mass
                print(f"  Package mass: {package_mass} (updated in environment)")

            gif_path = generate_gif_independent(
                env=env,
                policy_state=policy_state,
                policy_model=policy_def,
                num_agents=num_agents,
                num_envs=num_envs,
                obs_dim=obs_dim,
                agent_capabilities=agent_capabilities,
                checkpoint_dir=checkpoint_path,
                n_steps=args.gif_steps,
                torch_device=torch_device,
                jax_device=jax_device,
                use_cuda=use_cuda,
            )

            if gif_path:
                print(f"\nGIF saved to: {gif_path}")
            else:
                print("\nWarning: Failed to generate GIF")
        else:
            print("\nSkipping GIF generation (--no-gif flag set)")

    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
