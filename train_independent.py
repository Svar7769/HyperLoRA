import jax
import jax.numpy as jnp
import numpy as np
import torch
import optax
import distrax
from flax.training import train_state
import argparse
import yaml
from pathlib import Path
import os
from datetime import datetime
from functools import partial

from env_setup import make_vmas_env

# from lora_policy import LoRAPolicy
# from hypernetwork import Hypernetwork
from critic import CentralizedCritic
from render_gif import generate_policy_gif
from snd import (
    calculate_snd,
    calculate_snd_statistics,
    calculate_snd_independent,
    calculate_snd_statistics_independent,
)
from flax import linen as nn
from evaluate import run_independent_evaluation, print_training_eval_results
from typing import Sequence


class Actor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (64, 64)
    log_std_min: float = -2.0
    log_std_max: float = 1.0  # Reduced from 1.0 to prevent actions out of range

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action_and_log_prob(self, x, rng_key=None):
        """
        Get action and its log probability for PPO.
        Uses a Tanh-transformed distribution (matches lora_policy.py exactly).
        """
        mean, log_std = self(x)
        std = jnp.exp(log_std)

        # CRITICAL: Ensure std never collapses to 0 or becomes NaN
        # Increased minimum std to prevent entropy collapse
        std = jnp.maximum(std, 0.1)  # Minimum std for numerical stability
        std = jnp.nan_to_num(std, nan=1.0)  # Replace NaN with reasonable default

        # 1. Create the base (unbounded) Normal distribution
        base_dist = distrax.Normal(mean, std)

        # 2. Create the Tanh bijector to squash the output to [-1, 1]
        tanh_bijector = distrax.Tanh()

        # 3. Create the final transformed distribution
        dist = distrax.Transformed(base_dist, tanh_bijector)

        if rng_key is not None:
            # Sample action during rollout (automatically bounded to [-1, 1])
            action = dist.sample(seed=rng_key)

            # CRITICAL: Clip actions with epsilon to avoid exact boundary values
            # Clipping to exactly ±1.0 causes log_prob(1.0) = log(1 - 1.0^2) = log(0) = -inf
            action_epsilon = 1e-6
            action = jnp.clip(action, -1.0 + action_epsilon, 1.0 - action_epsilon)

            # Compute log probability AFTER clipping to match what's stored
            log_prob = dist.log_prob(action).sum(axis=-1)

            # Safety: Replace any remaining NaN/Inf values
            log_prob = jnp.nan_to_num(log_prob, nan=-1e10, posinf=-1e10, neginf=-1e10)
        else:
            # For evaluation, use the deterministic, squashed mean
            action = jnp.tanh(mean)
            log_prob = None

        return action, log_prob


# Configure JAX to use GPU if available
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


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

    # Visualization arguments
    parser.add_argument(
        "--render-final-policy",
        action="store_true",
        help="Generate a GIF visualization of the final trained policy",
    )
    parser.add_argument(
        "--gif-steps",
        type=int,
        default=100,
        help="Number of steps to render in the GIF (default: 100)",
    )

    return parser.parse_args()


# ============================================================================
# JAX Helper Functions (JIT-compiled)
# ============================================================================


@jax.jit
def _get_actions_and_log_probs(policy_params, obs_batch, rng_key):
    """
    Get actions and log probabilities from independent policies.

    Args:
        policy_params: Stacked policy parameters (num_agents, ...)
        obs_batch: Observations (num_envs, num_agents, obs_dim)
        rng_key: JAX random key

    Returns:
        actions: (num_envs, num_agents, action_dim)
        log_probs: (num_envs, num_agents)
    """
    # Transpose obs to (num_agents, num_envs, obs_dim)
    obs_batch_t = jnp.transpose(obs_batch, (1, 0, 2))

    # Split rng keys for each agent
    num_agents = obs_batch.shape[1]
    rng_keys = jax.random.split(rng_key, num_agents)

    def apply_agent(params, obs, key):
        return policy_def.apply(
            {"params": params}, obs, key, method=policy_def.get_action_and_log_prob
        )

    # vmap over agents
    actions_t, log_probs_t = jax.vmap(apply_agent)(policy_params, obs_batch_t, rng_keys)

    # Transpose back to (num_envs, num_agents, ...)
    actions = jnp.transpose(actions_t, (1, 0, 2))
    log_probs = jnp.transpose(log_probs_t, (1, 0))

    return actions, log_probs


@jax.jit
def _get_actions(policy_params, obs_batch):
    """
    Get deterministic actions from independent policies.
    """
    obs_batch_t = jnp.transpose(obs_batch, (1, 0, 2))

    def apply_agent(params, obs):
        mean, _ = policy_def.apply({"params": params}, obs)
        return jnp.clip(mean, -1.0, 1.0)

    actions_t = jax.vmap(apply_agent)(policy_params, obs_batch_t)
    return jnp.transpose(actions_t, (1, 0, 2))


@jax.jit
def _get_value(critic_params, global_state):
    """
    Get value estimate from centralized critic.

    Args:
        critic_params: Critic parameters
        global_state: Global state (all agents' observations concatenated)

    Returns:
        value: State value estimate
    """
    return critic.apply({"params": critic_params}, global_state)


@partial(jax.jit, static_argnames=["num_agents"])
def _train_step(
    policy_state,
    critic_state,
    obs_batch,
    global_state_batch,
    actions_batch,
    old_log_probs_batch,
    advantages_batch,
    returns_batch,
    clip_epsilon,
    entropy_coef,
    value_loss_coef,
    num_agents,
):
    """
    Perform one MAPPO training step for independent policies and centralized critic.

    PERFORMANCE IMPROVEMENTS (to match mappo_ff_nps.py):
    1. Value loss now uses normalized returns for stability
    2. Removed action clipping in log_prob computation to avoid distorting gradients
    3. Added value explained variance metric for monitoring
    4. Minibatch training added in outer loop for better sample efficiency
    """

    def loss_fn(policy_params, critic_params):
        # Critic loss (centralized)
        values = critic.apply({"params": critic_params}, global_state_batch)

        # Broadcast values to (batch_size, num_agents)
        values_expanded = jnp.tile(values[:, None], (1, num_agents))

        # Normalize returns for stable value learning (CRITICAL FIX)
        returns_mean = returns_batch.mean()
        returns_std = returns_batch.std()
        returns_normalized = (returns_batch - returns_mean) / (returns_std + 1e-8)
        values_normalized = (values_expanded - returns_mean) / (returns_std + 1e-8)
        value_loss = jnp.mean((values_normalized - returns_normalized) ** 2)

        # Calculate value explained variance for monitoring
        value_explained_var = 1 - jnp.var(returns_batch - values_expanded) / (
            jnp.var(returns_batch) + 1e-8
        )

        # Policy loss (independent)
        # Transpose inputs for vmap: (batch_size, num_agents, ...) -> (num_agents, batch_size, ...)

        obs_batch_t = jnp.transpose(obs_batch, (1, 0, 2))
        actions_batch_t = jnp.transpose(actions_batch, (1, 0, 2))
        old_log_probs_batch_t = jnp.transpose(old_log_probs_batch, (1, 0))
        advantages_batch_t = jnp.transpose(advantages_batch, (1, 0))

        def agent_loss(params, obs, act, old_lp, adv):
            # Get mean and log_std
            mean, log_std = policy_def.apply({"params": params}, obs)
            std = jnp.exp(log_std)
            std = jnp.maximum(std, 0.1)

            base_dist = distrax.Normal(mean, std)
            tanh_bijector = distrax.Tanh()
            dist = distrax.Transformed(base_dist, tanh_bijector)

            # Compute log probs without clipping (CRITICAL FIX)
            # Action clipping should only happen during environment interaction, not in loss
            new_log_probs = dist.log_prob(act).sum(axis=-1)

            log_prob_diff = new_log_probs - old_lp
            log_prob_diff = jnp.clip(log_prob_diff, -3.0, 3.0)
            ratio = jnp.exp(log_prob_diff)

            surr1 = ratio * adv
            surr2 = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
            actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            entropy = base_dist.entropy().sum(axis=-1).mean()

            loss = actor_loss - entropy_coef * entropy

            metrics = {
                "entropy": entropy,
                "mean_std": jnp.mean(std),
                "min_std": jnp.min(std),
                "max_std": jnp.max(std),
                "mean_ratio": jnp.mean(ratio),
                "approx_kl": jnp.mean(old_lp - new_log_probs),
            }
            return loss, metrics

        # Sum loss over all agents
        agent_losses, agent_metrics = jax.vmap(agent_loss)(
            policy_params,
            obs_batch_t,
            actions_batch_t,
            old_log_probs_batch_t,
            advantages_batch_t,
        )

        total_policy_loss = jnp.sum(agent_losses)
        total_loss = total_policy_loss + value_loss_coef * value_loss

        return total_loss, {
            "policy_loss": total_policy_loss,
            "value_loss": value_loss,
            "total_loss": total_loss,
            "entropy": jnp.mean(agent_metrics["entropy"]),
            "mean_std": jnp.mean(agent_metrics["mean_std"]),
            "min_std": jnp.min(agent_metrics["min_std"]),
            "max_std": jnp.max(agent_metrics["max_std"]),
            "mean_ratio": jnp.mean(agent_metrics["mean_ratio"]),
            "approx_kl": jnp.mean(agent_metrics["approx_kl"]),
            "advantages_mean": jnp.mean(advantages_batch),
            "advantages_std": jnp.std(advantages_batch),
            "advantages_max": jnp.max(advantages_batch),
            "value_explained_variance": value_explained_var,
        }

    (loss, info), grads = jax.value_and_grad(
        lambda p, c: loss_fn(p, c), argnums=(0, 1), has_aux=True
    )(policy_state.params, critic_state.params)

    policy_grads, critic_grads = grads

    # Calculate grad norms
    policy_grad_norm = optax.global_norm(policy_grads)
    critic_grad_norm = optax.global_norm(critic_grads)

    info["policy_grad_norm"] = policy_grad_norm
    info["critic_grad_norm"] = critic_grad_norm

    policy_state = policy_state.apply_gradients(grads=policy_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)

    return policy_state, critic_state, info


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

    # This script trains independent policies (no hypernetwork)
    use_hypernetwork = False

    # Print configuration
    print("=" * 80)
    print("HyperLoRA Training Configuration (Independent Policies)")
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

        print(f"Logging to: {log_dir}")
        print(f"Checkpoints to: {checkpoint_dir}")

    # ========================================================================
    # Setup Weights & Biases
    # ========================================================================
    use_wandb = args.wandb or (config["logging"].get("wandb_project") is not None)
    if use_wandb and WANDB_AVAILABLE:
        # Get wandb settings from args or config
        wandb_project = args.wandb_project or config["logging"].get(
            "wandb_project", "hyperlora-vmas"
        )
        wandb_entity = args.wandb_entity or config["logging"].get("wandb_entity")
        wandb_name = args.wandb_name or exp_name if not args.no_logging else None

        # Initialize wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config=config,
            tags=config["experiment"].get("tags", []),
        )
        print(f"Weights & Biases initialized: {wandb.run.url}")
    elif use_wandb and not WANDB_AVAILABLE:
        print(
            "Warning: wandb requested but not available. Install with: pip install wandb"
        )
        use_wandb = False
    else:
        use_wandb = False

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
    if use_cuda and torch.cuda.is_available():
        torch_device = f"cuda:{cuda_device_id}"
        print(f"PyTorch using CUDA device: {torch_device}")
    else:
        if use_cuda and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available for PyTorch. Using CPU.")
        torch_device = "cpu"
        print(f"PyTorch using device: {torch_device}")

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
    num_envs = config["env"]["num_envs"]
    continuous_actions = config["env"]["continuous_actions"]
    penalise_by_time = config["env"].get("penalise_by_time", False)
    share_reward = config["env"].get("share_reward", False)
    distance_shaping_coef = config["env"].get("distance_shaping_coef", 0.1)

    # Fixed capability and food position settings
    use_fixed_capabilities = config["env"].get("use_fixed_capabilities", False)
    fixed_food_positions = config["env"].get("fixed_food_positions", False)

    # Training
    num_episodes = config["training"]["num_episodes"]
    rollout_steps = config["training"]["rollout_steps"]
    log_interval = config["training"]["log_interval"]

    # Global state clipping
    clip_global_state = config["training"].get("clip_global_state", False)
    clip_global_state_min = config["training"].get("clip_global_state_min", -10.0)
    clip_global_state_max = config["training"].get("clip_global_state_max", 10.0)

    # Model dimensions (obs_dim and action_dim will be detected from environment)
    policy_hidden_dims = config["model"].get("policy_hidden_dims", [64, 64])
    lora_rank = config["model"]["lora_rank"]
    task_embed_dim = config["model"]["task_embed_dim"]

    # Context dimension: capability features (speed, lidar_range)
    # CTDE approach: use actual agent capabilities instead of one-hot IDs
    use_capability_context = config["model"].get("use_capability_context", True)
    context_dim = 2 if use_capability_context else 0  # [speed, lidar_range] or empty

    # Lidar context configuration
    use_lidar_context = config["model"].get("use_lidar_context", False)
    lidar_dim = 12 if use_lidar_context else 0  # Last 12 dims of obs are lidar readings

    # Hypernetwork Transformer
    transformer_dim = config["model"]["transformer_dim"]
    transformer_heads = config["model"]["transformer_heads"]
    transformer_layers = config["model"]["transformer_layers"]

    # ========================================================================
    # Environment Initialization (PyTorch/VMAS)
    # ========================================================================
    print("\nInitializing VMAS environment...")

    # Initialize RNG for capability randomization
    np_rng = np.random.default_rng(seed)

    # Setup agent capabilities (fixed or default for randomization)
    if use_fixed_capabilities:
        # Use fixed capabilities from config
        fixed_caps = config["env"].get("fixed_capabilities", {})
        fixed_speeds = fixed_caps.get("speeds", [])
        fixed_lidar_ranges = fixed_caps.get("lidar_ranges", [])

        # Auto-generate if not enough values provided or if lengths don't match
        if len(fixed_speeds) != num_agents or len(fixed_lidar_ranges) != num_agents:
            print(
                f"WARNING: Fixed capabilities list length doesn't match num_agents={num_agents}"
            )
            print(f"Auto-generating fixed capabilities with distinct values...")
            # Generate distinct capability combinations
            fixed_speeds = []
            fixed_lidar_ranges = []
            speed_values = [0.5, 1.5]  # Low and high speeds
            lidar_values = [0.3, 0.7]  # Low and high ranges
            for i in range(num_agents):
                fixed_speeds.append(speed_values[i % 2])
                fixed_lidar_ranges.append(lidar_values[(i // 2) % 2])

        default_speeds = fixed_speeds
        default_lidar_ranges = fixed_lidar_ranges
        print(f"\n{'='*80}")
        print("USING FIXED CAPABILITIES (no randomization)")
        print(f"{'='*80}")
        for i in range(num_agents):
            print(
                f"  Agent {i}: speed={default_speeds[i]:.1f}, lidar_range={default_lidar_ranges[i]:.1f}"
            )
        print(f"{'='*80}\n")
    else:
        # Use default capabilities for initial setup (will be randomized per episode)
        default_speeds = [1.0] * num_agents
        default_lidar_ranges = [0.5] * num_agents
        print(f"\nUsing RANDOMIZED capabilities (will vary each episode)")

    if fixed_food_positions:
        print(f"Using FIXED food positions in corners\n")
    else:
        print(f"Using RANDOM food positions\n")

    agent_capabilities = {"speed": default_speeds, "lidar_range": default_lidar_ranges}

    env = make_vmas_env(
        scenario_name,
        num_agents,
        num_envs,
        device=torch_device,
        continuous_actions=continuous_actions,
        penalise_by_time=penalise_by_time,
        share_reward=share_reward,
        distance_shaping_coef=distance_shaping_coef,
        agent_capabilities=agent_capabilities,
        fixed_food_positions=fixed_food_positions,
    )

    # Get actual observation and action dimensions from the environment
    print("Detecting environment dimensions...")
    temp_obs = env.reset()
    actual_obs_dim = temp_obs[0].shape[-1]  # Get obs dim from first agent
    actual_action_dim = env.get_agent_action_size(env.agents[0])

    print(f"Detected obs_dim: {actual_obs_dim}, action_dim: {actual_action_dim}")

    # Override config with actual dimensions
    obs_dim = actual_obs_dim
    action_dim = actual_action_dim

    # Calculate actual lidar dimension from observation
    # Observation structure: normalized_pos(2) + vel(2) + lidar(X)
    # So lidar_dim = obs_dim - 4 (NO food position in observations)
    if use_lidar_context:
        lidar_dim = obs_dim - 4  # Dynamically calculate based on actual obs_dim
        print(f"Calculated lidar_dim from observation: {lidar_dim}")

    print(
        f"Using obs_dim={obs_dim}, action_dim={action_dim}, context_dim={context_dim} (capability-based)"
    )
    print(f"Context encoding: Each agent gets a capability vector [speed, lidar_range]")

    # ========================================================================
    # Model Initialization (JAX/Flax)
    # ========================================================================
    print("Initializing models...")

    # Policy dimensions
    policy_dims = {
        "obs_dim": obs_dim,
        "hidden_dims": policy_hidden_dims,
        "action_dim": action_dim,
    }

    global policy_def
    policy_def = Actor(
        hidden_dims=tuple(policy_hidden_dims),
        action_dim=action_dim,
    )
    print("Using Independent PPO (separate policy per agent)")

    # Initialize centralized critic for MAPPO
    global critic
    critic_hidden_dim = config["model"]["critic_hidden_dim"]
    critic_num_layers = config["model"]["critic_num_layers"]
    global_state_dim = obs_dim * num_agents  # Concatenated observations from all agents

    critic = CentralizedCritic(
        hidden_dim=critic_hidden_dim, num_layers=critic_num_layers
    )

    # Initialize parameters with dummy inputs
    rng = jax.random.PRNGKey(0)
    rng, policy_rng, critic_rng = jax.random.split(rng, 3)

    dummy_obs = jnp.ones((1, obs_dim))
    dummy_global_state = jnp.ones((1, global_state_dim))

    # Initialize policy params for each agent
    policy_rngs = jax.random.split(policy_rng, num_agents)

    def init_policy(key):
        return policy_def.init(key, dummy_obs)["params"]

    # Stack params: (num_agents, ...)
    policy_params = jax.vmap(init_policy)(policy_rngs)

    critic_params = critic.init(critic_rng, dummy_global_state)["params"]

    # Get max grad norm from config
    max_grad_norm = config["training"].get("max_grad_norm", 0.5)

    policy_state = create_train_state(
        policy_def, policy_params, config, max_grad_norm, lr_key="learning_rate"
    )

    critic_state = create_train_state(
        critic, critic_params, config, max_grad_norm, lr_key="critic_learning_rate"
    )

    if use_cuda:
        with jax.default_device(jax_device):
            policy_state = jax.device_put(policy_state, jax_device)
            critic_state = jax.device_put(critic_state, jax_device)

    print("Models initialized successfully!")

    # ========================================================================
    # Main Training Loop
    # ========================================================================
    print("\nStarting training...")

    # Evaluation parameters from config
    eval_interval = config["evaluation"].get("eval_interval", 100)
    num_eval_episodes = config["evaluation"].get("num_eval_episodes", 10)
    max_eval_steps = config["evaluation"].get("max_eval_steps", 200)

    for episode in range(num_episodes):
        # Setup agent capabilities for this episode
        if use_fixed_capabilities:
            # Use fixed capabilities (no randomization)
            agent_speeds = default_speeds
            agent_lidar_ranges = default_lidar_ranges
        else:
            # Randomize agent capabilities for this episode (CTDE approach)
            # This enables the policy to generalize to different capability combinations
            agent_speeds = np_rng.uniform(0.5, 1.5, size=num_agents).tolist()
            agent_lidar_ranges = np_rng.uniform(0.3, 0.7, size=num_agents).tolist()

        # Update environment capabilities without recreating the environment
        agent_capabilities = {"speed": agent_speeds, "lidar_range": agent_lidar_ranges}
        env.scenario.update_agent_capabilities(agent_capabilities)

        # Reset environment (PyTorch)
        obs = env.reset()  # Shape: (num_envs, num_agents, obs_dim)

        # Create capability-based context vectors for each agent
        # Each agent gets a capability vector [speed, lidar_range]
        # Shape: (num_envs, num_agents, 2)
        if use_capability_context:
            capability_vectors = torch.tensor(
                [[agent_speeds[i], agent_lidar_ranges[i]] for i in range(num_agents)],
                device=torch_device,
                dtype=torch.float32,
            )  # (num_agents, 2)

            # Expand to include batch dimension
            static_context = capability_vectors.unsqueeze(0).expand(
                num_envs, -1, -1
            )  # (num_envs, num_agents, 2)
        else:
            # Empty context if disabled
            static_context = torch.zeros(
                num_envs, num_agents, 0, device=torch_device, dtype=torch.float32
            )

        # Extract initial lidar readings if enabled
        # Observation structure: pos(2) + vel(2) + food(4) + lidar(12) = 20
        # Lidar readings are the last 12 dimensions
        if use_lidar_context:
            # obs is a list of tensors: [agent0_obs, agent1_obs, ...]
            # Each agent_obs has shape: (num_envs, obs_dim)
            # Extract last lidar_dim dimensions from each agent's observation
            lidar_list = [agent_obs[:, -lidar_dim:] for agent_obs in obs]
            # Stack to (num_agents, num_envs, lidar_dim) then transpose to (num_envs, num_agents, lidar_dim)
            initial_lidar = torch.stack(lidar_list, dim=0).transpose(0, 1)
        else:
            initial_lidar = None

        # Print context info for first episode
        if episode == 0:
            cap_mode = "FIXED" if use_fixed_capabilities else "randomized per episode"
            if use_capability_context and static_context.shape[-1] >= 2:
                print(f"\nCapability vectors for all agents ({cap_mode}):")
                for agent_idx in range(num_agents):
                    print(
                        f"  Agent {agent_idx}: speed={static_context[0, agent_idx, 0].item():.3f}, "
                        f"lidar_range={static_context[0, agent_idx, 1].item():.3f}"
                    )
            else:
                print(f"\nCapability vectors disabled (using empty context)")

            print(f"Context shape: {static_context.shape}")
            if use_lidar_context:
                print(f"Using initial lidar readings as additional context")
                if initial_lidar is not None:
                    print(f"Lidar context shape: {initial_lidar.shape}")
                else:
                    print(f"Lidar context shape: None (lidar_dim={lidar_dim})")

        # Create task embedding (dummy for now, could be learned or scenario-specific)
        static_task = torch.ones(
            num_envs, num_agents, task_embed_dim, device=torch_device
        )

        # ====================================================================
        # Flatten for Batching
        # ====================================================================
        batch_size = num_envs * num_agents
        context_flat = static_context.reshape(batch_size, -1)  # (256, context_dim)
        task_flat = static_task.reshape(batch_size, -1)  # (256, task_embed_dim)

        if use_lidar_context:
            lidar_flat = initial_lidar.reshape(batch_size, -1)  # (256, lidar_dim)
        else:
            lidar_flat = None

        # ====================================================================
        # WRAPPER: PyTorch -> JAX (for context and task)
        # ====================================================================
        # Convert to numpy first, handling device properly
        context_np = (
            context_flat.detach().cpu().numpy()
            if context_flat.requires_grad
            else context_flat.cpu().numpy()
        )
        task_np = (
            task_flat.detach().cpu().numpy()
            if task_flat.requires_grad
            else task_flat.cpu().numpy()
        )

        jax_context = jnp.asarray(context_np)
        jax_task = jnp.asarray(task_np)

        if use_lidar_context:
            lidar_np = (
                lidar_flat.detach().cpu().numpy()
                if lidar_flat.requires_grad
                else lidar_flat.cpu().numpy()
            )
            # 1. Handle Infinity (Simulators often return inf for "no hit")
            lidar_np = np.nan_to_num(lidar_np, posinf=1.0, neginf=0.0)
            # 2. Hard Clip (Ensure values stay in -1 to 1 range for the Transformer)
            lidar_np = np.clip(lidar_np, -1.0, 1.0)
            jax_lidar = jnp.asarray(lidar_np)
        else:
            jax_lidar = None

        # Move to JAX device if using CUDA
        if use_cuda:
            jax_context = jax.device_put(jax_context, jax_device)
            jax_task = jax.device_put(jax_task, jax_device)
            if use_lidar_context:
                jax_lidar = jax.device_put(jax_lidar, jax_device)

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
        }

        # Create JAX RNG key for sampling (use different name to avoid shadowing NumPy rng)
        jax_rng = jax.random.PRNGKey(episode)

        for step in range(rollout_steps):  # Rollout steps per episode
            # ================================================================
            # WRAPPER: PyTorch -> JAX (for observations)
            # ================================================================
            # VMAS returns list of observations: [agent0_obs, agent1_obs, ...]
            # Each agent_obs has shape: (num_envs, obs_dim)
            # Stack them: (num_agents, num_envs, obs_dim)
            obs_stacked = torch.stack(obs, dim=0)  # (num_agents, num_envs, obs_dim)
            # Transpose to: (num_envs, num_agents, obs_dim)
            obs_transposed = obs_stacked.transpose(
                0, 1
            )  # (num_envs, num_agents, obs_dim)

            # For independent policies, we keep the agent dimension
            obs_np = (
                obs_transposed.detach().cpu().numpy()
                if obs_transposed.requires_grad
                else obs_transposed.cpu().numpy()
            )

            jax_obs = jnp.asarray(obs_np)

            # Move to JAX device if using CUDA
            if use_cuda:
                jax_obs = jax.device_put(jax_obs, jax_device)

            # Create global state for centralized critic (reshape to num_envs, num_agents * obs_dim)
            global_state_flat = obs_transposed.reshape(
                num_envs, -1
            )  # (num_envs, num_agents * obs_dim)
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
            values = _get_value(critic_state.params, jax_global_state)

            # ================================================================
            # Run Policy (JAX) - Get actions and log probs
            # ================================================================
            jax_rng, action_rng = jax.random.split(jax_rng)

            jax_actions, log_probs = _get_actions_and_log_probs(
                policy_state.params, jax_obs, action_rng
            )

            # ================================================================
            # WRAPPER: JAX -> PyTorch (for actions)
            # ================================================================
            actions_np = np.asarray(jax_actions)

            # Convert to list of tensors for VMAS
            # VMAS expects: [agent0_actions, agent1_actions, ...]
            # Each with shape: (num_envs, action_dim)
            torch_actions = []
            for agent_idx in range(num_agents):
                agent_actions = torch.tensor(
                    actions_np[:, agent_idx, :],
                    device=torch_device,
                    dtype=torch.float32,
                )
                torch_actions.append(agent_actions)

            # ================================================================
            # Environment Step (PyTorch)
            # ================================================================
            next_obs, rewards, dones, info = env.step(torch_actions)

            # Store trajectory data
            trajectory_data["obs"].append(jax_obs)
            trajectory_data["global_states"].append(jax_global_state)
            trajectory_data["actions"].append(jax_actions)
            trajectory_data["log_probs"].append(log_probs)
            trajectory_data["values"].append(values)

            # Convert rewards properly
            # VMAS returns list of rewards: [agent0_reward, agent1_reward, ...]
            # Each has shape: (num_envs,)
            rewards_stacked = torch.stack(rewards, dim=0)  # (num_agents, num_envs)
            rewards_transposed = rewards_stacked.transpose(
                0, 1
            )  # (num_envs, num_agents)
            rewards_np = (
                rewards_transposed.detach().cpu().numpy()
                if rewards_transposed.requires_grad
                else rewards_transposed.cpu().numpy()
            )
            trajectory_data["rewards"].append(np.asarray(rewards_np))

            # Calculate mean reward for logging
            episode_rewards.append(float(rewards_np.mean()))

            # Update observation
            obs = next_obs

            if dones.all():
                break

        # ====================================================================
        # Compute Advantages using GAE (Generalized Advantage Estimation)
        # ====================================================================
        gamma = config["training"]["gamma"]
        gae_lambda = config["training"]["gae_lambda"]
        reward_scale = config["training"].get("reward_scale", 1.0)

        rewards_array = np.array(
            trajectory_data["rewards"]
        )  # (num_steps, num_envs, num_agents)

        # Scale rewards to improve gradient magnitude (important for sparse rewards)
        rewards_array = rewards_array * reward_scale
        values_array = np.array(
            [np.asarray(v) for v in trajectory_data["values"]]
        )  # (num_steps, num_envs)
        num_steps = len(rewards_array)

        # Rewards are already (num_steps, num_envs, num_agents)
        rewards_reshaped = rewards_array
        # Average across agents for global reward signal (or sum, depending on task)
        rewards_per_env = rewards_reshaped.mean(axis=2)  # (num_steps, num_envs)

        # Compute GAE
        advantages = np.zeros_like(rewards_per_env)
        returns = np.zeros_like(rewards_per_env)
        gae = 0

        # Bootstrap value for last step (assume 0 for terminal states)
        next_value = np.zeros(num_envs)

        # Compute GAE backwards through time
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0  # Assume non-terminal
                next_val = next_value
            else:
                next_non_terminal = 1.0
                next_val = values_array[t + 1]

            delta = (
                rewards_per_env[t]
                + gamma * next_non_terminal * next_val
                - values_array[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values_array[t]

        # Expand advantages and returns back to per-agent
        # Each env's advantage is shared across its agents
        advantages_expanded = np.repeat(
            advantages[:, :, np.newaxis], num_agents, axis=2
        )  # (num_steps, num_envs, num_agents)
        returns_expanded = np.repeat(returns[:, :, np.newaxis], num_agents, axis=2)

        # Flatten to (num_steps * num_envs, num_agents)
        advantages_flat = advantages_expanded.reshape(-1, num_agents)
        returns_flat = returns_expanded.reshape(-1, num_agents)

        # Store raw advantages for debugging
        raw_advantages_mean = advantages_flat.mean()
        raw_advantages_std = advantages_flat.std()

        # CRITICAL: Normalize advantages to stabilize policy updates
        advantages_mean = advantages_flat.mean()
        advantages_std = advantages_flat.std()
        advantages_flat = (advantages_flat - advantages_mean) / (advantages_std + 1e-8)

        # Debug output for first episode and periodically
        if episode == 0 or episode == 1 or episode % 100 == 0:
            print(f"\n[DEBUG Episode {episode}]")
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
                f"  Final Advantages mean: {advantages_flat.mean():.6f}, std: {advantages_flat.std():.6f}"
            )

        advantages_batch = jnp.array(advantages_flat)
        returns_batch = jnp.array(returns_flat)

        # ====================================================================
        # Training Step (JAX)
        # ====================================================================
        # Stack trajectory data
        # obs_batch: (num_steps * num_envs, num_agents, obs_dim)
        obs_batch = jnp.concatenate(trajectory_data["obs"], axis=0)

        # global_state_batch: (num_steps * num_envs, num_agents * obs_dim)
        global_state_batch = jnp.concatenate(trajectory_data["global_states"], axis=0)

        # Optionally clip global state batch before critic in training
        if clip_global_state:
            global_state_batch = jnp.clip(
                global_state_batch, clip_global_state_min, clip_global_state_max
            )

        # actions_batch: (num_steps * num_envs, num_agents, action_dim)
        actions_batch = jnp.concatenate(trajectory_data["actions"], axis=0)

        # old_log_probs_batch: (num_steps * num_envs, num_agents)
        old_log_probs_batch = jnp.concatenate(trajectory_data["log_probs"], axis=0)

        # Get PPO hyperparameters from config
        clip_epsilon = config["training"]["ppo_clip_epsilon"]
        entropy_coef = config["training"]["entropy_coef"]
        value_loss_coef = config["training"]["value_loss_coef"]
        ppo_epochs = config["training"].get("ppo_epochs", 4)
        num_minibatches = config["training"].get("num_minibatches", 4)

        # Track parameter changes for debugging
        if episode % log_interval == 0 or episode == 1:
            # Get initial policy parameters for comparison
            initial_policy_params = jax.tree_util.tree_map(
                lambda x: x.copy(), policy_state.params
            )

        # Perform multiple PPO training epochs over the same data with minibatches
        batch_size = obs_batch.shape[0]
        minibatch_size = batch_size // num_minibatches

        for ppo_epoch in range(ppo_epochs):
            # Shuffle data for each epoch
            perm = jax.random.permutation(
                jax.random.PRNGKey(episode * ppo_epochs + ppo_epoch), batch_size
            )

            # Train on minibatches
            for mb_idx in range(num_minibatches):
                mb_start = mb_idx * minibatch_size
                mb_end = mb_start + minibatch_size
                mb_indices = perm[mb_start:mb_end]

                # Extract minibatch
                mb_obs = obs_batch[mb_indices]
                mb_global_state = global_state_batch[mb_indices]
                mb_actions = actions_batch[mb_indices]
                mb_old_log_probs = old_log_probs_batch[mb_indices]
                mb_advantages = advantages_batch[mb_indices]
                mb_returns = returns_batch[mb_indices]

                policy_state, critic_state, loss_info = _train_step(
                    policy_state,
                    critic_state,
                    mb_obs,
                    mb_global_state,
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                    clip_epsilon,
                    entropy_coef,
                    value_loss_coef,
                    num_agents,
                )

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
        # System Neural Diversity (SND) Calculation
        # ====================================================================
        # Calculate SND periodically to track behavioral diversity
        snd_interval = config["training"].get("snd_interval", log_interval)
        if episode % snd_interval == 0 or episode == 1:
            # Use observations from current rollout for SND calculation
            # obs_batch has shape (num_steps * num_envs, num_agents, obs_dim)
            # Flatten to (num_steps * num_envs * num_agents, obs_dim) for SND
            obs_batch_flat = obs_batch.reshape(-1, obs_batch.shape[-1])
            # Calculate actual number of timesteps in the batch
            num_timesteps = obs_batch.shape[0]
            # Use actual training policies (no random sampling)
            snd_value = calculate_snd_independent(
                policy_params=policy_state.params,
                obs_batch=obs_batch_flat,
                policy_model=policy_def,
                num_agents=num_agents,
                num_envs=num_timesteps,  # Pass num_timesteps as num_envs for SND calculation
            )
            loss_info["snd_total"] = snd_value

        # ====================================================================
        # Logging
        # ====================================================================
        if episode % log_interval == 0 or episode == 1:
            avg_reward = np.mean(episode_rewards)
            min_reward = np.min(episode_rewards)
            max_reward = np.max(episode_rewards)
            std_reward = np.std(episode_rewards)

            print(
                f"Episode {episode}/{num_episodes}: Avg Reward = {avg_reward:.3f}, "
                f"Policy Loss = {loss_info['policy_loss']:.4f}, "
                f"Value Loss = {loss_info['value_loss']:.4f}, "
                f"Entropy = {loss_info['entropy']:.4f}, "
                f"Mean Std = {loss_info['mean_std']:.4f} "
                f"[Min: {loss_info['min_std']:.4f}, Max: {loss_info['max_std']:.4f}], "
                f"Mean Ratio = {loss_info['mean_ratio']:.4f}"
            )
            print(
                f"  Grad Norms - Policy: {loss_info['policy_grad_norm']:.6f}, "
                f"Critic: {loss_info['critic_grad_norm']:.6f}"
            )
            print(
                f"  Param Changes - Avg: {loss_info.get('avg_param_change', 0):.8f}, "
                f"Max: {loss_info.get('max_param_change', 0):.8f}"
            )
            print(
                f"  Advantages - Mean: {loss_info['advantages_mean']:.6f}, "
                f"Std: {loss_info['advantages_std']:.6f}, "
                f"Max: {loss_info['advantages_max']:.6f}"
            )
            if "snd_total" in loss_info:
                print(f"  SND: {loss_info['snd_total']:.6f}")

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
                    "debug/mean_std": float(loss_info["mean_std"]),
                    "debug/mean_ratio": float(loss_info["mean_ratio"]),
                }

                # Add SND metrics if available
                if "snd_total" in loss_info:
                    log_dict["diversity/snd_total"] = float(loss_info["snd_total"])

                wandb.log(log_dict, step=episode)

        # ====================================================================
        # Evaluation Rollout
        # ====================================================================
        if episode % eval_interval == 0 and episode > 0:
            print(f"\n{'='*80}")
            print(f"Running evaluation at episode {episode}...")
            print(f"{'='*80}")

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
                num_eval_episodes=num_eval_episodes,
                max_eval_steps=max_eval_steps,
                randomize_capabilities=False,
                fixed_capabilities=config["env"].get("fixed_capabilities", {}),
                verbose=False,
                calculate_snd_metric=True,
            )

            print_training_eval_results(episode, eval_metrics)

            # Log evaluation metrics to wandb
            if use_wandb:
                log_dict = {
                    "eval/avg_reward": eval_metrics["avg_reward"],
                    "eval/avg_episode_length": eval_metrics["avg_episode_length"],
                    "eval/completion_rate": eval_metrics["completion_rate"],
                    "episode": episode,
                }

                # Add SND if available
                if "snd" in eval_metrics and eval_metrics["snd"] is not None:
                    log_dict["eval/snd"] = eval_metrics["snd"]

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

                wandb.log(log_dict)

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
        np.savez(final_checkpoint_path, **checkpoint_data)
        print(f"Final checkpoint saved to: {final_checkpoint_path}")

    # Generate visualization GIF
    if args.render_final_policy:
        print("\nGenerating visualization GIF of final policy...")
        gif_path = generate_policy_gif(
            env=env,
            policy_state=policy_state,
            hn_state=None,  # No hypernetwork in independent training
            adapters_dict=None,  # No adapters in independent training
            config=config,
            checkpoint_dir=checkpoint_dir,
            n_steps=args.gif_steps,
            use_hypernetwork=False,  # Always False for independent training
            num_agents=num_agents,
            obs_dim=obs_dim,
            context_dim=context_dim,
            task_embed_dim=task_embed_dim,
            use_cuda=use_cuda,
            jax_device=jax_device,
            torch_device=torch_device,
            get_static_adapters_fn=None,  # Not used in independent training
            get_actions_fn=_get_actions,
        )
        if gif_path:
            print(f"GIF saved to: {gif_path}")

    # Close wandb run
    if use_wandb:
        wandb.finish()
        print("Weights & Biases run finished")


if __name__ == "__main__":
    main()
