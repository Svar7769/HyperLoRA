"""
Quick test script to verify hypernetwork learns capability-to-behavior mapping.
Uses 4 agents with fixed, distinct capability vectors during training.
This allows the hypernetwork to overfit to specific capabilities, testing if it can
properly connect capabilities to desired outputs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
import optax
import distrax
from flax.training import train_state
import yaml
from pathlib import Path
import os
from datetime import datetime

from env_setup import make_vmas_env
from lora_policy import LoRAPolicy
from hypernetwork import Hypernetwork
from critic import CentralizedCritic
from snd import calculate_snd_statistics
from render_gif import generate_policy_gif

# Configure JAX
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Fixed capability vectors for 4 very different agents
FIXED_CAPABILITIES = {
    "agent_0": {"speed": 0.5, "lidar_range": 0.3},  # Slow, short range
    "agent_1": {"speed": 1.5, "lidar_range": 0.3},  # Fast, short range
    "agent_2": {"speed": 0.5, "lidar_range": 0.7},  # Slow, long range
    "agent_3": {"speed": 1.5, "lidar_range": 0.7},  # Fast, long range
}

print("=" * 80)
print("FIXED CAPABILITY TEST")
print("=" * 80)
print("Testing if hypernetwork can learn capability-to-behavior mapping")
print("Using 4 agents with fixed, distinct capabilities:")
for agent_name, caps in FIXED_CAPABILITIES.items():
    print(
        f"  {agent_name}: speed={caps['speed']:.1f}, lidar_range={caps['lidar_range']:.1f}"
    )
print(
    "\nFood positions: FIXED in 4 corners (bottom-left, bottom-right, top-left, top-right)"
)
print("=" * 80)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_train_state(model, params, lr, max_grad_norm=1.0):
    """Create a training state with optimizer and gradient clipping."""
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=lr),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def main():
    # Load base config
    config_path = Path("config.yaml")
    config = load_config(config_path)

    # Global variables for models (needed for JIT compilation)
    global shared_policy, hypernetwork, critic

    # Override settings for quick test
    config["env"]["num_agents"] = 4  # Fixed 4 agents
    config["env"]["num_envs"] = 128  # Fewer environments for faster testing
    config["training"]["num_episodes"] = 500  # Shorter training
    config["training"]["rollout_steps"] = 20  # Shorter rollouts
    config["model"]["use_hypernetwork"] = True  # Must use hypernetwork

    # Lidar context configuration
    use_lidar_context = config["model"].get("use_lidar_context", False)

    # Training hyperparameters
    seed = config["training"]["seed"]
    num_agents = 4
    num_envs = config["env"]["num_envs"]
    num_episodes = config["training"]["num_episodes"]
    rollout_steps = config["training"]["rollout_steps"]
    log_interval = 100  # Log more frequently

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device setup
    use_cuda = config["device"].get("use_cuda", False)
    if use_cuda:
        try:
            gpu_devices = jax.devices("gpu")
            jax_device = gpu_devices[0] if gpu_devices else jax.devices("cpu")[0]
            torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except:
            jax_device = jax.devices("cpu")[0]
            torch_device = "cpu"
    else:
        jax_device = jax.devices("cpu")[0]
        torch_device = "cpu"

    print(f"JAX device: {jax_device}")
    print(f"PyTorch device: {torch_device}")

    # Create environment with fixed capabilities
    agent_speeds = [FIXED_CAPABILITIES[f"agent_{i}"]["speed"] for i in range(4)]
    agent_lidar_ranges = [
        FIXED_CAPABILITIES[f"agent_{i}"]["lidar_range"] for i in range(4)
    ]

    agent_capabilities = {"speed": agent_speeds, "lidar_range": agent_lidar_ranges}

    env = make_vmas_env(
        config["env"]["scenario_name"],
        num_agents,
        num_envs,
        device=torch_device,
        continuous_actions=config["env"]["continuous_actions"],
        agent_capabilities=agent_capabilities,
        fixed_food_positions=False,  # Fix foods in corners for consistent testing
    )

    # Get dimensions
    temp_obs = env.reset()
    obs_dim = temp_obs[0].shape[-1]
    action_dim = env.get_agent_action_size(env.agents[0])

    # Calculate lidar dimension from observation
    # Observation structure: pos(2) + vel(2) + food(4) + lidar(X)
    lidar_dim = 0
    if use_lidar_context:
        lidar_dim = obs_dim - 4  # normalized_pos(2) + vel(2) = 4, rest is lidar
        print(f"Calculated lidar_dim from observation: {lidar_dim}")

    print(
        f"\nEnvironment dimensions: obs_dim={obs_dim}, action_dim={action_dim}, lidar_dim={lidar_dim}"
    )

    # Model configuration
    policy_hidden_dims = config["model"]["policy_hidden_dims"]
    lora_rank = config["model"]["lora_rank"]
    task_embed_dim = config["model"]["task_embed_dim"]
    context_dim = 2  # [speed, lidar_range]
    lora_mode = config["model"].get("lora_mode", "final_only")
    lora_scaling_factor = config["model"].get("lora_scaling_factor", 1.0)

    policy_dims = {
        "obs_dim": obs_dim,
        "hidden_dims": policy_hidden_dims,
        "action_dim": action_dim,
        "lora_rank": lora_rank,
    }

    # Initialize models
    shared_policy = LoRAPolicy(
        hidden_dims=tuple(policy_hidden_dims),
        action_dim=action_dim,
        lora_mode=lora_mode,
    )

    hypernetwork = Hypernetwork(
        policy_dims=policy_dims,
        context_dim=context_dim,
        task_embed_dim=task_embed_dim,
        lidar_dim=lidar_dim,
        transformer_dim=config["model"]["transformer_dim"],
        transformer_heads=config["model"]["transformer_heads"],
        transformer_layers=config["model"]["transformer_layers"],
        lora_mode=lora_mode,
        scaling_factor=lora_scaling_factor,
    )

    critic = CentralizedCritic(
        hidden_dim=config["model"]["critic_hidden_dim"],
        num_layers=config["model"]["critic_num_layers"],
    )

    # Define JIT-compiled functions that use the models
    @jax.jit
    def _get_static_adapters(hn_params, task_batch, context_batch, lidar_batch=None):
        """Generate LoRA adapters for all agents in the batch."""
        return hypernetwork.apply(
            {"params": hn_params}, task_batch, context_batch, lidar_batch
        )

    @jax.jit
    def _get_actions_and_log_probs(policy_params, obs_batch, adapters_dict, rng_key):
        """Get actions and log probabilities from the policy using LoRA adapters."""
        action, log_prob, mean, std = shared_policy.apply(
            {"params": policy_params},
            obs_batch,
            adapters_dict,
            rng_key,
            method=shared_policy.get_action_and_log_prob,
        )
        return action, log_prob

    @jax.jit
    def _get_value(critic_params, global_state):
        """Get value estimate from centralized critic."""
        return critic.apply({"params": critic_params}, global_state)

    @jax.jit
    def _train_step(
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
        clip_epsilon,
        entropy_coef,
        value_loss_coef,
        num_agents,
    ):
        """Perform one MAPPO training step."""

        def loss_fn(policy_params, hn_params, critic_params):
            # Generate adapters
            adapters_dict = hypernetwork.apply(
                {"params": hn_params}, task_batch, context_batch, lidar_batch
            )

            # Get mean and log_std for current policy
            mean, log_std = shared_policy.apply(
                {"params": policy_params}, obs_batch, adapters_dict
            )
            std = jnp.exp(log_std)
            std = jnp.maximum(std, 0.3)

            # Create tanh-transformed distribution
            base_dist = distrax.Normal(mean, std)
            tanh_bijector = distrax.Tanh()
            dist = distrax.Transformed(base_dist, tanh_bijector)

            # Clip actions to avoid boundary values
            action_epsilon = 1e-6
            actions_clipped = jnp.clip(
                actions_batch, -1.0 + action_epsilon, 1.0 - action_epsilon
            )

            # Compute log probabilities
            new_log_probs = dist.log_prob(actions_clipped).sum(axis=-1)

            # Clip log prob difference
            log_prob_diff = new_log_probs - old_log_probs_batch
            log_prob_diff = jnp.clip(log_prob_diff, -3.0, 3.0)
            ratio = jnp.exp(log_prob_diff)

            # Entropy from base distribution
            entropy = base_dist.entropy().sum(axis=-1).mean()

            # PPO clipped objective
            clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            surr1 = ratio * advantages_batch
            surr2 = clipped_ratio * advantages_batch
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            # Small regularization
            policy_regularization = 0.001 * jnp.mean(jnp.square(mean))
            policy_loss = policy_loss + policy_regularization

            # Std regularization
            min_std = 0.5
            std_penalty = jnp.mean(jnp.maximum(0.0, min_std - std) ** 2)
            std_regularization_coef = 0.1
            policy_loss = policy_loss + std_regularization_coef * std_penalty

            # Centralized value function loss
            values = critic.apply({"params": critic_params}, global_state_batch)
            total_length = returns_batch.shape[0]
            values_expanded = jnp.repeat(
                values, num_agents, total_repeat_length=total_length
            )

            # Normalize returns
            returns_normalized = (returns_batch - returns_batch.mean()) / (
                returns_batch.std() + 1e-8
            )
            value_loss = jnp.mean((values_expanded - returns_normalized) ** 2)

            # Adapter regularization
            adapter_norm = 0.0
            adapter_count = 0
            for v in adapters_dict.values():
                adapter_norm += jnp.mean(jnp.square(v))
                adapter_count += 1
            adapter_mean_norm = (
                jnp.sqrt(adapter_norm / max(adapter_count, 1))
                if adapter_count > 0
                else 0.0
            )

            adapter_regularization_coef = 0.001
            adapter_loss = adapter_regularization_coef * adapter_norm

            # Total loss
            total_loss = (
                policy_loss
                + value_loss_coef * value_loss
                - entropy_coef * entropy
                + adapter_loss
            )

            return total_loss, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": jnp.mean((new_log_probs - old_log_probs_batch) ** 2),
                "mean_std": jnp.mean(std),
                "min_std": jnp.min(std),
                "max_std": jnp.max(std),
                "mean_ratio": jnp.mean(ratio),
                "adapter_norm": adapter_mean_norm,
            }

        # Compute gradients
        (loss, info), grads = jax.value_and_grad(
            lambda p, h, c: loss_fn(p, h, c), argnums=(0, 1, 2), has_aux=True
        )(policy_state.params, hn_state.params, critic_state.params)

        policy_grads, hn_grads, critic_grads = grads

        # Update networks
        policy_state = policy_state.apply_gradients(grads=policy_grads)
        hn_state = hn_state.apply_gradients(grads=hn_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        return policy_state, hn_state, critic_state, info

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    rng, policy_rng, hn_rng, critic_rng = jax.random.split(rng, 4)

    dummy_obs = jnp.ones((1, obs_dim))
    dummy_global_state = jnp.ones((1, obs_dim * num_agents))
    dummy_adapters = {}
    input_dim = obs_dim
    for i, output_dim in enumerate(policy_hidden_dims):
        layer_idx = i + 1
        dummy_adapters[f"A{layer_idx}"] = jnp.ones((1, lora_rank, input_dim))
        dummy_adapters[f"B{layer_idx}"] = jnp.ones((1, output_dim, lora_rank))
        input_dim = output_dim
    final_idx = len(policy_hidden_dims) + 1
    dummy_adapters[f"A{final_idx}"] = jnp.ones((1, lora_rank, policy_hidden_dims[-1]))
    dummy_adapters[f"B{final_idx}"] = jnp.ones((1, action_dim, lora_rank))

    dummy_task = jnp.ones((1, task_embed_dim))
    dummy_context = jnp.ones((1, context_dim))
    dummy_lidar = jnp.ones((1, lidar_dim)) if lidar_dim > 0 else None

    policy_params = shared_policy.init(policy_rng, dummy_obs, dummy_adapters)["params"]
    hn_params = hypernetwork.init(hn_rng, dummy_task, dummy_context, dummy_lidar)[
        "params"
    ]
    critic_params = critic.init(critic_rng, dummy_global_state)["params"]

    # Create training states
    policy_state = create_train_state(
        shared_policy, policy_params, config["optimizer"]["learning_rate"]
    )
    hn_state = create_train_state(
        hypernetwork,
        hn_params,
        config["optimizer"]["hn_learning_rate"],
        max_grad_norm=0.1,
    )
    critic_state = create_train_state(
        critic, critic_params, config["optimizer"]["critic_learning_rate"]
    )

    if use_cuda:
        with jax.default_device(jax_device):
            policy_state = jax.device_put(policy_state, jax_device)
            hn_state = jax.device_put(hn_state, jax_device)
            critic_state = jax.device_put(critic_state, jax_device)

    print("\n" + "=" * 80)
    print("STARTING TRAINING WITH FIXED CAPABILITIES")
    print("=" * 80)

    # Create fixed capability context vectors (never change during training)
    capability_vectors = torch.tensor(
        [[agent_speeds[i], agent_lidar_ranges[i]] for i in range(num_agents)],
        device=torch_device,
        dtype=torch.float32,
    )

    static_context = capability_vectors.unsqueeze(0).expand(num_envs, -1, -1)
    static_task = torch.ones(num_envs, num_agents, task_embed_dim, device=torch_device)

    # Convert to JAX
    batch_size = num_envs * num_agents
    context_flat = static_context.reshape(batch_size, -1)
    task_flat = static_task.reshape(batch_size, -1)

    jax_context = jnp.asarray(context_flat.cpu().numpy())
    jax_task = jnp.asarray(task_flat.cpu().numpy())

    if use_cuda:
        jax_context = jax.device_put(jax_context, jax_device)
        jax_task = jax.device_put(jax_task, jax_device)

    # Extract initial lidar readings if enabled (FIXED - only done once at start)
    # NOTE: If food positions are fixed, lidar readings will be the same across episodes
    jax_lidar = None
    if use_lidar_context:
        # Get initial observation after reset
        initial_obs = env.reset()
        # Extract last lidar_dim dimensions from each agent's observation
        lidar_list = [agent_obs[:, -lidar_dim:] for agent_obs in initial_obs]
        # Stack to (num_agents, num_envs, lidar_dim) then transpose to (num_envs, num_agents, lidar_dim)
        initial_lidar = torch.stack(lidar_list, dim=0).transpose(0, 1)
        lidar_flat = initial_lidar.reshape(batch_size, -1)
        jax_lidar = jnp.asarray(lidar_flat.cpu().numpy())
        if use_cuda:
            jax_lidar = jax.device_put(jax_lidar, jax_device)
        print(f"\nInitial lidar context extracted: shape={jax_lidar.shape}")
        fixed_food = (
            env.scenario.fixed_food_positions
            if hasattr(env.scenario, "fixed_food_positions")
            else False
        )
        if fixed_food:
            print(
                f"Food positions are FIXED - lidar context will remain constant throughout training"
            )
        else:
            print(
                f"Food positions are RANDOM - lidar context is from initial reset only"
            )

    # Training loop
    for episode in range(num_episodes):
        # Reset environment (capabilities remain fixed)
        obs = env.reset()

        # Generate adapters once per episode (they're based on fixed capabilities + fixed lidar)
        adapters_dict = _get_static_adapters(
            hn_state.params, jax_task, jax_context, jax_lidar
        )

        # Rollout
        trajectory_data = {
            "obs": [],
            "global_states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
        }

        episode_rewards = []
        jax_rng = jax.random.PRNGKey(episode)

        for step in range(rollout_steps):
            # Convert observations
            obs_stacked = torch.stack(obs, dim=0)
            obs_transposed = obs_stacked.transpose(0, 1)
            obs_flat = obs_transposed.reshape(batch_size, -1)
            jax_obs = jnp.asarray(obs_flat.cpu().numpy())

            # Global state
            global_state_flat = obs_transposed.reshape(num_envs, -1)
            jax_global_state = jnp.asarray(global_state_flat.cpu().numpy())

            if use_cuda:
                jax_obs = jax.device_put(jax_obs, jax_device)
                jax_global_state = jax.device_put(jax_global_state, jax_device)

            # Get values
            values = _get_value(critic_state.params, jax_global_state)

            # Get actions
            jax_rng, action_rng = jax.random.split(jax_rng)
            jax_actions_flat, log_probs_flat = _get_actions_and_log_probs(
                policy_state.params, jax_obs, adapters_dict, action_rng
            )

            # Convert actions to PyTorch
            actions_flat = np.asarray(jax_actions_flat)
            actions_reshaped = actions_flat.reshape(num_envs, num_agents, -1)
            torch_actions = [
                torch.tensor(
                    actions_reshaped[:, i, :], device=torch_device, dtype=torch.float32
                )
                for i in range(num_agents)
            ]

            # Step environment
            next_obs, rewards, dones, info = env.step(torch_actions)

            # Store trajectory
            trajectory_data["obs"].append(jax_obs)
            trajectory_data["global_states"].append(jax_global_state)
            trajectory_data["actions"].append(jax_actions_flat)
            trajectory_data["log_probs"].append(log_probs_flat)
            trajectory_data["values"].append(values)

            rewards_stacked = torch.stack(rewards, dim=0)
            rewards_transposed = rewards_stacked.transpose(0, 1)
            rewards_np = rewards_transposed.cpu().numpy()
            trajectory_data["rewards"].append(rewards_np.flatten())

            episode_rewards.append(float(rewards_np.mean()))
            obs = next_obs

            if dones.all():
                break

        # Compute advantages using GAE
        gamma = config["training"]["gamma"]
        gae_lambda = config["training"]["gae_lambda"]
        reward_scale = config["training"].get("reward_scale", 1.0)

        rewards_array = np.array(trajectory_data["rewards"]) * reward_scale
        values_array = np.array([np.asarray(v) for v in trajectory_data["values"]])
        num_steps = len(rewards_array)

        rewards_reshaped = rewards_array.reshape(num_steps, num_envs, num_agents)
        rewards_per_env = rewards_reshaped.mean(axis=2)

        advantages = np.zeros_like(rewards_per_env)
        returns = np.zeros_like(rewards_per_env)
        gae = 0
        next_value = np.zeros(num_envs)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0
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

        advantages_expanded = np.repeat(
            advantages[:, :, np.newaxis], num_agents, axis=2
        )
        returns_expanded = np.repeat(returns[:, :, np.newaxis], num_agents, axis=2)

        advantages_flat = advantages_expanded.reshape(-1)
        returns_flat = returns_expanded.reshape(-1)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        advantages_batch = jnp.array(advantages_flat)
        returns_batch = jnp.array(returns_flat)

        # Stack trajectory data
        obs_batch = jnp.concatenate(trajectory_data["obs"], axis=0)
        global_state_batch = jnp.concatenate(trajectory_data["global_states"], axis=0)
        actions_batch = jnp.concatenate(trajectory_data["actions"], axis=0)
        old_log_probs_batch = jnp.concatenate(trajectory_data["log_probs"], axis=0)

        # Tile task and context
        num_trajectory_steps = obs_batch.shape[0]
        jax_task_batch = jnp.tile(jax_task, (num_trajectory_steps // batch_size, 1))
        jax_context_batch = jnp.tile(
            jax_context, (num_trajectory_steps // batch_size, 1)
        )
        if use_lidar_context:
            jax_lidar_batch = jnp.tile(
                jax_lidar, (num_trajectory_steps // batch_size, 1)
            )
        else:
            jax_lidar_batch = None

        # Training step
        clip_epsilon = config["training"]["ppo_clip_epsilon"]
        entropy_coef = config["training"]["entropy_coef"]
        value_loss_coef = config["training"]["value_loss_coef"]
        ppo_epochs = config["training"].get("ppo_epochs", 4)

        for ppo_epoch in range(ppo_epochs):
            policy_state, hn_state, critic_state, loss_info = _train_step(
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
                clip_epsilon,
                entropy_coef,
                value_loss_coef,
                num_agents,
            )

        # Logging
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards)

            # Calculate SND
            snd_value = 0.0
            try:
                obs_sample = obs_batch[:batch_size]
                snd_rng = jax.random.PRNGKey(episode * 1000)
                # Note: SND calculation doesn't support lidar_context parameter yet
                # It uses the observations directly which already contain lidar readings
                snd_stats = calculate_snd_statistics(
                    policy_state.params,
                    hn_state.params,
                    obs_sample,
                    jax_task,
                    jax_context,
                    shared_policy,
                    hypernetwork,
                    num_agents,
                    num_envs,
                    num_samples=10,
                    rng_key=snd_rng,
                )
                snd_value = snd_stats["snd_total"]
            except Exception as e:
                print(f"Warning: SND calculation failed: {e}")

            print(
                f"Episode {episode:4d}/{num_episodes}: "
                f"Reward={avg_reward:7.3f}, "
                f"PolicyLoss={loss_info['policy_loss']:.4f}, "
                f"ValueLoss={loss_info['value_loss']:.4f}, "
                f"Entropy={loss_info['entropy']:.4f}, "
                f"SND={snd_value:.6f}, "
                f"AdapterNorm={loss_info['adapter_norm']:.6f}"
            )

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nIf the hypernetwork is working correctly, you should see:")
    print("  1. Non-zero adapter norms (adapters are being generated)")
    print("  2. Improving rewards over episodes (policy is learning)")
    print(
        "  3. Non-zero SND values (different capabilities produce different behaviors)"
    )
    print("  4. Stable training (no NaN or exploding gradients)")
    print("=" * 80)

    # Generate visualization GIF
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION GIF")
    print("=" * 80)

    gif_path = generate_policy_gif(
        env=env,
        policy_state=policy_state,
        hn_state=hn_state,
        adapters_dict=None,  # Will be generated inside function
        config=config,
        checkpoint_dir=Path("./"),  # Save in current directory
        n_steps=100,
        use_hypernetwork=True,
        num_agents=num_agents,
        obs_dim=obs_dim,
        context_dim=context_dim,
        task_embed_dim=task_embed_dim,
        use_cuda=use_cuda,
        jax_device=jax_device,
        torch_device=torch_device,
        get_static_adapters_fn=_get_static_adapters,
        get_actions_fn=lambda p, o, a: shared_policy.apply({"params": p}, o, a)[
            0
        ],  # Return mean actions for deterministic evaluation
    )

    if gif_path:
        print(f"\n✓ GIF saved to: {gif_path}")
    else:
        print("\n✗ Failed to generate GIF")

    print("=" * 80)


if __name__ == "__main__":
    main()
