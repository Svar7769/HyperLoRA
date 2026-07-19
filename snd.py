"""
System Neural Diversity (SND) Calculation

This module implements the System Neural Diversity metric based on the DiCo paper approach.
SND measures behavioral diversity by computing pairwise Wasserstein distances between
agent policy outputs (mean and std), averaged over sampled observations.

For a population of N agents, SND is computed as:
    SND = E_{o ~ Buffer}[mean(W₂(πᵢ(·|o), πⱼ(·|o)))] for all pairs (i,j)

where W₂ is the 2-Wasserstein distance between Gaussian distributions.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


def wasserstein_distance_gaussian(
    mean1: jnp.ndarray,
    std1: jnp.ndarray,
    mean2: jnp.ndarray,
    std2: jnp.ndarray,
    just_mean: bool = True,
) -> jnp.ndarray:
    """
    Compute 2-Wasserstein distance between two Gaussian distributions.

    For diagonal covariance matrices, the Wasserstein distance simplifies to:
    W₂²(N(μ₁, Σ₁), N(μ₂, Σ₂)) = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²

    where σᵢ are the standard deviations (diagonal elements of Σᵢ).

    Following Bettini et al. (DiCo paper), we use just_mean=True by default,
    which only considers the mean component for behavioral diversity.

    Args:
        mean1: Mean of first distribution (..., action_dim)
        std1: Std dev of first distribution (..., action_dim)
        mean2: Mean of second distribution (..., action_dim)
        std2: Std dev of second distribution (..., action_dim)
        just_mean: If True, only use mean component (Bettini approach)

    Returns:
        Wasserstein distance (...,)
    """
    # Mean component: ||μ₁ - μ₂||₂
    mean_dist = jnp.linalg.norm(mean1 - mean2, ord=2, axis=-1)

    if just_mean:
        return mean_dist

    # Covariance component: ||σ₁ - σ₂||₂ (Frobenius norm for diagonal case)
    cov_dist = jnp.linalg.norm(std1 - std2, ord=2, axis=-1)

    # Combined Wasserstein distance
    return jnp.sqrt(mean_dist**2 + cov_dist**2)


def compute_pairwise_diversity(
    policy_outputs: list,
) -> jnp.ndarray:
    """
    Compute pairwise behavioral diversity between agents using Wasserstein distance.

    Uses JAX vmap for parallel computation of all pairwise distances.

    Args:
        policy_outputs: List of (mean, std) tuples for each agent
                       Each mean/std has shape (batch_size, action_dim)

    Returns:
        Array of pairwise distances with shape (batch_size, n_pairs)
        where n_pairs = n_agents * (n_agents - 1) / 2
    """
    n_agents = len(policy_outputs)

    if n_agents < 2:
        # Return empty array if not enough agents
        batch_size = policy_outputs[0][0].shape[0] if n_agents > 0 else 1
        return jnp.zeros((batch_size, 0))

    # Convert list of tuples to arrays for easier indexing
    means = jnp.stack(
        [output[0] for output in policy_outputs], axis=0
    )  # (n_agents, batch_size, action_dim)
    stds = jnp.stack(
        [output[1] for output in policy_outputs], axis=0
    )  # (n_agents, batch_size, action_dim)

    # Generate all pairs (i, j) where i < j using triu_indices (JIT-compatible)
    i_pairs, j_pairs = jnp.triu_indices(n_agents, k=1)

    # Vectorized function to compute Wasserstein distance for a single pair
    def compute_pair_wasserstein(i, j):
        """Compute Wasserstein distance between agents i and j."""
        return wasserstein_distance_gaussian(
            means[i], stds[i], means[j], stds[j]  # (batch_size, action_dim)
        )

    # Use vmap to compute all pairwise distances in parallel
    # Returns array of shape (n_pairs, batch_size)
    pair_distances = jax.vmap(compute_pair_wasserstein)(i_pairs, j_pairs)

    # Transpose to (batch_size, n_pairs) to match expected output format
    return pair_distances.T


def calculate_snd_statistics(
    policy_params: Dict[str, Any],
    hn_params: Dict[str, Any],
    obs_batch: jnp.ndarray,
    task_batch: jnp.ndarray,
    context_batch: jnp.ndarray,
    policy_model,
    hypernetwork,
    num_agents: int,
    num_envs: int,
    num_samples: int = 10,
    rng_key: jax.random.PRNGKey = None,
    lidar_batch: jnp.ndarray = None,
    mask: jnp.ndarray = None,
) -> Dict[str, float]:
    """
    Calculate Expected SND statistics by averaging over multiple observation samples.

    This implements the DiCo paper's Expected SND: E_{o ~ Buffer}[mean(W₂(πᵢ, πⱼ))]
    SND is computed by sampling observations, generating different agent policies via
    hypernetwork, and measuring behavioral diversity through Wasserstein distances.

    Args:
        policy_params: Policy network parameters
        hn_params: Hypernetwork parameters
        obs_batch: Current observations (num_envs * num_agents, obs_dim)
        task_batch: Task embeddings (num_envs, num_agents, task_dim)
        context_batch: Context vectors (num_envs, num_agents, context_dim)
        policy_model: Policy model instance
        hypernetwork: Hypernetwork model instance
        num_agents: Number of agents per environment
        num_envs: Number of parallel environments
        num_samples: Number of context samples to average over for Expected SND
        rng_key: JAX random key for sampling contexts
        lidar_batch: Optional lidar readings (num_envs, num_agents, lidar_dim)
        mask: Optional attention mask (num_envs, 1, num_agents, num_agents)

    Returns:
        Dictionary with SND statistics:
            - 'snd_total': Expected SND (mean pairwise Wasserstein distance)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Get dimensions from 3D structure
    context_dim = context_batch.shape[2]  # (num_envs, num_agents, context_dim)
    obs_dim = obs_batch.shape[1]

    # Accumulate SND over samples
    total_snd = 0.0

    for i in range(num_samples):
        # Split RNG key for this sample
        rng_key, sample_key = jax.random.split(rng_key)

        # Sample random contexts (capabilities) for diversity estimation
        # This creates different "agent types" to measure diversity across
        # Keep 3D structure: (num_envs, num_agents, context_dim)
        sampled_contexts = jax.random.uniform(
            sample_key,
            shape=(num_envs, num_agents, context_dim),
            minval=0.5,
            maxval=1.5,
        )

        # Use same observations and task embeddings
        sampled_obs = obs_batch
        sampled_tasks = task_batch

        # Compute SND for this sample
        snd_sample = calculate_snd(
            policy_params,
            hn_params,
            sampled_obs,
            sampled_tasks,
            sampled_contexts,
            policy_model,
            hypernetwork,
            num_agents,
            num_envs,
            lidar_batch,
            mask,
        )

        total_snd += snd_sample

    # Average over samples to get Expected SND
    expected_snd = total_snd / num_samples

    return {
        "snd_total": float(expected_snd),
    }


def calculate_snd_independent(
    policy_params: jnp.ndarray,
    obs_batch: jnp.ndarray,
    policy_model,
    num_agents: int,
    num_envs: int,
) -> float:
    """
    Calculate System Neural Diversity (SND) for independent policies (no parameter sharing).

    SND measures diversity by computing pairwise Wasserstein distances between
    agent policy outputs (mean and std). For independent policies, each agent
    has separate parameters, so we directly evaluate each agent's policy.

    Args:
        policy_params: Stacked policy parameters (num_agents, ...)
        obs_batch: Batch of observations (batch_size, obs_dim)
        policy_model: Policy model instance
        num_agents: Number of agents per environment
        num_envs: Number of parallel environments (inferred from batch)

    Returns:
        SND value (float): Mean pairwise Wasserstein distance across all agent pairs
    """
    # Infer actual num_envs from batch size
    actual_batch_size = obs_batch.shape[0]
    actual_num_envs = actual_batch_size // num_agents

    # Reshape observations: (batch_size, obs_dim) -> (num_envs, num_agents, obs_dim)
    obs_reshaped = obs_batch.reshape(actual_num_envs, num_agents, -1)

    # Transpose for vmapping: (num_agents, num_envs, obs_dim)
    obs_transposed = jnp.transpose(obs_reshaped, (1, 0, 2))

    # Check if using GRU policy
    is_gru_policy = hasattr(policy_model, "gru_hidden_dim")

    if is_gru_policy:
        # GRU policy requires special handling
        def get_policy_output(params, obs):
            """Apply GRU policy to get mean and log_std."""
            batch_size = obs.shape[0]

            # Initialize hidden states
            init_hidden = policy_model.initialize_carry(
                batch_size, policy_model.gru_hidden_dim
            )

            # Format inputs: add time dimension
            obs_seq = obs[None, ...]  # (1, batch_size, obs_dim)
            dones_seq = jnp.zeros((1, batch_size), dtype=bool)
            avail_seq = None  # Continuous actions

            policy_x = (obs_seq, dones_seq, avail_seq)

            # Empty adapters dict for independent policies
            adapters_dict = {}

            # Apply policy
            _, output = policy_model.apply(
                {"params": params}, init_hidden, policy_x, adapters_dict
            )
            mean_seq, log_std_seq = output
            mean = mean_seq[0]  # Remove time dimension
            log_std = log_std_seq[0]
            std = jnp.exp(log_std)

            return mean, std

    else:
        # Standard MLP policy
        def get_policy_output(params, obs):
            """Apply policy to get mean and log_std."""
            mean, log_std = policy_model.apply({"params": params}, obs)
            std = jnp.exp(log_std)
            # log_std is a parameter of shape (action_dim,), need to broadcast to (num_envs, action_dim)
            # mean has shape (num_envs, action_dim), std needs to match
            if std.ndim == 1:
                # Broadcast std to match mean's batch dimension
                std = jnp.broadcast_to(std, mean.shape)
            return mean, std

    # vmap over agents: (num_agents, num_envs, action_dim)
    means, stds = jax.vmap(get_policy_output)(policy_params, obs_transposed)

    # Transpose back: (num_envs, num_agents, action_dim)
    means = jnp.transpose(means, (1, 0, 2))
    stds = jnp.transpose(stds, (1, 0, 2))

    # Split by agent for pairwise comparison
    policy_outputs = [(means[:, i, :], stds[:, i, :]) for i in range(num_agents)]

    # Compute pairwise Wasserstein distances
    pairwise_distances = compute_pairwise_diversity(
        policy_outputs
    )  # (num_envs, n_pairs)

    # SND is the mean pairwise distance
    snd_value = float(jnp.mean(pairwise_distances))

    return snd_value


def calculate_snd_statistics_independent(
    policy_params: jnp.ndarray,
    obs_batch: jnp.ndarray,
    policy_model,
    num_agents: int,
    num_envs: int,
    num_samples: int = 10,
    rng_key: jax.random.PRNGKey = None,
) -> Dict[str, float]:
    """
    Calculate Expected SND statistics for independent policies.

    This computes Expected SND by averaging over multiple observation samples.
    Unlike the hypernetwork version, independent policies have fixed parameters
    per agent, so diversity comes from different observation inputs.

    Args:
        policy_params: Stacked policy parameters (num_agents, ...)
        obs_batch: Current observations (num_envs * num_agents, obs_dim)
        policy_model: Policy model instance
        num_agents: Number of agents per environment
        num_envs: Number of parallel environments
        num_samples: Number of observation samples to average over for Expected SND
        rng_key: JAX random key for sampling observations (not used for independent)

    Returns:
        Dictionary with SND statistics:
            - 'snd_total': Expected SND (mean pairwise Wasserstein distance)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # For independent policies, we use the current observations
    # (since agents are already differentiated by their parameters)
    snd_value = calculate_snd_independent(
        policy_params,
        obs_batch,
        policy_model,
        num_agents,
        num_envs,
    )

    return {
        "snd_total": float(snd_value),
    }


@partial(jax.jit, static_argnums=(2, 3, 4))
def calculate_snd_dico(
    policy_params: Dict[str, Any],
    obs_batch: jnp.ndarray,
    policy_model,
    num_agents: int,
    num_envs: int,
    diversity_scaling: float = 1.0,
) -> jnp.ndarray:
    """
    Calculate System Neural Diversity (SND) for DiCo policies.

    DiCo uses homogeneous + heterogeneous networks. SND measures diversity by
    computing pairwise Wasserstein distances between agent policy outputs.

    NOTE: This function is JIT-compiled for performance. The policy_model,
    num_agents, and num_envs arguments are static (must be known at compile time).
    diversity_scaling is a DYNAMIC argument to avoid recompilation on every change.

    Args:
        policy_params: DiCo policy parameters
        obs_batch: Batch of observations (num_envs * num_agents, obs_dim)
        policy_model: DiCo policy model instance (static)
        num_agents: Number of agents per environment (static)
        num_envs: Number of parallel environments (static)
        diversity_scaling: Diversity scaling factor (dynamic, default 1.0 for unscaled)
                          Can be a JAX array or Python scalar

    Returns:
        SND value (JAX scalar): Mean pairwise Wasserstein distance across all agent pairs
    """
    # Infer actual num_envs from batch size
    actual_batch_size = obs_batch.shape[0]
    actual_num_envs = actual_batch_size // num_agents

    # Create agent IDs for each observation
    # For DiCo: agent_ids must match the observation structure from train.py
    # train.py tiles as: [env0_agent0, env0_agent1, ..., env1_agent0, env1_agent1, ...]
    # So agent_ids should be: [0, 1, 2, ..., num_agents-1, 0, 1, 2, ..., num_agents-1, ...]
    # Shape: (num_envs * num_agents,)
    agent_ids = jnp.tile(jnp.arange(num_agents), actual_num_envs)

    # Get policy outputs with given diversity scaling
    # Note: We only use means for SND calculation, not stds
    mean_batch, _ = policy_model.apply(
        {"params": policy_params},
        obs_batch,
        agent_ids,
        diversity_scaling,
    )

    # Reshape to (num_envs, num_agents, action_dim)
    action_dim = mean_batch.shape[-1]
    means = mean_batch.reshape(actual_num_envs, num_agents, action_dim)

    # Compute pairwise W2 distances using the same formula as calculate_snd
    # W_2 = sqrt(mean over observations of ||pi_i(o) - pi_j(o)||^2)
    if num_agents < 2:
        snd_value = jnp.array(0.0)
    else:
        # Generate all pairs (i, j) where i < j
        i_pairs, j_pairs = jnp.triu_indices(num_agents, k=1)

        def compute_w2_distance(i, j):
            # Compute squared L2 norm between agent i and j's outputs for each observation
            # means[:, i, :] has shape (num_envs, action_dim)
            obs_wise_dist_sq = jnp.sum(
                (means[:, i, :] - means[:, j, :]) ** 2,
                axis=-1,  # Sum over action_dim
            )  # (num_envs,)
            # Take mean over observations, then square root
            return jnp.sqrt(jnp.mean(obs_wise_dist_sq))

        # Use vmap to compute all pairwise distances in parallel
        pair_distances = jax.vmap(compute_w2_distance)(i_pairs, j_pairs)

        # SND is the mean pairwise distance
        snd_value = jnp.mean(pair_distances)

    return snd_value


def calculate_snd(
    policy_params: Dict[str, Any],
    adapters_dict: Dict[str, jnp.ndarray],
    sample_obs: jnp.ndarray,
    policy_model,
    num_agents: int,
    sample_size: int,
    use_gru_policy: bool = False,
    # Optional: for computing both scaled and unscaled SND
    hypernetwork=None,
    hn_params: Dict[str, Any] = None,
    task_batch: jnp.ndarray = None,
    context_batch: jnp.ndarray = None,
    lidar_batch: jnp.ndarray = None,
    mask: jnp.ndarray = None,
    diversity_scaling: float = None,
) -> Tuple[float, float, jnp.ndarray]:
    """
    Calculate SND using unified observations from sampled environments.

    This implements the paper's definition: "O is created by unifying observations
    from all agents and all timesteps in a batch". Each agent is evaluated on the
    SAME unified set of observations from all agents and timesteps.

    Key aspects:
    - Uses PRE-GENERATED adapters (already environment-specific)
    - Handles adapter tiling for unified observation evaluation
    - Only computes mean-based Wasserstein distance (no std component)
    - Returns both SND value and agent means for further processing
    - Optionally computes both scaled and unscaled SND if hypernetwork info provided

    Args:
        policy_params: Policy network parameters
        adapters_dict: Pre-generated adapter dictionary (num_envs * num_agents, ...)
        sample_obs: Sampled observations (sample_size, num_agents, obs_dim)
        policy_model: Policy model instance
        num_agents: Number of agents per environment
        sample_size: Number of environments sampled
        use_gru_policy: Whether using GRU policy (requires special handling)
        hypernetwork: Optional hypernetwork model for computing unscaled SND
        hn_params: Optional hypernetwork parameters
        task_batch: Optional task batch for regenerating adapters
        context_batch: Optional context batch for regenerating adapters
        lidar_batch: Optional lidar batch for regenerating adapters
        mask: Optional mask for regenerating adapters
        diversity_scaling: Optional diversity scaling used for scaled adapters

    Returns:
        snd_scaled: Mean pairwise Wasserstein distance with given adapters
        snd_unscaled: Mean pairwise Wasserstein distance with unscaled adapters (or same as scaled if not computed)
        all_agent_means: Policy outputs (num_agents, num_unified_obs, action_dim)
    """
    # Create unified observation set from ALL sampled environments
    # sample_obs shape: (sample_size, num_agents, obs_dim)
    unified_obs = sample_obs.reshape(
        -1, sample_obs.shape[-1]
    )  # (sample_size * num_agents, obs_dim)
    num_unified_obs = unified_obs.shape[0]

    # For each agent role, evaluate on ALL observations using environment-specific adapters
    # Key insight: observation i comes from environment (i // num_agents)
    # So we use that environment's adapters for that observation
    agent_means_list = []

    for agent_idx in range(num_agents):
        # Build adapters for this agent role across all observations
        # For each observation, use the adapters from its source environment
        agent_adapters = {}

        for key, adapter in adapters_dict.items():
            # Reshape to (num_envs, num_agents, ...)
            # Note: adapters_dict was generated for all envs, we need to extract sampled envs
            adapter_shape = adapter.shape
            num_envs_total = adapter_shape[0] // num_agents
            adapter_reshaped = adapter.reshape(
                num_envs_total, num_agents, *adapter_shape[1:]
            )

            # Extract this agent's adapters from sampled environments: (sample_size, ...)
            agent_adapters_sampled = adapter_reshaped[:sample_size, agent_idx, ...]

            # Each environment contributes num_agents observations
            # So we need to tile each env's adapter num_agents times
            adapter_list = []
            for env_idx in range(sample_size):
                env_adapter = agent_adapters_sampled[env_idx : env_idx + 1, ...]
                # Tile for all observations from this environment
                tiled = jnp.tile(
                    env_adapter,
                    (num_agents, *([1] * (len(env_adapter.shape) - 1))),
                )
                adapter_list.append(tiled)

            # Concatenate: (sample_size * num_agents, ...)
            agent_adapters[key] = jnp.concatenate(adapter_list, axis=0)

        # Evaluate this agent role's policy on ALL unified observations
        if use_gru_policy:
            init_hidden = policy_model.initialize_carry(
                num_unified_obs, policy_model.gru_hidden_dim
            )
            obs_seq = unified_obs[None, ...]  # (1, num_unified_obs, obs_dim)
            dones_seq = jnp.zeros((1, num_unified_obs), dtype=bool)
            avail_seq = None
            policy_x = (obs_seq, dones_seq, avail_seq)
            _, output = policy_model.apply(
                {"params": policy_params},
                init_hidden,
                policy_x,
                agent_adapters,
            )
            mean_seq, _ = output
            agent_mean = mean_seq[0]  # (num_unified_obs, action_dim)
        else:
            agent_mean, _ = policy_model.apply(
                {"params": policy_params},
                unified_obs,
                agent_adapters,
            )

        agent_means_list.append(agent_mean)

    # Stack: (num_agents, num_unified_obs, action_dim)
    all_agent_means = jnp.stack(agent_means_list, axis=0)

    # Compute pairwise Wasserstein-2 distances (mean-only, following paper)
    # Use vmap for parallel computation
    if num_agents < 2:
        snd_scaled = 0.0
    else:
        # Generate all pairs (i, j) where i < j using triu_indices (JIT-compatible)
        i_pairs, j_pairs = jnp.triu_indices(num_agents, k=1)

        # Vectorized function to compute W2 distance for a single agent pair
        def compute_w2_distance(i, j):
            # W_2 = sqrt(mean over O of ||pi_i(o) - pi_j(o)||^2)
            obs_wise_dist_sq = jnp.sum(
                (all_agent_means[i] - all_agent_means[j]) ** 2,
                axis=-1,  # Sum over action_dim
            )  # (num_unified_obs,)
            return jnp.sqrt(jnp.mean(obs_wise_dist_sq))

        # Use vmap to compute all pairwise distances in parallel
        pair_distances = jax.vmap(compute_w2_distance)(i_pairs, j_pairs)
        snd_scaled = float(jnp.mean(pair_distances))

    # Compute UNSCALED SND if hypernetwork info is provided
    snd_unscaled = snd_scaled  # Default: same as scaled

    if (
        hypernetwork is not None
        and hn_params is not None
        and task_batch is not None
        and context_batch is not None
        and diversity_scaling is not None
        and diversity_scaling != 1.0
    ):

        # Generate unscaled adapters (diversity_scaling = 1.0)
        # Create dummy food position vectors if hypernetwork expects them
        food_position_dim = hypernetwork.food_position_dim
        if food_position_dim > 0:
            # Infer batch shape from context_batch
            batch_shape = context_batch.shape[:2]  # (num_envs, num_agents)
            food_batch_dummy = jnp.zeros((*batch_shape, food_position_dim))
        else:
            food_batch_dummy = None

        # Create dummy agent position vectors if hypernetwork expects them
        agent_position_dim = hypernetwork.agent_position_dim
        if agent_position_dim > 0:
            batch_shape = context_batch.shape[:2]  # (num_envs, num_agents)
            agent_position_batch_dummy = jnp.zeros((*batch_shape, agent_position_dim))
        else:
            agent_position_batch_dummy = None

        # Create dummy target_snd vectors if hypernetwork expects them
        target_snd_dim = hypernetwork.target_snd_dim
        if target_snd_dim > 0:
            batch_shape = context_batch.shape[:2]  # (num_envs, num_agents)
            target_snd_batch_dummy = jnp.zeros((*batch_shape, target_snd_dim))
        else:
            target_snd_batch_dummy = None

        # Create dummy env_context vectors if hypernetwork expects them
        env_context_dim = hypernetwork.env_context_dim
        if env_context_dim > 0:
            batch_shape = context_batch.shape[:2]  # (num_envs, num_agents)
            env_context_batch_dummy = jnp.zeros((*batch_shape, env_context_dim))
        else:
            env_context_batch_dummy = None

        adapters_dict_unscaled = hypernetwork.apply(
            {"params": hn_params},
            task_batch,
            context_batch,
            lidar_batch,
            food_batch_dummy,  # Dummy food positions to match expected feature dimensions
            agent_position_batch_dummy,  # Dummy agent positions to match expected feature dimensions
            target_snd_batch_dummy,  # Dummy target SND to match expected feature dimensions
            env_context_batch_dummy,  # Dummy env context to match expected feature dimensions
            mask,
            diversity_scaling=1.0,
        )

        # Recalculate agent means with unscaled adapters
        agent_means_list_unscaled = []

        for agent_idx in range(num_agents):
            agent_adapters_unscaled = {}

            for key, adapter in adapters_dict_unscaled.items():
                adapter_shape = adapter.shape
                num_envs_total = adapter_shape[0] // num_agents
                adapter_reshaped = adapter.reshape(
                    num_envs_total, num_agents, *adapter_shape[1:]
                )
                agent_adapters_sampled = adapter_reshaped[:sample_size, agent_idx, ...]

                adapter_list = []
                for env_idx in range(sample_size):
                    env_adapter = agent_adapters_sampled[env_idx : env_idx + 1, ...]
                    tiled = jnp.tile(
                        env_adapter,
                        (num_agents, *([1] * (len(env_adapter.shape) - 1))),
                    )
                    adapter_list.append(tiled)

                agent_adapters_unscaled[key] = jnp.concatenate(adapter_list, axis=0)

            # Evaluate with unscaled adapters
            if use_gru_policy:
                init_hidden = policy_model.initialize_carry(
                    num_unified_obs, policy_model.gru_hidden_dim
                )
                obs_seq = unified_obs[None, ...]
                dones_seq = jnp.zeros((1, num_unified_obs), dtype=bool)
                avail_seq = None
                policy_x = (obs_seq, dones_seq, avail_seq)
                _, output = policy_model.apply(
                    {"params": policy_params},
                    init_hidden,
                    policy_x,
                    agent_adapters_unscaled,
                )
                mean_seq, _ = output
                agent_mean_unscaled = mean_seq[0]
            else:
                agent_mean_unscaled, _ = policy_model.apply(
                    {"params": policy_params},
                    unified_obs,
                    agent_adapters_unscaled,
                )

            agent_means_list_unscaled.append(agent_mean_unscaled)

        # Stack and compute unscaled SND
        all_agent_means_unscaled = jnp.stack(agent_means_list_unscaled, axis=0)

        # Use vmap for parallel pairwise distance computation
        # Reuse the same pair indices from scaled SND calculation
        def compute_w2_distance_unscaled(i, j):
            obs_wise_dist_sq = jnp.sum(
                (all_agent_means_unscaled[i] - all_agent_means_unscaled[j]) ** 2,
                axis=-1,
            )
            return jnp.sqrt(jnp.mean(obs_wise_dist_sq))

        # Use vmap to compute all pairwise distances in parallel
        pair_distances_unscaled = jax.vmap(compute_w2_distance_unscaled)(
            i_pairs, j_pairs
        )
        snd_unscaled = float(jnp.mean(pair_distances_unscaled))

    return snd_scaled, snd_unscaled, all_agent_means


def calculate_adapter_snd(
    adapter_snd_buffer: list,
    sample_size: int,
    hn_params: Dict[str, Any],
    policy_params: Dict[str, Any],
    hypernetwork,
    policy_model,
    num_agents: int,
    rng_key: jax.random.PRNGKey,
    use_gru_policy: bool = False,
    return_action_outputs: bool = False,
    trajectory_obs: list = None,
    diversity_scaling: float = 1.0,
    scenario_name: str = None,
) -> float:
    """
    Calculate SND based on pairwise distance between different LoRA adapters.

    This function computes diversity by:
    1. Sampling entries from adapter SND buffer (one entry = one agent context)
    2. Each entry contains a single agent's task/context/lidar from when HN was queried
    3. Up to sample_size entries are randomly sampled from the buffer
    4. Generating one adapter per sampled entry
    5. Sampling observations from trajectory data (replay buffer)
    6. Evaluating all adapters on the same set of observations
    7. Computing pairwise Wasserstein distances between all adapters (averaged across observations)
    8. Averaging over all adapter pairs to get final SND value

    Buffer structure (refilled each episode):
    - Each entry is ONE agent's context (not grouped by environment)
    - Initial: num_envs * num_agents entries (one per agent)
    - Growth: Additional entries added when agents trigger events during rollout
    - Maximum buffer size: 2 * num_agents * num_envs (initial + one requery per agent)
    - Buffer size directly equals number of contexts available

    Example with sample_size=128, num_agents=2, num_envs=128:
    - Initial: 128 × 2 = 256 entries (one per agent)
    - During rollout: Additional entries when agents detect food/landmarks
    - Maximum: 128 × 2 × 2 = 512 entries (initial + one requery per agent)
    - Sample min(128, buffer_size) entries randomly from buffer
    - Generate one adapter per sampled entry
    - Evaluate each on sampled observations
    - Compute pairwise distances
    Args:
        adapter_snd_buffer: List of dicts containing single agent contexts (refilled each episode)
                           Each dict has keys: 'task', 'context', 'lidar', 'query_type', 'hidden_state', 'agent_idx'
                           - 'task': Task embedding (task_dim,)
                           - 'context': Agent context/capability (context_dim,)
                           - 'lidar': Lidar readings (lidar_dim,) or None
                           - 'query_type': 'initial' or 'requery'
                           - 'hidden_state': GRU hidden state at time of query (hidden_dim,) or None
                           - 'agent_idx': Agent index for food position extraction (int)
                           Buffer size: Up to 2 * num_agents * num_envs entries
        sample_size: Target number of adapters to generate and compare
        hn_params: Hypernetwork parameters
        policy_params: Policy network parameters
        hypernetwork: Hypernetwork model instance
        policy_model: Policy model instance
        num_agents: Number of agents per environment
        rng_key: JAX random key for sampling
        use_gru_policy: Whether policy uses GRU
        return_action_outputs: If True, also return action means for visualization
        trajectory_obs: List of observation arrays from rollout trajectory. If None, falls back to random obs.
                       Each entry has shape (batch_size, obs_dim) where batch_size = num_envs * num_agents
        diversity_scaling: Scaling factor for LoRA adapters (default 1.0 for unscaled diversity)
        scenario_name: Scenario name for determining if/how to extract food positions from observations
        trajectory_obs: List of observation arrays from rollout trajectory. If None, falls back to random obs.
                       Each entry has shape (batch_size, obs_dim) where batch_size = num_envs * num_agents
        diversity_scaling: Scaling factor for LoRA adapters (default 1.0 for unscaled diversity)

    Returns:
        If return_action_outputs is False:
            SND value (float): Mean pairwise Wasserstein distance across all adapter pairs
        If return_action_outputs is True:
            Tuple of (SND value, action_means) where action_means has shape (num_adapters, num_eval_obs, action_dim)
    """
    if adapter_snd_buffer is None or len(adapter_snd_buffer) == 0:
        if return_action_outputs:
            return 0.0, None
        else:
            return 0.0

    # Sample from adapter SND buffer for task/context/lidar
    # Buffer structure: One entry per agent context
    # Each entry contains: task (task_dim), context (context_dim), lidar (lidar_dim), query_type
    actual_buffer_size = len(adapter_snd_buffer)

    # Each entry IS one context, so buffer size = total contexts
    if actual_buffer_size <= sample_size:
        # Use all available entries
        sampled_entries = adapter_snd_buffer
    else:
        # Randomly sample exactly sample_size entries
        rng_key, subkey = jax.random.split(rng_key)
        sample_indices = jax.random.choice(
            subkey, actual_buffer_size, shape=(sample_size,), replace=False
        )
        sample_indices_np = np.array(sample_indices)
        sampled_entries = [adapter_snd_buffer[int(i)] for i in sample_indices_np]

    # Extract task/context/lidar/hidden_state from sampled entries
    # Each entry is already a single agent's context
    task_list = [entry["task"] for entry in sampled_entries]  # Each: (task_dim,)
    context_list = [
        entry["context"] for entry in sampled_entries
    ]  # Each: (context_dim,)
    lidar_list = [
        entry.get("lidar") for entry in sampled_entries
    ]  # Each: (lidar_dim,) or None
    hidden_state_list = [
        entry.get("hidden_state") for entry in sampled_entries
    ]  # Each: (hidden_dim,) or None
    agent_idx_list = [
        entry.get("agent_idx", 0) for entry in sampled_entries
    ]  # Each: int (default 0 if not present)

    # Number of adapters to generate = number of sampled entries
    num_adapters_to_generate = len(sampled_entries)

    if num_adapters_to_generate == 0:
        # No contexts available - return early
        if return_action_outputs:
            return 0.0, None
        else:
            return 0.0

    # Sample observations from trajectory data
    if trajectory_obs is not None and len(trajectory_obs) > 0:
        # trajectory_obs is a list of arrays, each (batch_size, obs_dim)
        # Concatenate all timesteps and flatten to get individual agent observations
        all_obs = jnp.concatenate(
            trajectory_obs, axis=0
        )  # (total_timesteps * batch_size, obs_dim)

        # Sample observations (use up to sample_size observations)
        total_obs_available = all_obs.shape[0]
        num_obs_to_sample = min(sample_size, total_obs_available)
        if num_obs_to_sample < total_obs_available:
            # Random sample without replacement
            rng_key, subkey = jax.random.split(rng_key)
            obs_indices = jax.random.choice(
                subkey, total_obs_available, shape=(num_obs_to_sample,), replace=False
            )
            obs_batch = all_obs[obs_indices]  # (num_obs_to_sample, obs_dim)
        else:
            obs_batch = all_obs  # Use all available
    else:
        # No trajectory observations available - cannot compute SND
        if return_action_outputs:
            return 0.0, None
        else:
            return 0.0

    # ============================================================
    # SPECIAL PATH: dispersion_vmas with food-position-only context
    # ============================================================
    # When food_position_dim > 0 and context/task dims are 0, the only
    # differentiating hypernetwork input is the per-agent food positions.
    # Feeding all agent slots the SAME food vector (as in the general path)
    # produces identical adapters for all agents → SND = 0.
    #
    # Correct approach: sample full environment snapshots from trajectory_obs,
    # feed each hypernetwork call with DIFFERENT food position vectors per agent
    # slot (drawn from each agent's actual observation in that environment), then
    # compute SND as pairwise diversity between the resulting agent-specific adapters.

    if scenario_name == "dispersion_vmas" and hypernetwork.food_position_dim == 0:
        food_position_dim_hn = hypernetwork.food_position_dim  # = num_agents * 2
        hn_max_agents = hypernetwork.max_agents
        task_embed_dim_hn = hypernetwork.task_embed_dim
        context_dim_hn = hypernetwork.context_dim
        agent_position_dim_hn = hypernetwork.agent_position_dim

        # Reshape trajectory into (num_steps, num_envs, num_agents, obs_dim)
        obs_dim_traj = trajectory_obs[0].shape[-1]
        batch_size_per_step = trajectory_obs[0].shape[0]  # num_envs * num_agents
        num_envs_traj = batch_size_per_step // num_agents
        num_steps_traj = len(trajectory_obs)

        # Stack and group: (num_steps, num_envs, num_agents, obs_dim)
        all_steps = jnp.stack(
            trajectory_obs, axis=0
        )  # (num_steps, num_envs * num_agents, obs_dim)
        all_steps_grouped = all_steps.reshape(
            num_steps_traj, num_envs_traj, num_agents, obs_dim_traj
        )

        # Sample (timestep, env) pairs
        total_pairs = num_steps_traj * num_envs_traj
        actual_sample_size = min(sample_size, total_pairs)
        rng_key, subkey = jax.random.split(rng_key)
        pair_indices = jax.random.choice(
            subkey,
            total_pairs,
            shape=(actual_sample_size,),
            replace=actual_sample_size > total_pairs,
        )
        timestep_indices = pair_indices // num_envs_traj
        env_indices_samp = pair_indices % num_envs_traj

        # Per-env snapshots: (actual_sample_size, num_agents, obs_dim)
        sampled_env_obs = all_steps_grouped[timestep_indices, env_indices_samp]

        # Build per-agent food position vectors for each snapshot
        # Agent a gets concat of [rel_x, rel_y] for ALL food items k
        # (i.e., positions are relative to agent a's own location)
        all_food_vecs = []
        for k in range(num_agents):
            food_start = 4 + k * 3
            food_k = sampled_env_obs[
                :, :, food_start : food_start + 2
            ]  # (S, num_agents, 2)
            all_food_vecs.append(food_k)
        food_positions_all = jnp.concatenate(
            all_food_vecs, axis=-1
        )  # (S, num_agents, num_agents*2)

        # Pad agent dimension to hn_max_agents if needed
        if num_agents < hn_max_agents:
            pad_food = jnp.zeros(
                (actual_sample_size, hn_max_agents - num_agents, food_position_dim_hn)
            )
            food_positions_all = jnp.concatenate([food_positions_all, pad_food], axis=1)

        # Agent absolute positions (first 2 dims of each observation = absolute pos)
        if agent_position_dim_hn > 0:
            agent_abs_pos = sampled_env_obs[:, :, :2]  # (S, num_agents, 2)
            if num_agents < hn_max_agents:
                pad_ap = jnp.zeros((actual_sample_size, hn_max_agents - num_agents, 2))
                agent_abs_pos = jnp.concatenate([agent_abs_pos, pad_ap], axis=1)
        else:
            agent_abs_pos = None

        # Zero task; context is the per-agent one-hot IDs (size = context_dim_hn = max_agents)
        # if one-hot IDs are used, otherwise zeros for the empty-context case.
        dummy_task_hn = jnp.zeros(
            (actual_sample_size, hn_max_agents, task_embed_dim_hn)
        )
        if context_dim_hn > 0:
            # Build per-agent one-hot identity: agent i gets e_i of length context_dim_hn.
            # Shape: (hn_max_agents, context_dim_hn) broadcast to (S, hn_max_agents, context_dim_hn)
            one_hot_ids = jnp.eye(
                hn_max_agents, context_dim_hn
            )  # (hn_max_agents, context_dim_hn)
            dummy_context_hn = jnp.broadcast_to(
                one_hot_ids[None, :, :],
                (actual_sample_size, hn_max_agents, context_dim_hn),
            )
        else:
            dummy_context_hn = jnp.zeros(
                (actual_sample_size, hn_max_agents, context_dim_hn)
            )

        # Create dummy target_snd vectors
        target_snd_dim_hn = hypernetwork.target_snd_dim
        if target_snd_dim_hn > 0:
            dummy_target_snd_hn = jnp.zeros(
                (actual_sample_size, hn_max_agents, target_snd_dim_hn)
            )
        else:
            dummy_target_snd_hn = None

        # Create dummy env_context vectors
        env_context_dim_hn = hypernetwork.env_context_dim
        if env_context_dim_hn > 0:
            dummy_env_context_hn = jnp.zeros(
                (actual_sample_size, hn_max_agents, env_context_dim_hn)
            )
        else:
            dummy_env_context_hn = None

        # Single hypernetwork call with per-agent food positions → diverse adapters
        adapters_all = hypernetwork.apply(
            {"params": hn_params},
            dummy_task_hn,
            dummy_context_hn,
            None,  # no lidar for dispersion_vmas
            food_positions_all,
            agent_abs_pos,
            dummy_target_snd_hn,  # Dummy target SND for SND calculation
            dummy_env_context_hn,  # Dummy env context for SND calculation
            mask=None,
            diversity_scaling=diversity_scaling,
        )

        # Reshape: (S * hn_max_agents, ...) → (S, num_agents, ...)
        adapters_per_env = {}
        for key, value in adapters_all.items():
            v_shape = value.shape
            adapters_per_env[key] = value.reshape(
                actual_sample_size, hn_max_agents, *v_shape[1:]
            )[
                :, :num_agents, ...
            ]  # (S, num_agents, ...)

        # Evaluate all agents' adapters on a COMMON observation per snapshot.
        # Using per-agent observations would add a baseline shared-policy diversity
        # (from observation differences) that is constant regardless of diversity_scaling.
        # That baseline prevents proportional SND scaling with s and causes the
        # diversity control loop to underestimate the achieved diversity.
        # By using the same observation for all agents within a snapshot, SND is
        # measured purely from adapter differences → scales as s² as expected.
        # Use agent 0's observation, broadcast to all num_agents slots.
        common_obs = sampled_env_obs[:, 0:1, :]  # (S, 1, obs_dim)
        common_obs_tiled = jnp.broadcast_to(
            common_obs, (actual_sample_size, num_agents, obs_dim_traj)
        )  # (S, num_agents, obs_dim)
        obs_flat = common_obs_tiled.reshape(
            -1, obs_dim_traj
        )  # (S * num_agents, obs_dim)

        adapters_flat = {
            k: v.reshape(-1, *v.shape[2:]) for k, v in adapters_per_env.items()
        }

        means_flat, log_stds_flat = policy_model.apply(
            {"params": policy_params},
            obs_flat,
            adapters_flat,
        )
        action_dim_snd = means_flat.shape[-1]
        means = means_flat.reshape(actual_sample_size, num_agents, action_dim_snd)
        stds = jnp.exp(log_stds_flat).reshape(
            actual_sample_size, num_agents, action_dim_snd
        )

        # SND: pairwise diversity between agent adapters, averaged over snapshots
        # compute_snd_from_action_means expects (num_adapters, num_obs, action_dim)
        means_for_snd = jnp.transpose(means, (1, 0, 2))  # (num_agents, S, action_dim)
        stds_for_snd = jnp.transpose(stds, (1, 0, 2))  # (num_agents, S, action_dim)
        snd = compute_snd_from_action_means(means_for_snd, stds_for_snd)

        if return_action_outputs:
            return snd, means_for_snd
        else:
            return snd

    # ============================================================
    # SPECIAL PATH: wind_flocking_position with position-based context
    # ============================================================
    # Keep this path for experimentation, but default to disabled so
    # wind_flocking_position uses the general adapter-SND flow.
    use_wind_flocking_special_sampling = False

    if (
        use_wind_flocking_special_sampling
        and scenario_name == "wind_flocking_position"
        and hypernetwork.agent_position_dim > 0
    ):
        agent_position_dim_hn = hypernetwork.agent_position_dim
        hn_max_agents = hypernetwork.max_agents
        task_embed_dim_hn = hypernetwork.task_embed_dim
        context_dim_hn = hypernetwork.context_dim
        target_snd_dim_hn = hypernetwork.target_snd_dim

        # Reshape trajectory into (num_steps, num_envs, num_agents, obs_dim)
        obs_dim_traj = trajectory_obs[0].shape[-1]
        batch_size_per_step = trajectory_obs[0].shape[0]  # num_envs * num_agents
        num_envs_traj = batch_size_per_step // num_agents
        num_steps_traj = len(trajectory_obs)

        # Stack and group: (num_steps, num_envs, num_agents, obs_dim)
        all_steps = jnp.stack(
            trajectory_obs, axis=0
        )  # (num_steps, num_envs * num_agents, obs_dim)
        all_steps_grouped = all_steps.reshape(
            num_steps_traj, num_envs_traj, num_agents, obs_dim_traj
        )

        # Sample ONLY from initial timestep (t=0) to get spawn positions
        # This ensures we use the same position contexts that were used during rollout
        # (adapters are generated from INITIAL positions, not mid-episode positions)
        actual_sample_size = min(sample_size, num_envs_traj)
        timestep_indices = jnp.zeros(actual_sample_size, dtype=jnp.int32)  # All t=0
        rng_key, subkey = jax.random.split(rng_key)
        env_indices_samp = jax.random.choice(
            subkey,
            num_envs_traj,
            shape=(actual_sample_size,),
            replace=actual_sample_size > num_envs_traj,
        )

        # Per-env snapshots: (actual_sample_size, num_agents, obs_dim)
        sampled_env_obs = all_steps_grouped[timestep_indices, env_indices_samp]

        # Extract agent positions (first 2 dims of observation = absolute position)
        # For wind_flocking: obs = pos(2) + vel(2) + rel_pos_others(...) + wind(2)
        agent_abs_pos = sampled_env_obs[:, :, :2]  # (actual_sample_size, num_agents, 2)

        # Compute relative positions (relative to center of mass per environment)
        center_of_mass = jnp.mean(
            agent_abs_pos, axis=1, keepdims=True
        )  # (actual_sample_size, 1, 2)
        agent_rel_pos = (
            agent_abs_pos - center_of_mass
        )  # (actual_sample_size, num_agents, 2)

        # Pad agent dimension to hn_max_agents if needed
        if num_agents < hn_max_agents:
            pad_pos = jnp.zeros(
                (actual_sample_size, hn_max_agents - num_agents, agent_position_dim_hn)
            )
            agent_positions_all = jnp.concatenate([agent_rel_pos, pad_pos], axis=1)
        else:
            agent_positions_all = agent_rel_pos

        # Create dummy task and context
        dummy_task_hn = jnp.zeros(
            (actual_sample_size, hn_max_agents, task_embed_dim_hn)
        )
        dummy_context_hn = jnp.zeros(
            (actual_sample_size, hn_max_agents, context_dim_hn)
        )

        # Create dummy target_snd vectors
        if target_snd_dim_hn > 0:
            dummy_target_snd_hn = jnp.zeros(
                (actual_sample_size, hn_max_agents, target_snd_dim_hn)
            )
        else:
            dummy_target_snd_hn = None

        # Single hypernetwork call with per-agent positions -> diverse adapters
        adapters_all = hypernetwork.apply(
            {"params": hn_params},
            dummy_task_hn,
            dummy_context_hn,
            None,  # no lidar for wind_flocking
            None,  # no food for wind_flocking
            agent_positions_all,  # Different positions per agent!
            dummy_target_snd_hn,
            None,  # no env_context
            mask=None,
            diversity_scaling=diversity_scaling,
        )

        # Reshape: (actual_sample_size * hn_max_agents, ...) -> (actual_sample_size, num_agents, ...)
        adapters_per_env = {}
        for key, value in adapters_all.items():
            v_shape = value.shape
            adapters_per_env[key] = value.reshape(
                actual_sample_size, hn_max_agents, *v_shape[1:]
            )[
                :, :num_agents, ...
            ]  # (actual_sample_size, num_agents, ...)

        # Evaluate all agents' adapters on a COMMON observation per snapshot
        # Use agent 0's observation, broadcast to all num_agents slots
        common_obs = sampled_env_obs[:, 0:1, :]  # (actual_sample_size, 1, obs_dim)
        common_obs_tiled = jnp.broadcast_to(
            common_obs, (actual_sample_size, num_agents, obs_dim_traj)
        )  # (actual_sample_size, num_agents, obs_dim)
        obs_flat = common_obs_tiled.reshape(
            -1, obs_dim_traj
        )  # (actual_sample_size * num_agents, obs_dim)

        adapters_flat = {
            k: v.reshape(-1, *v.shape[2:]) for k, v in adapters_per_env.items()
        }

        means_flat, log_stds_flat = policy_model.apply(
            {"params": policy_params},
            obs_flat,
            adapters_flat,
        )
        action_dim_snd = means_flat.shape[-1]
        means = means_flat.reshape(actual_sample_size, num_agents, action_dim_snd)
        stds = jnp.exp(log_stds_flat).reshape(
            actual_sample_size, num_agents, action_dim_snd
        )

        # SND: pairwise diversity between agent adapters, averaged over snapshots
        means_for_snd = jnp.transpose(
            means, (1, 0, 2)
        )  # (num_agents, actual_sample_size, action_dim)
        stds_for_snd = jnp.transpose(
            stds, (1, 0, 2)
        )  # (num_agents, actual_sample_size, action_dim)
        snd = compute_snd_from_action_means(means_for_snd, stds_for_snd)

        if return_action_outputs:
            return snd, means_for_snd
        else:
            return snd

    # ============================================================
    # GENERAL PATH: Standard adapter SND calculation
    # ============================================================

    # Handle task batch (might be None when task_embed_dim is 0)
    if task_list[0] is not None:
        task_batch = jnp.stack(
            task_list, axis=0
        )  # (num_adapters_to_generate, task_dim)
    else:
        # All entries are None - create zero array with appropriate shape
        task_batch = jnp.zeros((num_adapters_to_generate, 0))

    # Handle context batch (might be None when context_dim is 0)
    if context_list[0] is not None:
        context_batch = jnp.stack(
            context_list, axis=0
        )  # (num_adapters_to_generate, context_dim)
    else:
        # All entries are None - create zero array with appropriate shape
        context_batch = jnp.zeros((num_adapters_to_generate, 0))

    # Handle lidar (might be None for some entries)
    if lidar_list[0] is not None:
        lidar_batch = jnp.stack(
            lidar_list, axis=0
        )  # (num_adapters_to_generate, lidar_dim)
    else:
        lidar_batch = None

    # Extract food positions from observations if in dispersion scenario
    # For dispersion_vmas: obs structure is pos(2) + vel(2) + [food_0(3) + food_1(3) + ...]
    # Each agent i's matching food is at indices [4 + i*3 : 4 + i*3 + 2]
    #
    # CRITICAL FOR TRAINING: During actual training (in train.py), the hypernetwork receives
    # accurate, real-time food positions for each agent, enabling it to generate adapters
    # that help each agent reach its specific food target. This is essential for learning.
    #
    # FOR SND CALCULATION: Here we use food positions from sampled trajectory observations
    # as an approximation. This is acceptable because:
    # 1. SND measures behavioral diversity, not task performance
    # 2. Requerying the hypernetwork for each observation would be expensive
    # 3. Food positions are agent-relative and change each timestep anyway
    #
    # The hypernetwork generates adapters for ALL agent roles simultaneously,
    # so we provide food positions for all agents: agent i gets position to food i.
    if scenario_name == "dispersion_vmas" and obs_batch is not None:
        # Extract food positions from first observation only (avoid requerying hypernetwork)
        first_obs = obs_batch[0]  # (obs_dim,)

        # For each agent role, extract ALL food relative positions (not just matching food).
        # Observation structure: pos(2) + vel(2) + [food_0(3) + food_1(3) + ...]
        # Each food entry: [rel_x, rel_y, eaten_status] — food k starts at 4 + k*3.
        # We concatenate [rel_x, rel_y] for all k, giving (num_agents * 2,) per agent row.
        # This mirrors extract_food_positions() in train.py (food_position_dim = num_agents*2).
        all_foods_concat = []
        for k in range(num_agents):
            food_start_idx = 4 + k * 3
            all_foods_concat.append(first_obs[food_start_idx : food_start_idx + 2])
        # Concatenate into one vector of length num_agents*2
        food_row = jnp.concatenate(all_foods_concat, axis=0)  # (num_agents*2,)

        # Each agent gets the same full food vector: (num_agents, num_agents*2)
        all_food_positions = jnp.tile(food_row[None, :], (num_agents, 1))

        # Replicate for each sampled entry: (num_adapters_to_generate, num_agents, num_agents*2)
        food_batch = jnp.tile(
            all_food_positions[None, :, :], (num_adapters_to_generate, 1, 1)
        )
    else:
        food_batch = None

    # Reshape to match hypernetwork input format: (batch, max_agents, dim)
    # The hypernetwork expects (num_envs, max_agents, dim) where max_agents is the value
    # it was initialized with. Use hypernetwork.max_agents instead of env's num_agents.
    # Replicate each context across all agents since we're generating independent adapters
    # Each adapter is based on a single agent's context, but we need to match the expected shape
    hn_max_agents = hypernetwork.max_agents
    task_for_adapters = jnp.repeat(
        task_batch[:, None, :], hn_max_agents, axis=1
    )  # (num_adapters_to_generate, hn_max_agents, task_dim)
    context_for_adapters = jnp.repeat(
        context_batch[:, None, :], hn_max_agents, axis=1
    )  # (num_adapters_to_generate, hn_max_agents, context_dim)
    lidar_for_adapters = (
        jnp.repeat(lidar_batch[:, None, :], hn_max_agents, axis=1)
        if lidar_batch is not None
        else None
    )  # (num_adapters_to_generate, hn_max_agents, lidar_dim)

    # Handle food positions - use real positions if extracted, otherwise zeros/None
    food_position_dim = hypernetwork.food_position_dim
    if food_position_dim > 0:
        if food_batch is not None:
            # Food batch already has shape (num_adapters_to_generate, num_agents, food_position_dim)
            # Pad or trim to match hn_max_agents if needed
            current_num_agents = food_batch.shape[1]
            if current_num_agents < hn_max_agents:
                # Pad with zeros if we have fewer agents than hypernetwork expects
                padding = jnp.zeros(
                    (
                        num_adapters_to_generate,
                        hn_max_agents - current_num_agents,
                        food_position_dim,
                    )
                )
                food_for_adapters = jnp.concatenate([food_batch, padding], axis=1)
            elif current_num_agents > hn_max_agents:
                # Trim if we have more agents than hypernetwork expects
                food_for_adapters = food_batch[:, :hn_max_agents, :]
            else:
                # Perfect match
                food_for_adapters = food_batch
        else:
            # Fallback: use zeros if no food positions available
            food_for_adapters = jnp.zeros(
                (num_adapters_to_generate, hn_max_agents, food_position_dim)
            )
    else:
        food_for_adapters = None

    # Handle agent positions - extract from observations if available, otherwise zeros/None
    agent_position_dim = hypernetwork.agent_position_dim
    if agent_position_dim > 0:
        if (
            scenario_name in ["dispersion_vmas", "wind_flocking_position"]
            and obs_batch is not None
        ):
            # Extract agent positions from observations
            # For both scenarios, position is first 2 dims of observation

            if scenario_name == "wind_flocking_position":
                # CRITICAL: Create DIFFERENT positions for each adapter sample
                # This ensures non-zero adapter SND by giving each adapter different context

                # Generate random positions for each adapter (different spawn configurations)
                # Shape: (num_adapters_to_generate, num_agents, 2)
                from jax import random as jax_random
                import time

                # Create a key for randomization using current time to ensure variation
                # CRITICAL: Do NOT use fixed seed, or all episodes will have same positions!
                seed = int(time.time() * 1000000) % (2**32)  # Time-based seed
                key = jax_random.PRNGKey(seed)

                positions_list = []
                for adapter_idx in range(num_adapters_to_generate):
                    key, subkey = jax_random.split(key)

                    # Create random formation for this adapter
                    # Base spacing along X axis
                    spacing = 0.3
                    base_x = (
                        jnp.arange(num_agents) * spacing
                        - (num_agents - 1) * spacing / 2
                    )

                    # Add random offset and noise to create variation
                    x_offset = jax_random.uniform(
                        subkey, shape=(), minval=-1.0, maxval=1.0
                    )
                    key, subkey = jax_random.split(key)
                    y_offset = jax_random.uniform(
                        subkey, shape=(), minval=-0.5, maxval=0.5
                    )

                    # Create positions with noise
                    key, subkey = jax_random.split(key)
                    x_noise = jax_random.normal(subkey, shape=(num_agents,)) * 0.1
                    key, subkey = jax_random.split(key)
                    y_noise = jax_random.normal(subkey, shape=(num_agents,)) * 0.1

                    x_positions = base_x + x_offset + x_noise
                    y_positions = jnp.ones(num_agents) * y_offset + y_noise

                    agent_positions = jnp.stack(
                        [x_positions, y_positions], axis=1
                    )  # (num_agents, 2)

                    # Compute relative to center of mass
                    center_of_mass = jnp.mean(agent_positions, axis=0)
                    agent_positions_relative = agent_positions - center_of_mass

                    positions_list.append(agent_positions_relative)

                # Stack: (num_adapters_to_generate, num_agents, 2)
                all_agent_positions = jnp.stack(positions_list, axis=0)

            else:
                # dispersion_vmas: use same position for all (as before)
                # Collect positions for all agents from sampled observations
                num_eval_obs = obs_batch.shape[0]

                all_positions = []
                for obs_idx in range(
                    min(num_eval_obs, 10)
                ):  # Sample up to 10 observations
                    obs = obs_batch[obs_idx]  # (obs_dim,)
                    agent_pos = obs[:2]  # (2,) = (x, y) - first agent's position
                    all_positions.append(agent_pos)

                # Average position across sampled observations
                avg_pos = jnp.mean(jnp.stack(all_positions, axis=0), axis=0)  # (2,)

                # Replicate for all agent roles: (num_agents, 2)
                all_agent_positions_single = jnp.tile(avg_pos[None, :], (num_agents, 1))

                # Replicate for all adapters: (num_adapters_to_generate, num_agents, 2)
                all_agent_positions = jnp.tile(
                    all_agent_positions_single[None, :, :],
                    (num_adapters_to_generate, 1, 1),
                )

            # Pad or trim to hn_max_agents
            current_na = all_agent_positions.shape[1]
            if current_na < hn_max_agents:
                padding = jnp.zeros(
                    (
                        num_adapters_to_generate,
                        hn_max_agents - current_na,
                        agent_position_dim,
                    )
                )
                agent_position_for_adapters = jnp.concatenate(
                    [all_agent_positions, padding], axis=1
                )
            elif current_na > hn_max_agents:
                agent_position_for_adapters = all_agent_positions[:, :hn_max_agents, :]
            else:
                agent_position_for_adapters = all_agent_positions
        else:
            # Fallback: use zeros if no agent positions available
            agent_position_for_adapters = jnp.zeros(
                (num_adapters_to_generate, hn_max_agents, agent_position_dim)
            )
    else:
        agent_position_for_adapters = None

    # Create dummy target_snd vectors for all adapters
    target_snd_dim = hypernetwork.target_snd_dim
    if target_snd_dim > 0:
        target_snd_for_adapters = jnp.zeros(
            (num_adapters_to_generate, hn_max_agents, target_snd_dim)
        )
    else:
        target_snd_for_adapters = None

    # Create dummy env_context vectors for all adapters
    env_context_dim = hypernetwork.env_context_dim
    if env_context_dim > 0:
        env_context_for_adapters = jnp.zeros(
            (num_adapters_to_generate, hn_max_agents, env_context_dim)
        )
    else:
        env_context_for_adapters = None

    # Generate adapters - one per context
    adapters_dict_all_agents = hypernetwork.apply(
        {"params": hn_params},
        task_for_adapters,
        context_for_adapters,
        lidar_for_adapters,
        food_for_adapters,  # Real food positions from observations or zeros
        agent_position_for_adapters,  # Agent absolute positions or None
        target_snd_for_adapters,  # Dummy target SND for SND calculation
        env_context_for_adapters,  # Dummy env context for SND calculation
        mask=None,
        diversity_scaling=diversity_scaling,  # Use provided scaling (1.0 for intrinsic, >1.0 for scaled)
    )

    # Extract the correct agent's adapters from each batch entry
    # Hypernetwork flattens output: (num_adapters_to_generate * hn_max_agents, ...)
    # We need to:
    # 1. Reshape back to (num_adapters_to_generate, hn_max_agents, ...)
    # 2. Select the agent_idx-th adapter from each batch entry
    # Result should have shape (num_adapters_to_generate, ...)
    adapters_dict = {}
    for key, value in adapters_dict_all_agents.items():
        # value shape: (num_adapters_to_generate * hn_max_agents, ...)
        # Reshape to: (num_adapters_to_generate, hn_max_agents, ...)
        value_shape = value.shape
        value_reshaped = value.reshape(
            num_adapters_to_generate, hn_max_agents, *value_shape[1:]
        )

        # Use vmap to extract the correct agent's adapter from each batch entry
        def extract_agent_adapter(adapters_for_env, agent_idx):
            return adapters_for_env[agent_idx]

        # Stack agent indices to match batch dimension
        agent_indices = jnp.array(agent_idx_list)  # (num_adapters_to_generate,)

        # Extract adapters using vmap
        extracted_adapters = jax.vmap(extract_agent_adapter)(
            value_reshaped, agent_indices
        )
        adapters_dict[key] = extracted_adapters  # (num_adapters_to_generate, ...)

    # Number of adapters to evaluate (from all agents across sampled environments)
    num_adapters = num_adapters_to_generate

    # Number of observations to evaluate on (may be less if trajectory is short)
    num_eval_obs = obs_batch.shape[0]

    # Tile observations for all adapters
    # Each adapter evaluates on the same set of observations
    # Shape: (num_adapters * num_eval_obs, obs_dim)
    obs_tiled = jnp.tile(obs_batch, (num_adapters, 1))

    # Tile adapters to match the observation batch
    # Each adapter should be repeated num_eval_obs times
    # adapters_dict has tensors with shape (num_adapters, ...)
    # We need shape (num_adapters * num_eval_obs, ...)
    adapters_dict_tiled = {}
    for key, value in adapters_dict.items():
        # value shape: (num_adapters, ...)
        # Repeat each adapter num_eval_obs times along batch dimension
        # Use repeat to get: [adapter_0, adapter_0, ..., adapter_1, adapter_1, ...]
        value_repeated = jnp.repeat(
            value, num_eval_obs, axis=0
        )  # (num_adapters * num_eval_obs, ...)
        adapters_dict_tiled[key] = value_repeated

    # Get policy outputs for each adapter evaluated on all observations
    if use_gru_policy:
        # GRU policy requires hidden state and proper input format
        batch_size = num_adapters * num_eval_obs

        # CRITICAL FIX: Use saved hidden states instead of zeros!
        # Stack hidden states from sampled entries: (num_adapters, hidden_dim)
        if hidden_state_list[0] is not None:
            hidden_states_stacked = jnp.stack(hidden_state_list, axis=0)
            # Tile each hidden state to match num_eval_obs
            # We want: [h0, h0, ..., h1, h1, ..., h_n, h_n, ...]
            # Shape: (num_adapters * num_eval_obs, hidden_dim)
            init_hidden = jnp.repeat(hidden_states_stacked, num_eval_obs, axis=0)

            # Verification: Check that hidden states are not all zeros
            hidden_norms = jnp.linalg.norm(hidden_states_stacked, axis=1)
            num_nonzero = jnp.sum(hidden_norms > 1e-6)
            # Note: Initial queries will have zero hidden states, requeries should have non-zero
            # So we expect a mix of zero and non-zero states
        else:
            # Fallback: use zeros if no hidden states saved (shouldn't happen)
            init_hidden = policy_model.initialize_carry(
                batch_size, policy_model.gru_hidden_dim
            )

        # Format: (obs, dones, avail_actions) - add time dimension
        obs_seq = obs_tiled[None, ...]  # (1, batch, obs_dim)
        dones_seq = jnp.zeros((1, batch_size), dtype=bool)
        avail_seq = None  # Continuous actions
        policy_x = (obs_seq, dones_seq, avail_seq)

        _, output = policy_model.apply(
            {"params": policy_params},
            init_hidden,
            policy_x,
            adapters_dict_tiled,
        )
        # For continuous actions, output is (mean_seq, log_std_seq)
        mean_seq, log_std_seq = output
        action_means = mean_seq[0]  # Remove time dim: (batch, action_dim)
        action_log_stds = log_std_seq[0]  # Remove time dim: (batch, action_dim)
    else:
        # MLP policy: direct call
        action_means, action_log_stds = policy_model.apply(
            {"params": policy_params},
            obs_tiled,
            adapters_dict_tiled,
        )

    # Reshape action means and stds to separate adapters and observations
    # Shape: (num_adapters * num_eval_obs, action_dim) -> (num_adapters, num_eval_obs, action_dim)
    action_dim = action_means.shape[-1]
    means_reshaped = action_means.reshape(num_adapters, num_eval_obs, action_dim)
    log_stds_reshaped = action_log_stds.reshape(num_adapters, num_eval_obs, action_dim)
    stds_reshaped = jnp.exp(log_stds_reshaped)

    # Compute SND using helper function
    snd = compute_snd_from_action_means(means_reshaped, stds_reshaped)

    if return_action_outputs:
        return snd, means_reshaped  # (num_adapters, num_eval_obs, action_dim)
    else:
        return snd


def compute_snd_from_action_means(action_means, action_stds=None, just_mean=True):
    """
    Compute SND (average pairwise Wasserstein distance) from action means.

    Uses JAX vmap for parallel computation of all pairwise distances.

    Args:
        action_means: Array of shape (num_adapters, num_eval_obs, action_dim)
                     Action mean outputs for each adapter on each observation
        action_stds: Optional array of same shape with standard deviations
        just_mean: If True, only use means for distance (ignore stds)

    Returns:
        float: Average pairwise Wasserstein distance (SND)
    """
    num_adapters = action_means.shape[0]

    # Handle case with 0 or 1 adapters
    if num_adapters < 2:
        return 0.0

    # Prepare stds (use dummy values if not provided)
    if action_stds is None:
        action_stds = jnp.ones_like(action_means) * 1e-8

    # Generate all pairs (i, j) where i < j using triu_indices (JIT-compatible)
    i_pairs, j_pairs = jnp.triu_indices(num_adapters, k=1)

    # Vectorized function to compute distance for a single pair
    def compute_pair_distance(i, j):
        """Compute Wasserstein distance between adapters i and j."""
        dist = wasserstein_distance_gaussian(
            action_means[i],  # (num_eval_obs, action_dim)
            action_stds[i],
            action_means[j],
            action_stds[j],
            just_mean=just_mean,
        )
        # dist has shape (num_eval_obs,) - distance for each observation
        # Average over observations to get single distance for this adapter pair
        return jnp.mean(dist)

    # Use vmap to compute all pairwise distances in parallel
    # vmap over both i and j simultaneously
    pair_distances = jax.vmap(compute_pair_distance)(i_pairs, j_pairs)

    # Average over all adapter pairs
    snd = float(jnp.mean(pair_distances))
    return snd


def plot_adapter_impact_distribution(
    adapter_effect_buffer, episode, log_dir, use_wandb
):
    """
    Plot the distribution of adapter impacts over an episode.

    This creates histograms showing how the adapters affect each action dimension,
    including both positive and negative impacts (not just the norm).

    Args:
        adapter_effect_buffer: List of JAX arrays, each of shape (batch_size, action_dim)
                              containing raw adapter impacts (mean_with_adapters - mean_without_adapters)
        episode: Current episode number
        log_dir: Directory to save plots
        use_wandb: Whether to log to wandb
    """
    # Concatenate all impacts across the episode
    # Shape: (total_timesteps * batch_size, action_dim)
    all_impacts = jnp.concatenate(adapter_effect_buffer, axis=0)

    # Convert to numpy for plotting
    all_impacts_np = np.array(all_impacts)

    # Get dimensions
    num_samples, action_dim = all_impacts_np.shape

    # Create figure with subplots for each action dimension
    fig, axes = plt.subplots(1, action_dim, figsize=(5 * action_dim, 4))

    # Handle single action dimension case
    if action_dim == 1:
        axes = [axes]

    # Plot histogram for each action dimension
    for i, ax in enumerate(axes):
        impacts = all_impacts_np[:, i]

        # Create histogram
        ax.hist(impacts, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero impact")
        ax.set_xlabel(f"Adapter Impact (Action {i})", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(
            f"Action Dim {i}\nMean: {np.mean(impacts):.4f}, Std: {np.std(impacts):.4f}",
            fontsize=10,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Adapter Impact Distribution - Episode {episode}\n({num_samples:,} samples)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save to file
    if log_dir is not None:
        plot_path = log_dir / f"adapter_impact_dist_ep{episode}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved adapter impact distribution plot to {plot_path}")

    # Log to wandb if enabled
    if use_wandb:
        try:
            import wandb

            wandb.log(
                {"adapter_impact_distribution": wandb.Image(fig)},
                step=episode,
            )
        except Exception as e:
            print(f"Warning: Failed to log adapter impact plot to wandb: {e}")

    plt.close(fig)


def plot_action_distribution(
    action_buffer, episode, log_dir, use_wandb, plot_type="combined"
):
    """
    Plot the distribution of action values over an episode.

    This creates histograms showing the distribution of action values for each dimension.
    Can be used for backbone-only actions or combined (backbone + adapters) actions.

    Args:
        action_buffer: List of JAX arrays, each of shape (batch_size, action_dim)
                      containing action mean values
        episode: Current episode number
        log_dir: Directory to save plots
        use_wandb: Whether to log to wandb
        plot_type: Type of plot - "backbone_only" or "combined"
    """
    # Concatenate all actions across the episode
    # Shape: (total_timesteps * batch_size, action_dim)
    all_actions = jnp.concatenate(action_buffer, axis=0)

    # Convert to numpy for plotting
    all_actions_np = np.array(all_actions)

    # Get dimensions
    num_samples, action_dim = all_actions_np.shape

    # Create figure with subplots for each action dimension
    fig, axes = plt.subplots(1, action_dim, figsize=(5 * action_dim, 4))

    # Handle single action dimension case
    if action_dim == 1:
        axes = [axes]

    # Set title based on plot type
    if plot_type == "backbone_only":
        title_text = "Backbone-Only Action Distribution"
        wandb_key = "action_distribution/backbone_only"
        file_suffix = "backbone"
    else:
        title_text = "Combined (Backbone + Adapters) Action Distribution"
        wandb_key = "action_distribution/combined"
        file_suffix = "combined"

    # Plot histogram for each action dimension
    for i, ax in enumerate(axes):
        actions = all_actions_np[:, i]

        # Create histogram
        ax.hist(actions, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
        ax.set_xlabel(f"Action Value (Dim {i})", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(
            f"Action Dim {i}\nMean: {np.mean(actions):.4f}, Std: {np.std(actions):.4f}",
            fontsize=10,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{title_text} - Episode {episode}\n({num_samples:,} samples)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save to file
    if log_dir is not None:
        plot_path = log_dir / f"action_dist_{file_suffix}_ep{episode}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved {plot_type} action distribution plot to {plot_path}")

    # Log to wandb if enabled
    if use_wandb:
        try:
            import wandb

            wandb.log(
                {wandb_key: wandb.Image(fig)},
                step=episode,
            )
        except Exception as e:
            print(
                f"Warning: Failed to log {plot_type} action distribution plot to wandb: {e}"
            )

    plt.close(fig)
