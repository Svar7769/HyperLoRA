"""
DICO Policy Implementation

Based on "Diversity-Inducing Cooperative Multi-Agent Reinforcement Learning" (Bettini et al. 2024)

The DICO policy architecture consists of two parts:
1. Homogeneous policy φ_homo: Shared MLP across all agents (same parameters)
2. Heterogeneous policy φ_hetero_i: Per-agent MLPs (separate parameters for each agent)

The final policy output combines both:
    π_i(a|o) = φ_homo(o) + λ * φ_hetero_i(o)

where λ is the diversity scaling factor controlled by the diversity control mechanism.

Key insight: Unlike hypernetwork approaches, DICO uses simple per-agent networks that are
directly trained. This is simpler and follows the original paper implementation.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import distrax
from jax.nn import initializers
from gat_layers import DeepSetsEncoder


class HeteroDense(nn.Module):
    """
    XLA-Optimized HeteroDense layer.
    Uses one-hot matrix multiplications to dynamically construct per-agent weights,
    completely avoiding the slow scatter/gather operations in JAX's backward pass.
    """

    num_agents: int
    features: int

    @nn.compact
    def __call__(self, x, agent_ids):
        batch_size, in_features = x.shape

        # 1. Base parameters for all agents
        kernel = self.param(
            "kernel",
            initializers.xavier_uniform(),
            (self.num_agents, in_features, self.features),
        )
        bias = self.param("bias", initializers.zeros, (self.num_agents, self.features))

        # 2. Flatten the parameters so we can extract them via matmul
        kernel_flat = kernel.reshape(self.num_agents, -1)

        # 3. Convert agent_ids to one-hot encoding (Shape: batch_size, num_agents)
        one_hot = jax.nn.one_hot(agent_ids, self.num_agents)

        # 4. Extract weights using PURE DENSE MATMUL!
        # This completely bypasses XLA's scatter_add backward pass trap
        batch_kernel_flat = jnp.dot(one_hot, kernel_flat)
        batch_bias = jnp.dot(one_hot, bias)

        # 5. Reshape back into batched matrices (batch_size, in_features, features)
        batch_kernel = batch_kernel_flat.reshape(batch_size, in_features, self.features)

        # 6. Apply the batched matmul (Exact same blazing fast einsum as HyperLoRA)
        y = jnp.einsum("bi,bio->bo", x, batch_kernel) + batch_bias
        return y


class HeteroNetworks(nn.Module):
    num_agents: int
    hidden_dims: tuple
    action_dim: int

    @nn.compact
    def __call__(self, x, agent_ids):
        h = x
        for dim in self.hidden_dims:
            h = HeteroDense(num_agents=self.num_agents, features=dim)(h, agent_ids)
            h = nn.tanh(h)

        mean = HeteroDense(num_agents=self.num_agents, features=self.action_dim)(
            h, agent_ids
        )
        return mean


class DiCoPolicy(nn.Module):
    """
    DICO Policy with homogeneous and heterogeneous components.

    Architecture (following Bettini et al. 2024):
    - Homogeneous part φ_homo: Single MLP shared across all agents
    - Heterogeneous part φ_hetero_i: Per-agent MLPs (one for each agent)
    - Combination: π_i(a|o) = φ_homo(o) + λ * φ_hetero_i(o)

    The diversity scaling λ is dynamically adjusted based on System Neural Diversity (SND).

    Note: This implementation creates per-agent heterogeneous networks. Each agent
    has its own set of parameters for the heterogeneous component.
    """

    num_agents: int  # Number of agents (needed for per-agent networks)
    hidden_dims: tuple = (256, 256)  # Hidden layer dimensions (Bettini: num_cells=[256,256])
    action_dim: int = 2

    def setup(self):
        # Use Xavier/Glorot initialization for better gradient flow
        hidden_init = initializers.xavier_uniform()

        # ================================================================
        # Homogeneous network (shared across all agents)
        # ================================================================
        self.homo_hidden_layers = [
            nn.Dense(features=dim, name=f"homo_hidden_{i}", kernel_init=hidden_init)
            for i, dim in enumerate(self.hidden_dims)
        ]
        self.homo_mean_layer = nn.Dense(
            features=self.action_dim, name="homo_mean", kernel_init=hidden_init
        )

        # ================================================================
        # Heterogeneous networks (per-agent, EFFICIENT selective evaluation)
        # ================================================================
        # Use custom HeteroNetworks module that evaluates only the needed agent
        self.hetero_network = HeteroNetworks(
            num_agents=self.num_agents,
            hidden_dims=self.hidden_dims,
            action_dim=self.action_dim,
        )

        # ================================================================
        # Shared scale layer (common across both components)
        # Output is raw values; std = softplus(raw + 1.0) (biased_softplus_1.0)
        # ================================================================
        self.scale_layer = nn.Dense(
            features=self.action_dim, name="scale", kernel_init=hidden_init
        )

    def __call__(self, x, agent_ids, diversity_scaling=1.0):
        """
        Forward pass combining homogeneous and heterogeneous outputs.

        Args:
            x: Input observation (batch, obs_dim)
            agent_ids: Agent IDs for each observation (batch,) - integers 0 to num_agents-1
            diversity_scaling: Scaling factor λ for heterogeneous output

        Returns:
            mean: Action mean (batch, action_dim)
            log_std: Action log std (batch, action_dim)
        """
        # Protect against NaN diversity_scaling
        diversity_scaling_safe = jnp.where(
            jnp.isnan(diversity_scaling) | jnp.isinf(diversity_scaling),
            1.0,
            diversity_scaling,
        )
        diversity_scaling_safe = jnp.clip(diversity_scaling_safe, 0.0, 1000.0)

        # ================================================================
        # Homogeneous Policy (shared parameters)
        # ================================================================
        h_homo = x
        for layer in self.homo_hidden_layers:
            h_homo = nn.tanh(layer(h_homo))

        mean_homo = self.homo_mean_layer(h_homo)
        # process_shared: squash homo output to [-1, 1] via tanh (matching Bettini's tanh_squash)
        mean_homo = nn.tanh(mean_homo)

        # ================================================================
        # Heterogeneous Policy (per-agent parameters)
        # ================================================================
        # Shape: (batch_size, action_dim)
        mean_hetero = self.hetero_network(x, agent_ids)

        # Combine: π_i(o) = tanh(φ_homo(o)) + λ * φ_hetero_i(o)
        mean = mean_homo + diversity_scaling_safe * mean_hetero

        # Std via biased_softplus_1.0: std = softplus(raw + 1.0) = log(1 + exp(raw + 1.0))
        raw_scale = self.scale_layer(h_homo)
        std = jax.nn.softplus(raw_scale + 1.0)
        # FIX: Enforce both a minimum and a MAXIMUM std
        # A max std of 1.0 for tighter control and gradient stability
        std = jnp.clip(std, 0.1, 1.0)
        log_std = jnp.log(std)

        return mean, log_std

    def get_action_and_log_prob(self, x, agent_ids, diversity_scaling, rng_key=None):
        """
        Get action and its log probability for PPO.
        Uses a Tanh-transformed distribution (TanhNormal).

        Args:
            x: Input observation (batch, obs_dim)
            agent_ids: Agent IDs for each observation (batch,)
            diversity_scaling: Scaling factor for heterogeneous component
            rng_key: JAX random key for sampling

        Returns:
            action: Sampled action (bounded to [-1, 1])
            log_prob: Log probability of the action
            mean: Mean of the pre-tanh distribution
            std: Std of the pre-tanh distribution
        """
        mean, log_std = self(x, agent_ids, diversity_scaling)
        std = jnp.exp(log_std)

        # Create tanh-transformed distribution (TanhNormal)
        base_dist = distrax.Normal(mean, std)
        tanh_bijector = distrax.Tanh()
        dist = distrax.Transformed(base_dist, tanh_bijector)

        if rng_key is None:
            action = jnp.tanh(mean)
        else:
            action = dist.sample(seed=rng_key)

        # Clip to avoid ±1.0 boundary issues with tanh inverse
        action = jnp.clip(action, -0.9999, 0.9999)

        log_prob = dist.log_prob(action).sum(axis=-1)
        log_prob = jnp.where(
            jnp.isnan(log_prob) | jnp.isinf(log_prob), -100.0, log_prob
        )
        log_prob = jnp.clip(log_prob, -100.0, 100.0)

        return action, log_prob, mean, std


class DiCoHomogeneousPolicy(nn.Module):
    """
    DICO policy without diversity control (fixed scaling).

    This is the full DICO architecture:
    - Homogeneous policy φ_homo (shared parameters)
    - Heterogeneous policies φ_hetero_i (per-agent parameters)
    - Combination: π_i(o) = φ_homo(o) + λ * φ_hetero_i(o)

    But WITHOUT adaptive diversity control - uses a fixed λ value.
    This helps isolate the performance impact of the diversity control mechanism.
    """

    num_agents: int
    hidden_dims: tuple = (256, 256)
    action_dim: int = 2

    def setup(self):
        hidden_init = initializers.xavier_uniform()

        # Homogeneous network (shared across all agents)
        self.homo_hidden_layers = [
            nn.Dense(features=dim, name=f"homo_hidden_{i}", kernel_init=hidden_init)
            for i, dim in enumerate(self.hidden_dims)
        ]
        self.homo_mean_layer = nn.Dense(
            features=self.action_dim, name="homo_mean", kernel_init=hidden_init
        )

        # Heterogeneous networks (per-agent, using Data Routing trick)
        self.hetero_network = HeteroNetworks(
            num_agents=self.num_agents,
            hidden_dims=self.hidden_dims,
            action_dim=self.action_dim,
        )

        # Shared scale layer (biased_softplus_1.0)
        self.scale_layer = nn.Dense(
            features=self.action_dim, name="scale", kernel_init=hidden_init
        )

    def __call__(self, x, agent_ids, diversity_scaling=1.0):
        """
        Forward pass combining homogeneous and heterogeneous outputs.

        Args:
            x: Input observation (batch, obs_dim)
            agent_ids: Agent IDs (batch,)
            diversity_scaling: Fixed λ value (not adaptive)

        Returns:
            mean: Action mean (batch, action_dim)
            log_std: Action log std (batch, action_dim)
        """
        # Clamp inputs
        x = jnp.clip(x, -10.0, 10.0)
        x = jnp.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

        diversity_scaling_safe = jnp.clip(diversity_scaling, 0.0, 1000.0)

        # Homogeneous policy
        h_homo = x
        for layer in self.homo_hidden_layers:
            h_homo = nn.tanh(layer(h_homo))

        mean_homo = self.homo_mean_layer(h_homo)
        # process_shared: squash homo output via tanh
        mean_homo = nn.tanh(mean_homo)

        # Heterogeneous policy (per-agent networks with Data Routing)
        mean_hetero = self.hetero_network(x, agent_ids)

        # Combine: π_i(o) = tanh(φ_homo(o)) + λ * φ_hetero_i(o)
        mean = mean_homo + diversity_scaling_safe * mean_hetero

        # Std via biased_softplus_1.0
        raw_scale = self.scale_layer(h_homo)
        std = jax.nn.softplus(raw_scale + 1.0)
        # FIX: Enforce both a minimum and a MAXIMUM std
        # A max std of around 2.0 to e^2 (7.38) is standard in continuous control
        std = jnp.clip(std, 0.1, 1.0)
        log_std = jnp.log(std)

        return mean, log_std

    def get_action_and_log_prob(self, x, agent_ids, diversity_scaling, rng_key):
        """
        Get action and log probability.

        Uses fixed diversity_scaling (no adaptive control based on SND).
        """
        mean, log_std = self(x, agent_ids, diversity_scaling)
        std = jnp.exp(log_std)

        base_dist = distrax.Normal(mean, std)
        tanh_bijector = distrax.Tanh()
        dist = distrax.Transformed(base_dist, tanh_bijector)

        if rng_key is None:
            action = jnp.tanh(mean)
        else:
            action = dist.sample(seed=rng_key)

        action = jnp.clip(action, -0.9999, 0.9999)

        log_prob = dist.log_prob(action).sum(axis=-1)
        log_prob = jnp.where(
            jnp.isnan(log_prob) | jnp.isinf(log_prob), -100.0, log_prob
        )
        log_prob = jnp.clip(log_prob, -100.0, 100.0)

        return action, log_prob, mean, std


class DiCoDeepSetsPolicy(nn.Module):
    """
    DICO Policy with Deep Sets (GATv2) preprocessing.

    Architecture (following Bettini et al. 2024 for football):
    1. GATv2 layer: Processes agent observations with graph attention (local topology)
       - Outputs per-agent embeddings of dimension 32
    2. DICO layers: Homogeneous + Heterogeneous networks on embeddings
       - Homogeneous: Shared MLP across all agents
       - Heterogeneous: Per-agent MLPs
       - Combination: π_i(a|o) = φ_homo(h_i) + λ * φ_hetero_i(h_i)

    This matches the config_bettini.yaml architecture:
    - l1: gnn_deep_sets (GATv2, topology=from_pos, edge_radius=10) → 32-dim
    - l2: hetcontrolmlpempirical ([256, 256], desired_snd=0.2)

    Attributes:
        num_agents: Number of agents (for per-agent networks)
        hidden_dims: Hidden layer dimensions for DICO MLPs (default: [256, 256])
        action_dim: Action dimension
        gat_out_features: Output dimension of GAT layer (default: 32)
        gat_num_heads: Number of attention heads (default: 1)
        edge_radius: Maximum distance for local graph (default: 10.0)
        log_std_min: Minimum log std
        log_std_max: Maximum log std
        min_std: Minimum std for numerical stability
    """

    num_agents: int
    hidden_dims: tuple = (256, 256)
    action_dim: int = 2
    gat_out_features: int = 32
    edge_radius: float = 10.0

    def setup(self):
        hidden_init = initializers.xavier_uniform()

        # ================================================================
        # GATv2 encoder (Deep Sets layer)
        # ================================================================
        self.gat_encoder = DeepSetsEncoder(
            out_features=self.gat_out_features,
            edge_radius=self.edge_radius,
            aggr="mean",
        )

        # Fallback encoder for initialization (when batch_size < num_agents)
        self.fallback_encoder = nn.Dense(
            features=self.gat_out_features,
            name="fallback_encoder",
            kernel_init=hidden_init
        )

        # ================================================================
        # Homogeneous network (shared, operates on GAT embeddings)
        # ================================================================
        self.homo_hidden_layers = [
            nn.Dense(features=dim, name=f"homo_hidden_{i}", kernel_init=hidden_init)
            for i, dim in enumerate(self.hidden_dims)
        ]
        self.homo_mean_layer = nn.Dense(
            features=self.action_dim, name="homo_mean", kernel_init=hidden_init
        )

        # ================================================================
        # Heterogeneous networks (per-agent, operates on GAT embeddings)
        # ================================================================
        self.hetero_network = HeteroNetworks(
            num_agents=self.num_agents,
            hidden_dims=self.hidden_dims,
            action_dim=self.action_dim,
        )

        # ================================================================
        # Shared scale layer (biased_softplus_1.0)
        # ================================================================
        self.scale_layer = nn.Dense(
            features=self.action_dim, name="scale", kernel_init=hidden_init
        )

    def extract_positions(self, x, num_agents):
        """
        Extract positions from observations for graph construction.

        For football, assumes observations include position features.
        The observation structure should have position as the first 2 features.

        Args:
            x: Observations (batch, obs_dim)
            num_agents: Number of agents

        Returns:
            positions: (batch // num_agents, num_agents, 2)
        """
        batch_size = x.shape[0]
        agents_per_env = num_agents

        # Reshape to (num_envs, num_agents, obs_dim)
        num_envs = batch_size // agents_per_env
        x_reshaped = x.reshape(num_envs, agents_per_env, -1)

        # Extract first 2 features as positions
        positions = x_reshaped[:, :, :2]  # (num_envs, num_agents, 2)

        return positions

    def __call__(self, x, agent_ids, diversity_scaling=1.0, training=False):
        """
        Forward pass with GATv2 preprocessing.

        Args:
            x: Input observation (batch, obs_dim)
            agent_ids: Agent IDs (batch,)
            diversity_scaling: Scaling factor λ for heterogeneous output
            training: Whether in training mode

        Returns:
            mean: Action mean (batch, action_dim)
            log_std: Action log std (batch, action_dim)
        """
        batch_size = x.shape[0]

        # Clamp input observations
        x = jnp.clip(x, -10.0, 10.0)
        x = jnp.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

        diversity_scaling_safe = jnp.clip(diversity_scaling, 0.0, 1000.0)

        # ================================================================
        # Step 1: Extract positions and reshape for GATv2
        # ================================================================
        # Check if batch_size is divisible by num_agents
        if batch_size % self.num_agents == 0:
            # Normal case: batch contains complete environments
            num_envs = batch_size // self.num_agents
            x_reshaped = x.reshape(
                num_envs, self.num_agents, -1
            )  # (num_envs, num_agents, obs_dim)
            positions = self.extract_positions(
                x, self.num_agents
            )  # (num_envs, num_agents, 2)

            # ================================================================
            # Step 2: Apply GATv2 to get per-agent embeddings
            # ================================================================
            # h: (num_envs, num_agents, gat_out_features)
            h = self.gat_encoder(x_reshaped, positions=positions, training=training)

            # Flatten back to (batch, gat_out_features)
            h = h.reshape(batch_size, -1)
        else:
            # Initialization or edge case: batch_size < num_agents
            # Fall back to processing without graph structure
            # Use fallback encoder to get embeddings
            h = self.fallback_encoder(x)
            h = jax.nn.relu(h)

        # ================================================================
        # Step 3: Homogeneous Policy (on GAT embeddings)
        # ================================================================
        h_homo = h
        for layer in self.homo_hidden_layers:
            h_homo = nn.tanh(layer(h_homo))

        mean_homo = self.homo_mean_layer(h_homo)
        # process_shared: squash homo output via tanh
        mean_homo = nn.tanh(mean_homo)

        # ================================================================
        # Step 4: Heterogeneous Policy (on GAT embeddings)
        # ================================================================
        mean_hetero = self.hetero_network(h, agent_ids)

        # Combine: π_i(o) = tanh(φ_homo(o)) + λ * φ_hetero_i(o)
        mean = mean_homo + diversity_scaling_safe * mean_hetero

        # Std via biased_softplus_1.0
        raw_scale = self.scale_layer(h_homo)
        std = jax.nn.softplus(raw_scale + 1.0)
        # FIX: Enforce both a minimum and a MAXIMUM std
        # A max std of around 2.0 to e^2 (7.38) is standard in continuous control
        std = jnp.clip(std, 0.1, 1.0)
        log_std = jnp.log(std)
    
        return mean, log_std

    def get_action_and_log_prob(
        self, x, agent_ids, diversity_scaling, rng_key=None, training=False
    ):
        """
        Get action and log probability with GATv2 preprocessing.

        Args:
            x: Input observation (batch, obs_dim)
            agent_ids: Agent IDs (batch,)
            diversity_scaling: Scaling factor
            rng_key: Random key for sampling
            training: Whether in training mode

        Returns:
            action: Sampled action
            log_prob: Log probability
            mean: Mean of distribution
            std: Std of distribution
        """
        mean, log_std = self(x, agent_ids, diversity_scaling, training=training)
        std = jnp.exp(log_std)

        base_dist = distrax.Normal(mean, std)
        tanh_bijector = distrax.Tanh()
        dist = distrax.Transformed(base_dist, tanh_bijector)

        if rng_key is None:
            action = jnp.tanh(mean)
        else:
            action = dist.sample(seed=rng_key)

        action = jnp.clip(action, -0.9999, 0.9999)

        log_prob = dist.log_prob(action).sum(axis=-1)
        log_prob = jnp.where(
            jnp.isnan(log_prob) | jnp.isinf(log_prob), -100.0, log_prob
        )
        log_prob = jnp.clip(log_prob, -100.0, 100.0)

        return action, log_prob, mean, std
