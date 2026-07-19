import jax.numpy as jnp
from flax import linen as nn


class TransformerEncoderBlock(nn.Module):
    """Pre-Norm Transformer encoder block (Critical for RL Stability)."""

    num_heads: int
    dim: int
    mlp_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, train: bool = False):
        # --- SUB-LAYER 1: ATTENTION ---
        # 1. Pre-Norm: Normalize BEFORE the operation
        y = nn.LayerNorm()(x)

        # 2. Attention: Pass the mask here!
        # If mask is provided, it blocks interaction with padded agents.
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
        )(y, y, mask=mask)

        # 3. Residual: Add to the original stream
        x = x + attn_output

        # --- SUB-LAYER 2: MLP ---
        # 1. Pre-Norm
        y = nn.LayerNorm()(x)

        # 2. Feed-Forward
        mlp_output = nn.Dense(self.mlp_dim)(y)
        mlp_output = nn.gelu(mlp_output)
        mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(
            mlp_output
        )
        mlp_output = nn.Dense(self.dim)(mlp_output)
        mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(
            mlp_output
        )

        # 3. Residual
        x = x + mlp_output

        return x


class Hypernetwork(nn.Module):
    """
    A Flax hypernetwork module that generates LoRA adapter matrices using a
    Transformer context encoder, as described in the HyperVLA paper.

    The hypernetwork processes task and capability vectors through a Transformer
    to produce context-aware LoRA adapters for the policy.

    Supports gradual complexity increase via lora_mode:
    - 'final_only': Only generate adapters for final output layer
    - 'last_hidden': Generate adapters for last hidden + output layer
    - 'all': Generate adapters for all layers (full model)
    """

    policy_dims: dict  # Contains obs_dim, hidden_dims (list), action_dim, lora_rank
    context_dim: int  # Dimension of agent's capability vector
    task_embed_dim: int  # Dimension of task embedding
    lidar_dim: int = 0  # Dimension of initial lidar readings (0 if not used)
    food_position_dim: int = 0  # Dimension of food position vectors (0 if not used)
    agent_position_dim: int = 0  # Dimension of agent position vectors (0 if not used)
    target_snd_dim: int = (
        0  # Dimension of target SND input (0 if not used, typically 1)
    )
    env_context_dim: int = (
        0  # Dimension of environment context (e.g., package properties, 0 if not used)
    )
    max_agents: int = 10  # Maximum number of agents (for positional embeddings)
    transformer_dim: int = 256  # Internal dimension of Transformer
    transformer_heads: int = 4  # Number of attention heads
    transformer_layers: int = 2  # Number of encoder layers
    lora_mode: str = "final_only"  # 'final_only', 'last_hidden', or 'all'
    scaling_factor: float = 1.0  # Static scaling factor for LoRA adapters
    use_cross_agent_attention: bool = True  # Enable cross-agent attention

    @nn.compact
    def __call__(
        self,
        task_vectors=None,
        capability_vectors=None,
        lidar_vectors=None,
        food_position_vectors=None,
        agent_position_vectors=None,
        target_snd_vectors=None,
        env_context_vectors=None,
        mask=None,
        diversity_scaling=1.0,
    ):
        """
        Generate LoRA adapter matrices using cross-agent attention.

        Args:
            task_vectors: Optional task embeddings (num_envs, num_agents, task_embed_dim).
                          Pass None when task_embed_dim == 0.
            capability_vectors: Optional agent capability vectors (num_envs, num_agents, context_dim).
                                 Pass None when context_dim == 0.
            lidar_vectors: Optional initial lidar readings (num_envs, num_agents, lidar_dim)
            food_position_vectors: Optional relative position to matching food (num_envs, num_agents, food_position_dim)
            target_snd_vectors: Optional target SND values (num_envs, num_agents, target_snd_dim)
            env_context_vectors: Optional environment context (e.g., package properties) (num_envs, num_agents, env_context_dim)
            mask: Optional attention mask for dynamic agent counts (num_envs, 1, num_agents, num_agents)
            diversity_scaling: Additional scaling for final layer only (diversity control, default: 1.0)

        Returns:
            Dictionary containing reshaped adapter matrices for all agents:
                {'A1', 'B1', 'A2', 'B2', ..., 'An', 'Bn'}
                Each with shape (num_envs * num_agents, ...)
        """
        # Derive num_envs and num_agents from the first non-None input
        _ref = next(
            v
            for v in [
                task_vectors,
                capability_vectors,
                env_context_vectors,
                lidar_vectors,
                food_position_vectors,
                agent_position_vectors,
                target_snd_vectors,
            ]
            if v is not None
        )
        num_envs = _ref.shape[0]
        num_agents = _ref.shape[1]

        # Build agent feature vectors by concatenating all inputs
        # Each agent becomes one token in the sequence
        agent_features = []

        # Add capability features if context dimension is > 0
        if self.context_dim > 0:
            agent_features.append(capability_vectors)

        # Add lidar features if provided
        if lidar_vectors is not None and self.lidar_dim > 0:
            if len(lidar_vectors.shape) != 3:
                raise ValueError(
                    f"lidar_vectors must have 3 dimensions (num_envs, num_agents, lidar_dim), "
                    f"got shape {lidar_vectors.shape}"
                )
            agent_features.append(lidar_vectors)

        # Add food position features if provided
        if food_position_vectors is not None and self.food_position_dim > 0:
            if len(food_position_vectors.shape) != 3:
                raise ValueError(
                    f"food_position_vectors must have 3 dimensions (num_envs, num_agents, food_position_dim), "
                    f"got shape {food_position_vectors.shape}"
                )
            agent_features.append(food_position_vectors)

        # Add agent position features if provided
        if agent_position_vectors is not None and self.agent_position_dim > 0:
            if len(agent_position_vectors.shape) != 3:
                raise ValueError(
                    f"agent_position_vectors must have 3 dimensions (num_envs, num_agents, agent_position_dim), "
                    f"got shape {agent_position_vectors.shape}"
                )
            agent_features.append(agent_position_vectors)

        # Add target SND features if provided
        if target_snd_vectors is not None and self.target_snd_dim > 0:
            if len(target_snd_vectors.shape) != 3:
                raise ValueError(
                    f"target_snd_vectors must have 3 dimensions (num_envs, num_agents, target_snd_dim), "
                    f"got shape {target_snd_vectors.shape}"
                )
            agent_features.append(target_snd_vectors)
        # Add environment context features if provided (broadcast to all agents)
        if env_context_vectors is not None and self.env_context_dim > 0:
            if len(env_context_vectors.shape) != 3:
                raise ValueError(
                    f"env_context_vectors must have 3 dimensions (num_envs, num_agents, env_context_dim), "
                    f"got shape {env_context_vectors.shape}"
                )
            agent_features.append(env_context_vectors)
        # Add task features if task_embed_dim > 0
        if self.task_embed_dim > 0:
            agent_features.append(task_vectors)

        # Concatenate all features: (num_envs, num_agents, total_feature_dim)
        if agent_features:
            combined_features = jnp.concatenate(agent_features, axis=-1)
        else:
            # Fallback: use learnable embeddings if no features provided
            combined_features = jnp.zeros((num_envs, num_agents, 1))

        # Project to transformer dimension
        # Shape: (num_envs, num_agents, transformer_dim)
        agent_tokens = nn.Dense(
            self.transformer_dim,
            kernel_init=nn.initializers.lecun_normal(),
            name="agent_embed",
        )(combined_features)

        # Normalize input sequence (no positional encodings for permutation invariance)
        input_seq = nn.LayerNorm(name="input_layer_norm")(agent_tokens)

        # Create attention mask based on cross_agent_attention setting
        if not self.use_cross_agent_attention:
            # Create diagonal mask: each agent only attends to itself
            # Shape: (num_envs, 1, num_agents, num_agents)
            diagonal_mask = jnp.eye(num_agents, dtype=bool)
            diagonal_mask = jnp.broadcast_to(
                diagonal_mask[None, None, :, :], (num_envs, 1, num_agents, num_agents)
            )
            # Combine with provided mask (for dynamic agent counts) if present
            attention_mask = diagonal_mask if mask is None else (diagonal_mask & mask)
        else:
            # Use provided mask (allows cross-agent attention)
            attention_mask = mask

        # Pass through Transformer encoder blocks
        # With cross_agent_attention=True: agents attend to all other agents
        # With cross_agent_attention=False: each agent only attends to its own features
        x = input_seq
        for i in range(self.transformer_layers):
            x = TransformerEncoderBlock(
                num_heads=self.transformer_heads,
                dim=self.transformer_dim,
                mlp_dim=self.transformer_dim * 4,
                dropout_rate=0.0,
                name=f"transformer_block_{i}",
            )(x, mask=attention_mask, train=False)

        # Final LayerNorm (Standard practice for Pre-Norm architectures)
        x = nn.LayerNorm(name="final_layer_norm")(x)

        # Extract outputs for all agents and flatten to (num_envs * num_agents, transformer_dim)
        batch_size = num_envs * num_agents
        context_output = x.reshape(batch_size, self.transformer_dim)

        # Determine which adapters to create based on lora_mode
        hidden_dims_list = list(self.policy_dims["hidden_dims"])
        obs_dim = self.policy_dims["obs_dim"]
        action_dim = self.policy_dims["action_dim"]
        lora_rank = self.policy_dims["lora_rank"]
        num_hidden_layers = len(hidden_dims_list)

        # Check if using GRU policy (indicated by gru_hidden_dim in policy_dims)
        is_gru_policy = "gru_hidden_dim" in self.policy_dims
        if is_gru_policy:
            gru_hidden_dim = self.policy_dims["gru_hidden_dim"]

        adapter_configs = []

        if is_gru_policy:
            # For GRU policy: only create adapter for final layer (GRU hidden -> action)
            # The GRU itself is shared (not adapted), only the output projection is adapted
            final_idx = 1
            adapter_configs.append((final_idx, lora_rank, gru_hidden_dim, action_dim))
        else:
            # For MLP policy: create adapters based on lora_mode
            input_dim = obs_dim
            for i, output_dim in enumerate(hidden_dims_list):
                layer_idx = i + 1
                create_adapter = self.lora_mode == "all" or (
                    self.lora_mode == "last_hidden" and i == num_hidden_layers - 1
                )
                if create_adapter:
                    adapter_configs.append(
                        (layer_idx, lora_rank, input_dim, output_dim)
                    )
                input_dim = output_dim

            # Always add final layer
            final_idx = len(hidden_dims_list) + 1
            adapter_configs.append(
                (final_idx, lora_rank, hidden_dims_list[-1], action_dim)
            )

        # Generate adapters
        adapters = {}

        for layer_idx, lora_rank, input_dim, output_dim in adapter_configs:
            A_key = f"A{layer_idx}"
            B_key = f"B{layer_idx}"

            # Determine if this is the final layer
            is_final_layer = layer_idx == final_idx
            # Apply diversity_scaling only to final layer, self.scaling_factor to all layers
            layer_scaling = self.scaling_factor * (
                diversity_scaling if is_final_layer else 1.0
            )

            # Generate A adapter
            A_flat = nn.Dense(
                features=lora_rank * input_dim,
                kernel_init=nn.initializers.normal(stddev=0.001),
                bias_init=nn.initializers.zeros,
                name=A_key,
            )(context_output)
            A = A_flat.reshape(batch_size, lora_rank, input_dim)
            A = A * layer_scaling
            # Clip adapters after scaling to prevent extreme values with high diversity_scaling
            # This prevents numerical overflow in policy forward pass
            A = jnp.clip(A, -1000.0, 1000.0)
            A = jnp.nan_to_num(A, nan=0.0, posinf=1000.0, neginf=-1000.0)
            adapters[A_key] = A

            # Generate B adapter
            B_flat = nn.Dense(
                features=output_dim * lora_rank,
                kernel_init=nn.initializers.normal(stddev=0.001),
                bias_init=nn.initializers.zeros,
                name=B_key,
            )(context_output)
            B = B_flat.reshape(batch_size, output_dim, lora_rank)
            B = B * layer_scaling
            # Clip adapters after scaling to prevent extreme values
            B = jnp.clip(B, -1000.0, 1000.0)
            B = jnp.nan_to_num(B, nan=0.0, posinf=1000.0, neginf=-1000.0)
            adapters[B_key] = B

        return adapters
