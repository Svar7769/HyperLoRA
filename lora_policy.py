import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import distrax
import functools
from flax.linen.initializers import constant, orthogonal


class ScannedRNN(nn.Module):
    """
    Scanned RNN module that applies GRU cells across a sequence.

    Handles proper state reset based on episode termination flags,
    maintaining recurrent state across sequences of observations.
    Mirrors the HyperMARL implementation from mappo_rnn_smax_with_eval.py.
    """

    hidden_size: int  # Explicit hidden dimension for GRU

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """
        Apply GRU cell to input sequence.

        Args:
            carry: Recurrent state
            x: Tuple of (input, reset flags)

        Returns:
            New recurrent state and output
        """
        rnn_state = carry
        ins, resets = x
        # Reset state when episode terminates
        # resets is (batch,) from scan, expand to (batch, 1) for broadcasting with (batch, hidden_size)
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], self.hidden_size),
            rnn_state,
        )
        # Improved GRU initialization for better gradient flow
        new_rnn_state, y = nn.GRUCell(
            features=self.hidden_size,
            # Input-to-Hidden: Use Orthogonal gain sqrt(2) for ReLU-like activations
            # This gives the input sufficient influence on the gates
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            # Hidden-to-Hidden: Use Orthogonal gain 1.0 for unitary recurrence
            # This preserves hidden state norm over time, preventing vanishing gradients
            recurrent_kernel_init=nn.initializers.orthogonal(1.0),
            # Bias: Initialize to 0.0 (standard practice)
            bias_init=nn.initializers.constant(0.0),
        )(rnn_state, ins)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """
        Initialize recurrent state.

        Args:
            batch_size: Number of sequences in batch
            hidden_size: Size of hidden state

        Returns:
            Initial recurrent state
        """
        # Use a dummy key since the default state init fn is just zeros
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class GRULoRAPolicy(nn.Module):
    """
    A Flax neural network module with GRU backbone and LoRA adapters at the final layer.
    Matches HyperMARL ActorRNN architecture exactly:

    Architecture (matching HyperMARL's ActorRNN):
    - Initial embedding layer with ReLU activation
    - ScannedRNN processes sequential observations with proper done handling
    - Intermediate dense layer (gru_hidden_dim) with orthogonal(2) initialization
    - ReLU activation
    - Final output layer with LoRA adapter for agent-specific adaptation
    - Supports both continuous and discrete action spaces

    Key differences from previous version:
    - NO LayerNorm (removed for consistency with HyperMARL)
    - Intermediate layer uses orthogonal(2) instead of orthogonal(sqrt(2))
    - Cleaner architecture matching reference implementation
    """

    gru_hidden_dim: int = 64  # Hidden dimension of GRU
    fc_dim_size: int = 64  # Dimension of initial embedding layer
    action_dim: int = 2
    log_std_min: float = -2.0
    log_std_max: float = 0.0
    min_std: float = 0.3
    discrete_actions: bool = False

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """Initialize hidden state for GRU."""
        return ScannedRNN.initialize_carry(batch_size, hidden_size)

    @nn.compact
    def __call__(self, hidden, x, adapters):
        """
        Forward pass matching HyperMARL ActorRNN architecture.

        Args:
            hidden: Initial recurrent state (batch, gru_hidden_dim)
            x: Tuple of (observations, dones, avail_actions)
                - obs: (time, batch, obs_dim) for sequences or (batch, obs_dim) for single step
                - dones: (time, batch) or (batch,)
                - avail_actions: (time, batch, action_dim) or (batch, action_dim) - only used for discrete
            adapters: Dictionary containing LoRA adapter matrices

        Returns:
            hidden: Updated recurrent state (batch, gru_hidden_dim)
            output: Action logits (discrete) or mean (continuous)
        """
        obs, dones, avail_actions = x

        # 1. Initial embedding layer (matching HyperMARL ActorRNN)
        embedding = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        # 2. Apply RNN (handles done resets internally)
        rnn_in = (embedding, dones)
        hidden, rnn_out = ScannedRNN(hidden_size=self.gru_hidden_dim)(hidden, rnn_in)

        # 3. Intermediate actor layer (matching HyperMARL ActorRNN)
        # Reduced gain from 2 to 0.1 to prevent base policy from overwhelming adapter contributions
        actor_features = nn.Dense(
            self.gru_hidden_dim,
            kernel_init=orthogonal(0.1),
            bias_init=constant(0.0),
        )(rnn_out)
        actor_features = nn.relu(actor_features)

        # 4. Check if adapters exist and are non-empty
        use_adapters = len(adapters) > 0
        if use_adapters:
            first_key = list(adapters.keys())[0]
            lora_rank = (
                adapters[first_key].shape[1]
                if len(adapters[first_key].shape) > 1
                else 0
            )
            use_adapters = lora_rank > 0

        # 5. Compute LoRA adapter contribution
        # Both shared and adapter operate on the same input (actor_features)
        A_key = "A1"
        B_key = "B1"

        if use_adapters and A_key in adapters and B_key in adapters:
            # actor_features: (time, batch, gru_hidden_dim)
            # A: (batch, lora_rank, gru_hidden_dim), B: (batch, action_dim, lora_rank)
            # Project Input -> Rank
            adapter_out = jnp.einsum("tbh,brh->tbr", actor_features, adapters[A_key])
            # Project Rank -> Output
            adapter_out = jnp.einsum("tbr,bar->tba", adapter_out, adapters[B_key])
            # Clip adapter contribution - use large threshold to preserve diversity metrics
            adapter_out = jnp.clip(adapter_out, -1000.0, 1000.0)
        else:
            adapter_out = 0.0

        # 6. Generate output with shared weights + adapter
        if self.discrete_actions:
            # Shared transformation (reduced gain from 0.5 to 0.01 to allow adapter influence)
            shared_logits = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(actor_features)

            # Combine shared + adapter
            action_logits = shared_logits + adapter_out

            # Mask unavailable actions
            unavail_actions = 1 - avail_actions
            action_logits = action_logits - (unavail_actions * 1e10)

            return hidden, action_logits
        else:
            # Shared transformation (reduced gain from 0.5 to 0.01 to allow adapter influence)
            shared_mean = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(actor_features)

            # Combine shared + adapter
            mean = shared_mean + adapter_out
            # Clip action mean - use large threshold to preserve diversity metrics
            mean = jnp.clip(mean, -1000.0, 1000.0)

            # Log std (shared across agents, no adaptation)
            log_std = nn.Dense(features=self.action_dim, kernel_init=orthogonal(0.5))(
                actor_features
            )
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
            return hidden, (mean, log_std)

    def get_action_and_log_prob(self, hidden, x, adapters, rng_key=None):
        """
        Get action and its log probability for PPO.
        Updated to match HyperMARL's call signature: (hidden, x, adapters).

        Args:
            hidden: Recurrent hidden state
            x: Tuple of (observations, dones, avail_actions)
            adapters: Dictionary containing LoRA adapter matrices
            rng_key: JAX random key for sampling

        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            hidden: Updated hidden state
            output: Logits (discrete) or mean (continuous)
            std_or_none: Std (continuous) or None (discrete)
        """
        hidden, output1 = self(hidden, x, adapters)

        if self.discrete_actions:
            logits = output1
            dist = distrax.Categorical(logits=logits)

            if rng_key is not None:
                action = dist.sample(seed=rng_key)
                log_prob = dist.log_prob(action)
            else:
                action = jnp.argmax(logits, axis=-1)
                log_prob = None

            return action, log_prob, hidden, logits, None
        else:
            # For continuous actions, output1 is (mean, log_std) tuple
            mean, log_std = output1
            std = jnp.exp(log_std)

            base_dist = distrax.Normal(mean, std)
            tanh_bijector = distrax.Tanh()
            dist = distrax.Transformed(base_dist, tanh_bijector)

            if rng_key is not None:
                action = dist.sample(seed=rng_key)
                action_epsilon = 1e-6
                action = jnp.clip(action, -1.0 + action_epsilon, 1.0 - action_epsilon)
                log_prob = dist.log_prob(action).sum(axis=-1)
                log_prob = jnp.nan_to_num(
                    log_prob, nan=-1e10, posinf=-1e10, neginf=-1e10
                )
            else:
                action = jnp.tanh(mean)
                log_prob = None

            return action, log_prob, hidden, mean, std


class LoRAPolicy(nn.Module):
    """
    A Flax neural network module with LoRA (Low-Rank Adaptation) support.
    The policy uses a configurable list of hidden layer dimensions, with LoRA adapters applied to each layer.

    Supports both continuous and discrete action spaces:
    - Continuous: Outputs mean and log_std for Gaussian distribution with Tanh squashing
    - Discrete: Outputs logits for Categorical distribution

    Supports gradual complexity increase:
    - lora_mode='final_only': Only apply LoRA to final output layer
    - lora_mode='last_hidden': Apply LoRA to last hidden + output layer
    - lora_mode='all': Apply LoRA to all layers (full model)
    """

    hidden_dims: tuple = (64, 64)  # Tuple of hidden layer dimensions
    action_dim: int = 2
    log_std_min: float = -2.0
    log_std_max: float = 0.0
    min_std: float = 0.3  # Minimum std for numerical stability and exploration
    lora_mode: str = "final_only"  # 'final_only', 'last_hidden', or 'all'
    discrete_actions: bool = (
        False  # If True, output logits for Categorical distribution
    )

    def setup(self):
        # Create hidden layers based on hidden_dims
        self.hidden_layers = [nn.Dense(features=dim) for dim in self.hidden_dims]
        # Output layers
        if self.discrete_actions:
            # For discrete actions, only need logits output
            self.logits_layer = nn.Dense(features=self.action_dim)
        else:
            # For continuous actions, need mean and log_std
            self.mean_layer = nn.Dense(features=self.action_dim)
            self.log_std_layer = nn.Dense(features=self.action_dim)

    def __call__(self, x, adapters):
        # Check if adapters exist and are non-empty
        use_adapters = len(adapters) > 0
        if use_adapters:
            # Get rank from any available adapter
            first_key = list(adapters.keys())[0]
            lora_rank = (
                adapters[first_key].shape[1]
                if len(adapters[first_key].shape) > 1
                else 0
            )
            use_adapters = lora_rank > 0

        h = x
        num_hidden_layers = len(self.hidden_layers)

        # Apply each hidden layer with its corresponding LoRA adapter (if enabled for this layer)
        for i, layer in enumerate(self.hidden_layers):
            layer_idx = i + 1
            A_key = f"A{layer_idx}"
            B_key = f"B{layer_idx}"

            # Check if adapters exist for this layer (they may not, depending on lora_mode)
            if use_adapters and A_key in adapters and B_key in adapters:
                # Apply LoRA adapter for this layer
                # h: (batch, input_dim), A: (batch, lora_rank, input_dim)
                adapter_out = jnp.einsum("bi,bri->br", h, adapters[A_key])
                # adapter_out: (batch, lora_rank), B: (batch, output_dim, lora_rank)
                adapter_out = jnp.einsum("br,bor->bo", adapter_out, adapters[B_key])
                # Clip adapter contribution (loose bound to preserve adapter SND calculation)
                adapter_out = jnp.clip(adapter_out, -1000.0, 1000.0)
            else:
                adapter_out = 0.0

            h = nn.relu(layer(h) + adapter_out)

        # Apply final LoRA adapter to output mean (should exist in all modes)
        final_idx = len(self.hidden_layers) + 1
        A_key = f"A{final_idx}"
        B_key = f"B{final_idx}"

        if use_adapters and A_key in adapters and B_key in adapters:
            adapter_out = jnp.einsum("bh,brh->br", h, adapters[A_key])
            adapter_out = jnp.einsum("br,bar->ba", adapter_out, adapters[B_key])
            # Clip final adapter contribution - use large threshold to avoid affecting
            # SND calculations. Clipping compresses pairwise distances and breaks
            # the proportionality between scaled and unscaled diversity metrics.
            adapter_out = jnp.clip(adapter_out, -1000.0, 1000.0)
        else:
            adapter_out = 0.0

        if self.discrete_actions:
            # For discrete actions, output logits for categorical distribution
            logits = self.logits_layer(h) + adapter_out
            # Return logits and dummy log_std for API compatibility
            return logits, jnp.zeros_like(logits)
        else:
            # For continuous actions, output mean and log_std for Gaussian
            mean = self.mean_layer(h) + adapter_out
            # Clip action mean - use large threshold to preserve diversity metrics.
            # Actual actions are bounded by tanh anyway, so large means just saturate.
            mean = jnp.clip(mean, -1000.0, 1000.0)
            log_std = self.log_std_layer(h)
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std

    def get_action_and_log_prob(self, x, adapters, rng_key=None):
        """
        Get action and its log probability for PPO.

        For continuous actions: Uses a Tanh-transformed Gaussian distribution
        For discrete actions: Uses a Categorical distribution

        Args:
            x: Input observation (batch, obs_dim)
            adapters: Dictionary containing LoRA adapter matrices
            rng_key: JAX random key for sampling

        Returns:
            For continuous:
                action: Sampled action (bounded to [-1, 1])
                log_prob: Log probability of the action (correctly adjusted for tanh)
                mean: Mean of the pre-tanh distribution
                std: Std of the pre-tanh distribution
            For discrete:
                action: Sampled action index (integer)
                log_prob: Log probability of the action
                logits: Raw logits from policy
                None: Placeholder for std (not used in discrete case)
        """
        output1, output2 = self(x, adapters)

        if self.discrete_actions:
            # Discrete actions: output1 is logits, output2 is dummy
            logits = output1

            # Create categorical distribution
            dist = distrax.Categorical(logits=logits)

            if rng_key is not None:
                # Sample action during rollout
                action = dist.sample(seed=rng_key)
                log_prob = dist.log_prob(action)
            else:
                # For evaluation, use greedy action (argmax)
                action = jnp.argmax(logits, axis=-1)
                log_prob = None

            return action, log_prob, logits, None

        else:
            # Continuous actions: output1 is mean, output2 is log_std
            mean = output1
            log_std = output2
            std = jnp.exp(log_std)

            # CRITICAL: Ensure std never collapses to 0 or becomes NaN
            # Use configurable minimum std (can be adjusted per scenario)
            std = jnp.maximum(std, self.min_std)
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
                # This "poisons" the actions stored in the buffer, causing NaN during training
                action_epsilon = (
                    1e-6  # Small epsilon to keep actions away from boundaries
                )
                action = jnp.clip(action, -1.0 + action_epsilon, 1.0 - action_epsilon)

                # Compute log probability AFTER clipping to match what's stored
                # This ensures consistency between rollout and training
                log_prob = dist.log_prob(action).sum(axis=-1)

                # Safety: Replace any remaining NaN/Inf values
                # (shouldn't occur with epsilon clipping, but keep as failsafe)
                log_prob = jnp.nan_to_num(
                    log_prob, nan=-1e10, posinf=-1e10, neginf=-1e10
                )
            else:
                # For evaluation, use the deterministic, squashed mean
                action = jnp.tanh(mean)
                log_prob = None

            return action, log_prob, mean, std
