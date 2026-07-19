import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import functools
from flax.linen.initializers import constant, orthogonal
from gat_layers import DeepSetsEncoder


class ScannedRNN(nn.Module):
    """
    Scanned RNN module that applies GRU cells across a sequence.

    Handles proper state reset based on episode termination flags,
    maintaining recurrent state across sequences of observations.
    Shared by both Actor and Critic.
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
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], self.hidden_size),
            rnn_state,
        )
        # Improved GRU initialization for better gradient flow
        new_rnn_state, y = nn.GRUCell(
            features=self.hidden_size,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            recurrent_kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """Initialize recurrent state."""
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CentralizedCritic(nn.Module):
    """
    Centralized critic for MAPPO with MLP architecture (legacy).

    This is the simple MLP-based critic without recurrence.
    Use CentralizedCriticRNN for recurrent architecture.

    Outputs per-agent value predictions to support heterogeneous rewards.
    """

    hidden_dim: int = 128
    num_layers: int = 2
    num_agents: int = 1  # Number of agents for per-agent value predictions

    @nn.compact
    def __call__(self, global_state):
        """
        Estimate value function from global state.

        Args:
            global_state: Concatenated observations from all agents
                         Shape: (batch_size, global_state_dim)

        Returns:
            value: Estimated state value per agent (batch_size, num_agents)
        """
        x = global_state

        # Multi-layer MLP
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)

        # Output per-agent values
        value = nn.Dense(features=self.num_agents)(x)  # (batch_size, num_agents)

        return value


class CentralizedCriticRNN(nn.Module):
    """
    Centralized critic with recurrent architecture for MAPPO.

    Mirrors HyperMARL's CriticRNN architecture:
    - Initial embedding layer with ReLU
    - ScannedRNN (GRU) for temporal processing
    - Intermediate dense layer with ReLU
    - Final value output

    This implements centralized training with decentralized execution (CTDE),
    with the critic having access to global state and maintaining temporal
    context through recurrence.

    Outputs per-agent value predictions to support heterogeneous rewards.

    Attributes:
        gru_hidden_dim: Hidden dimension of GRU
        fc_dim_size: Dimension of initial embedding layer
        num_agents: Number of agents for per-agent value predictions
    """

    gru_hidden_dim: int = 64
    fc_dim_size: int = 64
    num_agents: int = 1  # Number of agents for per-agent value predictions

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """Initialize hidden state for GRU."""
        return ScannedRNN.initialize_carry(batch_size, hidden_size)

    @nn.compact
    def __call__(self, hidden, x):
        """
        Forward pass through the critic network.
        Matches HyperMARL's CriticRNN architecture exactly.

        Args:
            hidden: Initial recurrent state (batch, gru_hidden_dim)
            x: Tuple of (world_state, dones)
                - world_state: (time, batch, state_dim) global state
                - dones: (time, batch) episode termination flags

        Returns:
            hidden: Updated recurrent state (batch, gru_hidden_dim)
            value: Value estimates per agent (time, batch, num_agents)
        """
        world_state, dones = x

        # 1. Initial embedding layer
        embedding = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)

        # 2. Apply RNN (handles done resets internally)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.gru_hidden_dim)(hidden, rnn_in)

        # 3. Value head - intermediate layer
        critic = nn.Dense(
            self.gru_hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)

        # 4. Final per-agent value output
        critic = nn.Dense(
            self.num_agents,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, critic


class CentralizedCriticDeepSets(nn.Module):
    """
    Centralized critic with Deep Sets (GATv2) architecture for MAPPO.

    Architecture (following Bettini et al. 2024 for football):
    1. GATv2 layer: Processes all agent observations with graph attention (full topology)
       - Outputs per-agent embeddings of dimension 32
    2. MLP: Processes embeddings to output per-agent values
       - [256, 256] hidden layers with Tanh activation

    This matches the config_bettini.yaml critic architecture:
    - l1: gnn_deep_sets (GATv2, topology=full, edge_radius=null) → 32-dim
    - l2: mlp ([256, 256], activation=Tanh)

    Attributes:
        hidden_dim: Hidden dimension for MLP layers (default: 256)
        num_layers: Number of MLP layers (default: 2)
        num_agents: Number of agents for per-agent value predictions
        gat_out_features: Output dimension of GAT layer (default: 32)
        gat_num_heads: Number of attention heads (default: 1)
    """

    hidden_dim: int = 256
    num_layers: int = 2
    num_agents: int = 1
    gat_out_features: int = 32

    def setup(self):
        # GATv2 encoder (full topology, no edge radius)
        self.gat_encoder = DeepSetsEncoder(
            out_features=self.gat_out_features,
            edge_radius=None,  # Full connectivity
            aggr="mean",
        )

        # Fallback encoder for initialization
        self.fallback_encoder = nn.Dense(
            features=self.gat_out_features,
            name="fallback_encoder"
        )

        # MLP layers
        self.mlp_layers = [
            nn.Dense(features=self.hidden_dim, name=f"mlp_{i}")
            for i in range(self.num_layers)
        ]

        # Output layer (per-agent values)
        self.value_layer = nn.Dense(features=self.num_agents, name="value_out")

    def extract_positions(self, global_state, num_agents):
        """
        Extract positions from global state for graph construction.

        The global state is concatenated observations from all agents.
        We extract the first 2 features from each agent's observation as positions.

        Args:
            global_state: (batch, global_state_dim)
            num_agents: Number of agents

        Returns:
            positions: (batch, num_agents, 2)
            node_features: (batch, num_agents, obs_dim)
        """
        batch_size = global_state.shape[0]
        obs_dim = global_state.shape[1] // num_agents

        # Reshape to (batch, num_agents, obs_dim)
        node_features = global_state.reshape(batch_size, num_agents, obs_dim)

        # Extract first 2 features as positions
        positions = node_features[:, :, :2]  # (batch, num_agents, 2)

        return positions, node_features

    @nn.compact
    def __call__(self, global_state, training=False):
        """
        Estimate value function with GATv2 preprocessing.

        Args:
            global_state: Concatenated observations from all agents (batch, global_state_dim)
            training: Whether in training mode

        Returns:
            value: Estimated state value per agent (batch, num_agents)
        """
        batch_size = global_state.shape[0]
        global_state_dim = global_state.shape[1]
        obs_dim = global_state_dim // self.num_agents

        # Check if global_state_dim is divisible by num_agents
        if global_state_dim % self.num_agents == 0 and obs_dim > 0:
            # Normal case: properly formatted global state
            # Extract positions and node features
            positions, node_features = self.extract_positions(global_state, self.num_agents)

            # Apply GATv2 to get per-agent embeddings
            # h: (batch, num_agents, gat_out_features)
            h = self.gat_encoder(node_features, positions=positions, training=training)

            # Pool across agents to get global representation
            # Use mean pooling as in config_bettini (aggr: mean)
            h_global = jnp.mean(h, axis=1)  # (batch, gat_out_features)
        else:
            # Initialization or edge case: use fallback encoder
            h_global = self.fallback_encoder(global_state)
            h_global = jax.nn.relu(h_global)

        # Apply MLP layers
        x = h_global
        for layer in self.mlp_layers:
            x = layer(x)
            x = nn.tanh(x)  # Use tanh as in config_bettini

        # Output per-agent values
        value = self.value_layer(x)  # (batch, num_agents)

        return value
