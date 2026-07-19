"""
Graph Attention Network (GATv2) layers for JAX/Flax

JAX/Flax implementation of GATv2 for deep sets architecture in DICO.
Based on "How Attentive are Graph Attention Networks?" (Brody et al. 2022)
and used in Bettini et al. 2024's DICO football experiments.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import initializers
from typing import Optional


class GATv2Layer(nn.Module):
    """
    Simplified Graph Attention Network v2 (GATv2) layer for JAX/Flax.

    Uses mean aggregation with distance-based edge masking.
    Simplified from full GATv2 to be more stable and easier to optimize.

    Attributes:
        out_features: Output feature dimension
        use_bias: Whether to use bias in linear layers (default: True)
        edge_radius: Maximum distance for edges (None = fully connected)
    """

    out_features: int
    use_bias: bool = True
    edge_radius: Optional[float] = None

    def setup(self):
        # Define the linear transformation layer
        self.transform = nn.Dense(
            features=self.out_features,
            use_bias=self.use_bias,
            name="transform"
        )

    def compute_edge_mask(self, positions):
        """
        Compute edge mask based on distance threshold.

        Args:
            positions: Node positions (batch, num_nodes, 2)

        Returns:
            edge_mask: (batch, num_nodes, num_nodes) boolean mask
        """
        if self.edge_radius is None:
            # Fully connected graph
            batch_size, num_nodes, _ = positions.shape
            return jnp.ones((batch_size, num_nodes, num_nodes), dtype=bool)

        # Compute pairwise distances
        pos_i = jnp.expand_dims(positions, axis=2)  # (batch, num_nodes, 1, 2)
        pos_j = jnp.expand_dims(positions, axis=1)  # (batch, 1, num_nodes, 2)

        # Compute squared distances
        dist_sq = jnp.sum(
            (pos_i - pos_j) ** 2, axis=-1
        )  # (batch, num_nodes, num_nodes)

        # Create mask: True where distance <= radius
        edge_mask = dist_sq <= (self.edge_radius**2)

        return edge_mask

    def __call__(self, x, positions=None, training=False):
        """
        Forward pass with simplified attention.

        Args:
            x: Node features (batch, num_nodes, in_features)
            positions: Node positions for distance-based edges (batch, num_nodes, 2)
            training: Whether in training mode

        Returns:
            out: Updated node features (batch, num_nodes, out_features)
        """
        batch_size, num_nodes, in_features = x.shape

        # 1. Linear transformation of input features
        h = self.transform(x)
        # (batch, num_nodes, out_features)

        # 2. Compute edge mask from positions
        if positions is not None:
            edge_mask = self.compute_edge_mask(positions)
        else:
            edge_mask = jnp.ones((batch_size, num_nodes, num_nodes), dtype=bool)

        # Add self-loops
        eye = jnp.eye(num_nodes, dtype=bool)
        edge_mask = edge_mask | eye[None, :, :]

        # 3. Compute attention using dot product similarity
        # Normalize features for stable attention
        h_norm = h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)

        # Compute pairwise similarities: (batch, num_nodes, num_nodes)
        # attention_logits[b, i, j] = similarity between node i and node j
        attention_logits = jnp.einsum("bif,bjf->bij", h_norm, h_norm)

        # Apply edge mask
        attention_logits = jnp.where(edge_mask, attention_logits, -1e9)

        # Softmax to get attention weights
        attention_weights = jax.nn.softmax(
            attention_logits, axis=2
        )  # (batch, num_nodes, num_nodes)

        # 4. Aggregate neighbor features
        # out[b, i] = Σ_j attention_weights[b, i, j] * h[b, j]
        out = jnp.einsum("bij,bjf->bif", attention_weights, h)

        # 5. Apply nonlinearity
        out = jax.nn.relu(out)

        return out


class DeepSetsEncoder(nn.Module):
    """
    Deep Sets encoder using simplified GAT for permutation-invariant processing.

    This combines graph attention with mean pooling to create a permutation-invariant
    representation suitable for DICO.

    Attributes:
        out_features: Output feature dimension per node
        edge_radius: Maximum distance for local graph (None = fully connected)
        aggr: Aggregation method ('mean' - currently only mean supported)
    """

    out_features: int = 32
    edge_radius: Optional[float] = None
    aggr: str = "mean"

    def setup(self):
        """Setup the GAT layer."""
        self.gat = GATv2Layer(
            out_features=self.out_features,
            edge_radius=self.edge_radius,
        )

    def __call__(self, x, positions=None, training=False):
        """
        Encode node features with simplified GAT.

        Args:
            x: Node features (batch, num_nodes, in_features)
            positions: Node positions (batch, num_nodes, 2) for distance-based edges
            training: Whether in training mode

        Returns:
            node_embeddings: (batch, num_nodes, out_features)
        """
        # Apply simplified GAT
        h = self.gat(x, positions=positions, training=training)

        return h
