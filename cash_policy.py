import functools
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal


class ScannedRNN(nn.Module):
    """Scanned GRU with done-based hidden-state reset."""

    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], self.hidden_size),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(
            features=self.hidden_size,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            recurrent_kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CASHPolicy(nn.Module):
    """
    Capability-Aware Shared Hypernetworks (CASH) policy.

    At every timestep, the hyper-adapter consumes:
    - ego observation
    - ego capabilities
    - team capabilities (concatenated)

    and dynamically generates the decoder weights used to map GRU hidden states
    to action mean/log_std.
    """

    num_agents: int
    capability_dim: int
    gru_hidden_dim: int = 64
    fc_dim_size: int = 64
    action_dim: int = 2
    hyper_hidden_dim: int = 128
    hyper_num_layers: int = 2
    expected_hyper_input_dim: int = 0
    decoder_hidden_dim: int = 64
    use_two_layer_decoder: bool = True
    log_std_min: float = -2.0
    log_std_max: float = 0.0
    min_std: float = 0.3

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)

    def _decoder_param_dim(self):
        out_dim = 2 * self.action_dim
        if self.use_two_layer_decoder:
            return (
                self.decoder_hidden_dim * self.gru_hidden_dim
                + self.decoder_hidden_dim
                + out_dim * self.decoder_hidden_dim
                + out_dim
            )
        return out_dim * self.gru_hidden_dim + out_dim

    def _build_team_capabilities(self, capability_seq):
        # capability_seq: (time, batch, capability_dim)
        t, b, c = capability_seq.shape
        if b % self.num_agents != 0:
            # Fallback for JAX initialization / dummy batches
            pooled = jnp.mean(capability_seq, axis=1, keepdims=True)  # (t, 1, c)
            return jnp.repeat(pooled, b, axis=1)  # (t, b, c)

        num_envs = b // self.num_agents
        # caps_env: (time, num_envs, num_agents, capability_dim)
        caps_env = capability_seq.reshape(t, num_envs, self.num_agents, c)
        # KEY CHANGE: Aggregate capabilities using mean pooling across agents
        # This reduces the shape to (t, num_envs, capability_dim) regardless of num_agents
        team_caps_mean = jnp.mean(caps_env, axis=2)
        # Repeat the pooled team vector for each agent in the environment
        # Shape becomes (t, num_envs, num_agents, capability_dim)
        team_caps_repeated = jnp.repeat(
            team_caps_mean[:, :, None, :], self.num_agents, axis=2
        )
        # Return flattened batch: (t, batch, capability_dim)
        return team_caps_repeated.reshape(t, b, c)

    def _apply_dynamic_decoder(self, hidden_seq, flat_params):
        # hidden_seq: (time, batch, gru_hidden_dim)
        # flat_params: (time, batch, param_dim)
        t, b, hdim = hidden_seq.shape
        tb = t * b

        hidden_flat = hidden_seq.reshape(tb, hdim)
        params_flat = flat_params.reshape(tb, -1)

        out_dim = 2 * self.action_dim
        cursor = 0

        if self.use_two_layer_decoder:
            # W1: (tb, decoder_hidden_dim, gru_hidden_dim)
            w1_size = self.decoder_hidden_dim * self.gru_hidden_dim
            w1 = params_flat[:, cursor : cursor + w1_size].reshape(
                tb, self.decoder_hidden_dim, self.gru_hidden_dim
            )
            cursor += w1_size

            # b1: (tb, decoder_hidden_dim)
            b1 = params_flat[:, cursor : cursor + self.decoder_hidden_dim]
            cursor += self.decoder_hidden_dim

            # W2: (tb, out_dim, decoder_hidden_dim)
            w2_size = out_dim * self.decoder_hidden_dim
            w2 = params_flat[:, cursor : cursor + w2_size].reshape(
                tb, out_dim, self.decoder_hidden_dim
            )
            cursor += w2_size

            # b2: (tb, out_dim)
            b2 = params_flat[:, cursor : cursor + out_dim]

            # h1 = tanh(W1 * h + b1)
            h1 = jnp.einsum("bh,bdh->bd", hidden_flat, w1) + b1
            h1 = jnp.tanh(h1)
            # out = W2 * h1 + b2
            out = jnp.einsum("bd,bod->bo", h1, w2) + b2
        else:
            # W: (tb, out_dim, gru_hidden_dim)
            w_size = out_dim * self.gru_hidden_dim
            w = params_flat[:, cursor : cursor + w_size].reshape(
                tb, out_dim, self.gru_hidden_dim
            )
            cursor += w_size

            # b: (tb, out_dim)
            b_vec = params_flat[:, cursor : cursor + out_dim]
            out = jnp.einsum("bh,boh->bo", hidden_flat, w) + b_vec

        out = out.reshape(t, b, out_dim)
        mean = out[:, :, : self.action_dim]
        log_std = out[:, :, self.action_dim :]
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    @nn.compact
    def __call__(self, hidden, x):
        # x: (obs_seq, dones_seq, capability_seq)
        obs_seq, dones_seq, capability_seq = x

        embedding = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs_seq)
        embedding = nn.relu(embedding)

        hidden, rnn_out = ScannedRNN(hidden_size=self.gru_hidden_dim)(
            hidden, (embedding, dones_seq)
        )

        team_caps_seq = self._build_team_capabilities(capability_seq)

        # Hyper-adapter input per (time, batch, agent):
        # [ego_obs, ego_capabilities, team_capabilities]
        hyper_in = jnp.concatenate([obs_seq, capability_seq, team_caps_seq], axis=-1)

        # Evaluation safety: enforce a fixed hyper input width inferred from checkpoint
        # so parameter loading cannot fail due to accidental upstream feature drift.
        if self.expected_hyper_input_dim > 0:
            current_dim = int(hyper_in.shape[-1])
            if current_dim > self.expected_hyper_input_dim:
                hyper_in = hyper_in[..., : self.expected_hyper_input_dim]
            elif current_dim < self.expected_hyper_input_dim:
                pad = jnp.zeros(
                    hyper_in.shape[:-1]
                    + (self.expected_hyper_input_dim - current_dim,),
                    dtype=hyper_in.dtype,
                )
                hyper_in = jnp.concatenate([hyper_in, pad], axis=-1)

        hyper_x = hyper_in
        for i in range(self.hyper_num_layers):
            hyper_x = nn.Dense(
                self.hyper_hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name=f"hyper_dense_{i + 1}",
            )(hyper_x)
            # Required by CASH architecture: include LayerNorm in the hypernetwork.
            if i == 0:
                hyper_x = nn.LayerNorm(name="hyper_ln_1")(hyper_x)
            hyper_x = jnp.tanh(hyper_x)

        flat_decoder_params = nn.Dense(
            self._decoder_param_dim(),
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="hyper_out",
        )(hyper_x)

        mean, log_std = self._apply_dynamic_decoder(rnn_out, flat_decoder_params)
        return hidden, (mean, log_std)

    def get_action_and_log_prob(self, hidden, x, rng_key=None):
        hidden, (mean, log_std) = self(hidden, x)
        std = jnp.exp(log_std)
        std = jnp.maximum(std, self.min_std)
        std = jnp.nan_to_num(std, nan=1.0)

        base_dist = distrax.Normal(mean, std)
        tanh_bijector = distrax.Tanh()
        dist = distrax.Transformed(base_dist, tanh_bijector)

        if rng_key is not None:
            action = dist.sample(seed=rng_key)
            action_epsilon = 1e-6
            action = jnp.clip(action, -1.0 + action_epsilon, 1.0 - action_epsilon)
            log_prob = dist.log_prob(action).sum(axis=-1)
            log_prob = jnp.nan_to_num(log_prob, nan=-1e10, posinf=-1e10, neginf=-1e10)
        else:
            action = jnp.tanh(mean)
            log_prob = None

        return action, log_prob, hidden, mean, std
