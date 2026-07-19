"""
VMAS Football Environment Wrapper for HyperLoRA

This wrapper adapts the VMAS football environment to work with the HyperLoRA
hypernetwork approach. It handles:
- Observation extraction and formatting for both blue and red teams
- Capability context extraction (speed, size, shooting power for heterogeneous agents)
- Team-specific coordination and role differentiation
- Compatibility with the existing HyperLoRA training pipeline
"""

import torch
import numpy as np
from pathlib import Path
import importlib.util
from vmas.simulator.environment import Environment


class FootballWrapper:
    """
    Wrapper for VMAS Football environment compatible with HyperLoRA.

    This wrapper supports:
    - Heterogeneous agent capabilities (attackers, defenders, goalkeepers)
    - Team-based multi-agent learning (blue vs red)
    - Capability context extraction for hypernetwork
    - Observation normalization and formatting
    """

    def __init__(
        self,
        num_blue_agents=3,
        num_red_agents=3,
        num_envs=64,
        device="cpu",
        continuous_actions=True,
        ai_red_agents=True,
        ai_blue_agents=False,
        physically_different=False,
        enable_shooting=False,
        dense_reward=True,
        observe_teammates=True,
        observe_adversaries=True,
        scenario_kwargs=None,
    ):
        """
        Initialize the Football wrapper.

        Args:
            num_blue_agents: Number of blue team agents (learning agents)
            num_red_agents: Number of red team agents (opponents)
            num_envs: Number of parallel environments
            device: Device to run on ('cpu', 'cuda', or 'mps')
            continuous_actions: Whether to use continuous actions
            ai_red_agents: Whether red agents use heuristic AI
            ai_blue_agents: Whether blue agents use heuristic AI
            physically_different: Whether to use heterogeneous agent capabilities
            enable_shooting: Whether to enable shooting action (rotation + shoot)
            dense_reward: Whether to use dense reward shaping
            observe_teammates: Whether agents observe teammates
            observe_adversaries: Whether agents observe adversaries
            scenario_kwargs: Additional keyword arguments for scenario
        """
        self.num_blue_agents = num_blue_agents
        self.num_red_agents = num_red_agents
        self.num_envs = num_envs
        self.device = device
        self.physically_different = physically_different
        self.enable_shooting = enable_shooting
        self.observe_teammates = observe_teammates
        self.observe_adversaries = observe_adversaries

        # Load custom football scenario
        football_scenario_path = Path(__file__).parent / "football.py"
        if not football_scenario_path.exists():
            raise FileNotFoundError(
                f"Football scenario not found at {football_scenario_path}"
            )

        # Dynamically import the football scenario
        if "custom_football" in importlib.sys.modules:
            del importlib.sys.modules["custom_football"]

        spec = importlib.util.spec_from_file_location(
            "custom_football", football_scenario_path
        )
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        # Create scenario instance
        scenario = custom_module.Scenario()

        # Prepare scenario kwargs
        kwargs = {
            "n_blue_agents": num_blue_agents,
            "n_red_agents": num_red_agents,
            "ai_red_agents": ai_red_agents,
            "ai_blue_agents": ai_blue_agents,
            "physically_different": physically_different,
            "enable_shooting": enable_shooting,
            "dense_reward": dense_reward,
            "observe_teammates": observe_teammates,
            "observe_adversaries": observe_adversaries,
        }
        if scenario_kwargs:
            kwargs.update(scenario_kwargs)

        # Create VMAS environment
        self.env = Environment(
            scenario=scenario,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            **kwargs,
        )

        # Store agent capability information
        self.agent_capabilities = self._extract_agent_capabilities()

        # Calculate observation dimensions
        self._calculate_obs_dims()

        print(f"Football wrapper initialized:")
        print(f"  Blue agents: {num_blue_agents}")
        print(f"  Red agents: {num_red_agents}")
        print(f"  Physically different: {physically_different}")
        print(f"  Enable shooting: {enable_shooting}")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Capability dim: {self.capability_dim}")

        # Initialize goal tracking for win rate computation
        self.blue_goals = 0
        self.red_goals = 0
        self.total_episodes = 0

    def _extract_agent_capabilities(self):
        """
        Extract capability information for each blue agent.

        Returns:
            Dict with capability arrays for speed, size, and shooting power
        """
        blue_agents = self.env.world.blue_agents

        speeds = []
        sizes = []
        shoot_multipliers = []

        for agent in blue_agents:
            # Extract max speed
            speeds.append(float(agent.max_speed))

            # Extract agent size (radius)
            sizes.append(float(agent.shape.radius))

            # Extract shooting multiplier (if shooting enabled)
            if self.enable_shooting and len(agent.u_multiplier) >= 4:
                shoot_multipliers.append(float(agent.u_multiplier[3]))
            else:
                shoot_multipliers.append(0.0)

        capabilities = {
            "speed": np.array(speeds, dtype=np.float32),
            "size": np.array(sizes, dtype=np.float32),
            "shoot_power": np.array(shoot_multipliers, dtype=np.float32),
        }

        return capabilities

    def _calculate_obs_dims(self):
        """Calculate observation and action dimensions."""
        # Get a sample observation
        obs = self.env.reset()

        # Blue agents are the first n_blue_agents in the observation list
        self.obs_dim = obs[0].shape[-1]

        # Action dimension depends on shooting
        self.action_dim = 4 if self.enable_shooting else 2

        # Capability dimension: speed + size + shoot_power
        self.capability_dim = 3

        print(f"Observation breakdown:")
        print(f"  Per-agent obs dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
        if self.enable_shooting:
            print(f"  Actions: [vel_x, vel_y, rotation, shoot]")
        else:
            print(f"  Actions: [vel_x, vel_y]")

    def get_capability_vectors(self, normalize=True):
        """
        Get capability vectors for all blue agents.

        Args:
            normalize: Whether to normalize capabilities to [0, 1] range

        Returns:
            Capability vectors (num_blue_agents, capability_dim)
        """
        speeds = self.agent_capabilities["speed"]
        sizes = self.agent_capabilities["size"]
        shoot_powers = self.agent_capabilities["shoot_power"]

        if normalize:
            # Normalize speeds (typical range: 0.05 to 0.2)
            speeds_norm = (speeds - 0.05) / (0.2 - 0.05)
            speeds_norm = np.clip(speeds_norm, 0, 1)

            # Normalize sizes (typical range: 0.02 to 0.04)
            sizes_norm = (sizes - 0.02) / (0.04 - 0.02)
            sizes_norm = np.clip(sizes_norm, 0, 1)

            # Normalize shoot powers (typical range: 0.4 to 0.8)
            if self.enable_shooting:
                shoot_powers_norm = (shoot_powers - 0.4) / (0.8 - 0.4)
                shoot_powers_norm = np.clip(shoot_powers_norm, 0, 1)
            else:
                shoot_powers_norm = shoot_powers

            capabilities = np.stack(
                [speeds_norm, sizes_norm, shoot_powers_norm], axis=-1
            )
        else:
            capabilities = np.stack([speeds, sizes, shoot_powers], axis=-1)

        return capabilities.astype(np.float32)

    def get_role_embeddings(self):
        """
        Get one-hot role embeddings for agents (if physically_different=True).

        Returns:
            Role embeddings (num_blue_agents, 3) or None
            Roles: [is_attacker, is_defender, is_goalkeeper]
        """
        if not self.physically_different or self.num_blue_agents != 5:
            return None

        # For 5 agents: 2 attackers, 2 defenders, 1 goalkeeper
        roles = np.zeros((5, 3), dtype=np.float32)
        roles[0, 0] = 1.0  # Attacker 1
        roles[1, 0] = 1.0  # Attacker 2
        roles[2, 1] = 1.0  # Defender 1
        roles[3, 1] = 1.0  # Defender 2
        roles[4, 2] = 1.0  # Goalkeeper

        return roles

    def reset(self):
        """
        Reset the environment.

        Returns:
            observations: List of observations for blue agents (length = num_blue_agents)
                         Each obs has shape (num_envs, obs_dim)
        """
        obs_list = self.env.reset()

        # Return only blue agent observations (first num_blue_agents)
        blue_obs = obs_list[: self.num_blue_agents]

        return blue_obs

    def step(self, actions):
        """
        Step the environment with actions for blue agents.
        Red agents use their own policy (heuristic AI or learned).

        Args:
            actions: List of actions for blue agents (length = num_blue_agents)
                    Each action has shape (num_envs, action_dim)

        Returns:
            observations: List of observations for blue agents
            rewards: List of rewards for blue agents
            dones: Done flags (num_envs,)
            infos: List of info dicts for blue agents
        """
        # VMAS only expects actions for agents WITHOUT action_script
        # Since blue agents don't have action_script (ai_blue_agents=False)
        # and red agents + ball DO have action_script, we only pass blue actions

        # Step environment
        obs_list, rewards_list, dones, infos = self.env.step(actions)

        # Extract blue agent data
        blue_obs = obs_list[: self.num_blue_agents]
        blue_rewards = rewards_list[: self.num_blue_agents]
        blue_infos = infos[: self.num_blue_agents]

        # Track goals from sparse rewards (100 = blue scored, -100 = red scored)
        if len(blue_infos) > 0 and "sparse_reward" in blue_infos[0]:
            sparse_rewards = blue_infos[0]["sparse_reward"]  # (num_envs,)
            # Count goals: sparse_reward = 100 means blue scored, -100 means red scored
            blue_scored = (
                (sparse_rewards > 50).sum().item()
            )  # threshold at 50 to be safe
            red_scored = (sparse_rewards < -50).sum().item()
            self.blue_goals += blue_scored
            self.red_goals += red_scored
            self.total_episodes += dones.sum().item()

        return blue_obs, blue_rewards, dones, blue_infos

    def get_win_rate(self):
        """
        Get the current win rate (blue goals / total goals).

        Returns:
            dict with blue_goals, red_goals, total_episodes, win_rate
        """
        total_goals = self.blue_goals + self.red_goals
        win_rate = self.blue_goals / max(total_goals, 1)
        return {
            "blue_goals": self.blue_goals,
            "red_goals": self.red_goals,
            "total_episodes": self.total_episodes,
            "win_rate": win_rate,
        }

    def reset_stats(self):
        """Reset goal tracking statistics."""
        self.blue_goals = 0
        self.red_goals = 0
        self.total_episodes = 0

    def render(self, mode="human", agent_index_focus=None, visualize_when_rgb=False):
        """Render the environment."""
        return self.env.render(
            mode=mode,
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
        )

    def close(self):
        """Close the environment."""
        # VMAS Environment doesn't have a close method, so we can pass
        pass

    # ------------------------------------------------------------------
    # VMAS Environment API compatibility shims
    # These allow evaluate.py and other callers to treat FootballWrapper
    # identically to a raw VMAS Environment.
    # ------------------------------------------------------------------

    @property
    def agents(self):
        """Return the list of blue (learning) agents, matching VMAS API."""
        return self.env.world.blue_agents

    def get_agent_action_size(self, agent=None):
        """Return the action dimension for a blue agent, matching VMAS API."""
        return self.action_dim

    @property
    def scenario(self):
        """Expose the underlying VMAS scenario."""
        return self.env.scenario

    @property
    def world(self):
        """Access to the underlying VMAS world."""
        return self.env.world

    def get_initial_positions(self, normalize=True):
        """
        Get initial spawn positions for blue agents. Call after reset().

        Returns:
            Tensor of shape (num_envs, num_blue_agents, 2) with [x, y] positions.
            If normalize=True, positions are scaled to roughly [-1, 1] using
            pitch half-dimensions.
        """
        positions = torch.stack(
            [a.state.pos[:, :2] for a in self.env.world.blue_agents], dim=1
        )  # (num_envs, num_blue_agents, 2)
        if normalize:
            half_length = self.env.scenario.pitch_length / 2.0
            half_width = self.env.scenario.pitch_width / 2.0
            positions = positions.clone()
            positions[:, :, 0] = positions[:, :, 0] / half_length
            positions[:, :, 1] = positions[:, :, 1] / half_width
        return positions


def create_capability_context_batch(wrapper, num_envs, device="cpu"):
    """
    Create capability context tensors for hypernetwork in batch format.

    Args:
        wrapper: FootballWrapper instance
        num_envs: Number of parallel environments
        device: Device to create tensors on

    Returns:
        capability_vectors: (num_envs, num_agents, capability_dim)
    """
    # Get capability vectors for agents
    capabilities = wrapper.get_capability_vectors(normalize=True)

    # Expand to batch dimension: (num_agents, capability_dim) -> (num_envs, num_agents, capability_dim)
    capability_batch = (
        torch.from_numpy(capabilities).unsqueeze(0).expand(num_envs, -1, -1)
    )

    # Move to device
    if device != "cpu":
        capability_batch = capability_batch.to(device)

    return capability_batch


def create_task_embeddings_batch(wrapper, num_envs, device="cpu", task_embed_dim=16):
    """
    Create task embeddings for football environment.

    For football, the task embedding can encode:
    - Ball position relative to goal
    - Team formation/strategy
    - Score differential (if multi-episode)

    For simplicity, we use a learned embedding that's updated during training.

    Args:
        wrapper: FootballWrapper instance
        num_envs: Number of parallel environments
        device: Device to create tensors on
        task_embed_dim: Dimension of task embedding

    Returns:
        task_embeddings: (num_envs, num_agents, task_embed_dim)
    """
    num_agents = wrapper.num_blue_agents

    # Create zero task embeddings (will be learned parameters in practice)
    task_embeddings = torch.zeros(num_envs, num_agents, task_embed_dim)

    # Move to device
    if device != "cpu":
        task_embeddings = task_embeddings.to(device)

    return task_embeddings


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("VMAS Football Wrapper - Test Run")
    print("=" * 80)

    # Test 1: Homogeneous agents
    print("\n[Test 1] Homogeneous blue team (3 agents) vs AI red team (3 agents)")
    print("-" * 80)

    wrapper_homo = FootballWrapper(
        num_blue_agents=3,
        num_red_agents=3,
        num_envs=4,
        device="cpu",
        ai_red_agents=True,
        ai_blue_agents=False,
        physically_different=False,
        enable_shooting=False,
    )

    print(f"\nCapability vectors (normalized):")
    caps = wrapper_homo.get_capability_vectors(normalize=True)
    print(caps)

    print(f"\nTesting reset and step...")
    obs = wrapper_homo.reset()
    print(f"  Number of observations: {len(obs)}")
    print(f"  Observation shape: {obs[0].shape}")

    # Random actions
    actions = [torch.randn(4, 2).clamp(-1, 1) for _ in range(3)]
    obs, rewards, dones, infos = wrapper_homo.step(actions)
    print(f"  Rewards shape: {rewards[0].shape}")
    print(f"  Sample rewards: {rewards[0][:2]}")

    wrapper_homo.close()
    print("✓ Test 1 passed!")

    # Test 2: Heterogeneous agents (5 blue agents with roles)
    print("\n[Test 2] Heterogeneous blue team (5 agents) with roles")
    print("-" * 80)

    wrapper_hetero = FootballWrapper(
        num_blue_agents=5,
        num_red_agents=5,
        num_envs=4,
        device="cpu",
        ai_red_agents=True,
        ai_blue_agents=False,
        physically_different=True,
        enable_shooting=True,
    )

    print(f"\nCapability vectors (normalized):")
    caps = wrapper_hetero.get_capability_vectors(normalize=True)
    print(caps)
    print(f"\nCapability breakdown:")
    print(
        f"  Agent 0 (Attacker): speed={caps[0,0]:.3f}, size={caps[0,1]:.3f}, shoot={caps[0,2]:.3f}"
    )
    print(
        f"  Agent 1 (Attacker): speed={caps[1,0]:.3f}, size={caps[1,1]:.3f}, shoot={caps[1,2]:.3f}"
    )
    print(
        f"  Agent 2 (Defender): speed={caps[2,0]:.3f}, size={caps[2,1]:.3f}, shoot={caps[2,2]:.3f}"
    )
    print(
        f"  Agent 3 (Defender): speed={caps[3,0]:.3f}, size={caps[3,1]:.3f}, shoot={caps[3,2]:.3f}"
    )
    print(
        f"  Agent 4 (Goalkeeper): speed={caps[4,0]:.3f}, size={caps[4,1]:.3f}, shoot={caps[4,2]:.3f}"
    )

    roles = wrapper_hetero.get_role_embeddings()
    if roles is not None:
        print(f"\nRole embeddings:")
        print(roles)

    print(f"\nTesting reset and step with shooting...")
    obs = wrapper_hetero.reset()
    print(f"  Number of observations: {len(obs)}")
    print(f"  Observation shape: {obs[0].shape}")
    print(f"  Action dim: {wrapper_hetero.action_dim}")

    # Random actions (including rotation and shoot)
    actions = [torch.randn(4, 4).clamp(-1, 1) for _ in range(5)]
    obs, rewards, dones, infos = wrapper_hetero.step(actions)
    print(f"  Rewards shape: {rewards[0].shape}")

    wrapper_hetero.close()
    print("✓ Test 2 passed!")

    # Test 3: Context tensor creation for hypernetwork
    print("\n[Test 3] Creating context tensors for hypernetwork")
    print("-" * 80)

    wrapper = FootballWrapper(
        num_blue_agents=5,
        num_red_agents=5,
        num_envs=8,
        device="cpu",
        physically_different=True,
        enable_shooting=True,
    )

    capability_batch = create_capability_context_batch(wrapper, num_envs=8)
    task_batch = create_task_embeddings_batch(wrapper, num_envs=8, task_embed_dim=16)

    print(f"Capability batch shape: {capability_batch.shape}")
    print(f"  Expected: (8, 5, 3) = (num_envs, num_agents, capability_dim)")
    print(f"Task embedding batch shape: {task_batch.shape}")
    print(f"  Expected: (8, 5, 16) = (num_envs, num_agents, task_embed_dim)")

    wrapper.close()
    print("✓ Test 3 passed!")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print("\nYou can now integrate this wrapper with your HyperLoRA training pipeline.")
    print("Key integration points:")
    print("  1. Use FootballWrapper instead of make_vmas_env for football scenarios")
    print("  2. Extract capability contexts with wrapper.get_capability_vectors()")
    print("  3. Feed capability contexts to your hypernetwork")
    print("  4. The hypernetwork will generate role-specific LoRA adapters")
    print("  5. Train with PPO as usual, blue team learns against AI red team")
