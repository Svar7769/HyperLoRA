"""
Wrapper utilities to bridge between HyperMARL training script and local env_setup.
Provides a make_env function compatible with the baselines interface.
"""

import numpy as np
from env_setup import make_vmas_env


class FootballVMASWrapper:
    """Adapter that makes FootballWrapper compatible with the VMASWrapper dict interface."""

    def __init__(self, football_env, num_envs):
        self._env = football_env
        self.num_agents = football_env.num_blue_agents
        self.num_envs = num_envs
        self.observation_size = football_env.obs_dim
        self.action_dim = football_env.action_dim
        self.continuous_actions = True
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.possible_agents = self.agents

    def reset(self, seed=None):
        obs_list = self._env.reset()
        obs_dict = {
            agent: obs_list[i].cpu().numpy() for i, agent in enumerate(self.agents)
        }
        return obs_dict, {}

    def step(self, actions):
        import torch

        if isinstance(actions, dict):
            actions_list = [actions[agent] for agent in self.agents]
            actions_array = np.array(actions_list)
            actions_array = np.swapaxes(actions_array, 0, 1)
        else:
            actions_array = np.array(actions).reshape(
                self.num_envs, self.num_agents, self.action_dim
            )

        actions_tensor = [
            torch.tensor(actions_array[:, i, :], dtype=torch.float32)
            for i in range(self.num_agents)
        ]

        obs_list, rewards_list, dones, infos_list = self._env.step(actions_tensor)

        obs_dict = {
            agent: obs_list[i].cpu().numpy() for i, agent in enumerate(self.agents)
        }
        rewards_dict = {
            agent: rewards_list[i].cpu().numpy() for i, agent in enumerate(self.agents)
        }
        dones_np = dones.cpu().numpy()
        dones_dict = {agent: dones_np for agent in self.agents}
        truncs_dict = {
            agent: np.zeros_like(dones_np, dtype=bool) for agent in self.agents
        }

        return obs_dict, rewards_dict, dones_dict, truncs_dict, {}

    def get_win_rate(self):
        return self._env.get_win_rate()

    def reset_stats(self):
        self._env.reset_stats()

    def close(self):
        self._env.close()


class VMASWrapper:
    """Wrapper for VMAS environment to provide consistent interface."""

    def __init__(
        self,
        vmas_env,
        num_agents,
        num_envs,
        observation_size,
        action_dim=1,
        continuous_actions=True,
    ):
        self._env = vmas_env
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.observation_size = observation_size
        self.action_dim = action_dim
        self.continuous_actions = continuous_actions
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents

    def reset(self, seed=None):
        """Reset the environment."""
        obs = self._env.reset(seed=seed)
        # Convert tensor observations to dict format (no agent IDs added)
        obs_dict = {agent: obs[i].cpu().numpy() for i, agent in enumerate(self.agents)}
        infos = {}
        return obs_dict, infos

    def step(self, actions):
        """
        Step the environment.

        Args:
            actions: Either a flat array of shape (num_envs * num_agents,) or dict

        Returns:
            obs_dict, rewards_dict, dones_dict, truncs_dict, infos_dict
        """
        import torch

        # Convert actions to tensor format expected by VMAS
        if isinstance(actions, dict):
            # If dict, extract in agent order
            # actions[agent] has shape (num_envs, action_dim) or (num_envs,)
            actions_list = [actions[agent] for agent in self.agents]
            # Stack to get (num_agents, num_envs, action_dim) or (num_agents, num_envs)
            actions_array = np.array(actions_list)
            # Swap axes to get (num_envs, num_agents, action_dim) or (num_envs, num_agents)
            actions_array = np.swapaxes(actions_array, 0, 1)
        else:
            # If flat array, reshape by action space type
            if self.continuous_actions and self.action_dim > 1:
                # Continuous actions with multiple dimensions
                actions_array = np.array(actions).reshape(
                    self.num_envs, self.num_agents, self.action_dim
                )
            else:
                # Discrete actions (single dimension)
                actions_array = np.array(actions).reshape(
                    self.num_envs, self.num_agents
                )

        # Convert to tensor list format expected by VMAS
        if self.continuous_actions and self.action_dim > 1:
            # For continuous multi-dimensional actions
            actions_tensor = [
                torch.tensor(actions_array[:, i, :], dtype=torch.float32)
                for i in range(self.num_agents)
            ]
        else:
            # For discrete actions
            actions_tensor = [
                torch.tensor(actions_array[:, i], dtype=torch.int64)
                for i in range(self.num_agents)
            ]

        # Step environment
        obs, rewards, dones, infos = self._env.step(actions_tensor)

        # Convert outputs to dict format (no agent IDs added)
        obs_dict = {agent: obs[i].cpu().numpy() for i, agent in enumerate(self.agents)}

        rewards_dict = {
            agent: rewards[i].cpu().numpy() for i, agent in enumerate(self.agents)
        }

        dones_dict = {
            agent: dones.cpu().numpy() for agent in self.agents
        }  # Same for all agents
        truncs_dict = {
            agent: np.zeros_like(dones.cpu().numpy(), dtype=bool)
            for agent in self.agents
        }
        infos_dict = {}

        return obs_dict, rewards_dict, dones_dict, truncs_dict, infos_dict

    def close(self):
        """Close the environment."""
        pass


def make_env(env_name, num_envs=1, **kwargs):
    """
    Create environment with interface compatible with baselines.

    Args:
        env_name: Name of the environment/scenario
        num_envs: Number of parallel environments
        **kwargs: Additional environment parameters

    Returns:
        env: Wrapped environment
        possible_agents: List of agent IDs
        action_dim: Action dimension
        num_actions: Number of actions (same as action_dim for continuous)
        observation_size: Observation size including agent IDs
    """
    # Extract parameters from kwargs
    env_kwargs = kwargs.pop("ENV_KWARGS", {}) if "ENV_KWARGS" in kwargs else {}

    one_hot_encode_agent_id = env_kwargs.get("one_hot_encode_agent_id", True)
    requested_n_agents = env_kwargs.get("n_agents", 4)

    # Merge all kwargs (kwargs takes priority over env_kwargs)
    all_kwargs = {**env_kwargs, **kwargs}
    all_kwargs.pop(
        "one_hot_encode_agent_id", None
    )  # Remove as it's handled differently
    all_kwargs.pop("auto_reset", None)  # Not needed for VMAS

    # Extract agent capabilities configuration
    use_fixed_capabilities = all_kwargs.pop("use_fixed_capabilities", False)
    fixed_capabilities = all_kwargs.pop("fixed_capabilities", {})

    agent_capabilities = None

    if use_fixed_capabilities and fixed_capabilities:
        # Build agent_capabilities dictionary for make_vmas_env
        agent_capabilities = {}
        if "speed" in fixed_capabilities:
            agent_capabilities["speed"] = fixed_capabilities["speed"]
        if "lidar_range" in fixed_capabilities:
            agent_capabilities["lidar_range"] = fixed_capabilities["lidar_range"]
        if "max_speed" in fixed_capabilities:
            agent_capabilities["max_speed"] = fixed_capabilities["max_speed"]
        if "force_multiplier" in fixed_capabilities:
            agent_capabilities["force_multiplier"] = fixed_capabilities[
                "force_multiplier"
            ]

        print(f"\n[Environment Setup] Using fixed agent capabilities for {env_name}:")
        for key, values in agent_capabilities.items():
            print(f"  {key}: {values}")
        print()  # Add blank line for readability

    # Create VMAS environment
    vmas_env = make_vmas_env(
        scenario_name=env_name,
        num_agents=requested_n_agents,
        num_envs=num_envs,
        agent_capabilities=agent_capabilities,
        **all_kwargs,
    )

    # FootballWrapper has a different interface; wrap it with the dict adapter
    from football_wrapper import FootballWrapper

    if isinstance(vmas_env, FootballWrapper):
        env = FootballVMASWrapper(vmas_env, num_envs)
        observation_size = env.observation_size
        action_dim = env.action_dim
        possible_agents = env.agents
        return env, possible_agents, action_dim, action_dim, observation_size

    # IMPORTANT: derive the actual number of agents from the created environment.
    # Some scenarios (e.g., pressure_plate) may ignore `n_agents` and define their
    # own active agent count (via scenario-specific kwargs like n_ground_robots).
    actual_n_agents = len(vmas_env.agents)

    # Get environment dimensions
    # Reset to get observation shape
    obs = vmas_env.reset()
    obs_size = obs[0].shape[-1]

    # Observation size without agent IDs (agent IDs computed from batch position)
    observation_size = obs_size

    # Get action dimension
    action_dim = vmas_env.get_agent_action_size(vmas_env.agents[0])
    continuous_actions = all_kwargs.get("continuous_actions", True)

    # Wrap environment (no agent IDs added to observations)
    env = VMASWrapper(
        vmas_env,
        actual_n_agents,
        num_envs,
        observation_size,
        action_dim,
        continuous_actions=continuous_actions,
    )
    possible_agents = env.agents

    return env, possible_agents, action_dim, action_dim, observation_size
