#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import typing
from typing import Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = True
        self.viewer_zoom = 0.9

        # Reward weights
        self.energy_reward_weight = kwargs.pop("energy_reward_weight", 1.0)
        self.wind_reward_weight = kwargs.pop("wind_reward_weight", 1.0)
        self.formation_shaping_weight = kwargs.pop("formation_shaping_weight", 0.5)
        self.center_reward_weight = kwargs.pop("center_reward_weight", 1.0)

        # Position controller parameters
        self.position_gain = kwargs.pop(
            "position_gain", 2.0
        )  # Proportional gain for position control
        self.max_speed = kwargs.pop("max_speed", 0.5)  # Maximum velocity magnitude
        self.position_range = kwargs.pop(
            "position_range", 5.0
        )  # Action space: [-position_range, +position_range]

        # Wind configuration
        self.wind = torch.tensor(
            [0, -kwargs.pop("wind", 2.0)], device=device, dtype=torch.float32
        ).expand(batch_dim, 2)

        # Agent configuration
        self.n_agents = kwargs.pop("n_agents", 2)
        self.agent_radii = kwargs.pop("agent_radii", [0.05, 0.03])  # Configurable radii

        # Ensure we have enough radii specified
        if len(self.agent_radii) < self.n_agents:
            # Extend with the last radius value
            self.agent_radii = self.agent_radii + [self.agent_radii[-1]] * (
                self.n_agents - len(self.agent_radii)
            )

        self.agent_radii = self.agent_radii[: self.n_agents]  # Trim if too many

        # Episode configuration
        self.horizon = kwargs.pop("horizon", 200)

        # Shielding configuration
        self.cover_angle_tolerance = kwargs.pop("cover_angle_tolerance", 1.0)

        # Make world
        world = World(batch_dim, device, drag=0, linear_friction=0.1)

        # Create agents with configurable radii
        self.agents_list = []
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                render_action=True,
                shape=Sphere(radius=self.agent_radii[i]),
                max_speed=self.max_speed,  # For velocity clipping
                action_script=None,  # We'll use custom position controller
                gravity=self.wind.clone(),  # Each agent starts with full wind
            )
            world.add_agent(agent)
            self.agents_list.append(agent)

            # Initialize reward tracking
            agent.wind_rew = torch.zeros(batch_dim, device=device)
            agent.energy_rew = torch.zeros(batch_dim, device=device)
            agent.center_rew = torch.zeros(batch_dim, device=device)
            agent.prev_pos = torch.zeros(batch_dim, 2, device=device)

        # World-level reward tracking
        self.formation_rew = torch.zeros(batch_dim, device=device)
        self.t = torch.zeros(batch_dim, device=device, dtype=torch.int)

        return world

    def update_agent_capabilities(self, agent_capabilities):
        """
        Update agent capabilities (e.g., radii, speeds).

        For wind_flocking, we primarily care about agent radii.
        Other scenarios might update speeds or lidar ranges.

        Args:
            agent_capabilities: Dict with capability arrays (e.g., {'radii': [0.05, 0.03]})
        """
        # For wind_flocking, capabilities are primarily the radii
        # Radii are set at initialization and don't change during training
        # This method exists for compatibility with the training pipeline
        pass

    def reset_world_at(self, env_index: int = None):
        """Reset world state. Agents start in a line formation perpendicular to wind with random variations."""
        # Spawn agents in a line perpendicular to wind direction with randomization
        # Wind is [0, -wind_magnitude], so perpendicular is along X axis
        spacing = 0.3  # Base spacing between agents
        total_width = (self.n_agents - 1) * spacing

        # CRITICAL: Add randomization to spawn positions to ensure different hypernetwork inputs
        # This creates different initial position contexts each episode → non-zero adapter SND
        if env_index is None:
            # Reset all environments: add random offset per environment
            # X offset: random shift along the line
            x_offset = (
                torch.rand(self.world.batch_dim, 1, device=self.world.device) * 2.0
                - 1.0
            )  # [-1, 1]
            # Y offset: random shift perpendicular to line
            y_offset = (
                torch.rand(self.world.batch_dim, 1, device=self.world.device) * 1.0
                - 0.5
            )  # [-0.5, 0.5]
            # Spacing variation: randomize spacing slightly
            spacing_variation = (
                torch.rand(self.world.batch_dim, 1, device=self.world.device) * 0.2
                + 0.9
            )  # [0.9, 1.1]
        else:
            # Reset single environment
            x_offset = torch.rand(1, device=self.world.device).item() * 2.0 - 1.0
            y_offset = torch.rand(1, device=self.world.device).item() * 1.0 - 0.5
            spacing_variation = (
                torch.rand(1, device=self.world.device).item() * 0.2 + 0.9
            )

        for i, agent in enumerate(self.world.agents):
            if env_index is None:
                # Batch reset with randomization
                base_x = -total_width / 2 + i * spacing
                x_pos = (
                    base_x
                    + x_offset.squeeze(-1)
                    + torch.randn(self.world.batch_dim, device=self.world.device) * 0.1
                )
                y_pos = (
                    y_offset.squeeze(-1)
                    + torch.randn(self.world.batch_dim, device=self.world.device) * 0.1
                )

                positions = torch.stack([x_pos, y_pos], dim=1)
                agent.set_pos(positions, batch_index=None)
                agent.prev_pos = agent.state.pos.clone()
            else:
                # Single environment reset with randomization
                base_x = -total_width / 2 + i * spacing
                x_pos = (
                    base_x
                    + x_offset
                    + torch.randn(1, device=self.world.device).item() * 0.1
                )
                y_pos = y_offset + torch.randn(1, device=self.world.device).item() * 0.1

                agent.set_pos(
                    torch.tensor(
                        [x_pos, y_pos], device=self.world.device, dtype=torch.float32
                    ),
                    batch_index=env_index,
                )
                agent.prev_pos[env_index] = agent.state.pos[env_index].clone()

            # Reset velocity
            agent.set_vel(torch.zeros_like(agent.state.pos), batch_index=env_index)

        # Reset time counter
        if env_index is None:
            self.t = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.int
            )
        else:
            self.t[env_index] = 0

    def process_action(self, agent: Agent):
        """
        Position-based action controller.

        Action is a normalized position in [-1, 1] that gets scaled to environment coordinates.
        We then compute desired velocity as: v_desired = k_p * (pos_target - pos_current)
        """
        # Action is in range [-1, 1], scale to position space
        target_position = agent.action.u * self.position_range

        # Compute desired velocity using proportional control
        position_error = target_position - agent.state.pos
        desired_velocity = self.position_gain * position_error

        # Clip desired velocity to max_speed (only scale down if exceeding max_speed)
        velocity_norm = torch.linalg.norm(desired_velocity, dim=-1, keepdim=True)
        velocity_scale = torch.clamp(velocity_norm / self.max_speed, min=1.0)
        desired_velocity = desired_velocity / velocity_scale

        # Set velocity directly (simple position controller)
        # In VMAS, we can set velocity which will be integrated to position
        # For more realistic control, we could compute force instead
        agent.state.vel = desired_velocity

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.t += 1
            self.update_wind_shielding()

            # Energy reward: penalize distance moved (encourages staying in place)
            for a in self.world.agents:
                distance_moved = torch.linalg.norm(a.state.pos - a.prev_pos, dim=-1)
                a.energy_rew = -distance_moved * self.energy_reward_weight
                # Update previous position for next step
                a.prev_pos = a.state.pos.clone()

            # Wind reward: penalize wind exposure (after shielding is applied)
            for a in self.world.agents:
                wind_magnitude = torch.linalg.norm(a.gravity, dim=-1)
                a.wind_rew = -wind_magnitude * self.wind_reward_weight

            # Center reward: penalize distance from origin to prevent collective drift
            for a in self.world.agents:
                dist_to_center = torch.linalg.norm(a.state.pos, dim=-1)
                a.center_rew = -dist_to_center * self.center_reward_weight

            # Formation shaping reward: reward when small agents are downstream of big agent
            self.formation_rew = self.compute_formation_reward()

        # Per-agent reward is sum of all components
        return agent.energy_rew + agent.wind_rew + agent.center_rew + self.formation_rew

    def compute_formation_reward(self):
        """
        Compute shaping reward based on formation alignment.

        Reward when smaller agents are positioned downstream of the largest agent.
        Uses cosine similarity between (small_pos - big_pos) and wind_direction.
        """
        if self.n_agents < 2:
            return torch.zeros(self.world.batch_dim, device=self.world.device)

        # Find the agent with largest radius (the shield)
        big_agent_idx = torch.tensor(self.agent_radii).argmax().item()
        big_agent = self.world.agents[big_agent_idx]
        big_pos = big_agent.state.pos  # (batch_dim, 2)

        # Normalize wind direction
        wind_norm = self.wind / (
            torch.linalg.norm(self.wind, dim=-1, keepdim=True) + 1e-8
        )  # (batch_dim, 2)

        total_reward = torch.zeros(self.world.batch_dim, device=self.world.device)

        # For each other agent, compute alignment reward
        for i, agent in enumerate(self.world.agents):
            if i == big_agent_idx:
                continue

            # Vector from big agent to this agent
            vec = agent.state.pos - big_pos  # (batch_dim, 2)
            vec_norm = vec / (torch.linalg.norm(vec, dim=-1, keepdim=True) + 1e-8)

            # Cosine similarity with wind direction
            # If agent is directly downstream: cos = 1 (aligned with wind)
            # If perpendicular: cos = 0
            # If upstream: cos = -1
            alignment = torch.sum(vec_norm * wind_norm, dim=-1)  # (batch_dim,)

            # Reward alignment (value in [-1, 1])
            total_reward += alignment

        # Average over all small agents and scale by weight
        num_small_agents = self.n_agents - 1
        formation_reward = (
            total_reward / num_small_agents
        ) * self.formation_shaping_weight

        return formation_reward

    def update_wind_shielding(self):
        """
        Update wind exposure for agents based on shielding from larger agents.

        The big agent always faces the full wind (it is the shield).
        Small agents receive a reduced wind proportional to how well they are
        positioned downstream of the big agent and how close they are to it.
        """
        if self.n_agents < 2:
            return

        big_agent_idx = torch.tensor(self.agent_radii).argmax().item()
        big_agent = self.world.agents[big_agent_idx]
        big_pos = big_agent.state.pos
        big_radius = self.agent_radii[big_agent_idx]

        wind_norm = self.wind / (
            torch.linalg.norm(self.wind, dim=-1, keepdim=True) + 1e-8
        )

        # Define a max distance where drafting/shielding physically works
        max_shield_dist = 1.0

        for i, agent in enumerate(self.world.agents):
            if i == big_agent_idx:
                # The big agent is the shield; it always faces the full wind
                agent.gravity = self.wind.clone()
                continue

            # Vector from big agent to small agent
            vec = agent.state.pos - big_pos
            dist = torch.linalg.norm(vec, dim=-1, keepdim=True)
            vec_norm = vec / (dist + 1e-8)

            # Cosine similarity: 1 = perfectly downstream of the shield
            alignment = torch.sum(vec_norm * wind_norm, dim=-1, keepdim=True)

            # Map alignment from [-1, 1] to [0, 1]
            shielding_factor = torch.clamp((alignment + 1) / 2, 0.0, 1.0)

            # Define optimal distance as the sum of their radii (touching but not overlapping)
            optimal_dist = big_radius + self.agent_radii[i]

            # Shielding decays as the small agent moves further away from the optimal distance
            dist_error = torch.clamp(dist - optimal_dist, min=0.0)
            dist_factor = torch.clamp(1.0 - (dist_error / max_shield_dist), 0.0, 1.0)

            # Combined effective shielding
            effective_shielding = shielding_factor * dist_factor
            reduction = 1.0 - (effective_shielding * self.cover_angle_tolerance)
            reduction = torch.clamp(reduction, 0.0, 1.0)

            agent.gravity = self.wind * reduction

    def observation(self, agent: Agent):
        """
        Observation for each agent.

        Includes:
        - Agent's own position
        - Agent's own velocity
        - Relative positions to all other agents
        - Wind vector (same for all agents)
        """
        observations = [
            agent.state.pos,  # (batch_dim, 2)
            agent.state.vel,  # (batch_dim, 2)
        ]

        # Relative positions to other agents
        for other_agent in self.world.agents:
            if other_agent != agent:
                rel_pos = other_agent.state.pos - agent.state.pos
                observations.append(rel_pos)

        # Add wind direction (constant across batch)
        observations.append(self.wind)

        return torch.cat(observations, dim=-1)

    def done(self):
        """Episode is done after horizon steps."""
        return self.t >= self.horizon

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "energy_rew": agent.energy_rew,
            "wind_rew": agent.wind_rew,
            "center_rew": getattr(
                agent, "center_rew", torch.zeros_like(agent.energy_rew)
            ),
            "formation_rew": self.formation_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = []

        # Draw wind direction arrow
        wind_arrow_start = (0, 0)
        wind_vec = self.wind[env_index].cpu().numpy()
        wind_arrow_end = tuple(wind_vec * 0.5)  # Scale for visibility

        arrow = rendering.Line(wind_arrow_start, wind_arrow_end, width=2)
        arrow.set_color(0.8, 0.2, 0.2)  # Red for wind
        geoms.append(arrow)

        # Draw line connecting big agent to small agents (to visualize formation)
        big_agent_idx = torch.tensor(self.agent_radii).argmax().item()
        big_pos = self.world.agents[big_agent_idx].state.pos[env_index].cpu().numpy()

        for i, agent in enumerate(self.world.agents):
            if i == big_agent_idx:
                continue
            agent_pos = agent.state.pos[env_index].cpu().numpy()
            line = rendering.Line(tuple(big_pos), tuple(agent_pos), width=1)
            line.set_color(0.5, 0.5, 0.5)  # Gray line
            geoms.append(line)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
