#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
#
#  MODIFIED: Agent-specific food collection
#  - Each agent has a matching colored food (agent_0 → food_0, agent_1 → food_1, etc.)
#  - Agents can ONLY collect their own matching food
#  - Agents only get reward shaping for approaching their own food
#  - Colors match: Blue agent → Blue food, Orange agent → Orange food, etc.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.share_reward = kwargs.pop("share_reward", False)
        self.penalise_by_time = kwargs.pop("penalise_by_time", False)
        self.food_radius = kwargs.pop("food_radius", 0.1)
        self.pos_range = kwargs.pop(
            "pos_range", 1.0
        )  # Increased from 0.3 to 1.0 for bigger environment
        n_food = kwargs.pop("n_food", n_agents)  # Default: one food per agent
        # Reward shaping parameters
        self.distance_shaping_coef = kwargs.pop("distance_shaping_coef", 0.1)

        # Agent capabilities (speed and lidar_range)
        agent_capabilities = kwargs.pop("agent_capabilities", None)
        if agent_capabilities is None:
            # Default: all agents have same capabilities
            self.agent_speeds = [1.0] * n_agents
            self.agent_lidar_ranges = [0.5] * n_agents
        else:
            self.agent_speeds = agent_capabilities.get("speed", [1.0] * n_agents)
            self.agent_lidar_ranges = agent_capabilities.get(
                "lidar_range", [0.5] * n_agents
            )

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Store number of agents for agent-specific food coloring
        self.n_agents = n_agents
        self.n_food = n_food

        # Define colors for agent-specific foods (cycling through colors if more than 8 agents)
        agent_colors = [
            Color.BLUE,
            Color.ORANGE,
            Color.GREEN,
            Color.PINK,
            Color.PURPLE,
            Color.YELLOW,
            Color.RED,
        ]

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.pos_range,
            y_semidim=self.pos_range,
        )
        # Add agents with matching colors and capabilities
        for i in range(n_agents):
            # Assign color to agent matching their food
            agent_color = agent_colors[i % len(agent_colors)]

            agent = Agent(
                name=f"agent_{i}",
                collide=True,  # Must be True for lidar to detect
                shape=Sphere(radius=0.035),
                color=agent_color,  # Set agent color to match their food
                max_speed=self.agent_speeds[i],  # Set agent-specific speed
                obs_range=self.agent_lidar_ranges[i],  # Set agent-specific lidar range
                sensors=(
                    # Lidar for detecting food only (no agent detection needed)
                    Lidar(
                        world,
                        n_rays=12,  # Number of LIDAR rays (evenly distributed 360°)
                        max_range=self.agent_lidar_ranges[
                            i
                        ],  # Agent-specific LIDAR range
                        entity_filter=lambda e, target_name=f"food_{i}": e.name
                        == target_name,  # Detect only matching food
                        angle_start=0.0,  # Start angle (radians)
                        angle_end=2.0 * torch.pi,  # End angle (full 360°)
                    ),
                ),
            )
            world.add_agent(agent)
        # Add landmarks (one food per agent with matching color)
        for i in range(n_food):
            # Assign color based on agent index (cycling through available colors)
            food_color = agent_colors[
                i % len(agent_colors)
            ]  # Use same color scheme as agents

            food = Landmark(
                name=f"food_{i}",
                collide=True,  # Must be True for lidar to detect
                shape=Sphere(radius=self.food_radius),
                color=food_color,
            )
            # Store original and eaten colors for visualization
            food.original_color = food_color
            food.eaten_color = Color.GRAY
            world.add_landmark(food)

        return world

    def update_agent_capabilities(self, agent_capabilities):
        """
        Update agent capabilities (speed, lidar_range) without recreating the world.
        This allows for randomizing capabilities between episodes for generalization.

        Args:
            agent_capabilities: dict with 'speed' and 'lidar_range' lists
        """
        if agent_capabilities is None:
            return

        n_agents = len(self.world.agents)
        self.agent_speeds = agent_capabilities.get("speed", [1.0] * n_agents)
        self.agent_lidar_ranges = agent_capabilities.get(
            "lidar_range", [0.5] * n_agents
        )

        # Update each agent's properties
        # Note: max_speed and obs_range are read-only properties
        # We need to update the underlying attributes instead
        for i, agent in enumerate(self.world.agents):
            agent._max_speed = self.agent_speeds[i]
            agent._obs_range = self.agent_lidar_ranges[i]

            # Also update the Lidar sensor's max_range to match
            if hasattr(agent, "sensors") and agent.sensors is not None:
                for sensor in agent.sensors:
                    if hasattr(sensor, "_max_range"):
                        sensor._max_range = self.agent_lidar_ranges[i]

    def reset_world_at(self, env_index: int = None):
        # Randomize agent starting positions instead of all at (0, 0)
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.pos_range
                    * 0.3,  # Start agents in central region (30% of world size)
                    self.pos_range * 0.3,
                ),
                batch_index=env_index,
            )
        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.pos_range,
                    self.pos_range,
                ),
                batch_index=env_index,
            )
            if env_index is None:
                landmark.eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.just_eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.reset_render()
            else:
                landmark.eaten[env_index] = False
                landmark.just_eaten[env_index] = False
                landmark.is_rendering[env_index] = True

        # Initialize distance tracking for reward shaping
        if not hasattr(self, "prev_min_dist"):
            self.prev_min_dist = {}

        # Reset distance tracking for all agents
        if env_index is None:
            self.prev_min_dist.clear()
        else:
            # Reset only for specific environment
            for agent in self.world.agents:
                if agent.name in self.prev_min_dist:
                    self.prev_min_dist[agent.name][env_index] = float("inf")

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        # Initialize reward decomposition tracking on first call
        if is_first:
            if not hasattr(self, "reward_info"):
                self.reward_info = {}
            # Reset per-step tracking
            self.reward_info["food_collection"] = torch.zeros(
                self.world.batch_dim, device=self.world.device
            )
            self.reward_info["shaping_reward"] = torch.zeros(
                self.world.batch_dim, device=self.world.device
            )
            self.reward_info["time_penalty"] = torch.zeros(
                self.world.batch_dim, device=self.world.device
            )

        rews = torch.zeros(self.world.batch_dim, device=self.world.device)
        food_reward = torch.zeros(self.world.batch_dim, device=self.world.device)
        shaping_reward = torch.zeros(self.world.batch_dim, device=self.world.device)
        time_penalty = torch.zeros(self.world.batch_dim, device=self.world.device)

        # Extract agent index from agent name (e.g., "agent_0" -> 0)
        agent_idx = int(agent.name.split("_")[1])

        # Agent-specific food collection: each agent can only eat their matching food
        # agent_0 can only eat food_0, agent_1 can only eat food_1, etc.
        if agent_idx < len(self.world.landmarks):
            # Get the matching food for this agent
            matching_food = self.world.landmarks[agent_idx]

            # Check if agent is on their matching food
            on_matching_food = (
                torch.linalg.vector_norm(
                    agent.state.pos - matching_food.state.pos, dim=1
                )
                < agent.shape.radius + matching_food.shape.radius
            )

            # Give reward ONLY if:
            # 1. Agent is on their matching food
            # 2. Food has NOT been eaten yet
            # This ensures one-time reward per food
            newly_eating = on_matching_food & ~matching_food.eaten

            if self.share_reward:
                # All agents get reward when this agent eats their food
                food_reward[newly_eating] += 1.0
            else:
                # Only this agent gets reward for eating their matching food
                food_reward[newly_eating] += 1.0

            rews += food_reward

            # Mark food as eaten where agent just ate it
            # This immediately prevents further rewards
            matching_food.eaten |= newly_eating

            # Hide eaten food (turn off rendering)
            matching_food.is_rendering[matching_food.eaten] = False

        # Distance-based reward shaping (approach agent's matching food only)
        if self.distance_shaping_coef > 0 and agent_idx < len(self.world.landmarks):
            # Initialize tracking dict if needed
            if not hasattr(self, "prev_min_dist"):
                self.prev_min_dist = {}

            # Calculate distance to agent's matching food only
            matching_food = self.world.landmarks[agent_idx]

            dist_to_matching_food = torch.linalg.vector_norm(
                agent.state.pos - matching_food.state.pos, dim=1
            )

            # Mask out eaten food (set distance to infinity)
            dist_masked = torch.where(
                matching_food.eaten,
                torch.tensor(float("inf"), device=self.world.device),
                dist_to_matching_food,
            )

            # Potential-based shaping: reward for getting closer to matching food
            agent_key = agent.name
            if agent_key in self.prev_min_dist:
                # Reward = coefficient * (old_distance - new_distance)
                # Positive when getting closer, negative when moving away
                shaping_reward_val = self.distance_shaping_coef * (
                    self.prev_min_dist[agent_key] - dist_masked
                )
                # CRITICAL: Replace inf - inf = NaN cases (when food is eaten)
                # This happens when both prev_min_dist and dist_masked are inf
                shaping_reward_val = torch.nan_to_num(
                    shaping_reward_val, nan=0.0, posinf=0.0, neginf=0.0
                )
                shaping_reward += shaping_reward_val
                rews += shaping_reward_val

            # Update previous distance for next step
            self.prev_min_dist[agent_key] = dist_masked.clone()

        # Time penalty
        if self.penalise_by_time:
            time_penalty_val = torch.where(
                rews == 0,
                torch.tensor(-0.01, device=self.world.device),
                torch.tensor(0.0, device=self.world.device),
            )
            time_penalty += time_penalty_val
            rews += time_penalty_val

        # Accumulate reward components across agents (for env-level stats)
        self.reward_info["food_collection"] += food_reward
        self.reward_info["shaping_reward"] += shaping_reward
        self.reward_info["time_penalty"] += time_penalty

        # CRITICAL: Final safety check to prevent NaN/Inf from propagating to training
        # Clamp rewards to reasonable finite range
        rews = torch.clamp(rews, min=-10.0, max=10.0)

        # As a final failsafe, replace any remaining NaN/Inf
        # (e.g., from 0/0 divisions or other logical errors)
        rews = torch.nan_to_num(rews, nan=0.0, posinf=0.0, neginf=0.0)

        return rews

    def observation(self, agent: Agent):
        """
        Agent observations include:
        - Own position (2)
        - Own velocity (2)
        - Matching food: [rel_x, rel_y, eaten_status, in_range_flag] (4 values)
        - Lidar readings for matching food (12) - added automatically by VMAS

        Each agent only observes their own matching food (agent_0 sees food_0, etc.).
        The in_range_flag (0 or 1) tells the agent if the food is within sensing range.
        When in_range_flag=0, the relative position values are meaningless (set to 0).
        This makes agent capabilities (different obs_range) truly meaningful.
        """
        # Get agent's observation range
        obs_range = agent._obs_range

        # Extract agent index from agent name (e.g., "agent_0" -> 0)
        agent_idx = int(agent.name.split("_")[1])

        # Only observe the matching food for this agent
        if agent_idx < len(self.world.landmarks):
            matching_food = self.world.landmarks[agent_idx]

            # Relative position to matching food
            rel_pos = matching_food.state.pos - agent.state.pos

            # Check if food is within sensing range
            dist = torch.linalg.vector_norm(rel_pos, dim=1)
            in_range = dist < obs_range

            # If NOT in range, set position to zeros (meaningless when in_range=0)
            # If in range, provide real position
            masked_pos = torch.where(
                in_range.unsqueeze(-1).expand_as(rel_pos),
                rel_pos,
                torch.zeros_like(rel_pos),
            )

            # Eaten status: only meaningful when in_range=1
            masked_eaten = torch.where(
                in_range,
                matching_food.eaten.to(torch.int),
                torch.zeros_like(matching_food.eaten, dtype=torch.int),
            )

            # Add explicit in_range flag so agent knows when position is valid
            in_range_flag = in_range.to(torch.float32).unsqueeze(-1)

            food_obs = torch.cat(
                [masked_pos, masked_eaten.unsqueeze(-1), in_range_flag],
                dim=-1,
            )
        else:
            # Safety: if agent has no matching food, provide zeros
            food_obs = torch.zeros(
                (agent.state.pos.shape[0], 4),
                device=agent.state.pos.device,
                dtype=torch.float32,
            )

        # Get lidar sensor readings (if sensors exist)
        obs_parts = [agent.state.pos, agent.state.vel, food_obs]

        if hasattr(agent, "sensors") and agent.sensors:
            # Sensors are ordered: [agents_lidar, food_lidar]
            for sensor in agent.sensors:
                sensor_readings = sensor.measure()  # Get sensor readings
                # Debug: print sensor shape on first call
                if not hasattr(self, "_sensor_debug_printed"):
                    print(
                        f"DEBUG: Sensor {type(sensor).__name__} readings shape: {sensor_readings.shape}"
                    )
                obs_parts.append(sensor_readings)
            if not hasattr(self, "_sensor_debug_printed"):
                self._sensor_debug_printed = True
                print(
                    f"DEBUG: Total observation parts: {len(obs_parts)}, shapes: {[p.shape for p in obs_parts]}"
                )

        return torch.cat(obs_parts, dim=-1)

    def done(self):
        return torch.all(
            torch.stack(
                [landmark.eaten for landmark in self.world.landmarks],
                dim=1,
            ),
            dim=-1,
        )

    def info(self, agent: Agent) -> dict:
        """
        Return per-agent info including reward decomposition.
        VMAS calls this method for each agent and aggregates the results.
        """
        info_dict = {}

        # Only include reward decomposition if it's been computed
        if hasattr(self, "reward_info"):
            # Return decomposition as torch tensors (VMAS expects tensors, not floats)
            # Keep as tensors - do not call .item() which converts to Python float
            info_dict["reward_food_collection"] = self.reward_info[
                "food_collection"
            ].mean()
            info_dict["reward_shaping"] = self.reward_info["shaping_reward"].mean()
            info_dict["reward_time_penalty"] = self.reward_info["time_penalty"].mean()

        return info_dict

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        """
        Add extra rendering for LIDAR sensors.
        This method is called by VMAS to add custom visualization elements.
        """
        from vmas.simulator.rendering import Geom, Line, Color
        from typing import List
        import math

        geoms: List[Geom] = []

        # Render LIDAR rays for each agent
        for agent in self.world.agents:
            if agent.sensors:
                for sensor in agent.sensors:
                    # LIDAR sensors have a built-in render method
                    sensor_geoms = sensor.render(env_index=env_index)
                    geoms += sensor_geoms

            # Also render a circle showing the agent's lidar range for clarity
            from vmas.simulator.rendering import make_circle, Transform

            # Get agent position and lidar range
            agent_pos = agent.state.pos[env_index].cpu().numpy()
            lidar_range = agent._obs_range

            # Create a circle representing the lidar range
            range_circle = make_circle(radius=lidar_range, res=32, filled=False)
            range_circle.set_color(0.7, 0.7, 0.7, 0.3)  # Light gray, semi-transparent
            range_circle.add_attr(Transform(translation=(agent_pos[0], agent_pos[1])))
            geoms.append(range_circle)

        return geoms


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=4,
        share_reward=False,
        penalise_by_time=False,  # Fixed typo
        distance_shaping_coef=1.0,  # <-- Increased from 0.1 default
    )
