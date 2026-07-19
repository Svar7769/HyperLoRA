#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.package_width = kwargs.pop("package_width", 0.6)
        self.package_length = kwargs.pop("package_length", 0.6)
        self.package_mass = kwargs.pop("package_mass", 50)

        # Store mass range for color brightness calculation
        self.package_mass_range = kwargs.pop("package_mass_range", [1, 100])

        # Agent capabilities: heterogeneous max_speed and u_multiplier (force/push strength)
        agent_capabilities = kwargs.pop("agent_capabilities", None)
        if agent_capabilities is None:
            self.agent_speeds = [0.5] * n_agents
            self.agent_force_multipliers = [0.5] * n_agents
        else:
            self.agent_speeds = agent_capabilities.get("speed", [0.5] * n_agents)
            self.agent_force_multipliers = agent_capabilities.get(
                "force_multiplier", [0.5] * n_agents
            )

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.shaping_factor = 100

        # Make world
        world = World(
            batch_dim,
            device,
            contact_margin=6e-3,
            substeps=5,
            collision_force=500,
        )
        # Add agents
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(0.03),
                u_multiplier=self.agent_force_multipliers[i],
                max_speed=self.agent_speeds[i],
            )
            agent._max_speed = self.agent_speeds[i]  # for context extraction
            agent._force_multiplier = self.agent_force_multipliers[
                i
            ]  # for context extraction
            world.add_agent(agent)
        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.09),
            color=Color.YELLOW,
        )
        world.add_landmark(goal)

        self.package = Landmark(
            name=f"package {i}",
            collide=True,
            movable=True,
            mass=self.package_mass,
            shape=Box(
                length=self.package_length,
                width=self.package_width,
                hollow=True,
            ),
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)

        return world

    def update_agent_capabilities(self, agent_capabilities):
        """
        Update agent capabilities without recreating the world.

        Args:
            agent_capabilities: dict with:
              'speed'            – list of max_speed values per agent
              'force_multiplier' – list of u_multiplier (force/push) values per agent
        """
        if agent_capabilities is None:
            return

        n_agents = len(self.world.agents)
        self.agent_speeds = agent_capabilities.get("speed", [0.5] * n_agents)
        self.agent_force_multipliers = agent_capabilities.get(
            "force_multiplier", [0.5] * n_agents
        )

        for i, agent in enumerate(self.world.agents):
            agent._max_speed = self.agent_speeds[i]
            agent._force_multiplier = self.agent_force_multipliers[i]
            # u_multiplier scales the action force applied to the agent.
            # agent.u_multiplier is a read-only property that delegates to
            # agent.action._u_multiplier, so we must update it directly and
            # invalidate the cached tensor so the new value takes effect.
            agent.action._u_multiplier = self.agent_force_multipliers[i]
            agent.action._u_multiplier_tensor = None  # invalidate cached tensor

    def update_package_properties(
        self, package_mass=None, package_width=None, package_length=None
    ):
        """
        Update package properties dynamically during an episode.
        Returns True if any property was changed, False otherwise.

        Args:
            package_mass: New mass for the package (None = no change)
            package_width: New width for the package (None = no change)
            package_length: New length for the package (None = no change)

        Returns:
            bool: True if any property changed, False otherwise
        """
        changed = False
        size_changed = False

        if package_mass is not None and package_mass != self.package_mass:
            self.package_mass = package_mass
            self.package.mass = package_mass
            changed = True

        if package_width is not None and package_width != self.package_width:
            self.package_width = package_width
            # Update the shape width
            self.package.shape._width = package_width
            self.package.shape._width_tensor = None  # Invalidate cached tensor
            changed = True
            size_changed = True

        if package_length is not None and package_length != self.package_length:
            self.package_length = package_length
            # Update the shape length
            self.package.shape._length = package_length
            self.package.shape._length_tensor = None  # Invalidate cached tensor
            changed = True
            size_changed = True

        # Always reposition agents when any property changes (including mass)
        # to prevent agents from drifting outside due to physics changes
        if changed:
            self._reposition_agents_inside_package()

        return changed

    def _reposition_agents_inside_package(self):
        """
        Reposition all agents to ensure they stay inside the package boundaries.
        Handles edge cases where package might be smaller than agent size.
        """
        package_pos = self.package.state.pos  # (batch_dim, 2)

        for agent in self.world.agents:
            # Get agent position relative to package center
            rel_pos = agent.state.pos - package_pos  # (batch_dim, 2)

            # Calculate maximum allowed distance from center
            # Ensure bounds are positive (at least 0 if package is very small)
            max_x = max(0.0, self.package_length / 2 - agent.shape.radius)
            max_y = max(0.0, self.package_width / 2 - agent.shape.radius)

            # If package is too small for the agent, place agent at center
            if max_x == 0.0:
                clamped_x = torch.zeros_like(rel_pos[:, 0:1])
            else:
                clamped_x = torch.clamp(rel_pos[:, 0:1], -max_x, max_x)

            if max_y == 0.0:
                clamped_y = torch.zeros_like(rel_pos[:, 1:2])
            else:
                clamped_y = torch.clamp(rel_pos[:, 1:2], -max_y, max_y)

            # Update agent position (absolute = package_pos + clamped_relative)
            new_pos = torch.cat([clamped_x, clamped_y], dim=1) + package_pos
            agent.set_pos(new_pos, batch_index=None)

    def get_package_properties(self):
        """
        Get current package properties as a dict.

        Returns:
            dict with keys: 'mass', 'width', 'length'
        """
        return {
            "mass": self.package_mass,
            "width": self.package_width,
            "length": self.package_length,
        }

    def reset_world_at(self, env_index: int = None):
        package_pos = torch.zeros(
            (
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p)
            ),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -1.0,
            1.0,
        )

        self.package.set_pos(
            package_pos,
            batch_index=env_index,
        )

        # Calculate safe spawning bounds for agents inside package
        for agent in self.world.agents:
            # Calculate maximum distance from center, ensuring non-negative
            max_x_offset = max(0.0, self.package_length / 2 - agent.shape.radius)
            max_y_offset = max(0.0, self.package_width / 2 - agent.shape.radius)

            # Create position tensor
            shape_for_batch = (
                (1, 1) if env_index is not None else (self.world.batch_dim, 1)
            )

            # If package is large enough, spawn randomly; otherwise spawn at center
            if max_x_offset > 0:
                x_pos = torch.zeros(
                    shape_for_batch,
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(-max_x_offset, max_x_offset)
            else:
                x_pos = torch.zeros(
                    shape_for_batch,
                    device=self.world.device,
                    dtype=torch.float32,
                )

            if max_y_offset > 0:
                y_pos = torch.zeros(
                    shape_for_batch,
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(-max_y_offset, max_y_offset)
            else:
                y_pos = torch.zeros(
                    shape_for_batch,
                    device=self.world.device,
                    dtype=torch.float32,
                )

            agent.set_pos(
                torch.cat([x_pos, y_pos], dim=1) + package_pos,
                batch_index=env_index,
            )

        self.package.goal.set_pos(
            torch.zeros(
                (
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p)
                ),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                -1.0,
                1.0,
            ),
            batch_index=env_index,
        )

        if env_index is None:
            self.package.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
            self.package.on_goal = torch.zeros(
                self.world.batch_dim,
                dtype=torch.bool,
                device=self.world.device,
            )
        else:
            self.package.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )
            self.package.on_goal[env_index] = False

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )

            self.package.dist_to_goal = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )
            self.package.on_goal = self.world.is_overlapping(
                self.package, self.package.goal
            )

            # Adjust color brightness based on package mass
            # Use the actual mass range from config
            # Higher mass = darker color (lower brightness)
            min_mass = float(self.package_mass_range[0])
            max_mass = float(self.package_mass_range[1])

            # Avoid division by zero
            if max_mass - min_mass > 0:
                normalized_mass = max(
                    0.0,
                    min(1.0, (self.package_mass - min_mass) / (max_mass - min_mass)),
                )
            else:
                normalized_mass = 0.0

            # Color gradient: almost white (min mass) to pure red (max mass)
            # At min: (255, 250, 250) almost white
            # At max: (255, 0, 0) pure red
            red_channel = 255.0
            green_channel = 250.0 * (1.0 - normalized_mass)
            blue_channel = 250.0 * (1.0 - normalized_mass)

            # Normalize to [0, 1] range (VMAS uses [0, 1] for colors)
            color_rgb = torch.tensor(
                [red_channel / 255.0, green_channel / 255.0, blue_channel / 255.0],
                device=self.world.device,
                dtype=torch.float32,
            )

            # Set package color (RED gradient when not on goal)
            self.package.color = color_rgb.unsqueeze(0).repeat(self.world.batch_dim, 1)

            # GREEN color when on goal (also with same gradient)
            if self.package.on_goal.any():
                # Apply same gradient logic to green: almost white to pure green
                green_red = 250.0 * (1.0 - normalized_mass) / 255.0
                green_green = 255.0 / 255.0
                green_blue = 250.0 * (1.0 - normalized_mass) / 255.0
                color_green = torch.tensor(
                    [green_red, green_green, green_blue],
                    device=self.world.device,
                    dtype=torch.float32,
                )
                self.package.color[self.package.on_goal] = color_green

            package_shaping = self.package.dist_to_goal * self.shaping_factor
            self.rew[~self.package.on_goal] += (
                self.package.global_shaping[~self.package.on_goal]
                - package_shaping[~self.package.on_goal]
            )
            self.package.global_shaping = package_shaping

            self.rew[~self.package.on_goal] += (
                self.package.global_shaping[~self.package.on_goal]
                - package_shaping[~self.package.on_goal]
            )
            self.package.global_shaping = package_shaping

        return self.rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self.package.state.vel,
                self.package.state.pos - agent.state.pos,
                self.package.state.pos - self.package.goal.state.pos,
            ],
            dim=-1,
        )

    def done(self):
        return self.package.on_goal


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=4,
        package_width=0.6,
        package_length=0.6,
        package_mass=50,
    )
