#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_good_agents = kwargs.pop("num_good_agents", 1)
        num_adversaries = kwargs.pop("num_adversaries", 3)
        num_landmarks = kwargs.pop("num_landmarks", 2)
        self.shape_agent_rew = kwargs.pop("shape_agent_rew", False)
        self.shape_adversary_rew = kwargs.pop("shape_adversary_rew", False)
        self.agents_share_rew = kwargs.pop("agents_share_rew", False)
        self.adversaries_share_rew = kwargs.pop("adversaries_share_rew", True)
        self.observe_same_team = kwargs.pop("observe_same_team", True)
        self.observe_pos = kwargs.pop("observe_pos", True)
        self.observe_vel = kwargs.pop("observe_vel", True)
        self.bound = kwargs.pop("bound", 1.0)
        self.respawn_at_catch = kwargs.pop("respawn_at_catch", False)

        # Agent capabilities (speed and lidar_range) for HyperLoRA
        agent_capabilities = kwargs.pop("agent_capabilities", None)
        if agent_capabilities is None:
            # Default: all agents have same capabilities
            self.adversary_speeds = [1.0] * num_adversaries
            self.agent_speeds = [1.3] * num_good_agents
            self.adversary_lidar_ranges = [0.5] * num_adversaries
            self.agent_lidar_ranges = [0.6] * num_good_agents
        else:
            self.adversary_speeds = agent_capabilities.get(
                "adversary_speeds", [1.0] * num_adversaries
            )
            self.agent_speeds = agent_capabilities.get(
                "agent_speeds", [1.3] * num_good_agents
            )
            self.adversary_lidar_ranges = agent_capabilities.get(
                "adversary_lidar_ranges", [0.5] * num_adversaries
            )
            self.agent_lidar_ranges = agent_capabilities.get(
                "agent_lidar_ranges", [0.6] * num_good_agents
            )

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.visualize_semidims = False

        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=self.bound,
            y_semidim=self.bound,
            substeps=10,
            collision_force=500,
        )
        # set any world properties first
        num_agents = num_adversaries + num_good_agents
        self.adversary_radius = 0.075

        # Add agents
        for i in range(num_agents):
            adversary = True if i < num_adversaries else False
            name = f"adversary_{i}" if adversary else f"agent_{i - num_adversaries}"
            agent_idx = i if adversary else i - num_adversaries

            # Get agent-specific capabilities
            if adversary:
                agent_speed = self.adversary_speeds[agent_idx]
                agent_lidar_range = self.adversary_lidar_ranges[agent_idx]
            else:
                agent_speed = self.agent_speeds[agent_idx]
                agent_lidar_range = self.agent_lidar_ranges[agent_idx]

            agent = Agent(
                name=name,
                collide=True,
                shape=Sphere(radius=self.adversary_radius if adversary else 0.05),
                u_multiplier=3.0 if adversary else 4.0,
                max_speed=agent_speed,
                color=Color.RED if adversary else Color.GREEN,
                adversary=adversary,
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                shape=Sphere(radius=0.2),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def update_agent_capabilities(self, agent_capabilities):
        """
        Update agent capabilities (speed, lidar_range) without recreating the world.
        This allows for randomizing capabilities between episodes for generalization.

        Args:
            agent_capabilities: dict with 'adversary_speeds', 'agent_speeds',
                              'adversary_lidar_ranges', 'agent_lidar_ranges'
        """
        if agent_capabilities is None:
            return

        # Get capability lists for adversaries and good agents
        adversary_speeds = agent_capabilities.get("adversary_speeds", None)
        agent_speeds = agent_capabilities.get("agent_speeds", None)
        adversary_lidar_ranges = agent_capabilities.get("adversary_lidar_ranges", None)
        agent_lidar_ranges = agent_capabilities.get("agent_lidar_ranges", None)

        # Update internal state
        if adversary_speeds is not None:
            self.adversary_speeds = adversary_speeds
        if agent_speeds is not None:
            self.agent_speeds = agent_speeds
        if adversary_lidar_ranges is not None:
            self.adversary_lidar_ranges = adversary_lidar_ranges
        if agent_lidar_ranges is not None:
            self.agent_lidar_ranges = agent_lidar_ranges

        # Update each agent's properties
        adversary_idx = 0
        good_agent_idx = 0
        for agent in self.world.agents:
            if agent.adversary:
                if adversary_speeds is not None:
                    agent._max_speed = self.adversary_speeds[adversary_idx]
                if adversary_lidar_ranges is not None:
                    agent._obs_range = self.adversary_lidar_ranges[adversary_idx]
                adversary_idx += 1
            else:
                if agent_speeds is not None:
                    agent._max_speed = self.agent_speeds[good_agent_idx]
                if agent_lidar_ranges is not None:
                    agent._obs_range = self.agent_lidar_ranges[good_agent_idx]
                good_agent_idx += 1

    def reset_world_at(self, env_index: int = None):
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
                    -self.bound,
                    self.bound,
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
                    -(self.bound - 0.1),
                    self.bound - 0.1,
                ),
                batch_index=env_index,
            )

    def is_collision(self, agent1: Agent, agent2: Agent):
        delta_pos = agent1.state.pos - agent2.state.pos
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)
        dist_min = agent1.shape.radius + agent2.shape.radius
        return dist < dist_min

    # return all agents that are not adversaries
    def good_agents(self):
        return [agent for agent in self.world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self):
        return [agent for agent in self.world.agents if agent.adversary]

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            for a in self.world.agents:
                a.rew = (
                    self.adversary_reward(a) if a.adversary else self.agent_reward(a)
                )
            self.agents_rew = torch.stack(
                [a.rew for a in self.good_agents()], dim=-1
            ).sum(-1)
            self.adverary_rew = torch.stack(
                [a.rew for a in self.adversaries()], dim=-1
            ).sum(-1)
            if self.respawn_at_catch:
                for a in self.good_agents():
                    for adv in self.adversaries():
                        coll = self.is_collision(a, adv)
                        a.state.pos[coll] = torch.zeros(
                            (self.world.batch_dim, self.world.dim_p),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-self.bound, self.bound,)[coll]
                        a.state.vel[coll] = 0.0

        if agent.adversary:
            if self.adversaries_share_rew:
                return self.adverary_rew
            else:
                return agent.rew
        else:
            if self.agents_share_rew:
                return self.agents_rew
            else:
                return agent.rew

    def agent_reward(self, agent: Agent):
        # Good agent: survive and stay mobile, avoid walls
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        # Survival reward - encourage staying alive
        rew += 1.0

        # Movement reward - encourage active evasion
        vel_magnitude = torch.linalg.vector_norm(agent.state.vel, dim=-1)
        rew += 0.5 * vel_magnitude

        # Wall penalty - stay away from boundaries
        dist_to_wall = self.bound - torch.max(torch.abs(agent.state.pos), dim=-1)[0]
        near_wall = dist_to_wall < 0.2  # Within 0.2 of wall
        rew[near_wall] -= 5.0

        # Small reward for staying reasonably centered (but not dominant)
        dist_from_center = torch.linalg.vector_norm(agent.state.pos, dim=-1)
        center_reward = torch.clamp(1.0 - dist_from_center / self.bound, 0.0, 1.0)
        rew += center_reward * 0.5

        adversaries = self.adversaries()
        if self.shape_agent_rew:
            # Reward distance from adversaries
            for adv in adversaries:
                rew += 0.1 * torch.linalg.vector_norm(
                    agent.state.pos - adv.state.pos, dim=-1
                )
        if agent.collide:
            for a in adversaries:
                rew[
                    self.is_collision(a, agent)
                ] -= 10.0  # High penalty for being caught

        return rew

    def adversary_reward(self, agent: Agent):
        # Adversary: chase actively, close distance, catch without corner-trapping
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        # Movement reward - encourage active chasing
        vel_magnitude = torch.linalg.vector_norm(agent.state.vel, dim=-1)
        rew += 0.5 * vel_magnitude

        # Penalty for being near walls (discourage corner-trapping)
        dist_to_wall = self.bound - torch.max(torch.abs(agent.state.pos), dim=-1)[0]
        near_wall = dist_to_wall < 0.2
        rew[near_wall] -= 2.0

        agents = self.good_agents()
        if self.shape_adversary_rew:
            # Reward for getting closer to good agent
            min_dist = torch.min(
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            a.state.pos - agent.state.pos,
                            dim=-1,
                        )
                        for a in agents
                    ],
                    dim=-1,
                ),
                dim=-1,
            )[0]
            # Reward inversely proportional to distance (closer is better)
            rew += (2.0 - min_dist) * 0.5  # Max reward when distance is 0

        if agent.collide:
            for ag in agents:
                rew[
                    self.is_collision(ag, agent)
                ] += 3.0  # Reduced from 5.0 to balance with other rewards
        return rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)

        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            # All agents observe all other agents' positions and velocities
            other_pos.append(other.state.pos - agent.state.pos)
            if self.observe_vel:
                other_vel.append(other.state.vel)

        return torch.cat(
            [
                *([agent.state.vel] if self.observe_vel else []),
                *([agent.state.pos] if self.observe_pos else []),
                *entity_pos,
                *other_pos,
                *other_vel,
            ],
            dim=-1,
        )

    def done(self):
        # Episode ends when any good agent is caught by any adversary
        # Returns a boolean tensor of shape (batch_dim,)
        for good_agent in self.good_agents():
            for adversary in self.adversaries():
                # If collision detected in any batch, that batch is done
                collision = self.is_collision(good_agent, adversary)
                if collision.any():
                    # Return True for all batches where collision occurred
                    done = collision
                    # Check remaining adversaries for this good agent
                    for adv in self.adversaries():
                        if adv != adversary:
                            done = done | self.is_collision(good_agent, adv)
                    return done

        # No collisions in any batch
        return torch.zeros(
            self.world.batch_dim, dtype=torch.bool, device=self.world.device
        )

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        # Perimeter
        for i in range(4):
            geom = Line(
                length=2
                * ((self.bound - self.adversary_radius) + self.adversary_radius * 2)
            ).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                (
                    0.0
                    if i % 2
                    else (
                        self.bound + self.adversary_radius
                        if i == 0
                        else -self.bound - self.adversary_radius
                    )
                ),
                (
                    0.0
                    if not i % 2
                    else (
                        self.bound + self.adversary_radius
                        if i == 1
                        else -self.bound - self.adversary_radius
                    )
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)
        return geoms


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
