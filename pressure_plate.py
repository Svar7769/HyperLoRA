import typing
from typing import Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.simulator.dynamics.holonomic import Holonomic

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_ground_robots = kwargs.pop("n_ground_robots", 2)
        self.x_semidim = kwargs.pop("x_semidim", 2)
        self.y_semidim = kwargs.pop("y_semidim", 2)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.2)
        self.lidar_range = kwargs.pop(
            "lidar_range", 0.6
        )  # Increased for better detection
        self.drone_obs_range = kwargs.pop("drone_obs_range", 1.0)
        self.drone_noise_factor = kwargs.pop("drone_noise_factor", 0.01)
        self.ground_robot_noise_factor = kwargs.pop("ground_robot_noise_factor", 0.01)
        self.with_drone = kwargs.pop(
            "with_drone", False
        )  # Default to False for global obs
        self.grid_size = kwargs.pop("grid_size", 12)
        self.drone_occupancy_noise = kwargs.pop("drone_occupancy_noise", 0.0)
        self.use_global_obs = kwargs.pop(
            "use_global_obs", True
        )  # New: global observability mode

        self.pad_observations = kwargs.pop("pad_observations", False)

        _default_plate_reward = kwargs.pop("plate_reward", 0.1)
        self.plate_left_reward = kwargs.pop("plate_left_reward", _default_plate_reward)
        self.plate_right_reward = kwargs.pop(
            "plate_right_reward", _default_plate_reward
        )
        self.door_reward = kwargs.pop("door_reward", 1.0)
        self.goal_reward = kwargs.pop("goal_reward", 10.0)
        self.time_penalty = kwargs.pop("time_penalty", -0.01)
        self.distance_shaping_coef = kwargs.pop("distance_shaping_coef", 1.0)
        self.share_reward = kwargs.pop("share_reward", False)
        self.penalise_by_time = kwargs.pop("penalise_by_time", True)

        self.eval_mode = kwargs.pop("eval_mode", False)
        # "left"  → spawn left of the door (same as eval_mode)
        # "right" → spawn right of the door (goal side)
        # "both"  → full map (original training default)
        self.training_spawn_side = kwargs.pop("training_spawn_side", "both")
        self.reward_type = kwargs.pop("reward_type", "sparse")
        self.viewer_zoom = kwargs.pop(
            "viewer_zoom", max(self.x_semidim + 0.5, self.y_semidim + 0.5) ** 0.5
        )

        # Agent capabilities not used in this scenario; pop to avoid kwarg errors
        kwargs.pop("agent_capabilities", None)

        self.plate_radius = kwargs.pop("plate_radius", 0.15)
        self.plate_margin = kwargs.pop("plate_margin", 0.8)
        self.door_size = kwargs.pop("door_size", 0.6)
        self.goal_radius = kwargs.pop("goal_radius", 0.3)

        # Whether to randomize plate positions during reset (vs fixed corners)
        self.random_plate_positions = kwargs.pop("random_plate_positions", False)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.n_lidar_rays = 12
        self.wall_thickness = 0.1

        # World
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            collision_force=500,
            substeps=2,
            drag=0.25,
        )

        for i in range(self.n_ground_robots):
            # Add LiDAR only if not using global observations
            sensors = []
            if not self.use_global_obs:
                sensors.append(
                    Lidar(
                        world,
                        n_rays=self.n_lidar_rays,
                        max_range=self.lidar_range,
                        entity_filter=lambda e: "pressure_plate" in e.name,
                        noise=0.0,
                        render_color=Color.GREEN,
                    )
                )

            ground_robot = Agent(
                name=f"ground_robot_{i}",
                collide=True,
                shape=Sphere(radius=0.1),
                dynamics=Holonomic(),
                max_speed=1.0,
                sensors=sensors,
            )
            # render_shape defaults to the agent's shape (Sphere)
            ground_robot.color = Color.BLUE
            world.add_agent(ground_robot)

        if self.with_drone:
            drone = Agent(
                name="drone",
                collide=False,
                shape=Sphere(radius=0.05),
                dynamics=Holonomic(),
            )
            # render_shape defaults to the agent's shape (Sphere)
            drone.color = Color.BLUE
            world.add_agent(drone)

        # --- Landmarks ---

        # Walls
        # Top Wall
        self.wall_top = Landmark(
            name="wall_top",
            collide=True,
            movable=False,
            shape=Box(
                length=self.wall_thickness, width=self.y_semidim - (self.door_size / 2)
            ),
            color=Color.BLACK,
        )
        world.add_landmark(self.wall_top)

        # Bottom Wall
        self.wall_bottom = Landmark(
            name="wall_bottom",
            collide=True,
            movable=False,
            shape=Box(
                length=self.wall_thickness, width=self.y_semidim - (self.door_size / 2)
            ),
            color=Color.BLACK,
        )
        world.add_landmark(self.wall_bottom)

        # Door (Movable conceptually, but we toggle collision/pos)
        self.door = Landmark(
            name="door",
            collide=True,
            movable=False,
            shape=Box(length=self.wall_thickness, width=self.door_size),
            color=Color.ORANGE,  # Changed color for visibility
        )
        world.add_landmark(self.door)

        # Pressure Plates
        # Use a small physical radius to allow robots to stand "on" it (visually)
        # but keep collide=True for LiDAR detection.
        physical_plate_radius = 0.02

        self.plate_left = Landmark(
            name="pressure_plate_left",
            collide=True,
            movable=False,
            shape=Sphere(radius=physical_plate_radius),  # Small physical collider
            color=Color.LIGHT_GREEN,
        )
        self.plate_left.render_shape = Sphere(
            radius=self.plate_radius
        )  # Large visual representation
        world.add_landmark(self.plate_left)

        self.plate_right = Landmark(
            name="pressure_plate_right",
            collide=True,
            movable=False,
            shape=Sphere(radius=physical_plate_radius),  # Small physical collider
            color=Color.LIGHT_GREEN,
        )
        self.plate_right.render_shape = Sphere(
            radius=self.plate_radius
        )  # Large visual representation
        world.add_landmark(self.plate_right)

        # Goal
        self.goal = Landmark(
            name="goal",
            collide=False,
            movable=False,
            shape=Sphere(radius=self.goal_radius),
            color=Color.YELLOW,
        )
        world.add_landmark(self.goal)

        # --- State Tracking ---
        self.door_open = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.agents_at_goal = torch.zeros(
            batch_dim, self.n_ground_robots, dtype=torch.bool, device=device
        )
        self.prev_agents_at_goal = torch.zeros(
            batch_dim, self.n_ground_robots, dtype=torch.bool, device=device
        )

        # Distance shaping tracking for global obs
        self.prev_dist_to_plates = torch.zeros(
            batch_dim, self.n_ground_robots, device=device
        )
        self.prev_dist_to_goal = torch.zeros(
            batch_dim, self.n_ground_robots, device=device
        )

        self.occupancy_grid = torch.zeros(
            batch_dim, self.grid_size, self.grid_size, device=device, dtype=torch.float
        )
        self.plate_grid = torch.zeros(
            batch_dim, self.grid_size, self.grid_size, device=device, dtype=torch.long
        )
        self.plate_state = torch.zeros(batch_dim, 2, dtype=torch.float32, device=device)

        # Per-plate activation flags (set in post_step, read in reward)
        self.is_on_left = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.is_on_right = torch.zeros(batch_dim, dtype=torch.bool, device=device)

        # State for sequential reward
        self.stage = torch.zeros(batch_dim, dtype=torch.long, device=device)
        self.robot_left_idx = torch.full(
            (batch_dim,), -1, dtype=torch.long, device=device
        )
        self.robot_right_idx = torch.full(
            (batch_dim,), -1, dtype=torch.long, device=device
        )
        # Track if right plate has been activated (persistent across steps)
        self.right_plate_ever_activated = torch.zeros(
            batch_dim, dtype=torch.bool, device=device
        )

        # Discovery tracking for pressure plates only (0:Left, 1:Right)
        self.n_targets = 2  # Plates only
        self.discovered_targets = torch.zeros(
            batch_dim, self.n_targets, device=device, dtype=torch.bool
        )
        self.target_discovery_times = torch.full(
            (batch_dim, self.n_targets), float("inf"), device=device, dtype=torch.float
        )
        self.target_collection_times = torch.full(
            (batch_dim, self.n_targets), float("inf"), device=device, dtype=torch.float
        )
        self.current_step = 0

        # For specialized reward
        self.prev_occupancy_grid = self.occupancy_grid.clone()

        # Observation dimensions
        # Drone: Pos(2) + Vel(2) + Grid(size^2) + plate_grid(size^2) + GroundRobots(N*2) + Door(2) + Goal(2) + plate_state(2)
        # Ground (LiDAR): Pos(2) + Vel(2) + Lidar(N_rays) + TargetIDs(N_rays) + Door(2) + Goal(2) + plate_state(2) + am_i_on_left(1) + am_i_on_right(1)
        # Ground (Global): Pos(2) + Vel(2) + plate_left(2) + is_on_left(1) + plate_right(2) + is_on_right(1) + door(2) + door_open(1) + goal(2) + am_i_on_left(1) + am_i_on_right(1) + other_pos((N-1)*2) + other_vel((N-1)*2)

        # Calculating max obs dim for padding
        drone_dim = (
            4
            + self.grid_size**2
            + self.grid_size**2
            + (self.n_ground_robots * 2)
            + 4
            + 2
        )  # +4 for door/goal + 2 for plate_status
        robot_lidar_dim = (
            4 + 2 * self.n_lidar_rays + 4 + 2 + 2
        )  # +4 for door/goal + 2 for plate_status + 2 for am_i_on_plate flags
        robot_global_dim = (
            4 + 3 + 3 + 3 + 2 + 2 + (self.n_ground_robots - 1) * 4
        )  # pos+vel + plate_left+flag + plate_right+flag + door+flag + goal + am_i_flags + others
        self.max_obs_dim = max(drone_dim, robot_lidar_dim, robot_global_dim)

        return world

    def reset_world_at(self, env_index: int = None):
        # 1. Set Static Positions (Walls, Door, Goal)
        self.wall_top.set_pos(
            torch.tensor(
                [0.0, (self.y_semidim + self.door_size / 2) / 2],
                device=self.world.device,
            ),
            env_index,
        )
        self.wall_bottom.set_pos(
            torch.tensor(
                [0.0, -(self.y_semidim + self.door_size / 2) / 2],
                device=self.world.device,
            ),
            env_index,
        )

        # Door at 0,0 initially
        self.door.set_pos(torch.tensor([0.0, 0.0], device=self.world.device), env_index)

        # Goal right after the door (on the right side, close to door)
        goal_x = 0.4  # Just after the door
        goal_y = 0.0
        self.goal.set_pos(
            torch.tensor([goal_x, goal_y], device=self.world.device), env_index
        )

        # 2. Fixed Plate Positions: left plate top-left, right plate bottom-right
        if env_index is None:
            batch_size = self.world.batch_dim
        else:
            batch_size = 1

        # Helper for random pos (still used for agent spawning)
        def get_random_pos(x_min, x_max, y_min, y_max):
            x = torch.empty(batch_size, device=self.world.device).uniform_(x_min, x_max)
            y = torch.empty(batch_size, device=self.world.device).uniform_(y_min, y_max)
            return torch.stack([x, y], dim=-1)

        margin = self.plate_margin

        # Plate Positions: Fixed corners (default) or randomized (if flag enabled)
        if self.random_plate_positions:
            # Randomize plate positions within the arena (respecting margin)
            pos_left = get_random_pos(
                -self.x_semidim + margin,
                self.x_semidim - margin,
                -self.y_semidim + margin,
                self.y_semidim - margin,
            )
            pos_right = get_random_pos(
                -self.x_semidim + margin,
                self.x_semidim - margin,
                -self.y_semidim + margin,
                self.y_semidim - margin,
            )
        else:
            # Fixed positions: use plate_margin from config
            # Larger margin brings plates toward center and goal
            pos_left = torch.tensor(
                [
                    [
                        -self.x_semidim + margin,
                        self.y_semidim - margin,
                    ]
                ],
                device=self.world.device,
            ).expand(batch_size, -1)
            pos_right = torch.tensor(
                [
                    [
                        self.x_semidim - margin,
                        -self.y_semidim + margin,
                    ]
                ],
                device=self.world.device,
            ).expand(batch_size, -1)

        self.plate_left.set_pos(pos_left, env_index)
        self.plate_right.set_pos(pos_right, env_index)

        # 3. Place Agents: spawn region depends on eval_mode / training_spawn_side
        # eval_mode always forces left side regardless of training_spawn_side
        if self.eval_mode or self.training_spawn_side == "left":
            spawn_x_min = -self.x_semidim + margin
            spawn_x_max = -margin
        elif self.training_spawn_side == "right":
            spawn_x_min = margin
            spawn_x_max = self.x_semidim - margin
        else:  # "both" – original training default
            spawn_x_min = -self.x_semidim + margin
            spawn_x_max = self.x_semidim - margin
        for agent in self.world.agents:
            pos = get_random_pos(
                spawn_x_min,
                spawn_x_max,
                -self.y_semidim + margin,
                self.y_semidim - margin,
            )
            agent.set_pos(pos, env_index)

        # Reset States
        if env_index is None:
            self.door_open[:] = False
            self.agents_at_goal[:] = False
            self.prev_agents_at_goal[:] = False
            self.prev_dist_to_plates[:] = 0
            self.prev_dist_to_goal[:] = 0
            self.occupancy_grid[:] = 0
            self.plate_grid[:] = 0
            self.stage[:] = 0
            self.robot_left_idx[:] = -1
            self.robot_right_idx[:] = -1
            self.right_plate_ever_activated[:] = False
            self.discovered_targets[:] = False
            self.target_discovery_times[:] = float("inf")
            self.target_collection_times[:] = float("inf")
            self.current_step = 0
            if self.reward_type == "specialized":
                self.prev_occupancy_grid = self.occupancy_grid.clone()
        else:
            self.door_open[env_index] = False
            self.agents_at_goal[env_index] = False
            self.prev_agents_at_goal[env_index] = False
            self.prev_dist_to_plates[env_index] = 0
            self.prev_dist_to_goal[env_index] = 0
            self.occupancy_grid[env_index] = 0
            self.plate_grid[env_index] = 0
            self.plate_state[env_index] = 0
            self.stage[env_index] = 0
            self.robot_left_idx[env_index] = -1
            self.robot_right_idx[env_index] = -1
            self.right_plate_ever_activated[env_index] = False
            self.discovered_targets[env_index] = False
            self.target_discovery_times[env_index] = float("inf")
            self.target_collection_times[env_index] = float("inf")
            if self.reward_type == "specialized":
                self.prev_occupancy_grid[env_index] = self.occupancy_grid[
                    env_index
                ].clone()

    def post_step(self, env_index: int = None):
        # Check plates
        # Identify Ground Robots
        ground_robots = [a for a in self.world.agents if "ground_robot" in a.name]

        # Check Left Plate
        is_on_left = torch.zeros(
            self.world.batch_dim, dtype=torch.bool, device=self.world.device
        )
        for robot in ground_robots:
            dist = torch.linalg.norm(
                robot.state.pos - self.plate_left.state.pos, dim=-1
            )
            # Reduced tolerance to 0.05
            is_on_left |= dist < (self.plate_radius + robot.shape.radius + 0.05)

        # Color Feedback for Left Plate
        if is_on_left[0]:
            self.plate_left.color = Color.RED
        else:
            self.plate_left.color = Color.LIGHT_GREEN

        # Check Right Plate
        is_on_right = torch.zeros(
            self.world.batch_dim, dtype=torch.bool, device=self.world.device
        )
        for robot in ground_robots:
            dist = torch.linalg.norm(
                robot.state.pos - self.plate_right.state.pos, dim=-1
            )
            # Reduced tolerance to 0.05
            is_on_right |= dist < (self.plate_radius + robot.shape.radius + 0.05)

        # Color Feedback for Right Plate
        if is_on_right[0]:
            self.plate_right.color = Color.RED
        else:
            self.plate_right.color = Color.LIGHT_GREEN

        # Update target_collection_times for plates (0: Left, 1: Right)
        for i, is_on_plate in enumerate([is_on_left, is_on_right]):
            newly_collected_plate = (
                is_on_plate  # Collection is simply being on the plate
            )
            self.target_collection_times[newly_collected_plate, i] = float(
                self.current_step
            )

        # Persist per-plate activation for use in reward()
        self.is_on_left = is_on_left
        self.is_on_right = is_on_right

        # Track if right plate has ever been activated (for stage transitions)
        self.right_plate_ever_activated |= is_on_right

        # Door Open Logic
        prev_door_open = self.door_open.clone()
        self.door_open = is_on_left | is_on_right
        # Door is not a "target" for discovery, but its state is used.

        # Move door based on state
        door_pos = torch.zeros((self.world.batch_dim, 2), device=self.world.device)
        door_pos[self.door_open] = torch.tensor(
            [1000.0, 1000.0], device=self.world.device
        )
        self.door.set_pos(door_pos, None)  # Set for all envs

        # Check Goal Reached
        for i, robot in enumerate(ground_robots):
            dist = torch.linalg.norm(robot.state.pos - self.goal.state.pos, dim=-1)
            is_at_goal = dist < (self.goal_radius + robot.shape.radius + 0.05)
            self.agents_at_goal[:, i] = is_at_goal

            # Goal is not a "target" for discovery.

    def reward(self, agent: Agent):
        if self.reward_type == "sparse":
            return self._reward_sparse(agent)
        elif self.reward_type == "dense":
            return self._reward_dense(agent)
        elif self.reward_type == "specialized":
            return self._reward_specialized(agent)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def _reward_sparse(self, agent: Agent):
        # Shared reward
        reward = torch.full(
            (self.world.batch_dim,), self.time_penalty, device=self.world.device
        )

        if agent.name == "drone":
            return reward  # Drone just shares context, maybe no specific sparse reward?

        # Ground robots
        # Let's give reward if ANY robot is at goal
        any_at_goal = self.agents_at_goal.any(dim=-1)
        reward[any_at_goal] += self.goal_reward

        # Reward for door open (plate held)
        reward[self.door_open] += self.plate_reward

        return reward

    def _reward_dense(self, agent: Agent):
        # Compute shared reward once per step
        if agent == self.world.agents[0]:
            self.reward_val = torch.zeros(
                self.world.batch_dim, device=self.world.device
            )

            ground_robots = [a for a in self.world.agents if "ground_robot" in a.name]
            all_pos = torch.stack(
                [a.state.pos for a in ground_robots], dim=1
            )  # [B, N, 2]

            # Distances
            # Plate Left (Fixed pos per env)
            dists_left = torch.linalg.norm(
                all_pos - self.plate_left.state.pos.unsqueeze(1), dim=-1
            )  # [B, N]
            dists_right = torch.linalg.norm(
                all_pos - self.plate_right.state.pos.unsqueeze(1), dim=-1
            )  # [B, N]

            # Goal
            dists_goal = torch.linalg.norm(
                all_pos - self.goal.state.pos.unsqueeze(1), dim=-1
            )  # [B, N]

            # Door target is (0,0)
            door_target = torch.zeros(
                (self.world.batch_dim, 2), device=self.world.device
            )
            dists_door = torch.linalg.norm(
                all_pos - door_target.unsqueeze(1), dim=-1
            )  # [B, N]

            # Masks
            stage0 = self.stage == 0
            stage1 = self.stage == 1
            stage2 = self.stage == 2

            batch_indices = torch.arange(self.world.batch_dim, device=self.world.device)

            # --- Stage 0 ---
            if stage0.any():
                # Closest to Left
                closest_vals, closest_indices = torch.min(dists_left, dim=1)  # [B], [B]

                # Reward closest to go to Left
                self.reward_val[stage0] -= closest_vals[stage0]

                # Others: Go to Door
                mask_others = torch.ones(
                    (self.world.batch_dim, self.n_ground_robots),
                    dtype=torch.bool,
                    device=self.world.device,
                )
                mask_others[batch_indices, closest_indices] = False

                # Sum dist to door for others
                dist_others_door = (dists_door * mask_others.float()).sum(dim=1)
                self.reward_val[stage0] -= dist_others_door[stage0]

                # Transition Logic
                # Check if closest is on plate (using same tolerance as post_step logic roughly)
                is_on_plate = closest_vals < (self.plate_radius + 0.1 + 0.05)

                # Check if others passed door (x > 0)
                pos_x = all_pos[:, :, 0]
                passed_condition = (pos_x > 0.0) | (~mask_others)
                others_passed_all = passed_condition.all(dim=1)

                transition_0_1 = stage0 & is_on_plate & others_passed_all

                if transition_0_1.any():
                    self.stage[transition_0_1] = 1
                    self.robot_left_idx[transition_0_1] = closest_indices[
                        transition_0_1
                    ]

            # --- Stage 1 ---
            if stage1.any():
                # idx_L is fixed
                idx_L = self.robot_left_idx

                # Reward idx_L to go to Door
                dist_L_door = dists_door[batch_indices, idx_L]
                self.reward_val[stage1] -= dist_L_door[stage1]

                mask_L = torch.zeros(
                    (self.world.batch_dim, self.n_ground_robots),
                    dtype=torch.bool,
                    device=self.world.device,
                )
                mask_L[batch_indices, idx_L] = True
                mask_others = ~mask_L

                # Among others, find closest to Right Plate
                dists_right_masked = dists_right.clone()
                dists_right_masked[mask_L] = float("inf")

                closest_right_vals, closest_right_indices = torch.min(
                    dists_right_masked, dim=1
                )

                if self.n_ground_robots > 1:
                    self.reward_val[stage1] -= closest_right_vals[stage1]

                    # Others (not L, not R) -> Goal
                    mask_R = torch.zeros(
                        (self.world.batch_dim, self.n_ground_robots),
                        dtype=torch.bool,
                        device=self.world.device,
                    )
                    mask_R[batch_indices, closest_right_indices] = True

                    mask_rest = mask_others & (~mask_R)

                    dist_rest_goal = (dists_goal * mask_rest.float()).sum(dim=1)
                    self.reward_val[stage1] -= dist_rest_goal[stage1]

                # Transition Logic
                pos_x_L = all_pos[batch_indices, idx_L, 0]
                left_passed_door = pos_x_L > 0.0

                # Transition to stage 2 when right plate has been activated AND left agent passed door
                transition_1_2 = (
                    stage1 & self.right_plate_ever_activated & left_passed_door
                )

                if transition_1_2.any():
                    self.stage[transition_1_2] = 2

            # --- Stage 2 ---
            if stage2.any():
                # All to Goal
                self.reward_val[stage2] -= dists_goal[stage2].sum(dim=1)

        self.reward_val *= 0.01
        return self.reward_val

    def _reward_specialized(self, agent: Agent):
        reward = self._reward_dense(agent)

        # Drone exploration bonus
        if agent.name == "drone":
            newly_discovered_cells = (self.occupancy_grid != 0) & (
                self.prev_occupancy_grid == 0
            )
            reward += newly_discovered_cells.sum(dim=(-1, -2)) * 0.1
            self.prev_occupancy_grid = self.occupancy_grid.clone()

        return reward

    def observation(self, agent: Agent):
        # increment step counter once per step (on first agent's observation)
        is_first_agent = agent == self.world.agents[0]
        if is_first_agent:
            self.current_step += 1

        # Global observation mode: all agents see all entities (positions relative to this agent)
        if self.use_global_obs and "ground_robot" in agent.name:
            obs_parts = []
            agent_pos = agent.state.pos[:, :2]

            # Own state (absolute pos kept for world-grounding; vel already agent-centric)
            obs_parts.append(agent_pos)
            obs_parts.append(agent.state.vel[:, :2])

            ground_robots = [a for a in self.world.agents if "ground_robot" in a.name]

            # Plate left (relative)
            obs_parts.append(self.plate_left.state.pos[:, :2] - agent_pos)
            is_on_left = torch.zeros(
                self.world.batch_dim, dtype=torch.float32, device=self.world.device
            )
            for robot in ground_robots:
                dist = torch.linalg.norm(
                    robot.state.pos - self.plate_left.state.pos, dim=-1
                )
                is_on_left = torch.max(
                    is_on_left,
                    (dist < (self.plate_radius + robot.shape.radius)).float(),
                )
            obs_parts.append(is_on_left.unsqueeze(-1))

            # Plate right (relative)
            obs_parts.append(self.plate_right.state.pos[:, :2] - agent_pos)
            is_on_right = torch.zeros(
                self.world.batch_dim, dtype=torch.float32, device=self.world.device
            )
            for robot in ground_robots:
                dist = torch.linalg.norm(
                    robot.state.pos - self.plate_right.state.pos, dim=-1
                )
                is_on_right = torch.max(
                    is_on_right,
                    (dist < (self.plate_radius + robot.shape.radius)).float(),
                )
            obs_parts.append(is_on_right.unsqueeze(-1))

            # Door (relative; zeroed out when open since the entity is teleported far away)
            door_rel_pos = self.door.state.pos[:, :2] - agent_pos
            door_rel_pos = door_rel_pos * (~self.door_open).float().unsqueeze(-1)
            obs_parts.append(door_rel_pos)
            obs_parts.append(self.door_open.float().unsqueeze(-1))

            # Goal (relative)
            obs_parts.append(self.goal.state.pos[:, :2] - agent_pos)

            # Am I on a plate? (boolean flag for this specific agent)
            dist_to_left = torch.linalg.norm(
                agent.state.pos - self.plate_left.state.pos, dim=-1
            )
            dist_to_right = torch.linalg.norm(
                agent.state.pos - self.plate_right.state.pos, dim=-1
            )
            am_i_on_left = (
                dist_to_left < (self.plate_radius + agent.shape.radius + 0.05)
            ).float()
            am_i_on_right = (
                dist_to_right < (self.plate_radius + agent.shape.radius + 0.05)
            ).float()
            obs_parts.append(am_i_on_left.unsqueeze(-1))
            obs_parts.append(am_i_on_right.unsqueeze(-1))

            # Other agents: relative positions and absolute velocities
            for other in ground_robots:
                if other != agent:
                    obs_parts.append(other.state.pos[:, :2] - agent_pos)

            for other in ground_robots:
                if other != agent:
                    obs_parts.append(other.state.vel[:, :2])

            obs = torch.cat(obs_parts, dim=-1)
            return obs

        # 1. Drone Grid Observation
        if self.with_drone and agent.name == "drone":
            drone_pos = agent.state.pos[:, :2]

            cell_size = (2 * self.x_semidim) / self.grid_size
            grid_x = torch.linspace(
                -self.x_semidim + cell_size / 2,
                self.x_semidim - cell_size / 2,
                self.grid_size,
                device=self.world.device,
            )
            grid_y = torch.linspace(
                -self.y_semidim + cell_size / 2,
                self.y_semidim - cell_size / 2,
                self.grid_size,
                device=self.world.device,
            )
            grid_pos_x = grid_x.unsqueeze(0).expand(self.grid_size, -1)
            grid_pos_y = grid_y.unsqueeze(1).expand(-1, self.grid_size)
            grid_pos = torch.stack([grid_pos_x, grid_pos_y], dim=-1).expand(
                self.world.batch_dim, -1, -1, -1
            )  # [B, G, G, 2]

            # Visibility
            dist_to_cells = torch.linalg.norm(
                drone_pos.unsqueeze(1).unsqueeze(1) - grid_pos, dim=-1
            )
            visible_cells = dist_to_cells < self.drone_obs_range

            # Update Occupancy
            self.occupancy_grid[visible_cells] = -1  # Empty

            # Noise
            noise_mask = (
                torch.rand_like(self.occupancy_grid) < self.drone_occupancy_noise
            )
            self.occupancy_grid[visible_cells & noise_mask] *= -1

            # Mark Plates, Door, Goal on Grid
            targets_for_grid = [self.plate_left, self.plate_right, self.door, self.goal]
            for i, target_entity in enumerate(targets_for_grid):
                pos = target_entity.state.pos
                grid_indices_x = (
                    torch.floor((pos[:, 0] + self.x_semidim) / cell_size)
                    .long()
                    .clamp(0, self.grid_size - 1)
                )
                grid_indices_y = (
                    torch.floor((pos[:, 1] + self.y_semidim) / cell_size)
                    .long()
                    .clamp(0, self.grid_size - 1)
                )

                batch_indices = torch.arange(
                    self.world.batch_dim, device=self.world.device
                )
                target_mask_in_grid = torch.zeros_like(
                    self.occupancy_grid, dtype=torch.bool
                )
                target_mask_in_grid[batch_indices, grid_indices_y, grid_indices_x] = (
                    True
                )

                visible_target_cell = visible_cells & target_mask_in_grid
                self.occupancy_grid[visible_target_cell] = 1
                self.plate_grid[visible_target_cell] = (
                    i  # 0:LPlate, 1:RPlate, 2:Door, 3:Goal
                )

                # Update discovery for drone
                drone_discovered_mask = visible_target_cell.any(dim=(-1, -2))
                self._update_target_discovery(i, drone_discovered_mask)

            # Construct Obs
            ground_robots_pos = [
                a.state.pos for a in self.world.agents if "ground_robot" in a.name
            ]
            ground_robots_flat = (
                torch.cat(ground_robots_pos, dim=1)
                if ground_robots_pos
                else torch.tensor([], device=self.world.device)
            )

            obs = torch.cat(
                [
                    agent.state.pos[:, :2],
                    agent.state.vel[:, :2],
                    ground_robots_flat,
                    self.occupancy_grid.flatten(1),
                    self.plate_grid.flatten(1),  # Added plate grid
                    self.door.state.pos[:, :2],
                    self.goal.state.pos[:, :2],
                    self.plate_state,  # Added plate status at the end
                ],
                dim=-1,
            )

        elif "ground_robot" in agent.name:
            lidar = agent.sensors[0]
            lidar_measurements = lidar.measure()

            target_ids = torch.full(
                (self.world.batch_dim, lidar._angles.shape[1]),
                -1,
                device=self.world.device,
                dtype=torch.long,
            )

            targets_for_lidar = [
                self.plate_left,
                self.plate_right,
                self.door,
                self.goal,
            ]
            angles = lidar._angles[0]

            for i, target_entity in enumerate(targets_for_lidar):
                delta_pos = target_entity.state.pos - agent.state.pos
                dist = torch.linalg.norm(delta_pos, dim=-1)
                angle = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])

                angle_diff = torch.abs(angle.unsqueeze(1) - angles)
                angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)

                ang_radius = torch.atan2(
                    torch.full_like(dist, target_entity.shape.radius), dist
                )
                is_aligned = angle_diff < (
                    torch.pi / self.n_lidar_rays
                ) + ang_radius.unsqueeze(1)

                dist_diff = torch.abs(lidar_measurements - dist.unsqueeze(1))
                is_close = dist_diff <= target_entity.shape.radius * 1.5  # Tolerance

                is_hit = is_aligned & is_close
                target_ids[is_hit] = i

                # Update discovery for ground robot LiDAR
                lidar_discovered_mask = is_hit.any(dim=-1)
                self._update_target_discovery(i, lidar_discovered_mask)

            # Am I on a plate? (boolean flag for this specific agent)
            dist_to_left = torch.linalg.norm(
                agent.state.pos - self.plate_left.state.pos, dim=-1
            )
            dist_to_right = torch.linalg.norm(
                agent.state.pos - self.plate_right.state.pos, dim=-1
            )
            am_i_on_left = (
                dist_to_left < (self.plate_radius + agent.shape.radius + 0.05)
            ).float()
            am_i_on_right = (
                dist_to_right < (self.plate_radius + agent.shape.radius + 0.05)
            ).float()

            obs = torch.cat(
                [
                    agent.state.pos[:, :2],
                    agent.state.vel[:, :2],
                    lidar_measurements,
                    target_ids,
                    self.door.state.pos[:, :2],
                    self.goal.state.pos[:, :2],
                    self.plate_state,  # Added plate status at the end
                    am_i_on_left.unsqueeze(-1),
                    am_i_on_right.unsqueeze(-1),
                ],
                dim=-1,
            )

        # Padding
        if self.pad_observations and obs.shape[-1] < self.max_obs_dim:
            padding = torch.zeros(
                obs.shape[0], self.max_obs_dim - obs.shape[-1], device=obs.device
            )
            obs = torch.cat([obs, padding], dim=-1)

        return obs

    def done(self):
        # Done if all ground robots are at goal
        return self.agents_at_goal.all(dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        percentage_at_goal = self.agents_at_goal.float().mean(dim=-1)
        return {
            "discovered_targets": self.discovered_targets,
            "percentage_agents_at_goal": percentage_at_goal,
        }

    def _update_target_discovery(self, target_idx: int, env_mask: Tensor):
        """Update discovery tracking for a target.

        Args:
            target_idx: Index of the target being discovered (0:LPlate, 1:RPlate, 2:Door, 3:Goal)
            env_mask: Boolean tensor [B] indicating which envs discovered the target
        """
        newly_discovered = env_mask & ~self.discovered_targets[:, target_idx]
        self.discovered_targets[:, target_idx] |= env_mask
        self.target_discovery_times[newly_discovered, target_idx] = float(
            self.current_step
        )

    def _update_target_collection(self, target_idx: int, env_mask: Tensor):
        """Update collection tracking when ground robot collects a target.

        Args:
            target_idx: Index of the target being collected (0:LPlate, 1:RPlate, 2:Door, 3:Goal)
            env_mask: Boolean tensor [B] indicating which envs collected the target
        """
        # For pressure plate scenario, 'collection' is when an agent is on it or door is open/goal reached.
        # This is already handled in post_step by updating target_collection_times.
        # This helper is more for the cooperative search scenario.
        # However, to maintain consistency in terminology, we can still define it.
        newly_collected = (
            env_mask & ~self.discovered_targets[:, target_idx]
        )  # Assuming discovered_targets acts as a 'found' flag initially
        self.target_collection_times[newly_collected, target_idx] = float(
            self.current_step
        )

    def get_benchmark_data(self) -> Dict[str, Tensor]:
        """Get data needed for benchmark/optimality analysis.

        Should be called after running an episode to get discovery and collection times.

        Returns:
            Dictionary containing:
                - ground_robot_pos: [B, N_robots, 2] ground robot positions
                - drone_pos: [B, 2] drone position (if with_drone)
                - target_positions: [B, 4, 2] all conceptual target positions (LPlate, RPlate, Door, Goal)
                - target_discovery_times: [B, 4] step when each target became available
                - target_collection_times: [B, 4] step when ground robot collected each target
                - discovered_targets: [B, 4] bool mask of discovered targets
        """
        ground_robots = [a for a in self.world.agents if "ground_robot" in a.name]
        ground_robot_pos = torch.stack(
            [r.state.pos[:, :2] for r in ground_robots], dim=1
        )  # [B, N_robots, 2]

        all_target_pos = torch.stack(
            [
                self.plate_left.state.pos[:, :2],
                self.plate_right.state.pos[:, :2],
                self.door.state.pos[:, :2],
                self.goal.state.pos[:, :2],
            ],
            dim=1,
        )  # [B, 4, 2]

        result = {
            "ground_robot_pos": ground_robot_pos.clone(),
            "target_positions": all_target_pos.clone(),
            "target_discovery_times": self.target_discovery_times.clone(),
            "target_collection_times": self.target_collection_times.clone(),
            "discovered_targets": self.discovered_targets.clone(),
        }

        if self.with_drone:
            drone = [a for a in self.world.agents if a.name == "drone"][0]
            result["drone_pos"] = drone.state.pos[:, :2].clone()

        return result

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = []

        # Draw pressure plates at their visual radius (physical collider is tiny at 0.02,
        # so the built-in renderer shows them as invisible dots — we draw them here instead)
        is_on_left = getattr(
            self,
            "is_on_left",
            torch.zeros(
                self.world.batch_dim, dtype=torch.bool, device=self.world.device
            ),
        )
        is_on_right = getattr(
            self,
            "is_on_right",
            torch.zeros(
                self.world.batch_dim, dtype=torch.bool, device=self.world.device
            ),
        )
        for plate, is_active in [
            (self.plate_left, is_on_left),
            (self.plate_right, is_on_right),
        ]:
            color = Color.RED if is_active[env_index].item() else Color.LIGHT_GREEN
            circle = rendering.make_circle(self.plate_radius, filled=True)
            xform = rendering.Transform()
            xform.set_translation(*plate.state.pos[env_index][:2])
            circle.add_attr(xform)
            circle.set_color(*color.value)
            geoms.append(circle)

            # Outline ring so the plate is distinguishable when active
            outline = rendering.make_circle(self.plate_radius, filled=False)
            xform2 = rendering.Transform()
            xform2.set_translation(*plate.state.pos[env_index][:2])
            outline.add_attr(xform2)
            outline.set_color(0.0, 0.0, 0.0)
            geoms.append(outline)

        if self.with_drone:
            # Drone is now the LAST agent if added last.
            # Or search by name.
            drone = [a for a in self.world.agents if a.name == "drone"][0]

            range_circle = rendering.make_circle(self.drone_obs_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*drone.state.pos[env_index][:2])
            range_circle.add_attr(xform)
            range_circle.set_color(*Color.BLUE.value)
            geoms.append(range_circle)

            cell_size = (2 * self.x_semidim) / self.grid_size
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell_val = self.occupancy_grid[env_index, i, j].item()
                    if cell_val == 0:
                        color = Color.GRAY
                    elif cell_val == 1:
                        color = Color.GREEN
                    else:
                        color = Color.RED

                    cell = rendering.make_polygon(
                        [
                            (-cell_size / 2, -cell_size / 2),
                            (cell_size / 2, -cell_size / 2),
                            (cell_size / 2, cell_size / 2),
                            (-cell_size / 2, cell_size / 2),
                        ]
                    )
                    xform = rendering.Transform()
                    x = -self.x_semidim + cell_size / 2 + j * cell_size
                    y = -self.y_semidim + cell_size / 2 + i * cell_size
                    xform.set_translation(x, y)
                    cell.add_attr(xform)
                    cell.set_color(*color.value, alpha=0.2)
                    geoms.append(cell)

            for i in range(self.grid_size + 1):
                x = -self.x_semidim + i * cell_size
                line = rendering.Line((x, -self.y_semidim), (x, self.y_semidim))
                line.set_color(*Color.BLACK.value)
                geoms.append(line)
                y = -self.y_semidim + i * cell_size
                line = rendering.Line((-self.x_semidim, y), (self.x_semidim, y))
                line.set_color(*Color.BLACK.value)
                geoms.append(line)

        return geoms
