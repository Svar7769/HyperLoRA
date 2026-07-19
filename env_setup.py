import vmas
import torch
from pathlib import Path
import sys
import importlib.util


def make_vmas_env(
    scenario_name,
    num_agents,
    num_envs,
    device="cpu",
    continuous_actions=True,
    penalise_by_time=False,
    share_reward=False,
    distance_shaping_coef=1.0,
    agent_capabilities=None,
    fixed_food_positions=False,
    fixed_n_food=None,  # New parameter: fixed number of food items (None = use default)
    use_original_vmas_env=False,
    **kwargs,
):
    """
    Initialize a VMAS environment.

    Args:
        scenario_name: Name of the VMAS scenario to load
        num_agents: Number of agents in the environment
        num_envs: Number of parallel environments
        device: Device to use ('cpu', 'cuda', or 'mps')
        continuous_actions: Whether to use continuous actions
        penalise_by_time: Whether to penalise agents by time (scenario-specific)
        share_reward: Whether agents share rewards (scenario-specific)
        distance_shaping_coef: Coefficient for distance-based reward shaping
        agent_capabilities: Dict with 'speed' and 'lidar_range' arrays for each agent
                          e.g., {'speed': [1.0, 0.8, 1.2], 'lidar_range': [0.5, 0.6, 0.4]}
        fixed_food_positions: If True, place foods in corners (for testing with 4 agents)
        fixed_n_food: Fixed number of food items (None = default to num_agents). Useful for
                      scaling to different agent counts while keeping task difficulty constant.
        **kwargs: Additional scenario-specific parameters (e.g., for football)

    Returns:
        env: Initialized VMAS environment or FootballWrapper
    """
    if use_original_vmas_env and scenario_name not in {"football", "smax"}:
        env_kwargs = {
            "scenario": scenario_name,
            "num_envs": num_envs,
            "device": device,
            "continuous_actions": continuous_actions,
            **kwargs,
        }

        if "n_agents" not in env_kwargs:
            env_kwargs["n_agents"] = num_agents
        if penalise_by_time and "penalise_by_time" not in env_kwargs:
            env_kwargs["penalise_by_time"] = penalise_by_time
        if (
            share_reward
            and "share_reward" not in env_kwargs
            and "shared_rew" not in env_kwargs
        ):
            env_kwargs["share_reward"] = share_reward

        try:
            print(f"Using original VMAS scenario: {scenario_name}")
            return vmas.make_env(**env_kwargs)
        except Exception as e:
            print(
                f"Could not load original VMAS scenario '{scenario_name}' ({e}). Falling back to local scenario loader."
            )

    # Check if we should use custom dispersion scenario
    if scenario_name == "dispersion" and not use_original_vmas_env:
        # Use local custom dispersion.py file
        custom_scenario_path = Path(__file__).parent / "dispersion.py"
        if custom_scenario_path.exists():
            print(f"Using custom dispersion scenario from: {custom_scenario_path}")

            # Dynamically import the custom scenario module
            # Force reload by removing from sys.modules if it exists
            import sys

            if "custom_dispersion" in sys.modules:
                del sys.modules["custom_dispersion"]

            spec = importlib.util.spec_from_file_location(
                "custom_dispersion", custom_scenario_path
            )
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            # Create the scenario instance directly
            scenario = custom_module.Scenario()

            # Create the environment using VMAS's Environment class
            from vmas.simulator.environment import Environment

            env = Environment(
                scenario=scenario,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                # Scenario-specific parameters passed to scenario's make_world
                n_agents=num_agents,
                n_food=num_agents,  # Explicitly set one food per agent
                penalise_by_time=penalise_by_time,
                share_reward=share_reward,
                distance_shaping_coef=distance_shaping_coef,
                agent_capabilities=agent_capabilities,  # Pass capabilities to scenario
                fixed_food_positions=fixed_food_positions,  # Pass fixed positions flag
            )
        else:
            # Fallback to VMAS library version
            print("Custom dispersion.py not found, using VMAS library version")
            env = vmas.make_env(
                scenario=scenario_name,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                n_agents=num_agents,
                penalise_by_time=penalise_by_time,
                share_reward=share_reward,
            )
    elif scenario_name == "dispersion":
        # Explicitly use the built-in VMAS dispersion scenario
        print("Using original VMAS dispersion scenario")
        env = vmas.make_env(
            scenario=scenario_name,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            n_agents=num_agents,
            penalise_by_time=penalise_by_time,
            share_reward=share_reward,
        )
    elif scenario_name == "dispersion_vmas":
        # Use local dispersion_vmas.py file (global observability, agent-specific food)
        custom_scenario_path = Path(__file__).parent / "dispersion_vmas.py"
        if custom_scenario_path.exists():
            print(f"Using custom dispersion_vmas scenario from: {custom_scenario_path}")

            # Dynamically import the custom scenario module
            # Force reload by removing from sys.modules if it exists
            import sys

            if "custom_dispersion_vmas" in sys.modules:
                del sys.modules["custom_dispersion_vmas"]

            spec = importlib.util.spec_from_file_location(
                "custom_dispersion_vmas", custom_scenario_path
            )
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            # Create the scenario instance directly
            scenario = custom_module.Scenario()

            # Create the environment using VMAS's Environment class
            from vmas.simulator.environment import Environment

            env = Environment(
                scenario=scenario,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                fixed_n_food=fixed_n_food,  # Pass fixed food count (None = default to n_agents)e_world
                n_agents=num_agents,
                n_food=num_agents,  # Explicitly set one food per agent
                penalise_by_time=penalise_by_time,
                share_reward=share_reward,
            )
        else:
            # Fallback error message
            print("Custom dispersion_vmas.py not found")
            raise ValueError(
                f"dispersion_vmas scenario requires dispersion_vmas.py file"
            )
    elif scenario_name == "simple_tag":
        # Use local custom simple_tag.py file
        custom_scenario_path = Path(__file__).parent / "simple_tag.py"
        if custom_scenario_path.exists():
            print(f"Using custom simple_tag scenario from: {custom_scenario_path}")

            # Dynamically import the custom scenario module
            # Force reload by removing from sys.modules if it exists
            import sys

            if "custom_simple_tag" in sys.modules:
                del sys.modules["custom_simple_tag"]

            spec = importlib.util.spec_from_file_location(
                "custom_simple_tag", custom_scenario_path
            )
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            # Create the scenario instance directly
            scenario = custom_module.Scenario()

            # Create the environment using VMAS's Environment class
            from vmas.simulator.environment import Environment

            # Extract simple_tag-specific parameters
            num_good_agents = kwargs.get("num_good_agents", 1)
            num_adversaries = kwargs.get("num_adversaries", 3)
            num_landmarks = kwargs.get("num_landmarks", 2)
            shape_agent_rew = kwargs.get("shape_agent_rew", False)
            shape_adversary_rew = kwargs.get("shape_adversary_rew", True)
            agents_share_rew = kwargs.get("agents_share_rew", False)
            adversaries_share_rew = kwargs.get("adversaries_share_rew", True)
            observe_same_team = kwargs.get("observe_same_team", True)
            observe_pos = kwargs.get("observe_pos", True)
            observe_vel = kwargs.get("observe_vel", True)
            bound = kwargs.get("bound", 1.0)
            respawn_at_catch = kwargs.get("respawn_at_catch", False)

            env = Environment(
                scenario=scenario,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                # Scenario-specific parameters passed to scenario's make_world
                num_good_agents=num_good_agents,
                num_adversaries=num_adversaries,
                num_landmarks=num_landmarks,
                shape_agent_rew=shape_agent_rew,
                shape_adversary_rew=shape_adversary_rew,
                agents_share_rew=agents_share_rew,
                adversaries_share_rew=adversaries_share_rew,
                observe_same_team=observe_same_team,
                observe_pos=observe_pos,
                observe_vel=observe_vel,
                bound=bound,
                respawn_at_catch=respawn_at_catch,
                agent_capabilities=agent_capabilities,  # Pass capabilities to scenario
            )
        else:
            # Fallback to VMAS library version
            print("Custom simple_tag.py not found, using VMAS library version")
            env = vmas.make_env(
                scenario=scenario_name,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                n_agents=num_agents,
            )
    elif scenario_name == "sampling":
        # Use local custom sampling.py file
        custom_scenario_path = Path(__file__).parent / "sampling.py"
        if custom_scenario_path.exists():
            print(f"Using custom sampling scenario from: {custom_scenario_path}")

            # Dynamically import the custom scenario module
            # Force reload by removing from sys.modules if it exists
            import sys

            if "custom_sampling" in sys.modules:
                del sys.modules["custom_sampling"]

            spec = importlib.util.spec_from_file_location(
                "custom_sampling", custom_scenario_path
            )
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            # Create the scenario instance directly
            scenario = custom_module.Scenario()

            # Create the environment using VMAS's Environment class
            from vmas.simulator.environment import Environment

            # Extract sampling-specific parameters
            shared_rew = kwargs.get("shared_rew", True)
            comms_range = kwargs.get("comms_range", 0.0)
            lidar_range = kwargs.get("lidar_range", 0.2)
            agent_radius = kwargs.get("agent_radius", 0.025)
            xdim = kwargs.get("xdim", 1.0)
            ydim = kwargs.get("ydim", 1.0)
            grid_spacing = kwargs.get("grid_spacing", 0.05)
            n_gaussians = kwargs.get("n_gaussians", 3)
            cov = kwargs.get("cov", 0.05)
            collisions = kwargs.get("collisions", True)
            spawn_same_pos = kwargs.get("spawn_same_pos", False)
            norm = kwargs.get("norm", True)

            env = Environment(
                scenario=scenario,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                # Scenario-specific parameters passed to scenario's make_world
                n_agents=num_agents,
                shared_rew=shared_rew,
                comms_range=comms_range,
                lidar_range=lidar_range,
                agent_radius=agent_radius,
                xdim=xdim,
                ydim=ydim,
                grid_spacing=grid_spacing,
                n_gaussians=n_gaussians,
                cov=cov,
                collisions=collisions,
                spawn_same_pos=spawn_same_pos,
                norm=norm,
            )
        else:
            # Fallback to VMAS library version if it exists
            print("Custom sampling.py not found, using VMAS library version")
            env = vmas.make_env(
                scenario=scenario_name,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                n_agents=num_agents,
            )
    elif scenario_name == "grassland":
        # Use local custom grassland_vmas.py file
        custom_scenario_path = Path(__file__).parent / "grassland_vmas.py"
        if custom_scenario_path.exists():
            print(f"Using custom grassland scenario from: {custom_scenario_path}")

            # Dynamically import the custom scenario module
            # Force reload by removing from sys.modules if it exists
            import sys

            if "custom_grassland" in sys.modules:
                del sys.modules["custom_grassland"]

            spec = importlib.util.spec_from_file_location(
                "custom_grassland", custom_scenario_path
            )
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            # Create the scenario instance directly
            scenario = custom_module.Scenario()

            # Create the environment using VMAS's Environment class
            from vmas.simulator.environment import Environment

            # Extract grassland-specific parameters
            n_agents_good = kwargs.get("n_agents_good", num_agents)
            n_agents_adversaries = kwargs.get("n_agents_adversaries", num_agents)
            obs_agents = kwargs.get("obs_agents", True)
            ratio = kwargs.get("ratio", 5)

            env = Environment(
                scenario=scenario,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                # Scenario-specific parameters passed to scenario's make_world
                n_agents_good=n_agents_good,
                n_agents_adversaries=n_agents_adversaries,
                obs_agents=obs_agents,
                ratio=ratio,
            )
        else:
            # Fallback to VMAS library version if it exists
            print("Custom grassland_vmas.py not found")
            raise ValueError(f"Grassland scenario requires grassland_vmas.py file")
    elif scenario_name == "football":
        # Use football wrapper
        print(f"Using Football environment wrapper")
        from football_wrapper import FootballWrapper

        # Extract football-specific parameters from kwargs
        num_red_agents = kwargs.get("num_red_agents", num_agents)
        ai_red_agents = kwargs.get("ai_red_agents", True)
        ai_blue_agents = kwargs.get("ai_blue_agents", False)
        physically_different = kwargs.get("physically_different", False)
        enable_shooting = kwargs.get("enable_shooting", False)
        dense_reward = kwargs.get("dense_reward", True)
        observe_teammates = kwargs.get("observe_teammates", True)
        observe_adversaries = kwargs.get("observe_adversaries", True)

        # Extract curriculum-related parameters (AI strength and disable flags)
        # These will be passed through scenario_kwargs
        curriculum_kwargs = {}
        if "disable_ai_red" in kwargs:
            curriculum_kwargs["disable_ai_red"] = kwargs["disable_ai_red"]
        if "ai_strength" in kwargs:
            curriculum_kwargs["ai_strength"] = kwargs["ai_strength"]
        if "ai_decision_strength" in kwargs:
            curriculum_kwargs["ai_decision_strength"] = kwargs["ai_decision_strength"]
        if "ai_precision_strength" in kwargs:
            curriculum_kwargs["ai_precision_strength"] = kwargs["ai_precision_strength"]
        if "pos_shaping_factor_ball_goal" in kwargs:
            curriculum_kwargs["pos_shaping_factor_ball_goal"] = kwargs[
                "pos_shaping_factor_ball_goal"
            ]
        if "pos_shaping_factor_agent_ball" in kwargs:
            curriculum_kwargs["pos_shaping_factor_agent_ball"] = kwargs[
                "pos_shaping_factor_agent_ball"
            ]

        # Capability sweeps for football: map requested speed into scenario kwargs.
        # Football currently uses a team-level max_speed parameter.
        football_max_speed = kwargs.get("max_speed", None)
        if football_max_speed is None and agent_capabilities is not None:
            if isinstance(agent_capabilities, dict):
                if "max_speed" in agent_capabilities:
                    speed_values = agent_capabilities.get("max_speed", [])
                    if len(speed_values) > 0:
                        football_max_speed = float(speed_values[0])
                elif "speed" in agent_capabilities:
                    speed_values = agent_capabilities.get("speed", [])
                    if len(speed_values) > 0:
                        football_max_speed = float(speed_values[0])

        if football_max_speed is not None:
            curriculum_kwargs["max_speed"] = float(football_max_speed)

        env = FootballWrapper(
            num_blue_agents=num_agents,
            num_red_agents=num_red_agents,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            ai_red_agents=ai_red_agents,
            ai_blue_agents=ai_blue_agents,
            physically_different=physically_different,
            enable_shooting=enable_shooting,
            dense_reward=dense_reward,
            observe_teammates=observe_teammates,
            observe_adversaries=observe_adversaries,
            scenario_kwargs=curriculum_kwargs if curriculum_kwargs else None,
        )
    elif scenario_name == "smax":
        # Use SMAX environment with heuristic enemies (JAX-based)
        print(f"Using SMAX environment with heuristic enemies (JAX-based)")
        from smax.smax_env import map_name_to_scenario, Scenario
        from smax.heuristic_enemy_smax_env import HeuristicEnemySMAX

        # Extract SMAX-specific parameters from kwargs
        map_name = kwargs.get("map_name", "3m")
        num_allies = kwargs.get("num_allies", 3)
        num_enemies = kwargs.get("num_enemies", 3)
        map_width = kwargs.get("map_width", 32)
        map_height = kwargs.get("map_height", 32)
        world_steps_per_env_step = kwargs.get("world_steps_per_env_step", 8)
        time_per_step = kwargs.get("time_per_step", 1.0 / 16)
        observation_type = kwargs.get("observation_type", "unit_list")
        action_type = kwargs.get("action_type", "discrete")
        use_self_play_reward = kwargs.get("use_self_play_reward", False)
        see_enemy_actions = kwargs.get("see_enemy_actions", True)
        won_battle_bonus = kwargs.get("won_battle_bonus", 1.0)
        walls_cause_death = kwargs.get("walls_cause_death", True)
        max_steps = kwargs.get("max_steps", 100)
        smacv2_position_generation = kwargs.get("smacv2_position_generation", False)
        smacv2_unit_type_generation = kwargs.get("smacv2_unit_type_generation", False)

        # Curriculum learning parameters
        enemy_shoots = kwargs.get("enemy_shoots", True)
        enemy_damage_scale = kwargs.get("enemy_damage_scale", 1.0)
        damage_reward_multiplier = kwargs.get("damage_reward_multiplier", 1.0)
        death_penalty_multiplier = kwargs.get("death_penalty_multiplier", 0.0)
        distance_reward_scale = kwargs.get("distance_reward_scale", 0.0)

        # Get base scenario from map name
        base_scenario = map_name_to_scenario(map_name)

        # Create custom scenario with potentially different num_enemies (for curriculum)
        # Use base scenario's unit_types but override num_allies and num_enemies
        import jax.numpy as jnp

        # For 3m map: all marines (unit type 0), so unit_types is all zeros
        # Adjust unit_types array size based on num_allies + num_enemies
        total_units = num_allies + num_enemies
        unit_types = jnp.zeros((total_units,), dtype=jnp.uint8)

        # CRITICAL: Disable SMACv2 position generation for asymmetric battles (3v1, 3v2)
        # The reflection position generator assumes symmetric teams and will crash with mismatched sizes
        # For curriculum learning with different enemy counts, we must use fixed positions
        use_position_generation = (
            num_allies == num_enemies and smacv2_position_generation
        )

        scenario = Scenario(
            unit_types=unit_types,
            num_allies=num_allies,
            num_enemies=num_enemies,
            smacv2_position_generation=use_position_generation,
            smacv2_unit_type_generation=smacv2_unit_type_generation,
        )

        # Apply damage scaling to enemy unit types if specified
        if enemy_damage_scale != 1.0:
            # Scale enemy attack damage
            unit_type_attacks = jnp.array([9.0, 10.0, 13.0, 8.0, 5.0, 12.0])
            scaled_attacks = unit_type_attacks * enemy_damage_scale
        else:
            scaled_attacks = None

        print(f"  Map: {map_name} (modified for curriculum)")
        print(
            f"  Allies (trainable): {scenario.num_allies}, Enemies (heuristic): {scenario.num_enemies}"
        )
        if not use_position_generation and smacv2_position_generation:
            print(
                f"  Note: Position generation disabled for asymmetric battle ({num_allies}v{num_enemies})"
            )
        print(f"  Observation type: {observation_type}")
        print(f"  Action type: {action_type}")
        print(
            f"  Curriculum: enemy_shoots={enemy_shoots}, enemy_damage_scale={enemy_damage_scale:.2f}, damage_reward_multiplier={damage_reward_multiplier:.1f}, death_penalty={death_penalty_multiplier:.1f}, distance_reward={distance_reward_scale:.2f}"
        )

        env_kwargs = {
            "scenario": scenario,
            "map_width": map_width,
            "map_height": map_height,
            "world_steps_per_env_step": world_steps_per_env_step,
            "time_per_step": time_per_step,
            "use_self_play_reward": use_self_play_reward,
            "see_enemy_actions": see_enemy_actions,
            "won_battle_bonus": won_battle_bonus,
            "walls_cause_death": walls_cause_death,
            "max_steps": max_steps,
            "smacv2_position_generation": smacv2_position_generation,
            "smacv2_unit_type_generation": smacv2_unit_type_generation,
            "observation_type": observation_type,
            "action_type": action_type,
            "enemy_shoots": enemy_shoots,
            "attack_mode": "closest",
            "damage_reward_multiplier": damage_reward_multiplier,
            "death_penalty_multiplier": death_penalty_multiplier,
            "distance_reward_scale": distance_reward_scale,
        }

        # Add scaled attacks if damage scaling is applied
        if scaled_attacks is not None:
            env_kwargs["unit_type_attacks"] = scaled_attacks

        env = HeuristicEnemySMAX(**env_kwargs)
    elif scenario_name == "reverse_transport":
        # Use local custom reverse_transport.py file
        custom_scenario_path = Path(__file__).parent / "reverse_transport.py"
        if custom_scenario_path.exists():
            print(
                f"Using custom reverse_transport scenario from: {custom_scenario_path}"
            )

            import sys

            if "custom_reverse_transport" in sys.modules:
                del sys.modules["custom_reverse_transport"]

            spec = importlib.util.spec_from_file_location(
                "custom_reverse_transport", custom_scenario_path
            )
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            scenario = custom_module.Scenario()

            from vmas.simulator.environment import Environment

            # Extract reverse_transport-specific parameters
            package_width = kwargs.get("package_width", 0.6)
            package_length = kwargs.get("package_length", 0.6)
            package_mass = kwargs.get("package_mass", 50)
            package_mass_range = kwargs.get("package_mass_range", [1, 100])

            env = Environment(
                scenario=scenario,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                n_agents=num_agents,
                package_width=package_width,
                package_length=package_length,
                package_mass=package_mass,
                package_mass_range=package_mass_range,
                agent_capabilities=agent_capabilities,  # Pass heterogeneous speeds
            )
        else:
            raise ValueError(
                "reverse_transport scenario requires reverse_transport.py file"
            )
    elif scenario_name == "pressure_plate":
        # Use local custom pressure_plate.py file
        custom_scenario_path = Path(__file__).parent / "pressure_plate.py"
        if custom_scenario_path.exists():
            print(f"Using custom pressure_plate scenario from: {custom_scenario_path}")

            import sys

            if "custom_pressure_plate" in sys.modules:
                del sys.modules["custom_pressure_plate"]

            spec = importlib.util.spec_from_file_location(
                "custom_pressure_plate", custom_scenario_path
            )
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            scenario = custom_module.Scenario()

            from vmas.simulator.environment import Environment

            # Extract pressure_plate-specific parameters
            n_ground_robots = kwargs.get("n_ground_robots", 3)
            x_semidim = kwargs.get("x_semidim", 2.0)
            y_semidim = kwargs.get("y_semidim", 2.0)
            plate_radius = kwargs.get("plate_radius", 0.15)
            plate_margin = kwargs.get("plate_margin", 0.8)
            door_size = kwargs.get("door_size", 0.6)
            goal_radius = kwargs.get("goal_radius", 0.3)
            with_drone = kwargs.get("with_drone", False)
            use_global_obs = kwargs.get("use_global_obs", True)
            plate_reward = kwargs.get("plate_reward", 0.1)
            goal_reward = kwargs.get("goal_reward", 10.0)
            time_penalty = kwargs.get("time_penalty", -0.01)
            reward_type = kwargs.get("reward_type", "sparse")
            training_spawn_side = kwargs.get("training_spawn_side", "both")
            eval_mode = kwargs.get("eval_mode", False)

            env = Environment(
                scenario=scenario,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                n_ground_robots=n_ground_robots,
                x_semidim=x_semidim,
                y_semidim=y_semidim,
                plate_radius=plate_radius,
                plate_margin=plate_margin,
                door_size=door_size,
                goal_radius=goal_radius,
                with_drone=with_drone,
                use_global_obs=use_global_obs,
                plate_reward=plate_reward,
                goal_reward=goal_reward,
                time_penalty=time_penalty,
                reward_type=reward_type,
                training_spawn_side=training_spawn_side,
                share_reward=share_reward,
                agent_capabilities=agent_capabilities,
                eval_mode=eval_mode,
            )
        else:
            raise ValueError("pressure_plate scenario requires pressure_plate.py file")
    elif scenario_name == "wind_flocking_position":
        # Use local custom wind_flocking_position.py file
        custom_scenario_path = Path(__file__).parent / "wind_flocking_position.py"
        if custom_scenario_path.exists():
            print(
                f"Using custom wind_flocking_position scenario from: {custom_scenario_path}"
            )

            import sys

            if "custom_wind_flocking" in sys.modules:
                del sys.modules["custom_wind_flocking"]

            spec = importlib.util.spec_from_file_location(
                "custom_wind_flocking", custom_scenario_path
            )
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            scenario = custom_module.Scenario()

            from vmas.simulator.environment import Environment

            # Extract wind_flocking-specific parameters
            wind = kwargs.get("wind", 2.0)
            energy_reward_weight = kwargs.get("energy_reward_weight", 1.0)
            wind_reward_weight = kwargs.get("wind_reward_weight", 1.0)
            formation_shaping_weight = kwargs.get("formation_shaping_weight", 0.5)
            position_gain = kwargs.get("position_gain", 2.0)
            max_speed = kwargs.get("max_speed", 0.5)
            position_range = kwargs.get("position_range", 5.0)
            agent_radii = kwargs.get("agent_radii", [0.05, 0.03])
            cover_angle_tolerance = kwargs.get("cover_angle_tolerance", 1.0)
            horizon = kwargs.get("horizon", 200)

            env = Environment(
                scenario=scenario,
                num_envs=num_envs,
                device=device,
                continuous_actions=continuous_actions,
                n_agents=num_agents,
                wind=wind,
                energy_reward_weight=energy_reward_weight,
                wind_reward_weight=wind_reward_weight,
                formation_shaping_weight=formation_shaping_weight,
                position_gain=position_gain,
                max_speed=max_speed,
                position_range=position_range,
                agent_radii=agent_radii,
                cover_angle_tolerance=cover_angle_tolerance,
                horizon=horizon,
            )
        else:
            raise ValueError(
                "wind_flocking_position scenario requires wind_flocking_position.py file"
            )
    else:
        # Use VMAS library scenarios for other scenario names
        env = vmas.make_env(
            scenario=scenario_name,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            n_agents=num_agents,
            penalise_by_time=penalise_by_time,
            share_reward=share_reward,
        )

    return env
