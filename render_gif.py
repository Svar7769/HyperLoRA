"""
GIF rendering utilities for visualizing trained policies.
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
from datetime import datetime
from pathlib import Path


def generate_positional_encoding(num_agents, encoding_dim, device="cpu"):
    """
    Generate sinusoidal positional encodings for agent IDs (like in Transformers).

    This encoding scales better with variable agent counts compared to one-hot encoding,
    as it doesn't require the dimensionality to match max_agents.

    Args:
        num_agents: Number of agents to encode
        encoding_dim: Dimension of the positional encoding (typically 16-32)
        device: PyTorch device for tensor creation

    Returns:
        Tensor of shape (num_agents, encoding_dim) with positional encodings
    """
    # Create position indices: [0, 1, 2, ..., num_agents-1]
    positions = torch.arange(num_agents, dtype=torch.float32, device=device).unsqueeze(
        1
    )

    # Create dimension indices and compute frequencies
    dim_indices = torch.arange(encoding_dim, dtype=torch.float32, device=device)
    div_term = torch.exp(dim_indices * -(np.log(10000.0) / encoding_dim))

    # Compute sinusoidal encodings
    pos_encoding = torch.zeros(num_agents, encoding_dim, device=device)
    pos_encoding[:, 0::2] = torch.sin(positions * div_term[0::2])  # Even dimensions
    pos_encoding[:, 1::2] = torch.cos(positions * div_term[1::2])  # Odd dimensions

    return pos_encoding


def add_mass_legend_to_frame(frame, min_mass, max_mass, current_mass=None):
    """
    Add a color gradient legend showing mass-to-brightness mapping for reverse_transport.

    Args:
        frame: RGB array (H, W, 3) from env.render()
        min_mass: Minimum mass value (lightest/brightest red)
        max_mass: Maximum mass value (darkest red)
        current_mass: Optional current mass to highlight on legend

    Returns:
        Modified frame with legend overlay
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Warning: PIL not available, skipping legend")
        return frame

    # Convert numpy array to PIL Image
    img = Image.fromarray(frame.astype("uint8"), "RGB")
    draw = ImageDraw.Draw(img)

    # Legend parameters - made larger for visibility
    legend_width = 220
    legend_height = 300
    margin = 20
    bar_width = 50

    # Position in top-right corner
    legend_x = img.width - legend_width - margin
    legend_y = margin

    # Draw semi-transparent background for legend
    background_box = [
        legend_x - 10,
        legend_y - 10,
        legend_x + legend_width + 10,
        legend_y + legend_height + 10,
    ]
    # Create overlay for transparency
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(background_box, fill=(240, 240, 240, 200))
    # Composite overlay
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw gradient bar (top = light/min mass, bottom = dark/max mass)
    gradient_start_y = legend_y + 30
    gradient_height = legend_height - 60

    for i in range(gradient_height):
        # normalized_mass goes from 0 (top, min) to 1 (bottom, max)
        normalized_mass = i / gradient_height
        # Color gradient: almost white (min mass) to pure red (max mass)
        # At min (top): (255, 250, 250) almost white
        # At max (bottom): (255, 0, 0) pure red
        red_value = 255
        green_value = int(250 * (1.0 - normalized_mass))
        blue_value = int(250 * (1.0 - normalized_mass))
        color = (red_value, green_value, blue_value)

        y_pos = gradient_start_y + i
        draw.line(
            [(legend_x, y_pos), (legend_x + bar_width, y_pos)], fill=color, width=1
        )

    # Draw border around gradient
    draw.rectangle(
        [
            legend_x,
            gradient_start_y,
            legend_x + bar_width,
            gradient_start_y + gradient_height,
        ],
        outline=(0, 0, 0),
        width=2,
    )

    # Try to load Arial font with larger sizes, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
        font_small = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 32
        )
    except:
        try:
            # Try alternate Arial locations
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 40)
            font_small = ImageFont.truetype("/Library/Fonts/Arial.ttf", 32)
        except:
            try:
                # Linux Arial fallback
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    40,
                )
                font_small = ImageFont.truetype(
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    32,
                )
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()

    # Draw title
    draw.text((legend_x, legend_y), "Package Mass", fill=(0, 0, 0), font=font)

    # Draw min mass label (top - lightest)
    min_text = f"{min_mass:.1f}"
    draw.text(
        (legend_x + bar_width + 5, gradient_start_y - 5),
        min_text,
        fill=(0, 0, 0),
        font=font_small,
    )

    # Draw max mass label (bottom - darkest)
    max_text = f"{max_mass:.1f}"
    draw.text(
        (legend_x + bar_width + 5, gradient_start_y + gradient_height - 10),
        max_text,
        fill=(0, 0, 0),
        font=font_small,
    )

    # If current mass provided, draw indicator
    if current_mass is not None:
        if max_mass > min_mass:
            normalized = (current_mass - min_mass) / (max_mass - min_mass)
            normalized = max(0.0, min(1.0, normalized))
            indicator_y = gradient_start_y + int(normalized * gradient_height)

            # Draw arrow/indicator
            draw.polygon(
                [
                    (legend_x - 8, indicator_y),
                    (legend_x - 2, indicator_y - 4),
                    (legend_x - 2, indicator_y + 4),
                ],
                fill=(0, 0, 0),
            )

            # Draw current value text
            current_text = f"{current_mass:.1f}"
            draw.text(
                (legend_x - 10 - len(current_text) * 6, indicator_y - 6),
                current_text,
                fill=(0, 0, 0),
                font=font_small,
            )

    # Convert back to numpy array
    return np.array(img)


def add_agent_legend_to_frame(frame):
    """
    Add a legend explaining agent colors for reverse_transport.

    Args:
        frame: RGB array (H, W, 3) from env.render()

    Returns:
        Modified frame with agent legend overlay
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Warning: PIL not available, skipping agent legend")
        return frame

    # Convert numpy array to PIL Image
    img = Image.fromarray(frame.astype("uint8"), "RGB")
    draw = ImageDraw.Draw(img)

    # Legend parameters
    legend_width = 280
    legend_height = 120
    margin = 20
    circle_radius = 8

    # Position in bottom-right corner
    legend_x = img.width - legend_width - margin
    legend_y = img.height - legend_height - margin

    # Draw semi-transparent background for legend
    background_box = [
        legend_x - 10,
        legend_y - 10,
        legend_x + legend_width + 10,
        legend_y + legend_height + 10,
    ]
    # Create overlay for transparency
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(background_box, fill=(240, 240, 240, 200))
    # Composite overlay
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to load Arial font
    try:
        font_title = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 18
        )
        font_text = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 14
        )
    except:
        try:
            font_title = ImageFont.truetype("/Library/Fonts/Arial.ttf", 18)
            font_text = ImageFont.truetype("/Library/Fonts/Arial.ttf", 14)
        except:
            try:
                font_title = ImageFont.truetype(
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    18,
                )
                font_text = ImageFont.truetype(
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    14,
                )
            except:
                font_title = ImageFont.load_default()
                font_text = ImageFont.load_default()

    # Draw title
    draw.text((legend_x, legend_y), "Agents", fill=(0, 0, 0), font=font_title)

    # Draw agent color indicators
    y_offset = legend_y + 30

    # Regular agent (blue/original color)
    draw.ellipse(
        [
            legend_x,
            y_offset,
            legend_x + circle_radius * 2,
            y_offset + circle_radius * 2,
        ],
        fill=(100, 150, 200),
        outline=(0, 0, 0),
        width=1,
    )
    draw.text(
        (legend_x + circle_radius * 2 + 10, y_offset),
        "Agent (regular)",
        fill=(0, 0, 0),
        font=font_text,
    )

    # Green agent (hypernetwork requeried)
    y_offset += 35
    draw.ellipse(
        [
            legend_x,
            y_offset,
            legend_x + circle_radius * 2,
            y_offset + circle_radius * 2,
        ],
        fill=(0, 255, 0),
        outline=(0, 0, 0),
        width=1,
    )
    draw.text(
        (legend_x + circle_radius * 2 + 10, y_offset),
        "Hypernetwork Requeried",
        fill=(0, 0, 0),
        font=font_text,
    )

    # Convert back to numpy array
    return np.array(img)


def detect_food_in_range(obs_list, num_envs, num_agents):
    """
    Detect which agents have food within lidar range by checking the in_range_flag.

    Args:
        obs_list: List of observations from VMAS [agent0_obs, agent1_obs, ...]
                 Each has shape: (num_envs, obs_dim)
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment

    Returns:
        food_in_range: Boolean tensor of shape (num_envs, num_agents)
                      True if food is within lidar range for that agent
    """
    # Observation structure: pos(2) + vel(2) + food(4) + lidar(...)
    # Food observation: [rel_x, rel_y, eaten_status, in_range_flag]
    # in_range_flag is at index 7 (after pos[2], vel[2], food_pos[2], eaten_status[1])

    food_in_range_list = []
    for agent_obs in obs_list:
        # Extract in_range_flag (index 7)
        in_range_flag = agent_obs[:, 7]  # (num_envs,)
        food_in_range_list.append(in_range_flag > 0.5)  # Threshold to boolean

    # Stack to (num_envs, num_agents)
    food_in_range = torch.stack(food_in_range_list, dim=1)
    return food_in_range


def generate_policy_gif(
    env,
    policy_state,
    hn_state,
    adapters_dict,
    config,
    checkpoint_dir,
    n_steps=100,
    use_hypernetwork=True,
    use_dico=False,
    adaptive_hypernetwork=False,
    num_agents=2,
    obs_dim=18,
    action_dim=2,
    context_dim=6,
    task_embed_dim=8,
    lidar_dim=0,
    env_context_dim=0,
    target_snd=0.01,
    diversity_scaling=1.0,
    use_cuda=False,
    jax_device=None,
    torch_device=None,
    get_static_adapters_fn=None,
    get_actions_fn=None,
    use_gru_policy=False,
    gru_hidden_dim=None,
    max_agents=None,
):
    """Generate a GIF visualization of the trained policy.

    Args:
        env: VMAS environment
        policy_state: Trained policy state
        hn_state: Trained hypernetwork state (if using hypernetwork)
        adapters_dict: Static adapters dict (if not using hypernetwork)
        config: Configuration dictionary
        checkpoint_dir: Directory to save the GIF
        n_steps: Maximum number of steps to render (will stop early if episode ends)
        use_hypernetwork: Whether hypernetwork is used
        num_agents: Number of agents
        obs_dim: Observation dimension
        context_dim: Context embedding dimension
        task_embed_dim: Task embedding dimension
        lidar_dim: Lidar dimension (0 if not used)
        use_cuda: Whether to use CUDA
        jax_device: JAX device
        torch_device: PyTorch device
        get_static_adapters_fn: Function to generate static adapters
        get_actions_fn: Function to get actions from policy

    Returns:
        Path to saved GIF file
    """
    try:
        # Try moviepy 2.x import first
        from moviepy import ImageSequenceClip

        print("moviepy 2.x successfully imported")
    except ImportError:
        try:
            # Fallback to moviepy 1.x import
            from moviepy.editor import ImageSequenceClip

            print("moviepy 1.x successfully imported")
        except ImportError as e:
            print(
                f"Warning: moviepy not installed. Install with: uv pip install moviepy"
            )
            print(f"Import error details: {e}")
            return None

    # Setup virtual display for headless environments (servers without X11)
    display = None
    try:
        import os

        if "DISPLAY" not in os.environ or os.environ["DISPLAY"] == "":
            print("No display detected. Setting up virtual display for rendering...")
            try:
                from pyvirtualdisplay import Display

                display = Display(visible=False, size=(1400, 900))
                display.start()
                print("Virtual display started successfully")
            except ImportError:
                print("Warning: pyvirtualdisplay not installed.")
                print(
                    "On headless servers, install with: uv pip install pyvirtualdisplay"
                )
                print("You may also need: sudo apt-get install xvfb (on Ubuntu/Debian)")
                return None
            except Exception as e:
                print(f"Warning: Could not start virtual display: {e}")
                print("Attempting to render anyway...")
    except Exception as e:
        print(f"Display check failed: {e}. Attempting to render anyway...")

    print(
        f"Rendering up to {n_steps} steps with trained policy (will stop if episode ends)..."
    )

    # Get scenario name early (needed for various checks)
    scenario_name = config["env"]["scenario_name"]

    # Debug: Show key parameters for pressure_plate
    if scenario_name == "pressure_plate":
        print(f"\n[DEBUG] Pressure Plate GIF Parameters:")
        print(f"  use_hypernetwork={use_hypernetwork}")
        print(f"  adaptive_hypernetwork={adaptive_hypernetwork}")
        print(f"  env_context_dim={env_context_dim}")
        print(f"  hn_state is not None: {hn_state is not None}")
        print()

    # Reset environment
    # Check if this is a SMAX environment (JAX-based)
    if hasattr(env, "unit_type_names"):
        # SMAX environment requires JAX random key
        rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        obs_dict, env_state = env.reset(rng_key)
        obs = [obs_dict[agent] for agent in env.agents]
        # Convert JAX arrays to PyTorch tensors
        obs = [torch.from_numpy(np.array(o)).float().to(torch_device) for o in obs]
        # Add batch dimension for SMAX
        obs = [o.unsqueeze(0) for o in obs]
        num_envs = 1  # SMAX doesn't vectorize like VMAS
    else:
        # VMAS environment
        obs = env.reset()
        num_envs = len(obs[0])

    # Immediately fix any 4-channel colors set during environment reset
    if hasattr(env, "world") and hasattr(env.world, "entities"):
        for entity in env.world.entities:
            if hasattr(entity, "color") and entity.color is not None:
                if isinstance(entity.color, torch.Tensor):
                    if len(entity.color.shape) == 2 and entity.color.shape[-1] > 3:
                        entity.color = entity.color[..., :3].clone().contiguous()
                    elif len(entity.color.shape) == 1 and entity.color.shape[0] > 3:
                        entity.color = entity.color[:3].clone().contiguous()
                elif isinstance(entity.color, (tuple, list)) and len(entity.color) > 3:
                    entity.color = tuple(entity.color[:3])

    # Set goal color to yellow for reverse_transport
    if scenario_name == "reverse_transport" and hasattr(env, "scenario"):
        if hasattr(env.scenario, "goal"):
            goal = env.scenario.goal
            # Set goal to yellow color (RGB)
            from vmas.simulator.utils import Color

            yellow_color = torch.tensor(
                Color.YELLOW.value, device=torch_device, dtype=torch.float32
            )[
                :3
            ]  # Only RGB, not alpha
            goal.color = yellow_color.unsqueeze(0).expand(num_envs, 3).contiguous()

    # Debug: Print environment info
    print(f"Environment info:")
    print(f"  - Number of parallel environments: {num_envs}")
    print(f"  - Number of agents per environment: {num_agents}")
    print(f"  - Number of agents in env.agents: {len(env.agents)}")

    # Print VMAS-specific info if available
    if hasattr(env, "world"):
        print(f"  - Agent names: {[agent.name for agent in env.agents]}")
        print(f"  - Number of food landmarks: {len(env.world.landmarks)}")
        print(f"  - Food names: {[landmark.name for landmark in env.world.landmarks]}")

    print(f"  - Rendering environment index: 0 (first of {num_envs} parallel envs)")

    # Setup context and task embeddings (same as in training)
    # Use capability-based context (matching train.py)
    # Get agent capabilities from the environment

    if scenario_name == "simple_tag":
        # For simple_tag, use full initial observations as context (matching training)
        # obs is a list of tensors: [agent0_obs, agent1_obs, ...] each (num_envs, obs_dim)
        capability_list = []
        for i in range(num_agents):
            agent_obs = obs[i][0, :]  # (obs_dim,) - full observation from first env
            capability_list.append(agent_obs)
        capability_vectors = torch.stack(
            capability_list, dim=0
        )  # (num_agents, obs_dim)
    elif scenario_name == "grassland":
        # For grassland, use full initial observations as context (matching training)
        # obs is a list of tensors: [agent0_obs, agent1_obs, ...] each (num_envs, obs_dim)
        capability_list = []
        for i in range(num_agents):
            agent_obs = obs[i][0, :]  # (obs_dim,) - full observation from first env
            capability_list.append(agent_obs)
        capability_vectors = torch.stack(
            capability_list, dim=0
        )  # (num_agents, obs_dim)
    elif scenario_name == "sampling":
        # For sampling, agents don't have lidar, so no capability context
        # Use empty capability vectors (context_dim=0)
        capability_vectors = torch.zeros(
            (num_agents, 0), device=torch_device, dtype=torch.float32
        )
    elif scenario_name == "dispersion_vmas":
        # For dispersion_vmas: use capability context if enabled in config
        use_capability_context = config["model"].get("use_capability_context", False)
        use_onehot_context = config["model"].get("use_onehot_context", True)
        use_positional_context = config["model"].get("use_positional_context", False)
        positional_encoding_dim = config["model"].get("positional_encoding_dim", 16)

        if use_capability_context:
            # Use [speed, lidar_range] as context
            agent_speeds = []
            agent_lidar_ranges = []

            for agent in env.agents:
                speed = getattr(agent, "_max_speed", None)
                agent_speeds.append(1.0 if speed is None else float(speed))

                lidar_range = getattr(agent, "_obs_range", None)
                agent_lidar_ranges.append(
                    0.5 if lidar_range is None else float(lidar_range)
                )

            capability_vectors = torch.tensor(
                [[agent_speeds[i], agent_lidar_ranges[i]] for i in range(num_agents)],
                device=torch_device,
                dtype=torch.float32,
            )
        else:
            # No capability context (context_dim=0)
            capability_vectors = torch.zeros(
                (num_agents, 0),
                device=torch_device,
                dtype=torch.float32,
            )
    elif scenario_name == "smax":
        # For SMAX, use unit capability features (7-dim normalized stats)
        from smax_capabilities import get_unit_capabilities

        smax_state = env_state.state if hasattr(env_state, "state") else env_state
        unit_types = smax_state.unit_types[: env.num_allies]
        capability_features = get_unit_capabilities(env, unit_types)
        capability_vectors = (
            torch.from_numpy(np.array(capability_features)).float().to(torch_device)
        )  # (num_agents, 7)
    elif scenario_name == "reverse_transport":
        # For reverse_transport: 2-D capability vector = [max_speed, force_multiplier]
        # getattr returns None (not the default) when the attr exists but is None.
        def _rt(agent, attr, fallback=0.5):
            v = getattr(agent, attr, None)
            return float(v) if v is not None else float(fallback)

        capability_vectors = torch.tensor(
            [
                [
                    _rt(agent, "_max_speed"),
                    _rt(agent, "_force_multiplier"),
                ]
                for agent in env.agents[:num_agents]
            ],
            device=torch_device,
            dtype=torch.float32,
        )  # (num_agents, 2)
    elif scenario_name == "pressure_plate":
        # No capability context for pressure_plate (context_dim=0).
        capability_vectors = torch.zeros(
            num_agents, 0, device=torch_device, dtype=torch.float32
        )  # (num_agents, 0)
    elif scenario_name == "football" or hasattr(env, "get_capability_vectors"):
        # Football environment: extract 3D capabilities [speed, size, shoot_power]
        capability_array = env.get_capability_vectors(normalize=True)
        capability_vectors = torch.tensor(
            capability_array,
            device=torch_device,
            dtype=torch.float32,
        )  # (num_agents, 3)
    else:
        # For other scenarios, use [speed, lidar_range]
        # For dispersion_vmas with global obs, agents may not have explicit capabilities
        # Handle both missing attributes and None values
        agent_speeds = []
        agent_lidar_ranges = []

        for agent in env.agents:
            # Get speed with fallback to default
            speed = getattr(agent, "_max_speed", None)
            agent_speeds.append(1.0 if speed is None else float(speed))

            # Get lidar range with fallback to default
            lidar_range = getattr(agent, "_obs_range", None)
            agent_lidar_ranges.append(
                0.5 if lidar_range is None else float(lidar_range)
            )

        # Create capability vectors for each agent: [speed, lidar_range]
        capability_vectors = torch.tensor(
            [[agent_speeds[i], agent_lidar_ranges[i]] for i in range(num_agents)],
            device=torch_device,
            dtype=torch.float32,
        )  # (num_agents, 2 or obs_dim)

    # Expand to include batch dimension
    static_context = capability_vectors.unsqueeze(0).expand(
        num_envs, -1, -1
    )  # (num_envs, num_agents, context_dim) where context_dim is 2 or obs_dim

    # For dispersion_vmas: append per-agent positional OR one-hot encoding to the context (if enabled)
    # so the hypernetwork can reliably distinguish agents even when they are spatially symmetric.
    if scenario_name == "dispersion_vmas":
        max_agents = config["env"].get("max_agents", num_agents)

        if use_positional_context:
            # Generate positional encodings: scalable to any number of agents
            agent_pos_encoding = generate_positional_encoding(
                num_agents, positional_encoding_dim, device=torch_device
            )
            # Expand to batch: (num_envs, num_agents, positional_encoding_dim)
            agent_pos_encoding_expanded = agent_pos_encoding.unsqueeze(0).expand(
                num_envs, -1, -1
            )
            static_context = torch.cat(
                [static_context, agent_pos_encoding_expanded], dim=-1
            )  # (num_envs, num_agents, context_dim)
        elif use_onehot_context:
            # Create one-hot encodings for agent indices: [0, 1, 2, ..., num_agents-1]
            # Note: max_agents is passed as part of config to handle variable team sizes
            agent_one_hot = torch.nn.functional.one_hot(
                torch.arange(num_agents, device=torch_device), num_classes=max_agents
            ).float()  # (num_agents, max_agents)

            # Expand to batch: (num_envs, num_agents, max_agents)
            agent_one_hot_expanded = agent_one_hot.unsqueeze(0).expand(num_envs, -1, -1)
            static_context = torch.cat(
                [static_context, agent_one_hot_expanded], dim=-1
            )  # (num_envs, num_agents, context_dim)

    static_task = torch.ones(num_envs, num_agents, task_embed_dim, device=torch_device)

    # Extract initial lidar readings if lidar is used
    if lidar_dim > 0:
        # obs is a list of tensors: [agent0_obs, agent1_obs, ...]
        # Each agent_obs has shape: (num_envs, obs_dim)
        # Extract last lidar_dim dimensions from each agent's observation
        lidar_list = [agent_obs[:, -lidar_dim:] for agent_obs in obs]
        # Stack to (num_agents, num_envs, lidar_dim) then transpose to (num_envs, num_agents, lidar_dim)
        initial_lidar = torch.stack(lidar_list, dim=0).transpose(0, 1)
    else:
        initial_lidar = None

    batch_size = num_envs * num_agents

    # Convert to numpy (keep 3D shape for hypernetwork)
    context_np = (
        static_context.detach().cpu().numpy()
        if static_context.requires_grad
        else static_context.cpu().numpy()
    )
    task_np = (
        static_task.detach().cpu().numpy()
        if static_task.requires_grad
        else static_task.cpu().numpy()
    )

    # Keep 3D shape: (num_envs, num_agents, feature_dim) — None when dim is 0
    jax_context = jnp.asarray(context_np) if context_dim > 0 else None
    jax_task = jnp.asarray(task_np) if task_embed_dim > 0 else None

    # Convert lidar to JAX if present
    if initial_lidar is not None:
        lidar_np = (
            initial_lidar.detach().cpu().numpy()
            if initial_lidar.requires_grad
            else initial_lidar.cpu().numpy()
        )
        # Handle Infinity and clip values
        lidar_np = np.nan_to_num(lidar_np, posinf=1.0, neginf=0.0)
        lidar_np = np.clip(lidar_np, -1.0, 1.0)
        jax_lidar = jnp.asarray(lidar_np)
    else:
        jax_lidar = None

    if use_cuda:
        if jax_context is not None:
            jax_context = jax.device_put(jax_context, jax_device)
        if jax_task is not None:
            jax_task = jax.device_put(jax_task, jax_device)
        if jax_lidar is not None:
            jax_lidar = jax.device_put(jax_lidar, jax_device)

    # Extract food positions for dispersion_vmas and dispersion scenarios
    jax_food_positions = None
    if scenario_name in ["dispersion_vmas", "dispersion"]:
        # Extract relative position to matching food for each agent
        # For dispersion_vmas: obs structure is pos(2) + vel(2) + [food_0(3) + food_1(3) + ...]
        # Each food entry: [rel_x, rel_y, eaten_status]
        food_positions_list = []
        extraction_successful = True

        # Check if observation has been truncated (common in zero-shot generalization)
        # If obs_dim is smaller than expected, we can't extract all food positions
        first_agent_obs = obs[0]  # (num_envs, obs_dim)
        current_obs_dim = first_agent_obs.shape[1]

        num_food = num_agents  # dispersion_vmas has n_food == n_agents by default
        for agent_idx in range(num_agents):
            agent_obs = obs[agent_idx]  # (num_envs, obs_dim)
            # Collect relative positions to ALL foods for this agent
            all_food_cols = []
            agent_extraction_ok = True
            for food_j in range(num_food):
                food_start_idx = 4 + food_j * 3
                food_end_idx = food_start_idx + 2

                # Check if we have enough observation dimensions
                if food_end_idx > current_obs_dim:
                    print(
                        f"\nWarning: Cannot extract food {food_j} position for agent {agent_idx}"
                    )
                    print(
                        f"  Required obs_dim: {food_end_idx}, actual: {current_obs_dim}"
                    )
                    print(
                        f"  Skipping food position extraction (likely due to obs truncation in zero-shot deployment)"
                    )
                    print(
                        f"  GIF will still be generated, but without food position data for hypernetwork.\n"
                    )
                    agent_extraction_ok = False
                    extraction_successful = False
                    break

                food_pos_j = agent_obs[:, food_start_idx:food_end_idx]  # (num_envs, 2)
                all_food_cols.append(food_pos_j)

            if not agent_extraction_ok:
                break

            # Concatenate all food positions: (num_envs, 2*num_food)
            agent_all_food = torch.cat(all_food_cols, dim=-1)
            food_positions_list.append(agent_all_food)

        # Only stack if we successfully extracted all food positions
        if extraction_successful and len(food_positions_list) == num_agents:
            # Stack: (num_agents, num_envs, 2*num_food)
            food_positions_stacked = torch.stack(food_positions_list, dim=0)
            # Transpose: (num_envs, num_agents, 2*num_food)
            food_positions = food_positions_stacked.transpose(0, 1)

            # Convert to JAX
            food_positions_np = (
                food_positions.detach().cpu().numpy()
                if food_positions.requires_grad
                else food_positions.cpu().numpy()
            )
            jax_food_positions = jnp.asarray(food_positions_np)

            if use_cuda:
                jax_food_positions = jax.device_put(jax_food_positions, jax_device)

    # Agent positions are not passed to the hypernetwork (same for all agents,
    # provides no useful differentiation — one-hot IDs are used instead).
    jax_agent_positions = None

    # Create target_snd tensor (same value for all agents)
    target_snd_dim = config["model"].get("target_snd_dim", 0)
    if target_snd_dim > 0:
        # Create tensor of shape (num_envs, num_agents, target_snd_dim)
        static_target_snd = torch.full(
            (num_envs, num_agents, target_snd_dim),
            float(target_snd),
            device=torch_device,
            dtype=torch.float32,
        )
        target_snd_np = (
            static_target_snd.detach().cpu().numpy()
            if static_target_snd.requires_grad
            else static_target_snd.cpu().numpy()
        )
        jax_target_snd = jnp.asarray(target_snd_np)
        if use_cuda:
            jax_target_snd = jax.device_put(jax_target_snd, jax_device)
    else:
        jax_target_snd = None

    # Extract environment context for reverse_transport (package properties)
    if scenario_name == "reverse_transport" and env_context_dim > 0:
        # Get current package properties from environment
        package_props = env.scenario.get_package_properties()
        # Create environment context: [mass, width, length]
        env_context_vec = torch.tensor(
            [
                package_props["mass"],
                package_props["width"],
                package_props["length"],
            ],
            device=torch_device,
            dtype=torch.float32,
        )
        # Broadcast to all envs and agents: (num_envs, num_agents, env_context_dim)
        static_env_context = (
            env_context_vec.unsqueeze(0)
            .unsqueeze(0)
            .expand(num_envs, num_agents, env_context_dim)
        )
        # Convert to JAX
        env_context_np = (
            static_env_context.detach().cpu().numpy()
            if static_env_context.requires_grad
            else static_env_context.cpu().numpy()
        )
        jax_env_context = jnp.asarray(env_context_np)
        if use_cuda:
            jax_env_context = jax.device_put(jax_env_context, jax_device)

        # Initialize package color based on initial mass
        initial_mass = package_props["mass"]
        mass_range = config["env"].get("package_mass_range", [1, 100])
        min_mass, max_mass = float(mass_range[0]), float(mass_range[1])

        if max_mass - min_mass > 0:
            normalized_mass = max(
                0.0, min(1.0, (initial_mass - min_mass) / (max_mass - min_mass))
            )
        else:
            normalized_mass = 0.0

        # Set initial package color
        from vmas.simulator.utils import Color

        package = env.scenario.package
        # Color gradient: almost white (min mass) to pure red (max mass)
        # Light: (255, 250, 250), Heavy: (255, 0, 0)
        red_channel = torch.tensor(255.0, device=torch_device, dtype=torch.float32)
        green_channel = torch.tensor(
            250.0 * (1.0 - normalized_mass), device=torch_device, dtype=torch.float32
        )
        blue_channel = torch.tensor(
            250.0 * (1.0 - normalized_mass), device=torch_device, dtype=torch.float32
        )
        color_rgb = (
            torch.stack([red_channel, green_channel, blue_channel]) / 255.0
        )  # Normalize to [0, 1]
        package.color = color_rgb.unsqueeze(0).expand(num_envs, 3).contiguous()
    elif scenario_name == "pressure_plate" and env_context_dim > 0:
        # Get current environment state from pressure_plate scenario.
        # Build per-agent relative env context (plate/goal positions relative to
        # each agent, consistent with the policy observation convention).

        # Get configuration flags
        env_context_plate_positions = config["model"].get(
            "env_context_plate_positions", True
        )
        env_context_door_state = config["model"].get("env_context_door_state", True)
        env_context_goal_position = config["model"].get(
            "env_context_goal_position", True
        )
        use_agent_id_context = config["model"].get("use_agent_id_context", False)

        # Set max_agents if not already set (needed for one-hot encoding)
        if max_agents is None:
            max_agents = config["env"].get("max_agents", num_agents)

        # Agent positions: (num_envs, num_agents, 2) for per-agent relative context
        _ground_robots_gif = sorted(
            [a for a in env.agents if "ground_robot" in a.name],
            key=lambda a: a.name,
        )
        _agent_pos_gif = torch.stack(
            [a.state.pos[:, :2] for a in _ground_robots_gif], dim=1
        )  # (num_envs, num_agents, 2)

        # Build per-agent context parts: each (num_envs, num_agents, d)
        _gif_ctx_parts = []

        if env_context_plate_positions:
            # Plate positions (full batch)
            left_plate_pos = env.scenario.plate_left.state.pos[:, :2]  # (num_envs, 2)
            right_plate_pos = env.scenario.plate_right.state.pos[:, :2]  # (num_envs, 2)
            left_rel_gif = (
                left_plate_pos.unsqueeze(1) - _agent_pos_gif
            )  # (num_envs, num_agents, 2)
            right_rel_gif = right_plate_pos.unsqueeze(1) - _agent_pos_gif
            _gif_ctx_parts.extend([left_rel_gif, right_rel_gif])

        if env_context_door_state:
            door_open = env.scenario.door_open.float()  # (num_envs,)
            door_open_exp = door_open[:, None, None].expand(num_envs, num_agents, 1)
            _gif_ctx_parts.append(door_open_exp)

        if env_context_goal_position:
            goal_pos = env.scenario.goal.state.pos[:, :2]  # (num_envs, 2)
            goal_rel_gif = goal_pos.unsqueeze(1) - _agent_pos_gif
            _gif_ctx_parts.append(goal_rel_gif)

        if use_agent_id_context:
            # Add one-hot agent IDs for role differentiation
            # Shape: (num_envs, num_agents, max_agents)
            agent_ids_onehot = torch.zeros(
                num_envs, num_agents, max_agents, device=torch_device
            )
            for i in range(num_agents):
                agent_ids_onehot[:, i, i] = 1.0
            _gif_ctx_parts.append(agent_ids_onehot)

        # (num_envs, num_agents, env_context_dim)
        static_env_context = torch.cat(_gif_ctx_parts, dim=-1)

        # Convert to JAX
        env_context_np = (
            static_env_context.detach().cpu().numpy()
            if static_env_context.requires_grad
            else static_env_context.cpu().numpy()
        )
        jax_env_context = jnp.asarray(env_context_np)
        if use_cuda:
            jax_env_context = jax.device_put(jax_env_context, jax_device)

        print(
            f"\n[DEBUG] Environment context initialized for pressure_plate rendering (per-agent relative):"
        )
        if env_context_plate_positions:
            print(f"  Left plate position (env0): {left_plate_pos[0].cpu().numpy()}")
            print(f"  Right plate position (env0): {right_plate_pos[0].cpu().numpy()}")
            print(f"  Agent 0 pos (env0): {_agent_pos_gif[0, 0].cpu().numpy()}")
        if env_context_door_state:
            print(f"  Door open (env0): {door_open[0].item()}")
        if env_context_goal_position:
            print(f"  Goal position (env0): {goal_pos[0].cpu().numpy()}")
    else:
        jax_env_context = None

    # Generate adapters once (static for entire rollout)
    if use_hypernetwork and hn_state is not None:
        # No mask needed during rendering - we use actual num_agents without padding
        jax_mask = None

        # Pass lidar_batch, food_position_batch, agent_position_batch, target_snd_batch, env_context_batch, mask, and diversity_scaling for proper initialization
        adapters_dict = get_static_adapters_fn(
            hn_state.params,
            jax_task,
            jax_context,
            lidar_batch=jax_lidar,
            food_positions_batch=jax_food_positions,
            agent_positions_batch=jax_agent_positions,
            target_snd_batch=jax_target_snd,
            env_context_batch=jax_env_context,
            mask=jax_mask,
            diversity_scaling=diversity_scaling,
        )
    else:
        adapters_dict = {}

    # Initialize GRU hidden states if using GRU policy
    if use_gru_policy:
        gru_hidden_states = jnp.zeros((batch_size, gru_hidden_dim))
        if use_cuda:
            gru_hidden_states = jax.device_put(gru_hidden_states, jax_device)
    else:
        gru_hidden_states = None

    frame_list = []
    episode_done = False

    # For SMAX: collect state sequence for rendering at the end
    state_sequence = [] if hasattr(env, "unit_type_names") else None

    # Track which agents have already detected food to ensure we only requery once
    # NOTE: Only applicable for scenarios with partial observability (lidar-based)
    food_detected_already = torch.zeros(
        num_envs, num_agents, dtype=torch.bool, device=torch_device
    )

    # Track door state for pressure_plate mid-episode requerying
    prev_door_open_gif = torch.zeros(num_envs, dtype=torch.bool, device=torch_device)
    plate_activation_count_gif = torch.zeros(
        num_envs, dtype=torch.long, device=torch_device
    )

    # Track agents to highlight (for visual feedback when requerying)
    # Shape: (num_envs, num_agents) - stores remaining frames to highlight (0 = no highlight)
    agent_highlight_frames = torch.zeros(
        num_envs, num_agents, dtype=torch.long, device=torch_device
    )
    # Store original agent colors to restore after highlighting
    if not hasattr(env, "unit_type_names"):  # VMAS only
        original_agent_colors = []
        for agent in env.agents[:num_agents]:
            if isinstance(agent.color, torch.Tensor):
                # Only keep RGB (first 3 channels), discard alpha if present
                color = agent.color.clone()
                if color.shape[-1] > 3:
                    color = color[..., :3]
                original_agent_colors.append(color)
            else:
                # Color is a tuple - convert to tensor (RGB only)
                color_tuple = agent.color[:3] if len(agent.color) > 3 else agent.color
                original_agent_colors.append(
                    torch.tensor(
                        color_tuple, device=torch_device, dtype=torch.float32
                    ).repeat(num_envs, 1)
                )

    for step in range(n_steps):
        if step % 10 == 0:
            print(f"  Rendering step {step}/{n_steps}")

        # ================================================================
        # Dynamic environment changes (reverse_transport)
        # ================================================================
        if scenario_name == "reverse_transport":
            use_dynamic_env = config["env"].get("use_dynamic_env_changes", False)
            env_change_interval = config["env"].get("env_change_interval", 0)

            # Debug output on first step
            if step == 1:
                print(
                    f"\n[GIF Debug] env_change_interval from config: {env_change_interval}"
                )
                if env_change_interval > 0:
                    print(
                        f"[GIF Debug] Will change at steps: {[i for i in range(env_change_interval, n_steps, env_change_interval) if i <= 100][:10]}...\n"
                    )
                else:
                    print(f"[GIF Debug] No dynamic environment changes (interval=0)\n")

            if (
                use_dynamic_env
                and env_change_interval > 0
                and step > 0
                and step % env_change_interval == 0
            ):
                env_change_type = config["env"].get("env_change_type", "random")

                # Get property ranges from config
                mass_range = config["env"].get("package_mass_range", [1, 100])
                width_range = config["env"].get("package_width_range", [0.4, 0.8])
                length_range = config["env"].get("package_length_range", [0.4, 0.8])

                # Generate new properties based on change type
                if env_change_type == "single_transition":
                    # Single transition from max to min (heavy to light) at first change only
                    if step == env_change_interval:
                        # First change: go to minimum (light)
                        new_mass, new_width, new_length = (
                            mass_range[0],
                            width_range[0],
                            length_range[0],
                        )
                    else:
                        # No more changes after the first one
                        new_mass, new_width, new_length = None, None, None
                elif env_change_type == "random":
                    new_mass = np.random.uniform(mass_range[0], mass_range[1])
                    new_width = np.random.uniform(width_range[0], width_range[1])
                    new_length = np.random.uniform(length_range[0], length_range[1])
                elif env_change_type == "schedule":
                    # Cycle through predefined values
                    cycle_idx = (step // env_change_interval) % 3
                    if cycle_idx == 0:
                        new_mass, new_width, new_length = (
                            mass_range[0],
                            width_range[0],
                            length_range[0],
                        )
                    elif cycle_idx == 1:
                        new_mass = (mass_range[0] + mass_range[1]) / 2
                        new_width = (width_range[0] + width_range[1]) / 2
                        new_length = (length_range[0] + length_range[1]) / 2
                    else:
                        new_mass, new_width, new_length = (
                            mass_range[1],
                            width_range[1],
                            length_range[1],
                        )
                elif env_change_type == "gradual":
                    # Gradually increase properties from min to max
                    progress = min(step / n_steps, 1.0)
                    new_mass = (
                        mass_range[0] + (mass_range[1] - mass_range[0]) * progress
                    )
                    new_width = (
                        width_range[0] + (width_range[1] - width_range[0]) * progress
                    )
                    new_length = (
                        length_range[0] + (length_range[1] - length_range[0]) * progress
                    )
                elif env_change_type == "decreasing":
                    # Step-wise decrease: max → half → min, then stay at min
                    change_count = step // env_change_interval
                    if change_count == 1:
                        # First decrease: max to half (midpoint)
                        new_mass = (mass_range[0] + mass_range[1]) / 2
                        new_width = (width_range[0] + width_range[1]) / 2
                        new_length = (length_range[0] + length_range[1]) / 2
                    elif change_count >= 2:
                        # Second decrease and beyond: stay at min
                        new_mass = mass_range[0]
                        new_width = width_range[0]
                        new_length = length_range[0]
                    else:
                        # Should not happen (step == 0 is caught earlier)
                        new_mass, new_width, new_length = None, None, None
                else:
                    new_mass, new_width, new_length = None, None, None

                # Update properties in environment
                if new_mass is not None:
                    changed = env.scenario.update_package_properties(
                        package_mass=new_mass,
                        package_width=new_width,
                        package_length=new_length,
                    )
                    if changed:
                        print(
                            f"  [Step {step}] Changed package: "
                            f"mass={new_mass:.2f}, width={new_width:.2f}, length={new_length:.2f}"
                        )

                        # Manually update package color based on new mass
                        # (same logic as in reverse_transport.py reward function)
                        # Use the actual mass range from config
                        min_mass_color, max_mass_color = float(mass_range[0]), float(
                            mass_range[1]
                        )

                        if max_mass_color - min_mass_color > 0:
                            normalized_mass = max(
                                0.0,
                                min(
                                    1.0,
                                    (new_mass - min_mass_color)
                                    / (max_mass_color - min_mass_color),
                                ),
                            )
                        else:
                            normalized_mass = 0.0

                        # Import Color if not already imported
                        from vmas.simulator.utils import Color

                        # Update package color (check if on goal)
                        package = env.scenario.package
                        # Color gradient: almost white (min mass) to pure red (max mass)
                        # Light: (255, 250, 250), Heavy: (255, 0, 0)
                        red_channel = torch.tensor(
                            255.0, device=torch_device, dtype=torch.float32
                        )
                        green_channel = torch.tensor(
                            250.0 * (1.0 - normalized_mass),
                            device=torch_device,
                            dtype=torch.float32,
                        )
                        blue_channel = torch.tensor(
                            250.0 * (1.0 - normalized_mass),
                            device=torch_device,
                            dtype=torch.float32,
                        )
                        color_rgb = (
                            torch.stack([red_channel, green_channel, blue_channel])
                            / 255.0
                        )  # Normalize to [0, 1]
                        package.color = (
                            color_rgb.unsqueeze(0).expand(num_envs, 3).contiguous()
                        )

                        # If package is on goal, use green instead (with same gradient)
                        if hasattr(package, "on_goal") and package.on_goal.any():
                            # Apply same gradient logic to green: almost white to pure green
                            green_red = torch.tensor(
                                250.0 * (1.0 - normalized_mass),
                                device=torch_device,
                                dtype=torch.float32,
                            )
                            green_green = torch.tensor(
                                255.0, device=torch_device, dtype=torch.float32
                            )
                            green_blue = torch.tensor(
                                250.0 * (1.0 - normalized_mass),
                                device=torch_device,
                                dtype=torch.float32,
                            )
                            color_green = (
                                torch.stack([green_red, green_green, green_blue])
                                / 255.0
                            )
                            # Explicitly expand and assign to maintain shape (num_envs, 3)
                            for env_idx in range(num_envs):
                                if package.on_goal[env_idx]:
                                    package.color[env_idx] = color_green

                        # Update jax_env_context to reflect the new properties
                        if jax_env_context is not None:
                            env_context_vec = torch.tensor(
                                [new_mass, new_width, new_length],
                                device=torch_device,
                                dtype=torch.float32,
                            )
                            updated_env_context = (
                                env_context_vec.unsqueeze(0)
                                .unsqueeze(0)
                                .expand(num_envs, num_agents, env_context_dim)
                            )
                            env_context_np = (
                                updated_env_context.detach().cpu().numpy()
                                if updated_env_context.requires_grad
                                else updated_env_context.cpu().numpy()
                            )
                            jax_env_context = jnp.asarray(env_context_np)
                            if use_cuda:
                                jax_env_context = jax.device_put(
                                    jax_env_context, jax_device
                                )

                        # Regenerate adapters with updated environment context
                        if hn_state is not None:
                            adapters_dict = get_static_adapters_fn(
                                hn_state.params,
                                jax_task,
                                jax_context,
                                lidar_batch=jax_lidar,
                                food_positions_batch=jax_food_positions,
                                agent_positions_batch=jax_agent_positions,
                                target_snd_batch=jax_target_snd,
                                env_context_batch=jax_env_context,
                                mask=jax_mask,
                                diversity_scaling=diversity_scaling,
                            )

                            # Mark all agents in env 0 for highlighting (3 frames)
                            # to show that hypernetwork was requeried
                            agent_highlight_frames[0, :] = 3
                            print(
                                f"  [Step {step}] Environment changed — requeried hypernetwork (all agents highlighted)"
                            )

        # Render current state (skip for SMAX, will render at end from state sequence)
        if not hasattr(env, "unit_type_names"):
            # Ensure ALL entities (agents, landmarks, etc.) have RGB-only colors
            # to prevent "got multiple values for argument 'alpha'" error
            # This must be done RIGHT BEFORE rendering to catch all entities
            if hasattr(env, "world"):
                # Get all possible entities in the world
                all_entities = []
                if hasattr(env.world, "agents"):
                    all_entities.extend(env.world.agents)
                if hasattr(env.world, "landmarks"):
                    all_entities.extend(env.world.landmarks)
                if hasattr(env.world, "joints"):
                    all_entities.extend(env.world.joints)
                # Check for package, goal, or other special entities in scenario
                if hasattr(env, "scenario"):
                    for attr_name in dir(env.scenario):
                        if not attr_name.startswith("_"):  # Skip private attributes
                            special_entity = getattr(env.scenario, attr_name, None)
                            if special_entity is not None and hasattr(
                                special_entity, "color"
                            ):
                                all_entities.append(special_entity)

                # Fix colors for all entities - handle both tensors and tuples
                for entity in all_entities:
                    if hasattr(entity, "color") and entity.color is not None:
                        color = entity.color
                        if isinstance(color, torch.Tensor):
                            # Always ensure colors are exactly 3 channels (RGB only)
                            if len(color.shape) == 2:
                                # Shape: (num_envs, channels) - ensure last dim is 3
                                if color.shape[-1] != 3:
                                    entity.color = color[..., :3].clone().contiguous()
                            elif len(color.shape) == 1:
                                # Shape: (channels,) - ensure exactly 3 channels
                                if color.shape[0] != 3:
                                    entity.color = color[:3].clone().contiguous()
                            elif len(color.shape) == 0:
                                # Scalar tensor - skip
                                pass
                        elif isinstance(color, (tuple, list)):
                            # Color is a tuple or list - ensure only 3 elements
                            if len(color) != 3:
                                entity.color = (
                                    tuple(color[:3])
                                    if isinstance(color, tuple)
                                    else list(color[:3])
                                )

                # Extra pass: check env.agents directly (in case they're not in env.world.agents)
                if hasattr(env, "agents"):
                    for agent in env.agents:
                        if hasattr(agent, "color") and agent.color is not None:
                            if isinstance(agent.color, torch.Tensor):
                                if (
                                    len(agent.color.shape) == 2
                                    and agent.color.shape[-1] != 3
                                ):
                                    agent.color = (
                                        agent.color[..., :3].clone().contiguous()
                                    )
                                elif (
                                    len(agent.color.shape) == 1
                                    and agent.color.shape[0] != 3
                                ):
                                    agent.color = agent.color[:3].clone().contiguous()

                # Debug: Print entity colors on first step
                if step == 0 and scenario_name == "reverse_transport":
                    print(f"\n[DEBUG] Entity colors before rendering step 0:")
                    for i, entity in enumerate(all_entities):
                        if hasattr(entity, "color") and entity.color is not None:
                            color = entity.color
                            entity_name = getattr(entity, "name", f"entity_{i}")
                            if isinstance(color, torch.Tensor):
                                print(
                                    f"  {entity_name}: shape={color.shape}, dtype={color.dtype}"
                                )
                            else:
                                print(
                                    f"  {entity_name}: type={type(color)}, value={color}"
                                )
                    print()

            # For dispersion scenarios, ensure eaten food is not rendered
            if scenario_name in ["dispersion_vmas", "dispersion"] and hasattr(
                env, "world"
            ):
                for food in env.world.landmarks:
                    # Hide eaten food from rendering
                    food.is_rendering[food.eaten] = False

            # For reverse_transport: ensure package color is always RGB (3 channels) before rendering
            if scenario_name == "reverse_transport" and hasattr(env, "scenario"):
                package = env.scenario.package
                if hasattr(package, "color") and isinstance(
                    package.color, torch.Tensor
                ):
                    if package.color.shape[-1] != 3:
                        package.color = package.color[..., :3].clone().contiguous()

                # Also check goal
                if hasattr(env.scenario, "goal"):
                    goal = env.scenario.goal
                    if hasattr(goal, "color") and isinstance(goal.color, torch.Tensor):
                        if goal.color.shape[-1] != 3:
                            goal.color = goal.color[..., :3].clone().contiguous()

            # Final aggressive check: iterate through world.entities (what render actually uses)
            if hasattr(env, "world") and hasattr(env.world, "entities"):
                for entity in env.world.entities:
                    if hasattr(entity, "color") and entity.color is not None:
                        if isinstance(entity.color, torch.Tensor):
                            if (
                                len(entity.color.shape) == 2
                                and entity.color.shape[-1] != 3
                            ):
                                entity.color = (
                                    entity.color[..., :3].clone().contiguous()
                                )
                            elif (
                                len(entity.color.shape) == 1
                                and entity.color.shape[0] != 3
                            ):
                                entity.color = entity.color[:3].clone().contiguous()
                        elif (
                            isinstance(entity.color, (tuple, list))
                            and len(entity.color) != 3
                        ):
                            entity.color = (
                                tuple(entity.color[:3])
                                if isinstance(entity.color, tuple)
                                else list(entity.color[:3])
                            )

            # FINAL SAFETY CHECK: Right before rendering, force ALL entity colors to 3 channels
            # AND convert tensor colors to tuples to avoid indexing issues during render
            if hasattr(env, "world") and hasattr(env.world, "entities"):
                for entity in env.world.entities:
                    if hasattr(entity, "color") and entity.color is not None:
                        if isinstance(entity.color, torch.Tensor):
                            # Convert tensor colors to tuple to avoid any indexing/unpacking issues
                            # VMAS renders env_index=0, so we take the color for that environment
                            if len(entity.color.shape) == 2:
                                # Shape: (num_envs, 3) - take env 0 and convert to tuple
                                color_array = entity.color[0, :3].detach().cpu().numpy()
                                entity.color = tuple(float(c) for c in color_array)
                            elif len(entity.color.shape) == 1:
                                # Shape: (3,) or (4,) - ensure only 3 elements and convert to tuple
                                color_array = entity.color[:3].detach().cpu().numpy()
                                entity.color = tuple(float(c) for c in color_array)
                        elif isinstance(entity.color, (tuple, list)):
                            # Ensure exactly 3 elements
                            if len(entity.color) > 3:
                                entity.color = tuple(entity.color[:3])
                            elif not isinstance(entity.color, tuple):
                                entity.color = tuple(entity.color)

            # Apply highlighting to agents that recently requeried
            # This MUST be done AFTER the FINAL SAFETY CHECK to ensure colors aren't overwritten
            for agent_idx in range(num_agents):
                if agent_highlight_frames[0, agent_idx] > 0:
                    # Light yellow for pressure_plate, bright green for other scenarios
                    if scenario_name == "pressure_plate":
                        env.agents[agent_idx].color = (1.0, 1.0, 0.5)  # Light yellow
                    else:
                        env.agents[agent_idx].color = (0.0, 1.0, 0.0)  # Bright green
                elif agent_highlight_frames[0, agent_idx] == 0 and step > 0:
                    # Restore original color if highlight just ended
                    if isinstance(original_agent_colors[agent_idx], torch.Tensor):
                        # Convert to tuple for env 0
                        color_array = (
                            original_agent_colors[agent_idx][0, :3]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        env.agents[agent_idx].color = tuple(
                            float(c) for c in color_array
                        )
                    else:
                        env.agents[agent_idx].color = original_agent_colors[agent_idx]

            # Render environment index 0 (first of parallel envs)
            try:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Show all agents
                    visualize_when_rgb=True,
                    env_index=0,  # Explicitly render first environment
                )
            except TypeError as e:
                if "got multiple values for argument 'alpha'" in str(e):
                    print(f"\n[ERROR] Render failed at step {step} with alpha conflict")
                    print(f"Checking all entity colors in world.entities:")
                    if hasattr(env, "world") and hasattr(env.world, "entities"):
                        for i, entity in enumerate(env.world.entities):
                            if hasattr(entity, "color") and entity.color is not None:
                                entity_name = getattr(entity, "name", f"entity_{i}")
                                color = entity.color
                                if isinstance(color, torch.Tensor):
                                    print(
                                        f"  {entity_name}: tensor shape={color.shape}"
                                    )
                                else:
                                    print(
                                        f"  {entity_name}: {type(color).__name__} len={len(color) if hasattr(color, '__len__') else 'N/A'}"
                                    )
                    print(f"\nOriginal error: {e}")
                raise

            # Add mass legend for reverse_transport scenario
            # DISABLED: Legends removed per user request
            # if scenario_name == "reverse_transport":
            #     mass_range = config["env"].get("package_mass_range", [1, 100])
            #     min_mass, max_mass = float(mass_range[0]), float(mass_range[1])

            #     # Get current package mass from environment
            #     current_mass = None
            #     if hasattr(env, "scenario") and hasattr(env.scenario, "package"):
            #         package_props = env.scenario.get_package_properties()
            #         current_mass = package_props.get("mass", None)

            #     frame = add_mass_legend_to_frame(
            #         frame, min_mass, max_mass, current_mass
            #     )

            #     # Add agent legend
            #     frame = add_agent_legend_to_frame(frame)

            frame_list.append(frame)

            # Decrement highlight counters after rendering
            agent_highlight_frames = torch.clamp(agent_highlight_frames - 1, min=0)

        # Get actions from policy
        obs_np = [o.cpu().numpy() for o in obs]
        obs_stacked_np = np.stack(obs_np, axis=1)
        # Get actual observation dimension from the first observation
        actual_obs_dim = obs_np[0].shape[-1]
        obs_flat = obs_stacked_np.reshape(-1, actual_obs_dim)

        # For SMAX with curriculum, pad observations if needed to match policy input size
        if scenario_name == "smax" and actual_obs_dim < obs_dim:
            padding_size = obs_dim - actual_obs_dim
            padding = np.zeros((obs_flat.shape[0], padding_size), dtype=obs_flat.dtype)
            obs_flat = np.concatenate([obs_flat, padding], axis=1)

        obs_jax = jnp.asarray(obs_flat)

        if use_cuda:
            obs_jax = jax.device_put(obs_jax, jax_device)

        # For the shared-policy baseline (no HN, no DiCo) on scenarios that expose
        # env_context, append it to the observations to match training behaviour.
        if (
            not use_hypernetwork
            and not use_dico
            and scenario_name in ("reverse_transport", "pressure_plate")
            and jax_env_context is not None
            and env_context_dim > 0
        ):
            _batch_size_gif = num_envs * num_agents
            env_context_flat = jax_env_context.reshape(_batch_size_gif, env_context_dim)
            obs_jax = jnp.concatenate([obs_jax, env_context_flat], axis=-1)

        # Forward pass through policy (get mean actions for deterministic behavior)
        if use_gru_policy:
            # GRU policy: call with hidden state and get updated hidden state
            mean_actions, gru_hidden_states = get_actions_fn(
                policy_state.params, obs_jax, adapters_dict, gru_hidden_states
            )
        elif use_hypernetwork:
            mean_actions = get_actions_fn(policy_state.params, obs_jax, adapters_dict)
        elif use_dico:
            # For DiCo, create agent_ids and pass diversity_scaling
            agent_ids = np.tile(np.arange(num_agents), num_envs)
            agent_ids_jax = jnp.asarray(agent_ids)
            if use_cuda:
                agent_ids_jax = jax.device_put(agent_ids_jax, jax_device)
            mean_actions = get_actions_fn(
                policy_state.params,
                obs_jax,
                agent_ids_jax,
                diversity_scaling=diversity_scaling,
            )
        else:
            # For independent policies, use empty adapters (zeros)
            mean_actions = get_actions_fn(policy_state.params, obs_jax, adapters_dict)

        # Convert to numpy and reshape
        actions_np = np.array(mean_actions)

        # Check if shape is correct before reshaping
        expected_size = num_envs * num_agents * action_dim
        if actions_np.size != expected_size:
            raise ValueError(
                f"Action array size mismatch: got {actions_np.size}, "
                f"expected {expected_size} (num_envs={num_envs}, num_agents={num_agents}, action_dim={action_dim}). "
                f"Action shape from policy: {actions_np.shape}"
            )

        # Explicitly reshape to (num_envs, num_agents, action_dim)
        actions_per_env = actions_np.reshape(num_envs, num_agents, action_dim)

        # CRITICAL: Clip actions to valid range [-1.0, 1.0] for VMAS
        # The policy should output tanh-bounded actions, but due to numerical precision
        # they can be slightly outside this range, causing VMAS assertion errors
        actions_per_env = np.clip(actions_per_env, -1.0, 1.0)

        # Step environment - handle SMAX vs VMAS differently
        if hasattr(env, "unit_type_names"):
            # SMAX: discrete actions - take argmax over action_dim to get action indices
            # actions_per_env shape: (num_envs, num_agents, action_dim)
            # We need action indices: (num_envs, num_agents)
            actions_reshaped = np.argmax(
                actions_per_env, axis=2
            )  # (num_envs, num_agents)

            # SMAX steps single environment
            step_key = jax.random.fold_in(rng_key, step)
            actions_dict = {
                agent: int(actions_reshaped[0, i]) for i, agent in enumerate(env.agents)
            }
            next_obs_dict, env_state, rewards_dict, dones_dict, _ = env.step_env(
                step_key, env_state, actions_dict
            )

            # Collect state sequence for SMAX rendering
            if state_sequence is not None:
                state_sequence.append((step_key, env_state, actions_dict))

            # Update context from new state
            from smax_capabilities import get_unit_capabilities

            smax_state = env_state.state if hasattr(env_state, "state") else env_state
            unit_types = smax_state.unit_types[: env.num_allies]
            capability_features = get_unit_capabilities(env, unit_types)
            capability_vectors = (
                torch.from_numpy(np.array(capability_features)).float().to(torch_device)
            )
            static_context = capability_vectors.unsqueeze(0).expand(num_envs, -1, -1)

            # Convert to list format
            obs = [next_obs_dict[agent] for agent in env.agents]
            obs = [torch.from_numpy(np.array(o)).float().to(torch_device) for o in obs]
            obs = [o.unsqueeze(0).expand(num_envs, -1) for o in obs]

            rewards = [
                torch.full(
                    (num_envs,),
                    float(np.array(rewards_dict[agent])),
                    device=torch_device,
                )
                for agent in env.agents
            ]
            dones = [
                torch.full(
                    (num_envs,),
                    bool(np.array(dones_dict[agent])),
                    dtype=torch.bool,
                    device=torch_device,
                )
                for agent in env.agents
            ]
            info = {}
        else:
            # VMAS: continuous actions
            # Convert to list of tensors for VMAS (one tensor per agent)
            actions = [
                torch.tensor(
                    actions_per_env[:, i, :], dtype=torch.float32, device=torch_device
                )
                for i in range(num_agents)
            ]
            obs, rewards, dones, info = env.step(actions)

        # Debug: Check if any food has been eaten (for dispersion scenarios)
        if scenario_name in ["dispersion_vmas", "dispersion"] and hasattr(env, "world"):
            if step % 20 == 0:  # Check every 20 steps to avoid spam
                eaten_count = sum(food.eaten[0].item() for food in env.world.landmarks)
                if eaten_count > 0:
                    print(
                        f"  [Step {step}] Food eaten: {eaten_count}/{len(env.world.landmarks)}"
                    )

        # Check if Food Entered Lidar Range and Requery Hypernetwork
        # NOTE: Skip for scenarios without food/lidar or with custom requery logic
        if (
            use_hypernetwork
            and hn_state is not None
            and adaptive_hypernetwork
            and scenario_name in ["grassland", "simple_tag", "dispersion"]
        ):
            # Detect which agents have food in lidar range now
            current_food_in_range = detect_food_in_range(obs, num_envs, num_agents)

            # Find agents that just detected food for the FIRST time
            newly_detected = current_food_in_range & ~food_detected_already

            # If any agent newly detected food, requery hypernetwork for ONLY those agents
            if newly_detected.any():
                # Extract updated lidar readings from obs
                if lidar_dim > 0:
                    updated_lidar_list = [
                        agent_obs[:, -lidar_dim:] for agent_obs in obs
                    ]
                    # Keep 3D structure: (num_envs, num_agents, lidar_dim)
                    updated_lidar = torch.stack(updated_lidar_list, dim=0).transpose(
                        0, 1
                    )

                    # Convert to numpy and JAX (keep 3D structure)
                    updated_lidar_np = updated_lidar.cpu().numpy()
                    updated_lidar_np = np.nan_to_num(
                        updated_lidar_np, posinf=1.0, neginf=0.0
                    )
                    updated_lidar_np = np.clip(updated_lidar_np, -1.0, 1.0)
                    jax_lidar_updated = jnp.asarray(updated_lidar_np)

                    if use_cuda:
                        jax_lidar_updated = jax.device_put(
                            jax_lidar_updated, jax_device
                        )
                else:
                    jax_lidar_updated = None

                # Requery hypernetwork with updated lidar context, target_snd, mask, and diversity_scaling
                new_adapters = get_static_adapters_fn(
                    hn_state.params,
                    jax_task,
                    jax_context,
                    jax_lidar_updated,
                    jax_food_positions,
                    jax_agent_positions,
                    jax_target_snd,
                    jax_mask,
                    diversity_scaling=diversity_scaling,
                )

                # Update adapters ONLY for agents that newly detected food
                newly_detected_flat = newly_detected.reshape(-1).cpu().numpy()

                # Replace adapters only for agents that newly detected food
                for key in adapters_dict.keys():
                    mask = newly_detected_flat[:, None, None]
                    adapters_dict[key] = jnp.where(
                        mask, new_adapters[key], adapters_dict[key]
                    )

                num_requeried = newly_detected.sum().item()
                requeried_indices = torch.where(newly_detected)
                print(
                    f"  [Step {step}] Requeried HN for {num_requeried} agent(s): env={requeried_indices[0].tolist()}, agent={requeried_indices[1].tolist()}"
                )

                # Mark newly requeried agents for highlighting (3 frames)
                agent_highlight_frames[newly_detected] = 3

            # Update tracking
            food_detected_already = food_detected_already | current_food_in_range

        # ================================================================
        # Pressure plate: requery hypernetwork when door opens OR 2nd plate activated
        # (mirrors the logic in train.py)
        # ================================================================
        # Debug on first step for pressure_plate
        if step == 1 and scenario_name == "pressure_plate":
            print(f"\n[DEBUG] Pressure plate requerying check (step {step}):")
            print(f"  use_hypernetwork: {use_hypernetwork}")
            print(f"  hn_state is not None: {hn_state is not None}")
            print(
                f"  scenario_name == 'pressure_plate': {scenario_name == 'pressure_plate'}"
            )
            print(
                f"  env_context_dim > 0: {env_context_dim} > 0 = {env_context_dim > 0}"
            )
            print(f"  jax_env_context is not None: {jax_env_context is not None}")
            will_check = (
                use_hypernetwork
                and hn_state is not None
                and scenario_name == "pressure_plate"
                and env_context_dim > 0
                and jax_env_context is not None
            )
            print(f"  Will check for requerying: {will_check}\n")

        if (
            use_hypernetwork
            and hn_state is not None
            and scenario_name == "pressure_plate"
            and env_context_dim > 0
            and jax_env_context is not None
        ):
            current_door_open = env.scenario.door_open  # (num_envs,)

            # Check plate activations for env 0 (the rendered environment)
            ground_robots = [a for a in env.agents if "ground_robot" in a.name]
            left_plate_active = torch.zeros(
                num_envs, dtype=torch.bool, device=torch_device
            )
            right_plate_active = torch.zeros(
                num_envs, dtype=torch.bool, device=torch_device
            )

            for robot in ground_robots:
                # Check left plate
                dist_left = torch.linalg.norm(
                    robot.state.pos - env.scenario.plate_left.state.pos, dim=-1
                )
                left_plate_active |= dist_left < (
                    env.scenario.plate_radius + robot.shape.radius + 0.05
                )

                # Check right plate
                dist_right = torch.linalg.norm(
                    robot.state.pos - env.scenario.plate_right.state.pos, dim=-1
                )
                right_plate_active |= dist_right < (
                    env.scenario.plate_radius + robot.shape.radius + 0.05
                )

            # Count active plates
            current_active_plates = (
                left_plate_active.long() + right_plate_active.long()
            )  # 0, 1, or 2

            # Detect when we go from 1 to 2 plates (second plate just activated)
            second_plate_activated = (current_active_plates == 2) & (
                plate_activation_count_gif == 1
            )

            # Update plate activation count (keep max seen)
            plate_activation_count_gif = torch.maximum(
                plate_activation_count_gif, current_active_plates
            )

            # Only watch env 0 — that's the environment being rendered.
            door_opened_in_rendered_env = (
                current_door_open[0] and not prev_door_open_gif[0]
            )
            second_plate_in_rendered_env = second_plate_activated[0].item()

            should_requery = door_opened_in_rendered_env or second_plate_in_rendered_env

            if should_requery:
                # Rebuild env context with updated (per-agent relative) positions.
                # ONLY for env 0 (the rendered environment)
                env_context_plate_positions = config["model"].get(
                    "env_context_plate_positions", True
                )
                env_context_door_state = config["model"].get(
                    "env_context_door_state", True
                )
                env_context_goal_position = config["model"].get(
                    "env_context_goal_position", True
                )
                use_agent_id_context = config["model"].get(
                    "use_agent_id_context", False
                )

                # Agent positions at requery time for env 0 only: (1, num_agents, 2)
                _ground_robots_rq_gif = sorted(
                    [a for a in env.agents if "ground_robot" in a.name],
                    key=lambda a: a.name,
                )
                _agent_pos_rq_gif = torch.stack(
                    [a.state.pos[0:1, :2] for a in _ground_robots_rq_gif], dim=1
                )  # (1, num_agents, 2)

                # Build per-agent relative context parts (only env 0)
                _rq_gif_parts = []
                if env_context_plate_positions:
                    _lp = env.scenario.plate_left.state.pos[0:1, :2]  # (1, 2)
                    _rp = env.scenario.plate_right.state.pos[0:1, :2]  # (1, 2)
                    _rq_gif_parts.extend(
                        [
                            _lp.unsqueeze(1) - _agent_pos_rq_gif,  # (1, num_agents, 2)
                            _rp.unsqueeze(1) - _agent_pos_rq_gif,  # (1, num_agents, 2)
                        ]
                    )
                if env_context_door_state:
                    _door_exp = (
                        current_door_open[0:1]
                        .float()[:, None, None]
                        .expand(1, num_agents, 1)
                    )
                    _rq_gif_parts.append(_door_exp)
                if env_context_goal_position:
                    _gp = env.scenario.goal.state.pos[0:1, :2]  # (1, 2)
                    _rq_gif_parts.append(
                        _gp.unsqueeze(1) - _agent_pos_rq_gif
                    )  # (1, num_agents, 2)

                if use_agent_id_context:
                    # Add one-hot agent IDs for role differentiation
                    # Shape: (1, num_agents, max_agents)
                    agent_ids_onehot = torch.zeros(
                        1, num_agents, max_agents, device=torch_device
                    )
                    for i in range(num_agents):
                        agent_ids_onehot[:, i, i] = 1.0
                    _rq_gif_parts.append(agent_ids_onehot)

                # (1, num_agents, env_context_dim) - only env 0
                updated_ctx = torch.cat(_rq_gif_parts, dim=-1)
                # Expand to full batch for compatibility with get_static_adapters_fn
                # (We only care about env 0, but maintain batch shape)
                updated_ctx = updated_ctx.expand(
                    num_envs, -1, -1
                )  # (num_envs, num_agents, env_context_dim)
                env_context_np = (
                    updated_ctx.detach().cpu().numpy()
                    if updated_ctx.requires_grad
                    else updated_ctx.cpu().numpy()
                )
                jax_env_context = jnp.asarray(env_context_np)
                if use_cuda:
                    jax_env_context = jax.device_put(jax_env_context, jax_device)

                adapters_dict = get_static_adapters_fn(
                    hn_state.params,
                    jax_task,
                    jax_context,
                    lidar_batch=jax_lidar,
                    food_positions_batch=jax_food_positions,
                    agent_positions_batch=jax_agent_positions,
                    target_snd_batch=jax_target_snd,
                    env_context_batch=jax_env_context,
                    mask=jax_mask,
                    diversity_scaling=diversity_scaling,
                )

                # Log what triggered the requery
                events = []
                if door_opened_in_rendered_env:
                    events.append("door opened")
                if second_plate_in_rendered_env:
                    events.append("2nd plate activated")
                print(
                    f"  [Step {step}] Pressure plate event: {', '.join(events)} — requeried hypernetwork"
                )

                # Mark all agents in env 0 for highlighting (3 frames)
                agent_highlight_frames[0, :] = 3

            prev_door_open_gif = current_door_open.clone()

        # Check if episode ended (check first environment's done status)
        # For SMAX: dones is a list of tensors, for VMAS: dones is a single tensor
        if isinstance(dones, list):
            # SMAX: check the "__all__" flag to see if episode actually ended
            # Individual agents can die, but episode continues until all allies/enemies dead or time limit
            try:
                # dones_dict has "__all__" key for episode-level termination
                if "__all__" in dones_dict and dones_dict["__all__"].item():
                    episode_done = True
                    # Get termination reason from env_state
                    allies_alive = (
                        env_state.state.unit_alive[: env.num_allies].sum().item()
                    )
                    enemies_alive = (
                        env_state.state.unit_alive[env.num_allies :].sum().item()
                    )
                    print(
                        f"  Episode ended at step {step + 1}: Allies alive: {allies_alive}, Enemies alive: {enemies_alive}"
                    )
                    break
            except (AttributeError, KeyError):
                # Fallback: check if any agent is done (old behavior)
                if any(d[0].item() for d in dones):
                    episode_done = True
                    print(f"  Episode ended at step {step + 1}")
                    break
        else:
            # VMAS: dones is a 1D tensor with shape (num_envs,)
            if dones[0].item():  # Check if first environment is done
                episode_done = True
                print(f"  Episode ended at step {step + 1}")
                break

    # Render SMAX state sequence if applicable
    if state_sequence is not None and len(state_sequence) > 0:
        print(f"Rendering SMAX state sequence ({len(state_sequence)} states)...")
        try:
            from jaxmarl.viz.visualizer import SMAXVisualizer
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend

            # Create visualizer with the state sequence
            # state_sequence is a list of (key, state, actions) tuples
            viz = SMAXVisualizer(env, state_sequence)

            # Create output directory
            gif_dir = Path(checkpoint_dir) / "gifs"
            gif_dir.mkdir(exist_ok=True)

            # Generate filename
            scenario_name = config["env"]["scenario_name"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = gif_dir / f"{scenario_name}_policy_{timestamp}.gif"

            # Render and save directly
            print(
                f"Creating GIF animation (this may take 30-60 seconds with Pillow)..."
            )
            print(
                f"Tip: Install ffmpeg for faster rendering: conda install -c conda-forge ffmpeg"
            )
            viz.animate(view=False, save_fname=str(gif_path))

            print(f"SMAX GIF saved to: {gif_path}")

            # Clean up display
            if display is not None:
                display.stop()

            return gif_path

        except Exception as e:
            print(f"Warning: Failed to render SMAX GIF: {e}")
            if display is not None:
                display.stop()
            return None

    # Save GIF for VMAS
    scenario_name = config["env"]["scenario_name"]
    num_agents = config["env"]["num_agents"]
    use_hn = config["model"].get("use_hypernetwork", True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if use_hn:
        lora_mode = config["model"].get("lora_mode", "none")
        hn_str = "hn"
        lora_str = str(lora_mode)
        gif_filename = (
            f"{scenario_name}_{num_agents}agents_{hn_str}_{lora_str}_{timestamp}.gif"
        )
    else:
        gif_filename = f"{scenario_name}_{num_agents}agents_{timestamp}.gif"
    gif_path = checkpoint_dir / gif_filename

    print(f"Creating GIF with {len(frame_list)} frames...")
    fps = 30
    clip = ImageSequenceClip(frame_list, fps=fps)
    clip.write_gif(str(gif_path), fps=fps)

    # Clean up virtual display if we created one
    if display is not None:
        try:
            display.stop()
            print("Virtual display stopped")
        except Exception as e:
            print(f"Warning: Could not stop virtual display: {e}")

    return gif_path
