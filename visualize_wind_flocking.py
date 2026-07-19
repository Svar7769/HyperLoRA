"""
Visualize wind flocking behavior from a trained checkpoint.

This script loads a HyperLoRA checkpoint and renders rollouts for the
`wind_flocking_position` scenario into GIF files.

Usage examples:
    python visualize_wind_flocking.py \
        --checkpoint checkpoints/wind_flocking_run

    python visualize_wind_flocking.py \
        --checkpoint checkpoints/wind_flocking_run \
        --target-snd 0.3

    python visualize_wind_flocking.py \
        --checkpoint checkpoints/wind_flocking_run \
        --eval-snds 0.3,0.5,1.0,1.5
"""

import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import yaml
from flax.training.train_state import TrainState

from env_setup import make_vmas_env
from hypernetwork import Hypernetwork
from lora_policy import LoRAPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render wind flocking rollouts from a checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory or .npz checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: checkpoint_dir/config.yaml)",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Load checkpoint_<step>.npz when checkpoint is a directory",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Max rollout steps (default: config env.horizon)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="GIF frame rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--target-snd",
        type=float,
        default=None,
        help="Override target SND used as hypernetwork context",
    )
    parser.add_argument(
        "--current-snd-ma",
        type=float,
        default=None,
        help="Override current_snd_ma used for diversity control scaling (default: load from checkpoint)",
    )
    parser.add_argument(
        "--disable-diversity-control",
        action="store_true",
        help="Disable diversity-control scaling in visualization even if checkpoint/config provide it",
    )
    parser.add_argument(
        "--eval-snds",
        type=str,
        default=None,
        help="Comma-separated SND list to render multiple GIFs, e.g. 0.3,0.5,1.5",
    )
    parser.add_argument(
        "--snd-change-interval",
        type=int,
        default=None,
        help="Switch target SND every N rollout steps (uses --eval-snds list; default uses equal segments)",
    )
    parser.add_argument(
        "--snd-mode",
        type=str,
        choices=["cycle", "random"],
        default=None,
        help="SND switching mode when --snd-change-interval is set: cycle or random",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoint_dir/gifs)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="wind_flocking",
        help="Output GIF filename prefix",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device to use",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless rendering on servers without a display",
    )
    return parser.parse_args()


def setup_headless_rendering(force_headless=False):
    """Configure pyglet/VMAS rendering for servers without an X display."""
    display = None
    needs_headless = force_headless or os.environ.get("DISPLAY", "") == ""

    if not needs_headless:
        return None

    os.environ["PYGLET_HEADLESS"] = "1"

    try:
        from pyvirtualdisplay import Display

        display = Display(visible=False, size=(1400, 900))
        display.start()
        print("Headless rendering enabled via virtual display")
    except ImportError:
        print("Warning: pyvirtualdisplay is not installed.")
        print("Install with: pip install pyvirtualdisplay")
        print("On Linux you may also need xvfb installed.")
        print("Alternative:")
        print(f"  xvfb-run -a python {' '.join(sys.argv)}")
    except Exception as e:
        print(f"Warning: could not start virtual display: {e}")
        print("Attempting to render with pyglet headless mode only.")

    return display


def resolve_checkpoint_file(checkpoint_path: Path, checkpoint_step=None) -> Path:
    if checkpoint_path.is_file():
        return checkpoint_path

    if checkpoint_step is not None:
        checkpoint_file = checkpoint_path / f"checkpoint_{checkpoint_step}.npz"
        if checkpoint_file.exists():
            return checkpoint_file
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    candidates = [
        checkpoint_path / "final_checkpoint.npz",
        checkpoint_path / "checkpoint_final.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    numbered = sorted(checkpoint_path.glob("checkpoint_*.npz"))
    if numbered:
        return numbered[-1]

    raise FileNotFoundError(f"No checkpoint .npz file found in {checkpoint_path}")


def resolve_config_path(checkpoint_file: Path, user_config=None) -> Path:
    if user_config is not None:
        return Path(user_config)

    checkpoint_dir = checkpoint_file.parent
    config_path = checkpoint_dir / "config.yaml"
    if config_path.exists():
        return config_path

    raise FileNotFoundError(
        "Could not find config. Pass --config explicitly or place config.yaml next to checkpoint."
    )


def extract_relative_agent_positions(obs_list, num_agents, scenario_name):
    if scenario_name not in ["dispersion_vmas", "wind_flocking_position"]:
        return None

    agent_positions = []
    for agent_obs in obs_list:
        agent_positions.append(agent_obs[:, :2])

    # (num_agents, num_envs, 2) -> (num_envs, num_agents, 2)
    agent_positions_tensor = torch.stack(agent_positions, dim=0).transpose(0, 1)
    center_of_mass = agent_positions_tensor.mean(dim=1)
    return agent_positions_tensor - center_of_mass.unsqueeze(1)


def make_env_from_config(config, torch_device, num_envs=1):
    env_cfg = config["env"]
    scenario_name = env_cfg.get("scenario_name", "wind_flocking_position")
    num_agents = env_cfg["num_agents"]

    if scenario_name != "wind_flocking_position":
        raise ValueError(
            f"This script is specific to wind_flocking_position, got: {scenario_name}"
        )

    env = make_vmas_env(
        scenario_name=scenario_name,
        num_agents=num_agents,
        num_envs=num_envs,
        device=torch_device,
        continuous_actions=env_cfg.get("continuous_actions", True),
        wind=env_cfg.get("wind", 2.0),
        energy_reward_weight=env_cfg.get("energy_reward_weight", 1.0),
        wind_reward_weight=env_cfg.get("wind_reward_weight", 1.0),
        formation_shaping_weight=env_cfg.get("formation_shaping_weight", 0.5),
        position_gain=env_cfg.get("position_gain", 2.0),
        max_speed=env_cfg.get("max_speed", 0.5),
        position_range=env_cfg.get("position_range", 5.0),
        agent_radii=env_cfg.get("agent_radii", [0.05, 0.03]),
        cover_angle_tolerance=env_cfg.get("cover_angle_tolerance", 1.0),
        horizon=env_cfg.get("horizon", 200),
    )
    return env


def build_models_from_config(config, env, seed):
    model_cfg = config["model"]

    obs_dim = env.observation_space[0].shape[-1]
    action_dim = env.action_space[0].shape[-1]
    hidden_dims = tuple(model_cfg["policy_hidden_dims"])
    lora_mode = model_cfg.get("lora_mode", "final_only")
    lora_rank = model_cfg["lora_rank"]

    shared_policy = LoRAPolicy(
        hidden_dims=hidden_dims,
        action_dim=action_dim,
        lora_mode=lora_mode,
        log_std_max=model_cfg.get("log_std_max", 0.0),
        min_std=model_cfg.get("min_std", 0.3),
    )

    context_dim = model_cfg.get("context_dim", 0)
    task_embed_dim = model_cfg.get("task_embed_dim", 0)
    lidar_dim = (
        model_cfg.get("lidar_dim", 0) if model_cfg.get("use_lidar_context") else 0
    )
    food_position_dim = model_cfg.get("food_position_dim", 0)
    agent_position_dim = (
        model_cfg.get("agent_position_dim", 2)
        if model_cfg.get("use_agent_position_context", False)
        else 0
    )
    target_snd_dim = (
        model_cfg.get("target_snd_dim", 1)
        if model_cfg.get("use_target_snd_context", False)
        else 0
    )

    hypernetwork = Hypernetwork(
        policy_dims={
            "obs_dim": obs_dim,
            "hidden_dims": hidden_dims,
            "action_dim": action_dim,
            "lora_rank": lora_rank,
        },
        context_dim=context_dim,
        task_embed_dim=task_embed_dim,
        lidar_dim=lidar_dim,
        food_position_dim=food_position_dim,
        agent_position_dim=agent_position_dim,
        target_snd_dim=target_snd_dim,
        env_context_dim=0,
        max_agents=config["env"].get("max_agents", config["env"]["num_agents"]),
        transformer_dim=model_cfg["transformer_dim"],
        transformer_heads=model_cfg["transformer_heads"],
        transformer_layers=model_cfg["transformer_layers"],
        lora_mode=lora_mode,
        scaling_factor=model_cfg.get("lora_scaling_factor", 1.0),
        use_cross_agent_attention=model_cfg.get("use_cross_agent_attention", True),
    )

    # Initialize params with dummy inputs so module signatures are materialized.
    dummy_obs = jnp.ones((1, obs_dim))
    final_idx = len(hidden_dims) + 1
    dummy_adapters = {
        f"A{final_idx}": jnp.zeros((1, 0, hidden_dims[-1])),
        f"B{final_idx}": jnp.zeros((1, action_dim, 0)),
    }
    _ = shared_policy.init(jax.random.PRNGKey(seed), dummy_obs, dummy_adapters)

    num_agents = config["env"]["num_agents"]
    dummy_agent_positions = (
        jnp.ones((1, num_agents, agent_position_dim))
        if agent_position_dim > 0
        else None
    )
    dummy_target_snd = (
        jnp.ones((1, num_agents, target_snd_dim)) if target_snd_dim > 0 else None
    )

    _ = hypernetwork.init(
        jax.random.PRNGKey(seed + 1),
        None,
        None,
        None,
        None,
        dummy_agent_positions,
        dummy_target_snd,
        None,
        None,
        1.0,
    )

    return shared_policy, hypernetwork


def create_adapters(
    hypernetwork,
    hn_params,
    obs,
    scenario_name,
    target_snd,
    num_agents,
    diversity_scaling=1.0,
):
    agent_positions = extract_relative_agent_positions(obs, num_agents, scenario_name)
    agent_positions_jax = (
        jnp.asarray(agent_positions.cpu().numpy())
        if agent_positions is not None
        else None
    )

    num_envs = obs[0].shape[0]
    target_snd_batch = None
    if target_snd is not None:
        target_snd_batch = jnp.full(
            (num_envs, num_agents, 1),
            fill_value=float(target_snd),
            dtype=jnp.float32,
        )

    adapters = hypernetwork.apply(
        {"params": hn_params},
        None,
        None,
        lidar_vectors=None,
        food_position_vectors=None,
        agent_position_vectors=agent_positions_jax,
        target_snd_vectors=target_snd_batch,
        env_context_vectors=None,
        mask=None,
        diversity_scaling=diversity_scaling,
    )
    return adapters


def resolve_scalar_from_checkpoint(checkpoint, key):
    if key not in checkpoint:
        return None
    value = checkpoint[key]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(value.reshape(-1)[0])
    return float(value)


def compute_diversity_scaling(
    target_snd,
    current_snd_ma,
    use_diversity_control,
    min_snd_floor,
    max_diversity_scaling,
):
    if not use_diversity_control or target_snd is None or current_snd_ma is None:
        return 1.0

    ratio = float(target_snd) / max(float(current_snd_ma), float(min_snd_floor))
    scaling = np.sqrt(max(ratio, 0.0))
    max_scale = np.sqrt(float(max_diversity_scaling))
    return float(np.clip(scaling, 0.001, max_scale))


def render_rollout(
    env,
    shared_policy,
    policy_params,
    hypernetwork,
    hn_params,
    scenario_name,
    num_agents,
    n_steps,
    target_snd,
    use_diversity_control,
    current_snd_ma,
    min_snd_floor,
    max_diversity_scaling,
    snd_change_interval=None,
    snd_mode="cycle",
    rng_seed=42,
):
    """Render a single rollout.

    target_snd can be a scalar/None (constant adapters) or a list of floats.
    When a list is given:
    - if snd_change_interval > 0, use fixed-interval switching with snd_mode
    - otherwise divide rollout into equal segments (legacy behavior)
    The hypernetwork is re-queried at each switch boundary.
    """
    obs = env.reset()

    # Build the schedule of (step_start, snd) pairs.
    if isinstance(target_snd, (list, tuple)):
        snd_list = [float(v) for v in target_snd]
        if len(snd_list) == 0:
            raise ValueError("target_snd list is empty")

        if snd_change_interval is not None and snd_change_interval > 0:
            snd_schedule = [(0, snd_list[0])]
            local_rng = np.random.default_rng(rng_seed)
            for step_start in range(snd_change_interval, n_steps, snd_change_interval):
                if snd_mode == "random":
                    next_snd = float(local_rng.choice(snd_list))
                else:
                    next_idx = (step_start // snd_change_interval) % len(snd_list)
                    next_snd = snd_list[next_idx]
                snd_schedule.append((step_start, next_snd))
        else:
            segment_len = max(1, n_steps // len(snd_list))
            snd_schedule = [
                (i * segment_len, snd_list[i]) for i in range(len(snd_list))
            ]
    else:
        snd_schedule = [(0, target_snd)]

    print("  Planned hypernetwork query schedule:")
    for step_start, snd_value in snd_schedule:
        sched_scaling = compute_diversity_scaling(
            target_snd=snd_value,
            current_snd_ma=current_snd_ma,
            use_diversity_control=use_diversity_control,
            min_snd_floor=min_snd_floor,
            max_diversity_scaling=max_diversity_scaling,
        )
        print(
            f"    step={step_start:>4} target_snd={snd_value} diversity_scaling={sched_scaling:.4f}"
        )

    def _query_adapters(current_obs, snd_value):
        scaling = compute_diversity_scaling(
            target_snd=snd_value,
            current_snd_ma=current_snd_ma,
            use_diversity_control=use_diversity_control,
            min_snd_floor=min_snd_floor,
            max_diversity_scaling=max_diversity_scaling,
        )
        return create_adapters(
            hypernetwork=hypernetwork,
            hn_params=hn_params,
            obs=current_obs,
            scenario_name=scenario_name,
            target_snd=snd_value,
            num_agents=num_agents,
            diversity_scaling=scaling,
        )

    # Schedule index tracking
    schedule_idx = 0
    current_snd = snd_schedule[0][1]
    adapters = _query_adapters(obs, current_snd)
    step0_scaling = compute_diversity_scaling(
        target_snd=current_snd,
        current_snd_ma=current_snd_ma,
        use_diversity_control=use_diversity_control,
        min_snd_floor=min_snd_floor,
        max_diversity_scaling=max_diversity_scaling,
    )
    print(
        f"  Step 0: querying hypernetwork with target_snd={current_snd}, diversity_scaling={step0_scaling:.4f}"
    )
    query_events = [(0, current_snd, step0_scaling)]

    frames = []
    action_dim = env.action_space[0].shape[-1]
    num_envs = env.num_envs

    @jax.jit
    def _get_actions(params, obs_batch, adapters_dict):
        mean, _ = shared_policy.apply({"params": params}, obs_batch, adapters_dict)
        return jnp.clip(mean, -1.0, 1.0)

    for _step in range(n_steps):
        # Check if we advance to the next SND segment.
        if (
            schedule_idx + 1 < len(snd_schedule)
            and _step >= snd_schedule[schedule_idx + 1][0]
        ):
            schedule_idx += 1
            current_snd = snd_schedule[schedule_idx][1]
            adapters = _query_adapters(obs, current_snd)
            current_scaling = compute_diversity_scaling(
                target_snd=current_snd,
                current_snd_ma=current_snd_ma,
                use_diversity_control=use_diversity_control,
                min_snd_floor=min_snd_floor,
                max_diversity_scaling=max_diversity_scaling,
            )
            print(
                f"  Step {_step}: re-querying hypernetwork with target_snd={current_snd}, diversity_scaling={current_scaling:.4f}"
            )
            query_events.append((_step, current_snd, current_scaling))
        try:
            frame = env.render(
                mode="rgb_array", agent_index_focus=None, visualize_when_rgb=True
            )
        except Exception as e:
            error_text = str(e)
            if (
                "NoSuchDisplayException" in error_text
                or "Cannot connect to" in error_text
            ):
                raise RuntimeError(
                    "Rendering failed because no display is available. "
                    "Run with --headless, or use xvfb-run if pyvirtualdisplay/xvfb is not installed."
                ) from e
            raise
        frames.append(frame)

        obs_stacked = torch.stack(obs, dim=0).transpose(0, 1)
        obs_flat = obs_stacked.reshape(num_envs * num_agents, -1)
        obs_np = np.nan_to_num(obs_flat.cpu().numpy(), nan=0.0, posinf=1.0, neginf=-1.0)
        jax_obs = jnp.asarray(obs_np)

        jax_actions = _get_actions(policy_params, jax_obs, adapters)
        actions_np = np.array(jax_actions)
        actions_torch = torch.from_numpy(actions_np).float().to(env.device)
        actions_reshaped = actions_torch.reshape(num_envs, num_agents, action_dim)
        action_list = [actions_reshaped[:, i, :] for i in range(num_agents)]

        obs, _rewards, dones, _info = env.step(action_list)

        done_all = False
        if isinstance(dones, torch.Tensor):
            done_all = bool(dones.all().item())
        elif isinstance(dones, (list, tuple)) and len(dones) > 0:
            first = dones[0]
            if isinstance(first, torch.Tensor):
                done_all = bool(torch.stack(dones).all().item())

        if done_all:
            break

    print("  Executed hypernetwork queries:")
    for step_idx, snd_value, scaling in query_events:
        print(
            f"    step={step_idx:>4} target_snd={snd_value} diversity_scaling={scaling:.4f}"
        )

    return frames


def save_gif(frames, output_path: Path, fps: int):
    if not frames:
        raise ValueError("No frames collected; cannot save GIF")

    try:
        from moviepy import ImageSequenceClip
    except ImportError:
        from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_gif(str(output_path), fps=fps)


def main():
    args = parse_args()

    display = setup_headless_rendering(force_headless=args.headless)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_file = resolve_checkpoint_file(checkpoint_path, args.checkpoint_step)
    config_path = resolve_config_path(checkpoint_file, args.config)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.device == "auto":
        use_cuda = (
            config.get("device", {}).get("use_cuda", False)
            and torch.cuda.is_available()
        )
        torch_device = torch.device("cuda:0" if use_cuda else "cpu")
    elif args.device == "cuda":
        torch_device = torch.device("cuda:0")
    else:
        torch_device = torch.device("cpu")

    checkpoint = np.load(checkpoint_file, allow_pickle=True)
    if "policy_params" not in checkpoint:
        raise ValueError(f"policy_params missing in checkpoint: {checkpoint_file}")
    if "hn_params" not in checkpoint:
        raise ValueError(
            "hn_params missing in checkpoint. This script expects HyperLoRA checkpoints."
        )

    training_cfg = config.get("training", {})
    use_diversity_control = bool(training_cfg.get("use_diversity_control", False))
    if args.disable_diversity_control:
        use_diversity_control = False

    min_snd_floor = float(training_cfg.get("min_snd_floor", 1e-6))
    max_diversity_scaling = float(training_cfg.get("max_diversity_scaling", 100.0))

    checkpoint_current_snd_ma = resolve_scalar_from_checkpoint(
        checkpoint, "current_snd_ma"
    )
    if args.current_snd_ma is not None:
        current_snd_ma = float(args.current_snd_ma)
    else:
        current_snd_ma = checkpoint_current_snd_ma

    env = make_env_from_config(config, torch_device=torch_device, num_envs=1)

    shared_policy, hypernetwork = build_models_from_config(config, env, args.seed)

    policy_state = TrainState.create(
        apply_fn=shared_policy.apply,
        params=checkpoint["policy_params"].item(),
        tx=optax.adam(learning_rate=3e-4),
    )
    hn_state = TrainState.create(
        apply_fn=hypernetwork.apply,
        params=checkpoint["hn_params"].item(),
        tx=optax.adam(learning_rate=3e-4),
    )

    scenario_name = config["env"].get("scenario_name", "wind_flocking_position")
    num_agents = config["env"]["num_agents"]
    horizon = config["env"].get("horizon", 200)
    n_steps = args.steps if args.steps is not None else horizon

    default_target_snd = config.get("training", {}).get("target_snd", None)

    if args.eval_snds is not None:
        target_snd_values = [
            float(x.strip()) for x in args.eval_snds.split(",") if x.strip()
        ]
    else:
        configured_snd_list = config.get("training", {}).get("target_snd_list", None)
        if configured_snd_list is None:
            configured_snd_list = config.get("env", {}).get("target_snd_list", None)

        if configured_snd_list:
            target_snd_values = [float(v) for v in configured_snd_list]
        else:
            one_value = (
                args.target_snd if args.target_snd is not None else default_target_snd
            )
            target_snd_values = [one_value]

    configured_interval = config.get("training", {}).get(
        "target_snd_change_interval",
        config.get("env", {}).get("snd_change_interval", None),
    )
    snd_change_interval = (
        args.snd_change_interval
        if args.snd_change_interval is not None
        else configured_interval
    )
    snd_mode = (
        args.snd_mode
        if args.snd_mode is not None
        else config.get("training", {}).get(
            "target_snd_mode", config.get("env", {}).get("target_snd_mode", "cycle")
        )
    )

    if args.output_dir is None:
        output_dir = checkpoint_file.parent / "gifs"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Wind Flocking Visualization")
    print("=" * 80)
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Config: {config_path}")
    print(f"Device: {torch_device}")
    print(f"Scenario: {scenario_name}")
    print(f"Num agents: {num_agents}")
    print(f"Steps: {n_steps}")
    print(f"FPS: {args.fps}")
    print(f"Diversity control enabled: {use_diversity_control}")
    print(f"Checkpoint current_snd_ma: {checkpoint_current_snd_ma}")
    print(f"Using current_snd_ma: {current_snd_ma}")
    if snd_change_interval is not None:
        print(f"SND change interval: {snd_change_interval}")
    print(f"SND mode: {snd_mode}")

    try:
        if args.eval_snds is not None:
            # Single rollout with SND switching between segments.
            snd_label = "_".join(f"{v:.2f}" for v in target_snd_values)
            if snd_change_interval is not None and snd_change_interval > 0:
                output_name = f"{args.output_prefix}_snd_switch_{snd_mode}_{snd_change_interval}_{snd_label}.gif"
            else:
                output_name = f"{args.output_prefix}_snd_sweep_{snd_label}.gif"
            output_path = output_dir / output_name

            print("-" * 80)
            print(f"Rendering SND sweep: {target_snd_values}")
            if snd_change_interval is not None and snd_change_interval > 0:
                print(
                    f"Switching every {snd_change_interval} steps with mode={snd_mode}"
                )
            else:
                print(
                    f"Segment length: ~{max(1, n_steps // len(target_snd_values))} steps each"
                )

            frames = render_rollout(
                env=env,
                shared_policy=shared_policy,
                policy_params=policy_state.params,
                hypernetwork=hypernetwork,
                hn_params=hn_state.params,
                scenario_name=scenario_name,
                num_agents=num_agents,
                n_steps=n_steps,
                target_snd=target_snd_values,
                use_diversity_control=use_diversity_control,
                current_snd_ma=current_snd_ma,
                min_snd_floor=min_snd_floor,
                max_diversity_scaling=max_diversity_scaling,
                snd_change_interval=snd_change_interval,
                snd_mode=snd_mode,
                rng_seed=args.seed,
            )
            save_gif(frames, output_path=output_path, fps=args.fps)
            print(f"Saved GIF: {output_path}")
        else:
            target_snd = (
                target_snd_values
                if len(target_snd_values) > 1
                else target_snd_values[0]
            )
            if isinstance(target_snd, list):
                snd_label = "_".join(f"{v:.2f}" for v in target_snd)
                if snd_change_interval is not None and snd_change_interval > 0:
                    output_name = f"{args.output_prefix}_snd_switch_{snd_mode}_{snd_change_interval}_{snd_label}.gif"
                else:
                    output_name = f"{args.output_prefix}_snd_sweep_{snd_label}.gif"
            else:
                snd_label = "none" if target_snd is None else f"{target_snd:.2f}"
                output_name = f"{args.output_prefix}_snd_{snd_label}.gif"
            output_path = output_dir / output_name

            print("-" * 80)
            print(f"Rendering target_snd={target_snd}")
            if (
                isinstance(target_snd, list)
                and snd_change_interval is not None
                and snd_change_interval > 0
            ):
                print(
                    f"Switching every {snd_change_interval} steps with mode={snd_mode}"
                )

            frames = render_rollout(
                env=env,
                shared_policy=shared_policy,
                policy_params=policy_state.params,
                hypernetwork=hypernetwork,
                hn_params=hn_state.params,
                scenario_name=scenario_name,
                num_agents=num_agents,
                n_steps=n_steps,
                target_snd=target_snd,
                use_diversity_control=use_diversity_control,
                current_snd_ma=current_snd_ma,
                min_snd_floor=min_snd_floor,
                max_diversity_scaling=max_diversity_scaling,
                snd_change_interval=snd_change_interval,
                snd_mode=snd_mode,
                rng_seed=args.seed,
            )
            save_gif(frames, output_path=output_path, fps=args.fps)
            print(f"Saved GIF: {output_path}")
    finally:
        if display is not None:
            try:
                display.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
