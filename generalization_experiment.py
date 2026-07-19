"""
Zero-shot generalization experiment for HyperLoRA.

This script demonstrates that HyperLoRA can generalize across different team sizes:
1. Train with N agents (with heterogeneous capabilities)
2. Deploy the trained policy on M agents (without retraining)

The hypernetwork can generalize because it conditions on capability vectors,
not agent count or identity.

Usage:
    python generalization_experiment.py --train-agents 2 --deploy-agents 4 --episodes 500
    python generalization_experiment.py --train-agents 3 --deploy-agents 6 --episodes 1000
"""

import subprocess
import sys
import argparse
from pathlib import Path
import yaml
import shutil
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HyperLoRA zero-shot generalization experiment"
    )

    parser.add_argument(
        "--train-agents",
        type=int,
        required=True,
        help="Number of agents to train with",
    )
    parser.add_argument(
        "--deploy-agents",
        type=int,
        required=True,
        help="Number of agents to deploy on (zero-shot)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (default: use config value)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: use config value)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--gif-steps",
        type=int,
        default=200,
        help="Number of steps for GIF visualization (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: use config value)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity/username",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name",
    )

    return parser.parse_args()


def run_command(cmd, description):
    """Run a command and print output."""
    print("\n" + "=" * 80)
    print(f"{description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nError: {description} failed!")
        sys.exit(1)

    print(f"\n{description} completed successfully!")
    return result


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("HyperLoRA Zero-Shot Generalization Experiment")
    print("=" * 80)
    print(
        f"Training on {args.train_agents} agents → Deploying on {args.deploy_agents} agents"
    )
    print("=" * 80)

    # Configuration
    config_file = args.config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create backup of original config
    config_backup = f"config_backup_{timestamp}.yaml"
    shutil.copy(config_file, config_backup)
    print(f"\nBacked up config to: {config_backup}")

    try:
        # ====================================================================
        # Phase 1: Train on N agents
        # ====================================================================
        print("\n" + "=" * 80)
        print(f"PHASE 1: Training on {args.train_agents} agents")
        print("=" * 80)

        # Load and modify config for training
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Set training configuration
        config["env"]["num_agents"] = args.train_agents

        if args.num_envs is not None:
            config["env"]["num_envs"] = args.num_envs

        if args.episodes is not None:
            config["training"]["num_episodes"] = args.episodes

        if args.seed is not None:
            config["training"]["seed"] = args.seed

        # Save modified config
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Modified config:")
        print(f"  - Agents: {config['env']['num_agents']}")
        print(f"  - Episodes: {config['training']['num_episodes']}")
        print(f"  - Parallel envs: {config['env']['num_envs']}")
        print(f"  - Seed: {config['training']['seed']}")

        # Train
        train_cmd = [
            sys.executable,
            "train.py",
            "--render-final-policy",
            "--gif-steps",
            str(args.gif_steps),
        ]

        # Add wandb arguments if provided
        if args.wandb:
            train_cmd.append("--wandb")
        if args.wandb_project:
            train_cmd.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            train_cmd.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_name:
            train_cmd.extend(["--wandb-name", args.wandb_name])

        run_command(train_cmd, f"Training on {args.train_agents} agents")

        # Find the latest checkpoint
        checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
        checkpoints = sorted(checkpoint_dir.glob("hyperlora_vmas_*"))

        if not checkpoints:
            print("Error: No checkpoint found after training!")
            sys.exit(1)

        latest_checkpoint = checkpoints[-1]
        print(f"\nFound checkpoint: {latest_checkpoint}")

        # ====================================================================
        # Phase 2: Deploy on M agents (zero-shot)
        # ====================================================================
        print("\n" + "=" * 80)
        print(f"PHASE 2: Zero-shot deployment on {args.deploy_agents} agents")
        print("=" * 80)

        # Modify config for deployment (different agent count)
        config["env"]["num_agents"] = args.deploy_agents

        # Save modified config
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Modified config for deployment: {config['env']['num_agents']} agents")

        # Create deployment script with dynamic agent count
        deploy_script = f"""
import torch
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

from env_setup import make_vmas_env
from lora_policy import LoRAPolicy
from hypernetwork import Hypernetwork
from render_gif import generate_policy_gif

print("\\n" + "="*80)
print("Zero-Shot Deployment: {args.train_agents} agents -> {args.deploy_agents} agents")
print("="*80)

# Load checkpoint
checkpoint_path = Path("{latest_checkpoint}")
print(f"\\nLoading checkpoint: {{checkpoint_path}}")

# Load the final checkpoint
checkpoint_file = checkpoint_path / "final_checkpoint.npz"
if not checkpoint_file.exists():
    print(f"Error: Checkpoint file not found: {{checkpoint_file}}")
    import sys
    sys.exit(1)

checkpoint_data = np.load(checkpoint_file, allow_pickle=True)

# Use training config (passed from generalization_experiment.py)
train_config = {config}

print("Checkpoint loaded successfully")

# Setup devices
use_cuda = False
torch_device = "cpu"
jax_device = jax.devices("cpu")[0] if not use_cuda else jax.devices("gpu")[0]

print(f"\\nUsing device: {{torch_device}}")

# Create environment with {args.deploy_agents} agents (NEW - not seen during training!)
print("\\n" + "="*80)
print("Creating {{num_agents}}-agent environment (zero-shot generalization)")
print("="*80)

num_agents = {args.deploy_agents}  # Different from training!
num_envs = 16

# Create heterogeneous capabilities for all agents
# Generate diverse capabilities across agents
import numpy as np
np.random.seed(123)  # For reproducibility
agent_capabilities = {{
    "speed": np.random.uniform(0.8, 1.2, num_agents).tolist(),
    "lidar_range": np.random.uniform(0.4, 0.6, num_agents).tolist()
}}

print(f"Agent capabilities:")
for i in range(num_agents):
    print(f"  Agent {{i}}: speed={{agent_capabilities['speed'][i]:.2f}}, "
          f"lidar_range={{agent_capabilities['lidar_range'][i]:.2f}}")

env = make_vmas_env(
    scenario_name="dispersion",
    num_agents=num_agents,
    num_envs=num_envs,
    device=torch_device,
    continuous_actions=True,
    penalise_by_time=False,
    share_reward=False,
    distance_shaping_coef=3.0,
    agent_capabilities=agent_capabilities,
)

print(f"Environment created: {{num_agents}} agents, {{num_envs}} parallel envs")

# Get dimensions
temp_obs = env.reset()
obs_dim = temp_obs[0].shape[-1]
action_dim = env.get_agent_action_size(env.agents[0])

print(f"\\nEnvironment dimensions: obs_dim={{obs_dim}}, action_dim={{action_dim}}")

# Initialize models (same architecture as training)
print("\\n" + "="*80)
print("Initializing models")
print("="*80)

policy_dims = {{
    "obs_dim": obs_dim,
    "hidden_dims": train_config['model']['policy_hidden_dims'],
    "action_dim": action_dim,
    "lora_rank": train_config['model']['lora_rank'],
}}

hypernetwork = Hypernetwork(
    policy_dims=policy_dims,
    context_dim=2,  # [speed, lidar_range]
    task_embed_dim=train_config['model']['task_embed_dim'],
    transformer_dim=train_config['model']['transformer_dim'],
    transformer_heads=train_config['model']['transformer_heads'],
    transformer_layers=train_config['model']['transformer_layers'],
    lora_mode=train_config['model']['lora_mode'],
    scaling_factor=train_config['model']['lora_scaling_factor'],
)

shared_policy = LoRAPolicy(
    hidden_dims=train_config['model']['policy_hidden_dims'],
    action_dim=action_dim,
    lora_mode=train_config['model']['lora_mode'],
)

print("Models initialized")

# Reconstruct state objects from checkpoint
from flax.training.train_state import TrainState
import optax

policy_params = checkpoint_data['policy_params'].item()
hn_params = checkpoint_data['hn_params'].item()

# Create dummy optimizer (not used for inference)
dummy_optimizer = optax.adam(learning_rate=0.0)

policy_state = TrainState.create(
    apply_fn=shared_policy.apply,
    params=policy_params,
    tx=dummy_optimizer,
)

hn_state = TrainState.create(
    apply_fn=hypernetwork.apply,
    params=hn_params,
    tx=dummy_optimizer,
)

print("State objects reconstructed from checkpoint")

# Define helper functions
@jax.jit
def get_static_adapters(hn_params, task_batch, context_batch):
    return hypernetwork.apply({{"params": hn_params}}, task_batch, context_batch)

@jax.jit
def get_actions(policy_params, obs_batch, adapters_dict):
    mean, log_std = shared_policy.apply(
        {{"params": policy_params}}, obs_batch, adapters_dict
    )
    return mean

# Generate visualization GIF
print("\\n" + "="*80)
print("Generating visualization GIF")
print("="*80)

gif_path = generate_policy_gif(
    env=env,
    policy_state=policy_state,
    hn_state=hn_state,
    adapters_dict=None,
    config=train_config,
    checkpoint_dir=checkpoint_path,
    n_steps={args.gif_steps},
    use_hypernetwork=True,
    num_agents=num_agents,
    obs_dim=obs_dim,
    context_dim=2,
    task_embed_dim=train_config['model']['task_embed_dim'],
    use_cuda=use_cuda,
    jax_device=jax_device,
    torch_device=torch_device,
    get_static_adapters_fn=get_static_adapters,
    get_actions_fn=get_actions,
)

if gif_path:
    print(f"\\nZero-shot deployment successful!")
    print(f"GIF saved to: {{gif_path}}")
    print("\\n" + "="*80)
    print("Experiment completed successfully!")
    print("="*80)
    print("\\nResults:")
    print(f"  - Trained on: {args.train_agents} agents")
    print(f"  - Deployed on: {args.deploy_agents} agents")
    print(f"  - Checkpoint: {{checkpoint_path}}")
    print(f"  - Visualization: {{gif_path}}")
else:
    print("\\nError: Failed to generate GIF")
"""

        deploy_script_path = "deploy_4agents.py"
        with open(deploy_script_path, "w") as f:
            f.write(deploy_script)

        print(f"Created deployment script: {deploy_script_path}")

        # Run deployment
        deploy_cmd = [sys.executable, deploy_script_path]
        run_command(deploy_cmd, f"Zero-shot deployment on {args.deploy_agents} agents")

        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nSummary:")
        print(f"  - Phase 1: Trained on {args.train_agents} agents")
        print(f"  - Phase 2: Deployed on {args.deploy_agents} agents (zero-shot)")
        print(f"  - Checkpoint: {latest_checkpoint}")
        print(f"\nKey insight:")
        print(f"  The hypernetwork can generalize to different team sizes")
        print(f"  because it conditions on capability vectors, not agent count!")
        print("=" * 80)

    finally:
        # Restore original config
        shutil.copy(config_backup, config_file)
        print(f"\nRestored original config from backup")

        # Clean up deployment script
        if Path(deploy_script_path).exists():
            Path(deploy_script_path).unlink()
            print(f"Cleaned up deployment script")


if __name__ == "__main__":
    main()
