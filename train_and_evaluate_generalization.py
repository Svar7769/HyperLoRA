"""
Train with mixed agent training and evaluate generalization to unseen team sizes.

This script:
1. Trains dispersion with 4 agents (mixed: 2-4) for specified episodes
2. Finds the best checkpoint based on completion rate from wandb
3. Evaluates on 5 and 6 agents to test generalization

Usage:
    python train_and_evaluate_generalization.py --num-episodes 1000
    python train_and_evaluate_generalization.py --num-episodes 500 --no-wandb
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import time
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate generalization to unseen team sizes"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Base config file (default: config.yaml)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging during training",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per team size (default: 50)",
    )
    parser.add_argument(
        "--eval-team-sizes",
        type=str,
        default="5,6",
        help="Comma-separated team sizes to evaluate (default: 5,6)",
    )
    parser.add_argument(
        "--train-min-agents",
        type=int,
        default=2,
        help="Minimum agents during training (default: 2)",
    )
    parser.add_argument(
        "--train-max-agents",
        type=int,
        default=4,
        help="Maximum agents during training (default: 4)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456,789,1024",
        help="Comma-separated list of seeds to run (default: 42,123,456,789,1024)",
    )
    return parser.parse_args()


def find_best_checkpoint_from_wandb(checkpoint_dir: Path, run_name: str = None):
    """
    Find the best checkpoint based on completion rate from wandb logs.

    Args:
        checkpoint_dir: Directory containing checkpoints
        run_name: Optional wandb run name to filter by

    Returns:
        Path to best checkpoint directory
    """
    try:
        import wandb

        # Get wandb API
        api = wandb.Api()

        # Find runs in the project
        runs = api.runs("hyperlora-vmas")  # Default project name from config

        # Filter to recent runs
        recent_runs = []
        for run in runs:
            if run.state == "finished" or run.state == "running":
                # Check if this is our training run (you can filter by tags, name, etc.)
                if "dispersion" in run.config.get("env", {}).get("scenario_name", ""):
                    recent_runs.append(run)

        if not recent_runs:
            print("⚠️  No wandb runs found. Using latest checkpoint by timestamp.")
            return find_latest_checkpoint_by_time(checkpoint_dir)

        # Sort by completion rate
        recent_runs.sort(
            key=lambda r: r.summary.get("eval/completion_rate", 0), reverse=True
        )

        best_run = recent_runs[0]
        best_completion = best_run.summary.get("eval/completion_rate", 0)

        print(f"✓ Best run from wandb: {best_run.name}")
        print(f"  Completion rate: {best_completion:.2f}%")
        print(
            f"  Episodes: {best_run.config.get('training', {}).get('num_episodes', 'N/A')}"
        )

        # Find corresponding checkpoint
        # Wandb run names typically match checkpoint directory names
        checkpoint_path = checkpoint_dir / f"hyperlora_vmas_{best_run.name}"

        if not checkpoint_path.exists():
            # Try to find by timestamp or ID in the run
            print(
                f"⚠️  Checkpoint {checkpoint_path} not found. Searching by timestamp..."
            )
            return find_latest_checkpoint_by_time(checkpoint_dir)

        return checkpoint_path

    except ImportError:
        print("⚠️  wandb not installed. Using latest checkpoint by timestamp.")
        return find_latest_checkpoint_by_time(checkpoint_dir)
    except Exception as e:
        print(f"⚠️  Error accessing wandb: {e}. Using latest checkpoint.")
        return find_latest_checkpoint_by_time(checkpoint_dir)


def find_latest_checkpoint_by_time(checkpoint_dir: Path):
    """Find the most recent checkpoint by modification time."""
    # Try multiple directory patterns
    patterns = [
        "generalization_experiment_*",
        "hyperlora_vmas_*",
        "*_202*",  # Any directory with timestamp pattern
    ]

    checkpoints = []
    for pattern in patterns:
        checkpoints = sorted(checkpoint_dir.glob(pattern))
        if checkpoints:
            break

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint directories found in {checkpoint_dir}")

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = checkpoints[0]

    print(f"✓ Using latest checkpoint: {latest.name}")
    return latest


def find_best_checkpoint_in_dir(checkpoint_path: Path):
    """
    Find the best checkpoint file within a checkpoint directory.
    Looks for checkpoint files with various naming patterns and selects the latest.
    """
    # Try different checkpoint naming patterns
    patterns = [
        "final_checkpoint.npz",  # Primary pattern from train.py
        "checkpoint_episode_*.npz",
        "checkpoint_episode_*.pkl",
        "*.npz",
        "checkpoint_*.pkl",
        "*.pkl",
        "policy_*.pkl",
        "best_checkpoint.pkl",
    ]

    checkpoint_files = []
    for pattern in patterns:
        checkpoint_files = list(checkpoint_path.glob(pattern))
        if checkpoint_files:
            print(
                f"  Found {len(checkpoint_files)} checkpoint files matching '{pattern}'"
            )
            break

    if not checkpoint_files:
        # List what files ARE in the directory for debugging
        all_files = list(checkpoint_path.glob("*"))
        print(f"  Files in {checkpoint_path.name}:")
        for f in all_files[:10]:  # Show first 10 files
            print(f"    - {f.name}")
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")

    # Extract episode numbers and sort
    def get_episode_num(path):
        try:
            # Try to extract episode number from filename
            parts = path.stem.split("_")
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
            # If no number found, use modification time
            return int(path.stat().st_mtime)
        except:
            return 0

    checkpoint_files.sort(key=get_episode_num, reverse=True)
    best_checkpoint = checkpoint_files[0]

    print(f"  Using checkpoint: {best_checkpoint.name}")
    return best_checkpoint


def run_training(args, seed):
    """Run training with mixed agent configuration for a specific seed."""
    print("=" * 80)
    print(f"PHASE 1: TRAINING (Seed {seed})")
    print("=" * 80)
    print(
        f"Training dispersion with {args.train_min_agents}-{args.train_max_agents} agents (mixed)"
    )
    print(f"Episodes: {args.num_episodes}")
    print(f"Seed: {seed}")
    print(f"Config: {args.config}")
    print()

    # Build training command
    # NOTE: We rely on the config file to have mixed_agent_training=true
    # The --num-agents flag sets the max, but mixed training will vary from min_agents to this
    cmd = [
        "python",
        "train.py",
        "--config",
        args.config,
        "--num-episodes",
        str(args.num_episodes),
        "--seed",
        str(seed),
    ]

    # Add wandb flag if not disabled
    if not args.no_wandb:
        cmd.append("--wandb")

    print(f"Running: {' '.join(cmd)}")
    print(f"Expected behavior: Train with mixed teams of 2-4 agents")
    print()

    # Run training
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"❌ Training failed with exit code {result.returncode}")
        return False

    print()
    print(f"✓ Training completed successfully for seed {seed}!")
    print()
    return True


def run_evaluation(
    checkpoint_dir: Path, num_agents: int, num_eval_episodes: int, base_config_path: str
):
    """Run evaluation on a specific team size."""
    print(f"\nEvaluating on {num_agents} agents (UNSEEN team size)...")

    # Create a temporary config with updated max_agents for evaluation
    with open(base_config_path, "r") as f:
        eval_config = yaml.safe_load(f)

    # Update max_agents to accommodate evaluation team size
    eval_config["env"]["max_agents"] = max(
        num_agents, eval_config["env"].get("max_agents", num_agents)
    )

    # Save temporary eval config
    temp_config_path = (
        Path(base_config_path).parent / f"temp_eval_config_{num_agents}.yaml"
    )
    with open(temp_config_path, "w") as f:
        yaml.dump(eval_config, f)

    try:
        cmd = [
            "python",
            "evaluate.py",
            "--checkpoint",
            str(checkpoint_dir),
            "--num-agents",
            str(num_agents),
            "--num-eval-episodes",
            str(num_eval_episodes),
            "--config",
            str(temp_config_path),  # Use temporary config
            "--no-gif",
        ]

        print(f"  Command: {' '.join(cmd)}")

        # Capture output to parse completion rate
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        # Clean up temporary config
        if temp_config_path.exists():
            temp_config_path.unlink()

    if result.returncode != 0:
        print(f"❌ Evaluation failed with exit code {result.returncode}")
        print(f"STDERR:\n{result.stderr}")
        print(f"STDOUT (first 1000 chars):\n{result.stdout[:1000]}")
        return None

    # Parse completion rate from output
    output = result.stdout

    # Debug: Show output length and last part (where metrics should be)
    if len(output) < 100:
        print(f"  ⚠️  Suspiciously short output ({len(output)} chars)")
        print(f"  Full output:\n{output}")
        print(f"  STDERR:\n{result.stderr}")

    completion_rate = None
    avg_reward = None
    avg_length = None

    for line in output.split("\n"):
        if "Completion Rate" in line:
            try:
                # Extract percentage (format: "Completion Rate:    XX.XX% (Y/Z episodes)")
                parts = line.split(":")[1].strip().split("%")[0]
                completion_rate = float(parts)
            except:
                pass
        if "Avg Reward:" in line:
            try:
                # Extract reward (format: "Avg Reward:         X.XXX ± Y.YYY")
                parts = line.split(":")[1].strip().split("±")[0]
                avg_reward = float(parts)
            except:
                pass
        if "Avg Episode Length:" in line:
            try:
                # Extract length (format: "Avg Episode Length: XX.XX ± YY.YY steps")
                parts = line.split(":")[1].strip().split("±")[0]
                avg_length = float(parts)
            except:
                pass

    # Debug: show what was parsed and the last lines of output
    if completion_rate is None:
        print(f"  ⚠️  Failed to parse metrics from evaluate.py output")
        print(f"  Output length: {len(output)} chars")
        print(f"  Last 1000 chars of output:\n{output[-1000:]}")
        print(f"  STDERR (if any):\n{result.stderr}")

    return {
        "num_agents": num_agents,
        "completion_rate": completion_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
    }


def main():
    args = parse_args()

    # Convert paths
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Parse evaluation team sizes and seeds
    eval_team_sizes = [int(x.strip()) for x in args.eval_team_sizes.split(",")]
    seeds = [int(x.strip()) for x in args.seeds.split(",")]

    print()
    print("=" * 80)
    print(
        f"Training: Mixed agent training on {args.train_min_agents}-{args.train_max_agents} agents"
    )
    print(f"Evaluation: Test on UNSEEN team sizes: {eval_team_sizes}")
    print(f"Seeds: {seeds}")
    print(f"Training episodes per seed: {args.num_episodes}")
    print(f"Eval episodes per team size: {args.num_eval_episodes}")
    print(f"Config file: {args.config}")
    print("=" * 80)
    print()
    print("NOTE: The config file MUST have mixed_agent_training=true")
    print("      and max_agents >= max(eval_team_sizes)")
    print()

    # Check if config has mixed_agent_training enabled
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if not config.get("env", {}).get("mixed_agent_training", False):
        print("⚠️  WARNING: mixed_agent_training is not enabled in config!")
        print("   Training will use fixed team size instead of mixed.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != "y":
            print("Aborting.")
            sys.exit(0)

    # Store results across all seeds
    all_seed_results = []

    # Run experiment for each seed
    overall_start = time.time()

    for seed_idx, seed in enumerate(seeds, 1):
        print()
        print("#" * 80)
        print(f"# SEED {seed_idx}/{len(seeds)}: {seed}")
        print("#" * 80)
        print()

        # Phase 1: Training
        seed_start = time.time()
        success = run_training(args, seed)
        if not success:
            print(f"⚠️  Skipping seed {seed} due to training failure")
            continue

        training_time = time.time() - seed_start

        # Phase 2: Find best checkpoint for this seed
        print("=" * 80)
        print(f"PHASE 2: FINDING CHECKPOINT (Seed {seed})")
        print("=" * 80)

        # Find latest checkpoint directory (should be from this training run)
        try:
            checkpoint_dir_path = find_latest_checkpoint_by_time(checkpoint_dir)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print(f"⚠️  Skipping seed {seed} - no checkpoint found")
            continue

        print()

        # Phase 3: Evaluation on unseen team sizes
        print("=" * 80)
        print(f"PHASE 3: EVALUATION (Seed {seed})")
        print("=" * 80)

        seed_results = {
            "seed": seed,
            "training_time": training_time,
            "checkpoint": str(checkpoint_dir_path),
            "evaluations": {},
        }

        for num_agents in eval_team_sizes:
            metrics = run_evaluation(
                checkpoint_dir_path, num_agents, args.num_eval_episodes, args.config
            )
            if metrics and metrics.get("completion_rate") is not None:
                seed_results["evaluations"][num_agents] = metrics
                print(
                    f"  ✓ {num_agents} agents: {metrics['completion_rate']:.2f}% completion"
                )
            else:
                print(
                    f"  ❌ {num_agents} agents: evaluation failed or metrics not parsed"
                )
                if metrics:
                    print(f"      Metrics returned: {metrics}")

        all_seed_results.append(seed_results)

        print()
        print(f"✓ Seed {seed} completed in {(time.time() - seed_start)/60:.1f} minutes")

    total_time = time.time() - overall_start

    # Aggregate results across seeds
    print()
    print("=" * 80)
    print("AGGREGATED RESULTS ACROSS ALL SEEDS")
    print("=" * 80)
    print(f"Total experiment time: {total_time/60:.1f} minutes")
    print(f"Successful seeds: {len(all_seed_results)}/{len(seeds)}")
    print(f"Training range: {args.train_min_agents}-{args.train_max_agents} agents")
    print()

    # Compute statistics for each team size
    print("Generalization Performance (Mean ± Std):")
    print("-" * 80)
    print(
        f"{'Team Size':<12} {'Completion Rate':<25} {'Avg Reward':<25} {'Avg Length':<25}"
    )
    print("-" * 80)

    for num_agents in eval_team_sizes:
        # Collect metrics across seeds
        completion_rates = []
        rewards = []
        lengths = []

        for seed_result in all_seed_results:
            if num_agents in seed_result["evaluations"]:
                metrics = seed_result["evaluations"][num_agents]
                if metrics["completion_rate"] is not None:
                    completion_rates.append(metrics["completion_rate"])
                if metrics["avg_reward"] is not None:
                    rewards.append(metrics["avg_reward"])
                if metrics["avg_length"] is not None:
                    lengths.append(metrics["avg_length"])

        # Compute mean and std
        if completion_rates:
            comp_mean = sum(completion_rates) / len(completion_rates)
            comp_std = (
                sum((x - comp_mean) ** 2 for x in completion_rates)
                / len(completion_rates)
            ) ** 0.5
            comp_str = f"{comp_mean:.2f}% ± {comp_std:.2f}"
        else:
            comp_str = "N/A"

        if rewards:
            rew_mean = sum(rewards) / len(rewards)
            rew_std = (sum((x - rew_mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
            rew_str = f"{rew_mean:.2f} ± {rew_std:.2f}"
        else:
            rew_str = "N/A"

        if lengths:
            len_mean = sum(lengths) / len(lengths)
            len_std = (sum((x - len_mean) ** 2 for x in lengths) / len(lengths)) ** 0.5
            len_str = f"{len_mean:.2f} ± {len_std:.2f}"
        else:
            len_str = "N/A"

        print(f"{num_agents:<12} {comp_str:<25} {rew_str:<25} {len_str:<25}")

    print("=" * 80)
    print()

    # Detailed per-seed results
    print("Per-Seed Breakdown:")
    print("-" * 80)
    for seed_result in all_seed_results:
        print(f"\nSeed {seed_result['seed']}:")
        print(f"  Training time: {seed_result['training_time']/60:.1f} min")
        print(f"  Checkpoint: {Path(seed_result['checkpoint']).name}")
        for num_agents, metrics in seed_result["evaluations"].items():
            comp = (
                f"{metrics['completion_rate']:.2f}%"
                if metrics["completion_rate"]
                else "N/A"
            )
            print(f"  {num_agents} agents: {comp}")

    print()
    print("=" * 80)
    print(f"✓ Multi-seed experiment completed!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
