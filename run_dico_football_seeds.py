#!/usr/bin/env python3
"""
Run DICO football experiments with multiple Bettini seeds sequentially.

Uses the config_dico_football.yaml as base and iterates over seeds.

Example usage:
    # Run with default Bettini seeds [0, 42, 342, 3421, 34210]
    python run_dico_football_seeds.py

    # Run with custom seeds
    python run_dico_football_seeds.py --seeds 0 1 2

    # Run with multiple target SND values x seeds
    python run_dico_football_seeds.py --target-snd 0.1 0.2 0.5 --seeds 0 42 342

    # Enable wandb logging
    python run_dico_football_seeds.py --wandb

    # Custom wandb project name
    python run_dico_football_seeds.py --wandb --wandb-project my-dico-football

    # Stop on first failure
    python run_dico_football_seeds.py --stop-on-error
"""

import yaml
import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime
import shutil

BASE_CONFIG = "config_dico_football.yaml"
DEFAULT_SEEDS = [0, 42, 342, 3421, 34210]


def create_temp_config(seed, base_config_path, temp_dir, target_snd=None):
    """Create a temporary config with the given seed (and optional target_snd)."""
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    config["training"]["seed"] = seed

    if target_snd is not None:
        config["training"]["target_snd"] = target_snd

    # Build descriptive experiment name
    name_parts = ["dico_football"]
    if target_snd is not None:
        name_parts.append(f"snd{target_snd}")
    name_parts.append(f"seed{seed}")
    config["experiment"]["name"] = "_".join(name_parts)

    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)

    snd_tag = f"_snd{target_snd}" if target_snd is not None else ""
    temp_config_path = temp_path / f"config_dico_football{snd_tag}_seed{seed}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return temp_config_path


def run_training(
    config_path,
    seed,
    use_wandb=False,
    wandb_project=None,
    wandb_entity=None,
    target_snd=None,
):
    """Run a single DICO football training via train.py."""
    snd_str = f"snd{target_snd}" if target_snd is not None else ""
    print(f"\n{'=' * 80}")
    print(f"DICO Football | seed={seed} {snd_str}")
    print(f"Config : {config_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

    cmd = [
        sys.executable,
        "train.py",
        "--config",
        str(config_path),
        "--use-dico",
        "true",
    ]

    # Always pass --wandb and project/entity so train.py logs to wandb.
    # The config already has wandb_project set, but being explicit avoids
    # any ambiguity.
    if use_wandb or wandb_project:
        cmd.append("--wandb")
        if wandb_project:
            cmd.extend(["--wandb-project", wandb_project])
        if wandb_entity:
            cmd.extend(["--wandb-entity", wandb_entity])

        run_name_parts = ["dico", "football"]
        if target_snd is not None:
            run_name_parts.append(f"snd{target_snd}")
        run_name_parts.append(f"seed{seed}")
        cmd.extend(["--wandb-name", "_".join(run_name_parts)])

    result = subprocess.run(cmd, cwd=Path.cwd())
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run DICO football experiments with multiple seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=BASE_CONFIG,
        help=f"Base config file (default: {BASE_CONFIG})",
    )
    parser.add_argument(
        "--target-snd",
        type=float,
        nargs="+",
        default=None,
        help="Target SND value(s). If multiple, runs all combos with seeds.",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="temp_configs",
        help="Temp config directory (default: temp_configs)",
    )
    parser.add_argument(
        "--keep-configs", action="store_true", help="Keep temp configs after completion"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging (overrides --wandb)",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="W&B project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity/username"
    )
    parser.add_argument(
        "--stop-on-error", action="store_true", help="Stop on first failure"
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    # Read base config to pick up wandb settings
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    logging_cfg = base_config.get("logging", {})
    wandb_project = args.wandb_project or logging_cfg.get("wandb_project")
    wandb_entity = args.wandb_entity or logging_cfg.get("wandb_entity")

    snd_values = args.target_snd if args.target_snd is not None else [None]
    use_wandb = args.wandb and not args.no_wandb
    total_runs = len(snd_values) * len(args.seeds)

    print(f"\n{'=' * 80}")
    print(f"DICO Football Multi-Seed Runner")
    print(f"{'=' * 80}")
    print(f"Base config : {args.config}")
    print(f"Seeds       : {args.seeds}")
    print(f"Target SND  : {snd_values if args.target_snd else '(from config)'}")
    print(f"Total runs  : {total_runs}")
    print(f"W&B logging : {'Enabled' if use_wandb else 'Disabled'}")
    if use_wandb:
        print(f"W&B project : {wandb_project or '(none)'}")
    print(f"{'=' * 80}\n")

    results = []
    failed_runs = []

    try:
        run_num = 0
        for target_snd in snd_values:
            for seed in args.seeds:
                run_num += 1
                snd_str = f"{target_snd:.2f}" if target_snd is not None else "(config)"

                print(f"\n{'#' * 80}")
                print(f"# Run {run_num}/{total_runs}: SND={snd_str}, Seed={seed}")
                print(f"{'#' * 80}")

                temp_config = create_temp_config(
                    seed,
                    base_config_path=args.config,
                    temp_dir=args.temp_dir,
                    target_snd=target_snd,
                )

                rc = run_training(
                    temp_config,
                    seed,
                    use_wandb=use_wandb,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    target_snd=target_snd,
                )

                results.append((target_snd, seed, rc))

                if rc != 0:
                    print(f"\nWARNING: SND={snd_str}, seed={seed} failed (code {rc})")
                    failed_runs.append((target_snd, seed))
                    if args.stop_on_error:
                        print("\nStopping due to --stop-on-error")
                        break
                else:
                    print(f"\nSND={snd_str}, seed={seed} completed successfully")

            if args.stop_on_error and failed_runs:
                break

    finally:
        if not args.keep_configs:
            temp_path = Path(args.temp_dir)
            if temp_path.exists():
                shutil.rmtree(temp_path)
                print(f"\nCleaned up {args.temp_dir}/")
        else:
            print(f"\nKept temp configs in {args.temp_dir}/")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total runs : {len(results)}")
    print(f"Successful : {sum(1 for r in results if r[2] == 0)}")
    print(f"Failed     : {len(failed_runs)}")

    if failed_runs:
        print("\nFailed runs:")
        for tsnd, seed in failed_runs:
            snd_str = f"{tsnd:.2f}" if tsnd is not None else "(config)"
            print(f"  SND={snd_str}, Seed={seed}")

    print("\nDetailed results:")
    for tsnd, seed, rc in results:
        snd_str = f"{tsnd:.2f}" if tsnd is not None else "(config)"
        status = "OK" if rc == 0 else f"FAILED (code {rc})"
        print(f"  SND={snd_str:>8}, Seed {seed:5d}: {status}")

    print(f"{'=' * 80}\n")
    sys.exit(1 if failed_runs else 0)


if __name__ == "__main__":
    main()
