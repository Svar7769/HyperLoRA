import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- USER CONFIGURATION ---
ENTITY = "hbuechi"  # Replace with your WandB username/team name
PROJECT = "hyperlora-vmas"  # Replace with your Project name

# Metric to plot
METRIC = "eval/snd"

# Output directory
OUTPUT_DIR = "/Users/hannesbuchi/Documents/Cambridge/MT/Outputs/Eval"

# Run prefix to match
RUN_PREFIX = "hyperlora_run_4agents_rank8tran4_nodist_seed"

# Plot settings
PLOT_TITLE = "System Neural Diversity (SND)"
X_LABEL = "Episode"
Y_LABEL = "SND"
# --------------------------


def get_snd_data():
    """Fetch SND data over time for all matching runs."""
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    all_data = []

    print(f"Scanning runs in {ENTITY}/{PROJECT}...")
    print(f"Looking for runs starting with: {RUN_PREFIX}")

    for run in runs:
        if not run.name.startswith(RUN_PREFIX):
            continue

        print(f"  Found: {run.name}")

        try:
            # Download history for the SND metric
            # Also get _step for x-axis
            history = run.history(keys=[METRIC], x_axis="_step", pandas=True)

            if history.empty:
                print(f"    -> No {METRIC} data found, skipping.")
                continue

            # Clean up the data
            history = history.dropna(subset=[METRIC])

            if history.empty:
                print(f"    -> All {METRIC} values are NaN, skipping.")
                continue

            # Add run identifier
            history["Run"] = run.name

            # Extract seed number from run name for legend (only the part before underscore)
            seed_str = run.name.replace(RUN_PREFIX, "").strip("_")
            seed_num = seed_str.split("_")[0] if "_" in seed_str else seed_str
            history["Seed"] = f"Seed {seed_num}" if seed_num else run.name

            all_data.append(history)
            print(f"    -> Loaded {len(history)} data points")

        except Exception as e:
            print(f"    -> Error: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def plot_snd(df):
    """Create SND over time plot."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each run as a separate line
    for seed in df["Seed"].unique():
        seed_data = df[df["Seed"] == seed].sort_values("_step")
        ax.plot(seed_data["_step"], seed_data[METRIC], label=seed, alpha=0.8)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(PLOT_TITLE)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(OUTPUT_DIR, "snd_over_time.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {output_path}")

    # Also create a plot with mean and std across seeds
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Group by step and compute mean/std
    grouped = df.groupby("_step")[METRIC].agg(["mean", "std"]).reset_index()

    ax2.plot(grouped["_step"], grouped["mean"], color="blue", linewidth=2, label="Mean")
    ax2.fill_between(
        grouped["_step"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        alpha=0.3,
        color="blue",
        label="±1 Std",
    )

    ax2.set_xlabel(X_LABEL)
    ax2.set_ylabel(Y_LABEL)
    ax2.set_title(f"{PLOT_TITLE} (Mean ± Std)")
    ax2.legend(loc="best")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save plot
    output_path2 = os.path.join(OUTPUT_DIR, "snd_over_time_mean.png")
    fig2.savefig(output_path2, dpi=300)
    plt.close(fig2)
    print(f"Saved: {output_path2}")


# --- MAIN ---
if __name__ == "__main__":
    df = get_snd_data()

    if df.empty:
        print("\nNo matching runs found with SND data.")
        print("Please check:")
        print(f"  - Entity: {ENTITY}")
        print(f"  - Project: {PROJECT}")
        print(f"  - Run prefix: {RUN_PREFIX}")
        print(f"  - Metric: {METRIC}")
    else:
        print(f"\nLoaded {len(df)} total data points from {df['Run'].nunique()} runs.")
        plot_snd(df)
        print("\nDone!")
