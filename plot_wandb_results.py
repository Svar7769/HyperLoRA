import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- USER CONFIGURATION ---
ENTITY = "hbuechi"  # Replace with your WandB username/team name
PROJECT = "hyperlora-vmas"  # Replace with your Project name

# Metrics
METRIC_PRIMARY = "eval/completion_rate"  # We maximize this to find the "best" step
METRIC_SECONDARY = "eval/avg_episode_length"  # We grab this value at that same step

# Output directory
OUTPUT_DIR = "/Users/hannesbuchi/Documents/Cambridge/MT/Outputs/Eval"

# Run prefixes to compare and their labels (from WandB)
RUN_GROUPS = {
    "hyperlora_run_4agents_rank8tran4_nodist_seed": "HyperLoRA (4 agents trained)",
}

# Custom color palette for methods
METHOD_COLORS = {
    "HyperLoRA (capability input)": "#4CAF50",  # Green
    "NoPS Baseline": "#2196F3",  # Blue
    "HyperLoRA (LiDAR input)": "#9B59B6",  # Purple
    "HyperLoRA (4 agents trained)": "#FFC107",  # Yellow
    "2→4 Generalization": "#E53935",  # Red
    "Other": "#999999",  # Gray (for future use)
}

# --- MANUAL EVAL RESULTS ---
# Add your eval results here after running:
#   python evaluate.py --checkpoint <checkpoint> --num-agents 4 --num-eval-episodes 100 --no-gif
# Format: list of tuples (completion_rate, avg_episode_length)
MANUAL_EVAL_RESULTS = {
    "2→4 Generalization": [
        (11.94, 97.45),  # seed0
        (14.69, 96.69),  # seed1
        (12.97, 97.00),  # seed2
        (15.25, 97.02),  # seed3
        (12.38, 97.40),  # seed4
    ]
}
# --------------------------


def get_run_data():
    # Since you are logged in, this automatically finds your session
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    data_list = []

    print(f"Scanning runs in {ENTITY}/{PROJECT}...")

    for run in runs:
        # 1. PARSE THE NAME
        # Check if run name starts with one of our target prefixes
        method_name = None
        for prefix, label in RUN_GROUPS.items():
            if run.name.startswith(prefix):
                method_name = label
                break

        if method_name is None:
            # Skip runs that don't match any of our target prefixes
            continue

        # 2. FIND THE BEST EVAL POINT
        try:
            # Download full history for the two metrics
            history = run.history(keys=[METRIC_PRIMARY, METRIC_SECONDARY])

            if history.empty:
                continue

            # Find the step where Completion Rate is highest (Max)
            # idxmax() gives us the index of the maximum value
            best_idx = history[METRIC_PRIMARY].idxmax()
            best_step = history.iloc[best_idx]

            # 3. STORE DATA
            data_list.append(
                {
                    "Method": method_name,
                    "Run Name": run.name,
                    "Best Completion Rate": best_step[METRIC_PRIMARY],
                    "Episode Length": best_step[METRIC_SECONDARY],
                }
            )
            print(
                f"  ✓ {run.name} -> {method_name} (Best CR: {best_step[METRIC_PRIMARY]:.4f})"
            )

        except KeyError:
            print(f"Skipping {run.name}: Metrics not found.")
            continue

    # Print summary by method
    print("\n--- SUMMARY ---")
    for label in RUN_GROUPS.values():
        method_runs = [d for d in data_list if d["Method"] == label]
        print(f"{label}: {len(method_runs)} runs")
        for r in method_runs:
            print(f"  - {r['Run Name']}")

    return pd.DataFrame(data_list)


def add_manual_results(df):
    """Add manual evaluation results to the dataframe."""
    manual_data = []
    for method_name, results in MANUAL_EVAL_RESULTS.items():
        for i, (completion_rate, episode_length) in enumerate(results):
            manual_data.append(
                {
                    "Method": method_name,
                    "Run Name": f"{method_name}_seed{i}",
                    "Best Completion Rate": completion_rate,
                    "Episode Length": episode_length,
                }
            )

    if manual_data:
        manual_df = pd.DataFrame(manual_data)
        print(f"\n--- MANUAL RESULTS ---")
        print(f"{list(MANUAL_EVAL_RESULTS.keys())[0]}: {len(manual_data)} runs")
        df = pd.concat([df, manual_df], ignore_index=True)

    return df


# --- PLOTTING ---
df = get_run_data()
df = add_manual_results(df)

if not df.empty:
    print(f"Found {len(df)} runs. Generating plots...")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create filename suffix from method names
    method_suffix = "_".join(sorted(df["Method"].unique()))

    # Build color palette in order of methods appearing in data
    methods_in_order = df["Method"].unique()
    palette = [METHOD_COLORS.get(m, "#999999") for m in methods_in_order]

    # Plot 1: Completion Rate
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x="Method", y="Best Completion Rate", ax=ax1, palette=palette)
    ax1.set_title("Completion Rate")
    ax1.set_ylabel("Completion Rate [%]")
    ax1.set_xlabel("Method")
    ax1.set_ylim(10, 50)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig1.savefig(
        os.path.join(OUTPUT_DIR, f"completion_rate_{method_suffix}.png"), dpi=300
    )
    plt.close(fig1)
    print(f"Saved: {os.path.join(OUTPUT_DIR, f'completion_rate_{method_suffix}.png')}")

    # Plot 2: Episode Length
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x="Method", y="Episode Length", ax=ax2, palette=palette)
    ax2.set_title("Average Episode Length")
    ax2.set_ylabel("Average Episode Length [Steps]")
    ax2.set_xlabel("Method")
    ax2.set_ylim(80, 100)
    ax2.axhline(
        y=100,
        color="red",
        linestyle="--",
        linewidth=4.0,
        alpha=0.7,
        label="Max Episode Length",
    )
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(
        os.path.join(OUTPUT_DIR, f"episode_length_{method_suffix}.png"), dpi=300
    )
    plt.close(fig2)
    print(f"Saved: {os.path.join(OUTPUT_DIR, f'episode_length_{method_suffix}.png')}")

    print("Done!")
else:
    print("No matching runs found. Please check your Entity and Project names.")
