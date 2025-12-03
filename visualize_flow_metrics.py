#!/usr/bin/env python3
"""
Visualize flow-metrics results with:
    • Method comparison
    • Per-population heatmaps
    • Scatter plots
    • Dataset name extracted from symlink
    • Consistent colors per method
    • Pretty bars with metric values
    • Run selection or random run
"""

import argparse
import gzip
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")


# ================================================================
# Extract dataset name from symlink: dataset_name-<NAME>_transformed-...
# ================================================================
def extract_dataset_name(root_path):
    root = Path(root_path)

    # find symlinks or directories starting with dataset_name-
    for child in root.iterdir():
        name = child.name
        if name.startswith("dataset_name-"):
            # example: dataset_name-Levine_13dim_transformed-false
            # extract part after "dataset_name-"
            part = name[len("dataset_name-"):]
            dataset = part.split("_transformed")[0]
            return dataset

    return "unknown"


# ================================================================
# Extract method from path: analysis/<method>/default
# ================================================================
def extract_method(path):
    parts = Path(path).parts
    if "analysis" in parts:
        idx = parts.index("analysis")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


# ================================================================
# Locate all flow metrics files
# ================================================================
def find_metric_files(root):
    return list(Path(root).rglob("*.flow_metrics.json.gz"))


# ================================================================
# Load & flatten metrics
# ================================================================
def load_metrics(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)

    dataset_from_json = data.get("name", "unknown")
    method = extract_method(path)
    metrics = data["results"]

    rows = []
    for run_label, metric_block in metrics.items():
        row = {
            "json_dataset": dataset_from_json,
            "method": method,
            "run": run_label,
            "file": str(path)
        }
        # flatten everything except per_population
        for key, value in metric_block.items():
            if isinstance(value, dict):
                continue
            row[key] = value

        rows.append(row)

    return pd.DataFrame(rows), data


# ================================================================
# Consistent random colors per method
# ================================================================
def build_method_colors(methods):
    random.seed(42)
    return {
        method: "#%06x" % random.randint(0, 0xFFFFFF)
        for method in methods
    }


# ================================================================
# Run selection
# ================================================================
def filter_to_single_run(df, json_blocks, selected_run=None, random_run=False):
    if selected_run:
        df = df[df["run"] == selected_run]
        new_json = []
        for js in json_blocks:
            if selected_run in js["results"]:
                js["results"] = {selected_run: js["results"][selected_run]}
                new_json.append(js)
        return df, new_json

    if random_run:
        choices = df["run"].unique().tolist()
        pick = random.choice(choices)
        print(f"🎲 Random run selected: {pick}")
        df = df[df["run"] == pick]

        new_json = []
        for js in json_blocks:
            if pick in js["results"]:
                js["results"] = {pick: js["results"][pick]}
                new_json.append(js)
        return df, new_json

    return df, json_blocks


# ================================================================
# Save figure
# ================================================================
def save_plot(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# BARPLOTS comparing methods
# ================================================================
def plot_metric_bars(df, outdir, method_colors, dataset_name):
    numeric_cols = [
        c for c in df.columns
        if c not in ("json_dataset", "run", "method", "file")
        and df[c].dtype != object
    ]

    for metric in numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 4))

        # Draw bars with default colors
        sns.barplot(
            data=df,
            x="method",
            y=metric,
            color="lightgray",   # temporary color
            ax=ax
        )

        # Manually color bars by method
        for patch, (_, row) in zip(ax.patches, df.iterrows()):
            method = row["method"]
            patch.set_facecolor(method_colors[method])

        # Add metric values above bars
        for patch in ax.patches:
            height = patch.get_height()
            if not np.isnan(height):
                ax.text(
                    patch.get_x() + patch.get_width()/2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

        ax.set_title(f"{metric} comparison across methods (dataset: {dataset_name})")
        ax.set_ylabel(metric)
        ax.set_xlabel("Method")

        # ---- MANUAL LEGEND (METHODS) ----
        handles = [
            plt.matplotlib.patches.Patch(
                color=method_colors[m],
                label=m
            )
            for m in method_colors
        ]
        ax.legend(
            handles=handles,
            title="Method",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5)
        )

        save_plot(fig, outdir, f"metric_{metric}")


# ================================================================
# SCATTER PLOTS
# ================================================================
def plot_correlations(df, outdir, method_colors, dataset_name):
    combos = [
        ("scalability_seconds_per_item", "accuracy"),
        ("runtime_seconds", "accuracy"),
        ("mcc", "accuracy"),
        ("aucroc", "accuracy")
    ]

    for x, y in combos:
        if x not in df.columns or y not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue="method",
            palette=method_colors,
            s=100,
            ax=ax
        )
        ax.set_title(f"{y} vs {x} (dataset: {dataset_name})")
        ax.legend(title="Method", loc="center left", bbox_to_anchor=(1.02, 0.5))

        save_plot(fig, outdir, f"scatter_{y}_vs_{x}")


# ================================================================
# PER-POPULATION HEATMAPS
# ================================================================
def plot_population_heatmaps(json_blocks, outdir, dataset_name):
    for js in json_blocks:
        for run_label, block in js["results"].items():
            per_pop = block.get("per_population")
            if per_pop is None:
                continue

            df = pd.DataFrame(per_pop).T

            fig, ax = plt.subplots(figsize=(9, 5))
            sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", ax=ax)

            ax.set_title(
                f"Per-population metrics (method: {js['name']}, run: {run_label}, dataset: {dataset_name})"
            )

            save_plot(fig, outdir, f"per_population_{js['name']}_run_{run_label}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="flow_plots")
    parser.add_argument("--select-run", type=str, default=None)
    parser.add_argument("--random-run", action="store_true")
    args = parser.parse_args()

    metric_files = find_metric_files(args.root)
    if not metric_files:
        print("❌ No metric files found.")
        return

    df_list = []
    json_blocks = []

    for f in metric_files:
        df, js = load_metrics(f)
        df_list.append(df)
        json_blocks.append(js)

    df_all = pd.concat(df_list, ignore_index=True)

    # Run filtering
    df_all, json_blocks = filter_to_single_run(df_all, json_blocks,
                                               args.select_run,
                                               args.random_run)

    # Extract dataset name from root folder
    dataset_name = extract_dataset_name(args.root)

    # Build method colors
    methods = sorted(df_all["method"].unique())
    method_colors = build_method_colors(methods)

    print(f"🧪 Methods: {methods}")
    print(f"🧪 Runs used: {list(df_all['run'].unique())}")
    print(f"🧪 Dataset extracted: {dataset_name}")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    plot_metric_bars(df_all, outdir, method_colors, dataset_name)
    plot_correlations(df_all, outdir, method_colors, dataset_name)
    plot_population_heatmaps(json_blocks, outdir, dataset_name)

    print(f"📁 Plots saved to: {outdir}")


if __name__ == "__main__":
    main()
