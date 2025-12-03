#!/usr/bin/env python3
"""
Flow/CyTOF metrics visualization (OmniBenchmark compatible).

Implements:
    • Method comparison barplots
    • Scatter plots
    • Per-population heatmaps
    • Consistent method colors
    • Dataset auto-detection
    • Run filtering (first run only)
    • Loading all metric files using only ONE input file
"""

import sys
import gzip
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")


# =================================================================
# UTILS — extract dataset name from file path
# =================================================================
def extract_dataset_name_from_file(path: Path):
    """
    Input file name is e.g. dataset.flow_metrics.json.gz
    We extract "dataset"
    """
    stem = path.name
    if stem.endswith(".flow_metrics.json.gz"):
        return stem.replace(".flow_metrics.json.gz", "")
    return stem


# =================================================================
# Extract method from path folders: analysis/<method>/default
# =================================================================
def extract_method(path: Path):
    parts = path.parts
    if "analysis" in parts:
        idx = parts.index("analysis")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


# =================================================================
# Given ONE input file, locate ALL metric files for same dataset
# =================================================================
def discover_metric_files_from_single_file(single_file: Path):
    """
    single_file:
        .../analysis/<method>/default/metrics/flow_metrics/<hash>/dataset.flow_metrics.json.gz

    We find:
        .../analysis/*/default/metrics/flow_metrics/*/*.flow_metrics.json.gz
    """

    parts = single_file.parts
    idx = parts.index("analysis")
    analysis_root = Path(*parts[:idx+1])  # up to /analysis

    return list(analysis_root.rglob("*.flow_metrics.json.gz"))


# =================================================================
# Load metrics JSON → flat dataframe + full JSON block
# =================================================================
def load_metrics(path: Path):
    with gzip.open(path, "rt") as f:
        js = json.load(f)

    method = extract_method(path)

    rows = []
    for run_label, block in js["results"].items():
        row = {
            "method": method,
            "run": run_label,
            "file": str(path)
        }
        for k, v in block.items():
            if not isinstance(v, dict):  # skip per_population
                row[k] = v
        rows.append(row)

    return pd.DataFrame(rows), js


# =================================================================
# Color map: deterministic random colors
# =================================================================
def build_method_colors(methods):
    random.seed(42)
    return {
        m: "#%06x" % random.randint(0, 0xFFFFFF)
        for m in methods
    }


# =================================================================
# Save plots
# =================================================================
def save_plot(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =================================================================
# BARPLOTS
# =================================================================
def plot_metric_bars(df, outdir, method_colors, dataset_name):
    numeric_cols = [
        c for c in df.columns
        if c not in ("run", "method", "file")
        and np.issubdtype(df[c].dtype, np.number)
    ]

    for metric in numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 4))

        sns.barplot(
            data=df,
            x="method",
            y=metric,
            color="lightgray",
            ax=ax
        )

        # manually color patches
        for patch, (_, row) in zip(ax.patches, df.iterrows()):
            patch.set_facecolor(method_colors[row["method"]])

        # add text labels
        for patch in ax.patches:
            h = patch.get_height()
            if not np.isnan(h):
                ax.text(
                    patch.get_x() + patch.get_width()/2,
                    h,
                    f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8
                )

        ax.set_title(f"{metric} comparison (dataset: {dataset_name})")

        handles = [
            plt.matplotlib.patches.Patch(color=method_colors[m], label=m)
            for m in method_colors
        ]
        ax.legend(handles=handles, title="Method", bbox_to_anchor=(1.02, 0.5), loc="center left")

        save_plot(fig, outdir, f"metric_{metric}")


# =================================================================
# SCATTER PLOTS
# =================================================================
def plot_correlations(df, outdir, method_colors, dataset_name):
    pairs = [
        ("scalability_seconds_per_item", "accuracy"),
        ("runtime_seconds", "accuracy"),
        ("mcc", "accuracy"),
        ("aucroc", "accuracy"),
    ]

    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.scatterplot(
            data=df,
            x=x, y=y,
            hue="method",
            palette=method_colors,
            s=100,
            ax=ax
        )

        ax.set_title(f"{y} vs {x} (dataset: {dataset_name})")
        ax.legend(title="Method", bbox_to_anchor=(1.02, 0.5), loc="center left")

        save_plot(fig, outdir, f"scatter_{y}_vs_{x}")


# =================================================================
# PER-POPULATION HEATMAPS
# =================================================================
def plot_population_heatmaps(json_blocks, outdir, dataset_name):
    for js in json_blocks:
        for run_label, block in js["results"].items():
            perpop = block.get("per_population")
            if perpop is None:
                continue

            df = pd.DataFrame(perpop).T

            fig, ax = plt.subplots(figsize=(9, 5))
            sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", ax=ax)

            method = js.get("name", "unknown")
            ax.set_title(f"Per-population (method={method}, run={run_label}, dataset={dataset_name})")

            save_plot(fig, outdir, f"per_population_{method}_run_{run_label}")


# =================================================================
# MAIN — OmniBenchmark compatible
# =================================================================
def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: visualize_flow_metrics.py <metrics.json.gz> <outdir>")

    input_file = Path(sys.argv[1])
    outdir = Path(sys.argv[2])

    # discover dataset name
    dataset_name = extract_dataset_name_from_file(input_file)

    # find ALL metrics
    metric_files = discover_metric_files_from_single_file(input_file)
    if not metric_files:
        raise SystemExit("❌ No metric files found.")

    # load ALL methods
    df_list = []
    js_blocks = []
    for f in metric_files:
        df, js = load_metrics(f)
        df_list.append(df)
        js_blocks.append(js)

    df_all = pd.concat(df_list, ignore_index=True)

    # choose ONE run per method to simplify
    df_all = df_all[df_all["run"] == df_all["run"].unique()[0]]

    # color palette
    methods = sorted(df_all["method"].unique())
    method_colors = build_method_colors(methods)

    outdir.mkdir(parents=True, exist_ok=True)

    plot_metric_bars(df_all, outdir, method_colors, dataset_name)
    plot_correlations(df_all, outdir, method_colors, dataset_name)
    plot_population_heatmaps(js_blocks, outdir, dataset_name)

    print(f"📁 Plots saved to {outdir}")


if __name__ == "__main__":
    main()
