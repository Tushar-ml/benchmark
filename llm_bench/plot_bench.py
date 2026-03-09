#!/usr/bin/env python3
"""
Plot utility for LLM benchmark results.

Reads CSV files produced by bench_serving.py or locust load_test.py and
generates standard performance plots.  Multiple CSV files can be overlaid
for A/B comparison (e.g. vLLM vs SGLang, different cache configs).

Usage:
    # Single file
    python plot_bench.py --csv bench-serving-results.csv

    # Compare two runs
    python plot_bench.py --csv run-a.csv --csv run-b.csv --labels "vLLM" "SGLang"

    # Filter to specific output-token count
    python plot_bench.py --csv results.csv --filter-col output_tokens_avg --filter-val 64

    # Custom output directory
    python plot_bench.py --csv results.csv -o ./plots
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ── Colour palette (same as dynamo/benchmarks/utils/plot.py) ─────────────
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]


# ── CSV normalisation ────────────────────────────────────────────────────
# Map the two CSV schemas to a common column set so every plotting function
# can work with either source.

_LOCUST_RENAMES = {
    "Concurrency":                  "concurrency",
    "Total Latency":                "avg_latency_ms",
    "Time To First Token":          "avg_ttft_ms",
    "Latency Per Token":            "avg_tpot_ms",
    "Num Tokens":                   "output_tokens_avg",
    "Num Requests":                 "num_requests",
    "Qps":                          "throughput_req_per_s",
    "Prompt Tokens":                "prompt_tokens_avg",
    "Prompt Cache Max Len":         "prompt_cache_max_len",
    "Model":                        "model",
    "Generation Tokens":            "generation_tokens",
    "P50 Total Latency":            "p50_latency_ms",
    "P90 Total Latency":            "p90_latency_ms",
    "P95 Total Latency":            "p95_latency_ms",
    "P99 Total Latency":            "p99_latency_ms",
    "P50 Time To First Token":      "p50_ttft_ms",
    "P90 Time To First Token":      "p90_ttft_ms",
    "P95 Time To First Token":      "p95_ttft_ms",
    "P99 Time To First Token":      "p99_ttft_ms",
}

_BENCH_SERVING_RENAMES = {}  # already uses the canonical names


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Concurrency" in df.columns:
        df = df.rename(columns=_LOCUST_RENAMES)
    else:
        df = df.rename(columns=_BENCH_SERVING_RENAMES)

    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return df


def filter_df(df: pd.DataFrame, col: Optional[str], val) -> pd.DataFrame:
    if col is None or col not in df.columns:
        return df
    try:
        val = type(df[col].iloc[0])(val)
    except Exception:
        pass
    return df[df[col] == val].copy()


# ── Reusable line-plot helper ────────────────────────────────────────────

def _line_plot(
    ax,
    series: list,
    xlabel: str,
    ylabel: str,
    title: str,
    log_x: bool = True,
    log_y: bool = False,
):
    for i, (label, xs, ys) in enumerate(series):
        ax.plot(
            xs, ys,
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            linewidth=2, markersize=7, label=label,
        )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=9)


def _percentile_band(ax, xs, p50, p90, color, alpha=0.15):
    """Shade between p50 and p90 if both are available."""
    if p50 is not None and p90 is not None and len(p50) == len(xs):
        ax.fill_between(xs, p50, p90, color=color, alpha=alpha)


# ── Individual plot generators ───────────────────────────────────────────
# Each takes a list of (label, DataFrame) and an output directory.

def plot_latency_vs_concurrency(datasets, out_dir: Path, stat="avg"):
    col = f"{stat}_latency_ms" if stat == "avg" else f"p{stat}_latency_ms" if stat.isdigit() else f"{stat}_latency_ms"
    fig, ax = plt.subplots(figsize=(10, 6))
    series = []
    for i, (label, df) in enumerate(datasets):
        if col not in df.columns or "concurrency" not in df.columns:
            continue
        grouped = df.groupby("concurrency")[col].mean().sort_index()
        series.append((label, grouped.index.tolist(), grouped.values.tolist()))

        for pcol_suffix in ["p50_latency_ms", "p90_latency_ms"]:
            if pcol_suffix in df.columns:
                pass  # band drawn below
        if "p50_latency_ms" in df.columns and "p90_latency_ms" in df.columns:
            g50 = df.groupby("concurrency")["p50_latency_ms"].mean().sort_index()
            g90 = df.groupby("concurrency")["p90_latency_ms"].mean().sort_index()
            _percentile_band(ax, g50.index.tolist(), g50.values, g90.values, COLORS[i % len(COLORS)])

    _line_plot(ax, series, "Concurrency", "Latency (ms)",
               f"{'Avg' if stat == 'avg' else 'P' + stat} End-to-End Latency vs Concurrency")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stat}_latency_vs_concurrency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stat}_latency_vs_concurrency.png")


def plot_throughput_vs_concurrency(datasets, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    series = []
    for label, df in datasets:
        if "throughput_req_per_s" not in df.columns:
            continue
        grouped = df.groupby("concurrency")["throughput_req_per_s"].mean().sort_index()
        series.append((label, grouped.index.tolist(), grouped.values.tolist()))
    _line_plot(ax, series, "Concurrency", "Throughput (req/s)",
               "Request Throughput vs Concurrency")
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_vs_concurrency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved throughput_vs_concurrency.png")


def plot_token_throughput_vs_concurrency(datasets, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    series = []
    for label, df in datasets:
        if "throughput_output_tok_per_s" not in df.columns:
            continue
        grouped = df.groupby("concurrency")["throughput_output_tok_per_s"].mean().sort_index()
        series.append((label, grouped.index.tolist(), grouped.values.tolist()))
    if not series:
        plt.close(fig)
        return
    _line_plot(ax, series, "Concurrency", "Output Tokens/s",
               "Output Token Throughput vs Concurrency")
    fig.tight_layout()
    fig.savefig(out_dir / "token_throughput_vs_concurrency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved token_throughput_vs_concurrency.png")


def plot_ttft_vs_concurrency(datasets, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    series = []
    for i, (label, df) in enumerate(datasets):
        if "avg_ttft_ms" not in df.columns:
            continue
        grouped = df.groupby("concurrency")["avg_ttft_ms"].mean().sort_index()
        if grouped.isna().all() or (grouped == 0).all():
            continue
        series.append((label, grouped.index.tolist(), grouped.values.tolist()))

        if "p50_ttft_ms" in df.columns and "p90_ttft_ms" in df.columns:
            g50 = df.groupby("concurrency")["p50_ttft_ms"].mean().sort_index()
            g90 = df.groupby("concurrency")["p90_ttft_ms"].mean().sort_index()
            _percentile_band(ax, g50.index.tolist(), g50.values, g90.values, COLORS[i % len(COLORS)])

    if not series:
        plt.close(fig)
        return
    _line_plot(ax, series, "Concurrency", "TTFT (ms)",
               "Time to First Token vs Concurrency")
    fig.tight_layout()
    fig.savefig(out_dir / "ttft_vs_concurrency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved ttft_vs_concurrency.png")


def plot_tpot_vs_concurrency(datasets, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    series = []
    for label, df in datasets:
        if "avg_tpot_ms" not in df.columns:
            continue
        grouped = df.groupby("concurrency")["avg_tpot_ms"].mean().sort_index()
        if grouped.isna().all() or (grouped == 0).all():
            continue
        series.append((label, grouped.index.tolist(), grouped.values.tolist()))
    if not series:
        plt.close(fig)
        return
    _line_plot(ax, series, "Concurrency", "TPOT (ms)",
               "Time per Output Token vs Concurrency")
    fig.tight_layout()
    fig.savefig(out_dir / "tpot_vs_concurrency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved tpot_vs_concurrency.png")


def plot_latency_vs_throughput(datasets, out_dir: Path):
    """Pareto-style: throughput on X, latency on Y — ideal is bottom-right."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, df) in enumerate(datasets):
        if "throughput_req_per_s" not in df.columns or "avg_latency_ms" not in df.columns:
            continue
        grouped = df.groupby("concurrency").agg(
            tp=("throughput_req_per_s", "mean"),
            lat=("avg_latency_ms", "mean"),
        ).sort_values("tp")
        ax.plot(
            grouped["tp"], grouped["lat"],
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            linewidth=2, markersize=7, label=label,
        )
        for conc, row in grouped.iterrows():
            ax.annotate(
                str(conc), (row["tp"], row["lat"]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=8, fontweight="bold",
            )

    ax.set_xlabel("Throughput (req/s)", fontsize=11)
    ax.set_ylabel("Avg Latency (ms)", fontsize=11)
    ax.set_title("Latency vs Throughput (numbers = concurrency)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_throughput.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved latency_vs_throughput.png")


def plot_percentile_bars(datasets, out_dir: Path):
    """Bar chart of latency percentiles at the highest concurrency tested."""
    pct_cols = ["median_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    pct_labels = ["P50", "P90", "P95", "P99"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.8 / max(len(datasets), 1)
    x = np.arange(len(pct_labels))
    plotted = False

    for i, (label, df) in enumerate(datasets):
        available = [c for c in pct_cols if c in df.columns]
        if not available:
            continue
        max_conc = df["concurrency"].max()
        row = df[df["concurrency"] == max_conc].iloc[-1]
        vals = [float(row.get(c, 0)) for c in pct_cols]
        ax.bar(x + i * bar_width, vals, bar_width,
               label=f"{label} (c={int(max_conc)})", color=COLORS[i % len(COLORS)])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(pct_labels)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Latency Percentiles at Max Concurrency", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_percentiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved latency_percentiles.png")


# ── Orchestrator ─────────────────────────────────────────────────────────

def generate_all_plots(
    csv_paths: list[str],
    labels: Optional[list[str]] = None,
    out_dir: str = "plots",
    filter_col: Optional[str] = None,
    filter_val=None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = [Path(p).stem for p in csv_paths]
    while len(labels) < len(csv_paths):
        labels.append(Path(csv_paths[len(labels)]).stem)

    datasets = []
    for path, label in zip(csv_paths, labels):
        df = load_csv(path)
        df = filter_df(df, filter_col, filter_val)
        if df.empty:
            print(f"WARNING: {path} has no data after filtering, skipping")
            continue
        datasets.append((label, df))
        print(f"Loaded {len(df)} rows from {path} as '{label}'")

    if not datasets:
        print("No data to plot!")
        return

    print(f"\nGenerating plots in {out}/")
    plot_latency_vs_concurrency(datasets, out, stat="avg")
    plot_latency_vs_concurrency(datasets, out, stat="90")
    plot_throughput_vs_concurrency(datasets, out)
    plot_token_throughput_vs_concurrency(datasets, out)
    plot_ttft_vs_concurrency(datasets, out)
    plot_tpot_vs_concurrency(datasets, out)
    plot_latency_vs_throughput(datasets, out)
    plot_percentile_bars(datasets, out)
    print(f"\nDone — all plots saved to {out}/")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot LLM benchmark results from CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", action="append", required=True,
        help="Path to a results CSV (repeatable for comparison)",
    )
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="Legend labels for each CSV (order must match --csv)",
    )
    parser.add_argument(
        "-o", "--output-dir", default="plots",
        help="Directory to save plots (default: ./plots)",
    )
    parser.add_argument(
        "--filter-col", default=None,
        help="Column name to filter on before plotting",
    )
    parser.add_argument(
        "--filter-val", default=None,
        help="Value to keep in --filter-col",
    )
    args = parser.parse_args()
    generate_all_plots(args.csv, args.labels, args.output_dir, args.filter_col, args.filter_val)


if __name__ == "__main__":
    main()
