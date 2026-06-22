#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib_venn import venn3
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD

from paths import PARETO_DIR, FIGURES_DIR

# ------------------------------
# Load Pareto front from CSV
# ------------------------------
PARETO_SPF_CSV = PARETO_DIR / "pareto_spf.csv"
PARETO_ESPF_CSV = PARETO_DIR / "pareto_espf.csv"
PARETO_TPF_CSV = PARETO_DIR / "pareto_tpf.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PRED_COLS = [
    "pred_metric_valid_error",
    "pred_metric_runtime",
    "pred_metric_latency",
    "pred_metric_flops",
]
TRUE_COLS = [
    "metric_valid_error",
    "metric_runtime",
    "metric_latency",
    "metric_flops",
]


def load_pareto_dfs():
    df_spf = pd.read_csv(PARETO_SPF_CSV, index_col=0)
    df_espf = pd.read_csv(PARETO_ESPF_CSV, index_col=0)
    df_tpf = pd.read_csv(PARETO_TPF_CSV, index_col=0)
    return df_spf, df_espf, df_tpf


def plot_venn(df_spf, df_espf, df_tpf):
    tpf_set = set(df_tpf.index)
    espf_set = set(df_espf.index)
    spf_set = set(df_spf.index)

    overlap_tpf_espf = tpf_set & espf_set
    recall_espf = len(overlap_tpf_espf) / len(tpf_set) if tpf_set else 0
    precision_espf = len(overlap_tpf_espf) / len(espf_set) if espf_set else 0

    overlap_tpf_spf = tpf_set & spf_set
    recall_spf = len(overlap_tpf_spf) / len(tpf_set) if tpf_set else 0
    precision_spf = len(overlap_tpf_spf) / len(spf_set) if spf_set else 0

    print("=== TPF vs ESPF ===")
    print(f"TPF size: {len(tpf_set)}")
    print(f"ESPF size: {len(espf_set)}")
    print(f"Overlap size: {len(overlap_tpf_espf)}")
    print(f"Pareto Set Recall: {recall_espf:.4f}")
    print(f"Pareto Set Precision: {precision_espf:.4f}\n")

    print("=== TPF vs SPF ===")
    print(f"TPF size: {len(tpf_set)}")
    print(f"SPF size: {len(spf_set)}")
    print(f"Overlap size: {len(overlap_tpf_spf)}")
    print(f"Pareto Set Recall: {recall_spf:.4f}")
    print(f"Pareto Set Precision: {precision_spf:.4f}\n")

    plt.figure(figsize=(5, 5))
    v = venn3(
        [tpf_set, espf_set, spf_set],
        set_labels=("TPF", "ESPF", "SPF"),
        set_colors=("#fc5437", "green", "#9da7c2"),
        alpha=0.6,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fg06_venn_diagram.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_pareto_front_2panel(df_spf, df_espf, df_tpf):
    spf = df_spf[PRED_COLS].values
    espf = df_espf[TRUE_COLS].values
    tpf = df_tpf[TRUE_COLS].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    axes[0].scatter(
        spf[:, 0], spf[:, 1],
        c="#96a2c3", marker="o", s=50, alpha=0.3,
        label="Surrogate Pareto Front",
    )
    axes[1].scatter(
        spf[:, 2], spf[:, 3],
        c="#96a2c3", marker="o", s=50, alpha=0.3,
        label="Surrogate Pareto Front",
    )

    axes[0].scatter(
        espf[:, 0], espf[:, 1],
        c="green", marker="^", s=70, alpha=0.7,
        label="Evaluated Surrogate Pareto Front",
    )
    axes[1].scatter(
        espf[:, 2], espf[:, 3],
        c="green", marker="^", s=70, alpha=0.7,
        label="Evaluated Surrogate Pareto Front",
    )

    axes[0].scatter(
        tpf[:, 0], tpf[:, 1],
        c="red", marker="*", s=80, alpha=0.5,
        label="True Pareto Front",
    )
    axes[1].scatter(
        tpf[:, 2], tpf[:, 3],
        c="red", marker="*", s=80, alpha=0.5,
        label="True Pareto Front",
    )

    axes[0].set_xlabel("metric_valid_error", fontsize=14)
    axes[0].set_ylabel("metric_runtime", fontsize=14)
    # axes[0].grid(True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_xlabel("metric_latency", fontsize=14)
    axes[1].set_ylabel("metric_flops", fontsize=14)
    # axes[1].grid(True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=14)

    fig.savefig(FIGURES_DIR / "fg07_pareto_front.png", bbox_inches="tight", dpi=300)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def plot_pairplot(df_spf, df_espf, df_tpf):
    spf = df_spf[PRED_COLS].values
    espf = df_espf[TRUE_COLS].values
    tpf = df_tpf[TRUE_COLS].values

    plot_df = pd.DataFrame({
        "Valid Error": np.concatenate([spf[:, 0], espf[:, 0], tpf[:, 0]]),
        "Runtime": np.concatenate([spf[:, 1], espf[:, 1], tpf[:, 1]]),
        "Latency": np.concatenate([spf[:, 2], espf[:, 2], tpf[:, 2]]),
        "FLOPs": np.concatenate([spf[:, 3], espf[:, 3], tpf[:, 3]]),
        "Type": ["SPF"] * len(spf) + ["ESPF"] * len(espf) + ["TPF"] * len(tpf),
    })

    g = sns.pairplot(
        plot_df,
        hue="Type",
        palette={"SPF": "#96a2c3", "ESPF": "green", "TPF": "red"},
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 60},
        markers=["o", "^", "*"],
    )
    g.fig.suptitle("Pairwise Comparison of Three Pareto Fronts", y=1.02, fontsize=16)
    plt.savefig(FIGURES_DIR / "fg08_pareto_pairplot.png", dpi=300, bbox_inches="tight")
    plt.show()


def _rank_by_distance_to_ideal(df, obj_cols):
    """
    Assign rank 1..n within a front by distance to ideal point (min-max normalized).
    """
    vals = df[obj_cols].values
    data_min = vals.min(axis=0)
    data_max = vals.max(axis=0)
    range_vals = data_max - data_min + 1e-10
    scaled = (vals - data_min) / range_vals
    dist = np.linalg.norm(scaled, axis=1)
    # rank: smallest distance = rank 1
    order = np.argsort(dist)
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return pd.Series(ranks, index=df.index)


def _normalize_ranks(rank_series):
    """
    Convert absolute ranks (1..N) to percentile ranks in [0, 1]: r_norm = (r - 1) / (N - 1).
    """
    N = len(rank_series)
    if N <= 1:
        if N == 0:
            return pd.Series(dtype=float)
        return pd.Series(0.0, index=rank_series.index)
    return (rank_series - 1) / (N - 1)


def plot_rank_preservation(df_spf, df_espf, df_tpf):
    # Absolute ranks within each front (1 = best by distance-to-ideal)
    rank_tpf = _rank_by_distance_to_ideal(df_tpf, TRUE_COLS)
    rank_espf = _rank_by_distance_to_ideal(df_espf, TRUE_COLS)
    rank_spf = _rank_by_distance_to_ideal(df_spf, PRED_COLS)

    # Convert to normalized (percentile) ranks in [0, 1]
    rank_tpf_norm = _normalize_ranks(rank_tpf)
    rank_espf_norm = _normalize_ranks(rank_espf)
    rank_spf_norm = _normalize_ranks(rank_spf)

    common_espf = set(df_tpf.index) & set(df_espf.index)
    common_spf = set(df_tpf.index) & set(df_spf.index)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    # ----- Left: TPF vs ESPF -----
    ax = axes[0]
    if common_espf:
        x_espf = np.array([rank_tpf_norm.loc[i] for i in common_espf])
        y_espf = np.array([rank_espf_norm.loc[i] for i in common_espf])
        ax.scatter(x_espf, y_espf, c="green", marker="^", s=60, alpha=0.7, label="TPF ∩ ESPF", zorder=3)
        order = np.argsort(x_espf)
        ax.plot(x_espf[order], y_espf[order], color="green", alpha=0.3, linewidth=1, zorder=1)
        # Marginal rug bars: x at bottom (y=0), y at left (x=0); small black segments
        ax.vlines(x_espf, ymin=-0.05, ymax=-0.02, colors="black", alpha=0.3, linewidth=1, zorder=2)
        ax.hlines(y_espf, xmin=-0.05, xmax=-0.02, colors="black", alpha=0.3, linewidth=1, zorder=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.65, label="y = x (perfect)")
    ax.set_xlabel("Normalized rank in TPF", fontsize=12)
    ax.set_ylabel("Normalized rank in ESPF", fontsize=12)
    # ax.set_title("Rank Preservation: TPF vs ESPF", fontsize=13, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # ----- Right: TPF vs SPF -----
    ax = axes[1]
    if common_spf:
        x_spf = np.array([rank_tpf_norm.loc[i] for i in common_spf])
        y_spf = np.array([rank_spf_norm.loc[i] for i in common_spf])
        ax.scatter(x_spf, y_spf, c="#96a2c3", marker="o", s=60, alpha=0.7, label="TPF ∩ SPF", zorder=3)
        order = np.argsort(x_spf)
        ax.plot(x_spf[order], y_spf[order], color="#96a2c3", alpha=0.3, linewidth=1, zorder=1)
        # Marginal rug bars: x at bottom (y=0), y at left (x=0); small black segments
        ax.vlines(x_spf, ymin=-0.05, ymax=-0.02, colors="black", alpha=0.3, linewidth=1, zorder=2)
        ax.hlines(y_spf, xmin=-0.05, xmax=-0.02, colors="black", alpha=0.3, linewidth=1, zorder=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.65, label="y = x (perfect)")
    ax.set_xlabel("Normalized rank in TPF", fontsize=12)
    ax.set_ylabel("Normalized rank in SPF", fontsize=12)
    # ax.set_title("Rank Preservation: TPF vs SPF", fontsize=13, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # fig.suptitle(
    #     "Rank Comparison (normalized ranks): Prediction vs True Pareto Front (on diagonal = rank preserved)",
    #     fontsize=14, fontweight="bold", y=1.02,
    # )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fg09_rank_preservation.png", dpi=300, bbox_inches="tight")
    plt.show()



def main():
    df_spf, df_espf, df_tpf = load_pareto_dfs()
    print(f"Loaded SPF: {len(df_spf)}, ESPF: {len(df_espf)}, TPF: {len(df_tpf)} solutions\n")

    plot_venn(df_spf, df_espf, df_tpf)
    plot_pareto_front_2panel(df_spf, df_espf, df_tpf)
    plot_pairplot(df_spf, df_espf, df_tpf)
    plot_rank_preservation(df_spf, df_espf, df_tpf)


if __name__ == "__main__":
    main()
