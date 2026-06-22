#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error, r2_score)

from paths import SURROGATE_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# # -------------------------------------------------------
# # 00) Load predictions of base models
# # -------------------------------------------------------
# pred_df = pd.read_csv("trees_surr_preds.csv")

# # len(pred_df)/16 = 3125
# # residual = pred - truth
# # rel_error_pct = (pred-truth) / truth * 100


# # Calculate performance and select the best model for each target

# metrics_records = []

# targets = pred_df["target"].unique()
# models = pred_df["model"].unique()

# for target in targets:
#     for model in models:
#         df_sub = pred_df[
#             (pred_df["target"] == target) &
#             (pred_df["model"] == model)
#         ]

#         metrics_records.append({
#             "target": target,
#             "model": model,
#             "rmse": root_mean_squared_error(df_sub["y_true"], df_sub["y_pred"]),
#             "mae": mean_absolute_error(df_sub["y_true"], df_sub["y_pred"]),
#             "r2": r2_score(df_sub["y_true"], df_sub["y_pred"])
#         })

# metrics_df = pd.DataFrame(metrics_records)
# metrics_df.head()


# best_models_df = (
#     metrics_df
#     .sort_values(by=["rmse", "mae", "r2"], ascending=[True, True, False])
#     .groupby("target", as_index=False)
#     .first()
# )

# best_models_df


# final_pred_records = []

# for _, row in best_models_df.iterrows():
#     target = row["target"]
#     model = row["model"]

#     df_sub = pred_df[
#         (pred_df["target"] == target) &
#         (pred_df["model"] == model)
#     ]

#     final_pred_records.append(df_sub)

# final_pred_df = pd.concat(final_pred_records, ignore_index=True)
# final_pred_df.head()


# -------------------------------------------------------
# 1) Load Predictions for Tuned Model
# -------------------------------------------------------

final_pred_df = pd.read_csv(SURROGATE_DIR / "surrogate_tuned_preds.csv")
final_pred_df = final_pred_df.rename(columns={"model_tuned": "model"})

final_pred_df.head()
final_pred_df["sample_id"].nunique()


# -------------------------------------------------------
# 2) Visualization
# -------------------------------------------------------

# targets = final_pred_df["target"].unique()
targets = ["metric_valid_error", "metric_latency", "metric_flops", "metric_runtime"]
n_targets = len(targets)

# ---- True vs Predicted ----
fig, axes = plt.subplots(
    1, n_targets,
    figsize=(4 * n_targets, 4),
    sharex=False,
    sharey=False
)

for i, (ax, target) in enumerate(zip(axes, targets)):
    df_t = final_pred_df[final_pred_df["target"] == target]

    ax.scatter(
        df_t["y_true"],
        df_t["y_pred"],
        alpha=0.3,
        s=18
    )

    lim_min = np.min([df_t["y_true"].min(), df_t["y_pred"].min()])
    lim_max = np.max([df_t["y_true"].max(), df_t["y_pred"].max()])

    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle="--",
        linewidth=1,
        color="gray"
    )

    model_name = df_t["model"].iloc[0]
    # ax.set_title(f"{target}")
    ax.set_title(f"{target} | {model_name}")
    ax.set_xlabel("True value")

    if i == 0:
        ax.set_ylabel("Predicted value")
    else:
        ax.set_ylabel("")

# fig.suptitle("True vs Predicted Values", fontsize=16)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fg01_true_vs_predicted.png", dpi=300, bbox_inches="tight")

plt.show()



# ---- Relative error vs True ----
fig, axes = plt.subplots(
    nrows=1,
    ncols=n_targets,
    figsize=(4 * n_targets, 4),
    sharex=False,
    sharey=True
)

for ax, target in zip(axes, targets):
    df_t = final_pred_df[final_pred_df["target"] == target]

    ax.scatter(
        df_t["y_true"],
        df_t["rel_error_pct"],
        alpha=0.3
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.tick_params(axis="x", labelrotation=15)

    # model_name = df_t["model"].iloc[0]
    # ax.set_title(f"{target} | {model_name}")
    ax.set_title(target)
    ax.set_xlabel("True value")

axes[0].set_ylabel("Relative error (%)")
# fig.suptitle("Residual vs True values", fontsize=16)
plt.tight_layout()

plt.show()





# ---- Relative error distribution (histogram) ----
fig, axes = plt.subplots(
    1, n_targets,
    figsize=(4 * n_targets, 4),
    sharex=False,
    sharey=False
)

for i, (ax, target) in enumerate(zip(axes, targets)):
    df_t = final_pred_df[final_pred_df["target"] == target]

    # Histogram / Relative Error Distribution
    sns.histplot(df_t["rel_error_pct"], bins=40, ax=ax, alpha=0.5)

    ax.set_title(target)
    ax.set_xlabel("Relative error (%)")
    if i == 0:
        ax.set_ylabel("Count")
    else:
        ax.set_ylabel("")

    ax.tick_params(axis="x", labelrotation=0)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fg02_relative_error_distribution.png", dpi=300, bbox_inches="tight")

plt.show()



# ---- Error Threshold Analysis ----
thresholds = [1, 3, 5, 10, 20]

threshold_records = []

for target in final_pred_df["target"].unique():
    df_t = final_pred_df[final_pred_df["target"] == target]
    abs_rel_err = df_t["rel_error_pct"].abs()

    for th in thresholds:
        threshold_records.append({
            "target": target,
            "model": df_t["model"].iloc[0],
            "threshold_pct": th,
            "ratio_within_threshold": (abs_rel_err <= th).mean()
        })

final_threshold_df = pd.DataFrame(threshold_records)
# final_threshold_df.head(3)
# final_threshold_df


# ---- Relative error density + threshold curve (fg03_fg04) ----
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

targets = final_pred_df["target"].unique()
linestyles = ["-", "--", "-.", ":"]
palette = sns.color_palette("Set2", n_colors=len(targets))
color_map = dict(zip(targets, palette))
linestyle_map = dict(zip(targets, linestyles))

ax = axes[0]

for color, linestyle, target in zip(palette, linestyles, targets):
    df_t = final_pred_df[final_pred_df["target"] == target]

    sns.kdeplot(
        data=df_t,
        x="rel_error_pct",
        bw_method="silverman",
        fill=False,
        color=color,
        linestyle=linestyle,
        linewidth=2,
        label=target,
        clip=(-20, 20),
        ax=ax
    )

ax.axvline(0, linestyle="--", color="gray", linewidth=1)
ax.set_xlabel("Relative error (%)")
ax.set_ylabel("Density")

# ---- Error threshold curve ----
ax = axes[1]

for target in targets:
    df_t = final_threshold_df[final_threshold_df["target"] == target]

    ax.plot(
        df_t["threshold_pct"],
        df_t["ratio_within_threshold"],
        marker="o",
        markersize=5,
        color=color_map[target],
        label=target
    )

ax.set_xlabel("Relative error threshold (%)")
ax.set_ylabel("Proportion within threshold")
ax.set_xticks(thresholds)
ax.invert_xaxis()

# ---- Shared legend (outside) ----
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    # title="Target",
    frameon=False,
    loc="lower center",
    ncol=len(labels),          # 横向排开
    bbox_to_anchor=(0.5, -0.05),
    fontsize=9
)

# ---- Layout & save ----
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(
    FIGURES_DIR / "fg03_fg04_relative_error_analysis.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()



# ---- Benchmark ----
# -------------------------------------------------------
# Load CV results
# -------------------------------------------------------

cv_df = pd.read_csv(SURROGATE_DIR / "trees_surrogate_cv.csv")

# standardized
cv_df_std = cv_df.copy()

cv_df_std["rmse_std"] = (
    cv_df_std
    .groupby("target")["rmse"]
    .transform(lambda x: x / x.max())
)

cv_df_std["mae_std"] = (
    cv_df_std
    .groupby("target")["mae"]
    .transform(lambda x: x / x.max())
)

target_order = ["metric_valid_error", "metric_latency", "metric_flops", "metric_runtime"]

cv_df_std["target"] = pd.Categorical(
    cv_df_std["target"],
    categories=target_order,
    ordered=True
)

models = cv_df_std["model"].unique()
model_palette = dict(
    zip(
        models,
        sns.color_palette("Set2", n_colors=len(models))
    )
)

fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)

#  RMSE
for i, target in enumerate(target_order):
    ax = axes[0, i]
    sns.boxplot(
        data=cv_df_std[cv_df_std["target"] == target],
        x="model",
        y="rmse_std",
        palette=model_palette,
        hue="model", 
        showfliers=False,
        ax=ax
    )
    ax.set_title(target)
    ax.set_xlabel("")  
    ax.set_ylabel("") 
    # ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30, labelbottom=False) 

#  MAE
for i, target in enumerate(target_order):
    ax = axes[1, i]
    sns.boxplot(
        data=cv_df_std[cv_df_std["target"] == target],
        x="model",
        y="mae_std",
        palette=model_palette,
        hue="model", 
        showfliers=False,
        ax=ax
    )
    ax.set_title("") 
    ax.set_xlabel("") 
    ax.set_ylabel("") 
    # ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30) 

for ax in axes[1, :]:
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30)

axes[0, 0].set_ylabel("Standardized RMSE")
axes[1, 0].set_ylabel("Standardized MAE")

fig.savefig(FIGURES_DIR / "fg05_benchmark(max-normalization).png", bbox_inches="tight", dpi=300)

plt.tight_layout()
plt.show()

