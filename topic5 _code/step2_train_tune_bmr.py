#!/usr/bin/env python
# coding: utf-8
import math
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    KFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import xgboost as xgb
import lightgbm as lgb

from paths import NASBENCH_CSV, SURROGATE_DIR

RANDOM_STATE = 8
np.random.seed(RANDOM_STATE)
sns.set(style="whitegrid")

SURROGATE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Path to NASBench201 dataset
# ---------------------------------------------------------------------------
CSV_PATH = NASBENCH_CSV


# ==============================
# Step 2.1: Data loading & preprocessing
# ==============================

df = pd.read_csv(CSV_PATH)
print("Loaded", len(df), "rows")
print(df.head())

hp_cols = ["hp_x0", "hp_x1", "hp_x2", "hp_x3", "hp_x4", "hp_x5"]

target_cols = [
    "metric_latency",
    "metric_flops",
    "metric_runtime",
    "metric_valid_error",
]

categorical_features = hp_cols
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        )
    ],
    remainder="drop",
)


def build_pipeline(reg, use_preprocessing=True):
    if use_preprocessing:
        return Pipeline([("pre", preprocessor), ("reg", reg)])
    else:
        return Pipeline([("reg", reg)])


# ==============================
# Step 2.2: Baseline surrogate training
# ==============================

learners = {
    "RandomForest": build_pipeline(
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        use_preprocessing=True,
    ),
    "GradientBoosting": build_pipeline(
        GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
        use_preprocessing=True,
    ),
    "XGBoost": build_pipeline(
        xgb.XGBRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=4,
            enable_categorical=True,
        ),
        use_preprocessing=False,
    ),
    "LightGBM": build_pipeline(
        lgb.LGBMRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        use_preprocessing=False,
    ),
}

print("Models:", list(learners.keys()))

all_predictions = []
all_test_results = {}

for target_col in target_cols:

    print("\n" + "=" * 90)
    print(f"Training surrogate models for target: {target_col}")
    print("=" * 90)

    X = df[hp_cols].astype("category")
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = {}

    for name, model in learners.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        for i in range(len(y_test)):
            all_predictions.append({
                "sample_id": X_test.index[i],
                "target": target_col,
                "model": name,
                "y_true": y_test[i],
                "y_pred": y_pred[i],
                "residual": y_pred[i] - y_test[i],
                "rel_error_pct": (y_pred[i] - y_test[i]) / y_test[i] * 100,
            })

        results[name] = {
            "rmse": math.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        print(
            f"{name}: "
            f"RMSE={results[name]['rmse']:.4f}, "
            f"MAE={results[name]['mae']:.4f}, "
            f"R2={results[name]['r2']:.4f}"
        )

    all_test_results[target_col] = (
        pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    )

pred_df = pd.DataFrame(all_predictions)
pred_df.to_csv(SURROGATE_DIR / "trees_surr_preds.csv", index=False)

print("\nSaved prediction table:")
print(pred_df.shape)
print(pred_df.head())

# ---- CV benchmark ----
cv_rows = []

for target_col in target_cols:

    X = df[hp_cols].astype("category")
    y = df[target_col].values

    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    for model_name, model in learners.items():

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
        )

        for i in range(len(scores["test_rmse"])):
            cv_rows.append({
                "target": target_col,
                "model": model_name,
                "fold": i,
                "rmse": -scores["test_rmse"][i],
                "mae": -scores["test_mae"][i],
                "r2": scores["test_r2"][i],
            })

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(SURROGATE_DIR / "trees_surrogate_cv.csv", index=False)


# ==============================
# Step 2.3: Hyperparameter tuning
# ==============================

target_model_map = {
    "metric_flops": "GradientBoosting",
    "metric_latency": "LightGBM",
    "metric_runtime": "LightGBM",
    "metric_valid_error": "XGBoost",
}

param_spaces = {
    "GradientBoosting": {
        "reg__n_estimators": [200, 400, 600, 800],
        "reg__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "reg__max_depth": [3, 5, 7],
        "reg__subsample": [0.8, 1.0],
        "reg__min_samples_split": [2, 5, 10],
        "reg__min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "reg__n_estimators": [200, 300, 400, 500],
        "reg__learning_rate": [0.01, 0.05, 0.1],
        "reg__max_depth": [3, 5, 7],
        "reg__subsample": [0.6, 0.8, 1.0],
        "reg__colsample_bytree": [0.8, 1.0],
    },
    "LightGBM": {
        "reg__n_estimators": [200, 300, 400, 500],
        "reg__learning_rate": [0.01, 0.05, 0.1],
        "reg__max_depth": [3, 5, 7],
        "reg__num_leaves": [31, 50, 70, 100, 150],
        "reg__subsample": [0.8, 1.0],
        "reg__colsample_bytree": [0.8, 1.0],
    },
}

all_results = []
tuned_predictions = []

for target_col, best_model_name in target_model_map.items():

    print("\n" + "=" * 80)
    print(f"Target: {target_col} | Model: {best_model_name}")
    print("=" * 80)

    X = df[hp_cols].astype("category")
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    base_model = learners[best_model_name]

    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)

    all_results.append({
        "target": target_col,
        "model": best_model_name,
        "type": "Base",
        "rmse": root_mean_squared_error(y_test, y_pred_base),
        "mae": mean_absolute_error(y_test, y_pred_base),
        "r2": r2_score(y_test, y_pred_base),
    })

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_spaces[best_model_name],
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    y_pred_tuned = search.predict(X_test)

    for i in range(len(y_test)):
        tuned_predictions.append({
            "sample_id": X_test.index[i],
            "target": target_col,
            "model_tuned": best_model_name,
            "y_true": y_test[i],
            "y_pred": y_pred_tuned[i],
            "residual": y_pred_tuned[i] - y_test[i],
            "rel_error_pct": (y_pred_tuned[i] - y_test[i]) / y_test[i] * 100,
        })

    all_results.append({
        "target": target_col,
        "model": best_model_name,
        "type": "Tuned",
        "rmse": root_mean_squared_error(y_test, y_pred_tuned),
        "mae": mean_absolute_error(y_test, y_pred_tuned),
        "r2": r2_score(y_test, y_pred_tuned),
        "best_params": search.best_params_,
    })

results_df = pd.DataFrame(all_results)
pd.set_option("display.max_colwidth", None)

print("\nBase vs Tuned summary:")
print(results_df)

tuned_pred_df = pd.DataFrame(tuned_predictions)
tuned_pred_df.to_csv(SURROGATE_DIR / "surrogate_tuned_preds.csv", index=False)

print("\nSaved tuned instance-level predictions:")
print(tuned_pred_df.shape)
print(tuned_pred_df.head())

results_df.to_csv(SURROGATE_DIR / "surrogate_tuning_params.csv", index=False)

for target in results_df["target"].unique():
    for model in results_df["model"].unique():
        base_row = results_df[
            (results_df["target"] == target)
            & (results_df["model"] == model)
            & (results_df["type"] == "Base")
        ]
        tuned_row = results_df[
            (results_df["target"] == target)
            & (results_df["model"] == model)
            & (results_df["type"] == "Tuned")
        ]

        if not base_row.empty and not tuned_row.empty:
            base_vals = base_row.iloc[0]
            tuned_vals = tuned_row.iloc[0]

            rmse_delta = (tuned_vals["rmse"] - base_vals["rmse"]) / base_vals["rmse"] * 100
            mae_delta = (tuned_vals["mae"] - base_vals["mae"]) / base_vals["mae"] * 100
            r2_delta = (tuned_vals["r2"] - base_vals["r2"]) / base_vals["r2"] * 100

            print(f"Target: {target}, Model: {model}")
            print(f"  RMSE change: {rmse_delta:.2f}%")
            print(f"  MAE  change: {mae_delta:.2f}%")
            print(f"  R2   change: {r2_delta:.2f}%")
            print("-" * 50)
