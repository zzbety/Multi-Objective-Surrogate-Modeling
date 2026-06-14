#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
import numpy as np

from paths import SURROGATE_DIR

results_df = pd.read_csv(SURROGATE_DIR / "surrogate_tuning_params.csv")
results_df.head()


# ---- Tuned config ----
tuned_config = {}

for _, row in results_df.iterrows():
    if row["type"] == "Tuned":
        tuned_config[row["target"]] = {
            "model_name": row["model"],
            "params": eval(row["best_params"]) if isinstance(row["best_params"], str) else row["best_params"]
        }

tuned_config



# ---- Define learners ----
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

RANDOM_STATE = 8

hp_cols = ["hp_x0", "hp_x1", "hp_x2", "hp_x3", "hp_x4", "hp_x5"]

categorical_features = hp_cols
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_features)
    ],
    remainder="drop",
)

def build_pipeline(reg, use_preprocessing=True):
    if use_preprocessing:
        return Pipeline([("pre", preprocessor), ("reg", reg)])
    else:
        return Pipeline([("reg", reg)])

learners = {
    "RandomForest": build_pipeline(
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        use_preprocessing=True
    ),
    "GradientBoosting": build_pipeline(
        GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
        use_preprocessing=True
    ),
    "XGBoost": build_pipeline(
        xgb.XGBRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=4,
            enable_categorical=True
        ),
        use_preprocessing=False
    ),
    "LightGBM": build_pipeline(
        lgb.LGBMRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        ),
        use_preprocessing=False
    )
}


# ---- Define surrogate trainer ----
# for sequentially training surrogate during moo phase


from sklearn.base import clone

class SurrogateTrainer:
    def __init__(self, learners, tuned_config, hp_cols):
        self.learners = learners
        self.tuned_config = tuned_config
        self.hp_cols = hp_cols
        self.models = {}

    def fit(self, df):
        X = df[self.hp_cols].astype("category")

        for target, cfg in self.tuned_config.items():
            y = df[target].values

            base_model = self.learners[cfg["model_name"]]
            model = clone(base_model)

            if cfg["params"] is not None:
                model.set_params(**cfg["params"])

            model.fit(X, y)
            self.models[target] = model

    def predict(self, df):
        X = df[self.hp_cols].astype("category")
        preds = {}

        for target, model in self.models.items():
            preds[f"pred_{target}"] = model.predict(X)

        return pd.DataFrame(preds, index=df.index)



