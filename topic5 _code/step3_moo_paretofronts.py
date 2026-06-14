#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from step2_surrogate_pipeline import (
    SurrogateTrainer,
    learners,
    tuned_config,
    hp_cols,
)
from paths import NASBENCH_CSV, PARETO_DIR

# ------------------------------
# Config (match step3_moo.py)
# ------------------------------
CSV_PATH = NASBENCH_CSV
TEST_SIZE = 0.2
RANDOM_STATE = 8

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


def main():
    # ------------------------------
    # Load data & split (same as step3)
    # ------------------------------
    df = pd.read_csv(CSV_PATH)
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ------------------------------
    # Surrogate predict on test set
    # ------------------------------
    trainer = SurrogateTrainer(
        learners=learners,
        tuned_config=tuned_config,
        hp_cols=hp_cols,
    )
    trainer.fit(df_train)
    pred_df = trainer.predict(df_test)

    df_moo = pd.concat([df_test, pred_df], axis=1)

    # ------------------------------
    # 1. SPF: Surrogate Pareto Front (predicted space)
    # ------------------------------
    pred_targets = df_moo[PRED_COLS].values
    spf_indices = NonDominatedSorting().do(
        pred_targets, only_non_dominated_front=True
    )

    # ------------------------------
    # 2. ESPF: Evaluated Surrogate Pareto Front (true space)
    # ------------------------------
    espf_true = df_moo.iloc[spf_indices][TRUE_COLS].values
    espf_sub_indices = NonDominatedSorting().do(
        espf_true, only_non_dominated_front=True
    )
    espf_indices = spf_indices[espf_sub_indices]

    # ------------------------------
    # 3. TPF: True Pareto Front (true space, full test set)
    # ------------------------------
    true_targets = df_moo[TRUE_COLS].values
    tpf_indices = NonDominatedSorting().do(
        true_targets, only_non_dominated_front=True
    )

    # ------------------------------
    # Export to CSV
    # ------------------------------
    PARETO_DIR.mkdir(parents=True, exist_ok=True)

    df_spf = df_moo.iloc[spf_indices].copy()
    df_spf.index.name = "df_moo_index"
    df_spf.to_csv(PARETO_DIR / "pareto_spf.csv")
    print(f"Saved pareto_spf.csv ({len(df_spf)} solutions)")

    df_espf = df_moo.iloc[espf_indices].copy()
    df_espf.index.name = "df_moo_index"
    df_espf.to_csv(PARETO_DIR / "pareto_espf.csv")
    print(f"Saved pareto_espf.csv ({len(df_espf)} solutions)")

    df_tpf = df_moo.iloc[tpf_indices].copy()
    df_tpf.index.name = "df_moo_index"
    df_tpf.to_csv(PARETO_DIR / "pareto_tpf.csv")
    print(f"Saved pareto_tpf.csv ({len(df_tpf)} solutions)")

    # Combined file with 'front' column for easy filtering in viz
    df_spf["front"] = "SPF"
    df_espf["front"] = "ESPF"
    df_tpf["front"] = "TPF"
    df_all = pd.concat([df_spf, df_espf, df_tpf], axis=0, ignore_index=False)
    df_all.to_csv(PARETO_DIR / "pareto_fronts.csv")
    print(f"Saved pareto_fronts.csv ({len(df_all)} rows, combined SPF+ESPF+TPF)")


if __name__ == "__main__":
    main()
