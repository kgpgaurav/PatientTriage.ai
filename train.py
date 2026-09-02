import json
import os
import pickle

import pandas as pd

import models
from data_gen import generate_dataset
from features import to_dataframe


def main():
    os.makedirs("outputs", exist_ok=True)

    raw = generate_dataset(n=4000)
    raw.to_csv("outputs/synthetic_patients.csv", index=False)

    records = raw.to_dict(orient="records")
    feat_df = to_dataframe(records)
    feat_df["true_band"] = raw["true_band"].values
    feat_df["age_group"] = raw["age_group"].values

    y = (feat_df["true_band"] <= 2).astype(int)
    X, feature_cols = models.get_feature_matrix(feat_df)

    print("Running baseline comparison (LR / RF / XGB)...")
    baselines = models.baseline_comparison(X, y)

    print("Running cost-sensitivity sweep...")
    sensitivity = models.cost_sensitivity_sweep(X, y)

    print("Running age-aware ablation...")
    ablation = models.age_aware_ablation(feat_df, X, y)

    missingness_ablation = models.missingness_ablation(X, y)

    print("Training final calibrated critical-risk model...")
    critical_model = models.train_final_critical_model(X, y, scale_pos_weight=6.0)

    severity_target = feat_df["true_band"].apply(models.band_to_severity)
    severity_model = models.train_severity_model(X, severity_target)

    print("Computing global SHAP importance...")
    plain_xgb = models.build_xgb(6.0)
    plain_xgb.fit(X, y)
    global_shap = models.global_shap_importance(plain_xgb, X)

    y_prob_full = critical_model.predict_proba(X)[:, 1]
    subgroup = models.subgroup_recall(feat_df, y, y_prob_full)

    results = {
        "baselines": baselines,
        "cost_sensitivity_sweep": sensitivity,
        "age_aware_ablation": ablation,
        "missingness_ablation": missingness_ablation,
        "global_shap_importance": global_shap,
        "subgroup_recall_at_0.5": subgroup,
        "n_patients": len(feat_df),
        "critical_prevalence": float(y.mean()),
        "band_prevalence": feat_df["true_band"].value_counts(normalize=True).sort_index().to_dict(),
        "age_group_prevalence": feat_df["age_group"].value_counts(normalize=True).to_dict(),
    }

    with open("outputs/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open("outputs/critical_model.pkl", "wb") as f:
        pickle.dump(critical_model, f)
    with open("outputs/severity_model.pkl", "wb") as f:
        pickle.dump(severity_model, f)
    with open("outputs/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    print("Done. Results written to outputs/training_results.json")
    print(json.dumps({"xgb_vs_baselines_auprc": {k: v["mean"]["auprc"] for k, v in baselines.items()}}, indent=2))


if __name__ == "__main__":
    main()
