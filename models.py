import json
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

NON_FEATURE_COLS = {"true_band", "patient_id", "_input_completeness", "age_group"}


def get_feature_matrix(df):
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS and str(df[c].dtype) not in ("object", "str")]
    X = df[cols].fillna(0)
    return X, cols


def band1_2_recall(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    return recall_score(y_true, y_pred)


def evaluate_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "recall_critical": recall_score(y_true, y_pred),
        "precision_critical": precision_score(y_true, y_pred, zero_division=0),
        "false_negative_rate": 1 - recall_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def cross_validated_eval(X, y, build_model_fn, n_splits=5, n_seeds=3):
    results = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            model = build_model_fn()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            results.append(evaluate_binary(y.iloc[test_idx], prob))
    df = pd.DataFrame(results)
    return df.mean().to_dict(), df.std().to_dict()


def build_logreg():
    return LogisticRegression(max_iter=500, class_weight="balanced")


def build_rf():
    return RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=0)


def build_xgb(scale_pos_weight=6.0):
    return xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=0,
    )


def baseline_comparison(X, y):
    out = {}
    for name, fn in [("logistic_regression", build_logreg), ("random_forest", build_rf), ("xgboost", lambda: build_xgb(6.0))]:
        mean, std = cross_validated_eval(X, y, fn)
        out[name] = {"mean": mean, "std": std}
    return out


def cost_sensitivity_sweep(X, y, weights=(2, 4, 6, 10)):
    out = {}
    for w in weights:
        mean, std = cross_validated_eval(X, y, lambda w=w: build_xgb(w))
        out[str(w)] = {"mean": mean, "std": std}
    return out


def train_final_critical_model(X, y, scale_pos_weight=6.0):
    base = build_xgb(scale_pos_weight)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    calibrated.fit(X, y)
    return calibrated


def train_severity_model(X, severity_target):
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.07, random_state=0)
    model.fit(X, severity_target)
    return model


def band_to_severity(band):
    return {1: 95, 2: 78, 3: 55, 4: 30, 5: 10}.get(int(band), 50)


def severity_to_band(score):
    if score >= 85:
        return 1
    if score >= 65:
        return 2
    if score >= 42:
        return 3
    if score >= 20:
        return 4
    return 5


def global_shap_importance(model, X, sample_n=500):
    est = model.calibrated_classifiers_[0].estimator if hasattr(model, "calibrated_classifiers_") else model
    explainer = shap.TreeExplainer(est)
    sample = X.sample(min(sample_n, len(X)), random_state=0)
    shap_values = explainer.shap_values(sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [(X.columns[i], float(mean_abs[i])) for i in order[:12]]


def prediction_confidence(model, X_row, input_completeness="HIGH"):
    """Model-certainty estimate, distinct from `_input_completeness`.

    `_input_completeness` only says how much intake data was supplied; it
    says nothing about whether the model itself is sure of its answer. This
    looks at how much the model's own cross-validated sub-classifiers agree
    with each other on this specific patient (the calibrated critical model
    is a `CalibratedClassifierCV`, so it already carries 5 independently
    fit fold estimators -- we reuse them instead of training anything new).

    Two things lower confidence: (1) the fold estimators disagree with each
    other on this patient (high ensemble spread), and (2) the input used to
    reach that patient's prediction was itself thin (LOW/MEDIUM
    completeness). Either one alone is enough to cap the label at MEDIUM;
    both together cap it at LOW.

    Returns a dict that is safe to attach to every scored result -- callers
    should never emit `critical_probability` / `severity_score` without it.
    """
    try:
        fold_probs = [
            clf.estimator.predict_proba(X_row)[0, 1] if hasattr(clf, "estimator")
            else clf.predict_proba(X_row)[0, 1]
            for clf in model.calibrated_classifiers_
        ]
        spread = float(np.std(fold_probs))
        mean_prob = float(np.mean(fold_probs))
    except Exception:
        spread, mean_prob = None, None

    if spread is None:
        return {
            "confidence_score": None,
            "confidence_level": "LOW",
            "confidence_reason": "Could not compute model agreement for this input.",
        }

    # Ensemble spread -> raw confidence (0 spread == folds agree perfectly).
    ensemble_confidence = max(0.0, 1.0 - min(spread / 0.25, 1.0))

    completeness_penalty = {"HIGH": 0.0, "MEDIUM": 0.15, "LOW": 0.35}.get(input_completeness, 0.25)
    confidence_score = round(max(0.0, ensemble_confidence - completeness_penalty), 3)

    if confidence_score >= 0.7:
        level = "HIGH"
    elif confidence_score >= 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"

    reasons = []
    if spread > 0.1:
        reasons.append(f"model's cross-validated folds disagree on this patient (±{spread:.2f})")
    if input_completeness != "HIGH":
        reasons.append(f"{input_completeness.lower()} input completeness")
    reason = "; ".join(reasons) if reasons else "folds agree closely and intake data was complete"

    return {
        "confidence_score": confidence_score,
        "confidence_level": level,
        "confidence_reason": reason,
        "ensemble_fold_spread": round(spread, 4),
    }


def patient_shap_explanation(model, X_row, feature_cols, top_n=5):
    est = model.calibrated_classifiers_[0].estimator if hasattr(model, "calibrated_classifiers_") else model
    explainer = shap.TreeExplainer(est)
    shap_values = explainer.shap_values(X_row)
    vals = shap_values[0] if shap_values.ndim > 1 else shap_values
    pairs = sorted(zip(feature_cols, vals), key=lambda p: -abs(p[1]))[:top_n]
    return [{"feature": f, "contribution": float(v)} for f, v in pairs]


def age_aware_ablation(df, X, y):
    naive_cols = [c for c in X.columns if not c.endswith("_deviation") and c not in ("is_pediatric", "is_geriatric")]
    naive_cols = [c for c in naive_cols if c in X.columns]
    mean_a, std_a = cross_validated_eval(X[naive_cols], y, lambda: build_xgb(6.0))
    mean_b, std_b = cross_validated_eval(X, y, lambda: build_xgb(6.0))
    return {"naive_age_feature": mean_a, "age_conditioned": mean_b}

def missingness_ablation(X, y):
    missingness_cols = [c for c in X.columns if c.endswith("_missing")]
    without_cols = [c for c in X.columns if c not in missingness_cols]
    mean_with, std_with = cross_validated_eval(X, y, lambda: build_xgb(6.0))
    mean_without, std_without = cross_validated_eval(X[without_cols], y, lambda: build_xgb(6.0))
    return {
        "with_missingness_features": {"mean": mean_with, "std": std_with},
        "without_missingness_features": {"mean": mean_without, "std": std_without},
        "missingness_feature_count": len(missingness_cols),
    }


def subgroup_recall(df, y, y_prob, threshold=0.5):
    out = {}
    y_pred = (y_prob >= threshold).astype(int)
    for grp in ["pediatric", "adult", "geriatric"]:
        mask = df["age_group"] == grp if "age_group" in df.columns else None
        if mask is None or mask.sum() < 5:
            continue
        out[grp] = {
            "n": int(mask.sum()),
            "recall": float(recall_score(y[mask], y_pred[mask])) if mask.sum() > 0 else None,
        }
    return out
