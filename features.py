import numpy as np
import pandas as pd

from data_gen import VITAL_RANGES, SYMPTOMS, age_group

VITAL_KEYS = ["hr", "sbp", "rr", "temp", "spo2"]


def _deviation(value, group, key):
    if pd.isna(value):
        return 0.0
    lo, hi = VITAL_RANGES[group][key]
    mid = (lo + hi) / 2
    half_range = (hi - lo) / 2
    if value < lo:
        return (lo - value) / half_range
    if value > hi:
        return (value - hi) / half_range
    return 0.0


def build_age_conditioned_features(record):
    group = record.get("age_group") or age_group(record["age"])
    feats = {"age_group": group}
    for key in VITAL_KEYS:
        val = record.get(key)
        feats[f"{key}_deviation"] = _deviation(val, group, key)
        feats[f"{key}_abnormal"] = int(feats[f"{key}_deviation"] > 0)
    spo2 = record.get("spo2")
    feats["spo2_abs_deficit"] = max(0.0, 100 - spo2) if spo2 is not None and not pd.isna(spo2) else 0.0
    feats["is_pediatric"] = int(group == "pediatric")
    feats["is_geriatric"] = int(group == "geriatric")
    return feats


def build_temporal_features(record, history=None):
    feats = {}
    if not history:
        for key in VITAL_KEYS:
            feats[f"d_{key}"] = 0.0
            feats[f"{key}_slope"] = 0.0
        feats["n_worsening_trends"] = 0
        feats["time_since_prev_reading_min"] = np.nan
        return feats

    prev = history[-1]
    worsening = 0
    for key in VITAL_KEYS:
        cur_v = record.get(key)
        prev_v = prev.get(key)
        if cur_v is None or prev_v is None or pd.isna(cur_v) or pd.isna(prev_v):
            feats[f"d_{key}"] = 0.0
        else:
            delta = cur_v - prev_v
            feats[f"d_{key}"] = delta
            direction_bad = (key == "spo2" and delta < 0) or (key != "spo2" and abs(cur_v - _reference_mid(record, key)) > abs(prev_v - _reference_mid(record, key)))
            if direction_bad and abs(delta) > 0.01:
                worsening += 1

    if len(history) >= 2:
        for key in VITAL_KEYS:
            vals = [h.get(key) for h in history[-3:]] + [record.get(key)]
            vals = [v for v in vals if v is not None and not pd.isna(v)]
            if len(vals) >= 2:
                feats[f"{key}_slope"] = (vals[-1] - vals[0]) / max(1, len(vals) - 1)
            else:
                feats[f"{key}_slope"] = 0.0
    else:
        for key in VITAL_KEYS:
            feats[f"{key}_slope"] = 0.0

    feats["n_worsening_trends"] = worsening
    feats["time_since_prev_reading_min"] = record.get("minutes_since_prev", np.nan)
    return feats


def _reference_mid(record, key):
    group = record.get("age_group") or age_group(record["age"])
    lo, hi = VITAL_RANGES[group][key]
    return (lo + hi) / 2


def build_missingness_features(record):
    feats = {}
    for key in VITAL_KEYS:
        v = record.get(key)
        feats[f"{key}_missing"] = int(v is None or pd.isna(v))
    feats["history_missing"] = int(not record.get("has_prior_history", False))
    feats["symptoms_missing"] = int(record.get("no_symptom_data", False))
    feats["repeat_vitals_missing"] = int(not record.get("has_repeat_vitals", False))
    return feats


def build_symptom_features(record):
    return {s: int(bool(record.get(s, False))) for s in SYMPTOMS}


def input_completeness_label(missingness_feats):
    n_missing = sum(v for k, v in missingness_feats.items() if k.endswith("_missing"))
    if n_missing == 0:
        return "HIGH"
    if n_missing <= 2:
        return "MEDIUM"
    return "LOW"


def build_feature_row(record, history=None):
    feats = {}
    feats.update({"age": record["age"]})
    feats["mental_status_altered"] = int(bool(record.get("mental_status_altered", False)))
    feats["pregnancy"] = int(bool(record.get("pregnancy", False)))
    feats.update(build_age_conditioned_features(record))
    feats.update(build_temporal_features(record, history))
    miss = build_missingness_features(record)
    feats.update(miss)
    feats.update(build_symptom_features(record))
    feats["_input_completeness"] = input_completeness_label(miss)
    return feats


FEATURE_COLUMNS = None


def to_dataframe(records, histories=None):
    rows = []
    for i, r in enumerate(records):
        hist = histories[i] if histories else None
        rows.append(build_feature_row(r, hist))
    df = pd.DataFrame(rows)
    return df
