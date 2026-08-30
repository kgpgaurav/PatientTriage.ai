import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

import audit
from features import build_feature_row
from pipeline import TriagePipeline
from validation import ValidationError

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


@pytest.fixture(scope="module")
def pipeline():
    with open(os.path.join(OUT, "critical_model.pkl"), "rb") as f:
        critical_model = pickle.load(f)
    with open(os.path.join(OUT, "severity_model.pkl"), "rb") as f:
        severity_model = pickle.load(f)
    with open(os.path.join(OUT, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    return TriagePipeline(critical_model, severity_model, feature_cols)


def base_vitals(**overrides):
    r = {"patient_id": "T0", "age": 40, "age_group": "adult", "hr": 88, "sbp": 118,
         "rr": 16, "temp": 37.0, "spo2": 97, "has_prior_history": True}
    r.update(overrides)
    return r


def test_missing_vitals_do_not_crash(pipeline):
    record = {"patient_id": "T1", "age": 40, "age_group": "adult", "has_prior_history": True}
    result = pipeline.run(record, log_audit=False)
    assert result["final_recommended_band"] in range(1, 6)


def test_invalid_negative_vital_is_rejected(pipeline):
    record = base_vitals(patient_id="T2", hr=-5)
    with pytest.raises(ValidationError):
        pipeline.run(record, log_audit=False)


def test_invalid_spo2_over_100_is_rejected(pipeline):
    record = base_vitals(patient_id="T2b", spo2=250)
    with pytest.raises(ValidationError):
        pipeline.run(record, log_audit=False)


def test_invalid_negative_rr_and_temp_are_rejected(pipeline):
    record = base_vitals(patient_id="T2c", rr=-3, temp=-50)
    with pytest.raises(ValidationError):
        pipeline.run(record, log_audit=False)


def test_extreme_but_valid_spo2_is_not_rejected_and_escalates(pipeline):
    record = base_vitals(patient_id="T2d", spo2=82)
    result = pipeline.run(record, log_audit=False)
    assert result["final_recommended_band"] == 1
    assert "spo2_low" in result["safety_gate_triggers"]


def test_missing_vitals_stay_missing_not_zero(pipeline):
    record = {"patient_id": "T2e", "age": 40, "age_group": "adult", "has_prior_history": True, "spo2": None}
    result = pipeline.run(record, log_audit=False)
    assert result["feature_snapshot"]["spo2_missing"] == 1
    assert "spo2_low" not in result["safety_gate_triggers"]


def test_missing_feature_column_defaults_safely(pipeline):
    record = {"patient_id": "T3", "age": 30, "age_group": "adult", "has_prior_history": False}
    feats = build_feature_row(record)
    df = pd.DataFrame([feats])
    for col in pipeline.feature_cols:
        if col not in df.columns:
            df[col] = 0
    assert set(pipeline.feature_cols).issubset(set(df.columns))


def test_unexpected_schema_extra_columns_ignored(pipeline):
    record = {"patient_id": "T4", "age": 30, "age_group": "adult", "has_prior_history": True,
              "totally_unexpected_field": "xyz"}
    result = pipeline.run(record, log_audit=False)
    assert result["patient_id"] == "T4"


def test_model_unavailable_fallback_path(pipeline):
    pipeline.set_model_unavailable()
    record = {"patient_id": "T5", "age": 50, "age_group": "adult", "has_prior_history": True}
    result = pipeline.run(record, log_audit=False)
    assert result["model_status"] == "unavailable"
    assert result["final_recommended_band"] <= 3
    assert result["critical_probability"] is None
    assert result["severity_score"] is None
    assert result["recommendation_mode"] == "safety_fallback"
    assert result["fallback_band"] == 3
    pipeline.set_model_available()


def test_model_unavailable_with_dangerous_spo2_still_escalates(pipeline):
    pipeline.set_model_unavailable()
    record = base_vitals(patient_id="T5b", spo2=85)
    result = pipeline.run(record, log_audit=False)
    assert result["final_recommended_band"] == 1
    pipeline.set_model_available()


def test_model_available_recommendation_mode(pipeline):
    record = base_vitals(patient_id="T5c")
    result = pipeline.run(record, log_audit=False)
    assert result["recommendation_mode"] == "model"
    assert result["fallback_band"] is None


def test_zero_history_first_time_patient(pipeline):
    record = {"patient_id": "T6", "age": 22, "age_group": "adult", "has_prior_history": False,
              "mental_status_altered": True}
    result = pipeline.run(record, log_audit=False)
    assert result["final_recommended_band"] <= 3


def test_downgrade_requires_reason_code(pipeline):
    with pytest.raises(ValueError):
        pipeline.record_clinician_decision("T7", ai_recommendation_band=2, clinician_band=4, reason_code=None)


def test_end_to_end_free_text_flow(pipeline):
    record = base_vitals(patient_id="T8")
    result = pipeline.run(record, note="denies chest pain, denies shortness of breath", log_audit=False)
    assert result["extraction_status"] == "ok"


def test_age_group_derived_not_taken_from_client():
    cases = [(12, "pediatric"), (13, "adult"), (64, "adult"), (65, "geriatric")]
    for age, expected in cases:
        record = base_vitals(patient_id=f"AG-{age}", age=age, age_group=None)
        result = _run_without_model(record)
        assert result["age_group"] == expected


def test_contradictory_client_age_group_is_overridden_by_backend():
    record = base_vitals(patient_id="AG-CONTRA", age=70, age_group="adult", fall=True)
    result = _run_without_model(record)
    assert result["age_group"] == "geriatric"
    assert result["age_group_overridden"] is True


def test_matching_client_age_group_not_flagged_as_overridden():
    record = base_vitals(patient_id="AG-MATCH", age=70, age_group="geriatric")
    result = _run_without_model(record)
    assert result["age_group_overridden"] is False


def _run_without_model(record):
    """A minimal pipeline whose model is deliberately unavailable, so these
    age-group/mismatch tests don't depend on the trained model's threshold
    behavior -- they only exercise pipeline.run()'s own logic."""
    p = TriagePipeline(None, None, [])
    p.set_model_unavailable()
    return p.run(record, log_audit=False)


def test_mismatch_flag_is_informational_only_and_not_a_feature(pipeline):
    base = base_vitals(patient_id="MM1")
    r_mismatch = pipeline.run(dict(base), note="denies shortness of breath, appears anxious and in distress", log_audit=False)
    r_no_mismatch = pipeline.run(dict(base), note="denies shortness of breath", log_audit=False)
    assert r_mismatch["observed_reported_mismatch"] is True
    assert r_no_mismatch["observed_reported_mismatch"] is False
    assert r_mismatch["model_recommended_band"] == r_no_mismatch["model_recommended_band"]
    assert "observed_reported_mismatch" not in pipeline.feature_cols


def test_audit_entry_has_single_calibrated_probability_field(tmp_path, monkeypatch, pipeline):
    log_path = tmp_path / "audit_log.jsonl"
    monkeypatch.setattr(audit, "LOG_PATH", str(log_path))
    record = base_vitals(patient_id="AUD1")
    result = pipeline.run(record, log_audit=True)
    lines = log_path.read_text().strip().splitlines()
    entry = json.loads(lines[-1])
    assert "raw_model_output" not in entry
    assert entry["calibrated_probability"] == result["critical_probability"]
    assert entry["probability_stage"] == "calibrated_only"
