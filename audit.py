import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "outputs", "audit_log.jsonl")

SCHEMA_VERSION = "1.1"
MODEL_VERSION = "xgb-critical-v1"
CALIBRATION_VERSION = "sigmoid-cv5-v1"

# The saved critical_model is a single sklearn CalibratedClassifierCV: base
# XGBoost folds and their sigmoid calibrators are fit and averaged together
# internally, so there is no separately addressable "raw" (pre-calibration)
# probability to log -- only the calibrated output the classifier returns.
# `calibrated_probability` below is that value, and it is the same number
# used for the model_band decision downstream. See README "Model probability"
# section.
PROBABILITY_STAGE = "calibrated_only"


def write_record(entry):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = dict(entry)
    entry["logged_at"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def build_audit_entry(patient_id, input_snapshot, feature_snapshot, extraction_status,
                       calibrated_probability, gate_band, gate_triggers,
                       ai_recommendation, clinician_decision=None, override_reason=None,
                       fallback_status="none"):
    return {
        "patient_id": patient_id,
        "input_snapshot": input_snapshot,
        "feature_snapshot": feature_snapshot,
        "extraction_status": extraction_status,
        "model_version": MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "feature_schema_version": SCHEMA_VERSION,
        "probability_stage": PROBABILITY_STAGE,
        "calibrated_probability": calibrated_probability,
        "safety_gate_triggers": gate_triggers,
        "final_ai_recommendation_band": ai_recommendation,
        "clinician_decision_band": clinician_decision,
        "override_reason": override_reason,
        "fallback_status": fallback_status,
    }


def read_all():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]
