import pandas as pd

import audit
import models
from data_gen import age_group as canonical_age_group
from features import build_feature_row
from llm_extract import extract_from_note, merge_extraction_into_record
from safety_gate import FALLBACK_BAND, apply_safety_gate, gate_reason_text
from validation import validate_vitals


class TriagePipeline:
    def __init__(self, critical_model, severity_model, feature_cols):
        self.critical_model = critical_model
        self.severity_model = severity_model
        self.feature_cols = feature_cols
        self.model_status = "ok"

    def run(self, record, history=None, note=None, log_audit=True):
        validate_vitals(record)
        input_snapshot = dict(record)

        record = dict(record)
        derived_age_group = canonical_age_group(record["age"])
        client_age_group = record.get("age_group")
        age_group_overridden = bool(client_age_group and client_age_group != derived_age_group)
        record["age_group"] = derived_age_group

        extraction_status = "no_note"
        extraction_backend = None
        red_flag_sources = {}
        if note:
            extraction = extract_from_note(note, timeout_ok=self.model_status == "ok")
            record = merge_extraction_into_record(record, extraction)
            extraction_status = extraction["extraction_status"]
            extraction_backend = extraction.get("extraction_backend")
            red_flag_sources = extraction.get("red_flag_sources", {})

        feats = build_feature_row(record, history)
        X_row = pd.DataFrame([feats])[self.feature_cols].fillna(0)

        if self.model_status == "ok":
            critical_probability = float(self.critical_model.predict_proba(X_row)[0, 1])
            severity_score = float(self.severity_model.predict(X_row)[0])
            confidence = models.prediction_confidence(
                self.critical_model, X_row, input_completeness=feats.get("_input_completeness", "HIGH")
            )
        else:
            critical_probability = None
            severity_score = None
            # No score is being returned at all here (safety_gate fallback
            # takes over) -- but we still surface an explicit confidence
            # record rather than silently omitting one, per the "never
            # return a score without a confidence indicator" rule.
            confidence = {
                "confidence_score": None,
                "confidence_level": "LOW",
                "confidence_reason": "ML model unavailable -- rules-based safety fallback in use.",
            }

        if critical_probability is not None:
            model_band = 1 if critical_probability >= 0.55 else models.severity_to_band(severity_score)
        else:
            model_band = FALLBACK_BAND

        gate_band, triggers = apply_safety_gate(record, feats, model_band, self.model_status)

        explanation = None
        if self.model_status == "ok":
            try:
                explanation = models.patient_shap_explanation(self.critical_model, X_row, self.feature_cols)
            except Exception:
                explanation = None

        result = {
            "patient_id": record.get("patient_id"),
            "critical_probability": critical_probability,
            "severity_score": severity_score,
            "confidence_score": confidence.get("confidence_score"),
            "confidence_level": confidence.get("confidence_level"),
            "confidence_reason": confidence.get("confidence_reason"),
            "input_completeness": feats.get("_input_completeness"),
            "model_recommended_band": model_band,
            "final_recommended_band": gate_band,
            "safety_gate_triggers": triggers,
            "safety_gate_reason": gate_reason_text(triggers),
            "model_explanation": explanation,
            "extraction_status": extraction_status,
            "extraction_backend": extraction_backend,
            "model_status": self.model_status,
            "recommendation_mode": "model" if self.model_status == "ok" else "safety_fallback",
            "fallback_band": None if self.model_status == "ok" else FALLBACK_BAND,
            "age_group": derived_age_group,
            "age_group_overridden": age_group_overridden,
            "observed_reported_mismatch": record.get("observed_reported_mismatch", False),
            "red_flag_sources": red_flag_sources,
            "feature_snapshot": feats,
        }

        if log_audit:
            entry = audit.build_audit_entry(
                patient_id=record.get("patient_id"),
                input_snapshot=input_snapshot,
                feature_snapshot=feats,
                extraction_status=extraction_status,
                calibrated_probability=critical_probability,
                gate_band=gate_band,
                gate_triggers=triggers,
                ai_recommendation=gate_band,
                fallback_status="none" if self.model_status == "ok" else "rules_fallback",
            )
            entry["extraction_backend"] = extraction_backend
            entry["red_flag_sources"] = red_flag_sources
            entry["age_group_overridden"] = age_group_overridden
            entry["confidence_score"] = confidence.get("confidence_score")
            entry["confidence_level"] = confidence.get("confidence_level")
            entry["confidence_reason"] = confidence.get("confidence_reason")
            audit.write_record(entry)

        return result

    def record_clinician_decision(self, patient_id, ai_recommendation_band, clinician_band, reason_code=None):
        downgrade = clinician_band > ai_recommendation_band
        if downgrade and not reason_code:
            raise ValueError("Downgrading below the AI recommendation requires a structured reason code.")
        entry = {
            "event": "clinician_override" if clinician_band != ai_recommendation_band else "clinician_confirmed",
            "patient_id": patient_id,
            "ai_recommendation_band": ai_recommendation_band,
            "clinician_decision_band": clinician_band,
            "override_reason": reason_code,
            "is_downgrade": downgrade,
        }
        audit.write_record(entry)
        return entry

    def set_model_unavailable(self):
        self.model_status = "unavailable"

    def set_model_available(self):
        self.model_status = "ok"
