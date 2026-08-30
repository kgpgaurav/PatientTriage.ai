# Conservative operational default used when the ML model is unavailable.
# This is NOT a clinically validated "safe" band -- it is a prototype
# fallback ceiling: hard redlines below can still escalate a patient past it,
# but the fallback path itself never recommends anything less urgent than
# this. See README "Model-unavailable fallback" section.
FALLBACK_BAND = 3

HARD_REDLINES = {
    "spo2_low": lambda r: r.get("spo2") is not None and r.get("spo2") < 90,
    "sbp_low": lambda r: r.get("sbp") is not None and r.get("sbp") < 90,
    "hr_extreme": lambda r: r.get("hr") is not None and (r.get("hr") > 150 or r.get("hr") < 45),
    "rr_extreme": lambda r: r.get("rr") is not None and (r.get("rr") < 9 or r.get("rr") > 32),
    "red_flag_symptom": lambda r: bool(r.get("red_flags_from_text")),
    "active_bleeding": lambda r: bool(r.get("bleeding")),
    "altered_mental_status": lambda r: bool(r.get("mental_status_altered")),
}

BAND_ORDER = [1, 2, 3, 4, 5]


def _min_of(a, b):
    return min(a, b)


def apply_safety_gate(record, feats, model_band, model_status="ok"):
    triggered = []
    escalated_band = model_band

    for name, check in HARD_REDLINES.items():
        try:
            if check(record):
                triggered.append(name)
                escalated_band = _min_of(escalated_band, 1)
        except Exception:
            continue

    if feats.get("n_worsening_trends", 0) >= 2:
        triggered.append("deterioration_trend")
        escalated_band = _min_of(escalated_band, max(1, model_band - 1))

    completeness = feats.get("_input_completeness", "HIGH")
    if completeness == "LOW" and model_band <= 3:
        triggered.append("low_data_quality_uncertainty")
        escalated_band = _min_of(escalated_band, max(1, model_band - 1))

    if model_status != "ok":
        triggered.append("model_unavailable_fallback")
        escalated_band = _min_of(escalated_band, FALLBACK_BAND)

    if not record.get("has_prior_history", True) and (record.get("mental_status_altered") or feats.get("_input_completeness") == "LOW") and escalated_band > 3:
        triggered.append("zero_history_signal_floor")
        escalated_band = 3

    if record.get("pregnancy") and (record.get("bleeding") or record.get("abdominal_pain")) and escalated_band > 3:
        triggered.append("pregnancy_floor")
        escalated_band = 3

    if record.get("age_group") == "geriatric" and record.get("fall") and escalated_band > 3:
        triggered.append("geriatric_fall_floor")
        escalated_band = 3

    escalated_band = min(escalated_band, model_band)
    return escalated_band, triggered


def gate_reason_text(triggered):
    if not triggered:
        return "No safety-gate escalation triggered."
    labels = {
        "spo2_low": "SpO2 below hard threshold",
        "sbp_low": "Systolic BP below hard threshold",
        "hr_extreme": "Heart rate outside safe range",
        "rr_extreme": "Respiratory rate outside safe range",
        "red_flag_symptom": "Red-flag phrase detected in intake note",
        "active_bleeding": "Active bleeding reported",
        "altered_mental_status": "Altered mental status observed",
        "deterioration_trend": "Worsening trend across repeat vitals",
        "low_data_quality_uncertainty": "Low input completeness with residual risk signal",
        "model_unavailable_fallback": "ML model unavailable, rules fallback engaged",
        "zero_history_signal_floor": "First-time patient with meaningful signal — urgency floor applied",
        "pregnancy_floor": "Pregnancy with bleeding/abdominal pain — urgency floor applied",
        "geriatric_fall_floor": "Geriatric fall — urgency floor applied",
    }
    return "; ".join(labels.get(t, t) for t in triggered)
