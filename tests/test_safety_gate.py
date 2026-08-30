import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safety_gate import apply_safety_gate


def base_record(**overrides):
    r = {
        "spo2": 98, "sbp": 120, "hr": 80, "rr": 16,
        "mental_status_altered": False, "bleeding": False,
        "pregnancy": False, "age_group": "adult", "fall": False,
        "has_prior_history": True,
    }
    r.update(overrides)
    return r


def base_feats(**overrides):
    f = {"_input_completeness": "HIGH", "n_worsening_trends": 0}
    f.update(overrides)
    return f


def test_gate_never_increases_band_number_above_model():
    band, triggers = apply_safety_gate(base_record(), base_feats(), model_band=4)
    assert band <= 4


def test_hard_redline_spo2_forces_band_1():
    band, triggers = apply_safety_gate(base_record(spo2=85), base_feats(), model_band=5)
    assert band == 1
    assert "spo2_low" in triggers


def test_hard_redline_sbp_forces_band_1():
    band, triggers = apply_safety_gate(base_record(sbp=80), base_feats(), model_band=4)
    assert band == 1


def test_deterioration_cannot_reduce_urgency():
    band_no_det, _ = apply_safety_gate(base_record(), base_feats(n_worsening_trends=0), model_band=3)
    band_det, triggers = apply_safety_gate(base_record(), base_feats(n_worsening_trends=3), model_band=3)
    assert band_det <= band_no_det
    assert "deterioration_trend" in triggers


def test_low_quality_data_cannot_suppress_danger():
    band_high_q, _ = apply_safety_gate(base_record(), base_feats(_input_completeness="HIGH"), model_band=3)
    band_low_q, triggers = apply_safety_gate(base_record(), base_feats(_input_completeness="LOW"), model_band=3)
    assert band_low_q <= band_high_q
    assert "low_data_quality_uncertainty" in triggers


def test_model_unavailable_forces_at_most_band_3():
    band, triggers = apply_safety_gate(base_record(), base_feats(), model_band=5, model_status="unavailable")
    assert band <= 3
    assert "model_unavailable_fallback" in triggers


def test_zero_history_low_quality_gets_floor():
    band, triggers = apply_safety_gate(
        base_record(has_prior_history=False), base_feats(_input_completeness="LOW"), model_band=5)
    assert band <= 3
    assert "zero_history_signal_floor" in triggers


def test_pregnancy_abdominal_pain_floor():
    band, triggers = apply_safety_gate(
        base_record(pregnancy=True, abdominal_pain=True), base_feats(), model_band=5)
    assert band <= 3
    assert "pregnancy_floor" in triggers


def test_geriatric_fall_floor():
    band, triggers = apply_safety_gate(
        base_record(age_group="geriatric", fall=True), base_feats(), model_band=5)
    assert band <= 3
    assert "geriatric_fall_floor" in triggers


def test_gate_is_monotonic_never_worse_than_model_band():
    for model_band in range(1, 6):
        band, _ = apply_safety_gate(base_record(spo2=85), base_feats(), model_band=model_band)
        assert band <= model_band
