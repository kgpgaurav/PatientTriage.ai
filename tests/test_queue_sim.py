import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from queue_sim import (
    HARD_PROTECTED_BANDS,
    SAFE_WAIT_MINUTES,
    _priority_key,
    _run_policy,
    run_operational_scenario,
)


def test_3x_scenario_is_crisis():
    result = run_operational_scenario(surge_multiplier=3, duration_min=180, n_clinicians=4, seed=1)
    assert result["operational_state"]["state"] == "CRISIS"
    assert result["config"]["surge_multiplier"] == 3


def test_normal_scenario_generates_roughly_baseline_volume():
    result = run_operational_scenario(surge_multiplier=1, duration_min=180, n_clinicians=4, seed=1)
    # ~20/hr * 3h = ~60, allow generous stochastic slack
    assert 30 <= result["n_arrivals"] <= 100


def test_policy_comparison_has_all_three_policies():
    result = run_operational_scenario(surge_multiplier=3, duration_min=180, n_clinicians=4, seed=1)
    labels = {row["policy"] for row in result["policy_comparison"]}
    assert labels == {"FIFO", "STATIC_SEVERITY", "WAIT_PROTECTED"}


def test_metrics_report_required_fields():
    result = run_operational_scenario(surge_multiplier=3, duration_min=180, n_clinicians=4, seed=1)
    for policy_metrics in result["policies"].values():
        for key in ("avg_wait_min", "p95_wait_min", "max_queue_length", "safe_wait_breaches",
                    "reassessment_events", "throughput_per_hour", "patients_served"):
            assert key in policy_metrics


def test_wait_protection_never_lets_band5_beat_a_real_band1():
    """Requirement #8: hard protection for the most urgent patients. A
    Band 5 patient who has waited a very long time must never sort ahead
    of a Band 1 patient under the wait-protected policy."""
    band1_patient = {"patient_id": "B1", "band": 1, "arrival_min": 100.0}
    band5_patient = {"patient_id": "B5", "band": 5, "arrival_min": 0.0}  # waited much longer
    clock = 500.0  # band5 has waited 500 min, band1 only 400 min

    key1 = _priority_key("wait_decay", band1_patient, clock)
    key5 = _priority_key("wait_decay", band5_patient, clock)
    assert key1 < key5, "Band 1 must still sort ahead of a long-waiting Band 5 patient"


def test_protected_bands_are_one_and_two():
    assert HARD_PROTECTED_BANDS == {1, 2}


def test_wait_decay_still_reduces_starvation_relative_to_static():
    """Under sustained crisis load, more low-acuity (band 4/5) patients
    should get served under WAIT_PROTECTED than under pure STATIC_SEVERITY,
    demonstrating starvation prevention without breaking hard protection."""
    result = run_operational_scenario(surge_multiplier=3, duration_min=180, n_clinicians=4, seed=7)
    static_low = sum(v["n"] for b, v in result["policies"]["static"]["by_band"].items() if b in (4, 5))
    protected_low = sum(v["n"] for b, v in result["policies"]["wait_decay"]["by_band"].items() if b in (4, 5))
    assert protected_low >= static_low
