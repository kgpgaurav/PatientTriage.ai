import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from surge import BASELINE_ARRIVALS_PER_HOUR, classify_load_multiplier, determine_operational_state


def test_normal_threshold():
    assert classify_load_multiplier(1.0) == "NORMAL"
    assert classify_load_multiplier(1.25) == "NORMAL"


def test_surge_threshold():
    assert classify_load_multiplier(1.26) == "SURGE"
    assert classify_load_multiplier(2.0) == "SURGE"


def test_crisis_threshold():
    assert classify_load_multiplier(2.01) == "CRISIS"
    assert classify_load_multiplier(3.0) == "CRISIS"


def test_3x_baseline_is_crisis():
    """The Round 2 brief's headline scenario: 20/hr baseline x3 = 60/hr
    must be explicitly classified CRISIS."""
    result = determine_operational_state(arrival_rate=60, baseline_rate=20)
    assert result["state"] == "CRISIS"
    assert result["load_multiplier"] == 3.0
    assert result["arrival_rate"] == 60
    assert result["baseline_rate"] == 20


def test_default_baseline_used_when_not_given():
    result = determine_operational_state(arrival_rate=BASELINE_ARRIVALS_PER_HOUR)
    assert result["state"] == "NORMAL"


def test_queue_pressure_can_escalate_state():
    # Arrival rate alone looks NORMAL, but a very deep backlog relative to
    # clinician count pushes the state up -- a prototype heuristic, not a
    # clinical judgement.
    result = determine_operational_state(arrival_rate=20, baseline_rate=20, queue_length=30, n_clinicians=4)
    assert result["state"] in ("SURGE", "CRISIS")
    assert result["queue_pressure_override"] is True


def test_output_never_mentions_clinical_fields():
    result = determine_operational_state(arrival_rate=60, baseline_rate=20)
    forbidden = {"band", "clinical_band", "final_recommended_band", "severity", "safety_gate"}
    assert forbidden.isdisjoint(result.keys())
