"""
Live surge-arrival simulator (for the "Simulate patient surge" button).

Unlike `queue_sim.run_operational_scenario` -- an offline, in-memory what-if
tool used by `POST /surge/simulate` -- this module inserts real rows into the
live `patients` table through the exact same `pipeline.run()` +
`db.insert_triage_record()` path a real nurse's `POST /triage` submission
uses. Its only purpose is to give the already-existing, arrival-frequency
based auto-detector (`surge.py` / `db.get_live_operational_status`)
something real to detect. It never classifies, tags, or pre-labels a
patient -- every simulated patient is triaged by the real pipeline exactly
like any other arrival, and is stored/displayed identically.

Patients are logged one at a time, in real time (`arrival_time` is the
actual moment of insertion), starting immediately when triggered. After each
insert, the loop reads the same live operational status every other part of
the app reads, and stops itself the moment that status leaves NORMAL -- i.e.
the moment the system's own detector says SURGE (or CRISIS) -- rather than
stopping at a pre-decided patient count. Two safety caps (max patients, max
duration) guard against a run that never crosses the threshold, e.g. if the
baseline arrival rate has been reconfigured very high.

Every step is audited (see db.insert_audit): a `surge_simulation_started`
event when triggered, one `surge_simulation_patient_logged` event per
synthetic patient, and a `surge_simulation_completed` event when it stops
(with the reason). An unexpected error mid-run is itself audited
(`surge_simulation_error`) and stops the loop rather than retrying forever.
This keeps a full, separate history of how any given surge state was
reached -- without adding any visible marker to the patient's row itself.
"""

import threading
import time
from datetime import datetime, timezone

import numpy as np

import db
from data_gen import SYMPTOMS, VITAL_RANGES, age_group as canonical_age_group
from validation import PLAUSIBILITY_BOUNDS

# --- tunable prototype constants (not clinical or capacity standards) -----
POLL_INTERVAL_SEC = 2.0   # gap between simulated arrivals
MAX_PATIENTS = 80         # hard safety cap regardless of state reached
MAX_DURATION_SEC = 360    # hard safety cap (6 min) regardless of count
AGE_GROUP_MIX = {"pediatric": 0.15, "adult": 0.60, "geriatric": 0.25}
# Keep generated vitals a hair inside the plausibility bounds validate_vitals
# enforces, so a random draw is never rejected mid-run.
_MARGIN = 0.1

_lock = threading.Lock()
_state = {
    "running": False,
    "count_logged": 0,
    "started_at": None,
    "last_result": None,  # set once a run finishes: {"reason", "final_state", "total_logged", "run_id"}
}


def get_status():
    """Snapshot of the simulator's current/last run, merged into GET /surge/status."""
    with _lock:
        return dict(_state)


def _clip(value, key):
    lo, hi = PLAUSIBILITY_BOUNDS[key]
    return round(float(min(hi - _MARGIN, max(lo + _MARGIN, value))), 1)


def _sample_patient(rng, seq, run_id):
    """One synthetic patient record, shaped exactly like a real /triage
    submission's stored input (see patients.DEMO_PATIENTS / api.PatientInput)
    -- minus `note`, so this never calls the LLM extraction path."""
    bucket = rng.choice(list(AGE_GROUP_MIX.keys()), p=list(AGE_GROUP_MIX.values()))
    if bucket == "pediatric":
        age = int(rng.integers(0, 13))
    elif bucket == "adult":
        age = int(rng.integers(13, 65))
    else:
        age = int(rng.integers(65, 96))
    group = canonical_age_group(age)
    severity = float(rng.beta(1.5, 4))

    record = {"patient_id": f"SURGE-SIM-{run_id}-{seq:04d}", "age": age, "age_group": group}
    for key, (lo, hi) in VITAL_RANGES[group].items():
        center = rng.uniform(lo, hi)
        spread = (hi - lo) * (0.3 + severity)
        val = center + rng.normal(0, spread) * severity * 2
        if key == "spo2":
            val = center - severity * rng.uniform(0, 15)
        record[key] = _clip(val, key)

    n_symptoms = int(rng.integers(0, 3))
    active_symptoms = rng.choice(SYMPTOMS, size=n_symptoms, replace=False) if n_symptoms else []
    for s in SYMPTOMS:
        record[s] = bool(s in active_symptoms)

    record["mental_status_altered"] = bool(rng.random() < (0.05 + severity * 0.3))
    record["pregnancy"] = bool(group == "adult" and rng.random() < 0.05)
    record["has_prior_history"] = bool(rng.random() < 0.5)
    return record


def _run_loop(pipeline, n_clinicians):
    run_id = datetime.now(timezone.utc).strftime("%H%M%S")
    rng = np.random.default_rng()
    started_at = datetime.now(timezone.utc).isoformat()

    with _lock:
        _state.update(running=True, count_logged=0, started_at=started_at, last_result=None)

    db.insert_audit("surge_simulation_started", None, {
        "run_id": run_id, "target_state": "SURGE",
        "max_patients": MAX_PATIENTS, "max_duration_sec": MAX_DURATION_SEC,
    })

    reason = "unknown"
    final_state = None
    seq = 0
    start_time = time.monotonic()

    try:
        while True:
            seq += 1
            record = _sample_patient(rng, seq, run_id)
            patient_id = record["patient_id"]

            result = pipeline.run(record, history=None, note=None)
            record["age_group"] = result.get("age_group", record.get("age_group"))
            record["observed_reported_mismatch"] = result.get("observed_reported_mismatch", False)
            row_id, _ = db.insert_triage_record(patient_id, record, result, arrival_time=None)

            with _lock:
                _state["count_logged"] = seq

            db.insert_audit("surge_simulation_patient_logged", patient_id, {
                "run_id": run_id, "row_id": row_id, "sequence": seq,
                "final_recommended_band": result.get("final_recommended_band"),
            })

            status = db.get_live_operational_status(n_clinicians=n_clinicians)
            final_state = status["operational_state"]["state"]
            if final_state != "NORMAL":
                reason = "target_reached"
                break
            if seq >= MAX_PATIENTS:
                reason = "safety_cap_patients"
                break
            if time.monotonic() - start_time >= MAX_DURATION_SEC:
                reason = "safety_cap_duration"
                break

            time.sleep(POLL_INTERVAL_SEC)
    except Exception as exc:
        reason = "error"
        db.insert_audit("surge_simulation_error", None, {"run_id": run_id, "sequence": seq, "error": str(exc)})
    finally:
        result_summary = {"reason": reason, "final_state": final_state, "total_logged": seq, "run_id": run_id}
        db.insert_audit("surge_simulation_completed", None, result_summary)
        with _lock:
            _state.update(running=False, last_result=result_summary)


def start(pipeline, n_clinicians=4):
    """Kick off the loop in a background thread; returns immediately.
    Idempotent: a second call while one is already running is a no-op that
    reports the in-progress status instead of starting an overlapping loop.
    """
    with _lock:
        if _state["running"]:
            return {"status": "already_running", **_state}
    thread = threading.Thread(target=_run_loop, args=(pipeline, n_clinicians), daemon=True)
    thread.start()
    return {"status": "started"}