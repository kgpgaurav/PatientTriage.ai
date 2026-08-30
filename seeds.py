"""
Feed every entry in patients.DEMO_PATIENTS through the live API (POST /triage),
exactly the way the "Add patient" page does it.

Usage:
    uvicorn api:app --reload --port 8000     # in one terminal
    python seed_from_patients.py             # in another

Safe to re-run: each submission is a fresh row (patient_id repeats just
supersede the previous row for that same patient, same as a reassessment).
"""
import json
import sys
import time
import urllib.error
import urllib.request

from patients import DEMO_PATIENTS

API_BASE = "http://localhost:8000"

KNOWN_TOP_LEVEL = {
    "patient_id", "age", "age_group", "hr", "sbp", "rr", "temp", "spo2",
    "mental_status_altered", "pregnancy", "has_prior_history",
}


def to_payload(patient: dict) -> dict:
    """Split a DEMO_PATIENTS dict into the /triage request shape:
    top-level vitals/demographics + a `symptoms` dict + `note`.
    (history_readings isn't a real API field -- the API derives history
    from prior DB rows for the same patient_id automatically.)
    """
    payload = {k: v for k, v in patient.items() if k in KNOWN_TOP_LEVEL}
    payload["note"] = patient.get("note") or None
    payload["symptoms"] = {
        k: v for k, v in patient.items()
        if k not in KNOWN_TOP_LEVEL and k not in ("note", "history_readings")
    }
    return payload


def post_triage(payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/triage",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ok, failed = 0, []
    for patient in DEMO_PATIENTS:
        payload = to_payload(patient)
        try:
            result = post_triage(payload)
            print(
                f"  {payload['patient_id']:6s} -> final band {result['final_recommended_band']} "
                f"(model {result['model_recommended_band']}, "
                f"P(crit)={result['critical_probability']:.2f}, "
                f"completeness={result['input_completeness']})"
            )
            ok += 1
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            print(f"  {payload['patient_id']:6s} -> FAILED ({e.code}): {detail}")
            failed.append(payload["patient_id"])
        except Exception as e:
            print(f"  {payload['patient_id']:6s} -> FAILED: {e}")
            failed.append(payload["patient_id"])
        time.sleep(0.05)  # stay comfortably clear of any reassessment-timing edge cases

    print(f"\n{ok}/{len(DEMO_PATIENTS)} submitted.")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()