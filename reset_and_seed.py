"""
Seed the live demo database with patients.DEMO_PATIENTS, POSTed through a
*running* API exactly the way the "Add patient" page does it -- optionally
preceded by a full reset, so a demo/recording always starts from the same
clean state.

Usage:
    uvicorn api:app --reload --port 8000     # terminal 1, must already be running
    python3 reset_and_seed.py                # terminal 2 -- reset + reseed + randomized arrivals (default)
    python3 reset_and_seed.py --seed-only     # just POST the 20 demo patients, no reset, no arrival randomization

Default mode (no flags) -- for recording a demo video (SETUP.md §8):
    1. db.reset_demo_db() -- clears `patients`, `overrides`, and `audit_log`,
       and resets the surge watermark. Does NOT touch outputs/audit_log.jsonl
       (the separate flat-file log); delete that yourself first if you want a
       fully blank audit history too.
    2. Re-submits all 20 patients.DEMO_PATIENTS through the real running API
       (POST /triage), so audit entries and confidence_* fields are populated
       exactly as they would be from the UI.
    3. Randomizes arrival times (fix_demo_arrivals.BREACH_FRACTION) for a
       realistic breached/not-breached mix.

--seed-only mode -- for adding demo patients to an existing queue without
touching what's already there: just step 2 above, nothing wiped or reset.

If TRIAGE_API_KEYS is set (see .env.example / SETUP.md §6.1a) -- the same
variable the server itself reads -- this script automatically picks a key
from it with at least the "nurse" role, via auth.get_keys(). There's no
separate seeding key to configure.

Safe to re-run: each submission is a fresh row (patient_id repeats just
supersede the previous row for that same patient, same as a reassessment).
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

import auth
import db
import fix_demo_arrivals
from patients import DEMO_PATIENTS

API_BASE = "http://localhost:8000"

KNOWN_TOP_LEVEL = {
    "patient_id", "age", "age_group", "hr", "sbp", "rr", "temp", "spo2",
    "mental_status_altered", "pregnancy", "has_prior_history",
}


def _pick_seed_api_key():
    """Pick a key from TRIAGE_API_KEYS (via auth.get_keys(), so the parsing
    logic lives in one place) with at least the "nurse" role. Returns None
    (no header sent) if no keys are configured, which is correct when the
    server has auth off."""
    keys = auth.get_keys()
    for key, info in keys.items():
        if info["role"] == "nurse":
            return key
    for key, info in keys.items():
        if info["role"] in ("clinician", "admin"):
            return key
    return None


API_KEY = _pick_seed_api_key()


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
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    req = urllib.request.Request(
        f"{API_BASE}/triage",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def seed_demo_patients():
    """POST all 20 DEMO_PATIENTS to the live API. Returns (ok_count, failed_ids)."""
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
    return ok, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-only", action="store_true",
        help="just POST the demo patients -- skip the db reset and arrival-time randomization",
    )
    args = parser.parse_args()

    if not args.seed_only:
        print("Resetting live database (patients, overrides, audit_log, surge watermark)...")
        db.reset_demo_db()

    print(f"\nSeeding {len(DEMO_PATIENTS)} demo patients through the live API...")
    ok, failed = seed_demo_patients()
    print(f"\n{ok}/{len(DEMO_PATIENTS)} submitted.")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)

    if not args.seed_only:
        print("\nRandomizing arrival times for a realistic breached/not-breached mix...")
        fix_demo_arrivals.main()

    print("\nDone. Open the React app (cd frontend && npm run dev) to see the seeded queue.")


if __name__ == "__main__":
    main()