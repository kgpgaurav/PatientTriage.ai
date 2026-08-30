"""
Rewrites arrival_time for every patient currently in triage.db so the queue has
a realistic mix of breached and not-yet-breached patients, instead of everyone
either being brand new (0 breached) or, after enough wall-clock time has passed
since seeding, everyone breached (what you're seeing now).

Safe to re-run any time you want to reset the demo to a fresh, mixed state.

Usage:
    python3 fix_demo_arrivals.py
"""
import random
from datetime import datetime, timedelta, timezone

import db

BREACH_FRACTION = 0.35  # ~35% of patients will show as breached after this runs


def main():
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id, patient_id, final_recommended_band, clinician_decision_band FROM patients"
    ).fetchall()
    now = datetime.now(timezone.utc)

    for row in rows:
        band = row["clinician_decision_band"] or row["final_recommended_band"] or 3
        ceiling = db.SAFE_WAIT_MINUTES.get(band, 30)

        if random.random() < BREACH_FRACTION:
            waited = ceiling * random.uniform(1.1, 2.0)   # past the ceiling -> breached
        else:
            waited = ceiling * random.uniform(0.05, 0.9)  # under the ceiling -> not breached

        arrival = now - timedelta(minutes=waited)
        conn.execute("UPDATE patients SET arrival_time = ? WHERE id = ?", (arrival.isoformat(), row["id"]))

    conn.commit()
    conn.close()
    print(f"Updated arrival_time for {len(rows)} rows (~{int(BREACH_FRACTION*100)}% now breached).")


if __name__ == "__main__":
    main()