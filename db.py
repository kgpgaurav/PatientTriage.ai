import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "outputs", "triage.db")

SAFE_WAIT_MINUTES = {1: 5, 2: 10, 3: 30, 4: 60, 5: 120}

# Auto-generated live-intake patient IDs look like "P-1001", "P-1002", ...
# Deliberately a different shape from the fixed demo set (patients.py uses
# "P01".."P20", no dash) so the two ID spaces never collide or interact --
# next_patient_id() below only ever looks at IDs matching _AUTO_ID_PATTERN.
_AUTO_ID_PREFIX = "P-"
_AUTO_ID_START = 1001
_AUTO_ID_PATTERN = re.compile(r"^P-(\d+)$")

# ED disposition — where the patient is in the encounter. `waiting` is the
# default; a clinician moves them forward from the dashboard. `superseded`
# is set automatically (not clinician-chosen) when a reassessment reading
# replaces an older "waiting" row for the same patient_id.
ED_DISPOSITIONS = ("waiting", "in_treatment", "admitted", "treatment_successful", "discharged")
RESOLVED_DISPOSITIONS = ("admitted", "treatment_successful", "discharged")

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'waiting',      -- waiting | superseded | discharged
    arrival_time TEXT NOT NULL,                  -- ISO timestamp, carried forward across reassessments
    created_at TEXT NOT NULL,
    input_json TEXT NOT NULL,
    feature_snapshot_json TEXT,
    critical_probability REAL,
    severity_score REAL,
    input_completeness TEXT,
    model_recommended_band INTEGER,
    final_recommended_band INTEGER,
    safety_gate_triggers_json TEXT,
    safety_gate_reason TEXT,
    model_explanation_json TEXT,
    extraction_status TEXT,
    extraction_backend TEXT,
    model_status TEXT,
    clinician_decision_band INTEGER,
    override_reason TEXT,
    is_downgrade INTEGER,
    status_updated_at TEXT,                      -- when `status` (disposition) last changed
    reassessment_required INTEGER NOT NULL DEFAULT 0,
    reassessment_required_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_patients_status ON patients(status);

CREATE TABLE IF NOT EXISTS overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_row_id INTEGER,
    patient_id TEXT NOT NULL,
    ai_recommendation_band INTEGER,
    clinician_decision_band INTEGER,
    override_reason TEXT,
    is_downgrade INTEGER,
    decided_by_role TEXT,
    decided_by TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    patient_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Single-row watermark for the live operational (surge) monitor: tracks the
-- last classified state and the running max queue length so the API can
-- report a "max observed" figure and emit surge_state_changed /
-- queue_threshold_reached audit events on transitions. Purely operational
-- bookkeeping -- never read by the clinical pipeline.
CREATE TABLE IF NOT EXISTS surge_watermark (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_state TEXT,
    max_queue_length INTEGER NOT NULL DEFAULT 0,
    over_threshold INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT
);
"""

# Prototype operational threshold: a live queue deeper than this many
# multiples of the clinician count is flagged as a queue_threshold_reached
# audit event. Not a clinical or staffing standard.
QUEUE_THRESHOLD_MULTIPLIER = 3


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn, table, column, coltype):
    """Add `column` to `table` if it doesn't already exist. Safe to call every startup."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def _rename_column_if_present(conn, table, old, new, coltype):
    """One-time migration for a column rename. If `old` exists and `new` doesn't,
    rename in place (preserves existing row data). Otherwise falls back to
    ensuring `new` exists, so this is safe to call on every startup."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if old in existing and new not in existing:
        conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
    else:
        _ensure_column(conn, table, new, coltype)


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    _rename_column_if_present(conn, "patients", "data_quality", "input_completeness", "TEXT")
    _ensure_column(conn, "patients", "status_updated_at", "TEXT")
    _ensure_column(conn, "patients", "reassessment_required", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "patients", "reassessment_required_at", "TEXT")
    _ensure_column(conn, "patients", "confidence_score", "REAL")
    _ensure_column(conn, "patients", "confidence_level", "TEXT")
    _ensure_column(conn, "patients", "confidence_reason", "TEXT")
    _ensure_column(conn, "overrides", "decided_by_role", "TEXT")
    _ensure_column(conn, "overrides", "decided_by", "TEXT")
    conn.execute(
        "INSERT OR IGNORE INTO surge_watermark (id, last_state, max_queue_length, over_threshold) "
        "VALUES (1, NULL, 0, 0)"
    )
    conn.commit()
    conn.close()


def reset_demo_db():
    """Wipe all live-path data back to empty: `patients`, `overrides`,
    `audit_log`, and the surge watermark. Used by `reset_and_seed.py` so a
    demo/recording always starts from the same clean state, on demand,
    without deleting the `triage.db` file itself (so a running `uvicorn`
    process doesn't need to be restarted).

    Does NOT touch `outputs/audit_log.jsonl` -- the separate flat-file audit
    log written by `pipeline.run`/`audit.write_record` -- since that log is
    meant to be an append-only record independent of what's currently in the
    live queue. Delete that file yourself if you want a fully blank slate.

    Also resets next_patient_id()'s counter back to _AUTO_ID_START, since it
    derives the next number from whatever's currently in `patients` -- an
    empty table means the next demo take gets P-1001 again, not wherever the
    previous take left off.
    """
    conn = get_conn()
    conn.execute("DELETE FROM patients")
    conn.execute("DELETE FROM overrides")
    conn.execute("DELETE FROM audit_log")
    conn.execute(
        "UPDATE surge_watermark SET last_state = NULL, max_queue_length = 0, over_threshold = 0 WHERE id = 1"
    )
    conn.commit()
    conn.close()


def next_patient_id():
    """Suggest the next sequential live-intake patient ID (P-1001, P-1002,
    ...) for the "Add patient" form to pre-fill. This is a suggestion, not a
    reservation -- nothing is written here, and the ID only actually exists
    once a `/triage` submission uses it. The nurse can still type over it
    (e.g. to reassess an existing patient by entering that patient's ID
    instead), so there's no lock/reservation step to worry about either.

    Only considers IDs already in the P-<digits> shape; the fixed demo set
    (patients.py's P01..P20) doesn't match and is never touched or affected.
    """
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT patient_id FROM patients").fetchall()
    conn.close()
    max_n = _AUTO_ID_START - 1
    for row in rows:
        m = _AUTO_ID_PATTERN.match(row["patient_id"])
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"{_AUTO_ID_PREFIX}{max_n + 1}"


def seed_demo_patients_if_empty(pipeline=None):
    """Populate a brand-new database with the demo patient set once.

    This is intentionally guarded so that regular live submissions are never
    overwritten or replaced by the static demo dataset.
    """
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    conn.close()
    if count > 0:
        return []

    if pipeline is None:
        try:
            from pipeline import TriagePipeline
        except Exception:
            return []
        else:
            return []

    try:
        from patients import DEMO_PATIENTS
    except Exception:
        return []

    inserted = []
    for patient in DEMO_PATIENTS:
        note = patient.get("note")
        history = patient.get("history_readings")
        record = {k: v for k, v in patient.items() if k not in ("note", "history_readings")}
        result = pipeline.run(record, history=history, note=note, log_audit=False)
        record["age_group"] = result.get("age_group", record.get("age_group"))
        record["observed_reported_mismatch"] = result.get("observed_reported_mismatch", False)
        record_for_storage = {**record, "note": note}
        row_id, _ = insert_triage_record(
            patient["patient_id"],
            record_for_storage,
            result,
            arrival_time=datetime.now(timezone.utc).isoformat(),
        )
        inserted.append({"row_id": row_id, "patient_id": patient["patient_id"], "band": result.get("final_recommended_band")})

    return inserted


def get_active_patient_row(patient_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM patients WHERE patient_id = ? AND status = 'waiting' "
        "ORDER BY created_at DESC LIMIT 1",
        (patient_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def set_disposition(patient_id, disposition, note=None, decided_by_role=None, decided_by=None):
    """Move a patient's ED disposition forward (waiting -> in_treatment -> admitted/
    treatment_successful/discharged). Freezes that patient's wait clock at the moment
    of the change (see get_queue) and writes an audit_log entry so it shows up in history.

    `decided_by_role`/`decided_by` identify the caller who made the change (mirrors
    the pattern already used for clinician_override in api.py's /override handler)
    so every disposition change -- admit, discharge, move to in_treatment, etc. --
    is attributable to a specific person, not just "someone with clinician rights".
    """
    if disposition not in ED_DISPOSITIONS:
        raise ValueError(f"Unknown disposition '{disposition}'. Must be one of {ED_DISPOSITIONS}.")

    conn = get_conn()
    row = conn.execute(
        "SELECT id, status FROM patients WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
        (patient_id,),
    ).fetchone()
    if not row:
        conn.close()
        raise ValueError(f"No patient found with patient_id '{patient_id}'.")

    previous_status = row["status"]
    now = _now()
    conn.execute(
        "UPDATE patients SET status = ?, status_updated_at = ? WHERE id = ?",
        (disposition, now, row["id"]),
    )
    conn.commit()
    conn.close()

    insert_audit("disposition_change", patient_id, {
        "previous_status": previous_status,
        "new_status": disposition,
        "note": note,
        "decided_by_role": decided_by_role,
        "decided_by": decided_by,
    })
    return {"patient_id": patient_id, "status": disposition, "status_updated_at": now}


def get_patient_timeline(patient_id):
    """Chronological history for one patient: every vitals reading (from
    `patients` -- one row per original submission or reassessment, including
    superseded ones), every clinician band decision (from `overrides`), and
    every disposition change / wait-breach / reassessment event (from
    `audit_log`), oldest first. Powers the per-patient history view in the
    dashboard, including the "previous readings" table shown on reassessment.
    """
    conn = get_conn()
    reading_rows = conn.execute(
        "SELECT input_json, final_recommended_band, model_recommended_band, "
        "critical_probability, created_at FROM patients WHERE patient_id = ? "
        "ORDER BY created_at",
        (patient_id,),
    ).fetchall()
    override_rows = conn.execute(
        "SELECT ai_recommendation_band, clinician_decision_band, override_reason, "
        "is_downgrade, decided_by_role, decided_by, created_at FROM overrides "
        "WHERE patient_id = ? ORDER BY created_at",
        (patient_id,),
    ).fetchall()
    audit_rows = conn.execute(
        "SELECT event_type, payload_json, created_at FROM audit_log "
        "WHERE patient_id = ? AND event_type IN "
        "('disposition_change', 'wait_breach_detected', 'reassessment_performed') "
        "ORDER BY created_at",
        (patient_id,),
    ).fetchall()
    conn.close()

    timeline = []
    for row in reading_rows:
        rec = json.loads(row["input_json"])
        timeline.append({
            "type": "vitals_reading",
            "created_at": row["created_at"],
            "hr": rec.get("hr"),
            "sbp": rec.get("sbp"),
            "rr": rec.get("rr"),
            "temp": rec.get("temp"),
            "spo2": rec.get("spo2"),
            "mental_status_altered": rec.get("mental_status_altered"),
            "final_recommended_band": row["final_recommended_band"],
            "model_recommended_band": row["model_recommended_band"],
            "critical_probability": row["critical_probability"],
        })
    for row in override_rows:
        timeline.append({
            "type": "band_decision",
            "created_at": row["created_at"],
            "ai_recommendation_band": row["ai_recommendation_band"],
            "clinician_decision_band": row["clinician_decision_band"],
            "override_reason": row["override_reason"],
            "is_downgrade": bool(row["is_downgrade"]),
            "decided_by_role": row["decided_by_role"],
            "decided_by": row["decided_by"],
        })
    for row in audit_rows:
        payload = json.loads(row["payload_json"])
        if row["event_type"] == "disposition_change":
            timeline.append({
                "type": "disposition_change",
                "created_at": row["created_at"],
                "previous_status": payload.get("previous_status"),
                "new_status": payload.get("new_status"),
                "note": payload.get("note"),
                "decided_by_role": payload.get("decided_by_role"),
                "decided_by": payload.get("decided_by"),
            })
        else:
            timeline.append({
                "type": row["event_type"],
                "created_at": row["created_at"],
                **payload,
            })

    timeline.sort(key=lambda e: e["created_at"])
    return timeline


def get_patient_history(patient_id, limit=3):
    """Prior vitals for this patient, oldest first, for temporal/deterioration features."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT input_json, created_at FROM patients WHERE patient_id = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (patient_id, limit),
    ).fetchall()
    conn.close()
    history = []
    for row in reversed(rows):
        rec = json.loads(row["input_json"])
        history.append({k: rec.get(k) for k in ("hr", "sbp", "rr", "temp", "spo2")})
    return history, (rows[0]["created_at"] if rows else None)


def insert_triage_record(patient_id, input_record, result, arrival_time=None,
                          decided_by_role=None, decided_by=None):
    conn = get_conn()
    existing = get_active_patient_row(patient_id)
    was_reassessment_required = False
    if existing:
        conn.execute("UPDATE patients SET status = 'superseded' WHERE id = ?", (existing["id"],))
        arrival_time = existing["arrival_time"]
        was_reassessment_required = bool(existing.get("reassessment_required"))
    if arrival_time is None:
        arrival_time = _now()

    cur = conn.execute(
        """INSERT INTO patients (
            patient_id, status, arrival_time, created_at, input_json, feature_snapshot_json,
            critical_probability, severity_score, input_completeness, model_recommended_band,
            final_recommended_band, safety_gate_triggers_json, safety_gate_reason,
            model_explanation_json, extraction_status, extraction_backend, model_status,
            reassessment_required, reassessment_required_at,
            confidence_score, confidence_level, confidence_reason
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            patient_id, "waiting", arrival_time, _now(), json.dumps(input_record),
            json.dumps(result.get("feature_snapshot")) if result.get("feature_snapshot") else None,
            result.get("critical_probability"), result.get("severity_score"), result.get("input_completeness"),
            result.get("model_recommended_band"), result.get("final_recommended_band"),
            json.dumps(result.get("safety_gate_triggers", [])), result.get("safety_gate_reason"),
            json.dumps(result.get("model_explanation")) if result.get("model_explanation") else None,
            result.get("extraction_status"), result.get("extraction_backend"), result.get("model_status"),
            0, None,
            result.get("confidence_score"), result.get("confidence_level"), result.get("confidence_reason"),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()

    if was_reassessment_required:
        insert_audit("reassessment_performed", patient_id, {
            "previous_row_id": existing["id"],
            "new_row_id": row_id,
            "arrival_time": arrival_time,
            "decided_by_role": decided_by_role,
            "decided_by": decided_by,
        })

    return row_id, arrival_time


def apply_override(patient_id, ai_band, clinician_band, reason_code, decided_by_role=None, decided_by=None):
    is_downgrade = clinician_band > ai_band
    conn = get_conn()
    row = conn.execute(
        "SELECT id FROM patients WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
        (patient_id,),
    ).fetchone()
    row_id = row["id"] if row else None
    if row_id:
        conn.execute(
            "UPDATE patients SET clinician_decision_band = ?, override_reason = ?, is_downgrade = ? WHERE id = ?",
            (clinician_band, reason_code, int(is_downgrade), row_id),
        )
    conn.execute(
        """INSERT INTO overrides (patient_row_id, patient_id, ai_recommendation_band,
           clinician_decision_band, override_reason, is_downgrade, decided_by_role, decided_by, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (row_id, patient_id, ai_band, clinician_band, reason_code, int(is_downgrade),
         decided_by_role, decided_by, _now()),
    )
    conn.commit()
    conn.close()


def insert_audit(event_type, patient_id, payload):
    conn = get_conn()
    conn.execute(
        "INSERT INTO audit_log (event_type, patient_id, payload_json, created_at) VALUES (?,?,?,?)",
        (event_type, patient_id, json.dumps(payload, default=str), _now()),
    )
    conn.commit()
    conn.close()


def get_audit_tail(n=20):
    conn = get_conn()
    rows = conn.execute(
        "SELECT event_type, patient_id, payload_json, created_at FROM audit_log "
        "ORDER BY id DESC LIMIT ?", (n,),
    ).fetchall()
    conn.close()
    out = []
    for row in reversed(rows):
        out.append({
            "event_type": row["event_type"],
            "patient_id": row["patient_id"],
            "created_at": row["created_at"],
            **json.loads(row["payload_json"]),
        })
    return out


def _flag_new_breaches(conn, rows, now):
    """For every currently-`waiting` row whose wait has just exceeded its
    ceiling and isn't already flagged, persist REASSESSMENT_REQUIRED and audit
    the breach. Idempotent: a row already flagged is left alone (its
    reassessment_required_at timestamp is the original detection time, not
    refreshed on every poll). Returns {row_id: flagged_at_iso} for rows
    flagged in this call, so the caller can reflect it in the same response
    instead of waiting for the next poll."""
    newly_flagged = {}
    for row in rows:
        if row["status"] != "waiting" or row["reassessment_required"]:
            continue
        ceiling = SAFE_WAIT_MINUTES.get(
            row["clinician_decision_band"] if row["clinician_decision_band"] is not None else (row["final_recommended_band"] or 3),
            30,
        )
        arrival = datetime.fromisoformat(row["arrival_time"])
        waited_min = (now - arrival).total_seconds() / 60.0
        if waited_min > ceiling:
            newly_flagged[row["id"]] = (row, waited_min, ceiling)

    if not newly_flagged:
        return {}

    flagged_at = now.isoformat()
    for row_id in newly_flagged:
        conn.execute(
            "UPDATE patients SET reassessment_required = 1, reassessment_required_at = ? WHERE id = ?",
            (flagged_at, row_id),
        )
    conn.commit()

    for row_id, (row, waited_min, ceiling) in newly_flagged.items():
        payload = {
            "row_id": row_id,
            "waited_min": round(waited_min, 1),
            "ceiling_min": ceiling,
        }
        # Two events, same underlying moment: WAIT_BREACH_DETECTED is the
        # trigger (safe-wait ceiling crossed); REASSESSMENT_REQUIRED is the
        # resulting operational requirement it creates. Logged separately so
        # the audit trail reads the way the Round 2 terminology expects,
        # without ever implying the patient's clinical band changed.
        insert_audit("wait_breach_detected", row["patient_id"], payload)
        insert_audit("reassessment_required", row["patient_id"], payload)

    return {row_id: flagged_at for row_id in newly_flagged}


def _percentile(values, pct):
    """Linear-interpolation percentile with no numpy dependency, since
    db.py otherwise has no numeric/array dependencies."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def get_live_operational_status(now=None, n_clinicians=4, baseline_rate=None, arrival_window_min=60):
    """Live, DB-backed operational (queue-management) snapshot -- entirely
    independent of clinical triage. Computes the real arrival rate from
    actual patient arrivals in the trailing window, the current queue
    length/composition from the live waiting queue, and classifies
    NORMAL / SURGE / CRISIS via `surge.determine_operational_state`.

    Also maintains a running max-queue-length high-water mark and emits
    `surge_state_changed` / `queue_threshold_reached` audit events on
    transitions. Never reads or writes a clinical band; `n_clinicians` and
    the queue-depth threshold are prototype operational assumptions, not a
    real staffing model.
    """
    from surge import BASELINE_ARRIVALS_PER_HOUR, determine_operational_state

    now = now or datetime.now(timezone.utc)
    baseline_rate = baseline_rate or BASELINE_ARRIVALS_PER_HOUR
    window_start = (now - timedelta(minutes=arrival_window_min)).isoformat()

    conn = get_conn()
    arrivals_in_window = conn.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM patients WHERE arrival_time >= ?",
        (window_start,),
    ).fetchone()[0]
    placeholders = ",".join("?" * len(RESOLVED_DISPOSITIONS))
    resolved_in_window = conn.execute(
        f"SELECT COUNT(*) FROM patients WHERE status IN ({placeholders}) AND status_updated_at >= ?",
        (*RESOLVED_DISPOSITIONS, window_start),
    ).fetchone()[0]
    conn.close()

    arrival_rate = arrivals_in_window / (arrival_window_min / 60.0)
    throughput_per_hour = resolved_in_window / (arrival_window_min / 60.0)

    queue = get_queue(now=now)
    waiting = [e for e in queue if e["status"] == "waiting"]
    queue_length = len(waiting)
    waits = [e["waited_min"] for e in waiting]
    avg_wait = round(sum(waits) / len(waits), 1) if waits else 0.0
    p95_wait = round(_percentile(waits, 95), 1) if waits else 0.0
    safe_wait_breaches = sum(1 for e in waiting if e["reassessment_required"])

    state = determine_operational_state(
        arrival_rate=arrival_rate, baseline_rate=baseline_rate,
        queue_length=queue_length, n_clinicians=n_clinicians,
    )

    conn = get_conn()
    wm = conn.execute("SELECT * FROM surge_watermark WHERE id = 1").fetchone()
    prev_state = wm["last_state"] if wm else None
    prev_over_threshold = bool(wm["over_threshold"]) if wm else False
    max_queue_length = max(wm["max_queue_length"], queue_length) if wm else queue_length
    threshold = QUEUE_THRESHOLD_MULTIPLIER * n_clinicians
    over_threshold_now = queue_length > threshold

    if prev_state and prev_state != state["state"]:
        insert_audit("surge_state_changed", None, {
            "previous_state": prev_state, "new_state": state["state"],
            "arrival_rate": state["arrival_rate"], "queue_length": queue_length,
        })
    if over_threshold_now and not prev_over_threshold:
        insert_audit("queue_threshold_reached", None, {
            "queue_length": queue_length, "threshold": threshold, "state": state["state"],
        })

    conn.execute(
        "UPDATE surge_watermark SET last_state = ?, max_queue_length = ?, over_threshold = ?, updated_at = ? WHERE id = 1",
        (state["state"], max_queue_length, int(over_threshold_now), now.isoformat()),
    )
    conn.commit()
    conn.close()

    return {
        "operational_state": state,
        "queue_length": queue_length,
        "max_queue_length": max_queue_length,
        "n_clinicians": n_clinicians,
        "patients_served_last_hour": round(throughput_per_hour, 1),
        "arrival_window_min": arrival_window_min,
        "avg_wait_min": avg_wait,
        "p95_wait_min": p95_wait,
        "safe_wait_breaches": safe_wait_breaches,
        "reassessments_pending": safe_wait_breaches,
        "is_prototype_assumption": True,
    }


def get_queue(now=None):
    now = now or datetime.now(timezone.utc)
    conn = get_conn()
    # Only the latest row per patient_id -- otherwise a reassessed patient's
    # older (superseded) reading stays in the list as a duplicate entry.
    rows = conn.execute(
        """
        SELECT p.*
        FROM patients p
        INNER JOIN (
            SELECT patient_id, MAX(created_at) AS max_created
            FROM patients
            GROUP BY patient_id
        ) latest
            ON p.patient_id = latest.patient_id AND p.created_at = latest.max_created
        WHERE p.final_recommended_band IS NOT NULL
        ORDER BY p.created_at DESC
        """
    ).fetchall()

    newly_flagged = _flag_new_breaches(conn, rows, now)
    conn.close()

    out = []
    for row in rows:
        ai_band = row["final_recommended_band"] or 3
        # The clinician's logged decision is what's actually governing this
        # patient's priority once it exists -- the AI band is decision support,
        # not the live fact. We never overwrite the AI's own recorded output;
        # we just compute a separate "what's actually in effect" band from it.
        effective_band = row["clinician_decision_band"] if row["clinician_decision_band"] is not None else ai_band
        ceiling = SAFE_WAIT_MINUTES.get(effective_band, 30)

        disposition = row["status"] if row["status"] in ED_DISPOSITIONS else "waiting"
        # Once a patient has left the waiting state (in treatment, admitted, etc.)
        # their safe-wait clock should stop climbing -- it should freeze at
        # "how long they actually waited", not keep counting forever.
        if disposition != "waiting" and row["status_updated_at"]:
            clock_end = datetime.fromisoformat(row["status_updated_at"])
        else:
            clock_end = now
        arrival = datetime.fromisoformat(row["arrival_time"])
        waited_min = (clock_end - arrival).total_seconds() / 60.0

        input_record = json.loads(row["input_json"]) if row["input_json"] else {}
        safety_gate_triggers = json.loads(row["safety_gate_triggers_json"] or "[]")
        model_explanation = json.loads(row["model_explanation_json"] or "null")

        out.append({
            "patient_id": row["patient_id"],
            # --- bands: AI's own recommendation (never mutated) vs. what's actually in effect ---
            "model_recommended_band": row["model_recommended_band"],
            "final_recommended_band": ai_band,
            "clinician_decision_band": row["clinician_decision_band"],
            "override_reason": row["override_reason"],
            "effective_band": effective_band,
            "escalated_by_gate": bool(
                row["model_recommended_band"] is not None and ai_band < row["model_recommended_band"]
            ),
            # --- ED disposition ---
            "status": disposition,
            "status_updated_at": row["status_updated_at"],
            "is_resolved": disposition in RESOLVED_DISPOSITIONS,
            # --- model output ---
            "critical_probability": row["critical_probability"],
            "severity_score": row["severity_score"],
            "confidence_score": row["confidence_score"],
            "confidence_level": row["confidence_level"] or "LOW",
            "confidence_reason": row["confidence_reason"],
            "input_completeness": row["input_completeness"],
            "model_explanation": model_explanation,
            "model_status": row["model_status"],
            # --- safety gate ---
            "safety_gate_reason": row["safety_gate_reason"],
            "safety_gate_triggers": safety_gate_triggers,
            # --- vitals / demographics, for the collapsed row's "Vitals" column
            # and for pre-filling the "Reassess" form (age, gender, prior-history,
            # pregnancy don't change between readings, so they're carried over) ---
            "age": input_record.get("age"),
            "age_months": input_record.get("age_months"),
            "age_group": input_record.get("age_group"),
            "gender": input_record.get("gender"),
            "has_prior_history": input_record.get("has_prior_history"),
            "pregnancy": input_record.get("pregnancy"),
            "hr": input_record.get("hr"),
            "sbp": input_record.get("sbp"),
            "rr": input_record.get("rr"),
            "temp": input_record.get("temp"),
            "spo2": input_record.get("spo2"),
            # --- free-text note + which extraction path produced it ---
            "note": input_record.get("note"),
            "observed_reported_mismatch": input_record.get("observed_reported_mismatch", False),
            "extraction_status": row["extraction_status"],
            "extraction_backend": row["extraction_backend"],
            # --- wait tracking ---
            "arrival_time": row["arrival_time"],
            "waited_min": round(waited_min, 1),
            "ceiling_min": ceiling,
            "breached": waited_min > ceiling,
            "reassessment_required": bool(row["reassessment_required"]) or row["id"] in newly_flagged,
            "reassessment_required_at": row["reassessment_required_at"] or newly_flagged.get(row["id"]),
        })

    # Queue order is priority order, not submission order: most urgent
    # effective_band first, then whoever's been waiting longest within the
    # same band. effective_band is the clinician's logged decision if one
    # exists, otherwise the AI's gate-escalated band -- i.e. whichever number
    # actually governs that patient's priority right now.
    out.sort(key=lambda e: (e["effective_band"], e["arrival_time"]))
    return out


def get_patient_detail(patient_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM patients WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
        (patient_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["input"] = json.loads(d.pop("input_json"))
    d["safety_gate_triggers"] = json.loads(d.pop("safety_gate_triggers_json") or "[]")
    d["model_explanation"] = json.loads(d.pop("model_explanation_json") or "null")
    d.pop("feature_snapshot_json", None)
    return d