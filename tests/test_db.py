import importlib
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import db as db_module


@pytest.fixture()
def temp_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    monkeypatch.setattr(db_module, "DB_PATH", path)
    db_module.init_db()
    yield db_module
    if os.path.exists(path):
        os.remove(path)


def _fake_result(band=3, prob=0.2):
    return {
        "critical_probability": prob,
        "severity_score": 40.0,
        "input_completeness": "HIGH",
        "model_recommended_band": band,
        "final_recommended_band": band,
        "safety_gate_triggers": [],
        "safety_gate_reason": "No safety-gate escalation triggered.",
        "model_explanation": [{"feature": "spo2_abs_deficit", "contribution": 0.1}],
        "extraction_status": "no_note",
        "extraction_backend": None,
        "model_status": "ok",
        "feature_snapshot": {"age": 40},
    }


def test_insert_and_queue(temp_db):
    row_id, arrival = temp_db.insert_triage_record("Q1", {"patient_id": "Q1", "age": 40}, _fake_result(band=2))
    assert row_id == 1
    queue = temp_db.get_queue()
    assert len(queue) == 1
    assert queue[0]["patient_id"] == "Q1"
    assert queue[0]["final_recommended_band"] == 2
    assert queue[0]["input_completeness"] == "HIGH"


def test_reassessment_supersedes_previous_row_and_keeps_arrival_time(temp_db):
    row1, arrival1 = temp_db.insert_triage_record("Q2", {"patient_id": "Q2", "age": 40, "hr": 90}, _fake_result())
    row2, arrival2 = temp_db.insert_triage_record("Q2", {"patient_id": "Q2", "age": 40, "hr": 150}, _fake_result(band=1))
    assert arrival2 == arrival1
    queue = temp_db.get_queue()
    assert len(queue) == 1
    assert queue[0]["final_recommended_band"] == 1


def test_get_patient_history_returns_prior_vitals_oldest_first(temp_db):
    temp_db.insert_triage_record("Q3", {"patient_id": "Q3", "age": 50, "hr": 80, "sbp": 120, "rr": 16, "temp": 37.0, "spo2": 98}, _fake_result())
    temp_db.insert_triage_record("Q3", {"patient_id": "Q3", "age": 50, "hr": 100, "sbp": 110, "rr": 20, "temp": 37.5, "spo2": 94}, _fake_result())
    history, last_at = temp_db.get_patient_history("Q3", limit=3)
    assert len(history) == 2
    assert history[0]["hr"] == 80
    assert history[1]["hr"] == 100
    assert last_at is not None


def test_multiple_patients_isolated(temp_db):
    temp_db.insert_triage_record("A1", {"patient_id": "A1", "age": 30}, _fake_result(band=4))
    temp_db.insert_triage_record("A2", {"patient_id": "A2", "age": 60}, _fake_result(band=1))
    queue = temp_db.get_queue()
    bands = {e["patient_id"]: e["final_recommended_band"] for e in queue}
    assert bands == {"A1": 4, "A2": 1}


def test_queue_ordered_by_band_then_arrival(temp_db):
    temp_db.insert_triage_record("B1", {"patient_id": "B1", "age": 30}, _fake_result(band=5))
    temp_db.insert_triage_record("B2", {"patient_id": "B2", "age": 30}, _fake_result(band=1))
    temp_db.insert_triage_record("B3", {"patient_id": "B3", "age": 30}, _fake_result(band=3))
    queue = temp_db.get_queue()
    assert [e["patient_id"] for e in queue] == ["B2", "B3", "B1"]


def test_apply_override_updates_latest_row_and_logs(temp_db):
    row_id, _ = temp_db.insert_triage_record("C1", {"patient_id": "C1", "age": 30}, _fake_result(band=3))
    temp_db.apply_override("C1", ai_band=3, clinician_band=4, reason_code="stable on repeat exam")
    detail = temp_db.get_patient_detail("C1")
    assert detail["clinician_decision_band"] == 4
    assert detail["override_reason"] == "stable on repeat exam"
    assert detail["is_downgrade"] == 1


def test_audit_log_records_events_in_order(temp_db):
    temp_db.insert_audit("triage_submitted", "D1", {"final_recommended_band": 2})
    temp_db.insert_audit("clinician_override", "D1", {"clinician_decision_band": 3})
    tail = temp_db.get_audit_tail(10)
    assert len(tail) == 2
    assert tail[0]["event_type"] == "triage_submitted"
    assert tail[1]["event_type"] == "clinician_override"


def test_get_patient_detail_missing_returns_none(temp_db):
    assert temp_db.get_patient_detail("does-not-exist") is None


def test_seed_demo_patients_if_empty_uses_triage_pipeline(temp_db, monkeypatch):
    class DummyPipeline:
        def run(self, record, history=None, note=None, log_audit=True):
            return {
                "critical_probability": 0.55,
                "severity_score": 35.0,
                "input_completeness": "HIGH",
                "model_recommended_band": 3,
                "final_recommended_band": 3,
                "safety_gate_triggers": [],
                "safety_gate_reason": "No safety-gate escalation triggered.",
                "model_explanation": {"feature": "age", "contribution": 0.1},
                "extraction_status": "no_note",
                "extraction_backend": None,
                "model_status": "ok",
                "age_group": "adult",
                "age_group_overridden": False,
                "feature_snapshot": {"age": record.get("age")},
            }

    fake_pipeline = DummyPipeline()
    monkeypatch.setattr(
        "patients.DEMO_PATIENTS",
        [{"patient_id": "D1", "age": 40, "hr": 90, "sbp": 120, "rr": 18, "temp": 37.0, "spo2": 98}],
    )

    temp_db.seed_demo_patients_if_empty(fake_pipeline)

    queue = temp_db.get_queue()
    assert len(queue) == 1
    assert queue[0]["patient_id"] == "D1"
    assert queue[0]["final_recommended_band"] == 3


def test_breach_flagging_uses_real_elapsed_time(temp_db, monkeypatch):
    from datetime import datetime, timedelta, timezone
    row_id, arrival = temp_db.insert_triage_record("E1", {"patient_id": "E1", "age": 30}, _fake_result(band=1))
    future = datetime.fromisoformat(arrival) + timedelta(minutes=10)
    queue = temp_db.get_queue(now=future)
    assert queue[0]["breached"] is True
    assert queue[0]["ceiling_min"] == 5


def test_breach_flags_reassessment_required_and_audits_it(temp_db):
    from datetime import datetime, timedelta

    row_id, arrival = temp_db.insert_triage_record("F1", {"patient_id": "F1", "age": 30}, _fake_result(band=1))
    future = datetime.fromisoformat(arrival) + timedelta(minutes=10)

    queue = temp_db.get_queue(now=future)
    assert queue[0]["reassessment_required"] is True
    assert queue[0]["reassessment_required_at"] is not None

    tail = temp_db.get_audit_tail(10)
    breach_events = [e for e in tail if e["event_type"] == "wait_breach_detected"]
    assert len(breach_events) == 1
    assert breach_events[0]["patient_id"] == "F1"


def test_breach_flag_is_not_repeated_on_subsequent_polls(temp_db):
    from datetime import datetime, timedelta

    row_id, arrival = temp_db.insert_triage_record("F2", {"patient_id": "F2", "age": 30}, _fake_result(band=1))
    t1 = datetime.fromisoformat(arrival) + timedelta(minutes=10)
    t2 = datetime.fromisoformat(arrival) + timedelta(minutes=20)

    temp_db.get_queue(now=t1)
    temp_db.get_queue(now=t2)

    tail = temp_db.get_audit_tail(10)
    breach_events = [e for e in tail if e["event_type"] == "wait_breach_detected"]
    assert len(breach_events) == 1


def test_reassessment_clears_flag_and_audits_reassessment_performed(temp_db):
    from datetime import datetime, timedelta

    row_id, arrival = temp_db.insert_triage_record("F3", {"patient_id": "F3", "age": 30}, _fake_result(band=1))
    future = datetime.fromisoformat(arrival) + timedelta(minutes=10)
    temp_db.get_queue(now=future)  # triggers the breach flag

    queue = temp_db.get_queue(now=future)
    assert queue[0]["reassessment_required"] is True

    temp_db.insert_triage_record("F3", {"patient_id": "F3", "age": 30, "hr": 95}, _fake_result(band=2), arrival_time=None)

    queue_after = temp_db.get_queue(now=future)
    assert queue_after[0]["reassessment_required"] is False
    assert queue_after[0]["arrival_time"] == arrival  # original arrival time preserved

    tail = temp_db.get_audit_tail(10)
    reassessed_events = [e for e in tail if e["event_type"] == "reassessment_performed"]
    assert len(reassessed_events) == 1
    assert reassessed_events[0]["patient_id"] == "F3"


def test_resolved_patient_is_not_flagged_for_reassessment(temp_db):
    from datetime import datetime, timedelta

    row_id, arrival = temp_db.insert_triage_record("F4", {"patient_id": "F4", "age": 30}, _fake_result(band=1))
    temp_db.set_disposition("F4", "admitted")
    future = datetime.fromisoformat(arrival) + timedelta(minutes=10)

    queue = temp_db.get_queue(now=future)
    assert queue[0]["reassessment_required"] is False


def test_migration_renames_legacy_data_quality_column():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'waiting',
            arrival_time TEXT NOT NULL,
            created_at TEXT NOT NULL,
            input_json TEXT NOT NULL,
            data_quality TEXT
        )
    """)
    conn.execute(
        "INSERT INTO patients (patient_id, status, arrival_time, created_at, input_json, data_quality) "
        "VALUES ('LEGACY', 'waiting', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', '{}', 'HIGH')"
    )
    conn.commit()
    conn.close()

    try:
        importlib.reload(db_module)
        db_module.DB_PATH = path
        db_module.init_db()
        conn = db_module.get_conn()
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(patients)")}
        row = conn.execute("SELECT input_completeness FROM patients WHERE patient_id='LEGACY'").fetchone()
        conn.close()
        assert "data_quality" not in cols
        assert "input_completeness" in cols
        assert row["input_completeness"] == "HIGH"
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_next_patient_id_starts_at_base_on_empty_db(temp_db):
    assert temp_db.next_patient_id() == "P-1001"


def test_next_patient_id_increments_past_existing(temp_db):
    temp_db.insert_triage_record("P-1001", {"patient_id": "P-1001", "age": 40}, _fake_result())
    temp_db.insert_triage_record("P-1005", {"patient_id": "P-1005", "age": 40}, _fake_result())
    assert temp_db.next_patient_id() == "P-1006"


def test_next_patient_id_ignores_non_matching_ids(temp_db):
    # The fixed demo set (P01..P20) and any manually-typed ID (e.g. a real
    # MRN) shouldn't affect or be affected by the auto-sequence.
    temp_db.insert_triage_record("P18", {"patient_id": "P18", "age": 40}, _fake_result())
    temp_db.insert_triage_record("MRN-9981", {"patient_id": "MRN-9981", "age": 40}, _fake_result())
    assert temp_db.next_patient_id() == "P-1001"


def test_next_patient_id_is_a_suggestion_not_a_reservation(temp_db):
    # Calling it twice without an insert in between returns the same value --
    # nothing is written or reserved by the call itself.
    first = temp_db.next_patient_id()
    second = temp_db.next_patient_id()
    assert first == second == "P-1001"


def test_reset_demo_db_restarts_the_auto_id_sequence(temp_db):
    temp_db.insert_triage_record("P-1001", {"patient_id": "P-1001", "age": 40}, _fake_result())
    assert temp_db.next_patient_id() == "P-1002"
    temp_db.reset_demo_db()
    assert temp_db.next_patient_id() == "P-1001"


def test_get_queue_includes_demographics_for_reassess_prefill(temp_db):
    temp_db.insert_triage_record(
        "R1",
        {
            "patient_id": "R1", "age": 45, "age_months": None, "gender": "female",
            "has_prior_history": False, "pregnancy": True, "hr": 90,
        },
        _fake_result(),
    )
    entry = temp_db.get_queue()[0]
    assert entry["gender"] == "female"
    assert entry["has_prior_history"] is False
    assert entry["pregnancy"] is True
    assert entry["age_months"] is None