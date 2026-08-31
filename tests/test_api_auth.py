"""End-to-end tests for the access-control layer as wired into api.py --
complements tests/test_auth.py (which tests auth.py in isolation) by
checking the actual HTTP endpoints: role gates, per-caller audit
attribution, the admin-only key reload, and the CORS origin allowlist.

Requires train.py to have been run first (api.py loads outputs/*.pkl at
import time), same as test_pipeline.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

import auth
import db as db_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "test_triage.db"))
    monkeypatch.delenv("TRIAGE_API_KEYS", raising=False)
    monkeypatch.delenv("TRIAGE_ALLOWED_ORIGINS", raising=False)
    auth.reload_keys()
    db_module.init_db()

    import api  # cached after the first test; DB_PATH/env are patched fresh per test
    yield TestClient(api.app)

    monkeypatch.delenv("TRIAGE_API_KEYS", raising=False)
    auth.reload_keys()


def _patient_payload(patient_id="P-AUTH-1"):
    return {"patient_id": patient_id, "age": 40, "hr": 80, "sbp": 120, "rr": 16, "temp": 37.0, "spo2": 98}


def test_open_mode_allows_triage_without_key(client):
    resp = client.post("/triage", json=_patient_payload())
    assert resp.status_code == 200
    assert "final_recommended_band" in resp.json()


def test_configured_mode_rejects_missing_key(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "nurse-key:nurse:A. Fisher")
    auth.reload_keys()
    resp = client.get("/queue")
    assert resp.status_code == 401


def test_configured_mode_rejects_insufficient_role(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "nurse-key:nurse:A. Fisher")
    auth.reload_keys()
    resp = client.get("/audit", headers={"X-API-Key": "nurse-key"})
    assert resp.status_code == 403


def test_configured_mode_grants_correct_role(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "admin-key:admin:M. Otieno")
    auth.reload_keys()
    resp = client.get("/audit", headers={"X-API-Key": "admin-key"})
    assert resp.status_code == 200


def test_audit_trail_records_caller_name_not_just_role(client, monkeypatch):
    monkeypatch.setenv(
        "TRIAGE_API_KEYS",
        "nurse-key:nurse:A. Fisher,clin-key:clinician:Dr. J. Rao,admin-key:admin:M. Otieno",
    )
    auth.reload_keys()

    submit = client.post("/triage", json=_patient_payload("P-AUTH-2"), headers={"X-API-Key": "nurse-key"})
    assert submit.status_code == 200
    band = submit.json()["final_recommended_band"]
    other_band = 5 if band != 5 else 4

    override = client.post(
        "/override",
        json={
            "patient_id": "P-AUTH-2",
            "ai_recommendation_band": band,
            "clinician_band": other_band,
            "reason_code": "Bedside exam supports a different band.",
        },
        headers={"X-API-Key": "clin-key"},
    )
    assert override.status_code == 200

    tail = client.get("/audit?n=10", headers={"X-API-Key": "admin-key"})
    events = {e["event_type"]: e for e in tail.json()}
    assert events["triage_submitted"]["submitted_by"] == "A. Fisher"
    assert events["triage_submitted"]["submitted_by_role"] == "nurse"
    override_event = events.get("clinician_override") or events.get("clinician_confirmed")
    assert override_event["decided_by"] == "Dr. J. Rao"
    assert override_event["decided_by_role"] == "clinician"


def test_model_toggle_requires_admin(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "nurse-key:nurse:A. Fisher,admin-key:admin:M. Otieno")
    auth.reload_keys()

    denied_no_key = client.post("/model/unavailable")
    assert denied_no_key.status_code == 401

    denied_wrong_role = client.post("/model/unavailable", headers={"X-API-Key": "nurse-key"})
    assert denied_wrong_role.status_code == 403

    granted = client.post("/model/unavailable", headers={"X-API-Key": "admin-key"})
    assert granted.status_code == 200
    assert granted.json()["status"] == "unavailable"

    restored = client.post("/model/available", headers={"X-API-Key": "admin-key"})
    assert restored.status_code == 200
    assert restored.json()["status"] == "ok"


def test_reload_keys_endpoint_requires_admin_and_takes_effect(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "admin-key:admin:M. Otieno")
    auth.reload_keys()

    denied = client.post("/admin/reload-keys", headers={"X-API-Key": "nurse-key"})
    assert denied.status_code == 401

    # Simulate rotating in a new nurse key by editing the env var directly
    # (standing in for an edited .env file) and reloading without a restart.
    monkeypatch.setenv("TRIAGE_API_KEYS", "admin-key:admin:M. Otieno,new-nurse-key:nurse:B. Diaz")
    reload_resp = client.post("/admin/reload-keys", headers={"X-API-Key": "admin-key"})
    assert reload_resp.status_code == 200
    assert reload_resp.json()["key_count"] == 2

    now_works = client.post("/triage", json=_patient_payload("P-AUTH-3"), headers={"X-API-Key": "new-nurse-key"})
    assert now_works.status_code == 200


def test_cors_allows_known_frontend_origin(client):
    resp = client.get("/model/status", headers={"Origin": "http://localhost:5173"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    resp = client.get("/model/status", headers={"Origin": "http://evil.example.com"})
    assert "access-control-allow-origin" not in resp.headers


def test_next_patient_id_requires_nurse_and_returns_suggestion(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "nurse-key:nurse:A. Fisher")
    auth.reload_keys()

    denied = client.get("/patients/next-id")
    assert denied.status_code == 401

    granted = client.get("/patients/next-id", headers={"X-API-Key": "nurse-key"})
    assert granted.status_code == 200
    assert granted.json()["patient_id"] == "P-1001"