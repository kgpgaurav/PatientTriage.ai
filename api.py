import json
import os
import pickle
from datetime import datetime, timezone

from dotenv import load_dotenv

# Must run before `db`/`pipeline` are imported below -- they transitively import
# llm_extract, which reads OPENAI_API_KEY / OPENAI_MODEL / OPENAI_TIMEOUT_SECONDS
# as module-level constants at import time.
load_dotenv()

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import auth
import db
import surge_simulator
from pipeline import TriagePipeline
from queue_sim import run_operational_scenario
from surge import BASELINE_ARRIVALS_PER_HOUR
from validation import ValidationError, validate_band

app = FastAPI(title="PatientTriage.ai")

# Scoped to the known frontend origin(s), not "*" -- a wildcard origin next to
# an API-key-in-header auth scheme is a real combination to avoid, especially
# since the React app persists its key to localStorage (frontend/src/api.js).
# Override via TRIAGE_ALLOWED_ORIGINS (comma-separated) for a non-default
# frontend host/port or a real deployment; defaults cover the Vite dev server.
_allowed_origins_raw = os.environ.get(
    "TRIAGE_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
)
ALLOWED_ORIGINS = [origin.strip() for origin in _allowed_origins_raw.split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

# Resolve model/output paths relative to this file, not the process's current
# working directory -- so `python api.py` and `uvicorn api:app` behave the
# same whether launched from the project root or anywhere else.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.environ.get("PATIENT_TRIAGE_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))

with open(os.path.join(OUTPUTS_DIR, "critical_model.pkl"), "rb") as f:
    _critical_model = pickle.load(f)
with open(os.path.join(OUTPUTS_DIR, "severity_model.pkl"), "rb") as f:
    _severity_model = pickle.load(f)
with open(os.path.join(OUTPUTS_DIR, "feature_cols.json")) as f:
    _feature_cols = json.load(f)

pipeline = TriagePipeline(_critical_model, _severity_model, _feature_cols)


@app.on_event("startup")
def _startup():
    db.init_db()
    db.seed_demo_patients_if_empty(pipeline)


class PatientInput(BaseModel):
    patient_id: str
    age: int = Field(ge=0, le=120)
    age_months: int | None = Field(default=None, ge=0, le=11)
    gender: str | None = None  # male, female, other
    age_group: str | None = None  # advisory only -- the backend always derives this from `age`
    hr: float | None = Field(default=None, ge=0, le=300)
    sbp: float | None = Field(default=None, ge=0, le=300)
    rr: float | None = Field(default=None, ge=0, le=100)
    temp: float | None = Field(default=None, ge=25.0, le=45.0)
    spo2: float | None = Field(default=None, ge=0, le=100)
    mental_status_altered: bool = False
    pregnancy: bool = False
    has_prior_history: bool = True
    symptoms: dict[str, bool] = {}
    note: str | None = None


class OverrideInput(BaseModel):
    patient_id: str
    ai_recommendation_band: int
    clinician_band: int
    reason_code: str | None = None


class DispositionInput(BaseModel):
    patient_id: str
    disposition: str
    note: str | None = None


@app.post("/triage")
def triage(patient: PatientInput, caller=Depends(auth.require_role("nurse"))):
    record = patient.dict(exclude={"symptoms", "note"})
    record.update(patient.symptoms)

    history, last_reading_at = db.get_patient_history(patient.patient_id, limit=3)
    if last_reading_at:
        minutes_since_prev = (datetime.now(timezone.utc) - datetime.fromisoformat(last_reading_at)).total_seconds() / 60.0
        record["minutes_since_prev"] = minutes_since_prev

    try:
        result = pipeline.run(record, history=history if history else None, note=patient.note)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors)

    # The backend derives age_group from age regardless of what the client
    # sent (see pipeline.run) -- persist that canonical value, not whatever
    # the client submitted, so the stored/displayed record matches what the
    # Safety Gate actually used. Same for the mismatch flag, which only
    # exists after note extraction runs inside pipeline.run().
    record["age_group"] = result.get("age_group", record.get("age_group"))
    record["observed_reported_mismatch"] = result.get("observed_reported_mismatch", False)

    # `record` intentionally excludes `note` (it's not a model feature), but the
    # dashboard needs to display the original free-text note, so stash it in a
    # copy used only for persistence — this does not touch what the model saw.
    record_for_storage = {**record, "note": patient.note}
    row_id, arrival_time = db.insert_triage_record(
        patient.patient_id, record_for_storage, result,
        decided_by_role=caller["role"], decided_by=caller.get("name"),
    )
    db.insert_audit("triage_submitted", patient.patient_id, {
        "row_id": row_id,
        "final_recommended_band": result["final_recommended_band"],
        "model_recommended_band": result["model_recommended_band"],
        "confidence_level": result.get("confidence_level"),
        "safety_gate_triggers": result["safety_gate_triggers"],
        "extraction_backend": result.get("extraction_backend"),
        "reassessment": bool(history),
        "submitted_by_role": caller["role"],
        "submitted_by": caller.get("name"),
    })

    client_result = {k: v for k, v in result.items() if k != "feature_snapshot"}
    client_result["row_id"] = row_id
    client_result["arrival_time"] = arrival_time
    client_result["is_reassessment"] = bool(history)
    return client_result


@app.post("/override")
def override(payload: OverrideInput, caller=Depends(auth.require_role("clinician"))):
    try:
        validate_band(payload.ai_recommendation_band, "ai_recommendation_band")
        validate_band(payload.clinician_band, "clinician_band")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors)

    cleaned_reason = (payload.reason_code or "").strip()
    if payload.clinician_band == payload.ai_recommendation_band:
        raise HTTPException(status_code=400, detail="No clinician band change selected.")
    if payload.clinician_band > payload.ai_recommendation_band and not cleaned_reason:
        raise HTTPException(status_code=400, detail="Downgrading below the AI recommendation requires a reason code.")
    db.apply_override(
        payload.patient_id, payload.ai_recommendation_band, payload.clinician_band, cleaned_reason or None,
        decided_by_role=caller["role"], decided_by=caller.get("name"),
    )
    db.insert_audit(
        "clinician_override" if payload.clinician_band != payload.ai_recommendation_band else "clinician_confirmed",
        payload.patient_id,
        {
            "ai_recommendation_band": payload.ai_recommendation_band,
            "clinician_decision_band": payload.clinician_band,
            "override_reason": cleaned_reason or None,
            "is_downgrade": payload.clinician_band > payload.ai_recommendation_band,
            "decided_by_role": caller["role"],
            "decided_by": caller.get("name"),
        },
    )
    return {"status": "recorded"}


@app.get("/patients/next-id")
def next_patient_id(caller=Depends(auth.require_role("nurse"))):
    """Suggested next auto-generated patient ID for the intake form to
    pre-fill (see db.next_patient_id -- a suggestion, not a reservation)."""
    return {"patient_id": db.next_patient_id()}


@app.get("/queue")
def queue_status(caller=Depends(auth.require_role("nurse"))):
    entries = db.get_queue()
    if not entries:
        try:
            from patients import DEMO_PATIENTS
            entries = [
                {
                    "patient_id": p["patient_id"],
                    "final_recommended_band": p.get("final_recommended_band", 3),
                    "clinician_decision_band": p.get("clinician_decision_band"),
                    "critical_probability": p.get("critical_probability", 0.5),
                    "input_completeness": p.get("input_completeness", "MEDIUM"),
                    "safety_gate_reason": p.get("safety_gate_reason"),
                    "arrival_time": p.get("arrival_time", "2026-01-01T00:00:00Z"),
                    "waited_min": p.get("waited_min", 0),
                    "ceiling_min": p.get("ceiling_min", 30),
                    "breached": p.get("breached", False),
                    "reassessment_required": p.get("reassessment_required", False),
                }
                for p in DEMO_PATIENTS
            ]
        except Exception:
            pass
    breaches = [e for e in entries if e["breached"]]
    return {"summary": entries, "breaches": breaches}


@app.get("/patients/{patient_id}")
def patient_detail(patient_id: str, caller=Depends(auth.require_role("clinician"))):
    detail = db.get_patient_detail(patient_id)
    db.insert_audit("patient_record_accessed", patient_id, {
        "accessed_by_role": caller["role"], "accessed_by": caller.get("name"), "found": bool(detail),
    })
    if not detail:
        raise HTTPException(status_code=404, detail="patient not found")
    return detail


@app.get("/patients/{patient_id}/history")
def patient_history(patient_id: str, caller=Depends(auth.require_role("nurse"))):
    return db.get_patient_timeline(patient_id)


@app.post("/disposition")
def set_disposition(payload: DispositionInput, caller=Depends(auth.require_role("clinician"))):
    try:
        result = db.set_disposition(
            payload.patient_id,
            payload.disposition,
            payload.note,
            decided_by_role=caller["role"],
            decided_by=caller.get("name"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/audit")
def audit_tail(n: int = 20, caller=Depends(auth.require_role("admin"))):
    return db.get_audit_tail(n)


@app.get("/surge/status")
def surge_status(n_clinicians: int = 4):
    """Live, DB-backed ED operational status: NORMAL/SURGE/CRISIS derived
    from real arrivals and the real queue -- never from clinical bands.
    See surge.py / db.get_live_operational_status for the full contract.

    Also carries the live simulator's current/last run (see
    surge_simulator.get_status) so the dashboard button can show progress
    and disable itself while a run is active, from the one endpoint it's
    already polling.
    """
    status = db.get_live_operational_status(n_clinicians=n_clinicians)
    status["simulation"] = surge_simulator.get_status()
    return status


@app.post("/surge/simulate-arrivals")
def surge_simulate_arrivals(n_clinicians: int = 4):
    """Starts the live, real-time arrival simulator (surge_simulator.py) for
    the dashboard's "Simulate patient surge" button.

    Unlike POST /surge/simulate below (an offline, in-memory what-if
    calculation), this logs real rows into the live `patients` table, one at
    a time, through the same pipeline.run()/db.insert_triage_record() path a
    real nurse's POST /triage uses -- so the existing frequency-based
    auto-detector in db.get_live_operational_status has something real to
    detect, and the resulting state change is genuine, not asserted.

    Returns immediately; the run continues in a background thread and stops
    on its own once the live status leaves NORMAL (or a safety cap is hit).
    Calling this again while a run is already active is a no-op that reports
    the in-progress status instead of starting a second, overlapping run.
    """
    return surge_simulator.start(pipeline, n_clinicians=n_clinicians)


class SurgeSimInput(BaseModel):
    surge_multiplier: float = Field(default=1, ge=0.1, le=10)
    duration_min: int = Field(default=180, ge=10, le=720)
    n_clinicians: int = Field(default=4, ge=1, le=50)
    seed: int | None = Field(default=0)
    base_arrivals_per_hour: float = Field(default=BASELINE_ARRIVALS_PER_HOUR, ge=1, le=500)


@app.post("/surge/simulate")
def surge_simulate(payload: SurgeSimInput):
    """Offline what-if simulation for the surge demo screen: runs the
    requested arrival multiplier through FIFO / STATIC_SEVERITY /
    WAIT_PROTECTED and returns operational-state classification, per-policy
    metrics, a policy comparison table, and a queue-visualization snapshot.
    Every number here comes from the simulation run -- nothing is
    hardcoded. Uses a fixed default seed so a recorded demo is reproducible;
    pass a different seed for a fresh random draw.
    """
    return run_operational_scenario(
        surge_multiplier=payload.surge_multiplier,
        base_arrivals_per_hour=payload.base_arrivals_per_hour,
        duration_min=payload.duration_min,
        n_clinicians=payload.n_clinicians,
        seed=payload.seed,
    )


@app.get("/model/status")
def model_status():
    return {"status": pipeline.model_status}


@app.post("/model/unavailable")
def set_unavailable(caller=Depends(auth.require_role("admin"))):
    """Operational safety control, not a data read -- gated at admin even
    though it carries no PII, since it can force the whole pipeline into the
    rules-only fallback path for every patient triaged until it's reversed."""
    pipeline.set_model_unavailable()
    db.insert_audit("model_set_unavailable", None, {
        "triggered_by_role": caller["role"], "triggered_by": caller.get("name"),
    })
    return {"status": pipeline.model_status}


@app.post("/model/available")
def set_available(caller=Depends(auth.require_role("admin"))):
    pipeline.set_model_available()
    db.insert_audit("model_set_available", None, {
        "triggered_by_role": caller["role"], "triggered_by": caller.get("name"),
    })
    return {"status": pipeline.model_status}


@app.post("/admin/reload-keys")
def reload_api_keys(caller=Depends(auth.require_role("admin"))):
    """Re-reads TRIAGE_API_KEYS (via a fresh dotenv load, so an edited .env
    file is picked up) without restarting the server -- lets a rotated or
    revoked key take effect immediately instead of needing a redeploy."""
    load_dotenv(override=True)
    keys = auth.reload_keys()
    db.insert_audit("api_keys_reloaded", None, {
        "triggered_by_role": caller["role"], "triggered_by": caller.get("name"),
        "key_count": len(keys),
    })
    return {"status": "reloaded", "key_count": len(keys)}