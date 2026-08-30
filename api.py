import json
import os
import pickle
from datetime import datetime, timezone

from dotenv import load_dotenv

# Must run before `db`/`pipeline` are imported below -- they transitively import
# llm_extract, which reads OPENAI_API_KEY / OPENAI_MODEL / OPENAI_TIMEOUT_SECONDS
# as module-level constants at import time.
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db
from pipeline import TriagePipeline
from queue_sim import run_operational_scenario
from surge import BASELINE_ARRIVALS_PER_HOUR
from validation import ValidationError, validate_band

app = FastAPI(title="PatientTriage.ai")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
db.init_db()
db.seed_demo_patients_if_empty(pipeline)#new


@app.on_event("startup")
def _startup():
    db.init_db()
    db.seed_demo_patients_if_empty(pipeline)#new


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
def triage(patient: PatientInput):
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
    row_id, arrival_time = db.insert_triage_record(patient.patient_id, record_for_storage, result)
    db.insert_audit("triage_submitted", patient.patient_id, {
        "row_id": row_id,
        "final_recommended_band": result["final_recommended_band"],
        "model_recommended_band": result["model_recommended_band"],
        "safety_gate_triggers": result["safety_gate_triggers"],
        "extraction_backend": result.get("extraction_backend"),
        "reassessment": bool(history),
    })

    client_result = {k: v for k, v in result.items() if k != "feature_snapshot"}
    client_result["row_id"] = row_id
    client_result["arrival_time"] = arrival_time
    client_result["is_reassessment"] = bool(history)
    return client_result


@app.post("/override")
def override(payload: OverrideInput):
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
    db.apply_override(payload.patient_id, payload.ai_recommendation_band, payload.clinician_band, cleaned_reason or None)
    db.insert_audit(
        "clinician_override" if payload.clinician_band != payload.ai_recommendation_band else "clinician_confirmed",
        payload.patient_id,
        {
            "ai_recommendation_band": payload.ai_recommendation_band,
            "clinician_decision_band": payload.clinician_band,
            "override_reason": cleaned_reason or None,
            "is_downgrade": payload.clinician_band > payload.ai_recommendation_band,
        },
    )
    return {"status": "recorded"}


@app.get("/queue")
def queue_status():
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
def patient_detail(patient_id: str):
    detail = db.get_patient_detail(patient_id)
    if not detail:
        raise HTTPException(status_code=404, detail="patient not found")
    return detail


@app.get("/patients/{patient_id}/history")
def patient_history(patient_id: str):
    return db.get_patient_timeline(patient_id)


@app.post("/disposition")
def set_disposition(payload: DispositionInput):
    try:
        result = db.set_disposition(payload.patient_id, payload.disposition, payload.note)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/audit")
def audit_tail(n: int = 20):
    return db.get_audit_tail(n)


@app.get("/surge/status")
def surge_status(n_clinicians: int = 4):
    """Live, DB-backed ED operational status: NORMAL/SURGE/CRISIS derived
    from real arrivals and the real queue -- never from clinical bands.
    See surge.py / db.get_live_operational_status for the full contract.
    """
    return db.get_live_operational_status(n_clinicians=n_clinicians)


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
def set_unavailable():
    pipeline.set_model_unavailable()
    return {"status": pipeline.model_status}


@app.post("/model/available")
def set_available():
    pipeline.set_model_available()
    return {"status": pipeline.model_status}
