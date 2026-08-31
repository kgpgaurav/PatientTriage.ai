import json
import os
import pickle

import audit
from patients import DEMO_PATIENTS
from pipeline import TriagePipeline
from queue_sim import simulate_surge


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def load_pipeline():
    with open(os.path.join(OUTPUTS_DIR, "critical_model.pkl"), "rb") as f:
        critical_model = pickle.load(f)
    with open(os.path.join(OUTPUTS_DIR, "severity_model.pkl"), "rb") as f:
        severity_model = pickle.load(f)
    with open(os.path.join(OUTPUTS_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    return TriagePipeline(critical_model, severity_model, feature_cols)


def run_demo_patients(pipeline):
    print("\n=== DEMO PATIENT SCENARIOS ===")
    rows = []
    for p in DEMO_PATIENTS:
        note = p.get("note")
        history = p.get("history_readings")
        display = {k: v for k, v in p.items() if k not in ("note", "history_readings")}
        clean = {k: v for k, v in p.items() if k not in ("note", "history_readings")}
        result = pipeline.run(dict(clean), history=history, note=note)
        result["_display"] = display
        result["_note"] = note
        rows.append(result)
        print(f"{result['patient_id']}: band={result['final_recommended_band']} "
              f"(model={result['model_recommended_band']}) "
              f"p_crit={result['critical_probability']:.2f} "
              f"completeness={result['input_completeness']} "
              f"extraction=[{result.get('extraction_backend')}/{result['extraction_status']}] "
              f"gate=[{result['safety_gate_reason']}]")
    return rows


def run_override_demo(pipeline, rows):
    print("\n=== CLINICIAN OVERRIDE DEMO ===")
    entries = []
    target = next(r for r in rows if r["patient_id"] == "P18")
    entry = pipeline.record_clinician_decision(
        patient_id=target["patient_id"],
        ai_recommendation_band=target["final_recommended_band"],
        clinician_band=min(5, target["final_recommended_band"] + 1),
        reason_code="Bedside exam shows resolving symptoms and stable repeat vitals; clinician downgrades one level.",
    )
    entries.append(entry)
    print(json.dumps(entry, indent=2))

    try:
        pipeline.record_clinician_decision(
            patient_id=target["patient_id"],
            ai_recommendation_band=target["final_recommended_band"],
            clinician_band=target["final_recommended_band"] + 1,
            reason_code=None,
        )
    except ValueError as e:
        print("Downgrade without reason code correctly rejected:", e)
    return entries


def run_surge_demo():
    print("\n=== SURGE SIMULATION (3x normal volume, 3 hours) ===")
    normal = simulate_surge(base_arrivals_per_hour=20, surge_multiplier=1, duration_min=180)
    surge = simulate_surge(base_arrivals_per_hour=20, surge_multiplier=3, duration_min=180)
    print("Normal load:", json.dumps(normal["stats"], indent=2, default=str))
    print("3x surge:", json.dumps(surge["stats"], indent=2, default=str))
    print("Breach events -- normal:", normal["n_breach_events"], " surge:", surge["n_breach_events"])
    return {"normal": normal, "surge_3x": surge}


def main():
    pipeline = load_pipeline()
    rows = run_demo_patients(pipeline)
    run_override_demo(pipeline, rows)
    run_surge_demo()

    print("\n=== QUEUE POLICY COMPARISON (static vs FIFO vs wait-decay) ===")
    from queue_sim import compare_queue_policies
    policy_results = compare_queue_policies()
    print(json.dumps(policy_results, indent=2, default=str)[:2000])

    print("\n=== AUDIT LOG TAIL ===")
    entries = audit.read_all()
    for e in entries[-3:]:
        print(json.dumps(e, indent=2, default=str)[:600])


if __name__ == "__main__":
    main()
