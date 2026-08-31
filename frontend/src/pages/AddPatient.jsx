import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api, BAND_COLOR, SYMPTOMS } from "../api";
import ShapBars from "../components/ShapBars";

// These don't change between readings for the same patient -- carried over
// (and locked, via the isReassess flag below) when the form is opened via a
// queue row's "Reassess" button, instead of asking the nurse to re-enter
// identical answers every time. Vitals, mental status, symptoms, and the
// note are always freshly entered, reassess or not.
function blankForm(overrides = {}) {
  return {
    patient_id: "", age: "", age_months: "", gender: "other", hr: "", sbp: "", rr: "", temp: "", spo2: "",
    mental_status_altered: false, pregnancy: false, has_prior_history: "true",
    note: "",
    ...overrides,
  };
}

// Builds the carried-over fields from a queue entry (see QueueRow's
// "Reassess" button) -- converting types to match the form's own shape
// (has_prior_history is a "true"/"false" string here, a real boolean on
// the entry from GET /queue).
function reassessOverrides(entry) {
  return {
    patient_id: entry.patient_id || "",
    age: entry.age != null ? String(entry.age) : "",
    age_months: entry.age_months != null ? String(entry.age_months) : "",
    gender: entry.gender || "other",
    has_prior_history: String(!!entry.has_prior_history),
    pregnancy: !!entry.pregnancy,
  };
}

// Mirrors data_gen.age_group on the backend, purely for an immediate visual
// hint as the nurse types an age -- the backend always derives (and is the
// only source of truth for) the age group actually used in triage.
function deriveAgeGroup(ageStr) {
  const n = parseInt(ageStr, 10);
  if (ageStr === "" || Number.isNaN(n)) return "—";
  if (n < 13) return "pediatric";
  if (n < 65) return "adult";
  return "geriatric";
}

export default function AddPatient() {
  const location = useLocation();
  const reassessEntry = location.state?.reassess || null;

  const [form, setForm] = useState(() =>
    reassessEntry ? blankForm(reassessOverrides(reassessEntry)) : blankForm()
  );
  const [isReassess, setIsReassess] = useState(!!reassessEntry);
  const [symptoms, setSymptoms] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  // A fresh "new patient" open (not a reassess) gets a suggested next ID on
  // load -- just a starting point in an editable field, not a reservation
  // (see db.next_patient_id). Skipped when prefilled via "Reassess", since
  // that ID is the whole point of the button. assignFreshId is also called
  // explicitly after a successful submit and by "Submit another patient",
  // since each of those needs a new suggestion, not just a mount-time one.
  async function assignFreshId() {
    try {
      const { patient_id } = await api.getNextPatientId();
      setForm((f) => ({ ...f, patient_id }));
    } catch {
      // Non-fatal -- the field just stays blank and the nurse types an ID
      // by hand, exactly like before this existed.
    }
  }

  useEffect(() => {
    if (!isReassess) assignFreshId();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function update(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  function toggleSymptom(s) {
    setSymptoms((cur) => ({ ...cur, [s]: !cur[s] }));
  }

  function numOrNull(v) {
    return v === "" ? null : parseFloat(v);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setResult(null);

    const activeSymptoms = Object.fromEntries(Object.entries(symptoms).filter(([, v]) => v));

    const payload = {
      patient_id: form.patient_id.trim(),
      age: parseInt(form.age, 10),
      age_months: form.age_months === "" ? null : parseInt(form.age_months, 10),
      gender: form.gender,
      hr: numOrNull(form.hr), sbp: numOrNull(form.sbp), rr: numOrNull(form.rr),
      temp: numOrNull(form.temp), spo2: numOrNull(form.spo2),
      mental_status_altered: form.mental_status_altered,
      pregnancy: form.pregnancy,
      has_prior_history: form.has_prior_history === "true",
      symptoms: activeSymptoms,
      note: form.note.trim() || null,
    };

    try {
      const res = await api.submitTriage(payload);
      setResult(res);
      resetToFreshForm();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  function resetToFreshForm() {
    setIsReassess(false);
    setForm(blankForm());
    setSymptoms({});
    assignFreshId();
  }

  function submitAnother() {
    resetToFreshForm();
    setResult(null);
    setError(null);
  }

  return (
    <div className="grid-2">
      <div className="panel">
        <h2>New patient / reassessment</h2>
        {isReassess && (
          <div className="result-box" style={{ marginBottom: 12, fontSize: 12.5 }}>
            Reassessing <b>{form.patient_id}</b> — demographics below are carried over from their prior
            record and locked. Enter this reading's current vitals, symptoms, and note.
          </div>
        )}
        <form onSubmit={handleSubmit}>
          <div className="field">
            <label>Patient ID</label>
            <input
              id="f-patient-id"
              type="text" required placeholder="e.g. P-1042 or MRN" disabled={isReassess}
              value={form.patient_id} onChange={(e) => update("patient_id", e.target.value)}
            />
          </div>
          <div className="row3">
            <div className="field">
              <label>Age (years)</label>
              <input id="f-age" type="number" min="0" max="120" required disabled={isReassess} value={form.age} onChange={(e) => update("age", e.target.value)} />
            </div>
            <div className="field">
              <label>Months</label>
              <input id="f-age-months" type="number" min="0" max="11" disabled={isReassess} value={form.age_months} onChange={(e) => update("age_months", e.target.value)} />
            </div>
            <div className="field">
              <label>Gender</label>
              <select id="f-gender" value={form.gender} disabled={isReassess} onChange={(e) => update("gender", e.target.value)}>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>

          <div className="row2">
            <div className="field">
              <label>Age group</label>
              <div id="f-age-group-derived" style={{ padding: "7px 9px", color: "var(--muted)", fontSize: 13 }}>
                {deriveAgeGroup(form.age)} <span style={{ fontSize: 11 }}>(auto, from age)</span>
              </div>
            </div>
            <div className="field">
              <label>Has prior history?</label>
              <select id="f-history" value={form.has_prior_history} disabled={isReassess} onChange={(e) => update("has_prior_history", e.target.value)}>
                <option value="true">yes</option>
                <option value="false">no (first visit)</option>
              </select>
            </div>
          </div>

          <div className="row3">
            <div className="field"><label>HR</label><input id="f-hr" type="number" value={form.hr} onChange={(e) => update("hr", e.target.value)} /></div>
            <div className="field"><label>SBP</label><input id="f-sbp" type="number" value={form.sbp} onChange={(e) => update("sbp", e.target.value)} /></div>
            <div className="field"><label>RR</label><input id="f-rr" type="number" value={form.rr} onChange={(e) => update("rr", e.target.value)} /></div>
          </div>
          <div className="row2">
            <div className="field"><label>Temp (°C)</label><input id="f-temp" type="number" step="0.1" value={form.temp} onChange={(e) => update("temp", e.target.value)} /></div>
            <div className="field"><label>SpO2 (%)</label><input id="f-spo2" type="number" value={form.spo2} onChange={(e) => update("spo2", e.target.value)} /></div>
          </div>

          <div className="checks">
            <label className="chk">
              <input id="f-altered" type="checkbox" checked={form.mental_status_altered} onChange={(e) => update("mental_status_altered", e.target.checked)} />
              Altered mental status
            </label>
            <label className="chk">
              <input id="f-pregnant" type="checkbox" checked={form.pregnancy} disabled={isReassess} onChange={(e) => update("pregnancy", e.target.checked)} />
              Pregnancy
            </label>
          </div>

          <div className="field"><label>Symptoms observed</label></div>
          <div className="symptom-grid">
            {SYMPTOMS.map((s) => (
              <label className="chk" key={s}>
                <input type="checkbox" data-symptom={s} checked={!!symptoms[s]} onChange={() => toggleSymptom(s)} />
                {s.replace(/_/g, " ")}
              </label>
            ))}
          </div>

          <div className="field">
            <label>Free-text note (optional — parsed by extraction layer)</label>
            <textarea
              id="f-note"
              placeholder="e.g. pt c/o chest tightness x2 days, denies SOB, appears anxious"
              value={form.note} onChange={(e) => update("note", e.target.value)}
            />
          </div>

          <button type="submit" className="btn full" id="submit-btn" disabled={submitting}>
            {submitting ? "Submitting..." : "Submit to triage"}
          </button>
        </form>

        {error && <div className="error-box">Submission failed: {error}</div>}
      </div>

      <div className="panel">
        <h2>Result</h2>
        {!result && <div style={{ color: "var(--muted)", fontSize: 13 }}>
          Submit a patient on the left to see the triage decision here.
        </div>}
        {result && (
          <div className="result-box">
            <span
              className="result-band"
              style={{
                background: BAND_COLOR[result.final_recommended_band] + "22",
                color: BAND_COLOR[result.final_recommended_band],
                border: `1px solid ${BAND_COLOR[result.final_recommended_band]}`,
              }}
            >
              BAND {result.final_recommended_band}
            </span>
            <span style={{ fontSize: 11, color: "var(--muted)", marginLeft: 8 }}>
              {result.is_reassessment ? "(reassessment — history applied)" : "(new patient)"}
            </span>

            {result.recommendation_mode === "safety_fallback" && (
              <div className="error-box" style={{ marginTop: 10 }}>
                ML model unavailable — this is a safety-fallback recommendation (Band {result.fallback_band}), not a model prediction. Hard redlines and clinician review still apply.
              </div>
            )}

            <div className="result-line">
              P(critical) <b>{result.critical_probability != null ? result.critical_probability.toFixed(2) : "—"}</b> &nbsp;
              model band <b>{result.model_recommended_band}</b> &nbsp;
              input completeness <b>{result.input_completeness}</b> &nbsp;
              extraction <b>{result.extraction_backend || "none"}</b>
            </div>
            <div
              className="result-line"
              title="How much the model's own cross-validated folds agree on this patient, adjusted for how complete the intake data was. Distinct from input completeness above."
              style={{
                color:
                  result.confidence_level === "HIGH" ? "var(--band5)" :
                  result.confidence_level === "MEDIUM" ? "var(--band3)" : "var(--band1)",
              }}
            >
              model confidence <b>{result.confidence_level || "LOW"}</b>
              {result.confidence_score != null && <> ({result.confidence_score.toFixed(2)})</>}
              {result.confidence_reason && <span style={{ color: "var(--muted)" }}> — {result.confidence_reason}</span>}
            </div>

            {result.age_group_overridden && (
              <div className="result-line" style={{ color: "var(--band3)" }}>
                Age group entered didn't match age — using derived group: <b>{result.age_group}</b>
              </div>
            )}

            {result.observed_reported_mismatch && (
              <div className="result-line" title="Informational only — not used in risk scoring or the Safety Gate">
                Observed/reported mismatch flagged in note (informational only)
              </div>
            )}

            <div
              className="gate-note"
              style={{ color: result.final_recommended_band < result.model_recommended_band ? "var(--band1)" : "var(--band5)" }}
            >
              {result.safety_gate_reason}
            </div>

            <ShapBars explanation={result.model_explanation} />

            <div style={{ marginTop: 16, display: "flex", gap: 10 }}>
              <button className="btn secondary" onClick={() => navigate("/")}>View on live board</button>
              <button className="btn secondary" onClick={submitAnother}>Submit another patient</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}