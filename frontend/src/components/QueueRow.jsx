import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, attributionLabel, BAND_COLOR, dispositionLabel, dispositionColor, formatDateTime } from "../api";
import ShapBars from "./ShapBars";

function BandPill({ band }) {
  if (band == null) return <span style={{ color: "var(--muted)" }}>—</span>;
  return (
    <span
      className="band-pill"
      style={{
        background: BAND_COLOR[band] + "22",
        color: BAND_COLOR[band],
        border: `1px solid ${BAND_COLOR[band]}`,
      }}
    >
      {band}
    </span>
  );
}

function StatusBadge({ status }) {
  const color = dispositionColor(status);
  return (
    <span className="status-badge" style={{ color, borderColor: color }}>
      {dispositionLabel(status)}
    </span>
  );
}

function vitalsLine(entry) {
  const parts = [];
  if (entry.age != null) {
    parts.push(`${entry.age}${entry.age_group ? " · " + entry.age_group[0].toUpperCase() + entry.age_group.slice(1) : ""}`);
  }
  if (entry.hr != null) parts.push(<span key="hr">HR <b>{entry.hr}</b></span>);
  if (entry.sbp != null) parts.push(<span key="sbp">SBP <b>{entry.sbp}</b></span>);
  if (entry.rr != null) parts.push(<span key="rr">RR <b>{entry.rr}</b></span>);
  if (entry.spo2 != null) parts.push(<span key="spo2">SpO2 <b>{entry.spo2}</b></span>);
  if (entry.temp != null) parts.push(<span key="temp">T <b>{entry.temp}</b></span>);
  return parts.length ? parts.reduce((acc, p, i) => (i === 0 ? [p] : [...acc, "  \u00A0 ", p]), []) : "—";
}

// Local formatTime removed -- use the shared formatDateTime from ../api, which
// renders the backend's UTC-ISO timestamps in the browser's local timezone
// (and includes the date, since a wait can span past midnight).

// Which disposition buttons make sense from the current state.
// Once a patient reaches a resolved state, their clock is frozen and no
// further transitions are offered from here (matches an ED encounter ending).
function nextActions(status) {
  if (status === "waiting") {
    return [
      { value: "in_treatment", label: "Start treatment" },
      { value: "admitted", label: "Admit" },
      { value: "discharged", label: "Discharge" },
    ];
  }
  if (status === "in_treatment") {
    return [
      { value: "admitted", label: "Admit" },
      { value: "treatment_successful", label: "Mark treatment successful" },
      { value: "discharged", label: "Discharge" },
    ];
  }
  return []; // admitted / treatment_successful / discharged are resolved end-states
}

function HistoryPanel({ patientId }) {
  const [history, setHistory] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getPatientHistory(patientId)
      .then((data) => !cancelled && setHistory(data))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [patientId]);

  if (error) return <div className="error-box">Couldn't load history: {error}</div>;
  if (history === null) return <div className="history-loading">Loading history…</div>;
  if (history.length === 0) return <div className="history-empty">No decisions or status changes logged yet.</div>;

  const readings = history.filter((h) => h.type === "vitals_reading");
  const events = history.filter((h) => h.type !== "vitals_reading");

  return (
    <div>
      {readings.length > 0 && (
        <div className="readings-table-wrap" style={{ marginBottom: 14 }}>
          <div className="lbl" style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", marginBottom: 6 }}>
            Readings over time{readings.length > 1 ? ` (${readings.length}, oldest first)` : ""}
          </div>
          <table className="readings-table">
            <thead>
              <tr>
                <th>Time</th><th>HR</th><th>SBP</th><th>RR</th><th>Temp</th><th>SpO2</th>
                <th>P(crit)</th><th>Band</th>
              </tr>
            </thead>
            <tbody>
              {readings.map((r, i) => (
                <tr key={i}>
                  <td>{formatDateTime(r.created_at)}</td>
                  <td>{r.hr ?? "—"}</td>
                  <td>{r.sbp ?? "—"}</td>
                  <td>{r.rr ?? "—"}</td>
                  <td>{r.temp ?? "—"}</td>
                  <td>{r.spo2 ?? "—"}</td>
                  <td>{r.critical_probability != null ? r.critical_probability.toFixed(2) : "—"}</td>
                  <td>{r.final_recommended_band ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="history-list">
        {events.map((h, i) => {
          const who = attributionLabel(h.decided_by_role, h.decided_by);
          return (
            <div className="history-item" key={i}>
              <span className="history-time">{formatDateTime(h.created_at)}</span>
              {h.type === "band_decision" && (
                <span>
                  Clinician logged <b>Band {h.clinician_decision_band}</b>
                  {h.ai_recommendation_band != null && ` (AI recommended ${h.ai_recommendation_band})`}
                  {h.is_downgrade ? " — downgrade" : ""}
                  {h.override_reason ? ` — "${h.override_reason}"` : ""}
                </span>
              )}
              {h.type === "disposition_change" && (
                <span>
                  Status: <b>{dispositionLabel(h.previous_status)}</b> → <b>{dispositionLabel(h.new_status)}</b>
                  {h.note ? ` — "${h.note}"` : ""}
                </span>
              )}
              {h.type === "reassessment_performed" && <span>New reading logged for a required reassessment</span>}
              {h.type === "wait_breach_detected" && <span>Safe-wait ceiling breached</span>}
              {who && <span style={{ color: "var(--muted)" }}> — {who}</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function QueueRow({ entry, onOverridden }) {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [clinicianBand, setClinicianBand] = useState(entry.final_recommended_band);
  const [reason, setReason] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [justLogged, setJustLogged] = useState(false);
  const [dispositionSubmitting, setDispositionSubmitting] = useState(null); // which value is in flight
  const [dispositionError, setDispositionError] = useState(null);

  async function submitOverride() {
    const cleanedReason = reason.trim();
    const isNoChange = clinicianBand === entry.final_recommended_band;
    const isDowngrade = clinicianBand > entry.final_recommended_band;

    if (isNoChange) {
      setError("Select a different band before logging a decision.");
      return;
    }
    if (isDowngrade && !cleanedReason) {
      setError("A reason code is required for a downgrade.");
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      await api.submitOverride({
        patient_id: entry.patient_id,
        ai_recommendation_band: entry.final_recommended_band,
        clinician_band: clinicianBand,
        reason_code: cleanedReason || null,
      });
      setReason("");
      setJustLogged(true);
      onOverridden && onOverridden();
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function submitDisposition(value) {
    setDispositionSubmitting(value);
    setDispositionError(null);
    try {
      await api.setDisposition({ patient_id: entry.patient_id, disposition: value });
      onOverridden && onOverridden(); // reuse the same "refresh the queue" callback
    } catch (e) {
      setDispositionError(e.message);
    } finally {
      setDispositionSubmitting(null);
    }
  }

  // Opens the intake form pre-filled with this patient's demographics
  // (age, gender, prior-history, pregnancy) locked, so the nurse only has
  // to enter the current reading's vitals/symptoms/note -- same identity,
  // same patient_id, matched by ID exactly like a manually re-typed ID
  // already was before this button existed.
  function startReassess() {
    navigate("/add-patient", {
      state: {
        reassess: {
          patient_id: entry.patient_id,
          age: entry.age,
          age_months: entry.age_months,
          gender: entry.gender,
          has_prior_history: entry.has_prior_history,
          pregnancy: entry.pregnancy,
        },
      },
    });
  }

  const isNoChange = clinicianBand === entry.final_recommended_band;
  const isDowngrade = clinicianBand > entry.final_recommended_band;
  const disableSubmit = submitting || isNoChange || (isDowngrade && !reason.trim());

  const gateFired = entry.safety_gate_triggers && entry.safety_gate_triggers.length > 0;
  const escalated = entry.escalated_by_gate
    ?? (entry.model_recommended_band != null && entry.final_recommended_band < entry.model_recommended_band);

  const status = entry.status || "waiting";
  const actions = nextActions(status);
  const isResolved = entry.is_resolved ?? !["waiting", "in_treatment"].includes(status);
  // The effective band is what actually governs this patient right now: the
  // clinician's logged decision if one exists, otherwise the AI's own
  // recommendation. This -- not the frozen AI number -- is what should drive
  // the row's color and the big "Final band" pill; the AI's original call
  // stays visible as a secondary tag for audit/context.
  const effectiveBand = entry.effective_band ?? entry.clinician_decision_band ?? entry.final_recommended_band;
  const clinicianOverrode = entry.clinician_decision_band != null && entry.clinician_decision_band !== entry.final_recommended_band;

  return (
    <>
      <tr className="row" onClick={() => setOpen((o) => !o)}>
        <td className="band-cell">
          <span className="band-bar" style={{ background: BAND_COLOR[effectiveBand] }} />
        </td>
        <td className="pid">{entry.patient_id}</td>
        <td className="vitals">{vitalsLine(entry)}</td>
        <td>
          <span className={`quality ${entry.input_completeness}`} title="Reflects how much intake data was supplied — not clinical confidence or model certainty">{entry.input_completeness}</span>
        </td>
        <td className="prob">
          {entry.critical_probability != null ? entry.critical_probability.toFixed(2) : "—"}
          <br />
          <span
            className={`quality ${entry.confidence_level || "LOW"}`}
            style={{ fontSize: 10 }}
            title={entry.confidence_reason || "Model certainty for this prediction — never omitted alongside a score."}
          >
            conf: {entry.confidence_level || "LOW"}
          </span>
        </td>
        <td>
          <BandPill band={entry.model_recommended_band} />
        </td>
        <td>
          <BandPill band={effectiveBand} />
          {escalated && <span className="escalated-tag">GATE</span>}
          {clinicianOverrode && (
            <span
              className="clinician-tag"
              title={entry.override_reason ? `Reason: ${entry.override_reason}` : undefined}
              style={{ color: "var(--muted)", borderColor: "var(--line)" }}
            >
              AI: {entry.final_recommended_band}
            </span>
          )}
        </td>
        <td>
          <div className="status-stack">
            <StatusBadge status={status} />
            {entry.reassessment_required && (
              <span className="clinician-tag" style={{ color: "var(--band1)", borderColor: "var(--band1)" }} title="Wait-time safety envelope exceeded — reassessment required">
                REASSESS
              </span>
            )}
          </div>
        </td>
        <td className={`wait ${entry.breached ? "breach" : ""}`}>
          <div className="wait-cell">
            <span className="wait-text">{entry.waited_min}m / {entry.ceiling_min}m</span>
            {entry.breached && <span className="wait-alert" aria-label="Breached wait time">⚠</span>}
          </div>
          {isResolved && <span className="frozen-tag" title="Clock stopped when status changed">frozen</span>}
        </td>
      </tr>
      {open && (
        <tr className="detail">
          <td colSpan={9}>
            <div className="pipeline">
              <div className="pstep">
                <div className="lbl">Free text</div>
                <div className="val">
                  {entry.note
                    ? entry.extraction_backend === "openai"
                      ? "OpenAI extraction"
                      : "heuristic extraction" + (entry.extraction_status === "ok_heuristic_fallback" ? " (LLM fallback)" : "")
                    : "no note"}
                </div>
              </div>
              <div className="pstep">
                <div className="lbl">XGBoost</div>
                <div className="val">
                  {entry.model_status === "unavailable" ? (
                    <span style={{ color: "var(--band1)" }}>model unavailable — safety fallback (Band {entry.model_recommended_band})</span>
                  ) : (
                    <>
                      P(crit) {entry.critical_probability != null ? entry.critical_probability.toFixed(2) : "—"}
                      <br />
                      sev {entry.severity_score != null ? Math.round(entry.severity_score) : "—"} → band {entry.model_recommended_band ?? "—"}
                      <br />
                      <span title={entry.confidence_reason}>
                        confidence: {entry.confidence_level || "LOW"}
                        {entry.confidence_score != null && ` (${entry.confidence_score.toFixed(2)})`}
                      </span>
                    </>
                  )}
                </div>
              </div>
              <div className={`pstep ${gateFired ? "fired" : ""}`}>
                <div className="lbl">Safety gate</div>
                <div className="val">{gateFired ? `${entry.safety_gate_triggers.length} trigger(s)` : "no escalation"}</div>
              </div>
              <div className="pstep">
                <div className="lbl">Final</div>
                <div className="val">band {entry.final_recommended_band} · {entry.input_completeness} completeness</div>
              </div>
            </div>

            {entry.reassessment_required && (
              <div className="error-box" style={{ display: "block", marginBottom: 10 }}>
                Wait-time safety envelope exceeded — reassessment required (flagged {entry.reassessment_required_at ? formatDateTime(entry.reassessment_required_at) : ""}).
              </div>
            )}

            {entry.note && <div className="note-box">&ldquo;{entry.note}&rdquo;</div>}
            {entry.observed_reported_mismatch && (
              <div style={{ color: "var(--muted)", fontSize: 11, marginTop: -8, marginBottom: 8 }} title="Informational only — not used in risk scoring or the Safety Gate">
                observed/reported mismatch flagged (informational only)
              </div>
            )}

            <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
              <div style={{ flex: 1, minWidth: 260 }}>
                <div className="lbl" style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", marginBottom: 6 }}>
                  Model explanation (SHAP)
                </div>
                {entry.model_explanation && entry.model_explanation.length ? (
                  <ShapBars explanation={entry.model_explanation} />
                ) : (
                  <div style={{ color: "var(--muted)", fontSize: 12 }}>no explanation (model unavailable)</div>
                )}
              </div>

              <div style={{ flex: 1, minWidth: 220 }}>
                <div className="lbl" style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", marginBottom: 6 }}>
                  Safety gate
                </div>
                <div className={`gate-reason ${gateFired ? "" : "clear"}`}>{entry.safety_gate_reason}</div>

                {entry.clinician_decision_band != null && (
                  <div className="clinician-line">
                    On file: clinician logged <b>Band {entry.clinician_decision_band}</b>
                    {entry.clinician_decision_band !== entry.final_recommended_band ? " (override)" : " (confirmed)"}
                    {entry.override_reason ? ` — "${entry.override_reason}"` : ""}
                  </div>
                )}

                <div className="override-form">
                  <select value={clinicianBand} onChange={(e) => setClinicianBand(parseInt(e.target.value, 10))}>
                    {[1, 2, 3, 4, 5].map((b) => (
                      <option key={b} value={b}>
                        Band {b}
                      </option>
                    ))}
                  </select>
                  <input
                    type="text"
                    placeholder="reason code (required for downgrade)"
                    value={reason}
                    onChange={(e) => setReason(e.target.value)}
                    style={error ? { borderColor: "var(--band1)" } : undefined}
                  />
                  <button className="btn secondary" disabled={disableSubmit} onClick={submitOverride}>
                    Log clinician decision
                  </button>
                </div>
                {error && <div className="error-box" style={{ marginTop: 6 }}>{error}</div>}
                {justLogged && !error && (
                  <div style={{ marginTop: 6, color: "var(--band5)", fontSize: 12 }}>
                    ✓ Decision logged.
                  </div>
                )}
              </div>
            </div>

            <div className="disposition-block">
              <div className="lbl" style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", marginBottom: 6 }}>
                ED disposition — current: <StatusBadge status={status} />
              </div>

              {isResolved ? (
                <div className="disposition-resolved">
                  Resolved at {formatDateTime(entry.status_updated_at)}. Wait clock stopped
                  ({entry.waited_min}m of {entry.ceiling_min}m ceiling{entry.breached ? ", breached" : ""}).
                </div>
              ) : (
                <div className="disposition-actions">
                  {actions.map((a) => (
                    <button
                      key={a.value}
                      className="btn secondary"
                      disabled={dispositionSubmitting != null}
                      onClick={() => submitDisposition(a.value)}
                    >
                      {dispositionSubmitting === a.value ? "…" : a.label}
                    </button>
                  ))}
                  <button className="btn secondary" onClick={startReassess}>
                    Reassess
                  </button>
                </div>
              )}
              {dispositionError && <div className="error-box" style={{ marginTop: 6 }}>{dispositionError}</div>}
            </div>

            <div className="history-block">
              <button className="btn secondary" onClick={() => setShowHistory((s) => !s)}>
                {showHistory ? "Hide history" : "Show history"}
              </button>
              {showHistory && <HistoryPanel patientId={entry.patient_id} />}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}