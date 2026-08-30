import { useCallback, useEffect, useState } from "react";
import { api, BAND_COLOR } from "../api";
import QueueRow from "../components/QueueRow";

export default function Dashboard() {
  const [queue, setQueue] = useState([]);
  const [audit, setAudit] = useState([]);
  const [filter, setFilter] = useState("all");
  const [loadError, setLoadError] = useState(null);
  const [auditError, setAuditError] = useState(null);

  const refresh = useCallback(async () => {
    // Queue and audit are fetched independently, not via Promise.all: with
    // role-based access control (auth.py) on, a nurse/clinician key can read
    // the queue but not /audit (admin-only) -- one 403 on the audit call
    // must not take down the whole board.
    try {
      const q = await api.getQueue();
      setQueue(q.summary);
      setLoadError(null);
    } catch (e) {
      setLoadError(e.message);
    }
    try {
      const a = await api.getAudit(15);
      setAudit(a);
      setAuditError(null);
    } catch (e) {
      setAudit([]);
      setAuditError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
  }, [refresh]);
  const counts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
  queue.forEach((p) => { counts[p.final_recommended_band] = (counts[p.final_recommended_band] || 0) + 1; });

  const filtered = queue.filter((p) => {
    if (filter === "breached") return p.breached;
    if (filter === "reassess") return p.reassessment_required;
    if (filter === "low") return p.input_completeness === "LOW";
    return true;
  });

  return (
    <div>
      <div className="counts">
        {[1, 2, 3, 4, 5].map((b) => (
          <div className="count-chip" key={b}>
            <span className="sw" style={{ background: BAND_COLOR[b] }} />
            Band {b} <b>{counts[b] || 0}</b>
          </div>
        ))}
      </div>

      <div className="toolbar">
        <button className={`btn secondary ${filter === "all" ? "active" : ""}`} onClick={() => setFilter("all")}>
          All patients
        </button>
        <button className={`btn secondary ${filter === "breached" ? "active" : ""}`} onClick={() => setFilter("breached")}>
          Breached wait-time only
        </button>
        <button className={`btn secondary ${filter === "reassess" ? "active" : ""}`} onClick={() => setFilter("reassess")}>
          Reassessment required
        </button>
        <button className={`btn secondary ${filter === "low" ? "active" : ""}`} onClick={() => setFilter("low")}>
          Low input completeness
        </button>
      </div>

      {queue.some((p) => p.reassessment_required) && (
        <div className="error-box" style={{ display: "block", marginBottom: 12 }}>
          {queue.filter((p) => p.reassessment_required).length} patient(s) have exceeded their safe-wait ceiling and require reassessment.
        </div>
      )}

      {loadError && <div className="error-box">Can't reach the API: {loadError}</div>}

      <table className="queue">
        <thead>
          <tr>
            <th></th><th>Patient</th><th>Vitals</th><th>Completeness</th><th>P(critical)</th>
            <th>Model band</th><th>Final band</th><th>Status</th><th>Waited</th>
          </tr>
        </thead>
        <tbody>
          {filtered.length === 0 && (
            <tr><td colSpan={9} className="empty-row">Queue is empty — add a patient from the "Add patient" page.</td></tr>
          )}
          {filtered.map((entry) => (
            <QueueRow key={entry.patient_id} entry={entry} onOverridden={refresh} />
          ))}
        </tbody>
      </table>

      <div className="audit-panel panel">
        <h2>Audit trail</h2>
        {auditError ? (
          <div className="error-box" style={{ display: "block" }}>
            Audit trail requires an admin key ({auditError}).
          </div>
        ) : (
          <div className="audit-list">
            {[...audit].reverse().map((e, i) => (
              <div key={i}>
                {(e.created_at || "").slice(11, 19)} — <b>{e.event_type}</b> {e.patient_id}
                {e.final_recommended_band != null && `: band ${e.final_recommended_band}`}
                {e.clinician_decision_band != null && `: AI ${e.ai_recommendation_band} → clinician ${e.clinician_decision_band}`}
                {e.override_reason && ` ("${e.override_reason}")`}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}