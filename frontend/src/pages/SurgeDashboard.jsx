import { useCallback, useEffect, useState } from "react";
import { api, OPERATIONAL_STATE_COLOR } from "../api";

export default function SurgeDashboard() {
  const [liveStatus, setLiveStatus] = useState(null);
  const [liveError, setLiveError] = useState(null);

  const refreshLive = useCallback(async () => {
    try {
      const d = await api.getSurgeStatus();
      setLiveStatus(d);
      setLiveError(null);
    } catch (e) {
      setLiveError(e.message);
    }
  }, []);

  useEffect(() => {
    refreshLive();
    const id = setInterval(refreshLive, 10000);
    return () => clearInterval(id);
  }, [refreshLive]);

  return (
    <div>
      <div className="panel" style={{ marginBottom: 20 }}>
        <h2>Live ED operational status</h2>
        {liveError && <div className="error-box">Can't reach the API: {liveError}</div>}
        {liveStatus && (
          <div className="surge-live-row">
            <span
              className="surge-state-chip"
              style={{ color: OPERATIONAL_STATE_COLOR[liveStatus.operational_state.state], borderColor: OPERATIONAL_STATE_COLOR[liveStatus.operational_state.state] }}
            >
              {liveStatus.operational_state.state}
            </span>
            <span className="surge-live-detail">
              {liveStatus.operational_state.arrival_rate}/hr arrivals (baseline {liveStatus.operational_state.baseline_rate}/hr,
              {" "}{liveStatus.operational_state.load_multiplier}×) · queue {liveStatus.queue_length} · max queue observed {liveStatus.max_queue_length}
              {" "}· {liveStatus.safe_wait_breaches} awaiting reassessment
            </span>
          </div>
        )}
        <div className="note-box" style={{ marginTop: 12, marginBottom: 0 }}>
          This reflects real patients currently in the live queue. It is computed purely from arrival timing and wait times —
          clinical triage bands are never used to determine ED operational state. State changes here are automatic; there is
          no manual NORMAL/SURGE/CRISIS selector. To see this respond to a real load spike, use "Simulate patient surge" on
          the live board.
        </div>
      </div>

      <div className="panel">
        <h2>Surge mode — what changes and what doesn't</h2>
        <div className="result-line">
          <b>Clinical triage thresholds remain unchanged during surge conditions.</b> The Safety Gate, XGBoost models, and
          feature engineering run identically in NORMAL, SURGE, and CRISIS — surge state is never an input to a clinical
          recommendation. The system adapts queue management instead:
        </div>
        <ul className="result-line" style={{ marginTop: 8, paddingLeft: 20 }}>
          <li>Critical patients (Band 1–2) remain hard-protected in scheduling order, regardless of how long anyone else has waited.</li>
          <li>Waiting-time limits are continuously monitored against each patient's safe-wait ceiling.</li>
          <li>Lower-acuity patients receive bounded, starvation-prevention priority credit as they wait longer — this changes scheduling priority only, never their clinical band.</li>
          <li>Patients exceeding their safe-wait limit are flagged for mandatory reassessment.</li>
          <li>New or worsening repeat vitals trigger full re-triage through the same ML + Safety Gate pipeline used for any patient.</li>
        </ul>
      </div>
    </div>
  );
}