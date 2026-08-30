import { useCallback, useEffect, useState } from "react";
import { api, BAND_COLOR, OPERATIONAL_STATE_COLOR } from "../api";

const SCENARIOS = [
  { key: 1, label: "1× NORMAL" },
  { key: 2, label: "2× SURGE" },
  { key: 3, label: "3× CRISIS" },
];

const POLICY_NOTES = {
  FIFO: "Fair arrival ordering, but ignores clinical urgency entirely.",
  STATIC_SEVERITY: "Protects high-acuity patients, but can starve low-acuity patients indefinitely under sustained load.",
  WAIT_PROTECTED: "Keeps hard protection for the most urgent patients, while bounding how long lower-acuity patients wait.",
};

function StatCard({ label, value, sub }) {
  return (
    <div className="surge-stat">
      <div className="surge-stat-label">{label}</div>
      <div className="surge-stat-value">{value}</div>
      {sub && <div className="surge-stat-sub">{sub}</div>}
    </div>
  );
}

export default function SurgeDashboard() {
  const [liveStatus, setLiveStatus] = useState(null);
  const [liveError, setLiveError] = useState(null);
  const [multiplier, setMultiplier] = useState(3);
  const [sim, setSim] = useState(null);
  const [loading, setLoading] = useState(false);
  const [simError, setSimError] = useState(null);

  const refreshLive = useCallback(async () => {
    try {
      const d = await api.getSurgeStatus();
      setLiveStatus(d);
      setLiveError(null);
    } catch (e) {
      setLiveError(e.message);
    }
  }, []);

  const runScenario = useCallback(async (mult) => {
    setMultiplier(mult);
    setLoading(true);
    setSimError(null);
    try {
      const d = await api.simulateSurge({ surge_multiplier: mult, duration_min: 180, n_clinicians: 4 });
      setSim(d);
    } catch (e) {
      setSimError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshLive();
    const id = setInterval(refreshLive, 10000);
    return () => clearInterval(id);
  }, [refreshLive]);

  useEffect(() => {
    runScenario(3);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const headline = sim?.operational_state;
  const primary = sim?.policies?.wait_decay;

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
          clinical triage bands are never used to determine ED operational state.
        </div>
      </div>

      <div className="panel" style={{ marginBottom: 20 }}>
        <h2>Surge simulation</h2>
        <div className="toolbar" style={{ marginBottom: 16 }}>
          {SCENARIOS.map((s) => (
            <button
              key={s.key}
              className={`btn secondary ${multiplier === s.key ? "active" : ""}`}
              onClick={() => runScenario(s.key)}
              disabled={loading}
            >
              {s.label}
            </button>
          ))}
        </div>

        {simError && <div className="error-box">Simulation failed: {simError}</div>}

        {headline && (
          <>
            <div className="surge-banner" style={{ borderColor: OPERATIONAL_STATE_COLOR[headline.state] }}>
              <div className="surge-banner-state" style={{ color: OPERATIONAL_STATE_COLOR[headline.state] }}>
                {headline.state}
              </div>
              <div className="surge-banner-sub">
                {headline.load_multiplier}× normal arrival volume ({headline.arrival_rate}/hr vs {headline.baseline_rate}/hr baseline)
              </div>
            </div>

            <div className="surge-stat-grid">
              <StatCard label="Arrival rate" value={`${headline.arrival_rate} / hr`} />
              <StatCard label="Clinicians" value={sim.config.n_clinicians} />
              <StatCard label="Patients served" value={primary?.patients_served} sub="Wait-protected policy" />
              <StatCard label="Avg wait" value={`${primary?.avg_wait_min} min`} />
              <StatCard label="P95 wait" value={`${primary?.p95_wait_min} min`} />
              <StatCard label="Max queue length" value={primary?.max_queue_length} />
              <StatCard label="Safe-wait breaches" value={primary?.safe_wait_breaches} />
              <StatCard label="Reassessments" value={primary?.reassessment_events} />
            </div>

            <div className="note-box" style={{ marginTop: 16 }}>
              Baseline ({sim.config.base_arrivals_per_hour}/hr), safe-wait ceilings, and service-time assumptions are
              PROTOTYPE OPERATIONAL SIMULATION ASSUMPTIONS for this demo — not clinically validated targets. All figures
              above come directly from this simulation run (seed {sim.config.seed}); nothing is hardcoded.
            </div>
          </>
        )}
      </div>

      {sim && (
        <div className="panel" style={{ marginBottom: 20 }}>
          <h2>Queue policy comparison</h2>
          <table className="queue">
            <thead>
              <tr>
                <th>Policy</th><th>Avg wait</th><th>P95 wait</th><th>Safe-wait breaches</th>
                <th>Served</th><th>Left w/o being seen</th>
              </tr>
            </thead>
            <tbody>
              {sim.policy_comparison.map((row) => (
                <tr key={row.policy} className="row">
                  <td>
                    <b className="pid">{row.policy}</b>
                    <div className="vitals">{POLICY_NOTES[row.policy]}</div>
                  </td>
                  <td className="numeric">{row.avg_wait_min} min</td>
                  <td className="numeric">{row.p95_wait_min} min</td>
                  <td className="numeric">{row.safe_wait_breaches}</td>
                  <td className="numeric">{row.patients_served}</td>
                  <td className="numeric">{row.left_without_being_seen}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {sim?.queue_snapshot && (
        <div className="panel" style={{ marginBottom: 20 }}>
          <h2>Queue visualization — wait-protected policy, end of run</h2>
          <table className="queue">
            <thead>
              <tr><th></th><th>Patient</th><th>Band</th><th>Waited / safe-wait limit</th><th></th></tr>
            </thead>
            <tbody>
              {sim.queue_snapshot.length === 0 && (
                <tr><td colSpan={5} className="empty-row">No patients still waiting at the end of this run.</td></tr>
              )}
              {sim.queue_snapshot.map((row) => (
                <tr key={row.patient_id} className="row">
                  <td className="band-cell"><span className="band-bar" style={{ background: BAND_COLOR[row.band] }} /></td>
                  <td className="pid">{row.patient_id}</td>
                  <td><span className="band-pill" style={{ background: BAND_COLOR[row.band], color: "#fff" }}>Band {row.band}</span></td>
                  <td className={`wait ${row.reassessment_required ? "breach" : ""}`}>
                    {row.waited_min} min / {row.ceiling_min} min
                  </td>
                  <td>{row.reassessment_required && <span className="wait-alert" title="Reassessment required">⚠</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

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
