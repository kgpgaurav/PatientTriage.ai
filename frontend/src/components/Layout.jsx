import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api, getApiBase, setApiBase, getApiKey, setApiKey } from "../api";

export default function Layout() {
  const [connected, setConnected] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [baseInput, setBaseInput] = useState(getApiBase());
  const [keyInput, setKeyInput] = useState(getApiKey());
  const [authError, setAuthError] = useState(false);

  async function checkConn() {
    try {
      const d = await api.modelStatus();
      setConnected(true);
      setModelStatus(d.status);
      setAuthError(false);
    } catch (e) {
      setConnected(false);
      setModelStatus(null);
      // /model/status itself carries no PII and is left open, so a failure
      // here is a real connectivity problem, not an auth problem -- but a
      // subsequent 401/403 from a protected endpoint (queue, patient
      // detail, audit) means the key is missing/wrong, which is worth
      // surfacing distinctly from "server unreachable".
    }
  }

  async function checkAuth() {
    try {
      await api.getQueue();
      setAuthError(false);
    } catch (e) {
      if (/401|403|API key|Role/i.test(e.message || "")) setAuthError(true);
    }
  }

  useEffect(() => {
    checkConn();
    checkAuth();
    const id = setInterval(() => {
      checkConn();
      checkAuth();
    }, 5000);
    return () => clearInterval(id);
  }, []);

  function handleReconnect() {
    setApiBase(baseInput);
    setApiKey(keyInput);
    checkConn();
    checkAuth();
  }

  return (
    <div className="app-shell">
      <header className="nav">
        <div className="brand">
          PATIENTTRIAGE.AI
          <span className="sub">decision-support · escalation-biased · clinician has final say</span>
        </div>
        <nav className="links">
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
            Live board
          </NavLink>
          <NavLink to="/add-patient" className={({ isActive }) => (isActive ? "active" : "")}>
            Add patient
          </NavLink>
          <NavLink to="/surge" className={({ isActive }) => (isActive ? "active" : "")}>
            Surge simulation
          </NavLink>
        </nav>
        <div className="conn">
          <span className={`dot ${connected ? "ok" : "bad"}`}></span>
          <span>
            {connected === null ? "checking..." : connected ? `connected · model ${modelStatus}` : "not connected"}
          </span>
          <input
            type="text"
            value={baseInput}
            onChange={(e) => setBaseInput(e.target.value)}
            title="API base URL"
            style={{
              background: "var(--panel)", border: "1px solid var(--line)", color: "var(--text)",
              fontSize: 12, padding: "5px 8px", borderRadius: 3, width: 170,
            }}
          />
          <input
            type="password"
            value={keyInput}
            onChange={(e) => setKeyInput(e.target.value)}
            placeholder="staff API key"
            title="Your personal access key (X-API-Key) — required to view patient data. Never shared, stored only in this browser."
            style={{
              background: "var(--panel)",
              border: `1px solid ${authError ? "var(--band1)" : "var(--line)"}`,
              color: "var(--text)",
              fontSize: 12, padding: "5px 8px", borderRadius: 3, width: 130,
            }}
          />
          <button className="btn secondary" onClick={handleReconnect}>reconnect</button>
        </div>
      </header>
      {authError && (
        <div className="error-box" style={{ margin: "8px 16px" }}>
          Access denied — missing or insufficient API key for this data. Enter a valid staff key above and reconnect.
        </div>
      )}
      <Outlet />
    </div>
  );
}
