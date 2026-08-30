import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api, getApiBase, setApiBase } from "../api";

export default function Layout() {
  const [connected, setConnected] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [baseInput, setBaseInput] = useState(getApiBase());

  async function checkConn() {
    try {
      const d = await api.modelStatus();
      setConnected(true);
      setModelStatus(d.status);
    } catch (e) {
      setConnected(false);
      setModelStatus(null);
    }
  }

  useEffect(() => {
    checkConn();
    const id = setInterval(checkConn, 5000);
    return () => clearInterval(id);
  }, []);

  function handleReconnect() {
    setApiBase(baseInput);
    checkConn();
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
            style={{
              background: "var(--panel)", border: "1px solid var(--line)", color: "var(--text)",
              fontSize: 12, padding: "5px 8px", borderRadius: 3, width: 200,
            }}
          />
          <button className="btn secondary" onClick={handleReconnect}>reconnect</button>
        </div>
      </header>
      <Outlet />
    </div>
  );
}
