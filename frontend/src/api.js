const STORAGE_KEY = "patienttriage_api_base";
const API_KEY_STORAGE_KEY = "patienttriage_api_key";

export function getApiBase() {
  return localStorage.getItem(STORAGE_KEY) || import.meta.env.VITE_API_BASE || "http://localhost:8000";
}

export function setApiBase(url) {
  localStorage.setItem(STORAGE_KEY, url.replace(/\/$/, ""));
}

// The staff member's own key (never a shared/global secret) -- sent as
// X-API-Key so the backend's role-based access control (auth.py) can tell
// who's asking and log/scope access accordingly. Kept in localStorage only,
// not in source; entering it is a one-time login-style step per browser.
//
// Deliberately no `import.meta.env.VITE_API_KEY` fallback here: Vite inlines
// any VITE_-prefixed env var into the built JS bundle at build time, so a
// real key placed in frontend/.env would ship in plaintext to every visitor
// permanently, not just be readable in one person's local DevTools/storage.
// The in-app "reconnect" field -> localStorage is the only supported path.
export function getApiKey() {
  return localStorage.getItem(API_KEY_STORAGE_KEY) || "";
}

export function setApiKey(key) {
  localStorage.setItem(API_KEY_STORAGE_KEY, key);
}

async function request(path, options = {}) {
  const apiKey = getApiKey();
  const res = await fetch(getApiBase() + path, {
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { "X-API-Key": apiKey } : {}),
    },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {
      // no JSON body
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  modelStatus: () => request("/model/status"),
  getNextPatientId: () => request("/patients/next-id"),
  submitTriage: (payload) => request("/triage", { method: "POST", body: JSON.stringify(payload) }),
  getQueue: () => request("/queue"),
  getAudit: (n = 100) => request(`/audit?n=${n}`),
  getPatient: (patientId) => request(`/patients/${encodeURIComponent(patientId)}`),
  getPatientHistory: (patientId) => request(`/patients/${encodeURIComponent(patientId)}/history`),
  submitOverride: (payload) => request("/override", { method: "POST", body: JSON.stringify(payload) }),
  setDisposition: (payload) => request("/disposition", { method: "POST", body: JSON.stringify(payload) }),
  getSurgeStatus: () => request("/surge/status"),
  simulateSurge: (payload) => request("/surge/simulate", { method: "POST", body: JSON.stringify(payload) }),
  // Live, real-time arrival simulator for the dashboard's "Simulate patient
  // surge" button -- distinct from simulateSurge() above, which is the
  // offline what-if calculator. See surge_simulator.py / POST
  // /surge/simulate-arrivals for what this actually does.
  simulateArrivals: (n_clinicians = 4) =>
    request(`/surge/simulate-arrivals?n_clinicians=${n_clinicians}`, { method: "POST" }),
};

// Mirrors surge.py OPERATIONAL_STATES. Colors reuse the existing band
// palette so CRISIS reads the same visual "danger" language as Band 1.
export const OPERATIONAL_STATE_COLOR = {
  NORMAL: "var(--band5)",
  SURGE: "var(--band3)",
  CRISIS: "var(--band1)",
};

export const BAND_COLOR = {
  1: "var(--band1)",
  2: "var(--band2)",
  3: "var(--band3)",
  4: "var(--band4)",
  5: "var(--band5)",
};

// Mirrors db.ED_DISPOSITIONS on the backend. `waiting` is the default/starting
// state; the other four are clinician-chosen and each one freezes that
// patient's wait clock the moment it's set.
export const DISPOSITIONS = [
  { value: "waiting", label: "Waiting", color: "var(--muted)" },
  { value: "in_treatment", label: "In treatment", color: "var(--band3)" },
  { value: "admitted", label: "Admitted", color: "var(--band4)" },
  { value: "treatment_successful", label: "Treatment successful", color: "var(--band5)" },
  { value: "discharged", label: "Discharged", color: "var(--band5)" },
];

export function dispositionLabel(value) {
  return DISPOSITIONS.find((d) => d.value === value)?.label || value;
}

export function dispositionColor(value) {
  return DISPOSITIONS.find((d) => d.value === value)?.color || "var(--muted)";
}

// The backend stores every timestamp as a UTC ISO string (`datetime.now(timezone.utc).isoformat()`),
// e.g. "2026-09-01T06:40:56.123456+00:00". `new Date(iso)` parses the offset correctly, so
// everything below renders in the *browser's* local timezone -- not the raw UTC clock the API
// stores. Use these everywhere a timestamp is shown instead of slicing the raw ISO string.
export function formatDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// Short "who did this" label for audit/history entries -- e.g. "Dr. J. Rao (clinician)"
// or, when no name is on the API key (old "key:role" format), just "(clinician)".
export function attributionLabel(role, name) {
  if (!role && !name) return null;
  if (name && role) return `${name} (${role})`;
  return name || `(${role})`;
}

export const SYMPTOMS = [
  "chest_pain", "shortness_of_breath", "fever", "confusion", "headache",
  "abdominal_pain", "vomiting", "bleeding", "weakness", "dizziness",
  "cough", "rash", "fall", "back_pain",
];