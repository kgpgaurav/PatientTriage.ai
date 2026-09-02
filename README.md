# PatientTriage.ai — Prototype

Decision-support prototype for ED patient triage. Implements the architecture from
the Round 2 design doc: LLM-style free-text extraction → age-conditioned feature
engineering → calibrated XGBoost critical-risk model → deterministic, escalation-only
Safety Gate → clinician review → queue monitoring → full audit trail.

**Assumed jurisdiction: HIPAA (US).** The audit log captures the minimum necessary
fields to reconstruct a decision (input snapshot, feature snapshot, model +
calibration + schema versions, gate triggers, override reason) without storing
free-text notes beyond the extraction step, and treats every override reason code as
part of the legally-discoverable clinical record. A GDPR deployment would additionally
need explicit consent capture and a right-to-erasure path for the audit store — not
implemented here.

> Architecture, results, file roles, and design rationale are covered in the sections
> below. `SETUP.md` in this repo has the same execution instructions in more detail,
> plus troubleshooting notes — this section is the self-contained version for anyone
> reading only this file.

---

## 0. Setup & execution instructions

### 0.1 Requirements

- Python 3.10+ (built and tested on 3.12)
- Node.js + npm (for the React frontend, §10 / §0.6 below)
- pip
- Internet access only if you want to (a) `pip install` fresh, or (b) use the real
  OpenAI note-extraction backend (§0.7). Everything else runs fully offline.

### 0.2 Install

```bash
cd patient_triage
python3 -m venv .venv          # optional but recommended
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`, `fastapi`,
`uvicorn`, `openai`, `pytest`.

Then set up your environment file:

```bash
cp .env.example .env
```

`.env.example` documents two independent, optional settings — leave either blank if
you don't need it:
- `OPENAI_API_KEY` / `OPENAI_MODEL` — only needed for the real OpenAI note-extraction
  backend (§0.7); without it, the heuristic extractor is used automatically.
- `TRIAGE_API_KEYS` — only needed to turn on patient-data access control on the live
  API (§0.5a); without it, the API runs open (fine for a local-only demo).

### 0.3 Train the model

```bash
python3 train.py
```

Generates 4,000 synthetic ED patients, runs the baseline comparison (LR / RF /
XGBoost), a cost-sensitivity sweep, an age-aware ablation, then trains and calibrates
the final critical-risk model and the severity regressor, and computes global SHAP
importance. Takes under a minute on a laptop CPU. Writes to `outputs/`:
`critical_model.pkl`, `severity_model.pkl`, `feature_cols.json`,
`training_results.json`. **This must run before anything else below** — both the
offline demo and the live API load these files.

### 0.4 Run the offline demo (console only, no server/DB)

```bash
python3 simulate.py
```

Runs all 20 patients in `patients.py` through the full pipeline, logs a clinician
override, runs the 3× surge simulation and queue-policy comparison, and prints the
audit log tail. No dashboard, no database — a pure sanity check. Skip straight to §0.5
if you want the UI.

### 0.5 Live path — nurse enters a patient, it's saved to the database

This is the persistent workflow: a nurse fills in a form in the browser, submits it,
and the triage decision is written to `outputs/triage.db` (SQLite) — not held in
memory, not lost on restart.

**Start the API:**
```bash
uvicorn api:app --reload --port 8000
```
On first run this creates `outputs/triage.db` automatically (`db.init_db()`, no
manual migration step). You should see `INFO: Uvicorn running on http://127.0.0.1:8000`.

#### 0.5a (Optional but recommended) Turn on patient-data access control

By default — if `TRIAGE_API_KEYS` is unset in `.env` — the API runs **open**: anyone
who can reach it can read/write patient data. Fine for a solo local demo; turn this on
the moment more than one person can reach the server:

```bash
# in .env
TRIAGE_API_KEYS=nurse-demo-key:nurse:A. Fisher,clinician-demo-key:clinician:Dr. J. Rao,admin-demo-key:admin:M. Otieno
```

Each entry is **`key:role:name`** — one key per staff member, not one shared key per
role. The name is technically optional (`key:role` still parses) but you should always
set one, or the audit trail can only prove "a clinician did this," never "which
clinician." Roles rank `nurse` < `clinician` < `admin` — a higher role can reach
everything a lower one can. Restart `uvicorn` after editing `.env`, or call
`POST /admin/reload-keys` (admin-only) instead of restarting.

Every request now needs an `X-API-Key` header; a missing key gets `401`, an
insufficient role gets `403`. Quick check:
```bash
curl -s http://localhost:8000/queue                                    # -> 401, no key
curl -s http://localhost:8000/queue -H "X-API-Key: nurse-demo-key"      # -> 200
curl -s http://localhost:8000/audit -H "X-API-Key: nurse-demo-key"      # -> 403, wrong role
curl -s http://localhost:8000/audit -H "X-API-Key: admin-demo-key"      # -> 200
```

CORS is scoped, not wildcard — by default only `http://localhost:5173` /
`http://127.0.0.1:5173` (the Vite dev server) is accepted. Set
`TRIAGE_ALLOWED_ORIGINS` in `.env` (comma-separated) if your frontend runs elsewhere.

### 0.6 Open the intake form / dashboard

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`). Three routes: `/` (live
board), `/add-patient` (intake form), `/surge` (surge simulation tab). If you turned on
`TRIAGE_API_KEYS`, enter **`admin-demo-key`** in the nav bar's staff-key field for full
functionality (admin outranks nurse/clinician, so it can do everything they can, plus
see the audit panel) — a nurse/clinician key still works for the queue and intake, it
just can't see the audit panel.

Reassessment: use the **"Reassess"** button on any queue row (pre-fills the locked
fields — ID, age, gender, history, pregnancy — and clears vitals for a new reading), or
manually resubmit the same Patient ID with new vitals from `/add-patient` — both paths
are identical under the hood. The original arrival time is preserved (the wait-time
ceiling doesn't reset), and the prior reading is kept in the database (marked
`superseded`, never deleted), so the full history stays queryable.

### 0.7 Using the real OpenAI extraction backend (optional)

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # optional, this is the default
```
No code changes needed — restart `uvicorn api:app` after setting the key. Without it,
the heuristic extractor in `llm_extract.py` is used automatically, and any OpenAI call
failure (timeout, auth error, malformed response) falls back to it automatically too.

### 0.8 Run the tests

```bash
python3 -m pytest tests/ -v
```
99 tests across eight files (safety gate, LLM extraction, pipeline, DB, queue
simulation, surge classification, auth, and the auth-wired HTTP endpoints). If
`test_pipeline.py`/`test_api_auth.py` fail with `FileNotFoundError` on
`outputs/critical_model.pkl`, run `python3 train.py` first (§0.3).

### 0.9 Seeding the live demo (optional)

To make the dashboard show a realistic queue immediately instead of an empty board:
```bash
uvicorn api:app --reload --port 8000 &      # terminal 1, must already be running
python3 reset_and_seed.py                   # terminal 2 — wipes and reseeds all 20 demo patients
#set "TRIAGE_API_KEYS=admin-demo-key", set this before running reset_and_seed as it needs api key 
```
Add `--seed-only` to add the 20 demo patients without wiping existing data. If
`TRIAGE_API_KEYS` is set, `reset_and_seed.py` automatically picks a usable key from it.

### 0.10 Full clean-run checklist

```bash
cd patient_triage
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # fill in OPENAI_API_KEY and/or TRIAGE_API_KEYS if you want them
python3 train.py
python3 -m pytest tests/ -v

# offline demo path (20 fixed patients, console only) — optional
python3 simulate.py

# live path (real nurse intake, persisted to SQLite) — the only UI
uvicorn api:app --reload --port 8000 &
python3 reset_and_seed.py          # optional: clean, reseeded queue in one command
cd frontend && npm install && npm run dev
```

If you only want to look at the offline results without training anything yourself,
the `outputs/` folder shipped in the delivered project already contains a trained
model and a training-results file — §0.3–0.4 are only needed if you've deleted
`outputs/` or want to retrain with different settings. The live path (§0.5) always
needs `uvicorn api:app` running, since it's a real server, not a snapshot.

### 0.11 Repo hygiene — what NOT to commit

- `.env` — contains your real `OPENAI_API_KEY` and/or `TRIAGE_API_KEYS`. Only
  `.env.example` should be committed. Treat any real value that ever touched a public
  repo as compromised and rotate it, even after deleting it in a later commit.
- `__pycache__/`, `*.pyc`, `frontend/node_modules/`, `frontend/dist/` — all
  regenerated/installed automatically.
- `outputs/*.pkl`, `outputs/triage.db`, `outputs/synthetic_patients.csv`,
  `outputs/training_results.json`, `outputs/audit_log.jsonl`,
  `outputs/feature_cols.json` — all regenerated by `train.py` / `simulate.py` /
  `api.py`; `triage.db` is local runtime state and will conflict with a
  collaborator's own local data.
- **Keep:** `frontend/package-lock.json`, `frontend/.env.example`, `.env.example` —
  templates/lockfiles, not secrets or build output.

---

## Table of contents

0. [Setup & execution instructions](#0-setup--execution-instructions)
1. [Three systems in one repo](#1-three-systems-in-one-repo)
2. [End-to-end flow](#2-end-to-end-flow)
3. [Walking through one submission](#3-walking-through-one-submission)
4. [Repository structure](#4-repository-structure)
5. [Why the LLM is not the decision-maker](#5-why-the-llm-is-not-the-decision-maker)
6. [Safety Gate](#6-safety-gate)
7. [Queue / surge behavior](#7-queue--surge-behavior)
8. [ML results](#8-ml-results-4000-synthetic-patients-5-fold-cv-x-3-seeds)
9. [Demo patients](#9-demo-patients-patientspy-run-via-simulatepy)
10. [Dashboard](#10-dashboard-react)
11. [Live API reference](#11-live-api-apipy)
12. [Data handling notes](#12-data-handling-notes)
13. [Limitations](#13-limitations)
14. [Design decisions / open questions](#14-design-decisions--open-questions)

---

## 1. Three systems in one repo

This project has **three separate systems sharing the same ML pipeline**. Mixing them
up is the most common source of confusion — start here.

| # | System | What it's for | Talks to a database? |
|---|---|---|---|
| 1 | **Offline demo** — `train.py` → `simulate.py` | A fixed, reproducible console demonstration on 20 hand-built patients, for review/grading | No — reads/writes flat JSON/JSONL files only |
| 2 | **Live API + database** — `api.py`, `db.py`, `pipeline.py` | The actual running service — real patients, real persistence | Yes — SQLite, `outputs/triage.db` |
| 3 | **Live frontend** — React app (`frontend/`), **the only UI** | What a nurse actually looks at and types into | No state of its own — it's a client of system 2 |

Systems 2 and 3 together are "the live path." System 1 is untouched by the live path —
it's a separate, self-contained console demo kept for grading/review, independent of
the live path entirely (see [§10](#10-dashboard-react)).

---

## 2. End-to-end flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  BROWSER — React app (frontend/, npm run dev, e.g. localhost:5173)  │
│                                                                       │
│   /add-patient page                    /  (Dashboard page)          │
│   ┌─────────────────────┐              ┌──────────────────────────┐│
│   │ nurse fills in       │              │ live queue table         ││
│   │ vitals + symptoms    │              │ (polls GET /queue every  ││
│   │ + free-text note     │              │  4s)                     ││
│   │                      │              │                          ││
│   │ [Submit to triage]───┼──────┐       │ audit trail              ││
│   └─────────────────────┘      │       │ (polls GET /audit)       ││
│                                  │       │                          ││
│                                  │       │ click a row -> override ││
│                                  │       │ form -> POST /override  ││
│                                  │       └──────────────────────────┘│
└──────────────────────────────────┼───────────────────────────────────┘
                                    │  fetch() over HTTP (CORS open)
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI backend — api.py (uvicorn, e.g. localhost:8000)             │
│                                                                       │
│   POST /triage                                                       │
│     1. db.get_patient_history(patient_id)  <-- pulls prior vitals    │
│        for this patient, if any (for deterioration features)         │
│     2. pipeline.run(record, history=..., note=...)                   │
│          -> llm_extract.extract_from_note()   (OpenAI or heuristic)  │
│          -> features.build_feature_row()      (age-conditioned,      │
│                                                 temporal, missingness)│
│          -> critical_model.predict_proba()    (calibrated XGBoost)   │
│          -> severity_model.predict()                                 │
│          -> safety_gate.apply_safety_gate()   (escalation-only)      │
│          -> models.patient_shap_explanation() (SHAP)                 │
│          -> audit.write_record(...)           (outputs/audit_log.jsonl)│
│     3. db.insert_triage_record(...)  <-- writes to SQLite            │
│     4. db.insert_audit("triage_submitted", ...)                      │
│     5. returns the result JSON to the browser                        │
│                                                                       │
│   GET /queue      -> db.get_queue()       (real elapsed wait time)   │
│   POST /override  -> db.apply_override() + db.insert_audit()         │
│   GET /audit       -> db.get_audit_tail()                            │
│   GET /patients/{id} -> db.get_patient_detail()                      │
│   GET /surge/status -> db.get_live_operational_status()              │
│   POST /surge/simulate -> queue_sim.run_operational_scenario()       │
└──────────────────────────────────┼───────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  outputs/triage.db (SQLite — created automatically on first run)    │
│                                                                       │
│   patients table   — one row per triage submission (append-only;    │
│                       a reassessment marks the old row 'superseded'  │
│                       rather than deleting it)                       │
│   overrides table  — one row per clinician decision                  │
│   audit_log table  — full chronological event log                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Note on the audit trail:** `pipeline.py` calls `audit.write_record()` on *every*
run — live and offline alike — so `outputs/audit_log.jsonl` is written from both
paths. This is separate from (and in addition to) `db.py`'s own `audit_log` SQLite
table, which `api.py` writes to for the live path only. A live `/triage` call is
therefore logged in two places by design.

**The nurse never touches the database, a file, or a script.** Everything they do
goes through the two React pages, which only ever call the FastAPI endpoints. The API
is the only thing that writes to `triage.db`.

---

## 3. Walking through one submission

Say a nurse opens `/add-patient` and enters a 68-year-old with HR 110, SBP 88,
SpO2 91, and types "chest tightness, denies SOB, appears anxious":

1. **React (`AddPatient.jsx`)** builds a JSON payload matching the `PatientInput`
   shape `api.py` expects, and calls `api.submitTriage(payload)` (`frontend/src/api.js`),
   a `fetch()` to `POST {API_BASE}/triage`.
2. **FastAPI (`api.py`)** receives it, calls `db.get_patient_history("P-1042")` — empty,
   since this is a new patient — then calls `pipeline.run(...)`.
3. **`pipeline.py`** runs the note through `llm_extract.extract_from_note()`. If
   `OPENAI_API_KEY` is set, it's a real OpenAI call; otherwise the heuristic
   negation-aware extractor. Either way it comes back with `chest_pain: true`,
   `shortness_of_breath: false` (explicitly negated), and a `mismatch_flag` if the note
   also describes visible distress.
4. **`features.py`** turns the merged record into the full feature vector —
   age-conditioned vital deviations (geriatric SBP range, not adult), missingness
   flags, and (since there's no prior reading) zeroed-out temporal features.
5. **`models.py`**'s calibrated XGBoost model estimates `P(critical) = 0.66`; the
   severity regressor gives a raw score, mapped to a model-recommended band.
6. **`safety_gate.py`** checks hard redlines — SBP 88 is below the hard threshold of
   90 — and forces the band to 1 regardless of what the model said, recording
   `"sbp_low"` as the trigger.
7. **`models.py`**'s SHAP explainer runs against the same feature row, returning the
   top contributing features (`sbp_deviation`, `spo2_abs_deficit`, ...).
8. **`db.py`** writes one row to `patients` (status `waiting`, `arrival_time = now`)
   and one row to `audit_log`.
9. **`api.py`** strips the (not-JSON-safe) internal feature snapshot from the response
   and returns the rest to the browser.
10. **React** renders the Band 1 result panel with the SHAP bars and gate reasoning,
    and — because `Dashboard.jsx` is polling `GET /queue` every 4 seconds — the new
    patient appears in the live board within a few seconds without anyone refreshing
    anything.

If the same nurse later re-enters `P-1042` with worsening vitals: step 2's
`db.get_patient_history` now returns the prior reading, `pipeline.run` passes it in as
`history=[...]`, `features.py` computes real deltas (ΔSpO2, etc.), and if two or more
vitals are trending the wrong way, `safety_gate.py`'s `deterioration_trend` rule fires
on top of whatever the hard redlines already caught. `db.insert_triage_record` marks
the *first* row `superseded` and inserts a new one, but carries the *original*
`arrival_time` forward, so the wait-time ceiling is measured from when the patient
actually arrived, not from the reassessment.

---

## 4. Repository structure

### Backend (Python)

| File | Role |
|---|---|
| `db.py` | SQLite schema + all reads/writes. The only file that touches `triage.db` directly. Also suggests auto-generated patient IDs (`next_patient_id`). |
| `pipeline.py` | Orchestrates one triage decision: extraction → features → model → gate → explanation. Used by both `api.py` (live) and `simulate.py` (offline). |
| `features.py` | Age-conditioned vital deviations, temporal/deterioration deltas, missingness flags. |
| `llm_extract.py` | Free-text → structured symptoms. Real OpenAI call if `OPENAI_API_KEY` is set, heuristic regex/negation extractor otherwise, with automatic fallback on any API failure. |
| `safety_gate.py` | Deterministic, escalation-only rules. Can move a patient to a *more* urgent band than the model said, never less. |
| `models.py` | Trains and evaluates the ML models: Logistic Regression / Random Forest / XGBoost baselines, calibration, cost-sensitivity sweep, SHAP. |
| `data_gen.py` | Generates the synthetic 4,000-patient training set. |
| `queue_sim.py` | Safe-wait ceilings + the static/FIFO/wait-decay policy comparison. Used by the offline `simulate.py` demo *and* by `api.py`'s `POST /surge/simulate`. |
| `surge.py` | `determine_operational_state()` — classifies ED workload as NORMAL/SURGE/CRISIS from arrival rate alone (never clinical data). Used by `api.py` and `db.get_live_operational_status()`. |
| `surge_simulator.py` | Live, real-time arrival simulator behind `POST /surge/simulate-arrivals` — inserts real rows into `triage.db` through the same `pipeline.run()` / `db.insert_triage_record()` path a real submission uses, one at a time, until the live NORMAL/SURGE/CRISIS detector itself reports non-NORMAL (or a safety cap is hit). Distinct from `queue_sim.py`'s offline what-if simulator. |
| `validation.py` | Vitals/age plausibility bounds — enforced at the Pydantic layer in `api.py` (fast 422) and again inside `pipeline.run()` (so direct callers like tests/seeding are covered too). |
| `audit.py` | Append-only JSONL log at `outputs/audit_log.jsonl`. Written on every `pipeline.run()` call — live and offline alike. |
| `auth.py` | Role-based access control (`nurse` < `clinician` < `admin`) for the live API, keyed by the `TRIAGE_API_KEYS` env var — one key *per staff member*, not per role, so every audited action records who did it, not just their role. Runs open (no auth) if that var is unset — see [§11](#11-live-api-apipy) and `SETUP.md` §6.1a for setup and the full endpoint/role table. |
| `patients.py` | The 20 fixed demo patients — used by offline `simulate.py` *and* by `reset_and_seed.py` to populate the live database. |
| `api.py` | FastAPI app — the live HTTP interface described in [§2](#2-end-to-end-flow). |
| `train.py` | STEP 1 of the offline demo — trains + evaluates everything, writes `outputs/*.pkl` + `training_results.json`. Also produces the models `api.py` loads for the live path. |
| `simulate.py` | STEP 2 — runs the 20 demo patients + one override + a 3× surge + policy comparison, printing every result to the console. |
| `fix_demo_arrivals.py` | Convenience script: rewrites `arrival_time` on rows already in `triage.db` so the live queue shows a realistic breached / not-breached mix instead of everyone in the same state. Safe to re-run any time. Also called by `reset_and_seed.py`'s default mode. |
| `reset_and_seed.py` | Convenience script (not part of any automated chain): POSTs all 20 `patients.py` entries to a *running* `uvicorn api:app`, so the live DB / React dashboard has realistic data without manual entry. By default also wipes `patients`/`overrides`/`audit_log` first (`db.reset_demo_db()`) and randomizes arrival times afterward via `fix_demo_arrivals.py`, so every demo take starts from the same clean state; run with `--seed-only` to just POST the demo patients without resetting or randomizing anything. See `SETUP.md` §9. |

### Frontend

| File | Role |
|---|---|
| `frontend/` | **The React app — the only dashboard.** See `SETUP.md` §6.2 for dev commands. |
| `frontend/src/main.jsx` | Entry point, wraps `App` in `BrowserRouter`. |
| `frontend/src/App.jsx` | Route table: `/` → `Dashboard`, `/add-patient` → `AddPatient`, `/surge` → `SurgeDashboard`. All render inside `Layout`. |
| `frontend/src/api.js` | Every backend call in one place, plus the runtime-configurable API base URL (nav bar "reconnect" field, persisted to `localStorage`). |
| `frontend/src/index.css` | Design tokens (CSS variables) + all component classes. |
| `frontend/src/components/Layout.jsx` | Shared nav bar with page links and a live "connected / not connected" indicator against the API. |
| `frontend/src/components/QueueRow.jsx` | One expandable queue row + its override form (enforces the reason-code-on-downgrade rule client-side, same as the backend) + a "Reassess" button that opens the intake form pre-filled for that patient. |
| `frontend/src/components/ShapBars.jsx` | Renders a SHAP explanation array as signed horizontal bars. |
| `frontend/src/pages/Dashboard.jsx` | `/` — live queue, band counts, filters, audit trail. Polls, never mutates except through `QueueRow`'s override form. |
| `frontend/src/pages/AddPatient.jsx` | `/add-patient` — the intake form + result panel. The *only* place a patient gets created. A fresh open suggests the next auto-generated ID (editable); opened via a queue row's "Reassess" button instead, it locks the demographics that don't change between readings and leaves vitals/symptoms/note blank for fresh entry. |
| `frontend/src/pages/SurgeDashboard.jsx` | `/surge` — live operational status, 1×/2×/3× scenario selector, policy-comparison metrics, queue visualization. |
| `frontend/package-lock.json` | Locked dependency versions — commit this for reproducible `npm install`. |
| `frontend/.env.example` | Template for `VITE_API_BASE`; copy to `frontend/.env` to set a build-time default API URL. |

### Tests

| File | Covers |
|---|---|
| `tests/test_safety_gate.py` | The core safety invariant — escalation-only, for every rule. |
| `tests/test_llm_extract.py` | Negation handling, and the OpenAI success/failure/malformed-JSON paths against a mocked client. |
| `tests/test_pipeline.py` | Missing/invalid input, model-unavailable fallback, override reason-code enforcement. |
| `tests/test_db.py` | Persistence, reassessment history, queue ordering, override + audit writes, breach flagging by real elapsed time, auto-generated patient ID sequencing. |
| `tests/test_queue_sim.py` | Queue-policy simulation (FIFO / static-priority / wait-protected). |
| `tests/test_surge.py` | NORMAL/SURGE/CRISIS classification thresholds. |
| `tests/test_auth.py` | `auth.py` in isolation — key parsing (`key:role:name`), role grants/denials, key reload, constant-time match. |
| `tests/test_api_auth.py` | The same access control against the real HTTP endpoints (`TestClient`) — role gates, admin-only operational controls, audit attribution by name, CORS allowlist. |

### Generated (`outputs/`) — not checked in by hand

| File | Produced by |
|---|---|
| `outputs/synthetic_patients.csv` | `data_gen.py`, via `train.py` |
| `outputs/critical_model.pkl` | `train.py` |
| `outputs/severity_model.pkl` | `train.py` |
| `outputs/feature_cols.json` | `train.py` |
| `outputs/training_results.json` | `train.py` |
| `outputs/audit_log.jsonl` | Any `pipeline.run()` call, live or offline |
| `outputs/triage.db` | `api.py`, created automatically on first run |

### Root config

| File | Role |
|---|---|
| `requirements.txt` | Pinned Python dependencies. |
| `.env.example` | Template for `OPENAI_API_KEY` / `OPENAI_MODEL`. Copy to `.env` and fill in your own key — never commit the real `.env`. |
| `.gitignore` | Excludes secrets, build output, and generated files from git (see `SETUP.md` §10). |

---

## 5. Why the LLM is not the decision-maker

`llm_extract.py` turns free text ("pt c/o chest tightness x2 days, denies SOB, appears
anxious") into structured booleans with negation handling, red-flag detection, and an
observed-vs-reported mismatch flag. It never outputs a band.

A small, explicit set of the highest-risk phrases (stroke, facial droop, slurred
speech, unresponsive, active bleeding, seizure, anaphylaxis) is also scanned
deterministically (`deterministic_red_flag_scan`, same negation handling as the rest of
the heuristic layer) on every note, independent of which extraction backend served the
request. This is a narrow safety backstop, not a second triage engine: it exists so
that an LLM extraction failure or omission can never make one of these specific phrases
disappear from the Safety Gate's input. The merged `red_flags` list and a
`red_flag_sources` map (phrase → `llm` / `heuristic` / `deterministic_scan`) are both
written to the audit entry, so it's always possible to tell which path actually caught
a given phrase.

There are two backends behind one interface (`extract_from_note`):

- **OpenAI (`extract_from_note_llm`)** — real call to `client.chat.completions.create`
  (`gpt-4o-mini` by default, override with `OPENAI_MODEL`), JSON-mode response, parsed
  and schema-validated (`_validate_llm_payload` drops any symptom key or type the model
  hallucinates outside the fixed schema). Used automatically whenever `OPENAI_API_KEY`
  is set in the environment.
- **Heuristic (`extract_from_note_heuristic`)** — the original negation-aware
  regex/keyword extractor. Used when no API key is set, or as the automatic fallback
  if the OpenAI call raises anything (timeout, auth error, malformed JSON) — the
  exception is caught, `extraction_status` becomes `"ok_heuristic_fallback"` with the
  error message attached, and the pipeline continues rather than failing the triage.

**Sandbox note:** this development environment's network egress is restricted to a
fixed allowlist (PyPI, GitHub, npm, etc.) and does not include `api.openai.com`, so a
live call couldn't be executed from here. The integration is written and tested
against a mocked `OpenAI` client (`tests/test_llm_extract.py::test_llm_backend_used_and_parsed`,
`test_llm_failure_falls_back_to_heuristic`, `test_llm_malformed_json_falls_back`,
`test_llm_payload_strips_unknown_symptom_keys`) covering the success path, a raised
exception, and malformed/hallucinated output. To actually run it live: set
`OPENAI_API_KEY` in `.env` (copy `.env.example` first) and re-run `simulate.py` or hit
`POST /triage` in an environment with normal internet access — no code changes needed,
the pipeline picks up the key automatically and prints which backend served each
patient (`extraction=[openai/ok]` vs `extraction=[heuristic/ok]`).

---

## 6. Safety Gate

Deterministic, checked after the model, allowed only to move a patient to a *lower*
band number (more urgent) than the model recommended:

- **Hard redlines:** SpO2 < 90, SBP < 90, HR > 150 or < 45, RR outside 9–32, red-flag
  phrases from the note, active bleeding, altered mental status → forces Band 1
- **Deterioration:** ≥2 worsening vitals across repeat readings → escalates one band
- **Low input completeness** with residual risk → escalates one band
- **Model unavailable:** hard-caps at Band 3 (`FALLBACK_BAND`) and marks fallback
  status explicitly (`recommendation_mode: "safety_fallback"`, `critical_probability:
  null`, `severity_score: null`) — the UI never presents Band 3 in fallback mode as if
  it were an XGBoost prediction
- **Structural floors:** zero-history + weak signal, pregnancy + bleeding/abdominal
  pain, geriatric fall → floor of Band 3

`tests/test_safety_gate.py` asserts the gate can never move a patient to a *less*
urgent band than the model output, for every hard rule and every model band.

---

## 7. Queue / surge behavior

Safe-wait ceilings: Band 1 = 5 min, Band 2 = 10, Band 3 = 30, Band 4 = 60, Band 5 = 120.
`queue_sim.py` simulates Poisson arrivals; at 3× normal volume over 3 hours, breach
events roughly triple versus normal load (see `simulate.py` output).

`compare_queue_policies()` actually re-runs (not narrates) a static-priority vs FIFO vs
wait-decay comparison across normal / 2× / 3×-crisis load with 4 clinicians. In this
re-implementation, the finding is sharper than the design doc's version: under extreme
sustained overload, pure static-priority ordering starves low-acuity patients badly —
only 46 of 160 arrivals get seen at all (vs. FIFO's 89 and the hybrid's 75) — because a
steady stream of higher-band arrivals can indefinitely postpone anyone lower down the
queue. FIFO and the wait-decay hybrid both complete meaningfully more patients under
crisis than static priority alone.

This does **not** mean the deployed design should switch to FIFO — FIFO ignores
severity entirely, which is unacceptable for a genuinely critical patient arriving
late in a surge. It means ordering policy alone cannot both (a) always serve the
sickest patient next and (b) bound how long anyone waits under sustained overload —
those two goals conflict once demand exceeds capacity for long enough. That's why the
production design keeps severity-first ordering for who's served next, but adds the
independent safe-wait ceiling + mandatory reassessment loop (already implemented in
`TriageQueue.check_breaches`) as the actual backstop against starvation, rather than
trying to solve it through queue-ordering rules alone.

### The operational (surge) layer is separate from clinical triage

```
Patient clinical data → triage pipeline → ML → Safety Gate → clinical band
                                                                    │
                                                                    ▼
                                                            queue manager
                                                                    │
                                                    NORMAL / SURGE / CRISIS
```

- `surge.py` — `determine_operational_state()` classifies ED workload as
  `NORMAL` / `SURGE` / `CRISIS` from arrival rate (and, secondarily, queue depth
  relative to clinician count) alone. It never looks at a patient's vitals, symptoms,
  or triage band. Baseline (20/hr) and the multiplier thresholds (≤1.25× / ≤2× / >2×)
  are **prototype simulation assumptions**, not a validated capacity-planning model —
  a real deployment would derive these per-site.
- `queue_sim.py` — offline what-if simulator (`run_operational_scenario`) used by the
  surge demo screen. Arrivals are a stochastic Poisson process, not uniform spacing.
  Three scheduling policies are compared:
  - `FIFO` — arrival order only.
  - `STATIC_SEVERITY` — clinical band strictly first; can starve low-acuity patients
    under sustained load.
  - `WAIT_PROTECTED` — clinical band first, but Bands 3–5 accrue bounded wait-time
    scheduling-priority credit the longer they wait, to prevent starvation.
    **Bands 1–2 are hard-protected**: no amount of another patient's waiting can move
    them ahead of a real Band 1/2 patient. This changes *scheduling priority* only; it
    never rewrites `clinical_band`, which is stored and reported separately from queue
    position throughout.
- `db.get_live_operational_status()` — the same NORMAL/SURGE/CRISIS classification,
  computed live from the real running app's actual patient arrivals and real queue
  (not the simulator), exposed at `GET /surge/status`. Emits `surge_state_changed` and
  `queue_threshold_reached` audit events on transitions, and `reassessment_required`
  alongside the existing `wait_breach_detected` event when a patient crosses their
  safe-wait ceiling.
- `POST /surge/simulate` runs the offline scenario (`surge_multiplier`, `duration_min`,
  `n_clinicians`, `seed`) and returns operational-state classification, per-policy
  metrics (avg/P95 wait, max queue length, throughput, safe-wait breaches,
  reassessment events), a policy comparison table, and an end-of-run
  queue-visualization snapshot — every number comes from that run, nothing is
  hardcoded. The default seed (`0`) is fixed only so a recorded demo is repeatable;
  pass a different seed (or `null`) for a fresh random draw. All of
  `SAFE_WAIT_MINUTES`, the band mix, and the per-band service-time assumptions used by
  the simulator are prototype demo assumptions, not clinical or staffing standards,
  and are labeled as such in code and in the UI.
- The Safety Gate, XGBoost models, feature engineering, and `SAFE_WAIT_MINUTES`
  ceilings themselves are **unchanged by surge state** — see `safety_gate.py` /
  `pipeline.py`, neither of which imports `surge.py` or `queue_sim.py`. A patient's
  safe-wait breach still routes through the same repeat-vitals → features → ML →
  Safety Gate reassessment pipeline used for any other reassessment, surge or not.
- Frontend: the "Surge simulation" tab (`frontend/src/pages/SurgeDashboard.jsx`) shows
  live operational status, a 1×/2×/3× scenario selector, the resulting metrics and
  policy comparison, a queue visualization, and a short explanation panel reiterating
  that clinical thresholds are unchanged.

---

## 8. ML results (4,000 synthetic patients, 5-fold CV × 3 seeds)

> **Corrected 2026-09-02.** An earlier run of this benchmark had a label-imputation
> artifact: the synthetic generator scored a patient's ground-truth severity using `0`
> for any vital that was later marked missing, which trips several hard severity
> thresholds by construction and inflated the apparent critical rate and recall. The
> fix computes the label from the complete, pre-missingness vitals instead
> (`data_gen.py`, `generate_dataset()`) — see `FIX_LOG_label_imputation_artifact.md`
> for the full root cause, before/after numbers, and the ablation proving the leak is
> closed. **The numbers below are the corrected, post-fix numbers.**

| Model | Recall (critical) | Precision | AUPRC | Brier |
|---|---|---|---|---|
| Logistic Regression (balanced) | 0.83 | 0.50 | 0.716 | 0.106 |
| Random Forest (balanced) | 0.69 | 0.61 | 0.691 | 0.093 |
| XGBoost (6× critical weight) | **0.75** | 0.55 | 0.710 | 0.087 |

**Honest finding:** XGBoost's AUPRC is not meaningfully better than a plain balanced
logistic regression on this synthetic dataset — the gain from XGBoost shows up in
recall once its class weight is pushed to 6× (deliberately trading precision for
fewer missed critical cases), not in ranking quality. This is reported rather than
hidden, per the brief.

Cost-sensitivity sweep (critical-class weight):

| Weight | Recall | Precision | FN rate |
|---|---|---|---|
| 2× | 0.64 | 0.67 | 0.36 |
| 4× | 0.71 | 0.59 | 0.29 |
| **6× (chosen)** | **0.75** | 0.55 | **0.25** |
| 10× | 0.80 | 0.50 | 0.20 |

6× is used as the deployed operating point: it materially cuts false negatives
relative to 2×/4×, while 10×'s extra recall (5 points) costs 5 points of precision —
judged not worth the added over-triage. This should be revisited with real outcome
data and clinician input, not treated as final.

Age-aware ablation: age-conditioned features improve AUPRC (0.710 vs 0.625) and AUROC
(0.922 vs 0.888) over treating age as a plain numeric feature — a larger and more
consistent gap than the pre-correction run showed, in line with the design doc's
expectation, though it should still be treated as directional rather than conclusive
given synthetic subgroup sizes.

Subgroup recall at threshold 0.5: pediatric 0.81 (n=651), adult 0.83 (n=2,357),
geriatric 0.83 (n=992) — no subgroup was left materially behind.

Top global SHAP drivers: `mental_status_altered`, `spo2_abs_deficit`, `bleeding`,
`hr_deviation`, `sbp_deviation`, `rr_deviation`, `age`, `spo2_deviation` — consistent
with the hand-built Safety Gate's own priorities, which is the sanity check the design
doc calls for. Notably, **missingness flags no longer appear in the top drivers** —
under the buggy generator they did, which was itself a symptom of the label leak;
their absence now is further confirmation the leak is closed, not just reduced (the
ablation in `FIX_LOG_label_imputation_artifact.md` §6 shows less than a one-point
recall/AUPRC difference with vs. without the missingness features).

---

## 9. Demo patients (`patients.py`, run via `simulate.py`)

20 patients covering: an ambiguous presentation with a self-report/observed mismatch
(P01), pediatric fever (P02), a geriatric fall (P03), a zero-history patient with only
a mild complaint (P04 / P17), a zero-history unresponsive patient (P11), a pregnancy +
bleeding case (P06), a stroke red-flag case (P14), a deteriorating-vitals case with
repeat readings (P13), and a maximal-severity case (P20). Every run writes a full audit
entry; one clinician downgrade (P18) is captured with its reason code, and a second
attempted downgrade with no reason code is shown being rejected.

---

## 10. Dashboard (React)

`frontend/` (React) is the only dashboard. Band-colored queue rows, an
input-completeness badge, P(critical), model band vs. gate-escalated final band, a
working override control, live polling of `/queue` and `/audit`, and the `/surge`
operational tab. It's a client of the live API (system 2 in [§1](#1-three-systems-in-one-repo))
and reflects real, persisted data.

An earlier static-HTML offline dashboard (`bake_dashboard.py` / `dashboard_template.html`)
and a zero-dependency HTML fallback intake page (`frontend_nurse_intake.html`) have both
been removed now that `frontend/` is the only frontend. `simulate.py` ([§9](#9-demo-patients-patientspy-run-via-simulatepy))
still runs the same 20-patient scenario as a console-only script, independent of any
dashboard, if you want a no-server sanity check of the pipeline.

---

## 11. Live API (`api.py`)

`uvicorn api:app` exposes the actual pipeline, not a replay of pre-computed results.
Every endpoint except `/model/status` and the `/surge/*` trio is gated by `auth.py`'s
role-based access control — see the "Minimum role" column. `/model/unavailable` and
`/model/available` carry no PII but are gated at `admin` anyway, since they're an
operational safety control (they can force the whole pipeline into rules-only
fallback), not a data read. If `TRIAGE_API_KEYS` is unset, the API runs open and no
key is required (local-demo default); set it before any shared/real deployment. Full
setup: `SETUP.md` §6.1a.

| Endpoint | Method | Purpose | Minimum role |
|---|---|---|---|
| `/triage` | POST | Submit a new patient or a reassessment — runs the full pipeline, writes to `patients` table | `nurse` |
| `/patients/next-id` | GET | Suggests the next auto-generated patient ID (`P-1001`, `P-1002`, ...) for the intake form to pre-fill — a suggestion, not a reservation | `nurse` |
| `/override` | POST | Log a clinician decision — rejects an unexplained downgrade | `clinician` |
| `/queue` | GET | Current waiting patients with real elapsed wait time vs. safe-wait ceiling | `nurse` |
| `/patients/{patient_id}` | GET | Full latest record for one patient, including SHAP explanation | `clinician` |
| `/patients/{patient_id}/history` | GET | Patient's prior vitals only (no note/SHAP) | `nurse` |
| `/disposition` | POST | Move a patient's ED disposition (in treatment, admitted, discharged, ...) | `clinician` |
| `/audit` | GET | Recent audit log entries (`?n=` to control how many) | `admin` |
| `/admin/reload-keys` | POST | Re-reads `TRIAGE_API_KEYS` (via a fresh `.env` load) without a server restart, so a rotated/revoked key takes effect immediately | `admin` |
| `/model/status` | GET | Whether the pipeline is using the live model or the rules-only fallback | none |
| `/model/unavailable` / `/model/available` | POST | Toggle the safe-fallback path (for testing degraded mode) — an operational control, gated even though it touches no PII | `admin` |
| `/surge/status` | GET | Live NORMAL/SURGE/CRISIS classification from real arrivals | none |
| `/surge/simulate` | POST | Offline what-if scenario (multiplier, duration, clinician count, seed) | none |
| `/surge/simulate-arrivals` | POST | Starts the live, real-time arrival simulator (`surge_simulator.py`) that feeds the real NORMAL/SURGE/CRISIS detector | none |

This is the piece that matches the design doc's "Frontend → API → LLM extraction →
features → XGBoost → calibration → Safety Gate → clinician → audit" request flow as a
runnable service rather than only as an architecture diagram.

---

## 12. Data handling notes

A few field semantics that are easy to misread, documented explicitly here per the
project's own audit-trail philosophy:

- **Model probability.** The saved `critical_model` is a single
  `sklearn.CalibratedClassifierCV` (base XGBoost folds + sigmoid calibrators, fit and
  averaged together). There is no separately-addressable "raw, pre-calibration"
  probability behind it — `critical_probability` is the calibrated output, and it's the
  same number the Band-1 threshold and the audit entry both use. The audit field is
  named `calibrated_probability` (with `probability_stage: "calibrated_only"`)
  precisely so it doesn't imply a second, different quantity exists.
- **Age group.** `age_group` is always derived server-side from `age`
  (`data_gen.age_group`: <13 pediatric, <65 adult, else geriatric) and used for both
  feature engineering and the Safety Gate's geriatric-fall floor. A client-supplied
  `age_group` is accepted on the request only for backward compatibility with older
  clients — it is never trusted, and a contradictory value is silently overridden (with
  `age_group_overridden: true` returned so the caller knows the value it sent was
  replaced).
- **Input completeness.** `input_completeness` (`HIGH`/`MEDIUM`/`LOW`, called
  `data_quality` before this pass) is a count of missing intake fields, nothing more.
  It is not model confidence and not clinical certainty — a `LOW`-completeness,
  high-probability patient is exactly the case the Safety Gate treats cautiously
  (`low_data_quality_uncertainty` — the audit trigger name itself is left unchanged for
  continuity with historical audit logs, even though the field it reads is renamed).
- **Observed/reported mismatch.** `observed_reported_mismatch` is extracted from the
  note but is informational only — it is not in `feature_cols.json` and does not affect
  `model_recommended_band` or any Safety Gate rule. The synthetic training data has no
  ground-truth label correlating this signal with outcome, so adding it to the model
  would add noise, not signal, until real labeled data exists.
- **Physiological validation.** `validation.py` rejects vitals/age outside broad
  plausibility bounds (documented in that file) at both the Pydantic layer (fast 422)
  and inside `pipeline.run()` itself (so direct callers — tests, seeding — are covered
  too). These bounds are deliberately wide: a real but extreme value (e.g. SpO2 82) is
  never rejected, only flows through to the Safety Gate as intended; only physically
  impossible values (e.g. SpO2 250, negative HR) are rejected. Missing values (`null`)
  are always distinct from invalid ones and are never coerced to zero.
- **Wait-time breach → reassessment.** Exceeding a patient's safe-wait ceiling sets an
  explicit, persisted `reassessment_required` flag (with `reassessment_required_at`)
  and writes a `wait_breach_detected` audit event, rather than only being a derived
  `breached` boolean recomputed on each poll. Submitting a new `/triage` reading for
  that patient (a reassessment) re-runs the full pipeline against the *same, preserved*
  `arrival_time`, and a `reassessment_performed` audit event is written. The flag only
  clears if the freshly re-run pipeline no longer puts that patient in breach — a
  reassessment that still comes back critical and still exceeds its ceiling stays
  flagged, by design.

---

## 13. Limitations

- All patient data is synthetic; results are a controlled-development benchmark, not
  clinical validation.
- Ground truth is generated from a scoring function independent of the Safety Gate's
  rules, but it is still a simulated proxy for real clinical outcomes.
- The age-aware vs. naive comparison is directionally positive here but should not be
  treated as conclusive given synthetic subgroup sizes.
- SpO2 is weighted heavily because desaturation is a key safety signal, but pulse
  oximetry has known population-dependent accuracy limitations that need real-world,
  subgroup-stratified validation before deployment.
- The free-text extraction layer defaults to a heuristic when no `OPENAI_API_KEY` is
  set; the real OpenAI path is implemented and unit-tested against a mocked client but
  was not exercised against the live API in this environment (network egress here is
  restricted to a fixed allowlist that excludes `api.openai.com`) — a production
  deployment should run it live and add hallucination/robustness testing beyond the
  schema validation already in place.
- No real EHR, bed-management, or staffing-roster integration; the pipeline assumes
  data arrives in the shapes `features.py` expects.
- **Access control is prototype-scope, by design.** `auth.py` uses long-lived static
  API keys (one per staff member, with a name attached for audit accountability), not
  real IAM/SSO with short-lived tokens — see that file's own docstring. It also has no
  per-patient/department scoping (any valid `clinician` key can read any patient
  system-wide) and no automated response to repeated auth failures beyond logging them
  (`db.insert_audit`). A production deployment should replace the key store with
  OAuth/OIDC-issued tokens, add relationship- or department-based access, and add
  rate-limiting/alerting on repeated denials.
- This is a decision-support prototype, not an autonomous or clinically validated
  triage system, and should not be represented as one.

---