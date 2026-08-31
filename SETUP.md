# SETUP.md — PatientTriage.ai

Everything you need to get the project running from a clean checkout: what each file
does, how to install it, how to train the model, how to run the demo/dashboard, and
how to run the tests.

---

## 1. Requirements

- Python 3.10+ (built and tested on 3.12)
- pip
- Internet access only if you want to (a) `pip install` fresh, or (b) use the real
  OpenAI extraction backend. Everything else runs fully offline.

---

## 2. File structure

For a full description of every file's role, see [`README.md` §4](./README.md#4-repository-structure)
(backend, frontend, tests, generated files, and root config, each
as its own table). Summary of the two independent execution paths:

| Path | Chain | Talks to a DB? |
|---|---|---|
| **Offline demo** | `train.py` → `simulate.py` (console output only) | No — flat JSON/JSONL files only |
| **Live** | `api.py` (+ `frontend/`) | Yes — `outputs/triage.db` |

**Two separate paths, don't mix them up:**
- The offline demo (20 fixed patients) writes `outputs/audit_log.jsonl` and nothing else — no database involved.
- The live path — `api.py` + `frontend/` (React, the only UI) — is what persists real form submissions to
  `outputs/triage.db` and survives restarts. This is what you want for "a nurse enters details and it updates the
  database."

**Dependency order matters**: `train.py` must run before either path (both need the trained `.pkl` models). Beyond
that, `simulate.py` is a standalone console demo, and `api.py` → `frontend/` is its own — you don't need to
run `simulate.py` to use the live path.

---

## 3. Install

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
  backend (§6.6 below); without it, the heuristic extractor is used automatically.
- `TRIAGE_API_KEYS` — only needed to turn on patient-data access control on the live
  API (§6.1a below); without it, the API runs open (fine for a local-only demo).

---

## 4. Train the model

This is the step that actually builds the ML pipeline — synthetic data generation,
baseline comparison, cost-sensitivity sweep, the final calibrated model, and SHAP
importances all happen here.

```bash
python3 train.py
```

What it does, in order:

1. Calls `data_gen.generate_dataset(n=4000)` — synthetic ED patients, age-stratified
   (pediatric/adult/geriatric), with missing vitals and injected label noise. Also
   writes `outputs/synthetic_patients.csv` if you want to inspect the raw data.
2. Runs `features.to_dataframe(...)` to turn every patient into the full feature
   vector (age-conditioned deviations, temporal features, missingness flags).
3. `models.baseline_comparison(X, y)` — trains Logistic Regression, Random Forest, and
   XGBoost with 5-fold cross-validation × 3 random seeds each, and reports recall,
   precision, AUROC, AUPRC, and Brier score for each.
4. `models.cost_sensitivity_sweep(X, y)` — retrains XGBoost at critical-class weights
   2×/4×/6×/10× so you can see the recall/precision trade-off directly (6× is the one
   shipped in the final model).
5. `models.age_aware_ablation(...)` — age-conditioned features vs. naive age-as-a-number.
6. Trains and calibrates the **final** critical-risk model
   (`models.train_final_critical_model`, sigmoid calibration, 5-fold) and the
   **severity** regressor (`models.train_severity_model`).
7. Computes global SHAP feature importance.
8. Writes everything to `outputs/`:
   - `critical_model.pkl`, `severity_model.pkl` — the actual trained models
   - `feature_cols.json` — column order (needed by `pipeline.py` and `api.py` to build
     the right-shaped input row)
   - `training_results.json` — every metric above, in one file

Expect this to take under a minute on a laptop CPU (no GPU needed; XGBoost + SHAP on
4,000 rows is fast). You'll see console output like:

```
Running baseline comparison (LR / RF / XGB)...
Running cost-sensitivity sweep...
Running age-aware ablation...
Training final calibrated critical-risk model...
Computing global SHAP importance...
Done. Results written to outputs/training_results.json
```

**Re-running `train.py` overwrites the models.** If you want a different random seed
or a different critical-class weight than the default 6×, edit the call in `main()`
near the bottom of `train.py` (`train_final_critical_model(X, y, scale_pos_weight=6.0)`)
and re-run.

---

## 5. Run the demo patients + surge simulation

```bash
python3 simulate.py
```

This loads the models you just trained (`outputs/critical_model.pkl`,
`severity_model.pkl`, `feature_cols.json`) and:

- Runs all 20 patients in `patients.py` through the full pipeline (extraction →
  features → model → calibration → Safety Gate), printing each one's final band, the
  model's own recommendation, P(critical), input completeness, which extraction backend
  handled the note, and why the Safety Gate did or didn't escalate.
- Logs one clinician override (a downgrade on P18, with a required reason code) and
  demonstrates that an unexplained downgrade is rejected.
- Runs the 3× surge simulation and the static/FIFO/wait-decay queue policy comparison.
- Prints the tail of the audit log so you can see exactly what got recorded.

This is a console-only sanity check of the pipeline — no dashboard, no database, no
server needed. For a UI, use the live path below.

---

## 6. Live path — nurse enters a patient, it's saved to the database

This is the real, persistent workflow: a nurse fills in a form in the browser, submits
it, and the triage decision is written to `outputs/triage.db` (SQLite) — not held in
memory, not lost on restart.

### 6.1 Start the API

```bash
uvicorn api:app --reload --port 8000
```

On first run this creates `outputs/triage.db` automatically (via `db.init_db()` — no
manual migration step). You should see `INFO: Uvicorn running on http://127.0.0.1:8000`.

### 6.1a (Optional but recommended) Turn on patient-data access control

By default — if `TRIAGE_API_KEYS` is unset in `.env` — the API runs **open**: anyone
who can reach it can read/write patient data. That's fine for a solo local demo, but
turn on access control (`auth.py`) the moment more than one person can reach the
server, or before any submission/review where this matters:

```bash
# in .env (see .env.example for the exact format)
TRIAGE_API_KEYS=nurse-demo-key:nurse:A. Fisher,clinician-demo-key:clinician:Dr. J. Rao,admin-demo-key:admin:M. Otieno
```

Each entry is `key:role:name` — **one key per staff member, not one shared key per
role.** The name is technically optional (`key:role` still parses, for backward
compatibility) but you should always set one: without it, the audit trail can only
prove "a clinician did this," never "which clinician" — exactly the accountability
gap that matters for an overridable, liability-relevant recommendation. Roles:
`nurse` < `clinician` < `admin`.

Restart `uvicorn` after editing `.env` so it picks up the new keys — or, if the
server's already running, call `POST /admin/reload-keys` (admin-only, see §6.5)
instead of restarting it.

Once set:

| Role | Key (from the example above) | Can reach |
|---|---|---|
| `nurse` | `nurse-demo-key` | `POST /triage`, `GET /queue`, `GET /patients/{id}/history` |
| `clinician` | `clinician-demo-key` | everything a nurse can, plus `POST /override`, `POST /disposition`, `GET /patients/{id}` (full record) |
| `admin` | `admin-demo-key` | everything a clinician can, plus `GET /audit`, the `/model/*` fallback toggle, and `POST /admin/reload-keys` |

Every request now needs an `X-API-Key` header with one of these keys; a missing key
gets `401`, an insufficient role gets `403` — both are also written to the audit log
(`patient_record_accessed`, etc.), attributed to the *name* on the key where one is
configured, not just the role, so you can see denied attempts alongside normal
activity (`sqlite3 outputs/triage.db "SELECT * FROM audit_log ORDER BY id DESC LIMIT 10;"`).
Key comparison is constant-time (`hmac.compare_digest`), so a caller can't use
response timing to narrow down a valid key.

Quick check it's working:
```bash
curl -s http://localhost:8000/queue                                    # -> 401, no key
curl -s http://localhost:8000/queue -H "X-API-Key: nurse-demo-key"      # -> 200
curl -s http://localhost:8000/audit -H "X-API-Key: nurse-demo-key"      # -> 403, wrong role
curl -s http://localhost:8000/audit -H "X-API-Key: admin-demo-key"      # -> 200
```

**CORS is scoped, not wildcard.** By default the API only accepts browser requests
from `http://localhost:5173` / `http://127.0.0.1:5173` (the Vite dev server). If your
frontend runs somewhere else, set `TRIAGE_ALLOWED_ORIGINS` in `.env` to a
comma-separated list of the real origin(s) before deploying anywhere beyond a laptop
demo — this matters especially here because the React app persists its API key to
`localStorage` (`frontend/src/api.js`), so a wildcard origin next to a
header-based key is a combination worth avoiding.

### 6.2 Open the intake form

```bash
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (typically `http://localhost:5173`). Three routes: `/` (live
board), `/add-patient` (intake form), `/surge` (surge simulation tab) — all talking to
the API you started in 6.1. The nav bar's connection indicator, "reconnect" field, and
**staff API key field** (only needed if you set `TRIAGE_API_KEYS` in 6.1a) let you
point it at a different API host/port and authenticate at runtime.

**Which key to type in that field:** the live board (`/`) calls both `/queue`
(nurse+) and `/audit` (admin-only) to render the queue table and the audit panel
together — so for full app functionality, enter **`admin-demo-key`**. Admin is the
top of the role hierarchy, so it can do everything `nurse`/`clinician` keys can too
(submit intake, override, view patient detail). If you enter a `nurse` or `clinician`
key instead, the queue table still renders normally, but the audit panel shows
"Audit trail requires an admin key" instead of failing the whole page — useful if you
specifically want to demo what a lower-privilege login sees.

This gives you:

- The top-right field holds the API base URL — defaults to `http://localhost:8000`.
  Change it if you ran uvicorn on a different port or host, then click **reconnect**.
  A green dot means it's talking to the API; red means it can't reach it (check the
  URL and that uvicorn is still running).
- The **Patient ID** field is pre-filled with a suggested next ID (`P-1001`,
  `P-1002`, ... via `GET /patients/next-id`) the moment the form opens for a new
  patient — it's editable, not locked, so you can type over it (e.g. a real MRN, or
  an existing patient's ID to reassess them the old way, by hand). Age is the other
  required field; everything else (vitals, symptoms, note) is optional, matching how
  little may be known for a first-time patient.
- Click **Submit to triage**. The result panel shows the final band, P(critical), data
  quality, which extraction backend handled the note, the Safety Gate's reasoning, and
  the SHAP explanation — the same information a nurse would need to trust or challenge
  the recommendation in the few seconds they have.
- The **live queue** panel on the right polls `GET /queue` every 4 seconds and will
  show the new patient immediately, ordered by band and how long they've been waiting
  against their safe-wait ceiling (breached waits are flagged in red).
- Click a queue row to expand it and log a clinician decision (the override control).
  Downgrading below the AI's recommendation is rejected client-side and server-side
  without a reason code, exactly like `pipeline.record_clinician_decision`.
- The **audit trail** panel polls `GET /audit` and shows every triage submission and
  override as it happens.

### 6.3 Reassessment / deterioration tracking

Two ways to reassess a patient, same underlying mechanism:

- **The "Reassess" button** (in the live queue, on any non-resolved patient's expanded
  row) opens the intake form pre-filled with that patient's `patient_id`, age, gender,
  prior-history flag, and pregnancy status — locked, since those don't change between
  readings — leaving vitals, symptoms, and the note blank for the current reading.
- **Manually** — open a fresh `/add-patient` form and type in the **same Patient ID**
  by hand with new vitals (e.g. worsening SpO2). Works exactly the same way; the button
  is a shortcut, not a different code path.

Either way, the API automatically looks up that patient's prior readings from the
database (`db.get_patient_history`), passes them into the pipeline as temporal
features, and the Safety Gate will fire `deterioration_trend` if things are getting
worse — this is what "monitor patients already in the queue... if vitals are
re-recorded as worsening" from the brief actually looks like end to end. The patient's
original arrival time is preserved across reassessments (so their wait-time ceiling
doesn't reset), and the prior submission is marked `superseded` in the database rather
than deleted, so the full history stays queryable.

### 6.4 Inspecting the database directly

It's a normal SQLite file — no special tooling required:

```bash
sqlite3 outputs/triage.db "SELECT patient_id, final_recommended_band, input_completeness, created_at FROM patients ORDER BY id DESC LIMIT 10;"
sqlite3 outputs/triage.db "SELECT * FROM overrides;"
sqlite3 outputs/triage.db "SELECT * FROM audit_log ORDER BY id DESC LIMIT 10;"
```

Or from Python: `import db; db.get_queue()`, `db.get_patient_detail("P-1042")`,
`db.get_audit_tail(20)`.

### 6.5 API endpoints reference

| Endpoint | Method | Purpose | Minimum role (if `TRIAGE_API_KEYS` is set) |
|---|---|---|---|
| `/triage` | POST | Submit a new patient or a reassessment — runs the full pipeline, writes to `patients` table | `nurse` |
| `/patients/next-id` | GET | Suggests the next auto-generated patient ID (`P-1001`, `P-1002`, ...) for the intake form to pre-fill — a suggestion, not a reservation | `nurse` |
| `/override` | POST | Log a clinician decision — rejects an unexplained downgrade | `clinician` |
| `/queue` | GET | Current waiting patients with real elapsed wait time vs. safe-wait ceiling | `nurse` |
| `/patients/{patient_id}` | GET | Full latest record for one patient, including SHAP explanation | `clinician` |
| `/patients/{patient_id}/history` | GET | Patient's prior vitals only (no note/SHAP) | `nurse` |
| `/disposition` | POST | Move a patient's ED disposition (in treatment, admitted, discharged, ...) | `clinician` |
| `/audit` | GET | Recent audit log entries (`?n=` to control how many) | `admin` |
| `/admin/reload-keys` | POST | Re-reads `TRIAGE_API_KEYS` (fresh `.env` load) without restarting the server | `admin` |
| `/model/status` | GET | Whether the pipeline is using the live model or the rules-only fallback | none (no PII) |
| `/model/unavailable` / `/model/available` | POST | Toggle the safe-fallback path (for testing degraded mode) | `admin` (operational safety control, gated even though it's not a data read) |
| `/surge/status` | GET | Live NORMAL/SURGE/CRISIS operational status from real arrivals, plus the arrival simulator's current/last run | none (no PII) |
| `/surge/simulate` | POST | Offline what-if scenario (`surge_multiplier`, `duration_min`, `n_clinicians`, `seed`) — no DB writes | none (no PII) |
| `/surge/simulate-arrivals` | POST | Starts `surge_simulator.py`'s live arrival simulator, which inserts real rows and stops itself once the real detector leaves NORMAL | none (no PII) |

If `TRIAGE_API_KEYS` is unset, every endpoint is open (no key required) — see §6.1a.

### 6.6 Using the real OpenAI extraction backend

By default, free-text notes are parsed by the heuristic extractor in
`llm_extract.py`. To use a real OpenAI call instead:

```bash
export OPENAI_API_KEY=sk-...
# optional, defaults to gpt-4o-mini:
export OPENAI_MODEL=gpt-4o-mini
```

No code changes needed — restart `uvicorn api:app` after setting the key. If the call
fails for any reason (timeout, auth error, malformed response), the pipeline
automatically falls back to the heuristic extractor rather than failing the request;
you'll see `extraction_backend: "heuristic"` and `extraction_status:
"ok_heuristic_fallback"` in the result when that happens.

---

## 7. Run the tests

```bash
python3 -m pytest tests/ -v
```

You should see 99 tests pass, split across eight files:

- **`test_safety_gate.py`** (10 tests) — the core safety invariant: for every hard
  redline, deterioration trend, low-data-quality case, model-unavailable fallback, and
  structural floor (zero-history, pregnancy, geriatric fall), the gate is only ever
  allowed to move a patient to a *more* urgent band than the model recommended, never
  less.
- **`test_llm_extract.py`** (14 tests) — negation handling ("denies chest pain"),
  multiple symptoms, empty/`None` notes, and — using a mocked OpenAI client so no
  network call is made — the success path, a raised exception (falls back to
  heuristic), malformed JSON (falls back), and a payload with hallucinated/invalid
  keys (gets stripped down to the valid schema).
- **`test_pipeline.py`** (19 tests, requires `train.py` to have been run first since it
  loads the saved models) — missing vitals, invalid values, unexpected extra fields,
  the model-unavailable fallback path capping at Band 3, a zero-history patient, and
  the override reason-code requirement.
- **`test_db.py`** (21 tests, uses a temp SQLite file, does not touch
  `outputs/triage.db`) — insert-then-query, reassessment superseding the previous row
  while keeping the original arrival time, patient history ordering for temporal
  features, queue ordering by band then arrival, override writes updating the latest
  row, audit log ordering, breach flagging computed from real elapsed time, the
  auto-generated patient ID sequence (including that it resets on `reset_demo_db()`
  and ignores the fixed demo set's IDs), and the queue's reassess-prefill fields.
- **`test_queue_sim.py`** (7 tests) — the static/FIFO/wait-decay queue policy
  comparison and the offline what-if surge scenario used by `POST /surge/simulate`.
- **`test_surge.py`** (7 tests) — NORMAL/SURGE/CRISIS classification thresholds and
  the queue-pressure override, and that the operational-state output never leaks a
  clinical field.
- **`test_auth.py`** (11 tests) — `auth.py` in isolation: the `key:role:name` format
  (and its `key:role` backward-compatible form), open-mode fallback when no keys are
  configured, role grants/denials, `reload_keys()` picking up an env change without a
  restart, and the constant-time key match.
- **`test_api_auth.py`** (10 tests, requires `train.py` to have been run first, same as
  `test_pipeline.py`) — the same access control wired into the actual HTTP endpoints
  via FastAPI's `TestClient`: 401/403/200 paths on real routes, `/model/*` and
  `/admin/reload-keys` requiring `admin`, the reload endpoint actually taking effect
  against a live request, the audit trail recording a caller's name (not just role),
  the CORS origin allowlist accepting the configured frontend origin and rejecting
  an unlisted one, and `/patients/next-id` requiring `nurse`.

If `test_pipeline.py` fails with a `FileNotFoundError` on `outputs/critical_model.pkl`,
that means you skipped step 4 — run `python3 train.py` first.

Run a single file or test if you're iterating on one piece:

```bash
python3 -m pytest tests/test_safety_gate.py -v
python3 -m pytest tests/test_db.py::test_reassessment_supersedes_previous_row_and_keeps_arrival_time -v
```

---

## 8. Seeding the live demo (optional, React-first workflow)

If you want the React dashboard to show a realistic queue immediately instead of an
empty board, `reset_and_seed.py` handles it, in two modes:

**Full reset + reseed (default) — for recording a demo video:**

```bash
uvicorn api:app --reload --port 8000 &      # terminal 1, must already be running
python3 reset_and_seed.py                   # terminal 2
```

Wipes `outputs/triage.db` back to empty first (`db.reset_demo_db()` clears `patients`,
`overrides`, and `audit_log`), re-submits all 20 demo patients through the real running
API (so audit entries and `confidence_*` fields are populated exactly as they would be
from the UI), then randomizes arrival times for a realistic breached/not-breached mix.
Safe to re-run before every take. It does **not** delete `outputs/audit_log.jsonl` (the
separate flat-file log) — remove that yourself first if you want a fully blank audit
history too.

**Seed only, no reset — add demo patients without wiping what's already there:**

```bash
uvicorn api:app --reload --port 8000 &      # terminal 1
python3 reset_and_seed.py --seed-only       # terminal 2 — posts all 20 demo patients to /triage, nothing else
```

If you set `TRIAGE_API_KEYS` (§6.1a), `reset_and_seed.py` automatically picks a usable
key out of that same variable — nothing extra to export or configure.

Then open the React app (`cd frontend && npm run dev`) — the queue, band counts, and
audit trail will already be populated. Re-run `python3 fix_demo_arrivals.py` any time
you want to reset the demo to a fresh mixed state without re-seeding.

---

## 9. Repo hygiene — what NOT to commit

This project has generated artifacts mixed in with source files. Before pushing to
GitHub:

**Never commit (now covered by `.gitignore`):**
- `.env` — contains your real `OPENAI_API_KEY` and/or `TRIAGE_API_KEYS`. Only
  `.env.example` (a blank template) should be committed. If a real value was ever
  committed, **rotate it** — treat it as compromised the moment it touches a public
  repo, even if you delete it in a later commit. This applies to `TRIAGE_API_KEYS`
  just as much as `OPENAI_API_KEY`: a leaked staff key gives read access to patient
  records until you change it.
- `__pycache__/`, `*.pyc` — compiled Python, regenerated automatically.
- `frontend/node_modules/`, `frontend/dist/` — installed by `npm install` / built by
  `npm run build`, both fully reproducible from `package.json`.
- `outputs/*.pkl`, `outputs/triage.db`, `outputs/synthetic_patients.csv`,
  `outputs/training_results.json`, `outputs/audit_log.jsonl`,
  `outputs/feature_cols.json` — all regenerated by `train.py` / `simulate.py` /
  `api.py`. `triage.db` in particular is runtime state (whatever you've seeded
  locally) and will just conflict with a collaborator's own local data.

**Keep:** everything else, including `frontend/package-lock.json` (needed for
reproducible `npm install`), `frontend/.env.example`, and `.env.example` — these are
templates/lockfiles, not secrets or build output.

---

## 10. Full clean-run checklist

From a fresh checkout, this is the whole thing end to end:

```bash
cd patient_triage
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # fill in OPENAI_API_KEY and/or TRIAGE_API_KEYS if you want them
python3 train.py
python3 -m pytest tests/ -v

# offline demo path (20 fixed patients, console only) — optional, see README.md §10
python3 simulate.py

# live path (real nurse intake, persisted to SQLite) — the only UI
uvicorn api:app --reload --port 8000 &
python3 reset_and_seed.py          # optional: clean, reseeded queue in one command, see §8
cd frontend && npm install && npm run dev
```

If you only want to look at the offline results without training anything yourself,
the `outputs/` folder shipped in the delivered project already contains a trained
model and a training-results file — steps under §4–5 are only needed if you've deleted
`outputs/` or want to retrain with different settings. The live path (§6) always needs
`uvicorn api:app` running, since it's a real server, not a snapshot.