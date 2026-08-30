# PatientTriage.ai — Prototype

Decision-support prototype for ED patient triage. Implements the architecture from
the Round 2 design doc: LLM-style free-text extraction → age-conditioned feature
engineering → calibrated XGBoost critical-risk model → deterministic, escalation-only
Safety Gate → clinician review → queue monitoring → full audit trail.

**Assumed jurisdiction: HIPAA (US).** The audit log therefore captures the minimum
necessary fields to reconstruct a decision (input snapshot, feature snapshot, model
+ calibration + schema versions, gate triggers, override reason) without storing
free-text notes beyond the extraction step, and treats every override reason code as
part of the legally-discoverable clinical record. A GDPR deployment would additionally
need explicit consent capture and a right-to-erasure path for the audit store — not
implemented here.

## Layout

```
data_gen.py       synthetic patient generator (~4,000 patients)
features.py       age-conditioned / temporal / missingness feature engineering
llm_extract.py    free-text extraction (negation-aware heuristic; swap-in point for a real LLM call)
safety_gate.py    deterministic, escalation-only rule layer
models.py         LR/RF/XGBoost baselines, calibration, CV, SHAP, cost-sensitivity sweep
queue_sim.py      safe-wait ceilings, surge simulation, static/FIFO/wait-decay policy comparison
audit.py          append-only JSONL audit log
pipeline.py       end-to-end orchestration + clinician override capture
patients.py       20 hand-built demo patients (ambiguous / pediatric / geriatric / zero-history / deterioration)
train.py          trains + evaluates everything, writes outputs/training_results.json
simulate.py       runs the 20 demo patients, one override, a 3x surge + policy comparison, writes outputs/dashboard_data.json
api.py            FastAPI live service — the actual request path from section 24 (POST /triage, /override, GET /queue, /audit)
dashboard_template.html   static triage board UI, baked with the simulate.py snapshot into outputs/dashboard.html
requirements.txt  pinned dependencies
tests/            pytest suite for the safety invariants, extraction, and pipeline robustness
```

Run in order:
```
pip install -r requirements.txt
python3 train.py       # trains models, writes outputs/*.pkl + training_results.json
python3 simulate.py    # runs the 20 demo patients + surge + queue-policy experiment, writes outputs/dashboard_data.json
python3 -m pytest tests/ -q
```
Then open `outputs/dashboard.html` directly in a browser — it's a self-contained board built from the simulate.py
snapshot (click a row to expand the SHAP explanation, the safety-gate reasoning, and a working override form that
enforces the reason-code-on-downgrade rule).

For the live path instead of the static snapshot: `uvicorn api:app --reload`, then `POST /triage` with a patient
record — this runs the real trained model + calibration + Safety Gate + audit write on each call, exactly as
described in section 24 of the design doc, rather than replaying pre-computed results.

To use the real OpenAI extraction backend instead of the heuristic fallback: `export OPENAI_API_KEY=sk-...`
(optionally `export OPENAI_MODEL=gpt-4o-mini`, the default) before running `simulate.py` or the API — nothing
else changes, the pipeline detects the key automatically. See "Why the LLM is not the decision-maker" below.

## Why the LLM is not the decision-maker

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
exception, and malformed/hallucinated output. To actually run it live: `export
OPENAI_API_KEY=sk-...` and re-run `simulate.py` or hit `POST /triage` in an environment
with normal internet access — no code changes needed, the pipeline picks up the key
automatically and prints which backend served each patient (`extraction=[openai/ok]`
vs `extraction=[heuristic/ok]`).

## ML results (4,000 synthetic patients, 5-fold CV x 3 seeds)

| model | recall (critical) | precision | AUPRC | Brier |
|---|---|---|---|---|
| Logistic Regression (balanced) | 0.83 | 0.70 | 0.854 | 0.108 |
| Random Forest (balanced) | 0.79 | 0.72 | 0.818 | 0.130 |
| XGBoost (6x critical weight) | **0.91** | 0.61 | 0.848 | 0.135 |

**Honest finding:** XGBoost's AUPRC is not meaningfully better than a plain balanced
logistic regression on this synthetic dataset — the gain from XGBoost shows up in
recall once its class weight is pushed to 6x (deliberately trading precision for
fewer missed critical cases), not in ranking quality. This is reported rather than
hidden, per the brief.

Cost-sensitivity sweep (critical-class weight):

| weight | recall | precision | FN rate |
|---|---|---|---|
| 2x | 0.81 | 0.74 | 0.20 |
| 4x | 0.87 | 0.67 | 0.13 |
| **6x (chosen)** | **0.91** | 0.61 | **0.09** |
| 10x | 0.94 | 0.56 | 0.07 |

6x is used as the deployed operating point: it materially cuts false negatives
relative to 2x/4x, while 10x's extra recall (3 points) costs 5 points of precision —
judged not worth the added over-triage. This should be revisited with real outcome
data and clinician input, not treated as final.

Age-aware ablation: age-conditioned features improve AUPRC (0.848 vs 0.829) and AUROC
(0.925 vs 0.911) over treating age as a plain numeric feature — a small but consistent
gain in this synthetic run, in line with the design doc's expectation, though it should
still be treated as directional rather than conclusive given synthetic subgroup sizes.

Subgroup recall at threshold 0.5: pediatric 0.89 (n=633), adult 0.87 (n=2,421),
geriatric 0.91 (n=946) — no subgroup was left materially behind.

Top global SHAP drivers: `spo2_abs_deficit`, missingness flags on HR/RR/SBP/SpO2,
`mental_status_altered`, `bleeding`, `hr_deviation` — consistent with the hand-built
Safety Gate's own priorities, which is the sanity check the design doc calls for.

## Safety Gate

Deterministic, checked after the model, allowed only to move a patient to a *lower*
band number (more urgent) than the model recommended:

- hard redlines: SpO2 < 90, SBP < 90, HR > 150 or < 45, RR outside 9–32, red-flag
  phrases from the note, active bleeding, altered mental status → forces Band 1
- deterioration: ≥2 worsening vitals across repeat readings → escalates one band
- low input completeness with residual risk → escalates one band
- model unavailable → hard-caps at Band 3 (`FALLBACK_BAND`) and marks fallback status
  explicitly (`recommendation_mode: "safety_fallback"`, `critical_probability: null`,
  `severity_score: null`) — the UI never presents Band 3 in fallback mode as if it were
  an XGBoost prediction
- structural floors: zero-history + weak signal, pregnancy + bleeding/abdominal pain,
  geriatric fall → floor of Band 3

`tests/test_safety_gate.py` asserts the gate can never move a patient to a *less*
urgent band than the model output, for every hard rule and every model band.

## Queue / surge behavior

Safe-wait ceilings: Band 1 = 5 min, Band 2 = 10, Band 3 = 30, Band 4 = 60, Band 5 = 120.
`queue_sim.py` simulates Poisson arrivals; at 3x normal volume over 3 hours, breach
events roughly triple versus normal load (see `simulate.py` output).

`compare_queue_policies()` actually re-runs (not narrates) a static-priority vs FIFO vs
wait-decay comparison across normal / 2x / 3x-crisis load with 4 clinicians. In this
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

## Demo patients (`patients.py`, run via `simulate.py`)

20 patients covering: an ambiguous presentation with a self-report/observed mismatch
(P01), pediatric fever (P02), a geriatric fall (P03), a zero-history patient with only
a mild complaint (P04 / P17), a zero-history unresponsive patient (P11), a pregnancy +
bleeding case (P06), a stroke red-flag case (P14), a deteriorating-vitals case with
repeat readings (P13), and a maximal-severity case (P20). Every run writes a full audit
entry; one clinician downgrade (P18) is captured with its reason code, and a second
attempted downgrade with no reason code is shown being rejected.

## Dashboard

`outputs/dashboard.html` is a self-contained triage board (band-colored rows, an
input-completeness badge, P(critical), model band vs. gate-escalated final band). Clicking a
row expands the pipeline for that patient — the extracted note, the SHAP-driven model
explanation, the safety-gate reasoning, and a working override control that refuses a
downgrade without a reason code, same as `pipeline.record_clinician_decision`. It also
shows the queue-policy comparison and a live (session-local) audit trail of overrides
logged from the UI. It's driven entirely by the JSON `simulate.py` writes to
`outputs/dashboard_data.json` — re-run `simulate.py` and rebuild the HTML to refresh it
with new scenarios.

## Live API (`api.py`)

`uvicorn api:app` exposes the actual pipeline, not a replay of pre-computed results:

- `POST /triage` — runs a patient record through extraction → features → XGBoost →
  calibration → Safety Gate → SHAP explanation → audit write, and enqueues it
- `POST /override` — records a clinician decision, rejecting an unexplained downgrade
- `GET /queue` — current wait times against safe-wait ceilings and any new breaches
- `POST /model/unavailable` / `/model/available` — toggles the safe-fallback path
- `GET /audit` — tail of the audit log

This is the piece that matches section 24's "Frontend → API → LLM extraction →
features → XGBoost → calibration → Safety Gate → clinician → audit" request flow as a
runnable service rather than only as an architecture diagram.

## Data handling notes

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
- **Wait-time breach → reassessment.** Exceeding a patient's safe-wait ceiling now sets
  an explicit, persisted `reassessment_required` flag (with `reassessment_required_at`)
  and writes a `wait_breach_detected` audit event, rather than only being a derived
  `breached` boolean recomputed on each poll. Submitting a new `/triage` reading for
  that patient (a reassessment) re-runs the full pipeline against the *same, preserved*
  `arrival_time`, and a `reassessment_performed` audit event is written. The flag only
  clears if the freshly re-run pipeline no longer puts that patient in breach — a
  reassessment that still comes back critical and still exceeds its ceiling stays
  flagged, by design.



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
- This is a decision-support prototype, not an autonomous or clinically validated
  triage system, and should not be represented as one.

## ED surge handling (operational layer, not clinical)

The Round 2 "3x normal volume" requirement is implemented entirely at the
queue-management layer, deliberately separate from clinical triage:

```
Patient clinical data → triage pipeline → ML → Safety Gate → clinical band
                                                                    │
                                                                    ▼
                                                            queue manager
                                                                    │
                                                    NORMAL / SURGE / CRISIS
```

- `surge.py` — `determine_operational_state()` classifies ED workload as
  `NORMAL` / `SURGE` / `CRISIS` from arrival rate (and, secondarily, queue
  depth relative to clinician count) alone. It never looks at a patient's
  vitals, symptoms, or triage band. Baseline (20/hr) and the multiplier
  thresholds (≤1.25x / ≤2x / >2x) are **prototype simulation assumptions**,
  not a validated capacity-planning model — a real deployment would derive
  these per-site.
- `queue_sim.py` — offline what-if simulator (`run_operational_scenario`)
  used by the surge demo screen. Arrivals are a stochastic Poisson process,
  not uniform spacing. Three scheduling policies are compared:
  - `FIFO` — arrival order only.
  - `STATIC_SEVERITY` — clinical band strictly first; can starve low-acuity
    patients under sustained load.
  - `WAIT_PROTECTED` — clinical band first, but Bands 3–5 accrue bounded
    wait-time scheduling-priority credit the longer they wait, to prevent
    starvation. **Bands 1–2 are hard-protected**: no amount of another
    patient's waiting can move them ahead of a real Band 1/2 patient. This
    changes *scheduling priority* only; it never rewrites `clinical_band`,
    which is stored and reported separately from queue position throughout.
- `db.get_live_operational_status()` — the same NORMAL/SURGE/CRISIS
  classification, but computed live from the real running app's actual
  patient arrivals and real queue (not the simulator), exposed at
  `GET /surge/status`. Emits `surge_state_changed` and
  `queue_threshold_reached` audit events on transitions, and
  `reassessment_required` alongside the existing `wait_breach_detected`
  event when a patient crosses their safe-wait ceiling.
- `POST /surge/simulate` runs the offline scenario (`surge_multiplier`,
  `duration_min`, `n_clinicians`, `seed`) and returns operational-state
  classification, per-policy metrics (avg/P95 wait, max queue length,
  throughput, safe-wait breaches, reassessment events), a policy comparison
  table, and an end-of-run queue-visualization snapshot — every number
  comes from that run, nothing is hardcoded. The default seed (`0`) is
  fixed only so a recorded demo is repeatable; pass a different seed (or
  `null`) for a fresh random draw. All of `SAFE_WAIT_MINUTES`, the band
  mix, and the per-band service-time assumptions used by the simulator are
  prototype demo assumptions, not clinical or staffing standards, and are
  labeled as such in code and in the UI.
- The Safety Gate, XGBoost models, feature engineering, and
  `SAFE_WAIT_MINUTES` ceilings themselves are **unchanged by surge state** —
  see `safety_gate.py` / `pipeline.py`, neither of which imports `surge.py`
  or `queue_sim.py`. A patient's safe-wait breach still routes through the
  same repeat-vitals → features → ML → Safety Gate reassessment pipeline
  used for any other reassessment, surge or not.
- Frontend: the "Surge simulation" tab (`frontend/src/pages/SurgeDashboard.jsx`)
  shows live operational status, a 1x/2x/3x scenario selector, the resulting
  metrics and policy comparison, a queue visualization, and a short
  explanation panel reiterating that clinical thresholds are unchanged.
