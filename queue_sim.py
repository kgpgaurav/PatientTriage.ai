"""
Queue-management simulation for the ED surge demo.

This module is the *operational* layer only. It consumes a patient's
clinical triage band (already produced by the existing XGBoost + Safety
Gate pipeline) and decides scheduling order and wait-time monitoring. It
never computes, modifies, or overrides `clinical_band` / `final_recommended_band`
-- see pipeline.py / safety_gate.py for that. A patient's clinical band and
their queue-scheduling priority are always kept as two separate numbers.

All numeric assumptions here (safe-wait ceilings, service-time means,
decay rate, band mix) are PROTOTYPE SIMULATION ASSUMPTIONS for this demo,
not clinically validated targets.
"""

import numpy as np

from surge import BASELINE_ARRIVALS_PER_HOUR, determine_operational_state

# Prototype safe-wait ceilings (minutes) by clinical band. These mirror the
# same constants used by the live queue in db.py -- kept in one place here
# so the offline simulator and the live app never drift apart.
SAFE_WAIT_MINUTES = {1: 5, 2: 10, 3: 30, 4: 60, 5: 120}

# Prototype default clinical-band mix for simulated arrivals.
DEFAULT_BAND_MIX = {1: 0.08, 2: 0.15, 3: 0.30, 4: 0.25, 5: 0.22}

# Prototype mean service time (minutes) by band, used only to drive the
# discrete-event simulation -- not a clinical or staffing standard.
SERVICE_MEAN_MIN = {1: 45, 2: 35, 3: 20, 4: 12, 5: 8}

# Bands that hard wait-time protection is not allowed to erode: a patient
# whose *actual* clinical band is 1 or 2 is never overtaken in the queue by
# a lower-acuity patient's wait-time credit, no matter how long that lower
# acuity patient has waited. This directly implements requirement #8
# ("clinical urgency remains dominant; waiting time resolves priority
# within safe operational limits").
HARD_PROTECTED_BANDS = {1, 2}

# Minutes of waiting required to "earn" one clinical-band-level worth of
# scheduling-priority credit under the wait-protected policy. Prototype
# assumption, not a clinical or operational standard.
WAIT_DECAY_MINUTES_PER_LEVEL = 15.0

POLICY_KEYS = ("fifo", "static", "wait_decay")
POLICY_LABELS = {
    "fifo": "FIFO",
    "static": "STATIC_SEVERITY",
    "wait_decay": "WAIT_PROTECTED",
}

SCENARIOS = {"normal": 1, "surge_2x": 2, "crisis_3x": 3}


class QueueEntry:
    """Used only by the legacy `simulate_surge` breach-tally helper below."""

    def __init__(self, patient_id, band, arrival_min):
        self.patient_id = patient_id
        self.band = band
        self.arrival_min = arrival_min
        self.reassessed = False


class TriageQueue:
    def __init__(self):
        self.entries = []

    def enqueue(self, patient_id, band, arrival_min):
        self.entries.append(QueueEntry(patient_id, band, arrival_min))

    def check_breaches(self, current_min):
        breaches = []
        for e in self.entries:
            waited = current_min - e.arrival_min
            ceiling = SAFE_WAIT_MINUTES[e.band]
            if waited > ceiling and not e.reassessed:
                breaches.append({"patient_id": e.patient_id, "band": e.band, "waited_min": waited, "ceiling_min": ceiling})
                e.reassessed = True
        return breaches

    def summary(self, current_min):
        out = []
        for e in self.entries:
            waited = current_min - e.arrival_min
            out.append({
                "patient_id": e.patient_id, "band": e.band, "waited_min": waited,
                "ceiling_min": SAFE_WAIT_MINUTES[e.band],
                "breached": waited > SAFE_WAIT_MINUTES[e.band],
            })
        return out


def simulate_surge(base_arrivals_per_hour=20, surge_multiplier=3, duration_min=180, band_mix=None, seed=0):
    """Legacy helper retained for backward compatibility with simulate.py's
    console demo. Prefer `run_operational_scenario` for anything new."""
    rng = np.random.default_rng(seed)
    band_mix = band_mix or DEFAULT_BAND_MIX
    rate_per_min = (base_arrivals_per_hour * surge_multiplier) / 60.0

    queue = TriageQueue()
    t = 0.0
    pid = 0
    all_breaches = []
    while t < duration_min:
        t += rng.exponential(1 / rate_per_min)
        if t >= duration_min:
            break
        band = rng.choice(list(band_mix.keys()), p=list(band_mix.values()))
        pid += 1
        queue.enqueue(f"SURGE-{pid:04d}", int(band), t)
        all_breaches.extend(queue.check_breaches(t))

    final_breaches = queue.check_breaches(duration_min)
    all_breaches.extend(final_breaches)

    per_band_wait = {}
    for e in queue.entries:
        waited = duration_min - e.arrival_min
        per_band_wait.setdefault(e.band, []).append(waited)

    stats = {}
    for band, waits in per_band_wait.items():
        breached = sum(1 for w in waits if w > SAFE_WAIT_MINUTES[band])
        stats[band] = {
            "n": len(waits),
            "avg_wait_min": sum(waits) / len(waits),
            "breach_rate": breached / len(waits),
        }
    return {"n_patients": pid, "stats": stats, "n_breach_events": len(all_breaches)}


def _generate_arrivals(base_arrivals_per_hour, surge_multiplier, duration_min, band_mix, seed):
    """Stochastic (Poisson-process) arrivals -- deliberately NOT uniform
    spacing, so the demo shows realistic bursty queue behavior."""
    rng = np.random.default_rng(seed)
    rate_per_min = (base_arrivals_per_hour * surge_multiplier) / 60.0
    arrivals = []
    t = 0.0
    pid = 0
    while True:
        t += rng.exponential(1 / rate_per_min)
        if t >= duration_min:
            break
        pid += 1
        band = int(rng.choice(list(band_mix.keys()), p=list(band_mix.values())))
        arrivals.append({"patient_id": f"P{pid:04d}", "band": band, "arrival_min": t})
    return arrivals


def _priority_key(policy, patient, clock):
    """Return the sort key used to pick the next patient off the queue for
    the given policy. Lower sorts first. `patient['band']` is always the
    real clinical band and is never mutated by this function -- only the
    returned scheduling key changes with policy."""
    band = patient["band"]
    arrival = patient["arrival_min"]

    if policy == "fifo":
        return (0, arrival)

    if policy == "static":
        return (band, arrival)

    if policy == "wait_decay":
        waited = clock - arrival
        if band in HARD_PROTECTED_BANDS:
            # Hard clinical prioritization: bands 1-2 are scheduled purely
            # on (band, arrival) and are never made to look "less urgent"
            # by wait-decay math, and no amount of waiting by anyone else
            # can push a bucket-0 (protected) patient out of first place.
            return (0, band, arrival)
        # Unprotected bands (3-5): waiting time buys scheduling-priority
        # credit, bounded so it can never reach parity with a real
        # protected-band patient. effective_priority floors at 2.5, i.e.
        # strictly worse than the best possible protected-band key (0, 2, *).
        eff_priority = max(2.5, band - waited / WAIT_DECAY_MINUTES_PER_LEVEL)
        return (1, eff_priority, arrival)

    raise ValueError(policy)


def _run_policy(arrivals, duration_min, n_clinicians, policy, seed):
    """Simplified discrete-event server: n_clinicians process patients one
    at a time. Tracks, in addition to who got served when: a minute-by-
    minute queue-length series (for max-queue-length), and safe-wait
    breach events fired the instant a still-waiting patient crosses their
    ceiling (not just at time of service), matching the live app's
    behavior of flagging reassessment as soon as the threshold is crossed.
    """
    rng = np.random.default_rng(seed)

    pending = [dict(a, served=False, breach_flagged=False) for a in arrivals]
    clinician_free_at = [0.0] * n_clinicians
    completed = []
    lwbs = []
    breach_events = []
    queue_length_series = []  # (minute, queue_length), sampled within [0, duration_min]

    clock = 0.0
    step = 1.0
    while clock <= duration_min + 240:
        waiting = [p for p in pending if not p["served"] and p["arrival_min"] <= clock]

        if clock <= duration_min:
            queue_length_series.append((clock, len(waiting)))

        # Safe-wait breach detection for everyone still waiting, regardless
        # of whether they get served this tick.
        for p in waiting:
            waited = clock - p["arrival_min"]
            ceiling = SAFE_WAIT_MINUTES[p["band"]]
            if waited > ceiling and not p["breach_flagged"]:
                p["breach_flagged"] = True
                breach_events.append({
                    "patient_id": p["patient_id"], "band": p["band"],
                    "waited_min": round(waited, 1), "ceiling_min": ceiling,
                    "detected_at_min": round(clock, 1),
                })

        if waiting:
            free_idx = min(range(n_clinicians), key=lambda i: clinician_free_at[i])
            if clinician_free_at[free_idx] <= clock:
                key = lambda p: _priority_key(policy, p, clock)
                next_patient = min(waiting, key=key)
                waited = clock - next_patient["arrival_min"]
                ceiling = SAFE_WAIT_MINUTES[next_patient["band"]]
                # Prototype "left without being seen" hazard: only kicks in
                # well past a patient's own safe-wait ceiling, and is a
                # simulation device to demonstrate why starvation matters --
                # not a clinical claim.
                lwbs_hazard = 1 - np.exp(-max(0, waited - ceiling * 1.5) / 60.0) if waited > ceiling else 0.0
                if lwbs_hazard > 0 and rng.random() < lwbs_hazard * 0.1:
                    next_patient["served"] = True
                    lwbs.append(next_patient)
                else:
                    service_time = rng.exponential(SERVICE_MEAN_MIN[next_patient["band"]])
                    clinician_free_at[free_idx] = clock + service_time
                    next_patient["served"] = True
                    next_patient["wait_min"] = waited
                    next_patient["service_start_min"] = clock
                    completed.append(next_patient)
        clock += step
        if all(p["served"] for p in pending) and clock > duration_min:
            break

    return completed, lwbs, breach_events, queue_length_series


def _policy_metrics(arrivals, completed, lwbs, breach_events, queue_length_series, duration_min):
    waits = [p["wait_min"] for p in completed]
    n_served = len(completed)
    throughput_per_hour = n_served / (duration_min / 60.0) if duration_min else 0.0

    avg_wait = float(np.mean(waits)) if waits else 0.0
    p95_wait = float(np.percentile(waits, 95)) if waits else 0.0
    max_queue_length = max((q for _, q in queue_length_series), default=0)
    queue_length_at_end = queue_length_series[-1][1] if queue_length_series else 0

    by_band = {}
    for p in completed:
        by_band.setdefault(p["band"], []).append(p["wait_min"])
    by_band_stats = {}
    for band, w in by_band.items():
        breached = sum(1 for x in w if x > SAFE_WAIT_MINUTES[band])
        by_band_stats[band] = {
            "n": len(w),
            "avg_wait_min": round(float(np.mean(w)), 1),
            "p95_wait_min": round(float(np.percentile(w, 95)), 1),
            "breach_rate": round(breached / len(w), 3),
        }

    return {
        "policy": None,  # filled in by the caller, which knows the policy key
        "n_arrivals": len(arrivals),
        "patients_served": n_served,
        "patients_waiting_at_end": queue_length_at_end,
        "left_without_being_seen": len(lwbs),
        "throughput_per_hour": round(throughput_per_hour, 1),
        "avg_wait_min": round(avg_wait, 1),
        "p95_wait_min": round(p95_wait, 1),
        "max_queue_length": max_queue_length,
        "queue_length_at_end": queue_length_at_end,
        "safe_wait_breaches": len(breach_events),
        "reassessment_events": len(breach_events),  # every breach mandates reassessment; see README
        "by_band": by_band_stats,
    }


def compare_queue_policies(base_arrivals_per_hour=20, duration_min=180, n_clinicians=4, seed=0):
    """Legacy helper retained for backward compatibility with simulate.py's
    console demo. Prefer `run_operational_scenario` for anything new."""
    band_mix = DEFAULT_BAND_MIX
    scenarios = {"normal": 1, "moderate_surge_2x": 2, "extreme_crisis_3x": 3}
    results = {}

    for scenario_name, mult in scenarios.items():
        arrivals = _generate_arrivals(base_arrivals_per_hour, mult, duration_min, band_mix, seed)
        results[scenario_name] = {}
        for policy in POLICY_KEYS:
            completed, lwbs, breach_events, qseries = _run_policy(arrivals, duration_min, n_clinicians, policy, seed)
            by_band = {}
            for p in completed:
                by_band.setdefault(p["band"], []).append(p["wait_min"])
            band_stats = {}
            for band, w in by_band.items():
                breach = sum(1 for x in w if x > SAFE_WAIT_MINUTES[band])
                band_stats[band] = {
                    "n": len(w),
                    "avg_wait_min": round(sum(w) / len(w), 1),
                    "breach_rate": round(breach / len(w), 3),
                }
            results[scenario_name][policy] = {
                "n_arrivals": len(arrivals),
                "n_completed": len(completed),
                "n_lwbs": len(lwbs),
                "by_band": band_stats,
            }
    return results


def run_operational_scenario(surge_multiplier=1, base_arrivals_per_hour=BASELINE_ARRIVALS_PER_HOUR,
                              duration_min=180, n_clinicians=4, seed=0, band_mix=None):
    """Run one operational scenario (1x/2x/3x, or any multiplier) through
    all three queue policies and return everything the surge dashboard
    needs: operational-state classification, per-policy metrics, a policy
    comparison table, and a point-in-time queue-visualization snapshot.

    Uses a fixed default seed so the demo is reproducible run-to-run, while
    arrivals themselves remain a stochastic (Poisson) process -- i.e. the
    *shape* of the surge is always realistic/bursty, but the specific run
    shown in a recorded demo is repeatable. Pass a different seed for a
    fresh random draw.

    Nothing here reads or writes clinical bands -- `band` on each simulated
    arrival is a clinical-triage-band *input* to the queue, exactly as it
    would be for a real patient coming out of the Safety Gate.
    """
    band_mix = band_mix or DEFAULT_BAND_MIX
    arrivals = _generate_arrivals(base_arrivals_per_hour, surge_multiplier, duration_min, band_mix, seed)

    empirical_arrival_rate = len(arrivals) / (duration_min / 60.0) if duration_min else 0.0
    operational_state = determine_operational_state(
        arrival_rate=empirical_arrival_rate,
        baseline_rate=base_arrivals_per_hour,
        queue_length=None,
        n_clinicians=n_clinicians,
    )
    # The configured multiplier is what the demo is "showing"; the
    # empirical one reflects the actual stochastic draw. Surface both.
    operational_state["configured_multiplier"] = surge_multiplier

    policies = {}
    queue_snapshot = None
    for policy in POLICY_KEYS:
        completed, lwbs, breach_events, qseries = _run_policy(arrivals, duration_min, n_clinicians, policy, seed)
        metrics = _policy_metrics(arrivals, completed, lwbs, breach_events, qseries, duration_min)
        metrics["policy"] = POLICY_LABELS[policy]
        policies[policy] = metrics

        if policy == "wait_decay":
            queue_snapshot = _queue_visualization_snapshot(arrivals, completed, breach_events, duration_min, policy)

    comparison_table = [
        {
            "policy": POLICY_LABELS[key],
            "avg_wait_min": policies[key]["avg_wait_min"],
            "p95_wait_min": policies[key]["p95_wait_min"],
            "safe_wait_breaches": policies[key]["safe_wait_breaches"],
            "patients_served": policies[key]["patients_served"],
            "left_without_being_seen": policies[key]["left_without_being_seen"],
        }
        for key in POLICY_KEYS
    ]

    return {
        "operational_state": operational_state,
        "config": {
            "base_arrivals_per_hour": base_arrivals_per_hour,
            "surge_multiplier": surge_multiplier,
            "duration_min": duration_min,
            "n_clinicians": n_clinicians,
            "seed": seed,
            "is_prototype_assumption": True,
        },
        "n_arrivals": len(arrivals),
        "policies": policies,
        "policy_comparison": comparison_table,
        "queue_snapshot": queue_snapshot,
    }


def _queue_visualization_snapshot(arrivals, completed, breach_events, duration_min, policy, sample_size=12):
    """A point-in-time (end-of-simulation) list of still-waiting patients
    for the Section 16 queue visualization: patient, band, waited/ceiling,
    and whether they've crossed the safe-wait threshold. Sampled across
    bands so the demo table isn't dominated by one band."""
    served_ids = {p["patient_id"] for p in completed}
    breached_ids = {b["patient_id"] for b in breach_events}
    waiting = [a for a in arrivals if a["patient_id"] not in served_ids and a["arrival_min"] <= duration_min]

    rows = []
    for p in waiting:
        waited = duration_min - p["arrival_min"]
        ceiling = SAFE_WAIT_MINUTES[p["band"]]
        rows.append({
            "patient_id": p["patient_id"],
            "band": p["band"],
            "waited_min": round(waited, 1),
            "ceiling_min": ceiling,
            "reassessment_required": p["patient_id"] in breached_ids or waited > ceiling,
        })

    rows.sort(key=lambda r: (r["band"], -r["waited_min"]))
    if len(rows) > sample_size:
        # Keep a representative spread: worst-waiting patient per band,
        # capped to sample_size total, rather than an arbitrary slice.
        by_band = {}
        for r in rows:
            by_band.setdefault(r["band"], []).append(r)
        spread = []
        guard = 0
        while len(spread) < sample_size and any(by_band.values()) and guard <= sample_size:
            for band in sorted(by_band):
                if by_band[band]:
                    spread.append(by_band[band].pop(0))
                    if len(spread) >= sample_size:
                        break
            guard += 1
        rows = spread
    return rows
