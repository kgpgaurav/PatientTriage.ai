"""
Operational (queue-management) surge classification.

IMPORTANT: everything in this module is about *ED workload*, not clinical
risk. It never looks at a patient's vitals, symptoms, or triage band, and
nothing here is allowed to feed back into the Safety Gate, the XGBoost
models, or `final_recommended_band`. See pipeline.py / safety_gate.py for
the clinical path, which is untouched by surge state.

    Clinical severity  != scheduling priority
    Clinical severity  != operational (surge) state

All thresholds below are PROTOTYPE OPERATIONAL SIMULATION ASSUMPTIONS for
this demo -- not clinically validated capacity-planning standards, not a
real hospital's staffing model, and not tuned against real ED throughput
data. They exist only to make the "3x normal volume" surge requirement
explicit and demonstrable.
"""

# Prototype baseline used throughout the surge demo. In a real deployment
# this would be estimated per-site (time of day, day of week, season) --
# here it is a fixed assumption so the 3x scenario has a concrete meaning.
BASELINE_ARRIVALS_PER_HOUR = 20.0

# Prototype operational-state thresholds, expressed as a multiplier of the
# baseline arrival rate. These are simulation knobs, not clinical standards.
NORMAL_MAX_MULTIPLIER = 1.25   # arrival_rate <= 1.25x baseline -> NORMAL
SURGE_MAX_MULTIPLIER = 2.0     # 1.25x < arrival_rate <= 2x baseline -> SURGE
# anything above SURGE_MAX_MULTIPLIER -> CRISIS (3x baseline lands here)

OPERATIONAL_STATES = ("NORMAL", "SURGE", "CRISIS")


def classify_load_multiplier(load_multiplier: float) -> str:
    """Map a load multiplier (arrival_rate / baseline_rate) to an
    operational state. Pure function, no I/O, no clinical inputs."""
    if load_multiplier <= NORMAL_MAX_MULTIPLIER:
        return "NORMAL"
    if load_multiplier <= SURGE_MAX_MULTIPLIER:
        return "SURGE"
    return "CRISIS"


def determine_operational_state(arrival_rate: float, baseline_rate: float = BASELINE_ARRIVALS_PER_HOUR,
                                 queue_length: int | None = None, n_clinicians: int | None = None) -> dict:
    """Determine ED operational state from workload only.

    `arrival_rate` is patients/hour (observed or simulated). `queue_length`
    and `n_clinicians` are optional secondary signals: if the queue has
    grown far beyond what the current clinician count can be expected to
    clear, that alone can justify treating the state as more severe than
    the raw arrival rate would suggest (e.g. a slow-building backlog from a
    period of short-staffing). This is a prototype heuristic, not a
    validated capacity model.

    Returns a JSON-serializable dict:
        {"state": "CRISIS", "arrival_rate": 60.0, "baseline_rate": 20.0,
         "load_multiplier": 3.0, "queue_pressure_override": false}
    """
    baseline_rate = baseline_rate or BASELINE_ARRIVALS_PER_HOUR
    load_multiplier = (arrival_rate / baseline_rate) if baseline_rate else 0.0
    state = classify_load_multiplier(load_multiplier)

    queue_pressure_override = False
    # Prototype secondary signal: a queue much deeper than clinician
    # capacity can clear soon is itself a form of operational pressure,
    # even if the instantaneous arrival rate has already eased off.
    if queue_length is not None and n_clinicians:
        if queue_length > 6 * n_clinicians and state == "NORMAL":
            state = "SURGE"
            queue_pressure_override = True
        elif queue_length > 10 * n_clinicians and state != "CRISIS":
            state = "CRISIS"
            queue_pressure_override = True

    return {
        "state": state,
        "arrival_rate": round(arrival_rate, 2),
        "baseline_rate": round(baseline_rate, 2),
        "load_multiplier": round(load_multiplier, 2),
        "queue_pressure_override": queue_pressure_override,
        "is_prototype_assumption": True,
    }
