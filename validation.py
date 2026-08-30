# Broad physiological plausibility bounds -- NOT clinical normal ranges.
# A value outside these bounds is not a real measurement (device/entry error).
# A value inside these bounds but outside the normal range for the patient's
# age group (see data_gen.VITAL_RANGES) is "valid but extreme" and is left to
# the Safety Gate to react to, not rejected here.
#
# Sources for the outer bounds (rounded for a decision-support prototype, not
# a peer-reviewed reference): survivable/measurable extremes reported for
# each vital sign --
#   hr:   sustained bradycardia into the 20s-30s and tachyarrhythmias up to
#         ~300 bpm are both documented; outside 0-300 is not a real reading.
#   sbp:  0 is non-viable, hypertensive-crisis readings stay well under 300.
#   rr:   0 (apnea) is a real and critical value and must NOT be rejected;
#         sustained rates above ~60-80 in severe distress are documented, we
#         allow headroom to 100.
#   temp: survivable core temperature extremes are roughly 25-45 degC; a
#         thermometer fault is far more likely than a reading outside that.
#   spo2: a percentage, so 0-100 is the only physically possible range.
#   age:  0-120 years.
PLAUSIBILITY_BOUNDS = {
    "age": (0, 120),
    "hr": (0, 300),
    "sbp": (0, 300),
    "rr": (0, 100),
    "temp": (25.0, 45.0),
    "spo2": (0, 100),
}

BAND_RANGE = (1, 5)


class ValidationError(ValueError):
    def __init__(self, errors):
        self.errors = dict(errors)
        message = "; ".join(f"{field}: {msg}" for field, msg in self.errors.items())
        super().__init__(message or "validation failed")


def validate_vitals(record):
    errors = {}
    for field, (lo, hi) in PLAUSIBILITY_BOUNDS.items():
        if field not in record:
            continue
        value = record.get(field)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors[field] = f"must be numeric, got {record.get(field)!r}"
            continue
        if value != value:  # NaN
            continue
        if value < lo or value > hi:
            errors[field] = f"{value} is outside the plausible range [{lo}, {hi}]"
    if errors:
        raise ValidationError(errors)
    return record


def validate_band(value, field="band"):
    lo, hi = BAND_RANGE
    if value is None or not isinstance(value, int) or not (lo <= value <= hi):
        raise ValidationError({field: f"must be an integer between {lo} and {hi}, got {value!r}"})
    return value
