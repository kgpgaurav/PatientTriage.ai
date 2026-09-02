import os

import numpy as np
import pandas as pd

RNG_SEED = 42


def age_group(age):
    if age < 13:
        return "pediatric"
    if age < 65:
        return "adult"
    return "geriatric"


VITAL_RANGES = {
    "pediatric": {"hr": (100, 140), "sbp": (85, 110), "rr": (20, 30), "temp": (36.5, 37.5), "spo2": (97, 100)},
    "adult":     {"hr": (60, 100),  "sbp": (100, 130), "rr": (12, 20), "temp": (36.5, 37.3), "spo2": (96, 100)},
    "geriatric": {"hr": (60, 95),   "sbp": (105, 140), "rr": (12, 22), "temp": (36.0, 37.2), "spo2": (94, 99)},
}

SYMPTOMS = [
    "chest_pain", "shortness_of_breath", "fever", "confusion", "headache",
    "abdominal_pain", "vomiting", "bleeding", "weakness", "dizziness",
    "cough", "rash", "fall", "back_pain",
]


def _sample_vital(group, key, severity):
    lo, hi = VITAL_RANGES[group][key]
    center = rng.uniform(lo, hi)
    spread = (hi - lo) * (0.3 + severity)
    if key in ("hr", "rr"):
        val = center + rng.normal(0, spread) * (1 if rng.random() > 0.5 else -1) * severity * 2
    elif key == "spo2":
        val = center - severity * rng.uniform(0, 15)
    elif key == "sbp":
        val = center + rng.normal(0, spread) * (1 if rng.random() > 0.5 else -1) * severity * 2
    else:
        val = center + severity * rng.uniform(0, 3)
    return val


def _true_band(row):
    score = 0
    if row["spo2"] < 90:
        score += 4
    elif row["spo2"] < 94:
        score += 2
    if row["sbp"] < 90 or row["sbp"] > 180:
        score += 3
    if row["hr"] > 150 or row["hr"] < 45:
        score += 3
    if row["rr"] < 8 or row["rr"] > 32:
        score += 3
    if row["temp"] > 39.5:
        score += 1
    if row["mental_status_altered"]:
        score += 3
    if row["bleeding"]:
        score += 2
    if row["chest_pain"] and row["shortness_of_breath"]:
        score += 2
    if row["age_group"] == "geriatric" and row["fall"]:
        score += 1
    if row["pregnancy"] and (row["bleeding"] or row["abdominal_pain"]):
        score += 1

    score += rng.normal(0, 1.2)

    if score >= 7:
        return 1
    if score >= 4.5:
        return 2
    if score >= 2.5:
        return 3
    if score >= 1:
        return 4
    return 5


def generate_dataset(n=4000, seed=RNG_SEED):
    global rng
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        bucket = rng.choice(["pediatric", "adult", "geriatric"], p=[0.15, 0.6, 0.25])
        if bucket == "pediatric":
            age = int(rng.integers(0, 13))
        elif bucket == "adult":
            age = int(rng.integers(13, 65))
        else:
            age = int(rng.integers(65, 96))
        group = age_group(age)
        severity = rng.beta(1.5, 4)

        hr = _sample_vital(group, "hr", severity)
        sbp = _sample_vital(group, "sbp", severity)
        rr = _sample_vital(group, "rr", severity)
        temp = _sample_vital(group, "temp", severity)
        spo2 = np.clip(_sample_vital(group, "spo2", severity), 70, 100)

        active_symptoms = list(rng.choice(SYMPTOMS, size=rng.integers(0, 4), replace=False))
        row = {s: (s in active_symptoms) for s in SYMPTOMS}

        row.update({
            "patient_id": f"SIM-{i:05d}",
            "age": age,
            "age_group": group,
            "hr": hr, "sbp": sbp, "rr": rr, "temp": temp, "spo2": spo2,
            "mental_status_altered": bool(rng.random() < (0.05 + severity * 0.3)),
            "pregnancy": bool(group == "adult" and rng.random() < 0.05),
            "has_prior_history": bool(rng.random() < 0.5),
        })

        row["true_band"] = _true_band(row)

        for key in ("hr", "sbp", "rr", "temp", "spo2"):
            if rng.random() < 0.12:
                row[key] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    noise_mask = rng.random(len(df)) < 0.06
    shift = rng.choice([-1, 1], size=len(df))
    df.loc[noise_mask, "true_band"] = np.clip(df.loc[noise_mask, "true_band"] + shift[noise_mask], 1, 5)

    return df


if __name__ == "__main__":
    df = generate_dataset()
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/synthetic_patients.csv", index=False)
    print(df["true_band"].value_counts(normalize=True).sort_index())
    print(df["age_group"].value_counts(normalize=True))
    print("critical rate:", (df["true_band"] <= 2).mean())
