DEMO_PATIENTS = [
    {"patient_id": "P01", "age": 68, "age_group": "geriatric", "hr": 102, "sbp": 88, "rr": 24, "temp": 37.1, "spo2": 91,
     "mental_status_altered": False, "pregnancy": False, "has_prior_history": True,
     "chest_pain": True, "shortness_of_breath": True,
     "note": "pt c/o chest tightness x2 days, denies SOB, appears anxious"},

    {"patient_id": "P02", "age": 3, "age_group": "pediatric", "hr": 130, "sbp": 92, "rr": 26, "temp": 38.6, "spo2": 98,
     "mental_status_altered": False, "has_prior_history": False,
     "fever": True, "note": "3yo with fever since this morning, fussy but consolable, no rash"},

    {"patient_id": "P03", "age": 82, "age_group": "geriatric", "hr": 78, "sbp": 118, "rr": 18, "temp": 36.8, "spo2": 95,
     "mental_status_altered": False, "has_prior_history": True, "fall": True,
     "note": "found on floor by family, fell in bathroom, minor bruising, denies head strike"},

    {"patient_id": "P04", "age": 34, "age_group": "adult", "hr": 88, "sbp": 122, "rr": 16, "temp": 36.9, "spo2": 99,
     "mental_status_altered": False, "has_prior_history": False,
     "headache": True, "note": "first time here, mild headache for a few hours, no other symptoms reported"},

    {"patient_id": "P05", "age": 45, "age_group": "adult", "hr": 118, "sbp": 84, "rr": 22, "temp": 38.9, "spo2": 93,
     "mental_status_altered": True, "has_prior_history": True,
     "note": "confused, low blood pressure, febrile, family reports worsening over 6 hours"},

    {"patient_id": "P06", "age": 29, "age_group": "adult", "hr": 90, "sbp": 110, "rr": 18, "temp": 37.0, "spo2": 98,
     "mental_status_altered": False, "pregnancy": True, "has_prior_history": True,
     "abdominal_pain": True, "bleeding": True, "note": "20 weeks pregnant, vaginal bleeding and cramping since this morning"},

    {"patient_id": "P07", "age": 55, "age_group": "adult", "hr": 76, "sbp": 128, "rr": 14, "temp": 36.7, "spo2": 97,
     "mental_status_altered": False, "has_prior_history": True,
     "back_pain": True, "note": "chronic lower back pain, no new symptoms, wants refill on note"},

    {"patient_id": "P08", "age": 71, "age_group": "geriatric", "hr": 132, "sbp": 96, "rr": 28, "temp": 37.4, "spo2": 90,
     "mental_status_altered": False, "has_prior_history": True,
     "shortness_of_breath": True, "note": "known COPD, worsening breathlessness over 2 days, using rescue inhaler more often"},

    {"patient_id": "P09", "age": 9, "age_group": "pediatric", "hr": 118, "sbp": None, "rr": 22, "temp": 37.0, "spo2": 99,
     "mental_status_altered": False, "has_prior_history": False,
     "note": "twisted ankle playing soccer, swelling, able to bear some weight"},

    {"patient_id": "P10", "age": 61, "age_group": "adult", "hr": 84, "sbp": 132, "rr": 16, "temp": 36.9, "spo2": 97,
     "mental_status_altered": False, "has_prior_history": True,
     "weakness": True, "dizziness": True, "note": "feels generally weak and lightheaded for two days, vague symptoms, hard to localize"},

    {"patient_id": "P11", "age": 40, "age_group": "adult", "hr": 172, "sbp": 78, "rr": 30, "temp": 39.8, "spo2": 87,
     "mental_status_altered": True, "has_prior_history": False,
     "note": "found unresponsive briefly, now confused, family called ambulance, no prior records on file"},

    {"patient_id": "P12", "age": 25, "age_group": "adult", "hr": 92, "sbp": 118, "rr": 16, "temp": 37.0, "spo2": 99,
     "mental_status_altered": False, "has_prior_history": True,
     "rash": True, "note": "mild rash on arm for 3 days, no fever, no other symptoms"},

    {"patient_id": "P13", "age": 77, "age_group": "geriatric", "hr": 96, "sbp": 100, "rr": 20, "temp": 36.5, "spo2": 94,
     "mental_status_altered": False, "has_prior_history": True,
     "note": "vitals stable at intake, repeat check requested",
     "history_readings": [
         {"hr": 88, "sbp": 112, "rr": 18, "temp": 36.6, "spo2": 97},
         {"hr": 92, "sbp": 106, "rr": 19, "temp": 36.6, "spo2": 96},
     ]},

    {"patient_id": "P14", "age": 50, "age_group": "adult", "hr": 100, "sbp": 100, "rr": 20, "temp": 37.6, "spo2": 92,
     "mental_status_altered": False, "has_prior_history": True,
     "note": "stroke symptoms reported by bystander, facial droop and slurred speech noted on arrival"},

    {"patient_id": "P15", "age": 15, "age_group": "pediatric", "hr": 110, "sbp": 106, "rr": 20, "temp": 37.0, "spo2": 99,
     "mental_status_altered": False, "has_prior_history": True,
     "cough": True, "note": "mild dry cough for a week, no fever, otherwise well"},

    {"patient_id": "P16", "age": 88, "age_group": "geriatric", "hr": 70, "sbp": 122, "rr": 16, "temp": 36.6, "spo2": 96,
     "mental_status_altered": False, "has_prior_history": True,
     "note": "routine wound check, dressing change, no acute complaints"},

    {"patient_id": "P17", "age": 33, "age_group": "adult", "hr": 84, "sbp": 118, "rr": 16, "temp": 36.9, "spo2": 98,
     "mental_status_altered": False, "has_prior_history": False,
     "note": ""},

    {"patient_id": "P18", "age": 58, "age_group": "adult", "hr": 112, "sbp": 96, "rr": 24, "temp": 38.2, "spo2": 93,
     "mental_status_altered": False, "has_prior_history": True,
     "note": "abdominal pain and vomiting since last night, denies chest pain, denies shortness of breath",
     "abdominal_pain": True, "vomiting": True},

    {"patient_id": "P19", "age": 6, "age_group": "pediatric", "hr": 140, "sbp": 88, "rr": 34, "temp": 39.4, "spo2": 89,
     "mental_status_altered": True, "has_prior_history": False,
     "note": "high fever, lethargic, difficult to rouse, parents very worried, no prior visits"},

    {"patient_id": "P20", "age": 47, "age_group": "adult", "hr": 200, "sbp": 60, "rr": 6, "temp": 35.0, "spo2": 78,
     "mental_status_altered": True, "has_prior_history": False,
     "note": "found collapsed at bus stop, active bleeding from scalp laceration, unresponsive to voice"},
]
