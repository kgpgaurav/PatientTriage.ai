import json
import os
import re

from dotenv import load_dotenv

from data_gen import SYMPTOMS

load_dotenv()  # picks up OPENAI_API_KEY etc. from a .env file if present; no-op otherwise

SYMPTOM_PATTERNS = {
    "chest_pain": [r"chest pain", r"chest tightness", r"chest pressure"],
    "shortness_of_breath": [r"sob\b", r"shortness of breath", r"can'?t breathe", r"breathless", r"dyspnea"],
    "fever": [r"fever", r"febrile", r"chills"],
    "confusion": [r"confus", r"disoriented", r"altered mental"],
    "headache": [r"headache", r"head pain"],
    "abdominal_pain": [r"abdominal pain", r"stomach pain", r"belly pain"],
    "vomiting": [r"vomit", r"throwing up"],
    "bleeding": [r"bleeding", r"blood loss", r"hemorrhag"],
    "weakness": [r"weak(ness)?"],
    "dizziness": [r"dizz", r"lighthead"],
    "cough": [r"cough"],
    "rash": [r"rash"],
    "fall": [r"\bfell\b", r"fall(en)?\b"],
    "back_pain": [r"back pain"],
}

# Deterministic backup for the small set of highest-risk textual signals (see
# merge logic below). These are scanned with the same negation handling as
# SYMPTOM_PATTERNS and always run, independent of whether the LLM backend is
# used, so a red flag can never disappear purely because of an LLM
# extraction failure or omission. Keep this list narrow and explicit -- it is
# a safety backstop, not a second triage engine.
RED_FLAGS = [r"stroke", r"facial droop", r"slurred speech", r"unresponsive",
             r"active bleeding", r"seizure", r"anaphyla"]

NEGATION_WINDOW = 4
NEGATORS = {"denies", "denied", "no", "not", "without", "negative"}

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "6"))

EXTRACTION_SCHEMA_PROMPT = f"""You extract structured clinical facts from an ED nurse's free-text intake note.
You do NOT diagnose, triage, or assign urgency. You only report what the note states or negates.

Return strict JSON with this shape, nothing else:
{{
  "extracted_symptoms": {{"<symptom>": true|false, ...}},
  "red_flags": ["<phrase found in the note>", ...],
  "mismatch_flag": true|false,
  "reasoning_note": "<one short sentence, optional>"
}}

Rules:
- Only include a symptom key in "extracted_symptoms" if the note mentions it, either present (true) or explicitly
  negated ("denies", "no", "without" -> false).
- Valid symptom keys: {", ".join(SYMPTOMS)}.
- "red_flags" lists any of these exact phrases if present in the note: {", ".join(RED_FLAGS)}.
- "mismatch_flag" is true only if the note explicitly denies a symptom (e.g. denies SOB) while also describing
  visible distress, anxiety, or discomfort observed by staff -- i.e. self-report contradicts observed behavior.
- Never invent a symptom, vital sign, or fact not present in the note.
- Output JSON only, no prose, no markdown fences."""


def _tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def _is_negated(text, match_start):
    prefix = text[:match_start].lower()
    tokens = _tokenize(prefix)[-NEGATION_WINDOW:]
    return any(t in NEGATORS for t in tokens)


def deterministic_red_flag_scan(note):
    """Narrow, explicit, negation-aware scan for RED_FLAGS phrases. Always
    safe to run (no LLM/network dependency), so this is the floor every
    extraction path -- LLM, heuristic, or LLM-unavailable -- can fall back
    on for these specific high-risk phrases."""
    if not note:
        return []
    text = note.lower()
    found = []
    for phrase in RED_FLAGS:
        match = re.search(phrase, text)
        if match and not _is_negated(text, match.start()):
            found.append(phrase)
    return found


def extract_from_note_heuristic(note):
    text = note.lower()
    extracted = {}
    for symptom, patterns in SYMPTOM_PATTERNS.items():
        found = False
        negated = False
        for pat in patterns:
            for m in re.finditer(pat, text):
                found = True
                if _is_negated(text, m.start()):
                    negated = True
                else:
                    negated = False
                    break
        if found:
            extracted[symptom] = not negated

    red_flags = deterministic_red_flag_scan(note)

    mismatch = "appears" in text and any(
        extracted.get(s) is False for s in ("shortness_of_breath", "confusion")
    ) and ("anxious" in text or "distress" in text or "uncomfortable" in text)

    return {
        "extracted_symptoms": extracted,
        "red_flags": red_flags,
        "mismatch_flag": mismatch,
    }


def _validate_llm_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("LLM output was not a JSON object")
    symptoms = payload.get("extracted_symptoms", {})
    if not isinstance(symptoms, dict):
        raise ValueError("extracted_symptoms was not an object")
    clean_symptoms = {}
    for k, v in symptoms.items():
        if k in SYMPTOMS and isinstance(v, bool):
            clean_symptoms[k] = v
    red_flags = payload.get("red_flags", [])
    if not isinstance(red_flags, list):
        red_flags = []
    red_flags = [rf for rf in red_flags if isinstance(rf, str)]
    mismatch = bool(payload.get("mismatch_flag", False))
    return {
        "extracted_symptoms": clean_symptoms,
        "red_flags": red_flags,
        "mismatch_flag": mismatch,
    }


_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    _client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)
    return _client


def extract_from_note_llm(note):
    client = _get_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=400,
        messages=[
            {"role": "system", "content": EXTRACTION_SCHEMA_PROMPT},
            {"role": "user", "content": note},
        ],
    )
    raw = response.choices[0].message.content
    payload = json.loads(raw)
    return _validate_llm_payload(payload)


def _with_red_flag_backup(result, note, backend):
    """Union whatever the extraction backend found with the always-on
    deterministic scan, and record which path(s) found each phrase --
    the audit trail can then show whether a trigger came from the LLM,
    the deterministic scan, or both."""
    deterministic_flags = deterministic_red_flag_scan(note)
    backend_flags = result.get("red_flags", [])
    merged = sorted(set(backend_flags) | set(deterministic_flags))
    sources = {}
    for flag in merged:
        found_by = []
        if flag in backend_flags:
            found_by.append(backend)
        if flag in deterministic_flags:
            found_by.append("deterministic_scan")
        sources[flag] = found_by
    return {**result, "red_flags": merged, "red_flag_sources": sources}


def extract_from_note(note, timeout_ok=True, use_llm=True):
    if note is None or not note.strip():
        return {
            "extracted_symptoms": {},
            "red_flags": [],
            "red_flag_sources": {},
            "mismatch_flag": False,
            "note_present": False,
            "extraction_status": "no_note",
            "extraction_backend": None,
        }

    if not timeout_ok:
        deterministic_flags = deterministic_red_flag_scan(note)
        return {
            "extracted_symptoms": {},
            "red_flags": deterministic_flags,
            "red_flag_sources": {f: ["deterministic_scan"] for f in deterministic_flags},
            "mismatch_flag": False,
            "note_present": True,
            "extraction_status": "unavailable",
            "extraction_backend": None,
        }

    if use_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            result = extract_from_note_llm(note)
            result = _with_red_flag_backup(result, note, "llm")
            return {
                **result,
                "note_present": True,
                "extraction_status": "ok",
                "extraction_backend": "openai",
            }
        except Exception as e:
            fallback = extract_from_note_heuristic(note)
            fallback = _with_red_flag_backup(fallback, note, "heuristic")
            return {
                **fallback,
                "note_present": True,
                "extraction_status": "ok_heuristic_fallback",
                "extraction_backend": "heuristic",
                "llm_error": str(e),
            }

    result = extract_from_note_heuristic(note)
    result = _with_red_flag_backup(result, note, "heuristic")
    return {
        **result,
        "note_present": True,
        "extraction_status": "ok",
        "extraction_backend": "heuristic",
    }


def merge_extraction_into_record(record, extraction):
    merged = dict(record)
    for symptom, present in extraction.get("extracted_symptoms", {}).items():
        if symptom not in merged or not merged.get(symptom):
            merged[symptom] = present
    merged["red_flags_from_text"] = extraction.get("red_flags", [])
    merged["red_flag_sources"] = extraction.get("red_flag_sources", {})
    merged["observed_reported_mismatch"] = extraction.get("mismatch_flag", False)
    merged["extraction_status"] = extraction.get("extraction_status", "no_note")
    merged["extraction_backend"] = extraction.get("extraction_backend")
    return merged
