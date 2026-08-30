import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_extract import extract_from_note


def test_positive_symptom_detected():
    r = extract_from_note("patient reports chest pain since last night")
    assert r["extracted_symptoms"].get("chest_pain") is True


def test_negated_symptom_detected():
    r = extract_from_note("denies chest pain, denies shortness of breath")
    assert r["extracted_symptoms"].get("chest_pain") is False
    assert r["extracted_symptoms"].get("shortness_of_breath") is False


def test_multiple_symptoms():
    r = extract_from_note("fever, cough and vomiting for two days")
    syms = r["extracted_symptoms"]
    assert syms.get("fever") is True
    assert syms.get("cough") is True
    assert syms.get("vomiting") is True


def test_ambiguous_language_mismatch_flag():
    r = extract_from_note("denies SOB but appears in visible distress")
    assert r["extracted_symptoms"].get("shortness_of_breath") is False


def test_observed_vs_reported_mismatch():
    r = extract_from_note("denies shortness of breath, appears anxious and in distress")
    assert r["mismatch_flag"] is True


def test_empty_note():
    r = extract_from_note("")
    assert r["extraction_status"] == "no_note"
    assert r["extracted_symptoms"] == {}


def test_none_note():
    r = extract_from_note(None)
    assert r["note_present"] is False


def test_extraction_unavailable_status():
    r = extract_from_note("chest pain", timeout_ok=False)
    assert r["extraction_status"] == "unavailable"
    assert r["extracted_symptoms"] == {}


def test_red_flag_detection():
    r = extract_from_note("bystander reports facial droop and slurred speech")
    assert "facial droop" in r["red_flags"] or "slurred speech" in r["red_flags"]


def test_default_backend_is_heuristic_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = extract_from_note("chest pain")
    assert r["extraction_backend"] == "heuristic"
    assert r["extraction_status"] == "ok"


class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content=None, raise_exc=None):
        self._content = content
        self._raise_exc = raise_exc

    def create(self, **kwargs):
        if self._raise_exc:
            raise self._raise_exc
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, content=None, raise_exc=None):
        self.chat = _FakeChat(_FakeCompletions(content, raise_exc))


def test_llm_backend_used_and_parsed(monkeypatch):
    import llm_extract
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_json = json.dumps({
        "extracted_symptoms": {"chest_pain": True, "shortness_of_breath": False},
        "red_flags": [],
        "mismatch_flag": False,
    })
    monkeypatch.setattr(llm_extract, "_get_client", lambda: _FakeClient(content=fake_json))
    r = extract_from_note("pt c/o chest pain, denies SOB")
    assert r["extraction_backend"] == "openai"
    assert r["extraction_status"] == "ok"
    assert r["extracted_symptoms"]["chest_pain"] is True
    assert r["extracted_symptoms"]["shortness_of_breath"] is False


def test_llm_failure_falls_back_to_heuristic(monkeypatch):
    import llm_extract
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_extract, "_get_client", lambda: _FakeClient(raise_exc=TimeoutError("timed out")))
    r = extract_from_note("denies chest pain, denies shortness of breath")
    assert r["extraction_backend"] == "heuristic"
    assert r["extraction_status"] == "ok_heuristic_fallback"
    assert "llm_error" in r
    assert r["extracted_symptoms"].get("chest_pain") is False


def test_llm_malformed_json_falls_back(monkeypatch):
    import llm_extract
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_extract, "_get_client", lambda: _FakeClient(content="not valid json"))
    r = extract_from_note("chest pain")
    assert r["extraction_backend"] == "heuristic"
    assert r["extraction_status"] == "ok_heuristic_fallback"


def test_llm_payload_strips_unknown_symptom_keys(monkeypatch):
    import llm_extract
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_json = json.dumps({
        "extracted_symptoms": {"chest_pain": True, "not_a_real_symptom": True, "fever": "yes"},
        "red_flags": ["stroke"],
        "mismatch_flag": False,
    })
    monkeypatch.setattr(llm_extract, "_get_client", lambda: _FakeClient(content=fake_json))
    r = extract_from_note("chest pain and stroke symptoms")
    assert r["extraction_backend"] == "openai"
    assert "not_a_real_symptom" not in r["extracted_symptoms"]
    assert "fever" not in r["extracted_symptoms"]
    assert r["extracted_symptoms"]["chest_pain"] is True
    assert r["red_flags"] == ["stroke"]
