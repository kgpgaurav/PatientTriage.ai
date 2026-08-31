import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import HTTPException

import auth


@pytest.fixture(autouse=True)
def _isolated_auth_state(monkeypatch):
    # Every test here drives TRIAGE_API_KEYS explicitly -- keep that from
    # leaking into other test files' (or the live app's) module-level state.
    monkeypatch.delenv("TRIAGE_API_KEYS", raising=False)
    auth.reload_keys()
    yield
    monkeypatch.delenv("TRIAGE_API_KEYS", raising=False)
    auth.reload_keys()


def test_open_when_no_keys_configured():
    assert auth.auth_enabled() is False
    dependency = auth.require_role("admin")
    caller = dependency(x_api_key=None)
    assert caller == {"role": "dev-open", "key": None, "name": None}


def test_two_part_format_is_backward_compatible(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "key-1:nurse")
    auth.reload_keys()
    assert auth.get_keys() == {"key-1": {"role": "nurse", "name": None}}


def test_three_part_format_captures_name(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "key-1:clinician:Dr. J. Rao")
    auth.reload_keys()
    assert auth.get_keys() == {"key-1": {"role": "clinician", "name": "Dr. J. Rao"}}


def test_malformed_entries_are_skipped(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "no-colon-here, :nurse, key-2:not-a-role, key-3:admin:M. Otieno")
    auth.reload_keys()
    assert auth.get_keys() == {"key-3": {"role": "admin", "name": "M. Otieno"}}


def test_require_role_grants_exact_role_and_returns_name(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "nurse-key:nurse:A. Fisher")
    auth.reload_keys()
    caller = auth.require_role("nurse")(x_api_key="nurse-key")
    assert caller == {"role": "nurse", "key": "nurse-key", "name": "A. Fisher"}


def test_require_role_grants_higher_role_for_lower_minimum(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "clin-key:clinician:Dr. J. Rao")
    auth.reload_keys()
    caller = auth.require_role("nurse")(x_api_key="clin-key")
    assert caller["role"] == "clinician"


def test_require_role_denies_insufficient_role(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "nurse-key:nurse:A. Fisher")
    auth.reload_keys()
    with pytest.raises(HTTPException) as exc:
        auth.require_role("admin")(x_api_key="nurse-key")
    assert exc.value.status_code == 403


def test_require_role_denies_missing_key(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "nurse-key:nurse:A. Fisher")
    auth.reload_keys()
    with pytest.raises(HTTPException) as exc:
        auth.require_role("nurse")(x_api_key=None)
    assert exc.value.status_code == 401


def test_require_role_denies_unknown_key(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "nurse-key:nurse:A. Fisher")
    auth.reload_keys()
    with pytest.raises(HTTPException) as exc:
        auth.require_role("nurse")(x_api_key="not-a-real-key")
    assert exc.value.status_code == 401


def test_reload_keys_picks_up_env_change_without_restart(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "key-1:nurse:A. Fisher")
    auth.reload_keys()
    assert "key-1" in auth.get_keys()

    monkeypatch.setenv("TRIAGE_API_KEYS", "key-2:admin:M. Otieno")
    auth.reload_keys()
    keys = auth.get_keys()
    assert "key-1" not in keys
    assert keys["key-2"] == {"role": "admin", "name": "M. Otieno"}


def test_match_key_is_exact_and_not_substring(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEYS", "nurse-key:nurse:A. Fisher")
    auth.reload_keys()
    assert auth._match_key("nurse-key") == "nurse-key"
    assert auth._match_key("nurse-ke") is None
    assert auth._match_key("nurse-key-extra") is None