"""Role-based access control for patient data.

Prototype-scope security: keeps unauthenticated/unauthorized callers away
from patient-identifiable data and records who accessed what. This is not a
substitute for a real IAM/SSO integration in a production deployment (see
README "Data protection" section for what a production version would add:
OAuth/SSO, encryption at rest, per-field PII masking, key rotation).

Keys are configured via the TRIAGE_API_KEYS env var:
    TRIAGE_API_KEYS="nurse-key-123:nurse,clinician-key-456:clinician,audit-key-789:admin"

Roles, least-privilege:
    nurse      -- can submit intake (/triage) and read the live queue.
    clinician  -- everything a nurse can do, plus overrides, dispositions,
                  and full patient detail/history (needed for care decisions).
    admin      -- everything a clinician can do, plus the audit log.

Every request is logged via db.insert_audit("data_access", ...) with the
caller's role and outcome (granted/denied) -- so misuse or probing shows up
in the same audit trail clinicians already review, not a silo no one reads.
"""
import os

from fastapi import Header, HTTPException

ROLE_RANK = {"nurse": 1, "clinician": 2, "admin": 3}


def _load_keys():
    raw = os.environ.get("TRIAGE_API_KEYS", "")
    keys = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, role = pair.split(":", 1)
        key, role = key.strip(), role.strip().lower()
        if role in ROLE_RANK:
            keys[key] = role
    return keys


_API_KEYS = _load_keys()


def reload_keys():
    """Test/ops hook -- re-reads TRIAGE_API_KEYS from the environment."""
    global _API_KEYS
    _API_KEYS = _load_keys()
    return _API_KEYS


def auth_enabled():
    # Fails OPEN only in local/dev when no keys are configured at all, so the
    # prototype doesn't lock developers out with zero setup. Fails CLOSED
    # (see require_role) the moment even one key is configured -- there is
    # no "half enforced" state that would give a false sense of protection.
    return bool(_API_KEYS)


def require_role(minimum_role):
    """FastAPI dependency: 401 with no/unknown key, 403 with insufficient role."""

    def dependency(x_api_key: str | None = Header(default=None)):
        if not auth_enabled():
            return {"role": "dev-open", "key": None}

        if not x_api_key or x_api_key not in _API_KEYS:
            raise HTTPException(status_code=401, detail="Missing or invalid API key.")

        role = _API_KEYS[x_api_key]
        if ROLE_RANK[role] < ROLE_RANK[minimum_role]:
            raise HTTPException(status_code=403, detail=f"Role '{role}' cannot access this resource.")

        return {"role": role, "key": x_api_key}

    return dependency
