"""Role-based access control for patient data.

Prototype-scope security: keeps unauthenticated/unauthorized callers away
from patient-identifiable data and records who accessed what. This is not a
substitute for a real IAM/SSO integration in a production deployment (see
README "Data protection" section for what a production version would add:
OAuth/SSO, encryption at rest, per-field PII masking, short-lived tokens).

Keys are configured via the TRIAGE_API_KEYS env var, one entry per staff
member (not per role -- see below), comma-separated:
    TRIAGE_API_KEYS="key-123:nurse:A. Fisher,key-456:clinician:Dr. J. Rao,key-789:admin:M. Otieno"

Each entry is "key:role[:name]". The name is optional for backward
compatibility with the older "key:role" format, but strongly recommended:
without it, the audit trail can only prove "a clinician did this," never
"which clinician" -- and a shared per-role key makes every override
practically unattributable to one person. With a name on each key, every
audit entry records both the role and the individual (see api.py's
insert_audit calls, which log caller["role"] and caller["name"]).

Roles, least-privilege:
    nurse      -- can submit intake (/triage) and read the live queue.
    clinician  -- everything a nurse can do, plus overrides, dispositions,
                  and full patient detail/history (needed for care decisions).
    admin      -- everything a clinician can do, plus the audit log and
                  operational controls (model fallback toggle, key reload).

Every role-gated request is logged via db.insert_audit(...) with the
caller's role/name and outcome (granted/denied) -- so misuse or probing
shows up in the same audit trail clinicians already review, not a silo no
one reads.
"""
import hmac
import os

from fastapi import Header, HTTPException

ROLE_RANK = {"nurse": 1, "clinician": 2, "admin": 3}


def _load_keys():
    """Parse TRIAGE_API_KEYS into {key: {"role": ..., "name": ...}}.
    Each entry is "key:role" or "key:role:name" (name optional, so existing
    "key:role" deployments keep working -- callers just get name=None)."""
    raw = os.environ.get("TRIAGE_API_KEYS", "")
    keys = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        parts = entry.split(":", 2)
        key, role = parts[0].strip(), parts[1].strip().lower()
        name = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        if key and role in ROLE_RANK:
            keys[key] = {"role": role, "name": name}
    return keys


_API_KEYS = _load_keys()


def reload_keys():
    """Ops hook -- re-reads TRIAGE_API_KEYS from the environment (the process's
    os.environ, not the .env file itself -- callers that store keys in .env
    should re-run their dotenv loader with override=True first; see api.py's
    POST /admin/reload-keys). Lets a rotated/revoked key take effect without
    restarting the server."""
    global _API_KEYS
    _API_KEYS = _load_keys()
    return _API_KEYS


def get_keys():
    """Public accessor for the already-parsed TRIAGE_API_KEYS map
    (key -> {"role": ..., "name": ...}). Used by anything outside this
    module that needs to pick a valid key for a given role -- e.g.
    reset_and_seed.py -- so key-format parsing lives in one place instead
    of being duplicated per caller."""
    return dict(_API_KEYS)


def auth_enabled():
    # Fails OPEN only in local/dev when no keys are configured at all, so the
    # prototype doesn't lock developers out with zero setup. Fails CLOSED
    # (see require_role) the moment even one key is configured -- there is
    # no "half enforced" state that would give a false sense of protection.
    return bool(_API_KEYS)


def _match_key(x_api_key):
    """Constant-time lookup: compares against every configured key rather
    than short-circuiting on the first mismatched byte via `in`/`==`, so a
    caller can't use response timing to narrow down a valid key."""
    for candidate in _API_KEYS:
        if hmac.compare_digest(candidate, x_api_key):
            return candidate
    return None


def require_role(minimum_role):
    """FastAPI dependency: 401 with no/unknown key, 403 with insufficient role."""

    def dependency(x_api_key: str | None = Header(default=None)):
        if not auth_enabled():
            return {"role": "dev-open", "key": None, "name": None}

        matched_key = _match_key(x_api_key) if x_api_key else None
        if matched_key is None:
            raise HTTPException(status_code=401, detail="Missing or invalid API key.")

        info = _API_KEYS[matched_key]
        role = info["role"]
        if ROLE_RANK[role] < ROLE_RANK[minimum_role]:
            raise HTTPException(status_code=403, detail=f"Role '{role}' cannot access this resource.")

        return {"role": role, "key": matched_key, "name": info["name"]}

    return dependency