"""
Cryptographic audit trail — Step 3 addition
================================================
Every `/analyze` and `/analyze-change` call is appended to a local,
hash-chained ledger: each record's hash is computed over the previous
record's hash plus its own contents, so editing any past entry breaks
the chain from that point forward — the same tamper-evidence idea behind
a blockchain or a transparency log, minus the distributed-consensus
part, which an MVP genuinely doesn't need.

Being precise about what this is *not*: it is not hardware attestation
(no TEE/enclave — this app doesn't run on hardware that offers one), and
it's not a substitute for access control on the ledger file itself. What
it *does* guarantee is that if the JSONL file is edited after the fact,
`verify_chain()` will detect exactly which record stopped matching,
rather than silently trusting file contents. That's an honest,
independently-checkable claim, which is the more useful property for an
SIH demo than an unverifiable "cryptographically secure" label.

Privacy note: the ledger stores a SHA-256 fingerprint of each uploaded
image (`input_hash`), not the image itself, so the audit trail doesn't
become a second copy of user data.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import os
import time
import uuid

LEDGER_PATH = os.environ.get(
    "SATQUERY_LEDGER_PATH",
    os.path.join(os.path.dirname(__file__), "data", "audit_ledger.jsonl"),
)
GENESIS_HASH = "0" * 64


@dataclass
class AuditRecord:
    record_id: str
    timestamp: float
    endpoint: str
    input_hash: str      # sha256 of the raw uploaded image bytes — a fingerprint, not the image
    query: str
    intent: str
    answer: str
    prev_hash: str
    record_hash: str = ""


def _canonical(payload: dict) -> str:
    """Deterministic JSON encoding so hashing is reproducible regardless
    of dict key insertion order."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _last_hash() -> str:
    if not os.path.exists(LEDGER_PATH):
        return GENESIS_HASH
    last_line = None
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return GENESIS_HASH
    return json.loads(last_line)["record_hash"]


def append_record(endpoint: str, raw_image_bytes: bytes, query: str, intent: str, answer: str) -> AuditRecord:
    """Append one record to the ledger and return it. Failures here
    should never take down the main analysis response — callers are
    expected to wrap this in a try/except and log-but-continue."""
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    prev_hash = _last_hash()
    record = AuditRecord(
        record_id=str(uuid.uuid4()),
        timestamp=time.time(),
        endpoint=endpoint,
        input_hash=hashlib.sha256(raw_image_bytes).hexdigest(),
        query=query,
        intent=intent,
        answer=answer,
        prev_hash=prev_hash,
    )
    payload = asdict(record)
    payload.pop("record_hash")
    record.record_hash = hashlib.sha256((prev_hash + _canonical(payload)).encode("utf-8")).hexdigest()

    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    return record


def read_records(limit: int = 20) -> list[dict]:
    if not os.path.exists(LEDGER_PATH):
        return []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return list(reversed(records[-limit:]))


def verify_chain() -> dict:
    """Recompute every hash from genesis and confirm it matches what's
    stored. Returns where the chain broke (if anywhere) so a judge / demo
    can see the mechanism, not just a pass/fail badge."""
    if not os.path.exists(LEDGER_PATH):
        return {"valid": True, "records_checked": 0, "broken_at": None, "reason": None}

    prev_hash = GENESIS_HASH
    checked = 0
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if record["prev_hash"] != prev_hash:
                return {
                    "valid": False, "records_checked": checked,
                    "broken_at": record["record_id"], "reason": "prev_hash does not match the chain",
                }

            payload = dict(record)
            stored_hash = payload.pop("record_hash")
            recomputed = hashlib.sha256((prev_hash + _canonical(payload)).encode("utf-8")).hexdigest()
            if recomputed != stored_hash:
                return {
                    "valid": False, "records_checked": checked,
                    "broken_at": record["record_id"],
                    "reason": "record_hash does not match its contents — edited after being written",
                }

            prev_hash = stored_hash
            checked += 1

    return {"valid": True, "records_checked": checked, "broken_at": None, "reason": None}
