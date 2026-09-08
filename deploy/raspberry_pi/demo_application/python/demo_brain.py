"""Replace this deliberately selected local application/brain with reviewed code.

This checksum is a packaging smoke test, not an AI model or control policy.
"""

import hashlib


def infer(observation: bytes) -> bytes:
    return hashlib.sha256(observation).hexdigest().encode("ascii")