"""Stable named random streams, separate from the engine's dealing."""

import hashlib
import json


def derived_seed(seed: int, *parts: object) -> int:
    payload = json.dumps([seed, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
