"""Process-wide hardware authorisation. Real device backends are FAIL-CLOSED.

A config file can name a real backend (``realsense``, ``opencv``, ``feetech_readonly``) but can never enable hardware by
itself: the code that opens a camera, enumerates devices or opens a serial port must call ``require()``, which only
succeeds after ``authorise()`` - done by the CLI only when the operator passes ``--allow-hardware`` and confirms (typed
phrase on a terminal, or ``--hardware-confirmed-by NAME`` which is recorded as the authorisation provenance).

Offline tools (validate, replay, export, quality-report, acceptance, recover) and the explicit mock backends never call
``require()``. ``jsonl_tap`` reads files written by another process and opens no device, so it is not gated here; the
publishing process (``collector.lerobot_bridge``) has its own explicit flags.
"""
from __future__ import annotations

import threading

REAL_BACKENDS = frozenset({"realsense", "opencv", "feetech_readonly"})
CONFIRM_PHRASE = "OPEN HARDWARE"


class HardwareAccessDenied(RuntimeError):
    pass


_lock = threading.Lock()
_state: dict = {"authorised": False, "authorised_by": None, "attempts_denied": []}


def authorise(by: str) -> None:
    if not by or not by.strip():
        raise ValueError("hardware authorisation needs a non-empty provenance string")
    with _lock:
        _state.update(authorised=True, authorised_by=by.strip())


def revoke() -> None:
    with _lock:
        _state.update(authorised=False, authorised_by=None)


def is_authorised() -> bool:
    return bool(_state["authorised"])


def provenance() -> dict:
    return {"hardware_authorised": bool(_state["authorised"]), "authorised_by": _state["authorised_by"]}


def denied_attempts() -> list[str]:
    return list(_state["attempts_denied"])


def require(what: str) -> None:
    """Call immediately before touching a real device. Raises unless the operator authorised hardware for this process."""
    if not _state["authorised"]:
        with _lock:
            _state["attempts_denied"].append(what)
        raise HardwareAccessDenied(
            f"refusing to access hardware ({what}): not authorised in this process. Pass --allow-hardware and confirm "
            "(see docs/OFFLINE_SPRINT.md). A config file never enables hardware; mock backends need no authorisation.")


def config_uses_real_hardware(cfg: dict) -> list[str]:
    """Names of enabled sources in a loaded config that would open a real device."""
    out = [f"camera:{c['name']}({c['backend']})" for c in cfg.get("cameras", []) if c.get("enabled", True) and c["backend"] in REAL_BACKENDS]
    for k in ("follower", "leader"):
        b = cfg["robot"][k]["backend"]
        if b in REAL_BACKENDS:
            out.append(f"robot:{k}({b})")
    return out
