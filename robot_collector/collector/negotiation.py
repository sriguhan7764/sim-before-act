"""Stream-profile comparison shared by the camera adapters: requested vs actually negotiated, never silently substituted."""
from __future__ import annotations


class ProfileMismatch(RuntimeError):
    """The device negotiated a stream profile different from the one requested. The episode must not proceed."""


def normalise_format(f) -> str | None:
    return None if f is None else str(f).split(".")[-1].lower()


def compare_profile(requested: dict, actual: dict, fps_tolerance: float = 0.5) -> list[str]:
    """Return human-readable mismatches. A requested value that is None/0 means 'not requested' and is not compared."""
    out = []
    for k in ("width", "height"):
        if requested.get(k) and actual.get(k) != requested[k]:
            out.append(f"{k}: requested {requested[k]}, actual {actual.get(k)}")
    if requested.get("fps"):
        a = actual.get("fps")
        if a is None or abs(float(a) - float(requested["fps"])) > fps_tolerance:
            out.append(f"fps: requested {requested['fps']}, actual {a}")
    if requested.get("format") and normalise_format(actual.get("format")) != normalise_format(requested["format"]):
        out.append(f"format: requested {requested['format']}, actual {actual.get('format')}")
    return out


def negotiation_record(requested: dict[str, dict], actual: dict[str, dict]) -> dict:
    mism = {s: compare_profile(requested[s], actual.get(s, {})) for s in requested}
    return {"requested": requested, "actual": actual, "mismatches": {s: m for s, m in mism.items() if m},
            "matches_request": not any(mism.values()), "silent_substitution": False,
            "note": "a mismatch raises ProfileMismatch before any frame is recorded; nothing is substituted"}
