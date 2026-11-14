"""Configuration loading and validation. Unknown/unset values stay explicit (null) and are reported, never guessed."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .util import hash_obj

CONFIG_SCHEMA_VERSION = "sba-config/0.1"
ALLOWED_BUDGETS = (1, 3, 5, 10, 25, 50)


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    raw: dict
    path: Path
    hash: str

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def section(self, name: str) -> dict:
        if name not in self.raw:
            raise ConfigError(f"config section '{name}' missing")
        return self.raw[name]

    def scene_xml(self) -> Path:
        p = Path(self.section("twin")["scene_xml"])
        return p if p.is_absolute() else (self.base_dir / p).resolve()


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    if not isinstance(raw, dict) or raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    for k, v in (overrides or {}).items():
        sec, key = k.split(".", 1)
        raw.setdefault(sec, {})[key] = v
    for sec in ("twin", "contract", "rollout", "state", "perturbations", "verification", "gate"):
        if not isinstance(raw.get(sec), dict):
            raise ConfigError(f"config section '{sec}' missing")
    v = raw["verification"]
    n = v.get("n_rollouts")
    if not isinstance(n, int) or n < 1:
        raise ConfigError("verification.n_rollouts must be a positive integer")
    if n not in tuple(v.get("allowed_budgets", ALLOWED_BUDGETS)) and not v.get("allow_nonstandard_budget", False):
        raise ConfigError(f"n_rollouts={n} not in the study budgets {ALLOWED_BUDGETS} (set allow_nonstandard_budget to override)")
    if not (0 < float(v.get("min_valid_fraction", 0.5)) <= 1):
        raise ConfigError("verification.min_valid_fraction must be in (0, 1]")
    for name, spec in (raw["perturbations"].get("variables") or {}).items():
        for req in ("dist", "units", "source", "calibrated"):
            if req not in spec:
                raise ConfigError(f"perturbation '{name}' must declare '{req}'")
        if spec["dist"] == "normal" and not (spec.get("sigma", -1) >= 0 and spec.get("bound", -1) > 0):
            raise ConfigError(f"perturbation '{name}': normal needs sigma>=0 and bound>0 (truncation)")
        if spec["dist"] == "uniform" and not spec["low"] <= spec["high"]:
            raise ConfigError(f"perturbation '{name}': uniform needs low<=high")
        if spec["dist"] not in ("normal", "uniform"):
            raise ConfigError(f"perturbation '{name}': unsupported dist {spec['dist']!r}")
    return Config(raw=raw, path=path, hash=hash_obj(raw))
