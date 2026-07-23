from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_CONFIG = PROJECT_ROOT / "config" / "project.toml"


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def load_project_config(path: str | Path = DEFAULT_PROJECT_CONFIG) -> dict[str, Any]:
    config = load_toml(path)
    validate_project_config(config)
    return config


def validate_project_config(config: dict[str, Any]) -> None:
    market = config["market"]
    if market["model"] != "heston":
        raise ValueError("Stage 1 is frozen to the Heston generator.")
    if market["accounting"] != "discounted":
        raise ValueError("Only coherent discounted accounting is supported.")
    if market["payoff"] != "european_call":
        raise ValueError("Stage 1 is frozen to a European call.")
    if int(market["N"]) <= 0 or float(market["T"]) <= 0:
        raise ValueError("T and N must be positive.")

    public = config["leaderboards"]["public"]
    private = config["leaderboards"]["private"]
    if int(public["seed"]) == int(private["seed"]):
        raise ValueError("Public and private leaderboards must use different seeds.")
    if int(public["n_paths"]) <= 0 or int(private["n_paths"]) <= 0:
        raise ValueError("Leaderboard sizes must be positive.")

    features = config["features"]
    expected = [
        "log_moneyness",
        "variance_ratio",
        "time_ratio",
        "previous_hedge",
    ]
    if list(features["names"]) != expected:
        raise ValueError(f"Stage-1 feature contract must be {expected}.")
    if bool(features["use_running_pnl"]):
        raise ValueError("Running PnL is intentionally disabled in stage 1.")


def deep_copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config)
