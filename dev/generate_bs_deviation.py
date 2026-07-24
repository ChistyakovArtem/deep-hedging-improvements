from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.artifacts import atomic_write_json, atomic_write_toml, write_config_safely
from src.config import PROJECT_ROOT, load_project_config
from src.market import paths_sha256, sample_heston_numpy


EXPERIMENT_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-BSDeviation-v1"
XI_VALUES = (0.1, 0.3)
SEEDS = tuple(range(4))
PUBLIC_SEED = 20_260_724
PUBLIC_PATHS = 100_000


def _market(base: dict[str, Any], xi: float) -> dict[str, Any]:
    market = dict(base)
    market["xi"] = xi
    return market


def _market_name(xi: float) -> str:
    return f"heston-xi{int(round(10 * xi)):02d}"


def _leaderboard(market: dict[str, Any], xi: float) -> dict[str, Any]:
    paths = sample_heston_numpy(market, PUBLIC_PATHS, PUBLIC_SEED)
    return {
        "name": f"bs-deviation-xi{int(round(10 * xi)):02d}-public",
        "n_paths": PUBLIC_PATHS,
        "seed": PUBLIC_SEED,
        "paths_sha256": paths_sha256(paths),
        "allowed_uses": ["checkpoint_selection", "diagnostic_reporting"],
    }


def configs() -> list[tuple[Path, dict[str, Any]]]:
    project = load_project_config()
    result = []
    for xi in XI_VALUES:
        market = _market(project["market"], xi)
        market_name = _market_name(xi)
        leaderboard = _leaderboard(market, xi)
        for seed in SEEDS:
            directory = EXPERIMENT_ROOT / market_name / "public" / f"seed-{seed:03d}"
            payload = {
                "seed": seed,
                "experiment": {
                    "name": "DeepHedger-BSDeviation-v1",
                    "purpose": (
                        "merge NTBN state coordinates and a local-BS residual "
                        "target into the standard shared MLP"
                    ),
                    "market": market_name,
                    "split": "public",
                    "model": "DeepHedger",
                },
                "market": market,
                "objective": dict(project["objective"]),
                "leaderboard": leaderboard,
                "model": {
                    "architecture": "mlp",
                    "feature_mode": "local_bs_state",
                    "prediction_target": "local_bs_deviation",
                    "feature_names": [
                        "log_moneyness",
                        "time_to_expiry",
                        "instantaneous_volatility",
                        "previous_hedge",
                    ],
                    "time_parameterization": "shared",
                    "hidden_dims": [64, 32],
                    "activation": "leaky_relu",
                    "batch_norm": False,
                    "output_initialization": "zero_last",
                    "n_frequencies": 16,
                    "paf_sigma": 1.0,
                    "periodic_include_linear": False,
                    "action_rule": "local_black_scholes_delta_plus_network_residual",
                },
                "training": {
                    "optimizer": "adam",
                    "learning_rate": 1e-3,
                    "n_epochs": 10_000,
                    "paths_per_epoch": 3_000,
                    "validation_interval": 200,
                },
                "evaluation": {
                    "leaderboards": ["public"],
                    "save_predictions": True,
                    "private_access": False,
                },
                "comparison": {
                    "primary": "DeepHedger-NTBN-v1",
                    "paired_training_seed": seed,
                    "same_public_paths": True,
                    "ntbn_difference": (
                        "the previous hedge is an MLP input and the network "
                        "predicts an unconstrained residual instead of band widths"
                    ),
                },
                "provenance": {
                    "design": (
                        "standard ICAIF shared MLP with the paper-ordered NTBN "
                        "market state and the coursework local-BS deviation head"
                    ),
                    "zero_initialization": (
                        "the final layer is zeroed, so the initial policy is "
                        "exactly the local Black-Scholes delta"
                    ),
                    "fixed_components": (
                        "same objective, accounting, optimizer, update budget, "
                        "public paths, and training RNG convention as NTBN"
                    ),
                },
            }
            result.append((directory, payload))
    return result


def generate() -> None:
    generated = configs()
    counts = {"written": 0, "exists": 0}
    for directory, payload in generated:
        directory.mkdir(parents=True, exist_ok=True)
        status = write_config_safely(directory / "config.toml", payload)
        counts[status] += 1

    manifest = {
        "experiment": "DeepHedger-BSDeviation-v1",
        "n_markets": len(XI_VALUES),
        "n_seeds_per_market": len(SEEDS),
        "n_configs": len(generated),
        "public_seed": PUBLIC_SEED,
        "public_paths": PUBLIC_PATHS,
        "private_access": False,
        "primary_comparator": "DeepHedger-NTBN-v1 paired seeds 0-3",
        "common_protocol": {
            "market_differences": "xi only",
            "accounting": "discounted",
            "objective": "entropic_risk_gamma_1",
            "transaction_cost": 0.001,
            "n_steps": 30,
            "no_paf": True,
            "no_running_pnl": True,
            "stock_only": True,
        },
        "config_directories": [
            str(directory.relative_to(PROJECT_ROOT)) for directory, _ in generated
        ],
    }
    atomic_write_toml(EXPERIMENT_ROOT / "manifest.toml", manifest)
    summary = {"configs": len(generated), **counts}
    atomic_write_json(EXPERIMENT_ROOT / "generation-summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
