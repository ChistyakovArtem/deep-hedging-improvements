from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.artifacts import atomic_write_json, atomic_write_toml, write_config_safely
from src.config import PROJECT_ROOT, load_project_config
from src.market import paths_sha256, sample_heston_numpy


EXPERIMENT_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-NTBN-v1"
XI_VALUES = (0.1, 0.3)
SEEDS = tuple(range(8))
PUBLIC_SEED = 20_260_724
PUBLIC_PATHS = 100_000
SOURCE_REPOSITORY = "https://github.com/pfnet-research/NoTransactionBandNetwork"
SOURCE_COMMIT = "c061a1a673a82d63278ba6a3a4bf869a615a6c48"


def _market(base: dict[str, Any], xi: float) -> dict[str, Any]:
    market = dict(base)
    market["xi"] = xi
    return market


def _market_name(xi: float) -> str:
    return f"heston-xi{int(round(10 * xi)):02d}"


def _leaderboard(market: dict[str, Any], xi: float) -> dict[str, Any]:
    paths = sample_heston_numpy(market, PUBLIC_PATHS, PUBLIC_SEED)
    return {
        "name": f"ntbn-xi{int(round(10 * xi)):02d}-public",
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
        leaderboard = _leaderboard(market, xi)
        for seed in SEEDS:
            directory = (
                EXPERIMENT_ROOT
                / _market_name(xi)
                / "public"
                / f"seed-{seed:03d}"
            )
            payload = {
                "seed": seed,
                "experiment": {
                    "name": "DeepHedger-NTBN-v1",
                    "purpose": (
                        "same-objective No-Transaction Band Network baseline"
                    ),
                    "market": _market_name(xi),
                    "split": "public",
                    "model": "DeepHedger",
                },
                "market": market,
                "objective": dict(project["objective"]),
                "leaderboard": leaderboard,
                "model": {
                    "architecture": "ntbn",
                    "feature_mode": "ntbn_paper",
                    "feature_names": [
                        "log_moneyness",
                        "time_to_expiry",
                        "instantaneous_volatility",
                        "previous_hedge_for_clamp_only",
                    ],
                    "time_parameterization": "shared",
                    "hidden_dims": [32, 32, 32, 32],
                    "activation": "relu",
                    "batch_norm": False,
                    "initialization": "torch-default",
                    "n_frequencies": 16,
                    "paf_sigma": 1.0,
                    "periodic_include_linear": False,
                    "band_center": "local_black_scholes_delta",
                    "band_width_transform": "leaky_relu_0.01",
                    "inverted_band_rule": "midpoint",
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
                "provenance": {
                    "paper": (
                        "Imaki et al., No-Transaction Band Network: "
                        "A Neural Network Architecture for Efficient Deep Hedging"
                    ),
                    "paper_url": "https://arxiv.org/abs/2103.01775",
                    "source_repository": SOURCE_REPOSITORY,
                    "source_commit": SOURCE_COMMIT,
                    "faithful_components": (
                        "three network inputs in paper order; four hidden ReLU "
                        "layers of width 32; two asymmetric widths; leaky-ReLU "
                        "width transform; previous-position clamp; midpoint "
                        "fallback for inverted bands"
                    ),
                    "heston_adaptation": (
                        "paper GBM volatility is replaced by current sqrt(variance); "
                        "the local Black-Scholes center includes configured r and "
                        "therefore recovers the authors' formula when r=0"
                    ),
                    "common_protocol_change": (
                        "10000 updates x 3000 fresh paths for comparison with this "
                        "project; authors' notebook used 200 x 50000"
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
        "experiment": "DeepHedger-NTBN-v1",
        "n_markets": len(XI_VALUES),
        "n_seeds": len(SEEDS),
        "n_configs": len(generated),
        "public_seed": PUBLIC_SEED,
        "public_paths": PUBLIC_PATHS,
        "private_access": False,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
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
