from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.artifacts import atomic_write_json, atomic_write_toml, write_config_safely
from src.config import PROJECT_ROOT, load_project_config
from src.market import paths_sha256, sample_heston_numpy


EXPERIMENT_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-FeatureStudy-v1"
XI_VALUES = (0.1, 0.3)
PUBLIC_SEED = 20_260_724
PUBLIC_PATHS = 100_000

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "name": "legacy-raw-shared",
        "feature_mode": "legacy_raw",
        "time_parameterization": "shared",
        "hidden_dims": [64, 64],
        "activation": "leaky_relu",
        "batch_norm": False,
        "learning_rate": 1e-3,
        "paths_per_epoch": 3_000,
        "feature_names": ["spot", "variance", "time_ratio", "previous_hedge"],
        "provenance": "archived coursework Vanilla DH",
    },
    {
        "name": "normalized-shared",
        "feature_mode": "normalized",
        "time_parameterization": "shared",
        "hidden_dims": [64, 32],
        "activation": "leaky_relu",
        "batch_norm": False,
        "learning_rate": 1e-3,
        "paths_per_epoch": 3_000,
        "feature_names": [
            "log_moneyness",
            "variance_ratio",
            "time_ratio",
            "previous_hedge",
        ],
        "provenance": "ICAIF stage-1 normalized Vanilla DH",
    },
    {
        "name": "paper-shared-time",
        "feature_mode": "paper_log_state_with_time",
        "time_parameterization": "shared",
        "hidden_dims": [16, 16],
        "activation": "relu",
        "batch_norm": True,
        "learning_rate": 5e-3,
        "paths_per_epoch": 256,
        "feature_names": [
            "log_spot",
            "variance",
            "time_ratio",
            "previous_hedge",
        ],
        "provenance": (
            "Buehler et al. paper conventions with shared weights and explicit time"
        ),
    },
    {
        "name": "paper-per-step",
        "feature_mode": "paper_log_state",
        "time_parameterization": "per_step",
        "hidden_dims": [16, 16],
        "activation": "relu",
        "batch_norm": True,
        "learning_rate": 5e-3,
        "paths_per_epoch": 256,
        "feature_names": ["log_spot", "variance", "previous_hedge"],
        "provenance": (
            "stock-only adaptation of Buehler et al. with one network per date"
        ),
    },
)


def _market(base: dict[str, Any], xi: float) -> dict[str, Any]:
    market = dict(base)
    market["xi"] = xi
    return market


def _market_name(xi: float) -> str:
    return f"heston-xi{int(round(10 * xi)):02d}"


def _leaderboard(market: dict[str, Any], xi: float) -> dict[str, Any]:
    paths = sample_heston_numpy(market, PUBLIC_PATHS, PUBLIC_SEED)
    return {
        "name": f"feature-study-xi{int(round(10 * xi)):02d}-public",
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
        for config_id, variant in enumerate(VARIANTS):
            directory = (
                EXPERIMENT_ROOT
                / _market_name(xi)
                / "public"
                / str(variant["name"])
            )
            payload = {
                "seed": 0,
                "experiment": {
                    "name": "DeepHedger-FeatureStudy-v1",
                    "purpose": "estimate Vanilla DH versus mathematical deltas",
                    "market": _market_name(xi),
                    "split": "public",
                    "config_id": config_id,
                    "variant": variant["name"],
                    "model": "DeepHedger",
                },
                "market": market,
                "objective": dict(project["objective"]),
                "leaderboard": leaderboard,
                "model": {
                    "architecture": "mlp",
                    "feature_mode": variant["feature_mode"],
                    "feature_names": variant["feature_names"],
                    "time_parameterization": variant["time_parameterization"],
                    "hidden_dims": variant["hidden_dims"],
                    "activation": variant["activation"],
                    "batch_norm": variant["batch_norm"],
                    "initialization": (
                        "torch-default; exact paper distributions were not disclosed"
                        if str(variant["name"]).startswith("paper-")
                        else "torch-default"
                    ),
                    "n_frequencies": 16,
                    "paf_sigma": 1.0,
                    "periodic_include_linear": False,
                },
                "training": {
                    "optimizer": "adam",
                    "learning_rate": variant["learning_rate"],
                    "n_epochs": 10_000,
                    "paths_per_epoch": variant["paths_per_epoch"],
                    "validation_interval": 200,
                },
                "evaluation": {
                    "leaderboards": ["public"],
                    "save_predictions": True,
                    "private_access": False,
                },
                "provenance": {
                    "description": variant["provenance"],
                    "paper_architecture_rule": (
                        "two hidden layers of width d+15; d=1 gives 16"
                        if str(variant["name"]).startswith("paper-")
                        else "not applicable"
                    ),
                    "paper_training_conventions": (
                        "ReLU; batch norm before activation; Adam lr=0.005; batch=256"
                        if str(variant["name"]).startswith("paper-")
                        else "not applicable"
                    ),
                    "common_changes_from_paper": (
                        "stock only; T=1; 10000 updates; entropic objective; "
                        "full-truncation Euler"
                        if str(variant["name"]).startswith("paper-")
                        else "not applicable"
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
        "experiment": "DeepHedger-FeatureStudy-v1",
        "n_markets": 2,
        "n_variants": 4,
        "n_configs": len(generated),
        "training_seed": 0,
        "public_seed": PUBLIC_SEED,
        "public_paths": PUBLIC_PATHS,
        "private_access": False,
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
