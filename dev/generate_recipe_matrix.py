from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.artifacts import atomic_write_json, atomic_write_toml, write_config_safely
from src.config import PROJECT_ROOT, load_project_config
from src.market import paths_sha256, sample_heston_numpy


EXPERIMENT_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-RecipeMatrix-v1"
XI_VALUES = (0.1, 0.3)
SEEDS = tuple(range(8))
PUBLIC_SEED = 20_260_724
PUBLIC_PATHS = 100_000

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "name": "v00-vanilla-original",
        "article_label": "Vanilla DH",
        "features": "original",
        "action_head": "direct",
        "reference": "none",
    },
    {
        "name": "v01-vanilla-best-features",
        "article_label": "Vanilla DH + best features",
        "features": "best",
        "action_head": "direct",
        "reference": "none",
    },
    {
        "name": "v02-residual-local-bs",
        "article_label": "Best features + local-BS residual",
        "features": "best",
        "action_head": "delta_residual",
        "reference": "local_bs",
    },
    {
        "name": "v03-residual-best-pointwise",
        "article_label": "Best features + best-pointwise residual",
        "features": "best",
        "action_head": "delta_residual",
        "reference": "best_pointwise",
    },
    {
        "name": "v04-residual-best-overall",
        "article_label": "Best features + best-math-policy residual",
        "features": "best",
        "action_head": "delta_residual",
        "reference": "best_overall",
    },
    {
        "name": "v05-band-local-bs",
        "article_label": "Merged NTBN + local-BS centre",
        "features": "best",
        "action_head": "delta_band",
        "reference": "local_bs",
    },
    {
        "name": "v06-band-best-pointwise",
        "article_label": "Merged NTBN + best-pointwise centre",
        "features": "best",
        "action_head": "delta_band",
        "reference": "best_pointwise",
    },
    {
        "name": "v07-band-best-overall",
        "article_label": "Merged NTBN + best-math-policy centre",
        "features": "best",
        "action_head": "delta_band",
        "reference": "best_overall",
    },
)

REFERENCE_SELECTIONS = {
    0.1: {
        "best_pointwise": {
            "kind": "leland_local_vol",
            "leland_scale": 20.0,
            "no_trade_width": 0.0,
            "public_entropic_risk": 9.729814039984172,
        },
        "best_overall": {
            "kind": "leland_local_vol",
            "leland_scale": 20.0,
            "no_trade_width": 0.0,
            "public_entropic_risk": 9.729814039984172,
        },
    },
    0.3: {
        "best_pointwise": {
            "kind": "leland_local_vol",
            "leland_scale": 18.0,
            "no_trade_width": 0.0,
            "public_entropic_risk": 11.652543090135763,
        },
        "best_overall": {
            "kind": "heston_mv_no_trade",
            "leland_scale": 0.0,
            "no_trade_width": 0.04,
            "public_entropic_risk": 11.629454251545086,
        },
    },
}


def _market(base: dict[str, Any], xi: float) -> dict[str, Any]:
    market = dict(base)
    market["xi"] = xi
    return market


def _market_name(xi: float) -> str:
    return f"heston-xi{int(round(10 * xi)):02d}"


def _leaderboard(market: dict[str, Any], xi: float) -> dict[str, Any]:
    paths = sample_heston_numpy(market, PUBLIC_PATHS, PUBLIC_SEED)
    return {
        "name": f"recipe-matrix-xi{int(round(10 * xi)):02d}-public",
        "n_paths": PUBLIC_PATHS,
        "seed": PUBLIC_SEED,
        "paths_sha256": paths_sha256(paths),
        "allowed_uses": [
            "checkpoint_selection",
            "diagnostic_reporting",
        ],
    }


def _reference(xi: float, selector: str) -> dict[str, Any]:
    if selector == "none":
        return {
            "selector": "none",
            "kind": "local_bs",
            "used": False,
            "leland_scale": 0.0,
            "no_trade_width": 0.0,
            "selection_metric": "not applicable",
        }
    if selector == "local_bs":
        return {
            "selector": "local_bs",
            "kind": "local_bs",
            "used": True,
            "leland_scale": 0.0,
            "no_trade_width": 0.0,
            "selection_metric": "fixed untuned reference",
        }
    selected = dict(REFERENCE_SELECTIONS[xi][selector])
    return {
        "selector": selector,
        "used": True,
        "selection_metric": (
            "lowest entropic risk on the frozen public mathematical-delta board"
        ),
        **selected,
    }


def _model_and_training(
    variant: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if variant["features"] == "original":
        model = {
            "architecture": "mlp",
            "feature_mode": "paper_log_state",
            "feature_names": [
                "log_spot",
                "variance",
                "previous_hedge",
            ],
            "time_parameterization": "per_step",
            "hidden_dims": [16, 16],
            "activation": "relu",
            "batch_norm": True,
        }
        training = {
            "optimizer": "adam",
            "learning_rate": 5e-3,
            "n_epochs": 10_000,
            "paths_per_epoch": 256,
            "validation_interval": 200,
        }
    else:
        model = {
            "architecture": "mlp",
            "feature_mode": "normalized",
            "feature_names": [
                "log_moneyness",
                "variance_ratio",
                "time_ratio",
                "previous_hedge",
            ],
            "time_parameterization": "shared",
            "hidden_dims": [64, 32],
            "activation": "leaky_relu",
            "batch_norm": False,
        }
        training = {
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "n_epochs": 10_000,
            "paths_per_epoch": 3_000,
            "validation_interval": 200,
        }
    model.update(
        {
            "action_head": variant["action_head"],
            "prediction_target": "direct",
            "output_initialization": (
                "zero_last" if variant["action_head"] == "delta_residual" else "default"
            ),
            "reference_hedge": reference["kind"],
            "reference_used": reference["used"],
            "reference_leland_scale": reference["leland_scale"],
            "reference_no_trade_width": reference["no_trade_width"],
            "reference_grid_moneyness_points": 513,
            "reference_grid_volatility_points": 257,
            "reference_grid_log_moneyness_min": -1.5,
            "reference_grid_log_moneyness_max": 1.5,
            "reference_grid_volatility_min": 1e-4,
            "reference_grid_volatility_max": 1.0,
            "reference_cf_n_quad": 256,
            "reference_cf_phi_min": 1e-8,
            "reference_cf_phi_max": 200.0,
            "reference_cf_batch_size": 8192,
            "n_frequencies": 16,
            "paf_sigma": 1.0,
            "periodic_include_linear": False,
        }
    )
    return model, training


def configs() -> list[tuple[Path, dict[str, Any]]]:
    project = load_project_config()
    generated = []
    for xi in XI_VALUES:
        market = _market(project["market"], xi)
        market_name = _market_name(xi)
        leaderboard = _leaderboard(market, xi)
        for seed in SEEDS:
            for variant_id, variant in enumerate(VARIANTS):
                reference = _reference(xi, str(variant["reference"]))
                model, training = _model_and_training(variant, reference)
                directory = (
                    EXPERIMENT_ROOT
                    / market_name
                    / "public"
                    / f"seed-{seed:03d}"
                    / str(variant["name"])
                )
                payload = {
                    "seed": seed,
                    "experiment": {
                        "name": "DeepHedger-RecipeMatrix-v1",
                        "purpose": (
                            "factor features, action head, and mathematical "
                            "reference hedge without periodic embeddings"
                        ),
                        "market": market_name,
                        "split": "public",
                        "variant_id": variant_id,
                        "variant": variant["name"],
                        "article_label": variant["article_label"],
                        "model": "DeepHedger",
                    },
                    "market": market,
                    "objective": dict(project["objective"]),
                    "leaderboard": leaderboard,
                    "model": model,
                    "training": training,
                    "reference_selection": reference,
                    "evaluation": {
                        "leaderboards": ["public"],
                        "save_predictions": True,
                        "private_access": False,
                    },
                    "provenance": {
                        "matrix_axes": ("features x action head x reference hedge"),
                        "no_paf": True,
                        "no_running_pnl": True,
                        "paper_ntbn_difference": (
                            "merged band heads use the best normalized features "
                            "and feed previous hedge to the encoder as well as "
                            "the structural clamp"
                        ),
                        "best_overall_semantics": (
                            "a complete deterministic reference policy; when it "
                            "already has a no-trade band, its own reference "
                            "position is tracked independently"
                        ),
                    },
                }
                generated.append((directory, payload))
    return generated


def generate() -> None:
    generated = configs()
    counts = {"written": 0, "exists": 0}
    for directory, payload in generated:
        directory.mkdir(parents=True, exist_ok=True)
        status = write_config_safely(directory / "config.toml", payload)
        counts[status] += 1

    manifest = {
        "experiment": "DeepHedger-RecipeMatrix-v1",
        "n_markets": len(XI_VALUES),
        "n_variants": len(VARIANTS),
        "n_seeds_per_variant_market": len(SEEDS),
        "n_configs": len(generated),
        "public_seed": PUBLIC_SEED,
        "public_paths": PUBLIC_PATHS,
        "private_access": False,
        "variant_order": [variant["name"] for variant in VARIANTS],
        "reference_selections": {
            "heston-xi01": REFERENCE_SELECTIONS[0.1],
            "heston-xi03": REFERENCE_SELECTIONS[0.3],
        },
        "selection_notes": [
            (
                "xi=0.1 best-pointwise and best-overall are the same Leland "
                "scale-20 policy; both requested labels are retained."
            ),
            (
                "xi=0.3 best-overall is a full Heston-MV no-trade policy, not "
                "misrepresented as a pointwise delta."
            ),
            (
                "The 513x257 Heston-MV interpolation grid reproduces the exact "
                "xi=0.3 public no-trade risk within 0.0011 "
                "(11.62837 versus 11.62945)."
            ),
        ],
        "grouping": (
            "eight one-GPU jobs keyed by training seed; each queues all sixteen "
            "market-variant configs"
        ),
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
    atomic_write_json(
        EXPERIMENT_ROOT / "generation-summary.json",
        {"configs": len(generated), **counts},
    )
    print(json.dumps({"configs": len(generated), **counts}, indent=2))


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
