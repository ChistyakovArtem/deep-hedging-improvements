from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from src.artifacts import atomic_write_json, atomic_write_toml, write_config_safely
from src.config import PROJECT_ROOT, load_toml


EXPERIMENT_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-RunningPnL-v1"
BASELINE_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-RecipeMatrix-v1"
XI_VALUES = (0.1, 0.3)
SEEDS = tuple(range(8))


def market_name(xi: float) -> str:
    return f"heston-xi{int(round(10 * xi)):02d}"


def _baseline_config(xi: float, seed: int) -> tuple[Path, dict]:
    path = (
        BASELINE_ROOT
        / market_name(xi)
        / "public"
        / f"seed-{seed:03d}"
        / "v04-residual-best-overall"
        / "config.toml"
    )
    return path, load_toml(path)


def configs() -> list[tuple[Path, dict]]:
    generated = []
    for xi in XI_VALUES:
        for seed in SEEDS:
            baseline_path, baseline = _baseline_config(xi, seed)
            payload = deepcopy(baseline)
            payload["experiment"] = {
                "name": "DeepHedger-RunningPnL-v1",
                "purpose": (
                    "one-factor ablation adding observable running hedge PnL "
                    "to the public-selected residual recipe"
                ),
                "market": market_name(xi),
                "split": "public",
                "variant": "selected-residual-plus-running-pnl",
                "article_label": "Selected residual + running PnL",
                "model": "DeepHedger",
            }
            payload["model"]["feature_names"] = [
                "log_moneyness",
                "variance_ratio",
                "time_ratio",
                "previous_hedge",
                "running_pnl",
            ]
            payload["model"]["use_running_pnl"] = True
            payload["model"]["running_pnl_scale"] = 1.0
            payload["provenance"] = {
                "baseline_experiment": "DeepHedger-RecipeMatrix-v1",
                "baseline_variant": "v04-residual-best-overall",
                "baseline_config": str(baseline_path.relative_to(PROJECT_ROOT)),
                "one_factor_change": "append running_pnl to the policy state",
                "running_pnl_timing": (
                    "pre-decision at t, using only trades and prices observed through t"
                ),
                "running_pnl_definition": (
                    "discounted cash account + previous hedge * discounted current spot"
                ),
                "running_pnl_includes": "past proportional transaction fees",
                "running_pnl_excludes": (
                    "option mark, terminal payoff, future prices, and future fees"
                ),
                "running_pnl_scale": "raw currency units; divisor 1.0",
                "no_paf": True,
                "private_access": False,
            }
            directory = (
                EXPERIMENT_ROOT
                / market_name(xi)
                / "public"
                / f"seed-{seed:03d}"
            )
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
        "experiment": "DeepHedger-RunningPnL-v1",
        "n_markets": len(XI_VALUES),
        "n_seeds": len(SEEDS),
        "n_configs": len(generated),
        "private_access": False,
        "paired_baseline": (
            "DeepHedger-RecipeMatrix-v1/v04-residual-best-overall"
        ),
        "factor": {
            "name": "running_pnl",
            "levels": ["absent in paired baseline", "present in this experiment"],
            "definition": (
                "pre-decision discounted hedge wealth after all prior fees"
            ),
            "scale": 1.0,
        },
        "common_protocol": {
            "accounting": "discounted",
            "objective": "entropic_risk_gamma_1",
            "training_budget": "10000 updates x 3000 fresh paths",
            "checkpoint_selection": "frozen 100000-path public board",
            "no_paf": True,
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
