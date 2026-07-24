from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from src.artifacts import atomic_write_json, atomic_write_toml, write_config_safely
from src.config import PROJECT_ROOT, load_toml


EXPERIMENT_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-OptimizerStudy-v1"
BASELINE_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-RecipeMatrix-v1"
XI_VALUES = (0.1, 0.3)
SEEDS = tuple(range(8))
PAPER_URL = "https://arxiv.org/abs/2604.15297"
PAPER_CODE_COMMIT = "c34e33152db9eea8ed4b2d742b9bef5628d71fb5"


def _candidate(
    label: str,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
    ema_decay: float = 0.99,
    muon_learning_rate: float = 0.02,
) -> dict[str, Any]:
    return {
        "label": label,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "ema_decay": ema_decay,
        "muon_learning_rate": muon_learning_rate,
        "muon_momentum": 0.95,
    }


# Five center-out points per family. Each design begins with a strong anchor,
# then changes one optimizer-specific coordinate at a time. This is a compact
# guided search, not a Cartesian grid and not a random draw.
OPTIMIZER_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    "adam": (
        _candidate("anchor-lr-1e-3", learning_rate=1e-3),
        _candidate("lr-low-5e-4", learning_rate=5e-4),
        _candidate("lr-high-2e-3", learning_rate=2e-3),
        _candidate("lr-lower-2p5e-4", learning_rate=2.5e-4),
        _candidate("lr-higher-4e-3", learning_rate=4e-3),
    ),
    "adamw": (
        _candidate("anchor-lr-1e-3-wd-1e-4", learning_rate=1e-3, weight_decay=1e-4),
        _candidate("lr-low-5e-4", learning_rate=5e-4, weight_decay=1e-4),
        _candidate("lr-high-2e-3", learning_rate=2e-3, weight_decay=1e-4),
        _candidate("wd-mid-1e-3", learning_rate=1e-3, weight_decay=1e-3),
        _candidate("wd-high-1e-2", learning_rate=1e-3, weight_decay=1e-2),
    ),
    "adamw_ema": (
        _candidate(
            "anchor-lr-1e-3-ema-0p99",
            learning_rate=1e-3,
            weight_decay=1e-4,
            ema_decay=0.99,
        ),
        _candidate(
            "lr-low-5e-4",
            learning_rate=5e-4,
            weight_decay=1e-4,
            ema_decay=0.99,
        ),
        _candidate(
            "lr-high-2e-3",
            learning_rate=2e-3,
            weight_decay=1e-4,
            ema_decay=0.99,
        ),
        _candidate(
            "ema-fast-0p9",
            learning_rate=1e-3,
            weight_decay=1e-4,
            ema_decay=0.9,
        ),
        _candidate(
            "ema-slow-0p999",
            learning_rate=1e-3,
            weight_decay=1e-4,
            ema_decay=0.999,
        ),
    ),
    "schedule_free_adamw": (
        _candidate(
            "anchor-lr-2p5e-3-wd-1e-4",
            learning_rate=2.5e-3,
            weight_decay=1e-4,
        ),
        _candidate("lr-low-1e-3", learning_rate=1e-3, weight_decay=1e-4),
        _candidate("lr-high-5e-3", learning_rate=5e-3, weight_decay=1e-4),
        _candidate("wd-mid-1e-3", learning_rate=2.5e-3, weight_decay=1e-3),
        _candidate("wd-high-1e-2", learning_rate=2.5e-3, weight_decay=1e-2),
    ),
    "muon": (
        _candidate(
            "anchor-aux-1e-3-muon-2e-2",
            learning_rate=1e-3,
            weight_decay=1e-4,
            muon_learning_rate=2e-2,
        ),
        _candidate(
            "aux-lr-low-3e-4",
            learning_rate=3e-4,
            weight_decay=1e-4,
            muon_learning_rate=2e-2,
        ),
        _candidate(
            "muon-lr-low-5e-3",
            learning_rate=1e-3,
            weight_decay=1e-4,
            muon_learning_rate=5e-3,
        ),
        _candidate(
            "muon-lr-high-3e-2",
            learning_rate=1e-3,
            weight_decay=1e-4,
            muon_learning_rate=3e-2,
        ),
        _candidate(
            "wd-high-1e-2",
            learning_rate=1e-3,
            weight_decay=1e-2,
            muon_learning_rate=2e-2,
        ),
    ),
}


def market_name(xi: float) -> str:
    return f"heston-xi{int(round(10 * xi)):02d}"


def _baseline_config(xi: float, seed: int) -> tuple[Path, dict[str, Any]]:
    path = (
        BASELINE_ROOT
        / market_name(xi)
        / "public"
        / f"seed-{seed:03d}"
        / "v04-residual-best-overall"
        / "config.toml"
    )
    return path, load_toml(path)


def configs() -> list[tuple[Path, dict[str, Any]]]:
    generated = []
    for xi in XI_VALUES:
        for seed in SEEDS:
            baseline_path, baseline = _baseline_config(xi, seed)
            for optimizer, candidates in OPTIMIZER_CANDIDATES.items():
                for hp_id, candidate in enumerate(candidates):
                    payload = deepcopy(baseline)
                    hp_name = f"hp-{hp_id:02d}-{candidate['label']}"
                    payload["experiment"] = {
                        "name": "DeepHedger-OptimizerStudy-v1",
                        "purpose": (
                            "guided optimizer search for the selected no-PnL "
                            "mathematical-residual DeepHedger"
                        ),
                        "market": market_name(xi),
                        "split": "public",
                        "variant": f"{optimizer}/{hp_name}",
                        "article_label": f"{optimizer}: {candidate['label']}",
                        "model": "DeepHedger",
                        "optimizer": optimizer,
                        "hp_id": hp_id,
                        "hp_label": candidate["label"],
                    }
                    training = payload["training"]
                    training.update(
                        {
                            "optimizer": optimizer,
                            "learning_rate": candidate["learning_rate"],
                            "weight_decay": candidate["weight_decay"],
                            "ema_decay": candidate["ema_decay"],
                            "muon_learning_rate": candidate["muon_learning_rate"],
                            "muon_momentum": candidate["muon_momentum"],
                        }
                    )
                    payload["provenance"] = {
                        "baseline_experiment": "DeepHedger-RecipeMatrix-v1",
                        "baseline_variant": "v04-residual-best-overall",
                        "baseline_config": str(
                            baseline_path.relative_to(PROJECT_ROOT)
                        ),
                        "one_factor_change": "optimizer and optimizer hyperparameters",
                        "guided_search": (
                            "anchor followed by one-coordinate lower/higher probes"
                        ),
                        "optimizer_paper": PAPER_URL,
                        "optimizer_paper_code_commit": PAPER_CODE_COMMIT,
                        "muon_parameter_policy": (
                            "Muon on hidden Linear weights; auxiliary AdamW on "
                            "output head and vector parameters"
                        ),
                        "weight_decay_policy": (
                            "decoupled for AdamW-family methods; biases and "
                            "one-dimensional parameters exempt"
                        ),
                        "no_paf": True,
                        "no_running_pnl": True,
                        "private_access": False,
                    }
                    directory = (
                        EXPERIMENT_ROOT
                        / market_name(xi)
                        / "public"
                        / f"seed-{seed:03d}"
                        / optimizer
                        / hp_name
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
        "experiment": "DeepHedger-OptimizerStudy-v1",
        "n_markets": len(XI_VALUES),
        "n_seeds": len(SEEDS),
        "n_optimizer_families": len(OPTIMIZER_CANDIDATES),
        "n_hps_per_family": 5,
        "n_configs": len(generated),
        "private_access": False,
        "paired_baseline": "DeepHedger-RecipeMatrix-v1/v04-residual-best-overall",
        "optimizer_families": list(OPTIMIZER_CANDIDATES),
        "optimizer_paper": PAPER_URL,
        "optimizer_paper_code_commit": PAPER_CODE_COMMIT,
        "search_design": (
            "five center-out guided points per family; anchor plus "
            "one-coordinate lower/higher probes"
        ),
        "selection_protocol": (
            "select hyperparameters and optimizer by mean public entropic risk "
            "over eight paired training seeds; reserve private for final evaluation"
        ),
        "common_protocol": {
            "accounting": "discounted",
            "objective": "entropic_risk_gamma_1",
            "training_budget": "10000 updates x 3000 fresh paths",
            "checkpoint_selection": "frozen 100000-path public board",
            "features": "selected normalized no-PnL feature set",
            "action_head": "residual over selected mathematical policy",
            "no_paf": True,
            "no_running_pnl": True,
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
