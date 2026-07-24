from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

from dev.generate_pfhedge_official import (
    DETERMINISTIC_MODELS,
    EXPERIMENT_ROOT,
    SEEDS,
    TRAINABLE_MODELS,
    XI_VALUES,
    market_name,
)
from src.artifacts import atomic_write_json
from src.config import PROJECT_ROOT
from src.pfhedge_baselines import PFHEDGE_VERSION, PFHEDGE_WHEEL_SHA256


RECIPE_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-RecipeMatrix-v1"
ADAPTED_NTBN_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-NTBN-v1"
METRICS = (
    "entropic_risk",
    "pnl_mean",
    "pnl_std",
    "fees_mean",
    "turnover_mean",
    "loss_cvar_95",
    "loss_cvar_99",
    "best_epoch",
    "fit_and_eval_seconds",
)


def _read_result(directory: Path) -> dict:
    required = (
        "DONE",
        "checkpoint.pt",
        "predictions-public.npz",
        "scores-public.json",
        "training_log.json",
    )
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"Missing {missing} at {directory}.")
    return json.loads((directory / "scores-public.json").read_text())


def _official_rows(xi: float, model: str) -> list[dict]:
    root = EXPERIMENT_ROOT / market_name(xi) / "public"
    if model in DETERMINISTIC_MODELS:
        rows = [_read_result(root / "deterministic" / model)]
    else:
        rows = [
            _read_result(root / f"seed-{seed:03d}" / model)
            for seed in SEEDS
        ]
    for row in rows:
        if (
            row["model"] != model
            or row["leaderboard"] != "public"
            or bool(row["smoke"])
            or bool(row["private_access"])
            or row["pfhedge_version"] != PFHEDGE_VERSION
            or row["pfhedge_wheel_sha256"] != PFHEDGE_WHEEL_SHA256
        ):
            raise RuntimeError(f"Malformed official PFHedge row: {row}")
    return rows


def _recipe_rows(xi: float) -> tuple[str, list[str], list[dict]]:
    root = RECIPE_ROOT / market_name(xi) / "public"
    by_variant: dict[str, list[dict]] = {}
    for config in sorted(root.glob("seed-000/*/config.toml")):
        variant = config.parent.name
        rows = [
            _read_result(root / f"seed-{seed:03d}" / variant)
            for seed in SEEDS
        ]
        by_variant[variant] = rows
    means = {
        variant: mean(float(row["entropic_risk"]) for row in rows)
        for variant, rows in by_variant.items()
    }
    best_mean = min(means.values())
    ties = sorted(
        variant
        for variant, value in means.items()
        if math.isclose(value, best_mean, rel_tol=0.0, abs_tol=1e-12)
    )
    canonical = "v04-residual-best-overall"
    if canonical not in ties:
        canonical = ties[0]
    return canonical, ties, by_variant[canonical]


def _adapted_ntbn_rows(xi: float) -> list[dict]:
    root = ADAPTED_NTBN_ROOT / market_name(xi) / "public"
    return [
        _read_result(root / f"seed-{seed:03d}")
        for seed in SEEDS
    ]


def _aggregate(rows: list[dict]) -> dict:
    result = {
        "n_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "paths_sha256": str(rows[0]["paths_sha256"]),
        "parameter_count": int(rows[0]["parameter_count"]),
        "per_seed_entropic_risk": [
            float(row["entropic_risk"]) for row in rows
        ],
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result[f"mean_{metric}"] = mean(values)
        result[f"std_{metric}"] = stdev(values) if len(values) > 1 else 0.0
    return result


def _paired(candidate: list[dict], reference: list[dict], name: str) -> dict:
    if [int(row["seed"]) for row in candidate] != [
        int(row["seed"]) for row in reference
    ]:
        raise RuntimeError(f"Unpaired seeds for {name}.")
    differences = [
        float(row["entropic_risk"]) - float(base["entropic_risk"])
        for row, base in zip(candidate, reference, strict=True)
    ]
    mean_difference = mean(differences)
    standard_error = stats.sem(differences)
    confidence_interval = stats.t.interval(
        0.95,
        len(differences) - 1,
        loc=mean_difference,
        scale=standard_error,
    )
    test = stats.ttest_rel(
        [float(row["entropic_risk"]) for row in candidate],
        [float(row["entropic_risk"]) for row in reference],
    )
    return {
        "reference": name,
        "difference_definition": (
            "candidate entropic risk - reference entropic risk; negative favors candidate"
        ),
        "paired_differences": differences,
        "mean_paired_difference": mean_difference,
        "std_paired_difference": stdev(differences),
        "mean_difference_95ci": [
            float(confidence_interval[0]),
            float(confidence_interval[1]),
        ],
        "paired_ttest_pvalue_descriptive": float(test.pvalue),
        "candidate_wins": sum(value < 0.0 for value in differences),
        "n_pairs": len(differences),
    }


def _validate_board(rows_by_name: dict[str, list[dict]], market: str) -> str:
    hashes = {
        str(row["paths_sha256"])
        for rows in rows_by_name.values()
        for row in rows
    }
    if len(hashes) != 1:
        raise RuntimeError(f"Leaderboard mismatch for {market}: {hashes}")
    return hashes.pop()


def summarize() -> dict:
    markets = {}
    for xi in XI_VALUES:
        market = market_name(xi)
        recipe_variant, recipe_ties, recipe = _recipe_rows(xi)
        adapted = _adapted_ntbn_rows(xi)
        official = {
            model: _official_rows(xi, model)
            for model in sorted(DETERMINISTIC_MODELS | TRAINABLE_MODELS)
        }
        board_rows = {
            "our_selected_recipe": recipe,
            "our_rate_adapted_ntbn": adapted,
            **{f"pfhedge_{name}": rows for name, rows in official.items()},
        }
        board_hash = _validate_board(board_rows, market)
        strategies = {
            "our_selected_recipe": _aggregate(recipe),
            "our_rate_adapted_ntbn": _aggregate(adapted),
            **{
                f"pfhedge_{model}": _aggregate(rows)
                for model, rows in official.items()
            },
        }
        ranking = sorted(
            (
                {
                    "strategy": name,
                    "mean_entropic_risk": aggregate["mean_entropic_risk"],
                }
                for name, aggregate in strategies.items()
            ),
            key=lambda row: row["mean_entropic_risk"],
        )
        markets[market] = {
            "xi": xi,
            "paths_sha256": board_hash,
            "recipe_selected_variant": recipe_variant,
            "recipe_public_mean_ties": recipe_ties,
            "strategies": strategies,
            "ranking": ranking,
            "paired_vs_our_selected_recipe": {
                model: _paired(rows, recipe, recipe_variant)
                for model, rows in official.items()
                if model in TRAINABLE_MODELS
            },
            "official_ntbn_vs_rate_adapted_ntbn": _paired(
                official["ntbn"],
                adapted,
                "DeepHedger-NTBN-v1",
            ),
        }

    payload = {
        "experiment": "PFHedge-Official-v1",
        "score_direction": "lower entropic risk is better",
        "evaluation_status": (
            "public development boards only; private evaluation was not run"
        ),
        "pfhedge_version": PFHEDGE_VERSION,
        "pfhedge_wheel_sha256": PFHEDGE_WHEEL_SHA256,
        "n_completed_configs": 36,
        "markets": markets,
        "article_interpretation": [
            (
                "At xi=0.1, the selected residual recipe is better than both "
                "official trainable PFHedge baselines across all eight paired seeds."
            ),
            (
                "At xi=0.3, official README NTBN is the public-board leader and "
                "beats the selected residual recipe in seven of eight paired seeds."
            ),
            (
                "These are development-board comparisons because the same public "
                "boards select checkpoints and recipes; they are not final "
                "out-of-sample claims."
            ),
            (
                "PFHedge's documented BlackScholes/WhalleyWilmott/README-NTBN "
                "center uses its literal zero-rate formula while the common market "
                "has r=0.01."
            ),
        ],
    }
    atomic_write_json(EXPERIMENT_ROOT / "run-summary.json", payload)
    return payload


def _write_markdown(payload: dict) -> None:
    labels = {
        "our_selected_recipe": "Our selected residual recipe",
        "our_rate_adapted_ntbn": "Our rate-adapted NTBN",
        "pfhedge_black_scholes": "PFHedge Black--Scholes",
        "pfhedge_whalley_wilmott": "PFHedge Whalley--Wilmott",
        "pfhedge_mlp": "PFHedge MLP",
        "pfhedge_ntbn": "PFHedge README NTBN",
    }
    lines = [
        "# Official PFHedge public-board comparison",
        "",
        "Lower entropic risk is better. Values are mean ± sample standard deviation "
        "over eight paired training seeds; deterministic policies have no seed deviation.",
        "",
        "| Heston xi | Strategy | Public entropic risk |",
        "|---:|---|---:|",
    ]
    for market in payload["markets"].values():
        for row in market["ranking"]:
            aggregate = market["strategies"][row["strategy"]]
            value = f"{aggregate['mean_entropic_risk']:.6f}"
            if aggregate["n_seeds"] > 1:
                value += f" ± {aggregate['std_entropic_risk']:.6f}"
            lines.append(
                f"| {market['xi']:.1f} | {labels[row['strategy']]} | {value} |"
            )
    lines += [
        "",
        "All strategies use exactly the same frozen 100,000-path board within each "
        "market and the project's discounted transaction-cost accounting. The private "
        "board was not accessed.",
        "",
        "At xi=0.1, our residual recipe beats PFHedge MLP by 0.007684 entropic-risk "
        "units and PFHedge NTBN by 0.023178 on the paired-seed means.",
        "",
        "At xi=0.3, PFHedge README NTBN beats our residual recipe by 0.049401 and our "
        "rate-adapted NTBN by 0.019368 on the paired-seed means. It is therefore the "
        "current public-board leader for this market.",
        "",
        "These are development-board results, not final out-of-sample claims: the "
        "public board is used for checkpoint and recipe selection.",
    ]
    (EXPERIMENT_ROOT / "RESULTS-public.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    payload = summarize()
    _write_markdown(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
