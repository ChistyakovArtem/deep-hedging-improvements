from __future__ import annotations

import json
import math
from statistics import mean, stdev

from dev.generate_recipe_matrix import (
    EXPERIMENT_ROOT,
    SEEDS,
    VARIANTS,
    XI_VALUES,
    _market_name,
)
from src.artifacts import atomic_write_json


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


def _rows(market: str, variant: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        directory = EXPERIMENT_ROOT / market / "public" / f"seed-{seed:03d}" / variant
        if not (directory / "DONE").exists():
            raise RuntimeError(f"Missing DONE marker: {directory}")
        row = json.loads((directory / "scores-public.json").read_text())
        if (
            int(row["seed"]) != seed
            or row["variant"] != variant
            or row["leaderboard"] != "public"
            or bool(row["smoke"])
        ):
            raise RuntimeError(f"Malformed public result: {directory}")
        rows.append(row)
    if len({row["paths_sha256"] for row in rows}) != 1:
        raise RuntimeError(f"Board mismatch for {market}/{variant}.")
    return rows


def _aggregate(rows: list[dict]) -> dict:
    payload = {
        "n_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "paths_sha256": rows[0]["paths_sha256"],
        "article_label": rows[0]["article_label"],
        "feature_mode": rows[0]["feature_mode"],
        "action_head": rows[0]["action_head"],
        "reference_selector": rows[0]["reference_selector"],
        "parameter_count": rows[0]["parameter_count"],
        "per_seed_entropic_risk": [float(row["entropic_risk"]) for row in rows],
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        payload[f"mean_{metric}"] = mean(values)
        payload[f"std_{metric}"] = stdev(values)
        payload[f"sem_{metric}"] = stdev(values) / math.sqrt(len(values))
    return payload


def _paired(candidate: list[dict], reference: list[dict]) -> dict:
    differences = [
        float(row["entropic_risk"]) - float(base["entropic_risk"])
        for row, base in zip(candidate, reference, strict=True)
    ]
    return {
        "reference": "v01-vanilla-best-features",
        "difference_definition": (
            "candidate entropic risk - best-feature Vanilla DH; negative favors candidate"
        ),
        "paired_differences": differences,
        "mean_paired_difference": mean(differences),
        "std_paired_difference": stdev(differences),
        "candidate_wins": sum(value < 0.0 for value in differences),
        "n_pairs": len(differences),
    }


def summarize() -> dict:
    markets = {}
    for xi in XI_VALUES:
        market = _market_name(xi)
        rows_by_variant = {
            variant["name"]: _rows(market, str(variant["name"])) for variant in VARIANTS
        }
        reference = rows_by_variant["v01-vanilla-best-features"]
        markets[market] = {
            "variants": {name: _aggregate(rows) for name, rows in rows_by_variant.items()},
            "paired_vs_best_feature_vanilla": {
                name: _paired(rows, reference)
                for name, rows in rows_by_variant.items()
                if name != "v01-vanilla-best-features"
            },
        }
    payload = {
        "experiment": "DeepHedger-RecipeMatrix-v1",
        "score_direction": "lower entropic risk is better",
        "evaluation_status": ("public development boards only; private evaluation was not run"),
        "markets": markets,
    }
    atomic_write_json(EXPERIMENT_ROOT / "run-summary.json", payload)
    return payload


def main() -> None:
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
