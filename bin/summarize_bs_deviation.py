from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

from src.artifacts import atomic_write_json
from src.config import PROJECT_ROOT


EXPERIMENT_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-BSDeviation-v1"
NTBN_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-NTBN-v1"
MARKETS = ("heston-xi01", "heston-xi03")
SEEDS = tuple(range(4))
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
# Two-sided 95% Student-t critical value for four paired seeds (df=3).
T_CRITICAL_95_DF3 = 3.182446305


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _rows(root: Path, market: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        directory = root / market / "public" / f"seed-{seed:03d}"
        if not (directory / "DONE").exists():
            raise RuntimeError(f"Missing DONE marker: {directory}")
        row = _read_json(directory / "scores-public.json")
        if int(row["seed"]) != seed:
            raise RuntimeError(f"Seed mismatch at {directory}")
        if row["leaderboard"] != "public" or bool(row["smoke"]):
            raise RuntimeError(f"Invalid public result at {directory}")
        rows.append(row)
    return rows


def _aggregate(rows: list[dict]) -> dict:
    result = {
        "n_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "paths_sha256": str(rows[0]["paths_sha256"]),
        "parameter_count": int(rows[0]["parameter_count"]),
        "per_seed_entropic_risk": [float(row["entropic_risk"]) for row in rows],
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result[f"mean_{metric}"] = mean(values)
        result[f"std_{metric}"] = stdev(values)
        result[f"sem_{metric}"] = stdev(values) / math.sqrt(len(values))
    return result


def _paired_comparison(candidate: list[dict], ntbn: list[dict]) -> dict:
    candidate_by_seed = {int(row["seed"]): row for row in candidate}
    ntbn_by_seed = {int(row["seed"]): row for row in ntbn}
    if set(candidate_by_seed) != set(ntbn_by_seed):
        raise RuntimeError("Candidate and NTBN training seeds do not match.")

    candidate_hashes = {str(row["paths_sha256"]) for row in candidate}
    ntbn_hashes = {str(row["paths_sha256"]) for row in ntbn}
    if len(candidate_hashes) != 1 or len(ntbn_hashes) != 1 or candidate_hashes != ntbn_hashes:
        raise RuntimeError("Candidate and NTBN leaderboard hashes do not match.")

    differences = [
        float(candidate_by_seed[seed]["entropic_risk"])
        - float(ntbn_by_seed[seed]["entropic_risk"])
        for seed in SEEDS
    ]
    paired_mean = mean(differences)
    paired_std = stdev(differences)
    paired_sem = paired_std / math.sqrt(len(differences))
    margin = T_CRITICAL_95_DF3 * paired_sem
    return {
        "reference": "DeepHedger-NTBN-v1",
        "difference_definition": (
            "BS-deviation entropic risk - NTBN entropic risk; negative favors BS-deviation"
        ),
        "paired_seeds": list(SEEDS),
        "paired_differences": differences,
        "mean_paired_difference": paired_mean,
        "std_paired_difference": paired_std,
        "sem_paired_difference": paired_sem,
        "paired_mean_95pct_t_interval": [
            paired_mean - margin,
            paired_mean + margin,
        ],
        "bs_deviation_wins": sum(value < 0.0 for value in differences),
        "ntbn_wins": sum(value > 0.0 for value in differences),
        "ties": sum(value == 0.0 for value in differences),
        "n_pairs": len(differences),
    }


def summarize() -> dict:
    market_payload = {}
    for market in MARKETS:
        candidate = _rows(EXPERIMENT_ROOT, market)
        ntbn = _rows(NTBN_ROOT, market)
        market_payload[market] = {
            "bs_deviation": _aggregate(candidate),
            "ntbn_same_four_seeds": _aggregate(ntbn),
            "paired_comparison": _paired_comparison(candidate, ntbn),
        }

    payload = {
        "experiment": "DeepHedger-BSDeviation-v1",
        "score_direction": "lower entropic risk is better",
        "evaluation_status": ("public development boards only; private evaluation was not run"),
        "markets": market_payload,
        "interpretation_guardrails": [
            (
                "Every comparison uses the same frozen 100000-path public board "
                "and paired training seeds."
            ),
            (
                "The candidate changes the state coordinates, prediction target, "
                "initialization, and MLP shape relative to NTBN; this is a recipe "
                "comparison, not a one-factor architecture ablation."
            ),
            (
                "The 95% intervals quantify four-seed training variation on the "
                "development board and are not final out-of-sample uncertainty."
            ),
        ],
    }
    atomic_write_json(EXPERIMENT_ROOT / "run-summary.json", payload)
    return payload


def main() -> None:
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
