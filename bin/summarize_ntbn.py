from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

from src.artifacts import atomic_write_json
from src.config import PROJECT_ROOT


NTBN_ROOT = PROJECT_ROOT / "exp" / "DeepHedger-NTBN-v1"
STAGE1_ROOT = (
    PROJECT_ROOT
    / "exp"
    / "DeepHedger-Stage1-v1"
    / "heston-xi03"
    / "split-public"
)
MATH_ROOT = (
    PROJECT_ROOT
    / "exp"
    / "HestonMathDeltas-v1"
    / "heston-xi03"
    / "frozen-public-private"
)
SEEDS = tuple(range(8))
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _ntbn_rows(market: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        directory = NTBN_ROOT / market / "public" / f"seed-{seed:03d}"
        if not (directory / "DONE").exists():
            raise RuntimeError(f"Missing DONE marker: {directory}")
        row = _read_json(directory / "scores-public.json")
        if int(row["seed"]) != seed:
            raise RuntimeError(f"Seed mismatch at {directory}")
        if row["leaderboard"] != "public" or bool(row["smoke"]):
            raise RuntimeError(f"Invalid public result at {directory}")
        rows.append(row)
    hashes = {str(row["paths_sha256"]) for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(f"Leaderboard hash mismatch for {market}: {hashes}")
    return rows


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
        result[f"std_{metric}"] = stdev(values)
    return result


def _stage1_rows(config_id: int) -> list[dict]:
    return [
        _read_json(
            STAGE1_ROOT
            / f"seed-{seed:03d}"
            / f"cfg-{config_id:04d}"
            / "scores-public.json"
        )
        for seed in SEEDS
    ]


def _paired_comparison(
    ntbn_rows: list[dict],
    reference_rows: list[dict],
    reference: str,
) -> dict:
    differences = [
        float(ntbn["entropic_risk"]) - float(other["entropic_risk"])
        for ntbn, other in zip(ntbn_rows, reference_rows, strict=True)
    ]
    return {
        "reference": reference,
        "difference_definition": "NTBN entropic risk - reference entropic risk",
        "mean_reference_entropic_risk": mean(
            float(row["entropic_risk"]) for row in reference_rows
        ),
        "std_reference_entropic_risk": stdev(
            float(row["entropic_risk"]) for row in reference_rows
        ),
        "paired_differences": differences,
        "mean_paired_difference": mean(differences),
        "std_paired_difference": stdev(differences),
        "ntbn_wins": sum(value < 0.0 for value in differences),
        "n_pairs": len(differences),
    }


def _math_reference(relative: str, name: str) -> dict:
    row = _read_json(MATH_ROOT / relative / "scores-public.json")
    return {
        "reference": name,
        "entropic_risk": float(row["entropic_risk"]),
        "paths_sha256": str(row["paths_sha256"]),
    }


def summarize() -> dict:
    xi01_rows = _ntbn_rows("heston-xi01")
    xi03_rows = _ntbn_rows("heston-xi03")
    xi03_hash = str(xi03_rows[0]["paths_sha256"])
    math_references = [
        _math_reference(
            "leland_local_vol-scale-18",
            "Leland local-vol delta, scale 18",
        ),
        _math_reference(
            "heston_mv_no_trade-width-0.04",
            "Heston MV no-trade delta, width 0.04",
        ),
    ]
    if any(row["paths_sha256"] != xi03_hash for row in math_references):
        raise RuntimeError("The xi=0.3 mathematical references use another board.")

    payload = {
        "experiment": "DeepHedger-NTBN-v1",
        "score_direction": "lower entropic risk is better",
        "evaluation_status": (
            "public development board only; private evaluation was not run"
        ),
        "markets": {
            "heston-xi01": _aggregate(xi01_rows),
            "heston-xi03": _aggregate(xi03_rows),
        },
        "xi03_paired_comparisons": [
            _paired_comparison(
                xi03_rows,
                _stage1_rows(0),
                "Vanilla DH, normalized shared MLP",
            ),
            _paired_comparison(
                xi03_rows,
                _stage1_rows(8),
                "Public-selected featurewise PAF, sigma 1.0",
            ),
        ],
        "xi03_math_references": math_references,
        "interpretation_guardrails": [
            (
                "The xi=0.3 comparisons use the same frozen 100000-path public "
                "board and paired training seeds."
            ),
            (
                "The result is a development-board comparison, not the final "
                "out-of-sample claim."
            ),
            (
                "The archived xi=0.1 Leland value 9.301237 is excluded: it used "
                "5000 paths from NumPy seed 42 and undiscounted accounting, not "
                "the frozen 100000-path discounted board used here."
            ),
        ],
    }
    atomic_write_json(NTBN_ROOT / "run-summary.json", payload)
    return payload


def main() -> None:
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
