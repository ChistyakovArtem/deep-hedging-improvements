from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from dev.generate_stage1_configs import LEARNED_ROOT
from src.artifacts import atomic_write_json, save_predictions
from src.config import PROJECT_ROOT, load_project_config, load_toml
from src.evaluation import outputs_to_numpy, scores_from_outputs
from src.hedger import DeepHedger
from src.market import materialize_leaderboard


def aggregate_rows(rows: list[dict], config_id: int) -> dict:
    selected = [row for row in rows if row["config_id"] == config_id]
    metrics = (
        "entropic_risk",
        "pnl_mean",
        "pnl_std",
        "loss_cvar_95",
        "loss_cvar_99",
        "fees_mean",
        "turnover_mean",
        "entropic_weight_ess",
        "largest_entropic_weight_share",
        "worst_loss",
    )
    aggregate = {
        "config_id": config_id,
        "architecture": selected[0]["architecture"],
        "paf_sigma": selected[0]["paf_sigma"],
        "n_training_seeds": len(selected),
    }
    for metric in metrics:
        values = [row[metric] for row in selected]
        aggregate[f"mean_{metric}"] = float(np.mean(values))
        aggregate[f"std_{metric}"] = float(np.std(values, ddof=1))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    project = load_project_config()

    grouped: dict[int, list[dict]] = defaultdict(list)
    directories: dict[tuple[int, int], Path] = {}
    for path in sorted(LEARNED_ROOT.glob("seed-*/cfg-*/scores-public.json")):
        directory = path.parent
        config = load_toml(directory / "config.toml")
        seed = int(config["seed"])
        config_id = int(config["experiment"]["config_id"])
        grouped[config_id].append(json.loads(path.read_text()))
        directories[(seed, config_id)] = directory
    if set(grouped) != set(range(11)) or any(len(rows) != 8 for rows in grouped.values()):
        raise RuntimeError("Private evaluation requires all 88 public results.")

    aggregates = []
    for config_id, rows in sorted(grouped.items()):
        aggregates.append(
            {
                "config_id": config_id,
                "mean_public_entropic_risk": float(
                    np.mean([row["entropic_risk"] for row in rows])
                ),
                "std_public_entropic_risk": float(
                    np.std([row["entropic_risk"] for row in rows], ddof=1)
                ),
                "architecture": rows[0]["architecture"],
                "paf_sigma": rows[0]["paf_sigma"],
            }
        )
    selected_id = int(
        min(aggregates, key=lambda row: row["mean_public_entropic_risk"])[
            "config_id"
        ]
    )
    selection = {
        "selection_leaderboard": "public",
        "selected_config_id": selected_id,
        "vanilla_config_id": 0,
        "aggregates": aggregates,
    }
    atomic_write_json(LEARNED_ROOT / "public-selection.json", selection)

    private_numpy, private_metadata = materialize_leaderboard(
        PROJECT_ROOT,
        project["market"],
        project["leaderboards"]["private"],
    )
    private_paths = torch.as_tensor(private_numpy, device=args.device)
    rows = []
    for config_id in sorted({0, selected_id}):
        for seed in range(8):
            directory = directories[(seed, config_id)]
            hedger = DeepHedger.load(directory / "checkpoint.pt", device=args.device)
            hedger.policy.eval()
            with torch.no_grad():
                actions, outputs = hedger.evaluate_tensor(private_paths)
            scores = scores_from_outputs(
                outputs, float(project["objective"]["risk_aversion"])
            )
            scaled_loss = (
                -float(project["objective"]["risk_aversion"])
                * outputs["pnl"].detach().double()
            )
            weights = torch.exp(scaled_loss - scaled_loss.max())
            scores.update(
                {
                    "entropic_weight_ess": float(
                        (weights.sum().square() / weights.square().sum()).item()
                    ),
                    "largest_entropic_weight_share": float(
                        (weights.max() / weights.sum()).item()
                    ),
                    "worst_loss": float((-outputs["pnl"].min()).item()),
                }
            )
            config = load_toml(directory / "config.toml")
            scores.update(
                {
                    "leaderboard": "private",
                    "paths_sha256": private_metadata["paths_sha256"],
                    "seed": seed,
                    "config_id": config_id,
                    "architecture": config["model"]["architecture"],
                    "paf_sigma": config["model"]["paf_sigma"],
                    "role": "vanilla_dh" if config_id == 0 else "best_dh",
                }
            )
            atomic_write_json(directory / "scores-private.json", scores)
            save_predictions(
                directory / "predictions-private.npz",
                actions.cpu().numpy(),
                outputs_to_numpy(outputs),
            )
            rows.append(scores)
    atomic_write_json(LEARNED_ROOT / "leaderboard-private.json", rows)
    vanilla_by_seed = {
        row["seed"]: row for row in rows if row["config_id"] == 0
    }
    selected_by_seed = {
        row["seed"]: row for row in rows if row["config_id"] == selected_id
    }
    paired_differences = [
        selected_by_seed[seed]["entropic_risk"]
        - vanilla_by_seed[seed]["entropic_risk"]
        for seed in range(8)
    ]
    final_summary = {
        "selection": selection,
        "private_aggregates": [
            aggregate_rows(rows, 0),
            aggregate_rows(rows, selected_id),
        ],
        "paired_private_best_minus_vanilla": {
            "values": paired_differences,
            "mean": float(np.mean(paired_differences)),
            "std": float(np.std(paired_differences, ddof=1)),
            "best_dh_wins": int(sum(value < 0.0 for value in paired_differences)),
            "n_training_seeds": len(paired_differences),
        },
    }
    atomic_write_json(LEARNED_ROOT / "final-summary.json", final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
