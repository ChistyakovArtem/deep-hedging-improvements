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
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()

