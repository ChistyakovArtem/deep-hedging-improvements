from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch

from src.artifacts import atomic_write_json, save_predictions, touch_terminal
from src.config import (
    DEFAULT_PROJECT_CONFIG,
    PROJECT_ROOT,
    load_project_config,
    load_toml,
)
from src.evaluation import outputs_to_numpy, scores_from_outputs
from src.hedger import DeepHedger, DeepHedgerConfig
from src.market import materialize_leaderboard
from src.nirvana_io import publish_experiment


def _hedger_config(config: dict) -> DeepHedgerConfig:
    model = config["model"]
    training = config["training"]
    return DeepHedgerConfig(
        architecture=str(model["architecture"]),
        feature_mode=str(model.get("feature_mode", "normalized")),
        prediction_target=str(model.get("prediction_target", "direct")),
        time_parameterization=str(model.get("time_parameterization", "shared")),
        hidden_dims=tuple(int(x) for x in model["hidden_dims"]),
        activation=str(model.get("activation", "leaky_relu")),
        batch_norm=bool(model.get("batch_norm", False)),
        output_initialization=str(model.get("output_initialization", "default")),
        n_frequencies=int(model["n_frequencies"]),
        paf_sigma=float(model["paf_sigma"]),
        periodic_include_linear=bool(model["periodic_include_linear"]),
        learning_rate=float(training["learning_rate"]),
        optimizer=str(training["optimizer"]),
        n_epochs=int(training["n_epochs"]),
        paths_per_epoch=int(training["paths_per_epoch"]),
        validation_interval=int(training["validation_interval"]),
        seed=int(config["seed"]),
    )


def run(
    config_dir: Path,
    device: str,
    smoke: bool,
) -> Path:
    config_dir = config_dir.resolve()
    config = load_toml(config_dir / "config.toml")
    actual_project_hash = hashlib.sha256(DEFAULT_PROJECT_CONFIG.read_bytes()).hexdigest()
    expected_project_hash = str(config["experiment"]["project_config_sha256"])
    if expected_project_hash != actual_project_hash:
        raise RuntimeError(
            f"Stale project config in {config_dir}: "
            f"expected {expected_project_hash}, got {actual_project_hash}."
        )
    if config["evaluation"]["leaderboards"] != ["public"]:
        raise RuntimeError("Stage-1 candidate configs may only access public.")
    if bool(config["evaluation"]["private_access"]):
        raise RuntimeError("Candidate config illegally requests private access.")

    project = load_project_config()
    public_paths, public_metadata = materialize_leaderboard(
        PROJECT_ROOT,
        project["market"],
        project["leaderboards"]["public"],
    )
    hedger_config = _hedger_config(config)
    output_dir = config_dir
    if smoke:
        hedger_config = replace(
            hedger_config,
            n_epochs=3,
            paths_per_epoch=128,
            validation_interval=1,
        )
        public_paths = public_paths[:512]
        relative = config_dir.relative_to(PROJECT_ROOT)
        output_dir = PROJECT_ROOT / "local" / "smoke" / relative
        output_dir.mkdir(parents=True, exist_ok=True)

    touch_terminal(output_dir, "RUNNING")
    started = time.perf_counter()
    hedger = DeepHedger(
        market=project["market"],
        objective=project["objective"],
        config=hedger_config,
        device=device,
    )
    public_tensor = torch.as_tensor(public_paths, device=hedger.device)
    hedger.fit(public_tensor)
    hedger.policy.eval()
    with torch.no_grad():
        actions, outputs = hedger.evaluate_tensor(public_tensor)
    scores = scores_from_outputs(
        outputs, risk_aversion=float(project["objective"]["risk_aversion"])
    )
    scores.update(
        {
            "leaderboard": "public",
            "paths_sha256": public_metadata["paths_sha256"],
            "seed": int(config["seed"]),
            "architecture": str(config["model"]["architecture"]),
            "paf_sigma": float(config["model"]["paf_sigma"]),
            "best_epoch": int(hedger.best_epoch or -1),
            "fit_and_eval_seconds": time.perf_counter() - started,
            "smoke": smoke,
        }
    )
    hedger.save(output_dir / "checkpoint.pt")
    atomic_write_json(output_dir / "training_log.json", hedger.training_log)
    atomic_write_json(output_dir / "scores-public.json", scores)
    if bool(config["evaluation"]["save_predictions"]):
        save_predictions(
            output_dir / "predictions-public.npz",
            actions.detach().cpu().numpy(),
            outputs_to_numpy(outputs),
        )
    (output_dir / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in scores.items()) + "\n"
    )
    touch_terminal(output_dir, "DONE")
    if not smoke:
        publish_experiment(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = run(args.config_dir, args.device, args.smoke)
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
